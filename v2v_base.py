"""
V2V Trajectory Prediction — IMPROVED v2
=========================================

Improvements over v1:
 1. Filter heading-unstable vehicles (removes corrupted forward deltas)
 2. Residual velocity connection (predict correction from physics baseline)
 3. Per-channel weighted loss (forward weighted 3x more)
 4. Per-channel normalization (separate scaler per output)
 5. Position anchor loss at 1s, 2s, 3s checkpoints

Usage:
 python v2v_final_v2.py --mode quick
 python v2v_final_v2.py --mode full
 python v2v_final_v2.py --mode export_pi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d
import warnings
import joblib
import argparse
import gc
import time
import os

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

LATENCY_OFFSET_FRAMES = 3
PREDICTION_HORIZON = 35
SEQUENCE_LENGTH = 15
SEQUENCE_STRIDE = 5

MODES = {
    'quick': {
        'max_sequences': 500_000,
        'epochs': 60,
        'patience': 18,
        'batch_size': 256,
        'hidden_size': 128,
        'description': 'Quick test (~1-2 hours on CPU)'
    },
    'full': {
        'max_sequences': 2_500_000,
        'epochs': 150,
        'patience': 25,
        'batch_size': 128,
        'hidden_size': 256,
        'description': 'Full training (overnight on CPU)'
    }
}


# ============================================================================
# 1. DATASET
# ============================================================================

class TrajectoryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# 2. LOSS FUNCTIONS — IMPROVED with per-channel weighting + anchor loss
# ============================================================================

class ImprovedMultiScaleLoss(nn.Module):
    """
    Improvements:
    - Per-channel weights: forward gets 3x weight
    - Position anchor loss at 1s, 2s, 3s checkpoints
    - Weighted temporal ramp
    """
    def __init__(self, output_length=35, ramp_end=1.5, alpha=0.3,
                 channel_weights=None, anchor_weight=0.2):
        super().__init__()
        
        # Temporal weights (later predictions weighted more)
        weights = torch.linspace(1.0, ramp_end, steps=output_length)
        self.register_buffer('temporal_weights', weights)
        
        # Per-channel weights: [forward, lateral, velocity]
        if channel_weights is None:
            channel_weights = [3.0, 1.0, 1.5] # Forward 3x, velocity 1.5x
        cw = torch.FloatTensor(channel_weights)
        self.register_buffer('channel_weights', cw)
        
        self.alpha = alpha
        self.anchor_weight = anchor_weight
        
        # Anchor points: 1s (10 frames), 2s (20), 3s (30)
        self.anchor_frames = [10, 20, 30]

    def forward(self, pred, target):
        # 1. Per-channel weighted temporal MSE
        sq_err = (pred - target) ** 2 # (batch, time, 3)
        
        # Apply channel weights
        weighted = sq_err * self.channel_weights[None, None, :]
        
        # Apply temporal weights
        T = pred.size(1)
        temporal_w = self.temporal_weights[:T]
        weighted = weighted * temporal_w[None, :, None]
        
        delta_loss = weighted.mean()
        
        # 2. Cumulative position loss
        cum_pred = torch.cumsum(pred[:, :, :2], dim=1)
        cum_true = torch.cumsum(target[:, :, :2], dim=1)
        abs_loss = F.mse_loss(cum_pred, cum_true)
        
        # 3. Position anchor loss at 1s, 2s, 3s
        anchor_loss = 0.0
        count = 0
        for af in self.anchor_frames:
            if af <= T:
                anchor_pred = cum_pred[:, af-1, :] # position at anchor
                anchor_true = cum_true[:, af-1, :]
                anchor_loss = anchor_loss + F.mse_loss(anchor_pred, anchor_true)
                count += 1
        if count > 0:
            anchor_loss = anchor_loss / count
        
        total = delta_loss + self.alpha * abs_loss + self.anchor_weight * anchor_loss
        return total


# ============================================================================
# 3. MODEL — Same architecture, works for both full and Pi
# ============================================================================

class TemporalAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.attn = nn.Linear(encoder_dim + decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        src_len = encoder_outputs.size(1)
        hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(
            torch.cat([encoder_outputs, hidden_expanded], dim=2)
        ))
        attention_weights = torch.softmax(self.v(energy).squeeze(2), dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights


class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=256, num_layers=2,
                 output_size=3, output_length=35, dropout=0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_length = output_length

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attention = TemporalAttention(hidden_size, hidden_size)

        self.decoder_cell = nn.LSTMCell(
            input_size=output_size + hidden_size,
            hidden_size=hidden_size
        )
        self.decoder_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        proj_mid = max(64, hidden_size // 2)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, proj_mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_mid, output_size)
        )

        self.enc_to_dec_h = nn.Linear(hidden_size, hidden_size)
        self.enc_to_dec_c = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, target=None, teacher_forcing_ratio=0.0):
        batch_size = x.size(0)

        encoder_outputs, (h_n, c_n) = self.encoder(x)

        dec_h = torch.tanh(self.enc_to_dec_h(h_n[-1]))
        dec_c = self.enc_to_dec_c(c_n[-1])
        decoder_input = torch.zeros(batch_size, self.output_size, device=x.device)
        predictions = []

        for t in range(self.output_length):
            context, _ = self.attention(encoder_outputs, dec_h)
            dec_in = torch.cat([decoder_input, context], dim=1)

            dec_h, dec_c = self.decoder_cell(dec_in, (dec_h, dec_c))
            dec_h = self.layer_norm(dec_h)
            dec_h_drop = self.decoder_dropout(dec_h)

            output_input = torch.cat([dec_h_drop, context], dim=1)
            prediction = self.output_proj(output_input)
            predictions.append(prediction)

            if target is not None and np.random.random() < teacher_forcing_ratio:
                decoder_input = target[:, t, :]
            else:
                decoder_input = prediction.detach()

        return torch.stack(predictions, dim=1)


# ============================================================================
# 4. DATA PREPROCESSING
# ============================================================================

def load_data():
    print("Loading NGSIM data...")
    df = pd.read_csv('Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20251027.csv', low_memory=False)
    print(f" Rows: {len(df):,}")

    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['Location']:
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
            except:
                pass

    vehicle_counts = df['Vehicle_ID'].value_counts()
    good = vehicle_counts[vehicle_counts >= 100].index
    df = df[df['Vehicle_ID'].isin(good)].sort_values(['Vehicle_ID', 'Frame_ID'])
    print(f" After filter: {len(df):,} rows, {df['Vehicle_ID'].nunique()} vehicles")
    return df


def compute_steering_features(df):
    print("Computing steering features...")

    def calc(group):
        dx = np.diff(group['Local_X'].values, prepend=group['Local_X'].iloc[0])
        dy = np.diff(group['Local_Y'].values, prepend=group['Local_Y'].iloc[0])
        dx[dx == 0] = 1e-6

        steer = np.degrees(np.arctan2(dy, dx))
        if len(steer) > 3:
            steer = gaussian_filter1d(steer, sigma=0.5)
        steer = np.clip(steer, -90, 90) / 90.0

        steer_rate = np.diff(steer, prepend=steer[0])
        if len(steer_rate) > 3:
            steer_rate = gaussian_filter1d(steer_rate, sigma=0.3)

        lat_vel = np.diff(group['Local_Y'].values, prepend=group['Local_Y'].iloc[0])
        if len(lat_vel) > 3:
            lat_vel = gaussian_filter1d(lat_vel, sigma=0.5)
        lat_vel = np.clip(lat_vel, -10, 10) / 10.0

        group['steering_normalized'] = steer
        group['steering_rate'] = steer_rate
        group['lateral_velocity'] = lat_vel
        return group

    df = df.groupby('Vehicle_ID', group_keys=False).apply(calc)
    for col in ['steering_normalized', 'steering_rate', 'lateral_velocity']:
        df[col] = df.groupby('Vehicle_ID')[col].ffill().bfill()
    return df


def compute_surrounding_features(df):
    print("Computing surrounding vehicle features...")
    print(f" Rows before: {len(df):,}")

    has_preceding = 'Preceding' in df.columns
    if has_preceding:
        df['Preceding'] = pd.to_numeric(df['Preceding'], errors='coerce').fillna(0).astype(int)

        print(f" Building lookup tables...")
        lookup = df.set_index(['Frame_ID', 'Vehicle_ID'])
        y_dict = lookup['Local_Y'].to_dict()
        vel_dict = lookup['v_Vel'].to_dict()
        del lookup

        print(f" Computing headway...")
        keys = list(zip(df['Frame_ID'], df['Preceding']))
        prec_y = pd.Series(keys).map(y_dict)
        df['space_headway_raw'] = (prec_y.values - df['Local_Y'].values)
        df['space_headway_raw'] = df['space_headway_raw'].abs()
        med = df['space_headway_raw'].median()
        if pd.isna(med): med = 100.0
        df['space_headway_raw'] = df['space_headway_raw'].fillna(med)
        df['space_headway'] = df['space_headway_raw'].clip(0, 500) / 500.0

        print(f" Computing relative velocity...")
        prec_vel = pd.Series(keys).map(vel_dict)
        df['rel_velocity_raw'] = df['v_Vel'].values - prec_vel.fillna(df['v_Vel']).values
        df['rel_velocity'] = df['rel_velocity_raw'].clip(-50, 50) / 50.0

        print(f" Computing TTC...")
        hdwy = df['space_headway_raw'].values
        rv = df['rel_velocity_raw'].values
        ttc = np.where(rv > 0.5, hdwy / rv, 10.0)
        df['ttc'] = np.clip(ttc, 0, 10.0) / 10.0

        df = df.drop(columns=['space_headway_raw', 'rel_velocity_raw'], errors='ignore')
    else:
        df['space_headway'] = 0.5
        df['rel_velocity'] = 0.0
        df['ttc'] = 1.0

    for col in ['space_headway', 'rel_velocity', 'ttc']:
        df[col] = df.groupby('Vehicle_ID')[col].transform(
            lambda x: gaussian_filter1d(x.fillna(x.median() if x.notna().any() else 0.5).values, sigma=0.5)
        )

    print(f" Rows after: {len(df):,}")
    return df


def handle_missing_data(df):
    print("Handling missing data...")
    df = compute_steering_features(df)

    keep_cols = [
        'Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y',
        'v_Vel', 'v_Acc', 'Preceding',
        'steering_normalized', 'steering_rate', 'lateral_velocity'
    ]
    available = [c for c in keep_cols if c in df.columns]
    if 'Location' in df.columns:
        available.append('Location')
    df = df[available].sort_values(['Vehicle_ID', 'Frame_ID'])

    for col in ['Local_X', 'Local_Y', 'v_Vel', 'v_Acc']:
        if col in df.columns:
            df[col] = df.groupby('Vehicle_ID', group_keys=False)[col].apply(
                lambda x: x.ffill().bfill()
            )

    before = len(df)
    df = df.dropna(subset=['Local_X', 'Local_Y', 'v_Vel'])
    print(f" Dropped {before - len(df)} rows with missing data")
    return df


# ============================================================================
# 5. EGO-CENTRIC COORDINATE TRANSFORM — v2 stable heading
# ============================================================================

def compute_ego_centric_deltas(df):
    print("Computing ego-centric deltas (v2 — stable heading)...")

    def calc_ego_deltas(group):
        group = group.sort_values('Frame_ID')

        x = group['Local_X'].values.astype(np.float64)
        y = group['Local_Y'].values.astype(np.float64)
        vel = group['v_Vel'].values.astype(np.float64)

        n = len(x)

        if n > 7:
            x_smooth = gaussian_filter1d(x, sigma=3.0)
            y_smooth = gaussian_filter1d(y, sigma=3.0)
        else:
            x_smooth = x.copy()
            y_smooth = y.copy()

        window = min(5, n - 1)
        heading = np.zeros(n)

        for i in range(n):
            i_start = max(0, i - window // 2)
            i_end = min(n - 1, i + window // 2)

            if i_end - i_start < 2:
                i_start = max(0, i_end - 2)
                i_end = min(n - 1, i_start + 2)

            dx = x_smooth[i_end] - x_smooth[i_start]
            dy = y_smooth[i_end] - y_smooth[i_start]

            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                heading[i] = heading[i-1] if i > 0 else 0.0
            else:
                heading[i] = np.arctan2(dy, dx)

        if n > 5:
            sin_h = gaussian_filter1d(np.sin(heading), sigma=3.0)
            cos_h = gaussian_filter1d(np.cos(heading), sigma=3.0)
            heading = np.arctan2(sin_h, cos_h)

        dx_world = np.diff(x, prepend=x[0])
        dy_world = np.diff(y, prepend=y[0])
        dv = np.diff(vel, prepend=vel[0])

        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        ego_forward = dx_world * cos_h + dy_world * sin_h
        ego_lateral = -dx_world * sin_h + dy_world * cos_h

        if np.median(ego_forward[1:]) < -1.0:
            heading = heading + np.pi
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            ego_forward = dx_world * cos_h + dy_world * sin_h
            ego_lateral = -dx_world * sin_h + dy_world * cos_h

        ego_forward = np.clip(ego_forward, -15, 30)
        ego_lateral = np.clip(ego_lateral, -5, 5)
        dv = np.clip(dv, -10, 10)

        if n > 3:
            ego_forward = gaussian_filter1d(ego_forward, sigma=0.3)
            ego_lateral = gaussian_filter1d(ego_lateral, sigma=0.3)
            dv = gaussian_filter1d(dv, sigma=0.3)

        group['delta_forward'] = ego_forward
        group['delta_lateral'] = ego_lateral
        group['delta_vel'] = dv
        group['heading'] = heading

        return group

    df = df.groupby('Vehicle_ID', group_keys=False).apply(calc_ego_deltas)

    fwd = df['delta_forward']
    lat = df['delta_lateral']
    dv = df['delta_vel']

    print(f" delta_forward: mean={fwd.mean():.4f}, std={fwd.std():.4f}, "
          f"range=[{fwd.min():.4f}, {fwd.max():.4f}]")
    print(f" delta_lateral: mean={lat.mean():.4f}, std={lat.std():.4f}")
    print(f" delta_vel: mean={dv.mean():.4f}, std={dv.std():.4f}")

    pct_positive = (fwd > 0).mean() * 100
    print(f" delta_forward positive: {pct_positive:.1f}% (should be >80%)")

    median_fwd = fwd.median()
    print(f" delta_forward median: {median_fwd:.2f} ft")

    return df


# ============================================================================
# 6. ★ NEW: FILTER HEADING-UNSTABLE VEHICLES
# ============================================================================

def filter_heading_unstable(df):
    """
    FIX 1: Remove vehicles whose forward deltas oscillate wildly.
    These are vehicles where the heading estimation failed, causing
    forward deltas to flip between +15 and -15 every frame.
    
    Criteria:
    - forward delta std > 8.0 (normal vehicles have std < 4)
    - forward delta sign flips > 20% of frames
    """
    print("\nFiltering heading-unstable vehicles...")
    
    before_veh = df['Vehicle_ID'].nunique()
    before_rows = len(df)
    
    # Compute per-vehicle stats
    veh_stats = df.groupby('Vehicle_ID')['delta_forward'].agg(['std', 'mean']).reset_index()
    
    # Also compute sign flip percentage
    def sign_flip_pct(group):
        fwd = group['delta_forward'].values
        if len(fwd) < 10:
            return 1.0  # too short, exclude
        signs = np.sign(fwd[1:])  # skip first frame
        flips = np.sum(np.abs(np.diff(signs)) > 0) / len(signs)
        return flips
    
    flip_pcts = df.groupby('Vehicle_ID').apply(sign_flip_pct).reset_index()
    flip_pcts.columns = ['Vehicle_ID', 'flip_pct']
    
    veh_stats = veh_stats.merge(flip_pcts, on='Vehicle_ID')
    
    # Filter criteria
    stable = veh_stats[
        (veh_stats['std'] < 8.0) &       # std < 8 (normal is 2-4)
        (veh_stats['flip_pct'] < 0.25) & # less than 25% sign flips
        (veh_stats['mean'] > 0.5)        # mean forward is positive
    ]['Vehicle_ID']
    
    # Also show what we're removing
    unstable = veh_stats[~veh_stats['Vehicle_ID'].isin(stable)]
    if len(unstable) > 0:
        print(f" Unstable vehicles removed:")
        print(f" High std (>8): {(veh_stats['std'] >= 8.0).sum()}")
        print(f" High flip rate (>25%): {(veh_stats['flip_pct'] >= 0.25).sum()}")
        print(f" Low mean (<0.5): {(veh_stats['mean'] <= 0.5).sum()}")
    
    df = df[df['Vehicle_ID'].isin(stable)]
    
    after_veh = df['Vehicle_ID'].nunique()
    after_rows = len(df)
    
    print(f" Vehicles: {before_veh} → {after_veh} (removed {before_veh - after_veh})")
    print(f" Rows: {before_rows:,} → {after_rows:,} (removed {before_rows - after_rows:,})")
    
    # Show improved stats
    fwd = df['delta_forward']
    print(f" After filtering:")
    print(f" delta_forward: mean={fwd.mean():.4f}, std={fwd.std():.4f}")
    print(f" positive: {(fwd > 0).mean()*100:.1f}%")
    print(f" median: {fwd.median():.2f} ft")
    
    return df


# ============================================================================
# 7. ★ NEW: COMPUTE RESIDUAL TARGETS
# ============================================================================

def compute_residual_targets(df):
    """
    FIX 2: Instead of predicting raw delta_forward, predict the RESIDUAL
    from a physics baseline (velocity * dt).
    
    raw delta_forward ≈ velocity * 0.1 (at 10Hz)
    residual = delta_forward - velocity * 0.1
    
    The residual is much smaller and easier to predict.
    The model only needs to learn the CORRECTION from constant-velocity.
    """
    print("\nComputing residual targets...")
    
    dt = 0.1  # 10 Hz
    
    # Physics baseline: if car continues at current speed
    baseline_forward = df['v_Vel'].values * dt
    
    # Residual = actual - baseline
    df['residual_forward'] = df['delta_forward'].values - baseline_forward
    
    # Stats comparison
    raw_std = df['delta_forward'].std()
    res_std = df['residual_forward'].std()
    raw_mae = df['delta_forward'].abs().mean()
    res_mae = df['residual_forward'].abs().mean()
    
    print(f" Raw delta_forward: std={raw_std:.4f}, MAE={raw_mae:.4f}")
    print(f" Residual (after vel): std={res_std:.4f}, MAE={res_mae:.4f}")
    print(f" Reduction: std {(1-res_std/raw_std)*100:.1f}%, MAE {(1-res_mae/raw_mae)*100:.1f}%")
    
    return df


def remove_outliers(df):
    print("Removing outliers (IQR)...")

    for col in ['residual_forward', 'delta_lateral', 'delta_vel']:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR < 1e-10:
            print(f" ⚠ {col}: IQR ≈ 0 — SKIPPING")
            continue

        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        df[col] = df[col].clip(lower, upper)
        print(f" {col}: [{lower:.4f}, {upper:.4f}]")

    return df


# ============================================================================
# 8. SEQUENCE CREATION — Now uses residual_forward instead of delta_forward
# ============================================================================

def create_sequences(df, max_sequences, stride=SEQUENCE_STRIDE):
    print("Creating sequences...")
    print(f" Stride: {stride}, Max: {max_sequences:,}")

    feature_cols = ['Local_X', 'Local_Y', 'v_Vel', 'v_Acc',
                    'steering_normalized', 'steering_rate', 'lateral_velocity',
                    'space_headway', 'rel_velocity', 'ttc']
    
    # ★ CHANGED: residual_forward instead of delta_forward
    target_cols = ['residual_forward', 'delta_lateral', 'delta_vel']

    avail_feat = [c for c in feature_cols if c in df.columns]
    avail_tgt = [c for c in target_cols if c in df.columns]
    print(f" Features ({len(avail_feat)}): {avail_feat}")
    print(f" Targets ({len(avail_tgt)}): {avail_tgt}")

    total_needed = SEQUENCE_LENGTH + LATENCY_OFFSET_FRAMES + PREDICTION_HORIZON
    seqs_X, seqs_y = [], []
    veh_count = 0

    for vid, vdata in df.groupby('Vehicle_ID'):
        vdata = vdata.sort_values('Frame_ID').reset_index(drop=True)
        if len(vdata) < total_needed:
            continue

        veh_count += 1
        feat = vdata[avail_feat].values
        tgt = vdata[avail_tgt].values

        for i in range(0, len(vdata) - total_needed + 1, stride):
            seqs_X.append(feat[i:i + SEQUENCE_LENGTH])
            y_start = i + SEQUENCE_LENGTH + LATENCY_OFFSET_FRAMES
            seqs_y.append(tgt[y_start:y_start + PREDICTION_HORIZON])

        if len(seqs_X) >= max_sequences:
            break
            
    if len(seqs_X) >= max_sequences:
        print(f" Hit cap at vehicle {veh_count}")

    X = np.array(seqs_X, dtype=np.float32)
    y = np.array(seqs_y, dtype=np.float32)
    mem_gb = (X.nbytes + y.nbytes) / (1024**3)
    print(f" Created {len(X):,} sequences from {veh_count} vehicles")
    print(f" X: {X.shape}, y: {y.shape}, Memory: {mem_gb:.1f} GB")
    
    # Show target stats
    print(f"\n Target statistics (after filtering + residual):")
    for i, col in enumerate(avail_tgt):
        vals = y[:, :, i].flatten()
        print(f" {col:20s}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")
    
    return X, y, avail_feat, avail_tgt


def split_data(X, y):
    print("Splitting data...")
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    n_test = int(n * 0.2)
    n_val = int(n * 0.2)

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    print(f" Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}")
    return (X[train_idx], X[val_idx], X[test_idx],
            y[train_idx], y[val_idx], y[test_idx])


# ============================================================================
# 9. ★ NEW: PER-CHANNEL NORMALIZATION
# ============================================================================

def normalize_per_channel(X_tr, X_va, X_te, y_tr, y_va, y_te):
    """
    FIX 4: Normalize each output channel separately.
    
    Previously, one StandardScaler for all 3 targets meant:
    - forward (mean=4, std=14) dominated the scale
    - lateral (mean=0, std=0.3) was compressed to tiny range
    - model paid unequal attention
    
    Now each channel gets its own mean/std, so all three are in [-3, +3] range.
    """
    print("Normalizing (per-channel for targets)...")
    
    # Input: standard normalization (same as before)
    sX = StandardScaler()
    nfx = X_tr.shape[-1]
    sX.fit(X_tr.reshape(-1, nfx))
    
    def tx(d): return sX.transform(d.reshape(-1, nfx)).reshape(d.shape)
    Xtn, Xvn, Xten = tx(X_tr), tx(X_va), tx(X_te)
    
    # Output: per-channel normalization
    nfy = y_tr.shape[-1]
    y_scalers = []
    ytn = np.zeros_like(y_tr)
    yvn = np.zeros_like(y_va)
    yten = np.zeros_like(y_te)
    
    for ch in range(nfy):
        sc = StandardScaler()
        train_ch = y_tr[:, :, ch].reshape(-1, 1)
        sc.fit(train_ch)
        
        ytn[:, :, ch] = sc.transform(y_tr[:, :, ch].reshape(-1, 1)).reshape(y_tr[:, :, ch].shape)
        yvn[:, :, ch] = sc.transform(y_va[:, :, ch].reshape(-1, 1)).reshape(y_va[:, :, ch].shape)
        yten[:, :, ch] = sc.transform(y_te[:, :, ch].reshape(-1, 1)).reshape(y_te[:, :, ch].shape)
        
        y_scalers.append(sc)
        print(f" Channel {ch}: mean={sc.mean_[0]:.4f}, std={sc.scale_[0]:.4f}")
        
    print(f" y_train range: [{ytn.min():.2f}, {ytn.max():.2f}]")
    
    return Xtn, Xvn, Xten, ytn, yvn, yten, sX, y_scalers


def inverse_transform_per_channel(y_norm, y_scalers):
    """Convert normalized predictions back to original scale, per channel."""
    y_out = np.zeros_like(y_norm)
    for ch in range(y_norm.shape[-1]):
        y_out[:, :, ch] = y_scalers[ch].inverse_transform(
            y_norm[:, :, ch].reshape(-1, 1)
        ).reshape(y_norm[:, :, ch].shape)
    return y_out


# ============================================================================
# 10. TRAINING
# ============================================================================

def get_curriculum_horizon(epoch):
    if epoch < 20: return 15
    elif epoch < 40: return 25
    else: return PREDICTION_HORIZON


def train_model(model, train_loader, val_loader, epochs, patience, device='cpu'):
    print(f"\nTraining on {device}, {epochs} epochs, patience={patience}")
    model = model.to(device)

    # ★ CHANGED: Use improved loss with channel weights and anchors
    criterion_full = ImprovedMultiScaleLoss(
        PREDICTION_HORIZON, 1.5, 0.3,
        channel_weights=[3.0, 1.0, 1.5],  # Forward 3x
        anchor_weight=0.2
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)

    best_val = float('inf')
    best_epoch = 0
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        horizon = get_curriculum_horizon(epoch)
        criterion = ImprovedMultiScaleLoss(
            horizon, 1.5, 0.3,
            channel_weights=[3.0, 1.0, 1.5],
            anchor_weight=0.2
        ).to(device)
        tf = max(0.0, 0.4 * (1.0 - epoch / (epochs * 0.6)))

        model.train()
        tloss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx, target=by, teacher_forcing_ratio=tf)
            loss = criterion(out[:, :horizon], by[:, :horizon])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tloss += loss.item()
            
        tloss /= len(train_loader)
        scheduler.step()

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                vloss += criterion_full(out, by).item()
            vloss /= len(val_loader)

        train_losses.append(tloss)
        val_losses.append(vloss)

        if epoch == 0:
            model.eval()
            with torch.no_grad():
                sx, sy_sample = next(iter(val_loader))
                so = model(sx.to(device))
                pm, ps = so.mean().item(), so.std().item()
                tm, ts = sy_sample.mean().item(), sy_sample.std().item()
                print(f"\n Epoch 1 Diagnostic:")
                print(f" Pred — mean: {pm:.4f}, std: {ps:.4f}")
                print(f" Target — mean: {tm:.4f}, std: {ts:.4f}")
                if ps < 1e-4:
                    print(f" ⚠ WARNING: Near-constant predictions!")
                else:
                    print(f" ✓ Predictions have variance\n")

        if vloss < best_val:
            best_val = vloss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1

        lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f" Ep {epoch+1:>3d}/{epochs} | Train: {tloss:.2e} | Val: {vloss:.2e} | "
                  f"LR: {lr:.1e} | TF: {tf:.2f} | Hz: {horizon} | "
                  f"Best: {best_val:.2e} @{best_epoch+1}")

        if no_improve >= patience:
            print(f" Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load('best_model.pth'))
    print(f" ✓ Best: epoch {best_epoch+1}, val={best_val:.2e}")
    return train_losses, val_losses


# ============================================================================
# 11. EVALUATION — Updated for residual + per-channel
# ============================================================================

def evaluate(model, test_loader, X_test_raw, y_test_raw, y_scalers, device='cpu'):
    print("\nEvaluating...")
    model.to(device).eval()

    preds, targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            out = model(bx.to(device))
            preds.append(out.cpu().numpy())
            targets.append(by.numpy())

    y_pred_n = np.concatenate(preds)
    y_true_n = np.concatenate(targets)

    # Per-channel inverse transform
    y_pred = inverse_transform_per_channel(y_pred_n, y_scalers)
    y_true = inverse_transform_per_channel(y_true_n, y_scalers)

    # ★ IMPORTANT: predictions are residual_forward, delta_lateral, delta_vel
    # To get actual forward delta, add back the velocity baseline
    # For evaluation, we use the last input velocity
    last_vel = X_test_raw[:, -1, 2]  # v_Vel is feature index 2
    dt = 0.1
    vel_baseline = last_vel * dt  # (batch,)
    
    # Convert residual back to actual forward delta
    pred_fwd_actual = y_pred[:, :, 0] + vel_baseline[:, None]
    true_fwd_actual = y_true[:, :, 0] + vel_baseline[:, None]
    
    pred_lat = y_pred[:, :, 1]
    true_lat = y_true[:, :, 1]
    pred_dv = y_pred[:, :, 2]
    true_dv = y_true[:, :, 2]

    # Cumulative displacement
    pred_cum_fwd = np.cumsum(pred_fwd_actual, axis=1)
    pred_cum_lat = np.cumsum(pred_lat, axis=1)
    true_cum_fwd = np.cumsum(true_fwd_actual, axis=1)
    true_cum_lat = np.cumsum(true_lat, axis=1)

    fwd_mae = np.mean(np.abs(true_cum_fwd - pred_cum_fwd))
    lat_mae = np.mean(np.abs(true_cum_lat - pred_cum_lat))
    vel_mae = np.mean(np.abs(true_dv - pred_dv))

    delta_fwd_mae = np.mean(np.abs(true_fwd_actual - pred_fwd_actual))
    delta_lat_mae = np.mean(np.abs(true_lat - pred_lat))

    overall_mae = (np.mean(np.abs(true_fwd_actual - pred_fwd_actual)) +
                   np.mean(np.abs(true_lat - pred_lat)) +
                   np.mean(np.abs(true_dv - pred_dv))) / 3

    horizons = [5, 10, 15, 20, 25, 30, 35]
    h_fwd, h_lat = [], []

    print(f"\n{'='*65}")
    print(f" EVALUATION RESULTS (Ego-Centric, Residual)")
    print(f"{'='*65}")
    print(f" Horizon: {PREDICTION_HORIZON} frames ({PREDICTION_HORIZON/10:.1f}s)")
    print(f" Latency offset: {LATENCY_OFFSET_FRAMES} frames ({LATENCY_OFFSET_FRAMES*100}ms)")
    print(f" ★ Using residual velocity connection")

    print(f"\n CUMULATIVE DISPLACEMENT MAE:")
    print(f" Forward (along heading): {fwd_mae:.4f} ft")
    print(f" Lateral (perpendicular): {lat_mae:.4f} ft")
    print(f" Velocity change: {vel_mae:.4f} ft/s")

    print(f"\n PER-STEP DELTA MAE:")
    print(f" delta_forward (actual): {delta_fwd_mae:.4f} ft")
    print(f" delta_lateral: {delta_lat_mae:.4f} ft")

    print(f"\n TIME-HORIZON BREAKDOWN (cumulative):")
    for h in horizons:
        if h <= y_pred.shape[1]:
            hf = np.mean(np.abs(true_cum_fwd[:,:h] - pred_cum_fwd[:,:h]))
            hl = np.mean(np.abs(true_cum_lat[:,:h] - pred_cum_lat[:,:h]))
            h_fwd.append(hf)
            h_lat.append(hl)
            print(f" {h/10:.1f}s: Forward MAE = {hf:>7.2f} ft | Lateral MAE = {hl:>7.2f} ft")

    print(f"\n Overall MAE: {overall_mae:.4f}")
    print(f"{'='*65}\n")

    # Build y arrays for plotting (use actual forward, not residual)
    y_pred_plot = np.stack([pred_fwd_actual, pred_lat, pred_dv], axis=2)
    y_true_plot = np.stack([true_fwd_actual, true_lat, true_dv], axis=2)

    return y_pred_plot, y_true_plot, horizons, h_fwd, h_lat


# ============================================================================
# 12. VISUALIZATION
# ============================================================================

def plot_results(train_losses, val_losses, y_pred, y_true, horizons, h_fwd, h_lat):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.semilogy(train_losses, label='Train'); a1.semilogy(val_losses, label='Val')
    a1.axvline(20, color='orange', ls=':', alpha=0.6, label='15→25')
    a1.axvline(40, color='red', ls=':', alpha=0.6, label='25→35')
    a1.set_title('Loss (Log)'); a1.legend(); a1.grid(True, alpha=0.3)
    a2.plot(train_losses, label='Train'); a2.plot(val_losses, label='Val')
    a2.axvline(20, color='orange', ls=':', alpha=0.6)
    a2.axvline(40, color='red', ls=':', alpha=0.6)
    a2.set_title('Loss (Linear)'); a2.legend(); a2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('training_history_v2.png', dpi=150); plt.close()

    names = ['Forward', 'Lateral', 'Velocity']
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    for ti in range(3):
        for si in range(3):
            ax = axes[si, ti]
            t = np.arange(y_true.shape[1])
            ax.plot(t, y_true[si, :, ti], lw=2, color=colors[ti], label='True')
            ax.plot(t, y_pred[si, :, ti], lw=2, ls='--', color='red', label='Pred')
            mae = np.mean(np.abs(y_true[si, :, ti] - y_pred[si, :, ti]))
            ax.set_title(f'S{si+1} — {names[ti]} (MAE:{mae:.3f})', fontsize=9)
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('ego_predictions_v2.png', dpi=150); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ti in range(3):
        err = (y_true[:,:,ti] - y_pred[:,:,ti]).flatten()
        axes[ti].hist(err, bins=50, alpha=0.7, color=colors[ti])
        axes[ti].axvline(0, color='red', ls='--')
        axes[ti].set_title(f'{names[ti]} Error'); axes[ti].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('error_distributions_v2.png', dpi=150); plt.close()

    if h_fwd:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(h_fwd)); w = 0.35
        ax.bar(x - w/2, h_fwd, w, label='Forward', color='#2196F3', alpha=0.8)
        ax.bar(x + w/2, h_lat, w, label='Lateral', color='#4CAF50', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels([f'{h/10:.1f}s' for h in horizons[:len(h_fwd)]])
        ax.set_ylabel('MAE (ft)'); ax.set_title('Error by Horizon (Ego-Centric, Improved)')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig('horizon_analysis_v2.png', dpi=150); plt.close()

    print(" Saved: training_history_v2.png, ego_predictions_v2.png,")
    print("        error_distributions_v2.png, horizon_analysis_v2.png")


# ============================================================================
# 13. PI MODEL EXPORT
# ============================================================================

def export_pi_model(X_train, y_train, X_val, y_val, scaler_X, y_scalers, 
                    feature_cols, target_cols, device='cpu'):
    print("\n" + "="*65)
    print(" EXPORTING PI MODEL (lightweight)")
    print("="*65)

    nfx = X_train.shape[-1]
    nfy = y_train.shape[-1]

    # Normalize using same scalers
    Xtn = scaler_X.transform(X_train.reshape(-1, nfx)).reshape(X_train.shape)
    Xvn = scaler_X.transform(X_val.reshape(-1, nfx)).reshape(X_val.shape)
    
    # Per-channel normalization for targets
    ytn = np.zeros_like(y_train)
    yvn = np.zeros_like(y_val)
    for ch in range(nfy):
        ytn[:, :, ch] = y_scalers[ch].transform(y_train[:, :, ch].reshape(-1, 1)).reshape(y_train[:, :, ch].shape)
        yvn[:, :, ch] = y_scalers[ch].transform(y_val[:, :, ch].reshape(-1, 1)).reshape(y_val[:, :, ch].shape)

    train_loader = DataLoader(TrajectoryDataset(Xtn, ytn), batch_size=256, shuffle=True)
    val_loader = DataLoader(TrajectoryDataset(Xvn, yvn), batch_size=256, shuffle=False)

    pi_model = TrajectoryLSTM(
        input_size=nfx,
        hidden_size=64,
        num_layers=1,
        output_size=nfy,
        output_length=PREDICTION_HORIZON,
        dropout=0.1
    )

    params = sum(p.numel() for p in pi_model.parameters())
    print(f" Pi model: {params:,} parameters")

    train_losses, val_losses = train_model(
        pi_model, train_loader, val_loader,
        epochs=40, patience=12, device=device
    )

    pi_model.load_state_dict(torch.load('best_model.pth'))
    torch.save(pi_model.state_dict(), 'pi_model_v2.pth')

    pi_config = {
        'input_size': nfx,
        'hidden_size': 64,
        'num_layers': 1,
        'output_size': nfy,
        'output_length': PREDICTION_HORIZON,
        'dropout': 0.1,
        'latency_offset': LATENCY_OFFSET_FRAMES,
        'sequence_length': SEQUENCE_LENGTH,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'coordinate_system': 'ego_centric',
        'residual_forward': True,  # ★ NEW: flag for Pi code
    }
    joblib.dump(pi_config, 'pi_model_config_v2.pkl')

    joblib.dump(scaler_X, 'pi_scaler_X_v2.pkl')
    joblib.dump(y_scalers, 'pi_scaler_y_v2.pkl')  # ★ List of per-channel scalers

    print(f"\n Inference speed test...")
    dummy = torch.randn(1, SEQUENCE_LENGTH, nfx)
    pi_model.eval()
    times = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.time()
            _ = pi_model(dummy)
            times.append(time.time() - t0)
    avg_ms = np.mean(times) * 1000
    print(f" Average inference: {avg_ms:.1f} ms (on this machine)")

    print(f"\n Files to copy to Raspberry Pi:")
    print(f"   pi_model_v2.pth ({os.path.getsize('pi_model_v2.pth')/1024:.0f} KB)")
    print(f"   pi_model_config_v2.pkl")
    print(f"   pi_scaler_X_v2.pkl")
    print(f"   pi_scaler_y_v2.pkl")
    print(f"\n ★ Pi code must add velocity*dt back to residual_forward predictions")

    return pi_model


# ============================================================================
# 14. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='V2V Trajectory Prediction — Improved v2')
    parser.add_argument('--mode', choices=['quick', 'full', 'export_pi'],
                        default='quick', help='Training mode')
    args = parser.parse_args()

    if args.mode == 'export_pi':
        print("Loading saved training data for Pi export...")
        if not os.path.exists('train_data_v2.npz'):
            print("ERROR: No saved training data. Run --mode quick or full first.")
            return
        data = np.load('train_data_v2.npz')
        scaler_X = joblib.load('scaler_X_v2.pkl')
        y_scalers = joblib.load('scaler_y_v2.pkl')
        feat_cols = joblib.load('model_config_v2.pkl')['feature_cols']
        tgt_cols = joblib.load('model_config_v2.pkl')['target_cols']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        export_pi_model(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            scaler_X, y_scalers, feat_cols, tgt_cols, device
        )
        return

    mode = MODES[args.mode]
    print("=" * 65)
    print(f" V2V TRAJECTORY PREDICTION — IMPROVED v2 ({args.mode} mode)")
    print(f" {mode['description']}")
    print("=" * 65)
    print(f" ✓ Ego-centric coordinates (direction-independent)")
    print(f" ✓ Surrounding vehicle features")
    print(f" ★ NEW: Filter heading-unstable vehicles")
    print(f" ★ NEW: Residual velocity connection")
    print(f" ★ NEW: Per-channel weighted loss (forward 3x)")
    print(f" ★ NEW: Per-channel normalization")
    print(f" ★ NEW: Position anchor loss at 1s, 2s, 3s")
    print(f" ✓ Multi-scale loss + curriculum learning")
    print(f" ✓ V2V latency compensation ({LATENCY_OFFSET_FRAMES*100}ms)")
    print(f" ✓ Sequences: {mode['max_sequences']:,}, Epochs: {mode['epochs']}")
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Pipeline ──
    df = load_data()
    df = handle_missing_data(df)
    df = compute_surrounding_features(df)
    df = compute_ego_centric_deltas(df)
    df = filter_heading_unstable(df)  # ★ NEW: Fix 1
    df = compute_residual_targets(df)  # ★ NEW: Fix 2
    df = remove_outliers(df)

    X, y, feat_cols, tgt_cols = create_sequences(df, mode['max_sequences'])
    del df; gc.collect()

    if len(X) == 0:
        print("ERROR: No sequences created.")
        return

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y)
    del X, y; gc.collect()

    # ★ NEW: Per-channel normalization (Fix 4)
    Xtn, Xvn, Xten, ytn, yvn, yten, sX, y_scalers = normalize_per_channel(
        X_tr, X_va, X_te, y_tr, y_va, y_te
    )

    # Save for Pi export later
    np.savez_compressed('train_data_v2.npz',
                        X_train=X_tr, y_train=y_tr,
                        X_val=X_va, y_val=y_va,
                        X_test=X_te, y_test=y_te)
    joblib.dump(sX, 'scaler_X_v2.pkl')
    joblib.dump(y_scalers, 'scaler_y_v2.pkl')

    # Loaders
    tr_loader = DataLoader(TrajectoryDataset(Xtn, ytn),
                           batch_size=mode['batch_size'], shuffle=True)
    va_loader = DataLoader(TrajectoryDataset(Xvn, yvn),
                           batch_size=mode['batch_size'], shuffle=False)
    te_loader = DataLoader(TrajectoryDataset(Xten, yten),
                           batch_size=mode['batch_size'], shuffle=False)

    del Xtn, Xvn, Xten, ytn, yvn, yten; gc.collect()

    # Model
    model = TrajectoryLSTM(
        input_size=X_tr.shape[2],
        hidden_size=mode['hidden_size'],
        num_layers=2 if args.mode == 'full' else 1,
        output_size=y_tr.shape[2],
        output_length=PREDICTION_HORIZON,
        dropout=0.15
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"\n Model: {params:,} parameters")

    # Train
    t_losses, v_losses = train_model(
        model, tr_loader, va_loader,
        epochs=mode['epochs'], patience=mode['patience'], device=device
    )

    # Save config
    config = {
        'input_size': X_tr.shape[2],
        'hidden_size': mode['hidden_size'],
        'num_layers': 2 if args.mode == 'full' else 1,
        'output_size': y_tr.shape[2],
        'output_length': PREDICTION_HORIZON,
        'feature_cols': feat_cols,
        'target_cols': tgt_cols,
        'coordinate_system': 'ego_centric',
        'residual_forward': True,
        'mode': args.mode,
    }
    joblib.dump(config, 'model_config_v2.pkl')
    torch.save(model.state_dict(), f'model_{args.mode}_v2.pth')

    # Evaluate
    y_pred, y_true, horizons, h_fwd, h_lat = evaluate(
        model, te_loader, X_te, y_te, y_scalers, device
    )

    # Plot
    plot_results(t_losses, v_losses, y_pred, y_true, horizons, h_fwd, h_lat)

    # Auto-export Pi model
    print("\n Now exporting lightweight Pi model...")
    export_pi_model(X_tr, y_tr, X_va, y_va, sX, y_scalers, feat_cols, tgt_cols, device)

    print("\n" + "="*65)
    print(" ALL DONE! (v2 — improved)")
    print("="*65)
    print(f" Report model: model_{args.mode}_v2.pth ({params:,} params)")
    print(f" Pi model: pi_model_v2.pth")
    print(f" Improvements: heading filter + residual + channel weights + anchors")
    print(f" Plots: *_v2.png")


if __name__ == "__main__":
    main()