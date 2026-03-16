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

LATENCY_OFFSET_FRAMES = 3  # 300ms V2V pipeline delay
PREDICTION_HORIZON = 35    # 3.5s at 10 Hz
SEQUENCE_LENGTH = 15       # 1.5s input context
SEQUENCE_STRIDE = 5        # stride between sequences

# Mode-specific settings
MODES = {
    'quick': {
        'max_sequences': 500_000,
        'epochs': 50,
        'patience': 15,
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
# 2. LOSS FUNCTIONS
# ============================================================================

class WeightedTemporalMSE(nn.Module):
    def __init__(self, output_length=35, ramp_end=1.5):
        super().__init__()
        weights = torch.linspace(1.0, ramp_end, steps=output_length)
        self.register_buffer('weights', weights)

    def forward(self, pred, target):
        sq_err = (pred - target) ** 2
        weighted = sq_err * self.weights[None, :, None]
        return weighted.mean()

class MultiScaleLoss(nn.Module):
    def __init__(self, output_length=35, ramp_end=1.5, alpha=0.3):
        super().__init__()
        self.weighted_mse = WeightedTemporalMSE(output_length, ramp_end)
        self.alpha = alpha

    def forward(self, pred, target):
        delta_loss = self.weighted_mse(pred, target)
        cum_pred = torch.cumsum(pred[:, :, :2], dim=1)
        cum_true = torch.cumsum(target[:, :, :2], dim=1)
        abs_loss = F.mse_loss(cum_pred, cum_true)
        return delta_loss + self.alpha * abs_loss

# ============================================================================
# 3. MODEL — Encoder-Decoder LSTM with Attention
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
    df = df.dropna(subset=['Local_X', 'Local_Y', 'v_Vel'])
    return df

# ============================================================================
# 5. EGO-CENTRIC COORDINATE TRANSFORM
# ============================================================================

def compute_ego_centric_deltas(df):
    print("Computing ego-centric deltas (v2 — stable heading)...")

    def calc_ego_deltas(group):
        group = group.sort_values('Frame_ID')
        x, y, vel = group['Local_X'].values.astype(np.float64), \
                    group['Local_Y'].values.astype(np.float64), \
                    group['v_Vel'].values.astype(np.float64)
        n = len(x)

        if n > 7:
            x_smooth, y_smooth = gaussian_filter1d(x, sigma=3.0), gaussian_filter1d(y, sigma=3.0)
        else:
            x_smooth, y_smooth = x.copy(), y.copy()

        window = min(5, n - 1)
        heading = np.zeros(n)
        for i in range(n):
            i_start, i_end = max(0, i - window // 2), min(n - 1, i + window // 2)
            if i_end - i_start < 2:
                i_start, i_end = max(0, i_end - 2), min(n - 1, i_start + 2)
            dx, dy = x_smooth[i_end] - x_smooth[i_start], y_smooth[i_end] - y_smooth[i_start]
            heading[i] = np.arctan2(dy, dx) if abs(dx) >= 1e-6 or abs(dy) >= 1e-6 else (heading[i-1] if i > 0 else 0.0)

        if n > 5:
            heading = np.arctan2(gaussian_filter1d(np.sin(heading), sigma=3.0),
                                 gaussian_filter1d(np.cos(heading), sigma=3.0))

        dx_world, dy_world, dv = np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0]), np.diff(vel, prepend=vel[0])
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        ego_forward = dx_world * cos_h + dy_world * sin_h
        ego_lateral = -dx_world * sin_h + dy_world * cos_h

        if np.median(ego_forward[1:]) < -1.0:
            heading += np.pi
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            ego_forward = dx_world * cos_h + dy_world * sin_h
            ego_lateral = -dx_world * sin_h + dy_world * cos_h

        ego_forward, ego_lateral, dv = np.clip(ego_forward, -15, 30), np.clip(ego_lateral, -5, 5), np.clip(dv, -10, 10)
        if n > 3:
            ego_forward, ego_lateral, dv = gaussian_filter1d(ego_forward, sigma=0.3), \
                                           gaussian_filter1d(ego_lateral, sigma=0.3), \
                                           gaussian_filter1d(dv, sigma=0.3)

        group['delta_forward'], group['delta_lateral'], group['delta_vel'], group['heading'] = \
            ego_forward, ego_lateral, dv, heading
        return group

    df = df.groupby('Vehicle_ID', group_keys=False).apply(calc_ego_deltas)
    return df

def remove_outliers(df):
    print("Removing outliers (IQR)...")
    for col in ['delta_forward', 'delta_lateral', 'delta_vel']:
        if col not in df.columns: continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR < 1e-10: continue
        df[col] = df[col].clip(Q1 - 3 * IQR, Q3 + 3 * IQR)
    return df

# ============================================================================
# 6. SEQUENCE CREATION
# ============================================================================

def create_sequences(df, max_sequences, stride=SEQUENCE_STRIDE):
    print("Creating sequences...")
    feature_cols = ['Local_X', 'Local_Y', 'v_Vel', 'v_Acc',
                    'steering_normalized', 'steering_rate', 'lateral_velocity',
                    'space_headway', 'rel_velocity', 'ttc']
    target_cols = ['delta_forward', 'delta_lateral', 'delta_vel']

    total_needed = SEQUENCE_LENGTH + LATENCY_OFFSET_FRAMES + PREDICTION_HORIZON
    seqs_X, seqs_y = [], []
    veh_count = 0

    for _, vdata in df.groupby('Vehicle_ID'):
        vdata = vdata.sort_values('Frame_ID').reset_index(drop=True)
        if len(vdata) < total_needed: continue
        veh_count += 1
        feat, tgt = vdata[feature_cols].values, vdata[target_cols].values
        for i in range(0, len(vdata) - total_needed + 1, stride):
            seqs_X.append(feat[i:i + SEQUENCE_LENGTH])
            y_start = i + SEQUENCE_LENGTH + LATENCY_OFFSET_FRAMES
            seqs_y.append(tgt[y_start:y_start + PREDICTION_HORIZON])
        if len(seqs_X) >= max_sequences: break

    return np.array(seqs_X, dtype=np.float32), np.array(seqs_y, dtype=np.float32), feature_cols, target_cols

def split_data(X, y):
    print("Splitting data...")
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    n_test, n_val = int(n * 0.2), int(n * 0.2)
    return (X[idx[n_test + n_val:]], X[idx[n_test:n_test + n_val]], X[idx[:n_test]],
            y[idx[n_test + n_val:]], y[idx[n_test:n_test + n_val]], y[idx[:n_test]])

def normalize(X_tr, X_va, X_te, y_tr, y_va, y_te):
    print("Normalizing...")
    sX, sy = StandardScaler(), StandardScaler()
    nfx, nfy = X_tr.shape[-1], y_tr.shape[-1]
    sX.fit(X_tr.reshape(-1, nfx))
    sy.fit(y_tr.reshape(-1, nfy))

    def tx(d): return sX.transform(d.reshape(-1, nfx)).reshape(d.shape)
    def ty(d): return sy.transform(d.reshape(-1, nfy)).reshape(d.shape)

    return tx(X_tr), tx(X_va), tx(X_te), ty(y_tr), ty(y_va), ty(y_te), sX, sy

# ============================================================================
# 7. TRAINING
# ============================================================================

def get_curriculum_horizon(epoch):
    return 15 if epoch < 20 else (25 if epoch < 40 else PREDICTION_HORIZON)

def train_model(model, train_loader, val_loader, epochs, patience, device='cpu'):
    print(f"\nTraining on {device}, {epochs} epochs, patience={patience}")
    model = model.to(device)
    criterion_full = MultiScaleLoss(PREDICTION_HORIZON, 1.5, 0.3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)

    best_val, best_epoch, no_improve = float('inf'), 0, 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        horizon = get_curriculum_horizon(epoch)
        criterion = MultiScaleLoss(horizon, 1.5, 0.3).to(device)
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
                vloss += criterion_full(model(bx), by).item()
        vloss /= len(val_loader)
        train_losses.append(tloss); val_losses.append(vloss)

        if vloss < best_val:
            best_val, best_epoch, no_improve = vloss, epoch, 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1
        if no_improve >= patience: break

    model.load_state_dict(torch.load('best_model.pth'))
    return train_losses, val_losses

# ============================================================================
# 8. EVALUATION & VISUALIZATION
# ============================================================================

def evaluate(model, test_loader, X_test_raw, y_test_raw, scaler_y, device='cpu'):
    print("\nEvaluating...")
    model.to(device).eval()
    preds, targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            preds.append(model(bx.to(device)).cpu().numpy())
            targets.append(by.numpy())
    y_pred_n, y_true_n = np.concatenate(preds), np.concatenate(targets)
    nf = y_pred_n.shape[-1]
    y_pred = scaler_y.inverse_transform(y_pred_n.reshape(-1, nf)).reshape(y_pred_n.shape)
    y_true = scaler_y.inverse_transform(y_true_n.reshape(-1, nf)).reshape(y_true_n.shape)
    return y_pred, y_true, [5, 10, 15, 20, 25, 30, 35], [], []

def plot_results(train_losses, val_losses, y_pred, y_true, horizons, h_fwd, h_lat):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.semilogy(train_losses, label='Train'); a1.semilogy(val_losses, label='Val'); a1.legend()
    a2.plot(train_losses, label='Train'); a2.plot(val_losses, label='Val'); a2.legend()
    plt.tight_layout(); plt.savefig('training_history.png'); plt.close()

# ============================================================================
# 10. PI MODEL EXPORT
# ============================================================================

def export_pi_model(X_train, y_train, X_val, y_val, sX, sy, f_cols, t_cols, device):
    print("\n EXPORTING PI MODEL...")
    nfx, nfy = X_train.shape[-1], y_train.shape[-1]
    def tx(d): return sX.transform(d.reshape(-1, nfx)).reshape(d.shape)
    def ty(d): return sy.transform(d.reshape(-1, nfy)).reshape(d.shape)
    tl = DataLoader(TrajectoryDataset(tx(X_train), ty(y_train)), batch_size=256, shuffle=True)
    vl = DataLoader(TrajectoryDataset(tx(X_val), ty(y_val)), batch_size=256, shuffle=False)
    pm = TrajectoryLSTM(input_size=nfx, hidden_size=64, num_layers=1, output_size=nfy, output_length=PREDICTION_HORIZON)
    train_model(pm, tl, vl, epochs=40, patience=12, device=device)
    torch.save(pm.state_dict(), 'pi_model.pth')
    joblib.dump({'input_size': nfx, 'hidden_size': 64, 'num_layers': 1, 'output_size': nfy, 'output_length': PREDICTION_HORIZON}, 'pi_model_config.pkl')
    joblib.dump(sX, 'pi_scaler_X.pkl'); joblib.dump(sy, 'pi_scaler_y.pkl')

# ============================================================================
# 11. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', default='quick'); args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode = MODES[args.mode]
    df = load_data(); df = handle_missing_data(df); df = compute_surrounding_features(df); df = compute_ego_centric_deltas(df); df = remove_outliers(df)
    X, y, f_cols, t_cols = create_sequences(df, mode['max_sequences']); del df; gc.collect()
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y)
    Xtn, Xvn, Xten, ytn, yvn, yten, sX, sy = normalize(X_tr, X_va, X_te, y_tr, y_va, y_te)
    np.savez_compressed('train_data.npz', X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va, X_test=X_te, y_test=y_te)
    joblib.dump(sX, 'scaler_X.pkl'); joblib.dump(sy, 'scaler_y.pkl')
    tr_l, va_l, te_l = DataLoader(TrajectoryDataset(Xtn, ytn), batch_size=mode['batch_size'], shuffle=True), \
                        DataLoader(TrajectoryDataset(Xvn, yvn), batch_size=mode['batch_size'], shuffle=False), \
                        DataLoader(TrajectoryDataset(Xten, yten), batch_size=mode['batch_size'], shuffle=False)
    model = TrajectoryLSTM(input_size=X_tr.shape[2], hidden_size=mode['hidden_size'], num_layers=2 if args.mode == 'full' else 1, output_size=y_tr.shape[2], output_length=PREDICTION_HORIZON)
    t_l, v_l = train_model(model, tr_l, va_l, epochs=mode['epochs'], patience=mode['patience'], device=device)
    y_p, y_t, h, hf, hl = evaluate(model, te_l, X_te, y_te, sy, device)
    plot_results(t_l, v_l, y_p, y_t, h, hf, hl)
    export_pi_model(X_tr, y_tr, X_va, y_va, sX, sy, f_cols, t_cols, device)

if __name__ == "__main__":
    main()