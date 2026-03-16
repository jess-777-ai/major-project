import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d
import warnings
import joblib

warnings.filterwarnings('ignore')

# --- 1. Memory Efficient Dataset ---
class LazyTrajectoryDataset(Dataset):
    """
    Takes flat 2D arrays and slices them into 3D windows on-the-fly.
    This prevents the 4D ValueError and saves massive amounts of RAM.
    """
    def __init__(self, features, targets, seq_len=15, pred_hor=15):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = seq_len
        self.pred_hor = pred_hor
        
        # Total number of sliding windows possible
        self.num_samples = len(self.features) - seq_len - pred_hor

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Slice X: (seq_len, num_features) -> e.g., (15, 7)
        x = self.features[idx : idx + self.seq_len]
        
        # Slice Y: (pred_hor, num_targets) -> e.g., (15, 3)
        y = self.targets[idx + self.seq_len : idx + self.seq_len + self.pred_hor]
        
        return x, y

# --- 2. Multi-Output LSTM Model ---
class MultiOutputTrajectoryLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, output_length=15):
        super(MultiOutputTrajectoryLSTM, self).__init__()
        self.output_length = output_length
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Three separate prediction heads
        self.delta_x_head = nn.Linear(64, output_length)
        self.delta_y_head = nn.Linear(64, output_length)
        self.delta_vel_head = nn.Linear(64, output_length)

    def forward(self, x):
        # x shape: (batch_size, 15, 7)
        _, (h_n, _) = self.lstm(x)
        
        # Use the hidden state from the last LSTM layer
        last_hidden = h_n[-1] 
        shared = self.shared_fc(last_hidden)

        dx = self.delta_x_head(shared)
        dy = self.delta_y_head(shared)
        dv = self.delta_vel_head(shared)

        # Stack into (batch_size, 15, 3)
        predictions = torch.stack([dx, dy, dv], dim=2)
        return predictions

# --- 3. Preprocessing Functions ---
def load_and_preprocess_data(sample_size=None):
    print("Loading NGSIM dataset...")
    # Update filename to match yours
    df = pd.read_csv('Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20251027.csv', low_memory=False)

    if 'Direction' in df.columns:
        df = df[df['Direction'] == df['Direction'].value_counts().index[0]]

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['Location']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

    # Filter for long enough trajectories
    v_counts = df['Vehicle_ID'].value_counts()
    df = df[df['Vehicle_ID'].isin(v_counts[v_counts >= 100].index)]
    return df.sort_values(['Vehicle_ID', 'Frame_ID'])

def compute_features(df):
    """Calculates deltas and enhanced steering features"""
    def process_group(group):
        group = group.sort_values('Frame_ID')
        # Steering/Lateral logic
        dx = np.diff(group['Local_X'].values, prepend=group['Local_X'].iloc[0])
        dy = np.diff(group['Local_Y'].values, prepend=group['Local_Y'].iloc[0])
        dx[dx == 0] = 1e-6
        
        group['steering_normalized'] = np.clip(np.degrees(np.arctan2(dy, dx)), -90, 90) / 90.0
        group['lateral_velocity'] = np.clip(dy, -10, 10) / 10.0
        group['steering_rate'] = np.diff(group['steering_normalized'], prepend=group['steering_normalized'].iloc[0])
        
        # Targets (Deltas)
        group['delta_X'] = group['Local_X'].diff().fillna(0).clip(-100, 100)
        group['delta_Y'] = group['Local_Y'].diff().fillna(0).clip(-50, 50)
        group['delta_vel'] = group['v_Vel'].diff().fillna(0).clip(-50, 50)
        
        return group

    return df.groupby('Vehicle_ID', group_keys=False).apply(process_group)

# --- 4. Training Loop ---
def train_model(model, train_loader, val_loader, device, epochs=50):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                val_loss += criterion(model(batch_X), batch_y).item()
        
        avg_v = val_loss/len(val_loader)
        if avg_v < best_loss:
            best_loss = avg_v
            torch.save(model.state_dict(), 'best_model.pth')
            
        print(f"Epoch {epoch+1}: Val Loss {avg_v:.6f}")

# --- 5. Main Execution ---
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Process
    df = load_and_preprocess_data(sample_size=300000)
    df = compute_features(df)
    df = df.dropna()

    # Define Columns
    feature_cols = ['Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'steering_normalized', 'steering_rate', 'lateral_velocity']
    target_cols = ['delta_X', 'delta_Y', 'delta_vel']

    # Create 2D arrays
    X_data = df[feature_cols].values
    y_data = df[target_cols].values

    # Split (Split rows, not windows, to save memory)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # Normalize
    sc_X, sc_y = StandardScaler(), StandardScaler()
    X_train_n = sc_X.fit_transform(X_train)
    X_val_n = sc_X.transform(X_val)
    y_train_n = sc_y.fit_transform(y_train)
    y_val_n = sc_y.transform(y_val)

    # Dataset & Loader (Slices into 3D here)
    train_ds = LazyTrajectoryDataset(X_train_n, y_train_n)
    val_ds = LazyTrajectoryDataset(X_val_n, y_val_n)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # Model
    model = MultiOutputTrajectoryLSTM(input_size=len(feature_cols))
    
    # Run
    train_model(model, train_loader, val_loader, device)
    
    # Save Scalers
    joblib.dump(sc_X, 'scaler_X.pkl')
    joblib.dump(sc_y, 'scaler_y.pkl')
    print("Mission Success: Model Trained & Saved.")

if __name__ == "__main__":
    main()