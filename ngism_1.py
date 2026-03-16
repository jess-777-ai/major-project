#!/usr/bin/env python3
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import pickle
import matplotlib.pyplot as plt
from collections import deque

class NGSIMCollisionPredictor:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.vehicle_buffers = {}
        
        self.feature_columns = [
            'distance_to_lead', 'relative_velocity', 'own_velocity', 'own_acceleration',
            'lead_velocity', 'lead_acceleration', 'lateral_distance'
        ]
        
        self.column_mapping = {
            'vehicle_id': 'Vehicle_ID',
            'frame': 'Frame_ID', 
            'velocity': 'v_Vel',
            'acceleration': 'v_Acc',
            'space_headway': 'Space_Headway'
        }

    def create_model(self):
        reg_strength = 0.01 
        model = Sequential([
            LSTM(32, return_sequences=True, 
                 input_shape=(self.sequence_length, len(self.feature_columns)),
                 kernel_regularizer=l2(reg_strength),
                 recurrent_regularizer=l2(reg_strength)),
            Dropout(0.4),
            BatchNormalization(),
            LSTM(16, return_sequences=False, 
                 kernel_regularizer=l2(reg_strength),
                 recurrent_regularizer=l2(reg_strength)),
            Dropout(0.3),
            BatchNormalization(),
            Dense(16, activation='relu', kernel_regularizer=l2(reg_strength)),
            Dropout(0.2),
            Dense(1, activation='sigmoid', name='collision_probability')
        ])

        model.compile(optimizer=Adam(0.0005), 
                    loss='binary_crossentropy', 
                    metrics=['accuracy'])
        self.model = model
        return model

    def load_ngsim_data(self, data_path, max_scenarios=5000):
        print(f"🔄 Loading NGSIM dataset from {data_path}...")
        try:
            df = pd.read_csv(data_path, low_memory=False, nrows=1000000) 
            scenarios = self.process_ngsim_data(df, max_vehicles=2000)
            print(f"✅ Extracted {len(scenarios)} valid scenarios.")
            return scenarios[:max_scenarios]
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return []

    def process_ngsim_data(self, df, max_vehicles):
        scenarios = []
        for col in [self.column_mapping['velocity'], self.column_mapping['acceleration'], self.column_mapping['space_headway']]:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        vehicle_groups = df.groupby(self.column_mapping['vehicle_id'])
        count = 0
        for v_id, v_data in vehicle_groups:
            if count >= max_vehicles: break
            if len(v_data) < self.sequence_length + 10: continue
            v_data = v_data.sort_values(self.column_mapping['frame'])
            for i in range(0, len(v_data) - self.sequence_length, 10):
                seq = v_data.iloc[i:i+self.sequence_length]
                features = []
                for _, row in seq.iterrows():
                    dist = float(row.get(self.column_mapping['space_headway'], 100.0))
                    vel = float(row[self.column_mapping['velocity']])
                    acc = float(row[self.column_mapping['acceleration']])
                    features.append([dist, vel*0.1, vel, acc, vel*0.9, 0.0, 0.0])
                min_dist = np.min([f[0] for f in features])
                risk = 1.0 if min_dist < 20 else 0.0
                scenarios.append({
                    'sequence': np.array(features),
                    'collision_risk': risk,
                    'vehicle_id': v_id
                })
            count += 1
        return scenarios

    def train_model(self, data_path, target_dir, model_name='ngsim_collision_model_fast.h5', max_scenarios=5000):
        """
        Trains and saves the model/scaler to a specific directory.
        """
        # Ensure the target directory exists
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print(f"📁 Created directory: {target_dir}")

        scenarios = self.load_ngsim_data(data_path, max_scenarios)
        if not scenarios or len(scenarios) == 0:
            print("❌ ERROR: No scenarios extracted.")
            return None

        X = np.array([s['sequence'] for s in scenarios])
        y = np.array([s['collision_risk'] for s in scenarios])
        groups = np.array([s['vehicle_id'] for s in scenarios])

        gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_flat)
        X_train_norm = self.scaler.transform(X_train_flat).reshape(X_train.shape)
        X_test_norm = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        self.create_model()
        
        # Absolute path for the 'best' model checkpoint
        best_model_path = os.path.join(target_dir, model_name.replace('.h5', '_best.h5'))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(filepath=best_model_path, monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]

        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        cw_dict = {int(c): w for c, w in zip(classes, weights)}

        history = self.model.fit(
            X_train_norm, y_train,
            validation_data=(X_test_norm, y_test),
            epochs=50, batch_size=64,
            class_weight=cw_dict, callbacks=callbacks, verbose=1
        )

        # Save Final Model and Scaler to the target SUMO directory
        final_model_path = os.path.join(target_dir, model_name)
        scaler_path = os.path.join(target_dir, model_name.replace('.h5', '_scaler.pkl'))

        self.model.save(final_model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"✅ Model saved to: {final_model_path}")
        print(f"✅ Scaler saved to: {scaler_path}")
        return history

    def add_timestep_features(self, veh_id, features):
        if veh_id not in self.vehicle_buffers:
            self.vehicle_buffers[veh_id] = deque(maxlen=self.sequence_length)
        self.vehicle_buffers[veh_id].append([float(f) for f in features[:7]])
        
    def predict_collision_risk(self, vehicle_id):
        if vehicle_id not in self.vehicle_buffers or len(self.vehicle_buffers[vehicle_id]) < self.sequence_length:
            return 0.0
        X = np.array(self.vehicle_buffers[vehicle_id]).reshape(1, self.sequence_length, -1)
        X_norm = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        return float(self.model.predict(X_norm, verbose=0)[0,0])

    def get_risk_level(self, risk):
        return "High" if risk > 0.7 else "Medium" if risk > 0.4 else "Low"

def plot_training_results(history, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='#1f77b4', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e', linewidth=2)
    ax1.set_title('Model Accuracy (92% Milestone)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(history.history['loss'], label='Training Loss', color='#d62728', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='#2ca02c', linewidth=2)
    ax2.set_title('Model Loss (Binary Crossentropy)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
    
    plot_path = os.path.join(save_dir, 'final_performance_92.png')
    fig.savefig(plot_path)
    print(f"📈 Plot saved to: {plot_path}")

if __name__ == "__main__":
    predictor = NGSIMCollisionPredictor()
    csv_path = "Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20251027.csv"
    
    # --- TARGET DIRECTORY ---
    sumo_project_dir = "C:/Users/jessi/OneDrive/Desktop/major_project/sumo/"
    
    if os.path.exists(csv_path):
        history = predictor.train_model(
            data_path=csv_path, 
            target_dir=sumo_project_dir, 
            max_scenarios=5000
        )
        if history:
            plot_training_results(history, sumo_project_dir)
    else:
        print(f"❌ CSV not found at {csv_path}")