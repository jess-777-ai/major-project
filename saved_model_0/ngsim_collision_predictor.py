#!/usr/bin/env python3
"""
NGSIM Dataset-based V2V Collision Prediction Model
Uses full traffic trajectory data from NGSIM to train collision prediction
Supports both batch training and real-time testing
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
import random
from collections import deque  # <-- Import deque

class NGSIMCollisionPredictor:
    """
    Collision prediction model trained on NGSIM dataset
    Optimized for high recall, precision, F1-score, and >80% accuracy
    """

    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.vehicle_buffers = {}
        self.feature_columns = [
            'distance_to_lead', 'relative_velocity', 'own_velocity', 'own_acceleration',
            'lead_velocity', 'lead_acceleration', 'lateral_distance',
            'ego_x', 'ego_y', 'lead_x', 'lead_y'
        ]
        
        # Column mapping for NGSIM dataset
        self.column_mapping = {
            'vehicle_id': 'Vehicle_ID',
            'frame': 'Frame_ID', 
            'x': 'Local_X',
            'y': 'Local_Y',
            'velocity': 'v_Vel',
            'acceleration': 'v_Acc',
            'lane': 'Lane_ID',
            'preceding': 'Preceding',
            'space_headway': 'Space_Headway',
            'time_headway': 'Time_Headway'
        }

    ###########################
    # Data Loading & Processing
    ###########################

    def load_ngsim_data(self, data_path, chunk_size=50000, max_vehicles_per_chunk=1000, max_scenarios=5000):
        """Load limited NGSIM data to speed up preprocessing"""
        print("🔄 Loading limited NGSIM dataset...")
        scenarios = []

        try:
            chunk_iter = pd.read_csv(data_path, chunksize=chunk_size, low_memory=False)
            total_chunks = 0

            for chunk in chunk_iter:
                if len(scenarios) >= max_scenarios:
                    print(f"✅ Reached scenario limit: {max_scenarios}, stopping early.")
                    break

                total_chunks += 1
                print(f"📦 Processing chunk {total_chunks}...")

                chunk_scenarios = self.process_ngsim_data(chunk, max_vehicles=max_vehicles_per_chunk)
                scenarios.extend(chunk_scenarios)

                print(f"✅ Total collected: {len(scenarios)}")

            return scenarios[:max_scenarios]

        except Exception as e:
            print(f"❌ Error loading NGSIM data: {e}")
            return []


    def process_ngsim_data(self, ngsim_data, max_vehicles=1000):
        """Process a chunk of NGSIM data to extract vehicle sequences (Optimized)"""
        scenarios = []
        
        # --- OPTIMIZATION: Pre-group by Frame_ID for fast lookups ---
        print("    Optimizing chunk: grouping by Frame_ID for faster lookups...")
        try:
            frame_col = self.column_mapping['frame']
            frame_groups = ngsim_data.groupby(frame_col)
        except Exception as e:
            print(f"    Could not group by frame, will be slow. Error: {e}")
            frame_groups = None
        # --- END OPTIMIZATION ---

        vehicle_groups = ngsim_data.groupby(self.column_mapping['vehicle_id'])
        processed_count = 0
        for vehicle_id, vehicle_data in vehicle_groups:
            if processed_count >= max_vehicles:
                break
            if len(vehicle_data) < self.sequence_length + 5:
                continue

            vehicle_data = vehicle_data.sort_values(self.column_mapping['frame']).reset_index(drop=True)
            
            # Pass the pre-grouped frames for fast processing
            vehicle_scenarios = self.extract_ngsim_scenarios(vehicle_data, frame_groups, self.column_mapping)
            
            scenarios.extend(vehicle_scenarios)
            processed_count += 1
            if processed_count % 100 == 0:
                 print(f"    Processed {processed_count}/{max_vehicles} vehicles...")

        return scenarios

    def extract_ngsim_scenarios(self, ego_vehicle, frame_groups, column_mapping):
        """Extract collision scenarios from NGSIM vehicle trajectory"""
        scenarios = []

        x_col = column_mapping.get('x', 'x')
        y_col = column_mapping.get('y', 'y')
        vel_col = column_mapping.get('velocity', 'velocity')
        acc_col = column_mapping.get('acceleration', 'acceleration')
        frame_col = column_mapping.get('frame', 'frame')
        lane_col = column_mapping.get('lane', 'lane')

        for i in range(len(ego_vehicle) - self.sequence_length):
            sequence_data = ego_vehicle.iloc[i:i + self.sequence_length].copy()
            sequence_features = []

            for _, row in sequence_data.iterrows():
                try:
                    ego_x = float(row[x_col]) if x_col in row else 0.0
                    ego_y = float(row[y_col]) if y_col in row else 0.0
                    ego_vel = float(row[vel_col]) if vel_col in row else 20.0
                    ego_acc = float(row[acc_col]) if acc_col in row else 0.0
                    frame = row[frame_col] if frame_col in row else 0
                    ego_lane = row[lane_col] if lane_col in row else 1

                    # --- OPTIMIZATION: Use pre-grouped frame data ---
                    try:
                        current_frame_data = frame_groups.get_group(frame) if frame_groups else None
                    except KeyError:
                        current_frame_data = None # No other cars in this frame
                    
                    lead_vehicle = self.find_ngsim_lead_vehicle(
                        current_frame_data, # Pass the *small, pre-filtered* DataFrame
                        frame, ego_x, ego_y, ego_lane, column_mapping
                    )
                    # --- END OPTIMIZATION ---

                    if lead_vehicle is not None:
                        lead_x = float(lead_vehicle[x_col])
                        lead_y = float(lead_vehicle[y_col])
                        lead_vel = float(lead_vehicle[vel_col])
                        lead_acc = float(lead_vehicle[acc_col])

                        distance = np.sqrt((lead_x - ego_x)**2 + (lead_y - ego_y)**2)
                        relative_velocity = ego_vel - lead_vel

                        features = [
                            distance, relative_velocity, ego_vel, ego_acc,
                            lead_vel, lead_acc, abs(ego_y - lead_y),
                            ego_x, ego_y, lead_x, lead_y
                        ]
                    else:
                        features = [
                            100.0, 0.0, ego_vel, ego_acc,
                            ego_vel, 0.0, 0.0,
                            ego_x, ego_y, ego_x + 100, ego_y
                        ]

                    sequence_features.append(features)

                except Exception:
                    continue

            if len(sequence_features) == self.sequence_length:
                collision_label = self.analyze_collision_trajectory(sequence_features)
                scenarios.append({
                    'sequence': np.array(sequence_features),
                    'collision_risk': collision_label
                })

        return scenarios

    def analyze_collision_trajectory(self, sequence_features):
        """Analyze trajectory to determine collision risk"""
        sequence_array = np.array(sequence_features)
        distances = sequence_array[:, 0]
        rel_velocities = sequence_array[:, 1]
        ego_velocities = sequence_array[:, 2]
        ego_accelerations = sequence_array[:, 3]

        collision_risk = 0.0

        if len(distances) >= 10:
            distance_trend = np.polyfit(range(len(distances)), distances, 1)[0]
            avg_rel_velocity = np.mean(rel_velocities[-5:])
            min_distance = np.min(distances)

            if distance_trend < -2 and avg_rel_velocity > 3 and min_distance < 30:
                collision_risk = max(collision_risk, 0.85)
            elif distance_trend < -1 and avg_rel_velocity > 2 and min_distance < 50:
                collision_risk = max(collision_risk, 0.65)
            elif min_distance < 25 and avg_rel_velocity > 1:
                collision_risk = max(collision_risk, 0.45)

        hard_braking = np.any(ego_accelerations < -3.0)
        if hard_braking and np.min(distances) < 40:
            collision_risk = max(collision_risk, 0.70)

        high_speed_frames = np.sum((ego_velocities > 25) & (distances < 50))
        if high_speed_frames > len(sequence_features) * 0.5:
            collision_risk = max(collision_risk, 0.40)

        noise = np.random.normal(0, 0.05)
        final_risk = np.clip(collision_risk + noise, 0, 1)

        return 1.0 if final_risk > 0.5 else 0.0

    def find_ngsim_lead_vehicle(self, frame_data, frame, ego_x, ego_y, ego_lane, column_mapping):
        """Find lead vehicle in NGSIM data (Optimized)"""
        
        # If no data was passed (no cars in frame), just return
        if frame_data is None or frame_data.empty:
            return None

        x_col = column_mapping.get('x', 'x')
        y_col = column_mapping.get('y', 'y')

        # We NO LONGER filter by frame, 'frame_data' is *already* filtered
        potential_leads = frame_data[(frame_data[x_col] > ego_x) & (abs(frame_data[y_col] - ego_y) < 4.0)]

        if len(potential_leads) == 0:
            return None

        distances = np.sqrt((potential_leads[x_col] - ego_x)**2 + (potential_leads[y_col] - ego_y)**2)
        closest_idx = distances.idxmin()
        return potential_leads.loc[closest_idx]

    ###########################
    # Model Creation & Training
    ###########################

    def create_model(self):
        """Create LSTM model"""
        from tensorflow.keras.regularizers import l2

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, len(self.feature_columns)), kernel_regularizer=l2(0.001)),
            Dropout(0.4),
            BatchNormalization(),

            LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            BatchNormalization(),

            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid', name='collision_probability')
        ])

        def f1_score(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            def recall_m(y_true, y_pred):
                tp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
                pos = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
                return tp / (pos + tf.keras.backend.epsilon())

            def precision_m(y_true, y_pred):
                tp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
                pred_pos = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
                return tp / (pred_pos + tf.keras.backend.epsilon())

            precision = precision_m(y_true, y_pred)
            recall = recall_m(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))

        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall', f1_score])
        self.model = model
        return model

    def prepare_training_data(self, scenarios):
        """Prepare balanced training data"""
        X, y = [], []
        collision_scenarios = []
        safe_scenarios = []

        for scenario in scenarios:
            if scenario['collision_risk'] > 0.4:
                collision_scenarios.append((scenario['sequence'], 1))
            else:
                safe_scenarios.append((scenario['sequence'], 0))

        # Balance classes
        if len(collision_scenarios) > 0 and len(safe_scenarios) > len(collision_scenarios) * 2:
            collision_multiplier = min(3, len(safe_scenarios) // len(collision_scenarios))
            collision_scenarios = collision_scenarios * collision_multiplier
            random.shuffle(safe_scenarios)
            safe_scenarios = safe_scenarios[:len(collision_scenarios) * 2]

        all_scenarios = collision_scenarios + safe_scenarios
        random.shuffle(all_scenarios)

        for sequence, label in all_scenarios:
            X.append(sequence)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        return X, y

    def train_model(self, data_path, save_path='ngsim_collision_model.h5', max_scenarios=5000):
        """Train using limited preprocessed data"""
        scenarios = self.load_ngsim_data(data_path, max_scenarios=max_scenarios)

        if len(scenarios) == 0:
            print("❌ No scenarios extracted.")
            return None

        print(f"🎯 Using {len(scenarios)} scenarios for training...")
        X, y = self.prepare_training_data(scenarios)
        
        if len(X) == 0 or len(y) == 0:
            print("❌ No data after balancing (X, y are empty).")
            return None

        # Normalize features
        X_flat = X.reshape(-1, X.shape[-1])
        self.scaler.fit(X_flat)
        X_normalized = self.scaler.transform(X_flat).reshape(X.shape)

        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y, test_size=0.2, random_state=42, stratify=y
        )

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        self.create_model()

        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

        callbacks = [
            EarlyStopping(monitor='val_f1_score', mode='max', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_f1_score', mode='max', factor=0.5, patience=8, min_lr=1e-6),
            ModelCheckpoint(save_path.replace('.h5', '_best.h5'), monitor='val_f1_score',
                            mode='max', save_best_only=True)
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=64,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )

        # Save model and scaler
        self.model.save(save_path)
        with open(save_path.replace('.h5', '_scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        return history

    ###########################
    # Real-time Prediction Methods
    ###########################

    def add_timestep_features(self, veh_id, features):
        """
        Adds a new, complete 11-feature vector to the vehicle's buffer.
        """
        if veh_id not in self.vehicle_buffers:
            # Use a deque for efficient pop/append
            self.vehicle_buffers[veh_id] = deque(maxlen=self.sequence_length)
        
        # Ensure features is a list of floats
        processed_features = [float(f) for f in features]
        
        # Add the new feature vector to the buffer
        self.vehicle_buffers[veh_id].append(processed_features)
        
    def predict_collision_risk(self, vehicle_id):
        """Predict collision risk for a vehicle using last sequence_length data points"""
        if vehicle_id not in self.vehicle_buffers or len(self.vehicle_buffers[vehicle_id]) < self.sequence_length:
            return 0.0

        X = np.array(self.vehicle_buffers[vehicle_id]).reshape(1, self.sequence_length, len(self.feature_columns))
        
        # Reshape for scaling, transform, then reshape back
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled_flat = self.scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(X.shape)

        return float(self.model.predict(X_scaled, verbose=0)[0,0])

    def get_risk_level(self, risk):
        """Convert probability to risk category"""
        if risk > 0.7:
            return "High"
        elif risk > 0.4:
            return "Medium"
        else:
            return "Low"


if __name__ == "__main__":
    predictor = NGSIMCollisionPredictor(sequence_length=20)
    # !!! REPLACE THIS WITH THE ACTUAL PATH TO YOUR CSV FILE !!!
    ngsim_data_path = "Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20251027.csv"

    # ✅ Train faster by limiting scenarios extracted (change 3000 -> as needed)
    predictor.train_model(
        data_path=ngsim_data_path,
        save_path='ngsim_collision_model_fast.h5',
        max_scenarios=3000  # <-- Main change
    )

    # ✅ Optional: Real-time inference after training
    if predictor.model is not None:
        vehicle_id = 'vehicle_1'
        print("\n--- Running Test Prediction ---")
        for step in range(25):
            
            # --- BUG FIX: Create a full 11-feature vector for the test ---
            distance = max(1.0, 120 - step * 4)
            rel_vel = 20.0
            ego_vel = 30.0
            ego_acc = -1.5
            lead_vel = 10.0
            lead_acc = 0.0
            
            test_features = [
                distance, rel_vel, ego_vel, ego_acc,
                lead_vel, lead_acc, 0.0, # lateral_dist
                50.0, 5.0, # ego_x, ego_y
                50.0 + distance, 5.0 # lead_x, lead_y
            ]
            
            # Call the *correct* function
            predictor.add_timestep_features(vehicle_id, test_features)
            # --- END OF BUG FIX ---

            if step >= 19: # Wait for buffer to fill
                risk = predictor.predict_collision_risk(vehicle_id)
                level = predictor.get_risk_level(risk)
                print(f"Step {step}: Distance={distance:.1f}, Risk={risk:.3f} ({level})")