import os
import pickle
#from sumo.ngsim_collision_predictor import NGSIMCollisionPredictor

def main():
    # Paths
    model_path = 'ngsim_collision_model_fast_best.h5'  # using best model
    scaler_path = 'ngsim_collision_model_fast_scaler.pkl'
    ngsim_data_path = "Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20251027.csv"

    # Initialize predictor
    predictor = NGSIMCollisionPredictor(sequence_length=20)

    # Check if model exists
    if os.path.exists(model_path) and os.path.exists(model_path.replace('.h5', '_scaler.pkl')):
        print("🔄 Loading existing NGSIM model...")
        try:
            import tensorflow as tf
            # Custom F1 metric
            def f1_score(y_true, y_pred):
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

            predictor.model = tf.keras.models.load_model(model_path, custom_objects={'f1_score': f1_score})
            with open(model_path.replace('.h5', '_scaler.pkl'), 'rb') as f:
                predictor.scaler = pickle.load(f)
            print(f"✅ NGSIM model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return
    else:
        # Train model
        if not os.path.exists(ngsim_data_path):
            print(f"❌ NGSIM data file not found: {ngsim_data_path}")
            return
        print("🚀 Training NGSIM model...")
        predictor.train_model(ngsim_data_path, save_path=model_path)
        print(f"✅ Model trained and saved to {model_path}")

    # Real-time simulation
    print("\n🧪 Simulating vehicle collision scenario...")
    vehicle_id = 'test_vehicle'
    for i in range(25):
        distance = max(1.0, 120 - i * 4)
        rel_velocity = 20.0
        predictor.update_vehicle_data(
            vehicle_id=vehicle_id,
            distance_to_lead=distance,
            relative_velocity=rel_velocity,
            own_velocity=30.0,
            own_acceleration=-1.5,
            lead_velocity=10.0,
            lead_acceleration=0.0
        )
        if i >= 19:
            risk = predictor.predict_collision_risk(vehicle_id)
            level = predictor.get_risk_level(risk)
            print(f"Step {i}: Distance={distance:.1f}m, Risk={risk:.3f} ({level})")

    print("\n✅ Simulation completed successfully!")


if __name__ == "__main__":
    main()