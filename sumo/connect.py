#!/usr/bin/env python3
import os
import sys
import joblib
import torch
import numpy as np
from collections import deque

# --- 1. PATH & IMPORT SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import TraCI (SUMO Interface)
try:
    import traci
except ImportError:
    if 'SUMO_HOME' in os.environ:
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
        import traci
    else:
        sys.exit("❌ Error: TraCI not found. Please set SUMO_HOME.")

# Import Model Class
try:
    from lstm_b import MultiOutputTrajectoryLSTM
    print("✅ PyTorch Model Architecture loaded from lstm_b.py")
except ImportError:
    sys.exit("❌ Error: Could not find MultiOutputTrajectoryLSTM in lstm_b.py")

# ---------- CONFIG ----------
SEQ_LEN = 15  
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
SCALER_X_PATH = os.path.join(BASE_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "scaler_y.pkl")
SUMO_CFG = os.path.join(BASE_DIR, "sumo.sumocfg")

# --- 2. PREDICTOR CLASS ---
class PyTorchPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vehicle_buffers = {}
        self.model = MultiOutputTrajectoryLSTM(input_size=7)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.scaler_X = joblib.load(SCALER_X_PATH)
        self.scaler_y = joblib.load(SCALER_Y_PATH)

    def add_data(self, v_id, features):
        if v_id not in self.vehicle_buffers:
            self.vehicle_buffers[v_id] = deque(maxlen=SEQ_LEN)
        self.vehicle_buffers[v_id].append(features)

    def predict_movement(self, v_id):
        buffer = np.array(self.vehicle_buffers[v_id])
        scaled_input = self.scaler_X.transform(buffer)
        input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds_scaled = self.model(input_tensor).cpu().numpy().reshape(-1, 3)
            preds = self.scaler_y.inverse_transform(preds_scaled)
        return np.sum(preds[:, 0]) 

def select_test_case():
    print("\n" + "="*40)
    print("🚗 SUMO COLLISION PREDICTION TEST SUITE")
    print("="*40)
    print("1. Sudden Stop (Rear-End Collision)")
    print("2. Junction Conflict (T-Bone Collision)")
    print("3. Lateral Swerve (Side-Swipe Collision)")
    print("4. Exit")
    choice = input("\nSelect a test case (1-4): ")
    cases = {"1": "case_stop.rou.xml", "2": "case_junction.rou.xml", "3": "case_lane.rou.xml"}
    if choice == "4": sys.exit("👋 Exiting simulation.")
    selected_file = cases.get(choice)
    return selected_file if selected_file else "case_stop.rou.xml"

# --- 3. MAIN SIMULATION ---
def main():
    route_filename = select_test_case()
    route_path = os.path.join(BASE_DIR, route_filename)
    if not os.path.exists(route_path):
        sys.exit(f"❌ Error: {route_filename} not found.")

    predictor = PyTorchPredictor()
    print(f"🔄 Loading {route_filename}...")
    traci.start(["sumo-gui", "-c", SUMO_CFG, "--route-files", route_path])
    print("✅ System Ready. Press 'Play' in SUMO GUI.")

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            active_vehicles = traci.vehicle.getIDList()
            
            # --- INITIALIZE VARIABLE TO PREVENT NAMEERROR ---
            dist_to_conflict = 999.0 

            # --- CASE 1: Sudden Stop ---
            if route_filename == "case_stop.rou.xml" and "lead_1" in active_vehicles and "ego" in active_vehicles:
                leader_info = traci.vehicle.getLeader("ego", 50.0)
                if leader_info and leader_info[1] < 25.0:
                    traci.vehicle.setSpeedMode("lead_1", 0) 
                    traci.vehicle.setSpeed("lead_1", 0) 

            # --- CASE 2: Junction Conflict (Wait and Rush) ---
# --- CASE 2: Junction Conflict (Wait and Rush) ---
# --- CASE 2: Junction Conflict (Tripwire Logic) ---
            elif route_filename == "case_junction.rou.xml" and "attacker" in active_vehicles and "ego" in active_vehicles:
                ego_pos = np.array(traci.vehicle.getPosition("ego"))
                att_pos = np.array(traci.vehicle.getPosition("attacker"))
                
                # Use the distance between the two cars as the trigger
                dist_to_conflict = np.linalg.norm(ego_pos - att_pos)

                # TRIGGER 1: If EV is still far away, make the Attacker WAIT
                # This stops the lead vehicle from "coming and going" too early
                if dist_to_conflict > 35.0:
                    traci.vehicle.setSpeedMode("attacker", 0)
                    traci.vehicle.setSpeed("attacker", 0.5) # Almost stopped
                
                # TRIGGER 2: Once EV is within 35m, the Attacker "RUSHES"
                else:
                    traci.vehicle.setSpeedMode("attacker", 0)
                    traci.vehicle.setSpeed("attacker", 25.0) # Full speed rush
                    print(f"🚀 {sim_time}s:Lead Vehicle Rushing!")

            # --- CASE 3: Abrupt Lane Swerve ---
            elif route_filename == "case_lane.rou.xml" and "swarver" in active_vehicles and "ego" in active_vehicles:
                ego_x = traci.vehicle.getPosition("ego")[0]
                swarver_x = traci.vehicle.getPosition("swarver")[0]
                gap = swarver_x - ego_x
                if 0 < gap < 12.0:
                    traci.vehicle.setLaneChangeMode("swarver", 0) 
                    traci.vehicle.changeLane("swarver", 1, 0) 
                    print(f"⚠️ {sim_time}s: SWARVER triggered abrupt lane change!")

            # --- EGO VEHICLE CONTROL ---
            for v_id in active_vehicles:
                if v_id == "ego":
                    traci.vehicle.setLaneChangeMode(v_id, 0)
                    v_speed, v_pos, v_acc = traci.vehicle.getSpeed(v_id), traci.vehicle.getPosition(v_id), traci.vehicle.getAcceleration(v_id)
                    leader = traci.vehicle.getLeader(v_id, 100.0) 
                    
                    features = [v_pos[0], v_pos[1], v_speed, v_acc, 0.0, 0.0, 0.0]
                    predictor.add_data(v_id, features)

                    if len(predictor.vehicle_buffers[v_id]) == SEQ_LEN:
                        move_pred = predictor.predict_movement(v_id)
                        
                        # Logic: Use traci leader (Case 1/3) OR distance check (Case 2)
                        if (leader and leader[1] < 30.0) or (route_filename == "case_junction.rou.xml" and dist_to_conflict < 15.0): 
                            if move_pred < 2.0 and v_speed > 3.0:
                                traci.vehicle.setSpeed(v_id, 0)
                                traci.vehicle.setColor(v_id, (0, 0, 255, 255)) 
                                print(f"🚨 {sim_time}s: Hazard! LSTM BRAKING.")
                            else:
                                traci.vehicle.setSpeed(v_id, -1)
                        else:
                            traci.vehicle.setSpeed(v_id, -1)
                            traci.vehicle.setColor(v_id, (255, 0, 0, 255))
                
                else:
                    traci.vehicle.setSpeedMode(v_id, 31) 
                    if v_id == "lead_1" or "attacker" in v_id or "swarver" in v_id:
                        traci.vehicle.setColor(v_id, (255, 255, 0, 255)) 
                    else:
                        traci.vehicle.setColor(v_id, (255, 255, 255, 255))
                        
    except traci.exceptions.FatalTraCIError:
        print("ℹ️ SUMO closed or connection lost.")
    finally:
        traci.close()
        print("🏁 Simulation finished.")

if __name__ == "__main__":
    main()