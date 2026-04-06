#!/usr/bin/env python3
import os
import sys
import joblib
import torch
import csv
import numpy as np
from collections import deque

# --- 1. SYSTEM SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
from lstm_connect import TrajectoryLSTM

# ---------- CONFIG ----------
SEQ_LEN, PRED_LEN = 15, 35
INPUT_SIZE, HIDDEN_SIZE = 10, 64
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
SCALER_X_PATH = os.path.join(BASE_DIR, "scaler_X_v2.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "scaler_y_v2.pkl")
SUMO_CFG = os.path.join(BASE_DIR, "sumo.sumocfg")                                     
LOG_FILE = os.path.join(BASE_DIR, "lstm_live_output.csv")

# Trigger Constants
CASE1_START_DIST = 200.0  
CASE3_START_DIST = 150.0

prev_angles = {}
input_ema = {} 

# --- 2. PREDICTOR CLASS (V2 PHYSICS ADDED) ---
class TrajectoryPredictor:
    def __init__(self):
        self.device = torch.device('cpu')
        self.vehicle_buffers = {}
        self.prediction_buffers = {} 
        self.model = TrajectoryLSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=3, output_length=PRED_LEN)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.to(self.device).eval()
        
        raw_x = joblib.load(SCALER_X_PATH)
        self.scaler_X = raw_x[0] if isinstance(raw_x, list) else raw_x
        
        # Load the list of 3 separate channel scalers
        self.y_scalers = joblib.load(SCALER_Y_PATH)

    def update_buffer(self, v_id, features):
        if v_id not in self.vehicle_buffers:
            self.vehicle_buffers[v_id] = deque(maxlen=SEQ_LEN)
        feat_array = np.array(features)
        if v_id not in input_ema: input_ema[v_id] = feat_array
        else: input_ema[v_id] = 0.8 * input_ema[v_id] + 0.2 * feat_array
        self.vehicle_buffers[v_id].append(input_ema[v_id])

    def hallucinate_trajectory(self, v_id, current_pos, current_vel):
        buffer = np.array(self.vehicle_buffers[v_id])
        scaled_input = self.scaler_X.transform(buffer)
        input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds_scaled = self.model(input_tensor).cpu().numpy().reshape(PRED_LEN, 3)
            
            # Per-channel Inverse Transform
            deltas = np.zeros_like(preds_scaled)
            for ch in range(3):
                deltas[:, ch] = self.y_scalers[ch].inverse_transform(
                    preds_scaled[:, ch].reshape(-1, 1)
                ).flatten()
            
        # Apply the Residual Connection (Velocity * 0.1s dt)
        vel_baseline = current_vel * 0.1 
        deltas[:, 0] = deltas[:, 0] + vel_baseline
        
        deltas[:, 1] = np.where(np.abs(deltas[:, 1]) < 0.03, 0, deltas[:, 1])
        sync_offset = 0.96 

        if v_id in self.prediction_buffers:
            deltas = (deltas * 0.7) + (self.prediction_buffers[v_id][-1] * 0.3)
        self.prediction_buffers.setdefault(v_id, deque(maxlen=3)).append(deltas)
        
        path = []
        curr_x, curr_y = current_pos
        for df, dl, _ in deltas:
            curr_x += (df * sync_offset)
            curr_y += dl 
            path.append((curr_x, curr_y))
        return np.array(path)

# --- 3. KINEMATIC COLLISION CHECKER ---
def check_collision_risk(v_id, ego_path, obstacle_id):
    try:
        obs_pos = np.array(traci.vehicle.getPosition(obstacle_id))
        obs_speed = traci.vehicle.getSpeed(obstacle_id)
        obs_angle = traci.vehicle.getAngle(obstacle_id)
        obs_accel = traci.vehicle.getAcceleration(obstacle_id) 
        
        rad = np.radians(90 - obs_angle)
        vx, vy = obs_speed * np.cos(rad), obs_speed * np.sin(rad)
        ax, ay = obs_accel * np.cos(rad), obs_accel * np.sin(rad)

        for t_step, point in enumerate(ego_path):
            time_offset = (t_step + 1) * 0.1
            
            if obs_speed + (obs_accel * time_offset) < 0:
                time_offset = abs(obs_speed / obs_accel) if obs_accel < 0 else time_offset

            dx_proj = (vx * time_offset) + (0.5 * ax * time_offset**2)
            dy_proj = (vy * time_offset) + (0.5 * ay * time_offset**2)
            future_obs_pos = obs_pos + np.array([dx_proj, dy_proj])
            
            dx = abs(point[0] - future_obs_pos[0]) 
            dy = abs(point[1] - future_obs_pos[1]) 
            long_threshold = max(3.0, 6.0 - (t_step * 0.1)) 
            lat_threshold = 2.2 

            if dx < long_threshold and dy < lat_threshold:
                return True
        return False
    except:
        return False

def init_csv():
    header = ['Sim_Time', 'Ego_X', 'Ego_Y', 'Speed', 'Acc', 'Steer', 'SteerRate', 
              'LatVel', 'Headway', 'RelVel', 'TTC', 'Pred_X_Next', 'Pred_Y_Next', 'Hazard_Status']
    with open(LOG_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(header)

# --- 4. MAIN SIMULATION LOOP ---
def main():
    print("\n1. Sudden Stop | 2. Junction | 3. Adjacent Swerve")
    choice = input("Select Case: ")
    cases = {"1": "case_stop.rou.xml", "2": "case_junction.rou.xml", "3": "case_lane.rou.xml"}
    route_filename = cases.get(choice, "case_stop.rou.xml")

    init_csv()
    predictor = TrajectoryPredictor()
    traci.start(["sumo-gui", "-c", SUMO_CFG, "--route-files", os.path.join(BASE_DIR, route_filename)])

    ov_stop_time = -1 
    ego_initialized = False 
    hazard_cooldown = 0   

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        sim_time, active_vehicles = traci.simulation.getTime(), traci.vehicle.getIDList()

        # --- 🟢 INITIAL SETUP 🟢 ---
        if "ego" in active_vehicles and not ego_initialized:
            traci.vehicle.setMinGap("ego", 0.0) 
            traci.vehicle.setTau("ego", 0.1) 
            traci.vehicle.setSpeedFactor("ego", 1.2)
            traci.vehicle.setSpeedMode("ego", 0) 
            traci.vehicle.setSpeed("ego", 18.0)
            traci.vehicle.setLaneChangeMode("ego", 0)
            ego_initialized = True
            
        if "lead_1" in active_vehicles:
            traci.vehicle.setSpeedMode("lead_1", 31)
            traci.vehicle.setSpeed("lead_1", 10.0)
            
        if "swarver" in active_vehicles:
            traci.vehicle.setSpeedMode("swarver", 31)
            traci.vehicle.setSpeed("swarver", 15.0)

        # --- 🧠 IS THE LSTM ONLINE? ---
        lstm_ready = False
        if "ego" in active_vehicles and "ego" in predictor.vehicle_buffers:
            if len(predictor.vehicle_buffers["ego"]) == SEQ_LEN:
                lstm_ready = True

        # --- 🟢 THE "PUPPET MASTER" OVERRIDES 🟢 ---
        # CASE 1: Sudden Stop (Rear-End)
        if route_filename == "case_stop.rou.xml" and "lead_1" in active_vehicles and "ego" in active_vehicles:
            leader_info = traci.vehicle.getLeader("ego", 100.0)
            if leader_info and leader_info[0] == "lead_1" and leader_info[1] < 40.0:
                if ov_stop_time == -1 and lstm_ready:
                    traci.vehicle.setSpeedMode("lead_1", 31) 
                    traci.vehicle.setSpeed("lead_1", 0) 
                    ov_stop_time = sim_time
                    print(f"⚠️ {sim_time}s: CASE 1 Triggered. Gap is {leader_info[1]:.1f}m. Hard Brake.")
            
            # THE ESCAPE: After 3 seconds, the Lead car hits the gas and drives away!
            if ov_stop_time != -1 and (sim_time - ov_stop_time) > 6.0:
                traci.vehicle.setSpeed("lead_1", 20.0)

        # --- CASE 2: Junction Conflict (Tripwire Logic) ---
        elif route_filename == "case_junction.rou.xml" and "attacker" in active_vehicles and "ego" in active_vehicles:
            
            # THE FIX: Measure how far the Ego car has driven down its own road
            ego_distance = traci.vehicle.getDistance("ego")

            # Blinds the attacker to junction right-of-way rules
            traci.vehicle.setSpeedMode("attacker", 0)

            # TRIGGER 1: If EV hasn't driven 40 meters yet, make the Attacker WAIT
            if ego_distance < 101.0 or not lstm_ready:
                traci.vehicle.setSpeed("attacker", 0.5) 
            
            # TRIGGER 2: Once EV crosses the 40m distance mark, the Attacker "RUSHES"
            else:
                traci.vehicle.setSpeed("attacker", 20.0) 
                if ov_stop_time == -1: 
                    print(f"🚀 {sim_time}s: CASE 2 Triggered. Ego crossed 40m mark! Attacker Rushing!")
                    ov_stop_time = sim_time


        elif route_filename == "case_lane.rou.xml" and "swarver" in active_vehicles and "ego" in active_vehicles:
            ego_x = traci.vehicle.getPosition("ego")[0]
            swarver_x = traci.vehicle.getPosition("swarver")[0]
            gap = swarver_x - ego_x
            
            if 5.0 < gap < 15.0 and ov_stop_time == -1 and lstm_ready:
                traci.vehicle.setLaneChangeMode("swarver", 0) 
                traci.vehicle.changeLane("swarver", 1, 0.5) 
                traci.vehicle.setColor("swarver", (255, 0, 255, 255)) 
                print(f"🚨 {sim_time}s: CASE 3 Triggered. SWERVE INITIATED!")
                ov_stop_time = sim_time

        # --- EGO CONTROL & LOGGING ---
        if "ego" in active_vehicles:
            v_id = "ego"
            v_speed, v_pos, v_angle = traci.vehicle.getSpeed(v_id), traci.vehicle.getPosition(v_id), traci.vehicle.getAngle(v_id)
            lx, ly = traci.vehicle.getDistance(v_id), v_pos[1]
            px, py, status = "N/A", "N/A", "SAFE"

            steering = np.clip((v_angle - 180) / 90.0, -1, 1)
            steer_rate = v_angle - prev_angles.get(v_id, v_angle); prev_angles[v_id] = v_angle
            leader = traci.vehicle.getLeader(v_id, 100.0)
            
            if leader:
                target_id, d = leader
                rel_v = v_speed - traci.vehicle.getSpeed(target_id)
                h_n, rv_n = np.clip(d/500, 0, 1), np.clip(rel_v/50, -1, 1)
                ttc = np.clip((d/rel_v)/10, 0, 1) if rel_v > 0.1 else 1.0
            else: 
                h_n, rv_n, ttc = 1.0, 0.0, 1.0

            features = [lx, ly, v_speed, traci.vehicle.getAcceleration(v_id), steering, steer_rate, v_speed*np.sin(np.radians(v_angle)), h_n, rv_n, ttc]
            predictor.update_buffer(v_id, features)

            if lstm_ready:
                h_path = predictor.hallucinate_trajectory(v_id, v_pos, v_speed)
                px, py = h_path[0][0], h_path[0][1]

                # --- 🎨 DRAW THE LSTM'S BRAIN ---
                poly_id = f"lstm_path_{v_id}"
                shape = [(float(x), float(y)) for x, y in h_path] 
                try:
                    traci.polygon.setShape(poly_id, shape)
                except:
                    traci.polygon.add(poly_id, shape, color=(0, 255, 255, 255), fill=False, layer=100, lineWidth=0.2)
                # --------------------------------

                # --- 🛑 BRAKE LATCH LOGIC ---
                is_collision = any(check_collision_risk(v_id, h_path, oid) for oid in active_vehicles if oid != v_id)
                
                if is_collision:
                    hazard_cooldown = 15  # Force the brakes to stay on for at least 1.5 seconds (15 steps)
                
                if hazard_cooldown > 0:
                    status = "HAZARD"
                    traci.vehicle.setSpeedMode(v_id, 31) 
                    traci.vehicle.setSpeed(v_id, 0)
                    traci.vehicle.setColor(v_id, (255, 0, 0, 255))
                    hazard_cooldown -= 1  # Countdown the timer
                else:
                    status = "SAFE"
                    traci.vehicle.setSpeedMode(v_id, 0)
                    traci.vehicle.setSpeed(v_id, 18.0) 
                    traci.vehicle.setColor(v_id, (0, 255, 0, 255))

            with open(LOG_FILE, 'a', newline='') as f:
                csv.writer(f).writerow([sim_time, v_pos[0], v_pos[1], v_speed, features[3], steering, steer_rate, features[6], h_n, rv_n, ttc, px, py, status])
    traci.close()

if __name__ == "__main__":
    main()