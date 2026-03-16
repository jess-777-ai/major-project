#!/usr/bin/env python3
"""
connect.py
(Script description...)
 - 💥 NOW logs the full 20x11 input matrix on EVERY prediction
"""
# ... (all imports and config are the same) ...

import os
import sys
import time
import csv
import traceback
import pickle
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

import traci
import tensorflow as tf
from ngsim_1 import NGSIMCollisionPredictor

# ---------- CONFIG ----------
EGO_ID = "ego"
SEQ_LEN = 20

MODEL_FILENAME = os.path.join(PROJECT_ROOT, "sumo\ngsim_collision_model_fast_best.h5")
SCALER_FILENAME = os.path.join(PROJECT_ROOT, "sumo\ngsim_collision_model_fast_scaler.pkl")
SUMO_CFG = os.path.join(SCRIPT_DIR, "sumo.sumocfg")  # sumo/sumo.sumocfg
SUMO_GUI_BIN = "sumo-gui"  # change to full path if needed

LOG_DIR = SCRIPT_DIR
SUMO_LOG = os.path.join(LOG_DIR, "sumo_log.txt")
SUMO_ERROR_LOG = os.path.join(LOG_DIR, "sumo_errors.txt")
STEP_LOG_CSV = os.path.join(LOG_DIR, "traci_step_log.csv")
MATRIX_LOG_CSV = os.path.join(LOG_DIR, "traci_input_matrix_log.csv")
# ----------------------------

def ensure_overwrite(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def load_model_and_scaler(model_path, scaler_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

def start_sumo():
    ensure_overwrite(SUMO_LOG)
    ensure_overwrite(SUMO_ERROR_LOG)
    ensure_overwrite(STEP_LOG_CSV)
    ensure_overwrite(MATRIX_LOG_CSV)

    sumo_cmd = [
        SUMO_GUI_BIN,
        "-c", SUMO_CFG,
        "--log", SUMO_LOG,
        "--error-log", SUMO_ERROR_LOG
    ]
    print("Starting SUMO with command:", " ".join(sumo_cmd))
    traci.start(sumo_cmd)

def main():
    print("Project root:", PROJECT_ROOT)
    print("SUMO config:", SUMO_CFG)

    predictor = NGSIMCollisionPredictor(sequence_length=SEQ_LEN)

    try:
        predictor.model, predictor.scaler = load_model_and_scaler(MODEL_FILENAME, SCALER_FILENAME)
        print("✅ Model & scaler loaded.")
    except Exception as e:
        print("❌ Failed to load model/scaler:", e)
        traceback.print_exc()
        return

    try:
        start_sumo()
    except Exception as e:
        print("❌ Failed to start SUMO (is sumo-gui on PATH?):", e)
        traceback.print_exc()
        return

    csv_file = None
    matrix_csv_file = None
    
    try:
        # --- Setup STEP logger ---
        print(f"Attempting to create log file at: {STEP_LOG_CSV}")
        csv_file = open(STEP_LOG_CSV, "w", newline="")
        csv_writer = csv.writer(csv_file)
        print("✅ Step log file opened successfully.")
        
        csv_writer.writerow([
            "sim_time", "ego_present", "ego_x", "ego_y", "ego_speed", "ego_acc",
            "leader_id", "leader_headway", "leader_speed", "lead_acc", "lead_x", "lead_y",
            "relative_velocity", "lateral_distance",
            "follower_id", "follower_gap", "follower_speed",
            "prediction_risk", "risk_level"
        ])
        csv_file.flush()
        print("✅ Step log CSV header written.")

        # --- Setup MATRIX logger ---
        print(f"Attempting to create matrix log file at: {MATRIX_LOG_CSV}")
        matrix_csv_file = open(MATRIX_LOG_CSV, "w", newline="")
        matrix_csv_writer = csv.writer(matrix_csv_file)
        matrix_csv_writer.writerow([
            "1_distance_to_lead", "2_relative_velocity", "3_ego_speed", "4_ego_acc",
            "5_lead_speed", "6_lead_acc", "7_lateral_distance", "8_ego_x",
            "9_ego_y", "10_lead_x", "11_lead_y"
        ])
        matrix_csv_writer.writerow(["--- (Header: 11 features) ---"])
        matrix_csv_file.flush()
        print("✅ Matrix log file opened successfully.")


        traci.simulationStep()
        
        while traci.simulation.getMinExpectedNumber() > 0:
            
            sim_time = traci.simulation.getTime()
            ego_present = EGO_ID in traci.vehicle.getIDList()
            
            # (Initialize variables...)
            ego_x = ego_y = ego_speed = ego_acc = None
            leader_id = leader_headway = headway = None
            lead_speed = lead_acc = lead_x = lead_y = None
            distance_to_lead = relative_velocity = lateral_distance = None
            follower_id = follower_gap = follower_speed = None 
            prediction_risk = None
            risk_level = "NONE" 

            try:
                if "ego_alert_poi" in traci.poi.getIDList():
                    traci.poi.remove("ego_alert_poi")
            except Exception:
                pass 

            if ego_present:
                traci.vehicle.setSpeedMode(EGO_ID, 31)
                
                # --- 2. PERCEPTION ---
                ego_speed = traci.vehicle.getSpeed(EGO_ID)
                ego_acc = traci.vehicle.getAcceleration(EGO_ID)
                ego_x, ego_y = traci.vehicle.getPosition(EGO_ID)
                leader_data = traci.vehicle.getLeader(EGO_ID, 1000.0)
                if leader_data:
                    leader_id, headway = leader_data
                    try:
                        lead_speed = traci.vehicle.getSpeed(leader_id)
                        if lead_speed is None: raise traci.exceptions.TraCIException("Leader speed is None.")
                        lead_acc = traci.vehicle.getAcceleration(leader_id) 
                        lead_x, lead_y = traci.vehicle.getPosition(leader_id) 
                        distance_to_lead = headway
                        relative_velocity = ego_speed - lead_speed
                        lateral_distance = abs(ego_y - lead_y)
                    except traci.exceptions.TraCIException:
                        leader_id = None; distance_to_lead = 999.0; relative_velocity = 0.0
                        lead_speed = ego_speed; lead_acc = 0.0; lateral_distance = 0.0
                        lead_x = ego_x + 999.0; lead_y = ego_y
                else:
                    leader_id = None; distance_to_lead = 999.0; relative_velocity = 0.0
                    lead_speed = ego_speed; lead_acc = 0.0; lateral_distance = 0.0
                    lead_x = ego_x + 999.0; lead_y = ego_y

                # --- 3. UPDATE PREDICTOR ---
                current_timestep_features = [
                    distance_to_lead, relative_velocity, ego_speed, ego_acc,
                    lead_speed, lead_acc, lateral_distance, ego_x,
                    ego_y, lead_x, lead_y
                ]
                predictor.add_timestep_features(EGO_ID, current_timestep_features)

                # --- 4. PREDICTION & CONTROL ---
                follower_data = traci.vehicle.getFollower(EGO_ID)
                if follower_data:
                    try:
                        follower_id, follower_gap = follower_data
                        follower_speed = traci.vehicle.getSpeed(follower_id)
                    except traci.exceptions.TraCIException: follower_id = None 
                
                # --- Test Case 3: Junction Logic ---
                # (We removed the junction vehicle, so this will be false)
                junction_risk = False
                conflicting_vehicles = []
                if traci.vehicle.getRoadID(EGO_ID) == "E1" and traci.vehicle.getLanePosition(EGO_ID) > 150:
                    conflicting_vehicles.extend(traci.lane.getLastStepVehicleIDs("-E3_0"))
                for veh_id in conflicting_vehicles:
                    if veh_id == "veh_top_1":
                        try:
                            if traci.vehicle.getLanePosition(veh_id) > 70:
                                junction_risk = True; risk_level = "CRITICAL"; prediction_risk = 1.0
                        except traci.exceptions.TraCIException: pass 
                if junction_risk:
                    traci.vehicle.setStop(EGO_ID, edgeID="E1", pos=180, duration=5.0) 
                    traci.poi.add(poiID="ego_alert_poi", x=ego_x, y=ego_y + 5, color=(255, 0, 0, 255))

                # --- Test Cases 1 (Braking) ---
                elif leader_id: 
                    
                    if len(predictor.vehicle_buffers.get(EGO_ID, [])) >= SEQ_LEN:
                        # --- 1. ML MODEL IS READY ---
                        try:
                            prediction_risk = predictor.predict_collision_risk(EGO_ID)
                            risk_level = predictor.get_risk_level(prediction_risk)
                        except Exception as e:
                            print("Prediction error:", e)
                            prediction_risk = None; risk_level = "NONE"

                        # --- Log the matrix (we'll leave this on to see what's happening) ---
                        try:
                            current_buffer = predictor.vehicle_buffers.get(EGO_ID)
                            matrix_csv_writer.writerow([]) 
                            matrix_csv_writer.writerow([f"--- START MATRIX (SimTime: {sim_time}, Risk: {prediction_risk}, Level: {risk_level}) ---"])
                            for feature_vector in list(current_buffer):
                                matrix_csv_writer.writerow(feature_vector)
                            matrix_csv_writer.writerow([f"--- END MATRIX (SimTime: {sim_time}) ---"])
                            matrix_csv_file.flush()
                        except Exception as e:
                            print(f"Error logging matrix: {e}")

                        # --- CONTROL BASED ON ML PREDICTION ---
                        if risk_level == "HIGH" or risk_level == "CRITICAL":
                            # This is the action we WANT
                            print(f"--- 🚨 ML MODEL DETECTED {risk_level} RISK! BRAKING! ---")
                            traci.vehicle.setSpeed(EGO_ID, 0.0)
                            traci.poi.add(poiID="ego_alert_poi", x=ego_x, y=ego_y + 5, color=(255, 0, 0, 255))
                        
                        # --- 💥💥💥 LANE CHANGE LOGIC IS NOW DISABLED 💥💥💥 ---
                        #
                        # elif lead_speed < 1 and (risk_level == "LOW" or risk_level == "NONE"):
                        #     # (Lane change logic was here)
                        #     print(f"--- LANE CHANGE LOGIC TRIGGERED ---")
                        #     is_target_lane_safe = True
                        #     if "lane_blocker" in traci.vehicle.getIDList():
                        #         try:
                        #             blocker_pos = traci.vehicle.getLanePosition("lane_blocker")
                        #             ego_pos = traci.vehicle.getLanePosition(EGO_ID)
                        #             if abs(blocker_pos - ego_pos) < 20: is_target_lane_safe = False
                        #         except traci.exceptions.TraCIException: is_target_lane_safe = True 
                        #     if is_target_lane_safe:
                        #         traci.vehicle.changeLane(EGO_ID, 0, duration=3.0) 
                        #     else:
                        #         traci.vehicle.setSpeed(EGO_ID, 0.0) 
                        #         traci.poi.add(poiID="ego_alert_poi", x=ego_x, y=ego_y + 5, color=(255, 255, 0, 255))
                        #
                        # --- 💥💥💥 END OF DISABLED BLOCK 💥💥💥 ---
                        
                        else:
                            # --- SAFE-FOLLOWING LOGIC ---
                            # This is the FALLBACK. If the ML model fails (predicts NONE),
                            # this logic will run, making the car slow down as the leader slows.
                            print(f"--- ML Model predicted {risk_level}. Using safe-following. ---")
                            safe_gap = 5.0
                            max_speed = traci.lane.getMaxSpeed(traci.vehicle.getLaneID(EGO_ID))
                            speed_adjustment = (distance_to_lead - safe_gap) * 0.5
                            target_speed = lead_speed + speed_adjustment
                            target_speed = max(0, min(target_speed, max_speed))
                            traci.vehicle.setSpeed(EGO_ID, target_speed)
                    
                    else:
                        # (Buffer filling logic...)
                        print(f"[{sim_time:.2f}s] Buffer filling... ({len(predictor.vehicle_buffers.get(EGO_ID, []))}/{SEQ_LEN})")
                        safe_gap = 5.0
                        max_speed = traci.lane.getMaxSpeed(traci.vehicle.getLaneID(EGO_ID))
                        speed_adjustment = (distance_to_lead - safe_gap) * 0.5
                        target_speed = lead_speed + speed_adjustment
                        target_speed = max(0, min(target_speed, max_speed))
                        traci.vehicle.setSpeed(EGO_ID, target_speed)

                else: 
                    # (No leader logic)
                    print("--- No leader detected. Setting max speed. ---")
                    current_lane_id = traci.vehicle.getLaneID(EGO_ID)
                    if current_lane_id: 
                        max_speed = traci.lane.getMaxSpeed(current_lane_id)
                        traci.vehicle.setSpeed(EGO_ID, max_speed)
                    else:
                        try: traci.vehicle.setSpeed(EGO_ID, 0)
                        except traci.exceptions.TraCIException: pass 
            
            # --- 5. LOGGING (to step log) ---
            csv_writer.writerow([
                sim_time, bool(ego_present),
                ego_x, ego_y, ego_speed, ego_acc,
                leader_id, headway, lead_speed, lead_acc, lead_x, lead_y,
                relative_velocity, lateral_distance,
                follower_id, follower_gap, follower_speed,
                prediction_risk, risk_level
            ])
            csv_file.flush()
            
            traci.simulationStep() 
            # --- END OF LOOP ---

        print("Simulation finished or no more expected vehicles.")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error in main loop: {e}")
        traceback.print_exc()
    finally:
        # (Safe closing of both files...)
        try: traci.close()
        except Exception: pass
        if csv_file:
            try:
                csv_file.close()
                print("✅ Step log file closed.")
            except Exception as e: print(f"❌ Error closing step log file: {e}")
        if matrix_csv_file:
            try:
                matrix_csv_file.close()
                print("✅ Matrix log file closed.")
            except Exception as e: print(f"❌ Error closing matrix log file: {e}")
        print("TraCI closed, logs saved.")

if __name__ == "__main__":
    main()