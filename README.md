

***

# 🚗 Predictive V2V Collision Avoidance 
Traditional Autonomous Emergency Braking (AEB) is strictly reactive. This project implements a **Predictive Vehicle-to-Vehicle (V2V) pipeline** that allows cars to "see around corners." Using a deep learning LSTM model and an MQTT network, vehicles predict their physical trajectories up to 3.5 seconds into the future and broadcast their intent to avoid collisions before physical sensors even register a threat.

## 🛠️ Tech Stack
* **Hardware:** Raspberry Pi 5 & 4, Keyestudio KS0223 AWD Chassis, MPU6050 (IMU), HC-SR04 (Ultrasonic).
* **Software:** Python, PyTorch (TorchScript JIT), SUMO/TraCI (Simulation), Scikit-Learn.
* **Network:** MQTT (HiveMQ Cloud Broker), JSON.

## ⚙️ System Architecture

### 1. The Brain (AI & Inference)
* **Model:** PyTorch LSTM with Temporal Attention, trained on simulated SUMO traffic data.
* **Residual Velocity:** Predicts micro-adjustments (deltas) in speed and steering instead of absolute coordinates, mitigating vanishing gradients and simplifying the math.
* **Edge Inference:** JIT-compiled for real-time, multi-threaded execution on the Raspberry Pi's ARM architecture.

### 2. The Physical Execution (Robotics)
* **IMU Dead-Reckoning:** Integrates MPU6050 acceleration data at 10Hz to track the chassis' physical X/Y coordinates in feet without relying on GPS.
* **Software Differential Trim:** Counters physical motor drift by dynamically throttling right-side wheels (e.g., Left=30%, Right=26.5%) to force mathematically straight execution.
* **Electromagnetic Braking:** Fires `HIGH/HIGH` logic to the motor driver to short-circuit the motors, locking the wheels instantly instead of coasting.
* **Hardware Failsafe:** Hardwired HC-SR04 ultrasonic sensor provides a strictly reactive emergency override at < 25cm.

### 3. The Network (V2V Comms)
* **Pub/Sub Protocol:** Vehicles constantly extract their 2.0-second future prediction and broadcast it as a JSON payload over the HiveMQ cloud.
* **Fault Tolerance:** The receiving car timestamps incoming telemetry. If network latency exceeds 2.0 seconds, the software discards "stale" predictions and falls back to physical sensors.

## 🚦 Validated Test Scenarios
1. **Tailgate (Longitudinal):** The follower car intercepts the lead car's predicted deceleration and triggers its electromagnetic brakes preemptively based on a shrinking V2V geometric gap.
2. **Junction Yielding (Perpendicular):** Cars approach a blind 90-degree intersection. If predicted paths overlap within a 0.5m radius, the yielding car detects the mathematical collision, halts, and waits for network clearance before resuming.
