import time
import cv2
import traci
import sumolib
import torch
from ultralytics import YOLO

# 1. Initialize YOLOv8 Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)

# 2. Open Video Stream
cap = cv2.VideoCapture("sample_traffic.mp4")
if not cap.isOpened():
    print("[ERROR] Could not open sample_traffic.mp4")
    exit()

# 3. Start SUMO Simulation
sumo_binary = sumolib.checkBinary("sumo-gui")
traci.start([sumo_binary, "-c", "intersection.sumocfg", "--start", "true"])

tls_ids = traci.trafficlight.getIDList()
if not tls_ids:
    print("[ERROR] No traffic light found in SUMO!")
    exit()

tls_id = tls_ids[0]
print(f"[SUCCESS] Connected Balanced AI Controller to SUMO Traffic Light: {tls_id}")

# --- HYSTERESIS & GAP-OUT CONTROL PARAMETERS ---
MIN_GREEN_STEPS = 30     # Minimum green time before a switch can be evaluated
PRESSURE_THRESHOLD = 4   # Extra vehicle difference required on opposing lane to switch
GAP_OUT_STEPS = 5        # Consecutive idle steps required on active lane before allowing switch

current_phase = 0        # 0 = Lane 1 Green, 2 = Lane 2 Green
last_switch_step = 0
empty_gap_steps = 0      # Tracks idle frames on current green approach

step = 0
manual_emergency = False
manual_lane = "L1"

try:
    while cap.isOpened() and traci.simulation.getMinExpectedNumber() > 0:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (800, 480))
        results = model(frame, verbose=False)[0]

        height, width, _ = frame.shape
        roi_l1 = (0, int(height * 0.4), int(width * 0.5), height)
        roi_l2 = (int(width * 0.5), int(height * 0.4), width, height)

        l1_count = 0
        l2_count = 0
        auto_emergency_lane = None

        # Process vehicle detections
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            is_emergency = (cls_id == 5)  # Class 5 in COCO dataset (Bus/Emergency vehicle profile)

            if roi_l1[0] <= cx <= roi_l1[2] and roi_l1[1] <= cy <= roi_l1[3]:
                if is_emergency:
                    auto_emergency_lane = "L1"
                else:
                    l1_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255) if is_emergency else (0, 255, 255), 2)

            elif roi_l2[0] <= cx <= roi_l2[2] and roi_l2[1] <= cy <= roi_l2[3]:
                if is_emergency:
                    auto_emergency_lane = "L2"
                else:
                    l2_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255) if is_emergency else (255, 0, 255), 2)

        # Step SUMO physics
        traci.simulationStep()

        # Active Emergency Determination
        active_emergency = manual_emergency or (auto_emergency_lane is not None)
        active_lane = manual_lane if manual_emergency else auto_emergency_lane

        # --- HYSTERESIS & GAP-OUT AI CONTROLLER LOGIC ---
        if active_emergency:
            # Immediate Emergency Preemption Override
            target_emergency_phase = 0 if active_lane == "L1" else 2
            if current_phase != target_emergency_phase:
                traci.trafficlight.setPhase(tls_id, target_emergency_phase)
                current_phase = target_emergency_phase
                last_switch_step = step
                empty_gap_steps = 0
            
            mode_label = "MANUAL OVERRIDE" if manual_emergency else "AUTO DETECTED"
            cv2.putText(frame, f"[EMERGENCY PRIORITY: {active_lane} ({mode_label})]", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        else:
            # Normal Actuated AI Logic
            steps_in_current_phase = step - last_switch_step

            # 1. Track Gap-Out State (Check if current green approach is clearing)
            active_queue = l1_count if current_phase == 0 else l2_count
            if active_queue == 0:
                empty_gap_steps += 1
            else:
                empty_gap_steps = 0

            # 2. Evaluate Phase Switch with Hysteresis & Gap-Out Guardrails
            if steps_in_current_phase >= MIN_GREEN_STEPS:
                if current_phase == 0:
                    # Current Green: L1. Switch to L2 only if L2 has a significant pressure advantage or L1 gaps out
                    pressure_delta = l2_count - l1_count
                    if pressure_delta >= PRESSURE_THRESHOLD or (l1_count == 0 and empty_gap_steps >= GAP_OUT_STEPS and l2_count > 0):
                        current_phase = 2
                        traci.trafficlight.setPhase(tls_id, 2)
                        last_switch_step = step
                        empty_gap_steps = 0
                        print(f"[AI HYSTERESIS SWITCH] Step {step}: Switching Green to Lane 2 (L2: {l2_count} vs L1: {l1_count} | Delta: {pressure_delta})")

                elif current_phase == 2:
                    # Current Green: L2. Switch to L1 only if L1 has a significant pressure advantage or L2 gaps out
                    pressure_delta = l1_count - l2_count
                    if pressure_delta >= PRESSURE_THRESHOLD or (l2_count == 0 and empty_gap_steps >= GAP_OUT_STEPS and l1_count > 0):
                        current_phase = 0
                        traci.trafficlight.setPhase(tls_id, 0)
                        last_switch_step = step
                        empty_gap_steps = 0
                        print(f"[AI HYSTERESIS SWITCH] Step {step}: Switching Green to Lane 1 (L1: {l1_count} vs L2: {l2_count} | Delta: {pressure_delta})")

        # UI Overlay Metrics
        cv2.putText(frame, f"L1 Queue: {l1_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"L2 Queue: {l2_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        active_phase_name = "L1 GREEN" if current_phase == 0 else "L2 GREEN"
        cv2.putText(frame, f"Active Phase: {active_phase_name}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        status_text = f"EMERGENCY ACTIVE ({'MANUAL' if manual_emergency else 'AUTO'})" if active_emergency else "HYSTERESIS AI MODE"
        status_color = (0, 0, 255) if active_emergency else (0, 255, 0)
        cv2.putText(frame, f"Mode: {status_text}", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow("CV-Driven SUMO Traffic Controller", frame)

        # Keyboard Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            manual_emergency = not manual_emergency
        elif key == ord('r'):
            manual_emergency = False
        elif key == ord('q'):
            break

        step += 1

finally:
    cap.release()
    cv2.destroyAllWindows()
    try:
        traci.close()
    except Exception:
        pass