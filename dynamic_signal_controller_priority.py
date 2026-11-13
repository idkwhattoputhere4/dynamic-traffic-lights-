from collections import defaultdict
import csv
from datetime import datetime
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO


class PriorityTrafficLightController:
    """Finite State Machine with Emergency Priority Override."""

    def __init__(self, min_green=4, max_green=12, yellow_time=3):
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time

        self.current_state = "LANE1_GREEN"
        self.state_start_time = time.time()
        self.emergency_override = False
        self.target_emergency_lane = None

    def trigger_emergency(self, lane_number):
        """Forces immediate signal preemption for emergency vehicles."""
        self.emergency_override = True
        self.target_emergency_lane = lane_number

    def clear_emergency(self):
        """Clears emergency preemption and resumes normal FSM operation."""
        self.emergency_override = False
        self.target_emergency_lane = None

    def update(self, lane1_count, lane2_count):
        elapsed = time.time() - self.state_start_time

        # --- EMERGENCY OVERRIDE HANDLING ---
        if self.emergency_override:
            if self.target_emergency_lane == 2 and self.current_state != "LANE2_GREEN":
                if self.current_state == "LANE1_GREEN":
                    self.current_state = "LANE1_YELLOW"
                    self.state_start_time = time.time()
                elif self.current_state == "LANE1_YELLOW" and elapsed >= 2:  # Accelerated yellow
                    self.current_state = "LANE2_GREEN"
                    self.state_start_time = time.time()
            elif self.target_emergency_lane == 1 and self.current_state != "LANE1_GREEN":
                if self.current_state == "LANE2_GREEN":
                    self.current_state = "LANE2_YELLOW"
                    self.state_start_time = time.time()
                elif self.current_state == "LANE2_YELLOW" and elapsed >= 2:
                    self.current_state = "LANE1_GREEN"
                    self.state_start_time = time.time()
            return self.current_state, int(elapsed)

        # --- STANDARD FSM LOGIC ---
        if self.current_state == "LANE1_GREEN":
            if elapsed >= self.max_green or (elapsed >= self.min_green and lane2_count > lane1_count + 1):
                self.current_state = "LANE1_YELLOW"
                self.state_start_time = time.time()

        elif self.current_state == "LANE1_YELLOW":
            if elapsed >= self.yellow_time:
                self.current_state = "LANE2_GREEN"
                self.state_start_time = time.time()

        elif self.current_state == "LANE2_GREEN":
            if elapsed >= self.max_green or (elapsed >= self.min_green and lane1_count > lane2_count + 1):
                self.current_state = "LANE2_YELLOW"
                self.state_start_time = time.time()

        elif self.current_state == "LANE2_YELLOW":
            if elapsed >= self.yellow_time:
                self.current_state = "LANE1_GREEN"
                self.state_start_time = time.time()

        return self.current_state, int(elapsed)


class AnalyticsLogger:
    def __init__(self, filename="traffic_analytics.csv"):
        self.filename = filename
        self.last_log_time = time.time()
        with open(self.filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Active_Phase", "Phase_Elapsed_Sec",
                "Lane_1_Queue", "Lane_2_Queue", "Total_Serviced", "FPS", "Emergency_Active"
            ])

    def log(self, state, elapsed_time, l1_count, l2_count, total_serviced, fps, emergency):
        current_time = time.time()
        if current_time - self.last_log_time >= 1.0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, state, elapsed_time, l1_count, l2_count, total_serviced, round(fps, 1), emergency
                ])
            self.last_log_time = current_time


def draw_enhanced_hud(frame, state, elapsed_time, l1_count, l2_count, total_serviced, fps, emergency):
    # Expanded HUD Box for Emergency Indicator
    cv2.rectangle(frame, (10, 10), (350, 180), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, 10), (350, 180), (200, 200, 200), 1)

    cv2.putText(frame, f"Lane 1 Queue: {l1_count}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    cv2.putText(frame, f"Lane 2 Queue: {l2_count}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(frame, f"Serviced Vehicles: {total_serviced}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"Active Phase Time: {elapsed_time}s", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(frame, f"GPU System Speed: {fps:.1f} FPS", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Emergency Alert Banner
    if emergency:
        cv2.putText(frame, "EMERGENCY OVERRIDE: ACTIVE", (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "EMERGENCY MODE: NORMAL", (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    # Signal Indicators
    l1_color = (0, 0, 255)
    l2_color = (0, 0, 255)

    if state == "LANE1_GREEN":
        l1_color = (0, 255, 0)
    elif state == "LANE1_YELLOW":
        l1_color = (0, 255, 255)
    elif state == "LANE2_GREEN":
        l2_color = (0, 255, 0)
    elif state == "LANE2_YELLOW":
        l2_color = (0, 255, 255)

    cv2.putText(frame, "L1 Signal:", (20, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(frame, (110, 158), 9, l1_color, -1)

    cv2.putText(frame, "L2 Signal:", (180, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(frame, (270, 158), 9, l2_color, -1)


# ---------------------------------------------------------
# MAIN EXECUTION PIPELINE
# ---------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8n.pt')

video_path = "sample_traffic.mp4"
cap = cv2.VideoCapture(video_path)

signal_controller = PriorityTrafficLightController(min_green=4, max_green=12, yellow_time=3)
logger = AnalyticsLogger("traffic_analytics.csv")
seen_vehicle_ids = set()

LANE_1_POLY = np.array([[50, 360], [220, 180], [320, 180], [280, 360]], np.int32)
LANE_2_POLY = np.array([[300, 360], [330, 180], [450, 180], [580, 360]], np.int32)


def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


prev_frame_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time + 1e-6)
    prev_frame_time = new_frame_time

    results = model.track(
        frame, 
        device=device, 
        classes=[2, 3, 5, 7], 
        persist=True, 
        tracker="bytetrack.yaml", 
        verbose=False
    )

    lane_1_count = 0
    lane_2_count = 0

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            seen_vehicle_ids.add(track_id)

            in_lane_1 = is_point_in_polygon((cx, cy), LANE_1_POLY)
            in_lane_2 = is_point_in_polygon((cx, cy), LANE_2_POLY)

            color = (0, 255, 0)
            if in_lane_1:
                lane_1_count += 1
                color = (0, 165, 255)
            elif in_lane_2:
                lane_2_count += 1
                color = (255, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Controller & Logger Update
    current_state, elapsed_time = signal_controller.update(lane_1_count, lane_2_count)
    logger.log(
        current_state, elapsed_time, lane_1_count, lane_2_count, 
        len(seen_vehicle_ids), fps, signal_controller.emergency_override
    )

    # Overlay & HUD Render
    overlay = frame.copy()
    cv2.fillPoly(overlay, [LANE_1_POLY], (0, 165, 255))
    cv2.fillPoly(overlay, [LANE_2_POLY], (255, 0, 255))
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    draw_enhanced_hud(
        frame, current_state, elapsed_time, lane_1_count, lane_2_count, 
        len(seen_vehicle_ids), fps, signal_controller.emergency_override
    )

    cv2.imshow("Dynamic AI Traffic Controller - Priority Preemption", frame)

    # Interactive Keyboard Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):  # Press 'e' to trigger emergency override for Lane 2
        print("[ALERT] Emergency Vehicle Priority Override Triggered for Lane 2!")
        signal_controller.trigger_emergency(lane_number=2)
    elif key == ord('c'):  # Press 'c' to clear emergency override
        print("[INFO] Emergency Priority Cleared. Resuming Normal Operation.")
        signal_controller.clear_emergency()

cap.release()
cv2.destroyAllWindows()