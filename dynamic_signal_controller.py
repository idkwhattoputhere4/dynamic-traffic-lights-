from collections import defaultdict, Counter
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO


class TrafficLightController:
    """Finite State Machine for Dynamic Traffic Signal Control."""
    
    def __init__(self, min_green=5, max_green=15, yellow_time=3):
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time

        # Initial state: Lane 1 starts with Green
        self.current_state = "LANE1_GREEN"
        self.state_start_time = time.time()

    def update(self, lane1_count, lane2_count):
        """Updates light state based on real-time lane queue counts."""
        elapsed = time.time() - self.state_start_time

        # State 1: LANE 1 GREEN
        if self.current_state == "LANE1_GREEN":
            # Switch to yellow if max green expired OR lane 2 has higher demand after min green
            if elapsed >= self.max_green or (elapsed >= self.min_green and lane2_count > lane1_count + 1):
                self.current_state = "LANE1_YELLOW"
                self.state_start_time = time.time()

        # State 2: LANE 1 YELLOW (Clearance Phase)
        elif self.current_state == "LANE1_YELLOW":
            if elapsed >= self.yellow_time:
                self.current_state = "LANE2_GREEN"
                self.state_start_time = time.time()

        # State 3: LANE 2 GREEN
        elif self.current_state == "LANE2_GREEN":
            # Switch to yellow if max green expired OR lane 1 has higher demand after min green
            if elapsed >= self.max_green or (elapsed >= self.min_green and lane1_count > lane2_count + 1):
                self.current_state = "LANE2_YELLOW"
                self.state_start_time = time.time()

        # State 4: LANE 2 YELLOW (Clearance Phase)
        elif self.current_state == "LANE2_YELLOW":
            if elapsed >= self.yellow_time:
                self.current_state = "LANE1_GREEN"
                self.state_start_time = time.time()

        return self.current_state, int(elapsed)


def draw_signal_hud(frame, state, elapsed_time, l1_count, l2_count):
    """Draws a visual HUD panel with interactive traffic light signals."""
    h, w, _ = frame.shape

    # HUD Background Card
    cv2.rectangle(frame, (10, 10), (320, 140), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, 10), (320, 140), (200, 200, 200), 1)

    # Queue Metrics
    cv2.putText(frame, f"Lane 1 Queue: {l1_count}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
    cv2.putText(frame, f"Lane 2 Queue: {l2_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
    cv2.putText(frame, f"Active Phase Time: {elapsed_time}s", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # Signal Color Assignments
    l1_color = (0, 0, 255)  # Red
    l2_color = (0, 0, 255)  # Red

    if state == "LANE1_GREEN":
        l1_color = (0, 255, 0)
    elif state == "LANE1_YELLOW":
        l1_color = (0, 255, 255)
    elif state == "LANE2_GREEN":
        l2_color = (0, 255, 0)
    elif state == "LANE2_YELLOW":
        l2_color = (0, 255, 255)

    # Signal Status Indicators (Circular Light Indicators)
    cv2.putText(frame, "L1 Signal:", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(frame, (110, 111), 10, l1_color, -1)

    cv2.putText(frame, "L2 Signal:", (170, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(frame, (260, 111), 10, l2_color, -1)


# ---------------------------------------------------------
# MAIN TRACKING & SIGNAL EXECUTION PIPELINE
# ---------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8n.pt')

video_path = "sample_traffic.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize Controller
signal_controller = TrafficLightController(min_green=4, max_green=12, yellow_time=3)
track_class_history = defaultdict(list)
MAX_HISTORY_FRAMES = 10

# Lane Polygons (Using your established 360p boundaries)
LANE_1_POLY = np.array([[50, 360], [220, 180], [320, 180], [280, 360]], np.int32)
LANE_2_POLY = np.array([[300, 360], [330, 180], [450, 180], [580, 360]], np.int32)


def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Check Lane Occupancy
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

    # Update Signal Controller State Machine
    current_state, elapsed_time = signal_controller.update(lane_1_count, lane_2_count)

    # Overlay Lane Polygons
    overlay = frame.copy()
    cv2.fillPoly(overlay, [LANE_1_POLY], (0, 165, 255))
    cv2.fillPoly(overlay, [LANE_2_POLY], (255, 0, 255))
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    # Render Light HUD
    draw_signal_hud(frame, current_state, elapsed_time, lane_1_count, lane_2_count)

    cv2.imshow("Dynamic AI Traffic Controller - Phase 3", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()