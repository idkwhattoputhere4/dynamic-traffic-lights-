from collections import defaultdict, Counter
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 1. Load GPU model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8n.pt')

video_path = "sample_traffic.mp4"
cap = cv2.VideoCapture(video_path)

# Class history for stabilization
track_class_history = defaultdict(list)
MAX_HISTORY_FRAMES = 10

# ---------------------------------------------------------
# DEFINE VIRTUAL DETECTION ZONES (ROIs)
# Adjust these coordinates based on your video dimensions (360p example)
# ---------------------------------------------------------
# Polygon points format: np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)

LANE_1_POLY = np.array([
    [50, 360],    # Bottom Left
    [220, 180],   # Top Left
    [320, 180],   # Top Right
    [280, 360]    # Bottom Right
], np.int32)

LANE_2_POLY = np.array([
    [300, 360],   # Bottom Left
    [330, 180],   # Top Left
    [450, 180],   # Top Right
    [580, 360]    # Bottom Right
], np.int32)


def is_point_in_polygon(point, polygon):
    """Returns True if the (x, y) centroid is inside the polygon zone."""
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions if needed
    h, w, _ = frame.shape

    # Run ByteTrack
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
            # Calculate Vehicle Centroid (Center Point)
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Class stabilization
            track_class_history[track_id].append(class_id)
            if len(track_class_history[track_id]) > MAX_HISTORY_FRAMES:
                track_class_history[track_id].pop(0)
            stable_class_id = Counter(track_class_history[track_id]).most_common(1)[0][0]
            class_name = model.names[stable_class_id]

            # Check ROI Occupancy
            in_lane_1 = is_point_in_polygon((cx, cy), LANE_1_POLY)
            in_lane_2 = is_point_in_polygon((cx, cy), LANE_2_POLY)

            # Assign color based on zone detection
            color = (0, 255, 0) # Green default
            if in_lane_1:
                lane_1_count += 1
                color = (0, 165, 255) # Orange for Lane 1
            elif in_lane_2:
                lane_2_count += 1
                color = (255, 0, 255) # Purple for Lane 2

            # Draw vehicle bounding box & centroid dot
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(
                frame, 
                f"#{track_id} {class_name}", 
                (x1, y1 - 8), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )

    # ---------------------------------------------------------
    # DRAW ROI OVERLAYS & HUD
    # ---------------------------------------------------------
    # Draw transparent lane polygons
    overlay = frame.copy()
    cv2.fillPoly(overlay, [LANE_1_POLY], (0, 165, 255))
    cv2.fillPoly(overlay, [LANE_2_POLY], (255, 0, 255))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # Outline ROI borders
    cv2.polylines(frame, [LANE_1_POLY], True, (0, 165, 255), 2)
    cv2.polylines(frame, [LANE_2_POLY], True, (255, 0, 255), 2)

    # Display HUD Counters
    cv2.rectangle(frame, (10, 10), (240, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Lane 1 Queue: {lane_1_count}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    cv2.putText(frame, f"Lane 2 Queue: {lane_2_count}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.imshow("Dynamic Traffic Control - Lane ROI Analytics", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()