import cv2
import torch
from ultralytics import YOLO

# 1. Select GPU acceleration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running inference on: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

# 2. Load model using downloaded weights
model = YOLO('yolov8n.pt')

# 3. Load video source
video_path = "sample_traffic.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or failed to load frame.")
        break

    # Run ByteTrack object tracker on GPU
    # tracker="bytetrack.yaml" assigns persistent IDs across frames
    results = model.track(
        frame, 
        device=device, 
        classes=[2, 3, 5, 7], 
        persist=True, 
        tracker="bytetrack.yaml", 
        verbose=False
    )

    # Draw tracking boxes, IDs, and trajectory tails onto frame
    annotated_frame = results[0].plot()

    # Display real-time output window
    cv2.imshow("Dynamic Traffic Control - Object Tracking", annotated_frame)

    # Press 'q' to stop execution
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()