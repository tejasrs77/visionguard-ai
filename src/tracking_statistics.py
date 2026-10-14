
import cv2
from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = Path("data/videos/test.mp4")
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

unique_ids = set()
total_frames = 0
total_observations = 0
max_concurrent_tracks = 0

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.resize(frame, (1280, 720))

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        result = results[0]
        current_ids = []

        if result.boxes is not None and result.boxes.id is not None:
            current_ids = result.boxes.id.int().cpu().tolist()

        unique_ids.update(current_ids)

        total_observations += len(current_ids)

        max_concurrent_tracks = max(
            max_concurrent_tracks,
            len(current_ids)
        )

        total_frames += 1

        if total_frames % 30 == 0:
            print(f"Processed {total_frames} frames")

finally:
    cap.release()

average_tracks = (
    total_observations / total_frames
    if total_frames else 0
)

print("\n--- ByteTrack Statistics ---")
print("Frames processed:", total_frames)
print("Unique track IDs:", len(unique_ids))
print("Maximum simultaneous tracks:", max_concurrent_tracks)
print("Average tracks per frame:", round(average_tracks, 2))
