
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = Path("data/videos/test.mp4")
output_path = Path("runs/phase3/trajectory_video.mp4")
output_path.parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

size = (1280, 720)

writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    size
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Cannot create output video")

# Store the last 30 positions for each tracking ID
track_history = defaultdict(lambda: deque(maxlen=30))

frame_number = 0

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.resize(frame, size)

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        result = results[0]
        annotated_frame = result.plot()

        # Draw movement trajectories
        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes.xywh.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):

                x, y, w, h = box

                # Store the center position
                track_history[track_id].append(
                    (int(x), int(y))
                )

                # Convert previous positions to OpenCV format
                points = np.array(
                    track_history[track_id],
                    dtype=np.int32
                ).reshape((-1, 1, 2))

                # Draw the movement trail
                if len(points) >= 2:
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(0, 255, 255),
                        thickness=2
                    )

            if frame_number % 10 == 0:
                print(
                    f"Frame {frame_number}: "
                    f"{len(track_ids)} active tracks"
                )

        writer.write(annotated_frame)

        cv2.imshow(
            "VisionGuard AI - Object Trajectories",
            annotated_frame
        )

        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

print("Total frames processed:", frame_number)
print("Trajectory video saved to:", output_path)
