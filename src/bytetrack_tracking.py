
import cv2
from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = Path("data/videos/test.mp4")
output_path = Path("runs/phase3/tracked_video.mp4")

if not video_path.exists():
    raise FileNotFoundError(video_path)

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

        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            print(f"Frame {frame_number}: IDs {track_ids}")

        writer.write(annotated_frame)

        cv2.imshow(
            "VisionGuard AI - ByteTrack",
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
print("Tracked video saved to:", output_path)
