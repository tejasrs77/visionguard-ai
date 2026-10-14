
import cv2
from pathlib import Path
from ultralytics import YOLO

# Load pretrained pose model
model = YOLO("yolov8n-pose.pt")

video_path = Path("data/videos/test.mp4")
output_path = Path("runs/phase4/pose_video.mp4")

output_path.parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Cannot open input video")

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

        # Core pose estimation logic
        results = model.predict(
            source=frame,
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        # Draw skeletons and body keypoints
        annotated_frame = result.plot()

        # Count detected people
        people_count = (
            len(result.keypoints.data)
            if result.keypoints is not None
            else 0
        )

        cv2.putText(
            annotated_frame,
            f"People detected: {people_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        writer.write(annotated_frame)

        cv2.imshow(
            "VisionGuard AI - Video Pose Estimation",
            annotated_frame
        )

        if frame_number % 30 == 0:
            print(
                f"Frame {frame_number}: "
                f"{people_count} people detected"
            )

        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

print("Total frames processed:", frame_number)
print("Pose video saved to:", output_path)
