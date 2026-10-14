
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# STEP 1: Load YOLOv8 model
model = YOLO("yolov8n.pt")

# STEP 2: Set input and output paths
video_path = Path("data/videos/test.mp4")
output_path = Path("runs/phase1/detected_video.mp4")

if not video_path.exists():
    raise FileNotFoundError(f"Video not found: {video_path}")

output_path.parent.mkdir(parents=True, exist_ok=True)

# STEP 3: Open video
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Unable to open video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30.0

print("Original FPS:", fps)
print("Video Resolution:", width, "x", height)

# Save at 1280x720 to reduce output size
output_width = 1280
output_height = 720

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (output_width, output_height)
)

if not out.isOpened():
    cap.release()
    raise RuntimeError("Unable to create output video")

# STEP 4: Process video frames
frame_count = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    # Resize before inference for faster processing
    frame = cv2.resize(
        frame,
        (output_width, output_height)
    )

    start_time = time.perf_counter()

    # Perform object detection
    results = model.predict(
        source=frame,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    elapsed = time.perf_counter() - start_time

    processing_fps = 1.0 / elapsed if elapsed > 0 else 0

    # Show processing FPS
    cv2.putText(
        annotated_frame,
        f"Processing FPS: {processing_fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display frame
    cv2.imshow(
        "VisionGuard AI - Video Detection",
        annotated_frame
    )

    # Save processed frame
    out.write(annotated_frame)

    frame_count += 1

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# STEP 5: Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Total frames processed:", frame_count)
print("Output saved to:", output_path)
