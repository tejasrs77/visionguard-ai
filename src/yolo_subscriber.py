
from pathlib import Path
import queue
import threading
import time

import cv2
from ultralytics import YOLO


# =====================================
# 1. CONFIGURATION
# =====================================

VIDEO_PATH = Path("data/videos/test.mp4")
MODEL_PATH = "yolov8n.pt"

OUTPUT_DIR = Path("runs/phase7")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "yolo_subscriber.mp4"

MAX_FRAMES = 100
FRAME_SIZE = (640, 360)

frame_queue = queue.Queue(maxsize=5)


# =====================================
# 2. CAMERA PUBLISHER
# =====================================

def publisher():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    frame_id = 0

    if not cap.isOpened():
        print(f"Cannot open video: {VIDEO_PATH}")
        frame_queue.put(None)
        return

    try:
        while frame_id < MAX_FRAMES:
            success, frame = cap.read()

            if not success:
                break

            frame_id += 1

            frame = cv2.resize(frame, FRAME_SIZE)

            message = {
                "topic": "/camera/image_raw",
                "frame_id": frame_id,
                "timestamp": time.time(),
                "frame": frame
            }

            frame_queue.put(message)

            if frame_id % 10 == 0:
                print(
                    f"[PUBLISHER] Sent {frame_id} frames"
                )

    finally:
        cap.release()
        frame_queue.put(None)


# =====================================
# 3. YOLO SUBSCRIBER
# =====================================

def subscriber():
    model = YOLO(MODEL_PATH)

    writer = None
    processed_frames = 0
    total_detections = 0

    start_time = time.perf_counter()

    try:
        while True:
            message = frame_queue.get()

            if message is None:
                break

            frame = message["frame"]
            frame_id = message["frame_id"]

            # Perform YOLO inference
            results = model.predict(
                frame,
                conf=0.25,
                imgsz=640,
                device="cpu",
                verbose=False
            )

            result = results[0]

            detections = len(result.boxes)

            total_detections += detections
            processed_frames += 1

            # Draw bounding boxes
            annotated_frame = result.plot()

            if writer is None:
                height, width = annotated_frame.shape[:2]

                writer = cv2.VideoWriter(
                    str(OUTPUT_VIDEO),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    20,
                    (width, height)
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        "Could not create output video."
                    )

            writer.write(annotated_frame)

            if frame_id % 10 == 0:
                print(
                    f"[SUBSCRIBER] Frame {frame_id} "
                    f"| Detections: {detections}"
                )

    finally:
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - start_time

    print("\n========== RESULTS ==========")
    print("Frames processed:", processed_frames)
    print("Total detections:", total_detections)
    print("Processing time:", round(elapsed, 2), "seconds")

    if elapsed > 0:
        print(
            "Pipeline FPS:",
            round(processed_frames / elapsed, 2)
        )

    print("Saved video:", OUTPUT_VIDEO)


# =====================================
# 4. RUN THE PIPELINE
# =====================================

if __name__ == "__main__":
    publisher_thread = threading.Thread(
        target=publisher
    )

    subscriber_thread = threading.Thread(
        target=subscriber
    )

    subscriber_thread.start()
    publisher_thread.start()

    publisher_thread.join()
    subscriber_thread.join()

    print("\nYOLO subscriber pipeline completed!")
