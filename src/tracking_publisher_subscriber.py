
from pathlib import Path
import queue
import threading
import time

import cv2
from ultralytics import YOLO


# ==========================================
# 1. CONFIGURATION
# ==========================================

VIDEO_PATH = Path("data/videos/test.mp4")
MODEL_PATH = "yolov8n.pt"

OUTPUT_DIR = Path("runs/phase7")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "tracking_pipeline.mp4"

MAX_FRAMES = 100
FRAME_SIZE = (640, 360)

frame_queue = queue.Queue(maxsize=5)
tracking_queue = queue.Queue(maxsize=20)


# ==========================================
# 2. CAMERA PUBLISHER
# ==========================================

def camera_publisher():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    frame_id = 0

    try:
        if not cap.isOpened():
            print("ERROR: Cannot open video:", VIDEO_PATH)
            return

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
                    f"[CAMERA] Published frame {frame_id}"
                )

    finally:
        cap.release()
        frame_queue.put(None)


# ==========================================
# 3. YOLO + BYTETRACK SUBSCRIBER
# ==========================================

def tracking_subscriber():
    writer = None
    processed_frames = 0
    start = time.perf_counter()

    try:
        model = YOLO(MODEL_PATH)

        while True:
            message = frame_queue.get()

            if message is None:
                break

            frame = message["frame"]
            frame_id = message["frame_id"]

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.1,
                imgsz=640,
                device="cpu",
                verbose=False
            )

            result = results[0]

            tracks = []

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):
                boxes = result.boxes

                for i, track_id in enumerate(
                    boxes.id.int().cpu().tolist()
                ):
                    tracks.append({
                        "track_id": track_id,
                        "class_id": int(
                            boxes.cls[i].item()
                        ),
                        "confidence": round(
                            float(boxes.conf[i].item()), 3
                        ),
                        "bbox": [
                            round(float(value), 2)
                            for value in
                            boxes.xyxy[i].cpu().tolist()
                        ]
                    })

            tracking_message = {
                "topic": "/visionguard/tracks",
                "frame_id": frame_id,
                "timestamp": message["timestamp"],
                "tracks": tracks
            }

            tracking_queue.put(tracking_message)

            annotated = result.plot()

            if writer is None:
                height, width = annotated.shape[:2]

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

            writer.write(annotated)
            processed_frames += 1

            if frame_id % 10 == 0:
                print(
                    f"[TRACKER] Frame {frame_id} "
                    f"| Active tracks: {len(tracks)}"
                )

    finally:
        if writer is not None:
            writer.release()

        tracking_queue.put(None)

    elapsed = time.perf_counter() - start

    print("\nTracking processing finished.")
    print("Frames processed:", processed_frames)

    if elapsed > 0:
        print(
            "Pipeline FPS:",
            round(processed_frames / elapsed, 2)
        )

    print("Output video:", OUTPUT_VIDEO)


# ==========================================
# 4. TRACKING RESULTS RECEIVER
# ==========================================

def tracking_receiver():
    received_messages = 0
    unique_track_ids = set()

    while True:
        message = tracking_queue.get()

        if message is None:
            break

        received_messages += 1

        for track in message["tracks"]:
            unique_track_ids.add(track["track_id"])

        if message["frame_id"] % 10 == 0:
            print(
                f"[RECEIVER] Frame {message['frame_id']} "
                f"| Tracks received: "
                f"{len(message['tracks'])}"
            )

    print("\n========== TRACKING SUMMARY ==========")
    print("Messages received:", received_messages)
    print("Unique track IDs:", len(unique_track_ids))


# ==========================================
# 5. RUN ALL THREE COMPONENTS
# ==========================================

if __name__ == "__main__":
    threads = [
        threading.Thread(
            target=camera_publisher,
            name="CameraPublisher"
        ),
        threading.Thread(
            target=tracking_subscriber,
            name="TrackingSubscriber"
        ),
        threading.Thread(
            target=tracking_receiver,
            name="TrackingReceiver"
        )
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print("\nPhase 7 - Step 4 completed!")
