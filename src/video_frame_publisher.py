
from pathlib import Path
import queue
import threading
import time

import cv2


# =====================================
# 1. CONFIGURATION
# =====================================

VIDEO_PATH = Path("data/videos/test.mp4")
MAX_FRAMES = 100

frame_queue = queue.Queue(maxsize=5)


# =====================================
# 2. VIDEO PUBLISHER
# =====================================

def publisher():
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        frame_queue.put(None)
        return

    frame_id = 0

    try:
        while frame_id < MAX_FRAMES:
            ret, frame = cap.read()

            if not ret:
                break

            frame_id += 1

            # Resize to reduce processing cost
            frame = cv2.resize(frame, (640, 360))

            message = {
                "topic": "/camera/image_raw",
                "frame_id": frame_id,
                "timestamp": time.time(),
                "frame": frame
            }

            frame_queue.put(message)

            print(f"[PUBLISHER] Frame {frame_id} sent")

    finally:
        cap.release()
        frame_queue.put(None)


# =====================================
# 3. VIDEO SUBSCRIBER
# =====================================

def subscriber():
    received_frames = 0

    while True:
        message = frame_queue.get()

        if message is None:
            break

        frame = message["frame"]
        frame_id = message["frame_id"]

        received_frames += 1

        print(
            f"[SUBSCRIBER] Frame {frame_id} received "
            f"| Shape: {frame.shape}"
        )

    print(
        f"\nTotal frames received: {received_frames}"
    )


# =====================================
# 4. RUN BOTH COMPONENTS
# =====================================

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

print("\nVideo-frame communication completed!")
