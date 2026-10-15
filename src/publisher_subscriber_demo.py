
import queue
import threading
import time


# Shared message queue
message_queue = queue.Queue()


# Publisher
def publisher():
    for frame_id in range(1, 6):
        message = {
            "topic": "/camera/image_raw",
            "frame_id": frame_id,
            "timestamp": time.time()
        }

        message_queue.put(message)

        print(f"[PUBLISHER] Sent frame {frame_id}")

        time.sleep(1)

    # Signal that publishing has finished
    message_queue.put(None)


# Subscriber
def subscriber():
    while True:
        message = message_queue.get()

        if message is None:
            break

        print(
            f"[SUBSCRIBER] Received frame "
            f"{message['frame_id']} "
            f"from {message['topic']}"
        )


# Run publisher and subscriber concurrently
publisher_thread = threading.Thread(target=publisher)
subscriber_thread = threading.Thread(target=subscriber)

subscriber_thread.start()
publisher_thread.start()

publisher_thread.join()
subscriber_thread.join()

print("\nPublisher-Subscriber demo completed!")
