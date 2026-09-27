import cv2


video_path = "data/videos/test.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError("Video could not be opened.")


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


print("FPS:", fps)
print("Width:", width)
print("Height:", height)
print("Total Frames:", frame_count)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("VisionGuard Video", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()