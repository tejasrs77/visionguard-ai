
import cv2
from pathlib import Path
from ultralytics import YOLO

# Load pose estimation model
model = YOLO("yolov8n-pose.pt")

image_path = Path("data/images/test.png")
output_path = Path("runs/phase4/pose_image.jpg")

if not image_path.exists():
    raise FileNotFoundError(image_path)

output_path.parent.mkdir(parents=True, exist_ok=True)

# Run pose estimation
results = model.predict(
    source=str(image_path),
    conf=0.25,
    imgsz=640,
    verbose=False
)

result = results[0]

# Inspect pose keypoints
if result.keypoints is not None:
    keypoints = result.keypoints

    print("Keypoint tensor shape:", keypoints.data.shape)
    print("Number of detected people:", len(keypoints.data))

    for person_index, person_keypoints in enumerate(keypoints.xy):
        print(f"\nPerson {person_index + 1}")

        # Show nose and shoulder coordinates
        print("Nose:", person_keypoints[0].cpu().tolist())
        print("Left shoulder:", person_keypoints[5].cpu().tolist())
        print("Right shoulder:", person_keypoints[6].cpu().tolist())

else:
    print("No pose keypoints detected.")

# Draw skeleton and keypoints
annotated_image = result.plot()

# Save annotated image
cv2.imwrite(str(output_path), annotated_image)

# Resize for screen display
height, width = annotated_image.shape[:2]
scale = min(1280 / width, 720 / height, 1.0)

display_image = cv2.resize(
    annotated_image,
    (int(width * scale), int(height * scale))
)

cv2.imshow(
    "VisionGuard AI - Human Pose Estimation",
    display_image
)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nPose image saved to:", output_path)
