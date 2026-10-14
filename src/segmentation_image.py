
import cv2
from pathlib import Path
from ultralytics import YOLO

# Load pretrained segmentation model
model = YOLO("yolov8n-seg.pt")

# Input image
image_path = Path("data/images/test.png")

if not image_path.exists():
    raise FileNotFoundError(image_path)

# Run instance segmentation
results = model.predict(
    source=str(image_path),
    conf=0.25,
    imgsz=640,
    verbose=False
)

result = results[0]

# Inspect segmentation output
if result.masks is not None:
    print("Number of segmented objects:", len(result.masks.data))
    print("Mask tensor shape:", result.masks.data.shape)

    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        print(
            f"Object {i + 1}: "
            f"{class_name}, "
            f"confidence={confidence:.2f}"
        )
else:
    print("No segmentation masks detected.")

# Draw masks, bounding boxes, and labels
annotated_image = result.plot()

# Resize for screen display
height, width = annotated_image.shape[:2]
scale = min(1280 / width, 720 / height, 1.0)

display_image = cv2.resize(
    annotated_image,
    (int(width * scale), int(height * scale))
)

# Save the result
output_path = Path("runs/phase4/segmentation_image.jpg")
output_path.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(output_path), annotated_image)

# Display the result
cv2.imshow("VisionGuard AI - Instance Segmentation", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Segmentation image saved to:", output_path)
