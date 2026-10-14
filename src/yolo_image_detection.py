
import cv2
from pathlib import Path
from ultralytics import YOLO

# -----------------------------------
# STEP 1: Load YOLOv8 model
# -----------------------------------
model = YOLO("yolov8n.pt")

# -----------------------------------
# STEP 2: Set input image path
# -----------------------------------
image_path = Path("data/images/test.png")

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

# -----------------------------------
# STEP 3: Perform object detection
# -----------------------------------
results = model.predict(
    source=str(image_path),
    conf=0.25,
    save=True,
    project="runs",
    name="phase1",
    exist_ok=True
)

# -----------------------------------
# STEP 4: Print detected objects
# -----------------------------------
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        coordinates = box.xyxy[0].tolist()

        print("Object:", model.names[class_id])
        print("Confidence:", round(confidence, 3))
        print("Bounding Box:", coordinates)
        print("---------------------------")

# -----------------------------------
# STEP 5: Get annotated image
# -----------------------------------
annotated_image = results[0].plot(
    line_width=2,
    font_size=12
)

# -----------------------------------
# STEP 6: Resize for screen display
# -----------------------------------
height, width = annotated_image.shape[:2]

# Maximum display dimensions
max_width = 1200
max_height = 700

# Preserve original aspect ratio
scale = min(
    max_width / width,
    max_height / height,
    1.0
)

new_width = int(width * scale)
new_height = int(height * scale)

display_image = cv2.resize(
    annotated_image,
    (new_width, new_height),
    interpolation=cv2.INTER_AREA
)

# -----------------------------------
# STEP 7: Display complete image
# -----------------------------------
window_name = "VisionGuard AI - YOLOv8 Detection"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.imshow(window_name, display_image)

cv2.resizeWindow(
    window_name,
    new_width,
    new_height
)

print("Image displayed successfully!")
print("Press any key to close the window.")

cv2.waitKey(0)
cv2.destroyAllWindows()
