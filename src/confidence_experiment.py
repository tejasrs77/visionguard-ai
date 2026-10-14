
import cv2
from pathlib import Path
from ultralytics import YOLO

# STEP 1: Load pretrained YOLOv8
model = YOLO("yolov8n.pt")

# STEP 2: Set image path
image_path = Path("data/images/test.png")

if not image_path.exists():
    raise FileNotFoundError(
        f"Image not found: {image_path}"
    )

# STEP 3: Define confidence thresholds
thresholds = [0.25, 0.50, 0.80]

# STEP 4: Create output folder
output_dir = Path("runs/phase2/confidence")
output_dir.mkdir(parents=True, exist_ok=True)

# STEP 5: Perform detection at each threshold
for threshold in thresholds:

    results = model.predict(
        source=str(image_path),
        conf=threshold,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    print(f"\nConfidence Threshold: {threshold}")
    print("Number of detections:", len(result.boxes))

    # Print detected objects
    for box in result.boxes:
        class_id = int(box.cls[0])
        score = float(box.conf[0])

        print(
            f"{model.names[class_id]}: {score:.3f}"
        )

    # Draw bounding boxes
    annotated = result.plot(
        line_width=2,
        font_size=12
    )

    # Resize for screen display
    height, width = annotated.shape[:2]

    scale = min(
        1200 / width,
        700 / height,
        1.0
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    display_image = cv2.resize(
        annotated,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Add threshold information
    cv2.putText(
        display_image,
        f"Confidence: {threshold:.2f} | "
        f"Detections: {len(result.boxes)}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Save annotated image
    output_path = output_dir / f"conf_{threshold:.2f}.jpg"

    cv2.imwrite(str(output_path), display_image)

    # Display result
    window_name = f"YOLOv8 - Confidence {threshold:.2f}"

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

    print("Saved:", output_path)
    print("Press any key to continue.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\nConfidence experiment completed!")
