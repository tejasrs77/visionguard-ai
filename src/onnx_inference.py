
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

MODEL_PATH = Path("yolov8n.onnx")
IMAGE_PATH = Path("data/images/test.png")
OUTPUT_PATH = Path("runs/phase5/onnx_detection.jpg")

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

if not IMAGE_PATH.exists():
    raise FileNotFoundError(IMAGE_PATH)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load class names only, not the PyTorch model
CLASS_NAMES = YOLO(str(MODEL_PATH)).names

# Create ONNX Runtime inference session
session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Execution provider:", session.get_providers())
print("Input name:", input_name)
print("Output name:", output_name)

# Read original image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise RuntimeError("Failed to read input image")

original_height, original_width = image.shape[:2]

# Letterbox preprocessing: preserve aspect ratio
scale = min(
    INPUT_SIZE / original_width,
    INPUT_SIZE / original_height
)

new_width = round(original_width * scale)
new_height = round(original_height * scale)

resized = cv2.resize(
    image,
    (new_width, new_height)
)

pad_x = (INPUT_SIZE - new_width) // 2
pad_y = (INPUT_SIZE - new_height) // 2

canvas = np.full(
    (INPUT_SIZE, INPUT_SIZE, 3),
    114,
    dtype=np.uint8
)

canvas[
    pad_y:pad_y + new_height,
    pad_x:pad_x + new_width
] = resized

# BGR -> RGB, HWC -> CHW, uint8 -> float32
rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

input_tensor = (
    rgb.transpose(2, 0, 1)
    .astype(np.float32) / 255.0
)

input_tensor = np.expand_dims(input_tensor, axis=0)
input_tensor = np.ascontiguousarray(input_tensor)

# ONNX inference
outputs = session.run(
    [output_name],
    {input_name: input_tensor}
)

predictions = outputs[0][0].T

print("Raw prediction shape:", predictions.shape)

boxes = []
scores = []
class_ids = []

# Convert raw predictions into detections
for prediction in predictions:
    class_scores = prediction[4:]

    class_id = int(np.argmax(class_scores))
    confidence = float(class_scores[class_id])

    if confidence < CONF_THRESHOLD:
        continue

    cx, cy, w, h = prediction[:4]

    # Convert xywh to xyxy
    x1 = cx - w / 2
    y1 = cy - h / 2

    # Remove letterbox padding and rescale
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    w = w / scale
    h = h / scale

    # Clip to original image dimensions
    x1 = float(np.clip(x1, 0, original_width))
    y1 = float(np.clip(y1, 0, original_height))

    x2 = float(np.clip(x1 + w, 0, original_width))
    y2 = float(np.clip(y1 + h, 0, original_height))

    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)

    if box_w == 0 or box_h == 0:
        continue

    boxes.append([x1, y1, box_w, box_h])
    scores.append(confidence)
    class_ids.append(class_id)

# Class-aware non-maximum suppression
selected_indices = []

for class_id in set(class_ids):
    indices = [
        i for i, value in enumerate(class_ids)
        if value == class_id
    ]

    class_boxes = [boxes[i] for i in indices]
    class_scores = [scores[i] for i in indices]

    kept = cv2.dnn.NMSBoxes(
        class_boxes,
        class_scores,
        CONF_THRESHOLD,
        IOU_THRESHOLD
    )

    for local_index in np.array(kept).reshape(-1):
        selected_indices.append(indices[int(local_index)])

# Draw detections
for index in selected_indices:
    x, y, w, h = boxes[index]

    x1 = int(x)
    y1 = int(y)
    x2 = int(x + w)
    y2 = int(y + h)

    class_id = class_ids[index]
    confidence = scores[index]

    label = f"{CLASS_NAMES[class_id]} {confidence:.2f}"

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    print(label)

print("Final detections:", len(selected_indices))

cv2.imwrite(str(OUTPUT_PATH), image)

# Resize preview to fit screen
height, width = image.shape[:2]
display_scale = min(1280 / width, 720 / height, 1.0)

preview = cv2.resize(
    image,
    (
        int(width * display_scale),
        int(height * display_scale)
    )
)

cv2.imshow("VisionGuard AI - ONNX Runtime", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Saved to:", OUTPUT_PATH)
