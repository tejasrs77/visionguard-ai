
# VisionGuard-AI
# Phase 2 - YOLOv8 Model Evaluation

from ultralytics import YOLO

# STEP 1: Load pretrained model
model = YOLO("yolov8n.pt")

# STEP 2: Run evaluation on labeled sample dataset
metrics = model.val(
    data="coco8.yaml",
    split="val",
    imgsz=640,
    batch=1,
    device="cpu",
    plots=True,
    project="runs",
    name="phase2_evaluation",
    exist_ok=True
)

# STEP 3: Extract evaluation metrics
precision = metrics.box.mp
recall = metrics.box.mr

map50 = metrics.box.map50
map50_95 = metrics.box.map

# STEP 4: Print results
print("\nVisionGuard-AI Evaluation Results")
print("--------------------------------")

print(f"Mean Precision: {precision:.4f}")
print(f"Mean Recall: {recall:.4f}")
print(f"mAP@0.5: {map50:.4f}")
print(f"mAP@0.5:0.95: {map50_95:.4f}")

print("--------------------------------")
print("Evaluation completed successfully!")
