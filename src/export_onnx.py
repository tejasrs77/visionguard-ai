
from pathlib import Path

import onnx
from ultralytics import YOLO

# Load the pretrained PyTorch detection model
model = YOLO("yolov8n.pt")

# Export to ONNX
exported_path = model.export(
    format="onnx",
    imgsz=640,
    dynamic=False,
    simplify=True,
    opset=17
)

# Verify that the exported file exists
onnx_path = Path(exported_path)

if not onnx_path.exists():
    raise FileNotFoundError(
        f"ONNX export failed: {onnx_path}"
    )

# Validate the ONNX computational graph
onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)

print("\nONNX export successful!")
print("Model path:", onnx_path.resolve())
print("Model size (MB):", round(
    onnx_path.stat().st_size / (1024 * 1024), 2
))
print("ONNX graph validation: PASSED")
