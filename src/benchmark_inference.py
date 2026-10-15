
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

# Configuration
PT_MODEL = Path("yolov8n.pt")
ONNX_MODEL = Path("yolov8n.onnx")

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100
INPUT_SIZE = 640

torch.set_num_threads(4)

# Create one identical input for both models
rng = np.random.default_rng(42)

input_array = rng.random(
    (1, 3, INPUT_SIZE, INPUT_SIZE),
    dtype=np.float32
)

torch_input = torch.from_numpy(input_array)

# Load PyTorch model
yolo = YOLO(str(PT_MODEL))
pt_model = yolo.model
pt_model.eval()
pt_model.to("cpu")

# Load ONNX Runtime model
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 4
session_options.inter_op_num_threads = 1

session = ort.InferenceSession(
    str(ONNX_MODEL),
    sess_options=session_options,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def benchmark_pytorch():
    times = []

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            pt_model(torch_input)

        for _ in range(BENCHMARK_RUNS):
            start = time.perf_counter()

            pt_model(torch_input)

            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

    return np.array(times)


def benchmark_onnx():
    times = []

    for _ in range(WARMUP_RUNS):
        session.run(
            [output_name],
            {input_name: input_array}
        )

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()

        session.run(
            [output_name],
            {input_name: input_array}
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return np.array(times)


def print_results(name, times):
    avg_ms = float(np.mean(times))
    median_ms = float(np.median(times))
    p95_ms = float(np.percentile(times, 95))

    fps = 1000 / avg_ms

    print(f"\n{name}")
    print("-" * 35)
    print(f"Average latency: {avg_ms:.2f} ms")
    print(f"Median latency:  {median_ms:.2f} ms")
    print(f"P95 latency:     {p95_ms:.2f} ms")
    print(f"Inference FPS:   {fps:.2f}")

    return avg_ms, fps


print("Hardware: AMD Ryzen 5 5500U")
print("Device: CPU")
print("PyTorch version:", torch.__version__)
print("ONNX Runtime version:", ort.__version__)
print("Input shape:", input_array.shape)
print("Warm-up runs:", WARMUP_RUNS)
print("Benchmark runs:", BENCHMARK_RUNS)

print("\nBenchmarking PyTorch...")
pt_times = benchmark_pytorch()

print("Benchmarking ONNX Runtime...")
onnx_times = benchmark_onnx()

pt_latency, pt_fps = print_results(
    "PyTorch Results",
    pt_times
)

onnx_latency, onnx_fps = print_results(
    "ONNX Runtime Results",
    onnx_times
)

speedup = pt_latency / onnx_latency

print("\nFINAL COMPARISON")
print("=" * 35)
print(f"PyTorch FPS:     {pt_fps:.2f}")
print(f"ONNX Runtime FPS:{onnx_fps:9.2f}")
print(f"ONNX speed ratio: {speedup:.2f}x")

if speedup > 1:
    print("ONNX Runtime was faster.")
elif speedup < 1:
    print("PyTorch was faster.")
else:
    print("Both had similar average latency.")
