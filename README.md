
## ONNX Runtime CPU Benchmark

VisionGuard-AI supports YOLOv8n inference using both
PyTorch and ONNX Runtime.

### Benchmark Environment

- CPU: AMD Ryzen 5 5500U
- PyTorch: 2.14.1+cpu
- ONNX Runtime: 1.31.0
- Model: YOLOv8n
- Input: 1 x 3 x 640 x 640
- Warm-up iterations: 10
- Timed iterations: 100
- Execution: CPU

### Results

| Metric | PyTorch | ONNX Runtime |
|---|---:|---:|
| Average latency | 168.52 ms | 94.56 ms |
| Median latency | 166.98 ms | 96.11 ms |
| P95 latency | 196.59 ms | 104.83 ms |
| Inference FPS | 5.93 | 10.58 |

ONNX Runtime achieved a 1.78x model-inference speedup
and approximately 43.9% lower average latency.

These results are from a single CPU benchmark using
identical synthetic input tensors. They exclude
preprocessing, NMS, visualization, and video I/O.

### Run the Benchmark

```powershell
python src/export_onnx.py
python src/onnx_inference.py
python src/benchmark_inference.py
```
