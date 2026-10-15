
<div align="center">

# VisionGuard-AI

### Computer Vision | Object Tracking | Camera Geometry | Visual Odometry | State Estimation

**A modular computer vision project exploring object perception, multi-object tracking, 3D geometry, inference optimization, and motion estimation.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111F68?style=flat-square)](https://docs.ultralytics.com/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai/)

[Source Code](https://github.com/tejasrs77/visionguard-ai) · [Experiments](#implemented-modules) · [Benchmarks](#experimental-results)

</div>

---

## Overview

VisionGuard-AI is a Python-based computer vision experimentation project that combines deep learning, classical computer vision, multi-object tracking, camera geometry, and state estimation.

The project was developed incrementally across eight technical phases, progressing from YOLOv8 image detection to visual odometry and Kalman filtering.

It includes standalone experiments and selected integrated pipelines. It is intended for learning, evaluation, and reproducible demonstrations rather than production autonomous navigation.

### Core Capabilities

- Object detection on images and videos using YOLOv8
- Bounding-box analysis, Intersection over Union, and Non-Maximum Suppression
- Multi-object tracking with ByteTrack
- Instance segmentation and human pose estimation
- YOLOv8 ONNX export, CPU inference, and latency benchmarking
- Camera calibration, feature matching, relative pose estimation, and triangulation
- Python publisher/subscriber experiments inspired by ROS 2
- Synthetic visual odometry and Kalman state estimation

---

## Architecture

VisionGuard-AI consists of several related computer vision pipelines.

### 1. Object Perception and Tracking

```text
Image / Video Input
        |
        v
   OpenCV Loader
        |
        v
 YOLOv8 Inference
        |
        v
Bounding Boxes + Confidence
        |
        v
  ByteTrack Tracker
        |
        v
Track IDs + Trajectories
        |
        v
Visualization + Statistics
```

### 2. Camera Geometry and Visual Odometry

```text
Camera Frames / Synthetic 3D Scene
              |
              v
       Feature Matching
              |
              v
   Essential Matrix Estimation
              |
              v
       Relative Camera Pose
              |
              v
    Motion / Pose Evaluation
              |
              v
      Trajectory Visualization
```

The synthetic visual odometry experiment uses known landmark correspondences and ground-truth step lengths to resolve monocular scale ambiguity.

### 3. ONNX Inference Optimization

```text
YOLOv8n PyTorch Model
           |
           v
       ONNX Export
           |
           v
    ONNX Runtime CPU
           |
           v
      Raw Predictions
           |
           v
 Postprocessing + NMS
           |
           v
   Detection Predictions
```

### 4. Publisher/Subscriber Simulation

```text
Video Frame Publisher
          |
          v
     Python Queue
          |
          v
YOLOv8 + ByteTrack Worker
          |
          v
     Python Queue
          |
          v
  Tracking Result Receiver
          |
          v
    Output Video
```

This pipeline uses Python threading and queues. It is ROS 2-inspired but does not implement native ROS 2 nodes, DDS communication, or ROS message types.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch, Ultralytics YOLOv8 |
| Computer Vision | OpenCV |
| Object Tracking | ByteTrack |
| Numerical Computing | NumPy |
| Model Export | ONNX |
| Optimized Inference | ONNX Runtime |
| Visualization | Matplotlib |
| Camera Geometry | ORB, Essential Matrix, RANSAC, Triangulation |
| State Estimation | OpenCV Kalman Filter |
| Messaging Simulation | Python Queue, Threading |
| Development Environment | Windows 11, VS Code, Python Virtual Environment |
| Version Control | Git, GitHub |

---

## Implemented Modules

### Phase 1 — Computer Vision and YOLOv8

Introduced image processing, video handling, and pretrained YOLOv8 object detection.

**Scripts**

- `src/image_basics.py`
- `src/video_basics.py`
- `src/bounding_boxes.py`
- `src/yolo_image_detection.py`
- `src/yolo_video_detection.py`

**Concepts:** image arrays, video frames, bounding boxes, class labels, confidence scores, and inference.

### Phase 2 — Detection Evaluation

Implemented and explored key object-detection evaluation concepts.

**Scripts**

- `src/iou_calculation.py`
- `src/confidence_experiment.py`
- `src/nms_implementation.py`
- `src/precision_recall.py`
- `src/yolo_evaluation.py`

**Concepts:** Intersection over Union, confidence thresholds, Non-Maximum Suppression, precision, recall, and detection metrics.

The small COCO8 evaluation is a functionality demonstration, not a representative assessment of generalization performance.

### Phase 3 — Multi-Object Tracking

Applied ByteTrack to YOLOv8 video detections and analyzed track trajectories.

**Scripts**

- `src/bytetrack_tracking.py`
- `src/trajectory_tracking.py`
- `src/tracking_statistics.py`

**Concepts:** detection-to-track association, track IDs, trajectories, and frame-level statistics.

### Phase 4 — Segmentation and Pose Estimation

Implemented instance segmentation and human pose estimation with pretrained YOLOv8 models.

**Scripts**

- `src/segmentation_image.py`
- `src/segmentation_video.py`
- `src/pose_image.py`
- `src/pose_video.py`

**Models**

- `yolov8n-seg.pt`
- `yolov8n-pose.pt`

**Concepts:** pixel-level object masks, keypoints, and skeletal connections.

### Phase 5 — ONNX Deployment and Optimization

Exported YOLOv8n to ONNX, performed ONNX Runtime CPU inference, and compared inference latency with PyTorch.

**Scripts**

- `src/export_onnx.py`
- `src/onnx_inference.py`
- `src/benchmark_inference.py`

**Concepts:** model export, inference backends, tensor preprocessing, postprocessing, and performance benchmarking.

### Phase 6 — Camera Geometry and Structure from Motion

Explored camera projection, synthetic camera calibration, ORB feature matching, essential matrix estimation, relative camera pose, and triangulation.

**Scripts**

- `src/camera_projection.py`
- `src/generate_calibration_dataset.py`
- `src/camera_calibration.py`
- `src/orb_feature_matching.py`
- `src/relative_camera_pose.py`
- `src/triangulation_3d.py`

**Concepts:** camera intrinsic matrices, projection geometry, calibration, epipolar geometry, and 3D reconstruction.

Camera calibration used synthetic chessboard images. The relative-pose and triangulation experiments used generated views and should not be interpreted as metrically validated real-world reconstruction.

### Phase 7 — Publisher/Subscriber Communication

Built Python messaging demonstrations inspired by robotics publisher/subscriber architectures.

**Scripts**

- `src/publisher_subscriber_demo.py`
- `src/video_frame_publisher.py`
- `src/yolo_subscriber.py`
- `src/tracking_publisher_subscriber.py`

**Concepts:** producer/consumer communication, queues, threads, frame processing, and message passing.

This phase does not require or install ROS 2.

### Phase 8 — Visual Odometry and State Estimation

Implemented temporal feature matching, relative-motion diagnostics, synthetic visual odometry, and Kalman filtering.

**Scripts**

- `src/temporal_feature_matching.py`
- `src/relative_motion_video.py`
- `src/motion_diagnostics.py`
- `src/synthetic_visual_odometry.py`
- `src/kalman_state_estimation.py`

**Concepts:** temporal correspondences, relative camera pose, trajectory estimation, pose evaluation, Kalman prediction, and measurement correction.

The Kalman filter is a separate synthetic 2D tracking experiment; it is not yet fused with the visual odometry trajectory.

---

## Experimental Results

### 1. ONNX Runtime CPU Benchmark

**Benchmark environment**

| Parameter | Value |
|---|---|
| Processor | AMD Ryzen 5 5500U |
| Model | YOLOv8n |
| Input shape | 1 × 3 × 640 × 640 |
| Warm-up iterations | 10 |
| Timed iterations | 100 |
| Execution | CPU |
| ONNX Runtime | 1.31.0 |
| PyTorch package | 2.14.1 |

**Measured results**

| Metric | PyTorch | ONNX Runtime |
|---|---:|---:|
| Average latency | 168.52 ms | 94.56 ms |
| Median latency | 166.98 ms | 96.11 ms |
| P95 latency | 196.59 ms | 104.83 ms |
| Inference FPS | 5.93 | 10.58 |

**Outcome:** ONNX Runtime achieved a **1.78× model-inference speedup**, with approximately **43.9% lower average latency**.

These results are from a single CPU benchmark using identical synthetic input tensors. They exclude preprocessing, NMS, visualization, and video I/O.

### 2. Multi-Object Tracking

A video tracking experiment processed 704 frames and reported:

| Metric | Result |
|---|---:|
| Processed frames | 704 |
| Unique track IDs generated | 272 |
| Maximum simultaneous tracks | 23 |
| Average tracks per frame | 15.27 |

Track IDs are diagnostic outputs. They are not equivalent to the number of distinct physical objects, and these statistics do not establish tracking accuracy.

### 3. Synthetic Camera Calibration

The synthetic calibration experiment used 20 generated chessboard views.

| Metric | Result |
|---|---:|
| Calibration views | 20 |
| Reprojection RMS | 0.09755 pixels |
| Approximate recovered focal lengths | 900.11 px, 900.59 px |

The experiment validates the calibration procedure under controlled synthetic conditions, not physical-camera accuracy.

### 4. Synthetic Visual Odometry

The improved synthetic visual odometry experiment produced:

| Metric | Result |
|---|---:|
| Camera frames | 30 |
| Motion-estimation steps | 29 |
| Successful pose estimates | 28 |
| Failed pose estimates | 1 |
| Pose-estimation success rate | 96.55% |
| Trajectory RMSE | 0.5658 simulation units |
| Final position error | 0.6941 simulation units |

The experiment uses known landmark correspondences and ground-truth translation step lengths. The trajectory therefore has ground-truth-assisted scale and is not a scale-independent monocular VO system.

### 5. Kalman State Estimation

A synthetic 2D constant-velocity tracking experiment evaluated noisy measurements against Kalman-filtered estimates.

| Metric | Result |
|---|---:|
| Simulated frames | 100 |
| Measurement RMSE | 3.7346 |
| Kalman filter RMSE | 1.8043 |
| RMSE reduction | 51.69% |
| True velocity | [1.5, 0.8] |
| Final estimated velocity | [1.451, 0.689] |

The Kalman filter reduced trajectory-position error in this controlled simulation.

---

## Installation

### Prerequisites

- Windows 11
- Python and pip
- Git
- VS Code (recommended)

The project was developed and tested using a Windows Python virtual environment. Ubuntu, WSL, and ROS 2 are not required.

### 1. Clone the repository

```powershell
git clone https://github.com/tejasrs77/visionguard-ai.git
cd visionguard-ai
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Verify dependencies

```powershell
python -m pip check
```

The requirements file records package versions from the development environment. Installation compatibility on other Python versions or systems has not been independently validated.

---

## Running the Experiments

Run commands from the repository root with the virtual environment activated.

### YOLOv8 Image Detection

Place a test image at the input path expected by the script, then run:

```powershell
python src/yolo_image_detection.py
```

### YOLOv8 Video Detection

Provide the expected input video, then run:

```powershell
python src/yolo_video_detection.py
```

### ByteTrack Tracking

```powershell
python src/bytetrack_tracking.py
```

### Segmentation

```powershell
python src/segmentation_image.py
python src/segmentation_video.py
```

### Pose Estimation

```powershell
python src/pose_image.py
python src/pose_video.py
```

### ONNX Export and Benchmark

```powershell
python src/export_onnx.py
python src/onnx_inference.py
python src/benchmark_inference.py
```

### Camera Calibration

```powershell
python src/generate_calibration_dataset.py
python src/camera_calibration.py
```

### Relative Camera Pose and Triangulation

These scripts require the corresponding image inputs:

```powershell
python src/relative_camera_pose.py
python src/triangulation_3d.py
```

### Publisher/Subscriber Demonstration

```powershell
python src/publisher_subscriber_demo.py
python src/tracking_publisher_subscriber.py
```

### Synthetic Visual Odometry

```powershell
python src/synthetic_visual_odometry.py
```

### Kalman State Estimation

```powershell
python src/kalman_state_estimation.py
```

Some scripts expect local images, videos, or generated intermediate files. Check the input paths in each script before execution.

---

## Repository Structure

```text
visionguard-ai/
|
|-- src/
|   |-- YOLO detection scripts
|   |-- Tracking and trajectory scripts
|   |-- Segmentation and pose scripts
|   |-- ONNX inference and benchmarking scripts
|   |-- Camera calibration and SfM scripts
|   |-- Publisher/subscriber scripts
|   |-- Visual odometry and Kalman scripts
|
|-- README.md
|-- requirements.txt
|-- .gitignore
```

Local directories such as `data/`, `runs/`, and `outputs/` are excluded from version control.

---

## Limitations

VisionGuard-AI is an experimental computer vision repository rather than a production robotics platform.

- Detection and tracking rely on pretrained YOLOv8 models.
- Tracking counts are not ground-truth identity metrics.
- COCO8 is too small for representative model evaluation.
- Camera calibration uses synthetic chessboard images.
- Structure-from-motion experiments do not establish metric reconstruction accuracy.
- Synthetic visual odometry uses known correspondences and ground-truth scale.
- Kalman filtering is demonstrated separately from camera-motion estimation.
- Publisher/subscriber communication is simulated using Python queues, not native ROS 2.
- ONNX performance measurements are specific to one CPU benchmark.

---

## Future Improvements

Potential extensions include:

- Evaluation on larger, annotated datasets
- Tracking accuracy evaluation with ground-truth identities
- Calibration using real camera images
- Robust visual odometry on actual moving-camera sequences
- Visual-inertial state estimation
- Integration of Kalman filtering with tracking or camera motion
- Native ROS 2 nodes and message transport
- Automated testing and continuous integration
- End-to-end video-processing benchmarks

---

## Author

**Tejas R S**

Artificial Intelligence and Machine Learning

[GitHub Profile](https://github.com/tejasrs77)

---

<div align="center">

**VisionGuard-AI — Exploring perception, tracking, geometry, and motion estimation through practical computer vision experiments.**

</div>
