# Environment Setup & Code Execution Guide

This document explains how to set up the Python environments (Workstation/GPU and Raspberry Pi 5) and how to run every code file in the `face_yolov` repository.

---

## 1. Environment Setup

### Option A: Workstation / GPU Environment (Full Pipeline: Training, Export, Quantization & Evaluation)
Used for running `scripts/` and `tests/` (requires Python 3.10+, tested with Python 3.11 on Ubuntu 24.04 LTS).

#### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/face_yolov.git
cd face_yolov
```

#### 2. Create & Activate Environment
**Using Conda (Recommended):**
```bash
conda create -n ai-ml python=3.11 -y
conda activate ai-ml
```

**Or Using Python `venv`:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
# Optional: install Flask as well if testing the web server locally
pip install -e ".[pi]"
```

#### 4. Verify Setup
```bash
make test        # Run all unit tests
make preflight   # Verify Stage 1 model checksum, runtime, and calibration manifest
```

---

### Option B: Raspberry Pi 5 Edge Environment (Standalone Inference & Web Stream)
Used for running `pi/` on Raspberry Pi OS (64-bit Bookworm) **without** PyTorch or Ultralytics (~150 MB environment).

#### 1. Install System Packages & Camera Permissions
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libgl1 libglib2.0-0 v4l-utils
sudo usermod -aG video $USER
```
*(Log out and log back in if your user was added to the `video` group).*

#### 2. Create Virtual Environment & Install `pi/requirements.txt`
```bash
cd face_yolov/pi
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Quick Run via `Makefile`

From the repository root (`face_yolov/`), you can run common workflows directly with `make`:

```bash
make install       # Install Python dependencies from requirements.txt
make test          # Run automated unit tests (tests/)
make preflight     # Run Stage 2 preflight verification (scripts/stage2_preflight.py)
make eval-stage1   # Evaluate Stage 1 FP32 model on frozen test split
make eval-stage2   # Compare FP32 vs INT8 models on test and OOD splits
make benchmark     # Run Gate 5 hardware latency & memory benchmark
make bench-pi      # Run standalone ONNX CPU benchmark (pi/bench_pi.py)
make run-pi        # Run live webcam face detection (pi/detect_webcam.py)
make run-pi-web    # Run Flask MJPEG web streaming server (pi/app_web.py)
make report        # Regenerate Pi 5 benchmark plots and validate research report
make clean         # Remove __pycache__, .pyc, and temporary cache files
```

---

## 3. How to Run Each Code File

### 3.1 Edge Deployment & Inference Files (`pi/`)

#### 1. `pi/app_web.py` — Real-Time Web Streaming Dashboard (MJPEG)
Starts a Flask web server streaming annotated camera frames with live FPS, latency, face count, and interactive Confidence / IoU sliders.
```bash
# Default (camera 0, port 5000, 4 threads)
python pi/app_web.py

# Custom camera, port, resolution, and CPU thread count (3 threads optimal on Pi 5)
python pi/app_web.py --camera 0 --width 640 --height 480 --conf 0.25 --iou 0.45 --threads 3 --port 5000
```
Open **`http://<device-ip>:5000`** in any browser on the same network.

#### 2. `pi/detect_webcam.py` — Live Webcam CLI (HDMI GUI or Headless)
Runs real-time face detection from a USB/CSI camera, video file, or RTSP stream.
```bash
# Local HDMI/desktop window (press 'q' or ESC to exit)
python pi/detect_webcam.py --camera 0 --conf 0.25 --iou 0.45 --threads 3

# Headless mode (e.g., over SSH): save 100 annotated frames to ./output_frames
python pi/detect_webcam.py --camera 0 --headless --save-dir ./output_frames --max-frames 100
```

#### 3. `pi/bench_pi.py` — Standalone Edge CPU & Thermal Benchmark
Benchmarks Preprocess / ONNX Inference / NumPy NMS stage latencies, compares FP32 vs. INT8 ONNX models, sweeps CPU thread counts (`1,2,3,4`), and records RAM + SoC temperature.
```bash
# Default benchmark (30 warmup runs, 150 timed runs, threads 1,2,3,4)
python pi/bench_pi.py

# Sustained 1,000-run thermal stress test
python pi/bench_pi.py --warmup 50 --runs 1000 --threads 3 \
  --out-json reports/pi_benchmark_stress_1000runs.json \
  --out-md reports/pi_benchmark_stress_1000runs.md
```

#### 4. `pi/face_detector.py` — Core ONNX Runtime Detector Module
Imported by `detect_webcam.py`, `app_web.py`, and `bench_pi.py`. Can also be called directly in Python:
```python
import cv2
from pi.face_detector import FaceDetector, draw_detections

detector = FaceDetector("pi/best_int8.onnx", conf_thresh=0.25, iou_thresh=0.45, num_threads=3)
frame = cv2.imread("data/ood/images/ood_01_small_faces.jpg")
detections, timings = detector.detect_timed(frame)
draw_detections(frame, detections)
```

---

### 3.2 Stage 1 Scripts: Data Audit, Split & FP32 Training (`scripts/`)

#### 1. `scripts/environment_report.py` (Gate 0)
Records OS, Python, PyTorch, CUDA, and GPU hardware details to `reports/environment.json` and `reports/environment_report.md`.
```bash
python scripts/environment_report.py
```

#### 2. `scripts/audit_dataset.py` (Gate 1)
Scans the raw Roboflow export for perceptual duplicates (`dHash`), cross-split leakage, and label integrity.
```bash
python scripts/audit_dataset.py --dataset Face-Detection.v2-datasetv2.yolov8 --out reports
```

#### 3. `scripts/make_dedup_split.py` (Gate 2)
Builds a leak-free, cluster-deduplicated `80/10/10` symlink dataset under `datasets/face_v2_dedup` and writes `configs/face_v2_dedup.yaml` + `splits/*.txt`.
```bash
python scripts/make_dedup_split.py --dataset Face-Detection.v2-datasetv2.yolov8 --out datasets/face_v2_dedup --seed 42
```

#### 4. `scripts/train_stage1.py` (Gate 3 & Stage 1 Training)
Trains a reproducible FP32 YOLOv8 model and saves `models/stage1/<name>_best.pt`.
```bash
# Smoke test (2 epochs)
python scripts/train_stage1.py --name S0 --imgsz 416 --batch 8 --epochs 2

# Full resolution ablation runs
python scripts/train_stage1.py --name E0 --imgsz 640 --batch 4 --epochs 100
python scripts/train_stage1.py --name E1 --imgsz 416 --batch 4 --epochs 100
python scripts/train_stage1.py --name E2 --imgsz 800 --batch 4 --epochs 100
```

#### 5. `scripts/evaluate.py` (Stage 1 Split Evaluation)
Evaluates a checkpoint on a frozen split (`test` by default) and writes `reports/metrics_<name>.json` and `reports/evaluation_<name>.md`.
```bash
python scripts/evaluate.py --weights models/stage1/best.pt --name E1 --imgsz 416 --split test
```

#### 6. `scripts/stage1_summary.py` (Gate 4 Model Selection & Summary)
Compares `E0`, `E1`, and `E2` on validation `mAP50-95`, promotes the winner to `models/stage1/best.pt`, and generates `reports/stage1_summary.md` and `reports/run_comparison.md`.
```bash
python scripts/stage1_summary.py
```

---

### 3.3 Stage 2 Scripts: Calibration, Export, Quantization & Gate 5 (`scripts/`)

#### 1. `scripts/make_calibration_manifest.py`
Samples 300 stratified calibration images strictly from `train` and `valid` splits (never `test`) into `calibration/images.json`.
```bash
python scripts/make_calibration_manifest.py --count 300 --seed 42
```

#### 2. `scripts/stage2_preflight.py` (Gate 4.5)
Validates `configs/stage2.json`, verifies `models/stage1/best.pt` against `models/stage1/best.sha256`, and confirms zero test-split leakage in `calibration/images.json`.
```bash
python scripts/stage2_preflight.py --config configs/stage2.json
```

#### 3. `scripts/export_stage2.py` (Export X0)
Exports `models/stage1/best.pt` to a simplified static FP32 ONNX graph at `exports/stage2/onnxruntime/best.onnx`.
```bash
python scripts/export_stage2.py --config configs/stage2.json
```

#### 4. `scripts/quantize_ptq.py` (PTQ Q0)
Runs ONNX Runtime static QDQ MinMax quantization (`Conv`-isolated) using `calibration/images.json` and outputs `models/stage2/int8.onnx`.
```bash
python scripts/quantize_ptq.py --config configs/stage2.json
```

#### 5. `scripts/evaluate_stage2.py` (FP32 vs. INT8 Accuracy Gate)
Evaluates both `models/stage1/best.pt` (FP32) and `models/stage2/int8.onnx` (INT8) on the frozen `test` split and `data/ood`, writing `reports/stage2_fp32_vs_int8.md`.
```bash
python scripts/evaluate_stage2.py --config configs/stage2.json
```

#### 6. `scripts/bench_target.py` (Gate 5 Hardware Benchmark & Promotion)
Measures model latency and memory against `configs/stage2.json` budgets and promotes `models/stage2/int8.onnx` to `models/stage2/best_int8.onnx`.
```bash
python scripts/bench_target.py --config configs/stage2.json
```

#### 7. `scripts/stage2_summary.py`
Compiles all Stage 2 gate statuses and metrics into `reports/stage2_summary.md`.
```bash
python scripts/stage2_summary.py
```

---

### 3.4 Plotting, Report Validation & Shared Modules (`scripts/`)

#### 1. `scripts/plot_pi_benchmarks.py`
Generates the 3 Raspberry Pi 5 benchmark figures (`reports/pi_thread_scaling_thermal.png`, `reports/pi_burst_vs_endurance.png`, `reports/pi_stage_breakdown.png`) from `reports/pi_benchmark.json` and `reports/pi_benchmark_stress_1000runs.json`.
```bash
python scripts/plot_pi_benchmarks.py
```

#### 2. `scripts/check_report.py`
Validates that all image paths, sequential figure numbers, and LaTeX math delimiters in `docs/yolov8n_int8_ptq_pi5_report.md` are intact.
```bash
python scripts/check_report.py docs/yolov8n_int8_ptq_pi5_report.md
```

#### 3. `scripts/common.py` & `scripts/eval_core.py`
Internal shared library modules imported by the scripts above (`common.py` provides SHA-256 hashing, JSON I/O, and table formatting; `eval_core.py` provides IoU calculation, ground-truth loading, COCO size bucketing, and split evaluation).

---

### 3.5 Automated Unit Tests (`tests/`)

Run the entire test suite or individual test files using `unittest`:
```bash
# Run all tests
python -m unittest discover tests -v

# Run individual test modules
python -m unittest tests/test_face_detector.py -v
python -m unittest tests/test_detect_webcam.py -v
python -m unittest tests/test_bench_pi.py -v
python -m unittest tests/test_stage2_preflight.py -v
```
