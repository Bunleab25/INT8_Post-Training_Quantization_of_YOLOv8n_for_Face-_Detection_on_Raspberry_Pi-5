# YOLOv8 Face Detection: Two-Stage Lifecycle (FP32 & INT8)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Framework: Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-8.4.162-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![Runtime: ONNX Runtime](https://img.shields.io/badge/Runtime-ONNX_Runtime-orange.svg)](https://onnxruntime.ai/)

A production-grade, scientifically rigorous machine learning pipeline for **single-class face detection** based on YOLOv8. The project implements a complete two-stage lifecycle governed by strict quality gates:

1. **Stage 1 (FP32 Training & Data Engineering):** Comprehensive dataset audit, perceptual duplicate removal (0 cross-split leakage), controlled resolution ablation (416px, 640px, 800px), and frozen baseline checkpoint selection.
2. **Stage 2 (Quantization & Edge Export):** FP32 base reference profiling, train/val-only calibration, Conv-isolated Post-Training Quantization (PTQ) to INT8, multi-slice evaluation on test and OOD sets, and Gate 5 hardware latency/memory verification.

---

## Architecture & Lifecycle Pipeline

```
Gate 0: Environment Verification (GPU, CUDA, PyTorch)
   │
   ▼
Gate 1: Dataset Audit (40 cross-split duplicate clusters detected)
   │
   ▼
Gate 2: Cluster-Aware Deduplication (Zero cross-split leakage, 80/10/10 split)
   │
   ▼
Gate 3: Pipeline Smoke Test (S0 - Loss convergence & VRAM fit)
   │
   ▼
Stage 1 FP32 Training & Benchmark (E0: 640px, E1: 416px, E2: 800px)
   │
   ▼
Gate 4: Frozen Base Reference Selection (E1 @ 416px, mAP50-95: 0.6541)
   │
   ▼
Gate 4.5: Stage 2 Preflight (Lock runtime, calibration manifest, OOD harness)
   │
   ▼
Post-Training Quantization (PTQ INT8 with MinMax calibration)
   │
   ▼
Gate 5: Hardware & Accuracy Gate (Latency <= 20ms, Zero accuracy drop)
   │
   ▼
Production Model Promotion (best_int8.onnx)
```

---

## Benchmark Results

### 1. Stage 1 FP32 Model Selection (Frozen Test Split)

| Run | Resolution | Peak VRAM | Inference (GPU) | Val mAP50-95 | Test mAP50 | Test mAP50-95 | Test Recall (@0.25) |
|---|---|---|---|---|---|---|---|
| **E0** | 640×640 | 0.858 GiB | 4.83 ms | 0.7430 | 0.9884 | 0.6377 | 0.9912 |
| **E1 (Champion)** | **416×416** | **0.415 GiB** | **2.56 ms** | **0.7424** | **0.9950** | **0.6541** | **0.9825** |
| **E2** | 800×800 | 1.308 GiB | 7.29 ms | 0.7303 | 0.9946 | 0.6668 | 0.9825 |

> **Key Finding:** The source dataset is natively 416×416. Upscaling to 640px or 800px merely interpolated pixels without providing additional spatial information, while doubling/tripling latency and memory. **E1 (416px)** was selected and frozen to `models/stage1/best.pt`.

---

### 2. Stage 2: INT8 Judged Against FP32 Reference

Evaluation conducted on the identical frozen test split (114 images, 114 ground truth faces):

| Metric | FP32 Reference (`best.pt`) | INT8 Model (`best_int8.onnx`) | Delta ($\Delta$) | Status |
|---|---|---|---|---|
| **mAP50** | `0.9950` | `0.9893` | -0.0057 | PASS |
| **mAP50-95** | `0.6541` | `0.6549` | **+0.0008** | **PASS (Zero drop)** |
| **Precision** | `0.9993` | `0.9912` | -0.0081 | PASS |
| **Recall (@ conf=0.25)** | `0.9825` (112/114) | `0.9912` (113/114) | **+0.0087** | PASS |
| **Large Face Recall** | `0.9823` (111/113) | `0.9912` (112/113) | +0.0089 | PASS |
| **Medium Face Recall** | `1.0000` (1/1) | `1.0000` (1/1) | 0.0000 | PASS |
| **Model Size** | **11.6 MB** | **3.3 MB** | **3.5× smaller** | PASS |
| **Hardware Latency** | 2.56 ms (GPU) | 16.4 ms (CPU) | $\le 20.0\text{ ms}$ budget | **PASS (Gate 5)** |

---

## Repository Structure

```
├── calibration/
│   └── images.json                     # 300-image calibration manifest (Train/Val ONLY)
├── configs/
│   ├── face_v2_dedup.yaml              # Deduplicated YOLOv8 dataset definition
│   ├── ood_eval.yaml                   # Out-of-distribution evaluation definition
│   ├── stage2.json                     # Active Stage 2 deployment configuration
│   └── stage2.template.json            # Template for deployment & target specs
├── data/
│   └── ood/                            # Out-of-distribution test suite & annotations
├── docs/
│   ├── technical_report.md             # Full research & engineering technical report
│   ├── pi_benchmark_guide.md           # Step-by-step Raspberry Pi 5 benchmarking guide
│   ├── stage1_architecture.md          # Stage 1 FP32 training & audit specification
│   └── stage2_architecture.md          # Stage 2 INT8 PTQ & hardware gate specification
├── exports/
│   └── stage2/onnxruntime/             # Exported FP32 ONNX model & metadata
├── models/
│   ├── stage1/
│   │   ├── best.pt                     # [Stage 1 Winner] Frozen FP32 model checkpoint
│   │   ├── best.sha256                 # Cryptographic verification checksum
│   │   ├── best_info.json              # Selection criteria & benchmark metrics
│   │   └── best_args.yaml              # Training hyperparameter manifest
│   └── stage2/
│       ├── best_int8.onnx              # [Stage 2 Winner] Production INT8 quantized model
│       ├── best_int8.onnx.sha256       # Production model checksum
│       └── int8_info.json              # Quantization & calibrator metadata
├── pi/                                 # Standalone Raspberry Pi 5 edge deployment (no PyTorch)
│   ├── best_int8.onnx                  # Bundled INT8 ONNX model for self-contained deployment
│   ├── face_detector.py                # Pure NumPy NMS, letterboxing & ONNX Runtime CPU engine
│   ├── detect_webcam.py                # Live HDMI / headless webcam detection CLI
│   ├── app_web.py                      # Flask MJPEG real-time web stream & telemetry dashboard
│   ├── bench_pi.py                     # Automated FP32 vs INT8 & CPU thread scaling benchmark
│   ├── requirements.txt                # Minimal Pi dependencies (numpy, opencv, onnxruntime, flask)
│   └── README.md                       # Raspberry Pi 5 setup & usage guide
├── reports/                            # Generated evidence for Gate 0 through Gate 5
├── scripts/
│   ├── audit_dataset.py                # Gate 1: Perceptual hash & leakage audit
│   ├── make_dedup_split.py             # Gate 2: Cluster-aware deduplicated splitting
│   ├── train_stage1.py                 # Reproducible FP32 model training
│   ├── evaluate.py                     # Multi-metric evaluation on frozen splits
│   ├── stage1_summary.py               # Automated model selector & report generator
│   ├── stage2_preflight.py             # Gate 4.5: Stage 2 prerequisite validator
│   ├── make_calibration_manifest.py    # Builds train/val-only calibration manifest
│   ├── export_stage2.py                # Exports best.pt to optimized ONNX
│   ├── quantize_ptq.py                 # Post-training INT8 quantization engine
│   ├── evaluate_stage2.py              # Evaluates FP32 vs INT8 on test & OOD splits
│   ├── bench_target.py                 # Gate 5: Hardware latency & memory benchmark
│   └── stage2_summary.py               # Generates final Stage 2 summary report
├── splits/                             # Frozen image split manifests (train/val/test)
├── tests/
│   ├── test_bench_pi.py                # Edge benchmark & telemetry unit tests
│   ├── test_detect_webcam.py           # CLI & camera source resolution unit tests
│   ├── test_face_detector.py           # Letterbox, pure-NumPy NMS & coordinate scaling unit tests
│   └── test_stage2_preflight.py        # Stage 2 preflight & calibration guard unit tests
├── requirements.txt                    # Python package dependencies
├── LICENSE                             # MIT License
└── NOTICE                              # Dataset attribution & licensing notice
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/<your-username>/face_yolov.git
cd face_yolov
pip install -r requirements.txt
```

### 2. Verify Preflight & Tests

```bash
python -m unittest discover tests
python scripts/stage2_preflight.py
```

### 3. Edge Deployment on Raspberry Pi 5 (`pi/`)

Run lightweight CPU inference via ONNX Runtime without installing PyTorch or Ultralytics (see [`pi/README.md`](pi/README.md)):

```bash
# Live webcam detection (HDMI display or headless frame capture)
python pi/detect_webcam.py --camera 0

# Real-time MJPEG web streaming dashboard (view on any browser at http://<ip>:5000)
python pi/app_web.py --camera 0 --port 5000

# Run hardware benchmark suite (FP32 vs INT8, stage breakdown, 1-4 CPU threads, thermals)
python pi/bench_pi.py --warmup 30 --runs 150
```

See [`docs/pi_benchmark_guide.md`](docs/pi_benchmark_guide.md) for the step-by-step Raspberry Pi 5 measurement protocol and [`docs/technical_report.md`](docs/technical_report.md) for the full research report.

### 4. Run Inference with Ultralytics Python API

```python
from ultralytics import YOLO

# Load the production INT8 model
model = YOLO("models/stage2/best_int8.onnx", task="detect")

# Run inference
results = model.predict(source="data/ood/images/ood_01_small_faces.jpg", imgsz=416, conf=0.25)
print(f"Detected {len(results[0].boxes)} faces.")
```

### 5. Reproduce the Full Two-Stage Lifecycle

#### Stage 1: Data Audit, Split & Training
```bash
python scripts/environment_report.py
python scripts/audit_dataset.py
python scripts/make_dedup_split.py
python scripts/train_stage1.py --name E1 --imgsz 416 --batch 4 --epochs 100
python scripts/stage1_summary.py
```

#### Stage 2: Quantization, Evaluation & Promotion
```bash
python scripts/make_calibration_manifest.py --count 300
python scripts/export_stage2.py
python scripts/quantize_ptq.py
python scripts/evaluate_stage2.py
python scripts/bench_target.py
python scripts/stage2_summary.py
```

---

## License & Attribution

- **Code:** Licensed under the [MIT License](LICENSE).
- **Dataset Attribution:** See [NOTICE](NOTICE) for Roboflow Face-Detection v2 dataset attribution (licensed under CC BY 4.0).
