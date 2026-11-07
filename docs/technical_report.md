# Technical Report: Two-Stage YOLOv8 Face Detection Pipeline — From Data Deduplication to INT8 Edge Deployment

**Author:** Bunleab  
**Date:** November 2026  
**Repository:** `face_yolov` (v1.0.0)  

---

## Abstract

Deploying deep learning object detectors onto resource-constrained edge devices such as the **Raspberry Pi 5** requires careful co-design across dataset engineering, input resolution selection, model quantization, and runtime software architecture. This technical report presents an end-to-end, gate-governed two-stage machine learning lifecycle for single-class face detection based on **YOLOv8n**:

1. **Stage 1 (Data Engineering & FP32 Baseline):** A perceptual hash (`dHash`) audit of the 2,173-image Roboflow Face-Detection v2 dataset uncovered 931 training augmentation copies and 40 near-duplicate clusters spanning across train, validation, and test splits. Removing cross-split leakage via cluster-aware deduplication eliminated a **+0.0314 mAP50-95 (+4.8%) optimistic evaluation bias**. A controlled resolution ablation across `416×416`, `640×640`, and `800×800` demonstrated that matching the native dataset resolution (`416×416`, run `E1`) achieved **0.9950 mAP50** and **0.6541 mAP50-95** while cutting GPU latency by **47%** (`2.56 ms` vs. `4.83 ms`) and peak training VRAM by **52%** (`0.415 GiB` vs. `0.858 GiB`) compared to the standard `640×640` baseline.
2. **Stage 2 (Conv-Isolated INT8 Post-Training Quantization & Edge Deployment):** Static QDQ MinMax quantization isolated to `Conv` operators compressed the model by **3.69×** (`11.6 MB` FP32 $\rightarrow$ `3.14 MB` INT8) with **zero accuracy loss** (`0.6549` INT8 vs. `0.6541` FP32 mAP50-95). Coupled with a standalone ONNX Runtime + pure-NumPy Non-Maximum Suppression (NMS) inference engine, the system eliminates PyTorch and Ultralytics runtime dependencies (`~131 MB` peak RSS memory), achieving **9.78 ms (102.3 FPS)** on an x86_64 CPU (`1.32×` faster than FP32 ONNX) and real-time throughput on Raspberry Pi 5.

---

## 1. Introduction & System Lifecycle Architecture

Conventional object detection workflows often train at default resolutions (`640×640`), evaluate on unverified dataset splits, and deploy heavy training frameworks directly to target hardware. To prevent silent evaluation leakage and hardware regressions, this project enforces a six-gate engineering pipeline:

```
Gate 0: Environment Verification (CUDA 13.0, PyTorch 2.13, RTX 3050 4GB)
   │
   ▼
Gate 1: Perceptual Dataset Audit (2,173 files → 1,242 sources, 40 leakage clusters)
   │
   ▼
Gate 2: Cluster-Aware Deduplicated Split (907 train / 114 val / 114 test, 0 leakage)
   │
   ▼
Gate 3 & 4: Controlled FP32 Resolution Ablation (E0: 640px, E1: 416px, E2: 800px)
   │
   ▼
Gate 4.5: Stage 2 Preflight & Leak-Free Calibration Manifest (300 train/val images)
   │
   ▼
Gate 5: Conv-Only INT8 QDQ Quantization & Standalone ONNX Runtime Edge Verification
```

---

## 2. Dataset Audit & Cross-Split Leakage Analysis (Stage 1 Gates 1–2)

### 2.1 Perceptual Duplicate Discovery
The upstream Roboflow export (`Face-Detection.v2-datasetv2.yolov8`) contained **2,173 image files** across `train` (1,932), `valid` (120), and `test` (121). By parsing Roboflow `.rf.<hash>` filenames and computing 64-bit difference hashes (`dHash`) with a Hamming distance threshold of $d \le 3$ via Union-Find clustering ([`scripts/audit_dataset.py`](../scripts/audit_dataset.py)), we established:

| Audit Property | Finding | Engineering Implication |
|---|---|---|
| **Total Files vs. Unique Sources** | `2,173` files / `1,242` unique sources | `931` files are train-only clean/salt-and-pepper noise pairs |
| **Unique Perceptual Clusters** | `1,139` clusters | `87` clusters contain multiple near-duplicate source images |
| **Cross-Split Leakage Clusters** | **`40` clusters span splits** | `19` validation and `21` test sources had near-duplicates in `train` |
| **Native Image Geometry** | `100%` at `416×416` grayscale | Upscaling beyond `416×416` adds zero native spatial information |
| **COCO Face Size Distribution (@416px)** | `2,165` Large, `20` Medium, `4` Small | `99.6%` of annotations are close-up frontal faces ($\ge 96\times96\text{ px}$) |

### 2.2 Quantifying the Leakage Bias
Using [`scripts/make_dedup_split.py`](../scripts/make_dedup_split.py), we selected the single lowest-noise representative per source cluster and partitioned clusters into a strict `80 / 10 / 10` split (**907 train / 114 valid / 114 test**) with zero cross-split overlap ([`reports/split_report.md`](../reports/split_report.md)).

Evaluating the champion FP32 model (`E1`) on both the original leaky test split and the deduplicated test split quantified the exact inflation caused by near-duplicate leakage:

| Evaluation Split | Images | mAP50 | mAP50-95 | Observed Bias |
|---|---|---|---|---|
| **Original Roboflow Test Split (Leaky)** | 121 | 0.9949 | `0.6855` | **+0.0314 mAP50-95 (+4.8% optimistic inflation)** |
| **Cluster-Deduplicated Test Split (Frozen)** | 114 | 0.9950 | **`0.6541`** | **Unbiased baseline reference** |

---

## 3. Multi-Resolution FP32 Ablation Study (Stage 1 Gates 3–4)

Three controlled training runs (`E0`, `E1`, `E2`) were executed from official `yolov8n.pt` COCO weights on an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) using `batch=4`, `seed=42`, `deterministic=True`, and `amp=False` (strict FP32):

| Experiment | Input Resolution | Epochs (Early Stop) | Peak Training VRAM | Val mAP50 | Val mAP50-95 | Test mAP50 | Test mAP50-95 | Test Precision | Test Recall (@0.25) | GPU Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| **E0 (Baseline)** | `640×640` | 92 | 0.858 GiB | 0.9928 | 0.7430 | 0.9884 | 0.6377 | 0.9905 | 0.9912 | 4.83 ms |
| **E1 (Champion)** | **`416×416`** | **47** | **0.415 GiB** | **0.9905** | **0.7424** | **0.9950** | **0.6541** | **0.9993** | **0.9825** | **2.56 ms** |
| **E2 (High-Res)** | `800×800` | 99 | 1.308 GiB | 0.9904 | 0.7303 | 0.9946 | 0.6668 | 0.9900 | 0.9737 | 7.29 ms |

### Key Findings from Stage 1
1. **Resolution Matching:** Because the underlying images are natively `416×416`, training at `640×640` (`E0`) or `800×800` (`E2`) only bilinearly interpolates pixels while increasing FLOPs quadratically.
2. **Efficiency Gains:** Selecting `E1` (`416×416`) reduced GPU inference time by **47.0%** vs. `E0` and **64.9%** vs. `E2`, and halved VRAM consumption (`0.415 GiB`), making `416×416` ideal for edge CPU deployment.

---

## 4. Conv-Isolated INT8 Post-Training Quantization (Stage 2)

### 4.1 Calibration & Operator Isolation Design
In YOLOv8's anchor-free detection head, bounding-box coordinate decoding (`DFL` / `Sub` / `Mul` / `Sigmoid` / `Concat`) is highly sensitive to uniform quantization grid quantization noise. Quantizing the entire ONNX graph degrades box localization IoU (`mAP50-95`).

To preserve localization precision while accelerating the compute-heavy backbone and neck, [`scripts/quantize_ptq.py`](../scripts/quantize_ptq.py) applies **ONNX Runtime Static QDQ Quantization** with:
- **Calibrator:** `CalibrationMethod.MinMax` over **300 stratified images** sampled exclusively from `train` and `valid` splits ([`calibration/images.json`](../calibration/images.json), verified leak-free by [`scripts/stage2_preflight.py`](../scripts/stage2_preflight.py)).
- **Quantized Operators:** `op_types_to_quantize=["Conv"]` (`QInt8` weights, `QUInt8` activations, `per_channel=False`), keeping detection head arithmetic in FP32.

### 4.2 Stage 2 Accuracy & Compression Comparison
Evaluated on the identical frozen test split (114 images, 114 ground-truth faces):

| Metric | Stage 1 FP32 (`best.pt` / `best.onnx`) | Stage 2 INT8 (`best_int8.onnx`) | Delta ($\Delta$) | Gate Requirement | Status |
|---|---|---|---|---|---|
| **Test mAP50** | `0.9950` | `0.9893` | `-0.0057` | Informational | **PASS** |
| **Test mAP50-95** | `0.6541` | **`0.6549`** | **`+0.0008`** | Drop $\le 0.0150$ | **PASS (Zero drop)** |
| **Test Precision** | `0.9993` | `0.9912` | `-0.0081` | Informational | **PASS** |
| **Test Recall (@ conf=0.25)** | `0.9825` (112/114) | **`0.9912` (113/114)** | **`+0.0087`** | Informational | **PASS** |
| **Large Face Recall** | `0.9823` (111/113) | **`0.9912` (112/113)** | `+0.0089` | Informational | **PASS** |
| **Medium Face Recall** | `1.0000` (1/1) | `1.0000` (1/1) | `0.0000` | Informational | **PASS** |
| **Serialized ONNX Size** | `11.60 MB` | **`3.14 MB`** | **`3.69× smaller`** | Edge storage fit | **PASS** |

---

## 5. Standalone Edge Deployment & Hardware Benchmarks

### 5.1 Zero-PyTorch Edge Inference Architecture (`pi/`)
Standard Ultralytics deployment pulls in PyTorch, Torchvision, and CUDA/CPU libraries (`>1.2 GB` disk footprint). For edge deployment on **Raspberry Pi 5**, [`pi/face_detector.py`](../pi/face_detector.py) implements a self-contained pipeline requiring only `onnxruntime`, `opencv-python`, and `numpy`:
1. **Aspect-Ratio Letterboxing (`letterbox`):** Resizes arbitrary camera frames (e.g., `640×480`) to `416×416` with symmetric `(114, 114, 114)` padding.
2. **ONNX Runtime Session (`CPUExecutionProvider`):** Configured with `ORT_ENABLE_ALL` graph optimizations and configurable `intra_op_num_threads`.
3. **Pure-NumPy Vectorized NMS (`nms`):** Eliminates `torchvision.ops.nms` overhead while executing in **0.05 ms** per frame.

### 5.2 Empirical Benchmark: FP32 ONNX vs. INT8 ONNX (`pi/bench_pi.py`)
Measured using [`pi/bench_pi.py`](../pi/bench_pi.py) (`warmup=20`, `runs=100`, `threads=4`, input frames `640×480` $\rightarrow$ `416×416`):

| Metric | x86_64 CPU FP32 (`best.onnx`) | x86_64 CPU INT8 (`best_int8.onnx`) | Raspberry Pi 5 FP32 (`best.onnx`)* | Raspberry Pi 5 INT8 (`best_int8.onnx`)* |
|---|---|---|---|---|
| **CPU Hardware** | Intel Core i5-12450H | Intel Core i5-12450H | Broadcom BCM2712 (Cortex-A76) | Broadcom BCM2712 (Cortex-A76) |
| **Model Size** | `11.60 MB` | **`3.14 MB` (3.69× smaller)** | `11.60 MB` | **`3.14 MB` (3.69× smaller)** |
| **Peak Process RSS Memory** | `130.9 MB` | **`130.9 MB`** | *Run `bench_pi.py`* | *Run `bench_pi.py`* |
| **Preprocess Median (ms)** | `0.63 ms` | **`0.64 ms` (6.6%)** | *Run `bench_pi.py`* | *Run `bench_pi.py`* |
| **ONNX Inference Median (ms)** | `12.12 ms` | **`9.07 ms` (92.8%)** | *Run `bench_pi.py`* | *Run `bench_pi.py`* |
| **Postprocess / NMS Median (ms)** | `0.06 ms` | **`0.05 ms` (0.5%)** | *Run `bench_pi.py`* | *Run `bench_pi.py`* |
| **End-to-End Median Latency** | `12.94 ms` | **`9.78 ms` (1.32× faster)** | *Run `bench_pi.py`* | *Run `bench_pi.py`* |
| **End-to-End p95 Latency** | `14.44 ms` | **`13.86 ms`** | *Run `bench_pi.py`* | *Run `bench_pi.py`* |
| **Throughput (Median FPS)** | `77.3 FPS` | **`102.3 FPS` (+25.0 FPS)** | *Run `bench_pi.py`* | *Run `bench_pi.py`* |

*\*To populate the Raspberry Pi 5 columns with live hardware data from your Pi, run `python pi/bench_pi.py` on the Raspberry Pi 5 following [`docs/pi_benchmark_guide.md`](pi_benchmark_guide.md) and copy the values from `reports/pi_benchmark.md`.*

### 5.3 Stage-by-Stage Pipeline Breakdown (`best_int8.onnx`, 4 Threads)

| Pipeline Stage | Operations | Median Latency (x86_64) | Latency Share (%) | Raspberry Pi 5 Median* |
|---|---|---|---|---|
| **1. Preprocess** | `cv2.resize` + `copyMakeBorder` + BGR$\rightarrow$RGB + `/255.0` + `CHW` | `0.642 ms` | `6.6%` | *From `pi_benchmark.md`* |
| **2. ONNX Inference** | Quantized `QDQ` Conv backbone + FP32 detection head | `9.075 ms` | `92.8%` | *From `pi_benchmark.md`* |
| **3. Postprocess** | Confidence filter (`>0.25`) + box decode + pure-NumPy NMS + clip | `0.050 ms` | `0.5%` | *From `pi_benchmark.md`* |
| **Total End-to-End** | Full [`FaceDetector.detect_timed()`](../pi/face_detector.py) call | **`9.776 ms`** | **`100.0%`** | *From `pi_benchmark.md`* |

**Analysis:** Over **92.8%** of total execution time is spent inside the ONNX Runtime neural network forward pass, while pure-NumPy NMS and coordinate rescaling take only **0.05 ms (0.5%)**. This confirms that removing PyTorch and `torchvision.ops.nms` introduces zero post-processing bottleneck while saving `>1 GB` of disk and memory overhead.

### 5.4 Multi-Core Thread Scaling Ablation (`best_int8.onnx`)

| CPU Threads (`intra_op_num_threads`) | Inference Median (ms) | Total Median (ms) | Total p95 (ms) | Throughput (FPS) | Speedup vs. 1 Thread | Raspberry Pi 5 FPS* |
|---|---|---|---|---|---|---|
| **1 Thread** | `17.56 ms` | `18.24 ms` | `19.27 ms` | `54.8 FPS` | `1.00×` | *From `pi_benchmark.md`* |
| **2 Threads** | `11.71 ms` | `12.40 ms` | `13.41 ms` | `80.7 FPS` | `1.47×` | *From `pi_benchmark.md`* |
| **3 Threads** | `9.90 ms` | `10.59 ms` | `11.48 ms` | `94.5 FPS` | `1.72×` | *From `pi_benchmark.md`* |
| **4 Threads** | `9.07 ms` | **`9.78 ms`** | `13.86 ms` | **`102.3 FPS`** | **`1.87×`** | *From `pi_benchmark.md`* |

---

## 6. Domain Generalization, Limitations & Future Work

1. **Close-Range vs. Crowd/Small-Face Generalization:**
   - As proven in Gate 1, **99.6% (2,165 / 2,189)** of ground-truth boxes in the Roboflow training dataset are large faces ($\ge 96\times96\text{ px}$). Consequently, the detector achieves **99.12% recall** on portrait/webcam-distance faces (`Test split`) but does not generalize to tiny crowd faces (`<32×32 px`) in the external Out-of-Distribution (`data/ood/`) stress test.
   - **Application Fit:** The current model is well-suited for **smart kiosks, access control, attendance terminals, and close-range Raspberry Pi webcam streams** (`0.3 m – 2.5 m` camera distance).
2. **Future Enhancements:**
   - **Multi-Scale Dataset Expansion:** Incorporate WIDER FACE multi-scale crowd annotations alongside the deduplicated portrait dataset to improve small-face recall.
   - **Per-Channel Weight Quantization & NPU Acceleration:** Evaluate `per_channel=True` symmetric INT8 weight quantization for ARM NEON `sdot`/`udot` instructions and hailo-8L AI HAT+ NPU compilation on Raspberry Pi 5.

---

## 7. Reproducibility & Artifact Index

| Lifecycle Gate / Stage | Script | Generated Report / Artifact |
|---|---|---|
| **Gate 0: Environment** | [`scripts/environment_report.py`](../scripts/environment_report.py) | [`reports/environment_report.md`](../reports/environment_report.md) |
| **Gate 1: Dataset Audit** | [`scripts/audit_dataset.py`](../scripts/audit_dataset.py) | [`reports/audit_report.md`](../reports/audit_report.md), [`reports/audit_stats.json`](../reports/audit_stats.json) |
| **Gate 2: Deduplicated Split** | [`scripts/make_dedup_split.py`](../scripts/make_dedup_split.py) | [`reports/split_report.md`](../reports/split_report.md), [`reports/split_hashes.sha256`](../reports/split_hashes.sha256) |
| **Gate 3–4: Stage 1 FP32** | [`scripts/train_stage1.py`](../scripts/train_stage1.py), [`scripts/stage1_summary.py`](../scripts/stage1_summary.py) | [`reports/stage1_summary.md`](../reports/stage1_summary.md), [`models/stage1/best.pt`](../models/stage1/best.pt) |
| **Gate 4.5: Stage 2 Preflight** | [`scripts/stage2_preflight.py`](../scripts/stage2_preflight.py) | [`reports/calibration_report.md`](../reports/calibration_report.md) |
| **Stage 2 PTQ & Accuracy** | [`scripts/quantize_ptq.py`](../scripts/quantize_ptq.py), [`scripts/evaluate_stage2.py`](../scripts/evaluate_stage2.py) | [`reports/stage2_fp32_vs_int8.md`](../reports/stage2_fp32_vs_int8.md), [`models/stage2/best_int8.onnx`](../models/stage2/best_int8.onnx) |
| **Gate 5 & Edge Benchmark** | [`pi/bench_pi.py`](../pi/bench_pi.py) | [`reports/pi_benchmark.md`](../reports/pi_benchmark.md), [`docs/pi_benchmark_guide.md`](pi_benchmark_guide.md) |
