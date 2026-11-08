# Edge Hardware Benchmark Report: Extended Endurance Stress Test (1,000 Runs)

Generated: `2026-11-08T15:12:04`

## 1. System & Runtime Environment

| Property | Value |
|---|---|
| **Device / CPU** | `Raspberry Pi 5 Model B Rev 1.1` |
| **Architecture** | `aarch64` |
| **OS** | `Linux 6.18.60+rpt-rpi-2712` |
| **Python** | `3.13.5` |
| **ONNX Runtime** | `1.31.1` |
| **OpenCV / NumPy** | `5.0.0` / `2.5.4` |
| **Frame Source** | `/home/bunleab/Documents/yolov/PI_Quantize/deploy/bench/PI_Quantize/data/ood/images (5 images)` |
| **Warm-up / Measured Runs** | `50` / `1000` |
| **Peak Process RSS Memory** | `116.16 MB` |
| **Throttled Status** | `throttled=0xe0006` |

## 2. Model Comparison: FP32 ONNX vs. INT8 ONNX

| Metric | FP32 ONNX (`best.onnx`) | INT8 ONNX (`best_int8.onnx`) | Speedup / Reduction |
|---|---|---|---|
| **Model Size (MB)** | 11.6 MB | **3.14 MB** | **3.69× smaller** |
| **Cold Load Time (ms)** | 257.5 ms | **249.76 ms** | — |
| **Preprocess Median (ms)** | 2.51 ms | **2.15 ms** | — |
| **Inference Median (ms)** | 109.63 ms | **64.93 ms** | **1.67× faster** |
| **Postprocess / NMS (ms)** | 0.11 ms | **0.12 ms** | — |
| **End-to-End Median (ms)** | 112.27 ms | **67.22 ms** | **1.67× faster** |
| **End-to-End p95 (ms)** | 114.44 ms | **70.24 ms** | — |
| **Throughput (Median FPS)** | 8.9 FPS | **14.9 FPS** | **+6.0 FPS** |

## 3. INT8 Stage-by-Stage Pipeline Breakdown

| Pipeline Stage | Median Latency (ms) | Share of Total Time (%) |
|---|---|---|
| **1. Preprocess** (`letterbox` + RGB/CHW norm) | 2.146 ms | 3.2% |
| **2. ONNX Runtime Inference** (`session.run`) | 64.933 ms | 96.6% |
| **3. Postprocess** (Box decode + NumPy NMS) | 0.116 ms | 0.2% |
| **Total End-to-End** | **67.224 ms** | **100.0%** |

## 4. Multi-Core CPU Thread Scaling (`best_int8.onnx`)

| CPU Threads | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Total Median (ms) | Total p95 (ms) | Throughput (FPS) | Scaling vs 1T | Temp (°C) |
|---|---|---|---|---|---|---|---|---|
| **4** | 2.15 | 64.93 | 0.12 | **67.22** | 70.24 | **14.9** | 1.00× | 87.55°C |
