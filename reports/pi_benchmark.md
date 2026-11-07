# Edge Hardware Benchmark Report (ONNX Runtime CPU)

Generated: `2026-11-08T03:37:15`

## 1. System & Runtime Environment

| Property | Value |
|---|---|
| **Device / CPU** | `12th Gen Intel(R) Core(TM) i5-12450H` |
| **Architecture** | `x86_64` |
| **OS** | `Linux 7.0.0-42-generic` |
| **Python** | `3.11.15` |
| **ONNX Runtime** | `1.31.1` |
| **OpenCV / NumPy** | `5.0.0` / `2.4.4` |
| **Frame Source** | `/home/bunleab/face_yolov/data/ood/images (5 images)` |
| **Warm-up / Measured Runs** | `20` / `100` |
| **Peak Process RSS Memory** | `130.93 MB` |
| **Throttled Status** | `N/A (non-RPi or vcgencmd unavailable)` |

## 2. Model Comparison: FP32 ONNX vs. INT8 ONNX

| Metric | FP32 ONNX (`best.onnx`) | INT8 ONNX (`best_int8.onnx`) | Speedup / Reduction |
|---|---|---|---|
| **Model Size (MB)** | 11.6 MB | **3.14 MB** | **3.69× smaller** |
| **Cold Load Time (ms)** | 36.16 ms | **49.05 ms** | — |
| **Preprocess Median (ms)** | 0.63 ms | **0.64 ms** | — |
| **Inference Median (ms)** | 12.12 ms | **9.07 ms** | **1.32× faster** |
| **Postprocess / NMS (ms)** | 0.06 ms | **0.05 ms** | — |
| **End-to-End Median (ms)** | 12.94 ms | **9.78 ms** | **1.32× faster** |
| **End-to-End p95 (ms)** | 14.44 ms | **13.86 ms** | — |
| **Throughput (Median FPS)** | 77.3 FPS | **102.3 FPS** | **+25.0 FPS** |

## 3. INT8 Stage-by-Stage Pipeline Breakdown

| Pipeline Stage | Median Latency (ms) | Share of Total Time (%) |
|---|---|---|
| **1. Preprocess** (`letterbox` + RGB/CHW norm) | 0.642 ms | 6.6% |
| **2. ONNX Runtime Inference** (`session.run`) | 9.075 ms | 92.8% |
| **3. Postprocess** (Box decode + NumPy NMS) | 0.050 ms | 0.5% |
| **Total End-to-End** | **9.776 ms** | **100.0%** |

## 4. Multi-Core CPU Thread Scaling (`best_int8.onnx`)

| CPU Threads | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Total Median (ms) | Total p95 (ms) | Throughput (FPS) | Scaling vs 1T | Temp (°C) |
|---|---|---|---|---|---|---|---|---|
| **1** | 0.60 | 17.56 | 0.05 | **18.24** | 19.27 | **54.8** | 1.00× | 75.0°C |
| **2** | 0.61 | 11.71 | 0.05 | **12.40** | 13.41 | **80.7** | 1.47× | 74.0°C |
| **3** | 0.61 | 9.90 | 0.05 | **10.59** | 11.48 | **94.5** | 1.72× | 75.0°C |
| **4** | 0.64 | 9.07 | 0.05 | **9.78** | 13.86 | **102.3** | 1.87× | 76.0°C |

