# Stage 2 Gate 5 — Target Hardware Check

Generated: 2026-11-07T19:28:23

- Target device: `NVIDIA GeForce RTX 3050 Laptop GPU / x86_64 CPU`
- Latency definition: `model_only` (batch=1, imgsz=416)
- Warm-up / bench runs: 50 / 200
- Engine: `models/stage2/int8.onnx`

## Results

| Model | Median ms | p95 ms | Peak GPU GiB |
|---|---|---|---|
| INT8 | 16.400 | 25.340 | 0.0 |
| FP32 ref | 3.071 | 3.814 | 0.02 |

- Latency target: ≤ **20.0 ms** → **PASS**
- Memory limit: ≤ **3.5 GiB** → **PASS**
- Gate 5: **PASS**
- Accuracy gate (from stage2_fp32_vs_int8): **PASS**
- Promote INT8: **YES → models/stage2/best_int8.engine**

