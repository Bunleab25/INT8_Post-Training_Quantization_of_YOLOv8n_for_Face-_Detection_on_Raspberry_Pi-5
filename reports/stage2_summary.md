# Stage 2 Summary — INT8 Judged Against FP32

Generated: 2026-11-09T02:26:46

## Gate status

| Gate | Evidence | Status |
|---|---|---|
| Gate 4 — Stage 1 acceptance | `configs/stage2.json` → stage1_gate4_approved | PASS |
| Gate 4.5 — Preflight inputs | OOD + calibration + runtime targets | PASS |
| Export X0 | `exports/stage2/onnxruntime/export_info.json` | PASS |
| PTQ Q0 | `models/stage2/int8_info.json` | PASS |
| Accuracy (FP32 vs INT8) | `reports/stage2_fp32_vs_int8.md` | PASS |
| Gate 5 — Hardware | `reports/stage2_gate5.md` | PASS |
| Promote best_int8 | `models/stage2/best_int8.*` | PASS |

## Locked targets

| Item | Value |
|---|---|
| FP32 reference | `models/stage1/best.pt` (E1) |
| Runtime | `onnxruntime` |
| Target device | `NVIDIA GeForce RTX 3050 Laptop GPU / x86_64 CPU` |
| imgsz | 416 |
| Latency target | 20.0 ms (model_only) |
| Allowed test drop | 0.015 mAP50-95 |
| Allowed OOD drop | 0.02 mAP50-95 |
| Calibration count | 300 |

## Calibration

- Manifest images: **300** (requested 300); test-split images: **0**.

## FP32 vs INT8

| Set | FP32 mAP50-95 | INT8 mAP50-95 | Drop | Pass |
|---|---|---|---|---|
| Test | 0.6541 | 0.6549 | -0.0008 | PASS |
| OOD | 0.0007 | 0.0000 | 0.0007 | PASS |

## Gate 5 latency

- INT8 median latency: **16.400 ms** (target ≤ 20.0 ms)
- Peak GPU memory: **0.0 GiB**
- Gate 5: **PASS**; overall promote: **PASS**

## Environment

- torch `2.13.0+cu130`, CUDA `13.0`, GPU `NVIDIA GeForce RTX 3050 Laptop GPU`

## Artefacts

- `configs/stage2.template.json`, `configs/stage2.json`
- `calibration/images.json`, `reports/calibration_report.md`
- `exports/stage2/onnxruntime/best.onnx`
- `models/stage2/int8.onnx`
- `models/stage2/best_int8.onnx`
- `reports/stage2_fp32_vs_int8.md`, `reports/stage2_gate5.md`

## Next actions

1. Stage 2 complete; use `models/stage2/best_int8.onnx`.

