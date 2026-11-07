# YOLOv8 Face Detection – Stage 2 Architecture & Delivery Plan

Nov 7, 2026 · Stage 2: Quantization and export

## Executive summary

Stage 2 delivers an INT8 face detector judged only against the frozen Stage 1 FP32
checkpoint (`models/stage1/best.pt`, run E1, imgsz 416). Architecture and resolution
are not reopened. Success means the quantized export stays within the accuracy-drop
budget on the frozen test split and on an external OOD set, and meets the latency
target on the real target device (Gate 5).

Post-training quantization (PTQ) is the primary path. Quantization-aware training
(QAT) is a separate, labelled fallback if PTQ exceeds the allowed drop. The test
split is never used for calibration.

## Locked defaults

| Item | Value |
| --- | --- |
| FP32 reference | `models/stage1/best.pt` (E1); hash in `models/stage1/best.sha256` |
| Input size | 416 |
| Runtime | TensorRT via ONNX |
| Target device (Gate 5) | NVIDIA GeForce RTX 3050 Laptop GPU |
| Latency target | 10 ms, model-only, batch=1, warm-up 50, median of 200 |
| Allowed test mAP50-95 drop | 0.02 absolute vs FP32 |
| Allowed OOD mAP50-95 drop | 0.03 absolute vs FP32 |
| Calibration | ~300 train/valid images, stratified; never test |
| Config | `configs/stage2.json` (from `configs/stage2.template.json`) |

Override these in `configs/stage2.json` before Gate 4.5 if the deploy target differs.

## Judgment rule

INT8 is accepted only when both gates pass:

1. **Accuracy.** On the frozen deduplicated test split, INT8 mAP50-95 ≥ FP32 − 0.02.
   On the same OOD set, INT8 mAP50-95 ≥ FP32 − 0.03. Small-face recall is reported
   separately; the Roboflow test set may be n/a for small faces, so OOD carries that claim.
2. **Hardware (Gate 5).** Model-only latency ≤ 10 ms and memory fits the 4 GB device
   under the bench protocol in `configs/stage2.json`.

```
Frozen FP32 E1 → Export ONNX (X0) → PTQ INT8 (Q0) → Compare FP32 vs INT8
        │                                    │
        │                              fail accuracy → QAT0 → re-export → compare
        │                                    │
        └────────────────── pass ───────────→ Gate 5 latency/memory → promote best_int8
```

## Phases

| Phase | Name | Exit criteria |
| --- | --- | --- |
| A | Gate 4 close | OOD under `data/ood/`; `stage1_gate4_approved: true` in `configs/stage2.json` |
| B | Calibration | `calibration/images.json` + `reports/calibration_report.md`; train/valid only |
| C | Preflight (Gate 4.5) | `scripts/stage2_preflight.py` exits 0 |
| D | Export X0 | ONNX + checksum under `exports/stage2/tensorrt/` |
| E | PTQ Q0 | `models/stage2/int8.engine` (+ quantization config dump) |
| F | Compare | `reports/stage2_fp32_vs_int8.md` with pass/fail vs budgets |
| G | QAT0 (fallback) | Only if PTQ fails test drop; artefacts kept separate from Q0 |
| H | Gate 5 | `reports/stage2_gate5.md`; promote `models/stage2/best_int8.*` + sha256 |

## Experiment IDs

| ID | What | Variable |
| --- | --- | --- |
| X0 | Export FP32 ONNX control | runtime plumbing |
| Q0 | PTQ INT8 with calibration v1 | quantization |
| Q1 | PTQ retries (calib size/method) | calibration only — one change each |
| QAT0 | Quantization-aware training | training recipe (fallback) |
| H0 | Gate 5 timing on target device | hardware |

## Scripts

| Script | Role |
| --- | --- |
| `scripts/make_calibration_manifest.py` | Sample ~300 train/valid images; write calibration manifest |
| `scripts/stage2_preflight.py` | Gate 4.5 prerequisite check (no export) |
| `scripts/export_stage2.py` | Export frozen `best.pt` → ONNX @ 416 |
| `scripts/quantize_ptq.py` | TensorRT INT8 engine from ONNX + calibration manifest |
| `scripts/evaluate_stage2.py` | FP32 vs INT8 on frozen test + OOD; Δ report |
| `scripts/bench_target.py` | Gate 5 latency/memory harness |
| `scripts/stage2_summary.py` | Gate pass/fail summary |

## Artefact layout

```
data/ood/                          # owner-provided (images + labels)
calibration/images.json
configs/stage2.json
exports/stage2/tensorrt/
models/stage2/
reports/calibration_report.md
reports/stage2_fp32_vs_int8.md
reports/stage2_gate5.md
reports/stage2_summary.md
```

The Roboflow export and Stage 1 checkpoints are never modified in place.

## Reproduction (after Gate 4)

```bash
cd /home/bunleab/face_yolov
cp configs/stage2.template.json configs/stage2.json   # then set gate4 + paths
/home/bunleab/miniconda3/envs/ai-ml/bin/python scripts/make_calibration_manifest.py
/home/bunleab/miniconda3/envs/ai-ml/bin/python scripts/stage2_preflight.py
/home/bunleab/miniconda3/envs/ai-ml/bin/python scripts/export_stage2.py
/home/bunleab/miniconda3/envs/ai-ml/bin/python scripts/quantize_ptq.py
/home/bunleab/miniconda3/envs/ai-ml/bin/python scripts/evaluate_stage2.py
/home/bunleab/miniconda3/envs/ai-ml/bin/python scripts/bench_target.py
/home/bunleab/miniconda3/envs/ai-ml/bin/python scripts/stage2_summary.py
```

Environment: `/home/bunleab/miniconda3/envs/ai-ml` only. Do not reinstall PyTorch,
Torchvision or CUDA for Stage 2. Install TensorRT Python bindings only if preflight
reports them missing, and record the change in the environment report.

## Risks

| Risk | Mitigation |
| --- | --- |
| Portrait-only calib looks fine on test, fails on OOD | OOD mandatory; stratified calib; OOD drop budget |
| Test leakage into calibration | Preflight rejects test paths |
| Latency judged on wrong hardware | Gate 5 only on `target_device` |
| Stretch/grayscale mismatch at deploy | Document preprocess; match calib and bench |
| Small-face claim without OOD small faces | Report n/a; do not accept as general detector |

## Gates

| Gate | Criteria |
| --- | --- |
| Gate 4 (Stage 1) | FP32 accepted; OOD path and numeric budgets recorded |
| Gate 4.5 | `stage2_preflight.py` green |
| Gate 5 | INT8 within accuracy budgets and latency/memory on target device |

## Out of scope

- Retraining Stage 1 architecture or resolution (E3/E4 remain Stage 1 follow-ups)
- Using test or OOD images for calibration
- Judging deploy latency on a different machine without updating `target_device` and re-running Gate 5
