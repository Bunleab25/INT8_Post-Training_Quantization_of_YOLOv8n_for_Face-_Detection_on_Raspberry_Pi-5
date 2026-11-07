# Stage 1 Summary — FP32 YOLOv8 Face Detector

Generated: 2026-11-07T18:02:37

## Gate status

| Gate | Evidence | Status |
|---|---|---|
| Gate 0 — Environment | `reports/environment_report.md` | PASS — torch 2.13.0+cu130, CUDA 13.0, NVIDIA GeForce RTX 3050 Laptop GPU |
| Gate 1 — Audit | `reports/audit_report.md` | PASS with findings — 1242 unique sources, 40 cross-split duplicate clusters |
| Gate 2 — Split | `reports/split_report.md` | PASS — cluster-aware deduplicated split, 0 cross-split duplicates |
| Gate 3 — Smoke test | `runs/stage1/S0` | PASS — loss and validation work, GPU headroom |
| Gate 4 — Acceptance | this report | PENDING owner approval (OOD set not yet supplied) |

## Dataset audit headline

- Files: **2173** = **1242** unique sources + **931** train augmentation copies.
- Near-duplicate clusters: **87**, of which **40** span the original splits.
- Face-size buckets at 416 px: `{'small_lt_1024': 4, 'medium_1024_9216': 20, 'large_ge_9216': 2165}`.
- Contaminated original valid/test sources: valid **19**, test **21**.
- Domain: 416×416 grayscale, stretched, close-up portraits; no small-face/crowd/negative coverage.

## Split

| Split | Images |
|---|---|
| train | 907 |
| valid | 114 |
| test | 114 |

Config: `configs/face_v2_dedup.yaml`; frozen hashes in `reports/split_hashes.sha256`.

## Runs

| Run | imgsz | batch | epochs | peak VRAM (GiB) | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 | test precision | test recall | recall@conf=0.25 | inference ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 640 | 4 | 92 | 0.858 | 0.9928 | 0.7430 | 0.9884 | 0.6377 | 0.9905 | 0.9912 | 0.9912 | 4.83 |
| E1 | 416 | 4 | 47 | 0.415 | 0.9905 | 0.7424 | 0.9950 | 0.6541 | 0.9993 | 1.0000 | 0.9825 | 2.56 |
| E2 | 800 | 4 | 99 | 1.308 | 0.9904 | 0.7303 | 0.9946 | 0.6668 | 0.9900 | 0.9825 | 0.9737 | 7.29 |

## Best FP32 model

- Selected run: **E1** (frozen copy: `models/stage1/best.pt`).
- Validation mAP50-95 = **0.7424**; validation mAP50 = **0.9905**.
- Test mAP50-95 = **0.6541**; test mAP50 = **0.9950** (reported, not used for selection).
- Peak training VRAM = **0.415 GiB**; mean inference latency = **2.56 ms** on the training GPU.
- Selection rule: validation mAP50-95, with candidates within 0.005 treated as tied and the cheaper configuration chosen. This avoids tuning on the test split.

## Per-size recall (selected model, test split)

| Bucket | GT | Matched | Recall |
|---|---|---|---|
| small | 0 | 0 | n/a |
| medium | 1 | 1 | 1.0000 |
| large | 113 | 111 | 0.9823 |

## Leakage check (original Roboflow test split)

- E1 on the original (leaky) test split: mAP50-95 = **0.6855**.
- E1 on the deduplicated test split: mAP50-95 = **0.6541**.
- The leaky split is optimistic by **+0.0314** mAP50-95 (+4.8%). This is why the deduplicated split is used for decisions.

## Known limitations and pending inputs

- **OOD set is pending.** The dataset is close-up portraiture, so the reported test score is not a measure of general face detection. Gate 4 cannot be fully signed off without an external set (crowds, profile, occlusion, low light, small faces, negatives).
- The dataset has only 4 small-face boxes, so small-face recall is not measurable on the Roboflow data.
- The 416×416 stretch and grayscale conversion are baked into the export and may not match deployment images.
- Stage 2 (INT8) must not start until Gate 4 is approved.

## Artefacts

- `reports/environment_report.md`, `reports/audit_report.md`, `reports/split_report.md`
- `reports/run_comparison.md`, `reports/metrics_E*.json`, `reports/evaluation_E*.md`
- `runs/stage1/S0`, `runs/stage1/E0`, `runs/stage1/E1`, `runs/stage1/E2`
- `models/stage1/best.pt`, `models/stage1/best_info.json`
- `splits/*.txt`, `configs/face_v2_dedup.yaml`

