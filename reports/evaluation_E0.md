# Evaluation — E0

Generated: 2026-11-07T18:01:24
Weights: `/home/bunleab/face_yolov/models/stage1/E0_best.pt`
Split: `test` · imgsz 640 · conf 0.25 · IoU 0.5

## Ultralytics validation

| Metric | Value |
|---|---|
| `metrics/precision(B)` | 0.9905 |
| `metrics/recall(B)` | 0.9912 |
| `metrics/mAP50(B)` | 0.9884 |
| `metrics/mAP50-95(B)` | 0.6377 |
| `fitness` | 0.6377 |

## Recall at the operating threshold

| Size bucket | GT | Matched | Recall |
|---|---|---|---|
| small | 0 | 0 | n/a |
| medium | 1 | 1 | 1.0000 |
| large | 113 | 112 | 0.9912 |

- Overall matched recall: **0.9912280701754386** (113/114)
- Predictions above conf: 115 on 114 images
- Mean inference latency: 4.834017508823819 ms (p95 4.858413450074295 ms)

> Note: the Roboflow test split has almost no small faces, so the small bucket is not
> statistically meaningful. Use an external small-face set for that claim.

