# Evaluation — E2

Generated: 2026-11-07T18:01:36
Weights: `/home/bunleab/face_yolov/models/stage1/E2_best.pt`
Split: `test` · imgsz 800 · conf 0.25 · IoU 0.5

## Ultralytics validation

| Metric | Value |
|---|---|
| `metrics/precision(B)` | 0.9900 |
| `metrics/recall(B)` | 0.9825 |
| `metrics/mAP50(B)` | 0.9946 |
| `metrics/mAP50-95(B)` | 0.6668 |
| `fitness` | 0.6668 |

## Recall at the operating threshold

| Size bucket | GT | Matched | Recall |
|---|---|---|---|
| small | 0 | 0 | n/a |
| medium | 1 | 1 | 1.0000 |
| large | 113 | 110 | 0.9735 |

- Overall matched recall: **0.9736842105263158** (111/114)
- Predictions above conf: 117 on 114 images
- Mean inference latency: 7.291649412317176 ms (p95 7.377789000111079 ms)

> Note: the Roboflow test split has almost no small faces, so the small bucket is not
> statistically meaningful. Use an external small-face set for that claim.

