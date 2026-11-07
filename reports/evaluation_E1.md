# Evaluation — E1

Generated: 2026-11-07T18:01:30
Weights: `/home/bunleab/face_yolov/models/stage1/E1_best.pt`
Split: `test` · imgsz 416 · conf 0.25 · IoU 0.5

## Ultralytics validation

| Metric | Value |
|---|---|
| `metrics/precision(B)` | 0.9993 |
| `metrics/recall(B)` | 1.0000 |
| `metrics/mAP50(B)` | 0.9950 |
| `metrics/mAP50-95(B)` | 0.6541 |
| `fitness` | 0.6541 |

## Recall at the operating threshold

| Size bucket | GT | Matched | Recall |
|---|---|---|---|
| small | 0 | 0 | n/a |
| medium | 1 | 1 | 1.0000 |
| large | 113 | 111 | 0.9823 |

- Overall matched recall: **0.9824561403508771** (112/114)
- Predictions above conf: 118 on 114 images
- Mean inference latency: 2.557153473724994 ms (p95 2.7962025999840985 ms)

> Note: the Roboflow test split has almost no small faces, so the small bucket is not
> statistically meaningful. Use an external small-face set for that claim.

