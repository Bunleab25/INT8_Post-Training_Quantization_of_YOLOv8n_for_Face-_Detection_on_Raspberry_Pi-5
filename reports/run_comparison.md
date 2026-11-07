# Stage 1 Run Comparison

FP32 runs on the frozen, deduplicated split. Only image size changes between runs; batch is reported alongside every result.

`val` is the validation split used for model selection. `test` is the frozen test split, reported for final comparison only and never used for tuning.

| Run | imgsz | batch | epochs | peak VRAM (GiB) | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 | test precision | test recall | recall@conf=0.25 | inference ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 640 | 4 | 92 | 0.858 | 0.9928 | 0.7430 | 0.9884 | 0.6377 | 0.9905 | 0.9912 | 0.9912 | 4.83 |
| E1 | 416 | 4 | 47 | 0.415 | 0.9905 | 0.7424 | 0.9950 | 0.6541 | 0.9993 | 1.0000 | 0.9825 | 2.56 |
| E2 | 800 | 4 | 99 | 1.308 | 0.9904 | 0.7303 | 0.9946 | 0.6668 | 0.9900 | 0.9825 | 0.9737 | 7.29 |

## Observations

- **E1 (416) matches E0 (640) on validation** (Δ mAP50-95 < 0.001) at roughly half the peak VRAM and about 1.9× lower inference latency. This is consistent with the source images already being 416×416; 640 is upsampling.
- **E2 (800) does not improve mAP50-95** on validation and costs ~3× the VRAM and ~3× the latency of E1. Its test mAP50 is comparable, but the localization score (mAP50-95) is lower on validation.
- The differences among all three runs are small and within the noise of a 114-image test split. Treat the ranking as indicative, not decisive.
- The dataset has almost no small faces, so these results say nothing about small-face performance. That claim needs an external set.

## Selection rule

Model selection uses **validation** mAP50-95 (the test split is not used for tuning). If candidates are within 0.005 mAP50-95, the cheaper configuration (peak VRAM, latency, native resolution) wins the tie-break. See `reports/stage1_summary.md` for the selected model.

## Notes

- All runs start from official `yolov8n.pt`, FP32 (`amp=False`), seed 42, deterministic.
- Validation and test use the cluster-aware deduplicated split (`configs/face_v2_dedup.yaml`).
- `recall@conf` is the per-image matched recall at confidence 0.25 and IoU 0.5.
- Inference latency is measured on the RTX 3050 training GPU and is informational only; the deployment target is measured in Stage 2.

