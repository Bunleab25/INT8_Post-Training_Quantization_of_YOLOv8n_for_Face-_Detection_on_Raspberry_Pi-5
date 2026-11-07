# Calibration Report

Generated: 2026-11-07T18:48:59

- Dataset: `/home/bunleab/face_yolov/datasets/face_v2_dedup`
- Manifest: `calibration/images.json`
- Requested count: **300**; actual: **300**
- Seed: 42
- Candidate pool (train+valid): **1021**
- Test split images included: **0** (hard rule)

## Split counts

| Split | Count |
|---|---|
| train | 186 |
| valid | 114 |

## Face-size buckets

| Bucket | Count |
|---|---|
| small | 0 |
| medium | 5 |
| large | 295 |

## Notes

- Sampling prefers one image per Roboflow source within each split×bucket group.
- Use this manifest only for PTQ calibration. Never add test or OOD images.

