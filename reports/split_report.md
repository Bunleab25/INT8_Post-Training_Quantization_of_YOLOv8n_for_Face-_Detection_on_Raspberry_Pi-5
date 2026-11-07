# Split Report (Gate 2)

Source export: `/home/bunleab/face_yolov/Face-Detection.v2-datasetv2.yolov8`
Deduplicated dataset: `/home/bunleab/face_yolov/datasets/face_v2_dedup`
Config: `/home/bunleab/face_yolov/configs/face_v2_dedup.yaml`
Seed: 42

## Method

- Cluster all source images by perceptual hash (dHash <= 3) and source name.
- Keep one (cleanest) image per cluster.
- Assign **whole clusters** to train/valid/test, stratified by face-size bucket, 80/10/10.
- The Roboflow export is never modified; the new dataset contains symlinks.

## Counts

| Split | Images | Small | Medium | Large |
|---|---|---|---|---|
| train | 907 | 0 | 3 | 904 |
| valid | 114 | 0 | 1 | 113 |
| test | 114 | 0 | 1 | 113 |
| **total** | **1135** | | | |

## Checks

- Missing labels: **0**
- Cross-split near-duplicate clusters: **0**
- Original unique sources: 1242
- Unique clusters after dedup: 1135

## Frozen files

- `splits/train.txt`, `splits/valid.txt`, `splits/test.txt`
- `reports/split_hashes.sha256`

> **Gate 2 passes:** no cross-split near-duplicate clusters.
