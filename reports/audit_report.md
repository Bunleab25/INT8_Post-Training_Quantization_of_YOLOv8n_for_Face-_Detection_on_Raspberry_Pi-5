# Dataset Audit Report (Gate 1)

Generated: 2026-11-07T16:34:40
Dataset: `/home/bunleab/face_yolov/Face-Detection.v2-datasetv2.yolov8`

## Headline result

- **2173 files**, **1242 unique source names**, **1135 near-duplicate clusters**.
- Generation gap explained: **931** train-only augmentation copies (clean + salt-and-pepper pairs).
- Near-duplicate leakage: **40 clusters span splits**; valid contaminated **19**, test contaminated **21**.
- Face size: **2165 large**, 20 medium, 4 small at 416 px.

## Counts

| Split | Files | Unique sources | Boxes | Empty labels | Bad label lines |
|---|---|---|---|---|---|
| train | 1862 | 931 | 1878 | 0 | 0 |
| valid | 185 | 185 | 185 | 0 | 0 |
| test | 126 | 126 | 126 | 0 | 0 |
| **total** | **2173** | **1242** | **2189** | | |

## Integrity and labels

- Corrupt images: **0**
- Classes: `{'0': 2189}` (single class expected)
- Image dimensions: `{'416x416': 2173}`
- File modes: `{'RGB': 2173}`
- Grayscale content: **2173/2173**
- Border-touching boxes: **295**
- Face area fraction: min 0.0042, p5 0.1692, median 0.4292, p95 0.7531, max 1.0000
- COCO size buckets at 416 px: `{'small_lt_1024': 4, 'medium_1024_9216': 20, 'large_ge_9216': 2165}`

## Duplicates and leakage

- Exact cross-split duplicate hashes: **0**
- Near-duplicate clusters: **87** covering **194** source images
- Clusters spanning splits: **40**
- Contaminated unique sources: valid **19**, test **21**

## Augmentation

- Train clean/noisy pairs detected: **931**
- Median noise difference within a pair: **0.0044**

## Diversity and limitations

- All images are 416x416 and stretched; content is grayscale.
- The dataset is dominated by close-up, mostly frontal portraits; there is no crowd, small-face or low-light coverage to speak of.
- There are no negative/background images in the export.
- These limitations are recorded as findings, not assumed away. Per-size recall must be measured on an external set.

## Artefacts

- `dataset_manifest.csv` — every image with hashes, cluster id, boxes and noise score.
- `leakage_clusters.csv` — every cross-split near-duplicate cluster.
- `source_clusters.json` — cluster assignment used by the split builder.
- `audit_stats.json` — machine-readable results.
- `face_size_hist.png`, `boxes_per_image_hist.png`.

