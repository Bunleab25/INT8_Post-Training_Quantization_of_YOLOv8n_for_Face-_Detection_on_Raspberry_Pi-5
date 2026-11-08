#!/usr/bin/env python3
"""Gate 1: dataset audit.

Reads the Roboflow export without modifying it and writes:
  reports/audit_report.md
  reports/audit_stats.json
  reports/dataset_manifest.csv
  reports/leakage_clusters.csv
  reports/source_clusters.json
  reports/face_size_hist.png
  reports/boxes_per_image_hist.png
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from scripts.common import sha256_file, source_name
except ImportError:
    from common import sha256_file, source_name

SPLITS = ("train", "valid", "test")


def label_path(image_path: Path, dataset: Path) -> Path:
    split = image_path.parent.parent.name
    return dataset / split / "labels" / (image_path.stem + ".txt")


def read_boxes(path: Path):
    boxes, bad = [], 0
    if not path.exists():
        return boxes, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            if line.strip():
                bad += 1
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            bad += 1
            continue
        boxes.append((cls, x, y, w, h))
    return boxes, bad


def dhash(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize((9, 8), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.int16)
    return (arr[:, 1:] > arr[:, :-1]).flatten()


def noise_score(path: Path) -> float:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    med = cv2.medianBlur(img, 3).astype(np.float32)
    return float((np.abs(img.astype(np.float32) - med) > 40).mean())


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_sources(source_to_rep: dict[str, Path], hashes: dict[str, np.ndarray], threshold: int = 3):
    """Union sources whose representative images have dHash distance <= threshold."""
    sources = list(source_to_rep)
    n = len(sources)
    uf = UnionFind(n)
    mat = np.stack([hashes[s] for s in sources]).astype(np.uint8)
    for i in range(n):
        dist = (mat[i + 1 :] != mat[i]).sum(axis=1)
        for j in np.where(dist <= threshold)[0]:
            uf.union(i, int(j + i + 1))
    return {sources[i]: uf.find(i) for i in range(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Face-Detection.v2-datasetv2.yolov8")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--hash-threshold", type=int, default=3)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset = (root / args.dataset).resolve()
    out = (root / args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    corrupt = []
    dims = Counter()
    modes = Counter()
    class_counter = Counter()
    split_boxes = defaultdict(int)
    split_images = defaultdict(int)
    split_bad_lines = defaultdict(int)
    split_empty_labels = defaultdict(int)
    areas = []
    border_boxes = 0
    grayscale = 0

    for split in SPLITS:
        for image_path in sorted((dataset / split / "images").glob("*")):
            split_images[split] += 1
            try:
                with Image.open(image_path) as im:
                    im.verify()
                with Image.open(image_path) as im:
                    dims[im.size] += 1
                    modes[im.mode] += 1
                    arr = np.asarray(im.convert("RGB"))
                    if np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(
                        arr[:, :, 1], arr[:, :, 2]
                    ):
                        grayscale += 1
            except Exception as exc:  # noqa: BLE001
                corrupt.append((str(image_path), str(exc)))
                continue

            sha = sha256_file(image_path)
            dh = dhash(image_path)
            boxes, bad = read_boxes(label_path(image_path, dataset))
            if not boxes:
                split_empty_labels[split] += 1
            split_bad_lines[split] += bad
            for cls, x, y, w, h in boxes:
                class_counter[cls] += 1
                areas.append(w * h)
                if (
                    x - w / 2 <= 0.005
                    or y - h / 2 <= 0.005
                    or x + w / 2 >= 0.995
                    or y + h / 2 >= 0.995
                ):
                    border_boxes += 1

            rows.append(
                {
                    "path": str(image_path.relative_to(dataset)),
                    "split": split,
                    "source": source_name(image_path),
                    "sha256": sha,
                    "dhash": "".join("1" if b else "0" for b in dh),
                    "width": im.size[0],
                    "height": im.size[1],
                    "boxes": len(boxes),
                    "face_area_frac": float(sum(w * h for _, _, _, w, h in boxes)),
                    "touches_border": int(
                        any(
                            x - w / 2 <= 0.005
                            or y - h / 2 <= 0.005
                            or x + w / 2 >= 0.995
                            or y + h / 2 >= 0.995
                            for _, x, y, w, h in boxes
                        )
                    ),
                    "noise": noise_score(image_path),
                    "dhash_arr": dh,
                }
            )
            split_boxes[split] += len(boxes)

    # One representative (cleanest) per source name, then cluster by dHash.
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    source_to_rep = {}
    rep_hashes = {}
    for source, items in by_source.items():
        items.sort(key=lambda r: (r["noise"], r["path"]))
        source_to_rep[source] = items[0]["path"]
        rep_hashes[source] = items[0]["dhash_arr"]
    source_cluster = cluster_sources(source_to_rep, rep_hashes, args.hash_threshold)

    # Map every row to its cluster and choose the cleanest image per cluster.
    for row in rows:
        row["cluster_id"] = int(source_cluster[row["source"]])
    cluster_best = {}
    for row in rows:
        key = row["cluster_id"]
        if key not in cluster_best or (row["noise"], row["path"]) < (
            cluster_best[key]["noise"],
            cluster_best[key]["path"],
        ):
            cluster_best[key] = row

    # Split membership per cluster.
    cluster_splits = defaultdict(set)
    cluster_sources_map = defaultdict(set)
    for row in rows:
        cluster_splits[row["cluster_id"]].add(row["split"])
        cluster_sources_map[row["cluster_id"]].add(row["source"])

    # Exact duplicate hashes across splits.
    hash_splits = defaultdict(set)
    for row in rows:
        hash_splits[row["sha256"]].add(row["split"])
    exact_cross = {h: sorted(s) for h, s in hash_splits.items() if len(s) > 1}

    # Contamination: unique source representatives in valid/test near train.
    train_sources = {s for s, c in source_cluster.items() if "train" in cluster_splits[c]}
    contaminated = {"valid": 0, "test": 0}
    for split in ("valid", "test"):
        for source in {r["source"] for r in rows if r["split"] == split}:
            c = source_cluster[source]
            if "train" in cluster_splits[c]:
                contaminated[split] += 1

    # Augmentation pairs: train sources with >1 image.
    multi_train = {s: items for s, items in by_source.items() if len(items) > 1 and items[0]["split"] == "train"}
    aug_pairs = 0
    aug_noise_diffs = []
    for items in multi_train.values():
        if len(items) == 2:
            aug_pairs += 1
            aug_noise_diffs.append(abs(items[0]["noise"] - items[1]["noise"]))

    areas_arr = np.array(areas) if areas else np.zeros(1)
    px = areas_arr * 416 * 416
    stats = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "files": len(rows),
        "split_images": dict(split_images),
        "split_boxes": dict(split_boxes),
        "split_bad_lines": dict(split_bad_lines),
        "split_empty_labels": dict(split_empty_labels),
        "unique_sources": len(by_source),
        "source_splits": {
            split: len({r["source"] for r in rows if r["split"] == split})
            for split in SPLITS
        },
        "boxes": int(len(areas)),
        "classes": {str(k): v for k, v in sorted(class_counter.items())},
        "image_dimensions": {f"{w}x{h}": n for (w, h), n in dims.most_common()},
        "modes": dict(modes),
        "grayscale_content": grayscale,
        "corrupt_images": corrupt,
        "border_boxes": int(border_boxes),
        "face_area": {
            "min": float(areas_arr.min()),
            "p1": float(np.percentile(areas_arr, 1)),
            "p5": float(np.percentile(areas_arr, 5)),
            "median": float(np.median(areas_arr)),
            "p95": float(np.percentile(areas_arr, 95)),
            "max": float(areas_arr.max()),
        },
        "coco_sizes_at_416": {
            "small_lt_1024": int((px < 1024).sum()),
            "medium_1024_9216": int(((px >= 1024) & (px < 9216)).sum()),
            "large_ge_9216": int((px >= 9216).sum()),
        },
        "exact_cross_split_duplicates": len(exact_cross),
        "near_duplicate_clusters": sum(1 for c in cluster_splits if len(cluster_sources_map[c]) > 1),
        "near_duplicate_images": sum(len(cluster_sources_map[c]) for c in cluster_splits if len(cluster_sources_map[c]) > 1),
        "near_duplicate_clusters_spanning_splits": sum(
            1 for c in cluster_splits if len(cluster_splits[c]) > 1
        ),
        "contaminated": contaminated,
        "unique_clusters": len(cluster_splits),
        "augmentation_pairs_train": aug_pairs,
        "augmentation_noise_diff_median": float(np.median(aug_noise_diffs)) if aug_noise_diffs else 0.0,
    }

    # Write manifest (without the numpy hash array).
    manifest = out / "dataset_manifest.csv"
    fields = [
        "path", "split", "source", "cluster_id", "sha256", "dhash",
        "width", "height", "boxes", "face_area_frac", "touches_border", "noise",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["path"]):
            writer.writerow({k: row[k] for k in fields})

    # Leakage clusters.
    leaks = out / "leakage_clusters.csv"
    with leaks.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cluster_id", "size_sources", "splits", "sources", "best_image"])
        for cid in sorted(cluster_splits):
            if len(cluster_splits[cid]) > 1:
                writer.writerow([
                    cid,
                    len(cluster_sources_map[cid]),
                    "|".join(sorted(cluster_splits[cid])),
                    "|".join(sorted(cluster_sources_map[cid])),
                    cluster_best[cid]["path"],
                ])

    # Source -> cluster mapping for the split builder.
    (out / "source_clusters.json").write_text(
        json.dumps(
            {
                "threshold": args.hash_threshold,
                "source_cluster": source_cluster,
                "cluster_best": {str(k): v["path"] for k, v in cluster_best.items()},
                "cluster_splits": {str(k): sorted(v) for k, v in cluster_splits.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "audit_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    # Plots.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 3.5))
        plt.hist(areas_arr, bins=30, color="#4c72b0")
        plt.xlabel("Box area fraction of image")
        plt.ylabel("Count")
        plt.title("Face box area distribution")
        plt.tight_layout()
        plt.savefig(out / "face_size_hist.png", dpi=140)
        plt.close()

        per_image = np.array([r["boxes"] for r in rows])
        plt.figure(figsize=(6, 3.5))
        plt.hist(per_image, bins=range(0, int(per_image.max()) + 2), color="#dd8452", align="left")
        plt.xlabel("Annotated faces per image")
        plt.ylabel("Images")
        plt.title("Faces per image")
        plt.tight_layout()
        plt.savefig(out / "boxes_per_image_hist.png", dpi=140)
        plt.close()
    except Exception as exc:  # noqa: BLE001
        print(f"plot warning: {exc}")

    # Markdown report.
    md = []
    md.append("# Dataset Audit Report (Gate 1)")
    md.append("")
    md.append(f"Generated: {stats['generated']}")
    md.append(f"Dataset: `{dataset}`")
    md.append("")
    md.append("## Headline result")
    md.append("")
    md.append(
        f"- **{stats['files']} files**, **{stats['unique_sources']} unique source names**, "
        f"**{stats['unique_clusters']} near-duplicate clusters**."
    )
    md.append(
        f"- Generation gap explained: **{stats['files'] - stats['unique_sources']}** train-only "
        f"augmentation copies (clean + salt-and-pepper pairs)."
    )
    md.append(
        f"- Near-duplicate leakage: **{stats['near_duplicate_clusters_spanning_splits']} clusters span splits**; "
        f"valid contaminated **{stats['contaminated']['valid']}**, test contaminated **{stats['contaminated']['test']}**."
    )
    md.append(
        f"- Face size: **{stats['coco_sizes_at_416']['large_ge_9216']} large**, "
        f"{stats['coco_sizes_at_416']['medium_1024_9216']} medium, "
        f"{stats['coco_sizes_at_416']['small_lt_1024']} small at 416 px."
    )
    md.append("")
    md.append("## Counts")
    md.append("")
    md.append("| Split | Files | Unique sources | Boxes | Empty labels | Bad label lines |")
    md.append("|---|---|---|---|---|---|")
    for split in SPLITS:
        md.append(
            f"| {split} | {split_images[split]} | {stats['source_splits'][split]} | "
            f"{split_boxes[split]} | {split_empty_labels[split]} | {split_bad_lines[split]} |"
        )
    md.append(f"| **total** | **{stats['files']}** | **{stats['unique_sources']}** | **{stats['boxes']}** | | |")
    md.append("")
    md.append("## Integrity and labels")
    md.append("")
    md.append(f"- Corrupt images: **{len(corrupt)}**")
    md.append(f"- Classes: `{stats['classes']}` (single class expected)")
    md.append(f"- Image dimensions: `{stats['image_dimensions']}`")
    md.append(f"- File modes: `{stats['modes']}`")
    md.append(f"- Grayscale content: **{stats['grayscale_content']}/{stats['files']}**")
    md.append(f"- Border-touching boxes: **{stats['border_boxes']}**")
    md.append(f"- Face area fraction: min {stats['face_area']['min']:.4f}, p5 {stats['face_area']['p5']:.4f}, "
              f"median {stats['face_area']['median']:.4f}, p95 {stats['face_area']['p95']:.4f}, max {stats['face_area']['max']:.4f}")
    md.append(f"- COCO size buckets at 416 px: `{stats['coco_sizes_at_416']}`")
    md.append("")
    md.append("## Duplicates and leakage")
    md.append("")
    md.append(f"- Exact cross-split duplicate hashes: **{stats['exact_cross_split_duplicates']}**")
    md.append(f"- Near-duplicate clusters: **{stats['near_duplicate_clusters']}** "
              f"covering **{stats['near_duplicate_images']}** source images")
    md.append(f"- Clusters spanning splits: **{stats['near_duplicate_clusters_spanning_splits']}**")
    md.append(f"- Contaminated unique sources: valid **{stats['contaminated']['valid']}**, "
              f"test **{stats['contaminated']['test']}**")
    md.append("")
    md.append("## Augmentation")
    md.append("")
    md.append(f"- Train clean/noisy pairs detected: **{stats['augmentation_pairs_train']}**")
    md.append(f"- Median noise difference within a pair: **{stats['augmentation_noise_diff_median']:.4f}**")
    md.append("")
    md.append("## Diversity and limitations")
    md.append("")
    md.append("- All images are 416x416 and stretched; content is grayscale.")
    md.append("- The dataset is dominated by close-up, mostly frontal portraits; there is no crowd, small-face or low-light coverage to speak of.")
    md.append("- There are no negative/background images in the export.")
    md.append("- These limitations are recorded as findings, not assumed away. Per-size recall must be measured on an external set.")
    md.append("")
    md.append("## Artefacts")
    md.append("")
    md.append("- `dataset_manifest.csv` — every image with hashes, cluster id, boxes and noise score.")
    md.append("- `leakage_clusters.csv` — every cross-split near-duplicate cluster.")
    md.append("- `source_clusters.json` — cluster assignment used by the split builder.")
    md.append("- `audit_stats.json` — machine-readable results.")
    md.append("- `face_size_hist.png`, `boxes_per_image_hist.png`.")
    md.append("")
    (out / "audit_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'audit_report.md'}")
    print(json.dumps({k: stats[k] for k in (
        "files", "unique_sources", "unique_clusters", "near_duplicate_clusters",
        "near_duplicate_clusters_spanning_splits", "contaminated", "classes",
        "coco_sizes_at_416", "augmentation_pairs_train",
    )}, indent=2))


if __name__ == "__main__":
    main()
