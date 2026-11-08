#!/usr/bin/env python3
"""Build a train/valid-only calibration manifest for Stage 2 PTQ.

Samples ~calibration_count images from datasets/face_v2_dedup/{train,valid},
stratified by face-size bucket and split. Never includes the frozen test split.

Example:
  python scripts/make_calibration_manifest.py
  python scripts/make_calibration_manifest.py --count 300 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from scripts.common import IMAGE_SUFFIXES, load_json, source_name as source_key
    from scripts.eval_core import size_bucket
except ImportError:
    from common import IMAGE_SUFFIXES, load_json, source_name as source_key
    from eval_core import size_bucket

SPLITS = ("train", "valid")


def max_face_area(label_file: Path, width: int = 416, height: int = 416) -> float:
    if not label_file.is_file():
        return 0.0
    areas = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, _, _, w, h = map(float, parts)
        areas.append((w * width) * (h * height))
    return max(areas) if areas else 0.0


def collect_candidates(dataset: Path) -> list[dict]:
    items: list[dict] = []
    for split in SPLITS:
        image_dir = dataset / split / "images"
        label_dir = dataset / split / "labels"
        if not image_dir.is_dir():
            continue
        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = label_dir / (image.stem + ".txt")
            area = max_face_area(label)
            items.append({
                "path": image,
                "split": split,
                "bucket": size_bucket(area),
                "source": source_key(image),
                "area": area,
            })
    return items


def stratified_sample(items: list[dict], count: int, seed: int) -> list[dict]:
    """Sample up to count images, preferring one image per source, stratified by bucket+split."""
    rng = random.Random(seed)
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in items:
        by_group[(item["split"], item["bucket"])].append(item)

    # Deduplicate by source within each group, keep largest face.
    deduped_groups: dict[tuple[str, str], list[dict]] = {}
    for key, group in by_group.items():
        best_by_source: dict[str, dict] = {}
        for item in group:
            prev = best_by_source.get(item["source"])
            if prev is None or item["area"] > prev["area"]:
                best_by_source[item["source"]] = item
        values = list(best_by_source.values())
        rng.shuffle(values)
        deduped_groups[key] = values

    selected: list[dict] = []
    selected_sources: set[str] = set()
    groups = sorted(deduped_groups.keys())
    # Round-robin across groups until count is reached.
    indices = {key: 0 for key in groups}
    while len(selected) < count:
        progressed = False
        for key in groups:
            idx = indices[key]
            pool = deduped_groups[key]
            while idx < len(pool) and pool[idx]["source"] in selected_sources:
                idx += 1
            if idx >= len(pool):
                indices[key] = idx
                continue
            item = pool[idx]
            indices[key] = idx + 1
            selected.append(item)
            selected_sources.add(item["source"])
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break

    # If still short, allow additional images from remaining pools (same source ok).
    if len(selected) < count:
        leftovers = [item for group in deduped_groups.values() for item in group if item not in selected]
        rng.shuffle(leftovers)
        for item in leftovers:
            selected.append(item)
            if len(selected) >= count:
                break

    rng.shuffle(selected)
    return selected[:count]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="datasets/face_v2_dedup")
    ap.add_argument("--count", type=int, default=None,
                    help="Target image count (default: calibration_count from stage2 config or 300)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="configs/stage2.template.json")
    ap.add_argument("--out", default="calibration/images.json")
    ap.add_argument("--report", default="reports/calibration_report.md")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset = (root / args.dataset).resolve()
    config_path = root / args.config
    count = args.count
    if count is None and config_path.is_file():
        config = load_json(config_path)
        if isinstance(config.get("calibration_count"), int) and config["calibration_count"] > 0:
            count = config["calibration_count"]
    if count is None:
        count = 300

    if not dataset.is_dir():
        print(f"ERROR: dataset not found: {dataset}")
        return 2

    candidates = collect_candidates(dataset)
    if not candidates:
        print(f"ERROR: no train/valid images under {dataset}")
        return 2

    # Hard reject anything under test/.
    test_root = (dataset / "test").resolve()
    for item in candidates:
        if item["path"].resolve().is_relative_to(test_root):
            print(f"ERROR: refused test-split path during candidate scan: {item['path']}")
            return 2

    selected = stratified_sample(candidates, count, args.seed)
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "requested_count": count,
        "actual_count": len(selected),
        "dataset": str(dataset),
        "images": [
            {
                "path": str(item["path"].resolve().relative_to(root)),
                "split": item["split"],
                "bucket": item["bucket"],
                "source": item["source"],
            }
            for item in selected
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    bucket_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, int] = defaultdict(int)
    for item in selected:
        bucket_counts[item["bucket"]] += 1
        split_counts[item["split"]] += 1

    report_path = root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# Calibration Report",
        "",
        f"Generated: {payload['generated']}",
        "",
        f"- Dataset: `{dataset}`",
        f"- Manifest: `{out_path.relative_to(root)}`",
        f"- Requested count: **{count}**; actual: **{len(selected)}**",
        f"- Seed: {args.seed}",
        f"- Candidate pool (train+valid): **{len(candidates)}**",
        "- Test split images included: **0** (hard rule)",
        "",
        "## Split counts",
        "",
        "| Split | Count |",
        "|---|---|",
    ]
    for split in SPLITS:
        md.append(f"| {split} | {split_counts.get(split, 0)} |")
    md += [
        "",
        "## Face-size buckets",
        "",
        "| Bucket | Count |",
        "|---|---|",
    ]
    for bucket in ("small", "medium", "large"):
        md.append(f"| {bucket} | {bucket_counts.get(bucket, 0)} |")
    md += [
        "",
        "## Notes",
        "",
        "- Sampling prefers one image per Roboflow source within each split×bucket group.",
        "- Use this manifest only for PTQ calibration. Never add test or OOD images.",
        "",
    ]
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(out_path),
        "report": str(report_path),
        "actual_count": len(selected),
        "split_counts": dict(split_counts),
        "bucket_counts": dict(bucket_counts),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
