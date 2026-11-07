#!/usr/bin/env python3
"""Stage 1 evaluation: mAP plus per-size recall on a frozen split.

Example:
  python scripts/evaluate.py --weights models/stage1/E1_best.pt --name E1 --imgsz 416
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def load_gt(label_file: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    if not label_file.exists():
        return np.zeros((0, 4), dtype=np.float32)
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, x, y, w, h = map(float, parts)
        boxes.append([(x - w / 2) * width, (y - h / 2) * height,
                      (x + w / 2) * width, (y + h / 2) * height])
    return np.asarray(boxes, dtype=np.float32)


def size_bucket(area_px: float) -> str:
    if area_px < 32 * 32:
        return "small"
    if area_px < 96 * 96:
        return "medium"
    return "large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="configs/face_v2_dedup.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data = (root / args.data).resolve()
    resolved = check_det_dataset(str(data))
    split_dir = Path(resolved[args.split])
    images = sorted([p for p in split_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    model = YOLO(args.weights)
    val = model.val(
        data=str(data), split=args.split, imgsz=args.imgsz, batch=args.batch,
        conf=0.001, iou=0.7, device=args.device, plots=True, verbose=False,
    )
    val_metrics = {k: float(v) for k, v in val.results_dict.items()}

    # Per-size recall at the chosen confidence threshold.
    per_size = defaultdict(lambda: {"gt": 0, "tp": 0})
    total_gt = total_tp = 0
    total_pred = 0
    latencies = []
    failures = []
    for image_path in images:
        result = model.predict(
            source=str(image_path), imgsz=args.imgsz, conf=args.conf,
            iou=0.5, device=args.device, verbose=False,
        )[0]
        latencies.append(float(result.speed.get("inference", 0.0)))
        h, w = result.orig_shape
        gt = load_gt(image_path.parent.parent / "labels" / (image_path.stem + ".txt"), w, h)
        pred = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((0, 4))
        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.zeros((0,))
        keep = confs >= args.conf
        pred = pred[keep]
        total_pred += len(pred)
        ious = iou_matrix(pred, gt)
        used = set()
        matched_this = 0
        for gi, box in enumerate(gt):
            total_gt += 1
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            b = size_bucket(area)
            per_size[b]["gt"] += 1
            if ious.shape[0] and ious[:, gi].max() >= args.iou:
                pi = int(ious[:, gi].argmax())
                if pi not in used:
                    used.add(pi)
                    per_size[b]["tp"] += 1
                    total_tp += 1
                    matched_this += 1
        if len(gt) and matched_this == 0 and len(pred) == 0:
            failures.append(image_path.name)

    recall_by_size = {
        k: {"gt": v["gt"], "tp": v["tp"], "recall": (v["tp"] / v["gt"] if v["gt"] else None)}
        for k, v in per_size.items()
    }
    metrics = {
        "name": args.name,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "weights": str(Path(args.weights).resolve()),
        "data": str(data),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "ultralytics_val": val_metrics,
        "overall": {
            "gt": total_gt,
            "matched": total_tp,
            "recall": total_tp / total_gt if total_gt else None,
            "predictions": total_pred,
            "images": len(images),
        },
        "recall_by_size": recall_by_size,
        "latency_ms_mean": float(np.mean(latencies)) if latencies else None,
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
        "images_with_no_tp_and_no_pred": len(failures),
    }

    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    (reports / f"metrics_{args.name}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    md = [
        f"# Evaluation — {args.name}",
        "",
        f"Generated: {metrics['generated']}",
        f"Weights: `{metrics['weights']}`",
        f"Split: `{args.split}` · imgsz {args.imgsz} · conf {args.conf} · IoU {args.iou}",
        "",
        "## Ultralytics validation",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k, v in val_metrics.items():
        md.append(f"| `{k}` | {v:.4f} |")
    md += [
        "",
        "## Recall at the operating threshold",
        "",
        "| Size bucket | GT | Matched | Recall |",
        "|---|---|---|---|",
    ]
    for b in ("small", "medium", "large"):
        v = recall_by_size.get(b, {"gt": 0, "tp": 0, "recall": None})
        r = "n/a" if v["recall"] is None else f"{v['recall']:.4f}"
        md.append(f"| {b} | {v['gt']} | {v['tp']} | {r} |")
    md += [
        "",
        f"- Overall matched recall: **{metrics['overall']['recall']}** "
        f"({metrics['overall']['matched']}/{metrics['overall']['gt']})",
        f"- Predictions above conf: {metrics['overall']['predictions']} on {metrics['overall']['images']} images",
        f"- Mean inference latency: {metrics['latency_ms_mean']} ms (p95 {metrics['latency_ms_p95']} ms)",
        "",
        "> Note: the Roboflow test split has almost no small faces, so the small bucket is not",
        "> statistically meaningful. Use an external small-face set for that claim.",
        "",
    ]
    (reports / f"evaluation_{args.name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
