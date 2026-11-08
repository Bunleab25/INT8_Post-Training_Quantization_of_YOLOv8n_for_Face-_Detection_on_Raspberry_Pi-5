#!/usr/bin/env python3
"""Stage 1 evaluation: mAP plus per-size recall on a frozen split.

Example:
  python scripts/evaluate.py --weights models/stage1/E1_best.pt --name E1 --imgsz 416
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.eval_core import evaluate_weights, iou_matrix, load_gt, size_bucket
except ImportError:
    from eval_core import evaluate_weights, iou_matrix, load_gt, size_bucket

__all__ = ["iou_matrix", "load_gt", "size_bucket", "evaluate_weights"]


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
    metrics = evaluate_weights(
        weights=Path(args.weights),
        data_yaml=data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        name=args.name,
        batch=args.batch,
        plots=True,
    )
    val_metrics = metrics["ultralytics_val"]
    recall_by_size = metrics["recall_by_size"]

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
