#!/usr/bin/env python3
"""Compare frozen FP32 vs INT8 on the frozen test split and OOD.

Writes:
  reports/metrics_fp32_test.json, metrics_int8_test.json
  reports/metrics_fp32_ood.json, metrics_int8_ood.json (when OOD is labelled)
  reports/stage2_fp32_vs_int8.md
  reports/stage2_fp32_vs_int8.json

Example:
  python scripts/evaluate_stage2.py --config configs/stage2.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def ensure_ood_yaml(root: Path, ood_dir: Path) -> Path | None:
    """Build a YOLO data yaml for OOD when images+labels exist."""
    images_dir = ood_dir / "images" if (ood_dir / "images").is_dir() else ood_dir
    labels_dir = ood_dir / "labels"
    images = list_images(images_dir)
    if not images:
        return None
    labelled = 0
    for image in images:
        label = labels_dir / (image.stem + ".txt") if labels_dir.is_dir() else image.with_suffix(".txt")
        if label.is_file():
            labelled += 1
    if labelled == 0:
        return None

    # Prefer standard images/ layout for Ultralytics.
    if images_dir == ood_dir:
        # Flat OOD dir: synthesize a small dataset view via yaml path=ood_dir and train/val=.
        yaml_path = root / "configs" / "ood_eval.yaml"
        yaml_path.write_text(
            "\n".join([
                f"path: {ood_dir.resolve()}",
                "train: .",
                "val: .",
                "test: .",
                "nc: 1",
                "names: ['Face']",
                "",
            ]),
            encoding="utf-8",
        )
        return yaml_path

    yaml_path = root / "configs" / "ood_eval.yaml"
    yaml_path.write_text(
        "\n".join([
            f"path: {ood_dir.resolve()}",
            "train: images",
            "val: images",
            "test: images",
            "nc: 1",
            "names: ['Face']",
            "",
        ]),
        encoding="utf-8",
    )
    return yaml_path


def evaluate_weights(
    weights: Path,
    data_yaml: Path,
    split: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    name: str,
) -> dict:
    resolved = check_det_dataset(str(data_yaml))
    split_key = split if split in resolved else "test"
    split_dir = Path(resolved[split_key])
    images = list_images(split_dir)
    model = YOLO(str(weights))
    eval_device = "cpu" if weights.suffix.lower() == ".onnx" else device
    val = model.val(
        data=str(data_yaml),
        split=split_key,
        imgsz=imgsz,
        batch=8,
        conf=0.001,
        iou=0.7,
        device=eval_device,
        plots=False,
        verbose=False,
    )
    val_metrics = {k: float(v) for k, v in val.results_dict.items()}

    per_size = defaultdict(lambda: {"gt": 0, "tp": 0})
    total_gt = total_tp = total_pred = 0
    latencies = []
    for image_path in images:
        result = model.predict(
            source=str(image_path), imgsz=imgsz, conf=conf,
            iou=0.5, device=eval_device, verbose=False,
        )[0]
        latencies.append(float(result.speed.get("inference", 0.0)))
        h, w = result.orig_shape
        label_candidates = [
            image_path.parent.parent / "labels" / (image_path.stem + ".txt"),
            image_path.with_suffix(".txt"),
            image_path.parent / "labels" / (image_path.stem + ".txt"),
        ]
        label_file = next((p for p in label_candidates if p.is_file()), label_candidates[0])
        gt = load_gt(label_file, w, h)
        pred = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((0, 4))
        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.zeros((0,))
        pred = pred[confs >= conf]
        total_pred += len(pred)
        ious = iou_matrix(pred, gt)
        used = set()
        for gi, box in enumerate(gt):
            total_gt += 1
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            bucket = size_bucket(area)
            per_size[bucket]["gt"] += 1
            if ious.shape[0] and ious[:, gi].max() >= iou:
                pi = int(ious[:, gi].argmax())
                if pi not in used:
                    used.add(pi)
                    per_size[bucket]["tp"] += 1
                    total_tp += 1

    recall_by_size = {
        k: {"gt": v["gt"], "tp": v["tp"], "recall": (v["tp"] / v["gt"] if v["gt"] else None)}
        for k, v in per_size.items()
    }
    return {
        "name": name,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "weights": str(weights.resolve()),
        "data": str(data_yaml),
        "split": split_key,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
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
    }


def map5095(metrics: dict | None) -> float | None:
    if not metrics:
        return None
    return metrics.get("ultralytics_val", {}).get("metrics/mAP50-95(B)")


def fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/stage2.json")
    ap.add_argument("--fp32", default="models/stage1/best.pt")
    ap.add_argument("--int8", default="models/stage2/int8.engine")
    ap.add_argument("--data", default="configs/face_v2_dedup.yaml")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    imgsz = int(config.get("imgsz") or 416)
    allowed_test = float(config.get("allowed_map50_95_drop") or 0.02)
    allowed_ood = float(config.get("allowed_ood_map50_95_drop") or 0.03)

    fp32 = root / args.fp32
    int8 = root / args.int8
    if not int8.is_file():
        for candidate in (root / "models/stage2/int8.onnx", root / "models/stage2/int8.engine"):
            if candidate.is_file():
                int8 = candidate
                break
    data = root / args.data
    reports = root / "reports"
    reports.mkdir(exist_ok=True)

    if not fp32.is_file():
        print(f"ERROR: FP32 weights missing: {fp32}")
        return 2
    if not int8.is_file():
        print(f"ERROR: INT8 model missing: {int8}. Run quantize_ptq.py first.")
        return 2

    results = {}
    for tag, weights in (("fp32", fp32), ("int8", int8)):
        metrics = evaluate_weights(
            weights=weights,
            data_yaml=data,
            split="test",
            imgsz=imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            name=f"{tag}_test",
        )
        (reports / f"metrics_{tag}_test.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        results[f"{tag}_test"] = metrics

    ood_value = config.get("ood_dir") or "data/ood"
    ood_dir = Path(ood_value)
    if not ood_dir.is_absolute():
        ood_dir = root / ood_dir
    ood_yaml = ensure_ood_yaml(root, ood_dir)
    ood_note = None
    if ood_yaml is None:
        ood_note = (
            f"OOD directory `{ood_dir}` has no labelled images; "
            "OOD mAP comparison skipped. Provide YOLO labels under data/ood/labels."
        )
    else:
        for tag, weights in (("fp32", fp32), ("int8", int8)):
            metrics = evaluate_weights(
                weights=weights,
                data_yaml=ood_yaml,
                split="test",
                imgsz=imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                name=f"{tag}_ood",
            )
            (reports / f"metrics_{tag}_ood.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            results[f"{tag}_ood"] = metrics

    fp32_test = map5095(results.get("fp32_test"))
    int8_test = map5095(results.get("int8_test"))
    test_drop = None if fp32_test is None or int8_test is None else fp32_test - int8_test
    test_pass = test_drop is not None and test_drop <= allowed_test

    fp32_ood = map5095(results.get("fp32_ood"))
    int8_ood = map5095(results.get("int8_ood"))
    ood_drop = None if fp32_ood is None or int8_ood is None else fp32_ood - int8_ood
    ood_pass = ood_drop is not None and ood_drop <= allowed_ood
    if ood_note:
        ood_pass = False

    accuracy_pass = test_pass and (ood_pass if ood_yaml is not None else False)

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "imgsz": imgsz,
        "allowed_map50_95_drop": allowed_test,
        "allowed_ood_map50_95_drop": allowed_ood,
        "fp32_test_map50_95": fp32_test,
        "int8_test_map50_95": int8_test,
        "test_drop": test_drop,
        "test_pass": test_pass,
        "fp32_ood_map50_95": fp32_ood,
        "int8_ood_map50_95": int8_ood,
        "ood_drop": ood_drop,
        "ood_pass": ood_pass,
        "ood_note": ood_note,
        "accuracy_pass": accuracy_pass,
        "recall_by_size": {
            "fp32_test": results.get("fp32_test", {}).get("recall_by_size"),
            "int8_test": results.get("int8_test", {}).get("recall_by_size"),
            "fp32_ood": results.get("fp32_ood", {}).get("recall_by_size"),
            "int8_ood": results.get("int8_ood", {}).get("recall_by_size"),
        },
    }
    (reports / "stage2_fp32_vs_int8.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    md = [
        "# Stage 2 — FP32 vs INT8",
        "",
        f"Generated: {summary['generated']}",
        "",
        "INT8 is judged against the frozen Stage 1 FP32 checkpoint on the same splits.",
        "",
        "## Accuracy gate",
        "",
        f"| Set | FP32 mAP50-95 | INT8 mAP50-95 | Drop | Budget | Pass |",
        f"|---|---|---|---|---|---|",
        f"| Frozen test | {fmt(fp32_test)} | {fmt(int8_test)} | {fmt(test_drop)} | {allowed_test:.4f} | "
        f"{'PASS' if test_pass else 'FAIL'} |",
        f"| OOD | {fmt(fp32_ood)} | {fmt(int8_ood)} | {fmt(ood_drop)} | {allowed_ood:.4f} | "
        f"{'PASS' if ood_pass else 'FAIL'} |",
        "",
        f"- Overall accuracy gate: **{'PASS' if accuracy_pass else 'FAIL'}**",
    ]
    if ood_note:
        md += ["", f"> {ood_note}"]
    md += [
        "",
        "## Small-face recall (reported separately)",
        "",
        "| Set / model | small | medium | large |",
        "|---|---|---|---|",
    ]
    for key, label in (
        ("fp32_test", "test / FP32"),
        ("int8_test", "test / INT8"),
        ("fp32_ood", "OOD / FP32"),
        ("int8_ood", "OOD / INT8"),
    ):
        sizes = summary["recall_by_size"].get(key) or {}
        cells = []
        for bucket in ("small", "medium", "large"):
            recall = (sizes.get(bucket) or {}).get("recall")
            cells.append("n/a" if recall is None else f"{recall:.4f}")
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    md += [
        "",
        "## Next step",
        "",
    ]
    if accuracy_pass:
        md.append("- Accuracy gate passed. Run `scripts/bench_target.py` for Gate 5.")
    else:
        md.append(
            "- Accuracy gate failed. Do not promote INT8. "
            "Retry calibration (Q1) or run labelled QAT0; keep Q0 artefacts."
        )
    md.append("")
    (reports / "stage2_fp32_vs_int8.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if accuracy_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
