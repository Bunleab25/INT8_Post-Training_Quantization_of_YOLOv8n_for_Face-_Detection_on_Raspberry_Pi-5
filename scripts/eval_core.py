"""Shared object-detection evaluation and COCO size-bucketing utilities."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from scripts.common import IMAGE_SUFFIXES
except ImportError:
    from common import IMAGE_SUFFIXES


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between two sets of [x1, y1, x2, y2] boxes."""
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
    """Load YOLO normalized ground-truth boxes and scale to pixel [x1, y1, x2, y2]."""
    boxes = []
    if not label_file.exists():
        return np.zeros((0, 4), dtype=np.float32)
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, x, y, w, h = map(float, parts)
        boxes.append([
            (x - w / 2) * width,
            (y - h / 2) * height,
            (x + w / 2) * width,
            (y + h / 2) * height,
        ])
    return np.asarray(boxes, dtype=np.float32)


def size_bucket(area_px: float) -> str:
    """Return COCO size bucket ('small', 'medium', 'large') from pixel area."""
    if area_px < 32 * 32:
        return "small"
    if area_px < 96 * 96:
        return "medium"
    return "large"


def bucket_from_fraction(area_frac: float, ref_size: int = 416) -> str:
    """Return COCO size bucket from normalized area fraction at `ref_size`."""
    return size_bucket(area_frac * ref_size * ref_size)


def list_images(directory: Path) -> list[Path]:
    """Recursively list sorted image files in `directory`."""
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def evaluate_weights(
    weights: Path,
    data_yaml: Path,
    split: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    name: str,
    batch: int = 8,
    plots: bool = False,
) -> dict:
    """Run Ultralytics validation and per-size recall evaluation on a split."""
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

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
        batch=batch,
        conf=0.001,
        iou=0.7,
        device=eval_device,
        plots=plots,
        verbose=False,
    )
    val_metrics = {k: float(v) for k, v in val.results_dict.items()}

    per_size = defaultdict(lambda: {"gt": 0, "tp": 0})
    total_gt = total_tp = total_pred = 0
    latencies = []
    failures = []
    for image_path in images:
        result = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=0.5,
            device=eval_device,
            verbose=False,
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
        matched_this = 0
        for gi, box in enumerate(gt):
            total_gt += 1
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            b = size_bucket(area)
            per_size[b]["gt"] += 1
            if ious.shape[0] and ious[:, gi].max() >= iou:
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
    return {
        "name": name,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "weights": str(Path(weights).resolve()),
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
        "images_with_no_tp_and_no_pred": len(failures),
    }
