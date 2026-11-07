#!/usr/bin/env python3
"""Stage 1 training entry point (FP32 by default).

Examples:
  python scripts/train_stage1.py --name S0 --imgsz 416 --batch 8 --epochs 2
  python scripts/train_stage1.py --name E0 --imgsz 640 --batch 4 --epochs 100
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
from datetime import datetime
from pathlib import Path

import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/face_v2_dedup.yaml")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="runs/stage1")
    ap.add_argument("--device", default="0")
    ap.add_argument("--amp", action="store_true", help="Enable AMP (off for the FP32 baseline)")
    ap.add_argument("--exist-ok", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data = (root / args.data).resolve()
    project = (root / args.project).resolve()
    project.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = datetime.now()
    model = YOLO(args.weights)
    result = model.train(
        data=str(data),
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        amp=args.amp,
        cache=False,
        device=args.device,
        project=str(project),
        name=args.name,
        exist_ok=args.exist_ok,
        plots=True,
        val=True,
        verbose=True,
        pretrained=True,
    )

    save_dir = Path(result.save_dir)
    peak_gib = None
    if torch.cuda.is_available():
        peak_gib = round(torch.cuda.max_memory_allocated() / 1024**3, 3)

    # Validate the best checkpoint explicitly and copy it to a stable location.
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, root / "models" / "stage1" / f"{args.name}_best.pt")
    val_metrics = {}
    if best.exists():
        val_model = YOLO(str(best))
        val = val_model.val(data=str(data), imgsz=args.imgsz, batch=args.batch, split="val", device=args.device)
        for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"):
            val_metrics[key] = float(val.results_dict.get(key, float("nan")))

    info = {
        "name": args.name,
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "data": str(data),
        "save_dir": str(save_dir),
        "best": str(best),
        "peak_gpu_gib": peak_gib,
        "val_metrics": val_metrics,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "ultralytics": __import__("ultralytics").__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (save_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
