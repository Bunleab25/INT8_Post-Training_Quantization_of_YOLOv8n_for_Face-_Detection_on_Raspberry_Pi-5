#!/usr/bin/env python3
"""Stage 2 export X0: frozen Stage 1 best.pt → ONNX @ configured imgsz.

Example:
  python scripts/export_stage2.py
  python scripts/export_stage2.py --config configs/stage2.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

try:
    from scripts.common import load_expected_checkpoint_hash, load_json, sha256_file
except ImportError:
    from common import load_expected_checkpoint_hash, load_json, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/stage2.json")
    ap.add_argument("--weights", default="models/stage1/best.pt")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    config = load_json(config_path)
    imgsz = args.imgsz or int(config.get("imgsz") or 416)
    runtime = config.get("runtime") or "onnxruntime"

    weights = root / args.weights
    if not weights.is_file():
        print(f"ERROR: frozen checkpoint missing: {weights}")
        return 2

    expected_hash_file = root / "models/stage1/best.sha256"
    if expected_hash_file.is_file():
        expected = load_expected_checkpoint_hash(expected_hash_file, "models/stage1/best.pt")
        actual = sha256_file(weights)
        if expected and actual != expected:
            print("ERROR: models/stage1/best.pt does not match best.sha256; refusing export.")
            return 2

    out_dir = root / "exports" / "stage2" / str(runtime)
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(weights))
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        simplify=True,
        opset=12,
        device=args.device,
    )
    export_path = Path(export_path)
    dest = out_dir / "best.onnx"
    if export_path.resolve() != dest.resolve():
        shutil.copy2(export_path, dest)
        # Ultralytics writes next to the weights by default; keep a copy there too if present.
    digest = sha256_file(dest)
    (out_dir / "best.onnx.sha256").write_text(
        f"{digest}  {dest.relative_to(root).as_posix()}\n", encoding="utf-8"
    )
    info = {
        "experiment": "X0",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "onnx": str(dest),
        "onnx_sha256": digest,
        "imgsz": imgsz,
        "runtime": runtime,
        "format": "onnx",
    }
    (out_dir / "export_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
