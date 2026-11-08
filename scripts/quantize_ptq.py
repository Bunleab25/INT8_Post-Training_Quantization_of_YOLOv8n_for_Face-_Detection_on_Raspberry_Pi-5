#!/usr/bin/env python3
"""Stage 2 PTQ Q0: build an ONNX Runtime INT8 model from the Stage 2 ONNX export.

Uses only images listed in the calibration manifest (train/valid). Never reads test.

Example:
  python scripts/quantize_ptq.py
  python scripts/quantize_ptq.py --config configs/stage2.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

try:
    from scripts.common import load_json, sha256_file
    from scripts.stage2_preflight import resolve_calibration_images
except ImportError:
    from common import load_json, sha256_file
    from stage2_preflight import resolve_calibration_images


def build_int8_onnxruntime(
    onnx_path: Path,
    out_path: Path,
    images: list[Path],
    imgsz: int,
) -> dict:
    import cv2
    import numpy as np
    import onnxruntime.quantization as ort_quant

    try:
        from pi.face_detector import letterbox
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from pi.face_detector import letterbox

    class FaceDataReader(ort_quant.CalibrationDataReader):
        def __init__(self, image_paths: list[Path], target_imgsz: int):
            self.image_paths = image_paths
            self.imgsz = target_imgsz
            self.enum_data = None

        def get_next(self):
            if self.enum_data is None:
                self.enum_data = iter(self.image_paths)
            try:
                p = next(self.enum_data)
            except StopIteration:
                return None
            bgr = cv2.imread(str(p))
            if bgr is None:
                raise FileNotFoundError(f"Could not read calibration image: {p}")
            padded, _, _ = letterbox(bgr, self.imgsz)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            arr = rgb.astype(np.float32) / 255.0
            return {"images": np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...])}

        def rewind(self):
            self.enum_data = None

    reader = FaceDataReader(images, imgsz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ort_quant.quantize_static(
        model_input=str(onnx_path),
        model_output=str(out_path),
        calibration_data_reader=reader,
        quant_format=ort_quant.QuantFormat.QDQ,
        op_types_to_quantize=["Conv"],
        per_channel=False,
        weight_type=ort_quant.QuantType.QInt8,
        activation_type=ort_quant.QuantType.QUInt8,
        calibrate_method=ort_quant.CalibrationMethod.MinMax,
    )
    return {
        "calibrator": "onnxruntime_static_QDQ_MinMax_ConvOnly",
        "calibration_images": len(images),
        "imgsz": imgsz,
        "format": "onnx",
        "per_channel": False,
        "op_types_to_quantize": ["Conv"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/stage2.json")
    ap.add_argument("--onnx", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    if not config_path.is_file():
        print(f"ERROR: missing config {config_path}; copy stage2.template.json first")
        return 2
    config = load_json(config_path)
    imgsz = int(config.get("imgsz") or 416)
    runtime = config.get("runtime") or "onnxruntime"
    if runtime != "onnxruntime":
        print(f"ERROR: quantize_ptq.py supports runtime='onnxruntime', got {runtime!r}")
        return 2

    manifest_value = config.get("calibration_manifest") or "calibration/images.json"
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not manifest_path.is_file():
        print(f"ERROR: calibration manifest missing: {manifest_path}")
        return 2
    manifest = load_json(manifest_path)
    try:
        images = resolve_calibration_images(manifest, root)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 2

    models_dir = root / "models" / "stage2"
    models_dir.mkdir(parents=True, exist_ok=True)

    default_onnx = root / "exports/stage2/onnxruntime/best.onnx"
    onnx_path = Path(args.onnx) if args.onnx else default_onnx
    if not onnx_path.is_file():
        print(f"ERROR: ONNX export missing: {onnx_path}. Run export_stage2.py first.")
        return 2
    model_target = models_dir / "int8.onnx"
    quant_meta = build_int8_onnxruntime(onnx_path, model_target, images, imgsz)
    model_rel = "models/stage2/int8.onnx"

    digest = sha256_file(model_target)
    (models_dir / f"{model_target.name}.sha256").write_text(
        f"{digest}  {model_rel}\n", encoding="utf-8"
    )
    info = {
        "experiment": "Q0",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "runtime": runtime,
        "onnx": str(onnx_path),
        "engine": None,
        "quantized_model": str(model_target),
        "model_sha256": digest,
        "calibration_manifest": str(manifest_path),
        "calibration_count": len(images),
        "imgsz": imgsz,
        "quantization": quant_meta,
    }
    (models_dir / "int8_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (root / "reports" / "quantization_Q0.md").write_text(
        "\n".join([
            "# Quantization Q0 — PTQ INT8",
            "",
            f"Generated: {info['generated']}",
            "",
            f"- Quantized model: `{model_target.relative_to(root)}`",
            f"- SHA256: `{digest}`",
            f"- Calibration images: **{len(images)}** from `{manifest_path.relative_to(root)}`",
            f"- Method: `{quant_meta.get('calibrator')}`",
            f"- imgsz: {imgsz}",
            f"- Runtime: `{runtime}`",
            "",
            "Test-split images were not used.",
            "",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
