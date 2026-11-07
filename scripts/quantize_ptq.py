#!/usr/bin/env python3
"""Stage 2 PTQ Q0: build a TensorRT INT8 engine from the Stage 2 ONNX export.

Uses only images listed in the calibration manifest (train/valid). Never reads test.

Example:
  python scripts/quantize_ptq.py
  python scripts/quantize_ptq.py --config configs/stage2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_calibration_images(manifest: dict, root: Path) -> list[Path]:
    test_root = (root / "datasets/face_v2_dedup/test").resolve()
    images: list[Path] = []
    for index, item in enumerate(manifest.get("images") or []):
        if not isinstance(item, dict):
            raise ValueError(f"Calibration entry {index} must be an object")
        split = item.get("split")
        if split not in {"train", "valid"}:
            raise ValueError(f"Calibration entry {index} has disallowed split {split!r}")
        path_value = item.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"Calibration entry {index} missing path")
        path = Path(path_value)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Calibration image missing: {path}")
        if path.is_relative_to(test_root):
            raise ValueError(f"Calibration image points into test split: {path}")
        images.append(path)
    if not images:
        raise ValueError("Calibration manifest has no images")
    return images


def build_int8_onnxruntime(
    onnx_path: Path,
    out_path: Path,
    images: list[Path],
    imgsz: int,
) -> dict:
    import numpy as np
    from PIL import Image
    import onnxruntime.quantization as ort_quant

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
            im = Image.open(p).convert("RGB").resize((self.imgsz, self.imgsz), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            return {"images": arr.transpose(2, 0, 1)[None, ...]}

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


def build_engine_tensorrt(
    onnx_path: Path,
    engine_path: Path,
    images: list[Path],
    imgsz: int,
) -> dict:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise SystemExit(
            "ERROR: tensorrt is not installed in this environment. "
            "Install TensorRT Python bindings into ai-ml, then re-run preflight."
        ) from exc

    import numpy as np
    from PIL import Image

    logger = trt.Logger(trt.Logger.WARNING)

    class FaceCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, image_paths: list[Path], batch_size: int = 1):
            super().__init__()
            self.image_paths = image_paths
            self.batch_size = batch_size
            self.index = 0
            self.batch = np.empty((batch_size, 3, imgsz, imgsz), dtype=np.float32)
            import pycuda.driver as cuda  # noqa: WPS410
            import pycuda.autoinit  # noqa: F401,WPS433
            self.cuda = cuda
            self.device_input = cuda.mem_alloc(self.batch.nbytes)
            self.cache_path = engine_path.with_suffix(".calib.cache")

        def get_batch_size(self) -> int:
            return self.batch_size

        def get_batch(self, names):  # noqa: ANN001
            if self.index + self.batch_size > len(self.image_paths):
                return None
            for i in range(self.batch_size):
                image = Image.open(self.image_paths[self.index + i]).convert("RGB")
                image = image.resize((imgsz, imgsz), Image.BILINEAR)
                arr = np.asarray(image, dtype=np.float32) / 255.0
                # CHW, RGB — matches Ultralytics ONNX export convention.
                self.batch[i] = arr.transpose(2, 0, 1)
            self.index += self.batch_size
            self.cuda.memcpy_htod(self.device_input, self.batch)
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if self.cache_path.is_file():
                return self.cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache):  # noqa: ANN001
            self.cache_path.write_bytes(cache)

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    with onnx_path.open("rb") as stream:
        if not parser.parse(stream.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if not builder.platform_has_fast_int8:
        raise RuntimeError("This TensorRT platform does not report fast INT8 support.")
    config.set_flag(trt.BuilderFlag.INT8)
    calibrator = FaceCalibrator(images)
    config.int8_calibrator = calibrator

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT INT8 engine build failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    return {
        "calibrator": "IInt8EntropyCalibrator2",
        "calibration_images": len(images),
        "imgsz": imgsz,
        "workspace_bytes": 1 << 30,
    }


def build_engine_ultralytics_fallback(
    weights: Path,
    engine_path: Path,
    data_yaml: Path,
    imgsz: int,
    device: str,
) -> dict:
    """Fallback when native TensorRT calibrator deps are unavailable but Ultralytics can export engine."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = model.export(
        format="engine",
        imgsz=imgsz,
        int8=True,
        data=str(data_yaml),
        device=device,
        simplify=True,
    )
    exported = Path(exported)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != engine_path.resolve():
        engine_path.write_bytes(exported.read_bytes())
    return {
        "calibrator": "ultralytics_export_int8",
        "data": str(data_yaml),
        "imgsz": imgsz,
        "note": "Used Ultralytics TensorRT INT8 export; calibration images come from data yaml train/val.",
    }


def write_calib_data_yaml(root: Path, images: list[Path], imgsz: int) -> Path:
    """Create a temporary YOLO data yaml whose train list is the calibration set only."""
    calib_dir = root / "calibration" / "dataset"
    image_dir = calib_dir / "images"
    label_dir = calib_dir / "labels"
    if image_dir.exists() or label_dir.exists():
        # Rebuild cleanly each run.
        import shutil

        shutil.rmtree(calib_dir, ignore_errors=True)
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    for image in images:
        dest = image_dir / image.name
        if not dest.exists():
            dest.symlink_to(image)
        # Labels live beside images in the dedup layout: .../{split}/images/x.jpg → .../labels/x.txt
        label_src = image.parent.parent / "labels" / (image.stem + ".txt")
        label_dest = label_dir / (image.stem + ".txt")
        if label_src.is_file() and not label_dest.exists():
            label_dest.symlink_to(label_src.resolve())
        elif not label_dest.exists():
            label_dest.write_text("", encoding="utf-8")

    yaml_path = root / "calibration" / "calib_data.yaml"
    yaml_path.write_text(
        "\n".join([
            f"path: {calib_dir}",
            "train: images",
            "val: images",
            "nc: 1",
            "names: ['Face']",
            f"# imgsz={imgsz} calibration-only; never includes test",
            "",
        ]),
        encoding="utf-8",
    )
    return yaml_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/stage2.json")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--weights", default="models/stage1/best.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--backend", choices=("tensorrt", "ultralytics"), default="tensorrt")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    if not config_path.is_file():
        print(f"ERROR: missing config {config_path}; copy stage2.template.json first")
        return 2
    config = load_json(config_path)
    imgsz = int(config.get("imgsz") or 416)
    runtime = config.get("runtime") or "tensorrt"
    if runtime not in {"tensorrt", "onnxruntime"}:
        print(f"ERROR: quantize_ptq.py supports runtime='onnxruntime' or 'tensorrt', got {runtime!r}")
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

    quant_meta: dict
    if runtime == "onnxruntime":
        default_onnx = root / "exports/stage2/onnxruntime/best.onnx"
        onnx_path = Path(args.onnx) if args.onnx else default_onnx
        if not onnx_path.is_file():
            print(f"ERROR: ONNX export missing: {onnx_path}. Run export_stage2.py first.")
            return 2
        out_model = models_dir / "int8.onnx"
        quant_meta = build_int8_onnxruntime(onnx_path, out_model, images, imgsz)
        model_target = out_model
        model_rel = "models/stage2/int8.onnx"
    else:
        default_onnx = root / "exports/stage2/tensorrt/best.onnx"
        onnx_path = Path(args.onnx) if args.onnx else default_onnx
        engine_path = models_dir / "int8.engine"
        if args.backend == "tensorrt":
            if not onnx_path.is_file():
                print(f"ERROR: ONNX export missing: {onnx_path}. Run export_stage2.py first.")
                return 2
            try:
                quant_meta = build_engine_tensorrt(onnx_path, engine_path, images, imgsz)
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"Native TensorRT build failed ({exc}); trying Ultralytics INT8 export fallback.")
                calib_yaml = write_calib_data_yaml(root, images, imgsz)
                quant_meta = build_engine_ultralytics_fallback(
                    root / args.weights, engine_path, calib_yaml, imgsz, args.device
                )
                quant_meta["fallback_from"] = str(exc)
        else:
            calib_yaml = write_calib_data_yaml(root, images, imgsz)
            quant_meta = build_engine_ultralytics_fallback(
                root / args.weights, engine_path, calib_yaml, imgsz, args.device
            )
        model_target = engine_path
        model_rel = "models/stage2/int8.engine"

    digest = sha256_file(model_target)
    (models_dir / f"{model_target.name}.sha256").write_text(
        f"{digest}  {model_rel}\n", encoding="utf-8"
    )
    info = {
        "experiment": "Q0",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "runtime": runtime,
        "onnx": str(onnx_path) if onnx_path.is_file() else None,
        "engine": str(model_target) if runtime == "tensorrt" else None,
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
