#!/usr/bin/env python3
"""Validate Stage 2 prerequisites before export or quantization."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

try:
    from scripts.common import IMAGE_SUFFIXES, load_expected_checkpoint_hash, sha256_file
except ImportError:
    from common import IMAGE_SUFFIXES, load_expected_checkpoint_hash, sha256_file


RUNTIME_MODULES = {
    "tensorrt": ("tensorrt",),
    "openvino": ("openvino",),
    "onnxruntime": ("onnxruntime",),
    "tflite": ("tflite_runtime", "tensorflow"),
}
ALLOWED_RUNTIMES = set(RUNTIME_MODULES)
ALLOWED_LATENCY_DEFINITIONS = {"model_only", "e2e"}


def parse_calibration_manifest(
    source: Path | dict[str, Any],
    root: Path,
) -> tuple[list[Path], list[str]]:
    """Validate calibration manifest entries and return (resolved_paths, problems)."""
    problems: list[str] = []
    resolved_images: list[Path] = []

    if isinstance(source, Path):
        if not source.is_file():
            return [], [f"Calibration manifest does not exist: {source}"]
        try:
            manifest: Any = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return [], [f"Calibration manifest is not valid JSON: {exc}"]
    else:
        manifest = source

    images = manifest.get("images") if isinstance(manifest, dict) else None
    if not isinstance(images, list) or not images:
        return [], ["Calibration manifest must contain a non-empty 'images' list."]

    test_root = (root / "datasets/face_v2_dedup/test").resolve()
    for index, item in enumerate(images):
        prefix = f"Calibration image entry {index}"
        if not isinstance(item, dict):
            problems.append(f"{prefix} must be an object.")
            continue
        split = item.get("split")
        image_value = item.get("path")
        if split not in {"train", "valid"}:
            problems.append(f"{prefix} has disallowed split {split!r}; use train or valid only.")
        if not isinstance(image_value, str) or not image_value:
            problems.append(f"{prefix} is missing a non-empty path.")
            continue
        image = Path(image_value)
        if not image.is_absolute():
            image = root / image
        image = image.resolve()
        if not image.is_file():
            problems.append(f"{prefix} image does not exist: {image}")
        if image.is_relative_to(test_root):
            problems.append(f"{prefix} points into the frozen test split: {image}")
        resolved_images.append(image)
    return resolved_images, problems


def validate_calibration_manifest(path: Path, root: Path) -> list[str]:
    _, problems = parse_calibration_manifest(path, root)
    return problems


def resolve_calibration_images(source: Path | dict[str, Any], root: Path) -> list[Path]:
    """Return validated train/valid calibration image paths or raise ValueError."""
    images, problems = parse_calibration_manifest(source, root)
    if problems:
        raise ValueError("; ".join(problems))
    return images


def _positive_int(config: dict[str, Any], key: str, blockers: list[str]) -> None:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        blockers.append(f"A positive integer {key} is required.")


def validate_config(config: dict[str, Any], root: Path) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    notes: list[str] = []

    if config.get("stage1_gate4_approved") is not True:
        blockers.append("Stage 1 Gate 4 approval is not recorded.")

    target = config.get("target_device")
    if not isinstance(target, str) or not target.strip():
        blockers.append("Deployment target_device is not specified.")

    runtime = config.get("runtime")
    if runtime not in ALLOWED_RUNTIMES:
        blockers.append(f"Runtime must be one of: {', '.join(sorted(ALLOWED_RUNTIMES))}.")
    elif not any(importlib.util.find_spec(module) for module in RUNTIME_MODULES[runtime]):
        blockers.append(f"Runtime package for {runtime!r} is not installed in this environment.")

    latency = config.get("latency_target_ms")
    if not isinstance(latency, (int, float)) or isinstance(latency, bool) or latency <= 0:
        blockers.append("A positive latency_target_ms is required.")

    accuracy_drop = config.get("allowed_map50_95_drop")
    if (not isinstance(accuracy_drop, (int, float)) or isinstance(accuracy_drop, bool)
            or accuracy_drop < 0):
        blockers.append("A non-negative allowed_map50_95_drop is required.")

    ood_drop = config.get("allowed_ood_map50_95_drop")
    if (not isinstance(ood_drop, (int, float)) or isinstance(ood_drop, bool)
            or ood_drop < 0):
        blockers.append("A non-negative allowed_ood_map50_95_drop is required.")

    imgsz = config.get("imgsz")
    if not isinstance(imgsz, int) or isinstance(imgsz, bool) or imgsz <= 0:
        blockers.append("A positive integer imgsz is required.")

    latency_definition = config.get("latency_definition")
    if latency_definition not in ALLOWED_LATENCY_DEFINITIONS:
        blockers.append(
            f"latency_definition must be one of: {', '.join(sorted(ALLOWED_LATENCY_DEFINITIONS))}."
        )

    _positive_int(config, "warmup_runs", blockers)
    _positive_int(config, "bench_runs", blockers)
    _positive_int(config, "calibration_count", blockers)

    ood_value = config.get("ood_dir")
    if not isinstance(ood_value, str) or not ood_value.strip():
        blockers.append("An external OOD image directory is required.")
    else:
        ood_dir = Path(ood_value)
        if not ood_dir.is_absolute():
            ood_dir = root / ood_dir
        has_images = ood_dir.is_dir() and any(
            p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in ood_dir.rglob("*")
        )
        if not has_images:
            blockers.append(
                f"OOD directory is missing or has no images "
                f"({', '.join(sorted(IMAGE_SUFFIXES))}): {ood_dir.resolve()}"
            )

    calibration_value = config.get("calibration_manifest")
    if not isinstance(calibration_value, str) or not calibration_value.strip():
        blockers.append("A calibration_manifest containing train/validation images is required.")
    else:
        manifest_path = Path(calibration_value)
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
        blockers.extend(validate_calibration_manifest(manifest_path.resolve(), root))
        if isinstance(config.get("calibration_count"), int) and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                images = manifest.get("images") if isinstance(manifest, dict) else None
                if isinstance(images, list) and len(images) < config["calibration_count"]:
                    notes.append(
                        f"Calibration manifest has {len(images)} images; "
                        f"config calibration_count is {config['calibration_count']}."
                    )
            except json.JSONDecodeError:
                pass

    checkpoint = root / "models/stage1/best.pt"
    checksum_file = root / "models/stage1/best.sha256"
    if not checkpoint.is_file():
        blockers.append(f"Frozen Stage 1 checkpoint is missing: {checkpoint}")
    else:
        expected = load_expected_checkpoint_hash(checksum_file, Path("models/stage1/best.pt"))
        if expected is None:
            blockers.append(f"Expected checkpoint hash is missing from {checksum_file}.")
        elif sha256_file(checkpoint) != expected:
            blockers.append("Frozen Stage 1 checkpoint does not match models/stage1/best.sha256.")
        else:
            notes.append(f"Frozen checkpoint verified: {checkpoint.relative_to(root)}")

    return blockers, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage2.json",
                        help="Stage 2 JSON configuration (default: configs/stage2.json)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    if not config_path.is_file():
        print(f"BLOCKED: Stage 2 config does not exist: {config_path}")
        print("Copy configs/stage2.template.json and fill in the owner-approved values.")
        return 2
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"BLOCKED: invalid JSON in {config_path}: {exc}")
        return 2
    if not isinstance(config, dict):
        print(f"BLOCKED: Stage 2 config must be a JSON object: {config_path}")
        return 2

    blockers, notes = validate_config(config, root)
    for note in notes:
        print(f"PASS: {note}")
    if blockers:
        for blocker in blockers:
            print(f"BLOCKED: {blocker}")
        print(f"Stage 2 preflight failed with {len(blockers)} blocker(s); no export or quantization was run.")
        return 2
    print("Stage 2 prerequisites are satisfied. This check does not export or quantize the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
