#!/usr/bin/env python3
"""Gate 5: measure model-only latency and peak memory for the INT8 engine.

Protocol defaults from configs/stage2.json:
  latency_definition=model_only, warmup_runs=50, bench_runs=200, batch=1, imgsz=416

Example:
  python scripts/bench_target.py --config configs/stage2.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def load_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def median_ms(samples: list[float]) -> float:
    return float(np.median(samples))


def p95_ms(samples: list[float]) -> float:
    return float(np.percentile(samples, 95))


def bench_ultralytics(weights: Path, imgsz: int, warmup: int, runs: int, device: str) -> dict:
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights))
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warm-up
    for _ in range(warmup):
        model.predict(source=dummy, imgsz=imgsz, device=device, verbose=False)

    samples = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = model.predict(source=dummy, imgsz=imgsz, device=device, verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # Prefer Ultralytics model-only inference timing when present.
        infer_ms = float(result.speed.get("inference", 0.0))
        if infer_ms > 0:
            samples.append(infer_ms)
        else:
            samples.append((time.perf_counter() - t0) * 1000.0)

    peak_gib = None
    if torch.cuda.is_available():
        peak_gib = round(torch.cuda.max_memory_allocated() / 1024**3, 3)

    return {
        "backend": "ultralytics",
        "samples": len(samples),
        "latency_ms_median": median_ms(samples),
        "latency_ms_p95": p95_ms(samples),
        "latency_ms_mean": float(np.mean(samples)),
        "peak_gpu_gib": peak_gib,
    }


def promote_best_int8(root: Path, engine: Path, pass_gate: bool) -> Path | None:
    if not pass_gate or not engine.is_file():
        return None
    dest = root / "models" / "stage2" / f"best_int8{engine.suffix}"
    shutil.copy2(engine, dest)
    digest = __import__("hashlib").sha256(dest.read_bytes()).hexdigest()
    (root / "models" / "stage2" / f"best_int8{engine.suffix}.sha256").write_text(
        f"{digest}  models/stage2/best_int8{engine.suffix}\n", encoding="utf-8"
    )
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/stage2.json")
    ap.add_argument("--engine", default="models/stage2/int8.engine")
    ap.add_argument("--fp32", default="models/stage1/best.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--promote", action="store_true", default=True,
                    help="Promote best_int8 when Gate 5 and accuracy both pass (default on)")
    ap.add_argument("--no-promote", action="store_false", dest="promote")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    imgsz = int(config.get("imgsz") or 416)
    warmup = int(config.get("warmup_runs") or 50)
    runs = int(config.get("bench_runs") or 200)
    latency_target = float(config.get("latency_target_ms") or 10)
    latency_definition = config.get("latency_definition") or "model_only"
    target_device = config.get("target_device") or "unspecified"

    engine = root / args.engine
    if not engine.is_file():
        for candidate in (root / "models/stage2/int8.onnx", root / "models/stage2/int8.engine"):
            if candidate.is_file():
                engine = candidate
                break
    if not engine.is_file():
        print(f"ERROR: INT8 model missing: {engine}")
        return 2

    # Run benchmark; ONNX models evaluate on CPU provider if onnxruntime-gpu is not configured
    int8_device = "cpu" if engine.suffix.lower() == ".onnx" else args.device
    int8_bench = bench_ultralytics(engine, imgsz, warmup, runs, int8_device)
    fp32_path = root / args.fp32
    fp32_bench = None
    if fp32_path.is_file():
        fp32_bench = bench_ultralytics(fp32_path, imgsz, min(warmup, 20), min(runs, 50), args.device)

    latency_pass = int8_bench["latency_ms_median"] <= latency_target
    # 4 GB laptop GPU: require peak under ~3.5 GiB usable headroom.
    memory_limit_gib = 3.5
    memory_pass = (
        int8_bench["peak_gpu_gib"] is None
        or int8_bench["peak_gpu_gib"] <= memory_limit_gib
    )
    gate5_pass = latency_pass and memory_pass

    accuracy = {}
    accuracy_path = root / "reports" / "stage2_fp32_vs_int8.json"
    if accuracy_path.is_file():
        accuracy = json.loads(accuracy_path.read_text(encoding="utf-8"))
    accuracy_pass = bool(accuracy.get("accuracy_pass"))
    overall_pass = gate5_pass and accuracy_pass

    promoted = None
    if args.promote and overall_pass:
        promoted = promote_best_int8(root, engine, True)

    payload = {
        "experiment": "H0",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "target_device": target_device,
        "latency_definition": latency_definition,
        "latency_target_ms": latency_target,
        "imgsz": imgsz,
        "warmup_runs": warmup,
        "bench_runs": runs,
        "engine": str(engine),
        "int8": int8_bench,
        "fp32_reference": fp32_bench,
        "latency_pass": latency_pass,
        "memory_pass": memory_pass,
        "memory_limit_gib": memory_limit_gib,
        "gate5_pass": gate5_pass,
        "accuracy_pass": accuracy_pass,
        "overall_pass": overall_pass,
        "promoted": str(promoted) if promoted else None,
    }
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "stage2_gate5.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Stage 2 Gate 5 — Target Hardware Check",
        "",
        f"Generated: {payload['generated']}",
        "",
        f"- Target device: `{target_device}`",
        f"- Latency definition: `{latency_definition}` (batch=1, imgsz={imgsz})",
        f"- Warm-up / bench runs: {warmup} / {runs}",
        f"- Engine: `{engine.relative_to(root)}`",
        "",
        "## Results",
        "",
        "| Model | Median ms | p95 ms | Peak GPU GiB |",
        "|---|---|---|---|",
        f"| INT8 | {int8_bench['latency_ms_median']:.3f} | {int8_bench['latency_ms_p95']:.3f} | "
        f"{int8_bench['peak_gpu_gib']} |",
    ]
    if fp32_bench:
        md.append(
            f"| FP32 ref | {fp32_bench['latency_ms_median']:.3f} | {fp32_bench['latency_ms_p95']:.3f} | "
            f"{fp32_bench['peak_gpu_gib']} |"
        )
    md += [
        "",
        f"- Latency target: ≤ **{latency_target} ms** → **{'PASS' if latency_pass else 'FAIL'}**",
        f"- Memory limit: ≤ **{memory_limit_gib} GiB** → **{'PASS' if memory_pass else 'FAIL'}**",
        f"- Gate 5: **{'PASS' if gate5_pass else 'FAIL'}**",
        f"- Accuracy gate (from stage2_fp32_vs_int8): **{'PASS' if accuracy_pass else 'FAIL'}**",
        f"- Promote INT8: **{'YES → models/stage2/best_int8.engine' if promoted else 'NO'}**",
        "",
    ]
    (reports / "stage2_gate5.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
