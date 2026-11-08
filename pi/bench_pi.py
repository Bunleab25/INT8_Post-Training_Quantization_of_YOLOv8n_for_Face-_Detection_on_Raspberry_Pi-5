#!/usr/bin/env python3
"""Raspberry Pi 5 & Edge CPU Benchmark Suite for YOLOv8 Face Detection.

Measures:
  1. Stage-by-stage latency breakdown (Preprocess / ONNX Inference / NumPy NMS)
  2. FP32 ONNX vs. INT8 ONNX model comparison (latency, FPS, file size, load time)
  3. Multi-core CPU thread scaling (1, 2, 3, 4 threads)
  4. Peak process RSS memory (MB) and Raspberry Pi SoC temperature / throttling state

Outputs:
  - reports/pi_benchmark.json
  - reports/pi_benchmark.md

Usage:
  python pi/bench_pi.py
  python pi/bench_pi.py --warmup 30 --runs 150 --threads 1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

try:
    from pi.face_detector import FaceDetector, resolve_default_model_path
except ImportError:
    from face_detector import FaceDetector, resolve_default_model_path


def parse_thread_list(raw: str) -> list[int]:
    """Parse comma-separated positive integers (e.g. '1,2,3,4') into a list of ints."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Thread list cannot be empty")
    threads: list[int] = []
    for p in parts:
        val = int(p)
        if val <= 0:
            raise ValueError(f"Thread count must be positive, got {val}")
        threads.append(val)
    return threads


def read_cpu_temp_c(
    thermal_path: Path = Path("/sys/class/thermal/thermal_zone0/temp"),
) -> float | None:
    """Read Linux/Raspberry Pi SoC thermal zone temperature in degrees Celsius."""
    try:
        if not thermal_path.is_file():
            return None
        raw = thermal_path.read_text(encoding="utf-8").strip()
        return round(float(raw) / 1000.0, 2)
    except Exception:
        return None


def read_throttled_state() -> str | None:
    """Query Raspberry Pi `vcgencmd get_throttled` status if available."""
    vcgencmd = shutil.which("vcgencmd")
    if not vcgencmd:
        return None
    try:
        res = subprocess.run(
            [vcgencmd, "get_throttled"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return None


def read_peak_rss_mb() -> float:
    """Return peak resident set size (RSS) of the current process in MiB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Linux, ru_maxrss is in kilobytes; on macOS, bytes.
    if sys.platform == "darwin":
        return round(usage / (1024.0 * 1024.0), 2)
    return round(usage / 1024.0, 2)


def get_system_info() -> dict:
    """Collect hardware and software environment metadata."""
    cpu_model = platform.processor() or platform.machine()
    device_model = None

    # Check Raspberry Pi device-tree model if present
    dt_model = Path("/proc/device-tree/model")
    if dt_model.is_file():
        try:
            device_model = dt_model.read_bytes().decode("utf-8", errors="ignore").strip("\x00").strip()
        except Exception:
            pass

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass

    return {
        "device_model": device_model or cpu_model,
        "cpu_model": cpu_model,
        "architecture": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "onnxruntime_version": ort.__version__,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }


def load_eval_frames(images_dir: Path | None, count: int = 10) -> tuple[list[np.ndarray], str]:
    """Load evaluation images from disk or synthesize deterministic 640x480 test frames."""
    frames: list[np.ndarray] = []
    if images_dir and images_dir.is_dir():
        for p in sorted(images_dir.glob("*.jpg"))[:count]:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
        if frames:
            return frames, f"{images_dir} ({len(frames)} images)"

    rng = np.random.default_rng(42)
    synthetic = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    return [synthetic], "synthetic 640x480 frame (seed=42)"


def benchmark_model(
    model_path: Path,
    frames: list[np.ndarray],
    num_threads: int,
    warmup: int,
    runs: int,
    conf: float = 0.25,
    iou: float = 0.45,
) -> dict:
    """Run warm-up and timed iterations for a single ONNX model and thread setting."""
    model_size_mb = round(model_path.stat().st_size / (1024.0 * 1024.0), 2)
    temp_start = read_cpu_temp_c()

    t_load0 = time.perf_counter()
    detector = FaceDetector(
        model_path=str(model_path),
        conf_thresh=conf,
        iou_thresh=iou,
        num_threads=num_threads,
    )
    load_time_ms = round((time.perf_counter() - t_load0) * 1000.0, 2)

    # Warm-up pass
    for i in range(warmup):
        frame = frames[i % len(frames)]
        detector.detect_timed(frame)

    pre_samples: list[float] = []
    inf_samples: list[float] = []
    post_samples: list[float] = []
    tot_samples: list[float] = []
    det_counts: list[int] = []

    for i in range(runs):
        frame = frames[i % len(frames)]
        dets, timings = detector.detect_timed(frame)
        pre_samples.append(timings["preprocess_ms"])
        inf_samples.append(timings["inference_ms"])
        post_samples.append(timings["postprocess_ms"])
        tot_samples.append(timings["total_ms"])
        det_counts.append(len(dets))

    temp_end = read_cpu_temp_c()
    tot_median = float(np.median(tot_samples))
    tot_mean = float(np.mean(tot_samples))

    return {
        "model_path": str(model_path),
        "model_name": model_path.name,
        "model_size_mb": model_size_mb,
        "threads": num_threads,
        "warmup": warmup,
        "runs": runs,
        "load_time_ms": load_time_ms,
        "preprocess_ms_median": round(float(np.median(pre_samples)), 3),
        "inference_ms_median": round(float(np.median(inf_samples)), 3),
        "postprocess_ms_median": round(float(np.median(post_samples)), 3),
        "total_ms_median": round(tot_median, 3),
        "total_ms_mean": round(tot_mean, 3),
        "total_ms_p95": round(float(np.percentile(tot_samples, 95)), 3),
        "total_ms_p99": round(float(np.percentile(tot_samples, 99)), 3),
        "total_ms_std": round(float(np.std(tot_samples)), 3),
        "fps_median": round(1000.0 / tot_median, 2) if tot_median > 0 else 0.0,
        "fps_mean": round(1000.0 / tot_mean, 2) if tot_mean > 0 else 0.0,
        "avg_detections": round(float(np.mean(det_counts)), 2),
        "temp_start_c": temp_start,
        "temp_end_c": temp_end,
        "peak_rss_mb": read_peak_rss_mb(),
    }


def format_markdown_report(payload: dict) -> str:
    """Format the benchmark payload into a Markdown report with comparison tables."""
    sys_info = payload["system"]
    lines = [
        "# Edge Hardware Benchmark Report (ONNX Runtime CPU)",
        "",
        f"Generated: `{payload['generated']}`",
        "",
        "## 1. System & Runtime Environment",
        "",
        "| Property | Value |",
        "|---|---|",
        f"| **Device / CPU** | `{sys_info['device_model']}` |",
        f"| **Architecture** | `{sys_info['architecture']}` |",
        f"| **OS** | `{sys_info['os']}` |",
        f"| **Python** | `{sys_info['python_version']}` |",
        f"| **ONNX Runtime** | `{sys_info['onnxruntime_version']}` |",
        f"| **OpenCV / NumPy** | `{sys_info['opencv_version']}` / `{sys_info['numpy_version']}` |",
        f"| **Frame Source** | `{payload['frame_source']}` |",
        f"| **Warm-up / Measured Runs** | `{payload['warmup']}` / `{payload['runs']}` |",
        f"| **Peak Process RSS Memory** | `{payload['peak_rss_mb']} MB` |",
        f"| **Throttled Status** | `{payload['throttled_state'] or 'N/A (non-RPi or vcgencmd unavailable)'}` |",
        "",
    ]

    # Table 1: FP32 vs INT8 Comparison (if FP32 was evaluated)
    fp32 = payload.get("fp32_baseline")
    int8_primary = payload.get("int8_primary")
    if int8_primary:
        lines.extend([
            "## 2. Model Comparison: FP32 ONNX vs. INT8 ONNX",
            "",
            "| Metric | FP32 ONNX (`best.onnx`) | INT8 ONNX (`best_int8.onnx`) | Speedup / Reduction |",
            "|---|---|---|---|",
        ])
        if fp32:
            size_ratio = round(fp32["model_size_mb"] / int8_primary["model_size_mb"], 2) if int8_primary["model_size_mb"] else 0
            speedup = round(fp32["total_ms_median"] / int8_primary["total_ms_median"], 2) if int8_primary["total_ms_median"] else 0
            lines.extend([
                f"| **Model Size (MB)** | {fp32['model_size_mb']} MB | **{int8_primary['model_size_mb']} MB** | **{size_ratio}× smaller** |",
                f"| **Cold Load Time (ms)** | {fp32['load_time_ms']} ms | **{int8_primary['load_time_ms']} ms** | — |",
                f"| **Preprocess Median (ms)** | {fp32['preprocess_ms_median']:.2f} ms | **{int8_primary['preprocess_ms_median']:.2f} ms** | — |",
                f"| **Inference Median (ms)** | {fp32['inference_ms_median']:.2f} ms | **{int8_primary['inference_ms_median']:.2f} ms** | **{speedup}× faster** |",
                f"| **Postprocess / NMS (ms)** | {fp32['postprocess_ms_median']:.2f} ms | **{int8_primary['postprocess_ms_median']:.2f} ms** | — |",
                f"| **End-to-End Median (ms)** | {fp32['total_ms_median']:.2f} ms | **{int8_primary['total_ms_median']:.2f} ms** | **{speedup}× faster** |",
                f"| **End-to-End p95 (ms)** | {fp32['total_ms_p95']:.2f} ms | **{int8_primary['total_ms_p95']:.2f} ms** | — |",
                f"| **Throughput (Median FPS)** | {fp32['fps_median']:.1f} FPS | **{int8_primary['fps_median']:.1f} FPS** | **+{int8_primary['fps_median'] - fp32['fps_median']:.1f} FPS** |",
            ])
        else:
            lines.extend([
                f"| **Model Size (MB)** | *Not present* | **{int8_primary['model_size_mb']} MB** | — |",
                f"| **Cold Load Time (ms)** | *Not present* | **{int8_primary['load_time_ms']} ms** | — |",
                f"| **Preprocess Median (ms)** | *Not present* | **{int8_primary['preprocess_ms_median']:.2f} ms** | — |",
                f"| **Inference Median (ms)** | *Not present* | **{int8_primary['inference_ms_median']:.2f} ms** | — |",
                f"| **Postprocess / NMS (ms)** | *Not present* | **{int8_primary['postprocess_ms_median']:.2f} ms** | — |",
                f"| **End-to-End Median (ms)** | *Not present* | **{int8_primary['total_ms_median']:.2f} ms** | — |",
                f"| **End-to-End p95 (ms)** | *Not present* | **{int8_primary['total_ms_p95']:.2f} ms** | — |",
                f"| **Throughput (Median FPS)** | *Not present* | **{int8_primary['fps_median']:.1f} FPS** | — |",
            ])
        lines.append("")

        # Stage breakdown percentages
        tot = int8_primary["total_ms_median"] or 1.0
        pre_pct = (int8_primary["preprocess_ms_median"] / tot) * 100.0
        inf_pct = (int8_primary["inference_ms_median"] / tot) * 100.0
        post_pct = (int8_primary["postprocess_ms_median"] / tot) * 100.0
        lines.extend([
            "## 3. INT8 Stage-by-Stage Pipeline Breakdown",
            "",
            "| Pipeline Stage | Median Latency (ms) | Share of Total Time (%) |",
            "|---|---|---|",
            f"| **1. Preprocess** (`letterbox` + RGB/CHW norm) | {int8_primary['preprocess_ms_median']:.3f} ms | {pre_pct:.1f}% |",
            f"| **2. ONNX Runtime Inference** (`session.run`) | {int8_primary['inference_ms_median']:.3f} ms | {inf_pct:.1f}% |",
            f"| **3. Postprocess** (Box decode + NumPy NMS) | {int8_primary['postprocess_ms_median']:.3f} ms | {post_pct:.1f}% |",
            f"| **Total End-to-End** | **{int8_primary['total_ms_median']:.3f} ms** | **100.0%** |",
            "",
        ])

    # Table 3: Thread Scaling
    sweeps = payload.get("int8_thread_sweep", [])
    if sweeps:
        base_ms = sweeps[0]["total_ms_median"] or 1.0
        lines.extend([
            "## 4. Multi-Core CPU Thread Scaling (`best_int8.onnx`)",
            "",
            "| CPU Threads | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Total Median (ms) | Total p95 (ms) | Throughput (FPS) | Scaling vs 1T | Temp (°C) |",
            "|---|---|---|---|---|---|---|---|---|",
        ])
        for row in sweeps:
            scaling = base_ms / row["total_ms_median"] if row["total_ms_median"] > 0 else 1.0
            temp_str = f"{row['temp_end_c']}°C" if row["temp_end_c"] is not None else "N/A"
            lines.append(
                f"| **{row['threads']}** | {row['preprocess_ms_median']:.2f} | "
                f"{row['inference_ms_median']:.2f} | {row['postprocess_ms_median']:.2f} | "
                f"**{row['total_ms_median']:.2f}** | {row['total_ms_p95']:.2f} | "
                f"**{row['fps_median']:.1f}** | {scaling:.2f}× | {temp_str} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raspberry Pi 5 & Edge CPU Benchmark Suite for YOLOv8 Face Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    default_int8 = resolve_default_model_path()
    default_fp32 = str(repo_root / "exports" / "stage2" / "onnxruntime" / "best.onnx")
    default_images = str(repo_root / "data" / "ood" / "images")

    parser.add_argument(
        "--int8-model",
        type=str,
        default=default_int8,
        help="Path to INT8 quantized ONNX model",
    )
    parser.add_argument(
        "--fp32-model",
        type=str,
        default=default_fp32,
        help="Path to FP32 baseline ONNX model (optional; skipped if not found)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=default_images,
        help="Directory of sample .jpg frames to benchmark on (falls back to synthetic 640x480)",
    )
    parser.add_argument(
        "--threads",
        type=str,
        default="1,2,3,4",
        help="Comma-separated CPU thread counts to sweep for INT8 model",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Number of warm-up iterations before timing",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=150,
        help="Number of timed iterations per configuration",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(repo_root / "reports" / "pi_benchmark.json"),
        help="Output path for JSON benchmark metrics",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default=str(repo_root / "reports" / "pi_benchmark.md"),
        help="Output path for Markdown benchmark report",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    int8_path = Path(args.int8_model)
    if not int8_path.is_file():
        print(f"[ERROR] INT8 model not found: {int8_path}", file=sys.stderr)
        return 1

    try:
        thread_list = parse_thread_list(args.threads)
    except ValueError as exc:
        print(f"[ERROR] Invalid --threads argument: {exc}", file=sys.stderr)
        return 2

    sys_info = get_system_info()
    frames, frame_source = load_eval_frames(Path(args.images_dir) if args.images_dir else None)

    print("============================================================")
    print(" YOLOv8 Edge CPU Benchmark Suite (ONNX Runtime)")
    print("============================================================")
    print(f"Device / CPU : {sys_info['device_model']} ({sys_info['architecture']})")
    print(f"Frame Source : {frame_source}")
    print(f"Warmup / Runs: {args.warmup} / {args.runs}")
    print(f"Thread Sweep : {thread_list}")

    int8_sweep: list[dict] = []
    for t in thread_list:
        print(f"[BENCH] Running INT8 ({int8_path.name}) with threads={t}...")
        res = benchmark_model(
            model_path=int8_path,
            frames=frames,
            num_threads=t,
            warmup=args.warmup,
            runs=args.runs,
            conf=args.conf,
            iou=args.iou,
        )
        int8_sweep.append(res)
        print(
            f"        -> Median: {res['total_ms_median']:.2f} ms "
            f"(Inf: {res['inference_ms_median']:.2f} ms, NMS: {res['postprocess_ms_median']:.2f} ms) "
            f"| {res['fps_median']:.1f} FPS"
        )

    # Primary INT8 configuration (prefer 4 threads if in sweep, else last thread setting)
    int8_primary = next((r for r in int8_sweep if r["threads"] == 4), int8_sweep[-1])
    primary_threads = int8_primary["threads"]

    fp32_res = None
    fp32_path = Path(args.fp32_model) if args.fp32_model else None
    if fp32_path and fp32_path.is_file():
        print(f"[BENCH] Running FP32 baseline ({fp32_path.name}) with threads={primary_threads}...")
        fp32_res = benchmark_model(
            model_path=fp32_path,
            frames=frames,
            num_threads=primary_threads,
            warmup=args.warmup,
            runs=args.runs,
            conf=args.conf,
            iou=args.iou,
        )
        print(
            f"        -> Median: {fp32_res['total_ms_median']:.2f} ms "
            f"(Inf: {fp32_res['inference_ms_median']:.2f} ms) | {fp32_res['fps_median']:.1f} FPS"
        )
    else:
        print(f"[INFO] FP32 ONNX model not found at {fp32_path}; skipping FP32 baseline comparison.")

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "system": sys_info,
        "frame_source": frame_source,
        "warmup": args.warmup,
        "runs": args.runs,
        "throttled_state": read_throttled_state(),
        "peak_rss_mb": read_peak_rss_mb(),
        "int8_primary": int8_primary,
        "fp32_baseline": fp32_res,
        "int8_thread_sweep": int8_sweep,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(format_markdown_report(payload), encoding="utf-8")

    print("============================================================")
    print(f"[DONE] Saved JSON metrics : {out_json}")
    print(f"[DONE] Saved MD report    : {out_md}")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
