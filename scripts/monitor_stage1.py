#!/usr/bin/env python3
"""Live monitor for Stage 1 training runs (stdlib only).

Usage:
  python scripts/monitor_stage1.py            # one snapshot
  python scripts/monitor_stage1.py --watch    # refresh every 10 s
  python scripts/monitor_stage1.py -w -i 3    # refresh every 3 s
  python scripts/monitor_stage1.py --run E1   # pin a specific run
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "stage1"
LOGS = RUNS / "logs"
QUEUE = ["E0", "E1", "E2"]
BAR = "#"


# ---------------------------------------------------------------- discovery
def find_run(pinned: str | None) -> str | None:
    """Prefer an alive run; else the most recently updated results.csv."""
    alive = [name for name in QUEUE if pid_for(name)]
    if pinned:
        return pinned
    if alive:
        return alive[0]
    candidates = [p for p in RUNS.glob("*/results.csv") if p.stat().st_size > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime).parent.name


def pid_for(name: str) -> int | None:
    """PID of `train_stage1.py ... --name <name>`, if running."""
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            parts = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if not any("train_stage1.py" in p for p in parts):
            continue
        for i, part in enumerate(parts):
            if part == "--name" and i + 1 < len(parts) and parts[i + 1] == name:
                return int(proc.name)
    return None


# ---------------------------------------------------------------- parsing
def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def total_epochs(name: str) -> int:
    args = RUNS / name / "args.yaml"
    if args.exists():
        for line in args.read_text(encoding="utf-8").splitlines():
            if line.startswith("epochs:"):
                try:
                    return int(float(line.split(":", 1)[1].strip()))
                except ValueError:
                    break
    return 100


def num(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    return str(timedelta(seconds=int(round(seconds))))


def bar(fraction: float, width: int = 28) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "[" + BAR * filled + "-" * (width - filled) + "]"


def gpu_line() -> str:
    if not shutil.which("nvidia-smi"):
        return "GPU        : n/a"
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()
    except (subprocess.SubprocessError, OSError):
        return "GPU        : unavailable"
    return "GPU        : " + " | ".join(out) + "  (MiB, %, C, W)"


def gpu_watts() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,power.limit", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return out
    except (subprocess.SubprocessError, OSError):
        return None


# ---------------------------------------------------------------- rendering
def snapshot(run: str | None) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"Stage 1 training monitor  {now}", "=" * 74]

    if run is None:
        lines.append("No Stage 1 runs found under runs/stage1/")
        return "\n".join(lines)

    rows = read_csv(RUNS / run / "results.csv")
    total = total_epochs(run)
    pid = pid_for(run)

    status = f"RUNNING (pid {pid})" if pid else ("DONE" if (RUNS / run / "run_info.json").exists() else "STOPPED")
    lines += [f"Run        : {run}", f"Status     : {status}"]

    if not rows:
        lines.append("Waiting for the first epoch to finish ...")
        lines.append("")
        lines.append(gpu_line())
        return "\n".join(lines)

    last = rows[-1]
    epoch = int(float(last["epoch"])) + 1
    elapsed = num(last, "time") or 0.0
    per_epoch = elapsed / epoch if epoch else 0.0
    remaining = max(0, total - epoch) * per_epoch

    best = max(rows, key=lambda r: num(r, "metrics/mAP50-95(B)") or -1.0)
    best_epoch = int(float(best["epoch"])) + 1

    lines += [
        f"Progress   : {bar(epoch / total)} {epoch}/{total} epochs "
        f"({epoch / total * 100:5.1f}%)  {per_epoch:.1f}s/epoch",
        f"Elapsed    : {fmt_duration(elapsed):>9}   ETA: {fmt_duration(remaining) if pid else '—'}",
        "",
        "Current epoch",
        f"  train  box {num(last, 'train/box_loss') or 0:.4f}   "
        f"cls {num(last, 'train/cls_loss') or 0:.4f}   dfl {num(last, 'train/dfl_loss') or 0:.4f}",
        f"  val    box {num(last, 'val/box_loss') or 0:.4f}   "
        f"cls {num(last, 'val/cls_loss') or 0:.4f}   dfl {num(last, 'val/dfl_loss') or 0:.4f}",
        "",
        "Validation (last epoch)",
        f"  precision {num(last, 'metrics/precision(B)') or 0:.4f}   "
        f"recall {num(last, 'metrics/recall(B)') or 0:.4f}   "
        f"mAP50 {num(last, 'metrics/mAP50(B)') or 0:.4f}   "
        f"mAP50-95 {num(last, 'metrics/mAP50-95(B)') or 0:.4f}",
        "",
        f"Best so far: epoch {best_epoch}  mAP50 {num(best, 'metrics/mAP50(B)') or 0:.4f}  "
        f"mAP50-95 {num(best, 'metrics/mAP50-95(B)') or 0:.4f}  "
        f"recall {num(best, 'metrics/recall(B)') or 0:.4f}",
        f"LR (pg0)   : {num(last, 'lr/pg0') or 0:.6g}",
        "",
        gpu_line(),
    ]

    # Recent epochs table
    lines += ["", "Last epochs", "  ep   box_loss  cls_loss  dfl_loss    mAP50  mAP50-95   recall"]
    for row in rows[-8:]:
        ep = int(float(row["epoch"])) + 1
        lines.append(
            f"  {ep:>3}  {num(row, 'train/box_loss') or 0:8.4f}  {num(row, 'train/cls_loss') or 0:8.4f}  "
            f"{num(row, 'train/dfl_loss') or 0:8.4f}  {num(row, 'metrics/mAP50(B)') or 0:6.3f}  "
            f"{num(row, 'metrics/mAP50-95(B)') or 0:8.3f}  {num(row, 'metrics/recall(B)') or 0:7.3f}"
        )

    # Queue status
    lines += ["", "Experiment queue", "  run   imgsz  status"]
    for name in QUEUE:
        cfg = RUNS / name / "args.yaml"
        imgsz = "—"
        if cfg.exists():
            for line in cfg.read_text(encoding="utf-8").splitlines():
                if line.startswith("imgsz:"):
                    imgsz = line.split(":", 1)[1].strip()
                    break
        if (RUNS / name / "run_info.json").exists():
            state = "done"
        elif pid_for(name):
            state = f"training (pid {pid_for(name)})"
        elif (RUNS / name / "results.csv").exists():
            state = "stopped / incomplete"
        else:
            state = "queued"
        lines.append(f"  {name:<5} {imgsz:>5}  {state}")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Live monitor for Stage 1 YOLOv8 training.")
    ap.add_argument("--watch", "-w", action="store_true", help="refresh continuously")
    ap.add_argument("--interval", "-i", type=float, default=10.0, help="seconds between refreshes")
    ap.add_argument("--run", help="pin a run name (E0/E1/E2)")
    ap.add_argument("--once", action="store_true", help="print a single snapshot (default)")
    args = ap.parse_args()

    try:
        while True:
            text = snapshot(find_run(args.run))
            if args.watch:
                # Clear screen and repaint.
                sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
            if not args.watch:
                return 0
            time.sleep(max(0.5, args.interval))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
