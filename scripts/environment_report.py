#!/usr/bin/env python3
"""Gate 0: record the training environment without changing it."""
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "environment_report.md"


def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def main():
    import cv2
    import numpy as np
    import torch
    import ultralytics

    info = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "cpu": platform.processor() or run("lscpu | grep 'Model name' | head -1"),
        "cpu_count": run("nproc"),
        "ram": run("free -h | awk '/^Mem:/{print $2}'"),
        "os": run("lsb_release -ds 2>/dev/null || cat /etc/os-release | head -1"),
        "driver": run("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        "gpu_name": run("nvidia-smi --query-gpu=name --format=csv,noheader"),
        "gpu_memory_mib": run("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"),
        "compute_cap": run("nvidia-smi --query-gpu=compute_cap --format=csv,noheader"),
        "torchvision": run(f"{sys.executable} -c 'import torchvision; print(torchvision.__version__)'"),
    }
    if torch.cuda.is_available():
        info["torch_gpu"] = torch.cuda.get_device_name(0)
        info["torch_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        info["torch_memory_gib"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Environment Report (Gate 0)",
        "",
        f"Generated: {info['generated']}",
        "",
        "| Item | Value |",
        "|---|---|",
    ]
    for key, value in info.items():
        lines.append(f"| {key} | `{value}` |")
    lines += [
        "",
        "## Verification checks",
        "",
        "```text",
        run("nvidia-smi"),
        "```",
        "",
        "## Notes",
        "",
        "- Recorded only; no packages, drivers or environments were changed.",
        "- The training environment is the existing `ai-ml` Conda environment.",
        f"- Python executable: `{sys.executable}`",
        "",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "reports" / "environment.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUT}")
    for key in ("python", "torch", "torch_cuda", "cuda_available", "ultralytics", "gpu_name", "gpu_memory_mib", "compute_cap"):
        print(f"  {key}: {info[key]}")


if __name__ == "__main__":
    main()
