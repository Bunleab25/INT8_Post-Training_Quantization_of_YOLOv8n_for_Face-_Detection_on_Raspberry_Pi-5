#!/usr/bin/env python3
"""Build reports/stage2_summary.md from Stage 2 artefacts."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

try:
    from scripts.common import REPORTS, ROOT, fmt, load_json as load
except ImportError:
    from common import REPORTS, ROOT, fmt, load_json as load


def status(flag: bool | None, pending: str = "PENDING") -> str:
    if flag is True:
        return "PASS"
    if flag is False:
        return "FAIL"
    return pending


def main() -> int:
    config = load(ROOT / "configs" / "stage2.json")
    if not config:
        config = load(ROOT / "configs" / "stage2.template.json")
    calib = load(ROOT / "calibration" / "images.json")
    runtime = config.get("runtime") or "onnxruntime"
    export_info = load(ROOT / "exports" / "stage2" / str(runtime) / "export_info.json")
    quant_info = load(ROOT / "models" / "stage2" / "int8_info.json")
    compare = load(REPORTS / "stage2_fp32_vs_int8.json")
    gate5 = load(REPORTS / "stage2_gate5.json")
    env = load(REPORTS / "environment.json")

    gate4 = config.get("stage1_gate4_approved") is True
    preflight_ready = gate4 and bool(config.get("ood_dir")) and bool(config.get("calibration_manifest"))
    calib_ready = bool(calib.get("images"))
    export_ready = bool(export_info.get("onnx"))
    quant_ready = bool(quant_info.get("engine") or quant_info.get("quantized_model"))
    accuracy_pass = compare.get("accuracy_pass")
    gate5_pass = gate5.get("gate5_pass")
    overall = gate5.get("overall_pass")

    md = [
        "# Stage 2 Summary — INT8 Judged Against FP32",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Gate status",
        "",
        "| Gate | Evidence | Status |",
        "|---|---|---|",
        f"| Gate 4 — Stage 1 acceptance | `configs/stage2.json` → stage1_gate4_approved | "
        f"{'PASS' if gate4 else 'PENDING owner approval'} |",
        f"| Gate 4.5 — Preflight inputs | OOD + calibration + runtime targets | "
        f"{'PASS' if (preflight_ready and calib_ready) else 'PENDING'} |",
        f"| Export X0 | `exports/stage2/{runtime}/export_info.json` | "
        f"{'PASS' if export_ready else 'NOT RUN'} |",
        f"| PTQ Q0 | `models/stage2/int8_info.json` | {'PASS' if quant_ready else 'NOT RUN'} |",
        f"| Accuracy (FP32 vs INT8) | `reports/stage2_fp32_vs_int8.md` | "
        f"{status(accuracy_pass, 'NOT RUN')} |",
        f"| Gate 5 — Hardware | `reports/stage2_gate5.md` | {status(gate5_pass, 'NOT RUN')} |",
        f"| Promote best_int8 | `models/stage2/best_int8.*` | "
        f"{'PASS' if gate5.get('promoted') else status(overall, 'PENDING')} |",
        "",
        "## Locked targets",
        "",
        f"| Item | Value |",
        f"|---|---|",
        f"| FP32 reference | `models/stage1/best.pt` (E1) |",
        f"| Runtime | `{config.get('runtime')}` |",
        f"| Target device | `{config.get('target_device')}` |",
        f"| imgsz | {config.get('imgsz')} |",
        f"| Latency target | {config.get('latency_target_ms')} ms "
        f"({config.get('latency_definition')}) |",
        f"| Allowed test drop | {config.get('allowed_map50_95_drop')} mAP50-95 |",
        f"| Allowed OOD drop | {config.get('allowed_ood_map50_95_drop')} mAP50-95 |",
        f"| Calibration count | {config.get('calibration_count')} |",
        "",
        "## Calibration",
        "",
    ]
    if calib_ready:
        md.append(
            f"- Manifest images: **{calib.get('actual_count', len(calib.get('images', [])))}** "
            f"(requested {calib.get('requested_count', '—')}); test-split images: **0**."
        )
    else:
        md.append("- Calibration manifest not built yet. Run `scripts/make_calibration_manifest.py`.")

    md += [
        "",
        "## FP32 vs INT8",
        "",
        "| Set | FP32 mAP50-95 | INT8 mAP50-95 | Drop | Pass |",
        "|---|---|---|---|---|",
        f"| Test | {fmt(compare.get('fp32_test_map50_95'))} | {fmt(compare.get('int8_test_map50_95'))} | "
        f"{fmt(compare.get('test_drop'))} | {status(compare.get('test_pass'), '—')} |",
        f"| OOD | {fmt(compare.get('fp32_ood_map50_95'))} | {fmt(compare.get('int8_ood_map50_95'))} | "
        f"{fmt(compare.get('ood_drop'))} | {status(compare.get('ood_pass'), '—')} |",
        "",
    ]
    if compare.get("ood_note"):
        md.append(f"> {compare['ood_note']}")
        md.append("")

    md += [
        "## Gate 5 latency",
        "",
    ]
    int8 = gate5.get("int8") or {}
    if int8:
        md += [
            f"- INT8 median latency: **{fmt(int8.get('latency_ms_median'), 3)} ms** "
            f"(target ≤ {config.get('latency_target_ms')} ms)",
            f"- Peak GPU memory: **{int8.get('peak_gpu_gib')} GiB**",
            f"- Gate 5: **{status(gate5_pass)}**; overall promote: **{status(overall)}**",
        ]
    else:
        md.append("- Not measured yet. Run `scripts/bench_target.py` after accuracy comparison.")

    md += [
        "",
        "## Environment",
        "",
        f"- torch `{env.get('torch')}`, CUDA `{env.get('torch_cuda')}`, GPU `{env.get('gpu_name')}`",
        "",
        "## Artefacts",
        "",
        "- `configs/stage2.template.json`, `configs/stage2.json`",
        "- `calibration/images.json`, `reports/calibration_report.md`",
        f"- `exports/stage2/{runtime}/best.onnx`",
        f"- `models/stage2/int8.{'onnx' if runtime == 'onnxruntime' else 'engine'}`",
        f"- `models/stage2/best_int8.{'onnx' if runtime == 'onnxruntime' else 'engine'}`",
        "- `reports/stage2_fp32_vs_int8.md`, `reports/stage2_gate5.md`",
        "",
        "## Next actions",
        "",
    ]
    if not gate4:
        md.append("1. Supply OOD under `data/ood/`, set `stage1_gate4_approved: true`, run preflight.")
    elif not calib_ready:
        md.append("1. Run `scripts/make_calibration_manifest.py`.")
    elif not export_ready:
        md.append("1. Run `scripts/export_stage2.py`.")
    elif not quant_ready:
        md.append(f"1. Run `scripts/quantize_ptq.py`.")
    elif accuracy_pass is not True:
        md.append("1. Inspect `reports/stage2_fp32_vs_int8.md`; retry Q1 or labelled QAT0 if needed.")
    elif gate5_pass is not True:
        md.append("1. Run or re-run `scripts/bench_target.py` on the target device.")
    else:
        ext = 'onnx' if runtime == 'onnxruntime' else 'engine'
        md.append(f"1. Stage 2 complete; use `models/stage2/best_int8.{ext}`.")
    md.append("")

    out = REPORTS / "stage2_summary.md"
    REPORTS.mkdir(exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
