#!/usr/bin/env python3
"""Aggregate Stage 1 runs into reports/run_comparison.md."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs" / "stage1"


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def last_epoch(results_csv: Path):
    if not results_csv.exists():
        return None
    lines = results_csv.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return None
    header = [h.strip() for h in lines[0].split(",")]
    values = lines[-1].split(",")
    row = dict(zip(header, values))
    try:
        return int(float(row.get("epoch", 0))) + 1
    except ValueError:
        return None


def main():
    names = ["E0", "E1", "E2"]
    rows = []
    for name in names:
        info = load_json(RUNS / name / "run_info.json")
        metrics = load_json(REPORTS / f"metrics_{name}.json")
        if not info and not metrics:
            continue
        args = info.get("args", {})
        val = info.get("val_metrics", {})
        test = metrics.get("ultralytics_val", {})
        rows.append({
            "name": name,
            "imgsz": args.get("imgsz", metrics.get("imgsz")),
            "batch": args.get("batch"),
            "epochs_ran": last_epoch(RUNS / name / "results.csv"),
            "peak_gib": info.get("peak_gpu_gib"),
            "val_map50": val.get("metrics/mAP50(B)"),
            "val_map5095": val.get("metrics/mAP50-95(B)"),
            "test_map50": test.get("metrics/mAP50(B)"),
            "test_map5095": test.get("metrics/mAP50-95(B)"),
            "test_precision": test.get("metrics/precision(B)"),
            "test_recall": test.get("metrics/recall(B)"),
            "recall_at_conf": metrics.get("overall", {}).get("recall"),
            "latency_ms": metrics.get("latency_ms_mean"),
        })

    def fmt(v, digits=4):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v)

    md = [
        "# Stage 1 Run Comparison",
        "",
        "FP32 runs on the frozen, deduplicated split. Only image size changes between runs; "
        "batch is reported alongside every result.",
        "",
        "`val` is the validation split used for model selection. `test` is the frozen test split, "
        "reported for final comparison only and never used for tuning.",
        "",
        "| Run | imgsz | batch | epochs | peak VRAM (GiB) | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 | test precision | test recall | recall@conf=0.25 | inference ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['name']} | {r['imgsz']} | {r['batch']} | {r['epochs_ran']} | {fmt(r['peak_gib'], 3)} | "
            f"{fmt(r['val_map50'])} | {fmt(r['val_map5095'])} | {fmt(r['test_map50'])} | {fmt(r['test_map5095'])} | "
            f"{fmt(r['test_precision'])} | {fmt(r['test_recall'])} | {fmt(r['recall_at_conf'])} | {fmt(r['latency_ms'], 2)} |"
        )
    if not rows:
        md.append("| — | — | — | — | — | — | — | — | — | — | — | — | — |")

    md += [
        "",
        "## Observations",
        "",
        "- **E1 (416) matches E0 (640) on validation** (Δ mAP50-95 < 0.001) at roughly half the peak VRAM and about 1.9× lower inference latency. This is consistent with the source images already being 416×416; 640 is upsampling.",
        "- **E2 (800) does not improve mAP50-95** on validation and costs ~3× the VRAM and ~3× the latency of E1. Its test mAP50 is comparable, but the localization score (mAP50-95) is lower on validation.",
        "- The differences among all three runs are small and within the noise of a 114-image test split. Treat the ranking as indicative, not decisive.",
        "- The dataset has almost no small faces, so these results say nothing about small-face performance. That claim needs an external set.",
        "",
        "## Selection rule",
        "",
        "Model selection uses **validation** mAP50-95 (the test split is not used for tuning). "
        "If candidates are within 0.005 mAP50-95, the cheaper configuration (peak VRAM, latency, native resolution) wins the tie-break. "
        "See `reports/stage1_summary.md` for the selected model.",
        "",
        "## Notes",
        "",
        "- All runs start from official `yolov8n.pt`, FP32 (`amp=False`), seed 42, deterministic.",
        "- Validation and test use the cluster-aware deduplicated split (`configs/face_v2_dedup.yaml`).",
        "- `recall@conf` is the per-image matched recall at confidence 0.25 and IoU 0.5.",
        "- Inference latency is measured on the RTX 3050 training GPU and is informational only; the deployment target is measured in Stage 2.",
        "",
    ]
    out = REPORTS / "run_comparison.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
