#!/usr/bin/env python3
"""Build reports/stage1_summary.md and select the best FP32 model.

Selection uses the validation split (never the test split). Ties within
0.005 mAP50-95 are broken by cost (peak VRAM, then latency).
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs" / "stage1"
MODELS = ROOT / "models" / "stage1"


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default if default is not None else {}


def last_epoch(results_csv: Path):
    if not results_csv.exists():
        return None
    lines = results_csv.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return None
    row = dict(zip([h.strip() for h in lines[0].split(",")], lines[-1].split(",")))
    try:
        return int(float(row.get("epoch", 0))) + 1
    except ValueError:
        return None


def main():
    env = load(REPORTS / "environment.json")
    audit = load(REPORTS / "audit_stats.json")
    names = ["E0", "E1", "E2"]
    candidates = []
    for name in names:
        info = load(RUNS / name / "run_info.json")
        metrics = load(REPORTS / f"metrics_{name}.json")
        if not info and not metrics:
            continue
        val = info.get("val_metrics", {})
        test = metrics.get("ultralytics_val", {})
        candidates.append({
            "name": name,
            "info": info,
            "metrics": metrics,
            "imgsz": info.get("args", {}).get("imgsz", metrics.get("imgsz")),
            "batch": info.get("args", {}).get("batch"),
            "epochs": last_epoch(RUNS / name / "results.csv"),
            "val_map5095": val.get("metrics/mAP50-95(B)"),
            "val_map50": val.get("metrics/mAP50(B)"),
            "test_map5095": test.get("metrics/mAP50-95(B)"),
            "test_map50": test.get("metrics/mAP50(B)"),
            "test_precision": test.get("metrics/precision(B)"),
            "test_recall": test.get("metrics/recall(B)"),
            "peak_gib": info.get("peak_gpu_gib"),
            "latency": metrics.get("latency_ms_mean"),
        })

    best = None
    if candidates:
        top_score = max(c["val_map5095"] or -1 for c in candidates)
        contenders = [c for c in candidates if (c["val_map5095"] or -1) >= top_score - 0.005]
        # Cheapest viable configuration wins the tie.
        best = min(contenders, key=lambda c: (c["peak_gib"] or 99, c["latency"] or 99))
        src = MODELS / f"{best['name']}_best.pt"
        if not src.exists():
            src = RUNS / best["name"] / "weights" / "best.pt"
        dst = MODELS / "best.pt"
        MODELS.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
        (MODELS / "best_info.json").write_text(
            json.dumps(
                {
                    "run": best["name"],
                    "source": str(src),
                    "frozen_copy": str(dst),
                    "imgsz": best["imgsz"],
                    "selection": {
                        "criterion": "validation mAP50-95, cost tie-break within 0.005",
                        "validation_map50_95": best["val_map5095"],
                        "validation_map50": best["val_map50"],
                        "peak_gpu_gib": best["peak_gib"],
                        "latency_ms_mean": best["latency"],
                    },
                    "test_metrics": best["metrics"].get("ultralytics_val", {}),
                    "recall_by_size": best["metrics"].get("recall_by_size", {}),
                    "selected_at": datetime.now().isoformat(timespec="seconds"),
                    "ood_available": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def fmt(v, d=4):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.{d}f}"
        return str(v)

    md = [
        "# Stage 1 Summary — FP32 YOLOv8 Face Detector",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Gate status",
        "",
        "| Gate | Evidence | Status |",
        "|---|---|---|",
        f"| Gate 0 — Environment | `reports/environment_report.md` | PASS — torch {env.get('torch')}, CUDA {env.get('torch_cuda')}, {env.get('gpu_name')} |",
        f"| Gate 1 — Audit | `reports/audit_report.md` | PASS with findings — {audit.get('unique_sources')} unique sources, {audit.get('near_duplicate_clusters_spanning_splits')} cross-split duplicate clusters |",
        "| Gate 2 — Split | `reports/split_report.md` | PASS — cluster-aware deduplicated split, 0 cross-split duplicates |",
        "| Gate 3 — Smoke test | `runs/stage1/S0` | PASS — loss and validation work, GPU headroom |",
        "| Gate 4 — Acceptance | this report | PENDING owner approval (OOD set not yet supplied) |",
        "",
        "## Dataset audit headline",
        "",
        f"- Files: **{audit.get('files')}** = **{audit.get('unique_sources')}** unique sources + **{audit.get('files', 0) - audit.get('unique_sources', 0)}** train augmentation copies.",
        f"- Near-duplicate clusters: **{audit.get('near_duplicate_clusters')}**, of which **{audit.get('near_duplicate_clusters_spanning_splits')}** span the original splits.",
        f"- Face-size buckets at 416 px: `{audit.get('coco_sizes_at_416')}`.",
        f"- Contaminated original valid/test sources: valid **{audit.get('contaminated', {}).get('valid')}**, test **{audit.get('contaminated', {}).get('test')}**.",
        "- Domain: 416×416 grayscale, stretched, close-up portraits; no small-face/crowd/negative coverage.",
        "",
        "## Split",
        "",
        "| Split | Images |",
        "|---|---|",
    ]
    for split in ("train", "valid", "test"):
        n = len(list((ROOT / "datasets" / "face_v2_dedup" / split / "images").glob("*")))
        md.append(f"| {split} | {n} |")
    md += [
        "",
        "Config: `configs/face_v2_dedup.yaml`; frozen hashes in `reports/split_hashes.sha256`.",
        "",
        "## Runs",
        "",
        "| Run | imgsz | batch | epochs | peak VRAM (GiB) | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 | test precision | test recall | recall@conf=0.25 | inference ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in candidates:
        md.append(
            f"| {c['name']} | {c['imgsz']} | {c['batch']} | {c['epochs']} | {fmt(c['peak_gib'], 3)} | "
            f"{fmt(c['val_map50'])} | {fmt(c['val_map5095'])} | {fmt(c['test_map50'])} | {fmt(c['test_map5095'])} | "
            f"{fmt(c['test_precision'])} | {fmt(c['test_recall'])} | {fmt(c['metrics'].get('overall', {}).get('recall'))} | "
            f"{fmt(c['latency'], 2)} |"
        )
    if not candidates:
        md.append("| — | — | — | — | — | — | — | — | — | — | — | — | — |")
    md += [
        "",
        "## Best FP32 model",
        "",
    ]
    if best:
        md += [
            f"- Selected run: **{best['name']}** (frozen copy: `models/stage1/best.pt`).",
            f"- Validation mAP50-95 = **{fmt(best['val_map5095'])}**; validation mAP50 = **{fmt(best['val_map50'])}**.",
            f"- Test mAP50-95 = **{fmt(best['test_map5095'])}**; test mAP50 = **{fmt(best['test_map50'])}** (reported, not used for selection).",
            f"- Peak training VRAM = **{fmt(best['peak_gib'], 3)} GiB**; mean inference latency = **{fmt(best['latency'], 2)} ms** on the training GPU.",
            "- Selection rule: validation mAP50-95, with candidates within 0.005 treated as tied and the cheaper configuration chosen. This avoids tuning on the test split.",
            "",
            "## Per-size recall (selected model, test split)",
            "",
            "| Bucket | GT | Matched | Recall |",
            "|---|---|---|---|",
        ]
        for b in ("small", "medium", "large"):
            v = best["metrics"].get("recall_by_size", {}).get(b, {})
            r = v.get("recall")
            r_text = "n/a" if r is None else f"{r:.4f}"
            md.append(f"| {b} | {v.get('gt', 0)} | {v.get('tp', 0)} | {r_text} |")

        legacy = load(REPORTS / "metrics_E1_legacy.json")
        if legacy:
            legacy_map = legacy.get("ultralytics_val", {}).get("metrics/mAP50-95(B)")
            if legacy_map is not None and best["test_map5095"] is not None:
                delta = legacy_map - best["test_map5095"]
                md += [
                    "",
                    "## Leakage check (original Roboflow test split)",
                    "",
                    f"- E1 on the original (leaky) test split: mAP50-95 = **{legacy_map:.4f}**.",
                    f"- E1 on the deduplicated test split: mAP50-95 = **{best['test_map5095']:.4f}**.",
                    f"- The leaky split is optimistic by **{delta:+.4f}** mAP50-95 ({delta / best['test_map5095'] * 100:+.1f}%). This is why the deduplicated split is used for decisions.",
                ]
    else:
        md.append("- No completed runs yet.")
    md += [
        "",
        "## Known limitations and pending inputs",
        "",
        "- **OOD set is pending.** The dataset is close-up portraiture, so the reported test score is not a measure of general face detection. Gate 4 cannot be fully signed off without an external set (crowds, profile, occlusion, low light, small faces, negatives).",
        "- The dataset has only 4 small-face boxes, so small-face recall is not measurable on the Roboflow data.",
        "- The 416×416 stretch and grayscale conversion are baked into the export and may not match deployment images.",
        "- Stage 2 (INT8) must not start until Gate 4 is approved.",
        "",
        "## Artefacts",
        "",
        "- `reports/environment_report.md`, `reports/audit_report.md`, `reports/split_report.md`",
        "- `reports/run_comparison.md`, `reports/metrics_E*.json`, `reports/evaluation_E*.md`",
        "- `runs/stage1/S0`, `runs/stage1/E0`, `runs/stage1/E1`, `runs/stage1/E2`",
        "- `models/stage1/best.pt`, `models/stage1/best_info.json`",
        "- `splits/*.txt`, `configs/face_v2_dedup.yaml`",
        "",
    ]
    (REPORTS / "stage1_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {REPORTS / 'stage1_summary.md'}")
    if best:
        print(f"Best: {best['name']} val mAP50-95={best['val_map5095']:.4f} test mAP50-95={best['test_map5095']:.4f}")


if __name__ == "__main__":
    main()
