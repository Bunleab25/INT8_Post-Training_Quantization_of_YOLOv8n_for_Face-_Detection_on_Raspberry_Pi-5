#!/usr/bin/env bash
# Evaluate all Stage 1 candidates on the frozen test split, then compare.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/bunleab/miniconda3/envs/ai-ml/bin/python

for name in E0 E1 E2; do
  info="runs/stage1/${name}/run_info.json"
  [ -f "$info" ] || { echo "skip $name (no run_info.json)"; continue; }
  imgsz=$("$PY" -c "import json;print(json.load(open('$info'))['args']['imgsz'])")
  weights="models/stage1/${name}_best.pt"
  [ -f "$weights" ] || weights="runs/stage1/${name}/weights/best.pt"
  echo "=== evaluate $name (imgsz=$imgsz) ==="
  "$PY" scripts/evaluate.py --weights "$weights" --name "$name" --imgsz "$imgsz" \
    > "runs/stage1/logs/${name}_eval.log" 2>&1
done

"$PY" scripts/compare_runs.py
"$PY" scripts/stage1_summary.py
