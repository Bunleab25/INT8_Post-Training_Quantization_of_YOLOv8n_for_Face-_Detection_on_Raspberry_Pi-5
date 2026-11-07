#!/usr/bin/env bash
# Stage 1 experiment queue: E0 -> E1 -> E2, one variable (image size) each.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/bunleab/miniconda3/envs/ai-ml/bin/python
mkdir -p runs/stage1/logs

run() {
  local name="$1" imgsz="$2" batch="$3"
  echo "[$(date -Is)] START $name imgsz=$imgsz batch=$batch"
  "$PY" scripts/train_stage1.py --name "$name" --imgsz "$imgsz" --batch "$batch" --epochs 100 \
    > "runs/stage1/logs/${name}.log" 2>&1
  local rc=$?
  echo "[$(date -Is)] END $name rc=$rc"
  echo "$rc" > "runs/stage1/logs/${name}.rc"
  return $rc
}

# E0: 640, batch 4, FP32
run E0 640 4 || true

# E1: 416 (native), batch 4, FP32
run E1 416 4 || true

# E2: 800, batch 4; fall back to batch 2 if the batch-4 run fails.
if ! run E2 800 4; then
  echo "[$(date -Is)] E2 batch 4 failed; retrying batch 2 (memory-forced)"
  rm -rf runs/stage1/E2
  run E2 800 2 || true
fi

echo "[$(date -Is)] stage1 experiment queue finished" | tee runs/stage1/logs/queue.done
