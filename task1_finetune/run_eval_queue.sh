#!/bin/bash
# Run Task 1 evaluations one after another on one GPU (Kaggle T1-eval), one process per condition.
# usage: bash task1_finetune/run_eval_queue.sh <gpu> <condition|diagnostics> [...]
# A failed run does not stop the queue. Writes notes/kaggle_logs/eval_queue_gpu<gpu>.done when the queue ends.
cd "$(dirname "$0")/.." || exit 1
gpu=$1
shift
mkdir -p notes/kaggle_logs
for c in "$@"; do
  echo "=== $(date -u +%FT%TZ) GPU$gpu start $c ==="
  if [ "$c" = diagnostics ]; then
    CUDA_VISIBLE_DEVICES=$gpu python -m task1_finetune.diagnostics 2>&1 | tee notes/kaggle_logs/diagnostics.log
  else
    CUDA_VISIBLE_DEVICES=$gpu python -m task1_finetune.evaluate --condition "$c" 2>&1 | tee "notes/kaggle_logs/eval_$c.log"
  fi
  echo "=== $(date -u +%FT%TZ) GPU$gpu end $c exit ${PIPESTATUS[0]} ==="
done
date -u +%FT%TZ > "notes/kaggle_logs/eval_queue_gpu$gpu.done"
echo "=== $(date -u +%FT%TZ) GPU$gpu queue finished ==="
