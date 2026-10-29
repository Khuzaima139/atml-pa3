#!/bin/bash
# Run Task 1 trainings one after another on one GPU (Kaggle T1-train-all).
# usage: bash task1_finetune/run_queue.sh <gpu> <method> [<method> ...]
# A failed run does not stop the queue. After each run: verify sha256 of its kept files, then delete its
# recovery state (kept if the run or the verification failed). Writes notes/kaggle_logs/queue_gpu<gpu>.done
# when the queue ends.
cd "$(dirname "$0")/.." || exit 1
RECOVERY_ROOT=${RECOVERY_ROOT:-/kaggle/tmp/task1_recovery}
gpu=$1
shift
mkdir -p notes/kaggle_logs
for m in "$@"; do
  echo "=== $(date -u +%FT%TZ) GPU$gpu start $m ==="
  CUDA_VISIBLE_DEVICES=$gpu python -m task1_finetune.train --config "configs/task1/$m.yaml" \
    --recovery_dir "$RECOVERY_ROOT/$m" 2>&1 | tee "notes/kaggle_logs/train_$m.log"
  rc=${PIPESTATUS[0]}
  echo "=== $(date -u +%FT%TZ) GPU$gpu end $m exit $rc ==="
  if [ "$rc" -eq 0 ] && python -m task1_finetune.status --verify "results/task1/train_$m.json"; then
    rm -rf "${RECOVERY_ROOT:?}/$m"
    echo "=== GPU$gpu $m verified; recovery state deleted ==="
  else
    echo "=== GPU$gpu $m FAILED (exit $rc or verification); recovery state kept in $RECOVERY_ROOT/$m ==="
  fi
done
date -u +%FT%TZ > "notes/kaggle_logs/queue_gpu$gpu.done"
echo "=== $(date -u +%FT%TZ) GPU$gpu queue finished ==="
