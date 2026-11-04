#!/bin/bash
# Task 2 main run on one GPU (Kaggle cell 4): training, then the final evaluation of the Stage 3 endpoint.
# usage: bash task2_vlm/run_main.sh   (env FEATURES, RECOVERY; GPU chosen by CUDA_VISIBLE_DEVICES)
# Writes notes/kaggle_logs/task2_main.done with both exit codes when it ends.
cd "$(dirname "$0")/.." || exit 1
FEATURES=${FEATURES:-/kaggle/working/features}
RECOVERY=${RECOVERY:-/kaggle/tmp/task2_recovery}
mkdir -p notes/kaggle_logs
echo "=== $(date -u +%FT%TZ) train start (GPU ${CUDA_VISIBLE_DEVICES:-all}) ==="
python -m task2_vlm.train --features_dir "$FEATURES" --recovery_dir "$RECOVERY"
rc=$?
echo "=== $(date -u +%FT%TZ) train end exit $rc ==="
rc2=skipped
if [ "$rc" -eq 0 ]; then
  echo "=== $(date -u +%FT%TZ) final evaluation start ==="
  python -m task2_vlm.evaluate --features_dir "$FEATURES"
  rc2=$?
  echo "=== $(date -u +%FT%TZ) final evaluation end exit $rc2 ==="
fi
echo "train=$rc eval=$rc2 $(date -u +%FT%TZ)" > notes/kaggle_logs/task2_main.done
