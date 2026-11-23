#!/bin/bash
set -e
cd /pscratch/sd/b/binxia/ai4plasma
module load pytorch 2>/dev/null || true
export PYTHONPATH="/pscratch/sd/b/binxia/ai4plasma:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "=== START EVALUATION $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L | head -n 1)"
echo "Python: $(which python)"

python evaluations/matched_gkl_masking_continuation_20ep_20261121/run_evaluation.py

echo "=== DONE EVALUATION $(date) ==="
