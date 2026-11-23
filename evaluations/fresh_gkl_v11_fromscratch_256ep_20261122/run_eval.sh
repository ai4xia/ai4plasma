#!/bin/bash
# ==============================================================================
# run_eval.sh
# Scientific Evaluation for fresh_gkl_v11_fromscratch_256ep_20261122
# ==============================================================================
set -euo pipefail

PROJECT_DIR="/pscratch/sd/b/binxia/ai4plasma"
cd "${PROJECT_DIR}"

module load pytorch 2>/dev/null || true
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== START EVALUATION $(date) ==="
echo "Node: $(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU: $(nvidia-smi -L | head -n 1 || echo 'No GPU detected')"
else
    echo "GPU: None"
fi
echo "Python: $(which python3)"

python3 evaluations/fresh_gkl_v11_fromscratch_256ep_20261122/run_evaluation.py

echo "=== DONE EVALUATION $(date) ==="
