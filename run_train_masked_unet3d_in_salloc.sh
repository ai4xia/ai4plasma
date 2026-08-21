#!/bin/bash

set -euo pipefail

cd /pscratch/sd/b/binxia/ai4plasma

mkdir -p logs

# ------------------------------------------------------------
# Load NERSC PyTorch environment
# ------------------------------------------------------------
ml load pytorch

PYTHON=$(command -v python)

echo "Loaded PyTorch environment:"
echo "  Python: ${PYTHON}"

# Fail immediately if CUDA is unavailable.
"${PYTHON}" - <<'PY'
import torch

print(f"  PyTorch: {torch.__version__}")
print(f"  PyTorch CUDA: {torch.version.cuda}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  GPU count: {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Refusing to start CPU training."
    )

for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
PY

# ------------------------------------------------------------
# CPU threading
# ------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ------------------------------------------------------------
# Data / run configuration
# ------------------------------------------------------------
H5_DIR=/pscratch/sd/b/binxia/VPIC_PPPL_HDF5_by_beta_official2500_none_compat

# Bump this when the training configuration changes.
RUN_NAME=masked-unet3d_beta0p2_dt8_bc16_fourmask_v1

BETAS=(0.2)
DELTA_T=8
STRIDE_T=2
BATCH_SIZE=4
NUM_WORKERS=4
EPOCHS=80
LR=2e-4
BASE_CHANNELS=16

# Alternating pattern name and sampling weight.
MASK_PATTERNS=(
    spatial_random  1
    spatial_grid    1
    spatial_block   1
    temporal_random 1
)

# online, offline or disabled.
WANDB_MODE=online

OUT_DIR=runs/${RUN_NAME}
LOG=logs/train_${RUN_NAME}.out

TRAIN_ARGS=(
    --h5-dir "${H5_DIR}"
    --betas "${BETAS[@]}"
    --delta-t "${DELTA_T}"
    --stride-t "${STRIDE_T}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --epochs "${EPOCHS}"
    --lr "${LR}"
    --base-channels "${BASE_CHANNELS}"
    --amp
    --mask-patterns "${MASK_PATTERNS[@]}"
    --out-dir "${OUT_DIR}"
    --wandb-name "${RUN_NAME}"
    --wandb-mode "${WANDB_MODE}"

    # Extra arguments override defaults above when argparse accepts
    # repeated options with the last value taking precedence.
    "$@"
)

echo
echo "Start time: $(date)"
echo "Running inside allocation: ${SLURM_JOB_ID:-none}"
echo "Python: ${PYTHON}"
echo "Output dir: ${OUT_DIR}"
echo "Log: ${LOG}"
echo "Command: ${PYTHON} train_masked_unet3d.py ${TRAIN_ARGS[*]}"
echo

if [ -n "${SLURM_JOB_ID:-}" ]; then
    # Single-GPU training.
    srun \
        -N 1 \
        -n 1 \
        -c 32 \
        --gpus-per-task=1 \
        "${PYTHON}" train_masked_unet3d.py "${TRAIN_ARGS[@]}" \
        2>&1 | tee "${LOG}"
else
    echo "No SLURM allocation detected, running directly on this node."

    "${PYTHON}" train_masked_unet3d.py "${TRAIN_ARGS[@]}" \
        2>&1 | tee "${LOG}"
fi

echo
echo "End time: $(date)"
