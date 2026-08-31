#!/bin/bash

#SBATCH --job-name=ai4plasma-unet3d
#SBATCH --account=cusp
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/pscratch/sd/b/binxia/ai4plasma/logs/slurm-%x-%j.out
#SBATCH --error=/pscratch/sd/b/binxia/ai4plasma/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR=/pscratch/sd/b/binxia/ai4plasma
NNODES=4
GPUS_PER_NODE=4

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this file inside a 4-node allocation or submit it with: sbatch $0" >&2
    exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p logs

ml load pytorch
PYTHON=$(command -v python)

if (( SLURM_NNODES < NNODES )); then
    echo "Need at least ${NNODES} allocated nodes; got ${SLURM_NNODES}." >&2
    exit 2
fi

mapfile -t SLURM_HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
MASTER_ADDR=${SLURM_HOSTS[0]}
JOB_NUMBER=${SLURM_JOB_ID%%_*}
JOB_NUMBER=${JOB_NUMBER%%.*}
MASTER_PORT=${MASTER_PORT:-$((10000 + JOB_NUMBER % 50000))}
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

export MASTER_ADDR MASTER_PORT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NCCL_ASYNC_ERROR_HANDLING=1

echo "Loaded PyTorch environment:"
echo "  Python: ${PYTHON}"
"${PYTHON}" - <<'PY'
import torch

print(f"  PyTorch: {torch.__version__}")
print(f"  PyTorch CUDA: {torch.version.cuda}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; refusing to start DDP training.")
PY

H5_DIR=/pscratch/sd/b/binxia/VPIC_PPPL_HDF5_by_beta_official2500_none_compat
SOURCE_RUN_NAME=masked-resunet3d_beta0p2_dt24_bc24_depth4_ddp16_mixedB50D50_warmup10_cosine1500_v9_balancedloss
INIT_CHECKPOINT=runs/${SOURCE_RUN_NAME}/latest.pt
RUN_NAME=masked-resunet3d_beta0p2_dt24_bc24_depth4_ddp16_mixedB50D50_warmup10_cosine1500_v10_temporalboundary

BETAS=(0.2)
DELTA_T=24
STRIDE_T=2

# Per-GPU batch size. With 16 GPUs this gives global batch size 64.
BATCH_SIZE=4
NUM_WORKERS=4
EPOCHS=1500
LR=2e-4
WARMUP_EPOCHS=10
MIN_LR=2e-6
DENSITY_PROBE_MIN=0
DENSITY_PROBE_MAX=30
BASE_CHANNELS=24
CHANNEL_MULTS=(1 2 4 8)

MASK_PATTERNS=(
    spatial_random  1
    spatial_grid    1
    spatial_block   1
    temporal_random 1
)

WANDB_MODE=online
OUT_DIR=runs/${RUN_NAME}
LOG=logs/train_${RUN_NAME}_${SLURM_JOB_ID}.out

TRAIN_ARGS=(
    --h5-dir "${H5_DIR}"
    --betas "${BETAS[@]}"
    --delta-t "${DELTA_T}"
    --stride-t "${STRIDE_T}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --epochs "${EPOCHS}"
    --lr "${LR}"
    --warmup-epochs "${WARMUP_EPOCHS}"
    --min-lr "${MIN_LR}"
    --density-probe-min "${DENSITY_PROBE_MIN}"
    --density-probe-max "${DENSITY_PROBE_MAX}"
    --base-channels "${BASE_CHANNELS}"
    --channel-mults "${CHANNEL_MULTS[@]}"
    --amp
    --mask-patterns "${MASK_PATTERNS[@]}"
    --init-checkpoint "${INIT_CHECKPOINT}"
    --out-dir "${OUT_DIR}"
    --wandb-name "${RUN_NAME}"
    --wandb-mode "${WANDB_MODE}"
    "$@"
)

echo
echo "Start time: $(date)"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "Nodes: ${NNODES}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "DDP world size: ${WORLD_SIZE}"
echo "Per-GPU batch size: ${BATCH_SIZE}"
echo "Global batch size: $((BATCH_SIZE * WORLD_SIZE))"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Output dir: ${OUT_DIR}"
echo "Log: ${LOG}"
echo

# One torchrun controller per node, with four workers (one per local GPU).
srun \
    --nodes="${NNODES}" \
    --ntasks="${NNODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --cpu-bind=cores \
    --gpus-per-task="${GPUS_PER_NODE}" \
    --gpu-bind=none \
    --kill-on-bad-exit=1 \
    bash -c '
        exec "$1" -m torch.distributed.run \
            --nnodes="$2" \
            --nproc-per-node="$3" \
            --rdzv-backend=c10d \
            --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
            --rdzv-id="${SLURM_JOB_ID}" \
            "${@:4}"
    ' bash "${PYTHON}" "${NNODES}" "${GPUS_PER_NODE}" \
        train_masked_unet3d.py "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${LOG}"

echo
echo "End time: $(date)"
