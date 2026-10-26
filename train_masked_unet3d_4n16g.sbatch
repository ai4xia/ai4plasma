#!/bin/bash

#SBATCH --job-name=ai4plasma-unet3d
#SBATCH --account=m5240_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --output=/pscratch/sd/b/binxia/ai4plasma/logs/slurm-%x-%j.out
#SBATCH --error=/pscratch/sd/b/binxia/ai4plasma/logs/slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
NNODES=4
GPUS_PER_NODE=4
STAGE1_NLL=false
RESUME_STAGE1_NLL=false
CURRICULUM=false
ABCD_SYMMETRIC=false
GLOBAL_COSINE=false
RESUMING=true
CURRICULUM_STOP=200
FINAL_EPOCH=20
ARM_ARGS=()
MODEL_ARGS=()
case "${1:-}" in
    --stage1-gaussian-nll) STAGE1_NLL=true; RESUMING=false; shift ;;
    --resume-stage1-gaussian-nll) STAGE1_NLL=true; RESUME_STAGE1_NLL=true; shift ;;
    --curriculum) CURRICULUM=true; RESUMING=false; shift ;;
    --resume-curriculum) CURRICULUM=true; shift ;;
    --abcd-symmetric-lr)
        CURRICULUM=true; ABCD_SYMMETRIC=true; RESUMING=false
        CURRICULUM_STOP=240; FINAL_EPOCH=240; shift ;;
    --resume-abcd-symmetric-lr)
        CURRICULUM=true; ABCD_SYMMETRIC=true
        CURRICULUM_STOP=240; FINAL_EPOCH=240; shift ;;
    --global-cosine-lr)
        CURRICULUM=true; GLOBAL_COSINE=true; RESUMING=false
        CURRICULUM_STOP=240; FINAL_EPOCH=240; shift ;;
    --resume-global-cosine-lr)
        CURRICULUM=true; GLOBAL_COSINE=true
        CURRICULUM_STOP=240; FINAL_EPOCH=240; shift ;;
    "") ;;
    *) echo "Unknown training mode: $1" >&2; exit 2 ;;
esac
while (( $# )); do
    if ${CURRICULUM} && [[ "$1" == --arm-variance-ramp || "$1" == --arm-rollout-ramp ]]; then
        ARM_ARGS+=("$1"); shift
    elif ${CURRICULUM} && [[ "$1" == --stop-after-epochs ]] && (( $# >= 2 )) && [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
        CURRICULUM_STOP=$2; shift 2
    elif [[ "$1" == --final-epoch ]] && (( $# >= 2 )) && [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
        FINAL_EPOCH=$2; shift 2
    elif [[ "$1" == --model-version ]] && (( $# >= 2 )); then
        MODEL_ARGS=(--model-version "$2"); shift 2
    else
        echo "Usage: bash $0 [--stage1-gaussian-nll|--resume-stage1-gaussian-nll|--curriculum|--resume-curriculum|--abcd-symmetric-lr|--resume-abcd-symmetric-lr|--global-cosine-lr|--resume-global-cosine-lr] [--arm-variance-ramp|--arm-rollout-ramp] [--stop-after-epochs N] [--final-epoch N] [--model-version VERSION]" >&2
        exit 2
    fi
done
if (( ${#ARM_ARGS[@]} )) && ! ${RESUMING}; then
    echo "ERROR: ramp arming requires --resume-curriculum." >&2
    exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run this file inside an existing 4-node / 16-GPU Slurm allocation." >&2
    exit 2
fi

JOB_INFO=$(scontrol show job "${SLURM_JOB_ID}" -o)
if [[ "${JOB_INFO}" != *" JobState=RUNNING "* ]] ||
   [[ ! "${JOB_INFO}" =~ NumNodes=([0-9]+) ]] ||
   (( BASH_REMATCH[1] != NNODES )); then
    echo "ERROR: need a running ${NNODES}-node allocation (job ${SLURM_JOB_ID})." >&2
    exit 2
fi
if [[ ! "${JOB_INFO}" =~ AllocTRES=[^[:space:]]*gres/gpu=([0-9]+) ]] ||
   (( BASH_REMATCH[1] < NNODES * GPUS_PER_NODE )); then
    echo "ERROR: need ${GPUS_PER_NODE} allocated GPUs per node / 16 GPUs total." >&2
    exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p logs

ml load pytorch
PYTHON=$(command -v python)

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
print("  CUDA availability will be checked inside the srun workers.")
PY

H5_DIR=/pscratch/sd/b/binxia/VPIC_PPPL_HDF5_by_beta_official2500_none_compat
RUN_NAME=multi_beta_smoothl1_6ep_horizon20_20261018
if ${STAGE1_NLL}; then
    RUN_NAME=multi_beta_stage1_gaussian_nll_6ep_horizon20_20261019
elif ${CURRICULUM}; then
    if ${GLOBAL_COSINE}; then
        RUN_NAME=${CURRICULUM_RUN_NAME:-multi_beta_curriculum_direct_v3_fixedsplit_globalcosine_2e4_1e6_20261026}
    elif ${ABCD_SYMMETRIC}; then
        RUN_NAME=${CURRICULUM_RUN_NAME:-multi_beta_curriculum_direct_v3_fixedsplit_abcd_symmetriclr_20261026}
    else
        RUN_NAME=${CURRICULUM_RUN_NAME:-multi_beta_curriculum_manual_20261020}
        if [[ ${FINAL_EPOCH} -eq 20 ]]; then FINAL_EPOCH=${CURRICULUM_FINAL_EPOCH:-200}; fi
    fi
fi

BETAS=(0.01 0.02 0.04 0.06 0.08 0.10 0.12 0.15 0.18 0.20)
DELTA_T=24
STRIDE_T=2

# Per-GPU batch size. With 16 GPUs this gives global batch size 64.
BATCH_SIZE=4
NUM_WORKERS=4
EPOCHS=20
if ${ABCD_SYMMETRIC}; then EPOCHS=45; fi
if ${GLOBAL_COSINE}; then EPOCHS=240; fi
STOP_EPOCH=${EPOCHS}
if ${STAGE1_NLL} && ! ${RESUME_STAGE1_NLL}; then
    STOP_EPOCH=6
elif ${CURRICULUM}; then
    STOP_EPOCH=${CURRICULUM_STOP}
fi
LR=2e-5
WARMUP_EPOCHS=1
if ${ABCD_SYMMETRIC}; then WARMUP_EPOCHS=5; fi
MIN_LR=2e-6
if ${GLOBAL_COSINE}; then
    LR=2e-4
    WARMUP_EPOCHS=5
    MIN_LR=1e-6
fi
WEIGHT_DECAY=1e-4
DENSITY_PROBE_MIN=0
DENSITY_PROBE_MAX=30
DENSITY_VISIBLE_P_ZERO=0.05
DENSITY_VISIBLE_P_FULL=0.05
MAGNETIC_VISIBLE_P_ZERO=0.05
MAGNETIC_VISIBLE_P_FULL=0.05
BASE_CHANNELS=24
CHANNEL_MULTS=(1 2 4 8)

MASK_PATTERNS=(
    spatial_random  1
    spatial_grid    1
    spatial_block   1
    temporal_random 1
    temporal_block  1
)

OUT_DIR=runs/${RUN_NAME}
CHECKPOINT=${OUT_DIR}/latest.pt
LOG=logs/train_${RUN_NAME}_${SLURM_JOB_ID}.out

if [[ ! -d "${H5_DIR}" ]]; then
    echo "ERROR: HDF5 directory not found: ${H5_DIR}" >&2
    exit 2
fi
if [[ ! -f "${PROJECT_DIR}/split.json" ]]; then
    echo "ERROR: authoritative split not found: ${PROJECT_DIR}/split.json" >&2
    exit 2
fi
if ! ${RESUMING} && [[ -e "${OUT_DIR}" ]]; then
    echo "ERROR: fresh output directory already exists: ${OUT_DIR}; refusing overwrite/resume." >&2
    exit 2
fi
if ${RESUMING} && [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: required resume checkpoint not found: ${CHECKPOINT}; refusing fresh training." >&2
    exit 2
fi

TRAIN_ARGS=(
    "${MODEL_ARGS[@]}"
    --h5-dir "${H5_DIR}"
    --betas "${BETAS[@]}"
    --delta-t "${DELTA_T}"
    --stride-t "${STRIDE_T}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --epochs "${EPOCHS}"
    --final-epoch "${FINAL_EPOCH}"
    --lr "${LR}"
    --warmup-epochs "${WARMUP_EPOCHS}"
    --min-lr "${MIN_LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --seed 1234
    --grad-clip 1.0
    --density-probe-min "${DENSITY_PROBE_MIN}"
    --density-probe-max "${DENSITY_PROBE_MAX}"
    --density-visible-p-zero "${DENSITY_VISIBLE_P_ZERO}"
    --density-visible-p-full "${DENSITY_VISIBLE_P_FULL}"
    --magnetic-visible-p-zero "${MAGNETIC_VISIBLE_P_ZERO}"
    --magnetic-visible-p-full "${MAGNETIC_VISIBLE_P_FULL}"
    --base-channels "${BASE_CHANNELS}"
    --channel-mults "${CHANNEL_MULTS[@]}"
    --use-attention
    --spatial-only-pooling
    --amp
    --amp-dtype bfloat16
    --uncertainty-aware
    --uncertainty-loss-weight 1.0
    --mask-patterns "${MASK_PATTERNS[@]}"
    --out-dir "${OUT_DIR}"
    --wandb-name "${RUN_NAME}"
)
if ${STAGE1_NLL} || ${CURRICULUM}; then
    TRAIN_ARGS+=(--uncertainty-loss gaussian_nll --stop-after-epochs "${STOP_EPOCH}"
    )
    if ${CURRICULUM}; then
        TRAIN_ARGS+=(--curriculum-training "${ARM_ARGS[@]}")
        if ${ABCD_SYMMETRIC}; then
            TRAIN_ARGS+=(--curriculum-config abcd_symmetric_lr)
        elif ${GLOBAL_COSINE}; then
            TRAIN_ARGS+=(--curriculum-config global_cosine)
        fi
    fi
    if ${RESUMING}; then
        TRAIN_ARGS+=(--auto-resume)
    else
        TRAIN_ARGS+=(--no-auto-resume)
    fi
else
    TRAIN_ARGS+=(--auto-resume)
fi

# Check the same compatibility rules as the trainer before starting workers.
# Resume uses the LR horizon recorded by the selected curriculum configuration.
COMPLETED_EPOCH=0
if ${RESUMING}; then
COMPLETED_EPOCH=$("${PYTHON}" - "${TRAIN_ARGS[@]}" <<'PY'
from pathlib import Path
import json
import math
import torch
from data.vpic_hdf5_dataset import validate_checkpoint_split
from train_masked_unet3d import parse_args, validate_resume_parameters, make_curriculum_state

args = parse_args()
checkpoint = torch.load(Path(args.out_dir) / "latest.pt", map_location="cpu")
for key in ("model", "optimizer", "stats", "epoch", "args"):
    if key not in checkpoint or not checkpoint[key]:
        raise ValueError(f"Cannot resume: missing or empty checkpoint {key!r}")
if args.amp and args.amp_dtype == "float16" and not checkpoint.get("scaler"):
    raise ValueError("Cannot resume: missing or empty checkpoint 'scaler'")
validate_resume_parameters(args, checkpoint["args"])
validate_checkpoint_split(checkpoint)
epoch = int(checkpoint["epoch"])
limit = (args.stop_after_epochs or args.epochs) if args.curriculum_training else args.epochs
if not (1 if args.curriculum_training else 6) <= epoch <= limit:
    raise ValueError(f"Unexpected completed epoch {epoch} for final epoch {limit}")
if args.curriculum_training:
    make_curriculum_state(args, checkpoint["curriculum_state"]["updates_per_epoch"], checkpoint)
    if (args.arm_variance_ramp or args.arm_rollout_ramp) and epoch >= limit:
        raise ValueError("Arming requires --stop-after-epochs beyond the completed epoch")
if args.uncertainty_loss == "gaussian_nll":
    history = json.loads((Path(args.out_dir) / "history.json").read_text())
    if [row["epoch"] for row in history] != list(range(1, epoch + 1)):
        raise ValueError("Cannot resume: history does not match completed checkpoint epochs.")
print(epoch)
PY
)
if (( COMPLETED_EPOCH >= STOP_EPOCH )); then
    echo "Training is already complete through epoch ${STOP_EPOCH}; no workers launched."
    exit 0
fi
fi

echo
echo "Start time: $(date)"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "Nodes: ${NNODES}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "DDP world size: ${WORLD_SIZE}"
echo "Per-GPU batch size: ${BATCH_SIZE}"
echo "Global batch size: $((BATCH_SIZE * WORLD_SIZE))"
echo "Epochs: $((COMPLETED_EPOCH + 1)) through ${STOP_EPOCH} (original ${EPOCHS}-epoch LR horizon)"
echo "Predetermined official final epoch: ${FINAL_EPOCH}"
if ${CURRICULUM}; then
    echo "Adaptive curriculum: readiness automatically arms unarmed ramps; manual arming flags: ${ARM_ARGS[*]:-none}"
    if ${GLOBAL_COSINE}; then
        echo "Global cosine LR: epochs 1-5 ${MIN_LR}->${LR}; epochs 6-240 one cosine decay to ${MIN_LR}"
        echo "Curriculum transitions do not restart or modify LR; final epoch ${FINAL_EPOCH}"
    elif ${ABCD_SYMMETRIC}; then
        echo "ABCD symmetric LR: Phase A/D each use 5-epoch ${MIN_LR}->${LR} linear warmup plus 40-epoch cosine decay"
        echo "Phase C->D uses rollout validation MSE and Gaussian NLL convergence; final epoch ${FINAL_EPOCH}"
    else
        echo "40-epoch cosine ramps; original ${EPOCHS}-epoch LR horizon, then LR=${MIN_LR} until Phase D"
        echo "Phase D: 5-epoch linear reheat to LR=1e-5, fixed 40-epoch cosine decay to ${MIN_LR}, then fixed floor"
    fi
    echo "Authoritative split: ${PROJECT_DIR}/split.json; resume=${RESUMING}"
elif ${STAGE1_NLL}; then
    echo "Stage-1 Gaussian NLL: stop after epoch ${STOP_EPOCH}; binary v_in only"
    echo "FP32 NLL variance floor: 1e-6; uncertainty loss weight: 1.0; no output variance cap"
    echo "Authoritative split: ${PROJECT_DIR}/split.json"
    if ${RESUME_STAGE1_NLL}; then
        echo "Resume: ${CHECKPOINT} (model, optimizer, AMP scaler, channel stats, history)"
    else
        echo "Random initialization; no checkpoint weights loaded"
    fi
else
    echo "Resume: ${CHECKPOINT} (model, optimizer, AMP scaler, channel stats)"
fi
echo "Masking version: independentBD_fiveMask_logUniformCounts_orientedSpatialBlock_v9"
echo "Unified spatiotemporal attention: enabled"
echo "Spatial-only pooling: enabled"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Output dir: ${OUT_DIR}"
echo "Log: ${LOG}"
printf 'Training command:'
printf ' %q' "${PYTHON}" train_masked_unet3d.py "${TRAIN_ARGS[@]}"
printf '\n'
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
    2>&1 | tee -a "${LOG}"

echo
echo "End time: $(date)"
