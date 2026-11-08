#!/bin/bash

#SBATCH --job-name=ai4plasma-unet3d
#SBATCH --account=m5240_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --output=/pscratch/sd/b/binxia/ai4plasma/logs/slurm-%x-%j.out
#SBATCH --error=/pscratch/sd/b/binxia/ai4plasma/logs/slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="/pscratch/sd/b/binxia/ai4plasma"
NNODES=4
GPUS_PER_NODE=4
STAGE1_NLL=false
RESUME_STAGE1_NLL=false
CURRICULUM=false
ABCD_SYMMETRIC=false
GLOBAL_COSINE=false
POST_FINAL_CONTINUATION=false
PHASE_D_PROBE=false
PHASE_C_K2_PROBE=false
FROMSCRATCH_48H=false
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
    --post-final-continuation)
        CURRICULUM=true; GLOBAL_COSINE=true; POST_FINAL_CONTINUATION=true; RESUMING=false
        CURRICULUM_STOP=320; FINAL_EPOCH=320
        MODEL_ARGS=(--model-version direct_unet3d_no_activation_norm_v3); shift ;;
    --resume-post-final-continuation)
        CURRICULUM=true; GLOBAL_COSINE=true; POST_FINAL_CONTINUATION=true
        CURRICULUM_STOP=320; FINAL_EPOCH=320
        MODEL_ARGS=(--model-version direct_unet3d_no_activation_norm_v3); shift ;;
    --phase-d-stateful-probe)
        CURRICULUM=true; GLOBAL_COSINE=true; POST_FINAL_CONTINUATION=true
        PHASE_D_PROBE=true; RESUMING=false
        CURRICULUM_STOP=1020; FINAL_EPOCH=1020
        MODEL_ARGS=(--model-version direct_unet3d_no_activation_norm_v3); shift ;;
    --resume-phase-d-stateful-probe)
        CURRICULUM=true; GLOBAL_COSINE=true; POST_FINAL_CONTINUATION=true
        PHASE_D_PROBE=true
        CURRICULUM_STOP=1020; FINAL_EPOCH=1020
        MODEL_ARGS=(--model-version direct_unet3d_no_activation_norm_v3); shift ;;
    --phase-c-k2-probe)
        CURRICULUM=true; PHASE_C_K2_PROBE=true; shift ;;
    --fromscratch-global-cosine-48h)
        CURRICULUM=true; GLOBAL_COSINE=true; FROMSCRATCH_48H=true; RESUMING=false
        CURRICULUM_STOP=1000; FINAL_EPOCH=1000
        MODEL_ARGS=(--model-version direct_unet3d_no_activation_norm_v3); shift ;;
    --resume-fromscratch-global-cosine-48h)
        CURRICULUM=true; GLOBAL_COSINE=true; FROMSCRATCH_48H=true
        CURRICULUM_STOP=1000; FINAL_EPOCH=1000
        MODEL_ARGS=(--model-version direct_unet3d_no_activation_norm_v3); shift ;;
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
        echo "Usage: bash $0 [--stage1-gaussian-nll|--resume-stage1-gaussian-nll|--curriculum|--resume-curriculum|--abcd-symmetric-lr|--resume-abcd-symmetric-lr|--global-cosine-lr|--resume-global-cosine-lr|--post-final-continuation|--resume-post-final-continuation|--phase-d-stateful-probe|--resume-phase-d-stateful-probe|--phase-c-k2-probe|--fromscratch-global-cosine-48h|--resume-fromscratch-global-cosine-48h] [--arm-variance-ramp|--arm-rollout-ramp] [--stop-after-epochs N] [--final-epoch N] [--model-version VERSION]" >&2
        exit 2
    fi
done
if (( ${#ARM_ARGS[@]} )) && ! ${RESUMING}; then
    echo "ERROR: ramp arming requires --resume-curriculum." >&2
    exit 2
fi
HANDOFF_ACTIVE_SALLOC=${HANDOFF_ACTIVE_SALLOC:-0}
if [[ "${HANDOFF_ACTIVE_SALLOC}" != 0 && "${HANDOFF_ACTIVE_SALLOC}" != 1 ]]; then
    echo "ERROR: HANDOFF_ACTIVE_SALLOC must be 0 or 1." >&2
    exit 2
fi
if [[ "${HANDOFF_ACTIVE_SALLOC}" == 1 ]] && { ! ${FROMSCRATCH_48H} || ! ${RESUMING}; }; then
    echo "ERROR: HANDOFF_ACTIVE_SALLOC=1 is only valid for the 1000-epoch resume mode." >&2
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

if ${PHASE_C_K2_PROBE}; then
    : "${PHASE_C_PROBE_CHECKPOINT:?ERROR: PHASE_C_PROBE_CHECKPOINT must name an absolute checkpoint path}"
    : "${CURRICULUM_RUN_NAME:?ERROR: CURRICULUM_RUN_NAME must be set for --phase-c-k2-probe}"
    if [[ "${PHASE_C_PROBE_CHECKPOINT}" != /* || ! -f "${PHASE_C_PROBE_CHECKPOINT}" ]]; then
        echo "ERROR: PHASE_C_PROBE_CHECKPOINT must be an existing absolute file." >&2
        exit 2
    fi
    if [[ "${CURRICULUM_RUN_NAME}" == */* || "${CURRICULUM_RUN_NAME}" == "." || "${CURRICULUM_RUN_NAME}" == ".." ]]; then
        echo "ERROR: CURRICULUM_RUN_NAME must be a single run-directory name." >&2
        exit 2
    fi
    IFS=$'\t' read -r PROBE_SOURCE_EPOCH PROBE_EPOCHS PROBE_FINAL_EPOCH \
        PROBE_LR PROBE_WARMUP_EPOCHS PROBE_MIN_LR PROBE_CURRICULUM_CONFIG \
        PROBE_MODEL_VERSION PROBE_SOURCE_PHASE PROBE_LAMBDA_V PROBE_LAMBDA_R \
        PROBE_OPTIMIZER_LR < <(
        "${PYTHON}" - "${PHASE_C_PROBE_CHECKPOINT}" <<'PY'
from pathlib import Path
import sys
import torch

from train_masked_unet3d import curriculum_phase, curriculum_weights

path = Path(sys.argv[1])
checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
args = checkpoint["args"]
state = checkpoint["curriculum_state"]
epoch = int(checkpoint["epoch"])
lambda_v, lambda_r = curriculum_weights(state)
config = args.get("curriculum_config", "adaptive")
phase_index, phase_name = curriculum_phase(epoch + 1, state, lambda_v, lambda_r, config)
if phase_index != 2 or not (lambda_v == 1.0 and 0.0 < lambda_r < 1.0):
    raise ValueError(
        f"Phase-C K=2 probe requires an in-ramp Phase-C checkpoint; got "
        f"phase={phase_name}, lambda_v={lambda_v}, lambda_r={lambda_r}"
    )
if state.get("phase_d_start_epoch") is not None:
    raise ValueError("Phase-C K=2 probe source already contains Phase-D state")
stop_epoch = epoch + 20
if stop_epoch >= int(args["final_epoch"]):
    raise ValueError(
        f"Twenty probe epochs would reach/exceed final_epoch={args['final_epoch']}"
    )
completion_step = state["rollout_ramp_start_step"] + state["rollout_ramp_steps"]
maximum_probe_step = state["successful_optimizer_steps"] + 20 * state["updates_per_epoch"]
if maximum_probe_step >= completion_step:
    raise ValueError(
        "Twenty probe epochs could complete the rollout ramp and enter Phase D; "
        f"maximum_probe_step={maximum_probe_step}, completion_step={completion_step}"
    )
group_lrs = {float(group["lr"]) for group in checkpoint["optimizer"]["param_groups"]}
if len(group_lrs) != 1:
    raise ValueError(f"Expected one optimizer LR, got {sorted(group_lrs)}")
print("\t".join(map(str, (
    epoch, args["epochs"], args["final_epoch"], args["lr"],
    args["warmup_epochs"], args["min_lr"], config, args["model_version"],
    phase_name, lambda_v, lambda_r, group_lrs.pop(),
))))
PY
    )
    CURRICULUM_STOP=$((PROBE_SOURCE_EPOCH + 20))
    FINAL_EPOCH=${PROBE_FINAL_EPOCH}
    MODEL_ARGS=(--model-version "${PROBE_MODEL_VERSION}")
    case "${PROBE_CURRICULUM_CONFIG}" in
        adaptive) ;;
        abcd_symmetric_lr) ABCD_SYMMETRIC=true ;;
        global_cosine) GLOBAL_COSINE=true ;;
        *) echo "ERROR: unsupported source curriculum config: ${PROBE_CURRICULUM_CONFIG}" >&2; exit 2 ;;
    esac
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

H5_DIR=/pscratch/sd/b/binxia/VPIC_PPPL_HDF5_by_beta_official2500_none_compat
RUN_NAME=multi_beta_smoothl1_6ep_horizon20_20261018
if ${PHASE_C_K2_PROBE}; then
    RUN_NAME=${CURRICULUM_RUN_NAME}
elif ${FROMSCRATCH_48H}; then
    RUN_NAME=${FROMSCRATCH_RUN_NAME:-multi_beta_curriculum_direct_v3_fromscratch_globalcosine_2e4_1e5_1000ep_48h_20261027}
elif ${PHASE_D_PROBE}; then
    RUN_NAME=${CURRICULUM_RUN_NAME:-phase_d_stateful_sliding_probe_from_1000ep_20261105}
elif ${POST_FINAL_CONTINUATION}; then
    RUN_NAME=${CONTINUATION_RUN_NAME:-multi_beta_curriculum_direct_v3_globalcosine_uqcontinue_e240_e320_20261027}
elif ${STAGE1_NLL}; then
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
if ${FROMSCRATCH_48H}; then EPOCHS=1000; fi
if ${PHASE_D_PROBE}; then EPOCHS=1000; fi
if ${PHASE_C_K2_PROBE}; then EPOCHS=${PROBE_EPOCHS}; fi
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
if ${FROMSCRATCH_48H}; then
    LR=2e-4
    MIN_LR=1e-5
fi
if ${PHASE_D_PROBE}; then MIN_LR=1e-5; fi
if ${PHASE_C_K2_PROBE}; then
    LR=${PROBE_LR}
    WARMUP_EPOCHS=${PROBE_WARMUP_EPOCHS}
    MIN_LR=${PROBE_MIN_LR}
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
ACTIVE_JOB_REGISTRY=${OUT_DIR}/active_slurm_job_id
if ${PHASE_D_PROBE}; then
    CONTINUATION_SOURCE_CHECKPOINT=${CONTINUATION_SOURCE_CHECKPOINT:-runs/multi_beta_curriculum_direct_v3_fromscratch_globalcosine_2e4_1e5_1000ep_48h_20261027/final.pt}
else
    CONTINUATION_SOURCE_CHECKPOINT=${CONTINUATION_SOURCE_CHECKPOINT:-runs/multi_beta_curriculum_direct_v3_fixedsplit_globalcosine_2e4_1e6_20261026/final.pt}
fi
LOG=logs/train_${RUN_NAME}_${SLURM_JOB_ID}.out

slurm_job_is_active() {
    local listing
    listing=$(squeue --noheader --jobs="$1" --format='%i' 2>/dev/null || true)
    [[ -n "${listing}" ]]
}

if [[ "${HANDOFF_ACTIVE_SALLOC}" == 1 ]]; then
    if [[ ! -e "${ACTIVE_JOB_REGISTRY}" ]]; then
        echo "Handoff: no active-allocation registry exists; no salloc job to cancel."
    else
        REGISTERED_JOB_ID=$(<"${ACTIVE_JOB_REGISTRY}")
        if [[ -z "${REGISTERED_JOB_ID}" ]]; then
            echo "Handoff: active-allocation registry is empty; no salloc job to cancel."
        elif [[ ! "${REGISTERED_JOB_ID}" =~ ^[0-9]+$ ]]; then
            echo "ERROR: invalid Slurm job ID in ${ACTIVE_JOB_REGISTRY}: ${REGISTERED_JOB_ID@Q}" >&2
            exit 2
        elif [[ "${REGISTERED_JOB_ID}" == "${SLURM_JOB_ID}" ]]; then
            echo "ERROR: handoff registry points to the current batch job ${SLURM_JOB_ID}; refusing self-cancellation." >&2
            exit 2
        else
            REGISTERED_JOB_INFO=$(scontrol show job "${REGISTERED_JOB_ID}" -o 2>/dev/null || true)
            if [[ -z "${REGISTERED_JOB_INFO}" ]] || ! slurm_job_is_active "${REGISTERED_JOB_ID}"; then
                echo "Handoff: registered job ${REGISTERED_JOB_ID} is stale or already gone; nothing to cancel."
            else
                CURRENT_USER=$(id -un)
                if [[ "${REGISTERED_JOB_INFO}" != *" UserId=${CURRENT_USER}("* ]]; then
                    echo "ERROR: registered job ${REGISTERED_JOB_ID} is not owned by ${CURRENT_USER}; refusing cancellation." >&2
                    exit 2
                fi
                if [[ "${REGISTERED_JOB_INFO}" != *" Account=m5240_g "* ]] ||
                   [[ "${REGISTERED_JOB_INFO}" != *" BatchFlag=0 "* ]] ||
                   [[ ! "${REGISTERED_JOB_INFO}" =~ NumNodes=([0-9]+) ]] ||
                   (( BASH_REMATCH[1] != NNODES )) ||
                   [[ ! "${REGISTERED_JOB_INFO}" =~ AllocTRES=[^[:space:]]*gres/gpu=([0-9]+) ]] ||
                   (( BASH_REMATCH[1] < NNODES * GPUS_PER_NODE )); then
                    echo "ERROR: registered job ${REGISTERED_JOB_ID} is not the expected 4-node/16-GPU salloc allocation; refusing cancellation." >&2
                    exit 2
                fi
                echo "Handoff: cancelling registered active salloc job ${REGISTERED_JOB_ID}."
                scancel "${REGISTERED_JOB_ID}"
                HANDOFF_DEADLINE=$((SECONDS + 120))
                while slurm_job_is_active "${REGISTERED_JOB_ID}"; do
                    if (( SECONDS >= HANDOFF_DEADLINE )); then
                        echo "ERROR: registered job ${REGISTERED_JOB_ID} is still active after 120 seconds; aborting before checkpoint access/DDP." >&2
                        exit 2
                    fi
                    sleep 2
                done
                echo "Handoff: registered salloc job ${REGISTERED_JOB_ID} has terminated; checkpoint resume may proceed."
            fi
        fi
    fi
fi

echo "Loaded PyTorch environment:"
echo "  Python: ${PYTHON}"
"${PYTHON}" - <<'PY'
import torch

print(f"  PyTorch: {torch.__version__}")
print(f"  PyTorch CUDA: {torch.version.cuda}")
print("  CUDA availability will be checked inside the srun workers.")
PY

if [[ ! -d "${H5_DIR}" ]]; then
    echo "ERROR: HDF5 directory not found: ${H5_DIR}" >&2
    exit 2
fi
if [[ ! -f "${PROJECT_DIR}/split.json" ]]; then
    echo "ERROR: authoritative split not found: ${PROJECT_DIR}/split.json" >&2
    exit 2
fi
if ${PHASE_C_K2_PROBE} && [[ -e "${OUT_DIR}" ]]; then
    echo "ERROR: fresh output directory already exists: ${OUT_DIR}; refusing overwrite/resume." >&2
    exit 2
elif ! ${RESUMING} && [[ -e "${OUT_DIR}" ]]; then
    echo "ERROR: fresh output directory already exists: ${OUT_DIR}; refusing overwrite/resume." >&2
    exit 2
fi
if ${RESUMING} && ! ${PHASE_C_K2_PROBE} && [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: required resume checkpoint not found: ${CHECKPOINT}; refusing fresh training." >&2
    exit 2
fi
if ${POST_FINAL_CONTINUATION} && [[ ! -f "${CONTINUATION_SOURCE_CHECKPOINT}" ]]; then
    echo "ERROR: continuation source checkpoint not found: ${CONTINUATION_SOURCE_CHECKPOINT}" >&2
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
        if ${POST_FINAL_CONTINUATION}; then
            TRAIN_ARGS+=(--post-final-continuation "${CONTINUATION_SOURCE_CHECKPOINT}")
            if ${PHASE_D_PROBE}; then
                TRAIN_ARGS+=(--wandb-tags post-final-continuation phase-d-stateful sliding-probe from-1000ep)
            else
                TRAIN_ARGS+=(--wandb-tags post-final-continuation uq-convergence)
            fi
        fi
        if ${FROMSCRATCH_48H}; then
            TRAIN_ARGS+=(
                --wandb-tags from-scratch global-cosine-2e4-1e5 48h
            )
        fi
        if ${PHASE_C_K2_PROBE}; then
            TRAIN_ARGS+=(--wandb-tags phase-c-k2-probe)
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

if ${PHASE_C_K2_PROBE}; then
    "${PYTHON}" - "${PHASE_C_PROBE_CHECKPOINT}" "${OUT_DIR}" "${TRAIN_ARGS[@]}" <<'PY'
from pathlib import Path
import json
import os
import shutil
import sys
import torch

from data.vpic_hdf5_dataset import validate_checkpoint_split
from train_masked_unet3d import (
    curriculum_learning_rate_for_epoch, curriculum_phase, curriculum_weights,
    parse_args, validate_resume_parameters,
)

source = Path(sys.argv[1]).resolve()
out_dir = Path(sys.argv[2])
sys.argv = [sys.argv[0], *sys.argv[3:]]
args = parse_args()
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
validate_checkpoint_split(checkpoint)
validate_resume_parameters(args, checkpoint["args"])
state = checkpoint["curriculum_state"]
epoch = int(checkpoint["epoch"])
lambda_v, lambda_r = curriculum_weights(state)
phase_index, phase_name = curriculum_phase(
    epoch + 1, state, lambda_v, lambda_r, args.curriculum_config,
)
if phase_index != 2 or state.get("phase_d_start_epoch") is not None:
    raise ValueError(f"Source is not pre-Phase-D Phase C: {phase_name}")
completion_step = state["rollout_ramp_start_step"] + state["rollout_ramp_steps"]
maximum_probe_step = state["successful_optimizer_steps"] + 20 * state["updates_per_epoch"]
if maximum_probe_step >= completion_step:
    raise ValueError("Probe could complete the rollout ramp and enter Phase D")
if args.stop_after_epochs != epoch + 20:
    raise ValueError("Probe must stop exactly 20 epochs after the source checkpoint")
source_history = source.parent / "history.json"
if source_history.is_file():
    history = json.loads(source_history.read_text())
    if not history or int(history[-1]["epoch"]) != epoch:
        raise ValueError("Source history does not end at the checkpoint epoch")
next_lr = curriculum_learning_rate_for_epoch(epoch + 1, args, state)
out_dir.mkdir(parents=True, exist_ok=False)
checkpoint.pop("wandb_run_id", None)  # keep the source W&B run isolated
temporary = out_dir / "latest.pt.tmp"
torch.save(checkpoint, temporary)
os.replace(temporary, out_dir / "latest.pt")
if source_history.is_file():
    shutil.copy2(source_history, out_dir / "history.json")
print(f"Phase-C K=2 probe source checkpoint: {source}", file=sys.stderr)
print(f"Phase-C K=2 probe source epoch/phase: {epoch} / {phase_name}", file=sys.stderr)
print(f"Phase-C K=2 probe output directory: {out_dir}", file=sys.stderr)
print(f"Phase-C K=2 probe epochs: {epoch + 1}-{epoch + 20} (20 additional)", file=sys.stderr)
print(f"Phase-C K=2 probe LR next/current optimizer: {next_lr:g} / "
      f"{checkpoint['optimizer']['param_groups'][0]['lr']:g}", file=sys.stderr)
print(f"Phase-C K=2 probe lambda_v/lambda_r: {lambda_v:g} / {lambda_r:g}", file=sys.stderr)
PY
fi

# Check the same compatibility rules as the trainer before starting workers.
# Resume uses the LR horizon recorded by the selected curriculum configuration.
COMPLETED_EPOCH=0
if ${FROMSCRATCH_48H} && ! ${RESUMING}; then
FROMSCRATCH_PREFLIGHT_RESULT=$("${PYTHON}" - "${TRAIN_ARGS[@]}" <<'PY'
from pathlib import Path
import sys
import torch

from models.unet3d import UNet3D
from train_masked_unet3d import (
    curriculum_learning_rate_for_epoch, curriculum_phase, curriculum_weights,
    make_curriculum_state, parse_args, phase_a_convergence,
    resolve_checkpoint_plan,
)

args = parse_args()
out_dir = Path(args.out_dir).resolve()
if args.init_checkpoint is not None or args.post_final_continuation is not None:
    raise ValueError("From-scratch mode must not receive any source checkpoint")
if out_dir.exists():
    raise ValueError(f"Fresh output directory already exists: {out_dir}")
plan = resolve_checkpoint_plan(
    out_dir, auto_resume=args.auto_resume, init_checkpoint=args.init_checkpoint,
)
if plan.mode != "fresh" or plan.path is not None:
    raise ValueError(f"From-scratch checkpoint plan is not fresh: {plan}")

model = UNet3D(
    in_channels=8, out_channels=4, base_channels=args.base_channels,
    channel_mults=args.channel_mults, architecture=args.model_version,
    use_attention=args.use_attention,
    spatial_only_pooling=args.spatial_only_pooling,
    uncertainty_aware=args.uncertainty_aware,
)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
)
if optimizer.state:
    raise ValueError("From-scratch optimizer unexpectedly has state")
state = make_curriculum_state(args, updates_per_epoch=1)
if curriculum_weights(state) != (0.0, 0.0):
    raise ValueError("From-scratch curriculum does not begin with zero lambdas")
if curriculum_phase(1, state, 0.0, 0.0, args.curriculum_config) != (0, "mean_only"):
    raise ValueError("From-scratch curriculum does not begin in Phase A")
phase_a = phase_a_convergence(
    [{"epoch": 1, "val_mse": 1.0, "learning_rate": args.min_lr}], args, state,
)
if not phase_a["phase_a"] or phase_a["stable_count"] != 0 or phase_a["ready_for_variance_ramp"]:
    raise ValueError(f"Phase-A convergence state is not fresh: {phase_a}")
if (args.epochs, args.final_epoch, args.stop_after_epochs) != (1000, 1000, 1000):
    raise ValueError("From-scratch cosine horizon must be exactly 1000 epochs")
lrs = {
    1: curriculum_learning_rate_for_epoch(1, args, state),
    args.warmup_epochs: curriculum_learning_rate_for_epoch(args.warmup_epochs, args, state),
    args.final_epoch: curriculum_learning_rate_for_epoch(args.final_epoch, args, state),
}
if lrs != {1: 1e-5, 5: 2e-4, 1000: 1e-5}:
    raise ValueError(f"Unexpected from-scratch LR endpoints: {lrs}")
decay_lrs = [
    curriculum_learning_rate_for_epoch(epoch, args, state)
    for epoch in range(args.warmup_epochs, args.final_epoch + 1)
]
if any(left < right for left, right in zip(decay_lrs, decay_lrs[1:])):
    raise ValueError("From-scratch LR is not monotonically nonincreasing after warmup")
print(
    "From-scratch preflight confirmed: random model initialization; no source checkpoint; "
    "completed epoch=0; next epoch=1; optimizer=fresh; "
    "lambda_v=lambda_r=0; fresh Phase-A convergence; "
    "LR(epoch 1/5/1000)=1e-5/2e-4/1e-5; post-warmup monotonic=yes; "
    "cosine_horizon=1000; "
    f"split_sha256={args.split_sha256}; output={args.out_dir}; "
    f"wandb_name={args.wandb_name}",
    file=sys.stderr,
)
print(0)
PY
)
COMPLETED_EPOCH=${FROMSCRATCH_PREFLIGHT_RESULT}
elif ${PHASE_C_K2_PROBE}; then
    COMPLETED_EPOCH=${PROBE_SOURCE_EPOCH}
elif ${RESUMING} || ${POST_FINAL_CONTINUATION}; then
COMPLETED_EPOCH=$("${PYTHON}" - "${TRAIN_ARGS[@]}" <<'PY'
from pathlib import Path
import json
import math
import sys
import torch
from data.vpic_hdf5_dataset import validate_checkpoint_split
from train_masked_unet3d import (
    curriculum_weights, make_curriculum_state, parse_args,
    validate_post_final_continuation, validate_resume_parameters,
)

args = parse_args()
latest = Path(args.out_dir) / "latest.pt"
initial_continuation = bool(args.post_final_continuation and not latest.is_file())
checkpoint_path = Path(args.post_final_continuation) if initial_continuation else latest
checkpoint = torch.load(checkpoint_path, map_location="cpu")
for key in ("model", "optimizer", "stats", "epoch", "args"):
    if key not in checkpoint or not checkpoint[key]:
        raise ValueError(f"Cannot resume: missing or empty checkpoint {key!r}")
if args.amp and args.amp_dtype == "float16" and not checkpoint.get("scaler"):
    raise ValueError("Cannot resume: missing or empty checkpoint 'scaler'")
validate_checkpoint_split(checkpoint)
if initial_continuation:
    validate_post_final_continuation(args, checkpoint, checkpoint_path, Path(args.out_dir))
else:
    validate_resume_parameters(args, checkpoint["args"])
epoch = int(checkpoint["epoch"])
limit = (args.stop_after_epochs or args.epochs) if args.curriculum_training else args.epochs
minimum_epoch = args.epochs if args.post_final_continuation else (1 if args.curriculum_training else 6)
if not minimum_epoch <= epoch <= limit:
    raise ValueError(f"Unexpected completed epoch {epoch} for final epoch {limit}")
if args.curriculum_training:
    state = make_curriculum_state(args, checkpoint["curriculum_state"]["updates_per_epoch"], checkpoint)
    if args.post_final_continuation and curriculum_weights(state) != (1.0, 1.0):
        raise ValueError("Continuation did not restore lambda_v=lambda_r=1")
    if (args.arm_variance_ramp or args.arm_rollout_ramp) and epoch >= limit:
        raise ValueError("Arming requires --stop-after-epochs beyond the completed epoch")
if args.uncertainty_loss == "gaussian_nll":
    history_path = Path(args.out_dir) / "history.json"
    if initial_continuation:
        if history_path.exists():
            raise ValueError("Fresh continuation output unexpectedly contains history.json")
    else:
        history = json.loads(history_path.read_text())
        if args.post_final_continuation:
            source = torch.load(args.post_final_continuation, map_location="cpu", weights_only=False, mmap=True)
            first_epoch = int(source["epoch"]) + 1
        else:
            first_epoch = 1
        if [row["epoch"] for row in history] != list(range(first_epoch, epoch + 1)):
            raise ValueError("Cannot resume: history does not match completed checkpoint epochs.")
if args.post_final_continuation:
    group_lrs = [float(group["lr"]) for group in checkpoint["optimizer"]["param_groups"]]
    if not group_lrs or any(not math.isclose(lr, args.min_lr, rel_tol=0., abs_tol=1e-15) for lr in group_lrs):
        raise ValueError(f"Continuation optimizer LR is not fixed at {args.min_lr:g}: {group_lrs}")
    print(
        f"Continuation preflight confirmed: source/completed epoch={epoch}, "
        f"next epoch={epoch + 1}, final epoch={limit}, lambda_v=lambda_r=1, "
        f"optimizer restored ({len(checkpoint['optimizer']['state'])} parameter states), "
        f"LR={args.min_lr:g}, split_sha256={checkpoint['split_sha256']}, "
        f"output={args.out_dir}",
        file=sys.stderr,
    )
print(epoch)
PY
)
if (( COMPLETED_EPOCH >= STOP_EPOCH )); then
    echo "Training is already complete through epoch ${STOP_EPOCH}; no workers launched."
    exit 0
fi
fi

if ${FROMSCRATCH_48H} && [[ "${JOB_INFO}" == *" BatchFlag=0 "* ]]; then
    if [[ -e "${ACTIVE_JOB_REGISTRY}" ]]; then
        PREVIOUS_JOB_ID=$(<"${ACTIVE_JOB_REGISTRY}")
        if [[ "${PREVIOUS_JOB_ID}" =~ ^[0-9]+$ ]] &&
           [[ "${PREVIOUS_JOB_ID}" != "${SLURM_JOB_ID}" ]] &&
           slurm_job_is_active "${PREVIOUS_JOB_ID}"; then
            echo "ERROR: run registry still names active job ${PREVIOUS_JOB_ID}; refusing concurrent training from salloc ${SLURM_JOB_ID}." >&2
            exit 2
        fi
        if [[ -n "${PREVIOUS_JOB_ID}" && "${PREVIOUS_JOB_ID}" != "${SLURM_JOB_ID}" ]]; then
            echo "Active-allocation registry: replacing stale job ID ${PREVIOUS_JOB_ID@Q}."
        fi
    fi
    mkdir -p "${OUT_DIR}"
    REGISTRY_TMP=${ACTIVE_JOB_REGISTRY}.tmp.${SLURM_JOB_ID}.$$
    printf '%s\n' "${SLURM_JOB_ID}" > "${REGISTRY_TMP}"
    mv -f -- "${REGISTRY_TMP}" "${ACTIVE_JOB_REGISTRY}"
    echo "Active-allocation registry: atomically registered salloc job ${SLURM_JOB_ID} in ${ACTIVE_JOB_REGISTRY}."
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
    if ${PHASE_C_K2_PROBE}; then
        echo "Phase-C K=2 probe source checkpoint: ${PHASE_C_PROBE_CHECKPOINT}"
        echo "Phase-C K=2 probe source epoch/phase: ${PROBE_SOURCE_EPOCH} / ${PROBE_SOURCE_PHASE}"
        echo "Phase-C K=2 probe output directory: ${OUT_DIR}"
        echo "Phase-C K=2 probe duration: 20 additional epochs"
        echo "Phase-C K=2 probe optimizer LR: ${PROBE_OPTIMIZER_LR}"
        echo "Phase-C K=2 probe lambda_v/lambda_r: ${PROBE_LAMBDA_V} / ${PROBE_LAMBDA_R}"
        echo "Phase-C K=2 probe guard: rollout ramp cannot complete; Phase D remains inactive"
    elif ${FROMSCRATCH_48H}; then
        echo "Curriculum state: fresh adaptive Phase A from random model initialization"
    elif ${POST_FINAL_CONTINUATION}; then
        echo "Curriculum state: fixed full objective; adaptive transitions disabled"
    else
        echo "Adaptive curriculum: readiness automatically arms unarmed ramps; manual arming flags: ${ARM_ARGS[*]:-none}"
    fi
    if ${GLOBAL_COSINE}; then
        if ${FROMSCRATCH_48H}; then
            echo "True from-scratch run: random model; fresh optimizer/epoch/curriculum/W&B; no init checkpoint"
            echo "Global cosine LR: epochs 1-5 ${MIN_LR}->${LR}; epochs 6-${FINAL_EPOCH} cosine decay to ${MIN_LR}"
        elif ${POST_FINAL_CONTINUATION}; then
            echo "Post-final continuation: source epoch ${EPOCHS}; epochs $((EPOCHS + 1))-${FINAL_EPOCH}; lambda_v=lambda_r=1"
            echo "Fixed LR=${MIN_LR}; no warmup, reheat, convergence detector, or curriculum ramp"
            echo "Source checkpoint: ${CONTINUATION_SOURCE_CHECKPOINT}"
        else
            echo "Global cosine LR: epochs 1-5 ${MIN_LR}->${LR}; epochs 6-240 one cosine decay to ${MIN_LR}"
            echo "Curriculum transitions do not restart or modify LR; final epoch ${FINAL_EPOCH}"
        fi
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
