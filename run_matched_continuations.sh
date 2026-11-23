#!/bin/bash
set -euo pipefail

ALLOCATION_ID="61749951"
export SLURM_JOB_ID="${ALLOCATION_ID}"
export SLURM_JOB_NODELIST="nid[200288,200357,200441,200453]"
MAIN_DIR="/pscratch/sd/b/binxia/ai4plasma"
V9_DIR="/pscratch/sd/b/binxia/ai4plasma-v9-control"
LOG_DIR="${MAIN_DIR}/logs"
mkdir -p "${LOG_DIR}"

RUN_LOG="${LOG_DIR}/matched_continuations_${ALLOCATION_ID}.log"
echo "=== Starting Matched GKL Continuations at $(date) ===" | tee -a "${RUN_LOG}"
echo "Slurm Allocation: ${ALLOCATION_ID}" | tee -a "${RUN_LOG}"

# 1. Control Run (v9 masking)
echo "" | tee -a "${RUN_LOG}"
echo "=================================================================" | tee -a "${RUN_LOG}"
echo ">>> STEP 1: Launching Control Run (v9 masking) at $(date)" | tee -a "${RUN_LOG}"
echo "    Worktree: ${V9_DIR}" | tee -a "${RUN_LOG}"
echo "    Run Name: multi_beta_gkl_v9_control_20ep_20261121" | tee -a "${RUN_LOG}"
echo "=================================================================" | tee -a "${RUN_LOG}"

cd "${V9_DIR}"
PROJECT_DIR="${V9_DIR}" \
CURRICULUM_RUN_NAME="multi_beta_gkl_v9_control_20ep_20261121" \
bash train_masked_unet3d_4n16g.sbatch --gkl-masking-continuation-20ep 2>&1 | tee -a "${RUN_LOG}"

echo "" | tee -a "${RUN_LOG}"
echo ">>> STEP 1 COMPLETED SUCCESSFULLY at $(date)" | tee -a "${RUN_LOG}"

# 2. Experiment Run (v11 masking)
echo "" | tee -a "${RUN_LOG}"
echo "=================================================================" | tee -a "${RUN_LOG}"
echo ">>> STEP 2: Launching Experiment Run (v11 masking) at $(date)" | tee -a "${RUN_LOG}"
echo "    Worktree: ${MAIN_DIR}" | tee -a "${RUN_LOG}"
echo "    Run Name: multi_beta_gkl_v11_shared_temporal_20ep_20261121" | tee -a "${RUN_LOG}"
echo "=================================================================" | tee -a "${RUN_LOG}"

cd "${MAIN_DIR}"
PROJECT_DIR="${MAIN_DIR}" \
CURRICULUM_RUN_NAME="multi_beta_gkl_v11_shared_temporal_20ep_20261121" \
bash train_masked_unet3d_4n16g.sbatch --gkl-masking-continuation-20ep 2>&1 | tee -a "${RUN_LOG}"

echo "" | tee -a "${RUN_LOG}"
echo ">>> STEP 2 COMPLETED SUCCESSFULLY at $(date)" | tee -a "${RUN_LOG}"
echo "=== Both Matched GKL Continuations Completed at $(date) ===" | tee -a "${RUN_LOG}"
