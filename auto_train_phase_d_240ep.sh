#!/bin/bash
# ==============================================================================
# auto_train_phase_d_240ep.sh
#
# Single entry-point automatic 4-node Perlmutter salloc & resume monitor for:
#   Adaptive 240-Epoch From-Scratch A -> B -> C -> D Sliding Curriculum
#
# Usage (from Perlmutter login node):
#   bash auto_train_phase_d_240ep.sh [optional_initial_slurm_job_id]
# ==============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/pscratch/sd/b/binxia/ai4plasma}"
RUN_NAME="${CURRICULUM_RUN_NAME:-multi_beta_gkl_v11_adaptive_abcd_sliding_240ep}"
RUN_DIR="${PROJECT_DIR}/runs/${RUN_NAME}"
MONITOR_STATE_DIR="${MONITOR_STATE_DIR:-${PROJECT_DIR}/runs/.monitor_${RUN_NAME}}"

CHECKPOINT="${RUN_DIR}/latest.pt"
FINAL_CHECKPOINT="${RUN_DIR}/final.pt"
HISTORY_FILE="${RUN_DIR}/history.json"
ACTIVE_JOB_REGISTRY="${RUN_DIR}/active_slurm_job_id"
MONITOR_LOG="${MONITOR_STATE_DIR}/auto_resume_monitor.log"
LOCK_FILE="${MONITOR_STATE_DIR}/auto_resume.lock"

TARGET_EPOCH="${TARGET_EPOCH:-240}"
POLL_INTERVAL="${POLL_INTERVAL:-300}"
MAX_NO_PROGRESS="${MAX_NO_PROGRESS:-3}"
MAX_ALLOC_RETRIES="${MAX_ALLOC_RETRIES:-20}"
BASE_RETRY_DELAY="${BASE_RETRY_DELAY:-60}"
MIN_CKPT_BYTES="${MIN_CKPT_BYTES:-10000000}"
SALLOC_TIMEOUT_FLAG="${SALLOC_TIMEOUT_FLAG:-4:00:00}"
SALLOC_CMD="${SALLOC_CMD:-salloc}"
SQUEUE_CMD="${SQUEUE_CMD:-squeue}"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}"
    echo "${msg}" >> "${MONITOR_LOG}"
}

is_job_active() {
    local jid="$1"
    if [[ -z "${jid}" ]]; then
        return 1
    fi
    local listing
    listing=$("${SQUEUE_CMD}" --noheader --jobs="${jid}" --format='%i' 2>/dev/null || true)
    [[ -n "${listing//[[:space:]]/}" ]]
}

get_job_status() {
    local jid="$1"
    "${SQUEUE_CMD}" --noheader --jobs="${jid}" --format='%i %T elapsed=%M nodes=%N' 2>/dev/null || echo "Job ${jid} not in queue"
}

get_completed_epoch() {
    if [[ -f "${HISTORY_FILE}" ]]; then
        python3 -c "
import json, sys
try:
    with open('${HISTORY_FILE}') as f:
        data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            print(int(data[-1].get('epoch', 0)))
            sys.exit(0)
except Exception:
    pass
print(0)
" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

is_training_complete() {
    # Require final.pt to exist AND verify epoch >= TARGET_EPOCH and full A->B->C->D completion.
    if [[ ! -s "${FINAL_CHECKPOINT}" ]]; then
        return 1
    fi
    python3 -c "
import sys, torch
try:
    ckpt = torch.load('${FINAL_CHECKPOINT}', map_location='cpu', weights_only=False)
    ep = int(ckpt.get('epoch', 0))
    cs = ckpt.get('curriculum_state') or {}
    steps = cs.get('successful_optimizer_steps', -1)
    cc = cs.get('completion_conditions')
    cond_ok = True if cc is None else (
        bool(cc.get('lr_cosine_complete', False))
        and bool(cc.get('slide_cosine_complete', False))
        and bool(cc.get('phase_d_converged', False))
    )
    done = (
        ep >= ${TARGET_EPOCH}
        and bool(cs.get('adaptive_phase_d_sliding', False))
        and bool(cs.get('variance_ramp_armed', False))
        and bool(cs.get('rollout_ramp_armed', False))
        and bool(cs.get('slide_ramp_armed', False))
        and steps >= cs.get('slide_ramp_start_step', 0) + cs.get('slide_ramp_steps', 1)
        and cond_ok
    )
    sys.exit(0 if done else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

verify_checkpoint() {
    if [[ ! -f "${CHECKPOINT}" ]]; then
        log "ERROR: Checkpoint file does not exist: ${CHECKPOINT}"
        return 1
    fi
    local size
    size=$(stat -c%s "${CHECKPOINT}" 2>/dev/null || echo 0)
    if (( size < MIN_CKPT_BYTES )); then
        log "ERROR: Checkpoint file is suspiciously small (${size} bytes): ${CHECKPOINT}"
        return 1
    fi
    if ! python3 -m zipfile -t "${CHECKPOINT}" >/dev/null 2>&1; then
        log "ERROR: Checkpoint file failed zip archive integrity verification: ${CHECKPOINT}"
        return 1
    fi
    return 0
}

cleanup() {
    log "Monitor session ended for PID $$."
}
trap cleanup EXIT

# Concurrency Lock (kept outside RUN_DIR so fresh-run OUT_DIR does not pre-exist)
mkdir -p "${MONITOR_STATE_DIR}"
exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
    existing_holder=$(<"${LOCK_FILE}" 2>/dev/null || echo "unknown")
    echo "ERROR: Another auto_train_phase_d_240ep monitor instance is already active." >&2
    echo "Lock file: ${LOCK_FILE} (Holder: ${existing_holder})" >&2
    exit 1
fi
echo "$$ $(date '+%Y-%m-%d %H:%M:%S')" >&200

log "================================================================================"
log "ai4plasma 240-Epoch Adaptive A->B->C->D Monitor started (PID: $$)"
log "Project Directory: ${PROJECT_DIR}"
log "Run Directory:     ${RUN_DIR}"
log "Target Epoch:      ${TARGET_EPOCH}"
log "================================================================================"

if is_training_complete; then
    current_ep=$(get_completed_epoch)
    log "Training is already complete (verified final.pt at epoch ${current_ep}). Exiting."
    exit 0
fi

# Optional Phase 1: Monitor existing active Slurm allocation if provided or registered
INITIAL_JOB_ID="${1:-}"
if [[ -z "${INITIAL_JOB_ID}" && -f "${ACTIVE_JOB_REGISTRY}" ]]; then
    INITIAL_JOB_ID=$(tr -d '[:space:]' < "${ACTIVE_JOB_REGISTRY}" || true)
fi

if [[ -n "${INITIAL_JOB_ID}" && "${INITIAL_JOB_ID}" =~ ^[0-9]+$ ]] && is_job_active "${INITIAL_JOB_ID}"; then
    log "Existing allocation ${INITIAL_JOB_ID} is active. Monitoring..."
    while is_job_active "${INITIAL_JOB_ID}"; do
        status_line=$(get_job_status "${INITIAL_JOB_ID}")
        current_ep=$(get_completed_epoch)
        log "Job ${INITIAL_JOB_ID} active: [${status_line}]. Completed epoch: ${current_ep}/${TARGET_EPOCH}. Sleeping ${POLL_INTERVAL}s..."
        sleep "${POLL_INTERVAL}"
    done
    log "Existing allocation ${INITIAL_JOB_ID} has finished."
fi

# Phase 2: Allocation & Resume Loop
NO_PROGRESS_STREAK=0
ALLOC_RETRY_COUNT=0

while true; do
    if is_training_complete; then
        current_ep=$(get_completed_epoch)
        log "SUCCESS: Training & full A->B->C->D curriculum complete at epoch ${current_ep}. Exiting."
        exit 0
    fi

    current_ep=$(get_completed_epoch)
    if [[ -f "${CHECKPOINT}" ]]; then
        if ! verify_checkpoint; then
            log "FATAL: Resumable checkpoint check failed: ${CHECKPOINT}. Halting monitor."
            exit 1
        fi
        LAUNCH_FLAG="--resume-adaptive-abcd-sliding-240ep"
        log "Resumable checkpoint verified. Completed epoch: ${current_ep}/${TARGET_EPOCH}. Using ${LAUNCH_FLAG}."
    else
        if (( current_ep > 0 )); then
            log "FATAL: History shows epoch ${current_ep} > 0 but ${CHECKPOINT} is missing. Refusing to restart from scratch."
            exit 1
        fi
        LAUNCH_FLAG="--adaptive-abcd-sliding-240ep"
        log "No checkpoint found; launching fresh from-scratch run with ${LAUNCH_FLAG}."
    fi

    # Guard against concurrent active allocation in registry
    if [[ -f "${ACTIVE_JOB_REGISTRY}" ]]; then
        reg_job=$(tr -d '[:space:]' < "${ACTIVE_JOB_REGISTRY}" || true)
        if [[ "${reg_job}" =~ ^[0-9]+$ ]] && is_job_active "${reg_job}"; then
            log "Notice: Allocation ${reg_job} is active per registry; waiting..."
            while is_job_active "${reg_job}"; do
                sleep "${POLL_INTERVAL}"
            done
            continue
        fi
    fi

    epoch_before="${current_ep}"
    log "Requesting 4-node / 16-GPU interactive allocation (${LAUNCH_FLAG})..."
    start_time=$(date +%s)

    set +e
    "${SALLOC_CMD}" -N4 -q interactive -C 'gpu&hbm80g' -t "${SALLOC_TIMEOUT_FLAG}" \
        bash -c "cd '${PROJECT_DIR}' && CURRICULUM_RUN_NAME='${RUN_NAME}' bash train_masked_unet3d_4n16g.sbatch ${LAUNCH_FLAG}"
    salloc_rc=$?
    set -e

    end_time=$(date +%s)
    duration=$(( end_time - start_time ))
    log "salloc allocation ended after ${duration}s (exit code: ${salloc_rc})."

    if is_training_complete; then
        epoch_after=$(get_completed_epoch)
        log "SUCCESS: Full 240-epoch A->B->C->D curriculum complete (epoch ${epoch_after}). Exiting."
        exit 0
    fi

    epoch_after=$(get_completed_epoch)
    log "Epoch before allocation: ${epoch_before} | Epoch after allocation: ${epoch_after}"

    if (( epoch_after > epoch_before )); then
        NO_PROGRESS_STREAK=0
        ALLOC_RETRY_COUNT=0
        log "Progress confirmed: +$(( epoch_after - epoch_before )) epochs (now at ${epoch_after}/${TARGET_EPOCH})."
    else
        if (( duration < 120 )) && (( salloc_rc != 0 )); then
            ALLOC_RETRY_COUNT=$(( ALLOC_RETRY_COUNT + 1 ))
            log "Transient allocation failure/rejection (attempt ${ALLOC_RETRY_COUNT}/${MAX_ALLOC_RETRIES})."
            if (( ALLOC_RETRY_COUNT >= MAX_ALLOC_RETRIES )); then
                log "FATAL: Exceeded ${MAX_ALLOC_RETRIES} allocation retries. Halting monitor."
                exit 1
            fi
            delay=$(( BASE_RETRY_DELAY * (ALLOC_RETRY_COUNT < 5 ? ALLOC_RETRY_COUNT : 5) ))
            log "Sleeping ${delay}s before retrying allocation..."
            sleep "${delay}"
            continue
        else
            NO_PROGRESS_STREAK=$(( NO_PROGRESS_STREAK + 1 ))
            log "WARNING: No epoch progress (streak ${NO_PROGRESS_STREAK}/${MAX_NO_PROGRESS})."
            if (( NO_PROGRESS_STREAK >= MAX_NO_PROGRESS )); then
                log "FATAL: ${MAX_NO_PROGRESS} consecutive sessions made zero progress. Halting monitor."
                exit 1
            fi
            sleep "${BASE_RETRY_DELAY}"
        fi
    fi
done
