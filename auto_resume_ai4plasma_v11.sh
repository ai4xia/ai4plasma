#!/bin/bash
# ==============================================================================
# auto_resume_ai4plasma_v11.sh
#
# Automatic Slurm allocation and training-resume monitor for:
#   Run: multi_beta_gkl_v11_fromscratch_256ep_20261122
#   Target: 256 epochs
#
# Usage:
#   bash auto_resume_ai4plasma_v11.sh [initial_slurm_job_id]
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
PROJECT_DIR="/pscratch/sd/b/binxia/ai4plasma"
RUN_NAME="${CURRICULUM_RUN_NAME:-multi_beta_gkl_v11_fromscratch_256ep_20261122}"
RUN_DIR="${PROJECT_DIR}/runs/${RUN_NAME}"

CHECKPOINT="${RUN_DIR}/latest.pt"
FINAL_CHECKPOINT="${RUN_DIR}/final.pt"
HISTORY_FILE="${RUN_DIR}/history.json"
ACTIVE_JOB_REGISTRY="${RUN_DIR}/active_slurm_job_id"
MONITOR_LOG="${RUN_DIR}/auto_resume_monitor.log"
LOCK_FILE="${RUN_DIR}/.auto_resume.lock"

TARGET_EPOCH=256
POLL_INTERVAL=300          # 5 minutes between status checks
MAX_NO_PROGRESS=3          # Stop after 3 consecutive sessions without progress
MAX_ALLOC_RETRIES=20       # Maximum consecutive allocation request retries
BASE_RETRY_DELAY=60        # Seconds to wait after failed allocation request

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
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
    listing=$(squeue --noheader --jobs="${jid}" --format='%i' 2>/dev/null || true)
    [[ -n "${listing//[[:space:]]/}" ]]
}

get_job_status() {
    local jid="$1"
    squeue --noheader --jobs="${jid}" --format='%i %T elapsed=%M nodes=%N' 2>/dev/null || echo "Job ${jid} not in queue"
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
    if [[ -s "${FINAL_CHECKPOINT}" ]]; then
        return 0
    fi
    local ep
    ep=$(get_completed_epoch)
    if (( ep >= TARGET_EPOCH )); then
        return 0
    fi
    return 1
}

verify_checkpoint() {
    if [[ ! -f "${CHECKPOINT}" ]]; then
        log "ERROR: Checkpoint file does not exist: ${CHECKPOINT}"
        return 1
    fi
    local size
    size=$(stat -c%s "${CHECKPOINT}" 2>/dev/null || echo 0)
    if (( size < 10000000 )); then
        log "ERROR: Checkpoint file is suspiciously small (${size} bytes): ${CHECKPOINT}"
        return 1
    fi
    # Test zip archive integrity
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

# ------------------------------------------------------------------------------
# Initialization & Concurrency Lock
# ------------------------------------------------------------------------------
mkdir -p "${RUN_DIR}"
exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
    existing_holder=$(<"${LOCK_FILE}" 2>/dev/null || echo "unknown")
    echo "ERROR: Another auto_resume monitor instance is already active." >&2
    echo "Lock file: ${LOCK_FILE} (Holder: ${existing_holder})" >&2
    exit 1
fi
echo "$$ $(date '+%Y-%m-%d %H:%M:%S')" >&200

log "================================================================================"
log "ai4plasma Auto-Resume Monitor started (PID: $$)"
log "Project Directory: ${PROJECT_DIR}"
log "Run Directory:     ${RUN_DIR}"
log "Target Epoch:      ${TARGET_EPOCH}"
log "================================================================================"

# Parse Initial Slurm Allocation ID
INITIAL_JOB_ID="${1:-}"
if [[ -z "${INITIAL_JOB_ID}" ]]; then
    if [[ -f "${ACTIVE_JOB_REGISTRY}" ]]; then
        INITIAL_JOB_ID=$(<"${ACTIVE_JOB_REGISTRY}")
        INITIAL_JOB_ID=$(echo "${INITIAL_JOB_ID}" | tr -d '[:space:]')
        log "No job ID passed as argument; detected registered job ID: ${INITIAL_JOB_ID}"
    fi
fi

if [[ -z "${INITIAL_JOB_ID}" || ! "${INITIAL_JOB_ID}" =~ ^[0-9]+$ ]]; then
    log "ERROR: A valid numerical Slurm job ID must be provided as argument or registered in ${ACTIVE_JOB_REGISTRY}."
    echo "Usage: bash $0 <initial_slurm_job_id>" >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# Phase 1: Monitor Initial Active Allocation
# ------------------------------------------------------------------------------
if is_training_complete; then
    current_ep=$(get_completed_epoch)
    log "Training is already complete (final.pt exists or epoch ${current_ep} >= ${TARGET_EPOCH}). Exiting."
    exit 0
fi

if is_job_active "${INITIAL_JOB_ID}"; then
    log "Initial allocation ${INITIAL_JOB_ID} is currently active. Monitoring..."
    while is_job_active "${INITIAL_JOB_ID}"; do
        status_line=$(get_job_status "${INITIAL_JOB_ID}")
        current_ep=$(get_completed_epoch)
        log "Job ${INITIAL_JOB_ID} active: [${status_line}]. Completed epoch: ${current_ep}/${TARGET_EPOCH}. Sleeping ${POLL_INTERVAL}s..."
        sleep "${POLL_INTERVAL}"
    done
    log "Initial allocation ${INITIAL_JOB_ID} has finished."
else
    log "Initial allocation ${INITIAL_JOB_ID} is not currently running in squeue (already ended or inactive)."
fi

# ------------------------------------------------------------------------------
# Phase 2: Resume Loop
# ------------------------------------------------------------------------------
NO_PROGRESS_STREAK=0
ALLOC_RETRY_COUNT=0

while true; do
    # Check if target reached
    if is_training_complete; then
        current_ep=$(get_completed_epoch)
        log "================================================================================"
        log "SUCCESS: Training complete! Reached target epoch ${TARGET_EPOCH} (current completed epoch: ${current_ep})."
        log "Monitor exiting cleanly."
        log "================================================================================"
        exit 0
    fi

    # Verify checkpoint integrity
    if ! verify_checkpoint; then
        log "FATAL: Resumable checkpoint check failed. Halting monitor to protect experiment state."
        exit 1
    fi

    current_ep=$(get_completed_epoch)
    log "Resumable checkpoint verified. Completed epoch: ${current_ep} / ${TARGET_EPOCH}."

    # Guard: ensure no other allocation is currently running under this run's registry
    if [[ -f "${ACTIVE_JOB_REGISTRY}" ]]; then
        reg_job=$(<"${ACTIVE_JOB_REGISTRY}")
        reg_job=$(echo "${reg_job}" | tr -d '[:space:]')
        if [[ "${reg_job}" =~ ^[0-9]+$ ]] && is_job_active "${reg_job}"; then
            log "Notice: Slurm allocation ${reg_job} is currently running per ${ACTIVE_JOB_REGISTRY}."
            log "Waiting for active allocation ${reg_job} to finish before requesting a new one..."
            while is_job_active "${reg_job}"; do
                status_line=$(get_job_status "${reg_job}")
                current_ep=$(get_completed_epoch)
                log "Job ${reg_job} active: [${status_line}]. Completed epoch: ${current_ep}/${TARGET_EPOCH}. Sleeping ${POLL_INTERVAL}s..."
                sleep "${POLL_INTERVAL}"
            done
            log "Allocation ${reg_job} has finished."
            continue
        fi
    fi

    epoch_before="${current_ep}"
    log "Requesting new 4-node / 16-GPU allocation..."
    log "Command: salloc -N4 -q interactive -C 'gpu&hbm80g' -t4:00:00 bash -c \"cd '${PROJECT_DIR}' && CURRICULUM_RUN_NAME='${RUN_NAME}' bash train_masked_unet3d_4n16g.sbatch --resume-gkl-joint-grad-globalcosine-256ep\""

    start_time=$(date +%s)

    set +e
    salloc -N4 -q interactive -C 'gpu&hbm80g' -t4:00:00 bash -c "cd '${PROJECT_DIR}' && CURRICULUM_RUN_NAME='${RUN_NAME}' bash train_masked_unet3d_4n16g.sbatch --resume-gkl-joint-grad-globalcosine-256ep"
    salloc_rc=$?
    set -e

    end_time=$(date +%s)
    duration=$(( end_time - start_time ))
    log "salloc allocation ended after ${duration}s (exit code: ${salloc_rc})."

    # Assess progress
    epoch_after=$(get_completed_epoch)
    log "Epoch before allocation: ${epoch_before} | Epoch after allocation: ${epoch_after}"

    if is_training_complete; then
        log "================================================================================"
        log "SUCCESS: Training complete after allocation! Target epoch ${TARGET_EPOCH} reached."
        log "Monitor exiting cleanly."
        log "================================================================================"
        exit 0
    fi

    if (( epoch_after > epoch_before )); then
        advanced=$(( epoch_after - epoch_before ))
        log "Epoch progress confirmed: advanced +${advanced} epochs (now at epoch ${epoch_after})."
        NO_PROGRESS_STREAK=0
        ALLOC_RETRY_COUNT=0
    else
        # No progress made during this cycle
        if (( duration < 120 )) && (( salloc_rc != 0 )); then
            # Fast failure, likely transient allocation or queue rejection
            ALLOC_RETRY_COUNT=$(( ALLOC_RETRY_COUNT + 1 ))
            log "WARNING: Allocation request terminated quickly (${duration}s, exit code ${salloc_rc}). Retry attempt ${ALLOC_RETRY_COUNT}/${MAX_ALLOC_RETRIES}."
            if (( ALLOC_RETRY_COUNT >= MAX_ALLOC_RETRIES )); then
                log "FATAL: Exceeded maximum consecutive allocation retries (${MAX_ALLOC_RETRIES}). Aborting."
                exit 1
            fi
            retry_multiplier=$(( ALLOC_RETRY_COUNT > 5 ? 5 : ALLOC_RETRY_COUNT ))
            retry_delay=$(( BASE_RETRY_DELAY * retry_multiplier ))
            log "Sleeping ${retry_delay}s before retrying allocation request..."
            sleep "${retry_delay}"
        else
            # Job ran or crashed without saving any new epochs
            NO_PROGRESS_STREAK=$(( NO_PROGRESS_STREAK + 1 ))
            log "WARNING: Training session completed without advancing epochs (stuck at epoch ${epoch_after}). Zero-progress streak: ${NO_PROGRESS_STREAK}/${MAX_NO_PROGRESS}."
            if (( NO_PROGRESS_STREAK >= MAX_NO_PROGRESS )); then
                log "FATAL: Training repeatedly failed without making epoch progress for ${MAX_NO_PROGRESS} consecutive runs. Aborting to prevent infinite crash loop."
                exit 1
            fi
            log "Sleeping 30s before next resume attempt..."
            sleep 30
        fi
    fi
done
