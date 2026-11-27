#!/bin/bash
# Persistent 4-node salloc/resume controller for the 256-epoch LR-horizon
# Adaptive A->B->C->D run using the general_v12 masking sampler.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"
export CURRICULUM_RUN_NAME="${CURRICULUM_RUN_NAME:-multi_beta_gkl_v12_general_adaptive_abcd_cosine256_fromscratch_20261127}"
export TARGET_EPOCH="${TARGET_EPOCH:-256}"

exec bash "${SCRIPT_DIR}/auto_train_phase_d_240ep.sh" "$@"
