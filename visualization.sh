#!/usr/bin/env bash

set -euo pipefail

# Always run relative to this repository, even when launched from elsewhere.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

readonly RUN_DIR="runs/masked-resunet3d_beta0p2_dt24_bc24_depth4_ddp16_mixedB50D50_warmup10_cosine3000_v8"
readonly RUN_NAME="beta0.2_nu2_Bz0_dt2_tau70"
readonly WINDOW_T0=28
readonly PLASMOID_X_INDEX=130
readonly RELATIVE_ERROR_EPS=0.05
readonly INFO_OUT="${RUN_DIR}/figures_information_suite_plasmoid_merger"
readonly SLIDING_OUT="${RUN_DIR}/figures_sliding_density_reconstruction_plasmoid_merger"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: visualization.sh must be run inside an active Slurm GPU allocation." >&2
    echo "Start/enter the allocation first, then run: bash visualization.sh" >&2
    exit 1
fi

if [[ ! -f "${RUN_DIR}/best.pt" ]]; then
    echo "ERROR: checkpoint not found: ${RUN_DIR}/best.pt" >&2
    exit 1
fi

ml load pytorch

mkdir -p "$INFO_OUT" "$SLIDING_OUT"

echo
echo "[1/2] Rendering all 24-frame information-suite experiments"
srun -n 1 -c 32 -G 1 --gpu-bind=none \
    python visualize_mask_patterns_unet3d.py \
    --run-dir "$RUN_DIR" \
    --run-name "$RUN_NAME" \
    --t0 "$WINDOW_T0" \
    --experiment all \
    --all-times \
    --animation-format gif \
    --fps 2 \
    --relative-error-eps "$RELATIVE_ERROR_EPS" \
    --out-dir "$INFO_OUT"

echo
echo "[2/2] Rendering full-run sliding and bidirectional reconstruction"
srun -n 1 -c 32 -G 1 --gpu-bind=none \
    python visualize_sliding_density_reconstruction.py \
    --run-dir "$RUN_DIR" \
    --run-name "$RUN_NAME" \
    --analysis both \
    --slide-steps 8 4 2 1 \
    --refinement-step 8 \
    --refinement-passes 4 \
    --refinement-offset 12 \
    --density-visible-fraction 0.08 \
    --x-index "$PLASMOID_X_INDEX" \
    --animation-format gif \
    --fps 4 \
    --relative-error-eps "$RELATIVE_ERROR_EPS" \
    --out-dir "$SLIDING_OUT"

echo
echo "Visualization complete."
echo "Information suite: ${INFO_OUT}"
echo "Sliding analyses:  ${SLIDING_OUT}"
