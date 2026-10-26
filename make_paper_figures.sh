set -e

ml load pytorch

RUN_DIR=${RUN_DIR:?Set RUN_DIR to the completed fixed-split training run}

python make_paper_figures.py \
  --run-dir "${RUN_DIR}" \
  --checkpoint final.pt \
  --run-name beta0.2_nu2_Bz0_dt2_tau70 \
  --t0 28 \
  --force-recompute \
  --sliding-max-runs 25 "$@"

python evaluate_standardized_validation.py \
  --run-dir "${RUN_DIR}" \
  --checkpoint final.pt \
  --force-recompute
