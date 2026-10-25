set -e

ml load pytorch

python make_paper_figures.py \
  --run-dir runs/multi_beta_curriculum_manual_20261020 \
  --checkpoint latest.pt \
  --run-name beta0.2_nu7_Bz0_dt2_tau70 \
  --t0 28 \
  --force-recompute \
  --sliding-max-runs 25 "$@"

python evaluate_standardized_validation.py \
  --run-dir runs/multi_beta_curriculum_manual_20261020 \
  --checkpoint latest.pt \
  --force-recompute
