ml load pytorch

python make_paper_figures.py \
  --run-dir runs/masked-resunet3d_beta0p2_dt24_bc24_depth4_ddp16_v15_orientedSpatialBlock_independentBD_logUniformCounts_attention_spatialpool_b8_e4500 \
  --checkpoint latest.pt \
  --sliding-max-runs 25 \
