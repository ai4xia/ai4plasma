# Matched 20-Epoch GKL Masking Continuation: Control (v9) vs. Experiment (v11)

**Date**: 2026-11-21  
**Author**: Computational Plasma Physics & Machine Learning Team  
**Repository**: `/pscratch/sd/b/binxia/ai4plasma`  
**Control Worktree (v9)**: `/pscratch/sd/b/binxia/ai4plasma-v9-control`  
**Slurm Allocation ID**: `61749951` (Account: `cusp_g`, Partition: `urgent_gpu_ss11`, 4 nodes / 16 GPUs A100-SXM4-80GB)  
**Evaluation Status**: **COMPLETED & AUDITED**

---

## 1. Executive Summary

This study conducted a matched, controlled 20-epoch continuation experiment (epochs 257–276) from the completed **fresh Fully Differentiable GKL A+B epoch-256 checkpoint** (`runs/multi_beta_three_phase_gkl_joint_grad_globalcosine_256ep_20261119/final.pt`):
- **Control Run (`v9` masking)**: Historical production distribution with independent Magnetic/Density patterns and legacy boundary sampling. Checkpoint: `runs/multi_beta_gkl_v9_control_20ep_20261121/final.pt`.
- **Experiment Run (`v11` masking)**: Unified circular temporal block sampling paired with **shared temporal masks across all 4 channels** and independent spatial masks. Checkpoint: `runs/multi_beta_gkl_v11_shared_temporal_20ep_20261121/final.pt`.

### Key Numerical Results
- **Validation MSE (Overall)**: `v9` = 0.014927 $\rightarrow$ `v11` = **0.014116** (**-5.43% lower error**)
- **Validation GKL Loss**: `v9` = 0.015665 $\rightarrow$ `v11` = **0.014825** (**-5.36% lower loss**)
- **Test Set `temporal_block` MSE**: `v9` = 0.120447 $\rightarrow$ `v11` = **0.103458** (**-14.11% reduction**, $p < 10^{-4}$)
- **Forward Forecasting MSE ($h=16$)**: `v9` = 0.206237 $\rightarrow$ `v11` = **0.157123** (**-23.81% reduction**)
- **Reverse Extrapolation MSE ($h=20$)**: `v9` = 0.304196 $\rightarrow$ `v11` = **0.206043** (**-32.27% reduction**)
- **Long-Horizon Calibration ($h=22$)**: $u/\text{MSE}$ increased from **0.7527 to 0.8124**, alleviating uncertainty collapse.
- **Numerical Stability**: 0 nonfinite events, 0 AMP gradient scale skips across both runs.

---

## 2. Checkpoints & Provenance

### Source Checkpoint
- **Path**: `runs/multi_beta_three_phase_gkl_joint_grad_globalcosine_256ep_20261119/final.pt`
- **Model**: UNet3D (`base_channels=24`, `channel_mults=[1, 2, 4, 8]`)
- **Epoch**: 256
- **Split SHA-256**: `29837a718c54ff95b9c7b9668478d3eb`
- **Curriculum**: Settled Phase C rollout ($\lambda_v = 1.0, \lambda_r = 1.0$, $K=1$, LR = $1.0 \times 10^{-5}$)

### Evaluated Continuations
1. **Control Run (v9)**:
   - Checkpoint: `runs/multi_beta_gkl_v9_control_20ep_20261121/final.pt`
   - Config: `runs/multi_beta_gkl_v9_control_20ep_20261121/config.json`
   - History: `runs/multi_beta_gkl_v9_control_20ep_20261121/history.json`
   - Worktree: `/pscratch/sd/b/binxia/ai4plasma-v9-control` (pinned commit `2fa0716`)
2. **Experiment Run (v11)**:
   - Checkpoint: `runs/multi_beta_gkl_v11_shared_temporal_20ep_20261121/final.pt`
   - Config: `runs/multi_beta_gkl_v11_shared_temporal_20ep_20261121/config.json`
   - History: `runs/multi_beta_gkl_v11_shared_temporal_20ep_20261121/history.json`
   - Worktree: `/pscratch/sd/b/binxia/ai4plasma` (`main` branch)

---

## 3. Launcher Implementation Details (`train_masked_unet3d.py`)

The launcher was extended with the flag `--gkl-masking-continuation-20ep` to authorize continuation from settled GKL epoch-256 without resetting optimizers or curriculum ramps:
1. **Flags Required**: `--post-final-continuation`, `--curriculum-config three_phase_global_cosine`, `--uncertainty-loss generalized_kl`, `--differentiable-phase-c-reuse`.
2. **Resume Validation**: Added `("gkl_masking_continuation_20ep", False)` to `validate_resume_parameters`. In `validate_post_final_continuation`, authorized the source `...v9` to target `...v11` masking transition strictly and exclusively when this flag is active.
3. **Gradient Routing**: Connected flag to `enable_variance_feature_grad` (Path B ON) and `enable_mean_nll_grad` (Path A ON), maintaining $\lambda_r = 1.0$ differentiable reuse.

---

## 4. Evaluation Artifacts

All evaluation scripts, metrics CSVs, and plots reside in:
`evaluations/matched_gkl_masking_continuation_20ep_20261121/`

- **Scientific Report**: [`report.md`](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/matched_gkl_masking_continuation_20ep_20261121/report.md)
- **Training Audit**: [`training_audit.json`](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/matched_gkl_masking_continuation_20ep_20261121/training_audit.json)
- **Training Curves**: [`training_curves.png`](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/matched_gkl_masking_continuation_20ep_20261121/training_curves.png)
- **Standard Mask Comparisons**: [`standard_mask_metrics.csv`](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/matched_gkl_masking_continuation_20ep_20261121/standard_mask_metrics.csv) & [`standard_mask_comparison.png`](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/matched_gkl_masking_continuation_20ep_20261121/standard_mask_comparison.png)
- **Forecast Horizon Metrics**: [`forecast_horizon_metrics.csv`](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/matched_gkl_masking_continuation_20ep_20261121/forecast_horizon_metrics.csv) & [`forecast_horizon_comparison.png`](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/matched_gkl_masking_continuation_20ep_20261121/forecast_horizon_comparison.png)
- **Temporal Regimes (Reverse & Sparse)**: [`temporal_extrapolation_and_sparse_metrics.csv`](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/matched_gkl_masking_continuation_20ep_20261121/temporal_extrapolation_and_sparse_metrics.csv) & [`temporal_regimes_comparison.png`](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/matched_gkl_masking_continuation_20ep_20261121/temporal_regimes_comparison.png)
- **Paired Statistics & Bootstrap CIs**: [`paired_summary.csv`](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/matched_gkl_masking_continuation_20ep_20261121/paired_summary.csv)
- **Evaluation Log**: [`evaluation.log`](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/matched_gkl_masking_continuation_20ep_20261121/evaluation.log)

---

## 5. Instructions for Codex / Human Engineers

### Re-running the Scientific Evaluation (1 GPU, ~10 minutes):
```bash
cd /pscratch/sd/b/binxia/ai4plasma
srun -A cusp_g -C gpu -q gpu_interactive -N 1 -n 1 --gpus=1 -t 00:30:00 \
  evaluations/matched_gkl_masking_continuation_20ep_20261121/run_eval.sh
```

### Removing the Auxiliary Control Worktree (Optional Cleanup):
```bash
cd /pscratch/sd/b/binxia/ai4plasma
git worktree remove /pscratch/sd/b/binxia/ai4plasma-v9-control
```
