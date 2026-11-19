# Completed 100% temporal-extrapolation diagnostic

## Checkpoint and training audit

Compared the epoch-276 20%-exposure Student-t model against the epoch-286 100%-exposure continuation. Both checkpoints loaded successfully; the new model loaded strictly using the existing evaluator. All model parameters are finite. Training completed exactly epochs 277–286, with constant LR 1e-5, Student-t NLL (df=3), and settled Phase C: lambda_v=lambda_r=1, K=1, two forwards, one detached reuse, and zero weight on pass 2. Phase D remained disabled. History records 0 nonfinite events, 0 AMP-skipped steps, and 10690 optimizer updates. Both modalities recorded temporal_extrapolation share 1.0 throughout all ten epochs.

Final training/validation metrics (NLL is the logged pass-0 Student-t objective; lower is better). Training MSE is not directly comparable between models because their task distributions differ.

| Metric | 20% epoch 276 | 100% epoch 286 |
| --- | --- | --- |
| train_mse | 0.011010796 | 0.014439933 |
| val_mse | 0.0138128298 | 0.0163815181 |
| train_student_t_nll | -1.42460943 | -1.61781066 |
| val_student_t_nll | -1.28048139 | -0.817951573 |

## Paired evaluation protocol

Reused `evaluations/temporal_extrapolation_continuation_20ep_20261117/run_analysis.py` functions `load_model`, `evaluate_forecast_all`, and `evaluate_standard_all`; no second inference implementation was introduced. Archived rows labelled `extrapolation_continuation` were reused as the 20% baseline, without rerunning it. Dataset selection is identical: stride-1 dataset, deterministic `select_validation_statistics_indices(..., 24, 1)`, one first complete 24-frame window for each of the 125 test runs, and 125 validation runs for standard masks. The fixed split hash and normalization agree between checkpoints. Batch size 2, bfloat16 inference, and unknown-input noise seeds are unchanged. Forecast noise seed is 201234 + batch_number and reused across horizons. All four channels are exact at t=0..23-h and hidden at t=24-h..23. No horizon metadata is supplied.

Forecast MSE and mean raw predicted u are voxel-weighted over hidden suffixes only, in checkpoint-standardized squared units. Magnetic pools Bx/By/Bz equally; Density uses its own channel. Calibration is mean(u)/mean(squared error), not a mean of per-voxel ratios. The 256 sampled residual/u pairs per window/channel are used only for auxiliary sampled statistics, not the primary MSE or mean-u metrics. Any u/MSE below one indicates underestimation of actual squared error.

## Forecast difficulty and uncertainty response

| Channel | h | MSE20/MSE20(h1) | MSE100/MSE100(h1) | u20/u20(h1) | u100/u100(h1) | u-response change |
| --- | --- | --- | --- | --- | --- | --- |
| Density | 6 | 2.3296 | 1.9574 | 1.4107 | 1.5514 | +9.97% |
| Density | 12 | 4.3764 | 3.3584 | 1.6607 | 1.5554 | -6.34% |
| Density | 18 | 9.3673 | 6.1341 | 2.0086 | 1.7915 | -10.81% |
| Density | 23 | 45.6726 | 58.6468 | 2.0923 | 1.4455 | -30.92% |
| Magnetic | 6 | 2.5410 | 2.1725 | 1.0694 | 1.6474 | +54.05% |
| Magnetic | 12 | 4.3915 | 3.5863 | 1.3112 | 1.7079 | +30.25% |
| Magnetic | 18 | 9.8152 | 7.6416 | 1.7057 | 2.4443 | +43.31% |
| Magnetic | 23 | 38.4889 | 47.5211 | 0.8471 | 0.7726 | -8.79% |

## Selected absolute metrics

| Channel | h | MSE20 | MSE100 | u20 | u100 | u20/MSE20 | u100/MSE100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Density | 1 | 0.01025756 | 0.01019987 | 0.006061502 | 0.003114733 | 0.5909301 | 0.3053697 |
| Density | 6 | 0.02389621 | 0.01996538 | 0.008551098 | 0.004832273 | 0.3578432 | 0.2420326 |
| Density | 12 | 0.04489089 | 0.03425574 | 0.01006635 | 0.004844803 | 0.2242405 | 0.1414304 |
| Density | 18 | 0.096086 | 0.06256656 | 0.01217489 | 0.0055799 | 0.1267083 | 0.08918342 |
| Density | 23 | 0.4684895 | 0.5981903 | 0.01268244 | 0.004502194 | 0.02707091 | 0.007526358 |
| Magnetic | 1 | 0.02860856 | 0.02719389 | 0.006993687 | 0.005540026 | 0.2444614 | 0.2037232 |
| Magnetic | 6 | 0.07269557 | 0.05907879 | 0.007479125 | 0.009126518 | 0.1028828 | 0.1544805 |
| Magnetic | 12 | 0.1256343 | 0.09752443 | 0.009170234 | 0.009461542 | 0.07299149 | 0.09701715 |
| Magnetic | 18 | 0.2807982 | 0.2078057 | 0.01192887 | 0.01354163 | 0.042482 | 0.06516489 |
| Magnetic | 23 | 1.101113 | 1.292284 | 0.005924179 | 0.004280349 | 0.005380172 | 0.003312235 |

## Selective response versus global variance inflation

Every horizon is included below. A constant u100/u20 would indicate pure multiplicative inflation. A horizon-dependent ratio shows selective adaptation, but a normalized response can also rise because u(h1) falls; absolute changes must be inspected alongside response ratios.

| h | Density u100/u20 | Magnetic u100/u20 | Density MSE100/MSE20 | Magnetic MSE100/MSE20 |
| --- | --- | --- | --- | --- |
| 1 | 0.51385 | 0.79215 | 0.99438 | 0.95055 |
| 2 | 0.50501 | 1.13472 | 0.91764 | 0.94044 |
| 3 | 0.56745 | 1.18185 | 0.89873 | 0.89813 |
| 4 | 0.56231 | 1.37541 | 0.87549 | 0.84949 |
| 5 | 0.56537 | 1.26047 | 0.83085 | 0.81367 |
| 6 | 0.56511 | 1.22027 | 0.83550 | 0.81269 |
| 7 | 0.55713 | 1.21209 | 0.79002 | 0.77926 |
| 8 | 0.56140 | 1.19018 | 0.79047 | 0.76350 |
| 9 | 0.53946 | 1.15111 | 0.76195 | 0.76001 |
| 10 | 0.52664 | 1.11776 | 0.78247 | 0.77547 |
| 11 | 0.49267 | 1.06841 | 0.77395 | 0.78506 |
| 12 | 0.48129 | 1.03177 | 0.76309 | 0.77626 |
| 13 | 0.47083 | 1.00046 | 0.77156 | 0.76759 |
| 14 | 0.45403 | 0.94538 | 0.74060 | 0.73853 |
| 15 | 0.44220 | 0.90741 | 0.68207 | 0.71727 |
| 16 | 0.44519 | 0.92742 | 0.63071 | 0.68954 |
| 17 | 0.46096 | 0.98722 | 0.66019 | 0.74927 |
| 18 | 0.45831 | 1.13520 | 0.65115 | 0.74005 |
| 19 | 0.42220 | 1.18711 | 0.68256 | 0.77208 |
| 20 | 0.40877 | 1.33444 | 0.80766 | 0.82531 |
| 21 | 0.40690 | 1.27262 | 0.91018 | 0.86272 |
| 22 | 0.38734 | 1.19856 | 1.05754 | 0.98478 |
| 23 | 0.35499 | 0.72252 | 1.27685 | 1.17362 |

## Ordinary reconstruction regression

The full training-validation metric uses the existing balanced reconstruction objective and full validation sampling. It is distinct from the paired one-window-per-run hidden-only metrics below; their absolute values should not be equated.

| Standard validation mask | MSE20 | MSE100 | Change |
| --- | --- | --- | --- |
| spatial_random | 0.00281280 | 0.00700042 | +148.88% |
| spatial_grid | 0.00474749 | 0.00896481 | +88.83% |
| spatial_block | 0.03607035 | 0.04187505 | +16.09% |
| temporal_random | 0.00342075 | 0.00560405 | +63.83% |
| temporal_block | 0.02201277 | 0.01846327 | -16.12% |

Paired hidden-only pass-0 metrics, pooled over all four channels, using exactly the archived five standard masks. Full per-channel and pass-1 metrics are also retained in `standard_mask_metrics.csv`.

| Role | Mask | MSE20 | MSE100 | Change |
| --- | --- | --- | --- | --- |
| val | spatial_random | 0.00807881 | 0.02225279 | +175.45% |
| val | spatial_grid | 0.00893009 | 0.02322937 | +160.12% |
| val | spatial_block | 0.10369786 | 0.12865231 | +24.06% |
| val | temporal_random | 0.02497919 | 0.03373228 | +35.04% |
| val | temporal_block | 0.11382188 | 0.08893828 | -21.86% |
| test | spatial_random | 0.00793572 | 0.02187002 | +175.59% |
| test | spatial_grid | 0.00878427 | 0.02287506 | +160.41% |
| test | spatial_block | 0.11049995 | 0.13678020 | +23.78% |
| test | temporal_random | 0.02437001 | 0.03275977 | +34.43% |
| test | temporal_block | 0.10560084 | 0.08159434 | -22.73% |

## Scientific conclusion

1. **100% exposure did not produce a broad increase in horizon-dependent uncertainty amplitude.** Magnetic u(h18)/u(h1) improves from 1.71x to 2.44x (about +43%), while Density falls from 2.01x to 1.79x (about -11%). At h23, Density falls from 2.09x to 1.45x and Magnetic from 0.85x to 0.77x. This is a selective Magnetic response, not a solution to the multimodal dynamic-range problem. Actual MSE h18/h1 also falls: Density 9.37x to 6.13x, Magnetic 9.82x to 7.64x. The new 2.44x Magnetic u response still substantially trails its 7.64x MSE growth. At h23 the new MSE ratios grow to 58.65x (Density) and 47.52x (Magnetic), despite the weaker u responses.

2. **Absolute calibration worsens overall, with a Magnetic improvement at moderate horizons.** At h18, Density u/MSE falls from approximately 0.1267 to 0.0892, while Magnetic rises from approximately 0.0425 to 0.0652. At h23 both worsen: Density 0.0271 to 0.00753, Magnetic 0.00538 to 0.00331. All remain far below one. Density u100/u20 is below one at every horizon: 0.514 at h1, 0.458 at h18, 0.355 at h23. Magnetic changes are selective: 0.792 at h1, 1.135 at h18, 0.723 at h23. The Magnetic h18 response increase therefore combines a smaller h1 denominator (-20.8% u) with an absolute h18 increase (+13.5% u), alongside a 26.0% MSE reduction; it should not be read as a 43% absolute variance increase.

3. **Magnetic uncertainty still collapses at h23, more severely relative to h1.** At h22 the new mean u is 0.01242; at h23 it falls to 0.004280 (about -65.5%), while MSE rises from 0.7097 to 1.2923 (about +82.1%). Its u/MSE at h23 corresponds to underestimating observed error variance by approximately 302x.

4. **Forecasting improves at moderate horizons but ordinary reconstruction degrades materially.** At h18, MSE falls by 34.9% for Density and 26.0% for Magnetic. At h23 it instead rises by 27.7% and 17.4%. Full standard validation MSE rises 18.6%, from 0.01381283 to 0.01638152; validation NLL worsens from -1.28048 to -0.81795. Spatial-random, spatial-grid, spatial-block, and temporal-random validation MSE all worsen; temporal-block improves. This is a material general-reconstruction cost and is not a suitable production replacement for the 20% checkpoint on these criteria.

5. **Insufficient forecast exposure is not supported as the primary remaining explanation.** Exposure does affect reconstruction and the moderate-horizon Magnetic response, so task sampling matters in part. However, fivefold forecasting exposure over ten more epochs does not resolve Density compression or the extreme-horizon collapse. This bounded diagnostic does not rule out longer adaptation, and there is no matched additional-10-epoch 20%-exposure control to separate all effects of training duration from sampling. These are paired point estimates, not significance tests. The mechanism causing the variance/error mismatch cannot be identified uniquely by this experiment.

Recommended next action: retain the 20% checkpoint for general reconstruction. Before changing training, inspect frozen-model uncertainty-head outputs and conditioning features across the h22-to-h23 boundary, and compare squared residuals with predicted variance by modality and horizon. No new loss or training change is established by this audit.

## Artifacts and verification

![Forecast MSE and predicted variance](baseline_vs_100pct_horizon.png)

![Calibration and same-horizon model ratios](baseline_vs_100pct_calibration.png)

`forecast_horizon_metrics.csv` contains both models, all 23 horizons and six channel groups, actual MSE, raw mean u, calibration, relative-to-h1 ratios, and explicit u100/u20 and MSE100/MSE20 at every horizon. `standard_mask_metrics.csv` contains both roles, both passes, all five standard masks, and all six channel groups. `training_audit.json` and `evaluation_manifest.json` preserve audit details and evaluation provenance. Primary metrics are finite and paired voxel/window counts match the archive. Baseline CSV SHA256 hashes remain unchanged. Inference ran as a step in the user-owned existing allocation 61481634; no job was submitted. Training code, checkpoints, and prior evaluation results were not modified.

**100% Forecast Diagnostic Audit**

- Training completed: YES
- Density u(h18)/u(h1): 20% = 2.01x; 100% = 1.79x
- Magnetic u(h18)/u(h1): 20% = 1.71x; 100% = 2.44x
- Density u(h23)/u(h1): 20% = 2.09x; 100% = 1.45x
- Magnetic u(h23)/u(h1): 20% = 0.85x; 100% = 0.77x
- Difficulty-response amplitude: WORSE overall; moderate-horizon Magnetic improves
- Absolute calibration: WORSE overall; Magnetic h18 improves
- Ordinary reconstruction: WORSE
- Forecast-exposure hypothesis: NOT SUPPORTED as the primary remaining explanation
- Most important remaining bottleneck: predicted variance fails to track actual error, especially near-total temporal occlusion; mechanism unresolved
- Recommended next action: retain the 20% checkpoint and investigate frozen-model h22-to-h23 uncertainty response before another training change
