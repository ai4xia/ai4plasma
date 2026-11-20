# GKL reuse gradient ablation

**Recommendation. Use mean-only differentiable reuse for fresh 256-epoch training, with GKL Paths A and B enabled.** Mean-only recovered most of the reconstruction benefit of fully differentiable reuse in the matched continuation, with better Density spatial ranking during the advancing rollout and better Magnetic ranking on repeated temporal blocks. The severe sliding-window underestimation persisted in all three models. This recommendation extrapolates from continuations sharing one training source and seed.

The saved mean-only history covers epochs 257–276 with BF16, constant LR `1e-5`, settled Phase C, and two forwards. It records 295,044 cumulative successful optimizer steps and zero nonfinite events or AMP skips across the continuation. Final training MSE was 0.01383369 and final validation MSE was 0.01680755. The three saved run configurations differ in routing and output or logging fields. All name the same epoch-256 source and GKL A+B continuation.

| Reuse gradients | Epoch-276 validation MSE |
| --- | ---: |
| Detached | 0.01688173 |
| Mean only | 0.01680755 |
| Mean and uncertainty | 0.01691357 |

The checkpoint tensor audit passed for model and AdamW state. Inference used one GPU on authorized ai4plasma allocation `61632257` and evaluated only mean-only. It reused both controls' cached metrics, the archived 125-window validation and test splits, and the same 24 independent held-out runs for rollout and paired intervals. The advancing rollout used 24-frame windows, four-frame advances, and the original sparse probes with genuine values and zero input uncertainty. [Audit](training_audit.json) · [Protocol](protocol.json)

## Pass 0

Five-mask test MSE averaged 0.07199, 0.07226, and 0.07232 for detached, mean-only, and fully differentiable, respectively. Paired mean-only minus detached MSE intervals included zero for both Magnetic and Density. On the paired Density standard masks, mean run-level u/MSE was 1.034, 1.041, and 1.115. Enabling uncertainty reuse gradients in addition to mean gradients raised that ratio by 0.073 [0.055, 0.092], moving away from one.

| h23 metric | Detached | Mean-only | Fully differentiable |
| --- | ---: | ---: | ---: |
| Magnetic MSE | 1.1168 | 1.1081 | 1.0991 |
| Magnetic mean u | 0.5441 | 0.5488 | 0.5682 |
| Magnetic u/MSE | 0.487 | 0.495 | 0.517 |
| Magnetic spatial Spearman | 0.605 | 0.601 | 0.596 |
| Density MSE | 0.4484 | 0.4372 | 0.4316 |
| Density mean u | 0.2832 | 0.2760 | 0.3045 |
| Density u/MSE | 0.632 | 0.631 | 0.706 |
| Density spatial Spearman | 0.557 | 0.561 | 0.545 |

On frames common to h22 and h23, Magnetic uncertainty fell 20.9%, 18.5%, and 14.2%, while MSE rose 13.1%, 12.6%, and 11.9%. Magnetic window collapse frequency was 35.2%, 32.0%, and 29.6%. Density shared-frame uncertainty rose 7.7%, 9.8%, and 6.7%, while MSE rose 47.0%, 45.2%, and 41.8%. Density collapse frequency was 19.2%, 20.0%, and 21.6%. [Forecast curves](forecast_rollout.png) · [Paired pass-0 intervals](pass0_paired_summary.csv) · [Shared-frame metrics](extreme_horizon.json)

## Fixed-window and advancing rollout

Paired differences bracketed below give 95% percentile bootstrap intervals over independent runs. The first contrast isolates enabling mean reuse gradients with detached uncertainty reuse. The second isolates enabling uncertainty reuse gradients with differentiable mean reuse.

| MSE endpoint | Detached | Mean-only | Fully differentiable | Mean-only − detached | Fully differentiable − mean-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Temporal block pass 8, Magnetic | 0.13664 | 0.12281 | 0.12180 | −0.01383 [−0.02442, −0.00564] | −0.00101 [−0.00738, 0.00571] |
| Temporal block pass 8, Density | 0.03955 | 0.03665 | 0.03383 | −0.00290 [−0.00625, 0.00016] | −0.00282 [−0.00863, 0.00046] |
| Sliding update 6, Magnetic | 0.58696 | 0.56980 | 0.57058 | −0.01717 [−0.02693, −0.00862] | 0.00079 [−0.00697, 0.00869] |
| Sliding update 6, Density | 0.65572 | 0.64382 | 0.63607 | −0.01190 [−0.02135, −0.00280] | −0.00775 [−0.02272, 0.00239] |

Mean-only lowered temporal-block pass-1 MSE versus detached for both fields. On the fixed spatial grid at pass 8, mean-only Magnetic MSE was effectively tied with detached and Density MSE was slightly higher. At temporal-block pass 8, adding uncertainty reuse gradients lowered Magnetic calibration by 0.030 [−0.055, −0.011] and spatial Spearman by 0.0098 [−0.0158, −0.0044]. [Fixed spatial grid](spatial_grid_rollout.png) · [Fixed temporal block](temporal_block_rollout.png) · [Paired fixed-window intervals](repeated_reuse_summary.csv)

Mean-only uncertainty fell from 0.802 to 0.187 for Magnetic and from 0.312 to 0.042 for Density at the first window advance, similar to both controls. From updates 3 to 6, mean-only Magnetic MSE rose from 0.442 to 0.570 while uncertainty moved from 0.063 to 0.067. Density MSE rose from 0.456 to 0.644 while uncertainty moved from 0.0111 to 0.0119. The first drop and subsequent failure to track error remain.

At the final update, mean run-level u/MSE was 0.139, 0.132, and 0.133 for Magnetic and 0.032, 0.031, and 0.035 for Density. Enabling uncertainty reuse gradients raised Density calibration by 0.0035 [0.0016, 0.0057] and lowered Density spatial Spearman by 0.030 [−0.051, −0.009]. Density ranking remained negative in all three, with mean-only least negative at −0.127. At 15 frames since the last genuine probe, mean-only MSE remained lower than detached for both fields and Density spatial Spearman was −0.160 versus −0.197 for both controls. [Sliding-window curves](step_rollout.png) · [Observation-age curves](age_rollout.png) · [Paired sliding intervals](sliding_summary.csv) · [Transitions](sliding_transition_summary.csv)

Binned calibration error is descriptive. The archived controls used pooled control deciles. Their cache lacks voxel predictions and shared bin edges, so mean-only used its own deciles with the same sampling and absolute-gap formula. All other paired measures use the archived definitions.
