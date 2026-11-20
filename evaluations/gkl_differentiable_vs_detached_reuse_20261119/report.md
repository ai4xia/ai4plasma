# GKL differentiable versus detached prediction reuse

**Recommendation. Test mean-only differentiable reuse.** The joint change reduced reconstruction error under temporal-block and advancing-window reuse. At late iterations, predicted uncertainty substantially underestimated squared error for both checkpoints, and the joint change weakened spatial ranking in several conditions. The matched comparison estimates the combined effect of enabling both reuse gradients.

## Training and protocol

Both continuations completed epochs 257–276 from the same epoch-256 source with restored AdamW, constant LR `1e-5`, settled Phase C, GKL A+B, and the same optimizer-step alpha schedule. The saved configurations differed only in the reuse flag and output or logging fields. Both reached 295,044 cumulative optimizer steps and logged zero nonfinite events and AMP skips. Final train MSE was 0.0138383 detached and 0.0138377 differentiable. Final validation MSE was 0.0168817 and 0.0169136. See [training audit](training_audit.json) and [epoch history](training_history.csv).

The pass-0 benchmark used the original 125-window splits, masks, and seeds and reused the cached detached aggregates. Paired estimates use 24 independent held-out runs selected across beta strata, with 10,000 paired bootstrap resamples of runs. Differences are differentiable minus detached. The fixed-window test scored passes 0, 1, 2, 4, and 8 along each uninterrupted rollout. The advancing test used 24-frame windows, stride 4, and 32 probe sites at frames 0, 8, 16, 24, and 32. Both models used the same initial noise and uncertainty, copied predicted means and uncertainties into overlapping hidden positions, and used genuine probe values with zero input uncertainty. Observation age is measured from the latest genuine probe time. See the [protocol](protocol.json).

## Pass 0

Five-mask aggregate MSE changed from 0.07306 to 0.07332 on validation and from 0.07199 to 0.07232 on test. On the paired cohort the Magnetic MSE difference was +0.00061 (95% CI −0.00007 to +0.00142) and the Density difference was −0.00004 (−0.00026 to +0.00013). The Density mean run-level calibration ratio rose from 1.034 to 1.115, a difference of +0.080 (0.062 to 0.100).

| Horizon | Magnetic u/MSE | Density u/MSE | Magnetic spatial Spearman | Density spatial Spearman |
| --- | ---: | ---: | ---: | ---: |
| 18 | 0.962 → 0.949 | 1.152 → 1.220 | 0.620 → 0.613 | 0.439 → 0.426 |
| 22 | 0.697 → 0.675 | 0.876 → 0.954 | 0.604 → 0.596 | 0.454 → 0.434 |
| 23 | 0.487 → 0.517 | 0.632 → 0.706 | 0.605 → 0.596 | 0.557 → 0.545 |

At h23, the paired MSE differences were −0.01568 (−0.03622 to +0.00484) for Magnetic and −0.01498 (−0.02681 to −0.00309) for Density. Across frames hidden at both h22 and h23, Magnetic predicted uncertainty retained 85.8% of its h22 value versus 79.1% under detached reuse, while MSE rose 11.9% versus 13.1%. Magnetic window collapse fell from 35.2% to 29.6%. Density window collapse rose from 19.2% to 21.6%. See [horizon metrics](forecast_horizon_metrics.csv), [collapse statistics](extreme_horizon.json), and [paired intervals](pass0_paired_summary.csv).

## Reuse and rollout

| Evaluation point | Magnetic paired MSE difference (95% CI) | Density paired MSE difference (95% CI) |
| --- | ---: | ---: |
| Fixed temporal block, pass 1 | −0.00289 (−0.00509 to −0.00102) | −0.00101 (−0.00178 to −0.00036) |
| Fixed temporal block, pass 8 | −0.01484 (−0.02629 to −0.00526) | −0.00571 (−0.01069 to −0.00198) |
| Sliding window, final update | −0.01638 (−0.02666 to −0.00568) | −0.01964 (−0.03785 to −0.00471) |
| Sliding window, 15 frames since probes | −0.01704 (−0.02855 to −0.00552) | −0.02231 (−0.04156 to −0.00632) |

On the fixed spatial grid, both models improved with iteration. Differentiable reuse had higher Density MSE at pass 1 (+0.00022, 0.00016 to 0.00029), with little separation in endpoint MSE at pass 8. At temporal-block pass 8, differentiable reuse lowered absolute binned calibration error alongside MSE, while the uncertainty-to-MSE ratio fell from 0.319 to 0.276 for Magnetic and from 0.359 to 0.325 for Density. Spatial Spearman fell by 0.018 (0.011 to 0.025) and 0.021 (0.006 to 0.036), respectively. See [fixed-window metrics](repeated_reuse_summary.csv).

From the first to last advancing window, the change in MSE favored differentiable reuse by −0.01673 (−0.02740 to −0.00564) for Magnetic and −0.01841 (−0.03668 to −0.00429) for Density. Density MSE still increased from about 0.36 to 0.64. The final run-average u/MSE ratios were 0.139 → 0.133 for Magnetic and 0.032 → 0.035 for Density. Differentiable spatial Spearman was 0.267 for Magnetic and −0.157 for Density. At an observation age of 15 frames Density Spearman was approximately −0.197 for both. The rollout results give weak support for interpreting late predicted uncertainty as conditional expected squared error. See [window-step and observation-age metrics](sliding_summary.csv) and [first-to-last changes](sliding_change_summary.csv).
