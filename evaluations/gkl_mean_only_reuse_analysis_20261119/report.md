# Rollout uncertainty before the mean-only reuse ablation

The saved results cover the detached and fully differentiable checkpoints on 24 paired held-out runs. The sliding windows advance by four frames. The fixed-window tests feed predictions back on the same window. Each plot shows actual MSE, mean predicted uncertainty, mean run-level (u/\mathrm{MSE}), and within-frame spatial Spearman. This analysis used the existing CSVs and ran no inference.

[Sliding windows](step_rollout.png) · [Fixed spatial grid](spatial_grid_rollout.png) · [Fixed temporal block](temporal_block_rollout.png)

## Sliding windows

**Underestimation begins at the first advance in both fields and both checkpoints.** Magnetic uncertainty fell 76–77% from update 0 to 1, while MSE fell 24–25%. Density uncertainty fell 86–87% while MSE rose about 10–12%. At update 1, mean run-level (u/\mathrm{MSE}) had already fallen from 1.56–1.62 to 0.45–0.48 for Magnetic and from 1.32 to 0.20–0.21 for Density.

After update 3, Magnetic and Density uncertainty stayed close to their low levels while MSE rebounded. From update 3 to 6, Magnetic MSE rose 31–32% and Density MSE rose 41–43%. Mean uncertainty rose only 3–8% in Magnetic and 7–9% in Density. The initial deterioration therefore comes from a sharp drop in predicted uncertainty. Later, the small recovery in uncertainty fails to keep pace with rising error.

Values below are detached → fully differentiable.

| Field and update | Actual MSE | Mean predicted u | Mean run-level u/MSE | Spatial Spearman |
| --- | ---: | ---: | ---: | ---: |
| Magnetic 0 | 0.594 → 0.595 | 0.813 → 0.782 | 1.618 → 1.563 | 0.493 → 0.481 |
| Magnetic 1 | 0.452 → 0.446 | 0.199 → 0.180 | 0.484 → 0.448 | 0.410 → 0.398 |
| Magnetic 3 | 0.449 → 0.433 | 0.067 → 0.066 | 0.167 → 0.172 | 0.217 → 0.208 |
| Magnetic 6 | 0.587 → 0.571 | 0.072 → 0.068 | 0.139 → 0.133 | 0.264 → 0.267 |
| Density 0 | 0.362 → 0.361 | 0.326 → 0.321 | 1.316 → 1.323 | 0.543 → 0.527 |
| Density 1 | 0.406 → 0.396 | 0.045 → 0.043 | 0.212 → 0.203 | 0.463 → 0.375 |
| Density 3 | 0.458 → 0.453 | 0.012 → 0.013 | 0.087 → 0.071 | 0.086 → −0.002 |
| Density 6 | 0.656 → 0.636 | 0.012 → 0.014 | 0.032 → 0.035 | −0.165 → −0.157 |

Magnetic ranking reached its minimum at update 3 and recovered modestly. Density ranking reached approximately zero by update 3 for fully differentiable reuse and crossed zero at update 4 for detached reuse.

## Fixed windows

Calibration also deteriorated at pass 1 on the fixed spatial grid. Both fields' MSE improved throughout the sampled passes, while uncertainty fell faster. On the fixed temporal block, Density uncertainty underestimated error by pass 1. Magnetic uncertainty was near calibrated at pass 1 and clearly underestimated error by pass 2. From pass 4 to 8, uncertainty kept falling as both fields' MSE rose. The two existing checkpoints show the same qualitative deterioration. The mean-only continuation isolates the effect of permitting the cross-forward uncertainty gradient conditional on differentiable mean reuse and GKL A+B.
