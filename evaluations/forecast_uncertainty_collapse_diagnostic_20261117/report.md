# Forecast uncertainty collapse diagnostic — epoch 276

## Population, metrics, and reproducibility

Primary model: epoch-276 Student-t df=3, 20% extrapolation continuation. Reused the archived evaluator for model loading, normalization, deterministic window selection, batching, inference, fixed masks, and Phase-C reuse. The exact same 125 held-out test runs each contribute their first complete 24-frame window. Split hash: `0e6b4276bc7f40003eed23c3e952025a3e6ef0d47619b1fbe2a8939d091f4b99`. The manifest lists every run, index, and window start. All four channels have the same observed prefix and hidden suffix in pass 0. Unknown Gaussian inputs use the archived seed 201234 + batch number, batch size 2, and the same noise at every horizon. No epoch-286 or optional epoch-256 model was used.

MSE and mean raw predicted variance are voxel-weighted in checkpoint-standardized squared units; Magnetic pools Bx/By/Bz. Calibration is mean(u)/MSE. Frame 0 is the first local frame; forecast lead = local frame − last observed frame. The manifest records physical window starts. All primary metrics are finite. Re-aggregating the per-frame pass-0 results reproduces archived horizon MSE/u, with maximum relative difference 6.66e-16. Archived CSV hash is unchanged.

## Per-frame growth within each fixed horizon

Last/first ratios compare the last hidden frame with the first hidden frame under the same horizon, not h23 with h1. A single h1 frame has no measurable within-horizon slope.

| Channel | h | MSE last/first | u last/first | Spearman(lead,u) | u decreases | min u/MSE | max u/MSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bx | 1 | 1.000 | 1.000 | n/a | 0 | 0.3418 | 0.3418 |
| Bx | 6 | 7.573 | 0.813 | -0.600 | 3 | 0.05081 | 0.4735 |
| Bx | 12 | 11.694 | 0.559 | -0.923 | 7 | 0.02742 | 0.5736 |
| Bx | 18 | 13.249 | 0.420 | -0.942 | 10 | 0.008189 | 0.2581 |
| Bx | 21 | 30.463 | 0.903 | -0.739 | 12 | 0.005165 | 0.1743 |
| Bx | 22 | 42.569 | 0.944 | -0.573 | 13 | 0.004673 | 0.2108 |
| Bx | 23 | 54.185 | 0.742 | -0.550 | 14 | 0.003686 | 0.2693 |
| By | 1 | 1.000 | 1.000 | n/a | 0 | 0.2218 | 0.2218 |
| By | 6 | 4.472 | 0.992 | -0.429 | 2 | 0.06608 | 0.298 |
| By | 12 | 6.475 | 0.527 | -0.888 | 7 | 0.03133 | 0.4656 |
| By | 18 | 3.289 | 0.291 | -0.957 | 11 | 0.01407 | 0.2194 |
| By | 21 | 2.006 | 0.869 | -0.849 | 12 | 0.0109 | 0.06573 |
| By | 22 | 1.889 | 1.305 | -0.565 | 10 | 0.008107 | 0.02937 |
| By | 23 | 10.148 | 0.956 | -0.307 | 13 | 0.002806 | 0.1016 |
| Bz | 1 | 1.000 | 1.000 | n/a | 0 | 0.444 | 0.444 |
| Bz | 6 | 4.700 | 0.773 | -0.943 | 4 | 0.09859 | 0.5997 |
| Bz | 12 | 7.664 | 0.643 | -0.972 | 9 | 0.04745 | 0.566 |
| Bz | 18 | 12.717 | 0.541 | -0.953 | 9 | 0.01542 | 0.3628 |
| Bz | 21 | 12.054 | 0.850 | -0.722 | 15 | 0.009342 | 0.1325 |
| Bz | 22 | 5.981 | 0.839 | -0.584 | 13 | 0.006711 | 0.04785 |
| Bz | 23 | 6.095 | 0.721 | -0.687 | 13 | 0.003033 | 0.02608 |
| Density | 1 | 1.000 | 1.000 | n/a | 0 | 0.5909 | 0.5909 |
| Density | 6 | 5.645 | 1.366 | 0.943 | 1 | 0.2346 | 0.9694 |
| Density | 12 | 9.575 | 1.157 | 0.273 | 4 | 0.124 | 1.026 |
| Density | 18 | 14.856 | 0.489 | -0.913 | 12 | 0.04692 | 1.424 |
| Density | 21 | 13.920 | 0.197 | -0.938 | 14 | 0.02307 | 1.629 |
| Density | 22 | 4.769 | 0.277 | -0.933 | 14 | 0.021 | 0.3615 |
| Density | 23 | 1.474 | 0.183 | -0.910 | 16 | 0.01262 | 0.1016 |
| Magnetic | 1 | 1.000 | 1.000 | n/a | 0 | 0.2445 | 0.2445 |
| Magnetic | 6 | 4.796 | 0.950 | -0.429 | 2 | 0.06492 | 0.3279 |
| Magnetic | 12 | 7.066 | 0.537 | -0.923 | 7 | 0.03338 | 0.4811 |
| Magnetic | 18 | 4.381 | 0.314 | -0.965 | 11 | 0.01327 | 0.2263 |
| Magnetic | 21 | 3.077 | 0.872 | -0.849 | 12 | 0.00965 | 0.06931 |
| Magnetic | 22 | 2.850 | 1.197 | -0.593 | 10 | 0.007388 | 0.02751 |
| Magnetic | 23 | 11.278 | 0.899 | -0.332 | 12 | 0.003328 | 0.08746 |

![Per-frame MSE and uncertainty by forecast lead](per_frame_mse_vs_u.png)

## Critical comparison: common physical frames 2–23

h22 observes frames 0–1; h23 observes only frame 0. These comparisons exclude newly hidden frame 1 and match every target window and physical frame. The last column counts common frames whose aggregate u decreases (out of 22). The paired CSV also reports median window ratios and the fraction of windows with a decrease at every frame.

| Channel | MSE22 | MSE23 | MSE23/22 | u22 | u23 | u23/22 | C22 | C23 | frames u drops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bx | 0.319171 | 0.511606 | 1.603 | 0.00396832 | 0.00286534 | 0.722 | 0.01243 | 0.00560 | 22 |
| By | 1.70847 | 2.54708 | 1.491 | 0.0252078 | 0.0132187 | 0.524 | 0.01475 | 0.00519 | 22 |
| Bz | 0.134467 | 0.384483 | 2.859 | 0.00190093 | 0.00159386 | 0.838 | 0.01414 | 0.00415 | 20 |
| Density | 0.218913 | 0.474936 | 2.170 | 0.0154406 | 0.0117501 | 0.761 | 0.07053 | 0.02474 | 22 |
| Magnetic | 0.720704 | 1.14772 | 1.593 | 0.010359 | 0.00589265 | 0.569 | 0.01437 | 0.00513 | 22 |

Frame 1, hidden only under h23:

| Channel | MSE frame1 | u frame1 | C frame1 | full h23 u / common-frame u |
| --- | --- | --- | --- | --- |
| Bx | 0.0122415 | 0.00329689 | 0.26932 | 1.0065 |
| By | 0.145047 | 0.0147388 | 0.10161 | 1.0050 |
| Bz | 0.0697075 | 0.00181786 | 0.02608 | 1.0061 |
| Density | 0.326656 | 0.0331931 | 0.10161 | 1.0793 |
| Magnetic | 0.0756654 | 0.00661787 | 0.08746 | 1.0054 |

![Paired h22 and h23 at matching local frames](h22_vs_h23.png)

## Spatial uncertainty–error correspondence

Within-window Spearman uses all hidden space-time voxels. To isolate spatial ranking from temporal growth, within-frame Spearman ranks spatial voxels separately in each run/frame, then averages across frames and runs. Magnetic pools component/voxel pairs and therefore can include between-component scale effects; Bx/By/Bz are reported separately. Pooled Spearman uses a deterministic sample of 4096 voxels per run/channel group (512,000 total), and can include between-run difficulty. Exact empirical deciles use average ranks so tied u values stay together. The top/bottom error ratio below aggregates exact within-frame deciles; a ratio above one demonstrates spatial discrimination, not calibrated amplitude. All ten deciles for pooled, within-window, and within-frame ranking are in the spatial CSV.

| Channel | h | pooled rho | median window rho | mean frame spatial rho | top/bottom frame-decile MSE | top-decile u/MSE |
| --- | --- | --- | --- | --- | --- | --- |
| Bx | 18 | 0.095 | 0.104 | 0.116 | 1.90 | 0.0483 |
| By | 18 | -0.009 | -0.032 | -0.038 | 0.67 | 0.2862 |
| Bz | 18 | 0.034 | -0.008 | 0.088 | 0.88 | 0.0983 |
| Density | 18 | 0.292 | 0.294 | 0.296 | 3.54 | 0.3286 |
| Magnetic | 18 | 0.165 | 0.225 | 0.191 | 1.89 | 0.1433 |
| Bx | 22 | 0.131 | 0.155 | 0.136 | 1.83 | 0.0259 |
| By | 22 | 0.014 | 0.040 | 0.003 | 0.83 | 0.0890 |
| Bz | 22 | 0.072 | 0.103 | 0.061 | 0.93 | 0.0357 |
| Density | 22 | 0.407 | 0.374 | 0.383 | 6.83 | 0.1698 |
| Magnetic | 22 | 0.126 | 0.219 | 0.151 | 2.15 | 0.0448 |
| Bx | 23 | 0.150 | 0.162 | 0.159 | 1.34 | 0.0092 |
| By | 23 | 0.008 | 0.034 | 0.005 | 0.77 | 0.0238 |
| Bz | 23 | 0.142 | 0.157 | 0.138 | 1.06 | 0.0079 |
| Density | 23 | 0.467 | 0.502 | 0.511 | 11.90 | 0.0495 |
| Magnetic | 23 | 0.120 | 0.176 | 0.147 | 2.14 | 0.0136 |

## Phase-C pass 0 versus continuous-u pass 1

The established fixed-validation convention pairs temporal_block with spatial_random. We use that fresh spatial_random layout for every source horizon, via `rollout_validation_reuse_pattern` and `make_fixed_validation_mask`. `make_phase_c_reuse_input` places exact targets/u=0 at the fresh mask and detached mu0/raw-u0 elsewhere. Previous prefix observations are not automatically retained. Thus pass 1 is a spatial reconstruction/refinement condition, not a pure forecasting condition with the original prefix. Its effective observed fractions per run/channel/frame are recorded in `pass1_effective_masks.csv`. The all-frame per-pass CSV reports these fractions; its lead column explicitly refers to the source pass-0 prefix.

A matched binary control performs a first forward with this identical fresh spatial mask, targets/u=0 at observed locations, and Gaussian noise/u=1 elsewhere. Pass 1 and this control share exact observations and scored voxels, but pass 1 additionally carries prior prediction information. This measures their combined information/refinement effect, not a causal effect of changing u alone. The table scores only original future-frame positions for each source horizon; pass-0 calibration is contextual and is not an equal-information comparison.

| source h | Channel | C pass0 | MSE pass1 | MSE control | MSE1/control | u pass1 | u control | C pass1 | C control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 18 | Bx | 0.0249 | 0.0292728 | 0.0045102 | 6.490 | 0.0028984 | 0.00282076 | 0.0990 | 0.6254 |
| 18 | By | 0.0473 | 0.0941559 | 0.0175671 | 5.360 | 0.0155611 | 0.0154983 | 0.1653 | 0.8822 |
| 18 | Bz | 0.0440 | 0.00591771 | 0.00209629 | 2.823 | 0.0016973 | 0.00168189 | 0.2868 | 0.8023 |
| 18 | Density | 0.1267 | 0.0215455 | 0.00352729 | 6.108 | 0.00306128 | 0.0028472 | 0.1421 | 0.8072 |
| 18 | Magnetic | 0.0425 | 0.0431155 | 0.00805786 | 5.351 | 0.00671892 | 0.00666699 | 0.1558 | 0.8274 |
| 22 | Bx | 0.0124 | 0.0492128 | 0.00480492 | 10.242 | 0.00329192 | 0.00337631 | 0.0669 | 0.7027 |
| 22 | By | 0.0148 | 0.191299 | 0.0213099 | 8.977 | 0.0187114 | 0.0198537 | 0.0978 | 0.9317 |
| 22 | Bz | 0.0141 | 0.0148146 | 0.0022736 | 6.516 | 0.00174861 | 0.00184819 | 0.1180 | 0.8129 |
| 22 | Density | 0.0705 | 0.0365512 | 0.00371751 | 9.832 | 0.00483533 | 0.00322835 | 0.1323 | 0.8684 |
| 22 | Magnetic | 0.0144 | 0.0851086 | 0.00946279 | 8.994 | 0.0079173 | 0.00835941 | 0.0930 | 0.8834 |
| 23 | Bx | 0.0059 | 0.0491971 | 0.00482571 | 10.195 | 0.00324936 | 0.00345651 | 0.0660 | 0.7163 |
| 23 | By | 0.0054 | 0.279205 | 0.0211427 | 13.206 | 0.0189784 | 0.0203601 | 0.0680 | 0.9630 |
| 23 | Bz | 0.0043 | 0.0247943 | 0.00229818 | 10.789 | 0.00173417 | 0.00187765 | 0.0699 | 0.8170 |
| 23 | Density | 0.0271 | 0.0388095 | 0.00386146 | 10.050 | 0.0108176 | 0.00348025 | 0.2787 | 0.9013 |
| 23 | Magnetic | 0.0054 | 0.117732 | 0.00942218 | 12.495 | 0.0079873 | 0.00856475 | 0.0678 | 0.9090 |

Selected per-frame pass-1/control values (all frames and channels are in the pass CSV):

| source h | Channel | local frame | MSE pass1 | u pass1 | C pass1 | MSE control | C control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 18 | Magnetic | 6 | 0.02821 | 0.016822 | 0.5963 | 0.014497 | 0.9669 |
| 18 | Magnetic | 12 | 0.032679 | 0.006558 | 0.2007 | 0.007505 | 0.9352 |
| 18 | Magnetic | 23 | 0.067734 | 0.0055573 | 0.0820 | 0.0082843 | 0.3995 |
| 18 | Density | 6 | 0.0063057 | 0.004529 | 0.7182 | 0.0038244 | 1.0550 |
| 18 | Density | 12 | 0.015287 | 0.0028472 | 0.1863 | 0.0030812 | 0.9108 |
| 18 | Density | 23 | 0.036625 | 0.0040657 | 0.1110 | 0.0057695 | 0.4539 |
| 22 | Magnetic | 2 | 0.049773 | 0.015192 | 0.3052 | 0.01468 | 0.8920 |
| 22 | Magnetic | 12 | 0.078052 | 0.0063196 | 0.0810 | 0.007505 | 0.9352 |
| 22 | Magnetic | 23 | 0.098988 | 0.0059163 | 0.0598 | 0.0082843 | 0.3995 |
| 22 | Density | 2 | 0.018768 | 0.0068021 | 0.3624 | 0.0056918 | 1.0064 |
| 22 | Density | 12 | 0.029183 | 0.0055935 | 0.1917 | 0.0030812 | 0.9108 |
| 22 | Density | 23 | 0.058991 | 0.0050761 | 0.0860 | 0.0057695 | 0.4539 |
| 23 | Magnetic | 1 | 0.018385 | 0.012774 | 0.6948 | 0.0085288 | 1.5339 |
| 23 | Magnetic | 12 | 0.11167 | 0.0062778 | 0.0562 | 0.007505 | 0.9352 |
| 23 | Magnetic | 23 | 0.11723 | 0.0073872 | 0.0630 | 0.0082843 | 0.3995 |
| 23 | Density | 1 | 0.021397 | 0.012149 | 0.5678 | 0.0070283 | 1.2837 |
| 23 | Density | 12 | 0.036295 | 0.016224 | 0.4470 | 0.0030812 | 0.9108 |
| 23 | Density | 23 | 0.058289 | 0.0071555 | 0.1228 | 0.0057695 | 0.4539 |

## h23 calibration by within-frame predicted-u decile

Each decile is assigned within a single run and physical frame, then aggregated over all 125 runs and hidden frames. This avoids obtaining discrimination merely by ranking easier runs or earlier frames below harder ones.

| Channel | decile | MSE | mean u | u/MSE |
| --- | --- | --- | --- | --- |
| Magnetic | 1 | 1.07829 | 0.000697772 | 0.00065 |
| Magnetic | 2 | 0.60015 | 0.00105206 | 0.00175 |
| Magnetic | 3 | 0.523869 | 0.00131005 | 0.00250 |
| Magnetic | 4 | 0.544571 | 0.00161352 | 0.00296 |
| Magnetic | 5 | 0.656619 | 0.00203098 | 0.00309 |
| Magnetic | 6 | 0.803159 | 0.00264374 | 0.00329 |
| Magnetic | 7 | 1.08637 | 0.00365997 | 0.00337 |
| Magnetic | 8 | 1.52948 | 0.00558143 | 0.00365 |
| Magnetic | 9 | 1.87807 | 0.00915092 | 0.00487 |
| Magnetic | 10 | 2.3107 | 0.0315081 | 0.01364 |
| Density | 1 | 0.106896 | 0.000766762 | 0.00717 |
| Density | 2 | 0.131517 | 0.00114756 | 0.00873 |
| Density | 3 | 0.175546 | 0.00162357 | 0.00925 |
| Density | 4 | 0.231494 | 0.00256672 | 0.01109 |
| Density | 5 | 0.326355 | 0.00436253 | 0.01337 |
| Density | 6 | 0.433405 | 0.00680471 | 0.01570 |
| Density | 7 | 0.529915 | 0.00995961 | 0.01879 |
| Density | 8 | 0.648493 | 0.0144508 | 0.02228 |
| Density | 9 | 0.829067 | 0.0221973 | 0.02677 |
| Density | 10 | 1.27243 | 0.0629562 | 0.04948 |

## Answers to the scientific questions

1. **Does u increase with lead within a fixed horizon? Usually no for Magnetic; Density is horizon dependent.** At h18, Magnetic MSE rises from 0.094805 at lead 1 to 0.415320 at lead 18 (4.38x), while u falls from 0.021450 to 0.006733 (0.314x). Spearman(lead,u) is -0.965. Density at h18 has MSE growth 14.86x but u declines to 0.489x its first-hidden-frame value (lead/u rho -0.913). At h6, Density u does rise (1.366x), but much less than MSE (5.645x). At h23, Magnetic u is nonmonotonic (lead/u rho -0.332), with last/first u=0.899 versus MSE=11.278; Density has last/first u=0.183 versus MSE=1.474. The per-frame problem therefore includes incorrect temporal shape, not just insufficient aggregate scale.

2. **Both a broad lead-time failure and an abrupt h22-to-h23 transition are present.** On common frames 2–23, Magnetic u falls 43.1% (0.010359 to 0.005893), while MSE rises 59.3% (0.720704 to 1.147724). Calibration falls from 0.01437 to 0.00513. Aggregate u drops at all 22 common frames, and in 85.4% of paired run/frame means. Density common-frame u falls 23.9% while MSE rises 117.0%; its u drops at all 22 common frames and in 72.8% of run/frame pairs. By has the largest relative u suppression (u23/u22=0.524), compared with Bx=0.722 and Bz=0.838. The disappearance of the second exact observed frame changes both its values and the binary mask geometry; this experiment does not separate those causes.

**Newly hidden frame 1 is undercalibrated, but does not cause the mean-u collapse.** Its Magnetic MSE=0.075665, u=0.006618 and C=0.08746. Adding it to common frames raises h23 mean u by 0.54%, rather than depressing it. For Density, frame 1 has MSE=0.326656, u=0.033193 and C=0.10161; it raises mean u by 7.93%. Per-frame checking confirms underestimation throughout h23: Magnetic C spans 0.00333–0.08746, and Density C spans 0.01262–0.10161. At h18 the same statement would be incorrect for Density: its first hidden frame has C=1.424 before falling to 0.04692 at the last frame.

3. **Spatial ranking is weak for Magnetic and substantially better for Density.** At h23, Magnetic pooled rho=0.120 and mean within-run/frame spatial rho=0.147; the top predicted-u frame-decile has only 2.14x the bottom-decile MSE. By, which dominates Magnetic error, has within-frame rho=0.005 and top/bottom MSE=0.77, so the most uncertain locations do not even have larger aggregate error than the least uncertain locations. At h18 By is also weak/inverted (within-frame rho=-0.038, top/bottom MSE=0.67). In contrast, Density h23 has within-frame rho=0.511 and top/bottom MSE=11.90, yet its highest-u decile still has u/MSE=0.04948. Density therefore shows useful spatial ordering with severely insufficient amplitude; Magnetic additionally has poor ordering, especially By. Pooled Magnetic correlations partly mix channel scales and should not replace the separate component results.

4. **The failure is already present in binary-u pass 0.** All h22/h23 discontinuity, lead-time slopes, and spatial results above use a single forward with exact prefix/u=0 and Gaussian suffix/u=1. Continuous-u reuse cannot be the origin of this binary-input failure.

5. **Pass 1 improves raw errors relative to the much harder forecast input, but is harmful relative to the matched current-mask binary control.** Its fresh mask observes exactly 50% of spatial sites in every frame, shared across channels; this information is much richer than a one-frame prefix. For source h23 Magnetic, pass1 MSE=0.117732 versus 0.009422 for the same-mask binary control (12.50x), while u is nearly unchanged (0.007987 versus 0.008565). Calibration is 0.06784 versus 0.908999. Density pass1 MSE=0.038810 versus 0.003861 (10.05x), and C=0.27872 versus 0.90128. At source h18, Magnetic and Density MSE penalties are already 5.35x and 6.11x. Thus errors carried by reused (mu0,u0) are not adequately represented by the resulting u. This joint mean/variance reuse comparison does not isolate u alone. Nor is it a direct test of pass1 under a pure extrapolation mask: that condition was not introduced.

6. **Evidence specifically identifying a limited-global-context uncertainty head is INCONCLUSIVE.** The paired suppression and declining lead-time curves show a failure to track temporal difficulty, but do not localize it to a particular layer. Source inspection shows a pointwise 1x1x1 variance readout operating on contextual decoder features, after the existing spatiotemporal network. A pointwise readout does not itself establish absence of global context. No head-only or contextual-feature intervention was performed, so these measurements cannot distinguish shared features, variance readout, or learned objective behavior as the architectural cause.

7. **Smallest justified next experiment: a frozen-checkpoint, matched-mask 2x2 reuse-input ablation at h18/h22/h23.** Keep exact targets/u=0 at the same fresh mask. At still-hidden sites independently choose values from mu0 versus the already paired Gaussian noise, and variance from raw u0 versus one. Two corners (full reuse and the binary control) were measured here; only two additional forward conditions are needed. Compare same-voxel MSE, u and calibration to separate reused-mean contamination from the raw-u conditioning effect. This is a proposed diagnostic ablation, not a change to training semantics, and was not executed here. It addresses the measured 10–12.5x reuse penalty without prescribing architecture, loss, metadata, or exposure changes. The binary h23 failure would remain a separate issue to investigate.

## Execution and limitations

Inference ran on one GPU within user-owned existing allocation 61481634. No Slurm job was submitted and no training was run. All new scripts and artifacts are confined to this evaluation directory. Training/model/masking source, checkpoints and prior evaluation outputs were not modified. Both requested figures and all requested CSVs, including per-frame pass1 metrics, were generated. Exact per-window ranks and run/frame summaries are provided as auxiliary CSVs. Primary sums use all hidden voxels; only pooled ranking uses a sample. These are measured point estimates, without a formal significance test or a unique causal attribution.

**Forecast uncertainty collapse diagnostic**

- Primary model: epoch-276 Student-t, 20% extrapolation
- Per-frame uncertainty growth: NONMONOTONIC; often decreases as MSE grows
- h22→h23 discontinuity: PRESENT; common-frame Magnetic u -43.1%, MSE +59.3%
- Spatial uncertainty-error ranking: POOR for Magnetic (frame rho 0.147 at h23); Density is stronger (0.511)
- Pass-0 uncertainty behavior: declining/insufficient lead response and abrupt one-frame-prefix suppression are already present
- Pass-1 contribution: HARMFUL relative to matched binary current-mask control (h23 MSE penalties: Magnetic 12.50x, Density 10.05x)
- Main failure mechanism: incorrect temporal uncertainty response, additional one-observed-frame suppression, and weak Magnetic spatial ranking; reuse adds a separate error/variance mismatch
- Evidence for architectural limitation: INCONCLUSIVE
- Recommended next experiment: frozen-checkpoint 2x2 mean/variance reuse-input ablation; two additional conditions, no training
- Training code modified: NO
- Training launched: NO
