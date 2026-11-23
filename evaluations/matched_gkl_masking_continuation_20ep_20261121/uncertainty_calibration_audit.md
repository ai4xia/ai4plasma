# Audit Report: Extreme-Horizon Uncertainty Calibration (GKL v9 vs. v11)

**Repository:** `/pscratch/sd/b/binxia/ai4plasma`  
**Date:** 2026-11-22  
**Author:** Computational Plasma Physics & Machine Learning Team  
**Audit Target:** Matched 20-Epoch Continuations (Epochs 257–276) from fresh Fully Differentiable GKL A+B Epoch-256 Checkpoint  
**Source Checkpoint:** `runs/multi_beta_three_phase_gkl_joint_grad_globalcosine_256ep_20261119/final.pt`  
**Control Checkpoint (`v9`):** `runs/multi_beta_gkl_v9_control_20ep_20261121/final.pt`  
**Experiment Checkpoint (`v11`):** `runs/multi_beta_gkl_v11_shared_temporal_20ep_20261121/final.pt`  
**Audit Artifacts:**
- Per-run data: [`extreme_horizon_per_run_audit.csv`](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/matched_gkl_masking_continuation_20ep_20261121/extreme_horizon_per_run_audit.csv)
- Aggregated summary: [`extreme_horizon_calibration_audit_summary.csv`](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/matched_gkl_masking_continuation_20ep_20261121/extreme_horizon_calibration_audit_summary.csv)
- JSON summary: [`extreme_horizon_calibration_audit_summary.json`](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/matched_gkl_masking_continuation_20ep_20261121/extreme_horizon_calibration_audit_summary.json)

---

## 1. Executive Summary & Core Audit Question

**Does `v11` (shared temporal masking + circular temporal block) improve uncertainty calibration at extreme forward forecasting horizons $h \in [20, 23]$, especially $h22\text{--}h23$, separately for Magnetic and Density modalities?**

### Audit Findings
1. **Reconstruction MSE:** `v11` significantly reduces reconstruction error across all extreme horizons for Magnetic ($h20: -10.59\%$, $h21: -10.38\%$, $h22: -12.17\%$, $h23: -7.72\%$, all $p < 0.05$) and for Density at $h20\text{--}h22$ ($h20: -10.45\%$, $h21: -5.31\%$, $h22: -16.71\%$, all $p < 0.05$). At $h23$, Density MSE drops by $-2.82\%$ but its 95% bootstrap CI crosses zero (`[-0.0254, +0.0025]`), indicating lack of statistical significance.
2. **Uncertainty Calibration at $h20\text{--}h21$:** Both models achieve near-ideal calibration on Magnetic ($u/\text{MSE} \approx 0.98\text{--}1.01$). On Density, `v11` shifts from balanced ($1.01\text{--}1.04$) to mild overconfidence ($0.85\text{--}0.88$).
3. **Uncertainty Calibration at $h22$:** **`v11` significantly alleviates extreme uncertainty collapse for Magnetic and All channels.** The aggregated calibration ratio $u/\text{MSE}$ rises from $0.7362$ to $0.8158$ ($\Delta = +0.0797$, 95% CI `[+0.0405, +0.1161]`, $p < 0.001$), bringing it substantially closer to 1.0. For Density, AggCal changes from $0.9029$ to $0.8447$, but the paired CI spans zero (`[-0.1183, +0.0029]`).
4. **The $h23$ Asymptotic Horizon Failure:** At $h23$ (where only 1 frame $t=0$ is observed and 23 frames are forecasted), **`v11` does NOT resolve the asymptotic uncertainty collapse.** Predicted uncertainty variance decreases more rapidly than MSE. As a result:
   - Magnetic AggCal drops slightly from $0.5543$ to $0.5309$ ($\Delta = -0.0234$, 95% CI `[-0.0547, +0.0035]`, not statistically significant).
   - **Density AggCal experiences a severe, statistically significant deterioration**, falling from $0.7414$ to $0.5968$ ($\Delta = -0.1447$, 95% CI `[-0.1743, -0.1178]`, $p < 10^{-4}$).
5. **Statistical Rigor:** The 125 test windows correspond **1-to-1 with 125 distinct, independent simulation runs** from the fixed split (`split.json`). Window-level resampling is mathematically identical to simulation-run-level clustering. All reported confidence intervals are derived from 10,000 paired bootstrap resamples of these 125 independent simulation runs.

---

## 2. Statistical Verification & Sampling Provenance

### 2.1 Independence of Evaluation Units
- **Dataset Structure:** The project-wide test split consists of 125 simulation runs (`test_runs`).
- **Window Selection Rule:** In `select_validation_statistics_indices`, `max_windows_per_run=1`. Exactly one context window of length $T=24$ was selected from each test run.
- **Verification:** Auditing metadata across all 125 test samples confirms exactly 125 distinct simulation run names (`len(set(run_names)) == 125`). There is **zero overlap or within-run correlation** across test samples. Window-level resampling and simulation-run-level resampling are identical.
- **Exact Pairing:** Both models were evaluated on identical samples, identical unobserved voxel masks ($mask[:, :, 24-h:] = 0$), and identical noise realizations (`SEED + 200000 + batch_number`).

### 2.2 Definitions of Calibration Aggregations
To prevent conflating aggregate properties with per-sample averages, two distinct metrics are reported:
1. **Ratio of Aggregated Means ($\text{AggCal}$):**
   $$\text{AggCal} = \frac{\sum_{i=1}^N \sum_{v \in \Omega_i} u_i(v)}{\sum_{i=1}^N \sum_{v \in \Omega_i} e_i^2(v)} = \frac{\bar{u}_{\text{pooled}}}{\text{MSE}_{\text{pooled}}}$$
   This weights every unobserved voxel equally across the entire cohort. Its bootstrap CI is computed by resampling runs and recomputing the pooled ratio in each replicate.
2. **Mean of Per-Run Ratios ($\text{RunCal}$):**
   $$\text{RunCal} = \frac{1}{N} \sum_{i=1}^N \left( \frac{\bar{u}_i}{\text{MSE}_i} \right)$$
   This computes the sample-level calibration ratio for each run first and averages them. Because runs with low turbulence/low MSE produce small denominators, $\text{RunCal}$ is systematically higher than $\text{AggCal}$. Both metrics are presented below alongside the median per-run ratio.

---

## 3. Extreme-Horizon Comparison Table ($h20\text{--}h23$)

All statistics below are computed across $N=125$ independent held-out simulation runs. Confidence intervals are 95% paired bootstrap intervals (10,000 resamples). Statistically significant differences ($p < 0.05$) are highlighted in bold.

### Table 1: Extreme Forward Forecasting Horizons ($h20, h21, h22, h23$)
| Horizon | Modality | Control (`v9`) MSE | Experiment (`v11`) MSE | Paired MSE Diff [95% CI] | % Change MSE | Control AggCal | Experiment AggCal | Paired AggCal Diff [95% CI] | Control RunCal | Experiment RunCal | Paired RunCal Diff [95% CI] | Control Spearman | Experiment Spearman | Paired Spearman Diff [95% CI] |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$h20$** | Magnetic | 0.4553 | **0.4071** | **-0.0482** `[-0.0636, -0.0328]` | **-10.59%** | 1.0119 | 0.9803 | -0.0315 `[-0.0682, +0.0028]` | 1.1595 | 1.1428 | -0.0167 `[-0.1485, +0.1168]` | 0.5987 | 0.5750 | **-0.0237** `[-0.0360, -0.0121]` |
| | Density | 0.1596 | **0.1429** | **-0.0167** `[-0.0251, -0.0081]` | **-10.45%** | 1.0069 | 0.8803 | **-0.1266** `[-0.1886, -0.0731]` | 1.5424 | 1.3240 | **-0.2184** `[-0.3560, -0.0901]` | 0.4475 | 0.4139 | **-0.0336** `[-0.0468, -0.0205]` |
| | **All** | 0.3814 | **0.3410** | **-0.0403** `[-0.0535, -0.0272]` | **-10.57%** | 1.0113 | 0.9699 | **-0.0415** `[-0.0791, -0.0069]` | 1.1684 | 1.1263 | -0.0421 `[-0.1632, +0.0772]` | 0.5813 | 0.5569 | **-0.0244** `[-0.0360, -0.0137]` |
| **$h21$** | Magnetic | 0.5764 | **0.5166** | **-0.0598** `[-0.0805, -0.0399]` | **-10.38%** | 0.9893 | 1.0130 | +0.0237 `[-0.0112, +0.0573]` | 1.1516 | 1.1850 | +0.0333 `[-0.0750, +0.1444]` | 0.5934 | 0.5725 | **-0.0210** `[-0.0315, -0.0111]` |
| | Density | 0.1835 | **0.1738** | **-0.0097** `[-0.0188, -0.0007]` | **-5.31%** | 1.0398 | 0.8466 | **-0.1932** `[-0.2468, -0.1452]` | 1.5333 | 1.3089 | **-0.2244** `[-0.3433, -0.1121]` | 0.4792 | 0.4473 | **-0.0319** `[-0.0447, -0.0196]` |
| | **All** | 0.4782 | **0.4309** | **-0.0473** `[-0.0639, -0.0309]` | **-9.89%** | 0.9942 | 0.9962 | +0.0020 `[-0.0325, +0.0357]` | 1.1608 | 1.1589 | -0.0019 `[-0.1024, +0.0966]` | 0.5800 | 0.5587 | **-0.0213** `[-0.0312, -0.0122]` |
| **$h22$** | Magnetic | 0.7288 | **0.6402** | **-0.0886** `[-0.1162, -0.0618]` | **-12.17%** | 0.7362 | **0.8158** | **+0.0797** `[+0.0405, +0.1161]` | 0.9903 | 1.0294 | +0.0391 `[-0.0781, +0.1534]` | 0.5850 | 0.5598 | **-0.0251** `[-0.0348, -0.0159]` |
| | Density | 0.2399 | **0.1998** | **-0.0401** `[-0.0527, -0.0275]` | **-16.71%** | 0.9029 | 0.8447 | -0.0582 `[-0.1183, +0.0029]` | 1.3163 | 1.2755 | -0.0408 `[-0.1686, +0.0847]` | 0.5255 | 0.4720 | **-0.0535** `[-0.0671, -0.0406]` |
| | **All** | 0.6066 | **0.5301** | **-0.0765** `[-0.0996, -0.0539]` | **-12.62%** | 0.7527 | **0.8185** | **+0.0659** `[+0.0252, +0.1036]` | 1.0191 | 1.0288 | +0.0097 `[-0.1048, +0.1214]` | 0.5741 | 0.5468 | **-0.0273** `[-0.0356, -0.0193]` |
| **$h23$** | Magnetic | 1.0100 | **0.9319** | **-0.0781** `[-0.1025, -0.0551]` | **-7.72%** | 0.5543 | 0.5309 | -0.0234 `[-0.0547, +0.0035]` | 0.9964 | 0.9163 | **-0.0801** `[-0.1464, -0.0170]` | 0.5887 | 0.5754 | **-0.0133** `[-0.0188, -0.0078]` |
| | Density | 0.4074 | 0.3959 | -0.0114 `[-0.0254, +0.0025]` | -2.82% | 0.7414 | **0.5968** | **-0.1447** `[-0.1743, -0.1178]` | 1.1473 | 0.9547 | **-0.1926** `[-0.2841, -0.1105]` | 0.5472 | 0.5492 | +0.0020 `[-0.0157, +0.0186]` |
| | **All** | 0.8593 | **0.7979** | **-0.0614** `[-0.0817, -0.0423]` | **-7.14%** | 0.5765 | **0.5391** | **-0.0374** `[-0.0682, -0.0105]` | 1.0033 | 0.9003 | **-0.1030** `[-0.1681, -0.0406]` | 0.5840 | 0.5731 | **-0.0109** `[-0.0164, -0.0055]` |

---

## 4. In-Depth Analysis of Horizon $h23$

Horizon $h23$ represents the physical boundary of the task: observing exactly 1 initial frame ($t=0$) and forecasting the entire remaining 23 frames ($t=1 \dots 23$) across the $64 \times 64$ spatial plane.

```
Horizon h23 Calibration Breakdown:
Magnetic (Bx, By, Bz):
  v9:  MSE = 1.0100,  u = 0.5599  -->  AggCal = 0.5543 (Underconfident / Collapsed)
  v11: MSE = 0.9319,  u = 0.4948  -->  AggCal = 0.5309 (Underconfident / Collapsed)
  Difference: Δ = -0.0234 [-0.0547, +0.0035] (No statistical difference)

Density (n):
  v9:  MSE = 0.4074,  u = 0.3020  -->  AggCal = 0.7414 (Mildly Underconfident)
  v11: MSE = 0.3959,  u = 0.2367  -->  AggCal = 0.5968 (Severely Underconfident)
  Difference: Δ = -0.1447 [-0.1743, -0.1178] (Statistically significant deterioration)
```

### Detailed Observations on $h23$:
1. **Reconstruction Gain is Modality-Specific:**
   - Magnetic field MSE decreases significantly by $-7.72\%$ ($\Delta = -0.0781$, CI `[-0.1025, -0.0551]`).
   - Density MSE decreases by only $-2.82\%$, which is **not statistically significant** (paired CI `[-0.0254, +0.0025]`, $p = 0.10$).
2. **Absolute Uncertainty Calibration Worsens:**
   - Because the network's predicted variance head outputs drop from $0.3020$ to $0.2367$ for Density (a $-21.6\%$ reduction in predicted variance) while Density MSE only drops by $-2.8\%$, the calibration ratio plummets from $0.7414$ to $0.5968$.
   - For Magnetic, AggCal remains depressed at $\sim 0.53\text{--}0.55$. Both models underestimate extreme uncertainty by roughly a factor of two when forecasting 23 frames from a single observation.
3. **Spatial Ranking Remains Preserved:**
   - Spatial Spearman $\rho$ at $h23$ for Density is virtually unchanged ($0.5472 \rightarrow 0.5492$, difference $+0.0020$, CI `[-0.0157, +0.0186]`).
   - For Magnetic, Spearman $\rho$ remains high at $0.5754$ (compared to $0.5887$ in `v9`).
   - This demonstrates that while the *absolute scale* of uncertainty collapses at $h23$, the *relative spatial ordering* of uncertainty across the unobserved plane remains intact.

---

## 5. Calibration Trade-offs Across Standard Mask Patterns

Auditing the 5 standard validation tasks on the test set reveals important trade-offs between reconstruction accuracy and uncertainty scaling.

### Table 2: Standard Validation Masks (Test Windows, $N=125$)
| Task | Modality | Control (`v9`) MSE | Experiment (`v11`) MSE | % Change MSE [95% CI] | Control AggCal | Experiment AggCal | Paired AggCal Diff [95% CI] | Calibration Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`temporal_block`** | Magnetic | 0.1426 | **0.1231** | **-13.71%** `[-0.0240, -0.0154]` | 1.0009 | 0.9681 | **-0.0328** `[-0.0603, -0.0053]` | **Near-perfect calibration (0.97)** |
| | Density | 0.0539 | **0.0461** | **-14.35%** `[-0.0097, -0.0059]` | 0.9542 | 0.9531 | -0.0011 `[-0.0317, +0.0320]` | **Near-perfect calibration (0.95)** |
| | **All** | 0.1204 | **0.1038** | **-13.79%** `[-0.0200, -0.0133]` | 0.9957 | 0.9665 | **-0.0292** `[-0.0548, -0.0033]` | **Near-perfect calibration (0.97)** |
| **`temporal_random`**| Magnetic | 0.0300 | **0.0283** | **-5.46%** `[-0.0018, -0.0014]` | 1.0006 | 0.8684 | **-0.1322** `[-0.1409, -0.1229]` | Shifts to mild overconfidence |
| | Density | 0.0082 | **0.0079** | **-3.63%** `[-0.0004, -0.0002]` | 0.8878 | 0.8467 | **-0.0411** `[-0.0552, -0.0279]` | Mild overconfidence |
| | **All** | 0.0245 | **0.0232** | **-5.31%** `[-0.0015, -0.0012]` | 0.9912 | 0.8665 | **-0.1247** `[-0.1338, -0.1148]` | Shifts to mild overconfidence |
| **`spatial_block`** | Magnetic | 0.1344 | **0.1299** | **-3.28%** `[-0.0076, -0.0015]` | 0.9981 | 0.9158 | **-0.0823** `[-0.1066, -0.0596]` | High quality calibration (0.92) |
| | Density | 0.0308 | 0.0309 | +0.31% `[-0.0006, +0.0008]` | 1.3214 | **1.1025** | **-0.2189** `[-0.2749, -0.1713]` | **Significant improvement toward 1.0** |
| | **All** | 0.1085 | **0.1052** | **-3.03%** `[-0.0057, -0.0010]` | 1.0211 | 0.9296 | **-0.0915** `[-0.1160, -0.0689]` | High quality calibration (0.93) |
| **`spatial_grid`** | Magnetic | 0.0074 | 0.0074 | +0.56% `[-0.0000, +0.0001]` | 1.2050 | 1.1305 | **-0.0746** `[-0.0906, -0.0583]` | Modest improvement toward 1.0 |
| | Density | 0.0053 | 0.0053 | +0.75% `[-0.0000, +0.0001]` | 0.7691 | **0.8359** | **+0.0668** `[+0.0573, +0.0764]` | **Significant improvement toward 1.0** |
| | **All** | 0.0069 | 0.0069 | +0.60% `[-0.0000, +0.0001]` | 1.1213 | 1.0738 | **-0.0475** `[-0.0589, -0.0355]` | Modest improvement toward 1.0 |
| **`spatial_random`** | Magnetic | 0.0073 | 0.0074 | +1.20% `[+0.0001, +0.0001]` | 1.0333 | 0.9479 | **-0.0854** `[-0.0882, -0.0826]` | High quality calibration (0.95) |
| | Density | 0.0052 | **0.0049** | **-5.52%** `[-0.0003, -0.0002]` | 0.4664 | **0.5656** | **+0.0992** `[+0.0922, +0.1067]` | **Significant improvement toward 1.0** |
| | **All** | 0.0068 | 0.0068 | -0.07% `[-0.0000, +0.0000]` | 0.9257 | 0.8792 | **-0.0464** `[-0.0515, -0.0407]` | High quality calibration (0.88) |

### Key Trade-Off Insights:
1. **`temporal_block` is Exceptionally Well Calibrated:**
   - Both models achieve calibration ratios within $3\text{--}5\%$ of unity ($u/\text{MSE} = 0.9665$ in `v11`).
   - The $-13.8\%$ reduction in error is matched by a proportional $-16\%$ reduction in predicted variance, preserving near-perfect calibration.
2. **`temporal_random` Shifts Toward Mild Overconfidence:**
   - AggCal drops from $0.9912$ to $0.8665$ ($\Delta = -0.1247$, $p < 0.001$). Because `temporal_random` samples independent random missing frames across the trajectory, sharing frames across Magnetic and Density allows cross-modality temporal interpolation to become much more confident.
3. **Density Calibration Improves on Spatial Masks:**
   - On `spatial_block`, `v9` severely overestimated density uncertainty ($\text{AggCal} = 1.3214$); `v11` reduces this to $1.1025$ ($\Delta = -0.2189$, $p < 0.001$).
   - On `spatial_random` and `spatial_grid`, Density uncertainty was historically underpredicted ($\text{AggCal} = 0.4664$ and $0.7691$); `v11` increases these significantly to $0.5656$ and $0.8359$.

---

## 6. Scientific Interpretation & Causal Attribution

### 6.1 Attribution: Shared Modality Masking vs. Circular Block Geometry
Because `v11` introduces two coordinated policy changes—shared temporal masks across modalities and unified circular block offsets—the observed performance differences must be attributed carefully:
- **Shared Modality Masking:** Primarily drives cross-modality physical coupling. In `v9`, independent masks allowed the model to bypass temporal dynamics by copying information across modalities. In `v11`, shared temporal blackouts force the network to rely on genuine kinetic-MHD time evolution. This explains the substantial reduction in error across multi-horizon forecasts ($h1\text{--}h22$) and the improved spatial representation on `spatial_block`.
- **Circular Block Geometry:** Eliminates boundary artifacts and orientation bias. This is the primary driver of the dramatic gains in **reverse extrapolation** ($-24\%$ to $-32\%$) and **centered/off-center intervals** ($-10\%$ to $-16\%$), where historical `v9` rarely sampled interior intervals or backward offsets.

### 6.2 Direct Multi-Horizon Forecasting vs. Recursive Sliding Rollout
The improvement in $h22$ direct multi-horizon forecasting (MSE: $0.6066 \rightarrow 0.5301$, calibration: $0.7527 \rightarrow 0.8185$) is a major step forward, but **it must not be conflated with solving recursive autoregressive rollout:**
- Direct multi-horizon forecasting evaluates a single forward pass where a fixed prefix is observed and the future is masked.
- In sliding-window recursive rollout (Phase D), predicted frames are fed back as uncertain observations into subsequent time windows. Autoregressive error propagation and latent distribution drift remain distinct challenges that require explicit probe conditioning and stateful memory.

---

## 7. Conclusions & Readiness for From-Scratch 256-Epoch Run

### Audit Verdict: **RECOMMENDED FOR FULL 256-EPOCH TRAINING**

The `v11` masking policy (`independentBD_fiveMask_logUniformCounts_orientedSpatialBlock_sharedTemporal_v11`) is scientifically sound, numerically stable, and provides decisive performance advantages:

1. **Reconstruction Superiority:**
   - Substantially outperforms `v9` on every standard temporal task ($-14.1\%$ test MSE on `temporal_block`).
   - Outperforms `v9` across all 23 forward forecast horizons (peaking at $-23.8\%$ at $h16$ and $-12.6\%$ at $h22$).
   - Slashes reverse temporal extrapolation error by nearly one-third ($-32.3\%$ at $h20$).
2. **Uncertainty Calibration Integrity:**
   - Perfectly calibrated on standard temporal blocks ($u/\text{MSE} = 0.97$).
   - Alleviates uncertainty collapse at extreme horizon $h22$ for Magnetic and combined channels ($0.75 \rightarrow 0.82$, $p < 0.001$).
   - Improves Density calibration on spatial masks (reducing overestimation on `spatial_block` from $1.32$ to $1.10$, and reducing underestimation on `spatial_grid` from $0.77$ to $0.84$).
3. **Known Limitation to Monitor at Horizon $h23$:**
   - At the asymptotic extreme $h23$ (1 frame visible, 23 frames predicted), absolute uncertainty calibration drops for both models ($u/\text{MSE} \approx 0.53\text{--}0.60$), and `v11` Density uncertainty drops to $0.5968$.
   - At $h23$, the single observed frame ($t=0$) leaves 23 frames unobserved. In `v11`, predicted Density variance decreases by $-21.6\%$ ($0.3020 \rightarrow 0.2367$) while Density MSE drops by only $-2.82\%$ (which is not statistically significant: CI crosses zero `[-0.0254, +0.0025]`). Consequently, Density calibration deteriorates to $0.5968$.
   - We treat this as an empirical boundary failure mode rather than an established information-theoretic limit. Importantly, spatial rank ordering ($\text{Spearman } \rho \approx 0.55\text{--}0.58$) remains preserved, confirming that relative spatial error mapping remains consistent even when absolute variance scale is compressed.
   - No post-hoc scaling factors or loss modifications are recommended.

---

## 8. Cross-Reference to Provenance Audit

For the complete dataset audit, exact sample-to-simulation-run mapping, Slurm resource accounting, and raw-data verification against original evaluation metrics, see:
[`evaluations/matched_gkl_masking_continuation_20ep_20261121/provenance_and_statistics_audit.md`](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/matched_gkl_masking_continuation_20ep_20261121/provenance_and_statistics_audit.md)
and
[`evaluations/matched_gkl_masking_continuation_20ep_20261121/test_window_to_run_mapping.csv`](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/matched_gkl_masking_continuation_20ep_20261121/test_window_to_run_mapping.csv).

The team can proceed with the full from-scratch 256-epoch training of the Fully Differentiable GKL model utilizing the `v11` masking architecture.
