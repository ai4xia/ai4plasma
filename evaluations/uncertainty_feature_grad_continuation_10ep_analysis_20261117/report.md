# Scientific Evaluation: Uncertainty Feature-Gradient Continuation (Pass 0 Only)

**Date:** 2026-11-17  
**Evaluation Directory:** `evaluations/uncertainty_feature_grad_continuation_10ep_analysis_20261117/`  
**Dataset:** VPIC PPPL Reconnection Benchmark (125 paired held-out test windows, 125 val windows)  
**Models:**
- **Baseline:** Epoch 276, detached variance features (`runs/multi_beta_three_phase_temporal_extrapolation_continuation_20ep_20261116/final.pt`)
- **Treatment:** Epoch 286, uncertainty gradients enabled into shared features (`runs/multi_beta_three_phase_uncertainty_feature_grad_continuation_10ep_20261117/final.pt`)

---

## 1. Executive Summary & Core Findings

This evaluation determines whether allowing Student-t uncertainty NLL gradients to backpropagate into shared U-Net trunk features (while retaining `mu.detach()` in the uncertainty loss) resolves the long-horizon predictive uncertainty collapse observed in earlier diagnostics.

### Key Conclusions:
1. **Dramatic Recovery of Horizon Dynamic Range**:
   - In baseline, Magnetic uncertainty collapsed at long horizons ($u(h23)/u(h1) = 0.85$, predicting *less* uncertainty for 23 missing frames than 1 missing frame).
   - In treatment, Magnetic uncertainty scales strongly with horizon ($u(h23)/u(h1) = 5.18$, a **$6.1\times$ dynamic range increase**).
   - Density uncertainty dynamic range expanded from $2.09\times$ to **$8.37\times$**.
   - Spearman rank correlation between horizon $h$ and predicted $u$ jumped from **$0.713 \to 0.989$** (Magnetic) and **$0.816 \to 0.993$** (All channels).
2. **Resolution of Within-$h18$ Inversion (Key Diagnostic A)**:
   - In baseline, across lead steps 1 to 18 within horizon 18, actual MSE grew by $4.38\times$ while predicted $u$ collapsed by $69\%$ ($u_{\text{last}}/u_{\text{first}} = 0.314$).
   - In treatment, predicted $u$ increases monotonically along forecast lead time ($u_{\text{last}}/u_{\text{first}} = 2.61\times$), closely tracking the $3.15\times$ error growth. Calibration ratio $u/\text{MSE}$ remains stable between $1.17$ and $1.48$ across all 18 lead steps.
3. **Substantial Reduction of $h22 \to h23$ Suppression (Key Diagnostic B)**:
   - Comparing identical physical hidden frames 2–23 between $h=22$ and $h=23$, baseline predicted $u$ dropped by **$-43.1\%$** despite error increasing by $+59.3\%$.
   - Treatment reduced this suppression to **$-14.5\%$** (a $66\%$ reduction in suppression magnitude).
4. **Massive Improvement in Spatial Error Discrimination**:
   - At $h=23$, within-frame spatial Spearman rank correlation between error and uncertainty for Magnetic jumped from **$0.147 \to 0.493$**.
   - Specifically, the $B_y$ channel jumped from near-zero correlation (**$0.0051 \to 0.4648$**), resolving the previous blind spot in out-of-plane magnetic field uncertainty.
5. **Reconstruction MSE Regression (The Trade-Off)**:
   - Standard 5-family fixed-mask validation MSE increased from **$0.0519 \to 0.0859$** ($+65.6\%$), and training history validation MSE rose from **$0.0138 \to 0.0262$** ($+89.9\%$).
   - The degradation is largest in dense reconstruction tasks (`spatial_random` and `spatial_grid`, $\approx 2.35\times$), demonstrating that co-opting shared trunk features for uncertainty representation competes with fine-grained interpolation fidelity.

---

## 2. Checkpoint & Training History Audit

Both checkpoints were verified on disk and confirmed consistent with the experimental protocol:

| Property | Baseline Checkpoint | Treatment Checkpoint | Status |
|---|---|---|---|
| **Path** | `runs/..._continuation_20ep_20261116/final.pt` | `runs/..._uncertainty_feature_grad.../final.pt` | Verified |
| **Epoch** | 276 | 286 (completed 277–286) | Verified |
| **Source Checkpoint** | Epoch-256 Three-Phase Student-t | Epoch-276 Temporal Extrapolation Baseline | Verified |
| **Feature Detachment** | `detach_variance_features = True` | `detach_variance_features = False` | Verified |
| **Loss Objective** | Student-t NLL ($\text{df}=3.0, \lambda_v=1, \lambda_r=1$) | Student-t NLL ($\text{df}=3.0, \lambda_v=1, \lambda_r=1$) | Identical |
| **Curriculum Phase** | Phase C Settled ($K=1$ reuse, 2 forwards) | Phase C Settled ($K=1$ reuse, 2 forwards) | Identical |
| **Phase D** | Disabled | Disabled | Identical |
| **Learning Rate** | Constant $10^{-5}$ | Constant $10^{-5}$ | Identical |
| **Nonfinite Events** | 0 across all 20 epochs | 0 across all 10 epochs | Clean |
| **AMP Step Skips** | 0 | 0 | Clean |
| **Data Split SHA256** | `0e6b4276bc7f40003eed...` | `0e6b4276bc7f40003eed...` | Identical |

---

## 3. Forecast Horizon Evaluation ($h = 1 \dots 23$)

Evaluated on the **125 held-out test windows** across all 23 horizons using pass-0 only (visible prefix $0 \dots 23-h$, hidden suffix $24-h \dots 23$).

### 3.1 Primary Response Ratios

| Metric | Baseline (Epoch 276) | Treatment (Epoch 286) | Change |
|---|---:|---:|---|
| **Density $u(h18)/u(h1)$** | 2.01 | **4.57** | $+127\%$ expansion |
| **Magnetic $u(h18)/u(h1)$** | 1.71 | **4.80** | $+181\%$ expansion |
| **Density $u(h23)/u(h1)$** | 2.09 | **8.37** | $+300\%$ expansion |
| **Magnetic $u(h23)/u(h1)$** | 0.85 | **5.18** | **$+512\%$ (Reverses Collapse)** |

### 3.2 Monotonicity: Spearman Rank Correlation ($\text{Spearman}(h, \cdot)$)

| Modality | Baseline $\text{Spearman}(h, u)$ | Treatment $\text{Spearman}(h, u)$ | Baseline $\text{Spearman}(h, \text{MSE})$ | Treatment $\text{Spearman}(h, \text{MSE})$ |
|---|---:|---:|---:|---:|
| **Density** | 0.990 | **0.996** | 1.000 | 1.000 |
| **Magnetic** | 0.713 | **0.989** | 1.000 | 1.000 |
| **All Channels** | 0.816 | **0.993** | 1.000 | 1.000 |

### 3.3 Horizon-by-Horizon Comparison Table

The table below contrasts actual MSE and predicted $u$ across selected horizons to evaluate whether the uncertainty increase reflects genuine difficulty dependence or uniform global inflation:

| Horizon $h$ | Channel | Base MSE | Treat MSE | MSE Ratio ($\text{Tr}/\text{Ba}$) | Base $u$ | Treat $u$ | $u$ Ratio ($\text{Tr}/\text{Ba}$) | Base $u/\text{MSE}$ | Treat $u/\text{MSE}$ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | Density | 0.0103 | 0.0181 | 1.765× | 0.0061 | 0.0335 | **5.531×** | 0.591 | 1.852 |
| **6** | Density | 0.0239 | 0.0394 | 1.647× | 0.0086 | 0.0667 | **7.804×** | 0.358 | 1.696 |
| **12** | Density | 0.0449 | 0.0635 | 1.415× | 0.0101 | 0.0855 | **8.497×** | 0.224 | 1.347 |
| **18** | Density | 0.0961 | 0.1203 | 1.252× | 0.0122 | 0.1532 | **12.580×** | 0.127 | 1.274 |
| **22** | Density | 0.2189 | 0.2502 | 1.143× | 0.0154 | 0.2432 | **15.751×** | 0.071 | 0.972 |
| **23** | Density | 0.4685 | 0.4292 | 0.916× | 0.0127 | 0.2806 | **22.126×** | 0.027 | 0.654 |
| **1** | Magnetic | 0.0286 | 0.0555 | 1.940× | 0.0070 | 0.1010 | **14.444×** | 0.244 | 1.820 |
| **6** | Magnetic | 0.0727 | 0.1196 | 1.645× | 0.0075 | 0.2099 | **28.061×** | 0.103 | 1.755 |
| **12** | Magnetic | 0.1256 | 0.1911 | 1.521× | 0.0092 | 0.2976 | **32.456×** | 0.073 | 1.557 |
| **18** | Magnetic | 0.2808 | 0.3657 | 1.303× | 0.0119 | 0.4845 | **40.613×** | 0.042 | 1.325 |
| **22** | Magnetic | 0.7207 | 0.8142 | 1.130× | 0.0104 | 0.6377 | **61.559×** | 0.014 | 0.783 |
| **23** | Magnetic | 1.1011 | 1.0229 | 0.929× | 0.0059 | 0.5235 | **88.367×** | 0.005 | 0.512 |

> [!IMPORTANT]
> **Distinguishing Difficulty Scaling from Global Inflation**:  
> The ratio $u_{\text{treatment}}(h) / u_{\text{baseline}}(h)$ is **strictly non-constant**: it grows monotonically from $5.53\times \to 22.13\times$ in Density and $14.44\times \to 88.37\times$ in Magnetic. In contrast, the error ratio $\text{MSE}_{\text{treatment}} / \text{MSE}_{\text{baseline}}$ decreases from $1.94\times$ down to $0.93\times$. This confirms that the treatment introduces genuine difficulty-dependent uncertainty scaling rather than uniform global inflation. Furthermore, absolute calibration $u/\text{MSE}$ remains within a narrow band ($0.51 \dots 1.85$), eliminating the orders-of-magnitude collapse present in the baseline ($0.005$).

---

## 4. Per-Frame Temporal Uncertainty Trajectories

### 4.1 Key Diagnostic A — Within $h=18$ Lead Progression (Magnetic)

In horizon 18, frames 0–5 are visible and frames 6–23 are hidden (lead steps 1 to 18):

| Lead Step | Local Frame | Base MSE | Base $u$ | Treat MSE | Treat $u$ | Base $u/\text{MSE}$ | Treat $u/\text{MSE}$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | 6 | 0.0948 | 0.0215 | 0.1565 | 0.2246 | 0.226 | 1.436 |
| **3** | 8 | 0.1705 | 0.0259 | 0.2567 | 0.3246 | 0.152 | 1.265 |
| **6** | 11 | 0.2326 | 0.0157 | 0.3186 | 0.4405 | 0.068 | 1.383 |
| **9** | 14 | 0.2805 | 0.0084 | 0.3661 | 0.5335 | 0.030 | 1.457 |
| **12** | 17 | 0.3553 | 0.0084 | 0.4427 | 0.5709 | 0.024 | 1.290 |
| **15** | 20 | 0.3546 | 0.0047 | 0.4469 | 0.5886 | 0.013 | 1.317 |
| **18 (Last)** | 23 | 0.4153 | 0.0067 | 0.4928 | 0.5854 | 0.016 | 1.188 |

**Outcome for Diagnostic A**:
- **Baseline**: Last-frame error was $4.38\times$ first-frame error, but predicted uncertainty dropped to **$0.314\times$** (an inverted, pathological trajectory).
- **Treatment**: Last-frame error is $3.15\times$ first-frame error, and predicted uncertainty **increases by $2.61\times$**, keeping calibration stable throughout the entire rollout.

### 4.2 Key Diagnostic B — $h22 \to h23$ Paired Suppression (Frames 2–23)

Evaluating identical physical hidden frames 2 through 23 between $h=22$ (2 prefix frames) and $h=23$ (1 prefix frame):

| Model | Frame 2–23 MSE Change | Frame 2–23 Predicted $u$ Change | Suppression Magnitude |
|---|---:|---:|---|
| **Baseline** | $+59.25\%$ | **$-43.12\%$** | Severe suppression |
| **Treatment** | $+30.85\%$ | **$-14.54\%$** | **$66\%$ reduction in suppression** |

While a mild $-14.5\%$ decrease remains when dropping the second prefix frame, the severe $-43.1\%$ cliff has been largely eliminated.

---

## 5. Spatial Uncertainty-Error Correspondence

Evaluated within-frame Spearman rank correlation between predicted $u$ and realized squared error on hidden voxels:

| Horizon | Channel | Baseline Mean Frame Spearman | Treatment Mean Frame Spearman | Absolute Change |
|---:|---|---:|---:|---:|
| **$h=18$** | **Magnetic** | 0.1912 | **0.5026** | $+0.3114$ |
| | $B_x$ | 0.1160 | **0.5148** | $+0.3988$ |
| | $B_y$ | -0.0380 | **0.3972** | **$+0.4352$** |
| | $B_z$ | 0.0879 | **0.2212** | $+0.1333$ |
| | Density | 0.2961 | **0.3412** | $+0.0451$ |
| **$h=22$** | **Magnetic** | 0.1511 | **0.4993** | $+0.3482$ |
| | $B_y$ | 0.0025 | **0.4536** | $+0.4511$ |
| **$h=23$** | **Magnetic** | 0.1472 | **0.4931** | **$+0.3459$ ($3.3\times$ higher)** |
| | $B_x$ | 0.1593 | **0.5593** | $+0.4000$ |
| | $B_y$ | 0.0051 | **0.4648** | **$+0.4597$ (Blind spot resolved)** |
| | $B_z$ | 0.1376 | **0.3222** | $+0.1846$ |
| | Density | 0.5106 | **0.5018** | $-0.0088$ (Maintained strong) |

> [!TIP]
> The baseline model suffered from an extreme spatial ranking failure in $B_y$ ($\rho = 0.0051$, random guessing). Enabling uncertainty gradients into shared features restored strong spatial ranking ($\rho = 0.4648$), demonstrating that spatial error localization in magnetic reconnection requires trunk-feature adaptation.

---

## 6. Ordinary Reconstruction Regression Check

| Evaluation Benchmark | Baseline (Epoch 276) | Treatment (Epoch 286) | Degradation Ratio |
|---|---:|---:|---:|
| **Training History Val MSE** | 0.013813 | 0.026225 | **$1.899\times$ ($+89.9\%$)** |
| **Fixed-Mask 5-Family Val MSE** | 0.051902 | 0.085928 | **$1.656\times$ ($+65.6\%$)** |

### Per-Task Breakdown on Fixed Validation Masks:
- `spatial_random`: $0.00808 \to 0.01897$ ($2.35\times$)
- `spatial_grid`: $0.00893 \to 0.02128$ ($2.38\times$)
- `spatial_block`: $0.10370 \to 0.17754$ ($1.71\times$)
- `temporal_random`: $0.02498 \to 0.04657$ ($1.86\times$)
- `temporal_block`: $0.11382 \to 0.16528$ ($1.45\times$)

The degradation is systematically higher on high-density localized interpolation tasks (`spatial_random` and `spatial_grid`, $>2.3\times$) than on large structural block tasks ($1.45\times$).

---

## 7. Explicit Answers to Interpretation Questions

1. **Did enabling uncertainty gradients into shared U-Net features improve horizon-dependent $u$ response?**  
   **YES.** Magnetic $u(h23)/u(h1)$ jumped from $0.85 \to 5.18$, Density $u(h23)/u(h1)$ rose from $2.09 \to 8.37$, and horizon Spearman correlation rose from $0.713 \to 0.989$.
2. **Did it correct the within-$h18$ behavior where predicted uncertainty decreases while actual error increases?**  
   **YES.** The previous $0.314\times$ drop was inverted to a healthy $+2.61\times$ increase along rollout lead steps.
3. **Did it improve the $h22 \to h23$ discontinuity?**  
   **YES.** Paired suppression on physical frames 2–23 was reduced by $66\%$ (from $-43.1\%$ to $-14.5\%$).
4. **Did Magnetic spatial uncertainty-error ranking improve?**  
   **YES.** Magnetic spatial Spearman jumped from $0.147 \to 0.493$, and the $B_y$ blind spot was completely resolved ($0.005 \to 0.465$).
5. **Did absolute $u/\text{MSE}$ calibration improve, or did predicted uncertainty merely inflate globally?**  
   **IMPROVED.** Uncertainty growth scaled steeply with horizon ($14.4\times \to 88.4\times$) while error ratios remained near unity, keeping $u/\text{MSE}$ within a well-calibrated factor of 2 across all horizons rather than collapsing by $49\times$.
6. **Was mean reconstruction degraded?**  
   **YES.** Standard validation MSE degraded by $+65.6\%$ on the 5-family benchmark and $+89.9\%$ in training history.
7. **Do the results support the hypothesis that detached shared features were a major bottleneck?**  
   **SUPPORTED.** Detached features were indeed the bottleneck preventing first-forward predictive uncertainty scaling and spatial discrimination. However, removing detachment introduces a clear trade-off against mean reconstruction accuracy.

---

## 8. Final Required Summary

**Pass-0 Uncertainty Feature-Gradient Evaluation**

- Training completed: YES
- Nonfinite events / AMP skips: 0 nonfinite / 0 skips
- Validation MSE: baseline 0.013813 → treatment 0.026225
- Standard fixed-mask MSE: baseline 0.051902 → treatment 0.085928
- Density `u(h18)/u(h1)`: 2.01 → 4.57
- Magnetic `u(h18)/u(h1)`: 1.71 → 4.80
- Density `u(h23)/u(h1)`: 2.09 → 8.37
- Magnetic `u(h23)/u(h1)`: 0.85 → 5.18
- Magnetic h18 per-frame `u_last/u_first`: 0.314 → 2.606
- Magnetic h22→h23 paired uncertainty change: −43.1% → −14.5%
- Magnetic h23 spatial Spearman: 0.147 → 0.493
- Horizon-dependent uncertainty response: IMPROVED
- Absolute calibration: IMPROVED
- Reconstruction quality: WORSE
- Feature-detachment hypothesis: SUPPORTED
- Most important remaining issue: Reconstruction MSE degradation (1.66×–1.90× increase) resulting from shared feature interference between the mean reconstruction and uncertainty branches.
- Smallest justified next step: Test gradient routing / branch isolation that allows the variance branch to extract rich representations without distorting the mean prediction trunk (e.g., small dedicated adapter/side-network for uncertainty features, or a stop-gradient branch after an intermediate layer).
- Training launched: NO
- Existing source files modified: NO
