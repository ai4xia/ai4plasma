# Gradient-Scale & Directional Audit: Gaussian vs Student-t NLL under Joint A+B Routing

**Audit Date:** 2026-11-18  
**Repository:** `/pscratch/sd/b/binxia/ai4plasma`  
**Output Directory:** `evaluations/gaussian_vs_student_t_gradient_audit_20261118/`  
**Evaluated Checkpoint:** `runs/multi_beta_three_phase_joint_nll_grad_globalcosine_256ep_20261117/final.pt` (Epoch 256, Split `0e6b4276bc...`)  
**Hardware:** NVIDIA A100-PCIE-40GB (PyTorch 2.5.1+cu124)  

---

## 1. Executive Summary

This diagnostic audit evaluates the empirical gradient scale and directional alignment of **Gaussian NLL** versus **Student-t NLL ($\nu=3$)** under joint A+B gradient routing on our completed from-scratch 256-epoch model.

The audit was conducted across **8 paired test windows** drawn from distinct simulation runs spanning 8 plasma betas ($\beta \in \{0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20\}$) evaluated under **11 deterministic mask conditions** (5 standard masking families + pure temporal extrapolation at $h \in \{1, 6, 12, 18, 22, 23\}$), totaling **88 paired gradient evaluations**.

### Key Findings
1. **Uncertainty NLL Gradients Overwhelmingly Dominate Reconstruction MSE:**  
   On shared U-Net parameters ($\theta$), the median gradient norm ratio to reconstruction MSE is:
   $$\text{Student-t } R_T = \frac{\|g_T\|_2}{\|g_\mu\|_2} = \mathbf{261.84} \quad (16\text{th}\text{--}84\text{th}\text{ percentile: } 23.40 \text{ to } 1719.54)$$
   $$\text{Gaussian } R_G = \frac{\|g_G\|_2}{\|g_\mu\|_2} = \mathbf{197.38} \quad (16\text{th}\text{--}84\text{th}\text{ percentile: } 30.32 \text{ to } 963.12)$$
   In the unnormalized objective $L = \frac{L_\mu + \lambda_v L_{\text{NLL}}}{1 + \lambda_v}$ with $\lambda_v = 1$, shared features are driven $>200\times$ more strongly by uncertainty NLL than by reconstruction MSE.
2. **Path A Strongly Dominates Path B:**  
   For Student-t NLL, Path A (through predicted mean $\mu$) contributes **$24.05\times$** more gradient norm to shared features than Path B (through predicted variance $u$). For Gaussian NLL, Path A contributes **$3.46\times$** more than Path B.
3. **No Direct Gradient Opposition with Reconstruction:**  
   Cosine similarity between NLL gradients and reconstruction MSE on shared features is non-negative across all conditions:
   $$\cos(g_\mu, g_T) = +0.0710, \quad \cos(g_\mu, g_G) = +0.0900$$
   Path A and Path B are mutually near-orthogonal ($\cos \approx +0.02$ to $+0.03$), and both weakly align with reconstruction MSE. The observed reconstruction degradation in 256-epoch training was not due to adversarial gradient cancellation, but sheer gradient scale dominance.
4. **Extreme Error Saturation in Student-t vs Unbounded Scaling in Gaussian:**  
   At extreme forecast horizons ($h=22, 23$), $>50\%$ of hidden voxels have $r = (y-\mu)^2/u > 1$, and up to $1.3\%$ have $r > 100$ (with $r_{\max} \approx 1000$).
   - Student-t log-variance drive strictly **saturates at $-1.50$**, bounded regardless of error size.
   - Gaussian log-variance drive scales linearly with $r$, reaching **$-514.1$** ($343\times$ stronger).
   - In Gaussian NLL, the top 1% extreme error voxels contribute **$24.3\%$ to $32.3\%$** of the entire upward variance gradient, whereas in Student-t they contribute only **$2.3\%$ to $3.2\%$**.
5. **Normalization Scale Factor:**  
   Equalizing raw gradient norms requires an attenuation factor $s \approx 0.0038$ to $0.0051$ overall (or $s \approx 0.03$ to $0.06$ for long-horizon forecasting).

---

## 2. Checkpoint & Parameter Group Audit

- **Checkpoint:** `runs/multi_beta_three_phase_joint_nll_grad_globalcosine_256ep_20261117/final.pt`
- **Total Parameters:** 5,526,200 FP32 weights/biases
- **Architecture Partitioning:**
  - `shared_unet` ($\theta$): 58 parameter tensors, 5,526,000 parameters ($99.996\%$ of total)
  - `encoder`: 24 tensors, 2,005,752 parameters
  - `attention`: 12 tensors, 186,048 parameters
  - `decoder`: 22 tensors, 3,334,200 parameters
  - `mean_head` ($\phi$, `out`): 2 tensors, 100 parameters
  - `var_head` ($\psi$, `variance_out`): 2 tensors, 100 parameters
- **Additivity Verification:**  
  Analytical decomposition $g \approx g^{(A)} + g^{(B)}$ was verified numerically across all 88 evaluations:
  $$\max \frac{|g_G - (g_G^{(A)} + g_G^{(B)})|}{\|g_G\|_2} = 5.86 \times 10^{-4} \quad (<0.06\%)$$
  $$\max \frac{|g_T - (g_T^{(A)} + g_T^{(B)})|}{\|g_T\|_2} = 5.70 \times 10^{-4} \quad (<0.06\%)$$

---

## 3. Gradient Norm Scales Across Mask Conditions

Evaluated across the 8 representative windows on shared U-Net parameters ($\theta$):

| Mask Condition | Median $\|g_\mu\|_2$ | Median $\|g_G\|_2$ | Median $\|g_T\|_2$ | Median $R_G = \frac{\|g_G\|}{\|g_\mu\|}$ | Median $R_T = \frac{\|g_T\|}{\|g_\mu\|}$ | Median $\frac{\|g_G\|}{\|g_T\|}$ | Path A/B (G) | Path A/B (T) | Candidate $s_T$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Conditions** | **1.2272** | **195.29** | **234.49** | **197.38** | **261.84** | **0.55** | **3.46** | **24.05** | **0.0038** |
| `spatial_random` | 0.2288 | 158.42 | 239.56 | 836.34 | 1242.85 | 0.76 | 7.02 | 14.47 | 0.0009 |
| `spatial_grid` | 0.4076 | 288.62 | 127.97 | 725.10 | 314.00 | 1.63 | 2.97 | 5.27 | 0.0033 |
| `spatial_block` | 1.8384 | 227.18 | 321.43 | 163.91 | 317.13 | 0.52 | 3.28 | 27.55 | 0.0032 |
| `temporal_random` | 0.1774 | 208.52 | 309.84 | 1066.13 | 1759.00 | 0.64 | 11.85 | 34.69 | 0.0006 |
| `temporal_block` | 1.7005 | 179.99 | 363.36 | 109.09 | 219.09 | 0.50 | 4.44 | 34.32 | 0.0056 |
| `extrapolation_h1` | 0.2013 | 166.44 | 306.90 | 956.83 | 1846.60 | 0.49 | 4.52 | 40.16 | 0.0006 |
| `extrapolation_h6` | 0.8173 | 144.15 | 299.12 | 179.34 | 368.35 | 0.49 | 4.84 | 48.59 | 0.0027 |
| `extrapolation_h12` | 1.6441 | 140.40 | 281.33 | 99.24 | 198.20 | 0.50 | 4.28 | 34.26 | 0.0056 |
| `extrapolation_h18` | 2.7667 | 102.94 | 190.47 | 22.87 | 45.56 | 0.53 | 3.82 | 17.56 | 0.0329 |
| `extrapolation_h22` | 3.8744 | 107.13 | 55.45 | 20.29 | 15.50 | **3.86** | 0.76 | 1.11 | 0.0846 |
| `extrapolation_h23` | 4.8105 | 296.88 | 70.36 | 92.88 | 14.89 | **6.23** | 0.09 | 0.54 | 0.0674 |

### Breakdown by Network Sub-Component

| Network Group | Median $\|g_\mu\|_2$ | Median $\|g_G\|_2$ | Median $\|g_T\|_2$ | Median $R_G$ | Median $R_T$ | Median $\frac{\|g_G\|}{\|g_T\|}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Shared U-Net** | 1.2272 | 195.29 | 234.49 | 197.38 | 261.84 | 0.55 |
| **Encoder** | 0.8179 | 35.13 | 44.03 | 57.56 | 55.29 | 0.79 |
| **Attention** | 0.3132 | 4.47 | 5.65 | 21.70 | 18.20 | 0.86 |
| **Decoder** | 0.8459 | 188.41 | 216.57 | 259.83 | 324.02 | 0.55 |
| **Mean Head ($\phi$)** | 0.7087 | 194.12 | 290.52 | 335.84 | 444.12 | 0.52 |
| **Variance Head ($\psi$)**| 0.0000 | 2.0536 | 0.6970 | N/A | N/A | **3.20** |

---

## 4. Gradient Directional Alignment (Cosine Similarities)

Cosine similarities calculated on shared U-Net parameters ($\theta$):

| Relationship | Pair | Median Cosine | 16th Percentile | 84th Percentile | Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Path A vs MSE** | $\cos(g_\mu, g_G^{(A)})$ | **+0.0866** | +0.0242 | +0.2185 | Weakly aligned |
| | $\cos(g_\mu, g_T^{(A)})$ | **+0.0719** | +0.0205 | +0.1654 | Weakly aligned |
| **Path B vs MSE** | $\cos(g_\mu, g_G^{(B)})$ | **+0.0252** | -0.0152 | +0.0614 | Near-orthogonal |
| | $\cos(g_\mu, g_T^{(B)})$ | **+0.0465** | -0.0076 | +0.0883 | Near-orthogonal |
| **Full Joint vs MSE** | $\cos(g_\mu, g_G)$ | **+0.0900** | +0.0255 | +0.2173 | Positively aligned |
| | $\cos(g_\mu, g_T)$ | **+0.0710** | +0.0203 | +0.1648 | Positively aligned |
| **Within-NLL Internal** | $\cos(g_G^{(A)}, g_G^{(B)})$ | **+0.0314** | -0.0184 | +0.0872 | Independent / orthogonal |
| | $\cos(g_T^{(A)}, g_T^{(B)})$ | **+0.0206** | -0.0125 | +0.0631 | Independent / orthogonal |
| **Cross-Likelihood** | $\cos(g_G, g_T)$ | **+0.9641** | +0.9412 | +0.9855 | **Virtually collinear** |

---

## 5. Extreme Forecast Error Supervision ($h \in \{18, 22, 23\}$)

Analysis of residual difficulty ratio $r = (y-\mu)^2 / u$ on unobserved forecast frames across 125 test windows:

| Horizon | Channel | Median $r$ | p90 $r$ | p99 $r$ | Max $r$ | $\% r > 1$ | $\% r > 10$ | $\% r > 100$ | Min Theoretical Grad (G vs T) | Min Logit Grad (G vs T) | Top 1% Error Grad Share (G vs T) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$h=18$** | **Magnetic** | 0.289 | 2.01 | 6.53 | 108.7 | 22.5% | 0.2% | 0.00% | -53.9 vs -1.48 ($36\times$) | $-5.6\times 10^{-5}$ vs $-1.6\times 10^{-6}$ ($35\times$) | 27.1% vs 5.6% |
| | **Density** | 0.258 | 1.42 | 4.11 | 21.2 | 17.7% | 0.1% | 0.00% | -10.1 vs -1.41 ($7\times$) | $-1.1\times 10^{-5}$ vs $-1.5\times 10^{-6}$ ($7\times$) | 24.9% vs 6.0% |
| | **All** | 0.290 | 1.78 | 5.95 | 108.7 | 20.9% | 0.2% | 0.00% | -53.9 vs -1.48 ($36\times$) | $-5.6\times 10^{-5}$ vs $-1.6\times 10^{-6}$ ($36\times$) | 26.2% vs 5.5% |
| **$h=22$** | **Magnetic** | 1.052 | 9.72 | 49.25 | 1029.2 | 51.1% | 9.7% | 0.19% | -514.1 vs -1.50 ($343\times$) | $-5.6\times 10^{-4}$ vs $-1.6\times 10^{-6}$ ($341\times$) | 24.8% vs 3.2% |
| | **Density** | 0.793 | 5.40 | 29.54 | 174.8 | 44.5% | 4.5% | 0.03% | -86.9 vs -1.49 ($58\times$) | $-9.4\times 10^{-5}$ vs $-1.6\times 10^{-6}$ ($58\times$) | 19.7% vs 3.4% |
| | **All** | 1.040 | 8.82 | 47.01 | 1029.2 | 50.6% | 8.8% | 0.17% | -514.1 vs -1.50 ($343\times$) | $-5.6\times 10^{-4}$ vs $-1.6\times 10^{-6}$ ($341\times$) | 24.3% vs 3.2% |
| **$h=23$** | **Magnetic** | 1.772 | 21.69 | 108.38 | 995.8 | 56.7% | 18.9% | 1.29% | -497.4 vs -1.50 ($332\times$) | $-5.3\times 10^{-4}$ vs $-1.6\times 10^{-6}$ ($323\times$) | 28.9% vs 2.3% |
| | **Density** | 1.260 | 10.04 | 57.77 | 319.9 | 50.8% | 9.2% | 0.40% | -159.4 vs -1.49 ($107\times$) | $-1.7\times 10^{-4}$ vs $-1.6\times 10^{-6}$ ($106\times$) | 28.8% vs 2.7% |
| | **All** | 1.609 | 18.68 | 100.37 | 995.8 | 55.2% | 16.3% | 1.16% | -497.4 vs -1.50 ($332\times$) | $-5.3\times 10^{-4}$ vs $-1.6\times 10^{-6}$ ($323\times$) | 32.3% vs 2.4% |

---

## 6. Answers to Required Scientific Questions (Section 12)

### 1. Are Student-t NLL gradients substantially larger than reconstruction MSE gradients?
**YES.** On shared U-Net parameters, the median ratio is $R_T = \frac{\|g_T\|_2}{\|g_\mu\|_2} = \mathbf{261.84}$ (16th–84th percentile range: **$23.40$ to $1719.54$**). In the decoder, the ratio reaches $324.02$. Thus, under unnormalized multi-task training ($\lambda_v = 1$), NLL gradients exceed MSE gradients by more than two orders of magnitude.

### 2. Which pathway, A or B, contributes most strongly to NLL gradient magnitude?
**Pathway A (through predicted mean $\mu$) overwhelmingly dominates.** For Student-t NLL, the median norm ratio $\|g_T^{(A)}\|_2 / \|g_T^{(B)}\|_2$ is **$24.05$** (reaching $48.6\times$ at $h=6$). For Gaussian NLL, $\|g_G^{(A)}\|_2 / \|g_G^{(B)}\|_2 = \mathbf{3.46}$. Uncertainty-loss gradients act primarily as an adaptive reweighting of mean reconstruction errors on shared features rather than being driven by the variance head.

### 3. Does Path A align with reconstruction more strongly than Path B?
**YES.** Path A has a higher median cosine similarity with reconstruction MSE ($\cos = +0.0719$ for Student-t, $+0.0866$ for Gaussian) than Path B ($\cos = +0.0465$ for Student-t, $+0.0252$ for Gaussian). Furthermore, both paths are non-negative on average, confirming that the reconstruction degradation in 256-epoch training was not caused by conflicting gradient directions, but by the sheer magnitude of NLL gradient updates.

### 4. How much larger or smaller are Gaussian gradients compared with Student-t?
**On average, Gaussian gradients are moderately smaller ($\|g_G\|_2 / \|g_T\|_2 = 0.554$), but on extreme forecast horizons they become $4\times$ to $6\times$ LARGER.** At $h \le 18$, Gaussian gradients are roughly half the magnitude of Student-t because Student-t scale $s^2 = u/3$ increases the effective precision. However, at $h=22$ and $h=23$, Gaussian gradients surge to **$3.86\times$** and **$6.23\times$** Student-t gradients due to large outlier errors. Directly on the variance head ($\psi$), Gaussian gradients are **$3.20\times$ larger** than Student-t gradients.

### 5. Does Gaussian provide stronger supervision for extreme forecast errors?
**YES, by over two orders of magnitude.** At $h=22$ and $h=23$, where $55\%$ of voxels have $r > 1$ and up to $1.3\%$ have $r > 100$:
- Student-t theoretical upward variance gradient strictly saturates at **$-1.50$**.
- Gaussian theoretical upward variance gradient scales linearly ($r/2$), reaching **$-514.1$** ($343\times$ stronger).
- The actual logit gradient reaches $-5.57 \times 10^{-4}$ (Gaussian) vs $-1.63 \times 10^{-6}$ (Student-t) ($341\times$ stronger).
- Top 1% extreme error voxels contribute **$24.3\%$ to $32.3\%$** of the entire Gaussian upward gradient, but only **$2.3\%$ to $3.2\%$** in Student-t.

### 6. Is Gaussian likely to require additional gradient-scale normalization?
**YES.** Because Gaussian variance gradients scale linearly with squared residual error $r$, unnormalized Gaussian NLL produces massive gradient spikes on high-error windows ($R_G > 2000$). Without gradient-scale normalization, Gaussian NLL would cause severe optimization instability or gradient clipping during early Phase B when large errors occur.

### 7. What fixed normalization factors are suggested by the measured gradients?
Equalizing raw gradient norms ($\|g_{\text{NLL}}\|_2 \approx \|g_\mu\|_2$) suggests:
- Overall: $s_T = \frac{\|g_\mu\|}{\|g_T\|} \approx \mathbf{0.0038}$, and $s_G = \frac{\|g_\mu\|}{\|g_G\|} \approx \mathbf{0.0051}$.
- In the long-horizon forecast regime ($h=18$): $s_T \approx \mathbf{0.033}$, and $s_G \approx \mathbf{0.063}$.
- A candidate fixed scale factor of **$s \approx 0.01$ to $0.02$** would attenuate raw NLL gradients by $\sim 50\times$ to $100\times$, balancing reconstruction MSE and uncertainty learning.

### 8. Is a fixed scale adequate, or is there evidence supporting dynamic normalization?
**A fixed scale is PLAUSIBLE and recommended for initial trials.** While the empirical gradient ratio varies across mask conditions ($s_T \approx 0.0006$ on `temporal_random` to $0.085$ on `h22`), a fixed scale of $s \approx 0.01$–$0.02$ eliminates the dominant orders-of-magnitude discrepancy without introducing complex moving-average gradient estimators.

### 9. Does the evidence justify a fresh Gaussian A+B curriculum experiment?
**YES, but strictly contingent on gradient-scale normalization ($s_G \approx 0.01$–$0.02$).** The audit proves that Student-t saturation at $-1.50$ is directly responsible for the localized uncertainty suppression at $h=23$ (where $55\%$ of voxels have $r > 1$ but Student-t cannot push $u$ higher). Gaussian NLL provides the necessary non-saturating gradient drive ($340\times$ stronger on outliers) to resolve boundary suppression, provided its extreme gradient energy is normalized.

---

## 7. Artifact Manifest

The following files and figures have been generated and saved:
- Audit Report: [report.md](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/gaussian_vs_student_t_gradient_audit_20261118/report.md)
- Summary JSON: [diagnostic_summary.json](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/gaussian_vs_student_t_gradient_audit_20261118/diagnostic_summary.json)
- Gradient Norms CSV: [gradient_norms.csv](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/gaussian_vs_student_t_gradient_audit_20261118/gradient_norms.csv)
- Cosine Similarity CSV: [gradient_cosine_similarity.csv](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/gaussian_vs_student_t_gradient_audit_20261118/gradient_cosine_similarity.csv)
- Normalization Factor Summary CSV: [normalization_factor_summary.csv](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/gaussian_vs_student_t_gradient_audit_20261118/normalization_factor_summary.csv)
- Residual Tail Statistics CSV: [residual_tail_statistics.csv](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/gaussian_vs_student_t_gradient_audit_20261118/residual_tail_statistics.csv)
- Diagnostic Figures:
  - Figure 1: [figure1_gradient_norm_ratios.png](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/gaussian_vs_student_t_gradient_audit_20261118/figure1_gradient_norm_ratios.png)
  - Figure 2: [figure2_gradient_cosine_similarity.png](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/gaussian_vs_student_t_gradient_audit_20261118/figure2_gradient_cosine_similarity.png)
  - Figure 3: [figure3_extreme_horizon_gradient_behavior.png](file:///pscratch/sd/b/binxia/ai4plasma/evaluations/gaussian_vs_student_t_gradient_audit_20261118/figure3_extreme_horizon_gradient_behavior.png)

---

## 8. Final Audit Summary (Section 13)

**Gaussian vs Student-t Gradient Audit**

- Checkpoint: fresh gradual A+B epoch 256
- Evaluation: pass 0 only
- Number of paired windows: 8 (88 window-mask pairs)
- Median Student-t NLL / MSE gradient norm ratio: 261.84 (16-84%: 23.40 - 1719.54)
- Median Gaussian NLL / MSE gradient norm ratio: 197.38 (16-84%: 30.32 - 963.12)
- Gaussian / Student-t gradient norm ratio: 0.55 (16-84%: 0.48 - 2.20)
- Student-t Path A / Path B gradient norm ratio: 24.05
- Gaussian Path A / Path B gradient norm ratio: 3.46
- Student-t A versus MSE cosine similarity: +0.0719
- Student-t B versus MSE cosine similarity: +0.0465
- Gaussian A versus MSE cosine similarity: +0.0866
- Gaussian B versus MSE cosine similarity: +0.0252
- Student-t normalization factor estimate: 0.0038 (extrapolation h18: 0.0329)
- Gaussian normalization factor estimate: 0.0051 (extrapolation h18: 0.0625)
- Extreme-error gradient saturation: VERIFIED (Student-t saturates at -1.50; Gaussian reaches -514.1, 343x stronger)
- Fixed-scale normalization: PLAUSIBLE
- Gaussian training experiment justified: YES (requires gradient-scale normalization s ~ 0.01-0.02)
- Recommended next experiment: Fresh three-phase curriculum training with Gaussian NLL and gradient-scale normalization (s=0.01 to 0.02)
- Existing source code modified: NO
- Model parameters updated: NO
- Training launched: NO
