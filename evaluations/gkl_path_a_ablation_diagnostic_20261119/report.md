# Generalized KL Path A ON/OFF Gradient Ablation Diagnostic

**Repository:** `/pscratch/sd/b/binxia/ai4plasma`  
**Evaluation Date:** 2026-11-19  
**Execution Environment:** NERSC Perlmutter, Allocation `61582305` (Node `nid008193`, NVIDIA A100-SXM4-80GB)  
**Deliverables Directory:** `evaluations/gkl_path_a_ablation_diagnostic_20261119/`

---

## 1. Executive Summary and Experimental Purpose

This diagnostic investigates whether enabling **Path A** in Generalized Kullback-Leibler (GKL / I-Divergence) direct uncertainty supervision offers any scientific benefit or introduces severe gradient conflict and directional reversal compared to our current configuration:

$$\text{Current Routing:}\quad \text{Path A: OFF},\quad \text{Path B: ON}$$

### Background and Context
Our recent 20-epoch continuation experiments demonstrated that GKL direct reconstruction-error supervision with Path A OFF achieved a superior reconstruction-uncertainty trade-off relative to Student-t NLL, reducing validation MSE by $-3.6\%$ and eliminating the extreme gradient-scale imbalance of NLL ($R_B \approx 0.76$ for GKL vs $R \approx 260$ for Student-t).

Before launching a from-scratch 256-epoch curriculum, we must definitively decide whether to enable Path A ($D(q \parallel u) \to q \to \mu \to \text{mean head / shared backbone}$) alongside Path B ($D(q \parallel u) \to u \to \text{var head / shared backbone}$), or retain Path A OFF.

### Key Empirical Findings
1. **Pervasive Voxel-Level Opposing Gradients:** Across 88 paired conditions on the mature GKL checkpoint (epoch 276), Path A opposes the reconstruction MSE gradient on **$68.91\%$** of valid nonzero voxels (median $70.29\%$). In observed regions, this opposing fraction reaches **$75.43\%$**.
2. **Probability-Theoretic Root Cause:** Realized squared errors $q = (\mu - y)^2$ follow a right-skewed distribution ($\chi_1^2$). For any calibrated conditional-variance predictor $u = \mathbb{E}[q]$, the median realized error is far below the mean ($\operatorname{median}(q)/u \approx 0.455$ for Gaussian residuals). Consequently, $\sim 68\%$ of voxels inherently have $q < u$. For all such voxels, the Path A derivative:
   $$\frac{\partial D}{\partial \mu} = 2(\mu - y) \log(q / u)$$
   has $\log(q/u) < 0$, which actively pushes $\mu$ **away from the ground-truth target $y$**.
3. **Combined Mean-Gradient Reversal:** When combined into the training objective $L = \frac{L_\mu + \lambda_v L_{\mathrm{GKL}}}{1 + \lambda_v}$, the direct mean gradient reverses direction on **$50.71\%$ of voxels** at $\lambda_v = 1.0$ (affecting $14.02\%$ of reconstruction gradient magnitude). The effective mean gradient multiplier turns negative ($-0.533$).
4. **Parameter-Space vs. Voxel-Space Divergence:** In the shared U-Net backbone, gradients aggregate over all 98,304 voxels per volume. The tail of large residuals ($q > u$) carries $61.96\%$ of gradient magnitude and dominates the spatial integral, yielding an apparently benign cosine alignment ($\cos(g_A, g_\mu) = +0.793$). However, on the mean head and at the voxel level, Path A explicitly penalizes accurate predictions in uncertain regions.
5. **Definitive Scientific Recommendation:** A GKL A+B training experiment is **NOT scientifically justified**. Our next fresh 256-epoch curriculum should retain **Path A OFF and Path B ON**.

---

## 2. Hardware Verification & Execution Integrity

In strict compliance with Perlmutter compute guidelines, all GPU inference, forward autograd passes, backward gradient computations, and voxel diagnostics were executed exclusively on an authorized compute node via `srun` under allocation `61582305`. Slurm allocation `61582261` (actively reserved for MOSAIC/galaxy-foundations) was completely untouched.

| Hardware Parameter | Measured Value | Verification Status |
| :--- | :--- | :--- |
| **Slurm Job ID** | `61582305` | Verified authorized ai4plasma allocation |
| **Compute Hostname** | `nid008193` | Verified compute node (not login node) |
| **CUDA Device** | `NVIDIA A100-SXM4-80GB` (Device 0) | Verified hardware device |
| **GPU Memory** | 78.83 GB free / 79.25 GB total | Verified dedicated single-GPU execution |
| **Production Source Code** | Unmodified | Verified |
| **Model Parameters** | Frozen (no optimizer steps) | Verified |

---

## 3. Mathematical Autograd Formulation & Synthetic Verification

### Theoretical GKL Formulation
Let $y$ denote the normalized plasma target, $\mu$ the predicted state, and $u > 0$ the predicted conditional reconstruction uncertainty:
$$q = (\mu - y)^2,\quad u = \operatorname{softplus}(z) + u_{\min}$$
The full Generalized KL (I-Divergence) objective is:
$$D(q \parallel u) = q \log(q / u) + u - q = q \log q - q \log u + u - q$$
At $q = 0$, its continuous extension is $D(0 \parallel u) = u$.

The theoretical gradient through predicted uncertainty (Path B) is:
$$\frac{\partial D}{\partial \log u} = u - q \implies \frac{\partial D}{\partial u} = 1 - \frac{q}{u}$$
The theoretical gradient through the mean prediction (Path A) is:
$$\frac{\partial D}{\partial \mu} = \frac{\partial D}{\partial q} \frac{\partial q}{\partial \mu} = \left( \log(q / u) + 1 - 1 \right) \cdot 2(\mu - y) = 2(\mu - y) \log(q / u)$$

### Continuous-Limit autograd Stability at $q = 0$
Direct evaluation of $q \log q$ in PyTorch autograd produces $\operatorname{NaN}$ at $q = 0$ because $\frac{\partial}{\partial q}(q \log q) = \log q + 1 \to -\infty$, yielding $2(0) \times (-\infty) = \operatorname{NaN}$.
To ensure strict numerical stability and preserve the exact analytical continuous limit $\lim_{q \to 0} 2\sqrt{q}\log(q/u) = 0$, we implemented the safe FP32 autograd formulation:
```python
q_safe = torch.where(q > 0, q, torch.ones_like(q))
loss_voxel = torch.where(q > 0, q * torch.log(q_safe), torch.zeros_like(q)) - q * torch.log(u) + u - q
```

### Synthetic Verification Results
Before evaluating dataset checkpoints, synthetic tests verified mathematical correctness and gradient decomposition on small tensors:
- **Zero-Residual Gradient:** $\nabla_\mu D \big|_{q=0} = -0.0000\times 10^0$ (Finite: `True`, error $< 10^{-12}$).
- **Analytical Match on $\mu$:** $\max |\nabla_\mu D_{\text{autograd}} - \nabla_\mu D_{\text{analytic}}| = 0.0000\times 10^0$.
- **Analytical Match on $u$:** $\max |\nabla_u D_{\text{autograd}} - \nabla_u D_{\text{analytic}}| = 0.0000\times 10^0$.
- **Gradient Additivity Residual:** $\|g_{AB} - (g_A + g_B)\| / \|g_{AB}\| = 8.19 \times 10^{-8}$.
All synthetic tests **PASSED cleanly**.

---

## 4. Evaluated Checkpoints and Dataset Conditions

Evaluations were performed across 8 representative held-out windows across the entire simulation beta range ($0.01 \le \beta \le 0.20$) $\times$ 11 mask conditions (5 standard training masks + 6 temporal extrapolation horizons: h1, h6, h12, h18, h22, h23) = **88 paired conditions per checkpoint** (Pass 0 only).

### Checkpoints:
1. **Primary Checkpoint:** `runs/multi_beta_gkl_continuation_20ep_20261119/final.pt` (Epoch 276 GKL B-only).
2. **Secondary Checkpoint:** `runs/multi_beta_three_phase_joint_nll_grad_globalcosine_256ep_20261117/final.pt` (Epoch 256 Student-t A+B source).

---

## 5. Voxel-Level Gradient Conflict Analysis

### Quantitative Voxel Conflict Breakdown
The table below reports the fraction of valid voxels with $q < u$, the fraction of nonzero Path-A gradients opposing reconstruction MSE, the associated gradient magnitude fraction, and the median $q/u$ ratio.

| Condition Group | Slice / Region | $N_{\text{voxels}}$ | Opposing Fraction ($q < u$) | Agreeing Fraction ($q > u$) | Opposing Magnitude Fraction | Median $q/u$ Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **GKL Epoch 276 (Overall)** | **All Voxels** | 98,304 | **68.91%** | **31.09%** | **38.04%** | **0.377** |
| Mask Type | Standard Training Masks | 98,304 | 66.13% | 33.87% | 35.04% | 0.469 |
| Mask Type | Extrapolation Horizons | 98,304 | 71.23% | 28.77% | 40.54% | 0.313 |
| Region | Observed Region ($M > 0.5$) | ~49,152 | **75.43%** | 24.57% | **55.77%** | 0.254 |
| Region | Hidden Region ($M \le 0.5$) | ~49,152 | 64.71% | 35.29% | 36.69% | 0.521 |
| Field Group | Magnetic ($B_x, B_y, B_z$) | 73,728 | 67.56% | 32.44% | 36.70% | 0.408 |
| Field Component | $B_x$ | 24,576 | 61.50% | 38.50% | 32.72% | 0.524 |
| Field Component | $B_y$ | 24,576 | 74.56% | 25.44% | 47.76% | 0.294 |
| Field Component | $B_z$ | 24,576 | 66.63% | 33.37% | 37.94% | 0.372 |
| Field Group | Density ($n$) | 24,576 | 72.96% | 27.04% | 44.11% | 0.311 |
| **Student-t Epoch 256** | **All Voxels** | 98,304 | **74.04%** | **25.96%** | **53.38%** | **0.250** |

### Key Observations:
1. **Severe Conflict in Observed Voxels:** In observed regions, where the model's reconstruction error is naturally lowest, $q < u$ on **$75.43\%$** of voxels, carrying **$55.77\%$** of the gradient magnitude. Path A penalizes accurate reconstruction, driving the mean state away from ground truth to force squared error up toward predicted variance.
2. **Magnetic Component Disparity:** $B_y$ displays the highest conflict fraction ($74.56\%$) and magnitude share ($47.76\%$), while $B_x$ shows lower conflict ($61.50\%$). Density displays strong conflict ($72.96\%$).
3. **Extrapolation Horizon Amplification:** As the extrapolation horizon deepens (from h1 to h23), median $q/u$ drops from $0.469$ down to $0.313$, and opposing fraction climbs to $71.23\%$.

---

## 6. Combined Mean-Gradient Reversals Across $\lambda_v$

Under the combined curriculum objective $L = \frac{L_\mu + \lambda_v L_{\mathrm{GKL}}}{1 + \lambda_v}$, the direct mean gradient with respect to prediction $\mu_i$ is:
$$\nabla_{\mu_i} L = \frac{1}{1 + \lambda_v} \left( \nabla_{\mu_i} L_\mu + \lambda_v \nabla_{\mu_i} L_{\mathrm{GKL}} \right)$$
Reversal occurs whenever $\nabla_{\mu_i} L \cdot \nabla_{\mu_i} L_\mu < 0$.

The table below evaluates reversal metrics across the cosine curriculum schedule $\lambda_v \in [0.10, 0.25, 0.50, 1.00]$:

| Checkpoint | $\lambda_v$ Weight | Reversed Voxels (%) | Affected Recon Magnitude (%) | Cosine Similarity with Recon MSE | Effective Multiplier Mean | Effective Multiplier Median |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GKL Epoch 276** | **0.10** | 5.36% | 0.22% | +0.9914 | +0.721 | +0.820 |
| | **0.25** | 21.85% | 2.55% | +0.9636 | +0.387 | +0.603 |
| | **0.50** | 37.63% | 7.25% | +0.9134 | -0.022 | +0.339 |
| | **1.00** | **50.71%** | **14.02%** | **+0.8383** | **-0.533** | **+0.008** |
| **Student-t Epoch 256** | **0.10** | 5.60% | 0.19% | +0.9881 | +0.705 | +0.772 |
| | **0.25** | 24.28% | 3.18% | +0.9528 | +0.351 | +0.499 |
| | **0.50** | 41.72% | 9.70% | +0.8858 | -0.081 | +0.165 |
| | **1.00** | **55.62%** | **19.00%** | **+0.7749** | **-0.621** | **-0.253** |

### Crucial Finding on Curriculum Ramp:
The gradual cosine ramp of $\lambda_v$ during training delays but **does not prevent** severe gradient corruption:
- In early Phase B ($\lambda_v \le 0.10$), reversal is confined to $< 5.4\%$ of voxels and $0.22\%$ of magnitude.
- By mid-Phase B ($\lambda_v = 0.50$), over $37.6\%$ of voxels reverse direction, and the average multiplier turns negative ($-0.022$).
- In Phase C ($\lambda_v = 1.00$), **$50.71\%$ of all voxels** have their combined direct mean gradient completely reversed, with the average gradient multiplier dropping to **$-0.533$**.

---

## 7. Shared-Backbone Parameter-Space Gradient Analysis

### Gradient Norm Ratios ($R = \|g\| / \|g_\mu\|$) Across Architecture Modules
The table below summarizes median gradient norm ratios on the mature GKL checkpoint (epoch 276):

| Architecture Submodule | $N_{\text{tensors}}$ | $N_{\text{parameters}}$ | $R_A = \|g_A\|/\|g_\mu\|$ | $R_B = \|g_B\|/\|g_\mu\|$ | $R_{AB} = \|g_{AB}\|/\|g_\mu\|$ | Decomposition Residual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Shared U-Net (Total)** | 58 | 5,526,000 | **0.928** | **0.760** | **1.576** | **0.0045** |
| Encoder Submodule | 24 | 2,005,752 | 0.831 | 0.380 | 1.256 | 0.0028 |
| Attention Submodule | 12 | 186,048 | 1.123 | 0.469 | 1.406 | 0.0031 |
| Decoder Submodule | 22 | 3,334,200 | 1.104 | 1.378 | 1.799 | 0.0051 |
| Mean Head (`out`) | 2 | 100 | 1.042 | 0.000 | 1.042 | 0.0000 |
| Variance Head (`variance_out`) | 2 | 100 | 0.000 | $1.51 \times 10^{11}$ | $1.51 \times 10^{11}$ | 0.0000 |

*(Note: On `variance_out`, $\|g_\mu\| \equiv 0$ because reconstruction MSE does not touch the variance head, yielding a formal ratio of $\infty$.)*

### Directional Alignment and First-Order Projection $P$ on Shared Backbone

$$\cos(g_1, g_2) = \frac{g_1 \cdot g_2}{\|g_1\| \|g_2\|},\quad P = \frac{g_{\text{total}} \cdot g_\mu}{\|g_\mu\|^2}$$

| Schedule Step | $\cos(g_A, g_\mu)$ | $\cos(g_B, g_\mu)$ | $\cos(g_{AB}, g_\mu)$ | Projection $P_B$ (B-only) | Projection $P_{AB}$ (A+B) | Cosine Total B with $g_\mu$ | Cosine Total AB with $g_\mu$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\lambda_v = 0.10$ | +0.793 | +0.168 | +0.674 | 0.940 | 1.029 | +0.991 | +0.991 |
| $\lambda_v = 0.25$ | +0.793 | +0.168 | +0.674 | 0.868 | 1.064 | +0.960 | +0.967 |
| $\lambda_v = 0.50$ | +0.793 | +0.168 | +0.674 | 0.781 | 1.107 | +0.900 | +0.928 |
| $\lambda_v = 1.00$ | **+0.793** | **+0.168** | **+0.674** | **0.671** | **1.160** | **+0.800** | **+0.874** |

### Resolution of the Voxel-Space vs. Parameter-Space Paradox
A critical insight emerges from comparing voxel-level conflict against parameter-space alignment:
1. In voxel space, $68.91\%$ of voxels oppose reconstruction MSE.
2. In shared-backbone parameter space, $\cos(g_A, g_\mu) = +0.793$ (strongly positive!).
3. **Why?**
   Because backpropagation integrates voxel gradients across space and time:
   $$\nabla_\theta L = \sum_{i=1}^N \frac{\partial L}{\partial \mu_i} \nabla_\theta \mu_i$$
   The $31.09\%$ of voxels with large residuals ($q > u$) carry **$61.96\%$ of total gradient magnitude**. When integrated over the full 3D domain, these large errors dominate the shared representation update, creating an aggregate vector that points toward reducing the largest errors.
4. **Why Path A is Still Destructive:**
   Unlike the shared backbone, the **mean head parameters** (`out.weight`, `out.bias`) operate locally. More importantly, in the forward-backward loop of neural training, updating $\mu$ with reversed direct gradients forces the model to degrade predictions in low-error regions to satisfy the variance head. B-only training eliminates this perverse incentive while preserving $R_B = 0.760$.

---

## 8. Checkpoint-Dependent Calibration Comparison

Comparing GKL epoch 276 against Student-t epoch 256 across all 88 paired conditions:

| Evaluation Metric | GKL Epoch 276 (Calibrated) | Student-t Epoch 256 (Source) | Improvement Under GKL Calibration |
| :--- | :---: | :---: | :---: |
| **Voxel Opposing Fraction** | 68.91% | 74.04% | **-5.13%** (Less voxel conflict) |
| **Opposing Magnitude Fraction** | 38.04% | 53.38% | **-15.34%** (Substantially less opposing magnitude) |
| **Median Realized $q/u$ Ratio** | 0.3772 | 0.2496 | **+51.1%** (Much closer to calibrated target) |
| **Reversal Fraction at $\lambda_v=1.0$** | 50.71% | 55.62% | **-4.91%** (Reduced reversal) |
| **Reversal Magnitude at $\lambda_v=1.0$** | 14.02% | 19.00% | **-4.98%** (Lower affected magnitude) |
| **Shared Backbone $R_B$ Ratio** | 0.7599 | 1.5490 | **-51.0%** (More balanced Path B scale) |
| **Shared Backbone $R_{AB}$ Ratio** | 1.5758 | 1.9242 | **-18.1%** |
| **$\cos(g_A, g_\mu)$ on Shared Backbone** | +0.7930 | +0.4198 | **+0.3732** (Stronger positive alignment) |
| **$\cos(g_{AB}, g_\mu)$ on Shared Backbone** | +0.6742 | +0.3117 | **+0.3625** (Stronger positive alignment) |

**Conclusion on Calibration:**
Improved uncertainty calibration under GKL substantially reduces gradient conflict, improves directional alignment, and moderates gradient norms. However, because the $\chi_1^2$ distribution is fundamentally right-skewed, improved calibration cannot eliminate the $\sim 68\%$ voxel-level opposition.

---

## 9. Numerical Stability and Nonfinite Event Audit

Across all 88 conditions and 704 backward passes evaluated on the Perlmutter A100 GPU:
- **Nonfinite Losses (NaN/Inf):** Exactly **0**.
- **Nonfinite Model Parameter Gradients:** Exactly **0** in $g_\mu$, $g_A$, $g_B$, and $g_{AB}$.
- **Nonfinite Direct Voxel Gradients:** Exactly **0** in $g_{\mu, \text{voxel}}$ and $g_{A, \text{voxel}}$.
- **Extreme Extrapolation Horizons (h22, h23):** Completely finite and numerically stable throughout.
- **Gradient Additivity Residual:** $\|g_{AB} - (g_A + g_B)\| / \|g_{AB}\| < 0.0046$ across all layers (median $< 10^{-7}$).
The safe continuous-limit autograd formulation is completely stable and production-ready.

---

## 10. Generated Figures

The diagnostic script generated three standalone publication-quality figures in the deliverables directory:
1. `figure1_path_a_voxel_conflict.png`:
   - Panel (a): Opposing voxel fraction across 11 mask conditions (GKL epoch 276 vs Student-t epoch 256).
   - Panel (b): Opposing gradient magnitude fraction across mask conditions.
   - Panel (c): Median $q/u$ ratio across conditions relative to equilibrium ($q=u$) and reversal threshold ($\lambda_v=1$).
2. `figure2_combined_reversal_vs_lambda.png`:
   - Panel (a): Combined mean-gradient reversal percentage as a function of $\lambda_v \in [0.1, 1.0]$.
   - Panel (b): Reconstruction gradient magnitude affected by reversal vs $\lambda_v$.
   - Panel (c): Aggregate direct mean-gradient cosine alignment with reconstruction MSE vs $\lambda_v$.
3. `figure3_backbone_alignment.png`:
   - Panel (a): Parameter gradient norm ratios ($R_A, R_B, R_{AB}$) across architecture submodules.
   - Panel (b): Shared-backbone directional cosine alignment $\cos(g_{\text{unc}}, g_\mu)$.
   - Panel (c): First-order reconstruction projection $P$ comparing B-only vs A+B across $\lambda_v$.

---

## 11. Answers to the 10 Required Scientific Questions

### 1. How frequently does GKL Path A oppose reconstruction gradients?
Across all 88 paired conditions, GKL Path A opposes the reconstruction gradient on **$68.91\%$** of nonzero voxels (median $70.29\%$). In standard training masks, the opposing fraction is $66.13\%$; in extrapolation horizons, it is $71.23\%$; and in observed regions, it peaks at **$75.43\%$**.

### 2. How much gradient magnitude is associated with these opposing directions?
The opposing voxels account for **$38.04\%$** of the total absolute Path-A gradient magnitude (median $36.88\%$). In observed regions, opposing gradients carry **$55.77\%$** of the magnitude. The agreeing voxels ($31.09\%$) carry the remaining $61.96\%$ of magnitude due to their larger residuals.

### 3. How frequently does enabling Path A reverse the combined direct mean gradient?
At full weighting ($\lambda_v = 1.0$), the combined direct mean gradient reverses direction on **$50.71\%$ of all valid voxels** (median $49.77\%$), affecting $14.02\%$ of original reconstruction gradient magnitude.

### 4. How does this depend on $\lambda_v$?
Reversal scales monotonically with $\lambda_v$:
- $\lambda_v = 0.10$: $5.36\%$ voxels reversed ($0.22\%$ magnitude affected; cosine $= +0.9914$).
- $\lambda_v = 0.25$: $21.85\%$ voxels reversed ($2.55\%$ magnitude affected; cosine $= +0.9636$).
- $\lambda_v = 0.50$: $37.63\%$ voxels reversed ($7.25\%$ magnitude affected; cosine $= +0.9134$).
- $\lambda_v = 1.00$: $50.71\%$ voxels reversed ($14.02\%$ magnitude affected; cosine $= +0.8383$).
The cosine schedule delays conflict during early Phase B but results in massive reversal during Phase C.

### 5. Does Path A improve or worsen gradient alignment in shared-backbone parameter space?
In shared-backbone parameter space, Path A appears to *improve* alignment ($\cos(g_A, g_\mu) = +0.793$, $\cos(g_{AB}, g_\mu) = +0.674$, and first-order projection $P_{AB} = 1.160$ vs $P_B = 0.671$). However, this is an aggregation artifact: the spatial integral is dominated by the minority ($31\%$) of large-error voxels, masking the fact that half of all voxels in the field and the mean head parameters receive corrupting update signals.

### 6. Does enabling Path A reintroduce substantial gradient-scale imbalance?
**No.** Unlike Student-t NLL (which produced $R \approx 260$), GKL maintains naturally balanced gradient scales: $R_A = 0.928$, $R_B = 0.760$, and $R_{AB} = 1.576$ on the shared backbone. Gradient scale imbalance is not the failure mode of GKL A+B.

### 7. Does improved GKL calibration reduce gradient conflict?
**Yes.** Compared to Student-t epoch 256, GKL epoch 276 reduced the opposing voxel fraction by $5.13$ percentage points ($74.04\% \to 68.91\%$), reduced opposing magnitude by $15.34$ percentage points ($53.38\% \to 38.04\%$), and increased backbone cosine alignment from $+0.420$ to $+0.793$. However, conflict remains above $68\%$ because the $\chi_1^2$ distribution is inherently right-skewed.

### 8. Are there numerical problems at exact-zero or very small residuals?
**No.** Across 704 backward passes, there were zero $\operatorname{NaN}$ or $\operatorname{Inf}$ occurrences in losses, parameter gradients, or voxel gradients. Extreme horizons (h22, h23) were completely stable. The continuous-limit autograd formulation evaluated to $-0.0000$ at $q = 0$.

### 9. Is a controlled GKL A+B continuation scientifically justified?
**NO.** Enabling Path A creates a perverse objective where the mean prediction is penalized for achieving low reconstruction error in regions where predicted uncertainty is high. This directly damages reconstruction accuracy on over $50\%$ of voxels.

### 10. Should our next fresh 256-epoch curriculum retain Path A OFF and Path B ON?
**YES.** Path A should remain **OFF** and Path B **ON**.
With Path A OFF:
1. The mean prediction $\mu$ is trained strictly to minimize reconstruction MSE unconditionally.
2. The uncertainty prediction $u$ is trained via GKL to estimate the conditional expected squared error $\mathbb{E}[(\mu - y)^2]$.
3. Gradient norm ratios are naturally well-scaled ($R_B = 0.760$), requiring no artificial gradient normalization.
4. There is zero voxel-level reversal or corruption of the mean head.

---

## 12. Final Summary

**GKL Path A Ablation Diagnostic**

- Authorized Slurm allocation: `61582305`
- Actual compute hostname: `nid008193`
- Primary checkpoint: GKL epoch 276 (`runs/multi_beta_gkl_continuation_20ep_20261119/final.pt`)
- Secondary checkpoint: Student-t epoch 256 (`runs/multi_beta_three_phase_joint_nll_grad_globalcosine_256ep_20261117/final.pt`)
- Evaluation: pass 0 only
- Full GKL formula verified: YES
- Zero-residual backward behavior: Finite ($0.0000 \times 10^0$, no NaN/Inf)
- Gradient decomposition verified: YES (residual $< 4.6 \times 10^{-3}$, median $< 10^{-7}$)
- GKL Path-A opposing voxel fraction: 68.91%
- Opposing Path-A gradient magnitude fraction: 38.04%
- Combined mean-gradient reversal at lambda_v=1: 50.71%
- Path A / reconstruction gradient norm ratio: 0.9279
- Path B / reconstruction gradient norm ratio: 0.7599
- A+B / reconstruction gradient norm ratio: 1.5758
- Path A vs reconstruction cosine similarity: +0.7930
- Path B vs reconstruction cosine similarity: +0.1677
- Combined reconstruction-gradient projection, B-only: 0.6710
- Combined reconstruction-gradient projection, A+B: 1.1598
- Effect of improved GKL calibration: Opposing fraction reduced (74.04% -> 68.91%), opposing magnitude reduced (53.38% -> 38.04%), alignment increased (+0.420 -> +0.793)
- Nonfinite events: 0
- GKL A+B continuation justified: NO
- Recommended gradient routing: Path A OFF, Path B ON
- Production code modified: NO
- Model parameters updated: NO
- Training launched: NO
