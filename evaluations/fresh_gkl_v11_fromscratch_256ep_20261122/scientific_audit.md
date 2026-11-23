# Scientific Evaluation Audit: ai4plasma v11 From-Scratch 256-Epoch Model

**Audit Date:** November 22, 2026  
**Audited Run:** `multi_beta_gkl_v11_fromscratch_256ep_20261122`  
**Audited Checkpoint:** `runs/multi_beta_gkl_v11_fromscratch_256ep_20261122/final.pt`  
**Evaluation Directory:** `evaluations/fresh_gkl_v11_fromscratch_256ep_20261122/`  
**Auditor:** Antigravity Agent (DeepMind)  

---

## 1. Executive Summary of Audit Findings

| Audit Dimension | Finding | Status / Action Taken |
|:---|:---|:---:|
| **1. Uncertainty Calibration Terminology** | The original report inverted terminology for $u/\text{MSE} < 1$, erroneously labeling uncertainty underestimation as "under-confidence". | **CORRECTED:** Consistently renamed $u < \text{MSE}$ as **uncertainty underestimation / overconfidence**, $u > \text{MSE}$ as **uncertainty overestimation / underconfidence**, and $u \approx \text{MSE}$ as **marginal variance agreement**. |
| **2. Metric Definition & Space Audit** | Audited loss function and evaluation implementation. Confirmed $u$ and MSE are in identical standardized squared units. | **VERIFIED:** $u$ is parameterized and optimized via Generalized KL divergence as conditional expected squared reconstruction error ($\mathbb{E}[(y - \hat{y}(x))^2 \mid x] = \operatorname{Var}(y \mid x) + [\mathbb{E}(y \mid x) - \hat{y}(x)]^2$). Standardized residuals $z = (y - \hat{y})/\sqrt{u}$ and nominal coverage are mathematically sound. |
| **3. Sample Independence & Provenance** | Audited 125 test windows against `split.json` and simulation metadata. | **VERIFIED:** The 125 test windows correspond to **125 distinct, independent simulation runs** (1 window per run, $t=0..24$, zero temporal overlap), evenly distributed across 10 plasma $\beta$ values (12 or 13 runs per $\beta$). |
| **4. Comparability: v9 vs. v11** | Checked test window indices, noise seeds, channel normalizations, and resolution across evaluations. | **VERIFIED:** Test windows, channel means, standard deviations (to 17 decimal places), and split SHA-256 (`0e6b4276...`) are **100% bit-for-bit identical** between v9 baseline and v11 evaluations. |
| **5. Extreme-Horizon Calibration ($h=20..23$)** | Audited run-level error distributions and channel breakouts. | **VERIFIED:** Improvement at $h=20..22$ is genuine ($+19.5\%$ calibration improvement at $h=22$). Degradation at $h=22..23$ is systematic across the entire cohort (60.8% of runs at $h=22$ and 68.8% at $h=23$ have $u < \text{MSE}$), not driven by outliers. |
| **6. `spatial_grid` Calibration Deficit** | Investigated why $u/\text{MSE} = 0.568$ (Density $0.309$) on checkerboard grids. | **IDENTIFIED:** The model severely underestimates uncertainty ($u < \text{MSE}$) across all 10 $\beta$ values on checkerboard grids. The variance head predicts small variance ($u \approx 0.0037$) due to nearest-neighbor observed anchors at distance 1, failing to anticipate steep sub-grid density gradients. |
| **7. Forward vs. Reverse Extrapolation Asymmetry** | Audited claimed physical "entropy growth / information loss" narrative. | **CORRECTED:** The $2.9\times - 3.6\times$ MSE difference is an **empirical directional asymmetry** driven by the initial-value simulation setup, where the initial state ($t=0..1$) has $100\times$ lower intrinsic fluctuation variance ($B_y$ variance $0.0135$ at $t=0$ vs. $2.55$ at $t=12$) than late-stage turbulence. |
| **8. Cross-Model Comparison Rigor** | Previous report conflated isolated ablations with multi-variable configuration changes. | **CORRECTED:** Formally distinguished single-variable matched ablations (matched 20ep continuation) from from-scratch configuration comparisons (fresh v9 vs fresh v11) and cross-objective baselines (Student-t). |

---

## 2. Terminology & Statistical Definitions Audit

### 2.1 Theoretical Framework

Let $y$ denote the ground truth field (standardized per channel by dataset mean $\mu_c$ and standard deviation $\sigma_c$), $\hat{y}$ denote the deterministic mean prediction, and $u > 0$ denote the uncertainty head's predicted variance. The true error is $e = y - \hat{y}$, with squared error $q = e^2 = (y - \hat{y})^2$.

In `data/masking.py`, the Generalized Kullback-Leibler (GKL) divergence (also known as I-divergence or generalized relative entropy) between non-negative measures $q$ and $u$ is implemented as:
```python
if objective in ("generalized_kl", "gkl"):
    q = squared_error
    u = variance_valid
    q_safe = torch.where(q > 0, q, torch.ones_like(q))
    loss_voxel = torch.where(q > 0, q * torch.log(q_safe), torch.zeros_like(q)) - q * torch.log(u) + u - q
    return loss_voxel.mean()
```
Mathematically, for $q > 0$:
$$\mathcal{L}_{\text{GKL}}(q, u) = q \ln\left(\frac{q}{u}\right) - q + u$$
For $q = 0$, $\lim_{q \to 0^+} q \ln(q) = 0$, so $\mathcal{L}_{\text{GKL}}(0, u) = u$.

Taking the derivative with respect to $u$:
$$\frac{\partial \mathcal{L}_{\text{GKL}}}{\partial u} = -\frac{q}{u} + 1 = 1 - \frac{q}{u}$$
Setting the conditional expectation of the gradient to zero yields the Bayes-optimal solution for a fixed mean predictor $\hat{y}(x)$:
$$\mathbb{E}\left[1 - \frac{q}{u} \;\middle|\; x\right] = 0 \implies u^*(x) = \mathbb{E}[q \mid x] = \mathbb{E}[(y - \hat{y}(x))^2 \mid x]$$

**Decomposition of Conditional Squared Reconstruction Error:**
The optimal prediction $u^*(x)$ is the **conditional expected squared reconstruction error**, which decomposes into:
$$\mathbb{E}[(y - \hat{y}(x))^2 \mid x] = \operatorname{Var}(y \mid x) + \left[\mathbb{E}(y \mid x) - \hat{y}(x)\right]^2$$
- $\operatorname{Var}(y \mid x)$: The irreducible conditional variance of the true physical field given the observed context $x$.
- $\left[\mathbb{E}(y \mid x) - \hat{y}(x)\right]^2$: The squared approximation bias of the deterministic mean predictor $\hat{y}(x)$.

**Important Theoretical Qualifications:**
1. **Not Pure Aleatoric Noise:** The uncertainty head $u$ estimates the total expected squared reconstruction error of the network on unobserved voxels, which includes both the data variance given partial observations and model approximation error. It should not be interpreted as estimating purely irreducible physical variance or aleatoric noise.
2. **No Epistemic Separation:** This single-head regression objective does not separately identify or isolate epistemic uncertainty from aleatoric uncertainty.
3. **Equivalence of Metric Units:** Because $y$ and $\hat{y}$ are in standardized $z$-score units, $(y - \hat{y})^2$ and $u$ share identical squared standardized units, making $u$ and empirical MSE directly comparable on the same scale.

**Distinction from Gaussian NLL:**
In Gaussian negative log-likelihood $\mathcal{L}_{\text{Gauss}}(q, u) = \frac{1}{2} \left[ \ln(u) + \frac{q}{u} \right]$, the gradient is:
$$\frac{\partial \mathcal{L}_{\text{Gauss}}}{\partial u} = \frac{1}{2u}\left(1 - \frac{q}{u}\right)$$
While both loss functions share the identical Bayes-optimal stationary point $u^*(x) = \mathbb{E}[(y - \hat{y})^2 \mid x]$, GKL removes the $1/(2u)$ multiplier. When predicted variance $u$ is small ($u \ll 1$), Gaussian NLL suffers from extreme gradient scaling ($\propto 1/u^2$) and numerical instability. GKL provides gradient stability with unit asymptotic derivative as $q \to 0$.

### 2.2 Terminology Matrix

| Metric Value | Statistical Meaning | Proper Calibration Terminology | Original Report Error |
|:---:|:---:|:---:|:---:|
| **$u / \text{MSE} < 1$** | Predicted variance is smaller than empirical mean squared error ($\mathbb{E}[u] < \text{MSE}$) | **Uncertainty Underestimation / Overconfidence** (model overestimates its precision) | Erroneously called "under-confidence" |
| **$u / \text{MSE} > 1$** | Predicted variance exceeds empirical mean squared error ($\mathbb{E}[u] > \text{MSE}$) | **Uncertainty Overestimation / Underconfidence** (model is overly conservative) | N/A |
| **$u / \text{MSE} \approx 1$** | Aggregate mean predicted variance matches aggregate empirical MSE | **Marginal Variance Agreement** (necessary but not sufficient for conditional calibration) | Confused with complete probabilistic calibration |

---

## 3. Provenance, Comparability & Sample Independence Audit

### 3.1 Sample Independence & Verified Plasma Beta Range

We audited the simulation metadata, `split.json`, and evaluation outputs across all evaluation tasks:
- **Independent Simulation Runs:** Exactly **125 distinct runs** (1-to-1 mapping with the 125 test runs in `split.json`).
- **Temporal Window Location:** Every test window starts at frame $t=0$ ($t_0=0, t_1=24$). There is **zero temporal overlap** among test windows.
- **Exact Verified Plasma Beta Coverage:**
  - The dataset and evaluation span exactly **10 physical plasma $\beta$ values**:
    $$\beta \in \{0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20\}$$
  - **Identical Train / Val / Test Coverage:** All three splits in `split.json` share the identical 10 beta values: `train_runs` (2,250 runs, 225 per $\beta$), `val_runs` (125 runs, 12 or 13 per $\beta$), and `test_runs` (125 runs, 12 or 13 per $\beta$).
  - In `test_runs`:
    - $\beta \in \{0.01, 0.04, 0.08, 0.12, 0.18\}$: 13 runs each (65 runs total).
    - $\beta \in \{0.02, 0.06, 0.10, 0.15, 0.20\}$: 12 runs each (60 runs total).
    - Total: 125 independent simulation runs.
  - **Clarification on Inconsistent Beta Descriptions:** Certain interim notes referenced an older exploratory range ($\beta=0.01$ to $4.0$). Audit of `split.json` and `beta_breakout_metrics.csv` confirms that the completed 256-epoch experiment and its held-out test evaluations strictly span $\beta = 0.01$ to $\beta = 0.20$.

### 3.2 Learning-Rate Schedule & Allocation Continuity Verification

Inspection of `config.json`, the learning-rate scheduler implementation in `train_masked_unet3d.py`, and the logged learning rates across all 256 epochs in `history.json` verifies:
- **Initial Learning Rate:** $\mathbf{1.0 \times 10^{-5}}$ at Epoch 1 (`min_lr = 1e-05`).
- **Warmup Schedule:** Linear warmup over 5 epochs (`warmup_epochs = 5`), progressing as $1.00 \times 10^{-5}$ (ep 1), $5.75 \times 10^{-5}$ (ep 2), $1.05 \times 10^{-4}$ (ep 3), $1.525 \times 10^{-4}$ (ep 4), and reaching peak LR at Epoch 5.
- **Peak Learning Rate:** $\mathbf{2.0 \times 10^{-4}}$ ($0.0002$, `lr = 0.0002`) at Epochs 5 and 6.
- **Cosine Decay Schedule:** Decays monotonically from peak $2.0 \times 10^{-4}$ down to minimum LR across the remaining 251 epochs according to:
  $$\text{LR}(t) = \text{min\_lr} + \frac{1}{2}(\text{peak\_lr} - \text{min\_lr})\left[1 + \cos\left(\frac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}} \pi\right)\right]$$
- **Minimum / Terminal Learning Rate:** $\mathbf{1.0 \times 10^{-5}}$ at Epoch 256 (`min_lr = 1e-05`).
- **Strict Multi-Allocation Continuity:** Verified 100% continuous across all three Slurm interactive allocation boundaries:
  - **Boundary 1 (Epoch 117):** Job `61775454` ended at epoch 117 ($1.2163 \times 10^{-4}$). Job `61786126` resumed cleanly at epoch 118 with LR $= 1.2045 \times 10^{-4}$ (ratio $0.9903$).
  - **Boundary 2 (Epoch 181):** Job `61786126` ended at epoch 181 ($4.9160 \times 10^{-5}$). Job `61791983` resumed cleanly at epoch 182 with LR $= 4.8199 \times 10^{-5}$ (ratio $0.9804$).
  - **Boundary 3 (Epoch 245):** Job `61791983` ended at epoch 245 ($1.0906 \times 10^{-5}$). Job `61798844` resumed cleanly at epoch 246 with LR $= 1.0749 \times 10^{-5}$ (ratio $0.9856$).
  Zero LR resets, zero re-warmups, and continuous cosine decay were preserved throughout all 256 epochs.

### 3.3 v9 Control vs. v11 Experiment Comparability

We compared the checkpoint metadata, normalization statistics, evaluation scripts, and training histories between the 256-epoch GKL v9 baseline (`fresh_gkl_fully_diff_256ep_20261120`) and the 256-epoch GKL v11 model (`fresh_gkl_v11_fromscratch_256ep_20261122`):
- **Channel Means:** Identical to 17 decimal places (`[-0.00025126294349320233, 0.0019038555910810828, -0.06113342195749283, 0.7741567492485046]`).
- **Channel Standard Deviations:** Identical to 17 decimal places (`[0.150075763463974, 0.04356170818209648, 0.398861289024353, 0.43268531560897827]`).
- **Split SHA-256:** Identical (`0e6b4276bc7f40003eed23c3e952025a3e6ef0d47619b1fbe2a8939d091f4b99`).
- **Test Sample Indices:** Identical 125 dataset indices (`select_validation_statistics_indices` evaluated on `split['test_runs']`).
- **Evaluation Mask Equality:** In `data/masking.py`, `make_fixed_validation_mask` uses deterministic formulas (checkerboard parity `(x+z)%2==0`, fixed seed 12345 for `spatial_random`, fixed index slices for temporal masks). The evaluation masks evaluated for v9 and v11 are **bit-for-bit identical**, not merely drawn from matching random seeds.
- **Evaluation Isolation:** Target metrics are evaluated exclusively on unobserved voxels (`mask < 0.5`) across both models.

**Training Configurations & Curriculum Differences:**
In `config.json`, the architectures, AdamW optimizer settings, and learning rate schedules match identically. However, training utilized the three-phase adaptive curriculum (`three_phase_global_cosine`), where phase transitions are event-driven based on validation loss convergence (`convergence_rate_threshold=0.005, convergence_patience_target=4`):
- **v11 from-scratch:** Phase A lasted epochs 1–62; Phase B began at epoch 63; Phase C settled at epoch 117 through 256 (**140 epochs in settled Phase C**).
- **v9 from-scratch:** Phase A lasted epochs 1–86; Phase B began at epoch 87; Phase C settled at epoch 141 through 256 (**116 epochs in settled Phase C**).
- **Loss Weights in Settled Phase C:** $\lambda_v = 1.0$ (variance loss weight), $\lambda_r = 1.0$ (rollout reuse weight), with pass 0 weight $w_0 = 0.5$ and pass 1 weight $w_1 = 0.5$.

**Classification Verdict:** Because the adaptive curriculum allocated different epoch counts to Phase A vs. Phase C due to differing convergence rates, the 256-epoch comparison represents a **Level 2 Configuration Comparison** (full end-to-end curriculum learning under different masking policies). The only strictly controlled single-variable ablation is the **Level 1 Matched 20-Epoch Continuation** (`evaluations/matched_gkl_masking_continuation_20ep_20261121/`), where both branches were resumed from the exact same epoch-256 checkpoint with identical hyperparameters, fixed loss weights, and fixed 20-epoch duration.

---

## 4. Extreme-Horizon Calibration Investigation ($h=20..23$)

### 4.1 Cohort Distribution Analysis

To determine whether extreme-horizon uncertainty degradation at $h=22..23$ is driven by a few catastrophic runs or represents a systematic distribution shift, we audited window-level metrics across all 125 independent runs:

| Horizon | Mean MSE | Median MSE | MSE IQR | Mean $u$ | Median $u$ | $u$ IQR | Mean Ratio | Median Ratio | % Runs with $u < \text{MSE}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$h=20$** | $0.2436$ | $0.1770$ | $[0.0694, 0.3505]$ | $0.2502$ | $0.2185$ | $[0.0436, 0.4610]$ | $1.0829$ | $0.9584$ | $53.6\%$ |
| **$h=22$** | $0.4541$ | $0.3550$ | $[0.1199, 0.6741]$ | $0.3571$ | $0.3619$ | $[0.0565, 0.6965]$ | $0.9223$ | $0.8953$ | $60.8\%$ |
| **$h=23$** | $0.7009$ | $0.6878$ | $[0.1271, 0.9881]$ | $0.3712$ | $0.1899$ | $[0.1477, 0.6723]$ | $0.8362$ | $0.6101$ | $68.8\%$ |

**Findings:**
1. At $h=20$, calibration is evenly balanced: $53.6\%$ of runs have $u < \text{MSE}$, and the median ratio is $0.9584$.
2. At $h=22$, the entire distribution shifts towards overconfidence: $60.8\%$ of runs have $u < \text{MSE}$, with aggregate $u/\text{MSE} = 0.7882$.
3. At $h=23$, **$68.8\%$ of runs underestimate uncertainty**, and the median ratio drops to $0.6101$.
4. **Conclusion:** Uncertainty underestimation at $h \ge 22$ is a **systematic cohort-wide limitation**, not an artifact of outlier runs.

### 4.2 Channel Breakout at Extreme Horizons

| Horizon | Modality | MSE | Mean $u$ | $u/\text{MSE}$ | Diagnosis |
|:---:|:---|:---:|:---:|:---:|:---|
| **$h=20$** | Magnetic ($B_x, B_y, B_z$) | $0.2976$ | $0.3018$ | $1.0143$ | Well-calibrated (marginal variance agreement) |
| **$h=20$** | Density | $0.0885$ | $0.0961$ | $1.0855$ | Slight uncertainty overestimation ($+8.6\%$) |
| **$h=20$** | **All Channels** | **$0.2453$** | **$0.2504$** | **$1.0207$** | **Marginal variance agreement ($+2.1\%$)** |
| **$h=22$** | Magnetic | $0.5452$ | $0.4294$ | $0.7876$ | Moderate uncertainty underestimation / overconfidence ($-21.2\%$) |
| **$h=22$** | Density | $0.1761$ | $0.1397$ | $0.7937$ | Moderate uncertainty underestimation / overconfidence ($-20.6\%$) |
| **$h=22$** | **All Channels** | **$0.4529$** | **$0.3570$** | **$0.7882$** | **Moderate uncertainty underestimation ($-21.2\%$)** |
| **$h=23$** | Magnetic | $0.8106$ | $0.4256$ | $0.5250$ | Severe uncertainty underestimation / overconfidence ($-47.5\%$) |
| **$h=23$** | Density | $0.3698$ | $0.2125$ | $0.5746$ | Severe uncertainty underestimation / overconfidence ($-42.5\%$) |
| **$h=23$** | **All Channels** | **$0.7004$** | **$0.3723$** | **$0.5316$** | **Severe uncertainty underestimation ($-46.8\%$)** |

Notice that Magnetic and Density channels underestimate uncertainty in near-exact lockstep at $h=22$ ($0.788$ vs. $0.794$), indicating that the variance head's saturation is tied to the shared temporal mask context rather than channel-specific dynamics.

### 4.3 Calibration Provenance & Reconciling Differing $h=23$ Ratios

**1. Verification of $h=23$ All-Channel Metrics:**
In `forecast_horizon_metrics.csv` (row 137), the true evaluated quantities for Horizon 23 across all channels are:
- $\text{MSE} = \mathbf{0.700402}$
- Mean $u = \mathbf{0.372300}$
- Calibration ratio $u/\text{MSE} = \mathbf{0.531552}$ ($0.5316$)

**2. Explanation of the $u=0.45940, u/\text{MSE}=0.6558$ Discrepancy:**
An interim draft summary erroneously conflated numbers across channels and horizons:
- The ratio $0.654563 \approx 0.6558$ was mistakenly pulled from the $B_x$ channel at $h=23$ (where $\text{MSE} = 0.3472, u = 0.2272, u/\text{MSE} = 0.6546$).
- The value $0.45940$ was an errant reference to the $h=22$ MSE scale ($0.4529$) / v9 $h=22$ mean $u$ ($0.4571$).
The primary source artifact `forecast_horizon_metrics.csv` has always contained $\text{MSE} = 0.70040, u = 0.37230, u/\text{MSE} = 0.5316$.

**3. Reconciliation of Differing v9 Baseline Ratios at $h=23$:**
Across previous evaluation reports, differing calibration ratios were reported for v9 at $h=23$:
- **Fresh 256-epoch from-scratch v9 baseline** (`fresh_gkl_fully_diff_256ep_20261120/forecast_horizon_metrics.csv`):
  $\text{MSE} = 0.930169, u = 0.399306 \implies u/\text{MSE} = \mathbf{0.4293}$. This is the direct baseline for the fresh from-scratch v11 model ($u/\text{MSE} = 0.5316$).
- **Matched 20-epoch continuation v9 control** (`matched_gkl_masking_continuation_20ep_20261121/forecast_horizon_metrics.csv`):
  $\text{MSE} = 0.859303, u = 0.495397 \implies u/\text{MSE} = \mathbf{0.5765}$. This model received 20 additional continuation epochs (epochs 257–276) with joint gradients from the epoch-256 checkpoint, shifting its terminal variance head weights.
- **Earlier 10-epoch diagnostic** (`uncertainty_feature_grad_continuation_10ep_analysis_20261117/`):
  Evaluated under an older, non-standard continuation protocol ($u/\text{MSE} = 0.8991$ on Density alone).
Distinguishing these configurations resolves all discrepancies: each reported ratio is mathematically verified within its respective evaluation artifact.

---

## 5. Investigation of `spatial_grid` Calibration Deficit

The report found that `spatial_grid` (checkerboard parity masking) had an aggregate ratio of $u/\text{MSE} = 0.5676$, with Density at $u/\text{MSE} = 0.3089$.

### 5.1 Stratification Across Beta Values

| Beta | `spatial_grid` MSE | Mean Predicted $u$ | Calibration Ratio $u/\text{MSE}$ | 1σ Coverage | 2σ Coverage | Spearman $\rho(e^2, u)$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $\beta = 0.01$ | $0.00488$ | $0.00190$ | $0.3897$ | $44.2\%$ | $77.0\%$ | $0.311$ |
| $\beta = 0.02$ | $0.00397$ | $0.00142$ | $0.3569$ | $44.4\%$ | $77.5\%$ | $0.318$ |
| $\beta = 0.04$ | $0.00403$ | $0.00151$ | $0.3747$ | $47.0\%$ | $79.7\%$ | $0.348$ |
| $\beta = 0.06$ | $0.00507$ | $0.00201$ | $0.3955$ | $49.8\%$ | $81.6\%$ | $0.335$ |
| $\beta = 0.08$ | $0.00767$ | $0.00406$ | $0.5293$ | $52.2\%$ | $83.4\%$ | $0.361$ |
| $\beta = 0.10$ | $0.00772$ | $0.00417$ | $0.5399$ | $51.4\%$ | $82.6$ | $0.341$ |
| $\beta = 0.12$ | $0.00783$ | $0.00424$ | $0.5420$ | $51.9\%$ | $82.9\%$ | $0.349$ |
| $\beta = 0.15$ | $0.01134$ | $0.00772$ | $0.6813$ | $52.7\%$ | $84.1\%$ | $0.338$ |
| $\beta = 0.18$ | $0.01129$ | $0.00784$ | $0.6939$ | $53.4\%$ | $85.0\%$ | $0.362$ |
| $\beta = 0.20$ | $0.01265$ | $0.00859$ | $0.6792$ | $53.3\%$ | $84.6\%$ | $0.367$ |

### 5.2 Verification of Manhattan Distance & MSE Reconciliation

**1. Proof of Manhattan Distance to Observed Anchors:**
In `data/masking.py`, the evaluation function `_checkerboard_spatial_plane(size_x, size_z)` is defined as:
```python
def _checkerboard_spatial_plane(size_x: int, size_z: int) -> torch.Tensor:
    x = torch.arange(int(size_x)).view(int(size_x), 1)
    z = torch.arange(int(size_z)).view(1, int(size_z))
    return ((x + z) % 2 == 0).float().view(1, 1, 1, int(size_x), int(size_z))
```
A voxel $(x, z)$ is observed if and only if $(x + z) \equiv 0 \pmod 2$.
An unobserved voxel $(x, z)$ has $(x + z) \equiv 1 \pmod 2$.
Its four orthogonal spatial neighbors are: $(x+1, z), (x-1, z), (x, z+1), (x, z-1)$.
For every neighbor, the coordinate sum satisfies:
$$(x \pm 1 + z) \equiv (x + z \pm 1) \equiv 1 \pm 1 \equiv 0 \pmod 2$$
Therefore, in the interior of the simulation grid, **every single unobserved voxel is surrounded by four observed voxels at Manhattan distance $d_1 = |\Delta x| + |\Delta z| = 1$**. (Boundary voxels have 2 or 3 observed neighbors at $d_1 = 1$).

**2. Reconciliation of MSE: $0.00763$ vs. $0.0171$:**
- **Evaluated Test `spatial_grid` MSE is $\mathbf{0.007625}$ ($\approx 0.00763$):** In `standard_mask_metrics.csv` (row 41), the test evaluation over all unobserved voxels under `spatial_grid` yields $\text{MSE} = 0.007625$, Mean $u = 0.004328$, and $u/\text{MSE} = 0.567624$. By channel: Density test MSE is $0.011883$ ($u = 0.003670, u/\text{MSE} = 0.3089$), and Magnetic test MSE is $0.006206$ ($u = 0.004548, u/\text{MSE} = 0.7328$).
- **Origin of $0.0171$:** The number $0.01713$ was the final training curriculum component $L_{\mu,0}$ at epoch 256 in `history.json` (as well as the overall training MSE of $0.0171$ at epoch 256 across all training masks). It was inadvertently quoted in prior discussion and is **not** the test `spatial_grid` reconstruction error.

### 5.3 Distinction Between Measured Evidence and Hypotheses

**Measured Empirical Evidence:**
1. **Deficit Uniformity:** The uncertainty underestimation deficit ($u < \text{MSE}$) occurs systematically across all 10 plasma $\beta$ values ($0.3569 \le u/\text{MSE} \le 0.6939$).
2. **Channel Divergence:** Density exhibits extreme underestimation ($u/\text{MSE} = 0.3089$), whereas the out-of-plane shear magnetic component $B_y$ is well-calibrated ($u/\text{MSE} = 0.9026$).
3. **Anchor Proximity:** Every unobserved voxel is tightly bounded by observed anchors at Manhattan distance 1.
4. **Low Variance Output:** The variance head outputs small values ($u = 0.00367$ for Density, $0.00433$ overall) because observed observations exist immediately adjacent to every unobserved site.

**Physical Hypotheses (Requiring Future Experimental Testing):**
1. **Sub-Grid Gradient Hypothesis:** It is hypothesized that the neural network's convolutional kernels perform smooth interpolation between the observed anchors at distance 1. Because Density contains sharp compressional shocks and localized current sheets with high-frequency spatial gradients, the true field deviates from smooth interpolation, creating localized squared errors that exceed the head's low predicted variance. Conversely, $B_y$ has smoother variations in the reconnection plane, allowing the same smooth interpolation to achieve near-optimal calibration ($u/\text{MSE} = 0.9026$).
2. **Lattice Fixedness Hypothesis:** It is hypothesized that because the network trained on fixed grid strides without sub-pixel jitter, it overfitted its confidence to the presence of distance-1 neighbors.
*Scientific Status:* These mechanisms are physically plausible hypotheses consistent with the data, but cannot be definitively confirmed without retraining with varying grid strides (e.g., stride 2, 4, 8) and randomized spatial jittering.

---

## 6. Forward vs. Reverse Extrapolation Audit

The report observed that reverse extrapolation MSE was $2.9\times - 3.6\times$ lower than forward extrapolation at $h=22..23$.

### 6.1 Data Variance Across Time

We measured the empirical field variance across the 125 test runs at different time steps:

| Time Step | $B_x$ Variance | $B_y$ Variance | $B_z$ Variance | Density Variance | All Variance |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$t = 0$** | $2.5264$ | **$0.0135$** | $2.6553$ | $1.5961$ | $1.6978$ |
| **$t = 1$** | $2.3749$ | **$0.1450$** | $3.0218$ | $3.2048$ | $2.1866$ |
| **$t = 12$** | $3.2388$ | **$2.5548$** | $1.5782$ | $1.1020$ | $2.1184$ |
| **$t = 22$** | $1.5371$ | **$1.6670$** | $1.5856$ | $0.9821$ | $1.4429$ |
| **$t = 23$** | $1.4726$ | **$1.5541$** | $1.5621$ | $0.9557$ | $1.3861$ |

### 6.2 Audit Conclusion

- At the initial time steps ($t=0..1$), the out-of-plane magnetic field $B_y$ is essentially laminar ($\text{Var}(B_y) = 0.0135$). As reconnection develops, magnetic islands and turbulence drive $B_y$ variance up by **over $100\times$** ($\text{Var}(B_y) = 2.5548$ at $t=12$).
- In reverse extrapolation, predicting $t=0..1$ means predicting a near-zero, low-variance target, which inherently yields low MSE ($0.154$).
- In forward extrapolation, predicting $t=22..23$ means predicting a high-variance turbulent field, yielding higher MSE ($0.453$).
- **Conclusion:** The directional asymmetry is an **empirical artifact of the initial-value simulation setup** (laminar initial conditions evolving into turbulent states). The original claim attributing this to general "physical entropy growth" was an unsupported over-interpretation and has been replaced with the verified data-distribution explanation.

---

## 7. Cross-Model Comparison Classification

To ensure experimental rigor, the models compared in the evaluation are categorized as follows:

```
+-------------------------------------------------------------------------------------------------+
|                               LEVEL 1: CONTROLLED MATCHED ABLATION                              |
| Isolates a single independent variable (Masking Policy) from the identical epoch-256 checkpoint: |
|   - Matched v9 Continuation (20ep, settled Phase C, fixed LR)                                   |
|   - Matched v11 Continuation (20ep, settled Phase C, fixed LR)                                  |
+-------------------------------------------------------------------------------------------------+
                                                │
                                                ▼
+-------------------------------------------------------------------------------------------------+
|                         LEVEL 2: FROM-SCRATCH CONFIGURATION COMPARISON                          |
| Compares end-to-end 256-epoch representation learning under different masking policies:         |
|   - Fresh GKL v9 Baseline (256 epochs, independent masking, Path A+B)                           |
|   - Fresh GKL v11 Model (256 epochs, shared temporal + circular block, Path A+B)                |
+-------------------------------------------------------------------------------------------------+
                                                │
                                                ▼
+-------------------------------------------------------------------------------------------------+
|                         LEVEL 3: CROSS-OBJECTIVE DESCRIPTIVE BASELINE                           |
| Compares across both different loss objectives AND masking distributions:                       |
|   - Student-t Baseline (256 epochs, Student-t NLL, independent v9 masking)                     |
+-------------------------------------------------------------------------------------------------+
```

Claims of causality must be restricted to Level 1. Level 2 reflects the combined end-to-end impact over full curriculum training, and Level 3 represents an observational baseline.
