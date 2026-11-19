# Direct Reconstruction-Error Prediction: Uncertainty Objective Benchmark

**Repository:** `/pscratch/sd/b/binxia/ai4plasma`  
**Evaluation Directory:** `evaluations/direct_uncertainty_objective_benchmark_20261119/`  
**Checkpoint:** `runs/multi_beta_three_phase_joint_nll_grad_globalcosine_256ep_20261117/final.pt` (Epoch 256)  
**Execution Environment:** Perlmutter Interactive Allocation (Job 61552309, NVIDIA A100-SXM4-80GB)  
**Date:** November 19, 2026  

---

## 1. Executive Summary & Scientific Motivation

Our primary scientific goal is to reconstruct 3D spatiotemporal plasma states ($\mathbf{B}, n$) from incomplete observations and predict conditional reconstruction uncertainty:
$$u(x) = \mathbb{E}[(Y - \mu(x))^2 \mid x]$$
particularly for recursive sliding-window inference where predicted states and uncertainties are reinjected as inputs in subsequent passes.

Our previous audits identified that joint A+B Student-t and Gaussian NLL objectives induce a **$200$–$260\times$ gradient norm imbalance** relative to reconstruction MSE on the shared U-Net backbone ($R = 261.84$ for Student-t, $197.38$ for Gaussian, exceeding $415\times$ on standard training masks). This massive imbalance stems from:
1. **Path A Dominance:** Gradients backpropagating through the mean residual $\frac{\partial L_{\text{NLL}}}{\partial \mu} \approx \frac{1}{2u} \frac{\partial L_\mu}{\partial \mu}$. Because conditional variance $u \approx 0.005$, Path A intrinsically amplifies the gradient by $\sim 100$–$200\times$, destabilizing the shared representation and degrading validation reconstruction MSE.
2. **Inverse-Variance Division:** In Gaussian NLL, $\frac{\partial L_G}{\partial \log u} = \frac{1}{2}(1 - q/u)$, which blows up as $q/u \to \infty$.

This benchmark investigates whether **direct reconstruction-error supervision**—treating the error target as detached ($q = \text{stop\_gradient}((y - \mu)^2)$, so **Path A is OFF** and **Path B is ON**)—can provide reliable predictive uncertainty with naturally balanced gradient scales, potentially eliminating the need for dynamic gradient normalization.

We evaluated three candidate direct objectives and three reference objectives across the 8 representative held-out simulation windows (betas $0.01$ to $0.20$) and 11 masking conditions (88 paired evaluations), accompanied by per-voxel tail gradient diagnostics across extreme horizons ($h \in \{18, 22, 23\}$).

### Key Findings:
1. **Direct Objectives Naturally Eliminate Gradient Imbalance:**
   - **Candidate C (Generalized KL / I-Divergence):** Median shared-backbone gradient ratio is **$1.55$** overall and **$1.67$** on standard masks.
   - **Candidate A (Absolute-Error MSE):** Median shared-backbone gradient ratio is **$1.19$** overall and **$1.74$** on standard masks.
   - **Candidate B (Smooth L1):** Median shared-backbone gradient ratio is **$0.87$** overall and **$1.22$** on standard masks.
   - All three direct error-regression objectives achieve natural parity with reconstruction MSE ($R \approx 0.9$–$1.7$) without any dynamic normalization, reducing the previous $262\times$ mismatch by over $99.4\%$.
2. **Generalized KL Preserves Exact Conditional-MSE Semantics:**
   - Candidate C provably minimizes the conditional squared error: $u_C^*(x) = \mathbb{E}[(Y - \mu(x))^2 \mid x]$.
   - Its derivative with respect to log-variance is linear in squared error without inverse-variance division: $\frac{\partial L_C}{\partial \log u} = u - q$.
   - Candidate A predicts conditional MAE ($\mathbb{E}[|e| \mid x]$), whose square $(\mathbb{E}[|e|])^2 \le \mathbb{E}[e^2]$ underestimates variance by Jensen's inequality. Candidate B (Smooth L1) heavily suppresses extreme errors ($>1.0$) because its derivative saturates at $-1.0$.
3. **Extreme-Horizon Tail Stability:**
   - Under extreme extrapolation ($h \in \{18, 22, 23\}$), Gaussian NLL exhibits explosive tail gradients (maximum logit gradient $|dz| = 1.59 \times 10^{-2}$).
   - Direct squared-error MSE (Reference 1) suffers from severe tail dominance: the top 1% largest error voxels contribute **57.59%** of the entire upward uncertainty gradient, explaining its historical training divergence.
   - Generalized KL eliminates both pathologies: its peak gradient is $1.60 \times 10^{-4}$ ($100\times$ smaller than Gaussian), and top 1% error voxels contribute a well-balanced **28.92%** of upward gradient force.

---

## 2. Mathematical Framework and Candidate Objectives

Let $\mu(x)$ be the predicted plasma state and $u(x) > 0$ be the predicted uncertainty parameterized as $u = \text{softplus}(z) + 10^{-6}$. All direct error-supervision objectives use detached error targets (Path A OFF, Path B ON through shared decoder features):

### Candidate A — Absolute-Error Prediction with Ordinary MSE
- Target: $a_{\text{target}} = \text{detach}(|y - \mu|)$
- Prediction: $a = \sqrt{u}$ (providing dimensionally compatible initialization from existing variance logits without changing model architecture)
- Objective: $L_A = \mathbb{E}[(a - a_{\text{target}})^2]$
- Population Optimum: $a^*(x) = \mathbb{E}[|Y - \mu(x)| \mid x]$ (conditional MAE).
- By Jensen's inequality:
  $$\left(\mathbb{E}[|Y - \mu(x)| \mid x]\right)^2 \le \mathbb{E}[(Y - \mu(x))^2 \mid x]$$
  Equality holds if and only if $|Y - \mu(x)|$ is conditionally degenerate. Thus, using $a^2$ systematically underestimates true conditional MSE.

### Candidate B — Squared-Error Prediction with Smooth L1
- Target: $q = \text{detach}((y - \mu)^2)$
- Objective: $L_B = \mathbb{E}[\text{SmoothL1}(u, q; \beta=1.0)]$ where
  $$\text{SmoothL1}(u, q) = \begin{cases} \frac{1}{2}(u - q)^2 & \text{if } |u - q| < 1.0 \\ |u - q| - 0.5 & \text{otherwise} \end{cases}$$
- Derivative: $\frac{\partial L_B}{\partial u} = \text{sign}(u - q)$ for $|u - q| \ge 1.0$.
- Limitation: For large prediction errors ($q - u \ge 1.0$), the gradient is clipped to $-1.0$. Consequently, severe prediction failures receive identical supervision to mild threshold errors.

### Candidate C — Generalized KL Divergence (I-Divergence, Primary Candidate)
- Target: $q = \text{detach}((y - \mu)^2)$
- Objective:
  $$L_C = \mathbb{E}\left[ q \log\left(\frac{q}{u}\right) + u - q \right]$$
  with the continuous extension $D(0 \parallel u) = u$.
- Population Optimum:
  $$\frac{\partial L_C}{\partial u} = 1 - \frac{\mathbb{E}[q \mid x]}{u} = 0 \implies u^*(x) = \mathbb{E}[q \mid x] = \mathbb{E}[(Y - \mu(x))^2 \mid x]$$
  The theoretical minimum is provably the exact conditional MSE.
- Derivative w.r.t $\log u$:
  $$\frac{\partial L_C}{\partial \log u} = u \frac{\partial L_C}{\partial u} = u - q$$
  This derivative is linear in $q$ and $u$, completely eliminating the $\frac{q}{u}$ division that causes Gaussian NLL gradient explosions.

### References
- **Reference 1 (Direct Squared-Error MSE):** $L_{\text{MSE}} = \mathbb{E}[(u - q)^2]$, where $\frac{\partial L}{\partial u} = 2(u - q)$.
- **Reference 2 (Gaussian NLL):** $L_G = \frac{1}{2}\log(2\pi u) + \frac{q}{2u}$. Evaluated both Path-B only ($q$ detached) and Full A+B.
- **Reference 3 (Student-t NLL, df=3):** $L_T = \text{const} + \frac{1}{2}\log u + 2\log(1 + \frac{q}{u})$. Evaluated both Path-B only and Full A+B.

---

## 3. Mathematical and Implementation Verification

All 10 verification checks specified in Section 6 were implemented and validated:

1. **Finite at $q=0$:** Verified. For $q=0$, $L_C = u$, which is strictly positive and finite.
2. **Zero at $u=q>0$:** Verified. $|L_C| < 10^{-7}$ when $u=q$.
3. **Vanishing Gradient at $u=q$:** Verified. $\partial L_C / \partial u = 0$ at $u=q$.
4. **Directional Update when $u < q$:** Verified. $\partial L_C / \partial u < 0$, driving gradient descent to increase $u$.
5. **Directional Update when $u > q$:** Verified. $\partial L_C / \partial u > 0$, driving gradient descent to decrease $u$.
6. **Log-Variance Derivative:** Verified. $\partial L_C / \partial \log u = u - q$ matches autograd to numerical precision ($< 10^{-6}$).
7. **Softplus Parameterization Compatibility:** Verified. Derivative through $u = \text{softplus}(z) + 10^{-6}$ matches analytical chain rule: $\frac{\partial L_C}{\partial z} = (1 - q/u) \text{sigmoid}(z)$.
8. **Arithmetic Mean Optimum:** Verified via L-BFGS optimization on empirical sample batches; numerical minimum converges exactly to $\bar{q} = \frac{1}{N}\sum_i q_i$.
9. **Path A Isolation:** Verified. With detached $q$, autograd gradient on `model.out` (mean head) is identically zero (`False` for non-zero gradient).
10. **Path B Activation:** Verified. Uncertainty loss propagates non-zero gradients into `model.variance_out` and through shared U-Net parameters (`model.enc0`, etc.).
11. **Candidate A Optimum:** Verified via numerical optimization. Optimum converges to $\mathbb{E}[|e|]$, and $(\mathbb{E}[|e|])^2 < \mathbb{E}[e^2]$ strictly obeys Jensen's inequality.

---

## 4. Benchmark Results: Gradient Norms and Ratios

Evaluating all 8 windows $\times$ 11 masks (88 paired samples) produced 792 gradient norm measurements and 704 cosine similarities.

### Table 1: Gradient Norms and Ratios Relative to Reconstruction MSE ($R = \|\nabla_\theta L_u\|_2 / \|\nabla_\theta L_\mu\|_2$)

| Objective | Category | Subset | Median Loss | Shared Backbone Norm $\|\nabla_\theta L\|$ | Ratio $R$ Median | Ratio $R$ $[p_{16}, p_{84}]$ | Feature Ratio $R_{\text{feat}}$ | Head Norm $\|\nabla_{\theta_{\text{head}}} L\|$ | Cosine Sim $\cos(\nabla_\theta L_u, \nabla_\theta L_\mu)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reconstruction MSE ($L_\mu$)** | Reconstruct | All | 0.0401 | 1.8485 | **1.0000** | $[1.00, 1.00]$ | 1.0000 | 0.4072 | 1.0000 |
| | | Standard | 0.0135 | 0.7226 | **1.0000** | $[1.00, 1.00]$ | 1.0000 | 0.1691 | 1.0000 |
| | | Extrapolation | 0.1873 | 2.8773 | **1.0000** | $[1.00, 1.00]$ | 1.0000 | 1.2194 | 1.0000 |
| **Cand A: Absolute MSE** | Direct | All | 0.0236 | 2.1721 | **1.1905** | $[0.41, 4.01]$ | 4.1630 | 0.1706 | +0.0413 |
| | | Standard | 0.0088 | 1.2591 | **1.7432** | $[0.53, 4.80]$ | 3.8211 | 0.1191 | +0.0115 |
| | | Extrapolation | 0.0471 | 2.9170 | **1.0130** | $[0.38, 3.08]$ | 4.3045 | 0.2089 | +0.0891 |
| **Cand B: Smooth L1** | Direct | All | 0.0172 | 1.7214 | **0.8693** | $[0.11, 3.52]$ | 3.6086 | 0.1352 | +0.0362 |
| | | Standard | 0.0081 | 0.7303 | **1.2184** | $[0.03, 4.27]$ | 3.9217 | 0.0887 | +0.0002 |
| | | Extrapolation | 0.0437 | 2.1412 | **0.5215** | $[0.16, 2.04]$ | 3.2505 | 0.1493 | +0.1175 |
| **Cand C: Generalized KL** | Direct | All | 0.0378 | 2.1567 | **1.5489** | $[0.42, 5.34]$ | 8.1485 | 0.1960 | **+0.0765** |
| | | Standard | 0.0139 | 1.1995 | **1.6747** | $[0.58, 4.79]$ | 7.8922 | 0.1277 | +0.0189 |
| | | Extrapolation | 0.0695 | 3.4427 | **1.4225** | $[0.40, 6.99]$ | 8.1485 | 0.3412 | **+0.1166** |
| **Ref 1: Squared MSE** | Direct | All | 0.0552 | 4.7031 | **3.9011** | $[0.22, 12.08]$ | 20.9947 | 0.4958 | +0.0604 |
| | | Standard | 0.0446 | 2.5742 | **4.0769** | $[0.06, 14.73]$ | 22.0963 | 0.2581 | +0.0051 |
| | | Extrapolation | 0.1767 | 8.4632 | **3.7562** | $[0.35, 11.17]$ | 19.3496 | 0.5688 | +0.1386 |
| **Ref 2: Gaussian (Path B)** | NLL B | All | -2.9169 | 40.1468 | **45.4679** | $[11.59, 205.88]$ | 77.9148 | 2.0536 | +0.0252 |
| | | Standard | -2.9907 | 33.1526 | **99.9081** | $[23.09, 205.50]$ | 210.1416 | 1.8995 | +0.0115 |
| | | Extrapolation | -2.6165 | 42.6239 | **24.8213** | $[8.71, 195.79]$ | 39.8804 | 2.2155 | +0.0393 |
| **Ref 2: Gaussian (Full A+B)**| NLL A+B | All | -2.9169 | 195.2893 | **197.3857** | $[30.31, 963.09]$ | 417.6312 | 2.0536 | +0.0900 |
| | | Standard | -2.9907 | 195.2893 | **415.1545** | $[78.85, 1194.39]$| 694.0743 | 1.8995 | +0.0969 |
| | | Extrapolation | -2.6165 | 193.9053 | **90.5896** | $[20.33, 593.81]$ | 158.4239 | 2.2155 | +0.0822 |
| **Ref 3: Student-t (Path B)** | NLL B | All | -2.0316 | 12.2875 | **9.9612** | $[3.41, 55.62]$ | 22.9035 | 0.6970 | +0.0465 |
| | | Standard | -2.1433 | 10.4396 | **33.9891** | $[4.31, 99.42]$ | 45.7504 | 0.6909 | +0.0189 |
| | | Extrapolation | -1.7636 | 12.6309 | **7.7890** | $[2.55, 22.74]$ | 12.9231 | 0.7629 | +0.0985 |
| **Ref 3: Student-t (Full A+B)**| NLL A+B | All | -2.0316 | 234.4859 | **261.8188** | $[23.40, 1719.54]$| 437.7490 | 0.6970 | +0.0710 |
| | | Standard | -2.1433 | 251.9876 | **417.1772** | $[82.98, 2072.60]$| 671.3910 | 0.6909 | +0.0599 |
| | | Extrapolation | -1.7636 | 199.7514 | **78.7774** | $[12.34, 663.46]$ | 148.6190 | 0.7629 | +0.0914 |

---

## 5. Extreme-Error and Numerical-Stability Diagnostics ($h \in \{18, 22, 23\}$)

Under extreme forecast horizons, the model reconstructs up to 23 consecutive unobserved frames. Per-voxel errors reach $q_{\max} > 25.0$, and $q/u$ ratios reach $10^3$–$10^4$.

### Table 2: Extreme-Horizon Per-Voxel Gradient Magnitudes and Tail Concentrations

| Objective | Field Group | Median $\|dz\|$ | 95th %ile $\|dz\|$ | Max $\|dz\|$ | Top 1% Upward Gradient Share (%) | Nonfinite Voxels |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate A: Absolute MSE** | Magnetic | $2.50 \times 10^{-8}$ | $5.63 \times 10^{-7}$ | $7.81 \times 10^{-6}$ | 24.38% | 0 |
| | Density | $3.77 \times 10^{-8}$ | $3.72 \times 10^{-7}$ | $5.16 \times 10^{-6}$ | 24.85% | 0 |
| **Candidate B: Smooth L1** | Magnetic | $1.61 \times 10^{-9}$ | $3.12 \times 10^{-7}$ | $1.09 \times 10^{-6}$ | 18.17% | 0 |
| | Density | $4.69 \times 10^{-9}$ | $3.41 \times 10^{-7}$ | $1.09 \times 10^{-6}$ | 31.76% | 0 |
| **Candidate C: Generalized KL** | Magnetic | $4.46 \times 10^{-8}$ | $1.02 \times 10^{-6}$ | $1.60 \times 10^{-4}$ | **28.92%** | 0 |
| | Density | $5.55 \times 10^{-8}$ | $6.22 \times 10^{-7}$ | $5.87 \times 10^{-5}$ | **27.63%** | 0 |
| **Reference 1: Squared MSE** | Magnetic | $3.29 \times 10^{-9}$ | $1.44 \times 10^{-6}$ | $3.40 \times 10^{-4}$ | **57.59%** | 0 |
| | Density | $9.39 \times 10^{-9}$ | $7.50 \times 10^{-7}$ | $1.11 \times 10^{-4}$ | **46.99%** | 0 |
| **Reference 2: Gaussian NLL (Path B)**| Magnetic | $4.48 \times 10^{-7}$ | $3.58 \times 10^{-6}$ | **$1.59 \times 10^{-2}$** | 5.84% | 0 |
| | Density | $4.29 \times 10^{-7}$ | $2.27 \times 10^{-6}$ | **$2.99 \times 10^{-3}$** | 8.15% | 0 |
| **Reference 3: Student-t NLL (Path B)**| Magnetic | $4.53 \times 10^{-7}$ | $1.32 \times 10^{-6}$ | $1.64 \times 10^{-6}$ | 1.85% | 0 |
| | Density | $4.39 \times 10^{-7}$ | $1.23 \times 10^{-6}$ | $1.64 \times 10^{-6}$ | 2.55% | 0 |

### Critical Analytical Comparisons:
1. **Gaussian NLL Explodes at Small $u$ / Large $q$:**
   $$\frac{\partial L_G}{\partial z} = \frac{1}{2u}\left(1 - \frac{q}{u}\right)\text{sigmoid}(z)$$
   Because $u$ appears in the denominator squared w.r.t $q$, voxels with $q \approx 10$ and $u \approx 0.005$ produce $\partial L_G / \partial z \approx -2 \times 10^5 \times \frac{1}{N} \approx -0.5$. The empirical maximum reached **$1.59 \times 10^{-2}$**, four orders of magnitude larger than its median ($4.48 \times 10^{-7}$).
2. **Direct Squared-Error MSE is Hijacked by Outliers:**
   $$\frac{\partial L_{\text{MSE}}}{\partial z} = 2(u - q)\text{sigmoid}(z)$$
   At large errors, $\partial L / \partial z \propto -q = -(y - \mu)^2$. Although finite, the top 1% largest error voxels contribute **57.59%** of the entire upward gradient force! This means training steps under direct squared MSE are dominated by a handful of anomalous voxels rather than broad reconstruction errors.
3. **Smooth L1 Clips Supervision for Extreme Failures:**
   $$\frac{\partial L_B}{\partial z} = -\text{sigmoid}(z) \quad (\text{for } q - u > 1.0)$$
   Regardless of whether $q = 1.0$ or $q = 50.0$, the gradient is clamped to the identical constant. The maximum gradient is strictly capped at $1.09 \times 10^{-6}$.
4. **Generalized KL Provides Robust, Proportional Supervision:**
   $$\frac{\partial L_C}{\partial \log u} = u - q, \quad \frac{\partial L_C}{\partial z} = \left(1 - \frac{q}{u}\right)\text{sigmoid}(z)$$
   The gradient scales linearly with $q$ (providing stronger supervision for larger forecasting errors) without the squared inverse-variance explosion. Peak gradients ($1.60 \times 10^{-4}$) are $100\times$ lower than Gaussian NLL, and the top 1% voxel share is **28.92%**, maintaining balanced multi-scale supervision.

---

## 6. Answers to Mandatory Scientific Questions (Section 13)

### 1. Which direct uncertainty objective produces the most reasonable gradient scale relative to reconstruction MSE?
**Candidate C (Generalized KL / I-Divergence)** and **Candidate A (Absolute-Error MSE)** produce the most natural scales:
- Candidate C: Median ratio $R = \mathbf{1.55}$ overall, $\mathbf{1.67}$ on standard masks ($[p_{16}, p_{84}] = [0.42, 5.34]$).
- Candidate A: Median ratio $R = \mathbf{1.19}$ overall, $\mathbf{1.74}$ on standard masks ($[p_{16}, p_{84}] = [0.41, 4.01]$).
- Both objectives achieve natural order-of-magnitude parity ($R \approx 1.2$–$1.7$) with reconstruction MSE, completely eliminating the $262\times$ gradient blowup of Full A+B Student-t NLL.

### 2. Does absolute-error MSE substantially avoid the instability-related gradient behavior of squared-error MSE?
**YES.** Absolute-error MSE reduces maximum voxel gradient magnitude from $3.40 \times 10^{-4}$ (squared MSE) to $7.81 \times 10^{-6}$ (a **$43\times$ reduction**). The top 1% voxel gradient share drops from **57.59%** (squared MSE) to **24.38%**, preventing individual outlier voxels from dominating the batch update.

### 3. Does Smooth L1 excessively suppress supervision for extreme prediction errors?
**YES.** For all voxels with $|q - u| \ge 1.0$, Smooth L1's derivative $\partial L / \partial u$ saturates at $-1.0$. Consequently, its maximum voxel gradient is capped at $1.09 \times 10^{-6}$. Under extreme extrapolation ($h \in \{18, 22, 23\}$), where errors are largest, Candidate B's effective gradient ratio drops to $0.52$. It fails to provide proportionally stronger supervisory force when the model suffers severe forecasting degradation.

### 4. Does generalized KL provide a reasonable balance between strong error supervision and numerical stability?
**YES.**
- **Theoretical correctness:** Population minimizer is provably $\mathbb{E}[(Y - \mu)^2 \mid x]$.
- **Numerical scale:** Natural backbone ratio of $1.55$ ($1.67$ on standard masks) without artificial multipliers.
- **Tail behavior:** Peak gradient ($1.60 \times 10^{-4}$) is $100\times$ smaller than Gaussian NLL ($1.59 \times 10^{-2}$). Top 1% error voxels contribute $28.92\%$ of upward gradient force, providing active error correction while preventing batch updates from being hijacked by a few voxels.
- **Stability:** Zero nonfinite events across all 88 evaluations.

### 5. Are generalized KL gradients materially smaller than Gaussian and Student-t NLL gradients?
**YES.**
- **Versus Full A+B NLL:** Shared-backbone gradient norm is $2.16$ for Generalized KL vs **$234.5$** for Student-t ($108\times$ smaller) and **$195.3$** for Gaussian ($90\times$ smaller).
- **Versus Path-B Only NLL:** Shared-backbone gradient norm is $2.16$ for Generalized KL vs **$12.29$** for Student-t Path B ($5.7\times$ smaller) and **$40.15$** for Gaussian Path B ($18.6\times$ smaller). On standard training masks, Generalized KL ($1.20$) is $8.7\times$ smaller than Student-t Path B ($10.44$) and $27.6\times$ smaller than Gaussian Path B ($33.15$).

### 6. Do the direct objectives retain sufficiently strong uncertainty-head and shared-backbone gradients?
**YES.**
- Generalized KL imparts a shared-backbone gradient norm of $2.16$, which is fully aligned in magnitude with reconstruction MSE ($1.85$).
- Uncertainty-head parameter norm is $0.196$ (healthy, active, non-vanishing gradient).
- Cosine similarity with reconstruction MSE is weakly positive ($\cos = +0.0765$ overall, $+0.1166$ on extrapolation), confirming that uncertainty supervision does not destructively oppose reconstruction.

### 7. Is dynamic gradient normalization still justified?
**NO.** With Candidate C (Generalized KL), dynamic gradient normalization is **no longer necessary**:
- The natural gradient ratio $R = 1.55$ is already within a factor of $1.5$–$1.7$ of reconstruction MSE.
- Eliminating Path A removes the $1/u$ residual amplification.
- The training objective:
  $$L = \frac{L_\mu + \lambda_v L_C}{1 + \lambda_v}$$
  is intrinsically balanced at $\lambda_v = 1.0$, completely obviating runtime gradient-energy all-reduces and multi-pass normalization overhead.

### 8. Which objective best preserves our intended conditional-MSE uncertainty semantics?
**Candidate C (Generalized KL / I-Divergence).**
- Its mathematical population minimizer is provably $u^*(x) = \mathbb{E}[(Y - \mu(x))^2 \mid x]$.
- The output $u$ directly represents expected squared error, preserving identical dimensions, calibration definitions, and sliding-window recursive reinjection semantics without transformation.
- Candidate A predicts conditional MAE ($\mathbb{E}[|e|]$), which systematically underestimates variance due to Jensen's inequality. Candidate B (Smooth L1) predicts a pseudo-Huber smoothed statistic that under-reports variance in heavy-tailed distributions.

### 9. Which objective should be prioritized for the next controlled continuation experiment?
**Candidate C: Generalized KL Divergence with detached error target ($q = \text{stop\_gradient}((y - \mu)^2)$), Path A OFF, Path B ON.**
- It retains exact conditional-MSE uncertainty semantics.
- It naturally resolves the multi-order-of-magnitude gradient imbalance ($R = 1.67$ on standard masks) without runtime normalization machinery.
- It eliminates the Path-A mean degradation mechanism while maintaining robust supervision across extreme forecast horizons.

---

## 7. Deliverable Artifacts

The generated artifacts are preserved in `evaluations/direct_uncertainty_objective_benchmark_20261119/`:
- `run_benchmark.py`: Standalone benchmark script.
- `objective_gradient_norms.csv`: Complete raw gradient norms and ratios (792 rows).
- `gradient_cosine_similarity.csv`: Directional cosine similarities w.r.t reconstruction MSE (704 rows).
- `gradient_tail_statistics.csv`: Per-voxel error percentiles, gradient extremes, and tail concentrations (384 rows).
- `objective_comparison_summary.csv`: Aggregated median and percentile distributions (24 rows).
- `benchmark_summary.json`: Serialized JSON summary.
- `figure1_shared_backbone_gradient_ratios.png`: Boxplots of shared-backbone gradient ratios across objectives.
- `figure2_gradient_alignment.png`: Distribution of cosine similarities $\cos(\nabla_\theta L_u, \nabla_\theta L_\mu)$.
- `figure3_extreme_error_supervision.png`: Bar plots of logit gradient magnitudes and top 1% tail concentrations.

---

## 8. Final Synthesis

**Direct Uncertainty Objective Benchmark**

- Checkpoint: gradual A+B Student-t epoch 256
- Evaluation: pass 0 only
- Paired windows: 8 windows (betas 0.01 to 0.20) x 11 mask conditions (88 total paired samples)
- Direct error targets detached: YES
- Direct objective Path A: OFF
- Direct objective Path B: ON
- Original squared-error MSE / reconstruction gradient norm ratio: 3.9011 (all conditions), 4.0769 (standard masks)
- Absolute-error MSE / reconstruction gradient norm ratio: 1.1905 (all conditions), 1.7432 (standard masks)
- Smooth L1 / reconstruction gradient norm ratio: 0.8693 (all conditions), 1.2184 (standard masks)
- Generalized KL / reconstruction gradient norm ratio: 1.5489 (all conditions), 1.6747 (standard masks)
- Gaussian NLL Path-B / reconstruction gradient norm ratio: 45.4679 (all conditions), 99.9081 (standard masks)
- Student-t NLL Path-B / reconstruction gradient norm ratio: 9.9612 (all conditions), 33.9891 (standard masks)
- Generalized KL conditional-MSE interpretation: VERIFIED
- Generalized KL numerical checks: 10/10 verified (finite at q=0, zero at u=q, vanishing gradient at u=q, monotonic update directions, dL/dlog(u)=u-q, softplus compatibility, arithmetic mean optimum, Path A detached, Path B active)
- Extreme-error gradient behavior: Max |dz| = 1.60e-4 (100x lower than Gaussian 1.59e-2); top 1% error voxel share = 28.92% (vs 57.59% for squared MSE); zero nonfinite voxels
- Uncertainty-head gradient strength: Active and healthy (norm = 0.1960 vs reconstruction head norm = 0.4072)
- Dynamic gradient normalization necessary: NO (Generalized KL naturally achieves R = 1.55-1.67, eliminating the need for runtime normalization)
- Recommended uncertainty objective: Candidate C (Generalized KL Divergence, I-Divergence)
- Recommended next training experiment: 20-epoch continuation comparing Candidate C (Generalized KL, Path A OFF, Path B ON) against matched unnormalized Student-t control
- Production code modified: NO
- Model parameters updated: NO
- Training launched: NO
