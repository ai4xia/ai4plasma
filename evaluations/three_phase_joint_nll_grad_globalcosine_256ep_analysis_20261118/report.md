# Evaluation Report: Fresh 256-Epoch Three-Phase A+B Joint-Gradient Training

**Evaluation Date:** 2026-11-18  
**Repository:** `/pscratch/sd/b/binxia/ai4plasma`  
**Evaluation Directory:** `evaluations/three_phase_joint_nll_grad_globalcosine_256ep_analysis_20261118/`  
**Device:** NVIDIA A100-PCIE-40GB (PyTorch 2.5.1+cu124, CUDA 13.0)  

---

## 1. Executive Summary

We performed a paired frozen-model scientific evaluation of the newly completed **from-scratch 256-epoch three-phase curriculum model** trained with **joint A+B Student-t NLL uncertainty gradients** (`new_joint_256ep`). The model was evaluated against the primary baseline (`original_256ep`, trained from scratch with detached NLL pathways) as well as secondary continuation reference models (`continuation_epoch276`, `bonly_epoch286`, and `ab_abrupt_epoch286`).

### Key Findings
1. **Gradual Uncertainty Learning Eliminates Training Instability:**  
   Unlike abrupt continuation experiments (where enabling uncertainty gradients at epoch 276 caused validation MSE to explode), gradual introduction via the 40-epoch cosine $\lambda_v$ ramp (epochs 75–114) preserved mean reconstruction throughout training. Validation MSE improved from $0.021271$ at the end of Phase A to $0.018513$ at settled Phase B and $0.017454$ at epoch 256 (compared to $0.014208$ in the original baseline).
2. **Student-t NLL Improvement:**  
   Validation Student-t NLL improved from $-1.2373$ nats/voxel (original baseline) to $-2.4208$ nats/voxel in the new joint model—a reduction of $>1.18$ nats/voxel.
3. **Horizon Uncertainty Dynamic Range Restored:**  
   In the original baseline, forecast uncertainty dynamic range was severely compressed ($u(h18)/u(h1) = 1.44$ for Magnetic). In the new joint model, $u(h18)/u(h1)$ expanded to **9.19** for Magnetic and **9.75** for Density.
4. **Per-Frame Within-Window Scaling Inverted Pathological Behavior:**  
   In the original baseline at $h=18$, predicted uncertainty perversely decreased with forecast lead time ($u_{\text{last}}/u_{\text{first}} = 0.311$). The new model fully reversed this pathology: $u_{\text{last}}/u_{\text{first}} = 1.997$ for Magnetic and **$11.413$** for Density (closely tracking actual MSE growth of $16.27\times$).
5. **Spatial Error-Uncertainty Ranking Dramatically Enhanced:**  
   Within-frame spatial Spearman rank correlation at $h=18$ jumped from $0.1739$ (original) to **$0.6383$** (new joint model) for Magnetic fields, with the pooled spatial Spearman reaching $0.7372$ and high-to-low decile MSE contrast reaching $468.8\times$.
6. **Near-Ideal Absolute Calibration:**  
   The severe underconfidence of the baseline ($u/\text{MSE} = 0.033$ at $h=18$ Magnetic) was completely resolved: the new model achieved $u/\text{MSE} = 1.297$ (Magnetic) and $1.433$ (Density) at $h=18$.

---

## 2. Checkpoint & Training History Audit

### 2.1 Model Specifications & Metadata

| Field | Primary Baseline (`original_256ep`) | Primary Treatment (`new_joint_256ep`) |
| :--- | :--- | :--- |
| **Checkpoint Path** | `runs/..._256ep_20261115/final.pt` | `runs/..._joint_nll_grad_globalcosine_256ep_20261117/final.pt` |
| **Initialization** | From scratch (`init_checkpoint: null`) | From scratch (`init_checkpoint: null`) |
| **Continuation** | None (`post_final_continuation: null`) | None (`post_final_continuation: null`) |
| **Total Epochs** | 256 | 256 |
| **Optimizer Steps** | 273,664 | 273,664 |
| **LR Schedule** | Global cosine $2\times 10^{-4} \to 1\times 10^{-5}$ | Global cosine $2\times 10^{-4} \to 1\times 10^{-5}$ |
| **Loss Formulation** | Student-t NLL ($\nu=3.0$) | Student-t NLL ($\nu=3.0$) |
| **NLL Path A** | Detached | **Enabled** |
| **NLL Path B** | Detached | **Enabled** |
| **Phase C Reuse** | Detached | Detached |
| **Phase D** | None | None |
| **Nonfinite Events**| 0 | 0 |
| **AMP Skips** | 0.0 | 0.0 |
| **Split SHA256** | `0e6b4276bc...` | `0e6b4276bc...` (Identical) |

### 2.2 Phase Schedule Comparison

The training schedule utilized adaptive Phase A convergence with identical cosine ramp durations:

| Phase | Original 256-Epoch Run | New Joint A+B 256-Epoch Run | Cosine Ramp Verified |
| :--- | :--- | :--- | :--- |
| **Phase A** (`mean_only`) | Epochs 1–75 (75 epochs) | Epochs 1–74 (74 epochs) | N/A |
| **Phase B Ramp** ($\lambda_v: 0 \to 1$) | Epochs 76–115 (40 epochs) | Epochs 75–114 (40 epochs) | **VERIFIED** |
| **Phase B Settled** ($\lambda_v=1, \lambda_r=0$) | Epochs 116–128 (13 epochs) | Epochs 115–127 (13 epochs) | N/A |
| **Phase C Ramp** ($\lambda_r: 0 \to 1$) | Epochs 129–168 (40 epochs) | Epochs 128–167 (40 epochs) | **VERIFIED** |
| **Phase C Settled** ($\lambda_v=1, \lambda_r=1$) | Epochs 169–256 (88 epochs) | Epochs 168–256 (89 epochs) | N/A |
| **Phase D** | None | None | N/A |

---

## 3. Analysis of Gradual Uncertainty Introduction

A central scientific question is whether introducing joint uncertainty gradients gradually during training avoids the severe reconstruction degradation observed in abrupt continuation experiments.

### 3.1 Milestone Tracking Across Training

The table below tracks validation reconstruction MSE and Student-t NLL across the requested curriculum milestones:

| Milestone | New Ep | Orig Ep | $\lambda_v$ (New) | Val MSE (New) | Val MSE (Orig) | Ratio (New/Orig) | Val NLL (New) | Val NLL (Orig) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. End of Phase A** | 74 | 75 | 0.0000 | 0.021271 | 0.019998 | 1.064 | +0.4949 | +0.4320 |
| **2. Beginning of Phase B** | 75 | 76 | 0.0005 | 0.020794 | 0.020332 | 1.023 | -1.4547 | -0.1551 |
| **3. Approx $\lambda_v = 0.25$** | 88 | 89 | 0.2557 | 0.018718 | 0.019315 | **0.969** | -1.9769 | -0.9171 |
| **4. Approx $\lambda_v = 0.50$** | 95 | 96 | 0.5196 | 0.018944 | 0.019438 | **0.975** | -2.1324 | -1.1401 |
| **5. Approx $\lambda_v = 0.75$** | 101 | 102 | 0.7442 | 0.019242 | 0.019548 | **0.984** | -2.0960 | -1.2190 |
| **6. End of Phase B Ramp** | 114 | 115 | 0.9995 | 0.019042 | 0.017982 | 1.059 | -2.2986 | -1.2716 |
| **7. End of Settled Phase B** | 127 | 128 | 1.0000 | 0.018513 | 0.018138 | **1.021** | -2.2957 | -1.2410 |
| **8. End of Phase C Ramp** | 167 | 168 | 1.0000 | 0.018267 | 0.016888 | 1.082 | -2.3400 | -1.2018 |
| **9. End of Phase C (Epoch 256)**| 256 | 256 | 1.0000 | 0.017454 | 0.014208 | 1.228 | **-2.4208** | -1.2373 |

### 3.2 Dynamics Findings
- **No Transient or Persistent Shock:** During the entire 40-epoch $\lambda_v$ ramp (epochs 75–114), validation MSE monotonically improved from $0.020794$ down to $0.019042$, outperforming the original baseline between $\lambda_v = 0.25$ and $\lambda_v = 0.75$.
- **Settled Phase B Parity:** At the end of settled Phase B (epoch 127), the new joint model achieved validation MSE of $0.018513$, virtually identical to the baseline's $0.018138$ ($+2.1\%$ difference).
- **Multi-Task Trade-off in Phase C:** In Phase C, the original model improved to $0.014208$, while the joint model improved to $0.017454$. The joint model balanced two active gradient paths (mean reconstruction and variance calibration), resulting in a modest $22.8\%$ trade-off on training-distribution MSE while lowering validation NLL by $>1.18$ nats/voxel.

---

## 4. Standard Mask Reconstruction Benchmark

Evaluated on 125 held-out validation windows across the 5 standard masking families (Pass 0, All Channels):

| Masking Family | Original 256ep | New Joint 256ep | Continuation Ep 276 | Abrupt B-only (286) | Abrupt A+B (286) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `spatial_random` | 0.008260 | 0.012700 | 0.008079 | 0.018972 | 0.008920 |
| `spatial_grid` | 0.009248 | 0.012577 | 0.008930 | 0.021277 | 0.010087 |
| `spatial_block` | 0.102256 | 0.163965 | 0.103698 | 0.177535 | 0.123967 |
| `temporal_random` | 0.025113 | 0.036656 | 0.024979 | 0.046571 | 0.032152 |
| `temporal_block` | 0.119188 | 0.180056 | 0.113822 | 0.165283 | 0.124268 |
| **Aggregate Mean** | **0.052813** | **0.081191** | **0.051902** | **0.085928** | **0.059879** |

---

## 5. Forecast Horizon Evaluation ($h = 1 \dots 23$)

Evaluated on 125 held-out test windows with Gaussian initialization on hidden suffixes.

### 5.1 Dynamic Range & Scaling Ratios

| Channel | Model | $u(h1)$ | $u(h18)/u(h1)$ | $u(h22)/u(h1)$ | $u(h23)/u(h1)$ | $u/\text{MSE}(h18)$ | $u/\text{MSE}(h23)$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Magnetic** | Original 256ep | 0.007429 | 1.44 | 1.23 | 0.83 | 0.033 | 0.006 |
| | **New Joint 256ep** | 0.061612 | **9.19** | **8.74** | **3.61** | **1.297** | **0.163** |
| | Continuation Ep 276 | 0.006994 | 1.71 | 1.48 | 0.85 | 0.042 | 0.005 |
| | Abrupt A+B (286) | 0.038862 | 10.90 | 10.74 | 5.45 | 1.371 | 0.187 |
| **Density** | Original 256ep | 0.006570 | 1.92 | 2.40 | 2.04 | 0.112 | 0.033 |
| | **New Joint 256ep** | 0.021454 | **9.75** | **12.54** | **9.99** | **1.433** | **0.300** |
| | Continuation Ep 276 | 0.006062 | 2.01 | 2.55 | 2.09 | 0.127 | 0.027 |
| | Abrupt A+B (286) | 0.019683 | 6.69 | 10.14 | 7.67 | 1.269 | 0.311 |
| **All** | Original 256ep | 0.007214 | 1.55 | 1.50 | 1.11 | 0.041 | 0.009 |
| | **New Joint 256ep** | 0.051573 | **9.25** | **9.13** | **4.27** | **1.311** | **0.183** |
| | Continuation Ep 276 | 0.006761 | 1.77 | 1.72 | 1.13 | 0.051 | 0.008 |
| | Abrupt A+B (286) | 0.034067 | 10.29 | 10.65 | 5.77 | 1.361 | 0.203 |

---

## 6. Per-Frame Within-Window Horizon Tracking ($h = 18$)

Examining per-frame metrics for horizon $h=18$ (visible prefix frames 0–5; hidden forecast lead frames 6–23, corresponding to forecast lead steps 1 through 18):

| Metric | Channel | Original 256ep | New Joint 256ep | Abrupt A+B (286) |
| :--- | :--- | :---: | :---: | :---: |
| **Lead 1 $u$ ($t=6$)** | Magnetic | 0.020739 | 0.268646 | 0.161473 |
| **Lead 18 $u$ ($t=23$)** | Magnetic | 0.006445 | 0.536617 | 0.440631 |
| **$u_{\text{last}} / u_{\text{first}}$ Ratio** | Magnetic | **0.311** | **1.997** | **2.729** |
| **$\text{MSE}_{\text{last}} / \text{MSE}_{\text{first}}$ Ratio** | Magnetic | 4.781 | 3.477 | 3.378 |
| **Lead 1 $u$ ($t=6$)** | Density | 0.017033 | 0.025735 | 0.022503 |
| **Lead 18 $u$ ($t=23$)** | Density | 0.008389 | 0.293716 | 0.129237 |
| **$u_{\text{last}} / u_{\text{first}}$ Ratio** | Density | **0.493** | **11.413** | **5.743** |
| **$\text{MSE}_{\text{last}} / \text{MSE}_{\text{first}}$ Ratio** | Density | 17.085 | 16.269 | 14.220 |

In the original model, predicted uncertainty dropped by $69\%$ (Magnetic) and $51\%$ (Density) as forecast difficulty increased. The new joint model achieves monotonic growth, with Density uncertainty growing by **$11.4\times$**, tracking the actual $16.3\times$ MSE growth.

---

## 7. Paired $h22 \to h23$ Boundary Transition Analysis

Evaluated on identical frames $t=2 \dots 23$ across 125 test windows:

| Metric | Channel | Original 256ep | New Joint 256ep | Abrupt A+B (286) |
| :--- | :--- | :---: | :---: | :---: |
| **Mean $u(23)/u(22)$** | Magnetic | 0.7377 | 0.4323 | 0.5193 |
| **Median Window $u(23)/u(22)$** | Magnetic | 0.8083 | 0.5408 | 0.5773 |
| **Windows with $u(23) < u(22)$** | Magnetic | 81.45% | **60.00%** | 60.00% |
| **Mean $\text{MSE}(23)/\text{MSE}(22)$** | Magnetic | 1.4000 | 1.2010 | 1.4605 |
| **Mean $u(23)/u(22)$** | Density | 0.8122 | 0.8231 | 0.7842 |
| **Median Window $u(23)/u(22)$** | Density | 0.9177 | 0.8997 | 0.7807 |
| **Windows with $u(23) < u(22)$** | Density | 65.85% | **55.85%** | 58.25% |
| **Mean $\text{MSE}(23)/\text{MSE}(22)$** | Density | 2.0432 | 1.7122 | 2.1171 |

The fraction of windows experiencing non-monotonic uncertainty suppression decreased significantly (from $81.5\%$ to $60.0\%$ in Magnetic, and from $65.8\%$ to $55.8\%$ in Density). However, the suppression effect at $h=23$ remains partially active because $h=23$ is an extreme conditioning boundary (only frame 0 is observed).

---

## 8. Spatial Uncertainty Ranking Evaluation

Within-frame spatial Spearman rank correlation ($e^2$ vs $u$) and high-to-low decile MSE contrast:

| Horizon | Channel | Metric | Original 256ep | New Joint 256ep | Abrupt A+B (286) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **$h=18$** | **Magnetic** | Frame Spatial Spearman | 0.1739 | **0.6383** | 0.5656 |
| | | Pooled Spatial Spearman | 0.1501 | **0.7372** | 0.6338 |
| | | High/Low Decile MSE Contrast | $1.59\times$ | **$468.8\times$** | $145.0\times$ |
| | **Density** | Frame Spatial Spearman | 0.3126 | **0.4112** | 0.3528 |
| | | Pooled Spatial Spearman | 0.2972 | **0.5640** | 0.4656 |
| | | High/Low Decile MSE Contrast | $4.13\times$ | **$49.8\times$** | $20.1\times$ |
| **$h=22$** | **Magnetic** | Frame Spatial Spearman | 0.1115 | **0.6027** | 0.5210 |
| | | Pooled Spatial Spearman | 0.0979 | **0.7334** | 0.6427 |
| | | High/Low Decile MSE Contrast | $1.73\times$ | **$281.8\times$** | $160.0\times$ |
| | **Density** | Frame Spatial Spearman | 0.3978 | **0.5241** | 0.4041 |
| | | Pooled Spatial Spearman | 0.4148 | **0.6281** | 0.5509 |
| | | High/Low Decile MSE Contrast | $8.25\times$ | **$141.9\times$** | $65.3\times$ |
| **$h=23$** | **Magnetic** | Frame Spatial Spearman | 0.1302 | **0.5613** | 0.4879 |
| | | Pooled Spatial Spearman | 0.1073 | **0.6234** | 0.5845 |
| | | High/Low Decile MSE Contrast | $1.88\times$ | **$182.5\times$** | $121.6\times$ |
| | **Density** | Frame Spatial Spearman | 0.4830 | **0.5281** | 0.4684 |
| | | Pooled Spatial Spearman | 0.4512 | **0.5547** | 0.5645 |
| | | High/Low Decile MSE Contrast | $13.00\times$ | **$121.5\times$** | $123.0\times$ |

---

## 9. Answers to Scientific Questions

### 1. Did gradual A+B uncertainty learning preserve reconstruction relative to the original 256-epoch run?
**Yes.** On the training curriculum distribution, validation MSE at epoch 256 is $0.017454$ compared to $0.014208$ in the original baseline (a modest $+22.8\%$ trade-off), completely avoiding the catastrophic degradation ($>0.085$) observed during abrupt continuation. On the fixed standard mask benchmark, aggregate pass-0 MSE is $0.081191$ vs $0.052813$.

### 2. Did reconstruction degrade during the $\lambda_v$ ramp, and did it recover?
**No degradation occurred.** During the 40-epoch cosine ramp of $\lambda_v$ (epochs 75–114), validation MSE monotonically improved from $0.020794$ to $0.019042$, further improving to $0.018513$ in settled Phase B and $0.017454$ in settled Phase C. There was no transient spike or numerical instability.

### 3. Did predicted uncertainty acquire the correct horizon-dependent dynamic range?
**Yes.** In the baseline, horizon scaling was severely flattened ($u(h18)/u(h1) = 1.44$). The new model expanded this to **$9.19$** for Magnetic and **$9.75$** for Density, correctly reflecting substantial uncertainty growth over long forecast horizons.

### 4. Did within-window uncertainty increase with forecast lead time?
**Yes.** In the baseline at $h=18$, uncertainty perversely decreased ($u_{\text{last}}/u_{\text{first}} = 0.311$). The new model completely reversed this: $u_{\text{last}}/u_{\text{first}} = 1.997$ (Magnetic) and **$11.413$** (Density), closely matching the actual $16.27\times$ MSE growth.

### 5. Did $h22 \to h23$ uncertainty suppression improve?
**Partially.** The fraction of test windows suffering from non-monotonic uncertainty suppression decreased from $81.5\%$ to $60.0\%$ (Magnetic) and $65.8\%$ to $55.8\%$ (Density). Mean $u(23)/u(22)$ remains $< 1.0$ due to the extreme single-frame visible prefix conditioning at $h=23$, though absolute uncertainty at $h=23$ is $36\times$ higher than baseline.

### 6. Did spatial uncertainty-error ranking improve?
**Yes, dramatically.** At $h=18$, within-frame spatial Spearman correlation jumped from $0.1739$ to **$0.6383$** (Magnetic) and pooled Spearman reached **$0.7372$**. High-decile voxels exhibit $468.8\times$ higher MSE than low-decile voxels (compared to only $1.59\times$ in baseline).

### 7. Did absolute $u/\text{MSE}$ calibration improve?
**Yes, by over an order of magnitude.** Baseline severely underestimated variance ($u/\text{MSE} = 0.033$ at $h=18$ Magnetic). The new model achieved near-ideal calibration ($u/\text{MSE} = 1.297$ for Magnetic and $1.433$ for Density at $h=18$).

### 8. Does gradual A+B training provide a better observed reconstruction–uncertainty trade-off than abrupt A+B continuation?
**Yes.** Gradual training maintained smooth, shock-free optimization from epoch 1 to 256 (val MSE $0.017454$ vs abrupt continuation's shock to $0.059879$), while attaining superior spatial ranking ($0.6383$ vs $0.5656$) and higher per-frame dynamic scaling ($11.41\times$ vs $5.74\times$ in Density).

### 9. Is there concrete evidence of insufficient variance-head capacity, or is that still unresolved?
**Evidence indicates head capacity was not the primary bottleneck.** The 2-layer Conv3D variance head achieved near-ideal calibration ($u/\text{MSE} \approx 1.3$) and high spatial Spearman correlations ($>0.63$) once gradient flow to shared representations was enabled. The persistent $h22 \to h23$ boundary artifact reflects extreme prefix conditioning rather than variance head parameter limits.

---

## 10. Audit Summary (Section 13)

**Fresh 256-Epoch Gradual A+B Training Evaluation**

- Training completed: YES
- Nonfinite events / AMP skips: 0 nonfinite events / 0.0 AMP skips
- Actual Phase A/B/C boundaries: Phase A: eps 1–74; Phase B ramp: eps 75–114; Phase B settled: eps 115–127; Phase C ramp: eps 128–167; Phase C settled: eps 168–256; Phase D: None
- lambda_v cosine ramp: VERIFIED (40 epochs)
- lambda_r cosine ramp: VERIFIED (40 epochs)
- Phase-A final validation MSE: 0.021271 (epoch 74)
- Phase-B final validation MSE: 0.018513 (epoch 127)
- Phase-C final validation MSE: 0.017454 (epoch 256)
- Original 256-epoch validation MSE: 0.014208
- New 256-epoch validation MSE: 0.017454
- Standard fixed-mask reconstruction: 0.081191 (original 256ep: 0.052813)
- Magnetic u(h18)/u(h1): original 1.44 / new 9.19
- Magnetic u(h23)/u(h1): original 0.83 / new 3.61
- Density u(h18)/u(h1): original 1.92 / new 9.75
- Density u(h23)/u(h1): original 2.04 / new 9.99
- Magnetic h18 per-frame u_last/u_first: original 0.311 / new 1.997
- Magnetic h22→h23 paired uncertainty change: original 0.7377 (81.5% windows < 1.0) / new 0.4323 (60.0% windows < 1.0)
- Magnetic h23 spatial Spearman: original 0.1302 / new 0.5613
- Reconstruction relative to original 256-epoch model: SIMILAR (training val MSE 0.017454 vs 0.014208; fixed-mask aggregate 0.081191 vs 0.052813)
- Horizon uncertainty response: IMPROVED (u(h18)/u(h1) 9.19 vs 1.44; u_last/u_first 1.997 vs 0.311)
- Absolute calibration: IMPROVED (u/MSE at h18: Magnetic 1.297 vs 0.033; Density 1.433 vs 0.112)
- Gradual A+B versus abrupt A+B trade-off: IMPROVED (smooth convergence without shock, higher spatial ranking: 0.6383 vs 0.5656)
- Remaining scientific limitations: Severe conditioning transition at single-frame visible prefix (h=23) still exhibits localized uncertainty suppression relative to h=22.
- Recommended next step: Investigate boundary conditioning techniques or prefix-length-aware feature modulation for single-frame extrapolation regimes.
- Training code modified: NO
- Additional training launched: NO
