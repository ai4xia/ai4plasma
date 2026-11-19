# Gaussian uncertainty continuation: paired scientific evaluation

## Scope and pairing

The epoch-276 Gaussian continuation was evaluated on exactly the same one-window-per-run sample from all 125 held-out test runs as the archived epoch-256 Student-t evaluation. For every horizon `h=1..23`, all four channels are visible in frames `0..23-h` and hidden in the suffix of length `h`; metrics include only hidden suffix voxels. Dataset split hash, channel normalization, selected windows, masks, and random-noise seeds are identical. The archived Student-t inference was reused rather than recomputed.

## Checkpoint and training audit

The continuation restored the epoch-256 model and optimizer, ran epochs 257–276 at constant LR `1e-5`, and saved a valid epoch-276 `final.pt`. It retained `direct_unet3d_no_activation_norm_v3`, `lambda_v=lambda_r=1`, two-forward K=1 settled Phase C, and no Phase D. Slurm job 61434528 completed with exit code 0, zero AMP-skipped steps, and zero nonfinite events.

| Final epoch-276 metric | Value |
|---|---:|
| Train reconstruction MSE | 0.0112725 |
| Validation reconstruction MSE | 0.0142348 |
| Train Gaussian NLL | -1.318831 |
| Validation Gaussian NLL | -1.154384 |
| Pass-0 training MSE | 0.0112725 |
| Pass-1 training MSE | 0.00399506 |
| Preclip gradient norm, mean / max | 1.68227 / 24.03535 |

Gaussian and Student-t NLL values are not numerically compared because they are different objectives.

## Forecast response ratios

| Channel/model | MSE h6/h1 | u h6/h1 | MSE h12/h1 | u h12/h1 | MSE h18/h1 | u h18/h1 | MSE h23/h1 | u h23/h1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Density, Student-t | 2.422 | 1.355 | 4.471 | 1.570 | 10.851 | 1.921 | 39.168 | 2.044 |
| Density, Gaussian | 2.399 | 1.032 | 4.434 | 1.154 | 11.268 | 1.360 | 42.012 | 1.678 |
| Magnetic, Student-t | 2.549 | 0.969 | 4.458 | 1.147 | 11.152 | 1.439 | 35.339 | 0.832 |
| Magnetic, Gaussian | 2.613 | 0.949 | 4.496 | 1.020 | 11.125 | 1.129 | 37.372 | 1.099 |
| All channels, Student-t | 2.536 | 1.057 | 4.459 | 1.243 | 11.120 | 1.549 | 35.745 | 1.108 |
| All channels, Gaussian | 2.590 | 0.969 | 4.489 | 1.052 | 11.141 | 1.184 | 37.875 | 1.237 |

Gaussian NLL did **not** increase the desired response amplitude. Density `u(h18)/u(h1)` fell from 1.92 to 1.36 and `u(h23)/u(h1)` fell from 2.04 to 1.68. Magnetic response at h18 fell from 1.44 to 1.13; h23 recovered from the Student-t model's anomalous 0.83 to 1.10, but remained far below the 37.37-fold MSE rise.

## Global inflation versus difficulty-dependent change

| Channel | Gaussian u / Student-t u, h1 | h6 | h12 | h18 | h23 | Spearman(h, ratio) |
|---|---:|---:|---:|---:|---:|---:|
| Density | 1.552 | 1.182 | 1.141 | 1.099 | 1.274 | -0.564 |
| Magnetic | 1.459 | 1.430 | 1.297 | 1.144 | 1.928 | -0.586 |
| All channels | 1.480 | 1.357 | 1.252 | 1.132 | 1.653 | -0.625 |

The ratio generally declines with horizon rather than grows. The h23 magnetic rebound is caused partly by the Student-t model's h23 `u` collapse and does not reverse the overall negative trend. The evidence therefore supports **global variance inflation**, concentrated more strongly at easy horizons, not difficulty-dependent improvement.

## Calibration and horizon ranking

| Channel/model | u/MSE h1 | h6 | h12 | h18 | h23 |
|---|---:|---:|---:|---:|---:|
| Density, Student-t | 0.631 | 0.353 | 0.222 | 0.112 | 0.033 |
| Density, Gaussian | 0.971 | 0.418 | 0.253 | 0.117 | 0.039 |
| Magnetic, Student-t | 0.254 | 0.096 | 0.065 | 0.033 | 0.006 |
| Magnetic, Gaussian | 0.377 | 0.137 | 0.085 | 0.038 | 0.011 |

Absolute ratios move closer to one at every selected horizon, but most of that gain occurs at h1. Density calibration is not flatter: the endpoint decrease is about 25-fold for Gaussian versus 19-fold for Student-t. Magnetic endpoint flatness improves modestly, but h23 remains underestimated by roughly 90-fold.

| Channel/model | Spearman(h,MSE) | Spearman(h,u) | Spearman(MSE,u) | Pearson(MSE,u) |
|---|---:|---:|---:|---:|
| Density, Student-t | 1.000 | 0.990 | 0.990 | 0.761 |
| Density, Gaussian | 1.000 | 0.982 | 0.982 | 0.839 |
| Magnetic, Student-t | 1.000 | 0.685 | 0.685 | 0.286 |
| Magnetic, Gaussian | 1.000 | 0.927 | 0.927 | 0.721 |
| All, Student-t | 1.000 | 0.879 | 0.879 | 0.493 |
| All, Gaussian | 1.000 | 0.964 | 0.964 | 0.821 |

Gaussian improves magnetic horizon ordering/correlation, but correlation alone does not imply amplitude calibration. All absolute hard-horizon variances remain much too small.

## Ordinary reconstruction regression check

| Fixed validation mask | Student-t MSE | Gaussian MSE | Gaussian change |
|---|---:|---:|---:|
| spatial_random | 0.002876 | 0.002794 | -2.85% |
| spatial_grid | 0.005300 | 0.004978 | -6.07% |
| spatial_block | 0.036072 | 0.036621 | +1.52% |
| temporal_random | 0.003445 | 0.003416 | -0.84% |
| temporal_block | 0.023347 | 0.023364 | +0.07% |
| aggregate | 0.014208 | 0.014235 | +0.19% |

Ordinary reconstruction is **similar** overall. Forecast MSE is also close through h18, but Gaussian is worse at h23: Density rises by 8.25% and Magnetic by 4.01% relative to Student-t.

## Predicted-u distribution

| Channel/model | Mean | Median | p90 | p99 | Near floor | Fraction >1 |
|---|---:|---:|---:|---:|---:|---:|
| Bx, Student-t | 0.001968 | 0.001368 | 0.002635 | 0.010710 | 8.77e-5 | 5.76e-6 |
| Bx, Gaussian | 0.002603 | 0.002053 | 0.003953 | 0.011048 | 6.85e-10 | 1.77e-5 |
| By, Student-t | 0.010335 | 0.005570 | 0.013731 | 0.074283 | 2.88e-4 | 2.88e-4 |
| By, Gaussian | 0.012336 | 0.010064 | 0.019013 | 0.053238 | 0 | 5.60e-5 |
| Bz, Student-t | 0.001144 | 0.000830 | 0.001502 | 0.005570 | 2.85e-5 | 6.70e-6 |
| Bz, Gaussian | 0.001496 | 0.001285 | 0.002185 | 0.005234 | 4.57e-10 | 1.28e-5 |
| Density, Student-t | 0.003165 | 0.001550 | 0.005399 | 0.025111 | 5.73e-6 | 9.11e-5 |
| Density, Gaussian | 0.003998 | 0.001812 | 0.006116 | 0.032129 | 8.09e-8 | 1.78e-4 |

The Gaussian model raises typical variance, particularly medians, while p99 and `u>1` remain small. There is no global variance explosion. However, the horizon comparison shows that this increase is not preferentially assigned to difficult forecasts.

## Heavy-tail and numerical diagnostic

On 736,000 bounded residual/u samples per channel, Gaussian training reduced standardized-residual extremes substantially (for Density, `std(r/sqrt(u))` 4.33→3.37 and |z| p99 14.93→12.22; for By, 79.30→5.24 and 108.43→21.41). Nevertheless, all channels remain strongly underdispersed and heavy-tailed at hard horizons. Combined with finite training, zero skipped steps, modest validation p99 variance, and negligible `u>1`, there is no evidence of numerical instability or pathological global variance blow-up.

## Phase-C pass-0/pass-1 control

Across the five matched observation-pattern transitions, mean pass-1 MSE remains much lower than matched pass-0 MSE for both models: 0.00896 versus 0.05233 for Student-t and 0.00868 versus 0.05308 for Gaussian. Gaussian therefore preserves the benefit of one-step detached continuous-u reuse. Mean pass-1 `u/MSE` shifts from 1.027 to 1.142, indicating mild overprediction on this easier reuse task, while long-horizon forecasting remains severely underpredicted.

## Artifacts

- `forecast_horizon_metrics.csv`: both models, all 23 horizons and six channel groupings, including Gaussian/Student-t ratios.
- `standard_mask_metrics.csv`: fixed-mask reconstruction comparison.
- `predicted_u_distribution.csv`: per-channel validation variance distributions.
- `pass0_pass1_metrics.csv`: matched K=1 reuse results.
- `forecast_residual_likelihood.csv`: bounded heavy-tail diagnostics.
- `gaussian_vs_studentt_horizon.png`, `gaussian_vs_studentt_calibration.png`, and `gaussian_vs_studentt_standard_masks.png`.

## Final scientific judgment

```text
Gaussian continuation audit

Training completed cleanly:
  YES

Gaussian continuation epochs:
  20

Density predicted-u h18/h1:
  Student-t: 1.92x
  Gaussian: 1.36x

Density actual MSE h18/h1:
  Student-t: 10.85x
  Gaussian: 11.27x

Magnetic predicted-u h18/h1:
  Student-t: 1.44x
  Gaussian: 1.13x

Magnetic actual MSE h18/h1:
  Student-t: 11.15x
  Gaussian: 11.13x

Density u/MSE at h18:
  Student-t: 0.112
  Gaussian: 0.117

Density u/MSE at h23:
  Student-t: 0.033
  Gaussian: 0.039

Difficulty-response amplitude:
  WORSE

Absolute calibration:
  IMPROVED

Ordinary reconstruction:
  SIMILAR

Gaussian variance behavior:
  GLOBAL INFLATION

Student-t semantics hypothesis:
  NOT SUPPORTED

Most important remaining issue:
  Predicted u still grows far more slowly than actual forecast MSE, and the Gaussian increase is largest at easy horizons rather than difficult horizons.

Recommended next action:
  Do not replace the epoch-256 Student-t model with this Gaussian continuation on the basis of uncertainty response; retain the paired results as a negative diagnostic control.
```

Five most important numerical comparisons:

1. Density h18 response: `u` 1.92x→1.36x while MSE is 10.85x→11.27x.
2. Magnetic h18 response: `u` 1.44x→1.13x while MSE stays approximately 11.15x→11.13x.
3. Gaussian/Student-t Density `u` falls from 1.55x at h1 to 1.10x at h18 (Spearman with horizon −0.564).
4. Density h23 `u/MSE` improves only from 0.033 to 0.039 while forecast MSE worsens by 8.25%.
5. Aggregate fixed-mask reconstruction changes only +0.19%, so the failed response-amplitude hypothesis is not explained by a broad reconstruction collapse.
