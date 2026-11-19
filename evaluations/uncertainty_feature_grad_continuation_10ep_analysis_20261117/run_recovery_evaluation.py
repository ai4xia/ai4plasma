#!/usr/bin/env python3
"""Run treatment-only pass-0 forecast inference on 125 test windows,
verify identity with archived baseline, and combine all evaluation results.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
PROJECT = Path('/pscratch/sd/b/binxia/ai4plasma')
sys.path.insert(0, str(PROJECT))

import csv
import json
import math
import os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import rankdata, spearmanr
import torch

from data.masking import make_uncertain_input
from data.vpic_hdf5_dataset import VPICWindowDataset, load_fixed_split, validate_checkpoint_split
from models.unet3d import UNet3D
from visualize_mask_patterns_unet3d import select_validation_statistics_indices

OUTPUT = Path(__file__).resolve().parent
BASE_DIR = PROJECT / 'runs/multi_beta_three_phase_temporal_extrapolation_continuation_20ep_20261116'
TREAT_DIR = PROJECT / 'runs/multi_beta_three_phase_uncertainty_feature_grad_continuation_10ep_20261117'

ARCHIVE_HORIZON_DIR = PROJECT / 'evaluations/temporal_extrapolation_continuation_20ep_analysis_20261117'
ARCHIVE_DIAGNOSTIC_DIR = PROJECT / 'evaluations/forecast_uncertainty_collapse_diagnostic_20261117'

BASE_CKPT = BASE_DIR / 'final.pt'
TREAT_CKPT = TREAT_DIR / 'final.pt'

SEED = 1234
HORIZONS = list(range(1, 24))
FOCUS_HORIZONS = [18, 22, 23]
SAMPLE_LIMIT = 256

GROUPS = {
    'Bx': (0,), 'By': (1,), 'Bz': (2,), 'Density': (3,),
    'Magnetic': (0, 1, 2), 'All': (0, 1, 2, 3),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(v):
    if isinstance(v, dict):
        return {str(k): json_ready(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_ready(val) for val in v]
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    return v


def dump_json(path: Path, v) -> None:
    path.write_text(json.dumps(json_ready(v), indent=2, allow_nan=True))


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ckpt['args']
    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=int(args.get('base_channels', 24)),
        channel_mults=args.get('channel_mults', [1, 2, 4, 8]),
        architecture=args['model_version'],
        uncertainty_aware=bool(args['uncertainty_aware']),
        use_attention=bool(args.get('use_attention', False)),
        spatial_only_pooling=bool(args.get('spatial_only_pooling', False)),
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    stats_payload = ckpt['stats']
    mean = torch.tensor(stats_payload['mean'], device=device).view(1, 4, 1, 1, 1)
    std = torch.tensor(stats_payload['std'], device=device).view(1, 4, 1, 1, 1)
    return ckpt, model, mean, std


@torch.no_grad()
def infer(model, model_input):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        pred, var = model(model_input, detach_variance_features=True)
    return pred.float(), var.float()


def batches(dataset, indices: list[int], batch_size: int = 2):
    for start in range(0, len(indices), batch_size):
        chosen = indices[start:start + batch_size]
        samples = [dataset[index] for index in chosen]
        yield chosen, torch.stack([s['block'] for s in samples]), [s['metadata'] for s in samples]


def quantiles(e, u):
    bins = np.minimum((10 * (rankdata(u, method='average') - .5) / len(u)).astype(int), 9)
    return [(int(q), float(e[bins == q].sum()), float(u[bins == q].sum()), int((bins == q).sum()))
            for q in np.unique(bins)]


def new_bucket():
    return {
        'sum_e2': 0.0,
        'sum_u': 0.0,
        'count': 0,
        'window_mse': [],
        'window_u': [],
        'residual_samples': [],
        'u_samples': [],
    }


def add_metrics(bucket: dict, residual: torch.Tensor, variance: torch.Tensor,
                unknown: torch.Tensor, rng: np.random.Generator):
    for b in range(residual.shape[0]):
        choose = unknown[b]
        r = residual[b][choose].float().cpu().numpy()
        u = variance[b][choose].float().cpu().numpy()
        keep = np.isfinite(r) & np.isfinite(u) & (u > 0)
        r, u = r[keep], u[keep]
        if not r.size:
            continue
        e2 = np.square(r, dtype=np.float64)
        bucket['sum_e2'] += float(e2.sum())
        bucket['sum_u'] += float(u.astype(np.float64).sum())
        bucket['count'] += int(r.size)
        bucket['window_mse'].append(float(e2.mean()))
        bucket['window_u'].append(float(u.mean()))
        if r.size > SAMPLE_LIMIT:
            take = rng.choice(r.size, SAMPLE_LIMIT, replace=False)
            r, u = r[take], u[take]
        bucket['residual_samples'].append(r.astype(np.float32))
        bucket['u_samples'].append(u.astype(np.float32))


def finish_bucket(bucket: dict):
    residual = np.concatenate(bucket['residual_samples']) if bucket['residual_samples'] else np.empty(0)
    variance = np.concatenate(bucket['u_samples']) if bucket['u_samples'] else np.empty(0)
    mean_mse = bucket['sum_e2'] / bucket['count'] if bucket['count'] > 0 else 0.0
    mean_u = bucket['sum_u'] / bucket['count'] if bucket['count'] > 0 else 0.0
    e2_sample = np.square(residual)
    return {
        'voxel_count': bucket['count'],
        'window_count': len(bucket['window_mse']),
        'mean_mse': mean_mse,
        'median_window_mse': float(np.median(bucket['window_mse'])) if bucket['window_mse'] else 0.0,
        'mean_predicted_u': mean_u,
        'median_window_mean_u': float(np.median(bucket['window_u'])) if bucket['window_u'] else 0.0,
        'median_predicted_u_sampled': float(np.median(variance)) if variance.size else 0.0,
        'predicted_u_over_actual_mse': (mean_u / mean_mse) if mean_mse > 0 else 0.0,
        'actual_mse_over_predicted_u': (mean_mse / mean_u) if mean_u > 0 else 0.0,
        'sampled_pearson_e2_u': float(stats.pearsonr(e2_sample, variance).statistic) if len(residual) > 1 else float('nan'),
        'sampled_spearman_e2_u': float(stats.spearmanr(e2_sample, variance).statistic) if len(residual) > 1 else float('nan'),
    }


def main():
    print('Starting minimal recovery evaluation on treatment checkpoint...', flush=True)
    assert torch.cuda.is_available(), 'CUDA device required'
    device = torch.device('cuda')

    # 1. Load checkpoints and verify identical conditions
    base_ckpt = torch.load(BASE_CKPT, map_location='cpu', weights_only=False)
    treat_ckpt, model_treat, mean_treat, std_treat = load_model(TREAT_CKPT, device)

    assert base_ckpt['epoch'] == 276
    assert treat_ckpt['epoch'] == 286
    assert base_ckpt['stats'] == treat_ckpt['stats'], 'Normalization stats mismatch'
    assert base_ckpt['split_sha256'] == treat_ckpt['split_sha256'], 'Split sha256 mismatch'
    validate_checkpoint_split(base_ckpt)
    validate_checkpoint_split(treat_ckpt)

    treat_args = treat_ckpt['args']
    assert treat_args['uncertainty_loss'] == 'student_t_nll' and treat_args['student_t_df'] == 3.0
    assert treat_args['uncertainty_feature_grad_continuation'] is True
    assert Path(treat_args['post_final_continuation']).resolve() == BASE_CKPT.resolve()

    # Verify dataset split and test windows against archived manifest
    split = load_fixed_split()
    dataset = VPICWindowDataset(
        h5_dir=treat_args['h5_dir'], betas=treat_args['betas'],
        delta_t=24, stride_t=1, layout='C T X Z', return_metadata=True,
    )
    test_indices = select_validation_statistics_indices(dataset, set(split['test_runs']), 24, 1)
    assert len(test_indices) == 125

    with open(ARCHIVE_DIAGNOSTIC_DIR / 'evaluation_manifest.json') as f:
        archived_manifest = json.load(f)
    archived_test_indices = [w['index'] for w in archived_manifest['test_windows']]
    assert test_indices == archived_test_indices, 'Test window selection mismatch with archived baseline'
    print(f'Verified: 125 test windows, normalization, split, and seed identical to archived baseline.', flush=True)

    # 2. Run treatment forecast evaluation (h = 1..23)
    forecast_buckets = defaultdict(new_bucket)
    forecast_rng = np.random.default_rng(SEED + 9)

    frame_buckets = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    paired_frame_dict = {}

    spatial_window_records = defaultdict(list)
    spatial_pooled = defaultdict(list)

    print('Evaluating treatment forecast horizon sweep (h = 1..23)...', flush=True)
    for batch_number, (chosen, block, _meta) in enumerate(batches(dataset, test_indices)):
        target = ((block.to(device) - mean_treat) / (std_treat + 1e-8)).float()
        noise_gen = torch.Generator(device=device).manual_seed(SEED + 200000 + batch_number)
        noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)

        for h in HORIZONS:
            prefix = 24 - h
            mask = torch.ones_like(target)
            mask[:, :, prefix:] = 0.0
            input_var = 1.0 - mask
            model_input = torch.cat([make_uncertain_input(target, input_var, noise), input_var], dim=1)
            unknown_all = mask < 0.5

            pred, var = infer(model_treat, model_input)
            res = pred - target

            # Horizon buckets
            for ch, ci in GROUPS.items():
                ci = list(ci)
                add_metrics(
                    forecast_buckets[(h, ch)],
                    res[:, ci], var[:, ci], unknown_all[:, ci], forecast_rng,
                )

            # Per-frame tracking
            res_np = res.cpu().numpy()
            var_np = var.cpu().numpy()
            for ch, ci in GROUPS.items():
                ci = list(ci)
                for t in range(prefix, 24):
                    e_t = np.square(res_np[:, ci, t], dtype=np.float64)
                    u_t = var_np[:, ci, t].astype(np.float64)
                    frame_buckets[(h, ch, t)] += [e_t.sum(), u_t.sum(), e_t.size]
                    for b, idx in enumerate(chosen):
                        w_mse = float(e_t[b].mean())
                        w_u = float(u_t[b].mean())
                        paired_frame_dict[(h, ch, t, idx)] = (w_mse, w_u)

            # Spatial ranking strictly for focus horizons
            if h in FOCUS_HORIZONS:
                for ch, ci in GROUPS.items():
                    ci = list(ci)
                    for b, idx in enumerate(chosen):
                        e_frames = np.square(res_np[b, ci, prefix:], dtype=np.float64).transpose(1, 0, 2, 3).reshape(h, -1)
                        u_frames = var_np[b, ci, prefix:].astype(np.float64).transpose(1, 0, 2, 3).reshape(h, -1)
                        e_flat, u_flat = e_frames.ravel(), u_frames.ravel()

                        er = rankdata(e_frames, axis=1)
                        ur = rankdata(u_frames, axis=1)
                        er -= er.mean(1, keepdims=True)
                        ur -= ur.mean(1, keepdims=True)
                        denom = np.sqrt((er * er).sum(1) * (ur * ur).sum(1))
                        frame_rho = np.where(denom > 0, (er * ur).sum(1) / denom, 0.0)
                        mean_frame_spatial_spearman_exact = float(np.mean(frame_rho))

                        rho_st = float(spearmanr(e_flat, u_flat).statistic) if e_flat.size > 1 else float('nan')
                        qwindow = quantiles(e_flat, u_flat)
                        low, high = qwindow[0], qwindow[-1]
                        high_over_low = (high[1] / high[3]) / (low[1] / low[3]) if (low[1] > 0 and low[3] > 0) else float('nan')

                        spatial_window_records[(h, ch)].append({
                            'dataset_index': idx,
                            'mean_frame_spatial_spearman_exact': mean_frame_spatial_spearman_exact,
                            'spearman_space_time_exact': rho_st,
                            'high_u_decile_mse_over_low': high_over_low,
                        })

                        rng_sample = np.random.default_rng(SEED + 300000 + idx)
                        take = rng_sample.choice(e_flat.size, min(4096, e_flat.size), replace=False)
                        spatial_pooled[(h, ch)].append((e_flat[take], u_flat[take]))

        if batch_number == 0 or (batch_number + 1) % 10 == 0:
            print(f'Forecast evaluation: {min((batch_number + 1) * 2, len(test_indices))}/{len(test_indices)} windows complete', flush=True)

    print('Treatment GPU inference completed. Combining with archived baseline...', flush=True)

    # 3. Load Archived Baseline Results
    # 3a. Baseline forecast horizon metrics
    with open(ARCHIVE_HORIZON_DIR / 'forecast_horizon_metrics.csv') as f:
        archived_forecast_rows = [
            dict(r, model='baseline')
            for r in csv.DictReader(f)
            if r['model'] == 'extrapolation_continuation'
        ]
    assert len(archived_forecast_rows) == 138, f'Expected 138 baseline forecast rows, got {len(archived_forecast_rows)}'

    # 3b. Baseline per-frame metrics
    with open(ARCHIVE_DIAGNOSTIC_DIR / 'per_frame_metrics.csv') as f:
        archived_frame_rows = [
            dict(r, model='baseline')
            for r in csv.DictReader(f)
            if r.get('pass_name') == 'pass0' or 'pass_name' not in r
        ]

    # 3c. Baseline h22 vs h23 paired
    with open(ARCHIVE_DIAGNOSTIC_DIR / 'h22_vs_h23_paired.csv') as f:
        archived_paired_rows = [dict(r, model='baseline') for r in csv.DictReader(f)]

    # 3d. Baseline spatial metrics
    with open(ARCHIVE_DIAGNOSTIC_DIR / 'spatial_uncertainty_metrics.csv') as f:
        archived_spatial_rows = [dict(r, model='baseline') for r in csv.DictReader(f)]

    # 4. Format Treatment Results & Combine
    # 4a. Treatment forecast horizon metrics
    treat_forecast_rows = []
    for (h, ch), bucket in sorted(forecast_buckets.items()):
        summary = finish_bucket(bucket)
        treat_forecast_rows.append({
            'model': 'treatment', 'horizon': h, 'visible_prefix': 24 - h,
            'channel': ch, **summary,
        })

    # Combined forecast horizon metrics
    all_forecast_rows = archived_forecast_rows + treat_forecast_rows
    # Add u_over_h1 and mse_over_h1 for both models
    by_model_h_ch = {}
    for r in all_forecast_rows:
        for k in ('mean_mse', 'mean_predicted_u', 'predicted_u_over_actual_mse', 'actual_mse_over_predicted_u', 'median_window_mse', 'median_window_mean_u'):
            if k in r and r[k]:
                r[k] = float(r[k])
        by_model_h_ch[(r['model'], int(r['horizon']), r['channel'])] = r

    for r in all_forecast_rows:
        m, h, ch = r['model'], int(r['horizon']), r['channel']
        h1_row = by_model_h_ch[(m, 1, ch)]
        r['u_over_h1'] = r['mean_predicted_u'] / h1_row['mean_predicted_u'] if h1_row['mean_predicted_u'] > 0 else 0.0
        r['mse_over_h1'] = r['mean_mse'] / h1_row['mean_mse'] if h1_row['mean_mse'] > 0 else 0.0

    write_csv(OUTPUT / 'forecast_horizon_metrics.csv', all_forecast_rows)

    # 4b. forecast_horizon_comparison.csv
    comparison_rows = []
    for h in HORIZONS:
        for ch in ['All', 'Magnetic', 'Density', 'Bx', 'By', 'Bz']:
            b_row = by_model_h_ch[('baseline', h, ch)]
            t_row = by_model_h_ch[('treatment', h, ch)]
            comparison_rows.append({
                'horizon': h,
                'visible_prefix': 24 - h,
                'channel': ch,
                'baseline_mse': b_row['mean_mse'],
                'treatment_mse': t_row['mean_mse'],
                'mse_ratio_treatment_over_base': t_row['mean_mse'] / b_row['mean_mse'] if b_row['mean_mse'] > 0 else float('nan'),
                'baseline_mean_u': b_row['mean_predicted_u'],
                'treatment_mean_u': t_row['mean_predicted_u'],
                'u_ratio_treatment_over_base': t_row['mean_predicted_u'] / b_row['mean_predicted_u'] if b_row['mean_predicted_u'] > 0 else float('nan'),
                'baseline_u_over_mse': b_row['predicted_u_over_actual_mse'],
                'treatment_u_over_mse': t_row['predicted_u_over_actual_mse'],
            })
    write_csv(OUTPUT / 'forecast_horizon_comparison.csv', comparison_rows)

    # 4c. per_frame_metrics.csv
    treat_frame_rows = []
    for (h, ch, t), (se, su, n) in sorted(frame_buckets.items()):
        prefix = 24 - h
        mean_mse = se / n
        mean_u = su / n
        treat_frame_rows.append({
            'model': 'treatment',
            'pass_name': 'pass0',
            'horizon': h,
            'channel': ch,
            'local_frame': t,
            'last_pass0_observed_frame': prefix - 1,
            'forecast_lead_from_pass0': t - prefix + 1,
            'mean_mse': mean_mse,
            'mean_predicted_u': mean_u,
            'u_over_mse': mean_u / mean_mse if mean_mse > 0 else float('nan'),
            'voxel_count': int(n),
            'window_count': len(test_indices),
        })
    write_csv(OUTPUT / 'per_frame_metrics.csv', archived_frame_rows + treat_frame_rows)

    # 4d. h22_vs_h23_paired.csv
    treat_paired_rows = []
    frame_dict_indexed = {(r['horizon'], r['channel'], r['local_frame']): r for r in treat_frame_rows}
    for ch in ('Bx', 'By', 'Bz', 'Density', 'Magnetic', 'All'):
        for t in range(2, 24):
            a = frame_dict_indexed[(22, ch, t)]
            b = frame_dict_indexed[(23, ch, t)]
            mse_ratios = [paired_frame_dict[(23, ch, t, idx)][0] / paired_frame_dict[(22, ch, t, idx)][0] for idx in test_indices]
            u_ratios = [paired_frame_dict[(23, ch, t, idx)][1] / paired_frame_dict[(22, ch, t, idx)][1] for idx in test_indices]
            cal_ratios = [(paired_frame_dict[(23, ch, t, idx)][1] / paired_frame_dict[(23, ch, t, idx)][0]) /
                          (paired_frame_dict[(22, ch, t, idx)][1] / paired_frame_dict[(22, ch, t, idx)][0])
                          for idx in test_indices]
            treat_paired_rows.append({
                'model': 'treatment',
                'channel': ch,
                'local_frame': t,
                'lead_h22': t - 1,
                'lead_h23': t,
                'mse_h22': a['mean_mse'],
                'mse_h23': b['mean_mse'],
                'u_h22': a['mean_predicted_u'],
                'u_h23': b['mean_predicted_u'],
                'calibration_h22': a['u_over_mse'],
                'calibration_h23': b['u_over_mse'],
                'mse_h23_over_h22': b['mean_mse'] / a['mean_mse'],
                'mse_h23_over_h22_median_window': float(np.median(mse_ratios)),
                'mse_h23_over_h22_fraction_windows_below_one': float(np.mean(np.array(mse_ratios) < 1.0)),
                'u_h23_over_h22': b['mean_predicted_u'] / a['mean_predicted_u'],
                'u_h23_over_h22_median_window': float(np.median(u_ratios)),
                'u_h23_over_h22_fraction_windows_below_one': float(np.mean(np.array(u_ratios) < 1.0)),
                'calibration_h23_over_h22': b['u_over_mse'] / a['u_over_mse'],
                'calibration_h23_over_h22_median_window': float(np.median(cal_ratios)),
                'calibration_h23_over_h22_fraction_windows_below_one': float(np.mean(np.array(cal_ratios) < 1.0)),
            })
    write_csv(OUTPUT / 'h22_vs_h23_paired.csv', archived_paired_rows + treat_paired_rows)

    # 4e. spatial_uncertainty_metrics.csv
    treat_spatial_rows = []
    for (h, ch), win_list in sorted(spatial_window_records.items()):
        samples = spatial_pooled[(h, ch)]
        e_pooled = np.concatenate([x[0] for x in samples])
        u_pooled = np.concatenate([x[1] for x in samples])
        pooled_sp = float(spearmanr(e_pooled, u_pooled).statistic) if len(e_pooled) > 1 else float('nan')
        mean_frame_sp = float(np.mean([w['mean_frame_spatial_spearman_exact'] for w in win_list]))
        mean_win_sp = float(np.mean([w['spearman_space_time_exact'] for w in win_list]))
        med_win_sp = float(np.median([w['spearman_space_time_exact'] for w in win_list]))
        med_high_low = float(np.median([w['high_u_decile_mse_over_low'] for w in win_list]))

        treat_spatial_rows.append({
            'model': 'treatment', 'horizon': h, 'channel': ch, 'scope': 'ranking_summary', 'quantile': '',
            'mean_frame_spatial_spearman': mean_frame_sp,
            'pooled_sampled_spearman': pooled_sp,
            'pooled_sample_count': len(e_pooled),
            'mean_window_spearman': mean_win_sp,
            'median_window_spearman': med_win_sp,
            'median_window_high_over_low_mse': med_high_low,
        })
        for q, se, su, n in quantiles(e_pooled, u_pooled):
            treat_spatial_rows.append({
                'model': 'treatment', 'horizon': h, 'channel': ch, 'scope': 'pooled_sampled_decile', 'quantile': q + 1,
                'mean_mse': se / n, 'mean_predicted_u': su / n, 'u_over_mse': su / se if se > 0 else float('nan'),
                'voxel_count': int(n),
            })
    write_csv(OUTPUT / 'spatial_uncertainty_metrics.csv', archived_spatial_rows + treat_spatial_rows)

    # 5. Generate Figures
    print('Generating comparison plots...', flush=True)

    # Figure 1: forecast_horizon_comparison.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    comp_by_ch = defaultdict(dict)
    for r in comparison_rows:
        comp_by_ch[r['channel']][int(r['horizon'])] = r

    for channel, color in [('Density', 'blue'), ('Magnetic', 'red')]:
        values = comp_by_ch[channel]
        h_arr = np.array(sorted(values))
        base_mse = [values[x]['baseline_mse'] for x in h_arr]
        treat_mse = [values[x]['treatment_mse'] for x in h_arr]
        base_u = [values[x]['baseline_mean_u'] for x in h_arr]
        treat_u = [values[x]['treatment_mean_u'] for x in h_arr]
        base_cal = [values[x]['baseline_u_over_mse'] for x in h_arr]
        treat_cal = [values[x]['treatment_u_over_mse'] for x in h_arr]
        u_ratio = [values[x]['u_ratio_treatment_over_base'] for x in h_arr]
        mse_ratio = [values[x]['mse_ratio_treatment_over_base'] for x in h_arr]

        axes[0, 0].plot(h_arr, base_mse, marker='o', ms=3, ls='--', color=color, alpha=0.6, label=f'{channel} Baseline MSE')
        axes[0, 0].plot(h_arr, treat_mse, marker='o', ms=4, color=color, label=f'{channel} Feature-Grad MSE')

        axes[0, 1].plot(h_arr, base_u, marker='s', ms=3, ls='--', color=color, alpha=0.6, label=f'{channel} Baseline u')
        axes[0, 1].plot(h_arr, treat_u, marker='s', ms=4, color=color, label=f'{channel} Feature-Grad u')

        axes[1, 0].plot(h_arr, base_cal, marker='^', ms=3, ls='--', color=color, alpha=0.6, label=f'{channel} Baseline u/MSE')
        axes[1, 0].plot(h_arr, treat_cal, marker='^', ms=4, color=color, label=f'{channel} Feature-Grad u/MSE')

        axes[1, 1].plot(h_arr, u_ratio, marker='s', ms=4, color=color, label=f'{channel} u(Treat) / u(Base)')
        axes[1, 1].plot(h_arr, mse_ratio, marker='o', ms=3, ls=':', color=color, alpha=0.7, label=f'{channel} MSE(Treat) / MSE(Base)')

    axes[0, 0].set_title('Reconstruction MSE across Horizon h')
    axes[0, 0].set_xlabel('Forecast Horizon h')
    axes[0, 0].set_ylabel('Mean MSE')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].set_title('Predicted Uncertainty u across Horizon h')
    axes[0, 1].set_xlabel('Forecast Horizon h')
    axes[0, 1].set_ylabel('Mean Predicted u')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].set_title('Calibration Ratio u / MSE across Horizon h')
    axes[1, 0].set_xlabel('Forecast Horizon h')
    axes[1, 0].set_ylabel('u / MSE')
    axes[1, 0].axhline(1.0, color='gray', ls='--', alpha=0.5, label='Ideal u/MSE = 1.0')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].set_title('Treatment / Baseline Ratios (u vs MSE)')
    axes[1, 1].set_xlabel('Forecast Horizon h')
    axes[1, 1].set_ylabel('Ratio (Treatment / Baseline)')
    axes[1, 1].axhline(1.0, color='gray', ls='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle('Uncertainty Feature-Gradient Continuation vs Baseline (Epoch 276 -> 286)', fontsize=14)
    plt.savefig(OUTPUT / 'forecast_horizon_comparison.png', dpi=200)
    plt.close()

    # Figure 2: per_frame_mse_vs_u.png
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    frame_dict_all = {}
    for r in archived_frame_rows:
        frame_dict_all[('baseline', int(r['horizon']), r['channel'], int(r['local_frame']))] = {
            'mean_mse': float(r['mean_mse']), 'mean_predicted_u': float(r['mean_predicted_u'])
        }
    for r in treat_frame_rows:
        frame_dict_all[('treatment', int(r['horizon']), r['channel'], int(r['local_frame']))] = {
            'mean_mse': float(r['mean_mse']), 'mean_predicted_u': float(r['mean_predicted_u'])
        }

    for row_idx, h in enumerate([18, 22, 23]):
        prefix = 24 - h
        frames = list(range(prefix, 24))
        leads = [t - prefix + 1 for t in frames]

        for col_idx, (channel, color) in enumerate([('Density', 'blue'), ('Magnetic', 'red')]):
            ax = axes[row_idx, col_idx]
            base_mse_f = [frame_dict_all[('baseline', h, channel, t)]['mean_mse'] for t in frames]
            treat_mse_f = [frame_dict_all[('treatment', h, channel, t)]['mean_mse'] for t in frames]
            base_u_f = [frame_dict_all[('baseline', h, channel, t)]['mean_predicted_u'] for t in frames]
            treat_u_f = [frame_dict_all[('treatment', h, channel, t)]['mean_predicted_u'] for t in frames]

            ax.plot(leads, base_mse_f, marker='o', ms=3, ls='--', color='black', alpha=0.6, label='Base MSE')
            ax.plot(leads, treat_mse_f, marker='o', ms=4, color='black', label='Treat MSE')
            ax.plot(leads, base_u_f, marker='s', ms=3, ls='--', color=color, alpha=0.6, label=f'Base {channel} u')
            ax.plot(leads, treat_u_f, marker='s', ms=4, color=color, label=f'Treat {channel} u')

            ax.set_title(f'{channel} Per-Frame Lead Trajectory (h={h})')
            ax.set_xlabel('Forecast Lead Step')
            ax.set_ylabel('Magnitude')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    fig.suptitle('Per-Frame Error vs Predicted Uncertainty Trajectories (h=18, 22, 23)', fontsize=14)
    plt.savefig(OUTPUT / 'per_frame_mse_vs_u.png', dpi=200)
    plt.close()

    # 6. Build Diagnostic Summary JSON
    density_h1_b = by_model_h_ch[('baseline', 1, 'Density')]['mean_predicted_u']
    density_h18_b = by_model_h_ch[('baseline', 18, 'Density')]['mean_predicted_u']
    density_h23_b = by_model_h_ch[('baseline', 23, 'Density')]['mean_predicted_u']
    mag_h1_b = by_model_h_ch[('baseline', 1, 'Magnetic')]['mean_predicted_u']
    mag_h18_b = by_model_h_ch[('baseline', 18, 'Magnetic')]['mean_predicted_u']
    mag_h23_b = by_model_h_ch[('baseline', 23, 'Magnetic')]['mean_predicted_u']

    density_h1_t = by_model_h_ch[('treatment', 1, 'Density')]['mean_predicted_u']
    density_h18_t = by_model_h_ch[('treatment', 18, 'Density')]['mean_predicted_u']
    density_h23_t = by_model_h_ch[('treatment', 23, 'Density')]['mean_predicted_u']
    mag_h1_t = by_model_h_ch[('treatment', 1, 'Magnetic')]['mean_predicted_u']
    mag_h18_t = by_model_h_ch[('treatment', 18, 'Magnetic')]['mean_predicted_u']
    mag_h23_t = by_model_h_ch[('treatment', 23, 'Magnetic')]['mean_predicted_u']

    horizons_np = np.array(HORIZONS, dtype=float)
    spearman_summary = {}
    for m in ('baseline', 'treatment'):
        spearman_summary[m] = {}
        for ch in ('Density', 'Magnetic', 'All'):
            u_vals = np.array([by_model_h_ch[(m, h, ch)]['mean_predicted_u'] for h in HORIZONS])
            mse_vals = np.array([by_model_h_ch[(m, h, ch)]['mean_mse'] for h in HORIZONS])
            spearman_summary[m][ch] = {
                'spearman_h_u': float(stats.spearmanr(horizons_np, u_vals).statistic),
                'spearman_h_mse': float(stats.spearmanr(horizons_np, mse_vals).statistic),
            }

    mag_h18_first_b = frame_dict_all[('baseline', 18, 'Magnetic', 6)]
    mag_h18_last_b = frame_dict_all[('baseline', 18, 'Magnetic', 23)]
    mag_h18_first_t = frame_dict_all[('treatment', 18, 'Magnetic', 6)]
    mag_h18_last_t = frame_dict_all[('treatment', 18, 'Magnetic', 23)]

    mag_paired_b = [r for r in archived_paired_rows if r['channel'] == 'Magnetic']
    mag_paired_t = [r for r in treat_paired_rows if r['channel'] == 'Magnetic']

    mag_mse_change_b = sum(float(r['mse_h23']) for r in mag_paired_b) / sum(float(r['mse_h22']) for r in mag_paired_b) - 1.0
    mag_u_change_b = sum(float(r['u_h23']) for r in mag_paired_b) / sum(float(r['u_h22']) for r in mag_paired_b) - 1.0
    mag_mse_change_t = sum(float(r['mse_h23']) for r in mag_paired_t) / sum(float(r['mse_h22']) for r in mag_paired_t) - 1.0
    mag_u_change_t = sum(float(r['u_h23']) for r in mag_paired_t) / sum(float(r['u_h22']) for r in mag_paired_t) - 1.0

    spatial_dict = {}
    for r in archived_spatial_rows:
        if r.get('scope') == 'ranking_summary':
            spatial_dict[('baseline', int(r['horizon']), r['channel'])] = float(r['mean_frame_spatial_spearman'])
    for r in treat_spatial_rows:
        if r.get('scope') == 'ranking_summary':
            spatial_dict[('treatment', int(r['horizon']), r['channel'])] = float(r['mean_frame_spatial_spearman'])

    # Standard mask validation aggregate MSE
    with open(OUTPUT / 'standard_mask_metrics.csv') as f:
        std_rows = list(csv.DictReader(f))
    val_std_b = [float(r['mean_mse']) for r in std_rows if r['model'] == 'baseline' and r['role'] == 'val' and r['pass'] == 'pass0' and r['channel'] == 'All']
    val_std_t = [float(r['mean_mse']) for r in std_rows if r['model'] == 'treatment' and r['role'] == 'val' and r['pass'] == 'pass0' and r['channel'] == 'All']
    std_val_mse_b = sum(val_std_b) / len(val_std_b) if val_std_b else float('nan')
    std_val_mse_t = sum(val_std_t) / len(val_std_t) if val_std_t else float('nan')

    base_history = json.loads((BASE_DIR / 'history.json').read_text())
    treat_history = json.loads((TREAT_DIR / 'history.json').read_text())

    summary = {
        'response_ratios': {
            'density_u_h18_over_h1': {'baseline': density_h18_b / density_h1_b, 'treatment': density_h18_t / density_h1_t},
            'magnetic_u_h18_over_h1': {'baseline': mag_h18_b / mag_h1_b, 'treatment': mag_h18_t / mag_h1_t},
            'density_u_h23_over_h1': {'baseline': density_h23_b / density_h1_b, 'treatment': density_h23_t / density_h1_t},
            'magnetic_u_h23_over_h1': {'baseline': mag_h23_b / mag_h1_b, 'treatment': mag_h23_t / mag_h1_t},
        },
        'spearman_h_u': spearman_summary,
        'diagnostic_a_within_h18_magnetic': {
            'baseline': {
                'mse_last_over_first': mag_h18_last_b['mean_mse'] / mag_h18_first_b['mean_mse'],
                'u_last_over_first': mag_h18_last_b['mean_predicted_u'] / mag_h18_first_b['mean_predicted_u'],
            },
            'treatment': {
                'mse_last_over_first': mag_h18_last_t['mean_mse'] / mag_h18_first_t['mean_mse'],
                'u_last_over_first': mag_h18_last_t['mean_predicted_u'] / mag_h18_first_t['mean_predicted_u'],
            },
        },
        'diagnostic_b_h22_vs_h23_magnetic': {
            'baseline': {'mse_change': mag_mse_change_b, 'u_change': mag_u_change_b},
            'treatment': {'mse_change': mag_mse_change_t, 'u_change': mag_u_change_t},
        },
        'spatial_spearman_h23': {
            'magnetic': {'baseline': spatial_dict.get(('baseline', 23, 'Magnetic')),
                         'treatment': spatial_dict.get(('treatment', 23, 'Magnetic'))},
            'density': {'baseline': spatial_dict.get(('baseline', 23, 'Density')),
                        'treatment': spatial_dict.get(('treatment', 23, 'Density'))},
            'by': {'baseline': spatial_dict.get(('baseline', 23, 'By')),
                   'treatment': spatial_dict.get(('treatment', 23, 'By'))},
            'bx': {'baseline': spatial_dict.get(('baseline', 23, 'Bx')),
                   'treatment': spatial_dict.get(('treatment', 23, 'Bx'))},
            'bz': {'baseline': spatial_dict.get(('baseline', 23, 'Bz')),
                   'treatment': spatial_dict.get(('treatment', 23, 'Bz'))},
        },
        'standard_validation_mse': {
            'history_val_mse': {'baseline': base_history[-1]['val_mse'], 'treatment': treat_history[-1]['val_mse']},
            'fixed_mask_5family_val_mse': {'baseline': std_val_mse_b, 'treatment': std_val_mse_t},
        },
    }
    dump_json(OUTPUT / 'diagnostic_summary.json', summary)
    print('All CSVs, figures, and summaries generated successfully!', flush=True)


if __name__ == '__main__':
    main()
