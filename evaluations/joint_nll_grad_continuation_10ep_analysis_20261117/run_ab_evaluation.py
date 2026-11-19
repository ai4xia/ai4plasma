#!/usr/bin/env python3
"""Paired frozen-model evaluation of A+B joint NLL gradient continuation
against Baseline (epoch 276) and B-only (epoch 286).
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

from data.masking import (
    MASK_PATTERNS,
    make_fixed_validation_mask,
    make_uncertain_input,
)
from data.vpic_hdf5_dataset import (
    VPICWindowDataset,
    load_fixed_split,
    validate_checkpoint_split,
)
from models.unet3d import UNet3D
from train_masked_unet3d import (
    make_phase_c_reuse_input,
    rollout_validation_reuse_pattern,
)
from visualize_mask_patterns_unet3d import select_validation_statistics_indices

OUTPUT = Path(__file__).resolve().parent
BASE_DIR = PROJECT / 'runs/multi_beta_three_phase_temporal_extrapolation_continuation_20ep_20261116'
BONLY_DIR = PROJECT / 'runs/multi_beta_three_phase_uncertainty_feature_grad_continuation_10ep_20261117'
AB_DIR = PROJECT / 'runs/multi_beta_three_phase_joint_nll_grad_continuation_10ep_20261117'

ARCHIVE_DIR = PROJECT / 'evaluations/uncertainty_feature_grad_continuation_10ep_analysis_20261117'

BASE_CKPT = BASE_DIR / 'final.pt'
BONLY_CKPT = BONLY_DIR / 'final.pt'
AB_CKPT = AB_DIR / 'final.pt'

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
    print('Starting A+B Joint NLL Gradient Continuation Evaluation...', flush=True)
    assert torch.cuda.is_available(), 'CUDA device required'
    device = torch.device('cuda')

    # 1. Audit Checkpoints and Histories
    base_ckpt = torch.load(BASE_CKPT, map_location='cpu', weights_only=False)
    bonly_ckpt = torch.load(BONLY_CKPT, map_location='cpu', weights_only=False)
    ab_ckpt, model_ab, mean_ab, std_ab = load_model(AB_CKPT, device)

    assert base_ckpt['epoch'] == 276
    assert bonly_ckpt['epoch'] == 286
    assert ab_ckpt['epoch'] == 286

    assert base_ckpt['stats'] == ab_ckpt['stats'] == bonly_ckpt['stats'], 'Normalization stats mismatch'
    assert base_ckpt['split_sha256'] == ab_ckpt['split_sha256'] == bonly_ckpt['split_sha256'], 'Split sha256 mismatch'
    validate_checkpoint_split(base_ckpt)
    validate_checkpoint_split(bonly_ckpt)
    validate_checkpoint_split(ab_ckpt)

    base_history = json.loads((BASE_DIR / 'history.json').read_text())
    bonly_history = json.loads((BONLY_DIR / 'history.json').read_text())
    ab_history = json.loads((AB_DIR / 'history.json').read_text())

    assert [row['epoch'] for row in ab_history] == list(range(277, 287))

    ab_args = ab_ckpt['args']
    assert ab_args['uncertainty_loss'] == 'student_t_nll' and ab_args['student_t_df'] == 3.0
    assert ab_args['joint_nll_grad_continuation'] is True
    assert ab_args['temporal_extrapolation_continuation'] is True
    assert Path(ab_args['post_final_continuation']).resolve() == BASE_CKPT.resolve()

    total_nonfinite = sum(r.get('nonfinite_events', 0) for r in ab_history)
    total_amp_skips = sum(r.get('amp_skipped_steps', 0) for r in ab_history)
    assert total_nonfinite == 0, f'Found {total_nonfinite} nonfinite events'
    assert total_amp_skips == 0, f'Found {total_amp_skips} AMP skipped steps'
    for r in ab_history:
        assert r['learning_rate'] == 1e-5

    training_audit = {
        'baseline_checkpoint': str(BASE_CKPT),
        'baseline_epoch': 276,
        'bonly_checkpoint': str(BONLY_CKPT),
        'bonly_epoch': 286,
        'ab_checkpoint': str(AB_CKPT),
        'ab_epoch': 286,
        'ab_epochs_completed': [r['epoch'] for r in ab_history],
        'ab_args_joint_nll_grad': ab_args.get('joint_nll_grad_continuation'),
        'nonfinite_events': total_nonfinite,
        'amp_skipped_steps': total_amp_skips,
        'learning_rate': 1e-5,
        'baseline_final_train_history': {
            'train_mse': base_history[-1]['train_mse'],
            'val_mse': base_history[-1]['val_mse'],
            'train_nll': base_history[-1].get('train_student_t_nll', base_history[-1].get('train_nll')),
            'val_nll': base_history[-1].get('val_student_t_nll', base_history[-1].get('val_nll')),
        },
        'bonly_final_train_history': {
            'train_mse': bonly_history[-1]['train_mse'],
            'val_mse': bonly_history[-1]['val_mse'],
            'train_nll': bonly_history[-1].get('train_student_t_nll', bonly_history[-1].get('train_nll')),
            'val_nll': bonly_history[-1].get('val_student_t_nll', bonly_history[-1].get('val_nll')),
        },
        'ab_final_train_history': {
            'train_mse': ab_history[-1]['train_mse'],
            'val_mse': ab_history[-1]['val_mse'],
            'train_nll': ab_history[-1].get('train_student_t_nll', ab_history[-1].get('train_nll')),
            'val_nll': ab_history[-1].get('val_student_t_nll', ab_history[-1].get('val_nll')),
        },
    }
    dump_json(OUTPUT / 'training_audit.json', training_audit)
    print(f'Training audit verified: 0 nonfinite, 0 AMP skips, val_mse={ab_history[-1]["val_mse"]:.6f} (vs B-only {bonly_history[-1]["val_mse"]:.6f}, Base {base_history[-1]["val_mse"]:.6f})', flush=True)

    # 2. Setup Dataset & Windows
    split = load_fixed_split()
    dataset = VPICWindowDataset(
        h5_dir=ab_args['h5_dir'], betas=ab_args['betas'],
        delta_t=24, stride_t=1, layout='C T X Z', return_metadata=True,
    )
    role_indices = {
        role: select_validation_statistics_indices(dataset, set(split[f'{role}_runs']), 24, 1)
        for role in ('val', 'test')
    }
    assert len(role_indices['val']) == 125
    assert len(role_indices['test']) == 125

    manifest = {
        'baseline_checkpoint': str(BASE_CKPT),
        'bonly_checkpoint': str(BONLY_CKPT),
        'ab_checkpoint': str(AB_CKPT),
        'split_sha256': ab_ckpt['split_sha256'],
        'normalization': ab_ckpt['stats'],
        'device': torch.cuda.get_device_name(),
        'val_windows': len(role_indices['val']),
        'test_windows': len(role_indices['test']),
        'seed': SEED,
    }
    dump_json(OUTPUT / 'evaluation_manifest.json', manifest)

    # 3. Standard Mask Benchmark (Val & Test across 5 tasks)
    print('Evaluating standard mask benchmark for A+B model...', flush=True)
    # Load archived baseline and B-only metrics
    with open(ARCHIVE_DIR / 'standard_mask_metrics.csv') as f:
        archived_standard = list(csv.DictReader(f))
    # Map 'treatment' -> 'b_only'
    standard_rows = []
    for r in archived_standard:
        model_name = 'b_only' if r['model'] == 'treatment' else r['model']
        standard_rows.append(dict(r, model=model_name))

    # Run for A+B model
    rng = np.random.default_rng(SEED)
    for role, indices in role_indices.items():
        buckets = defaultdict(new_bucket)
        for batch_number, (_idx, block, _meta) in enumerate(batches(dataset, indices)):
            target = ((block.to(device) - mean_ab) / (std_ab + 1e-8)).float()
            noise_gen = torch.Generator(device=device).manual_seed(SEED + 100000 * (role == 'test') + batch_number)
            noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)
            for task in MASK_PATTERNS:
                mask, _ = make_fixed_validation_mask(task, target.shape, device=device, dtype=target.dtype)
                input_var = 1.0 - mask
                model_input = torch.cat([make_uncertain_input(target, input_var, noise), input_var], dim=1)
                pred0, var0 = infer(model_ab, model_input)
                res0 = pred0 - target
                pass_data = [('pass0', task, mask, res0, var0)]

                reuse_task = rollout_validation_reuse_pattern(task)
                reuse_mask, _ = make_fixed_validation_mask(reuse_task, target.shape, device=device, dtype=target.dtype)
                reuse_input, exact_mask = make_phase_c_reuse_input(target, pred0, var0, reuse_mask)
                pred1, var1 = infer(model_ab, reuse_input)
                pass_data.append(('pass1', reuse_task, exact_mask, pred1 - target, var1))

                for pass_name, eff_task, exact, residual, variance in pass_data:
                    unknown_all = exact < 0.5
                    for ch, ci in GROUPS.items():
                        add_metrics(
                            buckets[(pass_name, task, eff_task, ch)],
                            residual[:, list(ci)], variance[:, list(ci)],
                            unknown_all[:, list(ci)], rng,
                        )
        for (pass_name, source_task, eff_task, ch), bucket in buckets.items():
            summary = finish_bucket(bucket)
            standard_rows.append({
                'model': 'a_plus_b', 'role': role, 'pass': pass_name,
                'source_task': source_task, 'effective_observation_task': eff_task,
                'channel': ch, **summary,
            })
        print(f'[a_plus_b] standard {role} complete', flush=True)

    write_csv(OUTPUT / 'standard_mask_metrics.csv', standard_rows)

    # 4. Forecast Horizon Evaluation (h = 1..23 on 125 test windows for A+B)
    print('Evaluating A+B forecast horizon sweep (h = 1..23)...', flush=True)
    forecast_buckets = defaultdict(new_bucket)
    forecast_rng = np.random.default_rng(SEED + 9)

    frame_buckets = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    paired_frame_dict = {}

    spatial_window_records = defaultdict(list)
    spatial_pooled = defaultdict(list)

    test_indices = role_indices['test']
    for batch_number, (chosen, block, _meta) in enumerate(batches(dataset, test_indices)):
        target = ((block.to(device) - mean_ab) / (std_ab + 1e-8)).float()
        noise_gen = torch.Generator(device=device).manual_seed(SEED + 200000 + batch_number)
        noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)

        for h in HORIZONS:
            prefix = 24 - h
            mask = torch.ones_like(target)
            mask[:, :, prefix:] = 0.0
            input_var = 1.0 - mask
            model_input = torch.cat([make_uncertain_input(target, input_var, noise), input_var], dim=1)
            unknown_all = mask < 0.5

            pred, var = infer(model_ab, model_input)
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

    print('A+B GPU inference completed. Combining with archived baseline and B-only...', flush=True)

    # 5. Load Archived Baseline and B-only Results
    with open(ARCHIVE_DIR / 'forecast_horizon_metrics.csv') as f:
        archived_forecast_raw = list(csv.DictReader(f))
    archived_forecast_rows = []
    for r in archived_forecast_raw:
        m = 'b_only' if r['model'] == 'treatment' else r['model']
        archived_forecast_rows.append(dict(r, model=m))

    with open(ARCHIVE_DIR / 'per_frame_metrics.csv') as f:
        archived_frame_raw = list(csv.DictReader(f))
    archived_frame_rows = []
    for r in archived_frame_raw:
        m = 'b_only' if r.get('model') == 'treatment' else r.get('model', 'baseline')
        archived_frame_rows.append(dict(r, model=m))

    with open(ARCHIVE_DIR / 'h22_vs_h23_paired.csv') as f:
        archived_paired_raw = list(csv.DictReader(f))
    archived_paired_rows = []
    for r in archived_paired_raw:
        m = 'b_only' if r.get('model') == 'treatment' else r.get('model', 'baseline')
        archived_paired_rows.append(dict(r, model=m))

    with open(ARCHIVE_DIR / 'spatial_uncertainty_metrics.csv') as f:
        archived_spatial_raw = list(csv.DictReader(f))
    archived_spatial_rows = []
    for r in archived_spatial_raw:
        m = 'b_only' if r.get('model') == 'treatment' else r.get('model', 'baseline')
        archived_spatial_rows.append(dict(r, model=m))

    # 6. Format A+B Results & Combine All Three Models
    # 6a. Forecast horizon metrics
    ab_forecast_rows = []
    for (h, ch), bucket in sorted(forecast_buckets.items()):
        summary = finish_bucket(bucket)
        ab_forecast_rows.append({
            'model': 'a_plus_b', 'horizon': h, 'visible_prefix': 24 - h,
            'channel': ch, **summary,
        })

    all_forecast_rows = archived_forecast_rows + ab_forecast_rows
    by_model_h_ch = {}
    for r in all_forecast_rows:
        for k in ('mean_mse', 'mean_predicted_u', 'predicted_u_over_actual_mse', 'actual_mse_over_predicted_u', 'median_window_mse', 'median_window_mean_u'):
            if k in r and r[k] != '':
                r[k] = float(r[k])
        by_model_h_ch[(r['model'], int(r['horizon']), r['channel'])] = r

    for r in all_forecast_rows:
        m, h, ch = r['model'], int(r['horizon']), r['channel']
        h1_row = by_model_h_ch[(m, 1, ch)]
        r['u_over_h1'] = r['mean_predicted_u'] / h1_row['mean_predicted_u'] if h1_row['mean_predicted_u'] > 0 else 0.0
        r['mse_over_h1'] = r['mean_mse'] / h1_row['mean_mse'] if h1_row['mean_mse'] > 0 else 0.0

    write_csv(OUTPUT / 'forecast_horizon_metrics.csv', all_forecast_rows)

    # 6b. forecast_horizon_comparison.csv (3-model comparison)
    comparison_rows = []
    for h in HORIZONS:
        for ch in ['All', 'Magnetic', 'Density', 'Bx', 'By', 'Bz']:
            b_row = by_model_h_ch[('baseline', h, ch)]
            bonly_row = by_model_h_ch[('b_only', h, ch)]
            ab_row = by_model_h_ch[('a_plus_b', h, ch)]

            comparison_rows.append({
                'horizon': h,
                'visible_prefix': 24 - h,
                'channel': ch,
                'baseline_mse': b_row['mean_mse'],
                'bonly_mse': bonly_row['mean_mse'],
                'a_plus_b_mse': ab_row['mean_mse'],
                'mse_ratio_ab_over_base': ab_row['mean_mse'] / b_row['mean_mse'] if b_row['mean_mse'] > 0 else float('nan'),
                'mse_ratio_ab_over_bonly': ab_row['mean_mse'] / bonly_row['mean_mse'] if bonly_row['mean_mse'] > 0 else float('nan'),
                'mse_ratio_bonly_over_base': bonly_row['mean_mse'] / b_row['mean_mse'] if b_row['mean_mse'] > 0 else float('nan'),
                'baseline_mean_u': b_row['mean_predicted_u'],
                'bonly_mean_u': bonly_row['mean_predicted_u'],
                'a_plus_b_mean_u': ab_row['mean_predicted_u'],
                'u_ratio_ab_over_base': ab_row['mean_predicted_u'] / b_row['mean_predicted_u'] if b_row['mean_predicted_u'] > 0 else float('nan'),
                'u_ratio_ab_over_bonly': ab_row['mean_predicted_u'] / bonly_row['mean_predicted_u'] if bonly_row['mean_predicted_u'] > 0 else float('nan'),
                'u_ratio_bonly_over_base': bonly_row['mean_predicted_u'] / b_row['mean_predicted_u'] if b_row['mean_predicted_u'] > 0 else float('nan'),
                'baseline_u_over_mse': b_row['predicted_u_over_actual_mse'],
                'bonly_u_over_mse': bonly_row['predicted_u_over_actual_mse'],
                'a_plus_b_u_over_mse': ab_row['predicted_u_over_actual_mse'],
                'baseline_u_over_h1': b_row['u_over_h1'],
                'bonly_u_over_h1': bonly_row['u_over_h1'],
                'a_plus_b_u_over_h1': ab_row['u_over_h1'],
            })
    write_csv(OUTPUT / 'forecast_horizon_comparison.csv', comparison_rows)

    # 6c. per_frame_metrics.csv
    ab_frame_rows = []
    for (h, ch, t), (se, su, n) in sorted(frame_buckets.items()):
        prefix = 24 - h
        mean_mse = se / n
        mean_u = su / n
        ab_frame_rows.append({
            'model': 'a_plus_b',
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
    write_csv(OUTPUT / 'per_frame_metrics.csv', archived_frame_rows + ab_frame_rows)

    # 6d. h22_vs_h23_paired.csv
    ab_paired_rows = []
    frame_dict_indexed = {(r['horizon'], r['channel'], r['local_frame']): r for r in ab_frame_rows}
    for ch in ('Bx', 'By', 'Bz', 'Density', 'Magnetic', 'All'):
        for t in range(2, 24):
            a = frame_dict_indexed[(22, ch, t)]
            b = frame_dict_indexed[(23, ch, t)]
            mse_ratios = [paired_frame_dict[(23, ch, t, idx)][0] / paired_frame_dict[(22, ch, t, idx)][0] for idx in test_indices]
            u_ratios = [paired_frame_dict[(23, ch, t, idx)][1] / paired_frame_dict[(22, ch, t, idx)][1] for idx in test_indices]
            cal_ratios = [(paired_frame_dict[(23, ch, t, idx)][1] / paired_frame_dict[(23, ch, t, idx)][0]) /
                          (paired_frame_dict[(22, ch, t, idx)][1] / paired_frame_dict[(22, ch, t, idx)][0])
                          for idx in test_indices]
            ab_paired_rows.append({
                'model': 'a_plus_b',
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
    write_csv(OUTPUT / 'h22_vs_h23_paired.csv', archived_paired_rows + ab_paired_rows)

    # 6e. spatial_uncertainty_metrics.csv
    ab_spatial_rows = []
    for (h, ch), win_list in sorted(spatial_window_records.items()):
        samples = spatial_pooled[(h, ch)]
        e_pooled = np.concatenate([x[0] for x in samples])
        u_pooled = np.concatenate([x[1] for x in samples])
        pooled_sp = float(spearmanr(e_pooled, u_pooled).statistic) if len(e_pooled) > 1 else float('nan')
        mean_frame_sp = float(np.mean([w['mean_frame_spatial_spearman_exact'] for w in win_list]))
        mean_win_sp = float(np.mean([w['spearman_space_time_exact'] for w in win_list]))
        med_win_sp = float(np.median([w['spearman_space_time_exact'] for w in win_list]))
        med_high_low = float(np.median([w['high_u_decile_mse_over_low'] for w in win_list]))

        ab_spatial_rows.append({
            'model': 'a_plus_b', 'horizon': h, 'channel': ch, 'scope': 'ranking_summary', 'quantile': '',
            'mean_frame_spatial_spearman': mean_frame_sp,
            'pooled_sampled_spearman': pooled_sp,
            'pooled_sample_count': len(e_pooled),
            'mean_window_spearman': mean_win_sp,
            'median_window_spearman': med_win_sp,
            'median_window_high_over_low_mse': med_high_low,
        })
        for q, se, su, n in quantiles(e_pooled, u_pooled):
            ab_spatial_rows.append({
                'model': 'a_plus_b', 'horizon': h, 'channel': ch, 'scope': 'pooled_sampled_decile', 'quantile': q + 1,
                'mean_mse': se / n, 'mean_predicted_u': su / n, 'u_over_mse': su / se if se > 0 else float('nan'),
                'voxel_count': int(n),
            })
    write_csv(OUTPUT / 'spatial_uncertainty_metrics.csv', archived_spatial_rows + ab_spatial_rows)

    # 7. Generate Figures
    print('Generating comparison plots (3-model)...', flush=True)

    # Figure 1: forecast_horizon_comparison.png
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    comp_by_ch = defaultdict(dict)
    for r in comparison_rows:
        comp_by_ch[r['channel']][int(r['horizon'])] = r

    for channel, base_color in [('Density', 'blue'), ('Magnetic', 'red')]:
        values = comp_by_ch[channel]
        h_arr = np.array(sorted(values))
        base_mse = [values[x]['baseline_mse'] for x in h_arr]
        bonly_mse = [values[x]['bonly_mse'] for x in h_arr]
        ab_mse = [values[x]['a_plus_b_mse'] for x in h_arr]

        base_u = [values[x]['baseline_mean_u'] for x in h_arr]
        bonly_u = [values[x]['bonly_mean_u'] for x in h_arr]
        ab_u = [values[x]['a_plus_b_mean_u'] for x in h_arr]

        base_cal = [values[x]['baseline_u_over_mse'] for x in h_arr]
        bonly_cal = [values[x]['bonly_u_over_mse'] for x in h_arr]
        ab_cal = [values[x]['a_plus_b_u_over_mse'] for x in h_arr]

        u_ratio_ab_base = [values[x]['u_ratio_ab_over_base'] for x in h_arr]
        u_ratio_ab_bonly = [values[x]['u_ratio_ab_over_bonly'] for x in h_arr]
        mse_ratio_ab_base = [values[x]['mse_ratio_ab_over_base'] for x in h_arr]
        mse_ratio_ab_bonly = [values[x]['mse_ratio_ab_over_bonly'] for x in h_arr]

        # MSE
        axes[0, 0].plot(h_arr, base_mse, marker='o', ms=3, ls='--', color=base_color, alpha=0.4, label=f'{channel} Baseline')
        axes[0, 0].plot(h_arr, bonly_mse, marker='^', ms=4, ls='-.', color=base_color, alpha=0.7, label=f'{channel} B-only')
        axes[0, 0].plot(h_arr, ab_mse, marker='s', ms=5, ls='-', color=base_color, lw=2, label=f'{channel} A+B')

        # Predicted u
        axes[0, 1].plot(h_arr, base_u, marker='o', ms=3, ls='--', color=base_color, alpha=0.4, label=f'{channel} Baseline')
        axes[0, 1].plot(h_arr, bonly_u, marker='^', ms=4, ls='-.', color=base_color, alpha=0.7, label=f'{channel} B-only')
        axes[0, 1].plot(h_arr, ab_u, marker='s', ms=5, ls='-', color=base_color, lw=2, label=f'{channel} A+B')

        # Calibration u/MSE
        axes[1, 0].plot(h_arr, base_cal, marker='o', ms=3, ls='--', color=base_color, alpha=0.4, label=f'{channel} Baseline')
        axes[1, 0].plot(h_arr, bonly_cal, marker='^', ms=4, ls='-.', color=base_color, alpha=0.7, label=f'{channel} B-only')
        axes[1, 0].plot(h_arr, ab_cal, marker='s', ms=5, ls='-', color=base_color, lw=2, label=f'{channel} A+B')

        # Ratios
        axes[1, 1].plot(h_arr, u_ratio_ab_base, marker='s', ms=4, color=base_color, ls='-', label=f'{channel} u(A+B) / u(Base)')
        axes[1, 1].plot(h_arr, u_ratio_ab_bonly, marker='^', ms=4, color=base_color, ls='--', label=f'{channel} u(A+B) / u(B-only)')
        axes[1, 1].plot(h_arr, mse_ratio_ab_bonly, marker='o', ms=3, color=base_color, ls=':', alpha=0.7, label=f'{channel} MSE(A+B) / MSE(B-only)')

    axes[0, 0].set_title('Reconstruction MSE across Horizon h')
    axes[0, 0].set_xlabel('Forecast Horizon h')
    axes[0, 0].set_ylabel('Mean MSE')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title('Predicted Uncertainty u across Horizon h')
    axes[0, 1].set_xlabel('Forecast Horizon h')
    axes[0, 1].set_ylabel('Mean Predicted u')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_title('Calibration Ratio u / MSE across Horizon h')
    axes[1, 0].set_xlabel('Forecast Horizon h')
    axes[1, 0].set_ylabel('u / MSE')
    axes[1, 0].axhline(1.0, color='gray', ls='--', alpha=0.5, label='Ideal u/MSE = 1.0')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title('Comparative Ratios across Horizon h')
    axes[1, 1].set_xlabel('Forecast Horizon h')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].axhline(1.0, color='gray', ls='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle('Paired Horizon Evaluation: Baseline vs B-only vs A+B Joint NLL Continuation', fontsize=14)
    plt.savefig(OUTPUT / 'forecast_horizon_comparison.png', dpi=200)
    plt.close()

    # Figure 2: per_frame_mse_vs_u.png
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    frame_dict_all = {}
    for r in archived_frame_rows:
        frame_dict_all[(r['model'], int(r['horizon']), r['channel'], int(r['local_frame']))] = {
            'mean_mse': float(r['mean_mse']), 'mean_predicted_u': float(r['mean_predicted_u'])
        }
    for r in ab_frame_rows:
        frame_dict_all[('a_plus_b', int(r['horizon']), r['channel'], int(r['local_frame']))] = {
            'mean_mse': float(r['mean_mse']), 'mean_predicted_u': float(r['mean_predicted_u'])
        }

    for row_idx, h in enumerate([18, 22, 23]):
        prefix = 24 - h
        frames = list(range(prefix, 24))
        leads = [t - prefix + 1 for t in frames]

        for col_idx, (channel, color) in enumerate([('Density', 'blue'), ('Magnetic', 'red')]):
            ax = axes[row_idx, col_idx]
            base_mse_f = [frame_dict_all[('baseline', h, channel, t)]['mean_mse'] for t in frames]
            bonly_mse_f = [frame_dict_all[('b_only', h, channel, t)]['mean_mse'] for t in frames]
            ab_mse_f = [frame_dict_all[('a_plus_b', h, channel, t)]['mean_mse'] for t in frames]

            base_u_f = [frame_dict_all[('baseline', h, channel, t)]['mean_predicted_u'] for t in frames]
            bonly_u_f = [frame_dict_all[('b_only', h, channel, t)]['mean_predicted_u'] for t in frames]
            ab_u_f = [frame_dict_all[('a_plus_b', h, channel, t)]['mean_predicted_u'] for t in frames]

            ax.plot(leads, base_mse_f, marker='o', ms=3, ls=':', color='gray', alpha=0.6, label='Base MSE')
            ax.plot(leads, bonly_mse_f, marker='^', ms=3, ls='--', color='black', alpha=0.7, label='B-only MSE')
            ax.plot(leads, ab_mse_f, marker='s', ms=4, ls='-', color='black', lw=1.5, label='A+B MSE')

            ax.plot(leads, base_u_f, marker='o', ms=3, ls=':', color=color, alpha=0.5, label='Base u')
            ax.plot(leads, bonly_u_f, marker='^', ms=4, ls='--', color=color, alpha=0.8, label='B-only u')
            ax.plot(leads, ab_u_f, marker='s', ms=5, ls='-', color=color, lw=2, label='A+B u')

            ax.set_title(f'{channel} Per-Frame Lead Trajectory (h={h})')
            ax.set_xlabel('Forecast Lead Step')
            ax.set_ylabel('Magnitude')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

    fig.suptitle('Per-Frame Error vs Predicted Uncertainty Trajectories (h=18, 22, 23)', fontsize=14)
    plt.savefig(OUTPUT / 'per_frame_mse_vs_u.png', dpi=200)
    plt.close()

    # 8. Diagnostic Summary JSON
    models_list = ['baseline', 'b_only', 'a_plus_b']
    response_ratios = {}
    for m in models_list:
        den_h1 = by_model_h_ch[(m, 1, 'Density')]['mean_predicted_u']
        den_h18 = by_model_h_ch[(m, 18, 'Density')]['mean_predicted_u']
        den_h23 = by_model_h_ch[(m, 23, 'Density')]['mean_predicted_u']
        mag_h1 = by_model_h_ch[(m, 1, 'Magnetic')]['mean_predicted_u']
        mag_h18 = by_model_h_ch[(m, 18, 'Magnetic')]['mean_predicted_u']
        mag_h23 = by_model_h_ch[(m, 23, 'Magnetic')]['mean_predicted_u']

        response_ratios[m] = {
            'density_u_h18_over_h1': den_h18 / den_h1 if den_h1 > 0 else float('nan'),
            'magnetic_u_h18_over_h1': mag_h18 / mag_h1 if mag_h1 > 0 else float('nan'),
            'density_u_h23_over_h1': den_h23 / den_h1 if den_h1 > 0 else float('nan'),
            'magnetic_u_h23_over_h1': mag_h23 / mag_h1 if mag_h1 > 0 else float('nan'),
        }

    horizons_np = np.array(HORIZONS, dtype=float)
    spearman_summary = {}
    for m in models_list:
        spearman_summary[m] = {}
        for ch in ('Density', 'Magnetic', 'All'):
            u_vals = np.array([by_model_h_ch[(m, h, ch)]['mean_predicted_u'] for h in HORIZONS])
            mse_vals = np.array([by_model_h_ch[(m, h, ch)]['mean_mse'] for h in HORIZONS])
            spearman_summary[m][ch] = {
                'spearman_h_u': float(stats.spearmanr(horizons_np, u_vals).statistic),
                'spearman_h_mse': float(stats.spearmanr(horizons_np, mse_vals).statistic),
            }

    within_h18_mag = {}
    for m in models_list:
        first = frame_dict_all[(m, 18, 'Magnetic', 6)]
        last = frame_dict_all[(m, 18, 'Magnetic', 23)]
        within_h18_mag[m] = {
            'mse_last_over_first': last['mean_mse'] / first['mean_mse'],
            'u_last_over_first': last['mean_predicted_u'] / first['mean_predicted_u'],
        }

    h22_h23_mag = {}
    for m in models_list:
        paired = [r for r in (archived_paired_rows + ab_paired_rows) if r['model'] == m and r['channel'] == 'Magnetic']
        mse_change = sum(float(r['mse_h23']) for r in paired) / sum(float(r['mse_h22']) for r in paired) - 1.0
        u_change = sum(float(r['u_h23']) for r in paired) / sum(float(r['u_h22']) for r in paired) - 1.0
        h22_h23_mag[m] = {'mse_change': mse_change, 'u_change': u_change}

    spatial_dict = {}
    for r in (archived_spatial_rows + ab_spatial_rows):
        if r.get('scope') == 'ranking_summary':
            spatial_dict[(r['model'], int(r['horizon']), r['channel'])] = float(r['mean_frame_spatial_spearman'])

    spatial_summary = {}
    for m in models_list:
        spatial_summary[m] = {
            'magnetic_h23': spatial_dict.get((m, 23, 'Magnetic')),
            'density_h23': spatial_dict.get((m, 23, 'Density')),
            'by_h23': spatial_dict.get((m, 23, 'By')),
            'bx_h23': spatial_dict.get((m, 23, 'Bx')),
            'bz_h23': spatial_dict.get((m, 23, 'Bz')),
            'magnetic_h18': spatial_dict.get((m, 18, 'Magnetic')),
            'density_h18': spatial_dict.get((m, 18, 'Density')),
        }

    # Standard mask validation aggregate MSE
    val_std_b = [float(r['mean_mse']) for r in standard_rows if r['model'] == 'baseline' and r['role'] == 'val' and r['pass'] == 'pass0' and r['channel'] == 'All']
    val_std_bonly = [float(r['mean_mse']) for r in standard_rows if r['model'] == 'b_only' and r['role'] == 'val' and r['pass'] == 'pass0' and r['channel'] == 'All']
    val_std_ab = [float(r['mean_mse']) for r in standard_rows if r['model'] == 'a_plus_b' and r['role'] == 'val' and r['pass'] == 'pass0' and r['channel'] == 'All']

    std_val_mse = {
        'baseline': sum(val_std_b) / len(val_std_b) if val_std_b else float('nan'),
        'b_only': sum(val_std_bonly) / len(val_std_bonly) if val_std_bonly else float('nan'),
        'a_plus_b': sum(val_std_ab) / len(val_std_ab) if val_std_ab else float('nan'),
    }

    summary = {
        'response_ratios': response_ratios,
        'spearman_h_u': spearman_summary,
        'within_h18_magnetic': within_h18_mag,
        'h22_vs_h23_magnetic': h22_h23_mag,
        'spatial_spearman': spatial_summary,
        'validation_mse': {
            'history_val_mse': {
                'baseline': base_history[-1]['val_mse'],
                'b_only': bonly_history[-1]['val_mse'],
                'a_plus_b': ab_history[-1]['val_mse'],
            },
            'fixed_mask_5family_val_mse': std_val_mse,
        },
    }
    dump_json(OUTPUT / 'diagnostic_summary.json', summary)
    print('Evaluation completed successfully! All CSVs, plots, and summaries generated.', flush=True)


if __name__ == '__main__':
    main()
