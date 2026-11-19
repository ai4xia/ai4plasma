"""Read-only frame/ranking diagnostics using the archived evaluator and reuse helper."""
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

ROOT = Path('/pscratch/sd/b/binxia/ai4plasma')
OUT = Path(__file__).resolve().parent
REF = ROOT / 'evaluations/temporal_extrapolation_continuation_20ep_20261117/run_analysis.py'
ARCHIVE = ROOT / 'evaluations/temporal_extrapolation_continuation_20ep_analysis_20261117'
CHECKPOINT = ROOT / 'runs/multi_beta_three_phase_temporal_extrapolation_continuation_20ep_20261116/final.pt'
spec = importlib.util.spec_from_file_location('reference_evaluation', REF)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
HORIZONS = [1, 6, 12, 18, 21, 22, 23]
GROUPS = {k: v for k, v in ref.channel_groups().items() if k != 'All'}


def write(name, rows):
    ref.write_csv(OUT / name, rows)


def summarize(e, u):
    return {'mean_mse': float(e.mean()), 'mean_predicted_u': float(u.mean()),
            'u_over_mse': float(u.mean() / e.mean()), 'voxel_count': int(e.size)}


def quantiles(e, u):
    # Deciles of empirical average rank; equal predicted values stay together.
    bins = np.minimum((10 * (rankdata(u, method='average') - .5) / len(u)).astype(int), 9)
    return [(int(q), float(e[bins == q].sum()), float(u[bins == q].sum()), int((bins == q).sum()))
            for q in np.unique(bins)]


@torch.no_grad()
def main():
    assert not (OUT / 'per_frame_metrics.csv').exists()
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    checkpoint, model, mean, std = ref.load_model(CHECKPOINT, device)
    assert checkpoint['epoch'] == 276
    from data.vpic_hdf5_dataset import validate_checkpoint_split
    validate_checkpoint_split(checkpoint)
    args = checkpoint['args']
    split = ref.load_fixed_split()
    dataset = ref.VPICWindowDataset(h5_dir=args['h5_dir'], betas=args['betas'],
        delta_t=24, stride_t=1, layout='C T X Z', return_metadata=True)
    indices = ref.select_validation_statistics_indices(dataset, set(split['test_runs']), 24, 1)
    assert len(indices) == 125
    # The established validation convention maps temporal_block -> spatial_random.
    reuse_task = ref.rollout_validation_reuse_pattern('temporal_block')
    assert reuse_task == 'spatial_random'
    frame_buckets = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    mask_buckets = defaultdict(lambda: np.zeros(2, dtype=np.float64))
    sample_frames, spatial_windows, mask_rows = [], [], []
    pooled = defaultdict(list)
    quantile_sums = defaultdict(lambda: np.zeros(3, dtype=np.float64))

    for batch_number, (chosen, block, metadata) in enumerate(ref.batches(dataset, indices)):
        target = ((block.to(device) - mean) / (std + 1e-8)).float()
        noise = torch.randn(target.shape, generator=torch.Generator(device=device).manual_seed(
            ref.SEED + 200000 + batch_number), device=device, dtype=target.dtype)
        reuse_mask, _ = ref.make_fixed_validation_mask(reuse_task, target.shape, device=device, dtype=target.dtype)
        matched_input = torch.cat([ref.make_uncertain_input(target, 1-reuse_mask, noise), 1-reuse_mask], 1)
        matched_prediction, matched_u = ref.infer(model, matched_input)
        for b, index in enumerate(chosen):
            file_index, run_name, t0 = dataset.samples[index]
            for ch, ci in GROUPS.items():
                for t in range(24):
                    exact = reuse_mask[b, list(ci), t].float().mean().item()
                    mask_rows.append(dict(dataset_index=index, run=run_name, window_start=int(t0),
                        channel=ch, local_frame=t, physical_frame=int(t0)+t,
                        pass1_mask=reuse_task, exact_observed_fraction=exact))
        for horizon in HORIZONS:
            prefix = 24-horizon
            mask = torch.ones_like(target)
            mask[:, :, prefix:] = 0
            model_input = torch.cat([ref.make_uncertain_input(target, 1-mask, noise), 1-mask], 1)
            mu0, u0 = ref.infer(model, model_input)
            reuse_input, exact = ref.make_phase_c_reuse_input(target, mu0, u0, reuse_mask)
            assert not reuse_input.requires_grad
            mu1, u1 = ref.infer(model, reuse_input)
            for pass_name, prediction, variance, effective in [
                ('pass0', mu0, u0, mask), ('pass1', mu1, u1, exact),
                ('matched_binary_control', matched_prediction, matched_u, reuse_mask)]:
                residual = (prediction-target).cpu().numpy()
                uncertainty = variance.cpu().numpy()
                unknown = (effective < .5).cpu().numpy()
                assert np.isfinite(residual).all() and np.isfinite(uncertainty).all() and (uncertainty > 0).all()
                for ch, ci in GROUPS.items():
                    ci = list(ci)
                    for t in range(24):
                        e = np.square(residual[:, ci, t], dtype=np.float64)
                        u = uncertainty[:, ci, t].astype(np.float64)
                        keep = unknown[:, ci, t]
                        if not keep.any():
                            continue
                        frame_buckets[pass_name, horizon, ch, t] += [e[keep].sum(), u[keep].sum(), keep.sum()]
                        mask_buckets[pass_name, horizon, ch, t] += [(~keep).sum(), keep.size]
                        if pass_name == 'pass0':
                            for b, index in enumerate(chosen):
                                if keep[b].any():
                                    sample_frames.append(dict(dataset_index=index, horizon=horizon, channel=ch,
                                        local_frame=t, forecast_lead=t-prefix+1,
                                        **summarize(e[b][keep[b]], u[b][keep[b]])))
                    if pass_name != 'pass0' or horizon not in [18,22,23]:
                        continue
                    # Exact within-window and within-frame spatial ranks; sampled pooled ranks.
                    for b, index in enumerate(chosen):
                        e_frames = np.square(residual[b, ci, prefix:], dtype=np.float64).transpose(1,0,2,3).reshape(horizon,-1)
                        u_frames = uncertainty[b, ci, prefix:].astype(np.float64).transpose(1,0,2,3).reshape(horizon,-1)
                        e, u = e_frames.ravel(), u_frames.ravel()
                        rho = float(spearmanr(e, u).statistic)
                        er, ur = rankdata(e_frames,axis=1), rankdata(u_frames,axis=1)
                        er -= er.mean(1,keepdims=True); ur -= ur.mean(1,keepdims=True)
                        frame_rho = (er*ur).sum(1)/np.sqrt((er*er).sum(1)*(ur*ur).sum(1))
                        qwindow = quantiles(e,u)
                        qframe = defaultdict(lambda: np.zeros(3))
                        for et,ut in zip(e_frames,u_frames):
                            for q,se,su,n in quantiles(et,ut):
                                qframe[q] += [se,su,n]
                        for q,se,su,n in qwindow:
                            quantile_sums[horizon,ch,'within_window_decile_exact',q] += [se,su,n]
                        for q, values in qframe.items():
                            quantile_sums[horizon,ch,'within_frame_decile_exact',q] += values
                        low,high = qwindow[0],qwindow[-1]
                        spatial_windows.append(dict(dataset_index=index,horizon=horizon,channel=ch,
                            spearman_space_time_exact=rho,
                            mean_frame_spatial_spearman_exact=float(np.mean(frame_rho)),
                            min_frame_spatial_spearman=float(np.min(frame_rho)),
                            positive_frame_fraction=float(np.mean(frame_rho > 0)),
                            high_u_decile_mse_over_low=(high[1]/high[3])/(low[1]/low[3]),
                            **summarize(e,u)))
                        rng = np.random.default_rng(ref.SEED + 300000 + index)
                        take = rng.choice(e.size,min(4096,e.size),replace=False)
                        pooled[horizon,ch].append((e[take],u[take]))
        if batch_number == 0 or (batch_number+1)%10 == 0:
            print(f'{min((batch_number+1)*2,125)}/125 test windows complete',flush=True)

    all_frames = []
    for (pass_name,h,ch,t), (se,su,n) in sorted(frame_buckets.items()):
        visible,total = mask_buckets[pass_name,h,ch,t]
        all_frames.append(dict(pass_name=pass_name,horizon=h,channel=ch,local_frame=t,
            last_pass0_observed_frame=23-h,forecast_lead_from_pass0=t-(23-h),
            effective_mask='prefix_suffix' if pass_name=='pass0' else reuse_task,
            effective_observed_fraction=visible/total,mean_mse=se/n,mean_predicted_u=su/n,
            u_over_mse=su/se,voxel_count=int(n),window_count=125))
    write('per_frame_metrics.csv',[r for r in all_frames if r['pass_name']=='pass0'])
    write('pass0_vs_pass1_metrics.csv',all_frames)
    write('per_sample_frame_metrics.csv',sample_frames)
    write('pass1_effective_masks.csv',mask_rows)
    write('spatial_per_window_metrics.csv',spatial_windows)
    keyed={(r['horizon'],r['channel'],r['local_frame']):r for r in all_frames if r['pass_name']=='pass0'}
    sample_key={(r['horizon'],r['channel'],r['local_frame'],r['dataset_index']):r for r in sample_frames}
    paired=[]
    for ch in GROUPS:
        for t in range(2,24):
            a,b=keyed[22,ch,t],keyed[23,ch,t]
            pair=dict(channel=ch,local_frame=t,lead_h22=t-1,lead_h23=t,
                mse_h22=a['mean_mse'],mse_h23=b['mean_mse'],u_h22=a['mean_predicted_u'],
                u_h23=b['mean_predicted_u'],calibration_h22=a['u_over_mse'],calibration_h23=b['u_over_mse'])
            for field,output in [('mean_mse','mse_h23_over_h22'),('mean_predicted_u','u_h23_over_h22'),('u_over_mse','calibration_h23_over_h22')]:
                ratios=[sample_key[23,ch,t,i][field]/sample_key[22,ch,t,i][field] for i in indices]
                pair[output]=b[field]/a[field]
                pair[output+'_median_window']=float(np.median(ratios))
                pair[output+'_fraction_windows_below_one']=float(np.mean(np.array(ratios)<1))
            paired.append(pair)
    write('h22_vs_h23_paired.csv',paired)
    spatial=[]
    for (h,ch),samples in sorted(pooled.items()):
        e=np.concatenate([x[0] for x in samples]);u=np.concatenate([x[1] for x in samples])
        for q,se,su,n in quantiles(e,u):
            quantile_sums[h,ch,'pooled_sampled_decile',q] += [se,su,n]
        windows=[r for r in spatial_windows if r['horizon']==h and r['channel']==ch]
        spatial.append(dict(horizon=h,channel=ch,scope='ranking_summary',quantile='',
            pooled_sampled_spearman=float(spearmanr(e,u).statistic),pooled_sample_count=len(e),
            mean_window_spearman=float(np.mean([r['spearman_space_time_exact'] for r in windows])),
            median_window_spearman=float(np.median([r['spearman_space_time_exact'] for r in windows])),
            mean_frame_spatial_spearman=float(np.mean([r['mean_frame_spatial_spearman_exact'] for r in windows])),
            median_window_high_over_low_mse=float(np.median([r['high_u_decile_mse_over_low'] for r in windows]))))
    for (h,ch,scope,q),(se,su,n) in sorted(quantile_sums.items()):
        spatial.append(dict(horizon=h,channel=ch,scope=scope,quantile=q+1,
            mean_mse=se/n,mean_predicted_u=su/n,u_over_mse=su/se,voxel_count=int(n)))
    write('spatial_uncertainty_metrics.csv',spatial)
    manifest=dict(primary_checkpoint=str(CHECKPOINT),epoch=276,split_sha256=checkpoint['split_sha256'],
        reference_evaluator=str(REF),test_windows=[dict(index=i,run=dataset.samples[i][1],window_start=int(dataset.samples[i][2])) for i in indices],
        horizons=HORIZONS,forecast_noise_seed='201234 + batch_number',batch_size=2,
        normalization=checkpoint['stats'],inference='bfloat16 autocast; float32 predictions and residuals; float64 sums',
        pass1_mask=reuse_task,pass1_convention='rollout_validation_reuse_pattern(temporal_block); fixed validation layout',
        pass1_warning='Effective observations differ from pass0. matched_binary_control uses the same fresh mask and Gaussian initialization.',
        spatial_sampling='Exact per-window and per-frame ranks/deciles; pooled rank/deciles use 4096 deterministic voxels per window/group',
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),
        archive_sha256=hashlib.sha256((ARCHIVE/'forecast_horizon_metrics.csv').read_bytes()).hexdigest())
    ref.dump_json(OUT/'evaluation_manifest.json',manifest)
    dataset.close()
    print('Diagnostic inference complete.',flush=True)


if __name__=='__main__':
    main()
