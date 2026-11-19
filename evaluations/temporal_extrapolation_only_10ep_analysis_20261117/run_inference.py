"""Reuse the archived 20%-forecast evaluator; infer only the epoch-286 model."""
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import torch

PROJECT = Path('/pscratch/sd/b/binxia/ai4plasma')
OUTPUT = Path(__file__).resolve().parent
ARCHIVE = PROJECT / 'evaluations/temporal_extrapolation_continuation_20ep_analysis_20261117'
REFERENCE = PROJECT / 'evaluations/temporal_extrapolation_continuation_20ep_20261117/run_analysis.py'
BASE = PROJECT / 'runs/multi_beta_three_phase_temporal_extrapolation_continuation_20ep_20261116'
NEW = PROJECT / 'runs/multi_beta_three_phase_temporal_extrapolation_only_continuation_10ep_20261117'

spec = importlib.util.spec_from_file_location('archived_temporal_evaluator', REFERENCE)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
ref.OUTPUT = OUTPUT


def archived_rows(name):
    with (ARCHIVE / name).open() as handle:
        rows = [dict(row, model='forecast_20pct') for row in csv.DictReader(handle)
                if row['model'] == 'extrapolation_continuation']
    assert rows
    return rows


def main():
    assert not (OUTPUT / 'forecast_horizon_metrics.csv').exists()
    assert torch.cuda.is_available(), 'An allocated CUDA device is required'
    device = torch.device('cuda')
    context = ref.load_model(NEW / 'final.pt', device)
    checkpoint = context[0]
    base = torch.load(BASE / 'final.pt', map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 286 and base['epoch'] == 276
    assert checkpoint['stats'] == base['stats']
    assert checkpoint['split_sha256'] == base['split_sha256']
    from data.vpic_hdf5_dataset import validate_checkpoint_split
    validate_checkpoint_split(checkpoint)
    validate_checkpoint_split(base)
    history = json.loads((NEW / 'history.json').read_text())
    assert [row['epoch'] for row in history] == list(range(277, 287))
    args = checkpoint['args']
    assert args['uncertainty_loss'] == 'student_t_nll' and args['student_t_df'] == 3
    assert args['temporal_extrapolation_only_continuation']
    assert Path(args['post_final_continuation']).resolve() == (BASE / 'final.pt').resolve()
    assert checkpoint['curriculum_state']['phase_d_start_epoch'] is None
    for row in history:
        assert row['learning_rate'] == 1e-5
        assert row['amp_skipped_steps'] == row['nonfinite_events'] == 0
        for modality in ('magnetic', 'density'):
            shares = row[f'train_{modality}_pattern_share']
            assert shares['temporal_extrapolation'] == 1 and sum(shares.values()) == 1
        state = row['curriculum']
        assert state['lambda_v_start'] == state['lambda_v_end'] == 1
        assert state['lambda_r_start'] == state['lambda_r_end'] == 1
        assert state['two_forward_fraction'] == 1 and state['three_forward_fraction'] == 0
        assert state['L_mu_2'] == state['L_var_2'] == state['reuse_weight_2'] == 0
        for key in ('train_mse', 'val_mse', 'train_student_t_nll', 'val_student_t_nll'):
            assert math.isfinite(row[key])
    assert all(torch.isfinite(value).all().item() for value in checkpoint['model'].values())
    audit = {
        'epoch': checkpoint['epoch'], 'epochs': [row['epoch'] for row in history],
        'final_metrics': {key: history[-1][key] for key in (
            'train_mse', 'val_mse', 'train_student_t_nll', 'val_student_t_nll',
            'val_per_pattern', 'curriculum', 'gradient_norm_mean', 'gradient_norm_max')},
        'baseline_final_metrics': {key: json.loads((BASE / 'history.json').read_text())[-1][key]
                                   for key in ('train_mse', 'val_mse', 'train_student_t_nll', 'val_student_t_nll', 'val_per_pattern')},
        'amp_skipped_steps': sum(row['amp_skipped_steps'] for row in history),
        'nonfinite_events': sum(row['nonfinite_events'] for row in history),
        'optimizer_updates': sum(row['optimizer_updates'] for row in history),
        'optimizer_lrs': [group['lr'] for group in checkpoint['optimizer']['param_groups']],
        'model_parameters_finite': True,
    }
    ref.dump_json(OUTPUT / 'training_audit.json', audit)
    split = ref.load_fixed_split()
    dataset = ref.VPICWindowDataset(
        h5_dir=args['h5_dir'], betas=args['betas'], delta_t=24, stride_t=1,
        layout='C T X Z', return_metadata=True)
    roles = {role: ref.select_validation_statistics_indices(
        dataset, set(split[f'{role}_runs']), 24, 1) for role in ('val', 'test')}
    assert {role: len(indices) for role, indices in roles.items()} == {'val': 125, 'test': 125}
    manifest = {
        'baseline_checkpoint': str(BASE / 'final.pt'), 'new_checkpoint': str(NEW / 'final.pt'),
        'split_sha256': checkpoint['split_sha256'], 'normalization_identical': True,
        'reference_evaluator': str(REFERENCE), 'baseline_metrics_reused': str(ARCHIVE),
        'paired_window_indices': roles, 'paired_windows': {'val': 125, 'test': 125},
        'selection': 'select_validation_statistics_indices(dataset, split role runs, 24, 1); same as archived evaluator',
        'forecast_noise_seed': ref.SEED + 200000,
        'standard_noise_seed': '1234 + 100000*(role==test) + batch_number',
        'batch_size': 2, 'inference_dtype': 'bfloat16 autocast; float32 outputs',
        'forecast_geometry': 'all four channels visible prefix 24-h; hidden suffix h; h=1..23',
        'metric_region': 'hidden suffix only; voxel-weighted MSE and mean raw predicted variance',
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'gpu': torch.cuda.get_device_name(),
        'archived_csv_sha256': {name: hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest()
                                for name in ('forecast_horizon_metrics.csv', 'standard_mask_metrics.csv')},
    }
    ref.dump_json(OUTPUT / 'evaluation_manifest.json', manifest)
    models = {'forecast_100pct': context}
    standard, _ = ref.evaluate_standard_all(models, dataset, roles, device)
    ref.write_csv(OUTPUT / 'standard_mask_metrics.csv', archived_rows('standard_mask_metrics.csv') + standard)
    forecast, _ = ref.evaluate_forecast_all(models, dataset, roles['test'], device)
    combined = archived_rows('forecast_horizon_metrics.csv') + forecast
    keyed = {(row['model'], int(row['horizon']), row['channel']): row for row in combined}
    for row in combined:
        h, channel = int(row['horizon']), row['channel']
        old, new = [keyed[(model, h, channel)] for model in ('forecast_20pct', 'forecast_100pct')]
        first = keyed[(row['model'], 1, channel)]
        row['u_100pct_over_20pct'] = float(new['mean_predicted_u']) / float(old['mean_predicted_u'])
        row['mse_100pct_over_20pct'] = float(new['mean_mse']) / float(old['mean_mse'])
        row['u_over_h1'] = float(row['mean_predicted_u']) / float(first['mean_predicted_u'])
        row['mse_over_h1'] = float(row['mean_mse']) / float(first['mean_mse'])
    ref.write_csv(OUTPUT / 'forecast_horizon_metrics.csv', combined)
    dataset.close()
    print('Completed paired inference; archived baseline reused.', flush=True)


if __name__ == '__main__':
    main()
