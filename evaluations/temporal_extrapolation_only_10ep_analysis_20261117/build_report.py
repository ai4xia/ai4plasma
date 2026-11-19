"""Tables and figures from the reused evaluator's CSV outputs; no inference."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent


def read(name):
    with (OUT / name).open() as handle:
        return list(csv.DictReader(handle))


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    forecast = read('forecast_horizon_metrics.csv')
    standard = read('standard_mask_metrics.csv')
    audit = json.loads((OUT / 'training_audit.json').read_text())
    data = {(r['model'], r['channel'], int(r['horizon'])): r for r in forecast}
    old, new = 'forecast_20pct', 'forecast_100pct'
    channels = ['Density', 'Magnetic']
    horizons = list(range(1, 24))
    def value(model, channel, h, key):
        return float(data[(model, channel, h)][key])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for col, ch in enumerate(channels):
        for row, (key, label) in enumerate([('mean_mse', 'Actual hidden-suffix MSE'),
                                            ('mean_predicted_u', 'Mean predicted variance u')]):
            ax = axes[row, col]
            for model, title, color in [(old, '20% exposure (epoch 276)', '#377eb8'),
                                         (new, '100% exposure (epoch 286)', '#e6550d')]:
                ax.plot(horizons, [value(model, ch, h, key) for h in horizons],
                        'o-', ms=3, label=title, color=color)
            ax.set(title=f'{ch}: {label}', xlabel='Hidden future suffix h',
                   ylabel=label + ' (standardized squared units)', yscale='log', xticks=[1, 6, 12, 18, 23])
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.suptitle('Paired 125 held-out test windows; h=1..23; hidden suffix only')
    fig.savefig(OUT / 'baseline_vs_100pct_horizon.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for col, ch in enumerate(channels):
        ax = axes[0, col]
        for model, title, color in [(old, '20% exposure', '#377eb8'), (new, '100% exposure', '#e6550d')]:
            ax.plot(horizons, [value(model, ch, h, 'predicted_u_over_actual_mse') for h in horizons],
                    'o-', ms=3, label=title, color=color)
        ax.axhline(1, color='black', ls='--', lw=1, label='u = MSE')
        ax.set(title=f'{ch}: absolute calibration', yscale='log', xlabel='Hidden future suffix h',
               ylabel='Mean u / actual MSE', xticks=[1, 6, 12, 18, 23])
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
        ax = axes[1, col]
        for key, title, color in [('u_100pct_over_20pct', 'u(100%) / u(20%)', '#e6550d'),
                                   ('mse_100pct_over_20pct', 'MSE(100%) / MSE(20%)', '#377eb8')]:
            ax.plot(horizons, [value(new, ch, h, key) for h in horizons], 'o-', ms=3, label=title, color=color)
        ax.axhline(1, color='black', ls='--', lw=1)
        ax.set(title=f'{ch}: changes at each horizon', xlabel='Hidden future suffix h',
               ylabel='100% / 20% ratio', xticks=[1, 6, 12, 18, 23])
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    fig.savefig(OUT / 'baseline_vs_100pct_calibration.png', dpi=180)
    plt.close(fig)

    sections = ['# Completed 100% temporal-extrapolation diagnostic',
        '## Checkpoint and training audit',
        'Compared the epoch-276 20%-exposure Student-t model against the epoch-286 '
        '100%-exposure continuation. Both checkpoints loaded successfully; the new model '
        'loaded strictly using the existing evaluator. All model parameters are finite. '
        'Training completed exactly epochs 277–286, with constant LR 1e-5, Student-t NLL '
        '(df=3), and settled Phase C: lambda_v=lambda_r=1, K=1, two forwards, '
        'one detached reuse, and zero weight on pass 2. Phase D remained disabled. '
        f'History records {audit["nonfinite_events"]} nonfinite events, '
        f'{audit["amp_skipped_steps"]:g} AMP-skipped steps, and '
        f'{audit["optimizer_updates"]:g} optimizer updates. Both modalities recorded '
        'temporal_extrapolation share 1.0 throughout all ten epochs.',
        'Final training/validation metrics (NLL is the logged pass-0 Student-t objective; lower is better). '
        'Training MSE is not directly comparable between models because their task distributions differ.',
        table(['Metric', '20% epoch 276', '100% epoch 286'],
              [[key, f'{audit["baseline_final_metrics"][key]:.9g}', f'{audit["final_metrics"][key]:.9g}']
               for key in ['train_mse', 'val_mse', 'train_student_t_nll', 'val_student_t_nll']]),
        '## Paired evaluation protocol',
        'Reused `evaluations/temporal_extrapolation_continuation_20ep_20261117/run_analysis.py` '
        'functions `load_model`, `evaluate_forecast_all`, and `evaluate_standard_all`; no second '
        'inference implementation was introduced. Archived rows labelled `extrapolation_continuation` '
        'were reused as the 20% baseline, without rerunning it. Dataset selection is identical: '
        'stride-1 dataset, deterministic `select_validation_statistics_indices(..., 24, 1)`, '
        'one first complete 24-frame window for each of the 125 test runs, and 125 validation '
        'runs for standard masks. The fixed split hash and normalization agree between checkpoints. '
        'Batch size 2, bfloat16 inference, and unknown-input noise seeds are unchanged. '
        'Forecast noise seed is 201234 + batch_number and reused across horizons. '
        'All four channels are exact at t=0..23-h and hidden at t=24-h..23. '
        'No horizon metadata is supplied.',
        'Forecast MSE and mean raw predicted u are voxel-weighted over hidden suffixes only, '
        'in checkpoint-standardized squared units. Magnetic pools Bx/By/Bz equally; Density '
        'uses its own channel. Calibration is mean(u)/mean(squared error), not a mean of '
        'per-voxel ratios. The 256 sampled residual/u pairs per window/channel are used only '
        'for auxiliary sampled statistics, not the primary MSE or mean-u metrics. '
        'Any u/MSE below one indicates underestimation of actual squared error.',
        '## Forecast difficulty and uncertainty response',
        table(['Channel', 'h', 'MSE20/MSE20(h1)', 'MSE100/MSE100(h1)', 'u20/u20(h1)',
               'u100/u100(h1)', 'u-response change'],
              [[ch, h, f'{value(old,ch,h,"mse_over_h1"):.4f}', f'{value(new,ch,h,"mse_over_h1"):.4f}',
                f'{value(old,ch,h,"u_over_h1"):.4f}', f'{value(new,ch,h,"u_over_h1"):.4f}',
                f'{100*(value(new,ch,h,"u_over_h1")/value(old,ch,h,"u_over_h1")-1):+.2f}%']
               for ch in channels for h in [6,12,18,23]]),
        '## Selected absolute metrics',
        table(['Channel','h','MSE20','MSE100','u20','u100','u20/MSE20','u100/MSE100'],
              [[ch,h]+[f'{value(model,ch,h,key):.7g}' for model,key in
                [(old,'mean_mse'),(new,'mean_mse'),(old,'mean_predicted_u'),(new,'mean_predicted_u'),
                 (old,'predicted_u_over_actual_mse'),(new,'predicted_u_over_actual_mse')]]
               for ch in channels for h in [1,6,12,18,23]]),
        '## Selective response versus global variance inflation',
        'Every horizon is included below. A constant u100/u20 would indicate pure multiplicative '
        'inflation. A horizon-dependent ratio shows selective adaptation, but a normalized response '
        'can also rise because u(h1) falls; absolute changes must be inspected alongside response ratios.',
        table(['h','Density u100/u20','Magnetic u100/u20','Density MSE100/MSE20','Magnetic MSE100/MSE20'],
              [[h]+[f'{value(new,ch,h,key):.5f}' for key in ['u_100pct_over_20pct','mse_100pct_over_20pct']
                      for ch in channels] for h in horizons]),
        '## Ordinary reconstruction regression',
        'The full training-validation metric uses the existing balanced reconstruction objective '
        'and full validation sampling. It is distinct from the paired one-window-per-run hidden-only '
        'metrics below; their absolute values should not be equated.',
        table(['Standard validation mask','MSE20','MSE100','Change'],
              [[task, f'{metrics["mse"]:.8f}',
                f'{audit["final_metrics"]["val_per_pattern"][task]["mse"]:.8f}',
                f'{100*(audit["final_metrics"]["val_per_pattern"][task]["mse"]/metrics["mse"]-1):+.2f}%']
               for task,metrics in audit['baseline_final_metrics']['val_per_pattern'].items()]),
        'Paired hidden-only pass-0 metrics, pooled over all four channels, using exactly the '
        'archived five standard masks. Full per-channel and pass-1 metrics are also retained in '
        '`standard_mask_metrics.csv`.',
    ]
    by_standard = {(r['model'], r['role'], r['pass'], r['source_task'], r['channel']):r for r in standard}
    tasks = list(audit['baseline_final_metrics']['val_per_pattern'])
    rows = []
    for role in ['val', 'test']:
        for task in tasks:
            a,b=[float(by_standard[(m,role,'pass0',task,'All')]['mean_mse']) for m in [old,new]]
            rows.append([role,task,f'{a:.8f}',f'{b:.8f}',f'{100*(b/a-1):+.2f}%'])
    sections.append(table(['Role','Mask','MSE20','MSE100','Change'],rows))
    sections.extend([
        '## Scientific conclusion',
        '1. **100% exposure did not produce a broad increase in horizon-dependent uncertainty amplitude.** '
        'Magnetic u(h18)/u(h1) improves from 1.71x to 2.44x (about +43%), while Density falls '
        'from 2.01x to 1.79x (about -11%). At h23, Density falls from 2.09x to 1.45x and '
        'Magnetic from 0.85x to 0.77x. This is a selective Magnetic response, not a solution '
        'to the multimodal dynamic-range problem. Actual MSE h18/h1 also falls: Density '
        '9.37x to 6.13x, Magnetic 9.82x to 7.64x. The new 2.44x Magnetic u response still '
        'substantially trails its 7.64x MSE growth. At h23 the new MSE ratios grow to '
        '58.65x (Density) and 47.52x (Magnetic), despite the weaker u responses.',
        '2. **Absolute calibration worsens overall, with a Magnetic improvement at moderate horizons.** '
        'At h18, Density u/MSE falls from approximately 0.1267 to 0.0892, while Magnetic '
        'rises from approximately 0.0425 to 0.0652. At h23 both worsen: Density '
        '0.0271 to 0.00753, Magnetic 0.00538 to 0.00331. All remain far below one. '
        'Density u100/u20 is below one at every horizon: 0.514 at h1, 0.458 at h18, '
        '0.355 at h23. Magnetic changes are selective: 0.792 at h1, 1.135 at h18, '
        '0.723 at h23. The Magnetic h18 response increase therefore combines a smaller '
        'h1 denominator (-20.8% u) with an absolute h18 increase (+13.5% u), alongside '
        'a 26.0% MSE reduction; it should not be read as a 43% absolute variance increase.',
        '3. **Magnetic uncertainty still collapses at h23, more severely relative to h1.** '
        'At h22 the new mean u is 0.01242; at h23 it falls to 0.004280 (about -65.5%), '
        'while MSE rises from 0.7097 to 1.2923 (about +82.1%). Its u/MSE at h23 '
        'corresponds to underestimating observed error variance by approximately 302x.',
        '4. **Forecasting improves at moderate horizons but ordinary reconstruction degrades materially.** '
        'At h18, MSE falls by 34.9% for Density and 26.0% for Magnetic. At h23 it instead '
        'rises by 27.7% and 17.4%. Full standard validation MSE rises 18.6%, from '
        '0.01381283 to 0.01638152; validation NLL worsens from -1.28048 to -0.81795. '
        'Spatial-random, spatial-grid, spatial-block, and temporal-random validation MSE '
        'all worsen; temporal-block improves. This is a material general-reconstruction '
        'cost and is not a suitable production replacement for the 20% checkpoint on these criteria.',
        '5. **Insufficient forecast exposure is not supported as the primary remaining explanation.** '
        'Exposure does affect reconstruction and the moderate-horizon Magnetic response, '
        'so task sampling matters in part. However, fivefold forecasting exposure over '
        'ten more epochs does not resolve Density compression or the extreme-horizon '
        'collapse. This bounded diagnostic does not rule out longer adaptation, and '
        'there is no matched additional-10-epoch 20%-exposure control to separate all '
        'effects of training duration from sampling. These are paired point estimates, '
        'not significance tests. The mechanism causing the variance/error mismatch '
        'cannot be identified uniquely by this experiment.',
        'Recommended next action: retain the 20% checkpoint for general reconstruction. '
        'Before changing training, inspect frozen-model uncertainty-head outputs and '
        'conditioning features across the h22-to-h23 boundary, and compare squared '
        'residuals with predicted variance by modality and horizon. No new loss or '
        'training change is established by this audit.',
        '## Artifacts and verification',
        '![Forecast MSE and predicted variance](baseline_vs_100pct_horizon.png)\n\n'
        '![Calibration and same-horizon model ratios](baseline_vs_100pct_calibration.png)',
        '`forecast_horizon_metrics.csv` contains both models, all 23 horizons and six channel '
        'groups, actual MSE, raw mean u, calibration, relative-to-h1 ratios, and explicit '
        'u100/u20 and MSE100/MSE20 at every horizon. `standard_mask_metrics.csv` contains '
        'both roles, both passes, all five standard masks, and all six channel groups. '
        '`training_audit.json` and `evaluation_manifest.json` preserve audit details and '
        'evaluation provenance. Primary metrics are finite and paired voxel/window counts '
        'match the archive. Baseline CSV SHA256 hashes remain unchanged. Inference ran '
        'as a step in the user-owned existing allocation 61481634; no job was submitted. '
        'Training code, checkpoints, and prior evaluation results were not modified.',
        '**100% Forecast Diagnostic Audit**\n\n'
        '- Training completed: YES\n'
        '- Density u(h18)/u(h1): 20% = 2.01x; 100% = 1.79x\n'
        '- Magnetic u(h18)/u(h1): 20% = 1.71x; 100% = 2.44x\n'
        '- Density u(h23)/u(h1): 20% = 2.09x; 100% = 1.45x\n'
        '- Magnetic u(h23)/u(h1): 20% = 0.85x; 100% = 0.77x\n'
        '- Difficulty-response amplitude: WORSE overall; moderate-horizon Magnetic improves\n'
        '- Absolute calibration: WORSE overall; Magnetic h18 improves\n'
        '- Ordinary reconstruction: WORSE\n'
        '- Forecast-exposure hypothesis: NOT SUPPORTED as the primary remaining explanation\n'
        '- Most important remaining bottleneck: predicted variance fails to track actual error, '
        'especially near-total temporal occlusion; mechanism unresolved\n'
        '- Recommended next action: retain the 20% checkpoint and investigate frozen-model '
        'h22-to-h23 uncertainty response before another training change',
    ])
    for ch in channels:
        print(ch)
        for h in [1,6,12,18,22,23]:
            print(h, {key: value(new,ch,h,key) for key in ['mean_mse','mean_predicted_u','u_over_h1','mse_over_h1','u_100pct_over_20pct','mse_100pct_over_20pct','predicted_u_over_actual_mse']})
    print('VALIDATION_CHANGE_PERCENT',100*(audit['final_metrics']['val_mse']/audit['baseline_final_metrics']['val_mse']-1))
    (OUT / 'report.md').write_text('\n\n'.join(sections)+'\n')


if __name__ == '__main__':
    main()
