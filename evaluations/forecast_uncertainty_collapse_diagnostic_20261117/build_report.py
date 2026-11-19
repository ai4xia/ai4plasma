"""Summarize frame diagnostics; no model execution or changes to source artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

OUT=Path(__file__).resolve().parent
ARCHIVE=OUT.parent/'temporal_extrapolation_continuation_20ep_analysis_20261117'
CHANNELS=['Bx','By','Bz','Density','Magnetic']
HORIZONS=[1,6,12,18,21,22,23]


def read(name):
    return list(csv.DictReader((OUT/name).open()))


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,r))+' |' for r in rows])


def aggregate(rows):
    n=sum(int(r['voxel_count']) for r in rows)
    mse=sum(float(r['mean_mse'])*int(r['voxel_count']) for r in rows)/n
    u=sum(float(r['mean_predicted_u'])*int(r['voxel_count']) for r in rows)/n
    return dict(mse=mse,u=u,calibration=u/mse)


def main():
    frames=read('per_frame_metrics.csv')
    passes=read('pass0_vs_pass1_metrics.csv')
    spatial=read('spatial_uncertainty_metrics.csv')
    paired=read('h22_vs_h23_paired.csv')
    manifest=json.loads((OUT/'evaluation_manifest.json').read_text())
    assert hashlib.sha256((ARCHIVE/'forecast_horizon_metrics.csv').read_bytes()).hexdigest()==manifest['archive_sha256']
    data={(int(r['horizon']),r['channel'],int(r['local_frame'])):r for r in frames}
    old={ (int(r['horizon']),r['channel']):r for r in csv.DictReader((ARCHIVE/'forecast_horizon_metrics.csv').open())
          if r['model']=='extrapolation_continuation'}
    verification=[]
    for h in HORIZONS:
        for ch in CHANNELS:
            series=[data[h,ch,t] for t in range(24-h,24)]
            a=aggregate(series)
            for key,col in [('mse','mean_mse'),('u','mean_predicted_u')]:
                rel=abs(a[key]/float(old[h,ch][col])-1)
                assert rel<1e-6,(h,ch,key,rel)
                verification.append(dict(horizon=h,channel=ch,metric=key,relative_difference=rel))
    (OUT/'archive_reproduction_check.json').write_text(json.dumps(verification,indent=2))
    fig,axes=plt.subplots(2,5,figsize=(21,8),layout='constrained')
    for col,ch in enumerate(CHANNELS):
        for h in HORIZONS:
            rows=[data[h,ch,t] for t in range(24-h,24)]
            lead=[int(r['forecast_lead_from_pass0']) for r in rows]
            for row,key in [(0,'mean_mse'),(1,'mean_predicted_u')]:
                axes[row,col].plot(lead,[float(r[key]) for r in rows],'.-',label=f'h={h}',ms=3)
        for row,label in [(0,'Actual MSE'),(1,'Mean predicted u')]:
            ax=axes[row,col]
            ax.set(title=f'{ch}: {label}',xlabel='Lead from last observed frame',ylabel=label,
                   yscale='log',xticks=[1,6,12,18,23])
            ax.grid(alpha=.25)
        axes[0,col].legend(fontsize=7,ncol=2)
    fig.suptitle('Epoch 276, pass 0: 125 paired held-out windows; hidden frames only')
    fig.savefig(OUT/'per_frame_mse_vs_u.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(3,5,figsize=(21,11),layout='constrained')
    for col,ch in enumerate(CHANNELS):
        for h,style in [(22,'o-'),(23,'s-')]:
            rows=[data[h,ch,t] for t in range(24-h,24)]
            for row,key in [(0,'mean_mse'),(1,'mean_predicted_u'),(2,'u_over_mse')]:
                axes[row,col].plot([int(r['local_frame']) for r in rows],[float(r[key]) for r in rows],style,
                                  label=f'h={h}',ms=3)
        for row,label in [(0,'Actual MSE'),(1,'Mean predicted u'),(2,'u/MSE')]:
            ax=axes[row,col]
            ax.set(title=f'{ch}: {label}',xlabel='Local frame (same physical position)',ylabel=label,yscale='log')
            ax.axvspan(.7,1.3,color='gray',alpha=.15)
            ax.grid(alpha=.25)
        axes[0,col].legend(fontsize=8)
    fig.suptitle('h22 vs h23: compare common frames 2–23; shaded frame 1 is newly hidden only in h23')
    fig.savefig(OUT/'h22_vs_h23.png',dpi=180)
    plt.close(fig)

    growth=[]
    for ch in CHANNELS:
        for h in HORIZONS:
            rows=[data[h,ch,t] for t in range(24-h,24)]
            e=np.array([float(r['mean_mse']) for r in rows]);u=np.array([float(r['mean_predicted_u']) for r in rows])
            growth.append([ch,h,f'{e[-1]/e[0]:.3f}',f'{u[-1]/u[0]:.3f}',
                'n/a' if h==1 else f'{spearmanr(np.arange(h),u).statistic:.3f}',
                int((np.diff(u)<0).sum()),f'{min(u/e):.4g}',f'{max(u/e):.4g}'])
    critical=[]
    critical_dict={}
    for ch in CHANNELS:
        a=aggregate([data[22,ch,t] for t in range(2,24)])
        b=aggregate([data[23,ch,t] for t in range(2,24)])
        one=data[23,ch,1]
        full=aggregate([data[23,ch,t] for t in range(1,24)])
        pairs=[r for r in paired if r['channel']==ch]
        critical_dict[ch]=dict(h22_common=a,h23_common=b,h23_frame1=one,h23_full=full,
            common_frames_u_drops=sum(float(r['u_h23_over_h22'])<1 for r in pairs),
            mean_fraction_windows_u_drops=float(np.mean([float(r['u_h23_over_h22_fraction_windows_below_one']) for r in pairs])))
        critical.append([ch,f'{a["mse"]:.6g}',f'{b["mse"]:.6g}',f'{b["mse"]/a["mse"]:.3f}',
            f'{a["u"]:.6g}',f'{b["u"]:.6g}',f'{b["u"]/a["u"]:.3f}',
            f'{a["calibration"]:.5f}',f'{b["calibration"]:.5f}',
            critical_dict[ch]['common_frames_u_drops']])
    spatial_summary=[r for r in spatial if r['scope']=='ranking_summary']
    spatial_table=[]
    for r in spatial_summary:
        h,ch=int(r['horizon']),r['channel']
        exact=[s for s in spatial if int(s['horizon'])==h and s['channel']==ch and s['scope']=='within_frame_decile_exact']
        low,high=min(exact,key=lambda r:int(r['quantile'])),max(exact,key=lambda r:int(r['quantile']))
        spatial_table.append([ch,h]+[f'{float(r[k]):.3f}' for k in
            ['pooled_sampled_spearman','median_window_spearman','mean_frame_spatial_spearman']]+
            [f'{float(high["mean_mse"])/float(low["mean_mse"]):.2f}',f'{float(high["u_over_mse"]):.4f}'])
    pass_table=[]
    pass_results={}
    for h in [18,22,23]:
        for ch in CHANNELS:
            stats={p:aggregate([r for r in passes if r['pass_name']==p and int(r['horizon'])==h and
                               r['channel']==ch and int(r['local_frame'])>=24-h])
                   for p in ['pass0','pass1','matched_binary_control']}
            p,c=stats['pass1'],stats['matched_binary_control']
            pass_results[f'{h}_{ch}']=stats
            pass_table.append([h,ch,f'{stats["pass0"]["calibration"]:.4f}',
                f'{p["mse"]:.6g}',f'{c["mse"]:.6g}',f'{p["mse"]/c["mse"]:.3f}',
                f'{p["u"]:.6g}',f'{c["u"]:.6g}',f'{p["calibration"]:.4f}',f'{c["calibration"]:.4f}'])
    summary=dict(critical=critical_dict,pass_comparison=pass_results,growth_table=growth,
        spatial_summary=spatial_summary,spatial_table=spatial_table,
        max_archived_metric_relative_difference=max(r['relative_difference'] for r in verification))
    (OUT/'diagnostic_summary.json').write_text(json.dumps(summary,indent=2))
    sections=['# Forecast uncertainty collapse diagnostic — epoch 276',
        '## Population, metrics, and reproducibility',
        'Primary model: epoch-276 Student-t df=3, 20% extrapolation continuation. '
        'Reused the archived evaluator for model loading, normalization, deterministic window '
        'selection, batching, inference, fixed masks, and Phase-C reuse. The exact same 125 '
        'held-out test runs each contribute their first complete 24-frame window. Split hash: '
        f'`{manifest["split_sha256"]}`. The manifest lists every run, index, and window start. '
        'All four channels have the same observed prefix and hidden suffix in pass 0. '
        'Unknown Gaussian inputs use the archived seed 201234 + batch number, batch size 2, '
        'and the same noise at every horizon. No epoch-286 or optional epoch-256 model was used.',
        'MSE and mean raw predicted variance are voxel-weighted in checkpoint-standardized '
        'squared units; Magnetic pools Bx/By/Bz. Calibration is mean(u)/MSE. Frame 0 is the '
        'first local frame; forecast lead = local frame − last observed frame. The manifest '
        'records physical window starts. All primary metrics are finite. Re-aggregating the '
        'per-frame pass-0 results reproduces archived horizon MSE/u, with maximum relative '
        f'difference {summary["max_archived_metric_relative_difference"]:.3g}. Archived CSV hash is unchanged.',
        '## Per-frame growth within each fixed horizon',
        'Last/first ratios compare the last hidden frame with the first hidden frame under '
        'the same horizon, not h23 with h1. A single h1 frame has no measurable within-horizon slope.',
        table(['Channel','h','MSE last/first','u last/first','Spearman(lead,u)','u decreases','min u/MSE','max u/MSE'],growth),
        '![Per-frame MSE and uncertainty by forecast lead](per_frame_mse_vs_u.png)',
        '## Critical comparison: common physical frames 2–23',
        'h22 observes frames 0–1; h23 observes only frame 0. These comparisons exclude newly '
        'hidden frame 1 and match every target window and physical frame. The last column '
        'counts common frames whose aggregate u decreases (out of 22). The paired CSV also '
        'reports median window ratios and the fraction of windows with a decrease at every frame.',
        table(['Channel','MSE22','MSE23','MSE23/22','u22','u23','u23/22','C22','C23','frames u drops'],critical),
        'Frame 1, hidden only under h23:',
        table(['Channel','MSE frame1','u frame1','C frame1','full h23 u / common-frame u'],
            [[ch,f'{float(v["h23_frame1"]["mean_mse"]):.6g}',f'{float(v["h23_frame1"]["mean_predicted_u"]):.6g}',
              f'{float(v["h23_frame1"]["u_over_mse"]):.5f}',f'{v["h23_full"]["u"]/v["h23_common"]["u"]:.4f}']
             for ch,v in critical_dict.items()]),
        '![Paired h22 and h23 at matching local frames](h22_vs_h23.png)',
        '## Spatial uncertainty–error correspondence',
        'Within-window Spearman uses all hidden space-time voxels. To isolate spatial ranking '
        'from temporal growth, within-frame Spearman ranks spatial voxels separately in each '
        'run/frame, then averages across frames and runs. Magnetic pools component/voxel '
        'pairs and therefore can include between-component scale effects; Bx/By/Bz are '
        'reported separately. Pooled Spearman uses a deterministic sample of 4096 voxels '
        'per run/channel group (512,000 total), and can include between-run difficulty. '
        'Exact empirical deciles use average ranks so tied u values stay together. '
        'The top/bottom error ratio below aggregates exact within-frame deciles; a ratio '
        'above one demonstrates spatial discrimination, not calibrated amplitude. '
        'All ten deciles for pooled, within-window, and within-frame ranking are in the spatial CSV.',
        table(['Channel','h','pooled rho','median window rho','mean frame spatial rho',
               'top/bottom frame-decile MSE','top-decile u/MSE'],spatial_table),
        '## Phase-C pass 0 versus continuous-u pass 1',
        'The established fixed-validation convention pairs temporal_block with spatial_random. '
        'We use that fresh spatial_random layout for every source horizon, via '
        '`rollout_validation_reuse_pattern` and `make_fixed_validation_mask`. '
        '`make_phase_c_reuse_input` places exact targets/u=0 at the fresh mask and detached '
        'mu0/raw-u0 elsewhere. Previous prefix observations are not automatically retained. '
        'Thus pass 1 is a spatial reconstruction/refinement condition, not a pure forecasting '
        'condition with the original prefix. Its effective observed fractions per run/channel/frame '
        'are recorded in `pass1_effective_masks.csv`. The all-frame per-pass CSV reports these '
        'fractions; its lead column explicitly refers to the source pass-0 prefix.',
        'A matched binary control performs a first forward with this identical fresh spatial '
        'mask, targets/u=0 at observed locations, and Gaussian noise/u=1 elsewhere. Pass 1 '
        'and this control share exact observations and scored voxels, but pass 1 additionally '
        'carries prior prediction information. This measures their combined information/refinement '
        'effect, not a causal effect of changing u alone. The table scores only original '
        'future-frame positions for each source horizon; pass-0 calibration is contextual '
        'and is not an equal-information comparison.',
        table(['source h','Channel','C pass0','MSE pass1','MSE control','MSE1/control',
               'u pass1','u control','C pass1','C control'],pass_table),
    ]
    pointwise=[]
    for h in [18,22,23]:
        for ch in ['Magnetic','Density']:
            for t in [24-h,12,23]:
                selected={p:next(r for r in passes if r['pass_name']==p and int(r['horizon'])==h
                                and r['channel']==ch and int(r['local_frame'])==t)
                          for p in ['pass1','matched_binary_control']}
                a,b=selected['pass1'],selected['matched_binary_control']
                pointwise.append([h,ch,t,f'{float(a["mean_mse"]):.5g}',f'{float(a["mean_predicted_u"]):.5g}',
                    f'{float(a["u_over_mse"]):.4f}',f'{float(b["mean_mse"]):.5g}',f'{float(b["u_over_mse"]):.4f}'])
    sections.extend([
        'Selected per-frame pass-1/control values (all frames and channels are in the pass CSV):',
        table(['source h','Channel','local frame','MSE pass1','u pass1','C pass1','MSE control','C control'],pointwise),
        '## h23 calibration by within-frame predicted-u decile',
        'Each decile is assigned within a single run and physical frame, then aggregated over '
        'all 125 runs and hidden frames. This avoids obtaining discrimination merely by '
        'ranking easier runs or earlier frames below harder ones.',
        table(['Channel','decile','MSE','mean u','u/MSE'],
            [[ch,r['quantile'],f'{float(r["mean_mse"]):.6g}',f'{float(r["mean_predicted_u"]):.6g}',
              f'{float(r["u_over_mse"]):.5f}'] for ch in ['Magnetic','Density']
             for r in spatial if r['channel']==ch and r['horizon']=='23' and r['scope']=='within_frame_decile_exact']),
        '## Answers to the scientific questions',
        '1. **Does u increase with lead within a fixed horizon? Usually no for Magnetic; '
        'Density is horizon dependent.** At h18, Magnetic MSE rises from 0.094805 at lead 1 '
        'to 0.415320 at lead 18 (4.38x), while u falls from 0.021450 to 0.006733 (0.314x). '
        'Spearman(lead,u) is -0.965. Density at h18 has MSE growth 14.86x but u declines '
        'to 0.489x its first-hidden-frame value (lead/u rho -0.913). At h6, Density u '
        'does rise (1.366x), but much less than MSE (5.645x). At h23, Magnetic u is '
        'nonmonotonic (lead/u rho -0.332), with last/first u=0.899 versus MSE=11.278; '
        'Density has last/first u=0.183 versus MSE=1.474. The per-frame problem therefore '
        'includes incorrect temporal shape, not just insufficient aggregate scale.',
        '2. **Both a broad lead-time failure and an abrupt h22-to-h23 transition are present.** '
        'On common frames 2–23, Magnetic u falls 43.1% (0.010359 to 0.005893), while MSE '
        'rises 59.3% (0.720704 to 1.147724). Calibration falls from 0.01437 to 0.00513. '
        'Aggregate u drops at all 22 common frames, and in 85.4% of paired run/frame '
        'means. Density common-frame u falls 23.9% while MSE rises 117.0%; its u drops '
        'at all 22 common frames and in 72.8% of run/frame pairs. By has the largest '
        'relative u suppression (u23/u22=0.524), compared with Bx=0.722 and Bz=0.838. '
        'The disappearance of the second exact observed frame changes both its values '
        'and the binary mask geometry; this experiment does not separate those causes.',
        '**Newly hidden frame 1 is undercalibrated, but does not cause the mean-u collapse.** '
        'Its Magnetic MSE=0.075665, u=0.006618 and C=0.08746. Adding it to common frames '
        'raises h23 mean u by 0.54%, rather than depressing it. For Density, frame 1 has '
        'MSE=0.326656, u=0.033193 and C=0.10161; it raises mean u by 7.93%. '
        'Per-frame checking confirms underestimation throughout h23: Magnetic C spans '
        '0.00333–0.08746, and Density C spans 0.01262–0.10161. At h18 the same statement '
        'would be incorrect for Density: its first hidden frame has C=1.424 before '
        'falling to 0.04692 at the last frame.',
        '3. **Spatial ranking is weak for Magnetic and substantially better for Density.** '
        'At h23, Magnetic pooled rho=0.120 and mean within-run/frame spatial rho=0.147; '
        'the top predicted-u frame-decile has only 2.14x the bottom-decile MSE. By, '
        'which dominates Magnetic error, has within-frame rho=0.005 and top/bottom '
        'MSE=0.77, so the most uncertain locations do not even have larger aggregate '
        'error than the least uncertain locations. At h18 By is also weak/inverted '
        '(within-frame rho=-0.038, top/bottom MSE=0.67). In contrast, Density h23 has '
        'within-frame rho=0.511 and top/bottom MSE=11.90, yet its highest-u decile '
        'still has u/MSE=0.04948. Density therefore shows useful spatial ordering '
        'with severely insufficient amplitude; Magnetic additionally has poor ordering, '
        'especially By. Pooled Magnetic correlations partly mix channel scales and '
        'should not replace the separate component results.',
        '4. **The failure is already present in binary-u pass 0.** All h22/h23 discontinuity, '
        'lead-time slopes, and spatial results above use a single forward with exact '
        'prefix/u=0 and Gaussian suffix/u=1. Continuous-u reuse cannot be the origin '
        'of this binary-input failure.',
        '5. **Pass 1 improves raw errors relative to the much harder forecast input, but is '
        'harmful relative to the matched current-mask binary control.** Its fresh mask '
        'observes exactly 50% of spatial sites in every frame, shared across channels; '
        'this information is much richer than a one-frame prefix. For source h23 '
        'Magnetic, pass1 MSE=0.117732 versus 0.009422 for the same-mask binary control '
        '(12.50x), while u is nearly unchanged (0.007987 versus 0.008565). Calibration '
        'is 0.06784 versus 0.908999. Density pass1 MSE=0.038810 versus 0.003861 '
        '(10.05x), and C=0.27872 versus 0.90128. At source h18, Magnetic and Density '
        'MSE penalties are already 5.35x and 6.11x. Thus errors carried by reused '
        '(mu0,u0) are not adequately represented by the resulting u. This joint '
        'mean/variance reuse comparison does not isolate u alone. Nor is it a direct '
        'test of pass1 under a pure extrapolation mask: that condition was not introduced.',
        '6. **Evidence specifically identifying a limited-global-context uncertainty head is '
        'INCONCLUSIVE.** The paired suppression and declining lead-time curves show a '
        'failure to track temporal difficulty, but do not localize it to a particular '
        'layer. Source inspection shows a pointwise 1x1x1 variance readout operating '
        'on contextual decoder features, after the existing spatiotemporal network. '
        'A pointwise readout does not itself establish absence of global context. '
        'No head-only or contextual-feature intervention was performed, so these '
        'measurements cannot distinguish shared features, variance readout, or learned '
        'objective behavior as the architectural cause.',
        '7. **Smallest justified next experiment: a frozen-checkpoint, matched-mask 2x2 '
        'reuse-input ablation at h18/h22/h23.** Keep exact targets/u=0 at the same '
        'fresh mask. At still-hidden sites independently choose values from mu0 versus '
        'the already paired Gaussian noise, and variance from raw u0 versus one. '
        'Two corners (full reuse and the binary control) were measured here; only two '
        'additional forward conditions are needed. Compare same-voxel MSE, u and '
        'calibration to separate reused-mean contamination from the raw-u conditioning '
        'effect. This is a proposed diagnostic ablation, not a change to training '
        'semantics, and was not executed here. It addresses the measured 10–12.5x '
        'reuse penalty without prescribing architecture, loss, metadata, or exposure '
        'changes. The binary h23 failure would remain a separate issue to investigate.',
        '## Execution and limitations',
        'Inference ran on one GPU within user-owned existing allocation 61481634. '
        'No Slurm job was submitted and no training was run. All new scripts and '
        'artifacts are confined to this evaluation directory. Training/model/masking '
        'source, checkpoints and prior evaluation outputs were not modified. Both '
        'requested figures and all requested CSVs, including per-frame pass1 metrics, '
        'were generated. Exact per-window ranks and run/frame summaries are provided '
        'as auxiliary CSVs. Primary sums use all hidden voxels; only pooled ranking '
        'uses a sample. These are measured point estimates, without a formal '
        'significance test or a unique causal attribution.',
        '**Forecast uncertainty collapse diagnostic**\n\n'
        '- Primary model: epoch-276 Student-t, 20% extrapolation\n'
        '- Per-frame uncertainty growth: NONMONOTONIC; often decreases as MSE grows\n'
        '- h22→h23 discontinuity: PRESENT; common-frame Magnetic u -43.1%, MSE +59.3%\n'
        '- Spatial uncertainty-error ranking: POOR for Magnetic (frame rho 0.147 at h23); '
        'Density is stronger (0.511)\n'
        '- Pass-0 uncertainty behavior: declining/insufficient lead response and abrupt '
        'one-frame-prefix suppression are already present\n'
        '- Pass-1 contribution: HARMFUL relative to matched binary current-mask control '
        '(h23 MSE penalties: Magnetic 12.50x, Density 10.05x)\n'
        '- Main failure mechanism: incorrect temporal uncertainty response, additional '
        'one-observed-frame suppression, and weak Magnetic spatial ranking; reuse adds '
        'a separate error/variance mismatch\n'
        '- Evidence for architectural limitation: INCONCLUSIVE\n'
        '- Recommended next experiment: frozen-checkpoint 2x2 mean/variance reuse-input '
        'ablation; two additional conditions, no training\n'
        '- Training code modified: NO\n'
        '- Training launched: NO',
    ])
    (OUT/'report.md').write_text('\n\n'.join(sections)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
