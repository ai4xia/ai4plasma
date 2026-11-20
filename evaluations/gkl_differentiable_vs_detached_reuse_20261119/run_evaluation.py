"""Compare matched GKL A+B checkpoints on held-out pass-0 and rollout tasks."""
import csv
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CACHE = OUT.parent / "gkl_path_a_on_vs_off_continuation_20261119"
spec = importlib.util.spec_from_file_location("previous_evaluation", CACHE / "run_evaluation.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

from data.vpic_hdf5_dataset import parse_run_name
from train_masked_unet3d import make_phase_c_reuse_input
from visualize_sliding_density_reconstruction import sliding_window_starts

RUNS = {
    key: ROOT / "runs" / f"multi_beta_gkl_joint_grad{suffix}_continuation_20ep_20261119"
    for key, suffix in (("detached", ""), ("differentiable", "_differentiable_reuse"))
}
FOCUS = (18, 22, 23)
MODALITIES = {key: ref.GROUPS[key] for key in ("Magnetic", "Density")}
PASSED = (0, 1, 2, 4, 8)
REPEAT_MASKS = ("spatial_grid", "temporal_block")
WINDOW, STRIDE, LENGTH = 24, 4, 48


def audit():
    source = torch.load(ref.SRC_CKPT, map_location="cpu", weights_only=False)
    checkpoints = {name: torch.load(path / "final.pt", map_location="cpu", weights_only=False)
                   for name, path in RUNS.items()}
    a, b = checkpoints["detached"], checkpoints["differentiable"]
    differences = {key for key in a["args"].keys() | b["args"].keys()
                   if a["args"].get(key) != b["args"].get(key)}
    assert differences <= {"out_dir", "wandb_name", "wandb_tags", "differentiable_phase_c_reuse"}, differences
    assert a["stats"] == b["stats"] == source["stats"] and source["epoch"] == 256
    audit_rows = []
    for name, ckpt in checkpoints.items():
        args, state = ckpt["args"], ckpt["curriculum_state"]
        assert ckpt["epoch"] == args["final_epoch"] == args["stop_after_epochs"] == 276
        assert (ROOT / args["post_final_continuation"]).resolve() == ref.SRC_CKPT
        assert args["gkl_joint_grad_continuation_20ep"] and args["uncertainty_loss"] == "generalized_kl"
        assert args.get("differentiable_phase_c_reuse", False) == (name == "differentiable")
        assert args["curriculum_config"] == "three_phase_global_cosine"
        assert state["lambda_v"] == state["lambda_r"] == 1 and state["phase_d_start_epoch"] is None
        assert all(group["lr"] == 1e-5 for group in ckpt["optimizer"]["param_groups"])
        assert ckpt["split_sha256"] == source["split_sha256"]
        ref.validate_checkpoint_split(ckpt)
        assert all(torch.isfinite(v).all() for v in ckpt["model"].values())
        assert all(torch.isfinite(v).all() for p in ckpt["optimizer"]["state"].values()
                   for v in p.values() if torch.is_tensor(v))
        rows = json.loads((RUNS[name] / "history.json").read_text())
        assert [r["epoch"] for r in rows] == list(range(257, 277))
        for row in rows:
            c = row["curriculum"]
            assert row["learning_rate"] == 1e-5 and row["nonfinite_events"] == row["amp_skipped_steps"] == 0
            assert c["lambda_v_mean"] == c["lambda_r_mean"] == 1 and c["two_forward_fraction"] == 1
            assert c["successful_optimizer_steps"] == row["epoch"] * state["updates_per_epoch"]
            assert all(np.isfinite(row[k]) for k in ("train_mse", "val_mse", "gradient_norm_mean"))
            assert all(np.isfinite(c[k]) for k in ("L_gkl_0", "L_gkl_1", "gkl_alpha"))
            audit_rows.append(dict(model=name, epoch=row["epoch"], train_mse=row["train_mse"],
                                   val_mse=row["val_mse"], grad_norm=row["gradient_norm_mean"],
                                   gkl_alpha=c["gkl_alpha"], gkl_pass0=c["L_gkl_0"],
                                   gkl_pass1=c["L_gkl_1"], nonfinite_events=row["nonfinite_events"],
                                   amp_skipped_steps=row["amp_skipped_steps"],
                                   successful_optimizer_steps=c["successful_optimizer_steps"]))
    histories = [[r for r in audit_rows if r["model"] == m] for m in RUNS]
    assert [r["gkl_alpha"] for r in histories[0]] == [r["gkl_alpha"] for r in histories[1]]
    assert 0 < histories[0][0]["gkl_alpha"] < histories[0][1]["gkl_alpha"] < 1
    assert all(r["gkl_alpha"] == 1 for r in histories[0][2:])
    ref.write_csv(OUT / "training_history.csv", audit_rows)
    ref.dump_json(OUT / "training_audit.json", dict(
        checkpoints={k: str(v / "final.pt") for k, v in RUNS.items()},
        source=ref.SRC_CKPT, configuration_differences=sorted(differences),
        gradient_routing=dict(path_a=True, path_b=True, cross_forward=["detached", "cosine alpha"]),
        split_sha256=a["split_sha256"], final=[rows[-1] for rows in histories],
        finite_model_and_optimizer=True, fixed_lr=1e-5))
    return a["args"]


def read_cache(file):
    with (CACHE / file).open() as f:
        return [dict(r, model="detached") for r in csv.DictReader(f)
                if r["model"] == "gkl_ab_epoch276" and r.get("pass", "pass0") == "pass0"]


def metrics(pred, u, target, observed, groups=ref.GROUPS):
    squared = (pred - target).square()
    hidden = ~observed.bool()
    rows = {}
    for ch, ids in groups.items():
        e, v, keep = squared[:, list(ids)], u[:, list(ids)], hidden[:, list(ids)]
        dims = (1, 2, 3, 4)
        n = keep.sum(dims)
        q = (e * keep).sum(dims, dtype=torch.float64)
        s = (v * keep).sum(dims, dtype=torch.float64)
        rows[ch] = torch.stack((q, s, n)).T.cpu().numpy()
    return rows


def spatial(e, u, hidden):
    # Time-major, keeping the archived Magnetic pooling over all three components.
    rs = []
    for ee, uu, keep in zip(*(x.transpose(1, 0, 2, 3) for x in (e, u, hidden))):
        x = rankdata(ee[keep])
        y = rankdata(uu[keep])
        if len(x) < 2:
            continue
        x -= x.mean()
        y -= y.mean()
        denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
        rs.append(np.dot(x, y) / denom if denom else 0.)
    return float(np.mean(rs)) if rs else float("nan")


def paired_details(outputs, target, mask, seed, groups=MODALITIES, model_names=tuple(RUNS)):
    """Exact MSE/u/ranks and pooled predicted-u deciles on paired sampled voxels."""
    errs = [(p - target).square().cpu().numpy() for p, _ in outputs]
    us = [u.cpu().numpy() for _, u in outputs]
    hidden = (~mask.bool()).cpu().numpy()
    result = {}
    for ch, ids in groups.items():
        ids = list(ids)
        keep = hidden[ids]
        e = [a[ids] for a in errs]
        u = [a[ids] for a in us]
        samples = np.random.default_rng(seed).choice(int(keep.sum()), min(4096, int(keep.sum())), replace=False)
        qe = [v[keep][samples] for v in e]
        qu = [v[keep][samples] for v in u]
        edges = np.quantile(np.concatenate(qu), np.linspace(0, 1, 11)[1:-1])
        for j, name in enumerate(model_names):
            bins = np.searchsorted(edges, qu[j], side="right")
            gap = np.bincount(bins, weights=qu[j]-qe[j], minlength=10)
            mse = float(e[j][keep].mean(dtype=np.float64))
            mean_u = float(u[j][keep].mean(dtype=np.float64))
            result[ch, name] = dict(mse=mse, mean_u=mean_u, calibration=mean_u/mse,
                                    ace=float(np.abs(gap).sum()/len(samples)),
                                    spatial_spearman=spatial(e[j], u[j], keep))
    return result


def rows_from_details(details, **keys):
    return [dict(keys, model=m, channel=ch, **vals) for (ch, m), vals in details.items()]


def summaries(rows, keys, metric_keys):
    groups = defaultdict(dict)
    for r in rows:
        groups[tuple(r[k] for k in keys)][r["run"], r["model"]] = r
    out = []
    for key, records in groups.items():
        names = sorted({run for run, _ in records})
        for metric in metric_keys:
            values = [np.array([records[run, m][metric] for run in names]) for m in RUNS]
            assert all(np.isfinite(x).all() for x in values)
            delta, low, high = ref.paired_bootstrap_ci(values[1], values[0])
            out.append(dict(zip(keys, key), metric=metric, runs=len(names),
                            detached=float(values[0].mean()), differentiable=float(values[1].mean()),
                            difference=delta, ci95_low=low, ci95_high=high))
    return out


@torch.inference_mode()
def pass_zero(dataset, indices, cohort, models, mean, std, device):
    # Additional models are evaluated only on conditions used for paired intervals.
    model_names = tuple(models)
    primary = model_names[-1]
    standard = defaultdict(lambda: np.zeros(3))
    forecast = defaultdict(lambda: np.zeros(3))
    shared = defaultdict(lambda: np.zeros(3))
    ranks = defaultdict(list)
    full_runs = {}
    pairs = []
    repeat = []
    for role, selection in indices.items():
        for batch_no, (chosen, block, meta) in enumerate(ref.batches(dataset, selection)):
            target = (block.to(device) - mean) / (std + 1e-8)
            noise = torch.randn(target.shape, device=device, generator=torch.Generator(device=device)
                                .manual_seed(ref.SEED + 100000*(role == "test") + batch_no))
            selected = [i for i, idx in enumerate(chosen) if role == "test" and idx in cohort]
            for task in ref.MASK_PATTERNS:
                mask, _ = ref.make_fixed_validation_mask(task, target.shape, device=device, dtype=target.dtype)
                model_in = torch.cat([ref.make_uncertain_input(target, 1-mask, noise), 1-mask], 1)
                diff = ref.infer(models[primary], model_in)
                for ch, row in metrics(*diff, target, mask).items():
                    standard[role, task, ch] += row.sum(axis=0)
                if not selected:
                    continue
                # Keep archived batch size and index order for both networks.
                outputs = [ref.infer(models[name], model_in) for name in model_names[:-1]] + [diff]
                for i in selected:
                    name = meta[i]["run_name"]
                    initial = [tuple(v[i] for v in output) for output in outputs]
                    detail = paired_details(initial, target[i], mask[i], ref.SEED + chosen[i],
                                            model_names=model_names)
                    pairs += rows_from_details(detail, run=name, task=task)
                    if task in REPEAT_MASKS:
                        repeat += rows_from_details(detail, run=name, task=task, step=0)
                        state = [(p.unsqueeze(0), u.unsqueeze(0)) for p, u in initial]
                        for step in range(1, 9):
                            state = [ref.infer(model, make_phase_c_reuse_input(
                                target[i:i+1], *prev, mask[i:i+1])[0])
                                     for model, prev in zip(models.values(), state)]
                            if step in PASSED:
                                detail = paired_details([tuple(v[0] for v in o) for o in state],
                                                        target[i], mask[i], ref.SEED + chosen[i],
                                                        model_names=model_names)
                                repeat += rows_from_details(detail, run=name, task=task, step=step)
            if role != "test":
                continue
            noise = torch.randn(target.shape, device=device, generator=torch.Generator(device=device)
                                .manual_seed(ref.SEED + 200000 + batch_no))
            for h in range(1, 24):
                prefix = 24-h
                mask = torch.ones_like(target)
                mask[:, :, prefix:] = 0
                model_in = torch.cat([ref.make_uncertain_input(target, 1-mask, noise), 1-mask], 1)
                diff = ref.infer(models[primary], model_in)
                for ch, row in metrics(*diff, target, mask).items():
                    forecast[h, ch] += row.sum(axis=0)
                    if h in (22, 23):
                        for i, idx in enumerate(chosen):
                            full_runs[h, ch, idx] = (row[i, 0]/row[i, 2], row[i, 1]/row[i, 2])
                if h in (22, 23):
                    for ch, row in metrics(*(v[:, :, 2:] for v in diff), target[:, :, 2:], mask[:, :, 2:]).items():
                        shared[h, ch] += row.sum(axis=0)
                if h in FOCUS:
                    e = (diff[0] - target).square()[:, :, prefix:].cpu().numpy()
                    u = diff[1][:, :, prefix:].cpu().numpy()
                    for ch, ids in MODALITIES.items():
                        ee, uu = e[:, list(ids)], u[:, list(ids)]
                        for i in range(len(chosen)):
                            ranks[h, ch].append(spatial(ee[i], uu[i], np.ones_like(ee[i], dtype=bool)))
                    if selected:
                        outputs = [ref.infer(models[name], model_in) for name in model_names[:-1]] + [diff]
                        for i in selected:
                            detail = paired_details([tuple(v[i] for v in output) for output in outputs],
                                target[i], mask[i], ref.SEED+chosen[i], model_names=model_names)
                            pairs += rows_from_details(detail, run=meta[i]["run_name"], task=f"h{h}")
            if batch_no == 0 or (batch_no + 1) % 15 == 0:
                print(f"Pass-0 test sweep: {min((batch_no+1)*2, len(selection))}/{len(selection)}", flush=True)
    def row(triple):
        q,u,n = triple
        return dict(model=primary, voxel_count=int(n), mean_mse=q/n,
                    mean_predicted_u=u/n, predicted_u_over_actual_mse=u/q)
    standard_rows = [dict(row(v), role=r, source_task=t, channel=ch, **{"pass":"pass0"})
                     for (r,t,ch),v in standard.items()]
    forecast_rows = [dict(row(v), horizon=h, channel=ch, visible_prefix=24-h)
                     for (h,ch),v in forecast.items()]
    rank_rows = [dict(model=primary, horizon=h, channel=ch, scope="ranking_summary",
                      mean_frame_spatial_spearman=np.mean(v)) for (h,ch),v in ranks.items()]
    collapse = {ch: {primary: dict(
        shared_u23_over_u22=shared[23,ch][1]/shared[22,ch][1],
        shared_mse23_over_mse22=shared[23,ch][0]/shared[22,ch][0],
        collapse_fraction=float(np.mean([
            full_runs[23,ch,i][0]>full_runs[22,ch,i][0] and
            full_runs[23,ch,i][1]<full_runs[22,ch,i][1] for i in indices["test"]])))
        } for ch in (*MODALITIES, "All")}
    if len(models) == 1:
        return pairs, repeat, standard_rows, forecast_rows, rank_rows, collapse
    old_standard = read_cache("standard_mask_metrics.csv")
    old_forecast = read_cache("forecast_horizon_metrics.csv")
    assert len(old_standard) == 60 and len(old_forecast) == 138
    ref.write_csv(OUT/"standard_mask_metrics.csv", old_standard + standard_rows)
    ref.write_csv(OUT/"forecast_horizon_metrics.csv", old_forecast + forecast_rows)
    ref.write_csv(OUT/"spatial_uncertainty_metrics.csv", [
        r for r in read_cache("spatial_uncertainty_metrics.csv") if r["scope"]=="ranking_summary"
    ] + rank_rows)
    with (CACHE/"paired_comparison_summary.csv").open() as f:
        old_collapse = {r["channel"]: float(r["GKL A+B ep276"]) for r in csv.DictReader(f)
                        if r["domain"] == "extreme_horizon_collapse"}
    old_shared = read_cache("h22_vs_h23_paired.csv")
    for ch in (*MODALITIES, "All"):
        old = [r for r in old_shared if r["channel"]==ch]
        assert len(old)==22
        collapse[ch]["detached"] = dict(shared_u23_over_u22=sum(float(r["u_h23"]) for r in old)/sum(float(r["u_h22"]) for r in old),
                          shared_mse23_over_mse22=sum(float(r["mse_h23"]) for r in old)/sum(float(r["mse_h22"]) for r in old),
                          collapse_fraction=old_collapse[ch])
    ref.dump_json(OUT/"extreme_horizon.json",collapse)
    return pairs, repeat


@torch.inference_mode()
def slide(dataset, cohort, models, mean, std, device):
    rows = []
    for idx in cohort:
        _, name, offset = dataset.samples[idx]
        run = dataset.get_run(name)[:, offset:offset+LENGTH].to(device).unsqueeze(0)
        assert run.shape[2] == LENGTH
        run = (run-mean)/(std+1e-8)
        gen = torch.Generator(device=device).manual_seed(ref.SEED + 500000 + idx)
        noise = torch.randn(run.shape, device=device, generator=gen)
        sites = torch.randperm(run.shape[-2]*run.shape[-1], device=device, generator=gen)[:32]
        probe = torch.zeros_like(run, dtype=torch.bool).flatten(-2)
        probe[:, :, :33:8, sites] = True
        probe = probe.reshape_as(run)
        state = None
        for step, start in enumerate(sliding_window_starts(LENGTH, WINDOW, STRIDE)):
            end = start + WINDOW
            target = run[:, :, start:end]
            observed = probe[:, :, start:end]
            preds = []
            for j, model in enumerate(models.values()):
                val = noise[:, :, start:end].clone()
                u_in = torch.ones_like(val)
                if state is not None:
                    prior_mu, prior_u = state[j]
                    val[:, :, :-STRIDE] = prior_mu[:, :, STRIDE:]
                    u_in[:, :, :-STRIDE] = prior_u[:, :, STRIDE:]
                # Only exact probes read truth. New hidden slots remain Gaussian/u=1.
                model_in = torch.cat([torch.where(observed, target, val),
                                      torch.where(observed, 0., u_in)], 1)
                preds.append(ref.infer(model, model_in))
            state = preds
            # Score only the newest stride at each update, including the initial window.
            tail = [tuple(x[0, :, -STRIDE:] for x in pair) for pair in state]
            truth, mask = target[0, :, -STRIDE:], observed[0, :, -STRIDE:]
            rows += rows_from_details(paired_details(tail, truth, mask, ref.SEED+idx+step,
                                                     model_names=tuple(models)),
                                      run=name, axis="step", index=step)
            for t in range(STRIDE):
                frame = end-STRIDE+t
                age = frame-min(frame//8*8, 32)
                frame_results = paired_details([tuple(x[:, t:t+1] for x in pair) for pair in tail],
                    truth[:, t:t+1], mask[:, t:t+1], ref.SEED+idx+frame, model_names=tuple(models))
                rows += rows_from_details(frame_results, run=name, axis="age", index=age, frame=frame)
    return rows


def averages(rows, keys, metrics):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    return [dict(zip(keys,k), **{m:float(np.mean([r[m] for r in v])) for m in metrics}) for k,v in groups.items()]


def change_from_initial(rows, step, condition, fields):
    initial = {(r["run"], r["model"], r["channel"], r[condition]): r
               for r in rows if r[step] == 0}
    return [
        dict(r, **{m: r[m] - initial[r["run"], r["model"], r["channel"], r[condition]][m]
                   for m in fields})
        for r in rows if r[step] != 0
    ]


def main():
    assert torch.cuda.is_available() and os.environ.get("SLURM_JOB_ID")
    args = audit()
    print("Training audit passed", flush=True)
    device = torch.device("cuda")
    contexts = {name: ref.load_model(path/"final.pt", device) for name,path in RUNS.items()}
    models = {k: v[1] for k,v in contexts.items()}
    _, _, mean, std = contexts["detached"]
    split = ref.load_fixed_split()
    ds = ref.VPICWindowDataset(args["h5_dir"],betas=args["betas"],delta_t=24,stride_t=1,
                               layout="C T X Z",return_metadata=True)
    indices = {r:ref.select_validation_statistics_indices(ds,set(split[r+"_runs"]),24,1) for r in ("val","test")}
    assert all(len(x)==125 for x in indices.values())
    end_by_run = defaultdict(int)
    for _, name, start in ds.samples:
        end_by_run[name] = max(end_by_run[name],start+24)
    by_beta = defaultdict(list)
    for idx in indices["test"]:
        _,name,offset = ds.samples[idx]
        if end_by_run[name]-offset>=LENGTH:
            by_beta[parse_run_name(name)["beta"]].append(idx)
    rng = np.random.default_rng(20261119)
    for v in by_beta.values():
        rng.shuffle(v)
    cohort = []
    while len(cohort)<min(24,sum(map(len,by_beta.values()))+len(cohort)):
        for k in sorted(by_beta):
            if by_beta[k] and len(cohort)<24:
                cohort.append(by_beta[k].pop())
    assert len(cohort)>=16, len(cohort)
    ref.dump_json(OUT/"protocol.json",dict(
        checkpoint_audit="training_audit.json", historical_cache=str(CACHE),
        slurm_job_id=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),
        pass0="125 val/test windows each, archived first-window selector and noise seeds, hidden voxels only",
        paired_cohort=[dict(index=i,run=ds.samples[i][1],start=ds.samples[i][2]) for i in cohort],
        selection="Seed 20261119, shuffle within beta and draw round-robin from test runs with >=48 frames",
        repetition=dict(masks=REPEAT_MASKS, passes=PASSED, exact_mask="same layout each pass",
                        conditioning="raw previous mean/variance at hidden sites; no reset"),
        sliding=dict(window=WINDOW,stride=STRIDE,trajectory_frames=LENGTH,probe_sites=32,
                     probe_frames=[0,8,16,24,32],channels="all four; same sites",
                     scoring="newest four frames only; hide probes from metrics at age 0",
                     age="frames since last true probe timestamp anywhere, not since a voxel was observed",
                     initialization="Gaussian/u=1 for new hidden sites; overlap carries raw mu/u"),
        ace="Count-weighted absolute mean-u minus mean-error gap on shared deciles from pooled predictions of both models within each paired run/condition; <=4096 paired sampled hidden voxels per group",
        intervals="Percentile paired bootstrap, 10000 draws over independent run IDs, seed 42"))
    pass0, repeat = pass_zero(ds, indices, set(cohort), models, mean, std, device)
    fields = ("mse","mean_u","calibration","ace","spatial_spearman")
    ref.write_csv(OUT/"pass0_paired_runs.csv",pass0)
    ref.write_csv(OUT/"repeated_reuse_runs.csv",repeat)
    std_rows = [r for r in pass0 if not r["task"].startswith("h")]
    merged = averages(std_rows,("run","channel","model"),fields)
    pass0 += [dict(r,task="five_mask_mean") for r in merged]
    for ch in MODALITIES:
        for m in RUNS:
            for i in cohort:
                run = ds.samples[i][1]
                rr = {r["task"]:r for r in pass0 if r["channel"]==ch and r["model"]==m and r["run"]==run}
                a,b = rr["h22"],rr["h23"]
                b["collapse"] = float(b["mse"]>a["mse"] and b["mean_u"]<a["mean_u"])
    base = summaries(pass0,("task","channel"),fields)
    base += summaries([r for r in pass0 if r["task"]=="h23"],("task","channel"),("collapse",))
    ref.write_csv(OUT/"pass0_paired_summary.csv",base)
    ref.write_csv(OUT/"repeated_reuse_summary.csv",summaries(repeat,("task","step","channel"),fields))
    ref.write_csv(OUT/"repeated_reuse_change_summary.csv",summaries(
        change_from_initial(repeat, "step", "task", fields),("task","step","channel"),fields))
    print("Pass 0 and fixed-window reuse complete",flush=True)
    sliding = slide(ds,cohort,models,mean,std,device)
    ref.write_csv(OUT/"sliding_runs.csv",sliding)
    # Multiple frames at one observation age within a run are averaged before bootstrapping.
    per_run = averages(sliding,("run","axis","index","channel","model"),fields)
    ref.write_csv(OUT/"sliding_summary.csv",summaries(per_run,("axis","index","channel"),fields))
    ref.write_csv(OUT/"sliding_change_summary.csv",summaries(
        change_from_initial([r for r in per_run if r["axis"] == "step"], "index", "axis", fields),
        ("index","channel"), fields))
    ds.close()
    print("All comparisons complete",flush=True)


if __name__ == "__main__":
    main()
