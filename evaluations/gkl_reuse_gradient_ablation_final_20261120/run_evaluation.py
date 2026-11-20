"""Evaluate mean-only reuse and join paired, archived control measurements."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch


OUT = Path(__file__).resolve().parent
CACHE = OUT.parent / "gkl_differentiable_vs_detached_reuse_20261119"
NEW = "mean_only"
CONTROLS = ("detached", "differentiable")
MODELS = ("detached", NEW, "differentiable")
FIELDS = ("mse", "mean_u", "calibration", "ace", "spatial_spearman")


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


previous = load_script("previous_reuse", CACHE / "run_evaluation.py")
ref = previous.ref
RUN = previous.ROOT / "runs/multi_beta_gkl_joint_grad_mean_only_reuse_continuation_20ep_20261119"


def read_csv(path, numeric=()):
    with path.open() as f:
        return [dict(row, **{key: float(row[key]) for key in numeric}) for row in csv.DictReader(f)]


def audit():
    def load(path):
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)

    checkpoint = load(RUN / "final.pt")
    args, state = checkpoint["args"], checkpoint["curriculum_state"]
    assert checkpoint["epoch"] == args["final_epoch"] == args["stop_after_epochs"] == 276
    assert checkpoint["model"] and checkpoint["optimizer"]["state"]
    assert args["gkl_joint_grad_continuation_20ep"] and args["uncertainty_loss"] == "generalized_kl"
    assert args["differentiable_phase_c_reuse"] and args["mean_only_phase_c_reuse"]
    assert args["curriculum_config"] == "three_phase_global_cosine" and args["amp_dtype"] == "bfloat16"
    assert state["lambda_v"] == state["lambda_r"] == 1 and state["phase_d_start_epoch"] is None
    assert (previous.ROOT / args["post_final_continuation"]).resolve() == ref.SRC_CKPT
    assert all(group["lr"] == 1e-5 for group in checkpoint["optimizer"]["param_groups"])
    ref.validate_checkpoint_split(checkpoint)
    source = load(ref.SRC_CKPT)
    assert source["epoch"] == 256 and source["stats"] == checkpoint["stats"]
    assert source["split_sha256"] == checkpoint["split_sha256"]
    assert all(torch.isfinite(v).all() for v in checkpoint["model"].values())
    assert all(torch.isfinite(v).all() for entry in checkpoint["optimizer"]["state"].values()
               for v in entry.values() if torch.is_tensor(v))
    differences = {}
    for name in CONTROLS:
        control = load(previous.RUNS[name] / "final.pt")
        old = control["args"]
        assert control["epoch"] == 276 and control["stats"] == checkpoint["stats"]
        assert control["split_sha256"] == checkpoint["split_sha256"]
        assert bool(old.get("differentiable_phase_c_reuse")) == (name == "differentiable")
        assert not old.get("mean_only_phase_c_reuse")
        changed = {k for k in args.keys() | old.keys() if args.get(k) != old.get(k)}
        assert changed <= {"out_dir", "wandb_name", "wandb_tags",
                           "differentiable_phase_c_reuse", "mean_only_phase_c_reuse"}, changed
        differences[name] = sorted(changed)
    rows = json.loads((RUN / "history.json").read_text())
    assert [r["epoch"] for r in rows] == list(range(257, 277))
    history = []
    for r in rows:
        c = r["curriculum"]
        assert r["learning_rate"] == 1e-5 and r["nonfinite_events"] == r["amp_skipped_steps"] == 0
        assert c["lambda_v_mean"] == c["lambda_r_mean"] == c["two_forward_fraction"] == 1
        assert c["successful_optimizer_steps"] == r["epoch"] * state["updates_per_epoch"]
        values = dict(model=NEW, epoch=r["epoch"], train_mse=r["train_mse"], val_mse=r["val_mse"],
                      grad_norm=r["gradient_norm_mean"], gkl_alpha=c["gkl_alpha"], gkl_pass0=c["L_gkl_0"],
                      gkl_pass1=c["L_gkl_1"], nonfinite_events=r["nonfinite_events"],
                      amp_skipped_steps=r["amp_skipped_steps"],
                      successful_optimizer_steps=c["successful_optimizer_steps"])
        assert all(np.isfinite(values[k]) for k in values if k != "model")
        history.append(values)
    controls = read_csv(CACHE / "training_history.csv", ("gkl_alpha", "successful_optimizer_steps"))
    for name in CONTROLS:
        assert [(r["gkl_alpha"], r["successful_optimizer_steps"]) for r in history] == [
            (r["gkl_alpha"], r["successful_optimizer_steps"]) for r in controls if r["model"] == name]
    assert state["successful_optimizer_steps"] == history[-1]["successful_optimizer_steps"]
    ref.write_csv(OUT / "training_history.csv", controls + history)
    ref.dump_json(OUT / "training_audit.json", dict(
        checkpoint=RUN / "final.pt", source=ref.SRC_CKPT, configuration_differences=differences,
        split_sha256=checkpoint["split_sha256"], finite_model_and_optimizer=True, fixed_lr=1e-5,
        gradient_routing=dict(path_a=True, path_b=True, cross_forward_mean="cosine alpha",
                              cross_forward_uncertainty="detached"),
        final=history[-1]))
    return args


def summarize(rows, keys, metrics):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        by_pair = grouped.setdefault(key, {})
        pair = (row["run"], row["model"])
        assert pair not in by_pair, (key, pair)
        by_pair[pair] = row
    results = []
    for key, group in grouped.items():
        runs = sorted({run for run, _ in group})
        assert len(runs) == 24
        for metric in metrics:
            values = {m: np.array([group[run, m][metric] for run in runs]) for m in MODELS}
            assert all(np.isfinite(x).all() for x in values.values())
            row = dict(zip(keys, key), metric=metric, runs=len(runs),
                       **{m: float(x.mean()) for m, x in values.items()})
            if metric != "ace":
                for label, a, b in (("mean_minus_detached", NEW, "detached"),
                                    ("full_minus_mean", "differentiable", NEW),
                                    ("full_minus_detached", "differentiable", "detached")):
                    for field, value in zip(("delta", "low", "high"),
                                            ref.paired_bootstrap_ci(values[a], values[b])):
                        row[f"{label}_{field}"] = value
            results.append(row)
    return results


def combine(pairs, repeat, sliding):
    for stem, rows, ints in (
        ("pass0_paired", pairs, ()), ("repeated_reuse", repeat, ("step",)),
        ("sliding", sliding, ("index",)),
    ):
        cached = read_csv(CACHE / f"{stem}_runs.csv", FIELDS)
        for row in cached:
            for key in ints:
                row[key] = int(row[key])
        rows[:0] = cached
        ref.write_csv(OUT / f"{stem}_runs.csv", rows)
    average = previous.averages([r for r in pairs if not r["task"].startswith("h")],
                                ("run", "channel", "model"), FIELDS)
    pairs += [dict(r, task="five_mask_mean") for r in average]
    h22 = {(r["run"], r["channel"], r["model"]): r for r in pairs if r["task"] == "h22"}
    h23 = [r for r in pairs if r["task"] == "h23"]
    for r in h23:
        before = h22[r["run"], r["channel"], r["model"]]
        r["collapse"] = float(r["mse"] > before["mse"] and r["mean_u"] < before["mean_u"])
    ref.write_csv(OUT / "pass0_paired_summary.csv",
                  summarize(pairs, ("task", "channel"), FIELDS)
                  + summarize(h23, ("task", "channel"), ("collapse",)))
    per_run = previous.averages(sliding, ("run", "axis", "index", "channel", "model"), FIELDS)
    for stem, rows, keys, step in (
        ("repeated_reuse", repeat, ("task", "step", "channel"), "step"),
        ("sliding", per_run, ("axis", "index", "channel"), "index"),
    ):
        ref.write_csv(OUT / f"{stem}_summary.csv", summarize(rows, keys, FIELDS))
        step_rows = [r for r in rows if r.get("axis") != "age"]
        changes = FIELDS[:3] + FIELDS[4:]
        delta = previous.change_from_initial(step_rows, step, keys[0], changes)
        ref.write_csv(OUT / f"{stem}_change_summary.csv", summarize(delta, keys, changes))
    steps = {(r["run"], r["model"], r["channel"], r["index"]): r
             for r in per_run if r["axis"] == "step"}
    transitions = []
    for start, end in ((0, 1), (3, 6)):
        for (run, model, channel, index), row in steps.items():
            if index == end:
                earlier = steps[run, model, channel, start]
                transitions.append(dict(row, transition=f"{start} to {end}",
                                        **{m: row[m] / earlier[m] - 1 for m in FIELDS[:3]}))
    ref.write_csv(OUT / "sliding_transition_summary.csv",
                  summarize(transitions, ("transition", "channel"), FIELDS[:3]))


def plot(forecast, ranks):
    keyed = {(str(r["horizon"]), r["channel"], r["model"]): r["mean_frame_spatial_spearman"]
             for r in ranks}
    rows = []
    for h in range(1, 24):
        for ch in previous.MODALITIES:
            for metric, field in zip(FIELDS[:3], ("mean_mse", "mean_predicted_u",
                                                  "predicted_u_over_actual_mse")):
                values = {r["model"]: r[field] for r in forecast if r["channel"] == ch
                          and int(r["horizon"]) == h}
                rows.append(dict(axis="forecast", index=h, channel=ch, metric=metric, **values))
            if h in previous.FOCUS:
                rows.append(dict(axis="forecast", index=h, channel=ch, metric="spatial_spearman",
                                 **{m: keyed[str(h), ch, m] for m in MODELS}))
    ref.write_csv(OUT / "forecast_plot.csv", rows)
    plotter = load_script("reuse_plots", OUT.parent / "gkl_mean_only_reuse_analysis_20261119/plot_rollout.py")
    plotter.main(OUT, OUT, three_models=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    opts = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")
    args = audit()
    print("Checkpoint and histories audited", flush=True)
    if opts.audit_only:
        return
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    _, model, mean, std = ref.load_model(RUN / "final.pt", device)
    models = {NEW: model}
    split = ref.load_fixed_split()
    ds = ref.VPICWindowDataset(args["h5_dir"], betas=args["betas"], delta_t=24, stride_t=1,
                               layout="C T X Z", return_metadata=True)
    indices = {r: ref.select_validation_statistics_indices(ds, set(split[r+"_runs"]), 24, 1)
               for r in ("val", "test")}
    assert all(len(v) == 125 for v in indices.values())
    protocol = json.loads((CACHE / "protocol.json").read_text())
    cohort = [r["index"] for r in protocol["paired_cohort"]]
    assert len(cohort) == len(set(cohort)) == 24
    for r in protocol["paired_cohort"]:
        assert tuple(ds.samples[r["index"]][1:]) == (r["run"], r["start"])
        assert r["index"] in indices["test"]
    protocol.update(slurm_job_id=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                    historical_cache=str(CACHE), inferred_models=[NEW],
                    ace="Count-weighted absolute bin-mean gap using up to 4096 sampled hidden "
                    "voxels per group, with model-specific deciles for mean-only and archived "
                    "pooled deciles for the two controls.",
                    ace_comparability="Formula and sampled voxels match the archive. The archive "
                    "does not retain voxel predictions or shared decile boundaries. Mean-only uses "
                    "its own deciles, controls keep their pooled deciles. No paired ACE intervals.",
                    contrasts=dict(mean_minus_detached="effect of enabling mean reuse gradients",
                                   full_minus_mean="additional effect of uncertainty reuse gradients"))
    ref.dump_json(OUT / "protocol.json", protocol)
    pairs, repeat, standard, forecast, ranks, collapse = previous.pass_zero(
        ds, indices, set(cohort), models, mean, std, device)
    for name, new in (("standard_mask_metrics", standard), ("forecast_horizon_metrics", forecast),
                      ("spatial_uncertainty_metrics", ranks)):
        new[:0] = read_csv(CACHE / f"{name}.csv")
        ref.write_csv(OUT / f"{name}.csv", new)
    old = json.loads((CACHE / "extreme_horizon.json").read_text())
    for ch in collapse:
        collapse[ch].update(old[ch])
    ref.dump_json(OUT / "extreme_horizon.json", collapse)
    print("Pass 0 and fixed-window reuse complete", flush=True)
    sliding = previous.slide(ds, cohort, models, mean, std, device)
    combine(pairs, repeat, sliding)
    plot(forecast, ranks)
    print("All comparisons complete", flush=True)


if __name__ == "__main__":
    main()
