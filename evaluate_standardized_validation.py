# evaluate_standardized_validation.py
"""Post-hoc held-out evaluation on the five standardized ~50% validation masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from data.masking import (
    FIXED_VALIDATION_PATTERNS,
    FIXED_VALIDATION_SPATIAL_RANDOM_SEED,
    MASKING_VERSION,
    full_mse_loss,
    make_fixed_validation_mask,
    make_visible_input,
)
from data.vpic_hdf5_dataset import VPICWindowDataset
from models.unet3d import LEGACY_MODEL_VERSION, UNet3D
from visualize_mask_patterns_unet3d import (
    DEFAULT_RUN_DIR,
    JY_NRMSE_DEFINITION,
    JY_STATS_DEFINITION,
    checkpoint_cache_signature,
    compute_jy,
    denormalize,
    expand_path,
    get_train_runs,
    get_val_runs,
    jy_metric_metadata,
    load_checkpoint,
    load_or_compute_jy_training_stats,
    normalize,
    resolve_checkpoint_path,
)


CACHE_VERSION = 3
OUTPUT_DIRNAME = "paper_figures_v1/standardized_validation"
CACHE_STEM = "standardized_validation"
PATTERN_LABELS = {
    "spatial_random": "Spatial random",
    "spatial_grid": "Spatial grid",
    "spatial_block": "Spatial block",
    "temporal_random": "Temporal random",
    "temporal_block": "Temporal block",
}
OVERALL_MSE_DEFINITION = (
    "mean((prediction_normalized - target_normalized)^2) over all four channels "
    "and all T,X,Z locations"
)
DENSITY_NRMSE_DEFINITION = (
    "sqrt(mean((prediction_normalized - target_normalized)^2)) on the Density channel"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate latest.pt on the five standardized 50% validation masks "
            "and write a paper numerical table. Does not plot."
        )
    )
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", default="latest.pt")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=[-21.0, 21.0, -50.0, 50.0],
    )
    parser.add_argument("--h5-dir", default=None)
    return parser.parse_args()


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def build_standardized_masks(
    shape: Sequence[int],
    device=None,
    dtype=torch.float32,
) -> Tuple[List[torch.Tensor], List[Dict]]:
    """Call make_fixed_validation_mask for each standardized pattern, in order."""
    masks = []
    infos = []
    for pattern in FIXED_VALIDATION_PATTERNS:
        mask, info = make_fixed_validation_mask(
            pattern,
            shape,
            device=device,
            dtype=dtype,
        )
        masks.append(mask)
        infos.append(info)
    return masks, infos


def window_overall_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(np.square(residual)))


def window_density_nrmse(prediction: np.ndarray, target: np.ndarray) -> float:
    residual = np.asarray(prediction[3], dtype=np.float64) - np.asarray(
        target[3], dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(residual))))


def window_jy_nrmse(
    prediction: np.ndarray,
    target: np.ndarray,
    extent: Sequence[float],
    jy_std_train: float,
) -> float:
    """Jy NRMSE on already-physical (HDF5-stored) Bx/Bz fields."""
    scale = float(jy_std_train)
    if scale <= 0.0:
        raise ValueError(f"jy_std_train must be positive, got {scale}.")
    residual = compute_jy(prediction, extent) - compute_jy(target, extent)
    return float(np.sqrt(np.mean(np.square(residual))) / scale)


def per_run_means(records: Sequence[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Mean over windows inside each physical run, for each pattern."""
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    metrics = ("overall_mse", "density_nrmse", "jy_nrmse")
    for record in records:
        pattern = record["pattern"]
        run_name = record["run_name"]
        by_run = grouped.setdefault(pattern, {}).setdefault(
            run_name,
            {name: [] for name in metrics},
        )
        for name in metrics:
            by_run[name].append(float(record[name]))
    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pattern, runs in grouped.items():
        aggregated[pattern] = {}
        for run_name, values in runs.items():
            aggregated[pattern][run_name] = {
                name: float(np.mean(values[name])) for name in metrics
            }
            aggregated[pattern][run_name]["n_windows"] = float(len(values["overall_mse"]))
    return aggregated


def cross_run_stats(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty run list.")
    return {
        "median": float(np.median(array)),
        "p16": float(np.percentile(array, 16.0)),
        "p84": float(np.percentile(array, 84.0)),
        "n_runs": int(array.size),
    }


def format_median_interval(stats: Dict[str, float]) -> str:
    return f"{stats['median']:.4f} [{stats['p16']:.4f}, {stats['p84']:.4f}]"


def split_identity(split: Dict) -> Dict:
    return {
        "val_runs": sorted(split.get("val_runs", [])),
        "train_runs": sorted(split.get("train_runs", [])),
        "n_val_runs": len(split.get("val_runs", [])),
        "n_train_runs": len(split.get("train_runs", [])),
        "num_val_windows": split.get("num_val_windows"),
        "num_train_windows": split.get("num_train_windows"),
    }


def cache_signature(
    checkpoint_path: Path,
    checkpoint_epoch: int,
    split: Dict,
    mean: Sequence[float],
    std: Sequence[float],
    jy_stats: Dict,
    extent: Sequence[float],
) -> Dict:
    jy_path = checkpoint_path.parent / "jy_stats.json"
    jy_file = None
    if jy_path.exists():
        stat = jy_path.stat()
        jy_file = {
            "path": str(jy_path),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }
    return {
        "cache_version": CACHE_VERSION,
        "checkpoint": checkpoint_cache_signature(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_epoch),
        "split": split_identity(split),
        "normalization": {
            "mean": [float(value) for value in mean],
            "std": [float(value) for value in std],
        },
        "patterns": list(FIXED_VALIDATION_PATTERNS),
        "masking_version": MASKING_VERSION,
        "fixed_validation_spatial_random_seed": int(FIXED_VALIDATION_SPATIAL_RANDOM_SEED),
        "shared_channel_mask": True,
        "jy_stats": {
            "jy_definition": jy_stats.get("jy_definition", JY_STATS_DEFINITION),
            "jy_preprocessing": jy_stats.get("jy_preprocessing")
            or jy_stats.get("preprocessing"),
            "jy_stats_cache_version": jy_stats.get("jy_stats_cache_version"),
            "jy_includes_mu0": jy_stats.get("jy_includes_mu0"),
            "B_unit": jy_stats.get("B_unit", jy_stats.get("jy_b_units")),
            "source_coordinate_unit": jy_stats.get("source_coordinate_unit"),
            "derivative_coordinate_unit": jy_stats.get("derivative_coordinate_unit"),
            "jy_unit": jy_stats.get("jy_unit", jy_stats.get("jy_physical_units")),
            "jy_b_units": jy_stats.get("jy_b_units"),
            "jy_coordinate_units": jy_stats.get("jy_coordinate_units"),
            "jy_physical_units": jy_stats.get("jy_physical_units"),
            "jy_std_train": float(jy_stats["jy_std_train"]),
            "jy_mean_train": float(jy_stats["jy_mean_train"]),
            "n_runs": jy_stats.get("n_runs"),
            "count": jy_stats.get("count"),
            "file": jy_file,
        },
        "extent": [float(value) for value in extent],
    }


def load_cached_payload(out_dir: Path, signature: Dict) -> Dict | None:
    json_path = out_dir / f"{CACHE_STEM}.json"
    npz_path = out_dir / f"{CACHE_STEM}.npz"
    if not json_path.exists() or not npz_path.exists():
        return None
    payload = json.loads(json_path.read_text())
    if payload.get("signature") != signature:
        print(f"Cache signature mismatch at {json_path}; will recompute.")
        return None
    print(f"Reusing standardized validation cache: {json_path}")
    return payload


def write_csv(records: Sequence[Dict], path: Path) -> None:
    fieldnames = [
        "run_name",
        "t0",
        "pattern",
        "overall_mse",
        "density_nrmse",
        "jy_nrmse",
        "validate_style_mse",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})


def write_markdown_table(summary: Dict[str, Dict[str, Dict[str, float]]], path: Path) -> None:
    lines = [
        "| Task | Overall MSE | Density NRMSE | Jy NRMSE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for pattern in FIXED_VALIDATION_PATTERNS:
        stats = summary[pattern]
        lines.append(
            "| {label} | {overall} | {density} | {jy} |".format(
                label=PATTERN_LABELS[pattern],
                overall=format_median_interval(stats["overall_mse"]),
                density=format_median_interval(stats["density_nrmse"]),
                jy=format_median_interval(stats["jy_nrmse"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def write_tex_table(summary: Dict[str, Dict[str, Dict[str, float]]], path: Path) -> None:
    rows = [
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Task & Overall MSE & Density NRMSE & Jy NRMSE \\",
        r"\hline",
    ]
    for pattern in FIXED_VALIDATION_PATTERNS:
        stats = summary[pattern]
        rows.append(
            r"{label} & {overall} & {density} & {jy} \\".format(
                label=PATTERN_LABELS[pattern],
                overall=format_median_interval(stats["overall_mse"]),
                density=format_median_interval(stats["density_nrmse"]),
                jy=format_median_interval(stats["jy_nrmse"]),
            )
        )
    rows.extend([r"\hline", r"\end{tabular}", ""])
    path.write_text("\n".join(rows))


def summarize_from_records(records: Sequence[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    per_run = per_run_means(records)
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pattern in FIXED_VALIDATION_PATTERNS:
        run_values = per_run[pattern]
        summary[pattern] = {
            metric: cross_run_stats(
                [run_values[run_name][metric] for run_name in sorted(run_values)]
            )
            for metric in ("overall_mse", "density_nrmse", "jy_nrmse")
        }
    return summary


def global_window_weighted_overall_mse(records: Sequence[Dict]) -> Dict[str, float]:
    """Equal weight per validation window, matching validate() averaging."""
    by_pattern: Dict[str, List[float]] = {pattern: [] for pattern in FIXED_VALIDATION_PATTERNS}
    for record in records:
        by_pattern[record["pattern"]].append(float(record["validate_style_mse"]))
    metrics = {
        pattern: float(np.mean(values)) for pattern, values in by_pattern.items()
    }
    metrics["mean_over_patterns"] = float(
        np.mean([metrics[pattern] for pattern in FIXED_VALIDATION_PATTERNS])
    )
    return metrics


def write_tables_from_payload(payload: Dict, out_dir: Path) -> None:
    records = payload["records"]
    summary = payload["summary"]
    write_csv(records, out_dir / f"{CACHE_STEM}.csv")
    write_markdown_table(summary, out_dir / f"{CACHE_STEM}_table.md")
    write_tex_table(summary, out_dir / f"{CACHE_STEM}_table.tex")
    print(f"Wrote {out_dir / f'{CACHE_STEM}.csv'}")
    print(f"Wrote {out_dir / f'{CACHE_STEM}_table.md'}")
    print(f"Wrote {out_dir / f'{CACHE_STEM}_table.tex'}")


def save_payload(payload: Dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = payload["records"]
    np.savez_compressed(
        out_dir / f"{CACHE_STEM}.npz",
        overall_mse=np.asarray([row["overall_mse"] for row in records], dtype=np.float64),
        density_nrmse=np.asarray([row["density_nrmse"] for row in records], dtype=np.float64),
        jy_nrmse=np.asarray([row["jy_nrmse"] for row in records], dtype=np.float64),
        validate_style_mse=np.asarray(
            [row["validate_style_mse"] for row in records], dtype=np.float64
        ),
        t0=np.asarray([row["t0"] for row in records], dtype=np.int32),
        run_name=np.asarray([row["run_name"] for row in records]),
        pattern=np.asarray([row["pattern"] for row in records]),
    )
    (out_dir / f"{CACHE_STEM}.json").write_text(json.dumps(json_ready(payload), indent=2))
    print(f"Saved {out_dir / f'{CACHE_STEM}.json'}")
    print(f"Saved {out_dir / f'{CACHE_STEM}.npz'}")


def select_val_window_indices(dataset: VPICWindowDataset, val_runs: set[str]) -> List[int]:
    indices = [
        index
        for index, (_file_index, run_name, _t0) in enumerate(dataset.samples)
        if run_name in val_runs
    ]
    if not indices:
        raise RuntimeError("No validation windows found for split.json val_runs.")
    return indices


@torch.no_grad()
def evaluate_windows(
    model: torch.nn.Module,
    dataset: VPICWindowDataset,
    sample_indices: Sequence[int],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    extent: Sequence[float],
    jy_std_train: float,
) -> Tuple[List[Dict], List[Dict]]:
    records: List[Dict] = []
    pattern_infos: List[Dict] | None = None
    model.eval()
    for sample_number, dataset_index in enumerate(sample_indices, start=1):
        sample = dataset[dataset_index]
        target = sample["block"].unsqueeze(0).to(device)
        target_norm = normalize(target, mean, std)
        masks, infos = build_standardized_masks(
            target_norm.shape,
            device=device,
            dtype=target_norm.dtype,
        )
        if pattern_infos is None:
            pattern_infos = infos
        stacked_mask = torch.cat(masks, dim=0)
        stacked_target = target_norm.expand(len(FIXED_VALIDATION_PATTERNS), -1, -1, -1, -1)
        visible = make_visible_input(stacked_target, stacked_mask)
        prediction = model(torch.cat([visible, stacked_mask], dim=1)).float()
        target_np = stacked_target[0].detach().cpu().numpy()
        pred_np = prediction.detach().cpu().numpy()
        target_phys_np = target[0].detach().cpu().numpy()
        pred_phys_np = denormalize(prediction, mean, std).detach().cpu().numpy()
        metadata = sample["metadata"]
        for pattern_index, pattern in enumerate(FIXED_VALIDATION_PATTERNS):
            pred_i = pred_np[pattern_index]
            mask_i = stacked_mask[pattern_index : pattern_index + 1]
            target_i = stacked_target[pattern_index : pattern_index + 1]
            pred_i_t = prediction[pattern_index : pattern_index + 1]
            records.append(
                {
                    "run_name": metadata["run_name"],
                    "t0": int(metadata["t0"]),
                    "pattern": pattern,
                    "overall_mse": window_overall_mse(pred_i, target_np),
                    "density_nrmse": window_density_nrmse(pred_i, target_np),
                    "jy_nrmse": window_jy_nrmse(
                        pred_phys_np[pattern_index],
                        target_phys_np,
                        extent,
                        jy_std_train,
                    ),
                    "validate_style_mse": float(
                        full_mse_loss(pred_i_t, target_i, mask_i).detach().cpu()
                    ),
                }
            )
        if (
            sample_number == 1
            or sample_number % 10 == 0
            or sample_number == len(sample_indices)
        ):
            print(
                f"Standardized validation: {sample_number}/{len(sample_indices)} windows"
            )
    return records, pattern_infos or []


def main() -> None:
    args = parse_args()
    run_dir = expand_path(args.run_dir)
    checkpoint_path = resolve_checkpoint_path(run_dir, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    out_dir = (
        expand_path(args.out_dir)
        if args.out_dir is not None
        else run_dir / OUTPUT_DIRNAME
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    split_path = run_dir / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}")
    split = json.loads(split_path.read_text())

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt["args"]
    checkpoint_epoch = int(ckpt["epoch"])
    stats = ckpt["stats"]
    mean_list = [float(value) for value in stats["mean"]]
    std_list = [float(value) for value in stats["std"]]

    jy_path = run_dir / "jy_stats.json"
    if not jy_path.exists():
        raise FileNotFoundError(
            f"Missing {jy_path}; standardized evaluation reuses existing Jy stats."
        )
    jy_stats = json.loads(jy_path.read_text())
    signature = cache_signature(
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=checkpoint_epoch,
        split=split,
        mean=mean_list,
        std=std_list,
        jy_stats=jy_stats,
        extent=args.extent,
    )

    payload = None if args.force_recompute else load_cached_payload(out_dir, signature)
    if payload is None:
        device = torch.device(
            args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        ckpt, ckpt_path = load_checkpoint(run_dir, args.checkpoint, device=device)
        h5_dir = expand_path(args.h5_dir or ckpt_args["h5_dir"])
        delta_t = int(ckpt_args.get("delta_t", 24))
        dataset = VPICWindowDataset(
            h5_dir=h5_dir,
            betas=ckpt_args.get("betas", [0.2]),
            delta_t=delta_t,
            stride_t=int(ckpt_args.get("stride_t", 2)),
            layout="C T X Z",
            return_metadata=True,
        )
        val_runs = get_val_runs(run_dir)
        train_runs = get_train_runs(run_dir)
        if val_runs is None or train_runs is None:
            dataset.close()
            raise FileNotFoundError("split.json must contain val_runs and train_runs.")
        mean = torch.tensor(mean_list, dtype=torch.float32, device=device).view(1, 4, 1, 1, 1)
        std = torch.tensor(std_list, dtype=torch.float32, device=device).view(1, 4, 1, 1, 1)
        jy_stats = load_or_compute_jy_training_stats(
            run_dir=run_dir,
            dataset=dataset,
            train_runs=train_runs,
            mean=mean,
            std=std,
            extent=args.extent,
            checkpoint_path=ckpt_path,
            checkpoint_epoch=checkpoint_epoch,
        )
        model = UNet3D(
            in_channels=8,
            out_channels=4,
            base_channels=int(ckpt_args.get("base_channels", 16)),
            channel_mults=ckpt_args.get("channel_mults", [1, 2, 4]),
            architecture=ckpt_args.get("model_version", LEGACY_MODEL_VERSION),
            use_attention=bool(ckpt_args.get("use_attention", False)),
            spatial_only_pooling=bool(ckpt_args.get("spatial_only_pooling", False)),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        sample_indices = select_val_window_indices(dataset, val_runs)
        records, pattern_infos = evaluate_windows(
            model=model,
            dataset=dataset,
            sample_indices=sample_indices,
            mean=mean,
            std=std,
            device=device,
            extent=args.extent,
            jy_std_train=float(jy_stats["jy_std_train"]),
        )
        signature = cache_signature(
            checkpoint_path=ckpt_path,
            checkpoint_epoch=checkpoint_epoch,
            split=split,
            mean=mean_list,
            std=std_list,
            jy_stats=jy_stats,
            extent=args.extent,
        )
        summary = summarize_from_records(records)
        per_run = per_run_means(records)
        payload = {
            "signature": signature,
            "checkpoint": str(ckpt_path),
            "checkpoint_epoch": checkpoint_epoch,
            "patterns": list(FIXED_VALIDATION_PATTERNS),
            "pattern_labels": PATTERN_LABELS,
            "pattern_infos": pattern_infos,
            "n_windows": len(sample_indices),
            "n_runs": len(val_runs),
            "records": records,
            "per_run": per_run,
            "summary": summary,
            "global_window_weighted_overall_mse": global_window_weighted_overall_mse(
                records
            ),
            "aggregation": (
                "mean over windows within each validation run, then cross-run "
                "median and 16th-84th percentiles"
            ),
            "overall_mse_definition": OVERALL_MSE_DEFINITION,
            "density_nrmse_definition": DENSITY_NRMSE_DEFINITION,
            "jy_nrmse_definition": JY_NRMSE_DEFINITION,
            "jy_definition": JY_STATS_DEFINITION,
            "note": (
                "global_window_weighted_overall_mse uses data.masking.full_mse_loss "
                "averaged with equal window weight, like train_masked_unet3d.validate(); "
                "it is metadata only and is not the paper table statistic"
            ),
        }
        payload.update(
            jy_metric_metadata(
                jy_stats=jy_stats,
                checkpoint_path=ckpt_path,
                checkpoint_epoch=checkpoint_epoch,
            )
        )
        save_payload(payload, out_dir)
        dataset.close()

    write_tables_from_payload(payload, out_dir)
    manifest = {
        "out_dir": str(out_dir),
        "checkpoint": payload.get("checkpoint", str(checkpoint_path)),
        "checkpoint_epoch": payload.get("checkpoint_epoch", checkpoint_epoch),
        "cache_version": CACHE_VERSION,
        "force_recompute": bool(args.force_recompute),
        "files": [
            f"{CACHE_STEM}.npz",
            f"{CACHE_STEM}.json",
            f"{CACHE_STEM}.csv",
            f"{CACHE_STEM}_table.md",
            f"{CACHE_STEM}_table.tex",
            "manifest.json",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
