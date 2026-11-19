#!/usr/bin/env python3
"""Paired Gaussian continuation inference using the epoch-256 evaluation protocol."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

PROJECT = Path("/pscratch/sd/b/binxia/ai4plasma")
REFERENCE = PROJECT / "evaluations/three_phase_256ep_analysis_20261116/run_analysis.py"
OUTPUT = PROJECT / "evaluations/gaussian_u_continuation_20ep_analysis_20261116"
GAUSSIAN_RUN = PROJECT / "runs/multi_beta_three_phase_gaussian_u_continuation_20ep_20261116"
STUDENT_RUN = PROJECT / "runs/multi_beta_curriculum_direct_v3_three_phase_globalcosine_2e4_1e5_256ep_20261115"

spec = importlib.util.spec_from_file_location("three_phase_reference_evaluation", REFERENCE)
reference = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(reference)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if any(path.name not in {"run_inference.py", "__pycache__"} for path in OUTPUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing evaluation outputs in {OUTPUT}")
    device = torch.device("cuda")
    reference.OUTPUT = OUTPUT
    checkpoint = torch.load(
        GAUSSIAN_RUN / "final.pt", map_location="cpu", weights_only=False,
    )
    args = checkpoint["args"]
    model = reference.UNet3D(
        in_channels=8, out_channels=4,
        base_channels=int(args.get("base_channels", 16)),
        channel_mults=args.get("channel_mults", [1, 2, 4]),
        architecture=args["model_version"],
        uncertainty_aware=bool(args["uncertainty_aware"]),
        use_attention=bool(args.get("use_attention", False)),
        spatial_only_pooling=bool(args.get("spatial_only_pooling", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    stats_payload = checkpoint["stats"]
    mean = torch.tensor(stats_payload["mean"], device=device).view(1, 4, 1, 1, 1)
    std = torch.tensor(stats_payload["std"], device=device).view(1, 4, 1, 1, 1)
    if int(checkpoint["epoch"]) != 276:
        raise RuntimeError(f"Expected Gaussian epoch 276, got {checkpoint['epoch']}")
    if args["uncertainty_loss"] != "gaussian_nll":
        raise RuntimeError(f"Expected Gaussian NLL, got {args['uncertainty_loss']}")

    student = torch.load(STUDENT_RUN / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    if checkpoint["split_sha256"] != student["split_sha256"]:
        raise RuntimeError("Student-t and Gaussian checkpoints use different splits")
    if checkpoint["stats"] != student["stats"]:
        raise RuntimeError("Student-t and Gaussian checkpoints use different normalization")

    split = reference.load_fixed_split()
    dataset = reference.VPICWindowDataset(
        h5_dir=args["h5_dir"], betas=args["betas"], delta_t=int(args["delta_t"]),
        stride_t=1, layout="C T X Z", return_metadata=True,
    )
    role_indices = {
        role: reference.select_validation_statistics_indices(
            dataset, set(split[f"{role}_runs"]), 24, 1,
        )
        for role in ("val", "test")
    }
    if {key: len(value) for key, value in role_indices.items()} != {"val": 125, "test": 125}:
        raise RuntimeError("Expected exactly 125 paired validation and test windows")

    standard_rows, standard_samples = reference.evaluate_standard(
        model, mean, std, dataset, role_indices, device,
    )
    (OUTPUT / "standard_mask_detailed_gaussian.csv").write_bytes(
        (OUTPUT / "standard_mask_metrics.csv").read_bytes()
    )
    (OUTPUT / "standard_mask_metrics.csv").unlink()
    forecast_rows, forecast_samples = reference.evaluate_forecast(
        model, mean, std, dataset, role_indices["test"], device,
    )
    (OUTPUT / "forecast_horizon_metrics_gaussian.csv").write_bytes(
        (OUTPUT / "forecast_horizon_metrics.csv").read_bytes()
    )
    (OUTPUT / "forecast_horizon_metrics.csv").unlink()
    reference.residual_and_likelihood_rows(forecast_samples)
    (OUTPUT / "forecast_residual_likelihood_gaussian.csv").write_bytes(
        (OUTPUT / "forecast_residual_likelihood.csv").read_bytes()
    )
    (OUTPUT / "forecast_residual_likelihood.csv").unlink()

    manifest = {
        "student_checkpoint": str(STUDENT_RUN / "final.pt"),
        "gaussian_checkpoint": str(GAUSSIAN_RUN / "final.pt"),
        "student_epoch": 256,
        "gaussian_epoch": 276,
        "split_sha256": checkpoint["split_sha256"],
        "split_counts": {key: len(split[key]) for key in ("train_runs", "val_runs", "test_runs")},
        "paired_windows": {key: len(value) for key, value in role_indices.items()},
        "forecast_horizons": list(range(1, 24)),
        "forecast_geometry": "all four channels observed for prefix 24-h and unknown for suffix h",
        "paired_noise_seed": reference.SEED + 200000,
        "normalization_identical": True,
        "student_inference_reused": str(
            PROJECT / "evaluations/three_phase_256ep_analysis_20261116/forecast_horizon_metrics.csv"
        ),
    }
    (OUTPUT / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2))
    dataset.close()
    print(f"Gaussian paired inference complete: {OUTPUT}")


if __name__ == "__main__":
    main()
