#!/usr/bin/env python3
"""Create an evaluation-compatible copy (`final_audited.pt`) of the Phase D checkpoint.

Does not modify either original `final.pt`. Verifies architecture and normalization
parity with original v11, inherits verified architecture/evaluation metadata while
preserving Phase D weights and training provenance, and stages `jy_stats.json` if needed.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import torch

from data.vpic_hdf5_dataset import fixed_split_sha256, validate_final_checkpoint

INHERITED_ARCH_AND_EVAL_KEYS: List[str] = [
    "base_channels",
    "channel_mults",
    "model_version",
    "uncertainty_aware",
    "use_attention",
    "spatial_only_pooling",
    "variance_floor",
    "masking_version",
    "delta_t",
    "stride_t",
    "val_patterns",
    "val_mask_seed",
    "uncertainty_loss",
    "student_t_df",
]


def _sanitize_paths(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_paths(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_paths(v) for v in obj)
    return obj


def build_audited_checkpoint(
    v11_ckpt_path: Path,
    phased_ckpt_path: Path,
    out_ckpt_path: Path,
    copy_jy_stats: bool = True,
) -> Dict[str, Any]:
    """Build and validate `final_audited.pt` from Phase D and v11 checkpoints."""
    v11_ckpt = torch.load(v11_ckpt_path, map_location="cpu", weights_only=False)
    phd_ckpt = torch.load(phased_ckpt_path, map_location="cpu", weights_only=False)

    # 1. Verify state_dict architecture parity
    v11_keys = list(v11_ckpt["model"].keys())
    phd_keys = list(phd_ckpt["model"].keys())
    if v11_keys != phd_keys:
        raise ValueError("Model state_dict keys differ between v11 and Phase D checkpoints.")
    for k in v11_keys:
        if v11_ckpt["model"][k].shape != phd_ckpt["model"][k].shape:
            raise ValueError(
                f"Tensor shape mismatch for '{k}': "
                f"v11={v11_ckpt['model'][k].shape}, phd={phd_ckpt['model'][k].shape}"
            )

    # 2. Verify normalization stats parity
    if v11_ckpt["stats"] != phd_ckpt["stats"]:
        raise ValueError(
            f"Normalization stats differ between v11 and Phase D: "
            f"{v11_ckpt['stats']} vs {phd_ckpt['stats']}"
        )

    # 3. Preserve Phase D weights, epoch, history, and original args
    audited = copy.deepcopy(phd_ckpt)
    raw_phd_args = _sanitize_paths(dict(phd_ckpt["args"]))
    audited["phase_d_args"] = copy.deepcopy(raw_phd_args)

    v11_args = v11_ckpt["args"]
    merged_args = copy.deepcopy(raw_phd_args)
    inherited_fields: Dict[str, Any] = {}
    for key in INHERITED_ARCH_AND_EVAL_KEYS:
        if key not in merged_args and key in v11_args:
            merged_args[key] = copy.deepcopy(v11_args[key])
            inherited_fields[key] = copy.deepcopy(v11_args[key])

    merged_args["final_epoch"] = int(phd_ckpt["epoch"])
    audited["args"] = merged_args

    expected_sha = fixed_split_sha256()
    audited["split_path"] = str(raw_phd_args.get("split", "split.json"))
    audited["split_sha256"] = expected_sha
    audited["phase_d_provenance"] = {
        "source_phase_d_checkpoint": str(phased_ckpt_path),
        "reference_v11_checkpoint": str(v11_ckpt_path),
        "inherited_arch_and_eval_fields": inherited_fields,
        "added_validation_fields": {
            "split_sha256": expected_sha,
            "args.final_epoch": int(phd_ckpt["epoch"]),
        },
        "note": (
            "Model weights, epoch, global_step, optimizer, scaler, stats, and history "
            "are retained verbatim from Phase D; original Phase D args are preserved in "
            "`phase_d_args`."
        ),
    }

    # 4. Validate against official project rules before saving
    validate_final_checkpoint(audited)

    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(audited, out_ckpt_path)

    # 5. Stage jy_stats.json if missing in Phase D run dir and present in v11 run dir
    v11_jy = v11_ckpt_path.parent / "jy_stats.json"
    out_jy = out_ckpt_path.parent / "jy_stats.json"
    jy_staged = False
    if copy_jy_stats and v11_jy.exists() and not out_jy.exists():
        jy_data = json.loads(v11_jy.read_text())
        if (
            jy_data.get("channel_mean") != phd_ckpt["stats"]["mean"]
            or jy_data.get("channel_std") != phd_ckpt["stats"]["std"]
        ):
            raise ValueError("v11 jy_stats.json channel_mean/std do not match Phase D stats.")
        shutil.copy2(v11_jy, out_jy)
        jy_staged = True

    return {
        "out_checkpoint": str(out_ckpt_path),
        "epoch": int(audited["epoch"]),
        "split_sha256": expected_sha,
        "inherited_keys": sorted(inherited_fields.keys()),
        "jy_stats_staged": jy_staged,
        "jy_stats_path": str(out_jy) if out_jy.exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare evaluation-compatible final_audited.pt for Phase D."
    )
    parser.add_argument(
        "--v11-checkpoint",
        type=Path,
        default=Path("runs/multi_beta_gkl_v11_fromscratch_256ep_20261122/final.pt"),
    )
    parser.add_argument(
        "--phased-checkpoint",
        type=Path,
        default=Path("runs/multi_beta_gkl_v11_phase_d_pilot_6ep_20261123/final.pt"),
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("runs/multi_beta_gkl_v11_phase_d_pilot_6ep_20261123/final_audited.pt"),
    )
    args = parser.parse_args()

    info = build_audited_checkpoint(
        v11_ckpt_path=args.v11_checkpoint,
        phased_ckpt_path=args.phased_checkpoint,
        out_ckpt_path=args.output_checkpoint,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
