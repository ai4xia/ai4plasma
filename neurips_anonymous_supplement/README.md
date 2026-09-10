# Anonymous NeurIPS Supplementary Package

This is the anonymous supplementary package for the submitted plasma
reconstruction paper. It is intended for reviewers who need to inspect the
model, configuration, train/validation split, and evaluation code under
double-blind constraints.

## What is included

- Training, evaluation, and model source (`train_masked_unet3d.py`,
  `make_paper_figures.py`, `evaluate_standardized_validation.py`, helpers,
  `data/`, `models/`, `tests/`)
- A sanitized inference checkpoint (`checkpoint/latest.pt`, epoch 4500)
- Exact training configuration and split (`configs/`)
- Reproduction scripts (`scripts/`)
- Environment specification (`environment.yml`, `requirements.txt`)
- `LICENSE` (MIT, Anonymous Authors) and `THIRD_PARTY_LICENSES.md`

## What is not included

The processed VPIC corpus used in the paper is **not included in this
submission**. Provenance, attribution, and release permissions are still being
coordinated with the simulation contributors.

Therefore:

- **Code and the sanitized checkpoint are provided in this anonymous supplementary package.**
- **Checkpoint-based model inspection is possible** without the HDF5 corpus.
- **Exact end-to-end retraining on the paper's VPIC corpus is not possible**
  without access to that corpus.
- Standardized evaluation and paper-figure regeneration also require the
  corpus once it is supplied locally.

Do not assume a public download URL exists.

## Software environment (official training)

- Python 3.12.13
- PyTorch 2.13.0+cu130
- CUDA 13.0
- NCCL 2.29.7
- cuDNN 9.20.0

See `environment.yml` and `requirements.txt`.

## Formal training configuration

- 4 nodes × 4 NVIDIA A100-SXM4-80GB GPUs (16 GPUs total)
- batch size 8/GPU, global batch 128
- 4500 epochs
- validation after every epoch over all five standardized patterns
- AdamW, learning rate `2e-4`, weight decay `1e-4`
- 10-epoch linear warmup, then cosine decay to `2e-6`
- AMP + PyTorch DDP / NCCL
- independent B / Density five-mask training sampler with log-uniform exact
  visible counts (`p_zero = p_full = 0.05`)

The released `configs/config.json` records these hyperparameters. The
checkpoint is inference-only: optimizer state and logging metadata were
removed.

## Commands

All commands are run from the package root.

### Load the checkpoint (no HDF5 required)

```bash
python - <<'PY'
import torch
from models.unet3d import UNet3D

ckpt = torch.load("checkpoint/latest.pt", map_location="cpu", weights_only=False)
args = ckpt["args"]
model = UNet3D(
    in_channels=8,
    out_channels=4,
    base_channels=int(args["base_channels"]),
    channel_mults=args["channel_mults"],
    architecture=args["model_version"],
    use_attention=bool(args["use_attention"]),
    spatial_only_pooling=bool(args["spatial_only_pooling"]),
)
model.load_state_dict(ckpt["model"])
model.eval()
print("epoch", ckpt["epoch"], "best_val_mse", ckpt["best_val_mse"])
PY
```

### Standardized evaluation (requires local VPIC HDF5)

```bash
export H5_DIR=/path/to/processed_vpic_hdf5
bash scripts/evaluate_paper.sh
```

If `H5_DIR` is unset, the script looks in `data/hdf5/` and exits with an
explicit message when the corpus is absent.

### Paper figures (requires local VPIC HDF5)

`scripts/evaluate_paper.sh` also runs `make_paper_figures.py`. Equivalent:

```bash
python make_paper_figures.py \
  --run-dir . \
  --checkpoint checkpoint/latest.pt \
  --h5-dir "$H5_DIR" \
  --sliding-max-runs 25
```

### Formal 16-GPU training (requires local VPIC HDF5)

```bash
export H5_DIR=/path/to/processed_vpic_hdf5
export MASTER_ADDR=<master-node>
bash scripts/train_16gpu.sh
```

This launches the official 4500-epoch recipe with `--no-wandb`.
