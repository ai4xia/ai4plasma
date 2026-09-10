#!/usr/bin/env python3
"""Build an anonymized NeurIPS review supplement. Not itself part of the ZIP."""

from __future__ import annotations

import json
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SRC_RUN = ROOT / (
    "runs/masked-resunet3d_beta0p2_dt24_bc24_depth4_ddp16_v15_"
    "orientedSpatialBlock_independentBD_logUniformCounts_"
    "attention_spatialpool_b8_e4500"
)
RUN_NAME = SRC_RUN.name
RELEASE_NAME = "neurips_anonymous_supplement"
RELEASE = ROOT / RELEASE_NAME
ZIP_PATH = ROOT / f"{RELEASE_NAME}.zip"

SOURCE_FILES = [
    "train_masked_unet3d.py",
    "make_paper_figures.py",
    "evaluate_standardized_validation.py",
    "visualize_mask_patterns_unet3d.py",
    "visualize_sliding_density_reconstruction.py",
    "data/masking.py",
    "data/vpic_hdf5_dataset.py",
    "models/unet3d.py",
    "tests/test_masking.py",
    "tests/test_unet3d.py",
    "tests/test_training_metrics.py",
    "tests/test_distributed_training.py",
    "tests/test_visualization_masks.py",
    "tests/test_paper_figures.py",
    "tests/test_standardized_validation.py",
    "tests/test_sliding_density_reconstruction.py",
]

SCAN_PATTERNS = [
    ("Bin", re.compile(r"Bin")),
    ("binxia", re.compile(r"binxia", re.I)),
    ("xiabin", re.compile(r"xiabin", re.I)),
    ("ai4xia", re.compile(r"ai4xia", re.I)),
    ("gatech", re.compile(r"gatech", re.I)),
    ("Georgia Tech", re.compile(r"Georgia\s+Tech", re.I)),
    ("bjcpdayh", re.compile(r"bjcpdayh", re.I)),
    ("/pscratch/sd/b/", re.compile(r"/pscratch/sd/b/")),
    ("wandb.ai", re.compile(r"wandb\.ai", re.I)),
    ("wandb_project identifier", re.compile(r"wandb_project['\"]?\s*[:=]\s*['\"](?!anonymous-review['\"]|['\"])[^'\"]+['\"]")),
    ("wandb_entity identifier", re.compile(r"wandb_entity['\"]?\s*[:=]\s*['\"][^'\"]+['\"]")),
    ("ai4plasma", re.compile(r"ai4plasma", re.I)),
]


README = r'''# Anonymous NeurIPS Review Supplement

This archive contains the training, evaluation, and model code used for the
paper, plus a sanitized inference checkpoint and the official train/validation
split. It does **not** include the processed HDF5 corpus.

## Contents

- `train_masked_unet3d.py` — masked 3D residual U-Net training (PyTorch DDP)
- `make_paper_figures.py` / `evaluate_standardized_validation.py` — paper figures and the five-pattern standardized validation table
- `visualize_mask_patterns_unet3d.py`, `visualize_sliding_density_reconstruction.py` — shared inference/plot helpers
- `data/`, `models/` — dataset, masking, and network
- `tests/` — unit tests for masking, model, training metrics, and paper figures
- `runs/<run>/latest.pt` — inference-only checkpoint (weights, epoch, channel stats, architecture args)
- `runs/<run>/split.json`, `stats.json`, `config.json`, `jy_stats.json`
- `scripts/` — reproduction entry points
- `environment.yml`, `requirements.txt` — software stack used for the official run
- `LICENSE`, `THIRD_PARTY_LICENSES.md`

The checkpoint has optimizer state, W&B run IDs, Slurm metadata, user names, and
absolute filesystem paths removed. It is sufficient for evaluation and figures,
not for optimizer-level training resume.

## Data access

The official training set is a directory of VPIC-derived HDF5 files, **not**
shipped in this ZIP because of size.

Expected filenames:

```text
VPIC_PPPL_CSV_Data_official2500_beta0.2_none.h5
```

(and analogously for other beta values if used). The paper training run uses
`--betas 0.2` only.

HDF5 layout:

- `fields`: shape `(T, C, Nx, Nz)` with `C=4` channels `(Bx, By, Bz, Density)`
- spatial grid used in the paper: `Nx=154`, `Nz=62`
- physical `imshow` extent: `[zmin, zmax, xmin, xmax] = [-21, 21, -50, 50]` cm

Place the files in `data/hdf5/` (or pass `--h5-dir`). During review, request the
corpus through the anonymous OpenReview discussion so that author identity is
not revealed. A public snapshot will accompany the camera-ready release.

These fields were produced with the Vector Particle-In-Cell (VPIC) code.
Please cite Bowers et al. (2008) and respect the VPIC BSD-3-Clause license
(see `THIRD_PARTY_LICENSES.md`).

## Software environment

Official training used:

- Python 3.12.13
- PyTorch 2.13.0+cu130 (`torch.__version__`)
- CUDA 13.0 (`torch.version.cuda`)
- bundled NCCL 2.29.7, cuDNN 9.20.0
- NumPy 2.4.4, h5py 3.13.0
- 4 nodes × 4 NVIDIA A100 GPUs (16 GPUs), PyTorch DDP, NCCL, AMP, `torchrun`

Install from `environment.yml` or `requirements.txt`. GPU-enabled PyTorch with
CUDA 13 is required for the 16-GPU training recipe; evaluation can run on a
single GPU.

## Reproduce training

From the supplement root, with HDF5 available:

```bash
export H5_DIR=/path/to/hdf5
bash scripts/train_16gpu.sh
```

This launches the official 4500-epoch recipe: AdamW `2e-4`, 10-epoch warmup,
cosine to `2e-6`, batch 8/GPU (global 128), AMP, attention, spatial-only
pooling, independent B/Density five-mask sampler with log-uniform exact counts
(`p_zero=p_full=0.05`). Logging to W&B is disabled. Validation runs after every
epoch on all five standardized ~50% patterns.

## Reproduce evaluation and paper figures

```bash
export H5_DIR=/path/to/hdf5
bash scripts/evaluate_paper.sh
```

This scores `latest.pt` on the five standardized validation masks and redraws
the paper figures. Pass `--force-recompute` only if you need to ignore caches.

## Tests (no HDF5 required for most)

```bash
python -m pytest tests -q
```
'''

LICENSE = r'''MIT License

Copyright (c) 2026 Anonymous Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

THIRD_PARTY = r'''# Third-party licenses and attribution

## VPIC (Vector Particle-In-Cell)

The plasma fields used in this work were generated with VPIC, a first-principles
3D electromagnetic relativistic kinetic particle-in-cell code.

Please cite:

> K. J. Bowers, B. J. Albright, L. Yin, B. Bergen, and T. J. T. Kwan,
> "Ultrahigh performance three-dimensional electromagnetic relativistic
> kinetic plasma simulation," *Physics of Plasmas* **15**, 055703 (2008).
> https://doi.org/10.1063/1.2840133

VPIC is distributed under the BSD 3-Clause License. The following is the
standard BSD-3-Clause text as used by the VPIC open-source distribution
(Los Alamos National Laboratory / Triad National Security, LLC). This
supplement does not include VPIC source code; it only uses simulation output.

```
BSD 3-Clause License

Copyright (c) Los Alamos National Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Other Python dependencies

PyTorch, NumPy, h5py, and Matplotlib are used under their respective open-source
licenses. They are not redistributed in this archive.
'''

ENVIRONMENT = r'''name: masked-recon-anonymous
channels:
  - conda-forge
dependencies:
  - python=3.12.13
  - numpy=2.4.4
  - h5py=3.13.0
  - matplotlib
  - tqdm
  - pytest
  - pip
  - pip:
      - torch==2.13.0
'''

REQUIREMENTS = r'''# Official training stack (PyTorch 2.13.0+cu130, CUDA 13.0).
torch==2.13.0
numpy==2.4.4
h5py==3.13.0
matplotlib>=3.8
tqdm
pytest
'''

TRAIN_SCRIPT = rf'''#!/bin/bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")/.." && pwd)"
cd "${{ROOT}}"
H5_DIR="${{H5_DIR:-${{ROOT}}/data/hdf5}}"
OUT_DIR="${{OUT_DIR:-${{ROOT}}/runs/{RUN_NAME}}}"
NNODES="${{NNODES:-4}}"
NPROC="${{NPROC_PER_NODE:-4}}"
MASTER_ADDR="${{MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${{MASTER_PORT:-29500}}"

python -m torch.distributed.run \
    --nnodes="${{NNODES}}" \
    --nproc-per-node="${{NPROC}}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${{MASTER_ADDR}}:${{MASTER_PORT}}" \
    train_masked_unet3d.py \
        --h5-dir "${{H5_DIR}}" \
        --betas 0.2 \
        --delta-t 24 \
        --stride-t 2 \
        --batch-size 8 \
        --num-workers 4 \
        --epochs 4500 \
        --lr 2e-4 \
        --warmup-epochs 10 \
        --min-lr 2e-6 \
        --weight-decay 1e-4 \
        --density-visible-p-zero 0.05 \
        --density-visible-p-full 0.05 \
        --magnetic-visible-p-zero 0.05 \
        --magnetic-visible-p-full 0.05 \
        --base-channels 24 \
        --channel-mults 1 2 4 8 \
        --use-attention \
        --spatial-only-pooling \
        --amp \
        --mask-patterns spatial_random 1 spatial_grid 1 spatial_block 1 temporal_random 1 temporal_block 1 \
        --out-dir "${{OUT_DIR}}" \
        --no-wandb \
        "$@"
'''

EVAL_SCRIPT = rf'''#!/bin/bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")/.." && pwd)"
cd "${{ROOT}}"
H5_DIR="${{H5_DIR:-${{ROOT}}/data/hdf5}}"
RUN_DIR="${{RUN_DIR:-${{ROOT}}/runs/{RUN_NAME}}}"

python make_paper_figures.py \
    --run-dir "${{RUN_DIR}}" \
    --checkpoint latest.pt \
    --h5-dir "${{H5_DIR}}" \
    --sliding-max-runs 25

python evaluate_standardized_validation.py \
    --run-dir "${{RUN_DIR}}" \
    --checkpoint latest.pt \
    --h5-dir "${{H5_DIR}}"
'''

HDF5_README = r'''# HDF5 corpus (not included)

Put official `VPIC_PPPL_CSV_Data_official2500_beta*_none.h5` files in this
directory, or set `H5_DIR` / `--h5-dir` to their location.

The paper training run uses the beta=0.2 file only. Request the files through
anonymous OpenReview correspondence during review.
'''


def sanitize_source(rel: str, text: str) -> str:
    if rel == "train_masked_unet3d.py":
        text = text.replace('p.set_defaults(wandb=True)', 'p.set_defaults(wandb=False)')
        text = text.replace('default="ai4plasma"', 'default=""')
        text = text.replace(
            'Enable W&B tracking (default).',
            'Enable W&B tracking.',
        )
        text = text.replace('default="online"', 'default="disabled"')
    if rel == "data/vpic_hdf5_dataset.py":
        text = text.replace("# ai4plasma/data/vpic_hdf5_dataset.py", "# data/vpic_hdf5_dataset.py")
    return text


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def copy_sources() -> None:
    for rel in SOURCE_FILES:
        src = ROOT / rel
        dst = RELEASE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(sanitize_source(rel, src.read_text()))


def sanitize_checkpoint() -> None:
    src = SRC_RUN / "latest.pt"
    dst_dir = RELEASE / "runs" / RUN_NAME
    dst_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    args = dict(ckpt["args"])
    args["h5_dir"] = "data/hdf5"
    args["out_dir"] = f"runs/{RUN_NAME}"
    args["wandb"] = False
    args["wandb_mode"] = "disabled"
    args["wandb_project"] = ""
    args["wandb_entity"] = None
    args["wandb_name"] = None
    args["wandb_group"] = None
    args["wandb_tags"] = None
    args["wandb_log_model"] = False
    args["auto_resume"] = False
    args["init_checkpoint"] = None
    clean = {
        "model": ckpt["model"],
        "epoch": int(ckpt["epoch"]),
        "best_val_mse": float(ckpt["best_val_mse"]),
        "stats": ckpt["stats"],
        "args": args,
        "masking_version": args.get("masking_version"),
        "model_version": args.get("model_version"),
    }
    torch.save(clean, dst_dir / "latest.pt")

    split = json.loads((SRC_RUN / "split.json").read_text())
    (dst_dir / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    stats = json.loads((SRC_RUN / "stats.json").read_text())
    (dst_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    config = json.loads((SRC_RUN / "config.json").read_text())
    config["h5_dir"] = "data/hdf5"
    config["out_dir"] = f"runs/{RUN_NAME}"
    config["wandb"] = False
    config["wandb_mode"] = "disabled"
    config["wandb_project"] = ""
    config["wandb_entity"] = None
    config["wandb_name"] = None
    config["wandb_group"] = None
    config["wandb_tags"] = None
    config["wandb_log_model"] = False
    config["auto_resume"] = False
    config["init_checkpoint"] = None
    (dst_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    jy = json.loads((SRC_RUN / "jy_stats.json").read_text())
    jy["source_checkpoint"] = "latest.pt"
    jy["source_checkpoint_path"] = f"runs/{RUN_NAME}/latest.pt"
    (dst_dir / "jy_stats.json").write_text(json.dumps(jy, indent=2) + "\n")


def scan_bytes(path: Path, rel: str) -> list[str]:
    data = path.read_bytes()
    lowered = data.lower()
    hits = []
    byte_terms = [
        b"binxia",
        b"xiabin",
        b"ai4xia",
        b"gatech",
        b"georgia tech",
        b"bjcpdayh",
        b"/pscratch/sd/b/",
        b"wandb.ai",
        b"ai4plasma",
        b"bin xia",
    ]
    for term in byte_terms:
        start = 0
        while True:
            idx = lowered.find(term, start)
            if idx < 0:
                break
            ctx = data[max(0, idx - 20) : idx + len(term) + 20]
            hits.append(f"{rel}:byte[{idx}]: [{term.decode()}] {ctx!r}"[:240])
            start = idx + len(term)
    return hits


def scan_release() -> list[str]:
    hits: list[str] = []
    text_skip = {".png", ".pdf", ".npz", ".pkl", ".zip"}
    for path in RELEASE.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(RELEASE))
        if path.suffix.lower() == ".pt":
            hits.extend(scan_bytes(path, rel))
            continue
        if path.suffix.lower() in text_skip:
            continue
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        for label, pattern in SCAN_PATTERNS:
            for match in pattern.finditer(text):
                line_no = text[: match.start()].count("\n") + 1
                line = text.splitlines()[line_no - 1].strip()
                hits.append(f"{rel}:{line_no}: [{label}] {line[:200]}")
    return hits


def main() -> int:
    if RELEASE.exists():
        shutil.rmtree(RELEASE)
    RELEASE.mkdir(parents=True)
    copy_sources()
    sanitize_checkpoint()
    (RELEASE / "README.md").write_text(README)
    (RELEASE / "LICENSE").write_text(LICENSE)
    (RELEASE / "THIRD_PARTY_LICENSES.md").write_text(THIRD_PARTY)
    (RELEASE / "environment.yml").write_text(ENVIRONMENT)
    (RELEASE / "requirements.txt").write_text(REQUIREMENTS)
    scripts = RELEASE / "scripts"
    scripts.mkdir()
    write_executable(scripts / "train_16gpu.sh", TRAIN_SCRIPT)
    write_executable(scripts / "evaluate_paper.sh", EVAL_SCRIPT)
    hdf5 = RELEASE / "data" / "hdf5"
    hdf5.mkdir(parents=True, exist_ok=True)
    (hdf5 / "README.md").write_text(HDF5_README)

    hits = scan_release()
    report = RELEASE.parent / "anonymous_release_scan_report.txt"
    report.write_text(
        "Scan of "
        + str(RELEASE)
        + "\n"
        + (f"{len(hits)} match(es)\n" if hits else "0 matches\n")
        + "\n".join(hits)
        + "\n"
    )
    print(report.read_text())
    if hits:
        print("Refusing to zip while scan matches remain.", file=sys.stderr)
        return 2

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    subprocess.check_call(
        ["zip", "-r", "-q", str(ZIP_PATH), RELEASE_NAME],
        cwd=str(ROOT),
    )
    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    ckpt_mb = (RELEASE / "runs" / RUN_NAME / "latest.pt").stat().st_size / (1024 * 1024)
    print(f"checkpoint_mb={ckpt_mb:.2f}")
    print(f"zip={ZIP_PATH} size_mb={size_mb:.2f}")
    if size_mb >= 100:
        print("ZIP exceeds 100 MB.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
