#!/bin/bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
H5_DIR="${H5_DIR:-${ROOT}/data/hdf5}"
if ! ls "${H5_DIR}"/processed_vpic_beta*.h5 >/dev/null 2>&1; then
    echo "The processed VPIC corpus used in the paper is not distributed with this anonymous submission."
    echo "Place the HDF5 files in data/hdf5 or set H5_DIR / --h5-dir to that directory."
    exit 2
fi
OUT_DIR="${OUT_DIR:-${ROOT}/runs/paper_train}"
NNODES="${NNODES:-4}"
NPROC="${NPROC_PER_NODE:-4}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

python -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc-per-node="${NPROC}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    train_masked_unet3d.py \
        --h5-dir "${H5_DIR}" \
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
        --out-dir "${OUT_DIR}" \
        --no-wandb \
        "$@"
