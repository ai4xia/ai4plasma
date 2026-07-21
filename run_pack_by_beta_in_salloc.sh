#!/bin/bash

set -u

cd /pscratch/sd/b/binxia/ai4plasma

mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

BETAS=(0.01 0.02 0.04 0.06 0.08 0.1 0.12 0.15 0.18 0.2)

OUT_DIR=/pscratch/sd/b/binxia/VPIC_PPPL_HDF5_by_beta_official2500_lzf

MAX_PARALLEL=4

echo "Start time: $(date)"
echo "Running inside allocation: ${SLURM_JOB_ID:-none}"
echo "Output dir: ${OUT_DIR}"

for BETA in "${BETAS[@]}"; do
    echo "Launching beta=${BETA} at $(date)"

    srun \
      --exclusive \
      -N 1 \
      -n 1 \
      -c 32 \
      python pack_csvdata_by_beta_fast.py \
        --runs-dir /pscratch/sd/d/dcfy/VPIC_PPPL/runs \
        --out-dir "${OUT_DIR}" \
        --beta "${BETA}" \
        --compression none \
        --workers 24 \
        --batch-frames 4 \
      > "logs/pack_beta${BETA}.out" \
      2> "logs/pack_beta${BETA}.err" &

    while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
        sleep 30
    done
done

wait

echo "End time: $(date)"
echo "All beta jobs finished."
