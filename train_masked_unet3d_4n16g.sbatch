#!/bin/bash

#SBATCH --job-name=ai4plasma-unet3d
#SBATCH --account=cusp
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/pscratch/sd/b/binxia/ai4plasma/logs/slurm-%x-%j.out
#SBATCH --error=/pscratch/sd/b/binxia/ai4plasma/logs/slurm-%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this file with: sbatch $0" >&2
    exit 2
fi

exec /pscratch/sd/b/binxia/ai4plasma/run_train_masked_unet3d_in_salloc.sh "$@"
