#!/bin/bash

#SBATCH --job-name=convert_cq500
#SBATCH -p defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --error=convert_cq500.err
#SBATCH --output=convert_cq500.out
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

set -euo pipefail

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src

python -m datasets.cq500.convert \
  --root /common/ganesanv/tlab/data/cq500 \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --fail-on-error
