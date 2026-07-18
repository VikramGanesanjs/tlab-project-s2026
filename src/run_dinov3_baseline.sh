#!/bin/bash

#SBATCH --job-name=run_dinov3_baseline
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --time 48:00:00        
#SBATCH --error=run_dinov3_baseline.err  ## error log file
#SBATCH --output=run_dinov3_baseline.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL


module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src

python dinov3_baseline.py \
  --data-root /common/ganesanv/tlab/data/tcia/duke_breast_cancer_processed \
  --scan pre \
  --z-min 0.25 \
  --z-max 0.75 \
  --epochs 10 \
  --batch-size 32