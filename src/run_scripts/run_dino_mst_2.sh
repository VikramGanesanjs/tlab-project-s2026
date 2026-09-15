#!/bin/bash

#SBATCH --job-name=base_patch
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time 48:00:00        
#SBATCH --error=lldmmri_1.err  ## error log file
#SBATCH --output=lldmmri_1.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src

python -m classification.run --params-file /common/ganesanv/tlab/classification/lld-mmri/classification.yaml \
    --checkpoint-dir /common/ganesanv/tlab/classification/lld-mmri/dinov3/1 \
    --fold 1 \
    # --weights /common/ganesanv/tlab/runs/continued_pretraining_fixed/breastdm/0/ckpt/2999 \
