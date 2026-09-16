#!/bin/bash

#SBATCH --job-name=spatial_window
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time 48:00:00        
#SBATCH --error=spatial_window.err  ## error log file
#SBATCH --output=spatial_windowd.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src

python -m classification.run --params-file /common/ganesanv/tlab/classification/cq500/classification.yaml \
    --weights /common/ganesanv/tlab/runs/ssl_finetune_cq500/better/4/ckpt/2999 \
    --encoder dinov3 \
    --checkpoint-dir /common/ganesanv/tlab/classification/cq500_fixed/ich/ssl_finetune_fixed/4 \
    --fold 4\

