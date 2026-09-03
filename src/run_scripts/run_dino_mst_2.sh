#!/bin/bash

#SBATCH --job-name=base_patch
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time 48:00:00        
#SBATCH --error=fc_2.err  ## error log file
#SBATCH --output=fc_2.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src

python -m classification.run --params-file /common/ganesanv/tlab/runs/ssl_finetune_ablations/misc_experiments/full_context_mask_0.7/classification.yaml