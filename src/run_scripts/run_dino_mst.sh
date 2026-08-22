#!/bin/bash

#SBATCH --job-name=dinov3_mst_ssl_finetune
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --time 48:00:00        
#SBATCH --error=run_ssl_finetune_full_context.err  ## error log file
#SBATCH --output=run_ssl_finetune_full_context.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src

python -m classification.run --params-file /common/ganesanv/tlab/runs/adni_benchmark/ssl_finetune_best_ckpt.yaml
