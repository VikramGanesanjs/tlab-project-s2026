#!/bin/bash

#SBATCH --job-name=cq500_cls_transformer
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --time 48:00:00        
#SBATCH --error=cq500_cls_transformer_fixed.err  ## error log file
#SBATCH --output=cq500_cls_transformer_fixed.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src

python -m classification.run --params-file /common/ganesanv/tlab/runs/cq500/transformer.yaml