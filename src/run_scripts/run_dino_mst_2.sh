#!/bin/bash

#SBATCH --job-name=cp_mst
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time 48:00:00        
#SBATCH --error=run_classification_3-cls-bd.err  ## error log file
#SBATCH --output=run_classification_3-cls-bd.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src

python -m classification.run --params-file /common/ganesanv/tlab/runs/inf_tests/50patients_braindino.yaml
