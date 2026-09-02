#!/bin/bash

#SBATCH --job-name=dinov3_breastdm
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=120G
#SBATCH --array=1-4
#SBATCH --time 48:00:00        
#SBATCH --error=dinov3_breastdm_%a.err  ## error log file
#SBATCH --output=dinov3_breastdm_%a.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src

python -m classification.run --params-file /common/ganesanv/tlab/runs/breastdm/base_config.yaml --checkpoint-dir /common/ganesanv/tlab/runs/breastdm/dinov3/$SLURM_ARRAY_TASK_ID
