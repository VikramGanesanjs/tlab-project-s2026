#!/bin/bash

#SBATCH --job-name=dinov3_mst
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=120G
#SBATCH --time 48:00:00        
#SBATCH --array=1-3    
#SBATCH --error=dino_mst_%a.err  ## error log file
#SBATCH --output=dino_mst_%a.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src

python dino_mst.py --params-file ../runs/mst_configs/run${SLURM_ARRAY_TASK_ID}.yaml