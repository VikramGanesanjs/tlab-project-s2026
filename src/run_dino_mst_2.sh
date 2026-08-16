#!/bin/bash

#SBATCH --job-name=cp_mst
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time 48:00:00        
#SBATCH --error=run_dino_mst_25pts_bd.err  ## error log file
#SBATCH --output=run_dino_mst_25pts_bd.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src

python dino_mst.py --params-file /common/ganesanv/tlab/runs/adni_benchmark/25patients_braindino.yaml