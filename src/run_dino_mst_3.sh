#!/bin/bash

#SBATCH --job-name=dinov3_mst
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=120G
#SBATCH --time 48:00:00        
#SBATCH --error=run_dino_mst_dinov3_bench.err  ## error log file
#SBATCH --output=run_dino_mst_dinov3_bench.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src

python dino_mst.py --params-file /common/ganesanv/tlab/runs/adni_benchmark/25patients_higherlr.yaml