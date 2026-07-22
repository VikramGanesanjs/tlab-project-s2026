#!/bin/bash

#SBATCH --job-name=dinov3_mst
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --time 48:00:00        
#SBATCH --error=run_dino_mst.err  ## error log file
#SBATCH --output=run_dino_mst.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src

python dino_mst.py \
  --data-root /common/ganesanv/tlab/data/tcia/duke_breast_cancer_processed \
  --scan T1 \
  --n-slices 16 \
  --image-size 224 \
  --batch-size 16 \
  --epochs 100 \
  --min-epochs 70 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --encoder dinov3 \
  --features patch \
  --d-model 768 \
  --mst-depth 1 \
  --mst-heads 4 \
  --mst-ffn-dim 1024 \
  --mst-dropout 0.1 \
  --hidden-dim 256 \
  --augment \
  --val-frac 0.1 \
  --run-name dinov3_mst_t1_8slices