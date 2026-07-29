#!/bin/bash

#SBATCH --job-name=dinov3_supcon
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --time 48:00:00        
#SBATCH --error=run_dino_mst_dino_supcon_adni.err  ## error log file
#SBATCH --output=run_dino_mst_dino_supcon_adni.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab


python src/supcon_finetune.py \
  --dataset adni \
  --adni-task cn_mci_ad \
  --encoder dinov3 \
  --lora-r 16 \
  --temperature 0.07 \
  --batch-size 64