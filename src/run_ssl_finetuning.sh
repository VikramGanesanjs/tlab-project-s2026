#!/bin/bash

#SBATCH --job-name=ssl_finetune_vitb16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --error=run_ssl_finetuning_vitb16.err
#SBATCH --output=run_ssl_finetuning_vitb16.out
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab

python src/ssl_finetuning/train.py \
  --config-file /common/ganesanv/tlab/src/ssl_finetuning/config.yaml \
  --output-dir /common/ganesanv/tlab/runs/ssl_finetuning/adni_vitb16
