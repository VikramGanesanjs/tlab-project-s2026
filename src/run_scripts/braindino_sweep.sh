#!/bin/bash

#SBATCH --job-name=braindino_sweep
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=80G
#SBATCH --time 72:00:00        
#SBATCH --array=0-29
#SBATCH --error=bd_sweep_%a.err  ## error log file
#SBATCH --output=bd_sweep_%a.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL


train_props=("0.1" "0.2" "0.4" "0.6" "0.8" "1.0")
folds=("0" "1" "2" "3" "4")


train_prop=${train_props[$((SLURM_ARRAY_TASK_ID / 5))]}
fold=${folds[$((SLURM_ARRAY_TASK_ID % 5))]}

module load miniconda3 
. ~/conda_init
conda activate dinov3

write_dir=/common/ganesanv/tlab/runs/braindino_adni_sweep/$train_prop/$fold

mkdir -p "$write_dir"


cd /common/ganesanv/tlab/src

python -m classification.run --params-file /common/ganesanv/tlab/runs/braindino_adni_sweep/base_config.yaml --train-ratio $train_prop --fold $fold --checkpoint-dir $write_dir
