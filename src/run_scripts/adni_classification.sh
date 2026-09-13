#!/bin/bash

#SBATCH --job-name=adni
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time 72:00:00        
#SBATCH --array=0-9
#SBATCH --error=adni_%a.err  ## error log file
#SBATCH --output=adni_%a.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL


encs=("dinov3")
folds=("0" "1" "2" "3" "4")
tasks=("cn_ad" "cn_mci")

enc="dinov3"
fold=${folds[$(((SLURM_ARRAY_TASK_ID / 2)))]}
task=${tasks[$((SLURM_ARRAY_TASK_ID % 2))]}

weights=/common/ganesanv/tlab/runs/continued_pretraining_fixed/adni/$fold/ckpt/10999


module load miniconda3 
. ~/conda_init
conda activate dinov3

classification_dir=/common/ganesanv/tlab/classification/adni/$task/explora/$fold

cd /common/ganesanv/tlab/src

mkdir -p "$classification_dir"
python -m classification.run \
    --params-file /common/ganesanv/tlab/classification/adni/classification.yaml \
    --encoder $enc \
    --fold "$fold" \
    --checkpoint-dir "$classification_dir" \
    --adni-task $task \
    --weights $weights
