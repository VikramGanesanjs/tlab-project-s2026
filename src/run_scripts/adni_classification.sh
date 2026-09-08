#!/bin/bash

#SBATCH --job-name=loss_fn_ablation
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time 72:00:00        
#SBATCH --array=0-19
#SBATCH --error=adni_%a.err  ## error log file
#SBATCH --output=adni_%a.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL


encs=("braindino" "dinov3")
folds=("0" "1" "2" "3" "4")
tasks=("cn_ad" "cn_mci")

enc=${encs[$((SLURM_ARRAY_TASK_ID / 10))]}
fold=${folds[$(((SLURM_ARRAY_TASK_ID / 2) % 5))]}
task=${tasks[$((SLURM_ARRAY_TASK_ID % 2))]}



module load miniconda3 
. ~/conda_init
conda activate dinov3

classification_dir=/common/ganesanv/tlab/classification/adni/$task/$enc/$fold

cd /common/ganesanv/tlab/src

mkdir -p "$classification_dir"
python -m classification.run \
    --params-file /common/ganesanv/tlab/classification/adni/classification.yaml \
    --encoder $enc \
    --fold "$fold" \
    --checkpoint-dir "$classification_dir" \
    --adni-task $task \
