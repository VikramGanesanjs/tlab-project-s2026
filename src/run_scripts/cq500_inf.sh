#!/bin/bash

#SBATCH --job-name=cq500_inf
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time 72:00:00        
#SBATCH --array=0-4
#SBATCH --error=cq500_%a.err  ## error log file
#SBATCH --output=cq500_%a.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL


encs=("explora")
folds=("0" "1" "2" "3" "4")

enc="explora"
fold=${folds[$(((SLURM_ARRAY_TASK_ID) % 5))]}
task="ich"


weights=/common/ganesanv/tlab/runs/continued_pretraining_fixed/cq500/$fold/ckpt/6499


module load miniconda3 
. ~/conda_init
conda activate dinov3

classification_dir=/common/ganesanv/tlab/classification/cq500_fixed/$task/$enc/$fold

cd /common/ganesanv/tlab/src

mkdir -p "$classification_dir"

if [[ -f "$classification_dir/best_mst.pt" ]]; then
    echo "Skipping $enc fold $fold checkpoint: classification is complete"
    exit 0
fi

if [[ "$enc" == "explora" ]]; then 
    python -m classification.run \
        --params-file /common/ganesanv/tlab/classification/cq500/classification.yaml \
        --encoder dinov3 \
        --fold "$fold" \
        --checkpoint-dir "$classification_dir" \
        --cq500-task $task \
        --weights $weights
else
    python -m classification.run \
        --params-file /common/ganesanv/tlab/classification/cq500/classification.yaml \
        --encoder $enc \
        --fold "$fold" \
        --checkpoint-dir "$classification_dir" \
        --cq500-task $task

fi