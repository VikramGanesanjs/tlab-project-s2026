#!/bin/bash

#SBATCH --job-name=breastdm
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time 72:00:00        
#SBATCH --array=0-19
#SBATCH --error=breastdm_%a.err  ## error log file
#SBATCH --output=breastdm_%a.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

encs=("dinov3" "braindino" "explora" "ssl_finetune")
folds=("0" "1" "2" "3" "4")

enc=${encs[$(((SLURM_ARRAY_TASK_ID) / 5))]}
fold=${folds[$(((SLURM_ARRAY_TASK_ID) % 5))]}



module load miniconda3 
. ~/conda_init
conda activate dinov3

classification_dir=/common/ganesanv/tlab/classification/breastdm/$enc/$fold

cd /common/ganesanv/tlab/src

mkdir -p "$classification_dir"

if [[ -f "$classification_dir/best_mst.pt" ]]; then
    echo "Skipping $enc fold $fold checkpoint: classification is complete"
    exit 0
fi


if [[ $enc == "explora" ]]; then
    ckpt_num=$(find /common/ganesanv/tlab/runs/continued_pretraining_fixed/breastdm/$fold/ckpt -maxdepth 1 -mindepth 1 -type d -printf "%f\n" | sort -n | tail -1)
    weights=/common/ganesanv/tlab/runs/continued_pretraining_fixed/breastdm/$fold/ckpt/$ckpt_num
    python -m classification.run \
    --params-file /common/ganesanv/tlab/classification/breastdm/classification.yaml \
    --encoder dinov3 \
    --fold "$fold" \
    --checkpoint-dir "$classification_dir" \
    --weights $weights


elif [[ $enc == "ssl_finetune" ]]; then
    weights=/common/ganesanv/tlab/runs/ssl_breastdm/fixed_data/$fold/ckpt/999
    python -m classification.run \
    --params-file /common/ganesanv/tlab/classification/breastdm/classification.yaml \
    --encoder dinov3 \
    --fold "$fold" \
    --checkpoint-dir "$classification_dir" \
    --weights $weights
else
    python -m classification.run \
    --params-file /common/ganesanv/tlab/classification/breastdm/classification.yaml \
    --encoder $enc \
    --fold "$fold" \
    --checkpoint-dir "$classification_dir"
fi


