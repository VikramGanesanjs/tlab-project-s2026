#!/bin/bash

#SBATCH --job-name=organmnist3d
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time 72:00:00        
#SBATCH --array=0-19
#SBATCH --error=organmnist_%a.err  ## error log file
#SBATCH --output=organmnist_%a.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

encs=("dinov3" "meddinov3" "explora" "ssl_finetune")
folds=("0" "1" "2" "3" "4")

enc=${encs[$(((SLURM_ARRAY_TASK_ID) / 5))]}
fold=${folds[$(((SLURM_ARRAY_TASK_ID) % 5))]}



module load miniconda3 
. ~/conda_init
conda activate dinov3

classification_dir=/common/ganesanv/tlab/classification/organmnist3d/$enc/$fold

cd /common/ganesanv/tlab/src

mkdir -p "$classification_dir"

if [[ -f "$classification_dir/best_mst.pt" ]]; then
    echo "Skipping $enc fold $fold checkpoint: classification is complete"
    exit 0
fi


if [[ $enc == "explora" ]]; then
    weights=/common/ganesanv/tlab/runs/continued_pretraining_fixed/organmnist3d/$fold/ckpt/3499
    python -m classification.run \
    --params-file /common/ganesanv/tlab/classification/organmnist3d/classification.yaml \
    --encoder dinov3 \
    --fold "$fold" \
    --checkpoint-dir "$classification_dir" \
    --weights $weights


elif [[ $enc == "ssl_finetune" ]]; then
    weights=/common/ganesanv/tlab/runs/ssl_finetune_organmnist/base_run/$fold/ckpt/1999
    python -m classification.run \
    --params-file /common/ganesanv/tlab/classification/organmnist3d/classification.yaml \
    --encoder dinov3 \
    --fold "$fold" \
    --checkpoint-dir "$classification_dir" \
    --weights $weights
else
    python -m classification.run \
    --params-file /common/ganesanv/tlab/classification/organmnist3d/classification.yaml \
    --encoder $enc \
    --fold "$fold" \
    --checkpoint-dir "$classification_dir" \
    --seed $SLURM_ARRAY_TASK_ID
fi


