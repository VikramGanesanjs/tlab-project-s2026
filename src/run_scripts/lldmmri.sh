#!/bin/bash

#SBATCH --job-name=lldmmri
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time 72:00:00        
#SBATCH --array=0-19
#SBATCH --error=lldmmri_%a.err  ## error log file
#SBATCH --output=lldmmri_%a.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL


encs=("dinov3" "ssl_finetune" "braindino" "explora")
folds=("0" "1" "2" "3" "4")
ckpt_ssl=1499

enc=${encs[$(((SLURM_ARRAY_TASK_ID) / 5))]}
fold=${folds[$(((SLURM_ARRAY_TASK_ID) % 5))]}
scan_type="all"




module load miniconda3 
. ~/conda_init
conda activate dinov3

classification_dir=/common/ganesanv/tlab/classification/lld-mmri/$enc/$fold

cd /common/ganesanv/tlab/src

mkdir -p "$classification_dir"

if [[ -f "$classification_dir/best_mst.pt" ]]; then
    echo "Skipping $enc fold $fold checkpoint: classification is complete"
    exit 0
fi

if [[ $enc == "ssl_finetune" ]]; then
    weights=/common/ganesanv/tlab/runs/ssl_lldmmri/base_run/$fold/ckpt/$ckpt_ssl
    python -m classification.run \
        --params-file /common/ganesanv/tlab/classification/lld-mmri/classification.yaml \
        --encoder dinov3 \
        --fold "$fold" \
        --checkpoint-dir "$classification_dir" \
        --scan $scan_type \
        --weights $weights
elif [[ $enc == "explora" ]]; then
    weights=$(find /common/ganesanv/tlab/runs/continued_pretraining_fixed/lldmmri/$fold/ckpt -maxdepth 1 -mindepth 1 -type d -printf "%f\n" | sort -n | tail -1)
    python -m classification.run \
            --params-file /common/ganesanv/tlab/classification/lld-mmri/classification.yaml \
            --encoder dinov3 \
            --fold "$fold" \
            --checkpoint-dir "$classification_dir" \
            --scan $scan_type \
            --weights $weights

elif [[ $enc == "braindino" ]]; then
    python -m classification.run \
            --params-file /common/ganesanv/tlab/classification/lld-mmri/classification.yaml \
            --encoder braindino \
            --fold "$fold" \
            --checkpoint-dir "$classification_dir" \
            --scan $scan_type \

else
    python -m classification.run \
            --params-file /common/ganesanv/tlab/classification/lld-mmri/classification.yaml \
            --encoder dinov3 \
            --fold "$fold" \
            --checkpoint-dir "$classification_dir" \
            --scan $scan_type \

fi