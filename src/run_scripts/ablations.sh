#!/bin/bash

#SBATCH --job-name=ablations
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time 72:00:00        
#SBATCH --array=0-23
#SBATCH --error=ablations_%a.err  ## error log file
#SBATCH --output=ablations_%a.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

ablations=("lora_r" "window_size" "distance")
tasks=("cn_ad" "cn_mci")
fold=3
module load miniconda3 
. ~/conda_init
conda activate dinov3

ablation=${ablations[$((SLURM_ARRAY_TASK_ID / 8))]}

if [[ $ablation == "distance" ]]; then
    vals=("1" "3" "5" "10")
elif [[ $ablation == "lora_r" ]]; then
    vals=("2" "4" "8" "32")
elif [[ $ablation == "window_size" ]]; then
    vals=("3" "7" "9" "11")
fi


val=${vals[$(((SLURM_ARRAY_TASK_ID / 2) % 4))]}
task=${tasks[$((SLURM_ARRAY_TASK_ID % 2))]}
classification_dir=/common/ganesanv/tlab/ablations/adni/$ablation/$val/$task/4999
config_path=/common/ganesanv/tlab/classification/adni/classification.yaml

cd /common/ganesanv/tlab/src

mkdir -p "$classification_dir"

if [[ -f "$classification_dir/best_mst.pt" ]]; then
    echo "Skipping"
    exit 0
fi
weights=/common/ganesanv/tlab/runs/ssl_gram_penalty/$ablation/$val/ckpt/4999
python -m classification.run \
    --params-file $config_path \
    --encoder dinov3 \
    --fold "$fold" \
    --checkpoint-dir "$classification_dir" \
    --weights $weights \
    --adni-task $task


