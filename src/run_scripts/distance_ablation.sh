#!/bin/bash

#SBATCH --job-name=distance_ablation
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=80G
#SBATCH --time 72:00:00        
#SBATCH --array=0-3
#SBATCH --error=distance_ablation_%a.err  ## error log file
#SBATCH --output=distance_ablation_%a.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL


distances=("1" "3" "5" "10")
chosen_ckpts=("1999" "4999" "6999" "3999")



distance=${distances[$((SLURM_ARRAY_TASK_ID))]}

base_dir=/common/ganesanv/tlab/runs/ssl_finetune_ablations/distance/$distance

ckpt_idx=${chosen_ckpts[$((SLURM_ARRAY_TASK_ID))]}
ckpt_path=$base_dir/ckpt/$ckpt_idx

write_dir=/common/ganesanv/tlab/runs/ssl_finetune_ablations/distance/$distance/classification

mkdir -p "$write_dir"

module load miniconda3 
. ~/conda_init
conda activate dinov3


cd /common/ganesanv/tlab/src

python -m classification.run --params-file /common/ganesanv/tlab/runs/ssl_finetune_ablations/distance/classification.yaml --checkpoint-dir $write_dir --weights $ckpt_path
