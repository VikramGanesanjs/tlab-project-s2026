#!/bin/bash

#SBATCH --job-name=loss_fn_ablation
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=80G
#SBATCH --time 72:00:00        
#SBATCH --array=0-39
#SBATCH --error=loss_fn_%a.err  ## error log file
#SBATCH --output=loss_fn_%a.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL


runs=("no_mask_uwsd_only" "no_masking" "only_uwsd" "spatial_window")
folds=("0" "1" "2" "3" "4")
ckpts=("4999" "9999")
task="cn_mci"

run=${runs[$((SLURM_ARRAY_TASK_ID / 10))]}
fold=${folds[$(((SLURM_ARRAY_TASK_ID / 2) % 5))]}
ckpt=${ckpts[$((SLURM_ARRAY_TASK_ID % 2))]}



module load miniconda3 
. ~/conda_init
conda activate dinov3

source_run_dir=/common/ganesanv/tlab/runs/ssl_gram_penalty/$run
classification_dir=/common/ganesanv/tlab/runs/loss_fn_comparison/$run
weights=$source_run_dir/$fold/ckpt/$ckpt
write_dir=$classification_dir/$fold/$task/$ckpt

if [[ -f "$write_dir/run_summary.json" ]]; then
    echo "Skipping $run fold $fold checkpoint $ckpt: classification is complete"
    exit 0
fi

if [[ ! -d "$weights" ]]; then
    echo "Missing weights for $run fold $fold checkpoint $ckpt: $weights" >&2
    exit 1
fi

cd /common/ganesanv/tlab/src

mkdir -p "$write_dir"
echo "Running $run fold $fold checkpoint $ckpt"
python -m classification.run \
    --params-file /common/ganesanv/tlab/runs/loss_fn_comparison/classification.yaml \
    --fold "$fold" \
    --checkpoint-dir "$write_dir" \
    --weights "$weights" \
    --adni-task $task
