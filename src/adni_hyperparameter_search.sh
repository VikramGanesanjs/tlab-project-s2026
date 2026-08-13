#!/bin/bash

#SBATCH --job-name=adni_mst_tests
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=80G
#SBATCH --time 72:00:00        
#SBATCH --array=2,5,11,13  
#SBATCH --error=adni_gs_tests_%a.err  ## error log file
#SBATCH --output=adni_gs_tests_%a.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src

BASE_CONFIG_PATH=/common/ganesanv/tlab/runs/adni_gridsearch/base_config.yaml

lrs=("0.0001" "0.0002" "0.00005")
n_slices=("16" "64" "128")
stop_metrics=("auroc" "bce_loss")
seed=542

lr=${lrs[$SLURM_ARRAY_TASK_ID % 3]}
n_slice=${n_slices[($SLURM_ARRAY_TASK_ID / 3) % 3]}
stop_metric=${stop_metrics[($SLURM_ARRAY_TASK_ID / 9) % 2]}

seed_used=$(($seed + $SLURM_ARRAY_TASK_ID))

run_name="BDGS-lr-$lr-ns-$n_slice-sm-$stop_metric"

ckpt_dir="/common/ganesanv/tlab/runs/adni_gridsearch/$run_name"

mkdir $ckpt_dir




python dino_mst.py --params-file $BASE_CONFIG_PATH --lr $lr --n-slices $n_slice --early-stopping-metric $stop_metric --seed $seed_used --run-name $run_name --checkpoint-dir $ckpt_dir 
