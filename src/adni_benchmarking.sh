#!/bin/bash

#SBATCH --job-name=adni_mst_tests
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=80G
#SBATCH --time 72:00:00        
#SBATCH --array=1-5 
#SBATCH --error=adni_best_ckpt_%a.err  ## error log file
#SBATCH --output=adni_best_ckpt_%a.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src


TASK=25patients_braindino
BASE_CONFIG_PATH=/common/ganesanv/tlab/runs/adni_benchmark/$TASK.yaml
RUNS_DIR=/common/ganesanv/tlab/runs/adni_benchmark/$TASK

seed=543187

seed_used=$(($seed + $SLURM_ARRAY_TASK_ID))

run_name="run$SLURM_ARRAY_TASK_ID"

ckpt_dir="$RUNS_DIR/$run_name"

mkdir $ckpt_dir




python dino_mst.py --params-file $BASE_CONFIG_PATH --seed $seed_used --run-name $run_name --checkpoint-dir $ckpt_dir 
