#!/bin/bash

#SBATCH --job-name=adni_mst_tests
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=80G
#SBATCH --time 72:00:00        
#SBATCH --array=0-35
#SBATCH --error=collapse_test_%a.err  ## error log file
#SBATCH --output=collapse_test_%a.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

CKPT_PARENT_DIR=/common/ganesanv/tlab/runs/ssl_finetune_ablations/base_run_no_embed/ckpt
CONFIG_PATH=/common/ganesanv/tlab/runs/adni_benchmark/ssl_finetune_best_ckpt.yaml
WRITE_DIR=/common/ganesanv/tlab/runs/collapse_test


CKPT_DIRS=($CKPT_PARENT_DIR/*/)

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src


DIR=${CKPT_DIRS[$((SLURM_ARRAY_TASK_ID / 3))]}
DIR=${DIR%/} 
RUN_ID=$((SLURM_ARRAY_TASK_ID % 3))

WRITE_DIR_SPEC=$WRITE_DIR/${DIR##*/}/run$RUN_ID

mkdir -p $WRITE_DIR_SPEC

python -m classification.run --params-file $CONFIG_PATH --weights $DIR --checkpoint-dir $WRITE_DIR_SPEC