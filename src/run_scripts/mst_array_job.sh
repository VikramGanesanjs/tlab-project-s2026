#!/bin/bash

#SBATCH --job-name=adni_mst_tests
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=80G
#SBATCH --time 72:00:00        
#SBATCH --array=0    
#SBATCH --error=adni_mst_tests_%a.err  ## error log file
#SBATCH --output=adni_mst_tests_%a.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src


files=(../runs/mst_configs/adni/*.yaml)
file=${files[$SLURM_ARRAY_TASK_ID]}

python -m classification.run --params-file $file
