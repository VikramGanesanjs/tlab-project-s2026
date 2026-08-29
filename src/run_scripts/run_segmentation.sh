#!/bin/bash

#SBATCH --job-name=nnUNet_segmentation
#SBATCH -p gpu
#SBATCH --gpus=l40s:2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=120G
#SBATCH --time 48:00:00        
#SBATCH --error=nnUNet_seg_higher_lr.err  ## error log file
#SBATCH --output=nnUNet_seg_higher_lr.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

module load miniconda3 
. ~/conda_init
conda activate nnunetv2


export nnUNet_raw="/common/ganesanv/tlab/data/nnUNet_raw"
export nnUNet_preprocessed="/common/ganesanv/tlab/data/nnUNet_preprocessed"
export nnUNet_results="/common/ganesanv/tlab/data/nnUNet_results"

NUM_GPUS=2


nnUNetv2_train 001 2d 1 -tr meddinov3_base_primus_multiscale_Trainer --npz -num_gpus $NUM_GPUS