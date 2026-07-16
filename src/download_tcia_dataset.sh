#!/bin/bash

#SBATCH --job-name=download_tcia_dataset
#SBATCH -p defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time 48:00:00        
#SBATCH --error=download_tcia_dataset.err  ## error log file
#SBATCH --output=download_tcia_dataset.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL

COLLECTION_ID="duke_breast_cancer_mri"
DOWNLOAD_DIR="/common/ganesanv/tlab/data/tcia"

mkdir -p $DOWNLOAD_DIR

module load miniconda3
. ~/conda_init
conda activate data

cd /common/ganesanv/tlab/src
python download_tcia_dataset.py --collection_id $COLLECTION_ID --download_dir $DOWNLOAD_DIR