#!/bin/bash

#SBATCH --job-name=hf_download
#SBATCH -p defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time 48:00:00        
#SBATCH --error=hf_download.err  ## error log file
#SBATCH --output=hf_download.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL




module load miniconda3
. ~/conda_init
conda activate data

cd /common/ganesanv/tlab
hf download Bubenpo/BreastDividerDataset --repo-type dataset --include labels*/* --local-dir data/tcia/duke_breast_cancer_mri

