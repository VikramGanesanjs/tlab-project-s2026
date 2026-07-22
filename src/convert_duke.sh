#!/bin/bash

#SBATCH --job-name=convert_duke
#SBATCH -p defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time 48:00:00        
#SBATCH --error=convert_duke.err  ## error log file
#SBATCH --output=convert_duke.out ## output log file
#SBATCH --mail-user=Vikram.Ganesan@cshs.org
#SBATCH --mail-type=ALL


module load miniconda3
. ~/conda_init
conda activate dinov3

cd /common/ganesanv/tlab/src

RAW_ROOT=/common/ganesanv/tlab/data/tcia/duke_breast_cancer_mri
OUT_ROOT=/common/ganesanv/tlab/data/tcia/duke_breast_cancer_processed
CLINICAL_XLSX=/common/ganesanv/tlab/data/tcia/duke_breast_cancer_mri/Clinical_and_Other_Features.xlsx
MAPPING_XLSX=/common/ganesanv/tlab/data/tcia/duke_breast_cancer_mri/Breast-Cancer-MRI-filepath_filename-mapping.xlsx
SCAN_TYPES="pre T1"
python -m datasets.duke.convert convert --raw-root $RAW_ROOT --out-root $OUT_ROOT --clinical-xlsx $CLINICAL_XLSX --mapping-xlsx $MAPPING_XLSX --scan-types $SCAN_TYPES --split-breasts --workers 8