#!/bin/bash

# Submit the job through DINOv3's Submitit launcher. Run this script from a
# login node with: bash src/run_ssl_finetuning.sh

module load miniconda3
. ~/conda_init
conda activate dinov3

PROJECT_ROOT=/common/ganesanv/tlab
DINOV3_ROOT=${PROJECT_ROOT}/opt/dinov3
export PYTHONPATH="${DINOV3_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DINOV3_ROOT}" || exit 1

python -m dinov3.run.submit \
  --nodes 1 \
  --ngpus 2 \
  --timeout 2880 \
  --mem-gb 128 \
  --cpus-per-task 4 \
  --slurm-partition gpu \
  --slurm-qos "" \
  --output-dir "${PROJECT_ROOT}/runs/ssl_finetuning/adni_vitb16/%j" \
  "${PROJECT_ROOT}/src/ssl_finetuning/train.py" \
  --config-file "${PROJECT_ROOT}/src/ssl_finetuning/config.yaml"
