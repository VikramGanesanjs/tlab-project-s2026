#!/bin/bash

# Submit the job through DINOv3's Submitit launcher. Run this script from a
# login node with: bash run_scripts/run_ssl_finetuning.sh

module load miniconda3
. ~/conda_init
conda activate dinov3

PROJECT_ROOT=/common/ganesanv/tlab
DINOV3_ROOT=${PROJECT_ROOT}/opt/dinov3
export PYTHONPATH="${DINOV3_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DINOV3_ROOT}" || exit 1

python -m dinov3.run.submit \
  --nodes 1 \
  --ngpus 4 \
  --timeout 2880 \
  --mem-gb 256 \
  --cpus-per-task 4 \
  --slurm-partition gpu \
  --slurm-qos "" \
  --output-dir "${PROJECT_ROOT}/runs/ssl_finetune_cq500/fixed_prepro/4" \
  "${PROJECT_ROOT}/src/ssl_finetuning/train.py" \
  --config-file "/common/ganesanv/tlab/runs/ssl_finetune_cq500/base_config.yaml" \
  train.fold=4
  # train.fold=4
