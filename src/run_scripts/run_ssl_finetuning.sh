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

n_gpus=4
mem=$((n_gpus * 64))

python -m dinov3.run.submit \
  --nodes 1 \
  --ngpus $n_gpus \
  --timeout 2880 \
  --mem-gb $mem \
  --cpus-per-task 4 \
  --slurm-partition gpu \
  --slurm-qos "" \
  --output-dir "${PROJECT_ROOT}/runs/ssl_gram_penalty/lora_r/8" \
  "${PROJECT_ROOT}/src/ssl_finetuning/train.py" \
  --config-file "/common/ganesanv/tlab/runs/ssl_gram_penalty/spatial_window.yaml" \
  train.fold=3 \
  lora.rank=8
  # cross_slice_gram_spatial_window_size=11
