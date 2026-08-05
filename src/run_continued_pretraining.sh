#!/bin/bash


module load miniconda3
. ~/conda_init
conda activate dinov3


cd /common/ganesanv/tlab

PYTHONPATH=${PWD}/opt/dinov3:${PWD}/src \
python -m dinov3.run.submit \
  ${PWD}/src/continued_pretraining/train.py \
  --nodes 1 \
  --ngpus 2 \
  --timeout 2880 \
  --mem-gb 256 \
  --cpus-per-task 4 \
  --slurm-partition gpu \
  --slurm-qos "" \
  --config-file ${PWD}/src/continued_pretraining/config.yaml \
  --output-dir ${PWD}/runs/continued_pretraining/adni-unfreeze-last-2-r-16 \
  train.dataset_path=ADNI:root=${PWD}/data/ADNI \
  lora_r=8
