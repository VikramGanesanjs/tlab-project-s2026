#!/bin/bash


module load miniconda3
. ~/conda_init
conda activate dinov3


cd /common/ganesanv/tlab

n_gpus=2
mem=$((n_gpus * 64))


PYTHONPATH=${PWD}/opt/dinov3:${PWD}/src \
python -m dinov3.run.submit \
  ${PWD}/src/continued_pretraining/train.py \
  --nodes 1 \
  --ngpus $n_gpus \
  --timeout 2880 \
  --mem-gb $mem \
  --cpus-per-task 4 \
  --slurm-partition gpu \
  --slurm-qos "" \
  --config-file /common/ganesanv/tlab/runs/continued_pretraining_fixed/breastdm.yaml \
  --output-dir /common/ganesanv/tlab/runs/continued_pretraining_fixed/breastdm/4 \
  train.seed=4
