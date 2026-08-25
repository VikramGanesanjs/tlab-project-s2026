#!/bin/bash

# Submit all current runs with: sbatch src/run_scripts/eval_collapse_test_last.sh
# The current collapse_test directory contains 33 last_mst.pt files (11 checkpoints x 3 runs).
# Without SLURM_ARRAY_TASK_ID, this script evaluates every checkpoint serially.
#SBATCH --job-name=eval_collapse_last
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --array=0-32
#SBATCH --error=eval_collapse_last_%a.err
#SBATCH --output=eval_collapse_last_%a.out

set -euo pipefail

REPO_ROOT=/common/ganesanv/tlab
RUNS_DIR="$REPO_ROOT/runs/collapse_test"
SPLITS_FILE="$REPO_ROOT/runs/ssl_finetune_ablations/full_context/adni_patient_splits.json"

module load miniconda3
. ~/conda_init
conda activate dinov3

mapfile -t HEADS < <(find "$RUNS_DIR" -type f -name last_mst.pt | sort -V)
if (( ${#HEADS[@]} == 0 )); then
    echo "No last_mst.pt checkpoints found under $RUNS_DIR" >&2
    exit 1
fi

run_evaluation() {
    local head="$1"
    local output
    output="$(dirname "$head")/last_metrics_summary.json"
    echo "Evaluating $head"
    python -m classification.eval "$head" \
        --splits-file "$SPLITS_FILE" \
        --output "$output" \
        --batch-size 16 \
        --num-workers 2
}

cd "$REPO_ROOT/src"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if (( SLURM_ARRAY_TASK_ID >= ${#HEADS[@]} )); then
        echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID exceeds ${#HEADS[@]} checkpoints" >&2
        exit 1
    fi
    run_evaluation "${HEADS[$SLURM_ARRAY_TASK_ID]}"
else
    for head in "${HEADS[@]}"; do
        run_evaluation "$head"
    done
fi
