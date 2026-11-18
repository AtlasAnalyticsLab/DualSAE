#!/bin/bash
# slurm_template.sh -- generic batch wrapper for ONE steering unit (fill in the placeholders).
# Example: sbatch --array=0-80 scripts/slurm_template.sh s0
#SBATCH --job-name=steer
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=<HH:MM:SS>
#SBATCH --account=<ACCOUNT>
#SBATCH --output=<LOG_DIR>/%x_%A_%a.out
set -euo pipefail
cd "<REPO>/llm_steering"
export LANGDUAL_ROOT="<LANGDUAL_ROOT>" A16_ROOT="<A16_ROOT>"
bash run_stage.sh "$1" "${SLURM_ARRAY_TASK_ID:-0}"
