#!/bin/bash
# Generic SLURM array template for the resumable drivers. Fill in the placeholders for your
# cluster; one array task runs one outstanding item (the drivers skip finished work, so the
# array can simply be re-submitted until nothing is left).
#
#   sbatch --array=0-<N-1> slurm_template.sh python scripts/ablations/grid/run_grid.py --stage train --arms R0 S Z
#
#SBATCH --job-name=dualsae_abl
#SBATCH --account=<ACCOUNT>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=<HH:MM:SS>
#SBATCH --output=<LOG_DIR>/%x_%A_%a.out
set -euo pipefail
cd "<REPO>/vision"
export FEATURES_ROOT="<FEATURES_ROOT>" ABL_ROOT="<OUTPUT_ROOT>/ablations"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
"$@" --index "${SLURM_ARRAY_TASK_ID}"
