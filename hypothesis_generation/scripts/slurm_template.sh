#!/usr/bin/env bash
# Generic array-job template (fill in the placeholders). One task = one cell (54 cells per dataset).
#SBATCH --job-name=hg_cell
#SBATCH --account=<ACCOUNT>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=<HH:MM:SS>
#SBATCH --array=0-53
# Submit with:  DS=imagenet RULE=lasso sbatch --export=ALL scripts/slurm_template.sh   (after 01_stage1.sh)
set -euo pipefail
bash "$(dirname "${BASH_SOURCE[0]}")/02_cells.sh" "$DS" "$RULE" "$SLURM_ARRAY_TASK_ID"
