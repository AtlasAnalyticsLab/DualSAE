#!/bin/bash
# ============================================================================
# slurm_template.sh -- generic batch wrapper for ONE language cell (train + evaluate).
#
# Fill in the placeholders for your cluster, then submit one job per tag, e.g.
#   for t in $(cd language && python -m langdual.jobs --list main --layers 12); do
#     sbatch --export=ALL,CELL_TAG=$t language/scripts/slurm_template.sh
#   done
# Both steps are skip-guarded, so resubmitting a finished tag costs seconds.  The reported runs
# used a 45-minute (TopK/BatchTopK) or 1-hour (JumpReLU) limit per training cell on one H100.
# ============================================================================
#SBATCH --job-name=lang_cell
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=<LOG_DIR>/%x_%j.out
set -euo pipefail
: "${CELL_TAG:?set CELL_TAG to a cell tag}"
: "${LANG_DIR:=$PWD/language}"     # submit from the repository root, or export LANG_DIR
bash "$LANG_DIR/train_cell.sh" "$CELL_TAG"
# shellcheck disable=SC1091
source "$LANG_DIR/scripts/env.sh"
(cd "$LANG_DIR" && "$LD_PY" -m langdual.lmsae.eval_trackb --tag "$CELL_TAG")
