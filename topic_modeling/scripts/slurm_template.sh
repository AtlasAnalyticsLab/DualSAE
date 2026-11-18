#!/usr/bin/env bash
# Generic array-job template (fill in the placeholders for your scheduler). One task = one cell.
#SBATCH --job-name=tm_cell
#SBATCH --account=<ACCOUNT>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=<HH:MM:SS>
#SBATCH --array=0-<N-1>
# Submit with:  RUN=imagenet_paired sbatch --export=ALL scripts/slurm_template.sh
# where N = number of lines printed by:  python src/tm_driver.py list --run $RUN
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LINE=$("$PY" "$TM_DIR/src/tm_driver.py" list --run "$RUN" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
ENC=$(echo "$LINE" | cut -f1); SEG=$(echo "$LINE" | cut -f2); SEED=$(echo "$LINE" | cut -f3 | sed 's/seed//')
TR="${SEG%%/*}"; W=""; case "$SEG" in */warmup*) W="--warmup ${SEG##*/warmup}" ;; esac
"$PY" "$TM_DIR/src/tm_driver.py" cell --run "$RUN" --encoder "$ENC" --trainer "$TR" --seed "$SEED" $W
