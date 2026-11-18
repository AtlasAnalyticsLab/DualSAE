#!/usr/bin/env bash
# ============================================================================
# run_grid.sh — CB-SAE and CB-DualSAE for every (dataset, encoder, seed) whose
# Matryoshka-SAE and Dual-Matryoshka-SAE base runs exist. Sequential and
# resumable: re-running continues where it stopped (every stage skip-guards
# on its own outputs). Run from the repository root, after
# cb_sae/prepare_assets.sh and after the base runs of vision/scripts/main/.
#
# Environment:
#   CKPT_ROOT     root holding the vision/main.py run directories (./checkpoints/main)
#   DUAL_CHOICE   JSON selecting the reported Dual-Matryoshka setting per
#                 (dataset, encoder) -- {"chosen": {"imagenet/clip":
#                 {"span_coef": .., "warmup_steps": ..}, ...}}. Default: the
#                 main grid's selection, ./outputs/main/selection/
#                 dual_matryoshka_winners.json (written by
#                 vision/scripts/main/pick_dual_winners.py), when it exists.
#   ALL_ARMS=1    also run the robustness arms (R=2048 and k_cb=64)
#   FEATURES_ROOT, CB_WORK_DIR, CB_VOCAB_DIR: see cb_sae/README.md
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
WORK="${CB_WORK_DIR:-./outputs/cb_sae}"
BASES="$WORK/bases.json"

CHOICE=()
[ -n "${DUAL_CHOICE:-}" ] && CHOICE=(--dual-choice "$DUAL_CHOICE")
"$PY" "$HERE/cb_pipeline.py" bases --ckpt-root "${CKPT_ROOT:-./checkpoints/main}" \
    "${CHOICE[@]}" --out "$BASES"

# Operating point of the CB rows in the main comparison: R=1024 retained
# latents, k_cb=5 active concepts, scalar concepts (b_cb=1).
ARMS=(--retain 1024 --k-cb 5 --b-cb 1)
if [ "${ALL_ARMS:-0}" = "1" ]; then
  # Robustness arms of the same runs: R in {1024, 2048} x k_cb in {5, 64}.
  # R=2048 is admitted only for the (dataset, encoder) whose bases all keep
  # >= 2048 live latents (pair-matched over both arms and all seeds).
  ARMS=(--retain 1024 2048 --k-cb 5 64 --b-cb 1)
fi
"$PY" "$HERE/cb_pipeline.py" grid --bases "$BASES" "${ARMS[@]}"

COLLECT=(--retain 1024 --k-cb 5)
[ "${ALL_ARMS:-0}" = "1" ] && COLLECT=(--retain 1024 2048 --k-cb 5 64)
"$PY" "$HERE/cb_pipeline.py" collect --bases "$BASES" "${COLLECT[@]}" \
    --out "$WORK/cb_long.csv"
