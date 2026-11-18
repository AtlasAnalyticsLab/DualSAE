#!/usr/bin/env bash
# ============================================================================
# run_cell.sh DATASET ENCODER SEED — CB-SAE and CB-DualSAE for one cell
# (both arms: convert -> dissect -> retain -> prune -> CB train -> finalize
# -> eval), at the operating point of the main comparison (R=1024, k_cb=5,
# b_cb=1). Extra arguments are forwarded to cb_pipeline.py (e.g.
# --retain 1024 2048 --k-cb 5 64 for the robustness arms, --stages ...).
#
# Requires $CB_WORK_DIR/bases.json (written by run_grid.sh or by
# `python cb_sae/scripts/cb_pipeline.py bases ...`).
#
# Note: the retain check here sees only this seed's two bases; run_grid.sh
# applies it over all seeds of the (dataset, encoder), as the reported rows do.
# ============================================================================
set -euo pipefail
[ $# -ge 3 ] || { echo "usage: $0 DATASET ENCODER SEED [cb_pipeline.py args]" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS="$1"; ENC="$2"; SEED="$3"; shift 3
WORK="${CB_WORK_DIR:-./outputs/cb_sae}"
"${PYTHON:-python}" "$HERE/cb_pipeline.py" cell --bases "$WORK/bases.json" \
    --dataset "$DS" --encoder "$ENC" --seed "$SEED" \
    --retain 1024 --k-cb 5 --b-cb 1 "$@"
