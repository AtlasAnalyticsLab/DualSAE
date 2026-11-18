#!/usr/bin/env bash
# Topic-modeling tables from the evaluation outputs: every DualSAE configuration of configs/runs.json
# against its activation-matched baseline, every K, all three metrics. CPU.
# Outputs: $TM_OUT/tables/topic_table_long.csv and topic_table_<dataset>_<metric>.md
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
"$PY" "$TM_DIR/src/build_topic_table.py" "$@"
