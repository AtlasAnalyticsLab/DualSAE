#!/usr/bin/env bash
# Hypothesis-generation tables: every encoder x activation x dataset, one table per selection rule.
# Outputs: $HG_OUT/tables/hypogen_long.csv and hypogen_<dataset>_<rule>.md. CPU.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
"$PY" -m hypogen.build_results "$@"
