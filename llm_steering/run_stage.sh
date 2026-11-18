#!/bin/bash
# run_stage.sh -- run ONE unit of one steering stage (the unit index contract is steering/jobs.py).
#
#   bash run_stage.sh s0 7          # s0 rescore of SAE unit 7 (python -m steering.jobs --kind s0 --list)
#   bash run_stage.sh s9            # one-unit stages take no index
#
# Every stage re-checks its own skip-guard, so re-running a finished unit exits in seconds.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
STAGE="${1:?usage: run_stage.sh <stage> [index]}"
IDX="${2:-0}"
cd "$HERE"
eval "$("$PY" -m steering.jobs --kind "$STAGE" --index "$IDX" --shell)"
echo "=== ${A16_KIND}[${A16_INDEX}] unit=${A16_UNIT} root=${A16_ROOT}"
start=$(date +%s)
# shellcheck disable=SC2086
"$PY" -m "$A16_MODULE" $A16_ARGS
echo "=== ${A16_KIND}[${A16_INDEX}] done in $(( $(date +%s) - start ))s"
