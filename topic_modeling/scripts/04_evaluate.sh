#!/usr/bin/env bash
# SAE-TM's three default metrics (diversity, coherence, intruder) for every topic file of a run.
# Usage: bash scripts/04_evaluate.sh RUN [--encoders clip] [--trainer DualTopKTrainer] [--seed 42]
# Resumable (files with outputs are skipped). GPU (vLLM, microsoft/phi-4).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
RUN="${1:?run name}"; shift
"$PY" "$TM_DIR/src/tm_driver.py" evaluate --run "$RUN" "$@"
