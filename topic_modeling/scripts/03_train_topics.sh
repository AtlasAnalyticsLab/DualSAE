#!/usr/bin/env bash
# Train the SAEs of one run (configs/runs.json), interpret them and build topics for every K.
# Usage: bash scripts/03_train_topics.sh RUN [extra tm_driver.py args, e.g. --encoders clip --seeds 42]
# Runs: imagenet_paired imagenet_paired_span001 imagenet_warmup_sweep
#       openimages_paired openimages_warmup_sweep openimages_jr_warmup500
# Sequential and resumable (finished stages are skipped). GPU.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
RUN="${1:?run name}"; shift
"$PY" "$TM_DIR/src/tm_driver.py" grid --run "$RUN" "$@"
