#!/bin/bash
# prepare_assets.sh -- the ONLY network step of the steering pipeline.  Run it once on a machine
# with internet; every other script runs fully offline afterwards.
#
#   export LANGDUAL_ROOT=/path/to/language/data      # same root as the language pipeline
#   bash prepare_assets.sh                           # pythia, bge, phi-4, AxBench judge files,
#                                                    # Amazon / AG News / GoEmotions snapshots
#   bash prepare_assets.sh --with-second-judge       # + InternVL3.5-8B (optional s8j)
#
# Idempotent (a sentinel per snapshot).  See prepare_assets.py for the pinned revisions.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
"$PY" "$HERE/prepare_assets.py" "$@"
