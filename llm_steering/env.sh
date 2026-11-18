#!/bin/bash
# env.sh -- runtime environment of every steering stage (sourced by run_stage.sh / run_all.sh).
# Everything runs OFFLINE: models and data must already be on disk (prepare_assets.sh).
#
# Set before sourcing (or export in your shell):
#   LANGDUAL_ROOT   data root of the language pipeline (SAE run dirs, residual shards, tokens,
#                   corpora, LMDBs, raw/ snapshots).  Default: <repo>/language/outputs/language
#   A16_ROOT        output root of the steering pipeline.  Default: ./outputs/llm_steering
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LANGDUAL_ROOT="${LANGDUAL_ROOT:-$(cd "$HERE/.." && pwd)/language/outputs/language}"
export A16_ROOT="${A16_ROOT:-$PWD/outputs/llm_steering}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# vLLM: in-process engine.  Must be set BEFORE vllm is imported (common/llm_backend.py also
# setdefaults it and refuses any other value); the V1 multiprocessing EngineCore can deadlock at
# bootstrap.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python3}"
