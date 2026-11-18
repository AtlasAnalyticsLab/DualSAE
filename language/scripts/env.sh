#!/bin/bash
# ============================================================================
# env.sh -- runtime environment shared by every language driver and worker (sourced).
#
# Everything here is overridable from the caller's environment:
#   PYTHON           interpreter (default: python3 on PATH)
#   LANGDUAL_ROOT    root for downloads, token streams, shards, runs, results (default
#                    <repo>/language/outputs/language, the same default langdual/config.py and
#                    the LLM-steering component use; resolved to an absolute path once here)
#   HF_HOME          Hugging Face cache (only the tokenizer/model loads from the local snapshot
#                    under $LANGDUAL_ROOT/raw are used; nothing is fetched at runtime)
# ============================================================================

# Idempotent: sourcing twice (driver -> worker) is a no-op.
if [ -n "${LD_ENV_READY:-}" ]; then
  return 0 2>/dev/null || exit 0
fi

_LD_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LANG_DIR="$(cd "$_LD_SCRIPTS/.." && pwd)"
export REPO_DIR="$(cd "$LANG_DIR/.." && pwd)"

export LD_PY="${LD_PY:-${PYTHON:-python3}}"
# langdual (this component) and the vision core must both be importable: the vision dataset
# resolver imports langdual.lmsae.resid_shards to read activation shards.
export PYTHONPATH="$LANG_DIR:$REPO_DIR/vision${PYTHONPATH:+:$PYTHONPATH}"

export LANGDUAL_ROOT="$(realpath -m "${LANGDUAL_ROOT:-$LANG_DIR/outputs/language}")"

# Runtime is fully offline: every model and dataset file is fetched by prepare_assets.sh.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

export LD_ENV_READY=1
