#!/bin/bash
# ============================================================================
# run_pipeline.sh -- sequential, resumable driver for the language experiments.
#
#   bash language/run_pipeline.sh --stage STAGE [--grid GRID] [filters] [--dry-run]
#
# STAGE   pack      pack the token streams of the selected corpora (CPU)
#         extract   residual-stream activations, one (corpus, layer, split) at a time (GPU)
#         train     train every cell of the grid (GPU), via train_cell.sh
#         eval      evaluate every cell of the grid (GPU): delta CE, KL, R^2, MonoScore, L0
#         all       pack -> extract -> train -> eval
#         textemb   materialise AG News + GoEmotions and embed them with bge-large-en-v1.5
#                   (corpora/ and lmdb/bge/; required by llm_steering/, independent of --grid)
# GRID    main      the main grid: {owt, ghcode} x blocks {6,12,18} x {TopK, BatchTopK,
#                   JumpReLU} x {standard SAE, DualSAE} x seeds {42,43,44}   (default)
#         warmup    the DualSAE warm-up ablation (axis warmupB): warm-up {0,1250,5000,10000}
#                   x the three activations x blocks {6,12,18}, OpenWebText, DualSAE only
#         steering  the extra SAEs the LLM-steering component loads (axes tkuntied,
#                   dg0B1250); trained here, evaluated by llm_steering/
# filters --layers "12" --corpora "owt" --activations "topk batch_topk" --seeds "42"
#         --values "1250" (ablation axis values)
#
# Every unit has a skip-guard (packed streams, shard manifests, CELL_DONE.json, complete
# result JSONs), so re-running the same command continues where a killed run stopped.
# Environment: see scripts/env.sh (PYTHON, LANGDUAL_ROOT).
# ============================================================================
set -euo pipefail

LANG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$LANG_DIR/scripts/env.sh"

STAGE="" ; GRID="main" ; LAYERS="" ; CORPORA="" ; ACTS="" ; SEEDS="" ; VALUES="" ; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --grid) GRID="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --corpora) CORPORA="$2"; shift 2 ;;
    --activations) ACTS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --values) VALUES="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n 2,27p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$STAGE" ] || { echo "pass --stage pack|extract|train|eval|all|textemb" >&2; exit 2; }

run() {   # echo, then execute unless --dry-run
  echo "+ $*"
  if [ "$DRY" = "0" ]; then "$@"; fi
}

py() { (cd "$LANG_DIR" && "$LD_PY" "$@"); }

# The ablation grids read the OpenWebText shards; the main grid reads both corpora.
case "$GRID" in
  main) DEF_CORPORA="owt ghcode" ;;
  warmup|steering) DEF_CORPORA="owt" ;;
  *) echo "unknown --grid $GRID (main|warmup|steering)" >&2; exit 2 ;;
esac
CORPORA="${CORPORA:-$DEF_CORPORA}"
LAYERS="${LAYERS:-6 12 18}"

cell_tags() {
  local args=()
  [ -n "$LAYERS" ] && args+=(--layers $LAYERS)
  [ -n "$ACTS" ] && args+=(--activations $ACTS)
  [ -n "$SEEDS" ] && args+=(--seeds $SEEDS)
  case "$GRID" in
    main)     py -m langdual.jobs --list main --corpora $CORPORA "${args[@]}" ;;
    warmup)   py -m langdual.jobs --list ablation --axes warmupB ${VALUES:+--values $VALUES} "${args[@]}" ;;
    steering) py -m langdual.jobs --list ablation --axes tkuntied dg0B1250 ${VALUES:+--values $VALUES} "${args[@]}" ;;
  esac
}

stage_pack() {
  for c in $CORPORA; do
    run py -m langdual.materialize_corpora pack --lm-corpus "$c"
  done
}

stage_extract() {
  # One (corpus, layer, split) per invocation; "train" first, because a held-out split
  # inherits the train split's norm factor from its manifest.
  for c in $CORPORA; do
    for l in $LAYERS; do
      for s in train val test; do
        local d
        d="$(py -c "from langdual import config; print(config.resid_dir($l, '$s', '$c'))")"
        if [ -f "$d/manifest.json" ]; then
          echo "[skip] shards complete: $d"
          continue
        fi
        run py -m langdual.lmsae.extract_resid --corpus "$c" --layers "$l" --split "$s"
        if [ "$DRY" = "0" ] && [ ! -f "$d/manifest.json" ]; then
          echo "FATAL: extraction finished without a manifest: $d" >&2; exit 4
        fi
      done
    done
  done
}

stage_train() {
  local tags; tags="$(cell_tags)"
  echo "[train] $(echo "$tags" | wc -l) cells (grid=$GRID)"
  for t in $tags; do run bash "$LANG_DIR/train_cell.sh" "$t"; done
}

stage_eval() {
  if [ "$GRID" = "steering" ]; then
    echo "[eval] the steering SAEs are evaluated by llm_steering/; nothing to do here"
    return 0
  fi
  local tags; tags="$(cell_tags)"
  echo "[eval] $(echo "$tags" | wc -l) cells (grid=$GRID)"
  for t in $tags; do run py -m langdual.lmsae.eval_trackb --tag "$t"; done
}

stage_textemb() {
  # The labelled corpora and their bge-large document embeddings that llm_steering/ reads.
  # Both steps are skip-guarded (meta.json / a complete LMDB with __len__).
  for c in agnews goemotions; do
    run py -m langdual.materialize_corpora corpora --corpus "$c"
    run py -m langdual.extract_text_embeddings --encoder bge --corpus "$c"
  done
}

case "$STAGE" in
  textemb) stage_textemb ;;
  pack) stage_pack ;;
  extract) stage_extract ;;
  train) stage_train ;;
  eval) stage_eval ;;
  all) stage_pack; stage_extract; stage_train; stage_eval ;;
  *) echo "unknown --stage $STAGE" >&2; exit 2 ;;
esac
