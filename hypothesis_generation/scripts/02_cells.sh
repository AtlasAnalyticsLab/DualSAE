#!/usr/bin/env bash
# Stages 2-4 for every cell of one dataset under one selection rule:
#   lasso (HG_MODE=main)        : selection with all rules (K in {8,16}) -> proxy -> VLM interpretation of
#                                 the lasso K=16 union -> faithful evaluation (K=16)
#   correlation | separation_score (HG_MODE=selmethod): selection with that rule (K=16) -> interpretation
#                                 of its union (latents already interpreted in the lasso run are reused) ->
#                                 faithful evaluation (K=16); outputs under selmethod_<rule>/
# Usage: bash scripts/02_cells.sh imagenet|openimages lasso|correlation|separation_score [CELL_INDEX ...]
# Resumable. GPU (vLLM, InternVL2.5-8B).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export HG_DATASET="${1:?imagenet|openimages}"; RULE="${2:?lasso|correlation|separation_score}"; shift 2
if [ "$RULE" = lasso ]; then export HG_MODE=main; else export HG_MODE=selmethod HG_SEL_METHOD="$RULE"; fi
ROOT="$HG_OUT"; [ "$HG_DATASET" = openimages ] && ROOT="$HG_OUT/openimages"
N=$("$PY" -m hypogen.jobs --kind cells --count)
IDX=("$@"); [ ${#IDX[@]} -eq 0 ] && IDX=($(seq 0 $((N - 1))))
for i in "${IDX[@]}"; do
    read -r enc act var seed < <("$PY" -m hypogen.jobs --kind cells --index "$i")
    ARGS=(--encoder "$enc" --activation "$act" --variant "$var" --seed "$seed")
    echo "[02_cells] $HG_DATASET $RULE $enc/$act/$var/seed$seed"
    "$PY" -m hypogen.select_features "${ARGS[@]}"
    if [ "$RULE" = lasso ]; then
        "$PY" -m hypogen.evaluate --stage proxy "${ARGS[@]}"
    else
        # Reuse the lasso run's hypotheses for latents it already interpreted (same latent, same
        # images); only latents new to this rule's union are sent to the VLM.
        MAIN_HYP="$ROOT/interpreted/$enc/$act/$var/seed$seed/hypotheses.json"
        SM_INT="$ROOT/selmethod_${RULE}/interpreted/$enc/$act/$var/seed$seed"
        if [ -f "$MAIN_HYP" ] && [ ! -f "$SM_INT/hypotheses.json" ]; then
            mkdir -p "$SM_INT"; cp -n "$MAIN_HYP" "$SM_INT/hypotheses.json"
        fi
    fi
    "$PY" -m hypogen.interpret_latents "${ARGS[@]}" --vlm-batch 64
    "$PY" -m hypogen.evaluate --stage faithful "${ARGS[@]}"
done
