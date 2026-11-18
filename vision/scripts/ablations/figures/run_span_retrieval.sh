#!/bin/bash
# Effect of the decorrelation loss on branch separation: nearest-neighbour retrievals from x^M and
# r_hat for the same query, Dual-BatchTopK without (gamma = 0) and with the decorrelation loss.
# The runs drawn in the figure use ViT features of ImageNet-1K, seed 42, warm-up 17500, and
# gamma = 2.5 for the "with" arm; the figure shows validation query index 4.
#
#   NO_SPAN_RUN=<run dir> WITH_SPAN_RUN=<run dir>   reuse existing runs
#   QUERY_START / QUERY_END (default 1 / 9): validation indices to render
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
OUT="$ABL_ROOT/figures/span_retrieval"
RUNS="$ABL_ROOT/figure_runs"
feature_args imagenet vit
for GAMMA in 0.0 2.5; do
  EXP="fig_imagenet_vit_dual_batch_topk_span${GAMMA}_seed42"
  if [ "$GAMMA" = "0.0" ] && [ -n "${NO_SPAN_RUN:-}" ]; then continue; fi
  if [ "$GAMMA" = "2.5" ] && [ -n "${WITH_SPAN_RUN:-}" ]; then continue; fi
  train_once "$RUNS" "$EXP" --model dual --sparsity_mode batch_topk --dataset imagenet --extractor vit \
    "${FEATURE_ARGS[@]}" --mono_coef 0 --seed 42 --lr 0.0001 --span_coef "$GAMMA" --warmup_steps 17500 \
    --dead_check_interval 100 --poly_lr_factor 1 --skip_epoch_checkpoints
done
NO_SPAN_RUN="${NO_SPAN_RUN:-$RUNS/fig_imagenet_vit_dual_batch_topk_span0.0_seed42}"
WITH_SPAN_RUN="${WITH_SPAN_RUN:-$RUNS/fig_imagenet_vit_dual_batch_topk_span2.5_seed42}"
"$PY" "$ANALYSIS_DIR/span_retrieval.py" \
  --checkpoint-no-span "$NO_SPAN_RUN/best_val_monoscore_model.pt" \
  --checkpoint-with-span "$WITH_SPAN_RUN/best_val_monoscore_model.pt" \
  --lmdb-path "$(feature_path imagenet vit val)" --query-start "${QUERY_START:-1}" --query-end "${QUERY_END:-9}" \
  --result-dir "$OUT" ${IMAGE_ROOT:+--image-root "$IMAGE_ROOT"}
