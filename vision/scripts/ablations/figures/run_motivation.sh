#!/bin/bash
# Motivation figure: top-50 class purity of retrieval with the SAE reconstruction x_hat vs. the
# residual r = x - x_hat (left), and the retrievals for one query image (right).
#
# Model: BatchTopK SAE on CLIP features of ImageNet-1K, seed 42, main-table configuration
# (all main.py defaults for --model batch_topk), read at its epoch-50 checkpoint
# (autoencoder_final.pt). Query: validation index 1.
#
#   BTK_RUN=<run dir>   reuse an existing run of that configuration instead of training one
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
OUT="$ABL_ROOT/figures/motivation"
QUERIES="${QUERIES:-1}"

if [ -z "${BTK_RUN:-}" ]; then
  feature_args imagenet clip
  EXP=motivation_imagenet_clip_batch_topk_seed42
  train_once "$ABL_ROOT/figure_runs" "$EXP" --model batch_topk --dataset imagenet --extractor clip \
    "${FEATURE_ARGS[@]}" --mono_coef 0 --seed 42 --lr 0.0001
  BTK_RUN="$ABL_ROOT/figure_runs/$EXP"
fi
VAL=$(feature_path imagenet clip val)
CKPT="$BTK_RUN/autoencoder_final.pt"   # epoch-50 checkpoint, as in the figure

"$PY" "$ANALYSIS_DIR/motivation/top50_class_purity.py" --checkpoint "$CKPT" --lmdb "$VAL" \
  --topk 50 --output-dir "$OUT"
"$PY" "$ANALYSIS_DIR/motivation/retrieval_grid.py" --checkpoint "$CKPT" --lmdb "$VAL" \
  --queries $QUERIES --result-dir "$OUT" ${IMAGE_ROOT:+--image-root "$IMAGE_ROOT"}
echo "outputs in $OUT (histogram: top50_class_purity_overlay_histogram.pdf; grid: checkpoint_autoencoder_final_query_<i>.png)"
