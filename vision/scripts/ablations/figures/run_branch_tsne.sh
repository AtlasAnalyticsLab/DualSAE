#!/bin/bash
# Branch decomposition, panel (a): joint t-SNE of x^M, r_hat and x_hat for 3 ImageNet-1K classes
# (150 validation images), from a Dual-TopK model on CLIP features, seed 44, warm-up 2500, at its
# epoch-40 snapshot (epoch_checkpoints/epoch_040.pt).
#
# The run also passed --save_val_embeds (validation embeddings written every epoch; no effect on
# training) and recorded --poly_hidden 1 (not read at poly_layers = 1). Per-epoch checkpoints are
# kept for this run because the figure reads one of them.
#
#   DUAL_RUN=<run dir>   reuse an existing run
# Then panel (b) and the composite: analysis/ablations/plot_branch_bars.py --tsne-png <png written here>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
OUT="$ABL_ROOT/figures/branch_tsne"
RUNS="$ABL_ROOT/figure_runs"
if [ -z "${DUAL_RUN:-}" ]; then
  feature_args imagenet clip
  train_once "$RUNS" fig_imagenet_clip_dual_topk_w2500_seed44 --model dual --sparsity_mode topk \
    --dataset imagenet --extractor clip "${FEATURE_ARGS[@]}" --mono_coef 0 --seed 44 --lr 0.0001 \
    --span_coef 0.01 --warmup_steps 2500 --dead_check_interval 100 --poly_hidden 1 --poly_layers 1 \
    --poly_lr_factor 1
  DUAL_RUN="$RUNS/fig_imagenet_clip_dual_topk_w2500_seed44"
fi
"$PY" "$ANALYSIS_DIR/tsne_branches.py" --checkpoints "$DUAL_RUN/epoch_checkpoints/epoch_040.pt" \
  --encoder clip --num-classes 3 --num-samples 150 --random-state 0 --output-dir "$OUT"
