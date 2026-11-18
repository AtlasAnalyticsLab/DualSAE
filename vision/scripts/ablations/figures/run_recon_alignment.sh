#!/bin/bash
# Input-reconstruction alignment: joint t-SNE of x and x_hat (50 validation images of one random
# ImageNet-1K class, random state 0), one panel per model, CLIP features, best-validation-MonoScore
# checkpoints. The plot window is fixed to [-5, 5] (tsne_x_xhat.AXIS_LIMIT).
#
# BatchTopK: main-table run, seed 42. Dual-BatchTopK: the run drawn in the figure is seed 44 with
# gamma 0.01 and warm-up 17500 (not the main-table Dual-BatchTopK CLIP setting); trained here.
#
#   BTK_RUN=<run dir> DUAL_RUN=<run dir>   reuse existing runs
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
OUT="$ABL_ROOT/figures/recon_alignment"
RUNS="$ABL_ROOT/figure_runs"
feature_args imagenet clip
if [ -z "${BTK_RUN:-}" ]; then
  train_once "$RUNS" fig_imagenet_clip_batch_topk_seed42 --model batch_topk --dataset imagenet \
    --extractor clip "${FEATURE_ARGS[@]}" --mono_coef 0 --seed 42 --lr 0.0001
  BTK_RUN="$RUNS/fig_imagenet_clip_batch_topk_seed42"
fi
if [ -z "${DUAL_RUN:-}" ]; then
  train_once "$RUNS" fig_imagenet_clip_dual_batch_topk_w17500_seed44 --model dual --sparsity_mode batch_topk \
    --dataset imagenet --extractor clip "${FEATURE_ARGS[@]}" --mono_coef 0 --seed 44 --lr 0.0001 \
    --span_coef 0.01 --warmup_steps 17500 --dead_check_interval 100 --poly_lr_factor 1 --skip_epoch_checkpoints
  DUAL_RUN="$RUNS/fig_imagenet_clip_dual_batch_topk_w17500_seed44"
fi
for RUN in "$BTK_RUN" "$DUAL_RUN"; do
  NAME=$(basename "$RUN")
  case "$NAME" in *dual*) LABEL="Dual-BatchTopK SAE" ;; *) LABEL="BatchTopK SAE" ;; esac
  "$PY" "$ANALYSIS_DIR/tsne_x_xhat.py" --checkpoints "$RUN/best_val_monoscore_model.pt" --encoder clip \
    --num-samples 50 --random-state 0 --model-name "$LABEL" --output-dir "$OUT/$NAME"
done
