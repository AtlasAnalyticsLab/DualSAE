#!/bin/bash
# Interpretability of sparse latents: per-latent MonoScore / class-purity curves of all latents and
# top / mid / bottom activating images of latents paired by MonoScore rank.
#
# Models: BatchTopK and Dual-BatchTopK on SigLIP2 features of ImageNet-1K, seed 43, both with the
# main-table configuration, best-validation-MonoScore checkpoints. Evaluated on the validation split.
#
#   BTK_RUN=<run dir> DUAL_RUN=<run dir>   reuse main-table runs instead of training them
#   NUM_LATENTS (default 1000): how many MonoScore ranks to render; K (default 5): images per row.
#   The rank of the latent pair drawn in the figure is not recorded; choose it from the rendered ranks.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
OUT="$ABL_ROOT/figures/matched_latents"
RUNS="$ABL_ROOT/figure_runs"
feature_args imagenet sigclip
if [ -z "${BTK_RUN:-}" ]; then
  train_once "$RUNS" fig_imagenet_sigclip_batch_topk_seed43 --model batch_topk --dataset imagenet \
    --extractor sigclip "${FEATURE_ARGS[@]}" --mono_coef 0 --seed 43 --lr 0.0001
  BTK_RUN="$RUNS/fig_imagenet_sigclip_batch_topk_seed43"
fi
if [ -z "${DUAL_RUN:-}" ]; then
  # Main-table Dual-BatchTopK SigLIP2 seed 43: gamma 0.01, warm-up 17500, dead-latent check every
  # 100 steps (values used for the reported runs; main.py defaults differ).
  train_once "$RUNS" fig_imagenet_sigclip_dual_batch_topk_seed43 --model dual --sparsity_mode batch_topk \
    --dataset imagenet --extractor sigclip "${FEATURE_ARGS[@]}" --mono_coef 0 --seed 43 --lr 0.0001 \
    --span_coef 0.01 --warmup_steps 17500 --dead_check_interval 100 --poly_lr_factor 1 \
    --skip_epoch_checkpoints
  DUAL_RUN="$RUNS/fig_imagenet_sigclip_dual_batch_topk_seed43"
fi
mkdir -p "$OUT"
"$PY" "$ANALYSIS_DIR/latent_activation_grid.py" --dataset imagenet --extractor sigclip --split validation \
  --batch_topk_ckpt "$BTK_RUN/best_val_monoscore_model.pt" --dual_ckpt "$DUAL_RUN/best_val_monoscore_model.pt" \
  --num_latents "${NUM_LATENTS:-1000}" --min_active_samples 50 --k "${K:-5}" \
  --monoscore_plot_path "$OUT/monoscore_comparison.pdf" --class_purity_plot_path "$OUT/class_purity_comparison.pdf" \
  --output "$OUT/latent_grid.png" ${IMAGE_ROOT:+--image_root "$IMAGE_ROOT"}
