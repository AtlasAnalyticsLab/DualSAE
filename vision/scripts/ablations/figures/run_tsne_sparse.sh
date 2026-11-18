#!/bin/bash
# t-SNE of sparse representations: joint t-SNE of the sparse codes z of Dual-JumpReLU and JumpReLU
# for 20 ImageNet-1K classes (the fixed class list in tsne_sparse_codes.py), 50 validation images
# per class, SigLIP2 features, seed 43, best-validation-MonoScore checkpoints.
#
# JumpReLU: main-table run (lr 5e-5, normalised inputs, L0 coefficient 1e-3, threshold init 1.0).
# Dual-JumpReLU: the run drawn in the figure used gamma 0.01, lr 5e-5, L0 coefficient 1e-3,
# warm-up 17500 (not the main-table Dual-JumpReLU settings); it is trained here with those values.
#
#   JR_RUN=<run dir> DUAL_RUN=<run dir>   reuse existing runs
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
OUT="$ABL_ROOT/figures/tsne_sparse_codes"
RUNS="$ABL_ROOT/figure_runs"
feature_args imagenet sigclip
if [ -z "${JR_RUN:-}" ]; then
  train_once "$RUNS" fig_imagenet_sigclip_jumprelu_seed43 --model jumprelu --dataset imagenet \
    --extractor sigclip "${FEATURE_ARGS[@]}" --mono_coef 0 --seed 43 --normalize --l1_coef 0.001 \
    --bandwidth 0.001 --lr 0.00005 --jump_threshold_init 1.0
  JR_RUN="$RUNS/fig_imagenet_sigclip_jumprelu_seed43"
fi
if [ -z "${DUAL_RUN:-}" ]; then
  train_once "$RUNS" fig_imagenet_sigclip_dual_jump_relu_seed43 --model dual --sparsity_mode jump_relu \
    --dataset imagenet --extractor sigclip "${FEATURE_ARGS[@]}" --mono_coef 0 --seed 43 --normalize \
    --l1_coef 0.001 --bandwidth 0.001 --lr 0.00005 --jump_threshold_init 1.0 --span_coef 0.01 \
    --warmup_steps 17500 --dead_check_interval 100 --poly_lr_factor 1 --skip_epoch_checkpoints
  DUAL_RUN="$RUNS/fig_imagenet_sigclip_dual_jump_relu_seed43"
fi
"$PY" "$ANALYSIS_DIR/tsne_sparse_codes.py" --encoder sigclip --samples-per-class 50 \
  --pair "$DUAL_RUN/best_val_monoscore_model.pt" "$JR_RUN/best_val_monoscore_model.pt" --output-dir "$OUT"
