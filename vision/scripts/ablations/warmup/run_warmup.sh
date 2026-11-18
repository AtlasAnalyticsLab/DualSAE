#!/bin/bash
# Warm-up ablation (method section, "Effect of warm-up length" figure): DualSAE trained with
# different Stage-1 lengths (--warmup_steps = number of optimizer steps the Sparse Branch is
# trained before the Residual Branch and the decorrelation loss are switched on).
#
# Defaults = the runs drawn in the figure: OpenImagesV7, CLIP, Dual-BatchTopK, gamma = 0.01,
# warm-up in {0, 2500, 5000, 7500}, seeds 42-44. The full grid that was trained is
#   DATASET    open_images | imagenet
#   MODEL      dual_topk | dual_batch_topk | dual_jump_relu
#   ENCODERS   "clip sigclip vit"
#   WARMUPS    open_images: "0 100 500 1000 2500 5000 7500 10000 12500 15000 17500 20000" (seeds 42-44)
#              imagenet:    "0 100 500 1000 2500 5000 7500 10000 12500 15000 20000 22500 25000" (seed 42)
# e.g.  DATASET=imagenet SEEDS=42 WARMUPS="0 2500 5000 7500" bash run_warmup.sh
#
# Resumable: a run whose autoencoder_final.pt exists is skipped. Per-epoch validation R^2 and
# MonoScore go to TensorBoard ($ABL_ROOT/warmup/tensorboard/<run>), which
# analysis/ablations/plot_warmup_curves.py reads.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

DATASET="${DATASET:-open_images}"
MODEL="${MODEL:-dual_batch_topk}"
ENCODERS="${ENCODERS:-clip}"
WARMUPS="${WARMUPS:-0 2500 5000 7500}"
SEEDS="${SEEDS:-42 43 44}"
OUT="$ABL_ROOT/warmup"

case "$MODEL" in
  dual_topk) MODE=topk; EXTRA=() ;;
  dual_batch_topk) MODE=batch_topk; EXTRA=() ;;
  # JumpReLU Sparse Branch: normalised inputs, L0 coefficient 1e-4, bandwidth 1e-3, and the
  # threshold init theta = 1.0 used by these runs.
  dual_jump_relu) MODE=jump_relu; EXTRA=(--normalize --l1_coef 0.0001 --bandwidth 0.001 --jump_threshold_init 1.0) ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

for ENC in $ENCODERS; do
  feature_args "$DATASET" "$ENC"
  for SEED in $SEEDS; do
    for W in $WARMUPS; do
      EXP="warmup_${DATASET}_${ENC}_${MODEL}_w${W}_seed${SEED}"
      # --dead_check_interval 100: value used for the reported runs (main.py default 1000).
      train_once "$OUT/checkpoints" "$EXP" \
        --model dual --sparsity_mode "$MODE" --dataset "$DATASET" --extractor "$ENC" \
        "${FEATURE_ARGS[@]}" \
        --mono_coef 0 --seed "$SEED" --lr 0.0001 --dead_check_interval 100 \
        --span_coef 0.01 --warmup_steps "$W" --poly_lr_factor 1 \
        ${EXTRA[@]+"${EXTRA[@]}"} \
        --tb_dir "$(abspath "$OUT/tensorboard/$EXP")" --skip_epoch_checkpoints
    done
  done
done
echo "done. Plot: $PY $ANALYSIS_DIR/plot_warmup_curves.py --tb-root $OUT/tensorboard --dataset $DATASET --encoder ${ENCODERS%% *} --model $MODEL"
