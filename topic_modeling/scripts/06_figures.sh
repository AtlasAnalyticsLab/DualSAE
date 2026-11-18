#!/usr/bin/env bash
# Image-topic assignment figures (ImageNet-1K, run imagenet_paired, seed 42, K = 100):
#   1. LLM topic labels for the Dual and baseline topic cells          (annotate_topics.py; GPU)
#   2. LLM judge: per candidate image, Dual vs baseline dominant topic (judge_img2topic.py; GPU)
#   3. render one row per encoder/activation with judged Dual-win images (visualize_topics.py; CPU)
# The selection rule is documented in README.md and in judge_img2topic.py.
# Usage: bash scripts/06_figures.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
RUN_DIR="$TM_OUT/runs/imagenet_paired"; SEED=42; K=100
DOCS="$TM_OUT/imagenet_val/data/documents.jsonl"
CELLS=()
for ENC in clip siglip vit; do
  for TR in TopKTrainer DualTopKTrainer BatchTopKTrainer DualBatchTopKTrainer JumpReluTrainer DualJumpReluTrainer; do
    D="$RUN_DIR/$ENC/topics/$TR/seed$SEED/K$K"; [ -f "$D/clusters.csv" ] && CELLS+=("$D")
  done
done
"$PY" "$TM_DIR/src/annotate_topics.py" --model "$JUDGE_MODEL" --topics-cells "${CELLS[@]}"
"$PY" "$TM_DIR/src/judge_img2topic.py" --model "$JUDGE_MODEL" --results-dir "$RUN_DIR" --documents "$DOCS" \
    --seed "$SEED" --k "$K" --n-candidates 1500        # value used for the reported figures
"$PY" "$TM_DIR/src/visualize_topics.py" --figures img2topic-all --results-dir "$RUN_DIR" --documents "$DOCS" \
    --image-root "$IMAGENET_VAL_DIR" --seed "$SEED" --k "$K" --out-dir "$TM_OUT/figures"
