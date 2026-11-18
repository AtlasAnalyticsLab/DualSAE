#!/usr/bin/env bash
# Dense captions (InternVL3.5-8B) for the images whose captions form the topic-model vocabulary:
#   imagenet   : the ImageNet-1K validation split (50,000 images)
#   openimages : the OpenImagesV7 test split
# Usage: bash scripts/01_captions.sh imagenet|openimages [START END]
# Resumable: images already present in the output file are skipped. GPU.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
DS="${1:?imagenet|openimages}"; START="${2:-}"; END="${3:-}"
CAPTION_MODEL="${CAPTION_MODEL:-OpenGVLab/InternVL3_5-8B}"
# Resolve the local snapshot so trust_remote_code loads fully offline.
MODEL_DIR="$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$CAPTION_MODEL', local_files_only=True))")"
OUT_DIR="$DATA_ROOT/captions/$DS"; mkdir -p "$OUT_DIR"
RANGE=(); TAG="all"
if [ -n "$START" ]; then RANGE=(--start "$START" --end "$END"); TAG="${START}_${END}"; fi
if [ "$DS" = imagenet ]; then
    "$PY" "$TM_DIR/src/create_caption.py" --dataset imagenet --image-dir "$IMAGENET_VAL_DIR" \
        --model "$MODEL_DIR" --cache-dir "$HF_HOME/hub" --output "$OUT_DIR/InternVL3_5-8B_${TAG}.jsonl" \
        --resume-glob "$OUT_DIR/InternVL3_5-8B_*.jsonl" "${RANGE[@]}"
else
    "$PY" "$TM_DIR/src/create_caption.py" --dataset open_images --split test --image-dir "$OPENIMAGES_TEST_DIR" \
        --model "$MODEL_DIR" --cache-dir "$HF_HOME/hub" --output "$OUT_DIR/InternVL3_5-8B_${TAG}.jsonl" \
        --resume-glob "$OUT_DIR/InternVL3_5-8B_*.jsonl" "${RANGE[@]}"
fi
