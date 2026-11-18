#!/usr/bin/env bash
# Every download the hypothesis-generation pipeline needs (run once, with internet access).
# Afterwards all hypogen code runs offline (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1).
#
#   bash hypothesis_generation/prepare_assets.sh
#
#   * HypotheSAEs (Apache-2.0) at the pinned commit, installed into the current Python environment
#     without its own dependencies (only hypothesaes.evaluation.score_hypotheses is used)
#   * the VLM OpenGVLab/InternVL2_5-8B (proposer + annotator) -> $HF_HOME
#   * the three vision encoders (ImageNet-1K re-embedding)    -> $HF_HOME
#   * class-name files for the figures                       -> $ASSETS_DIR
set -euo pipefail
PY="${PY:-python}"
ASSETS_DIR="${ASSETS_DIR:-./assets}"
export HF_HOME="${HF_HOME:-$ASSETS_DIR/huggingface}"
VLM_MODEL="${HG_VLM_MODEL:-OpenGVLab/InternVL2_5-8B}"
HYPOTHESAES_COMMIT="4573abd87482ad3f02f850e0ca38897cac3e699b"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
mkdir -p "$HF_HOME" "$ASSETS_DIR"

echo "=== [1/4] HypotheSAEs @ $HYPOTHESAES_COMMIT ==="
"$PY" -m pip install --no-deps "hypothesaes @ git+https://github.com/rmovva/HypotheSAEs@${HYPOTHESAES_COMMIT}"
"$PY" -c "from hypothesaes.evaluation import score_hypotheses; print('  hypothesaes OK')"

echo "=== [2/4] VLM $VLM_MODEL -> $HF_HOME ==="
"$PY" - "$VLM_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id=sys.argv[1], allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model",
                                                           "tokenizer*", "*.py"])
print("  cached at:", p)
PY

echo "=== [3/4] vision encoders -> $HF_HOME ==="
"$PY" - <<'PY'
import open_clip
from transformers import AutoModel, AutoImageProcessor
open_clip.create_model_and_transforms("hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K")
for mid in ("google/siglip2-so400m-patch14-384", "google/vit-large-patch16-224"):
    AutoModel.from_pretrained(mid); AutoImageProcessor.from_pretrained(mid)
print("  encoders ready")
PY

echo "=== [4/4] class names -> $ASSETS_DIR ==="
fetch () {  # $1 = output file, remaining = mirror URLs (first that works wins)
    local out="$1"; shift
    [ -s "$out" ] && { echo "  have $out"; return 0; }
    for url in "$@"; do
        if curl -fsSL --connect-timeout 20 "$url" -o "$out.tmp" && [ -s "$out.tmp" ]; then
            mv "$out.tmp" "$out"; echo "  fetched $out"; return 0
        fi
    done
    rm -f "$out.tmp"; echo "  [warn] could not fetch $out (figures fall back to class ids)"
}
fetch "$ASSETS_DIR/imagenet_class_index.json" \
  "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json" \
  "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fetch "$ASSETS_DIR/oidv7-class-descriptions-boxable.csv" \
  "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv" \
  "https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv"
echo "Done."
