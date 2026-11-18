#!/usr/bin/env bash
# Every download the topic-modeling pipeline needs (run once, on a machine with internet access).
# After this, all topic-modeling code runs fully offline (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1).
#
#   bash topic_modeling/prepare_assets.sh
#
# Downloads into $ASSETS_DIR (default ./assets):
#   huggingface/  vision encoders (CLIP ViT-L/14 DataComp via open_clip, SigLIP2 so400m-14-384,
#                 ViT-L/16-224), the LLM judge microsoft/phi-4, and the captioning model
#                 OpenGVLab/InternVL3_5-8B
#   gensim-data/  word2vec-google-news-300 (.gz, read by SAE-TM sae_to_topic.py)
#   w2v/          embeddings.np.npy + vocabulary.json built from it (read by SAE-TM evaluation.py)
#   nltk_data/    tokenizer / stopword / wordnet / tagger data (BoW construction)
#   imagenet_class_index.json   class names (optional, used by visualize_topics.py --topic-select puritygap)
# and clones + patches SAE-TM at the pinned commit (setup_saetm.sh).
set -euo pipefail

TM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
ASSETS_DIR="${ASSETS_DIR:-./assets}"
export HF_HOME="${HF_HOME:-$ASSETS_DIR/huggingface}"
export GENSIM_DATA_DIR="${GENSIM_DATA_DIR:-$ASSETS_DIR/gensim-data}"
export NLTK_DATA="${NLTK_DATA:-$ASSETS_DIR/nltk_data}"
SAE_TM_W2V_DIR="${SAE_TM_W2V_DIR:-$ASSETS_DIR/w2v}"
JUDGE_MODEL="${JUDGE_MODEL:-microsoft/phi-4}"
CAPTION_MODEL="${CAPTION_MODEL:-OpenGVLab/InternVL3_5-8B}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
mkdir -p "$HF_HOME" "$GENSIM_DATA_DIR" "$NLTK_DATA" "$SAE_TM_W2V_DIR"

echo "=== [1/6] vision encoders -> $HF_HOME ==="
"$PY" - <<'PY'
import open_clip
from transformers import AutoModel, AutoImageProcessor
open_clip.create_model_and_transforms("hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K")
for mid in ("google/siglip2-so400m-patch14-384", "google/vit-large-patch16-224"):
    AutoModel.from_pretrained(mid)
    AutoImageProcessor.from_pretrained(mid)
print("  encoders ready")
PY

echo "=== [2/6] LLM judge ($JUDGE_MODEL) and captioning model ($CAPTION_MODEL) -> $HF_HOME ==="
"$PY" - "$JUDGE_MODEL" "$CAPTION_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    p = snapshot_download(repo_id=repo, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model",
                                                        "tokenizer*", "*.py", "*.tiktoken"])
    print("  cached:", p)
PY

echo "=== [3/6] word2vec-google-news-300 -> $GENSIM_DATA_DIR ==="
"$PY" - <<'PY'
import gensim.downloader as api, os
p = api.load("word2vec-google-news-300", return_path=True)
print("  word2vec at:", p, os.path.getsize(p) // 1024 // 1024, "MB")
PY

echo "=== [4/6] embeddings.np.npy + vocabulary.json -> $SAE_TM_W2V_DIR ==="
"$PY" - "$SAE_TM_W2V_DIR" <<'PY'
import os, sys, json, numpy as np
import gensim.downloader as api
out_dir = sys.argv[1]
emb_path, vocab_path = os.path.join(out_dir, "embeddings.np.npy"), os.path.join(out_dir, "vocabulary.json")
if os.path.exists(emb_path) and os.path.exists(vocab_path):
    print("  already built:", emb_path); raise SystemExit
kv = api.load("word2vec-google-news-300")             # cached from step 3
np.save(emb_path, np.asarray(kv.vectors, dtype=np.float32))   # (3M, 300)
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(list(kv.index_to_key), f)               # row i -> word
print("  built:", emb_path)
PY

echo "=== [5/6] NLTK data -> $NLTK_DATA ==="
"$PY" - <<'PY'
import nltk, os
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng", "omw-1.4"]:
    ok = nltk.download(pkg, quiet=True, download_dir=os.environ["NLTK_DATA"])
    print(f"  {pkg}: {'ok' if ok else 'FAILED'}")
PY
if [ ! -s "$ASSETS_DIR/imagenet_class_index.json" ]; then
    curl -fsSL --connect-timeout 20 -o "$ASSETS_DIR/imagenet_class_index.json" \
        "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json" \
        || echo "  [warn] imagenet_class_index.json not downloaded (only needed for --topic-select puritygap)"
fi

echo "=== [6/6] SAE-TM at the pinned commit + patch ==="
bash "$TM_DIR/setup_saetm.sh"

echo "Done. Export before running (see README):"
echo "  HF_HOME=$HF_HOME W2V_PATH=$GENSIM_DATA_DIR/word2vec-google-news-300/word2vec-google-news-300.gz"
echo "  SAE_TM_W2V_DIR=$SAE_TM_W2V_DIR NLTK_DATA=$NLTK_DATA"
