#!/bin/bash
# One-time downloads for the ablation / analysis figures (run on a machine with internet access;
# everything else in scripts/ablations runs offline afterwards).
#
#  1. ImageNet-1K validation images from the Hugging Face dataset `ILSVRC/imagenet-1k` (gated: accept
#     its terms on the Hub and log in with your own account first). The image-grid figures fetch a
#     tile by the LMDB index from this split when --image_root / IMAGE_ROOT is not given.
#  2. The WordNet corpus for nltk (superclass colouring of the t-SNE figure).
#
# Caches go to HF_HOME (default ~/.cache/huggingface) and NLTK_DATA (default ./assets/nltk_data);
# export the same variables when running the figure drivers.
set -euo pipefail
PY="${PY:-python}"
export NLTK_DATA="${NLTK_DATA:-./assets/nltk_data}"
mkdir -p "$NLTK_DATA"

"$PY" - <<'PYEOF'
from datasets import load_dataset
ds = load_dataset("ILSVRC/imagenet-1k", split="validation")
print("cached ImageNet-1K validation:", len(ds), "images")
PYEOF

"$PY" -c "import nltk, os; nltk.download('wordnet', download_dir=os.environ['NLTK_DATA']); nltk.download('omw-1.4', download_dir=os.environ['NLTK_DATA'])"
echo "assets ready (HF_HOME=${HF_HOME:-default}, NLTK_DATA=$NLTK_DATA)"
