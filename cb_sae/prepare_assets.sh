#!/usr/bin/env bash
# ============================================================================
# prepare_assets.sh — the ONLY step of the CB-SAE pipeline that needs internet.
# Run once, from the repository root, on a machine with network access.
#
#   1. concept vocabularies (not redistributed: their upstream repositories
#      carry no license) are downloaded at a pinned commit into CB_VOCAB_DIR
#      and verified by sha256:
#        * VLG-CBM ImageNet-filtered concept set  (C_user of the CB head)
#        * CLIP-Dissect Broden label set          (I_m ranking vocabulary)
#   2. the supervisor CLIP text/image model (open_clip, LAION
#      CLIP-ViT-L-14-DataComp.XL-s13B-b90K — the same checkpoint that extracted
#      the 'clip' feature LMDBs) is downloaded into the Hugging Face cache;
#   3. text embeddings of both vocabularies (+ ImageNet class names for the
#      zero-shot alignment gate) are computed OFFLINE from that cache
#      (python -m featbsf.cb_text_embeds --embeds) into
#      $CB_WORK_DIR/cb/embeds/.
#
# Everything after this script runs offline (HF_HUB_OFFLINE=1).
#
# Environment (all optional):
#   CB_VOCAB_DIR  where the vocabularies go     (default ./assets/cb_sae/concepts)
#   CB_WORK_DIR   CB outputs incl. embed caches (default ./outputs/cb_sae)
#   HF_HOME       Hugging Face cache            (default: the library default)
#   DEVICE        device for the text encoder   (default: cuda if available)
#   PYTHON        interpreter                   (default: python)
#
# Requirements: cb_sae/requirements.txt (open_clip_torch + huggingface_hub
# are only needed by this script).
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
export CB_VOCAB_DIR="$(realpath -m "${CB_VOCAB_DIR:-./assets/cb_sae/concepts}")"
export CB_WORK_DIR="$(realpath -m "${CB_WORK_DIR:-./outputs/cb_sae}")"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CB_VOCAB_DIR"

# --- [1/3] concept vocabularies --------------------------------------------
# Pinned upstream commits and the sha256 of the exact files the reported runs
# used (the files are byte-identical to upstream at these commits).
VLG_COMMIT=84031a293a7153ad89a739ef2096b9335ba66790
VLG_URL="https://raw.githubusercontent.com/Trustworthy-ML-Lab/VLG-CBM/${VLG_COMMIT}/concept_files/imagenet_filtered.txt"
VLG_SHA=18c9c85dfe4020dac4d2fdf1a38e0ae98a41f90dfb5af7936511c7d61e5df49e
BRODEN_COMMIT=21e7697feaea3bf7d7bc2d2cc8e4047d5f5fd502
BRODEN_URL="https://raw.githubusercontent.com/Trustworthy-ML-Lab/CLIP-dissect/${BRODEN_COMMIT}/data/broden_labels_clean.txt"
BRODEN_SHA=603036d4b138c66455e1285bd915f6cdcdf4728df7a5563843d1f89c9babe64b

fetch() {  # fetch <url> <dest> <sha256>
  local url="$1" dest="$2" sha="$3"
  if [ -s "$dest" ] && echo "$sha  $dest" | sha256sum -c --status; then
    echo "  [skip] $(basename "$dest") (sha256 ok)"
    return
  fi
  curl -sfL --retry 3 -o "$dest.tmp" "$url"
  echo "$sha  $dest.tmp" | sha256sum -c --status || {
    echo "FATAL: sha256 mismatch for $url" >&2; rm -f "$dest.tmp"; exit 1; }
  mv "$dest.tmp" "$dest"
  echo "  [ok]   $(basename "$dest")"
}
echo "=== [1/3] concept vocabularies -> $CB_VOCAB_DIR"
# File names are part of the artifact naming (cache files and dissect outputs
# are keyed by the vocabulary's file stem), so keep them as written here.
fetch "$VLG_URL"    "$CB_VOCAB_DIR/vlgcbm_imagenet_filtered.txt" "$VLG_SHA"
fetch "$BRODEN_URL" "$CB_VOCAB_DIR/broden_labels_clean.txt"      "$BRODEN_SHA"

# --- [2/3] supervisor CLIP weights -----------------------------------------
echo "=== [2/3] supervisor CLIP weights (open_clip, HF hub)"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
"$PYTHON" - <<'PY'
from huggingface_hub import snapshot_download
REPO = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
# Commit the reported embeddings were computed from. open_clip resolves the
# 'main' ref at load time, so the snapshot is taken at 'main' and checked.
PINNED = "84c9828e63dc9a9351d1fe637c346d4c1c4db341"
path = snapshot_download(REPO, revision="main",
                         allow_patterns=["open_clip_config.json",
                                         "open_clip_pytorch_model.bin"])
got = path.rstrip("/").rsplit("/", 1)[-1]
print(f"  {REPO} -> {path}")
if got != PINNED:
    print(f"  WARNING: 'main' now resolves to {got}, not the pinned {PINNED}; "
          f"text embeddings may differ from the original run.")
PY

# --- [3/3] text embeddings (offline from here on) ---------------------------
echo "=== [3/3] text embeddings -> $CB_WORK_DIR/cb/embeds"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DEV_ARGS=()
[ -n "${DEVICE:-}" ] && DEV_ARGS=(--device "$DEVICE")
"$PYTHON" -m featbsf.cb_text_embeds --embeds "${DEV_ARGS[@]}"
echo "done. Next (offline): python cb_sae/scripts/cb_pipeline.py prep"
