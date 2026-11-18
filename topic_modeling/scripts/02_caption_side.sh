#!/usr/bin/env bash
# Text side of the topic model for one dataset: merged/repaired captions, caption-aligned image
# embeddings for each encoder (rows in caption order, L2-normalised), documents.jsonl and the
# bag-of-words (vocabulary <= 20000 lemmas, min document frequency 5).
# Usage: bash scripts/02_caption_side.sh imagenet|openimages        (GPU for the embeddings)
# Outputs: $TM_OUT/{imagenet_val|openimages_test}/{emb_clip,emb_siglip,emb_vit,data/documents.jsonl,bow/}
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
DS="${1:?imagenet|openimages}"
if [ "$DS" = imagenet ]; then V="$TM_OUT/imagenet_val"; IMGROOT="$IMAGENET_VAL_DIR"
else V="$TM_OUT/openimages_test"; IMGROOT="$OPENIMAGES_TEST_DIR"; fi
CAPS="$V/data/captions_clean.jsonl"; DOCS="$V/data/documents.jsonl"
mkdir -p "$V/data"
# Stage 0: one caption file in a fixed order. OpenImagesV7: merge the shards and repair records whose
# caption contains unescaped newlines/quotes (repair_captions.py). ImageNet-1K: the caption records
# carry an extra class_folder field and were used as written, so shards are only concatenated
# (first occurrence of an image wins).
if [ ! -f "$CAPS" ]; then
    if [ "$DS" = openimages ]; then
        "$PY" "$TM_DIR/src/repair_captions.py" --in "$DATA_ROOT/captions/$DS/InternVL3_5-8B_*.jsonl" --out "$CAPS"
    else
        "$PY" - "$DATA_ROOT/captions/$DS" "$CAPS" <<'PYCAT'
import glob, json, os, sys
seen, tmp = set(), sys.argv[2] + ".tmp"
with open(tmp, "w", encoding="utf-8") as w:
    for fp in sorted(glob.glob(os.path.join(sys.argv[1], "InternVL3_5-8B_*.jsonl"))):
        for line in open(fp, encoding="utf-8"):
            if line.strip() and json.loads(line)["rel_path"] not in seen:
                seen.add(json.loads(line)["rel_path"]); w.write(line if line.endswith("\n") else line + "\n")
os.replace(tmp, sys.argv[2])
PYCAT
    fi
fi
# Stage 1: caption-aligned embeddings per encoder (resumable, shard-wise).
for ENC in clip siglip vit; do
    case "$ENC" in clip) BS=512 ;; *) BS=256 ;; esac
    "$PY" "$TM_DIR/src/prepare_topic_data.py" --encoder "$ENC" --captions "$CAPS" --image-root "$IMGROOT" \
        --out-dir "$V/emb_${ENC}" --documents-path "$DOCS" --batch-size "$BS" --shard-size 5000
done
# Stage 2: bag-of-words (SAE-TM baselines/utils; NLTK data must be present, never downloaded).
if [ ! -f "$V/bow/bow_meta.json" ]; then
    mkdir -p "$V/bow/vocab_shards" "$V/bow/bow_shards"
    BOWPY="$SAE_TM_DIR/baselines/utils/make_bow_cache_sharded.py"
    "$PY" "$BOWPY" build-vocab-shard --jsonl "$DOCS" --out-vocab-shard "$V/bow/vocab_shards/shard_0000.pt" --start 0 --num-rows -1
    "$PY" "$BOWPY" merge-vocab-shards --shards "$V/bow/vocab_shards/shard_*.pt" --out-vocab "$V/bow/vocab.pt" --vocab-size 20000 --min-df 5
    "$PY" "$BOWPY" build-shard --jsonl "$DOCS" --vocab "$V/bow/vocab.pt" --out-shard "$V/bow/bow_shards/shard_0000.pt" --start 0 --num-rows -1
    "$PY" "$BOWPY" merge-shards --vocab "$V/bow/vocab.pt" --out-dataset "$V/bow/bow_merged.pt" --shards "$V/bow/bow_shards/shard_*.pt"
    "$PY" "$SAE_TM_DIR/baselines/utils/convert_bow.py" --input-pt "$V/bow/bow_merged.pt" --out-base "$V/bow/bow"
fi
echo "caption side ready: $V"
