#!/usr/bin/env bash
# Download everything the vision experiments need. Run ONCE on a machine with internet access;
# feature extraction, training and evaluation then run fully offline.
#
#   HF_HOME      HuggingFace cache shared with the offline steps          (default ./assets/hf)
#   DATA_ROOT    raw datasets                                             (default ./data)
#   STEPS        which parts to run: models imagenet openimages           (default: all three)
#   OI_SPLITS    OpenImagesV7 splits                                      (default: "train valid test")
#
# ImageNet-1K is a gated dataset on the HuggingFace Hub: accept its terms on the Hub and log in
# (`huggingface-cli login`) before running the `imagenet` step. OpenImagesV7 annotations,
# Localized Narratives and images are public. The URLs below are the official download
# locations; if one has moved, take the file from the dataset's official download page and
# place it at the same local path.
set -euo pipefail

VISION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="${HF_HOME:-./assets/hf}"
DATA_ROOT="${DATA_ROOT:-./data}"
STEPS="${STEPS:-models imagenet openimages}"
OI_SPLITS="${OI_SPLITS:-train valid test}"
PY="${PY:-python}"
OI_DIR="$DATA_ROOT/open_images_v7"
RAW="$OI_DIR/raw"

has_step() { [[ " $STEPS " == *" $1 "* ]]; }
fetch() {  # fetch URL DEST (skips existing files; downloads to a temp name first)
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then echo "[have] $dest"; return 0; fi
  mkdir -p "$(dirname "$dest")"
  echo "[get ] $url"
  curl -fL --retry 3 -o "$dest.part" "$url" && mv "$dest.part" "$dest"
}

# --------------------------------------------------------------------------- encoder weights
if has_step models; then
  "$PY" - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",   # CLIP ViT-L/14 (open_clip)
             "google/siglip2-so400m-patch14-384",           # SigLIP2
             "google/vit-large-patch16-224"):               # supervised ViT-L/16
    print("downloading", repo)
    snapshot_download(repo)
PY
fi

# --------------------------------------------------------------------------- ImageNet-1K
if has_step imagenet; then
  "$PY" - <<'PY'
from datasets import load_dataset
for split in ("train", "validation"):
    ds = load_dataset("ILSVRC/imagenet-1k", split=split)   # cached under $HF_HOME
    print(split, len(ds))
PY
fi

# --------------------------------------------------------------------------- OpenImagesV7
if has_step openimages; then
  LN=https://storage.googleapis.com/localized-narratives/annotations
  OI=https://storage.googleapis.com/openimages
  fetch "$OI/v7/oidv7-class-descriptions-boxable.csv" "$RAW/oidv7-class-descriptions-boxable.csv"
  for split in $OI_SPLITS; do
    case "$split" in
      train)
        jsonl=()
        for i in $(seq 0 9); do
          f="open_images_train_v6_localized_narratives-0000${i}-of-00010.jsonl"
          fetch "$LN/$f" "$RAW/$f"; jsonl+=("$RAW/$f")
        done
        labels_url="$OI/v5/train-annotations-human-imagelabels-boxable.csv" ;;
      valid)
        fetch "$LN/open_images_validation_localized_narratives.jsonl" "$RAW/open_images_validation_localized_narratives.jsonl"
        jsonl=("$RAW/open_images_validation_localized_narratives.jsonl")
        labels_url="$OI/v5/validation-annotations-human-imagelabels-boxable.csv" ;;
      test)
        fetch "$LN/open_images_test_localized_narratives.jsonl" "$RAW/open_images_test_localized_narratives.jsonl"
        jsonl=("$RAW/open_images_test_localized_narratives.jsonl")
        labels_url="$OI/v5/test-annotations-human-imagelabels-boxable.csv" ;;
    esac
    labels_csv="$RAW/$(basename "$labels_url")"
    fetch "$labels_url" "$labels_csv"
    # Caption index (defines the sample order of the LMDBs), label CSVs, then the images.
    cap="$OI_DIR/captions/$split/simplified_open_images_${split}_localized_narratives.json"
    [ -s "$cap" ] || "$PY" "$VISION_DIR/prepare_open_images.py" captions --split "$split" \
        --jsonl "${jsonl[@]}" --dataset_dir "$OI_DIR"
    "$PY" "$VISION_DIR/prepare_open_images.py" labels --split "$split" \
        --class_csv "$RAW/oidv7-class-descriptions-boxable.csv" --labels_csv "$labels_csv" \
        --dataset_dir "$OI_DIR"
    "$PY" "$VISION_DIR/prepare_open_images.py" images --split "$split" --dataset_dir "$OI_DIR"
  done
fi
echo "assets ready (HF_HOME=$HF_HOME, DATA_ROOT=$DATA_ROOT)"
