#!/usr/bin/env bash
# Stage 1 for one dataset: image embeddings + manifests + labels, then dense SAE activations for
# every cell (3 encoders x 3 activations x {baseline, dual} x 3 seeds).
#   imagenet   : re-embeds the on-disk ImageFolder images (train: 300/class subsample; val: all)
#   openimages : reads the pre-extracted feature/label LMDBs (train role = test split, val role = valid split)
# Usage: bash scripts/01_stage1.sh imagenet|openimages        Resumable. GPU.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export HG_DATASET="${1:?imagenet|openimages}" HG_MODE=main
for ENC in clip siglip vit; do
    if [ "$HG_DATASET" = imagenet ]; then
        "$PY" -m hypogen.run_stage1 --encoder "$ENC"
    else
        for ROLE in val train; do "$PY" -m hypogen.build_oi_data --encoder "$ENC" --role "$ROLE"; done
        N=$("$PY" -m hypogen.jobs --kind cells --count)
        for i in $(seq 0 $((N - 1))); do
            read -r enc act var seed < <("$PY" -m hypogen.jobs --kind cells --index "$i")
            [ "$enc" = "$ENC" ] || continue
            for SPLIT in train val; do
                "$PY" -m hypogen.dump_activations --encoder "$enc" --activation "$act" --variant "$var" \
                    --seed "$seed" --split "$SPLIT"
            done
        done
    fi
done
