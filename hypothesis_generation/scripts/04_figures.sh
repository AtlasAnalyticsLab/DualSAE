#!/usr/bin/env bash
# Class-level hypothesis figures (ImageNet-1K, seed 42): one PDF per encoder, each stacking the three
# activation pairs (top-2 positive latents per model, five top-activating training images each).
# Selection rule per cell = 'auto' (see hypogen/encoder_figs.py and README). The two --override
# entries are the manually chosen classes used for the reported figures (bull mastiff = 243 for
# ViT/JumpReLU, class 4 for CLIP/JumpReLU); remove them to use the automatic class choice. CPU.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
HG_DATASET=imagenet HG_MODE=main "$PY" -m hypogen.encoder_figs --seed 42 \
    --override clip/jump_relu=4 vit/jump_relu=243 "$@"
