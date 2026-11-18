#!/usr/bin/env bash
# Shared environment for the hypothesis-generation scripts (sourced by every script here).
HG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PY="${PY:-python}"
export ASSETS_DIR="${ASSETS_DIR:-./assets}"
export DATA_ROOT="${DATA_ROOT:-./data}"
export HG_OUT="${HG_OUT:-./outputs/hypothesis_generation}"
export TM_OUT="${TM_OUT:-./outputs/topic_modeling}"          # SAE checkpoints: $TM_OUT/runs/<run>/...
export IMAGENET_TRAIN_DIR="${IMAGENET_TRAIN_DIR:-$DATA_ROOT/imagenet/train}"
export IMAGENET_VAL_DIR="${IMAGENET_VAL_DIR:-$DATA_ROOT/imagenet/val}"
export OPENIMAGES_TEST_DIR="${OPENIMAGES_TEST_DIR:-$DATA_ROOT/open_image_v7/test}"
export OPENIMAGES_VALID_DIR="${OPENIMAGES_VALID_DIR:-$DATA_ROOT/open_image_v7/valid}"
export OPENIMAGES_NARRATIVES_DIR="${OPENIMAGES_NARRATIVES_DIR:-$DATA_ROOT/open_image_v7}"
export FEATURES_IMAGENET="${FEATURES_IMAGENET:-./features/imagenet}"
export FEATURES_OPENIMAGES="${FEATURES_OPENIMAGES:-./features/open_images}"
export HF_HOME="${HF_HOME:-$ASSETS_DIR/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0     # vLLM engine in-process (must be set before vllm is imported)
export PYTHONPATH="$HG_DIR${PYTHONPATH:+:$PYTHONPATH}"
