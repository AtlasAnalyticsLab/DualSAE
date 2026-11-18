#!/usr/bin/env bash
# Shared environment for the topic-modeling scripts (sourced by every script here).
# Every location can be overridden by exporting the variable before running a script.
TM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PY="${PY:-python}"
export ASSETS_DIR="${ASSETS_DIR:-./assets}"
export DATA_ROOT="${DATA_ROOT:-./data}"
export TM_OUT="${TM_OUT:-./outputs/topic_modeling}"
export SAE_TM_DIR="${SAE_TM_DIR:-$TM_DIR/third_party/SAE-TM}"
# SAE training features (LMDBs written by the vision feature extraction):
#   $FEATURES_IMAGENET/{train,validation}/<encoder>.lmdb, $FEATURES_OPENIMAGES/{train,valid}/<encoder>.lmdb
export FEATURES_IMAGENET="${FEATURES_IMAGENET:-./features/imagenet}"
export FEATURES_OPENIMAGES="${FEATURES_OPENIMAGES:-./features/open_images}"
# Images captioned / shown in figures
export IMAGENET_VAL_DIR="${IMAGENET_VAL_DIR:-$DATA_ROOT/imagenet/val}"
export OPENIMAGES_TEST_DIR="${OPENIMAGES_TEST_DIR:-$DATA_ROOT/open_image_v7/test}"
# Offline assets (prepare_assets.sh)
export HF_HOME="${HF_HOME:-$ASSETS_DIR/huggingface}"
export NLTK_DATA="${NLTK_DATA:-$ASSETS_DIR/nltk_data}"
export W2V_PATH="${W2V_PATH:-$ASSETS_DIR/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz}"
export SAE_TM_W2V_DIR="${SAE_TM_W2V_DIR:-$ASSETS_DIR/w2v}"
export JUDGE_MODEL="${JUDGE_MODEL:-microsoft/phi-4}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0     # vLLM engine in-process (must be set before vllm is imported)
export PYTHONPATH="$TM_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
