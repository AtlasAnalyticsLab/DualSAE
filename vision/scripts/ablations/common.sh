#!/bin/bash
# Shared environment for the ablation / analysis drivers. Source it; do not execute it.
#
#   FEATURES_ROOT   feature LMDBs, laid out as described in feature_paths.py   (default ./features)
#   ABL_ROOT        all runs, evaluations and figures of these drivers         (default ./outputs/ablations)
#   PY              python interpreter                                          (default python)
#
# Everything runs offline: datasets/models needed by a figure are fetched once by
# prepare_assets.sh, and W&B is disabled unless the caller sets WANDB_MODE themselves.

ABL_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISION_DIR="$(cd "$ABL_SCRIPTS_DIR/../.." && pwd)"
ANALYSIS_DIR="$VISION_DIR/analysis/ablations"

export FEATURES_ROOT="${FEATURES_ROOT:-./features}"
export ABL_ROOT="${ABL_ROOT:-./outputs/ablations}"
PY="${PY:-python}"

# Offline by construction. HF_HOME is left to the caller so prepare_assets.sh and the runs share
# one cache.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPLBACKEND=Agg

# feature_args DATASET ENCODER -> fills the bash array FEATURE_ARGS with
# --train_path/--val_path[/--test_path] for main.py.
feature_args() {
  mapfile -t FEATURE_ARGS < <("$PY" "$ABL_SCRIPTS_DIR/feature_paths.py" --dataset "$1" --encoder "$2")
}

# feature_path DATASET ENCODER SPLIT -> prints one LMDB path (empty if the split does not exist).
feature_path() {
  "$PY" "$ABL_SCRIPTS_DIR/feature_paths.py" --dataset "$1" --encoder "$2" --split "$3"
}

# train_once OUT_DIR EXP_NAME ARGS... -> runs main.py unless the run already finished
# (autoencoder_final.pt is written only at the end of training, so it marks completion).
train_once() {
  local out_dir="$1" exp="$2"; shift 2
  if [ -f "$out_dir/$exp/autoencoder_final.pt" ]; then
    echo "[skip] $exp (finished)"
    return 0
  fi
  echo "[train] $exp"
  (cd "$VISION_DIR" && "$PY" main.py "$@" --exp_name "$exp" --output_dir "$(abspath "$out_dir")" \
     --skip_existing)
}

abspath() { "$PY" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"; }
