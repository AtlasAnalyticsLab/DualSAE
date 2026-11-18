#!/usr/bin/env bash
# End-to-end driver of the main SAE-vs-DualSAE comparison and the parameter-matched baselines.
# Sequential and resumable: every step skips work whose output already exists, so the script
# can simply be re-run after an interruption. Run from anywhere; paths default to ./ relative
# to the current directory.
#
#   FEATURES_ROOT   feature LMDBs                          (default ./features)
#   MAIN_CKPT_DIR   runs of configs/main_cells.json        (default ./checkpoints/main)
#   PMATCH_CKPT_DIR parameter-matched runs                 (default ./checkpoints/param_matched)
#   CB_CKPT_DIR     CB-SAE / CB-DualSAE runs (cb_sae/)     (default ./checkpoints/cb, optional)
#   CB_LONG         cb_sae/ collector output               (default ./outputs/cb_sae/cb_long.csv, optional)
#   MAIN_EVAL_DIR   branch_metrics.py results              (default ./outputs/main/branch_metrics)
#   OUT_DIR         collected CSV, tables, figures         (default ./outputs/main)
#   STEPS           subset of: extract train pmatch eval collect   (default: all)
#
# The Matryoshka / Gated cells are trained here with their final (selected) settings from
# configs/main_cells.json. To re-run the selection procedure that produced those settings
# (Gated lambda calibration, Dual gamma / warm-up sweep, winner picking), see
# scripts/main/stage_cells.py and the README.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISION_DIR="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python}"
export FEATURES_ROOT="${FEATURES_ROOT:-./features}"
export MAIN_CKPT_DIR="${MAIN_CKPT_DIR:-./checkpoints/main}"
PMATCH_CKPT_DIR="${PMATCH_CKPT_DIR:-./checkpoints/param_matched}"
CB_CKPT_DIR="${CB_CKPT_DIR:-./checkpoints/cb}"
export MAIN_EVAL_DIR="${MAIN_EVAL_DIR:-./outputs/main/branch_metrics}"
OUT_DIR="${OUT_DIR:-./outputs/main}"
STEPS="${STEPS:-extract train pmatch eval collect}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MPLBACKEND=Agg
has_step() { [[ " $STEPS " == *" $1 "* ]]; }

if has_step extract; then
  for model in clip siglip2 vit; do
    for split in train validation; do
      "$PY" "$VISION_DIR/extract_imagenet.py" --model "$model" --split "$split" \
        --output_dir "$FEATURES_ROOT/imagenet"
    done
    for split in train valid test; do
      "$PY" "$VISION_DIR/extract_open_images.py" --model "$model" --split "$split" \
        --dataset_dir "${DATA_ROOT:-./data}/open_images_v7" --output_dir "$FEATURES_ROOT/open_images"
    done
  done
fi

if has_step train; then
  "$PY" "$HERE/train_cells.py" --output_dir "$MAIN_CKPT_DIR" --features_root "$FEATURES_ROOT"
fi

if has_step pmatch; then
  "$PY" "$VISION_DIR/scripts/param_matched/train_param_matched.py" \
    --output_dir "$PMATCH_CKPT_DIR" --features_root "$FEATURES_ROOT"
fi

if has_step eval; then
  # Offline evaluation of every run (the TopK/BatchTopK/JumpReLU pairs and the parameter-matched
  # arms are reported from the trainers' own best_metrics.json; the offline results of those
  # runs are optional diagnostics, so they are evaluated too).
  runs=("$MAIN_CKPT_DIR"/* "$PMATCH_CKPT_DIR"/*)
  [ -d "$CB_CKPT_DIR" ] && runs+=("$CB_CKPT_DIR"/cbdual_*)
  "$PY" "$HERE/eval_runs.py" --runs "${runs[@]}" --out_dir "$MAIN_EVAL_DIR"
fi

if has_step collect; then
  # CB / CB-DUAL rows: from the CB pipeline's own collector (cb_sae/, CB_LONG) when present,
  # otherwise from the offline results of the cbdual_* runs in MAIN_EVAL_DIR.
  "$PY" "$VISION_DIR/analysis/main/collect_main.py" --ckpt_dir "$MAIN_CKPT_DIR" \
    --pmatch_dir "$PMATCH_CKPT_DIR" --eval_dir "$MAIN_EVAL_DIR" \
    --cb_long "${CB_LONG:-./outputs/cb_sae/cb_long.csv}" --out "$OUT_DIR/main_long.csv"
  "$PY" "$VISION_DIR/analysis/main/build_main_table.py" --long "$OUT_DIR/main_long.csv" \
    --latex "$OUT_DIR/main_table.tex" --markdown "$OUT_DIR/main_table.md"
  "$PY" "$VISION_DIR/analysis/main/build_param_matched_table.py" --long "$OUT_DIR/main_long.csv" \
    --latex "$OUT_DIR/param_matched_table.tex" --markdown "$OUT_DIR/param_matched_table.md"
  for ds in imagenet open_images; do
    "$PY" "$VISION_DIR/analysis/main/plot_radar.py" --long "$OUT_DIR/main_long.csv" --dataset "$ds" \
      --out "$OUT_DIR/radar_${ds}.pdf"
  done
fi
