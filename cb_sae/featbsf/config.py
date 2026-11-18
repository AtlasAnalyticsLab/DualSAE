"""Paths and protocol constants shared by the CB-SAE pipeline.

Single source of truth for where the CB stages read features from and write
their artifacts to, plus the training-protocol constants the CB head inherits
from the DualSAE runs it is built on.

This module deliberately imports ONLY the stdlib (no torch/numpy) so it is
importable everywhere (drivers, collectors, tests).

Every location is configurable through environment variables; the defaults
are relative to the directory the drivers are launched from (the repository
root) and are made absolute at import time, so a stage that changes its
working directory still resolves the same files:

  FEATURES_ROOT   pre-extracted feature LMDBs (default ./features), laid out as
                  $FEATURES_ROOT/<dataset>/<split dir>/<file name> -- the same
                  layout vision/ uses (vision/scripts/ablations/feature_paths.py).
  CB_LMDB_PATHS   optional JSON file {dataset: {encoder: {split: path}}} whose
                  entries override the layout above (any subset).
  CB_CKPT_DIR     model checkpoints (default ./checkpoints/cb): converted
                  bases (cvt_*), pruned bases (pruned_*), CB runs (cbdual_*).
  CB_WORK_DIR     every other CB output (default ./outputs/cb_sae): dissect +
                  prune artifacts, embedding caches, logs, TensorBoard event
                  files, evaluation results, collected CSVs.
"""
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------
# cb_sae/ (the directory holding the featbsf/ and bsf/ packages).
CB_SAE_ROOT = Path(__file__).resolve().parent.parent
# Repository root (holds vision/, whose model + dataset modules the CB
# pipeline imports; see featbsf/_saepath.py).
REPO_ROOT = CB_SAE_ROOT.parent

FEATURES_ROOT = Path(os.environ.get("FEATURES_ROOT", "./features")).resolve()
WORK_DIR = Path(os.environ.get("CB_WORK_DIR", "./outputs/cb_sae")).resolve()

# Converted bases, pruned bases, CB runs (one dir per run).
CKPT_DIR = Path(os.environ.get("CB_CKPT_DIR", "./checkpoints/cb")).resolve()
LOG_DIR = WORK_DIR / "logs"                  # driver stdout/stderr
TB_DIR = WORK_DIR / "tensorboard"            # TensorBoard event files

_ALL_DIRS = (WORK_DIR, CKPT_DIR, LOG_DIR, TB_DIR)


def ensure_dirs() -> None:
    """Create every output directory (idempotent)."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Experiment axes.
# ---------------------------------------------------------------------------
DATASETS = ("imagenet", "open_images")
ENCODERS = ("clip", "vit", "sigclip")

# Input feature dimensionality per encoder (fixed by the pre-extracted LMDBs:
# CLIP ViT-L/14 -> 768, ViT-L/16 -> 1024, SigLIP2 so400m -> 1152).
FEATURE_DIM = {"clip": 768, "vit": 1024, "sigclip": 1152}

# ---------------------------------------------------------------------------
# LMDB feature stores. Split KEYS follow the on-disk split directory names:
# ImageNet-1K 'validation' (no labelled test split), OpenImagesV7 'valid' and
# 'test'. The OpenImagesV7 multi-hot labels live in a companion
# 'openimages_labels.lmdb' inside each split directory (read by the dataset
# class and by the alignment gate in cb_text_embeds.py).
# ---------------------------------------------------------------------------
_SPLIT_DIRS = {
    "imagenet": ("train", "validation"),
    "open_images": ("train", "valid", "test"),
}
_FILE_NAMES = {
    "clip": "clip_image_features_CLIP-ViT-L-14-DataComp.lmdb",
    "vit": "vit_features_vit-large-p16-224.lmdb",
    "sigclip": "siglip2_image_features_siglip2-so400m-p14-384.lmdb",
}


def _build_lmdb_paths() -> dict:
    paths = {ds: {enc: {split: str(FEATURES_ROOT / ds / split / fname)
                        for split in splits}
                  for enc, fname in _FILE_NAMES.items()}
             for ds, splits in _SPLIT_DIRS.items()}
    override = os.environ.get("CB_LMDB_PATHS")
    if override:
        # Partial overrides are merged entry by entry so a user can point one
        # store elsewhere without restating the whole table.
        for ds, per_enc in json.loads(Path(override).read_text()).items():
            for enc, per_split in per_enc.items():
                paths.setdefault(ds, {}).setdefault(enc, {}).update(
                    {s: str(Path(p).resolve()) for s, p in per_split.items()})
    return paths


LMDB_PATHS = _build_lmdb_paths()

# Which split the parity check of the converters reads (the evaluation split:
# ImageNet validation; OpenImagesV7 'valid').
EVAL_SPLIT = {"imagenet": "validation", "open_images": "valid"}

# ---------------------------------------------------------------------------
# Training-protocol constants inherited by the CB head (the DualSAE runs'
# values: batch 2048, constant LR 1e-4, decorrelation coefficient 0.01).
# ---------------------------------------------------------------------------
BATCH = 2048
LR = 1e-4
LAM_SPAN = 0.01           # span-overlap (decorrelation, L_dcor) coefficient gamma
