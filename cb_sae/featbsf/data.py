"""LMDB feature loading for the CB-SAE pipeline.

Thin wrapper around vision/dataset/lmdb_features.py:LMDBFeatureDataset — the
SAME dataset class the SAE / DualSAE bases trained on — so the CB stages see
byte-identical inputs.

CRITICAL invariant: LMDBFeatureDataset L2-NORMALIZES every feature vector at
load time. That unit-norm space IS the
paper's input space; nothing downstream may renormalize inputs or "undo" it.

Label conventions (verified against lmdb_features.py):
  * imagenet: per-sample int class ids from __label__{idx} keys (or the
    dataset class's offline label-file fallback) -> int64 after collate.
  * open_images: features LMDB has no labels; a companion
    <split>/openimages_labels.lmdb stores float32 multi-hot vectors. With
    return_multilabel=True the raw multi-hot comes back; with False it is
    collapsed to the smallest active class index (degenerate single-label —
    only meaningful for quick checks, the eval uses multilabel).

Worker safety: LMDBFeatureDataset reads metadata in the parent process, then
opens its lmdb env LAZILY inside __getitem__ (_open_env), so forked DataLoader
workers each get their own handle — no worker_init_fn needed. This mirrors how
the vision/ trainers use it.
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from featbsf import config
from featbsf._saepath import ensure_sae_on_path

# vision/ modules import each other by bare top-level names; put vision/ on sys.path
# BEFORE importing them (single shared shim, see _saepath.py).
ensure_sae_on_path()
from dataset.lmdb_features import LMDBFeatureDataset  # noqa: E402

_LABEL_KINDS = ("none", "class", "multilabel")

# The known number of OpenImages-V7 label classes in the companion label LMDB;
# num_classes() reads the real width from disk and cross-checks against this.
_OPEN_IMAGES_CLASSES_EXPECTED = 601


def _resolve_lmdb(dataset: str, encoder: str, split: str) -> str:
    """LMDB path lookup with actionable errors (a bare KeyError from a batch
    job wastes a queue slot; name the valid options instead)."""
    try:
        per_enc = config.LMDB_PATHS[dataset]
    except KeyError:
        raise ValueError(f"unknown dataset {dataset!r}; "
                         f"expected one of {sorted(config.LMDB_PATHS)}") from None
    try:
        per_split = per_enc[encoder]
    except KeyError:
        raise ValueError(f"unknown encoder {encoder!r} for {dataset}; "
                         f"expected one of {sorted(per_enc)}") from None
    try:
        return per_split[split]
    except KeyError:
        raise ValueError(f"unknown split {split!r} for {dataset}/{encoder}; "
                         f"expected one of {sorted(per_split)}") from None


def feature_dim(dataset: str, encoder: str) -> int:
    """Input feature dimensionality (from the pinned FEATURE_DIM table)."""
    if dataset not in config.DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; "
                         f"expected one of {config.DATASETS}")
    if encoder not in config.FEATURE_DIM:
        raise ValueError(f"unknown encoder {encoder!r}; "
                         f"expected one of {tuple(config.FEATURE_DIM)}")
    return config.FEATURE_DIM[encoder]


# num_classes() opens an LMDB to read metadata — cache so repeated calls
# (metrics assemble it per run) touch disk once per process.
_NUM_CLASSES_CACHE: dict[str, int] = {}


def num_classes(dataset: str) -> int:
    """Number of label classes for a dataset.

    imagenet is a fixed 1000-way single-label problem. open_images' class
    count is read from the companion label LMDB's stored vector width (the
    on-disk source of truth), not hardcoded — with a loud warning if it ever
    deviates from the expected 601 so a swapped label store can't silently
    change every purity/KNN/LP number.
    """
    if dataset == "imagenet":
        return 1000
    if dataset != "open_images":
        raise ValueError(f"unknown dataset {dataset!r}; "
                         f"expected one of {config.DATASETS}")
    if dataset in _NUM_CLASSES_CACHE:
        return _NUM_CLASSES_CACHE[dataset]
    # Any encoder's train split works: the companion openimages_labels.lmdb
    # lives in the split DIRECTORY, shared by all encoders. Constructing the
    # dataset only reads LMDB metadata (O(1), no feature rows).
    ds = LMDBFeatureDataset(_resolve_lmdb(dataset, "clip", "train"),
                            return_class_idx=True, return_multilabel=True,
                            verbose=False)
    try:
        if ds.labels_feat_shape is None:
            raise RuntimeError(
                "open_images companion label LMDB (openimages_labels.lmdb) "
                "not found next to the features LMDB — cannot determine the "
                "multilabel class count")
        n = int(ds.labels_feat_shape[-1])
    finally:
        ds.close()
    if n != _OPEN_IMAGES_CLASSES_EXPECTED:
        print(f"[featbsf.data] WARNING: open_images label width {n} != "
              f"expected {_OPEN_IMAGES_CLASSES_EXPECTED}; using on-disk value")
    _NUM_CLASSES_CACHE[dataset] = n
    return n


def build_loader(dataset: str, encoder: str, split: str, batch_size: int,
                 labels: str = "none", shuffle: bool = False,
                 num_workers: int = 4, drop_last: bool = False,
                 generator: torch.Generator | None = None) -> DataLoader:
    """DataLoader over pre-extracted (L2-normalized) LMDB features.

    labels:
      'none'       -> each batch is a plain (B, d) float32 tensor
      'class'      -> (x, class_idx int64). imagenet: true class ids.
                      open_images: multi-hot collapsed to smallest active
                      index (degenerate; prefer 'multilabel' there).
      'multilabel' -> (x, multihot float32 (B, C)). open_images only —
                      imagenet's LMDB stores scalar ids, so this mode would
                      silently return ints; rejected up front instead.

    generator: optional torch.Generator for the shuffling sampler. The paper
    trainer seeds a fresh one per epoch (seed*10000+epoch) so a mid-epoch
    resume can fast-forward a bit-identical batch stream (plain shuffle=True
    is NOT deterministic across process restarts otherwise).

    No renormalization happens here or anywhere downstream: the dataset class
    already L2-normalizes per sample (see module docstring).
    """
    if labels not in _LABEL_KINDS:
        raise ValueError(f"labels={labels!r}; expected one of {_LABEL_KINDS}")
    if labels == "multilabel" and dataset != "open_images":
        raise ValueError(
            "labels='multilabel' is only valid for open_images; imagenet "
            "labels are scalar class ids (return_multilabel is a no-op there)")

    path = _resolve_lmdb(dataset, encoder, split)
    # return_class_idx drives BOTH label modes; return_multilabel only changes
    # what the companion-LMDB branch yields (raw multi-hot vs collapsed index).
    ds = LMDBFeatureDataset(
        path,
        return_class_idx=(labels != "none"),
        return_multilabel=(labels == "multilabel"),
        verbose=False,
    )

    # Sanity: the LMDB's stored feature width must match the pinned table —
    # catches pointing an encoder at the wrong store before hours of training.
    d = feature_dim(dataset, encoder)
    if ds.get_feature_dim() != d:
        raise RuntimeError(f"{path} stores dim {ds.get_feature_dim()}, but "
                           f"FEATURE_DIM[{encoder!r}]={d}")

    # pin_memory=True mirrors the vision/ trainers (same host->GPU copy path).
    # Default collate handles all three label modes: float32 arrays stack to
    # float32 tensors, python ints to int64 — verified against
    # lmdb_features.py's return types.
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        generator=generator,
    )
