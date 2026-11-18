"""OpenImages Stage-1a equivalent — build the hypogen data layer from the PRE-EXTRACTED feature/label
LMDBs (no GPU re-extraction needed; the features already exist).

Unlike ImageNet (extract_embeddings re-embeds on-disk ImageFolder images), OpenImages features are already
in LMDB but keyed by a bare integer index with NO stored image-id. We reconstruct index->image-id from each
split's localized-narratives file using the VERIFIED 'grouped' ordering (group by image-id in first-
occurrence order, expand one row per caption — the feature extractor's image -> captions iteration
order; check it on your copy with oi_verify_order.py, which should report cos~1.0). We then DEDUP to unique images (a few
images have >1 caption -> duplicate feature rows) taking each image's FIRST row, and emit the hypogen format:

  <role>_manifest.json   [{rel_path: "<image_id>.jpg"}]  (rel to config.IMAGE_ROOTS[role])
  <role>_labels.npy      float32 (N, 601) multi-hot   (config.labels_path(role))
  <encoder>/<role>_emb.npy   float16 (N, D) L2-NORMALIZED  (config.embeddings_path)

Role->split mapping (see config): role 'val' (faithful) <- OpenImages valid; role 'train' (selection +
interpretation sourcing) <- OpenImages test. L2-normalization matches LMDBFeatureDataset.__getitem__, which
unit-norms every feature on read at train time, so the SAEs saw unit-norm inputs -> the hypogen embeddings
must be unit-norm too (the raw LMDB features have norm ~21).

Run (CPU, offline): python -m hypogen.build_oi_data --encoder clip --role val
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict

import lmdb
import numpy as np

from hypogen import config

# OpenImages feature-LMDB filename per encoder (matches the pre-extracted files).
_FEAT_LMDB = {
    "clip": "clip_image_features_CLIP-ViT-L-14-DataComp.lmdb",
    "siglip": "siglip2_image_features_siglip2-so400m-p14-384.lmdb",
    "vit": "vit_features_vit-large-p16-224.lmdb",
}
# role -> on-disk OpenImages split that serves it (see config._DATASET_CFG openimages note).
_ROLE_SPLIT = {"val": "valid", "train": "test"}
# Pre-extracted OpenImagesV7 feature/label LMDBs: <FEATURES_OPENIMAGES>/<split>/<encoder lmdb> and
# <FEATURES_OPENIMAGES>/<split>/openimages_labels.lmdb; localized-narratives JSONL files from the
# OpenImages V7 release under OPENIMAGES_NARRATIVES_DIR.
_FEAT_ROOT = os.environ.get("FEATURES_OPENIMAGES", "./features/open_images")
_NARR_DIR = os.environ.get("OPENIMAGES_NARRATIVES_DIR", "./data/open_image_v7")
_NARR = {
    "valid": os.path.join(_NARR_DIR, "open_images_validation_localized_narratives.jsonl"),
    "test": os.path.join(_NARR_DIR, "open_images_test_localized_narratives.jsonl"),
}


def _atomic_save_npy(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _atomic_write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _grouped_first_row(split: str):
    """Return (unique_image_ids, first_row_index) under the verified 'grouped' ordering: image-ids in
    first-occurrence order, each mapped to the LMDB row index of its FIRST caption. The grouped expansion
    lists all of an image's caption-rows consecutively, so an image's first row index = cumulative count
    of all caption-rows for images that first-appeared before it."""
    counts = OrderedDict()  # image_id -> #captions, in first-occurrence order
    with open(_NARR[split], encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            iid = json.loads(line)["image_id"]
            counts[iid] = counts.get(iid, 0) + 1
    image_ids, first_idx, running = [], [], 0
    for iid, c in counts.items():
        image_ids.append(iid)
        first_idx.append(running)  # first row of this image in the grouped expansion
        running += c
    return image_ids, np.array(first_idx, dtype=np.int64), running


def build(encoder: str, role: str):
    split = _ROLE_SPLIT[role]
    emb_path = config.embeddings_path(encoder, role)
    lab_path = config.labels_path(role)
    man_path = config.manifest_path(role)

    image_ids, first_idx, n_rows = _grouped_first_row(split)
    n = len(image_ids)
    print(f"[oi {role}<-{split}/{encoder}] {n} unique images ({n_rows} caption-rows)")

    # Resume: skip if embeddings already complete for this encoder+role.
    if os.path.isfile(emb_path):
        e = np.load(emb_path, mmap_mode="r")
        if e.shape[0] == n:
            print(f"[oi {role}/{encoder}] embeddings already complete: {emb_path} {e.shape}")
            return
        print(f"[oi {role}/{encoder}] stale ({e.shape[0]} != {n}); rebuilding")

    # Manifest + multi-hot labels are encoder-independent: build once per role, reuse across encoders.
    if not os.path.isfile(man_path) or not os.path.isfile(lab_path):
        lab_lmdb = os.path.join(_FEAT_ROOT, split, "openimages_labels.lmdb")
        env = lmdb.open(lab_lmdb, readonly=True, lock=False)
        labels = np.zeros((n, config.NUM_CLASSES), dtype=np.float32)
        with env.begin() as t:
            for i, ridx in enumerate(first_idx):
                v = t.get(str(int(ridx)).encode("ascii"))
                if v is None:
                    raise KeyError(f"label row {ridx} missing in {lab_lmdb}")
                row = np.frombuffer(v, dtype=np.float32)
                assert row.shape[0] == config.NUM_CLASSES, f"label dim {row.shape[0]} != {config.NUM_CLASSES}"
                labels[i] = (row > 0).astype(np.float32)  # binarize multi-hot (defensive; values are 0/1)
        env.close()
        _atomic_save_npy(labels, lab_path)
        _atomic_write_json([{"rel_path": f"{iid}.jpg"} for iid in image_ids], man_path)
        print(f"[oi {role}] wrote manifest + multi-hot labels {labels.shape} "
              f"(avg {labels.sum(1).mean():.2f} pos/img, {int((labels.sum(0) > 0).sum())} classes present)")

    # Features: read each unique image's first row, L2-normalize (training read convention), fp16.
    feat_lmdb = os.path.join(_FEAT_ROOT, split, _FEAT_LMDB[encoder])
    env = lmdb.open(feat_lmdb, readonly=True, lock=False)
    with env.begin() as t:
        d = np.frombuffer(t.get(b"0"), dtype=np.float32).shape[0]
        emb = np.empty((n, d), dtype=np.float16)
        for i, ridx in enumerate(first_idx):
            v = t.get(str(int(ridx)).encode("ascii"))
            x = np.frombuffer(v, dtype=np.float32)
            nrm = np.linalg.norm(x)
            emb[i] = (x / (nrm + 1e-8)).astype(np.float16)  # unit-norm (matches LMDBFeatureDataset)
            if i % 20000 == 0:
                print(f"  [{role}/{encoder}] {i}/{n}", flush=True)
    env.close()
    _atomic_save_npy(emb, emb_path)
    print(f"[oi {role}/{encoder}] saved embeddings {emb.shape} (d={d}) -> {emb_path}")


def main():
    ap = argparse.ArgumentParser(description="Build OpenImages hypogen data layer from pre-extracted LMDBs.")
    ap.add_argument("--encoder", required=True, choices=config.ENCODERS)
    ap.add_argument("--role", required=True, choices=["train", "val"])
    a = ap.parse_args()
    assert config.DATASET == "openimages", "build_oi_data requires HG_DATASET=openimages"
    build(a.encoder, a.role)


if __name__ == "__main__":
    main()
