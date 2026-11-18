"""Stage 1a — re-extract image embeddings from the on-disk ImageFolder images.

WHY re-extract instead of reading the pre-extracted ImageNet feature LMDB: that LMDB's row order
follows the dataset-loader order it was extracted from (not ImageFolder order) and stores no image
paths. To ground every activation in a KNOWN image file (required for the VLM steps), we instead
embed the on-disk images directly, in ImageFolder
order, with the exact encoder the SAE was trained on (config.ENCODER_MODELS, reused via
prepare_topic_data.build_encoder) and the same L2-normalization. row i <-> ImageFolder.samples[i].

Outputs (under config.EMB_DIR), resumable + atomic:
  <split>_manifest.json   ordered [{rel_path, label}, ...]  (encoder-independent; written once)
  <split>_labels.npy      int16 [n] class labels aligned to the manifest rows
  <encoder>/<split>_emb.npy   float16 [n, D] L2-normalized embeddings aligned to the manifest rows

For 'train' we keep a fixed, deterministic stratified subsample (config.TRAIN_SUBSAMPLE_PER_CLASS per
class) — identical across encoders so cells compare on the same images. 'val' uses all images.

Run (GPU node): python -m hypogen.extract_embeddings --encoder clip --split val
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from hypogen import config
from prepare_topic_data import build_encoder  # reuse the exact training-time encoder construction


def _atomic_save_npy(arr: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _atomic_write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def build_manifest(split: str) -> list:
    """Deterministic ordered [(rel_path, label)] for a split.

    'val'  -> every ImageFolder sample, in ImageFolder order.
    'train'-> a fixed stratified subsample of TRAIN_SUBSAMPLE_PER_CLASS per class (seeded), so the
              chosen images are identical across encoders and runs.
    """
    root = config.IMAGE_ROOTS[split]
    folder = ImageFolder(root)  # scans <root>/<wnid>/*; samples sorted by (class, filename)
    samples = folder.samples    # list of (abs_path, class_idx); class_idx = sorted-wnid order = std ImageNet idx

    if split == "train":
        per_class = config.TRAIN_SUBSAMPLE_PER_CLASS
        by_class = {}
        for i, (_, c) in enumerate(samples):
            by_class.setdefault(c, []).append(i)
        rng = np.random.default_rng(config.SUBSAMPLE_SEED)
        chosen = []
        for c in sorted(by_class):
            idxs = np.array(by_class[c])
            rng.shuffle(idxs)
            chosen.extend(idxs[:per_class].tolist())
        chosen.sort()  # stable row order
        selected = [samples[i] for i in chosen]
    else:
        selected = samples

    manifest = [{"rel_path": os.path.relpath(p, root), "label": int(c)} for (p, c) in selected]
    return manifest


class _PathDataset(Dataset):
    """Open + preprocess each manifest image (label unused here)."""

    def __init__(self, manifest, root, preprocess):
        self.manifest, self.root, self.preprocess = manifest, root, preprocess

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, i):
        rel = self.manifest[i]["rel_path"]
        img = Image.open(os.path.join(self.root, rel)).convert("RGB")
        return self.preprocess(img)


@torch.no_grad()
def extract(encoder: str, split: str, batch_size: int, num_workers: int, device: str) -> None:
    root = config.IMAGE_ROOTS[split]
    man_path = config.manifest_path(split)
    lab_path = config.labels_path(split)
    emb_path = config.embeddings_path(encoder, split)

    # Manifest + labels are encoder-independent: build once, reuse (idempotent). If a manifest already
    # exists, trust it (so every encoder embeds the SAME images in the SAME order).
    if os.path.isfile(man_path):
        with open(man_path) as f:
            manifest = json.load(f)
        print(f"[{split}] reusing manifest ({len(manifest)} rows): {man_path}")
    else:
        manifest = build_manifest(split)
        _atomic_write_json(manifest, man_path)
        _atomic_save_npy(np.array([m["label"] for m in manifest], dtype=np.int16), lab_path)
        print(f"[{split}] wrote manifest + labels ({len(manifest)} rows)")

    n = len(manifest)
    # Resume: skip if embeddings already present with the right row count.
    if os.path.isfile(emb_path):
        existing = np.load(emb_path, mmap_mode="r")
        if existing.shape[0] == n:
            print(f"[{encoder}/{split}] embeddings already complete: {emb_path} {existing.shape}")
            return
        print(f"[{encoder}/{split}] stale embeddings ({existing.shape[0]} != {n}); recomputing")

    print(f"[{encoder}/{split}] loading encoder {config.ENCODER_MODELS[encoder]} on {device}")
    preprocess, embed_fn = build_encoder(encoder, config.ENCODER_MODELS[encoder], device)
    loader = DataLoader(_PathDataset(manifest, root, preprocess), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=(device != "cpu"))

    feats = []
    use_cuda = device != "cpu"
    for bi, imgs in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=use_cuda):
            f = embed_fn(imgs)
        f = F.normalize(f.float(), p=2, dim=-1)  # L2-norm: training convention (matches LMDB reads)
        feats.append(f.to(torch.float16).cpu())
        if bi % 50 == 0:
            print(f"  [{encoder}/{split}] batch {bi}/{len(loader)}", flush=True)
    emb = torch.cat(feats, dim=0).numpy()
    assert emb.shape[0] == n, f"row mismatch: {emb.shape[0]} != {n}"
    _atomic_save_npy(emb, emb_path)
    print(f"[{encoder}/{split}] saved embeddings {emb.shape} -> {emb_path}")


def main():
    ap = argparse.ArgumentParser(description="Stage 1a: re-extract per-(encoder,split) embeddings.")
    ap.add_argument("--encoder", required=True, choices=config.ENCODERS)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    extract(args.encoder, args.split, args.batch_size, args.num_workers, args.device)


if __name__ == "__main__":
    main()
