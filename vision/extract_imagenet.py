#!/usr/bin/env python3
"""Extract frozen-encoder features of ImageNet-1K and save them to LMDB.

Encoders (``--model``):

    clip     laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K (open_clip), ``encode_image``     d=768
    siglip2  google/siglip2-so400m-patch14-384, ``get_image_features``                   d=1152
    vit      google/vit-large-patch16-224 (supervised), CLS token of last_hidden_state   d=1024
    dinov2   facebook/dinov2-large, CLS token (available, not used by the paper)

Images come from the HuggingFace ImageNet-1K dataset (``--hf_dataset``, default
``ILSVRC/imagenet-1k``). Sample ``i`` of the LMDB is sample ``i`` of the HF split, and its
integer class id is stored under ``__label__{i}`` (ASCII), which is what
``dataset.lmdb_features.LMDBFeatureDataset`` reads for purity / KNN / LP. (The class ids were
originally written by a separate pass over the same HF split in the same order; writing them
here produces the same keys.)

Runs fully offline: the dataset and the encoder weights are read from the local HuggingFace
cache, which ``prepare_assets.sh`` fills on a machine with internet access. Set ``HF_HOME`` to
the same directory for both steps.

Output: ``<output_dir>/<split>/<encoder>_..._<tag>.lmdb`` (default ``--output_dir
./features/imagenet``), the layout ``main.py --features_root ./features`` expects.

Memory: features are accumulated in RAM and written to LMDB in one transaction at the end, so
peak memory is roughly ``num_samples * feature_dim * 4`` bytes (~5 GB for the train split at
1024 dims). Extract one split at a time.
"""

import argparse
import gc
import os
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Offline by default: everything must already be in the local cache (see prepare_assets.sh).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# (model id, output tag) per encoder. The tag is part of the LMDB file name.
MODEL_CONFIGS = {
    'dinov2': ('facebook/dinov2-large', 'dinov2-large'),
    'clip': ('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', 'CLIP-ViT-L-14-DataComp'),
    'siglip2': ('google/siglip2-so400m-patch14-384', 'siglip2-so400m-p14-384'),
    'vit': ('google/vit-large-patch16-224', 'vit-large-p16-224'),
}
IMAGENET_HF_ID = "ILSVRC/imagenet-1k"


class ImageNetHFDataset(Dataset):
    """Wrapper around a HuggingFace ImageNet-1K split: (image, label, idx)."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')  # some ImageNet images are grayscale / CMYK
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': label, 'idx': idx}


def pil_collate(batch):
    """Keep PIL images (for encoders whose HF processor does its own preprocessing)."""
    return {
        'image': [b['image'] for b in batch],
        'label': torch.tensor([b['label'] for b in batch], dtype=torch.long),
        'idx': torch.tensor([b['idx'] for b in batch], dtype=torch.long),
    }


def write_to_lmdb(lmdb_path, features_dict, feat_shape, labels_dict):
    """Write raw float32 features (+ ``__label__`` keys) in LMDBFeatureDataset's format."""
    num_samples = len(features_dict)
    sample_size = np.prod(feat_shape) * 4  # float32
    map_size = int(sample_size * num_samples * 2.5 + 1024 ** 3)  # B-tree overhead + 1 GB
    env = lmdb.open(lmdb_path, map_size=map_size)
    with env.begin(write=True) as txn:
        for idx, feat in tqdm(features_dict.items(), desc="Writing to LMDB"):
            txn.put(str(idx).encode('ascii'), feat.astype(np.float32).tobytes())
            label = labels_dict.get(int(idx))
            if label is not None:
                txn.put(f'__label__{idx}'.encode('ascii'), str(int(label)).encode('ascii'))
        txn.put(b'__len__', str(num_samples).encode('ascii'))
        txn.put(b'__shape__', str(feat_shape).encode('ascii'))
    env.close()
    print(f"Saved {num_samples} samples to {lmdb_path}")


def extract_features_streaming(loader, model_fn, desc):
    """Run the encoder over the loader; returns ({idx: feature}, {idx: label}, feature shape)."""
    features, labels, feat_shape = {}, {}, None
    for batch in tqdm(loader, desc=desc):
        idxs = batch['idx'].numpy()
        with torch.no_grad():
            feats = model_fn(batch)
        if feat_shape is None:
            feat_shape = tuple(feats.shape[1:])
        for i, idx in enumerate(idxs):
            features[int(idx)] = feats[i]
            labels[int(idx)] = int(batch['label'][i])
    return features, labels, feat_shape


def resolve_hf_cache_path(model_name):
    """Local snapshot directory of a cached HF model (avoids any network call offline)."""
    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    hub = os.environ.get('HF_HUB_CACHE', os.path.join(hf_home, 'hub'))
    snapshot_dir = os.path.join(hub, f"models--{model_name.replace('/', '--')}", 'snapshots')
    if os.path.isdir(snapshot_dir):
        snapshots = sorted(os.listdir(snapshot_dir))
        if snapshots:
            return os.path.join(snapshot_dir, snapshots[-1])
    return model_name  # let from_pretrained resolve it (works when online or already cached)


def _image_features_tensor(out):
    """``get_image_features`` returns a tensor in older transformers releases and an output
    object whose ``pooler_output`` holds the same projected embedding in newer ones."""
    return out.pooler_output if hasattr(out, 'pooler_output') else out


def build_model_fn(model_key, model_name, device):
    """(model_fn, dataset transform or None, collate_fn) for one encoder."""
    if model_key == 'clip':
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        model = model.to(device).eval()

        def model_fn(batch):
            imgs = batch['image'].to(device)
            with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                feats = model.encode_image(imgs)
            return feats.cpu().float().numpy()
        return model, model_fn, preprocess, None

    from transformers import AutoImageProcessor, AutoModel, Dinov2Model
    local = resolve_hf_cache_path(model_name)
    processor = AutoImageProcessor.from_pretrained(local)
    if model_key == 'siglip2':
        model = AutoModel.from_pretrained(local).to(device).eval()

        def model_fn(batch):
            inputs = processor(images=batch['image'], return_tensors='pt')
            with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                feats = _image_features_tensor(
                    model.get_image_features(pixel_values=inputs['pixel_values'].to(device)))
            return feats.cpu().float().numpy()
    else:  # vit / dinov2: CLS token of the last hidden state
        model = (Dinov2Model if model_key == 'dinov2' else AutoModel).from_pretrained(local)
        model = model.to(device).eval()

        def model_fn(batch):
            inputs = processor(images=batch['image'], return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                feats = model(**inputs).last_hidden_state[:, 0, :]
            return feats.cpu().float().numpy()
    return model, model_fn, None, pil_collate


LMDB_PREFIX = {'clip': 'clip_image_features', 'siglip2': 'siglip2_image_features',
               'vit': 'vit_features', 'dinov2': 'dino_features'}


def main():
    parser = argparse.ArgumentParser(description="Extract ImageNet-1K features to LMDB")
    parser.add_argument('--model', choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--output_dir', type=str, default='./features/imagenet')
    parser.add_argument('--hf_dataset', type=str, default=IMAGENET_HF_ID,
                        help='HF dataset id (read from the local HF cache)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    from datasets import load_dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name, tag = MODEL_CONFIGS[args.model]
    lmdb_path = out_dir / f'{LMDB_PREFIX[args.model]}_{tag}.lmdb'
    if (lmdb_path / 'data.mdb').exists():
        print(f"[skip] {lmdb_path} exists")
        return

    print(f"Loading {args.hf_dataset} [{args.split}] from the local HF cache...")
    hf_dataset = load_dataset(args.hf_dataset, split=args.split)
    print(f"Loaded {len(hf_dataset)} samples")

    model, model_fn, transform, collate = build_model_fn(args.model, model_name, device)
    loader = DataLoader(ImageNetHFDataset(hf_dataset, transform), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    features, labels, feat_shape = extract_features_streaming(loader, model_fn, f'{args.model} ({tag})')
    # Write to a temporary directory first and rename, so an interrupted write never leaves a
    # half-filled LMDB that the skip-guard above would later accept.
    tmp_path = out_dir / f'.{lmdb_path.name}.tmp'
    write_to_lmdb(str(tmp_path), features, feat_shape, labels)
    os.replace(tmp_path, lmdb_path)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("Done!")


if __name__ == "__main__":
    main()
