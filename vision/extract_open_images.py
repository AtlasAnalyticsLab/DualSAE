#!/usr/bin/env python3
"""Extract frozen-encoder features of OpenImagesV7 and save them to LMDB.

Expects a local copy of OpenImagesV7 under ``--dataset_dir`` with the layout read by
``dataset/open_images.py`` (built by ``prepare_open_images.py``)::

    <dataset_dir>/images/<split>/<image_id>.jpg
    <dataset_dir>/captions/<split>/simplified_open_images_<split>_localized_narratives.json
    <dataset_dir>/labels/oidv7-class-descriptions-boxable.csv
    <dataset_dir>/labels/<split>-annotations-human-imagelabels-boxable.csv

with ``<split>`` in {train, valid, test}. One sample is one (image, Localized-Narratives caption)
pair, so an image with several captions appears several times; only images that have a
Localized-Narratives caption are included. The image-level labels (human-verified, Confidence ==
1) become a 601-dimensional multi-hot target.

Each run writes, under ``<output_dir>/<split>/``:

* ``<encoder>_..._<tag>.lmdb`` -- the features (raw float32), indexed like the dataset samples;
* ``openimages_labels.lmdb``    -- the companion multi-hot label LMDB (written once per split;
  it depends only on the annotations). ``LMDBFeatureDataset`` finds it next to the feature
  LMDB and switches to multi-label evaluation.

Encoders (``--model``): clip (open_clip ViT-L-14, DataComp.XL weights, ``encode_image``),
siglip2 (``get_image_features``), vit (CLS token), dinov2 (torch.hub; not used by the paper).

Runs fully offline from the local HuggingFace cache filled by ``prepare_assets.sh``.
"""

import argparse
import gc
import json
import os
from pathlib import Path

import lmdb
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.open_images import OpenImagesDataset

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

CLIP_HF_REPO_ID = 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K'
# (model name, output tag) per encoder.
MODEL_CONFIGS = {
    'dinov2': ('dinov2_vitl14_reg', 'dinov2_vitl14_reg'),
    'clip': ('ViT-L-14', 'CLIP-ViT-L-14-DataComp'),
    'siglip2': ('google/siglip2-so400m-patch14-384', 'siglip2-so400m-p14-384'),
    'vit': ('google/vit-large-patch16-224', 'vit-large-p16-224'),
}
LMDB_PREFIX = {'clip': 'clip_image_features', 'siglip2': 'siglip2_image_features',
               'vit': 'vit_features', 'dinov2': 'dino_features'}


def _hub_cache() -> Path:
    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    return Path(os.environ.get('HF_HUB_CACHE', os.path.join(hf_home, 'hub'))).resolve()


def resolve_cached_open_clip_checkpoint(repo_id):
    """Path of the cached open_clip checkpoint file for ``repo_id`` (no network access)."""
    snapshots_dir = _hub_cache() / f"models--{repo_id.replace('/', '--')}" / 'snapshots'
    if not snapshots_dir.exists():
        raise FileNotFoundError(f'Model cache not found at {snapshots_dir}; run prepare_assets.sh')
    for pattern in ('open_clip_pytorch_model.bin', 'open_clip_model.safetensors',
                    'pytorch_model.bin', 'model.safetensors'):
        matches = sorted(snapshots_dir.glob(f'*/{pattern}'))
        if matches:
            return matches[-1].resolve()
    raise FileNotFoundError(f'No open_clip checkpoint found under {snapshots_dir}')


def resolve_cached_hf_snapshot_dir(repo_id):
    """Cached snapshot directory of a HuggingFace model (no network access)."""
    snapshots_dir = _hub_cache() / f"models--{repo_id.replace('/', '--')}" / 'snapshots'
    if not snapshots_dir.exists():
        raise FileNotFoundError(f'Model cache not found at {snapshots_dir}; run prepare_assets.sh')
    snapshot_dirs = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
    if not snapshot_dirs:
        raise FileNotFoundError(f'No snapshot directories found under: {snapshots_dir}')
    return snapshot_dirs[-1].resolve()


def pil_collate(batch):
    """Collate only the fields the extractors need (labels_text is ragged)."""
    return {'image': [b['image'] for b in batch],
            'idx': torch.tensor([b['idx'] for b in batch], dtype=torch.long)}


def tensor_collate(batch):
    return {'image': torch.stack([b['image'] for b in batch]),
            'idx': torch.tensor([b['idx'] for b in batch], dtype=torch.long)}


def write_to_lmdb(lmdb_path, features_dict, feat_shape, sample_paths=None):
    """Raw float32 features in LMDBFeatureDataset's format (+ optional ``__path__`` keys)."""
    num_samples = len(features_dict)
    sample_size = np.prod(feat_shape) * 4
    map_size = int(sample_size * num_samples * 2.5 + 1024 ** 3)
    env = lmdb.open(str(lmdb_path), map_size=map_size)
    with env.begin(write=True) as txn:
        for idx, feat in tqdm(features_dict.items(), desc="Writing to LMDB"):
            txn.put(str(idx).encode('ascii'), feat.astype(np.float32).tobytes())
            if sample_paths is not None and int(idx) in sample_paths:
                txn.put(f'__path__{idx}'.encode('ascii'), str(sample_paths[int(idx)]).encode('utf-8'))
        txn.put(b'__len__', str(num_samples).encode('ascii'))
        txn.put(b'__shape__', str(feat_shape).encode('ascii'))
    env.close()
    print(f"Saved {num_samples} samples to {lmdb_path}")


def write_labels_to_lmdb(lmdb_path, dataset):
    """Per-sample multi-hot targets, indexed by dataset sample idx (+ ``__classes__`` names)."""
    num_samples = len(dataset)
    feat_shape = (dataset.num_classes,)
    map_size = int(np.prod(feat_shape) * 4 * num_samples * 2.5 + 1024 ** 3)
    env = lmdb.open(str(lmdb_path), map_size=map_size)
    zero_label = np.zeros(dataset.num_classes, dtype=np.float32)
    with env.begin(write=True) as txn:
        for idx, (image_id, _) in tqdm(enumerate(dataset.samples), total=num_samples,
                                       desc="Writing labels to LMDB"):
            label_tensor = dataset.image_to_label_tensor.get(image_id)
            label_np = label_tensor.numpy() if label_tensor is not None else zero_label
            txn.put(str(idx).encode('ascii'), label_np.astype(np.float32).tobytes())
        txn.put(b'__len__', str(num_samples).encode('ascii'))
        txn.put(b'__shape__', str(feat_shape).encode('ascii'))
        txn.put(b'__classes__', json.dumps(dataset.all_classes).encode('utf-8'))
    env.close()
    print(f"Saved labels for {num_samples} samples to {lmdb_path}")


def extract_features_streaming(loader, model_fn, desc):
    features, feat_shape = {}, None
    for batch in tqdm(loader, desc=desc):
        idxs = batch['idx'].numpy()
        with torch.no_grad():
            feats = model_fn(batch)
        if feat_shape is None:
            feat_shape = tuple(feats.shape[1:])
        for i, idx in enumerate(idxs):
            features[int(idx)] = feats[i]
    return features, feat_shape


def _image_features_tensor(out):
    """Tensor in older transformers releases, output object (pooler_output) in newer ones."""
    return out.pooler_output if hasattr(out, 'pooler_output') else out


_IMAGENET_MEAN, _IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DINO_TRANSFORM = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def build_encoder(model_key, model_name, device):
    """(model, model_fn, dataset transform, collate)."""
    if model_key == 'clip':
        import open_clip
        ckpt = resolve_cached_open_clip_checkpoint(CLIP_HF_REPO_ID)
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=str(ckpt))
        model = model.to(device).eval()

        def model_fn(batch):
            img = torch.stack([preprocess(im) for im in batch['image']]).to(device)
            with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                feats = model.encode_image(img)
            return feats.cpu().float().numpy()
        return model, model_fn, None, pil_collate

    if model_key == 'dinov2':
        model = torch.hub.load('facebookresearch/dinov2', model_name).to(device).eval()

        def model_fn(batch):
            with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                feats = model(batch['image'].to(device))
            return feats.cpu().float().numpy()
        return model, model_fn, DINO_TRANSFORM, tensor_collate

    from transformers import AutoImageProcessor, AutoModel, AutoProcessor
    snapshot = resolve_cached_hf_snapshot_dir(model_name)
    if model_key == 'siglip2':
        model = AutoModel.from_pretrained(str(snapshot), local_files_only=True).to(device).eval()
        processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)

        def model_fn(batch):
            inputs = processor(images=batch['image'], return_tensors='pt')
            with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                feats = _image_features_tensor(
                    model.get_image_features(pixel_values=inputs['pixel_values'].to(device)))
            return feats.cpu().float().numpy()
        return model, model_fn, None, pil_collate

    # vit: CLS token of the last hidden state (no pooling layer).
    processor = AutoImageProcessor.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), add_pooling_layer=False,
                                      local_files_only=True).to(device).eval()

    def model_fn(batch):
        inputs = processor(images=batch['image'], return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            feats = model(**inputs).last_hidden_state[:, 0, :]
        return feats.cpu().float().numpy()
    return model, model_fn, None, pil_collate


def main():
    parser = argparse.ArgumentParser(description="Extract OpenImagesV7 features to LMDB")
    parser.add_argument('--model', choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument('--split', choices=['train', 'valid', 'test'], required=True)
    parser.add_argument('--dataset_dir', type=str, default='./data/open_images_v7',
                        help='OpenImagesV7 root (see the module docstring for the layout)')
    parser.add_argument('--output_dir', type=str, default='./features/open_images')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--write_paths', action='store_true',
                        help='also store __path__{idx} = images/<split>/<id>.jpg keys (useful '
                             'for visualization; the feature values are unaffected)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name, tag = MODEL_CONFIGS[args.model]
    lmdb_path = out_dir / f'{LMDB_PREFIX[args.model]}_{tag}.lmdb'

    model, model_fn, transform, collate = build_encoder(args.model, model_name, device)
    dataset = OpenImagesDataset(args.dataset_dir, args.split, transform)

    # The label LMDB depends only on the annotations, so whichever encoder runs first writes it.
    labels_path = out_dir / 'openimages_labels.lmdb'
    if not (labels_path / 'data.mdb').exists():
        tmp = out_dir / '.openimages_labels.lmdb.tmp'
        write_labels_to_lmdb(tmp, dataset)
        os.replace(tmp, labels_path)

    if (lmdb_path / 'data.mdb').exists():
        print(f"[skip] {lmdb_path} exists")
        return
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)
    features, feat_shape = extract_features_streaming(loader, model_fn, f'{args.model} ({tag})')
    paths = ({i: f"images/{args.split}/{image_id}.jpg" for i, (image_id, _) in enumerate(dataset.samples)}
             if args.write_paths else None)
    tmp = out_dir / f'.{lmdb_path.name}.tmp'
    write_to_lmdb(tmp, features, feat_shape, sample_paths=paths)
    os.replace(tmp, lmdb_path)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("Done!")


if __name__ == "__main__":
    main()
