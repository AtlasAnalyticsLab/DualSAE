#!/usr/bin/env python3
"""Map (dataset, encoder) to the feature LMDBs that `vision/main.py` trains on.

The ablation drivers pass every feature path explicitly (`--train_path/--val_path/--test_path`),
so a run never depends on a path table baked into the trainer. The layout below is

    $FEATURES_ROOT/<dataset>/<split>/<file name>

with the split directory names and LMDB file names that the feature-extraction step writes
(ImageNet-1K has no labelled test split, so its test path is None). `FEATURES_ROOT` defaults to
`./features`, relative to the directory the driver is launched from, and is made absolute.

Usage from bash (prints the three flags, one per line, ready for `mapfile`):

    python feature_paths.py --dataset imagenet --encoder clip
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

# Directory name of each split, per dataset.
SPLIT_DIRS: Dict[str, Dict[str, Optional[str]]] = {
    "imagenet": {"train": "train", "val": "validation", "test": None},
    "open_images": {"train": "train", "val": "valid", "test": "test"},
}

# LMDB file name per (dataset, encoder). CLIP = ViT-L/14 DataComp, SigLIP2 = so400m-p14-384,
# ViT = ViT-L/16 (224); "sigclip" is the code name of the SigLIP2 encoder throughout.
FILE_NAMES: Dict[str, Dict[str, str]] = {
    "imagenet": {
        "clip": "clip_image_features_CLIP-ViT-L-14-DataComp.lmdb",
        "sigclip": "siglip2_image_features_siglip2-so400m-p14-384.lmdb",
        "vit": "vit_features_vit-large-p16-224.lmdb",
    },
    "open_images": {
        "clip": "clip_image_features_CLIP-ViT-L-14-DataComp.lmdb",
        "sigclip": "siglip2_image_features_siglip2-so400m-p14-384.lmdb",
        "vit": "vit_features_vit-large-p16-224.lmdb",
    },
}


def features_root() -> Path:
    # Absolute, because the drivers launch main.py from vision/ rather than from the caller's
    # working directory, against which a relative FEATURES_ROOT is meant.
    return Path(os.environ.get("FEATURES_ROOT", "./features")).resolve()


def lmdb_path(dataset: str, encoder: str, split: str, root: Optional[Path] = None) -> Optional[str]:
    """Path of one split's LMDB, or None when the dataset has no such split."""
    split_dir = SPLIT_DIRS[dataset][split]
    if split_dir is None:
        return None
    root = features_root() if root is None else Path(root)
    return str(root / dataset / split_dir / FILE_NAMES[dataset][encoder])


def path_args(dataset: str, encoder: str, root: Optional[Path] = None) -> List[str]:
    """`--train_path X --val_path Y [--test_path Z]` for main.py.

    The test flag is omitted when the split does not exist, which leaves main.py's value at None
    exactly as in the ImageNet-1K runs (their training_config.json records test_path = null).
    """
    out: List[str] = []
    for split in ("train", "val", "test"):
        path = lmdb_path(dataset, encoder, split, root)
        if path is not None:
            out += [f"--{split}_path", path]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=sorted(SPLIT_DIRS))
    ap.add_argument("--encoder", required=True, choices=sorted(FILE_NAMES["imagenet"]))
    ap.add_argument("--split", choices=("train", "val", "test"), default=None,
                    help="print only this split's path instead of the main.py flags")
    args = ap.parse_args()
    if args.split:
        print(lmdb_path(args.dataset, args.encoder, args.split) or "")
    else:
        print("\n".join(path_args(args.dataset, args.encoder)))


if __name__ == "__main__":
    main()
