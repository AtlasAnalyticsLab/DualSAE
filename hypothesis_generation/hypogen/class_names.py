"""Human-readable class names for the qualitative figures.

Maps a dataset class index (as used in the hypogen labels) -> a readable name, using the files that
prepare_assets.sh downloads into $ASSETS_DIR (default ./assets); read offline.

ImageNet: our class index is ImageFolder sorted-wnid order, which is NOT the ILSVRC order keyed in
imagenet_class_index.json. So we map by WNID, not by index: the caller resolves idx -> wnid from the
manifest (rel_path's top dir), and we look the wnid up in the {wnid: name} table built from that file's
*values*. OpenImages: the label order of the pre-extracted label LMDB is sorted(DisplayName) of the
boxable classes, so idx -> sorted-DisplayName[idx] directly.

Missing name files -> graceful fallback to the raw wnid / numeric id.
"""

from __future__ import annotations

import csv
import functools
import json
import os

ASSETS = os.environ.get("ASSETS_DIR", "./assets")
_IMAGENET_JSON = os.path.join(ASSETS, "imagenet_class_index.json")
_OPENIMAGES_CSV = os.path.join(ASSETS, "oidv7-class-descriptions-boxable.csv")


@functools.lru_cache(maxsize=1)
def _imagenet_wnid_to_name() -> dict:
    """{wnid: display_name}. Empty dict if the file is missing."""
    if not os.path.isfile(_IMAGENET_JSON):
        return {}
    with open(_IMAGENET_JSON) as f:
        d = json.load(f)  # {"0": ["n01440764", "tench"], ...}
    return {wnid: name.replace("_", " ") for wnid, name in d.values()}


@functools.lru_cache(maxsize=1)
def _openimages_names() -> list:
    """sorted list of DisplayName, so idx -> names[idx] matches the training label order. [] if missing."""
    if not os.path.isfile(_OPENIMAGES_CSV):
        return []
    with open(_OPENIMAGES_CSV) as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    start = 1 if rows[0] and rows[0][0].lower() in ("labelname", "label") else 0  # skip header if present
    return sorted(r[1] for r in rows[start:] if len(r) >= 2)


def imagenet_name(wnid: str) -> str:
    """wnid (e.g. 'n01440764') -> 'tench', or the wnid itself if unknown."""
    return _imagenet_wnid_to_name().get(wnid, wnid)


def openimages_name(idx: int) -> str:
    """OpenImages class idx -> DisplayName (sorted order), or 'class <idx>' if unavailable."""
    names = _openimages_names()
    return names[idx] if 0 <= idx < len(names) else f"class {idx}"


def build_idx_to_wnid(manifest: list) -> dict:
    """ImageNet: {label_idx: wnid} from a manifest of [{rel_path:'n01440764/..', label:int}], first
    occurrence per label (rel_path's top directory IS the wnid)."""
    out = {}
    for m in manifest:
        lab = m.get("label")
        if lab is None or lab in out:
            continue
        out[lab] = m["rel_path"].split("/")[0]
    return out


def readable(dataset: str, idx: int, idx_to_wnid: dict | None = None) -> str:
    """Readable label for a class idx. ImageNet needs idx_to_wnid (from build_idx_to_wnid); OpenImages
    uses idx directly. Falls back to wnid / 'class <idx>' when name files are absent."""
    if dataset == "imagenet":
        wnid = (idx_to_wnid or {}).get(idx)
        return imagenet_name(wnid) if wnid else f"class {idx}"
    return openimages_name(idx)
