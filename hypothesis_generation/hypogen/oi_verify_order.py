"""OpenImages data-layer VERIFICATION GATE (do FIRST, before any hypogen OpenImages run).

The pre-extracted OpenImages feature LMDBs are keyed by integer index 0..N-1 with NO stored image-id /
path, so we cannot trust an index->image mapping until we PROVE it. The feature extractor built its row
order from an image_id -> captions mapping expanded into (image_id, caption_idx) pairs — one LMDB row PER
CAPTION. The per-line localized-narratives JSONL has exactly as many lines as the LMDB has rows. Two
plausible orderings reproduce that count; they differ only around the few images that have >1 caption:
  - "grouped": dict grouped by image_id in FIRST-OCCURRENCE order, then caption_idx (matches the extractor's
               image_id_to_captions.items() iteration IF the simplified json grouped captions per image).
  - "line":    raw JSONL line order (index i <-> line i).

This script reconstructs BOTH candidate index->image_id maps, re-extracts CLIP features for a sample of
indices with the EXACT training encoder (open_clip create_model_and_transforms + encode_image, same as the
extractor), and reports the max cosine to the stored LMDB feature at each index. The correct ordering
yields cos ~ 1.0 at every sampled index. PASS => that ordering is the manifest (build_oi_data.py uses
'grouped'); FAIL on both => the LMDB was built from a different caption file and must be re-extracted.

Run (GPU): python oi_verify_order.py --split valid
"""

from __future__ import annotations

import argparse
import json
import os

import lmdb
import numpy as np
import torch

_NARR_DIR = os.environ.get("OPENIMAGES_NARRATIVES_DIR", "./data/open_image_v7")
NARR = {  # per-split full localized-narratives JSONL (1 line per (image,caption))
    "valid": os.path.join(_NARR_DIR, "open_images_validation_localized_narratives.jsonl"),
    "test": os.path.join(_NARR_DIR, "open_images_test_localized_narratives.jsonl"),
}
IMG_DIR = {"valid": os.environ.get("OPENIMAGES_VALID_DIR", "./data/open_image_v7/valid"),
           "test": os.environ.get("OPENIMAGES_TEST_DIR", "./data/open_image_v7/test")}
FEAT_LMDB = os.path.join(os.environ.get("FEATURES_OPENIMAGES", "./features/open_images"),
                         "{split}", "clip_image_features_CLIP-ViT-L-14-DataComp.lmdb")
CLIP_ID = "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"


def _load_lines(split: str):
    """Return the JSONL rows in file order: list of image_id (one per line = one (image,caption) row)."""
    rows = []
    with open(NARR[split], encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line)["image_id"])
    return rows


def build_candidate(order: str, line_image_ids):
    """index -> image_id under a candidate ordering.
    'line'    : index i = line i's image_id (raw file order).
    'grouped' : group by image_id (first-occurrence order), expand captions -> [(img,ci)]; matches the
                extractor's image_id_to_captions.items() x caption_idx iteration."""
    if order == "line":
        return list(line_image_ids)
    # grouped: preserve first-occurrence order, repeat each image_id once per caption it has
    from collections import OrderedDict
    counts = OrderedDict()
    for iid in line_image_ids:
        counts[iid] = counts.get(iid, 0) + 1
    out = []
    for iid, c in counts.items():
        out.extend([iid] * c)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--n-sample", type=int, default=48)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    line_ids = _load_lines(a.split)
    feat_path = FEAT_LMDB.format(split=a.split)
    env = lmdb.open(feat_path, readonly=True, lock=False)
    with env.begin() as t:
        n_lmdb = int(t.get(b"__len__"))
        shape = eval(t.get(b"__shape__").decode())  # noqa: S307 (trusted, our own meta)
    print(f"[verify {a.split}] narrative lines={len(line_ids)}  LMDB len={n_lmdb}  shape={shape}")
    assert len(line_ids) == n_lmdb, f"line count {len(line_ids)} != LMDB len {n_lmdb} — wrong narratives file"

    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_ID)
    model = model.to(device).eval()
    from PIL import Image

    # deterministic spread of sample indices across the full range (incl. first & last)
    rng = np.random.default_rng(0)
    idxs = sorted(set([0, n_lmdb - 1] + rng.choice(n_lmdb, size=min(a.n_sample, n_lmdb), replace=False).tolist()))

    results = {}
    for order in ("grouped", "line"):
        cand = build_candidate(order, line_ids)
        cos_list, missing = [], 0
        with env.begin() as t:
            for i in idxs:
                iid = cand[i]
                p = os.path.join(IMG_DIR[a.split], f"{iid}.jpg")
                if not os.path.isfile(p):
                    missing += 1
                    continue
                img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    f = model.encode_image(img).cpu().float().numpy()[0]
                stored = np.frombuffer(t.get(str(i).encode()), dtype=np.float32)
                cos = float(np.dot(f, stored) / (np.linalg.norm(f) * np.linalg.norm(stored) + 1e-8))
                cos_list.append(cos)
        cos_arr = np.array(cos_list) if cos_list else np.array([np.nan])
        n_match = int((cos_arr > 0.99).sum())
        results[order] = (cos_arr, missing, n_match)
        print(f"[verify {a.split}] order={order:8s}  n={len(cos_list)} missing={missing}  "
              f"cos[min/median/mean]={np.nanmin(cos_arr):.4f}/{np.nanmedian(cos_arr):.4f}/{np.nanmean(cos_arr):.4f}  "
              f">0.99: {n_match}/{len(cos_list)}")

    # verdict
    best = max(results, key=lambda o: results[o][2])
    arr, missing, n_match = results[best]
    n = len(arr)
    if n_match >= 0.95 * n and missing == 0:
        print(f"\n[verify {a.split}] PASS ✓ ordering='{best}' (cos>0.99 on {n_match}/{n}, 0 missing). "
              f"Use this ordering to build the OpenImages manifest.")
    else:
        print(f"\n[verify {a.split}] FAIL ✗ best='{best}' only {n_match}/{n} match (missing={missing}). "
              f"The LMDB order came from a different caption file; re-extract the features.")


if __name__ == "__main__":
    main()
