#!/usr/bin/env python3
"""Build the local OpenImagesV7 layout that ``extract_open_images.py`` reads.

Three sub-commands (``prepare_assets.sh`` calls them in order; only ``images`` needs internet
access beyond the files it is given):

``captions``  Turn the official Localized Narratives ``.jsonl`` file(s) of one split into
              ``captions/<split>/simplified_open_images_<split>_localized_narratives.json``:
              a JSON object ``{image_id: [record, ...]}`` keeping, per narrative, only
              ``dataset_id``, ``image_id``, ``annotator_id`` and ``caption``. Images appear in
              the order of their first narrative in the input file(s) (files are read in the
              order given), and each image's narratives keep their input order. That order
              defines the sample index of every feature/label LMDB.
``labels``    Copy the class-description CSV and the split's human-verified image-level label
              CSV into ``labels/`` under the names the dataset class expects.
``images``    Download the JPEGs of every image in the split's caption JSON from the public
              Open Images S3 bucket into ``images/<split>/<image_id>.jpg`` (skips existing
              files, so it can be re-run to resume).

Splits are named train / valid / test locally; the bucket uses train / validation / test.
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

KEEP_FIELDS = ("dataset_id", "image_id", "annotator_id", "caption")
BUCKET_NAME = "open-images-dataset"
BUCKET_SPLIT = {"train": "train", "valid": "validation", "test": "test"}


def build_captions(jsonl_paths, out_path):
    """Group narratives by image id, in first-appearance order (see the module docstring)."""
    grouped = {}
    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                grouped.setdefault(rec["image_id"], []).append({k: rec.get(k) for k in KEEP_FIELDS})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(grouped, f)
    os.replace(tmp, out_path)
    n_caps = sum(len(v) for v in grouped.values())
    print(f"{out_path}: {len(grouped)} images, {n_caps} (image, caption) samples")


def copy_labels(dataset_dir, split, class_csv, labels_csv):
    labels_dir = Path(dataset_dir) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(class_csv, labels_dir / "oidv7-class-descriptions-boxable.csv")
    shutil.copyfile(labels_csv, labels_dir / f"{split}-annotations-human-imagelabels-boxable.csv")
    print(f"labels for {split} -> {labels_dir}")


def download_images(dataset_dir, split, workers):
    import boto3
    import botocore

    cap = Path(dataset_dir) / "captions" / split / f"simplified_open_images_{split}_localized_narratives.json"
    image_ids = sorted(json.loads(cap.read_text()).keys())
    out_dir = Path(dataset_dir) / "images" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [i for i in image_ids if not (out_dir / f"{i}.jpg").exists()]
    print(f"{split}: {len(image_ids)} images, {len(missing)} to download")
    if not missing:
        return
    bucket = boto3.resource("s3", config=botocore.config.Config(
        signature_version=botocore.UNSIGNED)).Bucket(BUCKET_NAME)

    def fetch(image_id):
        target = out_dir / f"{image_id}.jpg"
        tmp = out_dir / f".{image_id}.jpg.tmp"
        bucket.download_file(f"{BUCKET_SPLIT[split]}/{image_id}.jpg", str(tmp))
        os.replace(tmp, target)  # atomic: a killed download never leaves a truncated JPEG

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, i): i for i in missing}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as exc:  # report and continue; a re-run retries the rest
                errors.append((futures[fut], str(exc)))
            if k % 10000 == 0:
                print(f"  {k}/{len(missing)}")
    print(f"done: {len(missing) - len(errors)} downloaded, {len(errors)} failed")
    for image_id, err in errors[:20]:
        print(f"  {image_id}: {err}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("captions")
    c.add_argument("--split", choices=sorted(BUCKET_SPLIT), required=True)
    c.add_argument("--jsonl", nargs="+", required=True, help="official Localized Narratives .jsonl file(s), in order")
    c.add_argument("--dataset_dir", default="./data/open_images_v7")
    lab = sub.add_parser("labels")
    lab.add_argument("--split", choices=sorted(BUCKET_SPLIT), required=True)
    lab.add_argument("--class_csv", required=True)
    lab.add_argument("--labels_csv", required=True)
    lab.add_argument("--dataset_dir", default="./data/open_images_v7")
    im = sub.add_parser("images")
    im.add_argument("--split", choices=sorted(BUCKET_SPLIT), required=True)
    im.add_argument("--dataset_dir", default="./data/open_images_v7")
    im.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    if args.cmd == "captions":
        out = Path(args.dataset_dir) / "captions" / args.split / f"simplified_open_images_{args.split}_localized_narratives.json"
        build_captions(args.jsonl, out)
    elif args.cmd == "labels":
        copy_labels(args.dataset_dir, args.split, args.class_csv, args.labels_csv)
    else:
        download_images(args.dataset_dir, args.split, args.workers)


if __name__ == "__main__":
    main()
