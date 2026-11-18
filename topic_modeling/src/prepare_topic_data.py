"""
Prepare caption-aligned image embeddings for the SAE-TM topic-modeling pipeline,
for a chosen vision encoder (clip / siglip / vit).

WHY (read before editing):
  In SAE-TM the SAE trains on IMAGE embeddings; captions are only the text side (BoW) used to
  learn the word-emission matrix and to evaluate topics. interpret_sae.py pairs data POSITIONALLY
  (`embeddings[i]` ↔ BoW row `i`, with `assert embeddings.shape[0]==bow.shape[0]`), so the
  image-embedding rows MUST be in the SAME ORDER as the caption rows that build the BoW.

  We therefore embed each captioned val image IN CAPTION ORDER. The captions/BoW are
  ENCODER-INDEPENDENT and shared; only the image embeddings differ per encoder. Running this
  once per encoder (clip/siglip/vit) yields per-encoder embedding sets that all align 1:1 with
  the single shared documents.jsonl / BoW, because the kept-caption list is identical across
  encoders (every val image has a caption).

RESUMABILITY: embeddings are written shard-by-shard, atomically (tmp+rename); a re-run skips
  shards that already exist with the right row count. `manifest.json` (written last) is the
  per-encoder "done" sentinel. Keep --shard-size fixed across resumes.

OUTPUTS:
  <out-dir>/emb_000.pt, ...     float32 [n, D] shards in caption order (L2-normalized), D=encoder dim
  <out-dir>/manifest.json       {n, dim, encoder, model, ...}
  <documents-path>              shared documents.jsonl: {"document": <caption>, "id": <rel_path>}
                                (written once; identical across encoders)

USAGE (GPU; weights pre-fetched offline by prepare_assets.sh):
  python prepare_topic_data.py --encoder siglip \
      --captions   ./data/captions/imagenet/imagenet_val_captions.jsonl \
      --image-root ./data/imagenet/val \
      --out-dir    ./outputs/topic_modeling/imagenet_val/emb_siglip \
      --documents-path ./outputs/topic_modeling/imagenet_val/data/documents.jsonl \
      --batch-size 256 --shard-size 5000
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Default HF / open_clip ids per encoder (match the extracted LMDB features).
ENCODER_MODELS = {
    "clip":   "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",  # open_clip, 768-d
    "siglip": "google/siglip2-so400m-patch14-384",                 # transformers, 1152-d
    "vit":    "google/vit-large-patch16-224",                      # transformers, 1024-d (CLS)
}


def read_caption_records(captions_path: str, image_root: str, limit):
    """Ordered (rel_path, caption) for captions whose image exists on disk (caption file order).

    Robust to imperfect caption files: LLM-generated caption JSONL (e.g. the OpenImages InternVL
    captions) can contain invalid UTF-8 bytes or an occasional malformed line. We therefore (a) pin
    the encoding to UTF-8 with errors='replace' — a stray bad byte becomes U+FFFD and the record is
    kept (the bytes-level locale of the compute node no longer matters, which previously crashed the
    reader), and (b) skip any line that still fails to parse or lacks the required keys, counting it,
    instead of aborting the whole prep job on one bad line. The few U+FFFD that survive are harmless
    downstream: the BoW tokenizer keeps only ^[a-z]+$ tokens, so replacement chars are dropped.
    ImageNet's clean caption file produces byte-identical output under this path.
    """
    records, missing, bad = [], 0, 0
    with open(captions_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rel_path, caption = obj["rel_path"], obj["caption"]
            except (json.JSONDecodeError, KeyError, TypeError):
                bad += 1
                continue
            if not os.path.isfile(os.path.join(image_root, rel_path)):
                missing += 1
                continue
            records.append((rel_path, caption))
            if limit is not None and len(records) >= limit:
                break
    if missing:
        print(f"  WARNING: {missing} captioned image(s) not found under {image_root}; skipped.")
    if bad:
        print(f"  WARNING: {bad} malformed/undecodable caption line(s) skipped.")
    return records


class _ImageDataset(Dataset):
    """Loads + preprocesses images for an ordered rel_path slice. `preprocess(PIL)->Tensor`."""

    def __init__(self, records, image_root, preprocess):
        self.records, self.image_root, self.preprocess = records, image_root, preprocess

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rel_path, _ = self.records[idx]
        img = Image.open(os.path.join(self.image_root, rel_path)).convert("RGB")
        return self.preprocess(img)


def build_encoder(encoder: str, model_name: str, device: str):
    """Return (preprocess(PIL)->Tensor, embed_fn(batch)->features) for the chosen encoder.

    Same encoder families (and output tensors) as the feature extraction that produced the SAE
    training LMDBs:
      clip   : open_clip `model.encode_image` (post-projection, 768-d)
      siglip : transformers `AutoModel.get_image_features` (pooled, 1152-d)
      vit    : transformers `AutoModel(...).last_hidden_state[:, 0]` (CLS token, 1024-d)
    """
    if encoder == "clip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        model = model.to(device).eval()

        def embed_fn(imgs):
            return model.encode_image(imgs)

        return preprocess, embed_fn

    # transformers-based (siglip / vit): apply the image processor per-image so the DataLoader
    # collates pixel tensors, then run the model on the batched pixel_values.
    from transformers import AutoModel, AutoImageProcessor
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_name)

    def preprocess(img):
        return processor(images=img, return_tensors="pt")["pixel_values"][0]

    if encoder == "siglip":
        # transformers>=5 returns a BaseModelOutputWithPooling from get_image_features; the
        # pooled image embedding (.pooler_output) is the tensor the train LMDB stored. Older
        # versions returned that tensor directly — handle both.
        def embed_fn(imgs):
            out = model.get_image_features(pixel_values=imgs)
            if torch.is_tensor(out):
                return out
            po = getattr(out, "pooler_output", None)
            if torch.is_tensor(po):
                return po
            if isinstance(out, (tuple, list)):
                for v in out:
                    if torch.is_tensor(v):
                        return v
            raise TypeError(f"Unexpected SigLIP image-feature output type: {type(out)}")
    elif encoder == "vit":
        def embed_fn(imgs):
            return model(pixel_values=imgs).last_hidden_state[:, 0, :]  # CLS token (matches extract)
    else:
        raise ValueError(f"Unknown encoder {encoder!r}; expected clip/siglip/vit.")
    return preprocess, embed_fn


def _shard_path(out_dir, s, pad):
    return os.path.join(out_dir, f"emb_{s:0{pad}d}.pt")


def _valid_shard(path, expected_rows):
    if not os.path.isfile(path):
        return False
    try:
        t = torch.load(path, map_location="cpu")
    except Exception:
        return False
    return t.ndim == 2 and t.shape[0] == expected_rows


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _embed_chunk(records_chunk, image_root, preprocess, embed_fn, device, batch_size,
                 num_workers, normalize):
    loader = DataLoader(
        _ImageDataset(records_chunk, image_root, preprocess),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )
    feats, use_cuda = [], device != "cpu"
    for imgs in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_cuda):
            f = embed_fn(imgs)
        f = f.float()
        if normalize:
            f = F.normalize(f, p=2, dim=-1)   # L2-normalize (matches LMDBFeatureDataset reads)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)


def main():
    parser = argparse.ArgumentParser(
        description="Caption-ordered image embeddings per encoder (resumable).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--captions", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out-dir", required=True, help="Per-encoder dir for emb_*.pt + manifest.json.")
    parser.add_argument("--documents-path", default=None,
                        help="Where to write the shared documents.jsonl (default <out-dir>/documents.jsonl).")
    parser.add_argument("--encoder", choices=list(ENCODER_MODELS), default="clip")
    parser.add_argument("--model", default=None, help="Override the encoder's HF/open_clip id.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    doc_path = args.documents_path or os.path.join(args.out_dir, "documents.jsonl")
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    normalize = not args.no_normalize
    model_name = args.model or ENCODER_MODELS[args.encoder]

    print(f"Reading captions: {args.captions}")
    records = read_caption_records(args.captions, args.image_root, args.limit)
    n = len(records)
    if n == 0:
        raise SystemExit("No captioned images found on disk — check --captions/--image-root.")
    n_shards = math.ceil(n / args.shard_size)
    pad = max(3, len(str(n_shards)))
    print(f"  encoder={args.encoder} ({model_name}); {n} images → {n_shards} shard(s).")

    def chunk(s):
        return records[s * args.shard_size:(s + 1) * args.shard_size]

    todo = [s for s in range(n_shards) if not _valid_shard(_shard_path(args.out_dir, s, pad), len(chunk(s)))]
    if n_shards - len(todo):
        print(f"  resume: {n_shards - len(todo)}/{n_shards} shard(s) already valid.")

    if todo:
        print(f"Loading encoder {args.encoder} (device={args.device})")
        preprocess, embed_fn = build_encoder(args.encoder, model_name, args.device)
        for s in todo:
            emb = _embed_chunk(chunk(s), args.image_root, preprocess, embed_fn, args.device,
                               args.batch_size, args.num_workers, normalize)
            assert emb.shape[0] == len(chunk(s)), "shard row count mismatch"
            _atomic_save(emb.contiguous(), _shard_path(args.out_dir, s, pad))
            print(f"  saved shard {s + 1}/{n_shards}: {_shard_path(args.out_dir, s, pad)} "
                  f"({emb.shape[0]} rows, dim={emb.shape[1]})")

    dim = int(torch.load(_shard_path(args.out_dir, 0, pad), map_location="cpu").shape[1])

    # Shared documents.jsonl (encoder-independent). Idempotent + atomic.
    os.makedirs(os.path.dirname(os.path.abspath(doc_path)), exist_ok=True)
    doc_tmp = doc_path + ".tmp"
    with open(doc_tmp, "w") as w:
        for rel_path, caption in records:
            w.write(json.dumps({"document": caption, "id": rel_path}) + "\n")
    os.replace(doc_tmp, doc_path)

    mtmp = manifest_path + ".tmp"
    with open(mtmp, "w") as w:
        json.dump({"n": n, "dim": dim, "encoder": args.encoder, "model": model_name,
                   "image_root": args.image_root, "captions": args.captions,
                   "normalized": normalize, "shard_size": args.shard_size, "n_shards": n_shards}, w, indent=2)
    os.replace(mtmp, manifest_path)

    print(f"Done [{args.encoder}]. {n} rows, dim={dim}; embeddings ↔ documents.jsonl are 1:1.")


if __name__ == "__main__":
    main()
