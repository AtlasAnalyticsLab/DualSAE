"""Stage 1b — encode cached embeddings with one SAE checkpoint into dense latent activations.

For a cell (encoder x activation x variant x seed) and split, load the Stage-1a embeddings, load the
SAE via the uniform old-format loader, and stream `sae.encode` to produce a dense float16 activation
matrix (n_rows, dict_size). Dense (not sparse CSR) because Dual/baseline JumpReLU activations are not
k-sparse (~thousands of nonzeros per row); dense fp16 is uniform and simple for all activations.

Both span001 (Dual) and oldpipe (baseline) checkpoints are in the OLD `sae/` format, so both load via
old_sae_adapter.load_old_sae(path, sae_type, device); `.encode(x)` returns forward(x)[1] (the sparse
mono code for Dual, the sparse code for baselines). Old models run in float32 (as trained).

Run (GPU node):
  python -m hypogen.dump_activations --encoder clip --activation topk --variant dual --seed 42 --split val
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from numpy.lib.format import open_memmap

from hypogen import config
from old_sae_adapter import load_old_sae


@torch.no_grad()
def dump(cell: config.Cell, split: str, batch_size: int, device: str) -> None:
    emb_path = config.embeddings_path(cell.encoder, split)
    if not os.path.isfile(emb_path):
        raise FileNotFoundError(f"Missing Stage-1a embeddings: {emb_path} (run extract_embeddings first)")
    out_path = config.acts_path(cell, split)
    meta_path = config.acts_meta_path(cell, split)

    emb = np.load(emb_path, mmap_mode="r")  # float16 [n, D]
    n, d = emb.shape

    # Resume: skip if activations already complete + meta written.
    if os.path.isfile(out_path) and os.path.isfile(meta_path):
        existing = np.load(out_path, mmap_mode="r")
        if existing.shape[0] == n:
            print(f"[{cell.tag}/{split}] activations already complete: {existing.shape}")
            return

    if not os.path.isfile(cell.ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {cell.ckpt_path}")
    print(f"[{cell.tag}/{split}] loading SAE ({cell.sae_type}) {cell.ckpt_path}")
    sae = load_old_sae(cell.ckpt_path, cell.sae_type, device=device).eval()

    # Probe dict_size on one row.
    probe = torch.from_numpy(np.asarray(emb[:1], dtype=np.float32)).to(device)
    dict_size = int(sae.encode(probe).shape[1])
    print(f"[{cell.tag}/{split}] n={n} D={d} dict_size={dict_size}; encoding...")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp.npy"
    # Preallocate the output .npy on disk (avoids holding the whole fp16 matrix in RAM).
    out = open_memmap(tmp_path, mode="w+", dtype=np.float16, shape=(n, dict_size))
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        xb = torch.from_numpy(np.asarray(emb[s:e], dtype=np.float32)).to(device)
        z = sae.encode(xb).float().cpu().numpy().astype(np.float16)
        out[s:e] = z
        if (s // batch_size) % 20 == 0:
            print(f"  [{cell.tag}/{split}] {e}/{n}", flush=True)
    out.flush()
    del out
    os.replace(tmp_path, out_path)

    with open(meta_path + ".tmp", "w") as f:
        json.dump({"encoder": cell.encoder, "activation": cell.activation, "variant": cell.variant,
                   "seed": cell.seed, "sae_type": cell.sae_type, "split": split, "n_rows": n,
                   "dict_size": dict_size, "embed_dim": d, "ckpt": cell.ckpt_path}, f)
    os.replace(meta_path + ".tmp", meta_path)
    print(f"[{cell.tag}/{split}] saved activations ({n}, {dict_size}) -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Stage 1b: encode embeddings -> dense fp16 activations.")
    ap.add_argument("--encoder", required=True, choices=config.ENCODERS)
    ap.add_argument("--activation", required=True, choices=[p.name for p in config.ACTIVATION_PAIRS])
    ap.add_argument("--variant", required=True, choices=config.ALLOWED_VARIANTS)
    ap.add_argument("--seed", required=True, type=int, choices=config.SEEDS)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cell = config.Cell(args.encoder, args.activation, args.variant, args.seed)
    dump(cell, args.split, args.batch_size, args.device)


if __name__ == "__main__":
    main()
