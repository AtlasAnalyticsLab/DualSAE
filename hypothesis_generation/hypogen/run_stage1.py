"""Stage 1 (consolidated) — for ONE encoder: extract val+train embeddings, then dump activations for
all of that encoder's cells.

Each step is individually skip-guarded (resumable), so re-running only does what's missing. The vision
encoder is loaded once (by extract); each cell's SAE is small and loaded per-dump.

Run (GPU): python -m hypogen.run_stage1 --encoder clip
"""

from __future__ import annotations

import argparse

import torch

from hypogen import config
from hypogen.extract_embeddings import extract
from hypogen.dump_activations import dump


def main():
    ap = argparse.ArgumentParser(description="Stage 1: extract embeddings + dump activations for one encoder.")
    ap.add_argument("--encoder", required=True, choices=config.ENCODERS)
    ap.add_argument("--batch-size", type=int, default=256, help="vision-encoder batch (extract)")
    ap.add_argument("--dump-batch-size", type=int, default=8192, help="SAE encode batch (dump)")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 1) embeddings + manifest + labels (val first: smaller, and needed by proxy/faithful).
    for split in ("val", "train"):
        extract(args.encoder, split, args.batch_size, args.num_workers, args.device)

    # 2) activations for every cell of this encoder, both splits: 3 act x 2 variants x 3 seeds = 18.
    cells = list(config.iter_cells(encoders=[args.encoder]))
    print(f"[run_stage1:{args.encoder}] mode={config.MODE} dataset={config.DATASET}: "
          f"dumping activations for {len(cells)} cells")
    for cell in cells:
        for split in ("train", "val"):
            dump(cell, split, args.dump_batch_size, args.device)
    print(f"[run_stage1:{args.encoder}] done")


if __name__ == "__main__":
    main()
