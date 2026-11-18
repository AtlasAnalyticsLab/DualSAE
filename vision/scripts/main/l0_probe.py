#!/usr/bin/env python3
"""Cheap realized-L0 / dead-fraction probe for a trained run.

Why this exists
---------------
GatedSAE has no k: its sparsity is EMERGENT from l1_coef, so lambda must be
calibrated per (dataset, encoder) against the L0 the model actually reaches.
That calibration needs one number per candidate lambda, and running the full
branch_metrics suite (train bank + KNN + linear probe) for it would cost ~5
minutes a cell for a number that takes one encode pass over the eval split.

This also serves any model whose L0 is not pinned by construction (JumpReLU,
Vanilla) and the honesty columns the report prints next to every row.

Measured on the SAME checkpoint the evaluation reports -- best_val_monoscore_model.pt,
the epoch chosen by val MonoScore -- not on the final epoch, because those can
differ and the reported row is the former.

encode_batch_size defaults to 2048 and is RECORDED in the output: batch-level
gates (BatchTopK, Matryoshka) make L0 a function of the batch size, so a probe
run at a different size is not comparable to training.

Usage (GPU, ~10-30 s):
    python scripts/main/l0_probe.py --run-dir <dir> --eval-split val
Writes <out-dir>/<run>__<split>.json atomically; skip-guarded on existence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Vision core (two levels up) on sys.path for the flat imports below.
_VISION = Path(__file__).resolve().parents[2]
if str(_VISION) not in sys.path:
    sys.path.insert(0, str(_VISION))
from dataset import resolve_multilabel  # noqa: E402
from knn_lp_sparse import load_run, make_loader
from utils import unpack_batch

DEFAULT_OUT = Path(os.environ.get("MAIN_STATE_DIR", "./outputs/main/selection")) / "l0"


@torch.no_grad()
def probe(model, loader, device, dict_size: int) -> dict:
    """One pass: mean/percentile L0, per-unit fire counts, gate density."""
    fire_counts = torch.zeros(dict_size, dtype=torch.long, device=device)
    n_rows = 0
    nnz_total = 0
    per_row = []
    gate_open_sum, gate_batches = 0.0, 0
    for batch in loader:
        x, _, _ = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        _, latents, _ = model(x)
        active = latents > 0
        fire_counts += active.sum(dim=0).long()
        row_l0 = active.sum(dim=-1)
        nnz_total += int(row_l0.sum().item())
        n_rows += x.shape[0]
        # Percentiles need the distribution, but storing every row is wasteful;
        # the counts are tiny (one int per row) so keep them on CPU.
        per_row.append(row_l0.to("cpu"))
        # Gated only: density of the gate BEFORE the magnitude head's ReLU can
        # close a latent. A run where this sits far above L0/U is one whose two
        # heads disagree, which no L0 number alone would reveal.
        gate = getattr(model, "_gate_pre", None)
        if gate is not None:
            gate_open_sum += float((gate > 0).float().mean().item())
            gate_batches += 1

    l0 = torch.cat(per_row).float()
    dead = int((fire_counts == 0).sum().item())
    return {
        "n_rows": n_rows,
        "l0_mean": nnz_total / max(n_rows, 1),
        "l0_p05": float(l0.quantile(0.05)),
        "l0_p50": float(l0.median()),
        "l0_p95": float(l0.quantile(0.95)),
        "dict_size": dict_size,
        "n_dead": dead,
        "dead_frac": dead / dict_size,
        "n_active_units": dict_size - dead,
        "gate_open_frac": (gate_open_sum / gate_batches) if gate_batches else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--encode-batch-size", type=int, default=2048,
                    help="MUST match training: batch-level gates make L0 a "
                         "function of it")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{run_dir.name}__{args.eval_split}.json"
    if out.exists() and not args.overwrite:
        print(f"skip (exists): {out}")
        return 0

    device = torch.device(args.device)
    model, meta = load_run(run_dir, device)
    model.eval()

    eval_path = meta.get(f"{args.eval_split}_path")
    if not eval_path:
        raise ValueError(f"{run_dir}: no {args.eval_split}_path in training_config.json")
    # Same precedence knn_lp_sparse uses: the run's recorded flag wins over any
    # path heuristic, and the TRAIN path is the reference (a run is multi-label
    # or not as a whole, not per split).
    multilabel = resolve_multilabel(meta.get("train_path"), meta.get("multilabel"))
    loader = make_loader(eval_path, args.encode_batch_size, args.num_workers,
                         multilabel)

    stats = probe(model, loader, device, int(meta["dict_size"]))
    payload = {
        "schema_version": 1,
        "run_dir": str(run_dir), "exp_name": meta.get("exp_name"),
        "eval_split": args.eval_split, "eval_path": eval_path,
        "model_type": meta.get("model_type"),
        "sparsity_mode": meta.get("sparsity_mode"),
        "seed": meta.get("seed"),
        "dataset_name": meta.get("dataset_name"),
        "extractor_name": meta.get("extractor_name"),
        # The knobs that DETERMINE the number, recorded so a probe can never be
        # compared against one taken under different conditions.
        "encode_batch_size": args.encode_batch_size,
        "l1_coef": meta.get("l1_coef"),
        "nominal_k": meta.get("effective_k"),
        "best_epoch": meta.get("best_epoch"),
        **stats,
    }
    tmp = out.with_suffix(f".tmp{os.getpid()}.json")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(out)
    print(f"{meta.get('exp_name')}: L0 {stats['l0_mean']:.1f} "
          f"(p50 {stats['l0_p50']:.0f}, p95 {stats['l0_p95']:.0f})  "
          f"dead {stats['dead_frac']:.3f}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
