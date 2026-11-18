#!/usr/bin/env python3
"""List finished training runs as a manifest the ablation evaluators and collectors read.

Scans <checkpoint root>/*/training_config.json and writes a JSON list of rows

    {"dataset", "encoder", "method", "seed", "run_dir", "eval_split"}

where method is the main-table method name (topk, batch_topk, jumprelu, dual_topk,
dual_batch_topk, dual_jump_relu) and eval_split is "val" for ImageNet-1K and "test" for
OpenImagesV7 (purity is additionally read from val). Only finished runs (autoencoder_final.pt and
best_val_monoscore_model.pt present) are listed. Use it on the directory that holds the
main-table runs, e.g.

    python make_run_manifest.py --checkpoints ./outputs/main/checkpoints --out ./outputs/main_runs.json
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

# training_config.json records the trainer's model_type; dual runs add sparsity_mode.
SINGLE_METHODS = {"topk": "topk", "batch_topk": "batch_topk", "jumprelu": "jumprelu"}
DUAL_METHODS = {"topk": "dual_topk", "batch_topk": "dual_batch_topk", "jump_relu": "dual_jump_relu"}
EVAL_SPLIT = {"imagenet": "val", "open_images": "test"}


def method_of(cfg: dict):
    if cfg.get("model_type") == "dual":
        return DUAL_METHODS.get(cfg.get("sparsity_mode"))
    return SINGLE_METHODS.get(cfg.get("model_type"))


def dataset_of(cfg: dict):
    """training_config.json records dataset_name / extractor_name (current trainer)."""
    return cfg.get("dataset_name"), cfg.get("extractor_name")


def scan(root: Path, pattern: str = "*"):
    rows, skipped = [], []
    for cfg_path in sorted(root.glob("*/training_config.json")):
        run = cfg_path.parent
        if not fnmatch.fnmatch(run.name, pattern):
            continue
        if not ((run / "autoencoder_final.pt").exists() and (run / "best_val_monoscore_model.pt").exists()):
            skipped.append(f"{run.name}: unfinished")
            continue
        cfg = json.loads(cfg_path.read_text())
        method = method_of(cfg)
        dataset, encoder = dataset_of(cfg)
        if method is None or dataset not in EVAL_SPLIT or encoder is None:
            skipped.append(f"{run.name}: model/dataset not recognised")
            continue
        rows.append({"dataset": dataset, "encoder": encoder, "method": method,
                     "seed": int(cfg["seed"]), "run_dir": str(run.resolve()),
                     "eval_split": EVAL_SPLIT[dataset]})
    return rows, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoints", type=Path, required=True, help="directory holding run folders")
    ap.add_argument("--pattern", default="*", help="glob on run folder names")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows, skipped = scan(args.checkpoints, args.pattern)
    keys = [(r["dataset"], r["encoder"], r["method"], r["seed"]) for r in rows]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        # Two runs for one (dataset, encoder, method, seed) cannot both be "the" main-table run.
        sys.exit(f"duplicate cells (narrow --pattern): {dupes[:5]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"wrote {args.out}: {len(rows)} runs" + (f"; skipped {len(skipped)}" if skipped else ""))
    for s in skipped[:20]:
        print("  skipped", s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
