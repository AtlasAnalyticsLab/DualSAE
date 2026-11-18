#!/usr/bin/env python3
"""Score runs listed in a manifest with branch_metrics.py on every representation.

Used for (a) the main-table runs (Sparse Branch x^M vs. Residual Branch r_hat bars on ImageNet-1K
and OpenImagesV7; the DualSAE reference of the joint-vs-sequential figure; the decorrelation-loss
table) and (b) any other list of runs written by make_run_manifest.py.

Each run is scored on its eval split ("val" for ImageNet-1K; "test" for OpenImagesV7, plus "val"
there because purity is read from val), at the best-validation-MonoScore checkpoint, on the
representations x, x^M, r_hat and x_hat, with KNN k = 20 (cosine, full train bank), LP for 10
epochs, retrieval purity with k = 50 (self and bank modes) and the training batch size 2048 for
encoding (BatchTopK selects over the whole batch, so the batch size changes z).

Resumable: a row whose result JSON is complete is skipped. --index / --shard split the work.

    python run_branch_metrics.py --manifest ./outputs/main_runs.json \
        --out-dir ./outputs/ablations/main_branch_metrics/results
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

VISION_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(VISION_DIR))

REPS = ["x", "xM", "r_hat", "xhat"]
SPLITS = {"imagenet": ["val"], "open_images": ["test", "val"]}


def rows_from_manifest(path: Path, methods=None):
    rows = []
    for r in json.loads(path.read_text()):
        if methods and r["method"] not in methods:
            continue
        for split in SPLITS[r["dataset"]]:
            rows.append({**r, "eval_split": split, "checkpoint": r.get("checkpoint", "best")})
    return rows


def outstanding(row, out_dir: Path) -> bool:
    import branch_metrics as BM
    tag = "" if row["checkpoint"] == "best" else f"__{row['checkpoint']}"
    path = out_dir / f"{Path(row['run_dir']).name}__{row['eval_split']}{tag}.json"
    if not path.exists():
        return True
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return True
    return not BM.result_is_complete(payload, REPS, ["self", "bank"], BM.TRAINING_BATCH_SIZE,
                                     BM.KNN_K, 50)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(os.environ.get("ABL_ROOT", "./outputs/ablations")) / "main_branch_metrics" / "results")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="restrict to these methods (e.g. dual_topk dual_batch_topk dual_jump_relu)")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = rows_from_manifest(args.manifest, args.methods)
    if args.index is not None:
        rows = rows[args.index:args.index + 1]   # index into the full, stable list
    rows = [r for r in rows if outstanding(r, args.out_dir)]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        rows = rows[i::n]
    print(f"{len(rows)} evaluation(s) to run")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    for row in rows:
        cmd = [sys.executable, "branch_metrics.py", "--run-dir", row["run_dir"],
               "--eval-split", row["eval_split"], "--checkpoint", row["checkpoint"],
               "--reps", *REPS, "--retrieval-modes", "self", "bank", "--retrieval-k", "50",
               "--encode-batch-size", "2048", "--knn-k", "20",
               "--num-workers", str(args.num_workers), "--out-dir", str(args.out_dir.resolve())]
        print("$ " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=VISION_DIR, env=env)
        if result.returncode != 0:
            print(f"FAILED: {row['run_dir']} {row['eval_split']}", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
