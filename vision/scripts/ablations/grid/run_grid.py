#!/usr/bin/env python3
"""Run the DualSAE ablation grid locally, one cell at a time, resumably.

Stages
------
    train     S, Z and the projector arms (default), or any arms given with --arms
              (gamma arms: --arms G0 G1e-3 G1e-2 G1e-1 G1 G10;
               expansion arms: --arms E4 E8 E16 SE4 SE8 SE16 [E32 SE32];
               k arms: --arms K16 K32 K128 K256 SK16 SK32 SK128 SK256)
    frozen    F (sequential training), only for cells whose S, Z and R0 have finished
    reruns    the two x16 expansion points retrained with restricted checkpoint selection
    eval      branch_metrics.py on every finished, unscored reading
    print     print the main.py command of every selected cell and exit

Every stage skips finished work (training: autoencoder_final.pt + best checkpoint, and every F
snapshot; evaluation: a complete branch_metrics JSON), so re-running a stage continues where it
stopped. Use --index (one cell) or --shard i/n to split a stage across machines or array jobs;
scripts/ablations/slurm_template.sh shows one way.

Examples
--------
    python run_grid.py --stage train --arms R0 S Z --only-dataset imagenet
    python run_grid.py --stage frozen
    python run_grid.py --stage train --arms G0 G1e-3 G1e-2 G1e-1 G1 G10
    python run_grid.py --stage train --arms E4 E8 E16 SE4 SE8 SE16 --seeds 42
    python run_grid.py --stage reruns
    python run_grid.py --stage eval
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
VISION_DIR = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
import ablation_grid as C  # noqa: E402
import eval_rows as E  # noqa: E402


def select_cells(args) -> List[dict]:
    if args.stage == "frozen":
        grid = C.build_grid(["F"])
    elif args.stage == "reruns":
        grid = C.build_selection_reruns()
    else:
        arms = args.arms or list(C.INDEPENDENT_ARMS)
        if "F" in arms:
            sys.exit("arm F depends on finished runs; run it with --stage frozen")
        grid = C.build_grid(arms)
    out = []
    for cell in grid:
        if args.only_dataset and cell["dataset"] != args.only_dataset:
            continue
        if args.only_encoder and cell["encoder"] != args.only_encoder:
            continue
        if args.only_activation and cell["activation"] != args.only_activation:
            continue
        if args.seeds and cell["seed"] not in args.seeds:
            continue
        out.append(cell)
    return out


def shard(items: list, args) -> list:
    """--shard i/n over the outstanding items (--index is applied to the full list earlier)."""
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        return items[i::n]
    return items


def child_env() -> dict:
    env = dict(os.environ)
    # Offline and W&B-free unless the caller explicitly chose otherwise.
    env.setdefault("WANDB_MODE", "disabled")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        env.setdefault(key, "1")
    return env


def run_training(args) -> int:
    cells = select_cells(args)
    if args.index is not None:
        # Index into the full, stable cell list (not the outstanding one, which shrinks while
        # array tasks finish), so a task always maps to the same cell.
        cells = cells[args.index:args.index + 1]
    todo, done, blocked = [], 0, 0
    for cell in cells:
        if cell["arm"] == "F" and not C.f_prerequisites_met(cell):
            blocked += 1          # not an error: re-run this stage as S/Z/R0 finish
            continue
        if C.cell_is_complete(cell):
            done += 1
            continue
        todo.append(cell)
    todo = shard(todo, args)
    print(f"stage={args.stage}  cells={len(cells)}  done={done}  "
          f"waiting_on_prerequisites={blocked}  to_run={len(todo)}")
    for cell in todo:
        cmd = [sys.executable, "main.py", *cell["argv"]]
        print("\n$ " + " ".join(cmd), flush=True)
        if args.stage == "print" or args.dry_run:
            continue
        C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(cmd, cwd=VISION_DIR, env=child_env())
        if result.returncode != 0:
            print(f"FAILED: {cell['exp_name']} (exit {result.returncode})", file=sys.stderr)
            return result.returncode
        # Cached validation intermediates (~2 GB per run) that nothing in this suite reads.
        (C.OUTPUT_DIR / cell["exp_name"] / "best_val_monoscore_intermediates.pt").unlink(missing_ok=True)
    return 0


def eval_row_outstanding(row: dict) -> bool:
    """True if this row still has work, judged by branch_metrics' own completion rule."""
    sys.path.insert(0, str(VISION_DIR))
    import branch_metrics as BM
    run = Path(row["run_dir"])
    tag = "" if row["checkpoint"] == "best" else f"__{row['checkpoint']}"
    path = C.BRANCH_METRICS_RESULTS / f"{run.name}__{row['eval_split']}{tag}.json"
    if not path.exists():
        return True
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return True
    return not BM.result_is_complete(payload, row["reps"].split(), ["self", "bank"],
                                     BM.TRAINING_BATCH_SIZE, BM.KNN_K, 50)


def run_eval(args) -> int:
    rows = []
    cells = C.build_grid(args.arms) + (C.build_selection_reruns() if not args.arms else [])
    for cell in cells:
        if args.seeds and cell["seed"] not in args.seeds:
            continue
        rows.extend(E.rows_for_cell(cell))
    # Roles sharing an F snapshot produce identical work; only the first of them is run.
    unique, seen = [], set()
    for row in rows:
        if args.only_dataset and row["dataset"] != args.only_dataset:
            continue
        if args.only_encoder and row["encoder"] != args.only_encoder:
            continue
        if args.only_activation and row["activation"] != args.only_activation:
            continue
        key = (row["run_dir"], row["eval_split"], row["checkpoint"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    if args.index is not None:
        # Index into the full list of finished runs' readings; stable once training is complete.
        unique = unique[args.index:args.index + 1]
    todo, done = [], 0
    for row in unique:
        if eval_row_outstanding(row):
            todo.append(row)
        else:
            done += 1
    todo = shard(todo, args)
    print(f"stage=eval  rows={len(rows)}  unique={len(seen)}  done={done}  to_run={len(todo)}")
    C.BRANCH_METRICS_RESULTS.mkdir(parents=True, exist_ok=True)
    for row in todo:
        # The evaluation protocol of every ablation table: KNN with k=20 cosine neighbours over the
        # full train bank, LP for 10 epochs, retrieval purity with k=50 (self and bank modes).
        cmd = [sys.executable, "branch_metrics.py", "--run-dir", row["run_dir"],
               "--eval-split", row["eval_split"], "--checkpoint", row["checkpoint"],
               "--reps", *row["reps"].split(), "--retrieval-modes", "self", "bank",
               "--retrieval-k", "50", "--encode-batch-size", "2048", "--knn-k", "20",
               "--num-workers", str(args.num_workers), "--out-dir", str(C.BRANCH_METRICS_RESULTS)]
        print("\n$ " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=VISION_DIR, env=child_env())
        if result.returncode != 0:
            print(f"FAILED: {row['run_dir']} {row['eval_split']}", file=sys.stderr)
            return result.returncode
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["train", "frozen", "reruns", "eval", "print"])
    ap.add_argument("--arms", nargs="+", choices=list(C.ALL_ARMS), default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--only-dataset", choices=C.DATASETS, default=None)
    ap.add_argument("--only-encoder", choices=C.ENCODERS, default=None)
    ap.add_argument("--only-activation", choices=C.ACTIVATIONS, default=None)
    ap.add_argument("--index", type=int, default=None,
                    help="run only the i-th item of the full (stable) list; skipped if already done")
    ap.add_argument("--shard", default=None, help="i/n: run every n-th outstanding item from i")
    ap.add_argument("--num-workers", type=int, default=8, help="data-loader workers for evaluation")
    ap.add_argument("--root", type=Path, default=None,
                    help="grid output directory (default: $ABL_ROOT/grid)")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running them")
    args = ap.parse_args()
    if args.root is not None:
        C.set_root(args.root)
    if args.stage == "eval":
        return run_eval(args)
    return run_training(args)


if __name__ == "__main__":
    sys.exit(main())
