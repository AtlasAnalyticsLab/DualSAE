#!/usr/bin/env python3
"""Offline evaluation of trained runs with branch_metrics.py (sequential, resumable).

Each run directory is scored on the split(s) its dataset reports: ImageNet-1K on ``val``
(it has no labelled test split); OpenImagesV7 on ``test`` and ``val`` (the collector reads
R^2 / MonoScore / KNN / LP on test and binary purity on val). Every result is written to
``<out_dir>/<run name>__<split>.json``; branch_metrics.py itself skips a (run, split) whose
result file is already complete, so re-running this driver continues where it stopped.

Settings are the ones used for every evaluated run: representations x, xM, r_hat, xhat;
retrieval modes self and bank with k=50; KNN k=20; encode batch size 2048 (the training batch
size -- BatchTopK selects its top-k over the whole batch, so this value matters).

Examples
--------
    python scripts/main/eval_runs.py --runs ./checkpoints/main/main_imagenet_clip_*
    python scripts/main/eval_runs.py --runs ./checkpoints/main/* --print     # commands only
    python scripts/main/eval_runs.py --runs ./checkpoints/cb/* --out_dir ./outputs/main/branch_metrics
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

VISION = Path(__file__).resolve().parents[2]


def splits_for(run_dir: Path) -> list:
    cfg = json.loads((run_dir / "training_config.json").read_text())
    ds = cfg.get("dataset_name") or ("open_images" if "open_image" in str(cfg.get("train_path", "")).lower()
                                     else "imagenet")
    return ["test", "val"] if ds == "open_images" else ["val"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run directories or glob patterns")
    ap.add_argument("--out_dir", default=os.environ.get("MAIN_EVAL_DIR", "./outputs/main/branch_metrics"))
    ap.add_argument("--features_root", default=None,
                    help="remap the runs' recorded feature paths (see branch_metrics.py --features-root)")
    ap.add_argument("--checkpoint", default="best", choices=["best", "final"])
    ap.add_argument("--splits", nargs="+", default=None, help="override the per-dataset split choice")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--print", action="store_true", help="print the commands instead of running them")
    args = ap.parse_args()

    run_dirs = []
    for pat in args.runs:
        run_dirs += [Path(p) for p in sorted(glob.glob(pat))] if any(ch in pat for ch in "*?[") else [Path(pat)]
    # Only trained models are evaluable: the CB pipeline also leaves converted bases (cvt_*) and
    # pruned bases (pruned_*) next to its runs, which branch_metrics cannot score.
    run_dirs = [d.resolve() for d in run_dirs if (d / "training_config.json").exists()
                and not d.name.startswith(("cvt_", "pruned_"))]
    out_dir = str(Path(args.out_dir).resolve())
    for d in run_dirs:
        for split in (args.splits or splits_for(d)):
            cmd = [args.python, str(VISION / "branch_metrics.py"), "--run-dir", str(d),
                   "--eval-split", split, "--checkpoint", args.checkpoint,
                   "--reps", "x", "xM", "r_hat", "xhat", "--retrieval-modes", "self", "bank",
                   "--retrieval-k", "50", "--encode-batch-size", "2048", "--knn-k", "20",
                   "--num-workers", str(args.num_workers), "--out-dir", out_dir]
            if args.features_root:
                cmd += ["--features-root", str(Path(args.features_root).resolve())]
            if args.print:
                print(" ".join(shlex.quote(x) for x in cmd))
                continue
            print(f"[eval] {d.name} [{split}]", flush=True)
            subprocess.run(cmd, check=True, cwd=str(VISION))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
