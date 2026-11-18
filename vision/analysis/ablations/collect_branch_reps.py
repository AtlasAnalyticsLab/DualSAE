#!/usr/bin/env python3
"""Collect per-representation metrics of the main-table runs from branch_metrics.py results.

For every run in the manifest (scripts/ablations/make_run_manifest.py) and every representation
(x, x^M = Sparse Branch output `xM`, r_hat = Residual Branch output `r_hat`, x_hat = `xhat`) this
reads KNN and LP (top-1 accuracy on ImageNet-1K val, micro-F1 on OpenImagesV7 test), R^2 and
MonoScore, and writes one long CSV:

    dataset, encoder, method, seed, rep, metric, value

It is the input of plot_branch_bars.py (class information per branch) and of
make_dcor_table.py. Results that fail branch_metrics' own self-checks are skipped and listed.

    python collect_branch_reps.py --manifest ./outputs/main_runs.json \
        --results ./outputs/ablations/main_branch_metrics/results \
        --out ./outputs/ablations/main_branch_metrics/branch_reps.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PRIMARY_SPLIT = {"imagenet": "val", "open_images": "test"}
REPS = ("x", "xM", "r_hat", "xhat")


def read_result(results: Path, run_dir: str, split: str, checkpoint: str = "best"):
    tag = "" if checkpoint == "best" else f"__{checkpoint}"
    path = results / f"{Path(run_dir).name}__{split}{tag}.json"
    if not path.exists():
        return None, f"missing {path.name}"
    payload = json.loads(path.read_text())
    if not payload.get("self_checks", {}).get("all_ok", False):
        return None, f"self-checks failed: {path.name}"
    return payload, None


def rows_for_run(row: dict, results: Path):
    out, problems = [], []
    payload, err = read_result(results, row["run_dir"], PRIMARY_SPLIT[row["dataset"]],
                               row.get("checkpoint", "best"))
    if payload is None:
        return out, [f"{Path(row['run_dir']).name}: {err}"]
    base = {k: row[k] for k in ("dataset", "encoder", "method", "seed")}
    for rep in REPS:
        block = payload["reps"].get(rep)
        if block is None:
            continue
        values = {
            "knn": block.get("knn", {}).get("metric"),
            "lp": block.get("lp", {}).get("metric"),
            "r2_perfeat": block.get("r2_perfeat"),
            "r2_global": block.get("r2_global"),
            "monoscore": block.get("monoscore", {}).get("mean_all"),
        }
        for metric, value in values.items():
            if value is not None:
                out.append({**base, "rep": rep, "metric": metric, "value": value})
    return out, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = Path(os.environ.get("ABL_ROOT", "./outputs/ablations")) / "main_branch_metrics"
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=root / "results")
    ap.add_argument("--out", type=Path, default=root / "branch_reps.csv")
    args = ap.parse_args()
    rows, problems = [], []
    for r in json.loads(args.manifest.read_text()):
        got, bad = rows_for_run(r, args.results)
        rows += got
        problems += bad
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "encoder", "method", "seed", "rep",
                                                    "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} values)")
    for p in problems:
        print("  unavailable:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
