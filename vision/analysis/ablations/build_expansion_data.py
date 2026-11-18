#!/usr/bin/env python3
"""Build the plot data of the expansion-ratio figures from the collected evaluation outputs.

Input:  <expansion>/long.csv written by collect_scale_ablation.py (one value per cell, arm, seed and
        metric), plus the branch_metrics results of the two selection reruns
        (ablation_grid.SELECTION_RERUNS), which replace their original points.
Output: <expansion>/expansion_data.json, keyed "<dataset>|<encoder>|<activation>" ->
        {"dual"|"sae": {"<ratio>"|"pub": {"n", "x", "r2_xhat", "ms", "purity", "knn", "lp",
                                           "dead_frac", "best_epoch"}}}

"pub" is the 8192-latent point (arms R0 / S); its ratio x = 8192 / encoder width, so for ViT it is
the x8 point. Values are means over the requested seeds (the expansion figures use seed 42 only).

    python build_expansion_data.py --seeds 42
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
VISION_DIR = _HERE.parents[1]
for _p in (str(_HERE), str(VISION_DIR), str(VISION_DIR / "scripts" / "ablations" / "grid")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ablation_grid as C  # noqa: E402
import collect_grid as R  # noqa: E402

FIELDS = ("r2_xhat", "ms", "purity", "knn", "lp", "dead_frac", "best_epoch")


def point_key(arm: str) -> str:
    """"pub" for the 8192/64 anchor, else the expansion ratio as a string ("4", "8", ...)."""
    if arm in ("R0", "S"):
        return "pub"
    dual = C.SCALE_SAE_ARMS.get(arm, arm)
    return str(C.SCALE_DUAL_ARMS[dual]["expansion"])


def is_expansion_arm(arm: str) -> bool:
    dual = C.SCALE_SAE_ARMS.get(arm, arm)
    return arm in ("R0", "S") or (dual in C.SCALE_DUAL_ARMS and "expansion" in C.SCALE_DUAL_ARMS[dual])


def build(long_csv: Path, seeds, grid_root: Path | None = None) -> dict:
    if grid_root is not None:
        C.set_root(grid_root)
    acc = defaultdict(list)
    for r in csv.DictReader(open(long_csv)):
        if int(r["seed"]) not in seeds or not is_expansion_arm(r["arm"]) or r["metric"] not in FIELDS:
            continue
        if r["value"] in ("", "nan"):
            continue
        model = "sae" if C.is_sae_arm(r["arm"]) else "dual"
        acc[(r["dataset"], r["encoder"], r["activation"], model, point_key(r["arm"]),
             int(r["n_latents"]), r["metric"])].append(float(r["value"]))

    data: dict = {}
    for (ds, enc, act, model, key, n, metric), vals in acc.items():
        if len(vals) != len(seeds):
            continue  # a mean over fewer seeds than requested would be a different quantity
        rec = data.setdefault(f"{ds}|{enc}|{act}", {}).setdefault(model, {}).setdefault(
            key, {"n": n, "x": round(n / C.ACT_DIM[enc], 4)})
        rec[metric] = statistics.fmean(vals)

    # Substitute the retrained points (restricted checkpoint selection), read the same way as the
    # collector reads every other run.
    for spec in C.SELECTION_RERUNS:
        if spec["seed"] not in seeds:
            continue
        run = C.OUTPUT_DIR / f"{C.exp_name(spec['dataset'], spec['encoder'], spec['activation'], spec['arm'], spec['seed'])}_sel{spec['select_from_epoch']}"
        if not C.run_is_complete(run):
            print(f"warning: selection rerun {run.name} not finished; keeping the original point")
            continue
        metrics, problems, _ = R.metrics_from_branch_files(run, spec["dataset"], "best",
                                                           C.BRANCH_METRICS_RESULTS)
        if problems or not metrics:
            print(f"warning: {run.name}: {problems}; keeping the original point")
            continue
        model = "sae" if C.is_sae_arm(spec["arm"]) else "dual"
        cell = f"{spec['dataset']}|{spec['encoder']}|{spec['activation']}"
        rec = data.setdefault(cell, {}).setdefault(model, {}).setdefault(point_key(spec["arm"]), {})
        setting = C.scale_setting(spec["encoder"], spec["arm"])
        rec.update({"n": setting["n_latents"],
                    "x": round(setting["n_latents"] / C.ACT_DIM[spec["encoder"]], 4),
                    **{m: metrics[m] for m in FIELDS if m in metrics},
                    "best_epoch": float(C.read_best_epoch(run)),
                    "selection_rerun": run.name})
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default = Path(os.environ.get("ABL_ROOT", "./outputs/ablations")) / "expansion"
    ap.add_argument("--long-csv", type=Path, default=default / "long.csv")
    ap.add_argument("--out", type=Path, default=default / "expansion_data.json")
    ap.add_argument("--grid-root", type=Path, default=None,
                    help="ablation grid directory (default: $ABL_ROOT/grid)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = ap.parse_args()
    data = build(args.long_csv, set(args.seeds), args.grid_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out} ({len(data)} settings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
