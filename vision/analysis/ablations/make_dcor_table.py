#!/usr/bin/env python3
"""Decorrelation-loss ablation table: metric change when L_dcor is removed (gamma = 0).

Delta = (DualSAE trained without L_dcor) - (DualSAE with L_dcor), mean +- sd over seeds of the
per-seed differences, for R^2 (x_hat), MonoScore (x), binary purity (sparse code), KNN and LP
(x_hat); ImageNet-1K on val (top-1 accuracy), OpenImagesV7 on test (micro-F1; purity from val).

Arms (scripts/ablations/dcor/run_dcor.py):
  ImageNet-1K   ablated  = dcor_imagenet_<enc>_<act>_span0_seed<s>   (main-table config, gamma = 0)
                reference = the main-table DualSAE run of the same cell and seed
  OpenImagesV7  ablated  = dcor_open_images_<enc>_<act>_warm2500_span0_seed<s>
                reference = dcor_open_images_<enc>_<act>_warm2500_span0.01_seed<s>

All metrics are read from branch_metrics.py results with the same read path as the ablation grid
(collect_grid.metrics_from_branch_files). Writes dcor_ablation.{tex,md} and dcor_ablation.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
VISION_DIR = _HERE.parents[1]
for _p in (str(_HERE), str(VISION_DIR), str(VISION_DIR / "scripts" / "ablations" / "grid")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import collect_grid as R  # noqa: E402

ENCS = [("vit", "ViT"), ("clip", "CLIP"), ("sigclip", "SL2")]
ACTS = [("topk", "TK"), ("batch_topk", "BTK"), ("jump_relu", "JR")]
DATA = [("imagenet", "IN"), ("open_images", "OI")]
METRICS = [("r2_xhat", r"$\Delta R^2$"), ("ms", r"$\Delta\mathrm{MS}$"), ("purity", r"$\Delta\mathrm{Pur.}$"),
           ("knn", r"$\Delta\mathrm{KNN}$"), ("lp", r"$\Delta\mathrm{LP}$")]
SEEDS = (42, 43, 44)
DUAL_METHOD = {"topk": "dual_topk", "batch_topk": "dual_batch_topk", "jump_relu": "dual_jump_relu"}


def metrics(run_dir: Path, dataset: str, results: Path):
    vals, problems, _ = R.metrics_from_branch_files(run_dir, dataset, "best", results)
    return (vals if vals and not problems else None), problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = Path(os.environ.get("ABL_ROOT", "./outputs/ablations"))
    ap.add_argument("--dcor-root", type=Path, default=root / "dcor")
    ap.add_argument("--main-manifest", type=Path, required=True,
                    help="main-table runs (scripts/ablations/make_run_manifest.py)")
    ap.add_argument("--main-results", type=Path, default=root / "main_branch_metrics" / "results")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: --dcor-root")
    args = ap.parse_args()
    out_dir = args.out_dir or args.dcor_root
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.dcor_root / "checkpoints"
    results = args.dcor_root / "branch_metrics" / "results"
    main_runs = {(r["dataset"], r["encoder"], r["method"], int(r["seed"])): Path(r["run_dir"])
                 for r in json.loads(args.main_manifest.read_text())}

    table, problems, csv_rows = {}, [], []
    for ds, _ in DATA:
        for enc, _ in ENCS:
            for act, _ in ACTS:
                diffs = {m: [] for m, _ in METRICS}
                for seed in SEEDS:
                    if ds == "imagenet":
                        abl = ckpt / f"dcor_imagenet_{enc}_{act}_span0_seed{seed}"
                        ref_dir = main_runs.get((ds, enc, DUAL_METHOD[act], seed))
                        ref_results = args.main_results
                    else:
                        abl = ckpt / f"dcor_open_images_{enc}_{act}_warm2500_span0_seed{seed}"
                        ref_dir = ckpt / f"dcor_open_images_{enc}_{act}_warm2500_span0.01_seed{seed}"
                        ref_results = results
                    a, pa = metrics(abl, ds, results)
                    b, pb = (metrics(ref_dir, ds, ref_results) if ref_dir is not None
                             else (None, [f"no main-table run for {ds}/{enc}/{act}/seed{seed}"]))
                    problems += pa + pb
                    for m, _ in METRICS:
                        if a is not None and b is not None and m in a and m in b:
                            diffs[m].append(a[m] - b[m])
                            csv_rows.append({"dataset": ds, "encoder": enc, "activation": act, "seed": seed,
                                             "metric": m, "ablated": a[m], "reference": b[m],
                                             "delta": a[m] - b[m]})
                # A cell is reported only when every seed has both arms.
                table[(ds, enc, act)] = {
                    m: ((statistics.fmean(v), statistics.stdev(v)) if len(v) == len(SEEDS) else None)
                    for m, v in diffs.items()}

    def cell(st, latex):
        if st is None:
            return "–"
        return f"{st[0]:+.3f}" + (r"{\scriptsize$\pm$" + f"{st[1]:.3f}" + "}" if latex else f" ± {st[1]:.3f}")

    tex = [r"\begin{tabular}{ll" + "c" * len(METRICS) * len(DATA) + "}", r"\toprule",
           r"\multirow{2}{*}{Enc.} & \multirow{2}{*}{SAE} & "
           + " & ".join(rf"\multicolumn{{{len(METRICS)}}}{{c}}{{{dl}}}" for _, dl in DATA) + r" \\",
           " & & " + " & ".join(lab for _ in DATA for _, lab in METRICS) + r" \\", r"\midrule"]
    md = ["| Enc. | SAE | " + " | ".join(f"{dl} {m}" for _, dl in DATA for m, _ in METRICS) + " |",
          "|" + "---|" * (2 + len(METRICS) * len(DATA))]
    for enc, el in ENCS:
        for act, al in ACTS:
            vals = [table[(ds, enc, act)][m] for ds, _ in DATA for m, _ in METRICS]
            tex.append(f"{el} & {al} & " + " & ".join(cell(v, True) for v in vals) + r" \\")
            md.append(f"| {el} | {al} | " + " | ".join(cell(v, False) for v in vals) + " |")
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex.append(r"\end{tabular}")
    (out_dir / "dcor_ablation.tex").write_text("\n".join(tex) + "\n")
    (out_dir / "dcor_ablation.md").write_text("\n".join(md) + "\n")
    with (out_dir / "dcor_ablation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "encoder", "activation", "seed", "metric",
                                                    "ablated", "reference", "delta"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print("\n".join(md))
    if problems:
        print(f"\n{len(set(problems))} unavailable reading(s), e.g. {sorted(set(problems))[:5]}")
    print(f"wrote {out_dir}/dcor_ablation.(tex|md|csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
