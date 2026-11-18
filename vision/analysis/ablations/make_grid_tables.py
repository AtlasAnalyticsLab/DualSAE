#!/usr/bin/env python3
"""Supplementary tables built from the ablation-grid collector output (collect_grid.py -> long.csv).

1. Joint training vs. sequential training (frozen_vs_dualsae.{tex,md}):
     Sequential Training = arm F read after the same number of Residual Branch updates that the
                           jointly trained DualSAE (arm R0) made up to its selected checkpoint
                           (role at_D_best);
     Joint Training      = --joint published_dual (default; the main-table DualSAE runs, scored by
                           branch_metrics.py) or --joint R0 (the main-table config retrained in the grid).
2. Projector of the Residual Branch (projector_ablation.{tex,md}): arms R0 (Default,
   Linear(A,A)+GELU), B256, B512, D3, D4 at their best-validation-MonoScore checkpoint. The grid
   also trains L1, B64, B2048 and B512G; add them with --projector-arms.

Columns: R^2 (x_hat), MonoScore (on x), binary purity (sparse code), KNN and LP (x_hat); top-1
accuracy on ImageNet-1K, micro-F1 on OpenImagesV7. Mean +- sd over seeds; bold marks the higher /
highest mean in each setting.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import statistics
import sys
from pathlib import Path

DATA = [("imagenet", "IN"), ("open_images", "OI")]
ENCS = [("clip", "CLIP"), ("sigclip", "SL2"), ("vit", "ViT")]
ACTS = [("topk", "TK"), ("batch_topk", "BTK")]
METRICS = [("r2_xhat", r"$R^2$", "R²"), ("ms", "MS", "MS"), ("purity", "Pur.", "Pur."),
           ("knn", "KNN", "KNN"), ("lp", "LP", "LP")]
PROJ_LABEL = {"R0": "Default", "L1": "L1", "B64": "B64", "B256": "B256", "B512": "B512",
              "B2048": "B2048", "B512G": "B512G", "D3": "D3", "D4": "D4"}
SEEDS = (42, 43, 44)
DIGITS = 4


def load(long_csv: Path):
    raw = collections.defaultdict(dict)
    for r in csv.DictReader(open(long_csv)):
        raw[(r["dataset"], r["encoder"], r["activation"], r["arm"], r["role"], r["metric"])][int(r["seed"])] = float(r["value"])
    return raw


def stat(raw, key):
    per_seed = raw.get(key, {})
    vals = [per_seed[s] for s in SEEDS if s in per_seed]
    if len(vals) != len(SEEDS):
        return None  # never a mean over fewer seeds than the others
    return statistics.fmean(vals), statistics.stdev(vals)


def cell_text(st, bold, latex):
    if st is None:
        return "–"
    m, s = st
    if latex:
        x = f"{m:.{DIGITS}f}" + r"{\scriptsize$\pm$" + f"{s:.{DIGITS}f}" + "}"
        return r"\textbf{" + x + "}" if bold else x
    x = f"{m:.{DIGITS}f} ± {s:.{DIGITS}f}"
    return f"**{x}**" if bold else x


def rows_block(raw, setting, entries):
    """entries: [(label, arm, role)] -> rows of formatted cells, bold = best mean per metric."""
    d, e, a = setting
    stats = {lab: [stat(raw, (d, e, a, arm, role, m)) for m, _, _ in METRICS] for lab, arm, role in entries}
    best = []
    for j in range(len(METRICS)):
        vals = [round(stats[lab][j][0], DIGITS) for lab, _, _ in entries if stats[lab][j] is not None]
        best.append(max(vals) if vals else None)
    out = []
    for lab, _, _ in entries:
        row = []
        for j in range(len(METRICS)):
            st = stats[lab][j]
            row.append((st, st is not None and round(st[0], DIGITS) == best[j]))
        out.append((lab, row))
    return out


def write_table(path_stem: Path, title: str, blocks) -> None:
    """blocks: [((dataset label, enc label, act label), [(row label, [(stat, bold)])])]"""
    header_md = "| Data | Enc. | SAE | Method | " + " | ".join(m[2] for m in METRICS) + " |"
    md = [f"**{title}** (mean ± sd over seeds {', '.join(map(str, SEEDS))})", "", header_md,
          "|" + "---|" * (4 + len(METRICS))]
    tex = [r"\begin{tabular}{@{}llll" + "c" * len(METRICS) + "@{}}", r"\toprule",
           "Data & Enc. & SAE & Method & " + " & ".join(m[1] for m in METRICS) + r" \\", r"\midrule"]
    for (dl, el, al), rows in blocks:
        for lab, cells in rows:
            md.append(f"| {dl} | {el} | {al} | {lab} | " + " | ".join(cell_text(s, b, False) for s, b in cells) + " |")
            tex.append(f"{dl} & {el} & {al} & {lab} & " + " & ".join(cell_text(s, b, True) for s, b in cells) + r" \\")
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex.append(r"\end{tabular}")
    path_stem.with_suffix(".md").write_text("\n".join(md) + "\n")
    path_stem.with_suffix(".tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default = Path(os.environ.get("ABL_ROOT", "./outputs/ablations")) / "grid"
    ap.add_argument("--long-csv", type=Path, default=default / "long.csv")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: next to --long-csv")
    ap.add_argument("--joint", choices=["published_dual", "R0"], default="published_dual",
                    help="which DualSAE runs form the Joint Training rows")
    ap.add_argument("--projector-arms", nargs="+", default=["R0", "B256", "B512", "D3", "D4"],
                    choices=list(PROJ_LABEL))
    args = ap.parse_args()
    out_dir = args.out_dir or args.long_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load(args.long_csv)

    settings = [((d, e, a), (dl, el, al)) for d, dl in DATA for e, el in ENCS for a, al in ACTS]
    frozen = [(labels, rows_block(raw, s, [("Sequential Training", "F", "at_D_best"),
                                           ("Joint Training", args.joint, "best")]))
              for s, labels in settings]
    write_table(out_dir / "frozen_vs_dualsae", "Sequential vs. joint training", frozen)
    proj = [(labels, rows_block(raw, s, [(PROJ_LABEL[arm], arm, "best") for arm in args.projector_arms]))
            for s, labels in settings]
    write_table(out_dir / "projector_ablation", "Projector of the Residual Branch", proj)
    print(f"wrote {out_dir}/frozen_vs_dualsae.(tex|md) and projector_ablation.(tex|md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
