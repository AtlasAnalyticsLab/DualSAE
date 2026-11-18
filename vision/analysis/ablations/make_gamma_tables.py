#!/usr/bin/env python3
"""Tables of R^2(x_hat) and MonoScore vs. gamma (coefficient of the decorrelation loss L_dcor).

Reads <gamma>/long_final.csv written by `collect_span_ablation.py --checkpoint final`, i.e. the
EPOCH-50 checkpoint of every run (not the best-validation-MonoScore checkpoint used elsewhere;
at large gamma the best-validation selection can fall inside warm-up, where gamma has not acted
yet). Mean +- sd over seeds 42-44. Bold = best setting per row (over the gamma grid and the
main-table gamma). The gamma = 1e-2 column is arm G1e-2, or R0 where the cell's main-table gamma
is exactly 0.01 (ablation_grid.span_alias).

Writes, next to the input:
    gamma_ablation_table.tex / .md   one table, both metrics, with the main-table-gamma column
    gamma_r2.tex, gamma_ms.tex       one table per metric (gamma grid only)
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import statistics as st
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
VISION_DIR = _HERE.parents[1]
for _p in (str(_HERE), str(VISION_DIR), str(VISION_DIR / "scripts" / "ablations" / "grid")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import collect_span_ablation as CS  # noqa: E402

ARMS = [("G0", "0"), ("G1e-3", "10^{-3}"), ("G1e-2", "10^{-2}"), ("G1e-1", "10^{-1}"), ("G1", "1"),
        ("G10", "10")]
ACTS = [("topk", "TopK"), ("batch_topk", "BTopK")]
DS = [("imagenet", "ImageNet-1k"), ("open_images", "Open Images v7")]
ENCS = [("clip", "CLIP"), ("sigclip", "SigLIP2"), ("vit", "ViT")]
METRICS = [("r2_xhat", r"$R^2(\hat{x})$", "R²(x̂)", 4), ("ms", "MonoScore", "MonoScore", 3)]
SEEDS = (42, 43, 44)


def load(long_csv: Path):
    acc = collections.defaultdict(list)
    for r in csv.DictReader(open(long_csv)):
        if r["value"] not in ("", "nan"):
            acc[(r["dataset"], r["encoder"], r["activation"], r["arm"], r["metric"])].append(float(r["value"]))
    stats = {k: (st.mean(v), st.stdev(v), len(v)) for k, v in acc.items()}
    short = [k for k, (_, _, n) in stats.items() if n != len(SEEDS)]
    if short:
        raise SystemExit(f"expected {len(SEEDS)} seeds everywhere; incomplete: {short[:5]}")
    return stats


def pub(d, e, a):
    g = sorted(set(CS.published_gamma((d, e, a), list(SEEDS))))
    return "/".join(f"{x:g}" for x in g)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default = Path(os.environ.get("ABL_ROOT", "./outputs/ablations")) / "gamma"
    ap.add_argument("--long-csv", type=Path, default=default / "long_final.csv")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: next to --long-csv")
    args = ap.parse_args()
    out_dir = args.out_dir or args.long_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    S = load(args.long_csv)

    tex, md = [], []
    cols = ["G0", "G1e-3", "G1e-2", "G1e-1", "G1", "G10", "R0"]
    tex += [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
            r"\caption{DualSAE $\gamma$ ablation (weight on $\mathcal{L}_\mathrm{dcor}$). Mean $\pm$ sd over 3 seeds, "
            r"epoch-50 checkpoint. Every run replays the main-table configuration and changes only $\gamma$. "
            r"Bold: best setting per row. Main: the main-table $\gamma$ retrained.}",
            r"\label{tab:gamma-ablation}",
            r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lll" + "c" * 7 + "c}", r"\toprule",
            r" & & & \multicolumn{6}{c}{$\gamma$} & \multicolumn{2}{c}{Main} \\",
            r"\cmidrule(lr){4-9}\cmidrule(lr){10-11}",
            r"Dataset & Enc. & Act. & " + " & ".join(f"${g}$" for _, g in ARMS) + r" & $\gamma$ & value \\",
            r"\midrule"]
    per_metric_tex = {}
    for mk, mtex, mmd, dp in METRICS:
        tex.append(rf"\multicolumn{{11}}{{l}}{{\textit{{{mtex}}}}} \\")
        md.append(f"\n**{mmd}** (mean ± sd, 3 seeds, epoch 50)\n")
        md.append("| Dataset | Enc. | Act. | γ=0 | γ=1e-3 | γ=1e-2 | γ=1e-1 | γ=1 | γ=10 | main γ | Dual @ main γ |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        single = [r"\begin{tabular}{lll" + "c" * 6 + "}", r"\toprule",
                  r"Data & Enc. & SAE & " + " & ".join(f"$\\gamma={g}$" for _, g in ARMS) + r" \\",
                  r"\midrule"]
        for di, (d, dl) in enumerate(DS):
            for e, el in ENCS:
                for a, al in ACTS:
                    vals = {c: S[(d, e, a, c, mk)] for c in cols}
                    best = max(round(v[0], dp) for v in vals.values())
                    best_grid = max(round(vals[c][0], dp) for c in cols[:-1])

                    def f(c, latex, ref=best):
                        m, s, _ = vals[c]
                        b = round(m, dp) == ref
                        if latex:
                            x = f"{m:.{dp}f}" + r"{\scriptsize$\pm$" + f"{s:.{dp}f}" + "}"
                            return r"\textbf{" + x + "}" if b else x
                        x = f"{m:.{dp}f} ± {s:.{dp}f}"
                        return f"**{x}**" if b else x
                    tex.append(f"{dl if (e, a) == ('clip', 'topk') else ''} & {el if a == 'topk' else ''} & {al} & "
                               + " & ".join(f(c, True) for c in cols[:-1]) + f" & {pub(d, e, a)} & {f('R0', True)} \\\\")
                    md.append(f"| {dl} | {el} | {al} | " + " | ".join(f(c, False) for c in cols[:-1])
                              + f" | {pub(d, e, a)} | {f('R0', False)} |")
                    single.append(f"{dl} & {el} & {al} & "
                                  + " & ".join(f(c, True, best_grid) for c in cols[:-1]) + r" \\")
                if (d, e) != (DS[-1][0], ENCS[-1][0]):
                    tex.append(r"\addlinespace[2pt]")
            if di == 0:
                tex.append(r"\cmidrule(lr){1-11}")
                single.append(r"\midrule")
        if mk != METRICS[-1][0]:
            tex.append(r"\midrule")
        single += [r"\bottomrule", r"\end{tabular}"]
        per_metric_tex[mk] = single
    tex += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (out_dir / "gamma_ablation_table.tex").write_text("\n".join(tex) + "\n")
    (out_dir / "gamma_ablation_table.md").write_text("\n".join(md).lstrip() + "\n")
    (out_dir / "gamma_r2.tex").write_text("\n".join(per_metric_tex["r2_xhat"]) + "\n")
    (out_dir / "gamma_ms.tex").write_text("\n".join(per_metric_tex["ms"]) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out_dir}/gamma_ablation_table.(tex|md), gamma_r2.tex, gamma_ms.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
