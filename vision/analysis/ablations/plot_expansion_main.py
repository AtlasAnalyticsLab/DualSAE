"""Main-paper expansion-ratio figure: R^2 and MonoScore of the standard SAE (DualSAE's Sparse Branch
trained without the Residual Branch) and DualSAE at expansion ratios m/d = 4, 8, 16, BatchTopK on
ImageNet-1K, CLIP / SigLIP2 / ViT. Same data as the supplementary figures (plot_expansion.py).

Usage: python plot_expansion_main.py <expansion_data.json> <outdir>
       (expansion_data.json is written by build_expansion_data.py)
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA, OUT = sys.argv[1], sys.argv[2]
D = json.load(open(DATA))

DS, ACT = "imagenet", "batch_topk"
ENCS = [("clip", "CLIP"), ("sigclip", "SigLIP2"), ("vit", "ViT")]
XS = (4.0, 8.0, 16.0)
METRICS = [("r2_xhat", "$R^2$", (0.75, 1.01), [0.8, 0.9, 1.0]),
           ("ms", "MonoScore", (0.2, 0.7), [0.2, 0.4, 0.6])]
STYLE = {"sae": dict(color="#5b9bd5", marker="o", ms=3.6, label="SAE"),
         "dual": dict(color="#e8822a", marker="*", ms=6.0, label="DualSAE")}


def series(enc, model, met):
    recs = [r for r in D[f"{DS}|{enc}|{ACT}"][model].values() if r and r["x"] in XS]
    assert sorted(r["x"] for r in recs) == list(XS), (enc, model, [r["x"] for r in recs])
    recs.sort(key=lambda r: r["x"])
    return [r["x"] for r in recs], [r[met] for r in recs]


# Print the plotted values so the figure can be checked against the data file.
for enc, _ in ENCS:
    for met, *_ in METRICS:
        _, ys = series(enc, "sae", met)
        _, yd = series(enc, "dual", met)
        print(f"{enc:8s} {met:8s} SAE {[round(v, 3) for v in ys]}  DualSAE {[round(v, 3) for v in yd]}")

FS = 6.5
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": FS, "axes.labelsize": FS,
                     "axes.titlesize": FS + 0.5, "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5,
                     "axes.linewidth": 0.6, "axes.edgecolor": "0.35", "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
                     "xtick.minor.size": 0, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5})

fig, axs = plt.subplots(2, 3, figsize=(3.3, 1.45), sharex=True, sharey="row")
fig.subplots_adjust(wspace=0.10, hspace=0.22)
for j, (enc, name) in enumerate(ENCS):
    for i, (met, ylab, ylim, yticks) in enumerate(METRICS):
        ax = axs[i, j]
        for model in ("sae", "dual"):
            st = STYLE[model]
            xs, ys = series(enc, model, met)
            ax.plot(xs, ys, color=st["color"], lw=1.0, marker=st["marker"], ms=st["ms"], mec="0.1", mew=0.4,
                    zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(XS)
        ax.set_xticklabels([r"$\times$4", r"$\times$8", r"$\times$16"])
        ax.set_xlim(3.4, 19)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.grid(alpha=0.35, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if i == 0:
            ax.set_title(name, pad=2)
        if j == 0:
            ax.set_ylabel(ylab, labelpad=2)
axs[1, 1].set_xlabel("Expansion ratio $m/d$", labelpad=1)
fig.align_ylabels(axs[:, 0])
h = [Line2D([], [], color=s["color"], lw=1.0, marker=s["marker"], ms=s["ms"] * 1.1, mec="0.1", mew=0.4,
            label=s["label"]) for s in STYLE.values()]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False, fontsize=FS - 0.5,
           handlelength=1.8, columnspacing=1.6)
fig.savefig(f"{OUT}/expansion_ratio_main.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(f"{OUT}/expansion_ratio_main.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
