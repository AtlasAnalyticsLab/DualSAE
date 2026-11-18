"""Expansion-ratio figures of the supplementary material (ImageNet-1K and OpenImagesV7).

Rows: R^2, MonoScore, purity, KNN, LP; columns: encoder x Sparse-Branch activation; x axis: expansion
ratio m/d (log scale). DualSAE vs. the standard SAE (DualSAE's Sparse Branch trained without the
Residual Branch, grid arms SE<r> / S).

Usage: python plot_expansion.py <expansion_data.json> <outdir>
       (expansion_data.json is written by build_expansion_data.py)
"""
import json, sys, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, FixedLocator, NullLocator
from matplotlib.lines import Line2D

D = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
# Points to draw hollow from a different checkpoint ("<cell>|<model>|<key>" -> metrics); none are.
OVERRIDE = {}
# Points left as gaps. The x32 runs are trained by the grid but not plotted (SHOW_X); the ViT x32
# points selected an epoch-1 checkpoint and are listed here so they stay hidden if SHOW_X grows.
# The two x16 points with an early selection are replaced by their selection reruns in
# build_expansion_data.py.
FL = {"imagenet|vit|topk|dual|32", "imagenet|vit|batch_topk|dual|32", "imagenet|vit|topk|sae|32", "imagenet|vit|batch_topk|sae|32"}
KEYS = ["4", "8", "pub", "16", "32"]
SHOW_X = (4.0, 8.0, 16.0)  # plotted expansion ratios; ViT's 8192-latent run is its x8 point
DUAL, SAE = "#2a78d6", "#eb6834"          # reference palette slots 1-2 (validated, light)
INK, INK2, GRID, AXIS = "#1f1f1f", "#52514e", "#e4e4e2", "#9a9994"
SETTINGS = [("clip", "topk", "CLIP · TopK"), ("clip", "batch_topk", "CLIP · BatchTopK"),
            ("sigclip", "topk", "SigLIP2 · TopK"), ("sigclip", "batch_topk", "SigLIP2 · BatchTopK"),
            ("vit", "topk", "ViT · TopK"), ("vit", "batch_topk", "ViT · BatchTopK")]
# (key, y-label, scale, share-y-across-row, tick format)
METRICS = [("r2_xhat", r"$R^2$ $\uparrow$", 1, True, "%.2f"),
           ("ms", r"MS $\uparrow$", 1, True, "%.1f"),
           ("purity", r"Purity $\uparrow$", 1, True, "%.1f"),
           ("knn", r"KNN $\uparrow$", 1, True, "%.3f"),
           ("lp", r"LP $\uparrow$", 1, True, "%.3f")]

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5, "font.family": "DejaVu Sans",
    "axes.linewidth": 0.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.titlesize": 6.8,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "axes.labelsize": 6.8,
})

def series(cell, model, met, scale):
    pts = []
    for k in KEYS:
        r = D.get(cell, {}).get(model, {}).get(k)
        if r is None or r["x"] not in SHOW_X:
            continue
        key = f"{cell}|{model}|{k}"
        if key in OVERRIDE:
            pts.append((r["x"], OVERRIDE[key][met] * scale, True))
        elif key in FL:
            pts.append((r["x"], math.nan, False))
        else:
            pts.append((r["x"], r[met] * scale, False))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]

def make(ds, ds_name, fname):
    nr, nc = len(METRICS), len(SETTINGS)
    PANEL_H, GAP_H, TOP_IN, BOT_IN = 0.637, 0.14, 0.508, 0.321  # inches
    H = TOP_IN + BOT_IN + nr * PANEL_H + (nr - 1) * GAP_H
    fig, axes = plt.subplots(nr, nc, figsize=(6.875, H), sharex=True)
    fig.subplots_adjust(left=0.085, right=0.982, top=1 - TOP_IN / H, bottom=BOT_IN / H,
                        wspace=0.18, hspace=GAP_H / PANEL_H)
    for ri, (met, ylab, scale, share, fmt) in enumerate(METRICS):
        row_vals = []
        for ci, (enc, act, title) in enumerate(SETTINGS):
            ax = axes[ri, ci]
            cell = f"{ds}|{enc}|{act}"
            ax.set_xscale("log", base=2)
            for model, col, mk in [("sae", SAE, "s"), ("dual", DUAL, "o")]:
                xs, ys, hollow = series(cell, model, met, scale)
                row_vals += [y for y in ys if not math.isnan(y)]
                z = 3 if model == "dual" else 2
                ax.plot(xs, ys, color=col, lw=1.1, marker=mk, ms=3.0, mec="white", mew=0.45,
                        markevery=[i for i, h in enumerate(hollow) if not h], zorder=z, clip_on=False)
                hx = [x for x, h in zip(xs, hollow) if h]
                hy = [y for y, h in zip(ys, hollow) if h]
                if hx:
                    ax.plot(hx, hy, ls="none", marker=mk, ms=3.0, mfc="white", mec=col, mew=0.9,
                            zorder=z + 2, clip_on=False)
            ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
            ax.xaxis.set_major_locator(FixedLocator([4, 8, 16]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticklabels([r"$\times$4", r"$\times$8", r"$\times$16"])
            ax.set_xlim(3.3, 19.4)
            if ri == 0:
                ax.set_title(title, color=INK, pad=4)
            if ci == 0:
                ax.set_ylabel(ylab)
        if share:  # one y-range per row; tick labels only on the first panel
            lo, hi = min(row_vals), max(row_vals)
            pad = 0.08 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            if met != "r2_xhat":
                lo = max(lo, 0) if met != "dead_frac" else max(lo, -3)
            for ci in range(nc):
                axes[ri, ci].set_ylim(lo, hi)
                if ci > 0:
                    axes[ri, ci].tick_params(labelleft=False)
        else:  # same y-span in every panel of the row, centred on each panel's data (level depends on encoder)
            rng = []
            for enc, act, _ in SETTINGS:
                v = [y for m in ("dual", "sae") for y in series(f"{ds}|{enc}|{act}", m, met, scale)[1] if not math.isnan(y)]
                rng.append((min(v), max(v)))
            span = 1.3 * max(h - l for l, h in rng)
            for ci, (l, h) in enumerate(rng):
                c = (l + h) / 2
                axes[ri, ci].set_ylim(c - span / 2, c + span / 2)
    fig.canvas.draw()  # fix tick decimals from the final tick spacing (avoids duplicate labels)
    for ax in axes.flat:
        t = ax.yaxis.get_majorticklocs()
        dec = next(d for d in range(7) if all(abs(round(v, d) - v) < 1e-9 for v in t))
        ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{dec}f"))
    fig.align_ylabels(axes[:, 0])
    fig.text(0.085, 1 - 0.134 / H, ds_name, fontsize=7.5, fontweight="bold", color=INK, va="center", ha="left")
    handles = [Line2D([], [], color=DUAL, lw=1.1, marker="o", ms=3.5, mec="white", mew=0.45, label="DualSAE"),
               Line2D([], [], color=SAE, lw=1.1, marker="s", ms=3.5, mec="white", mew=0.45, label="Standard SAE")]
    if any(k.startswith(ds + "|") for k in OVERRIDE):
        handles.append(Line2D([], [], ls="none", marker="o", ms=3.5, mfc="white", mec=INK2, mew=0.9,
                              label="Final checkpoint (see caption)"))
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.982, 1 - 0.134 / H), ncol=len(handles), frameon=False,
               fontsize=6.8, handlelength=2.2, columnspacing=1.6, labelcolor=INK)
    fig.text(0.54, 0.064 / H, r"Expansion ratio $m/d$ (log scale)", ha="center", va="bottom", fontsize=6.8, color=INK)
    fig.savefig(f"{OUT}/{fname}.pdf")
    fig.savefig(f"{OUT}/{fname}.png", dpi=220)
    plt.close(fig)

make("imagenet", "ImageNet-1K", "expansion_ratio_imagenet")
make("open_images", "OpenImagesV7", "expansion_ratio_openimages")
