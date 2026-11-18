#!/usr/bin/env python3
"""Class information in each DualSAE branch: KNN / LP of the Sparse Branch output x^M vs. the
Residual Branch output r_hat (branch-decomposition figure, panel b; OpenImagesV7 supplementary figure).

Reads branch_reps.csv (collect_branch_reps.py) and draws, for every (dataset, encoder, activation)
cell of the DualSAE runs, a pair of bars (x^M, r_hat) with mean +- sd over seeds. Nothing is
averaged except seeds. Outputs (in --out-dir):

    xm_vs_rhat_grid.{pdf,png}          2x2 small multiples (rows ImageNet-1K / OpenImagesV7, cols KNN / LP)
    branch_decomposition.{pdf,png}     single-column figure: (a) joint t-SNE panel (--tsne-png, written
                                       by tsne_branches.py), (b) ImageNet-1K KNN over LP
    xm_vs_rhat_openimages.{pdf,png}    OpenImagesV7 KNN and LP in one row

KNN: k = 20 cosine neighbours over the full train bank; LP: 10 training epochs (branch_metrics.py).
ImageNet-1K values are top-1 accuracy (val), OpenImagesV7 values are micro-F1 (test).
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ENCODERS = ["ViT", "CLIP", "SL2"]
ENC_KEY = {"ViT": "vit", "CLIP": "clip", "SL2": "sigclip"}
ACTS = ["TK", "BTK", "JR"]
ACT_METHOD = {"TK": "dual_topk", "BTK": "dual_batch_topk", "JR": "dual_jump_relu"}
DS_KEY = {"IN": "imagenet", "OI": "open_images"}
DATASET_NAME = {"IN": "ImageNet-1K", "OI": "OpenImagesV7"}
# The two datasets report different metrics, so the y-label changes per row.
DATASET_YLABEL = {"IN": "Top-1 accuracy", "OI": "Micro-F1"}
METRICS = ["KNN", "LP"]
SEEDS = (42, 43, 44)

# Style: matplotlib defaults, tab:blue / tab:orange at alpha 0.6 with a thin black edge, y-only
# grid, 300 dpi. x^M takes blue (reconstruction side), r_hat orange (residual side).
C_XM = "tab:blue"
C_RHAT = "tab:orange"
BAR_ALPHA = 0.6
EDGE_LW = 0.4
GRID_ALPHA = 0.25
ERR = "black"
BAR_W = 0.4
BAR_GAP = 0.0
ENC_GAP = 0.6
LABELS = [r"Sparse Branch $x^{M}$", r"Residual Branch $\hat{r}$"]

DATA: dict = {}


def load(csv_path: str) -> dict:
    """DATA[ds][metric][enc][act] = {"xM": (mean, sd), "rhat": (mean, sd)} over the three seeds."""
    acc = collections.defaultdict(dict)
    for r in csv.DictReader(open(csv_path)):
        if r["metric"] not in ("knn", "lp") or r["rep"] not in ("xM", "r_hat"):
            continue
        acc[(r["dataset"], r["encoder"], r["method"], r["rep"], r["metric"])][int(r["seed"])] = float(r["value"])
    data = {}
    for ds, dsk in DS_KEY.items():
        data[ds] = {}
        for metric in METRICS:
            data[ds][metric] = {}
            for enc in ENCODERS:
                data[ds][metric][enc] = {}
                for act in ACTS:
                    cell = {}
                    for rep, name in (("xM", "xM"), ("r_hat", "rhat")):
                        per_seed = acc.get((dsk, ENC_KEY[enc], ACT_METHOD[act], rep, metric.lower()), {})
                        vals = [per_seed[s] for s in SEEDS if s in per_seed]
                        # A bar over fewer seeds than its neighbours would be a different quantity.
                        cell[name] = ((statistics.fmean(vals), statistics.stdev(vals))
                                      if len(vals) == len(SEEDS) else (np.nan, np.nan))
                    data[ds][metric][enc][act] = cell
    return data


def _x_positions():
    """Cell centres along x, with an ENC_GAP break between encoder triplets."""
    centres, mids, x = [], [], 0.0
    for _ in ENCODERS:
        group = []
        for _ in ACTS:
            group.append(x)
            x += 1.0
        centres.extend(group)
        mids.append(float(np.mean(group)))
        x += ENC_GAP
    return np.array(centres), mids


def _panel(ax, ds, metric, show_ylabel):
    """Draw one (dataset, metric) panel: 9 cells x {x^M, r_hat}."""
    centres, mids = _x_positions()
    cells = [DATA[ds][metric][e][a] for e in ENCODERS for a in ACTS]
    xm_m = np.array([c["xM"][0] for c in cells])
    xm_s = np.array([c["xM"][1] for c in cells])
    r_m = np.array([c["rhat"][0] for c in cells])
    r_s = np.array([c["rhat"][1] for c in cells])
    off = (BAR_W + BAR_GAP) / 2.0
    bar_kw = dict(alpha=BAR_ALPHA, edgecolor="black", linewidth=EDGE_LW, zorder=3)
    err_kw = dict(ecolor=ERR, elinewidth=0.8, capsize=2.0, capthick=0.8)
    ax.bar(centres - off, xm_m, BAR_W, yerr=xm_s, color=C_XM, error_kw=err_kw, label=LABELS[0], **bar_kw)
    ax.bar(centres + off, r_m, BAR_W, yerr=r_s, color=C_RHAT, error_kw=err_kw, label=LABELS[1], **bar_kw)
    # Activation names as x-ticks; encoder names as a second, centred tier, offset in points so it
    # clears the tick labels regardless of the panel height.
    ax.set_xticks(centres)
    ax.set_xticklabels(ACTS * len(ENCODERS))
    for enc, mid in zip(ENCODERS, mids):
        ax.annotate(enc, xy=(mid, 0), xycoords=("data", "axes fraction"), xytext=(0, -20),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=plt.rcParams["axes.labelsize"])
    ax.grid(alpha=GRID_ALPHA, axis="y")
    ax.set_axisbelow(True)
    ax.set_xlim(centres[0] - 0.65, centres[-1] + 0.65)
    if show_ylabel:
        ax.set_ylabel(DATASET_YLABEL[ds])


def _ylim(ds):
    """Baseline 0 (bar lengths are honest ratios) and ~10% headroom above the tallest bar + sd."""
    tops = [c[r][0] + c[r][1] for m in METRICS for e in ENCODERS for c in DATA[ds][m][e].values()
            for r in ("xM", "rhat") if np.isfinite(c[r][0])]
    return 0.0, (max(tops) * 1.1 if tops else 1.0)


def _style(ax, ds, metric, tier=-7.5):
    """Compact variant of _panel for the single-column figures."""
    _panel(ax, ds, metric, show_ylabel=False)
    for t in ax.texts:
        t.xyann = (0, tier)
    for cont in ax.containers:  # thinner error bars at this size
        if hasattr(cont, "lines") and cont.lines and len(cont.lines) > 2:
            for line in cont.lines[2]:
                line.set_linewidth(0.5)
        if hasattr(cont, "lines") and cont.lines and cont.lines[1]:
            for cap in cont.lines[1]:
                cap.set_markersize(1.5)
                cap.set_markeredgewidth(0.5)
    c, _ = _x_positions()
    ax.set_xlim(c[0] - 0.55, c[-1] + 0.55)


def grid_figure(out_dir):
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.4), sharey="row")
    for r, ds in enumerate(["IN", "OI"]):
        for c, metric in enumerate(METRICS):
            _panel(axes[r, c], ds, metric, show_ylabel=(c == 0))
            axes[r, c].set_title(f"{DATASET_NAME[ds]} · {metric}")
        axes[r, 0].set_ylim(*_ylim(ds))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"xm_vs_rhat_grid.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def compact_rc():
    fs = 6.5
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": fs,
                         "axes.labelsize": fs, "axes.titlesize": fs - 0.5,
                         "xtick.labelsize": 4.6, "ytick.labelsize": fs - 1.0,
                         "legend.fontsize": fs - 0.5, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 1.5, "ytick.major.size": 2.0,
                         "xtick.major.pad": 1.5, "ytick.major.pad": 1.5})
    return fs


def decomposition_figure(out_dir, tsne_png, ph=0.47):
    """(a) the joint t-SNE panel (image written by tsne_branches.py), (b) ImageNet-1K KNN over LP."""
    global ENC_GAP
    fs = compact_rc()
    saved_gap, ENC_GAP = ENC_GAP, 0.5
    leg = dict(handlelength=1.3, borderpad=0.3, columnspacing=1.0, handletextpad=0.4, frameon=True)
    W, TW = 3.3, (1.42 if tsne_png else 0.0)
    if tsne_png:
        from PIL import Image
        px = np.asarray(Image.open(tsne_png).convert("RGB"))
        TH = TW * px.shape[0] / px.shape[1]
    else:
        TH = 0.0
    leg_h, gap_h, tick_h = 0.19, 0.06, 0.25
    H = max(TH, leg_h + ph + gap_h + ph + tick_h)
    fig = plt.figure(figsize=(W, H))
    if tsne_png:
        ax_img = fig.add_axes([0, (H - TH) / 2 / H, TW / W, TH / H])
        ax_img.imshow(px, interpolation="none")
        ax_img.set_axis_off()
    x0 = TW + 0.36
    pw = W - x0 - 0.02
    ax_lp = fig.add_axes([x0 / W, tick_h / H, pw / W, ph / H])
    ax_knn = fig.add_axes([x0 / W, (tick_h + ph + gap_h) / H, pw / W, ph / H], sharey=ax_lp)
    _style(ax_knn, "IN", "KNN")
    _style(ax_lp, "IN", "LP")
    for t in list(ax_knn.texts):  # encoder labels only under the bottom panel
        t.remove()
    ax_knn.tick_params(labelbottom=False)
    ax_lp.set_ylim(*_ylim("IN"))
    ax_knn.set_ylabel("KNN", labelpad=2)
    ax_lp.set_ylabel("LP", labelpad=2)
    h, _ = ax_knn.get_legend_handles_labels()
    fig.legend(h, LABELS, loc="upper center", ncol=2, bbox_to_anchor=((x0 + pw / 2) / W, 1.0),
               fontsize=fs - 1.0, **leg)
    if tsne_png:
        for txt, x in (("a)", 0.0), ("b)", TW + 0.03)):
            fig.text(x / W, 1.0, txt, ha="left", va="top", fontsize=6.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"branch_decomposition.{ext}"), dpi=300, pad_inches=0.01,
                    bbox_inches="tight")
    plt.close(fig)
    ENC_GAP = saved_gap


def openimages_figure(out_dir):
    global ENC_GAP
    fs = compact_rc()
    saved_gap, ENC_GAP = ENC_GAP, 0.5
    plt.rcParams.update({"xtick.labelsize": 5.0, "axes.titlesize": fs + 0.5})
    leg = dict(handlelength=1.3, borderpad=0.3, columnspacing=1.0, handletextpad=0.4, frameon=True)
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.45), sharey=True)
    for ax, m in zip(axes, METRICS):
        _style(ax, "OI", m)
        ax.set_title(f"OpenImagesV7 · {m}", pad=2.5)
    axes[0].set_ylabel("Micro-F1")
    axes[0].set_ylim(*_ylim("OI"))
    h, _ = axes[0].get_legend_handles_labels()
    fig.legend(h, LABELS, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), **leg)
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=0.6, pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"xm_vs_rhat_openimages.{ext}"), dpi=300, bbox_inches="tight",
                    pad_inches=0.01)
    plt.close(fig)
    ENC_GAP = saved_gap


def main():
    global DATA
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = os.path.join(os.environ.get("ABL_ROOT", "./outputs/ablations"), "main_branch_metrics")
    ap.add_argument("--csv", default=os.path.join(root, "branch_reps.csv"))
    ap.add_argument("--out-dir", default=os.path.join(root, "figures"))
    ap.add_argument("--tsne-png", default=None,
                    help="panel (a) image, written by tsne_branches.py (omit to draw panel b only)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    DATA = load(args.csv)
    # Print the plotted values (mean, sd) so the figure can be checked against the CSV.
    for ds in ("IN", "OI"):
        for metric in METRICS:
            for enc in ENCODERS:
                for act in ACTS:
                    c = DATA[ds][metric][enc][act]
                    print(f"{ds} {metric} {enc:4s} {act:3s} xM {c['xM'][0]:.4f}±{c['xM'][1]:.4f}  "
                          f"rhat {c['rhat'][0]:.4f}±{c['rhat'][1]:.4f}")
    grid_figure(args.out_dir)
    decomposition_figure(args.out_dir, args.tsne_png)
    openimages_figure(args.out_dir)
    print("wrote figures to", args.out_dir)


if __name__ == "__main__":
    main()
