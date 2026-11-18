"""R^2 vs. MonoScore plots for joint vs. sequential training (Residual analysis figure).

Three measured models per setting, all from the same evaluation (long.csv of collect_grid.py):
  SAE                    arm S, best checkpoint: the standard SAE (DualSAE's Sparse Branch code, no
                         Residual Branch) that sequential training freezes
  SAE + Residual         arm F at DualSAE's Residual Branch update count (role at_D_best,
                         "Sequential Training"); its Sparse Branch is S, so its MonoScore equals S's
  DualSAE                --dual published_dual (default: the main-table DualSAE runs, both branches
                         trained jointly) or --dual R0 (the main-table config retrained in the grid)
Dashed dimension arrows give relative changes: R^2 from SAE to SAE + Residual (under the markers),
R^2 from SAE + Residual to DualSAE (only if they differ by >= 2%), and MonoScore from SAE +
Residual to DualSAE (right of the markers).

Usage: python plot_joint_vs_seq.py <outdir> <long.csv> [--dual published_dual|R0]
  -> <outdir>/joint_vs_seq_tradeoff.pdf      BatchTopK, ImageNet-1K, three encoders
  -> <outdir>/joint_vs_seq_tradeoff_all.pdf  all 12 settings
"""
import argparse
import collections
import csv
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_ap = argparse.ArgumentParser(description="joint vs. sequential training plots")
_ap.add_argument("outdir")
_ap.add_argument("long_csv")
_ap.add_argument("--dual", choices=["published_dual", "R0"], default="published_dual")
_args = _ap.parse_args()
outdir, LONG_CSV = _args.outdir, _args.long_csv

DATA, ENCS, SAES = ["imagenet", "open_images"], ["clip", "sigclip", "vit"], ["topk", "batch_topk"]
KEYS = [(d, e, s) for d in DATA for e in ENCS for s in SAES]       # row order of tab:frozen_vs_dualsae
NAME = {"imagenet": "ImageNet-1K", "open_images": "OpenImagesV7", "clip": "CLIP", "sigclip": "SigLIP2",
        "vit": "ViT", "topk": "TopK", "batch_topk": "BatchTopK"}
SRC = {"SAE": ("S", "best"), "Seq": ("F", "at_D_best"), "Dual": (_args.dual, "best")}

raw = collections.defaultdict(list)
for r in csv.DictReader(open(LONG_CSV)):
    raw[(r["dataset"], r["encoder"], r["activation"], r["arm"], r["role"], r["metric"])].append(float(r["value"]))
P = {}
for k in KEYS:
    P[k] = {}
    for m, (arm, role) in SRC.items():
        r2, ms = raw[k + (arm, role, "r2_xhat")], raw[k + (arm, role, "ms")]
        assert len(r2) == 3 and len(ms) == 3, (k, m, len(r2), len(ms))
        P[k][m] = (statistics.mean(r2), statistics.mean(ms))
    # the frozen arm keeps S's sparse codes, so its MonoScore must be S's (also asserted by the collector)
    assert abs(P[k]["Seq"][1] - P[k]["SAE"][1]) < 1e-9, ("frozen arm MonoScore != S", k)

for k in KEYS:
    s, f, d = P[k]["SAE"], P[k]["Seq"], P[k]["Dual"]
    print(f"{'/'.join(k):30s} SAE ({s[0]:.4f},{s[1]:.4f})  Frozen+Res ({f[0]:.4f},{f[1]:.4f})  "
          f"DualSAE ({d[0]:.4f},{d[1]:.4f})  dR2 {100 * (d[0] / f[0] - 1):+.1f}%  dMS {100 * (d[1] / f[1] - 1):+.1f}%")

STYLE = {"SAE": dict(marker="o", color="#5b9bd5", ms=6.0, label="SAE"),
         "Seq": dict(marker="s", color="#2e5597", ms=5.5, label="SAE + Residual"),
         "Dual": dict(marker="*", color="#e8822a", ms=10.0, label="DualSAE")}
FS = 6.5
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": FS, "axes.labelsize": FS,
                     "axes.titlesize": FS + 0.5, "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5,
                     "axes.linewidth": 0.6, "axes.edgecolor": "0.35", "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
                     "xtick.major.pad": 1.5, "ytick.major.pad": 1.5})

from matplotlib.transforms import ScaledTranslation

C_RES, C_DUAL = "#2e5597", "#b35806"   # dimension arrows: Residual Branch on the frozen SAE / DualSAE
R2_SIMILAR = 0.02                      # |relative R^2 change| below this: DualSAE ~ SAE + Residual


def pct(new, old):
    return f"{100 * (new / old - 1):+.1f}%".replace("-", "−")


def shifted(ax, dx_pt, dy_pt):
    """Data coordinates shifted by a fixed offset in points (keeps arrows clear of the markers)."""
    return ax.transData + ScaledTranslation(dx_pt / 72, dy_pt / 72, ax.figure.dpi_scale_trans)


def dim_arrow(ax, p0, p1, trans, color):
    ax.annotate("", xy=p1, xycoords=trans, xytext=p0, textcoords=trans, zorder=2,
                arrowprops=dict(arrowstyle="<|-|>", color=color, lw=0.8, ls=(0, (2.4, 1.4)),
                                mutation_scale=4.5, shrinkA=0, shrinkB=0))


def panel(ax, k, xlim, ylim, fs_note=FS - 0.5, below=7.0, right=6.0):
    pts = P[k]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    s, f, d = pts["SAE"], pts["Seq"], pts["Dual"]
    note = dict(fontsize=fs_note, textcoords="offset points", zorder=2.5)
    under = shifted(ax, 0, -below)
    # R^2: SAE -> SAE + Residual, a dimension arrow under the markers
    dim_arrow(ax, (s[0], f[1]), (f[0], f[1]), under, C_RES)
    second = abs(d[0] / f[0] - 1) >= R2_SIMILAR
    if second:   # two R^2 arrows: labels meet at the square instead of being centred (they would overlap)
        ax.annotate(pct(f[0], s[0]), xy=(f[0], f[1]), xycoords=under, xytext=(-1.0, -1.5),
                    ha="right", va="top", color=C_RES, **note)
        # R^2: SAE + Residual -> DualSAE, only when the two differ noticeably
        dim_arrow(ax, (f[0], f[1]), (d[0], f[1]), under, C_DUAL)
        ax.annotate(pct(d[0], f[0]), xy=(f[0], f[1]), xycoords=under, xytext=(1.0, -1.5),
                    ha="left", va="top", color=C_DUAL, **note)
    else:
        ax.annotate(pct(f[0], s[0]), xy=((s[0] + f[0]) / 2, f[1]), xycoords=under, xytext=(0, -1.5),
                    ha="center", va="top", color=C_RES, **note)
    # MonoScore: SAE + Residual -> DualSAE, a dimension arrow right of the square and star
    side = shifted(ax, right, 0)
    x = max(f[0], d[0])
    px2pt = 72 / ax.figure.dpi
    y0, y1 = ax.transData.transform([(x, f[1]), (x, d[1])])[:, 1] * px2pt      # square / star height, pt
    if abs(y1 - y0) > 6:                       # skip the arrow if it would be shorter than its heads
        dim_arrow(ax, (x, f[1]), (x, d[1]), side, C_DUAL)
    # its label goes on the arrow's left: between the square and the star if they are far enough apart
    # vertically, otherwise just left of both markers so it never covers them
    r_sq, r_star = STYLE["Seq"]["ms"] / 2 + 0.5, STYLE["Dual"]["ms"] / 2 + 0.5
    lo = min(y0 + r_sq, y1 + r_star) if y1 < y0 else y0 + r_sq          # top of the lower marker
    hi = y1 - r_star if y1 >= y0 else y0 - r_sq                           # bottom of the upper marker
    if hi - lo >= fs_note + 1.0:
        y_mid = ax.transData.inverted().transform((0, (lo + hi) / 2 / px2pt))[1]
        ax.annotate(pct(d[1], f[1]), xy=(x, y_mid), xycoords=side, xytext=(-2.0, 0),
                    ha="right", va="center", color=C_DUAL, **note)
    else:
        ax.annotate(pct(d[1], f[1]), xy=(min(f[0], d[0]), (f[1] + d[1]) / 2),
                    xytext=(-(STYLE["Dual"]["ms"] / 2 + 2.0), 0), ha="right", va="center", color=C_DUAL, **note)
    for m, st in STYLE.items():
        ax.plot(*pts[m], linestyle="none", marker=st["marker"], ms=st["ms"], mfc=st["color"],
                mec="0.1", mew=0.5, zorder=3)
    ax.grid(alpha=0.35, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_bounds(xlim[0], 1.0)   # R^2 <= 1; the space beyond holds the MonoScore arrow


def legend(fig, y, fs):
    h = [Line2D([], [], linestyle="none", marker=s["marker"], ms=s["ms"] * 0.8, mfc=s["color"], mec="0.1",
                mew=0.5, label=s["label"]) for s in STYLE.values()]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, y), ncol=3, frameon=False, fontsize=fs,
               handletextpad=0.3, columnspacing=1.4)


# ---------------- main paper: BatchTopK, ImageNet-1K ----------------
fig, axs = plt.subplots(1, 3, figsize=(3.3, 1.15), sharey=True)
fig.subplots_adjust(wspace=0.12)
for ax, e in zip(axs, ENCS):
    panel(ax, ("imagenet", e, "batch_topk"), (0.76, 1.035), (0.20, 0.70), right=5.0)
    ax.set_title(NAME[e], pad=2)
    ax.set_xticks([0.8, 0.9, 1.0])
axs[0].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
axs[0].set_ylabel("MonoScore", labelpad=2)
axs[1].set_xlabel("$R^2$", labelpad=1)
legend(fig, 0.93, FS - 0.5)
fig.savefig(f"{outdir}/joint_vs_seq_tradeoff.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(f"{outdir}/joint_vs_seq_tradeoff.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
plt.close(fig)

# ---------------- supplement: all 12 settings ----------------
ROWS = [(d, s) for d in DATA for s in SAES]
fig, axs = plt.subplots(4, 3, figsize=(6.85, 5.2), sharey="row")
fig.subplots_adjust(wspace=0.10, hspace=0.32)
LIM = {"imagenet": ((0.75, 1.025), (0.20, 0.70)), "open_images": ((0.65, 1.025), (0.05, 0.60))}
for i, (d, s) in enumerate(ROWS):
    for j, e in enumerate(ENCS):
        ax = axs[i, j]
        panel(ax, (d, e, s), *LIM[d], fs_note=FS)
        ax.set_xticks([0.8, 0.9, 1.0] if d == "imagenet" else [0.7, 0.8, 0.9, 1.0])
        if i == 0:
            ax.set_title(NAME[e], pad=3)
        if i == len(ROWS) - 1:
            ax.set_xlabel("$R^2$", labelpad=1)
    axs[i, 0].set_ylabel("MonoScore", labelpad=2)
for i, (d, s) in enumerate(ROWS):                 # row label, left of the y-axis label
    pos = axs[i, 0].get_position()
    fig.text(pos.x0 - 0.072, (pos.y0 + pos.y1) / 2, f"{NAME[d]}\n{NAME[s]}", rotation=90,
             ha="center", va="center", fontsize=FS + 0.5, fontweight="bold", linespacing=1.1)
legend(fig, 0.905, FS + 0.5)
fig.savefig(f"{outdir}/joint_vs_seq_tradeoff_all.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(f"{outdir}/joint_vs_seq_tradeoff_all.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
plt.close(fig)
