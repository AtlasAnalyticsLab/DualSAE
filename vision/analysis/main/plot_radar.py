#!/usr/bin/env python3
"""Radar figures of the main comparison: every SAE variant against its DualSAE twin.

One figure per dataset (``--dataset imagenet`` / ``open_images``), a grid of polar plots with
one row per encoder (ViT, CLIP, SigLIP2) and one column per pair (TopK, BatchTopK, JumpReLU,
MatryoshkaSAE, GatedSAE, CB-SAE). Each axis is one metric (R^2, MS, Pur., KNN, LP), min-max
scaled over every series drawn in the figure (the ranges are printed once, in the footer).

Data: the seed means of the long CSV written by ``collect_main.py`` -- nothing is hard-coded.

Usage:
    python analysis/main/plot_radar.py --long ./outputs/main/main_long.csv --dataset imagenet
    python analysis/main/plot_radar.py --long ./outputs/main/main_long.csv --dataset open_images
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_analysis_common as A  # noqa: E402

METRICS = list(A.METRICS)
METRIC_LABEL = {"R2": "R$^2$", "MS": "MS", "PUR": "Pur.", "KNN": "KNN", "LP": "LP"}

# (baseline key, dual key, column title), in the table's order.
PAIRS = [
    ("topk", "dual_topk", "TopK"),
    ("batch_topk", "dual_batch_topk", "BatchTopK"),
    ("jumprelu", "dual_jump_relu", "JumpReLU"),
    ("matryoshka", "dual_matryoshka", "MatryoshkaSAE"),
    ("gated", "dual_gated", "GatedSAE"),
    ("cb_sae", "cb_dual_sae", "CB-SAE"),
]
ENCODERS = [("vit", "ViT"), ("clip", "CLIP"), ("sigclip", "SigLIP2")]
DATASET_LABEL = {"imagenet": "ImageNet-1K", "open_images": "OpenImagesV7"}

# Colour pair checked for colour-vision-deficiency separation.
C_BASE, C_DUAL = "#1f77b4", "#ff7f0e"

# One knob for the whole type scale (sizes are set for a figure placed at text width).
FONT_SCALE = 1.0


def load(dataset: str, long_csv: Path, checkpoint: str = "best") -> dict:
    """{(encoder, method): {metric: seed mean}} for one dataset, from the collector's CSV."""
    out: dict = {}
    for (ds, enc, meth, metric), row in A.load_long(long_csv, checkpoint=checkpoint).items():
        if ds == dataset and row["mean"] is not None:
            out.setdefault((enc, meth), {})[metric] = row["mean"]
    # A series is drawn only when all five metrics exist.
    return {k: v for k, v in out.items() if all(m in v for m in METRICS)}


def build_ranges(vals: dict) -> dict:
    """Per-metric min/max over EVERY series drawn, so all panels share a scale.

    Computed across the pairs actually plotted; a radar axis is only
    meaningful against a stated range, which is why the footer prints them.
    """
    ranges = {}
    keys = [k for pair in PAIRS for k in pair[:2]]
    for m in METRICS:
        pool = [v[m] for (enc, meth), v in vals.items() if meth in keys and m in v]
        ranges[m] = (min(pool), max(pool))
    return ranges


def scale(values, ranges) -> np.ndarray:
    out = []
    for v, m in zip(values, METRICS):
        lo, hi = ranges[m]
        out.append(0.5 if hi == lo else (v - lo) / (hi - lo))
    return np.asarray(out, float)


def close(a):
    return np.concatenate([a, a[:1]])


def draw(ax, base, dual, ranges, fs):
    n = len(METRICS)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ac = np.concatenate([ang, ang[:1]])

    for vals, color, lab, lw in ((base, C_BASE, "SAE", 1.8),
                                 (dual, C_DUAL, "DualSAE", 2.4)):
        if vals is None:
            continue
        s = close(scale(vals, ranges))
        ax.plot(ac, s, lw=lw, color=color, label=lab, solid_joinstyle="round")
        ax.fill(ac, s, color=color, alpha=0.13)

    ax.set_xticks(ang)
    # Metric name only. The ranges are identical across panels and live in the
    # footer; repeating them here is what made the original mostly whitespace.
    ax.set_xticklabels([METRIC_LABEL[m] for m in METRICS], fontsize=11 * fs)
    ax.tick_params(axis="x", pad=1)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.grid(alpha=0.45, lw=0.6)
    ax.spines["polar"].set_linewidth(0.8)
    ax.spines["polar"].set_alpha(0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="imagenet",
                    choices=["imagenet", "open_images"])
    ap.add_argument("--long", type=Path, default=Path("./outputs/main/main_long.csv"),
                    help="long CSV written by collect_main.py")
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=float, default=10.5)
    ap.add_argument("--font-scale", type=float, default=FONT_SCALE)
    # wspace is tiny on purpose: with 6 columns the horizontal gutters were
    # the bulk of the blank space. hspace cannot go as low -- each radar
    # carries a label at the top (MS) and the bottom (LP), so consecutive rows
    # need a band between them or those two labels collide. That band is label
    # space, not waste.
    # 0.16 is nearly free: the circle is inscribed in min(cell_w, cell_h) and
    # is height-limited here, so cell_w carries ~0.16in of slack that is dead
    # space at wspace=0.05. Widening the gutter to ~0.20 spends that slack
    # without shrinking a single radar; past 0.20 the circles start to shrink.
    ap.add_argument("--wspace", type=float, default=0.16)
    # 0.38 was sized for full-radius circles, where row i's LP label sat close
    # to row i+1's MS label. Shrinking the radius pulls both labels inward by
    # (1 - radius_scale) * cell_h, which buys back most of that band, so the
    # gutter itself can be much smaller for the same clearance.
    ap.add_argument("--hspace", type=float, default=0.18)
    ap.add_argument("--radius-scale", type=float, default=0.85,
                    help="shrink each radar about its own centre, keeping the "
                         "grid, the gutters and the type sizes fixed. 1.0 = "
                         "circle inscribed in the cell.")
    ap.add_argument("--aspect", type=float, default=0.62,
                    help="figure height as a fraction of --width; set so a "
                         "cell is about square, since the circle is inscribed "
                         "in min(cell_w, cell_h) and any excess in the other "
                         "dimension is dead space")
    args = ap.parse_args()
    fs = args.font_scale

    vals = load(args.dataset, args.long, args.checkpoint)
    if not vals:
        raise SystemExit(f"no complete series for {args.dataset} in {args.long}")
    ranges = build_ranges(vals)

    ncol, nrow = len(PAIRS), len(ENCODERS)
    # Panel aspect ~1:1 plus room for the header/footer strips.
    fig, axes = plt.subplots(nrow, ncol, subplot_kw=dict(polar=True),
                             figsize=(args.width, args.width * args.aspect))

    # Lay the grid out first. Everything below positions itself against the
    # final axes rectangles, so this cannot run after them.
    fig.subplots_adjust(left=0.048, right=0.955, top=0.860, bottom=0.085,
                        wspace=args.wspace, hspace=args.hspace)

    missing = []
    for i, (enc, enc_lab) in enumerate(ENCODERS):
        for j, (bkey, dkey, title) in enumerate(PAIRS):
            ax = axes[i, j]
            b = vals.get((enc, bkey))
            d = vals.get((enc, dkey))
            if b is None or d is None:
                missing.append(f"{enc}/{title}")
                ax.set_axis_off()
                continue
            draw(ax, [b[m] for m in METRICS], [d[m] for m in METRICS], ranges, fs)
            if i == 0:
                ax.set_title(title, fontsize=15 * fs, pad=30 * fs)
        axes[i, 0].set_ylabel("")

    # Shrink each radar about its own centre. Scaling the AXES RECTANGLE (not
    # the r-limit) is what actually changes the drawn radius: a polar axes
    # draws its outer ring at the edge of its box, so set_ylim only rescales
    # the data inside a circle of unchanged size.
    #
    # Cell centres, gutters, margins and every font size are untouched, so the
    # grid and the type scale survive verbatim and only the ink shrinks.
    if args.radius_scale != 1.0:
        k = args.radius_scale
        for ax in axes.flat:
            bb = ax.get_position()
            cx, cy = bb.x0 + bb.width / 2, bb.y0 + bb.height / 2
            w, h = bb.width * k, bb.height * k
            ax.set_position([cx - w / 2, cy - h / 2, w, h])

    # Row labels are anchored to the first column's actual left edge, not to a
    # fixed figure coordinate. With a fixed x they drifted away from the plots
    # whenever the axes shrank (--radius-scale) or the margins moved, leaving a
    # gap that tracked nothing. The offset clears the "Pur."/"KNN" tick labels,
    # which sit OUTSIDE the axes box on the left -- anchoring flush to x0 put
    # the encoder name on top of them.
    for i, (_, enc_lab) in enumerate(ENCODERS):
        pos = axes[i, 0].get_position()
        fig.text(pos.x0 - 0.030, pos.y0 + pos.height / 2, enc_lab, rotation=90,
                 va="center", ha="center", fontsize=15 * fs)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Anchored ABOVE the title band, not level with it: at bbox y=1.0 the
    # legend landed on top of the middle column titles.
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.995), fontsize=14 * fs,
               columnspacing=2.2, handlelength=2.4)

    foot = "   ".join(
        f"{METRIC_LABEL[m]} [{ranges[m][0]:.2f}, {ranges[m][1]:.2f}]" for m in METRICS)
    fig.text(0.5, 0.012,
             "each axis min–max scaled over all series shown:   " + foot,
             ha="center", va="bottom", fontsize=9.5 * fs, color="0.30")

    # `top` leaves a clear band for the legend + column titles; `right` leaves
    # room for the last column's R^2 tick label, which was being clipped at
    # 0.995.
    # Margins are the minimum that clears the overhanging tick labels: the
    # last column's R^2 sits outside the axes on the right, the row labels on
    # the left, and the footer spans the full width.
    out = args.out or f"./outputs/main/radar_{args.dataset}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    fig.savefig(str(Path(out).with_suffix(".pdf")))
    print(f"wrote {out} and {Path(out).with_suffix('.pdf')}")
    print(f"data: {args.long}")
    print(f"panels: {nrow}x{ncol} = {nrow*ncol}" +
          (f"   MISSING: {missing}" if missing else "   (all present)"))
    for m in METRICS:
        print(f"  {m:4} range [{ranges[m][0]:.4f}, {ranges[m][1]:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
