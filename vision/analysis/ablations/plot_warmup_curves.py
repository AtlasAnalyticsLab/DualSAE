#!/usr/bin/env python3
"""Effect of warm-up length: validation R^2 (left) and validation MonoScore (right) per epoch,
one curve per warm-up length (method-section figure).

Reads the per-epoch validation scalars that train_dual.py logs (`val_r2`, R^2 of x_hat; and
`val_mono_mean`, the mean per-latent MonoScore of the sparse code), either from the TensorBoard
event files of the runs (default; written by scripts/ablations/warmup/run_warmup.sh) or, with
--logs, from training stdout logs (lines "Epoch e/E - Val   R2: ..." and
"Epoch e/E - Val Mono (sparse): mean=..."). Curves are the mean over the selected seeds; pass a
single --seeds value to draw one run per warm-up length.

    python plot_warmup_curves.py --tb-root ./outputs/ablations/warmup/tensorboard \
        --dataset open_images --encoder clip --model dual_batch_topk --warmups 0 2500 5000 7500
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

R2_LINE = re.compile(r"^Epoch (\d+)/\d+ - Val   R2: ([\-0-9.eE+]+)")
MS_LINE = re.compile(r"^Epoch (\d+)/\d+ - Val Mono \(sparse\): mean=([\-0-9.eE+]+)")
EXP_RE = re.compile(r"_w(\d+)_seed(\d+)$")


def from_tensorboard(tb_root: Path, prefix: str):
    """{(warmup, seed): {"r2": {epoch: v}, "ms": {epoch: v}}} with 1-based epochs."""
    from collect_grid import read_tb_scalars  # tensorboardX event reader
    out = {}
    for run in sorted(tb_root.glob(prefix + "_w*_seed*")):
        m = EXP_RE.search(run.name)
        if not m:
            continue
        tags = read_tb_scalars(run, ("val_r2", "val_mono_mean"))
        # train_dual logs epoch e (1-based) at TensorBoard step e - 1.
        out[(int(m.group(1)), int(m.group(2)))] = {
            "r2": {s + 1: v for s, v in tags["val_r2"].items()},
            "ms": {s + 1: v for s, v in tags["val_mono_mean"].items()}}
    return out


def from_logs(pattern: str, name_re: str):
    """Parse stdout logs; warm-up and seed are read from the file name with `name_re`."""
    out = {}
    rx = re.compile(name_re)
    for path in sorted(glob.glob(pattern)):
        m = rx.search(os.path.basename(path))
        if not m:
            continue
        r2, ms = {}, {}
        for line in open(path, errors="ignore"):
            a, b = R2_LINE.match(line), MS_LINE.match(line)
            if a:
                r2[int(a.group(1))] = float(a.group(2))
            if b:
                ms[int(b.group(1))] = float(b.group(2))
        out[(int(m.group("warmup")), int(m.group("seed")))] = {"r2": r2, "ms": ms}
    return out


def seed_mean(runs, warmup, seeds, key):
    series = [runs[(warmup, s)][key] for s in seeds if (warmup, s) in runs]
    if len(series) != len(seeds):
        return None  # a curve over fewer seeds than requested would be a different quantity
    epochs = sorted(set.intersection(*(set(s) for s in series)))
    return epochs, [statistics.fmean(s[e] for s in series) for e in epochs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tb-root", type=Path, default=None)
    ap.add_argument("--logs", default=None, help="glob of stdout logs (instead of --tb-root)")
    ap.add_argument("--log-name-re", default=r"warmup_(?P<warmup>\d+)_seed_(?P<seed>\d+)",
                    help="regex with named groups warmup and seed, matched on log file names")
    ap.add_argument("--dataset", default="open_images")
    ap.add_argument("--encoder", default="clip")
    ap.add_argument("--model", default="dual_batch_topk")
    ap.add_argument("--warmups", type=int, nargs="+", default=[0, 2500, 5000, 7500])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--seed-per-warmup", nargs="+", default=None, metavar="W:S",
                    help="draw a single run per warm-up length, e.g. 0:43 2500:42 (overrides --seeds)")
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("ABL_ROOT", "./outputs/ablations"))
                    / "warmup" / "warmup_r2_monoscore.png")
    args = ap.parse_args()

    if args.logs:
        runs = from_logs(args.logs, args.log_name_re)
    else:
        tb_root = args.tb_root or Path(os.environ.get("ABL_ROOT", "./outputs/ablations")) / "warmup" / "tensorboard"
        runs = from_tensorboard(tb_root, f"warmup_{args.dataset}_{args.encoder}_{args.model}")

    try:  # the figure uses seaborn's whitegrid style and pastel palette when available
        import seaborn as sns
        sns.set_theme(style="whitegrid", palette="pastel", font_scale=1.2)
        colors = sns.color_palette("pastel", len(args.warmups))
    except ImportError:
        colors = ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff", "#debb9b"][:len(args.warmups)]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.85))
    pick = {int(a): [int(b)] for a, b in (x.split(":") for x in args.seed_per_warmup or [])}
    for color, w in zip(colors, args.warmups):
        seeds = pick.get(w, args.seeds)
        for ax, key in zip(axes, ("r2", "ms")):
            got = seed_mean(runs, w, seeds, key)
            if got is None:
                print(f"warning: warm-up {w}: not every seed in {seeds} has data; curve omitted")
                continue
            ax.plot(*got, color=color, lw=2.0, label=str(w))
    axes[0].set_ylabel(r"Validation $R^2$")
    axes[1].set_ylabel("Validation MonoScore")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(title="Warm-up steps", loc="lower right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"wrote {args.out} ({len(runs)} runs found)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
