#!/usr/bin/env python3
"""Tables for the DualSAE expansion-ratio / sparsity-level (scale) ablation.

One-at-a-time design (ablation_grid.SCALE_ARMS):

    E<r>  / SE<r>   dictionary = r x encoder width (r in 4, 8, 16, 32), k = 64
    K<k>  / SK<k>   k in 16, 32, 128, 256, dictionary = 8192

Each Dual arm replays its cell's main-table config (gamma, warm-up, lr) and changes only
--n_latents or --topk_k; each "S" arm is the matched plain SAE on DualSAE's own Sparse Branch
code (grid arm S) at the same setting. The anchor point, 8192 latents / k = 64, is R0 (Dual) and
S (SAE); ViT x8 IS that anchor and is read from R0/S (marked †).

Writes <out>/REPORT.md and <out>/long.csv (the input of build_expansion_data.py):

    python collect_scale_ablation.py --seeds 42

Only seeds are averaged; dataset, encoder and activation stay separate rows; L0 is shown. A value
is shown only when EVERY requested seed has it.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
VISION_DIR = _HERE.parents[1]
for _p in (str(_HERE), str(VISION_DIR), str(VISION_DIR / "scripts" / "ablations" / "grid")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import collect_grid as R  # noqa: E402  (shares its metric read path)
import ablation_grid as C  # noqa: E402

DEFAULT_OUT = Path(os.environ.get("ABL_ROOT", "./outputs/ablations")) / "expansion"
METRICS: Tuple[str, ...] = ("r2_xhat", "ms", "purity", "knn", "lp", "r2_xm_global", "l0",
                            "dead_frac")
DELTA_METRICS: Tuple[str, ...] = ("r2_xhat", "ms", "purity", "knn", "lp")
DIGITS = {"l0": 1, "dead_frac": 3}

# Sweep points as (column label, Dual arm, SAE arm). "pub" is the anchor (R0/S).
DICT_POINTS: List[Tuple[str, str, str]] = (
    [(f"×{r}", f"E{r}", f"SE{r}") for r in C.EXPANSIONS[:2]] + [("pub 8192", "R0", "S")]
    + [(f"×{r}", f"E{r}", f"SE{r}") for r in C.EXPANSIONS[2:]])
K_POINTS: List[Tuple[str, str, str]] = (
    [(f"k={k}", f"K{k}", f"SK{k}") for k in C.K_VALUES if k < 64] + [("k=64 (pub)", "R0", "S")]
    + [(f"k={k}", f"K{k}", f"SK{k}") for k in C.K_VALUES if k > 64])
ARMS = ("R0", "S") + C.SCALE_ARMS


def resolve(encoder: str, arm: str) -> Tuple[str, bool]:
    """(arm whose run holds this point, whether it is an alias). ViT x8 is the 8192/64 anchor."""
    if arm in C.SCALE_ARMS:
        alias = C.scale_alias(encoder, arm)
        if alias:
            return alias, True
    return arm, False


class ScaleStore:
    """values[(cell, arm)][seed] = {metric: value}."""

    def __init__(self, seeds: Sequence[int]) -> None:
        self.seeds = tuple(seeds)
        self.values: Dict[Tuple[R.CellKey, str], Dict[int, Dict[str, float]]] = {}
        self.problems: List[str] = []
        self.flags: List[str] = []
        self.missing_runs: List[str] = []

    def load(self) -> "ScaleStore":
        r0_grid = C.build_grid(["R0"])
        for cell in R.cells():
            dataset, encoder, activation = cell
            pub = {r["seed"]: r for r in r0_grid
                   if (r["dataset"], r["encoder"], r["activation"]) == cell}
            for arm in ARMS:
                src, _ = resolve(encoder, arm)
                if src != arm:
                    continue  # read through its alias at report time
                for seed in self.seeds:
                    run = C.run_dir(dataset, encoder, activation, arm, seed)
                    if not C.run_is_complete(run):
                        self.missing_runs.append(run.name)
                        continue
                    metrics, problems, _ = R.metrics_from_branch_files(
                        run, dataset, "best", C.BRANCH_METRICS_RESULTS)
                    self.problems.extend(problems)
                    if not metrics:
                        continue
                    metrics = dict(metrics)
                    epoch = C.read_best_epoch(run)
                    metrics["best_epoch"] = float(epoch)
                    self.values.setdefault((cell, arm), {})[seed] = metrics
                    warm = 0 if C.is_sae_arm(arm) or seed not in pub else pub[seed]["published_warmup"]
                    early = epoch < R.MIN_BEST_EPOCH_FRAC * C.EPOCHS
                    pre_unfreeze = not C.is_sae_arm(arm) and \
                        epoch * C.STEPS_PER_EPOCH[dataset] <= warm + 1
                    if early or pre_unfreeze:
                        why = "pre-unfreeze" if pre_unfreeze else "early"
                        self.flags.append(f"{'/'.join(cell)}/{arm}/seed{seed}: best epoch "
                                          f"{epoch} ({why})")
        return self

    def per_seed(self, cell: R.CellKey, arm: str) -> Dict[int, Dict[str, float]]:
        return self.values.get((cell, resolve(cell[1], arm)[0]), {})

    def stat(self, cell: R.CellKey, arm: str, metric: str) -> Optional[Tuple[float, float]]:
        ps = self.per_seed(cell, arm)
        return R.seed_stats([ps.get(s, {}).get(metric) for s in self.seeds],
                            expected=len(self.seeds))

    def delta(self, cell: R.CellKey, dual: str, sae: str,
              metric: str) -> Optional[Tuple[float, float]]:
        """Paired per-seed Dual − SAE at one setting."""
        a, b = self.per_seed(cell, dual), self.per_seed(cell, sae)
        diffs = []
        for s in self.seeds:
            x, y = a.get(s, {}).get(metric), b.get(s, {}).get(metric)
            diffs.append(None if x is None or y is None else x - y)
        return R.seed_stats(diffs, expected=len(self.seeds))


def fmt(stat: Optional[Tuple[float, float]], n_seeds: int, digits: int, signed: bool = False) -> str:
    if stat is None:
        return "–"
    sign = "+" if signed else ""
    if n_seeds == 1:
        return f"{stat[0]:{sign}.{digits}f}"
    return f"{stat[0]:{sign}.{digits}f} ± {stat[1]:.{digits}f}"


def setting_row(cell: R.CellKey, points) -> List[str]:
    """Dictionary size (or k) actually trained at each point, so the header's ×r reads concretely."""
    out = []
    for _, dual, _ in points:
        if dual == "R0":
            out.append("8192/64")
        else:
            s = C.scale_setting(cell[1], dual)
            out.append(f"{s['n_latents']}/{s['top_k']}")
    return out


def sweep_table(store: ScaleStore, metric: str, points, title: str) -> List[str]:
    """Rows = cells; each point gets a Dual and an SAE column."""
    digits = DIGITS.get(metric, 4)
    header = ["Dataset", "Encoder", "Act"]
    for label, _, _ in points:
        header += [f"{label} Dual", f"{label} SAE"]
    rows = []
    for cell in R.cells():
        row = list(cell)
        for _, dual, sae in points:
            tag = "†" if resolve(cell[1], dual)[1] else ""
            row += [fmt(store.stat(cell, dual, metric), len(store.seeds), digits) + tag,
                    fmt(store.stat(cell, sae, metric), len(store.seeds), digits) + tag]
        rows.append(row)
    return [f"### {title}: {R.LABEL[metric]}", ""] + R.md_table(header, rows)


def delta_table(store: ScaleStore, metric: str, points, title: str) -> List[str]:
    header = ["Dataset", "Encoder", "Act"] + [label for label, _, _ in points]
    rows = []
    for cell in R.cells():
        rows.append(list(cell) + [fmt(store.delta(cell, d, s, metric), len(store.seeds), 4, True)
                                  for _, d, s in points])
    return [f"### {title}: Δ {R.LABEL[metric]} (Dual − SAE, paired by seed)", ""] + \
        R.md_table(header, rows)


def settings_table() -> List[str]:
    header = ["Encoder", "width d"] + [p[0] for p in DICT_POINTS]
    rows = []
    for enc in C.ENCODERS:
        cell = ("imagenet", enc, "topk")
        rows.append([enc, str(C.ACT_DIM[enc])] + setting_row(cell, DICT_POINTS))
    lines = ["### Dictionary / k trained at each expansion point", ""] + R.md_table(header, rows)
    lines.append(f"8192 latents = ×{8192 / C.ACT_DIM['clip']:.2f} (CLIP), "
                 f"×{8192 / C.ACT_DIM['sigclip']:.2f} (SigLIP2), ×{8192 / C.ACT_DIM['vit']:.0f} (ViT).")
    lines.append("")
    return lines


def flat_rows(store: ScaleStore) -> List[dict]:
    rows = []
    for (cell, arm), per_seed in sorted(store.values.items()):
        if arm in C.SCALE_ARMS:
            s = C.scale_setting(cell[1], arm)
        else:
            s = {"n_latents": 8192, "top_k": 64}
        for seed, metrics in sorted(per_seed.items()):
            for metric, value in metrics.items():
                rows.append({"dataset": cell[0], "encoder": cell[1], "activation": cell[2],
                             "arm": arm, "model": "sae" if C.is_sae_arm(arm) else "dual",
                             "n_latents": s["n_latents"], "top_k": s["top_k"],
                             "expansion": round(s["n_latents"] / C.ACT_DIM[cell[1]], 4),
                             "seed": seed, "metric": metric, "value": value})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid-root", type=Path, default=None,
                    help="ablation grid directory (default: $ABL_ROOT/grid)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="report directory")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.SEEDS))
    args = ap.parse_args()
    unknown = set(args.seeds) - set(C.SEEDS)
    if unknown:
        ap.error(f"unknown seeds {sorted(unknown)}; the grid has {C.SEEDS}")
    if args.grid_root is not None:
        C.set_root(args.grid_root)
    args.root = args.out
    args.root.mkdir(parents=True, exist_ok=True)
    store = ScaleStore(args.seeds).load()

    expected = sum(1 for c in C.build_grid(list(C.SCALE_ARMS)) if c["seed"] in args.seeds)
    done = sum(len(ps) for (cell, arm), ps in store.values.items() if arm in C.SCALE_ARMS)
    lines = ["# DualSAE scale ablation: expansion factor and sparsity level k", "",
             f"Seeds: {', '.join(map(str, args.seeds))}. Scale-arm runs scored: "
             f"{done}/{expected}.", "",
             "One-at-a-time: the dictionary sweep keeps k = 64; the k sweep keeps 8192 latents. "
             "Every Dual run replays its cell's main-table config and changes only the "
             "dictionary size or k; **γ stays at the main-table value, which was chosen at "
             "8192/64**. SAE = plain SAE on DualSAE's own sparse-branch code at the same setting. "
             "Dead-latent resampling is effectively off in the main-table config "
             "(dead_steps_threshold 1e7), so read the Dead rows next to large dictionaries. "
             "† = the 8192/64 anchor (R0/S) standing in for ViT ×8. "
             "MonoScore on x, R²/KNN/LP on x̂, purity on the sparse code (OIv7: val; other OIv7 "
             "metrics: test).", ""]
    lines += ["## Checks", ""]
    lines.append(f"- Selection: {len(store.flags)} run(s) selected a checkpoint before warm-up "
                 f"ended or before {R.MIN_BEST_EPOCH_FRAC:.0%} of training"
                 + (": " + "; ".join(sorted(store.flags)) if store.flags else ""))
    if store.problems:
        lines.append(f"- Unavailable readings: {len(set(store.problems))} "
                     f"(first: {sorted(set(store.problems))[:5]})")
    missing = sorted(set(store.missing_runs))
    if missing:
        lines.append(f"- Unfinished/missing training runs: {len(missing)} (first: {missing[:5]})")
    lines.append("")
    lines += settings_table()

    for title, points in (("Expansion factor (k = 64)", DICT_POINTS),
                          ("Sparsity level k (8192 latents)", K_POINTS)):
        lines += [f"## {title}", ""]
        for metric in METRICS:
            lines += sweep_table(store, metric, points, title)
        for metric in DELTA_METRICS:
            lines += delta_table(store, metric, points, title)

    (args.root / "REPORT.md").write_text("\n".join(lines) + "\n")
    with (args.root / "long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "encoder", "activation", "arm",
                                                    "model", "n_latents", "top_k", "expansion",
                                                    "seed", "metric", "value"])
        writer.writeheader()
        writer.writerows(flat_rows(store))
    print(f"wrote {args.root / 'REPORT.md'} and long.csv ({done}/{expected} scale runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
