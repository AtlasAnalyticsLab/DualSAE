#!/usr/bin/env python3
"""Tables for the DualSAE decorrelation-coefficient ablation (gamma = span_coef).

The objective is L = L_rec + lambda * L_sparse + gamma * L_dcor. For the TopK/BatchTopK cells this
ablation covers, L_sparse is 0 by construction, so gamma is the only live loss coefficient. Each
arm in ablation_grid.SPAN_ARMS replays the main-table config of its cell and seed and changes only
--span_coef. Two arms of the same grid anchor every row:

    S    the plain SAE on DualSAE's own Sparse Branch code (no Residual Branch at all)
    R0   the main-table config retrained, i.e. the cell's main-table gamma (which differs per cell)

Writes <out>/REPORT[_final].md and <out>/long[_final].csv. The supplementary gamma tables are built
from the epoch-50 checkpoint (`--checkpoint final`), see make_gamma_tables.py:

    python collect_span_ablation.py --checkpoint final
    python collect_span_ablation.py                     # best-validation-MonoScore checkpoint

Only seeds are averaged; dataset, encoder and activation stay separate rows; L0 is shown. A value is
shown only when EVERY requested seed has it, so a mean never silently covers fewer runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

DEFAULT_OUT = Path(os.environ.get("ABL_ROOT", "./outputs/ablations")) / "gamma"

# Report order: the five main-table metrics, then decomposition/sparsity diagnostics.
METRICS: Tuple[str, ...] = ("r2_xhat", "ms", "purity", "knn", "lp", "r2_xm_global", "l0",
                            "dead_frac")
# TB scalars that show the MECHANISM: val_span is L_dcor itself (the penalised overlap of r_hat
# with the active mono decoder columns), val_r_rhat_cosine is how aligned the residual is with
# the input residual. Read at the SELECTED epoch so they describe the reported model.
TB_TAGS: Tuple[str, ...] = ("val_span", "val_r_rhat_cosine")
TB_LABEL = {"val_span": "L_dcor (val)", "val_r_rhat_cosine": "cos(r, r̂) (val)"}
# Higher is better for these; used only to mark the per-row best column with bold.
HIGHER_IS_BETTER = {"r2_xhat", "ms", "purity", "knn", "lp", "r2_xm_global"}


def published_gamma(cell: R.CellKey, seeds: Sequence[int]) -> List[float]:
    """The main-table gamma of each requested seed. Can differ by seed (e.g. 0.005/0.006/0.007)."""
    dataset, encoder, activation = cell
    out = []
    for row in C.load_published_rows():
        if (row["dataset"], row["encoder"], row["method"]) == (dataset, encoder,
                                                               C.DUAL_METHOD[activation]) \
                and row["seed"] in seeds:
            out.append(float(C.published_config(row)["span_coef"]))
    return out


class SpanStore:
    """values[(cell, arm)][seed] = {metric: value}; plus best_epoch and TB readouts per run."""

    def __init__(self, seeds: Sequence[int], checkpoint: str = "best") -> None:
        # "best" = best val MonoScore over epochs (the selection rule of every other table);
        # "final" = epoch 50. Both are
        # needed: at large gamma MonoScore drops once the residual unfreezes, so "best" can pick
        # the last WARM-UP epoch, where gamma has not acted yet (gamma is forced to 0 in warm-up)
        # and different gammas yield bit-identical models.
        self.seeds = tuple(seeds)
        self.checkpoint = checkpoint
        self.values: Dict[Tuple[R.CellKey, str], Dict[int, Dict[str, float]]] = {}
        self.problems: List[str] = []
        self.flags: List[str] = []
        self.missing_runs: List[str] = []

    def load(self) -> "SpanStore":
        for cell in R.cells():
            dataset, encoder, activation = cell
            pub_rows = {r["seed"]: r for r in C.load_published_rows()
                        if (r["dataset"], r["encoder"], r["method"]) ==
                        (dataset, encoder, C.DUAL_METHOD[activation])}
            for arm in ("S", "R0") + tuple(C.SPAN_ARMS):
                for seed in self.seeds:
                    # A gamma arm equal to this seed's main-table gamma was never trained: R0 is
                    # that exact run (ablation_grid.span_alias).
                    pub = pub_rows.get(seed)
                    pub_span = C.published_config(pub)["span_coef"] if pub else None
                    src = (C.span_alias(arm, pub_span) if pub_span is not None else None) or arm
                    run = C.run_dir(dataset, encoder, activation, src, seed)
                    if not C.run_is_complete(run):
                        self.missing_runs.append(run.name)
                        continue
                    epoch = C.read_best_epoch(run)
                    # A run that selected epoch 50 has identical best/final weights, and the eval
                    # stage then scores only "best" (eval_rows skips final).
                    ckpt = "best" if (self.checkpoint == "best" or epoch == C.EPOCHS) else "final"
                    metrics, problems, meta = R.metrics_from_branch_files(
                        run, dataset, ckpt, C.BRANCH_METRICS_RESULTS)
                    self.problems.extend(problems)
                    if not metrics:
                        continue
                    read_epoch = epoch if self.checkpoint == "best" else C.EPOCHS
                    metrics = dict(metrics)
                    metrics["best_epoch"] = float(read_epoch)
                    if arm != "S":
                        # Only Dual arms have a residual and therefore an L_dcor to read.
                        tb = R.read_tb_scalars(C.TB_ROOT / run.name, TB_TAGS)
                        for tag in TB_TAGS:
                            # TB steps are 0-based epochs; best_epoch is 1-based.
                            if (read_epoch - 1) in tb[tag]:
                                metrics[tag] = tb[tag][read_epoch - 1]
                    self.values.setdefault((cell, arm), {})[seed] = metrics
                    if self.checkpoint == "best":
                        # The selection rule only applies to the selected checkpoint.
                        self._flag(cell, arm, seed, epoch, pub_rows.get(seed))
        return self

    def _flag(self, cell: R.CellKey, arm: str, seed: int, epoch: int, pub: Optional[dict]) -> None:
        """A checkpoint from before 20% of training, or (for Dual arms) from before the Residual
        Branch unfroze, does not represent its setting; such selections are listed."""
        warm = 0 if arm == "S" or pub is None else int(C.published_config(pub)["warmup_steps"])
        early = epoch < R.MIN_BEST_EPOCH_FRAC * C.EPOCHS
        pre_unfreeze = arm != "S" and epoch * C.STEPS_PER_EPOCH[cell[0]] <= warm + 1
        if early or pre_unfreeze:
            why = "pre-unfreeze" if pre_unfreeze else "early"
            self.flags.append(f"{'/'.join(cell)}/{arm}/seed{seed}: best epoch {epoch} ({why})")

    def stat(self, cell: R.CellKey, arm: str, metric: str) -> Optional[Tuple[float, float]]:
        per_seed = self.values.get((cell, arm), {})
        return R.seed_stats([per_seed.get(s, {}).get(metric) for s in self.seeds],
                            expected=len(self.seeds))

    def flat_rows(self) -> List[dict]:
        rows = []
        for (cell, arm), per_seed in sorted(self.values.items()):
            gamma = (C.SPAN_ARMS.get(arm) if arm in C.SPAN_ARMS else None)
            for seed, metrics in sorted(per_seed.items()):
                for metric, value in metrics.items():
                    rows.append({"dataset": cell[0], "encoder": cell[1], "activation": cell[2],
                                 "arm": arm, "gamma": "" if gamma is None else gamma,
                                 "seed": seed, "metric": metric, "value": value})
        return rows


def fmt(stat: Optional[Tuple[float, float]], n_seeds: int, digits: int) -> str:
    """Mean ± sd with several seeds; the bare value with one (an sd of one run is meaningless)."""
    if stat is None:
        return "–"
    if n_seeds == 1:
        return f"{stat[0]:.{digits}f}"
    return f"{stat[0]:.{digits}f} ± {stat[1]:.{digits}f}"


def span_arms_by_gamma() -> List[str]:
    return sorted(C.SPAN_ARMS, key=C.SPAN_ARMS.get)


def metric_table(store: SpanStore, metric: str, label: str, digits: int = 4) -> List[str]:
    """One table per metric: rows = cells; fixed columns SAE | γ grid | main-table γ.

    The main-table γ differs per cell (0.004..5.0), so its value gets its own column instead of
    a position in the grid; read it against the grid columns on either side.
    """
    grid = span_arms_by_gamma()
    header = (["Dataset", "Encoder", "Act", "SAE"] + [f"γ={C.SPAN_ARMS[a]:g}" for a in grid]
              + ["pub γ", "Dual @ pub γ"])
    rows = []
    for cell in R.cells():
        dual_arms = grid + ["R0"]
        stats = {a: store.stat(cell, a, metric) for a in ["S"] + dual_arms}
        # Bold the best DUAL setting; the SAE column is a reference, not a candidate gamma.
        best = None
        if metric in HIGHER_IS_BETTER:
            cands = [(stats[a][0], a) for a in dual_arms if stats[a] is not None]
            if cands:
                best = max(cands)[1]

        def show(arm: str) -> str:
            txt = fmt(stats[arm], len(store.seeds), digits)
            return f"**{txt}**" if arm == best else txt

        pub = sorted(set(published_gamma(cell, store.seeds)))
        rows.append([cell[0], cell[1], cell[2], show("S")] + [show(a) for a in grid]
                    + ["/".join(f"{g:g}" for g in pub), show("R0")])
    return [f"### {label}", ""] + R.md_table(header, rows)


def delta_table(store: SpanStore, metric: str, label: str) -> List[str]:
    """Paired per-seed difference (gamma arm − R0), so encoder/seed effects cancel.

    This is the cleanest read of "what does changing gamma alone do", since R0 is the same
    replayed config at the main-table gamma.
    """
    header = (["Dataset", "Encoder", "Act", "pub γ"]
              + [f"γ={C.SPAN_ARMS[a]:g}" for a in span_arms_by_gamma()])
    rows = []
    for cell in R.cells():
        pub = sorted(set(published_gamma(cell, store.seeds)))
        out = [cell[0], cell[1], cell[2], "/".join(f"{g:g}" for g in pub)]
        ref = store.values.get((cell, "R0"), {})
        for arm in span_arms_by_gamma():
            va = store.values.get((cell, arm), {})
            diffs = []
            for s in store.seeds:
                x, y = va.get(s, {}).get(metric), ref.get(s, {}).get(metric)
                diffs.append(None if x is None or y is None else x - y)
            st = R.seed_stats(diffs, expected=len(store.seeds))
            if st is None:
                out.append("–")
            elif len(store.seeds) == 1:
                out.append(f"{st[0]:+.4f}")
            else:
                out.append(f"{st[0]:+.4f} ± {st[1]:.4f}")
        rows.append(out)
    return [f"### Δ {label} vs the main-table γ (R0), paired by seed", ""] + R.md_table(header, rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid-root", type=Path, default=None,
                    help="ablation grid directory (default: $ABL_ROOT/grid)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="report directory")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.SEEDS),
                    help="seeds to report; a value appears only if ALL of them have it")
    ap.add_argument("--checkpoint", choices=["best", "final"], default="best",
                    help="best = best-validation-MonoScore epoch; final = epoch 50")
    args = ap.parse_args()
    unknown = set(args.seeds) - set(C.SEEDS)
    if unknown:
        ap.error(f"unknown seeds {sorted(unknown)}; the grid has {C.SEEDS}")
    if args.grid_root is not None:
        C.set_root(args.grid_root)
    args.root = args.out
    args.root.mkdir(parents=True, exist_ok=True)
    store = SpanStore(args.seeds, args.checkpoint).load()
    suffix = "" if args.checkpoint == "best" else "_final"

    n_span_expected = 12 * len(C.SPAN_ARMS) * len(args.seeds)
    n_span_done = sum(1 for (cell, arm), ps in store.values.items()
                      if arm in C.SPAN_ARMS for s in ps)
    lines = ["# DualSAE loss-coefficient ablation: γ (span_coef, weight on L_dcor)", "",
             f"Seeds: {', '.join(map(str, args.seeds))}. γ-arm runs scored: "
             f"{n_span_done}/{n_span_expected}. Checkpoint: **{args.checkpoint}** "
             + ("(best validation MonoScore over epochs)."
                if args.checkpoint == "best" else "(epoch 50, the converged model)."), "",
             "Objective L = L_rec + λ·L_sparse + γ·L_dcor. For TopK/BatchTopK L_sparse = 0, so γ is "
             "the only live loss coefficient. Every Dual arm replays the main-table config "
             "of its cell and seed and changes only γ (warm-up, lr, projector unchanged). "
             "**SAE** = plain SAE on DualSAE's own Sparse Branch code (grid arm S); "
             "**Dual @ main γ** = the main-table γ retrained (grid arm R0). The best Dual setting per "
             "row is bold. MonoScore is on rep x, R²/KNN/LP on x̂, "
             "purity on the sparse code (OIv7: val; other OIv7 metrics: test).", ""]
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

    lines += ["## Dose-response per metric", ""]
    for metric in METRICS:
        lines += metric_table(store, metric, R.LABEL[metric],
                              digits=1 if metric == "l0" else 4)
    lines += ["## Mechanism (read at the selected epoch)", ""]
    for tag in TB_TAGS:
        lines += metric_table(store, tag, TB_LABEL[tag], digits=6 if tag == "val_span" else 4)
    if args.checkpoint == "best":
        lines += metric_table(store, "best_epoch", "Selected epoch (of 50)", digits=0)
    lines += ["## Paired differences vs the main-table γ", ""]
    for metric in ("r2_xhat", "ms", "purity", "knn", "lp"):
        lines += delta_table(store, metric, R.LABEL[metric])

    (args.root / f"REPORT{suffix}.md").write_text("\n".join(lines) + "\n")
    with (args.root / f"long{suffix}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "encoder", "activation", "arm",
                                                    "gamma", "seed", "metric", "value"])
        writer.writeheader()
        writer.writerows(store.flat_rows())
    print(f"wrote {args.root / f'REPORT{suffix}.md'} and long{suffix}.csv ({n_span_done}/{n_span_expected} γ runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
