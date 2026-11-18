#!/usr/bin/env python3
"""Collector of the DualSAE ablation grid (joint vs. sequential training, projector ablation).

Reads the branch_metrics.py results of every grid run and writes
<root>/REPORT.md (diagnostic tables and consistency checks), <root>/long.csv (one value per
cell, seed, arm, reading and metric; the input of the table/figure builders) and
<root>/curves.csv (per-epoch validation curves from the runs' TensorBoard event files).

Conventions:
* Only seeds are averaged. Dataset, encoder and activation stay separate dimensions, and
  comparisons stay within an activation pair.
* L0 is printed beside every row.
* A reading that fails a self-check, or is missing, shows as a blank rather than a partial mean.

Readings are keyed on (dataset, encoder, activation, arm, role). Main-table DualSAE / SAE runs
(arms `published_dual` / `published_baseline`) are included when --main-manifest lists them and
--main-results holds their branch_metrics results (scripts/ablations/branch_metrics).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
VISION_DIR = _HERE.parents[1]
for _p in (str(VISION_DIR), str(VISION_DIR / "scripts" / "ablations" / "grid")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ablation_grid as C  # noqa: E402

# Where each metric comes from on OpenImagesV7: purity from val, everything else from test
# (the main-table convention). ImageNet-1K has only val.
METRIC_SPLIT = {"open_images": {"purity": "val"}}
PRIMARY_SPLIT = {"imagenet": "val", "open_images": "test"}

# Decomposition and comparison metrics, in report order.
METRICS: Tuple[str, ...] = ("r2_xhat", "r2_xhat_global", "r2_xm", "r2_xm_global",
                            "ms", "purity", "knn", "lp", "l0", "dead_frac")
LABEL = {"r2_xhat": "R²(x̂)", "r2_xhat_global": "R²(x̂) glob", "r2_xm": "R²(xM)",
         "r2_xm_global": "R²(xM) glob", "ms": "MS", "purity": "Pur", "knn": "KNN", "lp": "LP",
         "l0": "L0", "dead_frac": "Dead"}
DECOMPOSED: Tuple[str, ...] = ("r2_xhat", "r2_xm_global", "ms", "purity", "knn", "lp")
# With z frozen these cannot move, so any difference between S and F is a defect, not a result.
FROZEN_INVARIANT: Tuple[str, ...] = ("r2_xm", "r2_xm_global", "ms", "purity", "l0", "dead_frac")
FROZEN_TOL = 1e-6
# Tolerances of the R0-vs-main-table reproduction check (R0 replays the main-table config).
PILOT_TOL = {"r2_xhat": 2e-3, "ms": 9e-3, "purity": 1.3e-2, "knn": 2e-3, "lp": 2e-3}
# Below this fraction of training, a selected checkpoint is not a valid calibration point:
# a setting that degrades keeps an early (often epoch-1) best-validation checkpoint.
MIN_BEST_EPOCH_FRAC = 0.2


# --------------------------------------------------------------------------- #
# Reading results
# --------------------------------------------------------------------------- #

def _load(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _result_path(root: Path, run: Path, split: str, checkpoint: str) -> Path:
    tag = "" if checkpoint == "best" else f"__{checkpoint}"
    return root / f"{run.name}__{split}{tag}.json"


def metrics_from_branch_files(run: Path, dataset: str, checkpoint: str,
                              root: Path) -> Tuple[Dict[str, float], List[str], Dict[str, Any]]:
    """The reported metrics for one reading, each from its proper split.

    Returns (metric -> value, problems, meta). A metric that is absent is simply not in the dict,
    so a mean over seeds cannot silently use fewer of them: seed_stats requires all three.
    """
    out: Dict[str, float] = {}
    problems: List[str] = []
    meta: Dict[str, Any] = {}
    primary = _load(_result_path(root, run, PRIMARY_SPLIT[dataset], checkpoint))
    if primary is None:
        return out, [f"no branch_metrics file for {run.name}"], meta
    if not primary.get("self_checks", {}).get("all_ok", False):
        # A cell failing its own self-check must not reach a table.
        return out, [f"self-checks failed for {run.name} ({PRIMARY_SPLIT[dataset]})"], meta
    meta = primary.get("meta", {})
    reps = primary["reps"]
    out["r2_xhat"] = reps["xhat"]["r2_perfeat"]
    out["r2_xhat_global"] = reps["xhat"]["r2_global"]
    out["r2_xm"] = reps["xM"]["r2_perfeat"]
    out["r2_xm_global"] = reps["xM"]["r2_global"]
    # MonoScore probes the raw x (the main-table MonoScore column).
    out["ms"] = reps["x"]["monoscore"]["mean_all"]
    out["knn"] = reps["xhat"]["knn"]["metric"]
    out["lp"] = reps["xhat"]["lp"]["metric"]
    level = primary["model_level"]
    out["l0"] = level["eval_l0_mean"]
    out["dead_frac"] = 1.0 - level["n_active_latents_eval"] / level["dict_size"]
    purity_split = METRIC_SPLIT.get(dataset, {}).get("purity", PRIMARY_SPLIT[dataset])
    purity_file = (primary if purity_split == PRIMARY_SPLIT[dataset]
                   else _load(_result_path(root, run, purity_split, checkpoint)))
    if purity_file is None:
        problems.append(f"no {purity_split} file for purity: {run.name}")
    else:
        out["purity"] = purity_file["model_level"]["binary_purity"]["mean_active"]
    return out, problems, meta


def read_tb_scalars(run_tb_dir: Path, tags: Iterable[str]) -> Dict[str, Dict[int, float]]:
    """Read scalars straight from tfevents files (needs only tensorboardX, not tensorboard).

    Format: a sequence of records, each u64 length, u32 crc, payload, u32 crc. CRCs are not
    verified because this reads our own files; a torn tail from a killed job ends the loop.
    A restarted run's later files overwrite earlier values at the same step.
    """
    from tensorboardX.proto import event_pb2
    wanted = set(tags)
    out: Dict[str, Dict[int, float]] = {t: {} for t in wanted}
    for path in sorted(run_tb_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime):
        with path.open("rb") as handle:
            while True:
                head = handle.read(12)
                if len(head) < 12:
                    break
                length = struct.unpack("<Q", head[:8])[0]
                payload = handle.read(length)
                handle.read(4)
                if len(payload) < length:
                    break
                event = event_pb2.Event()
                event.ParseFromString(payload)
                for value in event.summary.value:
                    if value.tag in wanted:
                        out[value.tag][int(event.step)] = float(value.simple_value)
    return out


def seed_stats(values: Sequence[Optional[float]], expected: int = len(C.SEEDS)) -> Optional[Tuple[float, float]]:
    """(mean, sd) across seeds, or None unless every seed is present and finite.

    Averaging only the seeds that happened to finish would produce a mean of a different quantity
    than its neighbours, with nothing in the table to show it.
    """
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if len(clean) != expected:
        return None
    return statistics.fmean(clean), (statistics.stdev(clean) if len(clean) > 1 else 0.0)


def fmt(stat: Optional[Tuple[float, float]], digits: int = 4, signed: bool = False) -> str:
    if stat is None:
        return "–"
    sign = "+" if signed else ""
    return f"{stat[0]:{sign}.{digits}f} ± {stat[1]:.{digits}f}"


def md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


# --------------------------------------------------------------------------- #
# Collecting readings
# --------------------------------------------------------------------------- #

CellKey = Tuple[str, str, str]   # dataset, encoder, activation


def cells() -> Iterable[CellKey]:
    for dataset in C.DATASETS:
        for encoder in C.ENCODERS:
            for activation in C.ACTIVATIONS:
                yield dataset, encoder, activation


class Store:
    """All readings: store[(cell, arm, role)][seed] = {metric: value}, with per-seed metadata."""

    def __init__(self) -> None:
        self.values: Dict[Tuple[CellKey, str, str], Dict[int, Dict[str, float]]] = defaultdict(dict)
        self.meta: Dict[Tuple[CellKey, str, str], Dict[int, Dict[str, Any]]] = defaultdict(dict)
        self.problems: List[str] = []

    def add(self, cell: CellKey, arm: str, role: str, seed: int, run: Path, checkpoint: str,
            root: Path) -> None:
        metrics, problems, meta = metrics_from_branch_files(run, cell[0], checkpoint, root)
        self.problems.extend(problems)
        if metrics:
            self.values[(cell, arm, role)][seed] = metrics
            self.meta[(cell, arm, role)][seed] = meta

    def stat(self, cell: CellKey, arm: str, role: str, metric: str) -> Optional[Tuple[float, float]]:
        per_seed = self.values.get((cell, arm, role), {})
        return seed_stats([per_seed.get(s, {}).get(metric) for s in C.SEEDS])

    def delta(self, cell: CellKey, a: Tuple[str, str], b: Tuple[str, str],
              metric: str) -> Optional[Tuple[float, float]]:
        """Paired per-seed difference a - b, averaged over seeds."""
        va = self.values.get((cell, *a), {})
        vb = self.values.get((cell, *b), {})
        diffs: List[Optional[float]] = []
        for seed in C.SEEDS:
            x, y = va.get(seed, {}).get(metric), vb.get(seed, {}).get(metric)
            diffs.append(None if x is None or y is None else x - y)
        return seed_stats(diffs)

    def flat_rows(self) -> List[dict]:
        rows = []
        for (cell, arm, role), per_seed in sorted(self.values.items()):
            for seed, metrics in sorted(per_seed.items()):
                for metric, value in metrics.items():
                    rows.append({"dataset": cell[0], "encoder": cell[1], "activation": cell[2],
                                 "arm": arm, "role": role, "seed": seed, "metric": metric,
                                 "value": value})
        return rows


def snapshot_plan_for(cell: CellKey, seed: int) -> Dict[str, int]:
    """F's role -> step mapping, recomputed from the finished Z and R0 runs."""
    dataset, encoder, activation = cell
    for row in C.load_published_rows():
        if (row["dataset"], row["encoder"], row["method"], row["seed"]) == (
                dataset, encoder, C.DUAL_METHOD[activation], seed):
            return C.snapshot_plan(row, C.published_config(row)) or {}
    return {}


def load_main_manifest(path: Optional[Path]) -> List[dict]:
    """Rows {dataset, encoder, method, seed, run_dir} of the main-table runs (TopK / BatchTopK
    pairs), as written by scripts/ablations/make_run_manifest.py. Empty when not given."""
    if path is None:
        return []
    rows = json.loads(Path(path).read_text())
    wanted = set(C.DUAL_METHOD.values()) | set(C.BASELINE_METHOD.values())
    return [r for r in rows if r["method"] in wanted]


def load_all(main_rows: Sequence[dict] = (), main_results: Optional[Path] = None) -> Store:
    store = Store()
    for row in main_rows:
        cell = (row["dataset"], row["encoder"], C.activation_of(row["method"]))
        arm = "published_dual" if row["method"].startswith("dual_") else "published_baseline"
        store.add(cell, arm, "best", row["seed"], Path(row["run_dir"]), "best", main_results)
    for cell in cells():
        dataset, encoder, activation = cell
        for seed in C.SEEDS:
            for arm in C.INDEPENDENT_ARMS:
                run = C.run_dir(dataset, encoder, activation, arm, seed)
                if not C.run_is_complete(run):
                    continue
                store.add(cell, arm, "best", seed, run, "best", C.BRANCH_METRICS_RESULTS)
                if C.read_best_epoch(run) == C.EPOCHS:
                    # Best and final are the same weights, so the final reading already exists.
                    store.values[(cell, arm, "final")][seed] = store.values[(cell, arm, "best")].get(seed, {})
                    store.meta[(cell, arm, "final")][seed] = store.meta[(cell, arm, "best")].get(seed, {})
                else:
                    store.add(cell, arm, "final", seed, run, "final", C.BRANCH_METRICS_RESULTS)
            f_run = C.run_dir(dataset, encoder, activation, "F", seed)
            if C.run_is_complete(f_run):
                store.add(cell, "F", "end_of_training", seed, f_run, "best", C.BRANCH_METRICS_RESULTS)
                for role, step in snapshot_plan_for(cell, seed).items():
                    snap = f_run.parent / f"{f_run.name}__step{step}"
                    store.add(cell, "F", f"at_{role}", seed, snap, "best", C.BRANCH_METRICS_RESULTS)
    return store


# --------------------------------------------------------------------------- #
# Experiment 2
# --------------------------------------------------------------------------- #

def mixed_settings_note(cell: CellKey) -> str:
    """Name the main-table settings that vary across seeds within a cell, since a seed mean there
    averages over more than seeds."""
    spans, warms = set(), set()
    for row in C.load_published_rows():
        if (row["dataset"], row["encoder"], row["method"]) == (cell[0], cell[1], C.DUAL_METHOD[cell[2]]):
            spans.add(C.published_config(row)["span_coef"])
            warms.add(C.published_config(row)["warmup_steps"])
    notes = []
    if len(spans) > 1:
        notes.append("span " + "/".join(str(s) for s in sorted(spans)))
    if len(warms) > 1:
        notes.append("warm-up " + "/".join(str(w) for w in sorted(warms)))
    return "; ".join(notes)


def frozen_invariant_gate(store: Store, gates: List[str]) -> None:
    """F against S: same tensors and same z-dependent metrics, or the arm is not what it claims."""
    from train_dual import sparse_state_sha256
    import torch
    worst = 0.0
    n_pairs = n_sha = n_sha_bad = 0
    for cell in cells():
        dataset, encoder, activation = cell
        for seed in C.SEEDS:
            s_vals = store.values.get((cell, "S", "best"), {}).get(seed)
            f_run = C.run_dir(dataset, encoder, activation, "F", seed)
            s_run = C.run_dir(dataset, encoder, activation, "S", seed)
            if s_vals is None or not C.run_is_complete(f_run):
                continue
            n_sha += 1
            sha_s = sparse_state_sha256(torch.load(s_run / "best_val_monoscore_model.pt", map_location="cpu",
                                                   weights_only=False)["model_state_dict"])
            sha_f = sparse_state_sha256(torch.load(f_run / "best_val_monoscore_model.pt", map_location="cpu",
                                                   weights_only=False)["model_state_dict"])
            n_sha_bad += int(sha_s != sha_f)
            for (c, arm, role), per_seed in store.values.items():
                if c != cell or arm != "F" or seed not in per_seed:
                    continue
                n_pairs += 1
                for metric in FROZEN_INVARIANT:
                    if metric in s_vals and metric in per_seed[seed]:
                        worst = max(worst, abs(s_vals[metric] - per_seed[seed][metric]))
    if n_sha == 0:
        gates.append("F-vs-S invariants: no F runs yet")
        return
    verdict = "PASS" if worst <= FROZEN_TOL and n_sha_bad == 0 else "FAIL"
    gates.append(f"F-vs-S invariants ({', '.join(FROZEN_INVARIANT)}): worst |Δ| = {worst:.2e} "
                 f"over {n_pairs} readings; sparse-branch SHA mismatches {n_sha_bad}/{n_sha}: **{verdict}**")


def experiment_2(store: Store, lines: List[str], gates: List[str]) -> None:
    lines += ["## Joint vs. sequential training", "",
              "**S** is a standard SAE running DualSAE's exact sparse-branch code. **F** freezes S and "
              "trains only the residual branch. **Z** is normal two-stage training with r̂ held at 0 "
              "during warm-up. **D** is the retrained DualSAE.", "",
              "`F−S` is the residual branch alone, `Z−F` is joint training, and `D−Z` is the frozen "
              "random projector during warm-up. They sum to `D−S`. Main-table checkpoints are shown "
              "for reference and are outside the decomposition, because S differs from the main-table "
              "SAE baselines (decoder normalization axis, no AuxK, no clipping, uses respawn).", ""]
    frozen_invariant_gate(store, gates)
    sum_errors: List[float] = []

    for reading, s_role, f_role, z_role, d_role, heading in (
            ("best", "best", "at_Z_best", "best", "best",
             "Primary: selected checkpoints, with F at Z's residual-update count"),
            ("final", "final", "at_dual_final", "final", "final",
             "Converged: final checkpoints, with F at the final Dual residual-update count")):
        lines += [f"### {heading}", ""]
        rows = []
        for cell in cells():
            for metric in DECOMPOSED:
                a = store.delta(cell, ("F", f_role), ("S", s_role), metric)
                b = store.delta(cell, ("Z", z_role), ("F", f_role), metric)
                c = store.delta(cell, ("R0", d_role), ("Z", z_role), metric)
                total = store.delta(cell, ("R0", d_role), ("S", s_role), metric)
                if None not in (a, b, c, total):
                    sum_errors.append(abs(a[0] + b[0] + c[0] - total[0]))
                rows.append([cell[0], cell[1], cell[2], LABEL[metric],
                             fmt(store.stat(cell, "S", s_role, metric)), fmt(a, signed=True),
                             fmt(b, signed=True), fmt(c, signed=True), fmt(total, signed=True)])
        lines += md_table(["Dataset", "Encoder", "Activation", "Metric", "S",
                           "F−S (residual)", "Z−F (joint)", "D−Z (warm-up projector)", "D−S (total)"],
                          rows)
    if sum_errors:
        worst = max(sum_errors)
        gates.append(f"Decomposition sums to D−S: worst |error| = {worst:.2e}: "
                     f"**{'PASS' if worst < 1e-9 else 'FAIL'}**")

    lines += ["### All readings, including main-table references and the 50-epoch F", ""]
    arms = [("published_baseline", "best", "Main-table SAE"), ("S", "best", "S"),
            ("F", "at_Z_best", "F @ Z's updates"), ("F", "at_D_best", "F @ D's updates"),
            ("F", "at_dual_final", "F @ final Dual updates"), ("F", "end_of_training", "F @ 50 epochs"),
            ("Z", "best", "Z"), ("R0", "best", "D"), ("published_dual", "best", "Main-table DualSAE")]
    rows = []
    for cell in cells():
        note = mixed_settings_note(cell)
        for arm, role, label in arms:
            rows.append([cell[0], cell[1], cell[2], label]
                        + [fmt(store.stat(cell, arm, role, m)) for m in METRICS]
                        + [note if arm == "published_dual" else ""])
    lines += md_table(["Dataset", "Encoder", "Activation", "Arm"] + [LABEL[m] for m in METRICS]
                      + ["Settings varying across seeds"], rows)


# --------------------------------------------------------------------------- #
# Projector ablation
# --------------------------------------------------------------------------- #

def poly_param_count(arm: str, act_size: int) -> int:
    """Projector parameters, including bias terms. S has none."""
    hidden = {"R0": [], "L1": [], "B64": [64], "B256": [256], "B512": [512], "B2048": [2048],
              "B512G": [512], "D3": [512, 512], "D4": [512, 512, 512]}[arm]
    widths = [act_size] + hidden + [act_size]
    return sum(a * b + b for a, b in zip(widths, widths[1:]))


ACT_SIZE = {"clip": 768, "sigclip": 1152, "vit": 1024}
ARM_LABEL = {"R0": "R0: Linear(A,A)+GELU", "L1": "L1: Linear(A,A)",
             "B64": "B64: h=64", "B256": "B256: h=256", "B512": "B512: h=512",
             "B2048": "B2048: h=2048", "B512G": "B512G: h=512 + terminal GELU",
             "D3": "D3: A→512→512→A", "D4": "D4: A→512→512→512→A"}


def warmup_epochs_in_full(dataset: str, warmup: int) -> int:
    """Last 1-based epoch that ended entirely inside warm-up (0 if warm-up ended within epoch 1).

    Warm-up takes warmup + 1 optimizer updates, so an epoch ending exactly at update warmup + 1
    still counts as warm-up: its validation pass precedes any projector training.
    """
    return (warmup + 1) // C.STEPS_PER_EPOCH[dataset]


def curves_for(cell: CellKey, arm: str, seed: int) -> Dict[str, Dict[int, float]]:
    tb_dir = C.TB_ROOT / C.exp_name(*cell, arm, seed)
    if not tb_dir.exists():
        return {}
    return read_tb_scalars(tb_dir, ("val_r2", "val_r2_mono", "val_mono_mean"))


def experiment_3(store: Store, lines: List[str], root: Path) -> None:
    lines += ["## Projector of the Residual Branch", "",
              "Each architecture uses its own projector in both stages (literal two-stage schedule), so the "
              "frozen random projector during warm-up differs between arms. The warm-up-end columns "
              "show whether an arm was already affected before its projector began training. A "
              "projector that is still rising at the end of training may be limited by the fixed "
              "budget rather than by capacity; the last-5-epochs column shows that.", ""]
    published = {(r["dataset"], r["encoder"], r["method"], r["seed"]): r for r in C.load_published_rows()}
    curve_rows: List[dict] = []
    per_arm_curves: Dict[Tuple[CellKey, str], Dict[int, Dict[str, Dict[int, float]]]] = {}
    for cell in cells():
        for arm in tuple(C.ARCH_ARMS) + ("S", "Z", "F"):
            for seed in C.SEEDS:
                curves = curves_for(cell, arm, seed)
                if not curves:
                    continue
                per_arm_curves.setdefault((cell, arm), {})[seed] = curves
                for tag, series in curves.items():
                    for step, value in sorted(series.items()):
                        curve_rows.append({"dataset": cell[0], "encoder": cell[1], "activation": cell[2],
                                           "arm": arm, "seed": seed, "epoch": step + 1,
                                           "metric": tag, "value": value})
    with (root / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "encoder", "activation", "arm", "seed",
                                                    "epoch", "metric", "value"])
        writer.writeheader()
        writer.writerows(curve_rows)

    for role, heading in (("best", "Selected checkpoints"), ("final", "Final checkpoints")):
        lines += [f"### {heading}", ""]
        rows = []
        for cell in cells():
            ref_label = "S: no projector (different objective)"
            rows.append([cell[0], cell[1], cell[2], ref_label, "0"]
                        + [fmt(store.stat(cell, "S", role, m)) for m in METRICS]
                        + ["–", "–", "–", "–"])
            for arm in C.ARCH_ARMS:
                curves = per_arm_curves.get((cell, arm), {})
                warm_r2, warm_xm, tail = [], [], []
                for seed in C.SEEDS:
                    row = published.get((cell[0], cell[1], C.DUAL_METHOD[cell[2]], seed))
                    series = curves.get(seed)
                    if not row or not series:
                        warm_r2.append(None); warm_xm.append(None); tail.append(None)
                        continue
                    warm_epoch = warmup_epochs_in_full(cell[0], int(C.published_config(row)["warmup_steps"]))
                    warm_r2.append(series["val_r2"].get(warm_epoch - 1) if warm_epoch else None)
                    warm_xm.append(series["val_r2_mono"].get(warm_epoch - 1) if warm_epoch else None)
                    last, earlier = series["val_r2"].get(C.EPOCHS - 1), series["val_r2"].get(C.EPOCHS - 6)
                    tail.append(None if last is None or earlier is None else last - earlier)
                best_epochs = [store.meta.get((cell, arm, "best"), {}).get(s, {}).get("best_epoch")
                               for s in C.SEEDS]
                rows.append([cell[0], cell[1], cell[2], ARM_LABEL[arm],
                             f"{poly_param_count(arm, ACT_SIZE[cell[1]]):,}"]
                            + [fmt(store.stat(cell, arm, role, m)) for m in METRICS]
                            + [fmt(seed_stats(warm_r2)), fmt(seed_stats(warm_xm)),
                               fmt(seed_stats(tail), signed=True),
                               "/".join("–" if e is None else str(e) for e in best_epochs)])
        lines += md_table(["Dataset", "Encoder", "Activation", "Projector", "Params"]
                          + [LABEL[m] for m in METRICS]
                          + ["Warm-up-end R²(x̂)", "Warm-up-end R²(xM)", "Val R² change, last 5 epochs",
                             "Best epochs"], rows)


# --------------------------------------------------------------------------- #
# Cross-experiment gates
# --------------------------------------------------------------------------- #

def pilot_gate(store: Store, gates: List[str], lines: List[str]) -> None:
    """R0 against the main-table checkpoint it replays, seed by seed (when main readings exist)."""
    rows, misses, compared = [], 0, 0
    for cell in cells():
        for seed in C.SEEDS:
            retrained = store.values.get((cell, "R0", "best"), {}).get(seed)
            original = store.values.get((cell, "published_dual", "best"), {}).get(seed)
            if not retrained or not original:
                continue
            compared += 1
            row = [cell[0], cell[1], cell[2], str(seed)]
            miss = False
            for metric, tol in PILOT_TOL.items():
                if metric not in retrained or metric not in original:
                    row.append("–")
                    continue
                diff = retrained[metric] - original[metric]
                row.append(f"{diff:+.4f}" + ("" if abs(diff) <= tol else " ✗"))
                miss = miss or abs(diff) > tol
            misses += int(miss)
            rows.append(row)
    if not rows:
        gates.append("R0 vs main-table run: no pairs with both readings")
        return
    gates.append(f"R0 vs main-table run it replays: {misses}/{compared} runs outside "
                 "the tolerances (see table)")
    lines += ["## R0 vs main-table run", "",
              "Retrained reference minus the main-table checkpoint it replays. ✗ marks a difference "
              "outside PILOT_TOL. Training is not bit-deterministic across runs; all grid arms are "
              "trained under the same code, so comparisons between grid arms are unaffected.", ""]
    lines += md_table(["Dataset", "Encoder", "Activation", "Seed"] + [LABEL[m] for m in PILOT_TOL], rows)


def selection_flags(store: Store, gates: List[str]) -> None:
    """Flag selected checkpoints too early to represent their setting, or before warm-up ended."""
    published = {(r["dataset"], r["encoder"], r["method"], r["seed"]): r for r in C.load_published_rows()}
    flagged: List[str] = []
    for (cell, arm, role), per_seed in store.meta.items():
        if role != "best" or arm not in C.INDEPENDENT_ARMS:
            continue
        for seed, meta in per_seed.items():
            epoch = meta.get("best_epoch")
            if epoch is None:
                continue
            row = published.get((cell[0], cell[1], C.DUAL_METHOD[cell[2]], seed))
            warm = 0 if arm == "S" or row is None else int(C.published_config(row)["warmup_steps"])
            early = epoch < MIN_BEST_EPOCH_FRAC * C.EPOCHS
            pre_unfreeze = arm != "S" and epoch * C.STEPS_PER_EPOCH[cell[0]] <= warm + 1
            if early or pre_unfreeze:
                flagged.append(f"{'/'.join(cell)}/{arm}/seed{seed} (best epoch {epoch})")
    gates.append(f"Selection: {len(flagged)} run(s) selected a checkpoint before warm-up ended or "
                 f"before {MIN_BEST_EPOCH_FRAC:.0%} of training"
                 + (": " + ", ".join(sorted(flagged)) if flagged else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None,
                    help="grid directory (default: $ABL_ROOT/grid); reports are written here")
    ap.add_argument("--main-manifest", type=Path, default=None,
                    help="JSON list of main-table runs (scripts/ablations/make_run_manifest.py)")
    ap.add_argument("--main-results", type=Path, default=None,
                    help="branch_metrics results directory of the main-table runs")
    args = ap.parse_args()
    if args.root is not None:
        C.set_root(args.root)
    root = C.ROOT
    # Before any section runs: the projector section writes curves.csv as it reads the event files.
    root.mkdir(parents=True, exist_ok=True)
    main_rows = load_main_manifest(args.main_manifest)
    if main_rows and args.main_results is None:
        ap.error("--main-manifest needs --main-results")
    store = load_all(main_rows, args.main_results)

    gates: List[str] = []
    body: List[str] = []
    experiment_2(store, body, gates)
    experiment_3(store, body, root)
    tail: List[str] = []
    pilot_gate(store, gates, tail)
    selection_flags(store, gates)

    expected = len(C.CORE_ARMS) * len(C.DATASETS) * len(C.ENCODERS) * len(C.ACTIVATIONS) * len(C.SEEDS)
    # CORE_ARMS, not ALL_ARMS: the gamma and scale arms share the grid but are reported by
    # collect_span_ablation.py / collect_scale_ablation.py.
    finished = sum(1 for cell in cells() for arm in C.CORE_ARMS for seed in C.SEEDS
                   if C.run_is_complete(C.run_dir(*cell, arm, seed)))
    head = ["# DualSAE ablation grid: joint vs. sequential training and projector", "",
            f"Training runs finished: {finished}/{expected}.", "", "## Checks", ""]
    head += [f"- {g}" for g in gates] + [""]
    if store.problems:
        shown = sorted(set(store.problems))
        head += [f"## Unavailable readings ({len(shown)})", ""]
        head += [f"- {p}" for p in shown[:40]]
        if len(shown) > 40:
            head.append(f"- ... and {len(shown) - 40} more")
        head.append("")

    (root / "REPORT.md").write_text("\n".join(head + tail + body) + "\n")
    with (root / "long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "encoder", "activation", "arm", "role",
                                                    "seed", "metric", "value"])
        writer.writeheader()
        writer.writerows(store.flat_rows())
    print(f"wrote {root / 'REPORT.md'}, long.csv, curves.csv")
    for gate in gates:
        print(" -", gate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
