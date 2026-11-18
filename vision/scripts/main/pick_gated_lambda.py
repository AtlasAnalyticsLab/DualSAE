#!/usr/bin/env python3
"""Choose Gated-SAE's lambda per (dataset, encoder) from the L0 sweep.

Gated SAE has no k. Its sparsity is emergent from l1_coef, so a Gated row can
only be placed beside a TopK/BatchTopK row (both pinned at L0 = 64) if lambda
has been calibrated to land the realized L0 in the same neighbourhood. This
reads the l0_probe.py outputs for the sweep and applies a PRE-REGISTERED rule.

Selection rule
--------------
    winner = argmin_lambda | log( L0_val(lambda) / 64 ) |
             subject to dead_frac(lambda) <= 0.90
    ties -> smaller lambda (less regularised, so better R2)

Log-space distance because L0 is multiplicative: 32 and 128 are equally far
from 64, which absolute distance gets wrong (it would prefer 32). The
dead-fraction side condition rejects the degenerate solution where L0 hits 64
only because almost every latent is dead -- a model with 300 live units firing
64 at a time is not the same object as one with 8192.

Selection uses seed 42 ONLY; seeds 43/44 never inform it.

Honest failure
--------------
"Near 64" means realized L0 in [48, 85] (x1.33 either way). If the grid
brackets 64 but no point is in band, log-bisect (up to --bisect rounds). If it
never crosses 64, extend the grid geometrically. If it still does not bracket,
DO NOT fudge: the cell is recorded as unresolved and its Gated row is reported
at whatever L0 it reached, with L0 and dead-fraction printed beside every
metric -- exactly the treatment the JumpReLU pair already gets.

Usage (CPU, seconds):
    python scripts/main/pick_gated_lambda.py                 # baseline arm
    python scripts/main/pick_gated_lambda.py --arm dual      # Dual-Gated arm
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

# Make the vision core (two levels up) and this directory importable.
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import grid_config as C  # noqa: E402

PROBE_DIR = Path(C.STATE_DIR) / "l0"
DEFAULT_OUT = Path(C.STATE_DIR) / "gated_lambda_choice.json"
# Probe split per dataset: selection must read the split the row is reported
# on wherever possible. ImageNet has no test split at all.
PROBE_SPLIT = {"imagenet": "val", "open_images": "test"}


BRANCH_METRICS = Path(C.EVAL_RESULTS)


TRAJ_PATH = Path(C.STATE_DIR) / "l0_trajectory.json"


def _load_trajectories() -> dict:
    """exp_name -> per-epoch L0 summary, written by scripts/l0_trajectory.py.

    Separate file because reading TensorBoard events needs the `tensorboard`
    package, which this picker does not. Absent file is not an error: every
    reading that needs it degrades to "unavailable" and says so.
    """
    try:
        return json.loads(TRAJ_PATH.read_text())
    except Exception:
        return {}


def _from_branch_metrics(run_dir: Path, split: str) -> dict | None:
    """L0 + dead-fraction out of a branch_metrics result, if one exists.

    Preferred over a dedicated l0_probe run purely to avoid duplicating GPU
    work: the evaluation wave already streams the same split through the same
    checkpoint and records exactly these two numbers, so a separate probe pass
    would be a second encode of the same data for the same answer.
    """
    p = BRANCH_METRICS / f"{run_dir.name}__{split}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    ml = d.get("model_level", {})
    if "eval_l0_mean" not in ml:
        return None
    dict_size = float(ml["dict_size"])
    n_active = float(ml["n_active_latents_eval"])
    meta = d.get("meta", {})
    return {"l0_mean": float(ml["eval_l0_mean"]),
            "dead_frac": 1.0 - n_active / dict_size,
            "gate_open_frac": None, "source": "branch_metrics",
            "best_epoch": meta.get("best_epoch"),
            "val_monoscore": meta.get("val_monoscore"),
            "exp_name": meta.get("exp_name", run_dir.name)}


def _from_l0_probe(probe_dir: Path, run_dir: Path, split: str) -> dict | None:
    p = probe_dir / f"{run_dir.name}__{split}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {"l0_mean": float(d["l0_mean"]),
            "dead_frac": float(d["dead_frac"]),
            "gate_open_frac": d.get("gate_open_frac"),
            "best_epoch": d.get("best_epoch"),
            "val_monoscore": None,
            "source": "l0_probe", "exp_name": d.get("exp_name", run_dir.name)}


# Base-grid exp_name patterns, one per arm. Parsing lambda back out of the
# directory name is safe because the scheme is ours and deterministic
# (`grid_config.dual_name` / the gated baseline name).
_BASELINE_PAT = re.compile(
    r"^t3x_(imagenet|open_images)_(clip|vit|sigclip)_gated_"
    r"n8192k64_l1([0-9.eE+-]+)_seed(\d+)$")
_DUAL_PAT = re.compile(
    r"^t3x_(imagenet|open_images)_(clip|vit|sigclip)_dual_gated_"
    r"n8192k64_l1([0-9.eE+-]+)_span([0-9.eE+-]+)_warm(\d+)_seed(\d+)$")


def extra_probe_cells(arm: str, seen: set) -> list[dict]:
    """REFINEMENT points on disk that the base grid does not enumerate.

    These are bisect probes (baseline arm) and the lambda=0 ceiling probes
    (dual arm). A purely grid-driven read never sees them, so a cell stays
    "bracketed" forever however many refinement rounds finish -- and a probe
    that cost 45 minutes of H100 is silently thrown away.

    Discovered by SCANNING the checkpoint tree rather than by re-deriving them
    from the previous round's brackets: that derivation makes an already-trained
    probe invisible the moment the brackets change or are cleared.

    The BASE grid still defines completeness (see `missing` in main): a
    refinement is a bonus measurement, never a substitute for a grid point.

    The dual scan is pinned to the (span, warm-up) CENTRE and the sweep seed.
    That filter is load-bearing: the 65-cell Dual-Gated sweep varies exactly
    those two axes at the CHOSEN lambda, and a point at a different span is not
    a measurement of lambda at all -- letting one in would calibrate lambda
    against a span effect.
    """
    out = []
    glob = ("t3x_*_gated_n8192k64_l1*_seed*" if arm == "baseline"
            else "t3x_*_dual_gated_n8192k64_l1*_span*_warm*_seed*")
    pat = _BASELINE_PAT if arm == "baseline" else _DUAL_PAT
    for d in sorted(Path(C.OUTPUT_DIR).glob(glob)):
        m = pat.match(d.name)
        if not m or d.name in seen:
            continue
        if arm == "baseline":
            ds, enc, lam_s, seed = m.groups()
            if int(seed) != C.SWEEP_SEED:
                continue
        else:
            ds, enc, lam_s, span_s, warm_s, seed = m.groups()
            if (int(seed) != C.SWEEP_SEED
                    or int(warm_s) != C.DUAL_WARMUP_CENTRE[ds]
                    or abs(float(span_s) - C.DUAL_SPAN_CENTRE) > 1e-12):
                continue
        out.append({"dataset": ds, "encoder": enc, "seed": int(seed),
                    "l1_coef": float(lam_s), "exp_name": d.name})
    return out


def load_probes(probe_dir: Path, arm: str,
                baseline_lambda: dict | None = None) -> dict:
    """(dataset, encoder, l1_coef) -> measurement, driven BY THE GRID.

    Iterating the grid rather than scanning a directory is deliberate: it makes
    a missing cell explicit (the picker can then say "3/8 lambdas probed" and
    refuse) instead of silently selecting the best of whatever happened to be
    on disk -- which, on a partially finished sweep, is exactly how you end up
    calibrating against the fastest-converging lambda.
    """
    grid = (C.gated_lambda_grid() if arm == "baseline"
            else C.dual_gated_lambda_grid(baseline_lambda or {}))
    # Also read every REFINEMENT point already trained -- see
    # `extra_probe_cells`, which is where the scan and its guards live.
    seen = {c["exp_name"] for c in grid}
    grid = grid + extra_probe_cells(arm, seen)
    traj = _load_trajectories()
    try:
        traj_mtime = TRAJ_PATH.stat().st_mtime
    except OSError:
        traj_mtime = None
    out = {}
    for cell in grid:
        ds, enc, lam = cell["dataset"], cell["encoder"], float(cell["l1_coef"])
        run = Path(C.OUTPUT_DIR) / cell["exp_name"]
        if not (run / "autoencoder_final.pt").exists():
            continue                      # still training; not a missing probe
        split = PROBE_SPLIT[ds]
        got = (_from_branch_metrics(run, split)
               or _from_l0_probe(probe_dir, run, split))
        if got is None:
            continue
        # l0_final is the CONVERGED model's L0, read from the training
        # trajectory rather than from the selected checkpoint. It is the
        # fallback reading when checkpoint selection is untrustworthy for a
        # whole cell; see pick().
        t = traj.get(cell["exp_name"]) or {}
        # STALENESS GUARD. The trajectory JSON is regenerated by a separate
        # script, so it can lag the checkpoints -- and a run retrained under the
        # same exp_name would otherwise be read with its PREVIOUS trajectory. Since l0_final can decide a reported lambda via
        # the fallback in pick(), a stale reading is worse than none.
        l0_final = t.get("l0_final")
        if l0_final is not None and traj_mtime is not None:
            try:
                if (run / "autoencoder_final.pt").stat().st_mtime > traj_mtime:
                    l0_final = None       # trajectory predates this checkpoint
            except OSError:
                l0_final = None
        out[(ds, enc, lam)] = {**got, "l1_coef": lam, "dataset_name": ds,
                               "extractor_name": enc, "eval_split": split,
                               "l0_final": l0_final,
                               "n_epochs_validated": t.get("n_epochs_validated")}
    return out


# A branch that has stopped firing is not a sparse model, it is an absent one.
# A too-large lambda can close every gate of the Dual-Gated Sparse Branch, so
# this must be an explicit rejection rather than something the log-distance
# rule happens to score badly.
MIN_ALIVE_L0 = 1.0

# A run whose BEST-val-MonoScore epoch is very early did not converge at that
# lambda -- it peaked and then degraded, and the trainer kept the early
# checkpoint. Its L0 is the L0 of an undertrained model, not of that lambda.
#
# This is not hypothetical: in the sweep, imagenet/clip at lambda 0.3/1/3 and
# open_images/clip at 0.3/1/3 all selected EPOCH 1 of 50, and their L0s
# (351, 110, 16 / 765, 589, 320) are what made L0-vs-lambda look wildly
# non-monotone. Filtered to converged points the relationship is clean.
MIN_BEST_EPOCH_FRAC = 0.2          # >= 20% of the run (epoch 10 of 50)
TOTAL_EPOCHS = C.EPOCHS


def _probe_lambda(below: dict, above: dict, clamp: bool = True) -> float:
    """Where to place the next lambda, from two CONVERGED measurements.

    Log-log linear interpolation (regula falsi) rather than a geometric
    midpoint. L0 vs lambda is smooth and close to a power law once
    early-peaking runs are filtered out, so fitting a line through
    (log lambda, log L0) and solving for the target lands far closer than
    halving the bracket -- and each round costs a full 50-epoch training run.

    When one endpoint is an order of magnitude from the target and the other
    is close, the geometric midpoint is far from the root; interpolation in
    log-log space reaches in one round what bisection needs several for.

    The result is CLAMPED strictly inside the bracket: a near-flat or
    noisy segment can otherwise extrapolate outside it, and a probe outside its
    own bracket is not a refinement.
    """
    import math
    x0, y0 = math.log(below["l1_coef"]), math.log(max(below["l0_mean"], 1e-9))
    x1, y1 = math.log(above["l1_coef"]), math.log(max(above["l0_mean"], 1e-9))
    target = math.log(C.L0_TARGET)
    if abs(y1 - y0) < 1e-9:
        x = 0.5 * (x0 + x1)                      # flat segment: fall back
    else:
        x = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    if clamp:
        lo, hi = min(x0, x1), max(x0, x1)
        # Keep it a genuine interior refinement (5% inside each end).
        margin = 0.05 * (hi - lo)
        x = min(max(x, lo + margin), hi - margin)
    return math.exp(x)


def pick(cands: list[dict]) -> tuple[dict | None, str]:
    """Apply the rule. Returns (winner or None, reason)."""
    min_ep = max(1, int(MIN_BEST_EPOCH_FRAC * TOTAL_EPOCHS))
    early = [c for c in cands
             if c.get("best_epoch") is not None and c["best_epoch"] < min_ep]
    converged = [c for c in cands if c not in early]
    collapsed = [c for c in converged
                 if c["l0_mean"] < MIN_ALIVE_L0 or c["dead_frac"] >= 0.999]
    live = [c for c in converged
            if c["dead_frac"] <= C.MAX_DEAD_FRAC and c["l0_mean"] >= MIN_ALIVE_L0]
    if not converged:
        eps = ", ".join(f"{c['l1_coef']:g}@ep{c.get('best_epoch')}"
                        for c in cands)
        # FALLBACK TO THE CONVERGED MODEL'S OWN L0.
        #
        # The convergence filter exists to stop a MonoScore-selected epoch-1
        # checkpoint being read as "the L0 of this lambda". But when it rejects
        # EVERY point, refusing throws away a cell that may be perfectly well
        # behaved: every lambda can select epoch 1 (val MonoScore is then a
        # property of the initialisation, nearly identical across lambdas)
        # while the trained models converge to well-separated L0s.
        #
        # Sound rather than a loosened threshold: the quantity the calibration
        # wants is "what L0 does a model trained at this lambda have", a
        # property of the TRAINED model. Reading it off a checkpoint chosen by
        # a different criterion (MonoScore) conflates two selection rules; the
        # trajectory is the direct measurement.
        #
        # Deliberately a last resort. Where checkpoint selection IS
        # trustworthy the two readings agree (drift ~1 on every cell that
        # currently resolves), so switching globally would churn working cells
        # for nothing. It never fires silently: the reason string and the JSON
        # both record that the cell was decided this way.
        #
        # NOTE this targets the CONVERGED L0, which for such a cell is NOT the
        # L0 of the checkpoint the evaluation protocol reports. That divergence
        # is real and should be reported next to the row, not hidden here.
        fb = [dict(c, l0_mean=c["l0_final"], _l0_from="trajectory")
              for c in cands if c.get("l0_final") is not None]
        if fb:
            best_fb = min(fb, key=lambda c: abs(math.log(
                max(c["l0_mean"], 1e-9) / C.L0_TARGET)))
            alive = (best_fb["l0_mean"] >= MIN_ALIVE_L0
                     and best_fb["dead_frac"] <= C.MAX_DEAD_FRAC)
            lo, hi = C.L0_BAND
            if alive and lo <= best_fb["l0_mean"] <= hi:
                return best_fb, ("in band (CONVERGED-L0 fallback: every lambda "
                                 f"peaked before epoch {min_ep}, so the "
                                 f"selected checkpoints are untrustworthy; "
                                 f"read L0 at the end of training instead)")
            return None, (f"every lambda peaked before epoch {min_ep} of "
                          f"{TOTAL_EPOCHS} ({eps}). Converged-L0 fallback "
                          f"applied but its best point is L0 "
                          f"{best_fb['l0_mean']:.1f}, outside the band "
                          f"{lo:g}-{hi:g}: extend the grid.")
        return None, (f"every lambda peaked before epoch {min_ep} of "
                      f"{TOTAL_EPOCHS} ({eps}): these runs degraded rather than "
                      f"converged, so their L0 is not a property of lambda, and "
                      f"no trajectory is available to fall back on (run "
                      f"scripts/main/l0_trajectory.py first).")
    if collapsed and not live:
        lams = ", ".join(f"{c['l1_coef']:g}" for c in collapsed)
        return None, (f"COLLAPSED at every lambda ({lams}): the branch stopped "
                      f"firing entirely (L0 < {MIN_ALIVE_L0}). The gate cannot "
                      f"reopen on its own -- L_sparse and L_aux both see "
                      f"ReLU(pi_gate), which is 0 once it shuts. Extend the "
                      f"grid to smaller lambda, or report the arm as degenerate.")
    if not live:
        return None, (f"every lambda exceeds dead_frac {C.MAX_DEAD_FRAC}: the "
                      f"dictionary is collapsing, not sparsifying")
    # argmin |log(L0/target)|, tie -> smaller lambda
    def key(c):
        l0 = max(c["l0_mean"], 1e-9)
        return (abs(math.log(l0 / C.L0_TARGET)), c["l1_coef"])
    best = min(live, key=key)
    lo, hi = C.L0_BAND
    if lo <= best["l0_mean"] <= hi:
        return best, "in band"
    l0s = [c["l0_mean"] for c in live]
    if min(l0s) < C.L0_TARGET < max(l0s):
        # Record the TIGHTEST straddling pair, so the bisect stage acts on the
        # narrowest bracket rather than the grid endpoints.
        below = max((c for c in live if c["l0_mean"] < C.L0_TARGET),
                    key=lambda c: c["l0_mean"])
        above = min((c for c in live if c["l0_mean"] > C.L0_TARGET),
                    key=lambda c: c["l0_mean"])
        best["_bracket"] = {
            "lo": min(below["l1_coef"], above["l1_coef"]),
            "hi": max(below["l1_coef"], above["l1_coef"]),
            "probe": _probe_lambda(above, below),   # above/below by L0
        }
        return best, ("BRACKETED but no grid point in band — log-bisect "
                      "between the two straddling CONVERGED lambdas (early-"
                      "peaking runs are excluded, so the bracket is on the "
                      "clean part of the curve)")
    if min(l0s) > C.L0_TARGET:
        # The converged runs stop ABOVE the target. Two very different causes,
        # and they need opposite actions:
        #
        #  (a) the grid simply does not reach far enough -> extend it;
        #  (b) larger lambdas DO exist in the grid but every one of them
        #      degraded (peaked at epoch ~1). Then the limit is STABILITY, not
        #      range: the run destabilises before it reaches the target
        #      sparsity, and extending further is guaranteed to waste GPU on
        #      runs that will also degrade.
        #
        # Distinguish them by looking at what the early-peaking runs were.
        # E.g. a cell that converges only up to some lambda while all larger
        # lambdas peak at epoch 1: the useful probe is BETWEEN the last
        # converged lambda and the first that broke.
        lam_max_conv = max(c["l1_coef"] for c in live)
        broke = [c["l1_coef"] for c in early if c["l1_coef"] > lam_max_conv]
        if broke:
            lo, hi = lam_max_conv, min(broke)
            # Only ONE converged endpoint here (the sparsest), so there is no
            # bracket to interpolate inside. Extrapolate the trend from the two
            # sparsest converged points instead, clamped below the lambda that
            # broke -- going past it is a run we already know degrades.
            conv = sorted(live, key=lambda c: c["l1_coef"])[-2:]
            # clamp=False: the fit runs through two points BELOW this bracket,
            # so clamping to their own interval would pin the probe under `lo`
            # -- the opposite of the intended extrapolation.
            probe = (_probe_lambda(conv[0], conv[1], clamp=False)
                     if len(conv) == 2 else (lo * hi) ** 0.5)
            probe = min(max(probe, lo * 1.05), hi * 0.95)
            best["_bracket"] = {"lo": lo, "hi": hi, "probe": probe}
            return best, (f"STABILITY-LIMITED — the sparsest CONVERGED lambda "
                          f"({lam_max_conv:g}) still sits at L0 "
                          f"{min(l0s):.0f}, and every larger lambda in the grid "
                          f"degraded (peaked at epoch < {min_ep}). Extending "
                          f"further would only add more degraded runs; probe "
                          f"between {lo:g} and {hi:g} instead.")
        return best, "NOT BRACKETED — extend the grid toward larger lambda (sparser)"
    return best, "NOT BRACKETED — extend the grid toward smaller lambda (denser)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe-dir", type=Path, default=PROBE_DIR)
    ap.add_argument("--arm", choices=("baseline", "dual"), default="baseline",
                    help="'baseline' = GatedSAE; 'dual' = Dual-Gated, whose "
                         "lambda is calibrated SEPARATELY (the same value can "
                         "leave the baseline dense and kill the dual)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = (DEFAULT_OUT if args.arm == "baseline" else
                    DEFAULT_OUT.with_name("dual_gated_lambda_choice.json"))

    baseline_lambda = None
    if args.arm == "dual":
        base_path = DEFAULT_OUT
        if not base_path.exists():
            sys.exit(f"missing {base_path}: the dual lambda bracket is built "
                     f"around the BASELINE winner, so calibrate that first.")
        raw = json.loads(base_path.read_text())
        baseline_lambda = {(k.split("/")[0], k.split("/")[1]): v["l1_coef"]
                           for k, v in raw["chosen"].items()}
        if not baseline_lambda:
            sys.exit(f"{base_path} resolves 0 cells; cannot build the dual "
                     f"lambda bracket.")
    probes = load_probes(args.probe_dir, args.arm, baseline_lambda)

    # The full set of (cell, lambda) the grid SHOULD contain -- the yardstick
    # for "partial" above. Built from the same grid function load_probes uses.
    _grid = (C.gated_lambda_grid() if args.arm == "baseline"
             else C.dual_gated_lambda_grid(baseline_lambda or {}))
    probes_expected = {(g["dataset"], g["encoder"], float(g["l1_coef"]))
                       for g in _grid}

    chosen, unresolved, table, brackets = {}, [], [], {}
    for ds in C.DATASETS:
        for enc in C.ENCODERS:
            cands = [probes[k] for k in probes if k[0] == ds and k[1] == enc]
            cands.sort(key=lambda c: c["l1_coef"])
            expected = [k[2] for k in probes_expected if k[0] == ds and k[1] == enc]
            missing = [l for l in expected if (ds, enc, float(l)) not in probes]
            cell = f"{ds}/{enc}"
            if not cands:
                unresolved.append(f"{cell}: no probes at all")
                continue
            if missing:
                # Selecting from a PARTIAL grid picks the best of whatever
                # finished first, which on a sweep correlates with how fast a
                # lambda converges -- not with which lambda is right. Refuse
                # and say what is outstanding.
                unresolved.append(
                    # Denominator is THIS cell's expected grid, not the
                    # baseline constant: the dual arm has 4 points per cell,
                    # and the baseline arm grows past 8 as refinements land.
                    # A wrong denominator here is how "4/4 done" reads as
                    # "4/8, keep waiting".
                    f"{cell}: {len(missing)} of {len(expected)} base-grid "
                    f"lambdas not yet measured "
                    f"({[f'{m:g}' for m in missing]}) "
                    f"— selecting now would pick the best of a partial grid")
                table.append({"cell": cell, "n_probed": len(cands),
                              "missing": missing,
                              "grid": [{"l1_coef": c["l1_coef"],
                                        "l0_mean": round(c["l0_mean"], 2),
                                        "dead_frac": round(c["dead_frac"], 4),
                                        "gate_open_frac": c.get("gate_open_frac")}
                                       for c in cands],
                              "winner": None, "reason": "partial grid"})
                continue
            winner, reason = pick(cands)
            table.append({
                "cell": cell, "n_probed": len(cands), "missing": missing,
                "grid": [{"l1_coef": c["l1_coef"], "l0_mean": round(c["l0_mean"], 2),
                          "dead_frac": round(c["dead_frac"], 4),
                          "best_epoch": c.get("best_epoch"),
                          "val_monoscore": c.get("val_monoscore"),
                          "gate_open_frac": (round(c["gate_open_frac"], 4)
                                             if c.get("gate_open_frac") is not None
                                             else None)}
                         for c in cands],
                "winner": (None if winner is None else
                           {"l1_coef": winner["l1_coef"],
                            "l0_mean": round(winner["l0_mean"], 2),
                            "dead_frac": round(winner["dead_frac"], 4),
                            "exp_name": winner["exp_name"]}),
                "reason": reason,
            })
            # startswith, not ==: the CONVERGED-L0 fallback returns
            # "in band (CONVERGED-L0 fallback: ...)" so that the provenance
            # travels with the decision. An equality check here silently
            # demoted a resolved cell to unresolved.
            if winner is not None and reason.startswith("in band"):
                chosen[cell] = {"l1_coef": winner["l1_coef"],
                                "l0_mean": round(winner["l0_mean"], 2),
                                "dead_frac": round(winner["dead_frac"], 4)}
            else:
                unresolved.append(f"{cell}: {reason}")
                if winner is not None and "_bracket" in winner:
                    # Covers both the straddling bracket and the
                    # stability-limited one; the bisect stage treats them the
                    # same way (geometric midpoint), which is right in both
                    # cases because lambda acts multiplicatively.
                    # NOT list(): the bracket is a dict {lo, hi, probe} and
                    # list() would yield its KEY NAMES.
                    brackets[cell] = dict(winner["_bracket"])

    hdr = f"{'cell':22} {'lambda':>8} {'L0':>9} {'dead':>7}  status"
    print(hdr); print("-" * (len(hdr) + 30))
    for row in table:
        for g in row["grid"]:
            mark = ("<--" if row["winner"] and
                    g["l1_coef"] == row["winner"]["l1_coef"] else "")
            inband = C.L0_BAND[0] <= g["l0_mean"] <= C.L0_BAND[1]
            ep = g.get("best_epoch")
            min_ep = max(1, int(MIN_BEST_EPOCH_FRAC * TOTAL_EPOCHS))
            if ep is not None and ep < min_ep:
                flag = f"early(ep{ep})"
            elif g["l0_mean"] < MIN_ALIVE_L0:
                flag = "COLLAPSED"
            else:
                flag = "in-band" if inband else ""
            print(f"{row['cell']:22} {g['l1_coef']:>8g} {g['l0_mean']:>9.1f} "
                  f"{g['dead_frac']:>7.3f}  {flag:13}{mark}")
        print(f"{'':22} {'':>8} {'':>9} {'':>7}  => {row['reason']}")
        if row["missing"]:
            print(f"{'':22} MISSING probes for lambda {row['missing']}")
        print()

    print(f"resolved {len(chosen)}/{len(C.DATASETS) * len(C.ENCODERS)} cells")
    for line in unresolved:
        print("  !", line)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "rule": "argmin |log(L0/64)| s.t. dead_frac <= 0.90; ties -> smaller lambda",
        "l0_target": C.L0_TARGET, "l0_band": list(C.L0_BAND),
        "max_dead_frac": C.MAX_DEAD_FRAC, "selection_seed": C.SWEEP_SEED,
        "min_best_epoch": max(1, int(MIN_BEST_EPOCH_FRAC * TOTAL_EPOCHS)),
        "probe_split": PROBE_SPLIT,
        "arm": args.arm, "min_alive_l0": MIN_ALIVE_L0,
        "chosen": chosen, "unresolved": unresolved,
        # (lambda_low, lambda_high) per cell that straddles 64 without an
        # in-band point -- the input to the bisect stage.
        "brackets": brackets, "table": table,
    }, indent=1))
    print(f"\nwrote {args.out}")
    # Non-zero while any cell is unresolved: stage_cells.py reads `chosen` and
    # would otherwise silently train a partial grid.
    return 0 if not unresolved else 1


if __name__ == "__main__":
    sys.exit(main())
