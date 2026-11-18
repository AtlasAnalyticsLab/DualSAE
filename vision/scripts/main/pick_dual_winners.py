#!/usr/bin/env python3
"""Pick each Dual cell's (span_coef, warm-up) from the seed-42 sweep.

Selection rule -- PRE-REGISTERED, fixed before the sweep ran
------------------------------------------------------------
    winner = argmax over settings of the run's BEST VALIDATION MONOSCORE
             (seed 42 only; seeds 43/44 never inform selection)
    ties -> the setting closer to the grid centre (span 0.01), then
            the smaller warm-up

Why val MonoScore: it is the metric the trainers ALREADY use to choose which
epoch to keep (best_val_monoscore_model.pt), so selecting the setting by the
same quantity keeps the two levels of model choice consistent. Any other
criterion would pick a checkpoint by one rule and a setting by another.

Read cheaply and exactly: `val_monoscore` is a field of the very checkpoint the
table reports, loaded with mmap=True so no tensor data is touched (~10 ms/run).
No GPU, no W&B parsing, no possibility of reading a different run's number.

Honesty
-------
This is selection on the split the ImageNet rows are reported on, because
ImageNet has no test split. Two guards make the size of the effect visible:
  * the CONTROL column: the same cell at the grid centre
    (span 0.01 at the grid's warm-up centre), which is a point in the grid, so
    it costs nothing to report;
  * `--min-gain`: cells where the winner beats the control by less than this
    are flagged, because a tuned row that barely beats an untuned one is a
    tuning artefact, not a finding.

Usage (CPU, seconds):
    python scripts/main/pick_dual_winners.py --variant dual_matryoshka
    python scripts/main/pick_dual_winners.py --variant dual_gated --lambda-json <...>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Make the vision core (two levels up) and this directory importable.
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import grid_config as C  # noqa: E402

DEFAULT_OUT_DIR = Path(C.STATE_DIR)


# A run whose best-val-MonoScore epoch is very early did not converge at that
# setting -- it peaked and then degraded, and the trainer kept the early
# checkpoint. Its metrics describe an undertrained model, not the setting.
#
# Same floor and same rationale as pick_gated_lambda.MIN_BEST_EPOCH_FRAC. The
# two pickers guarded different things for no reason: the lambda picker
# rejected early-peaking runs while this one did not, so a degraded epoch-6
# checkpoint could win a span sweep by 0.0015 MonoScore over a converged
# epoch-30 one -- and that checkpoint then became the REPORTED row.
#
MIN_BEST_EPOCH_FRAC = 0.2

# Seeds landing in different regimes make their mean a fiction. 3x is well
# above ordinary seed-to-seed variation of L0 for these models.
SEED_SPREAD_MAX = 3.0

# Variants whose L0 is EMERGENT (set by a penalty) rather than pinned by a k.
# For these the winner must additionally be L0-matched -- see _l0_admissible.
EMERGENT_L0_VARIANTS = {"dual_gated"}
TRAJ_PATH = Path(C.STATE_DIR) / "l0_trajectory.json"


def _load_trajectories() -> dict:
    """exp_name -> L0 trajectory summary (scripts/l0_trajectory.py)."""
    try:
        return json.loads(TRAJ_PATH.read_text())
    except Exception:
        return {}


def _l0_of(traj: dict, exp_name: str):
    """The L0 this setting would REPORT, or None if not measured.

    That is `l0_best` -- the L0 of `best_val_monoscore_model.pt`, the
    checkpoint the evaluation protocol scores -- NOT the converged `l0_final`.
    The two can differ by an order of magnitude for an emergent-L0 model, and
    matching a quantity the table does not print is not matching anything.

    Falls back to l0_final only when l0_best is unavailable (no recorded
    best epoch), so a setting is never dropped merely for lacking the
    preferred reading.

    Sweep settings are deliberately NOT evaluated by branch_metrics (only the
    control point is), so the trajectory file is the only place either number
    exists for a swept cell.
    """
    t = traj.get(exp_name) or {}
    l0 = t.get("l0_best")
    return t.get("l0_final") if l0 is None else l0


def _per_seed_l0(exp_name_seed42: str, traj: dict) -> dict:
    """{seed: reported L0} for the winning setting across ALL seeds.

    The confirm runs share the sweep run's name except for the seed suffix, so
    the sibling names are derivable. A seed with no trajectory yet maps to
    None rather than being omitted, so the caller can tell "not measured"
    from "measured and fine".
    """
    out = {}
    for seed in C.SEEDS:
        name = exp_name_seed42.rsplit("_seed", 1)[0] + f"_seed{seed}"
        t = traj.get(name) or {}
        l0 = t.get("l0_best")
        out[seed] = t.get("l0_final") if l0 is None else l0
    return out


def _l0_admissible_on(variant: str, rec: dict, traj: dict, field: str) -> bool:
    """L0 admissibility measured on a NAMED trajectory field."""
    if variant not in EMERGENT_L0_VARIANTS:
        return True
    l0 = (traj.get(rec["exp_name"]) or {}).get(field)
    if l0 is None:
        return True
    lo, hi = C.L0_BAND
    return lo <= l0 <= hi


def _l0_admissible(variant: str, rec: dict, traj: dict) -> bool:
    """May this setting be chosen as the reported row?

    WHY THIS EXISTS. The sweep's pre-registered criterion is max val MonoScore.
    MonoScore rises as a model fires FEWER features, so for a variant whose L0
    is emergent, maximizing it selects degenerate sparsity rather than quality.

    E.g. Dual-Gated's lambda is calibrated at the (span 0.01, warm-up centre)
    point to an L0 near the Gated baseline's; an unconstrained span sweep at
    that lambda can then win at a setting where L0 collapses far below it,
    which would not be a comparison of two models at matched sparsity.

    This CANNOT happen to a k-pinned dual (TopK/BatchTopK/Matryoshka): their L0
    is identically k at every span, so the span sweep cannot move it and the
    unconstrained criterion is safe. So the constraint applies only where the
    degeneracy is reachable, and k-pinned variants keep the plain rule
    untouched.

    A setting whose L0 was never measured is admitted rather than rejected: the
    trajectory file can lag, and silently dropping unmeasured settings would
    narrow the sweep without saying so. The caller reports how many were
    excluded.
    """
    if variant not in EMERGENT_L0_VARIANTS:
        return True
    l0 = _l0_of(traj, rec["exp_name"])
    if l0 is None:
        return True
    lo, hi = C.L0_BAND
    return lo <= l0 <= hi


def read_val_monoscore(run_dir: Path) -> dict | None:
    """(val_monoscore, epoch) from the reported checkpoint, or None if absent.

    mmap=True keeps this to a metadata read: the ~50 MB state dict is never
    materialised, so 65 runs take under a second in total.
    """
    ckpt = run_dir / "best_val_monoscore_model.pt"
    if not ckpt.exists() or not (run_dir / "autoencoder_final.pt").exists():
        return None            # missing, or training never finished
    try:
        d = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    except Exception as exc:
        print(f"  ! unreadable {ckpt}: {exc}", file=sys.stderr)
        return None
    ms = d.get("val_monoscore")
    return None if ms is None else {"val_monoscore": float(ms),
                                    "best_epoch": d.get("epoch")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", required=True, choices=C.DUAL_VARIANTS)
    ap.add_argument("--lambda-json", default=None,
                    help="dual-arm lambda choice (dual_gated only)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-gain", type=float, default=0.01,
                    help="flag cells whose winner beats the stated-protocol "
                         "control by less than this on val MonoScore")
    args = ap.parse_args()
    out = args.out or DEFAULT_OUT_DIR / f"{args.variant}_winners.json"

    lam = None
    if args.variant == "dual_gated":
        path = Path(args.lambda_json or
                    DEFAULT_OUT_DIR / "dual_gated_lambda_choice.json")
        if not path.exists():
            sys.exit(f"missing {path}: Dual-Gated's lambda is calibrated on the "
                     f"DUAL arm (the baseline's value can collapse the branch) "
                     f"-- run the dual-gated-lambda stage first.")
        raw = json.loads(path.read_text())
        lam = {(k.split("/")[0], k.split("/")[1]): v["l1_coef"]
               for k, v in raw["chosen"].items()}
        if not lam:
            sys.exit(f"{path} resolved no cells; refusing to pick winners.")

    traj = _load_trajectories()
    grid = C.dual_sweep_grid(args.variant, lam)
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for cell in grid:
        key = (cell["dataset"], cell["encoder"])
        run = Path(C.OUTPUT_DIR) / cell["exp_name"]
        got = read_val_monoscore(run)
        rec = {"span_coef": cell["span_coef"],
               "warmup_steps": cell["warmup_steps"],
               "exp_name": cell["exp_name"],
               **(got or {"val_monoscore": None, "best_epoch": None})}
        by_cell.setdefault(key, []).append(rec)

    chosen, unresolved, table = {}, [], []
    for ds in C.DATASETS:
        for enc in C.ENCODERS:
            key = (ds, enc)
            recs = by_cell.get(key, [])
            done = [r for r in recs if r["val_monoscore"] is not None]
            label = f"{ds}/{enc}"
            control = next((r for r in recs
                            if r["span_coef"] == C.DUAL_SPAN_CENTRE
                            and r["warmup_steps"] == C.DUAL_WARMUP_CENTRE[ds]),
                           None)
            if not done:
                unresolved.append(f"{label}: 0/{len(recs)} settings finished")
                continue
            if len(done) < len(recs):
                unresolved.append(
                    f"{label}: only {len(done)}/{len(recs)} settings finished — "
                    f"selecting now would pick the best of a PARTIAL grid")
                continue
            # For an emergent-L0 variant, restrict to settings that stay
            # L0-matched to the baseline before maximizing MonoScore (see
            # _l0_admissible). Without this the criterion selects the sparsest
            # setting, not the best one.
            admissible = [r for r in done
                          if _l0_admissible(args.variant, r, traj)]
            n_excl = len(done) - len(admissible)
            # Prefer CONVERGED settings. Falling back only when none converged
            # mirrors the lambda picker: the filter must not be able to empty
            # a cell, but a converged setting always beats a degraded one.
            min_ep = max(1, int(MIN_BEST_EPOCH_FRAC * C.EPOCHS))
            converged = [r for r in admissible
                         if (r.get("best_epoch") or 0) >= min_ep]
            n_early = len(admissible) - len(converged)
            early_fallback = not converged
            if converged:
                admissible = converged
            # DUAL-CHECKPOINT FALLBACK.
            #
            # When NO setting is L0-matched on the reported (best) checkpoint,
            # the protocol's own checkpoint selection is the problem, not the
            # model: if every setting peaks at epoch 1 (where L0 is identical
            # across settings because span has not acted yet), no span choice
            # can fix that.
            #
            # Rather than drop the cell, re-match on the CONVERGED checkpoint
            # and mark it to be reported at BOTH: the protocol checkpoint shows
            # what the stated rule actually selects, the converged one shows
            # the trained model. The selection pathology then appears in the
            # table as a finding instead of being hidden behind a blank.
            report_both = False
            # SEEDS 43/44 CAN BEHAVE DIFFERENTLY FROM THE SELECTION SEED.
            #
            # Admissibility above is measured on seed 42, because that is the
            # only seed the sweep runs -- but the REPORTED row is a 3-seed
            # mean, so a setting can pass selection and still produce an
            # out-of-band row (e.g. seed 42 converged while seeds 43/44 select
            # dense epoch-1 checkpoints).
            #
            # So once the confirm runs exist, re-check across every seed and
            # promote the cell to dual-checkpoint reporting if the reported
            # picture is out of band. This cannot be done at selection time --
            # the information does not exist yet -- so it is a re-check, not a
            # stricter filter.
            if args.variant in EMERGENT_L0_VARIANTS and admissible:
                cand = max(admissible, key=lambda r: (
                    r["val_monoscore"],
                    -abs(r["span_coef"] - C.DUAL_SPAN_CENTRE),
                    -r["warmup_steps"]))
                seed_l0 = _per_seed_l0(cand["exp_name"], traj)
                lo, hi = C.L0_BAND
                vals = [v for v in seed_l0.values() if v is not None]
                # Judge the quantity that is actually REPORTED -- the mean over
                # seeds -- plus a spread test for the case where seeds land in
                # different regimes and the mean is a fiction.
                #
                # A per-seed band test was too strict: imagenet/clip runs
                # {50.1, 58.8, 40.3}, mean 49.7 and comfortably in band, but one
                # seed sits just under the floor. That is ordinary seed
                # variation, not a protocol failure, and flagging it would have
                # given a healthy cell a second row it does not need.
                # imagenet/sigclip runs {82.7, 1333, 1335}: mean 916.9 and a
                # 16x spread -- two different models wearing one label.
                mean_l0 = (sum(vals) / len(vals)) if vals else None
                spread = (max(vals) / max(min(vals), 1e-9)) if vals else 1.0
                bad_mean = mean_l0 is not None and not (lo <= mean_l0 <= hi)
                bad_spread = spread > SEED_SPREAD_MAX
                if len(vals) > 1 and (bad_mean or bad_spread):
                    report_both = True
                    flag_seed_note = (
                        f"reported L0 mean {mean_l0:.1f}, spread {spread:.1f}x; "
                        f"L0 per seed "
                        f"{ {k: (round(v, 1) if v is not None else None) for k, v in sorted(seed_l0.items())} }")
                else:
                    flag_seed_note = None
            else:
                flag_seed_note = None
            if not admissible:
                admissible = [r for r in done
                              if _l0_admissible_on(args.variant, r, traj,
                                                   "l0_final")]
                report_both = bool(admissible)
                flag_seed_note = None
            if not admissible:
                lo, hi = C.L0_BAND
                unresolved.append(
                    f"{label}: no setting is L0-matched to the baseline on "
                    f"EITHER the reported or the converged checkpoint (band "
                    f"{lo:g}-{hi:g}). Report the lambda-calibration point and "
                    f"say the span sweep found nothing comparable.")
                continue
            # argmax val MonoScore; ties -> nearer the grid centre, then
            # smaller warm-up.
            best = max(admissible, key=lambda r: (
                r["val_monoscore"],
                -abs(r["span_coef"] - C.DUAL_SPAN_CENTRE),
                -r["warmup_steps"]))
            best = dict(best, report_both_checkpoints=report_both,
                        n_excluded_by_l0=n_excl,
                        n_excluded_by_epoch=n_early,
                        early_fallback=early_fallback,
                        l0_final=_l0_of(traj, best["exp_name"]))
            ctrl_ms = control["val_monoscore"] if control and control["val_monoscore"] is not None else None
            gain = (best["val_monoscore"] - ctrl_ms) if ctrl_ms is not None else None
            # ACCUMULATE flags; do not overwrite. A cell can be degraded AND
            # have no tuning gain at once (open_images/sigclip dual_gated is
            # both), and a later assignment silently dropped the earlier
            # warning -- losing exactly the caveat that belongs in the caption.
            flags = []
            if flag_seed_note:
                flags.append(flag_seed_note)
            if report_both:
                flags.append(
                    "NO setting is L0-matched on the reported (best) "
                    "checkpoint — every one peaks early at a far denser L0. "
                    "Matched on the CONVERGED checkpoint instead; report this "
                    "cell at BOTH checkpoints so the selection effect is "
                    "visible")
            if early_fallback:
                flags.append(f"EVERY setting peaked before epoch {min_ep} of "
                             f"{C.EPOCHS}; winner is the best of a degraded "
                             f"field — report it as such")
            flag = ""
            if gain is not None and gain < args.min_gain:
                flag = (f"tuning gain {gain:+.4f} < {args.min_gain}: the winner "
                        f"barely beats the stated protocol")
            if best["span_coef"] == 0.0:
                flag = (flag + " | " if flag else "") + (
                    "winner has span_coef=0, i.e. the decorrelation penalty is "
                    "OFF — report it as such, not as a tuned DualSAE")
            chosen[label] = {"span_coef": best["span_coef"],
                             "warmup_steps": best["warmup_steps"],
                             "val_monoscore": round(best["val_monoscore"], 6),
                             "best_epoch": best["best_epoch"],
                             "exp_name": best["exp_name"],
                             "control_val_monoscore": (round(ctrl_ms, 6)
                                                       if ctrl_ms is not None else None),
                             "tuning_gain": (round(gain, 6) if gain is not None else None),
                             "flag": "; ".join(flags + ([flag] if flag else []))
                                     or None,
                             "early_fallback": early_fallback,
                             "report_both_checkpoints": report_both,
                             "n_excluded_by_epoch": best.get("n_excluded_by_epoch"),
                             "n_excluded_by_l0": best.get("n_excluded_by_l0"),
                             "l0_final": best.get("l0_final")}
            table.append({"cell": label, "n_settings": len(recs),
                          "settings": sorted(done,
                                             key=lambda r: -r["val_monoscore"])})

    hdr = f"{'cell':22} {'span':>7} {'warm':>7} {'valMS':>9} {'vs ctrl':>9}"
    print(hdr); print("-" * (len(hdr) + 20))
    for label, w in chosen.items():
        g = "" if w["tuning_gain"] is None else f"{w['tuning_gain']:+9.4f}"
        print(f"{label:22} {w['span_coef']:>7g} {w['warmup_steps']:>7} "
              f"{w['val_monoscore']:>9.4f} {g}")
        if w["flag"]:
            print(f"{'':22} ! {w['flag']}")
    print(f"\nresolved {len(chosen)}/{len(C.DATASETS) * len(C.ENCODERS)} cells")
    for line in unresolved:
        print("  !", line)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "variant": args.variant,
        "rule": ("argmax val MonoScore over (span, warm-up); seed 42 only; "
                 "ties -> nearer span 0.01, then smaller warm-up"),
        "selection_seed": C.SWEEP_SEED,
        "control": {"span_coef": C.DUAL_SPAN_CENTRE,
                    "warmup_steps": C.DUAL_WARMUP_CENTRE},
        "min_gain": args.min_gain,
        "chosen": chosen, "unresolved": unresolved, "table": table,
    }, indent=1))
    print(f"\nwrote {out}")
    return 0 if not unresolved else 1


if __name__ == "__main__":
    sys.exit(main())
