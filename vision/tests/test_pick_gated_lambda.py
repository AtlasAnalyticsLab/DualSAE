#!/usr/bin/env python3
"""Tests for the Gated lambda selection rule (pick_gated_lambda.pick).

Gated SAE has no k, so this rule is what decides where every Gated and
Dual-Gated row sits on the sparsity axis. It is pre-registered, so it needs to
be pinned: a silent change here would move the rows without moving any number
in the code that computes them.

CPU-only, no data. Run:  pytest tests/test_pick_gated_lambda.py -q  (from vision/)
"""

import sys
from pathlib import Path

import pytest

SAE_DIR = Path(__file__).resolve().parents[1]
for p in (str(SAE_DIR), str(SAE_DIR / "scripts" / "main")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pick_gated_lambda import (MIN_ALIVE_L0, MIN_BEST_EPOCH_FRAC,  # noqa: E402
                               TOTAL_EPOCHS, pick)
import grid_config as C  # noqa: E402


def cand(lam, l0, dead=0.1, epoch=None):
    """epoch=None means "no epoch recorded", which must behave exactly as it
    did before the convergence guard existed (backward compatibility for
    l0_probe outputs that predate it)."""
    return {"l1_coef": lam, "l0_mean": l0, "dead_frac": dead,
            "best_epoch": epoch}


def test_picks_nearest_in_log_space_not_absolute():
    """L0 is multiplicative: 32 and 128 are equidistant from 64. An absolute
    distance rule would prefer 32 (|32-64|=32 < |128-64|=64) and silently bias
    every Gated row sparse."""
    lo, hi = cand(0.3, 32.0), cand(0.1, 128.0)
    winner, _ = pick([lo, hi])
    # Exactly equidistant in log space -> the tie-break takes the smaller
    # lambda, which is the denser (higher-R2) model.
    assert winner["l1_coef"] == 0.1
    # And a genuinely closer point wins regardless of absolute distance.
    near = cand(0.2, 70.0)
    winner2, reason = pick([lo, hi, near])
    assert winner2 is near and reason == "in band"


def test_rejects_the_dead_dictionary_solution():
    """L0 == 64 reached only because almost every latent is dead is not the
    same object as a healthy 8192-latent model at L0 64."""
    dead_exact = cand(1.0, 64.0, dead=0.97)
    alive_near = cand(0.1, 80.0, dead=0.10)
    winner, reason = pick([dead_exact, alive_near])
    assert winner is alive_near, "a 97%-dead model must not win on L0 alone"
    assert reason == "in band"


def test_collapse_is_rejected_outright_with_a_named_reason():
    """A branch that has stopped firing (L0 ~ 0) is an absent model, not a
    sparse one, and the rule must say so rather than score it badly."""
    winner, reason = pick([cand(0.1, 0.0, dead=1.0), cand(0.3, 0.05, dead=1.0)])
    assert winner is None
    assert "COLLAPSED" in reason
    assert str(MIN_ALIVE_L0) in reason or "reopen" in reason


def test_reports_bracketed_vs_not_bracketed():
    """The two failure modes need different fixes -- bisect vs extend the grid
    -- so they must be distinguishable from the message alone."""
    _, r_bracket = pick([cand(0.03, 200.0), cand(1.0, 10.0)])
    assert "BRACKET" in r_bracket
    _, r_high = pick([cand(0.001, 900.0), cand(0.01, 300.0)])
    assert "NOT BRACKETED" in r_high and "larger lambda" in r_high
    _, r_low = pick([cand(1.0, 20.0), cand(3.0, 5.0)])
    assert "NOT BRACKETED" in r_low and "smaller lambda" in r_low


def test_in_band_definition_matches_the_config():
    """The band is x1.33 either side of 64; pin it so a config edit that
    widens it has to be deliberate."""
    lo, hi = C.L0_BAND
    assert lo < C.L0_TARGET < hi
    assert pick([cand(0.1, lo + 0.1)])[1] == "in band"
    assert pick([cand(0.1, hi + 5.0)])[1] != "in band"


MIN_EP = max(1, int(MIN_BEST_EPOCH_FRAC * TOTAL_EPOCHS))


def test_early_peaking_lambdas_are_excluded():
    """A run whose best-val-MonoScore epoch is epoch 1 of 50 DEGRADED; the
    trainer just kept the least-bad checkpoint. Its L0 describes an
    undertrained model, not that lambda (and makes L0-vs-lambda look
    non-monotone)."""
    early_perfect = cand(0.3, 64.0, epoch=1)      # bang on target, but ep 1
    converged_ok = cand(0.1, 70.0, epoch=TOTAL_EPOCHS)
    winner, reason = pick([early_perfect, converged_ok])
    assert winner is converged_ok, "an epoch-1 checkpoint must not win on L0"
    assert reason == "in band"


def test_all_early_is_reported_as_degradation_not_a_bracket():
    """If nothing converged, do not report a bracket over undertrained points.

    When every lambda peaked at epoch 1, "extend toward smaller lambda" can be
    the wrong direction (their L0s describe undertrained models). Direction
    advice inferred from untrustworthy L0s is itself untrustworthy, so the
    message points at the measurement that settles it."""
    winner, reason = pick([cand(1.0, 300.0, epoch=1), cand(3.0, 20.0, epoch=2)])
    assert winner is None
    assert "degraded" in reason
    assert "l0_trajectory.py" in reason


def test_missing_epoch_is_treated_as_converged():
    """Backward compatibility: measurements without a recorded epoch must not
    be silently discarded."""
    c = cand(0.1, 66.0, epoch=None)
    winner, reason = pick([c])
    assert winner is c and reason == "in band"


def test_stability_limit_is_distinguished_from_a_short_grid():
    """Two ways the converged runs can stop ABOVE the target, needing OPPOSITE
    actions:

      * the grid does not reach far enough        -> extend it;
      * larger lambdas exist but all DEGRADED     -> the limit is stability,
        and extending would only add more degraded runs.

    E.g. a cell that converges only up to some lambda while all larger lambdas
    peak at epoch 1: advising "extend toward larger lambda" there would spend
    compute on runs certain to destabilise.
    """
    cands = [cand(0.03, 515.0, epoch=14), cand(0.1, 114.9, epoch=50),
             cand(0.3, 765.0, epoch=1), cand(1.0, 588.0, epoch=1)]
    winner, reason = pick(cands)
    assert "STABILITY-LIMITED" in reason
    assert "0.1" in reason and "0.3" in reason, "must name the probe interval"
    br = winner["_bracket"]
    assert (br["lo"], br["hi"]) == (0.1, 0.3)
    # The probe must land INSIDE the stability bracket. It is extrapolated from
    # the two sparsest CONVERGED points, which both lie below `lo`, so a clamp
    # to their own interval would pin it under 0.1 -- the opposite of intended.
    assert br["lo"] < br["probe"] < br["hi"], br


def test_short_grid_still_says_extend():
    """When no larger lambda was tried at all, extending IS the right advice."""
    cands = [cand(0.001, 900.0, epoch=50), cand(0.01, 300.0, epoch=50)]
    winner, reason = pick(cands)
    assert "NOT BRACKETED" in reason and "larger lambda" in reason
    assert "_bracket" not in winner


def test_too_sparse_everywhere_says_go_denser():
    cands = [cand(1.0, 20.0, epoch=50), cand(3.0, 5.0, epoch=50)]
    _, reason = pick(cands)
    assert "NOT BRACKETED" in reason and "smaller lambda" in reason


def test_probe_uses_log_log_interpolation_not_the_midpoint():
    """Each refinement round costs a full 50-epoch run, so the probe is placed
    by fitting (log lambda, log L0) and solving for the target rather than
    halving the bracket: with one endpoint an order of magnitude from the
    target and the other close, the root is far from the geometric midpoint.
    """
    import math
    cands = [cand(0.03, 601.6, epoch=17), cand(0.1, 36.7, epoch=50)]
    winner, reason = pick(cands)
    br = winner["_bracket"]
    midpoint = math.sqrt(0.03 * 0.1)
    assert br["probe"] > midpoint * 1.2, (br["probe"], midpoint)
    assert br["lo"] < br["probe"] < br["hi"]


def test_probe_stays_inside_its_bracket_on_a_flat_segment():
    """A near-flat segment makes the interpolation ill-conditioned; the result
    must still be an interior refinement, not an extrapolation."""
    cands = [cand(0.03, 65.0, epoch=50), cand(0.1, 64.5, epoch=50)]
    winner, _ = pick(cands)
    if "_bracket" in winner:
        br = winner["_bracket"]
        assert br["lo"] < br["probe"] < br["hi"]


# --------------------------------------------------------------------------
# extra_probe_cells: the scan that makes refinement/ceiling probes visible.
# --------------------------------------------------------------------------
def _mk(tmp_path, *names):
    for n in names:
        (tmp_path / n).mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_dual_scan_picks_up_the_lambda0_ceiling_probe(tmp_path, monkeypatch):
    """The lambda=0 probe is not in the 4-point grid; without the scan it is
    trained and then never read -- an invisible stall."""
    import pick_gated_lambda as P
    _mk(tmp_path,
        "t3x_imagenet_clip_dual_gated_n8192k64_l10_span0.01_warm17500_seed42")
    monkeypatch.setattr(P.C, "OUTPUT_DIR", str(tmp_path))
    got = P.extra_probe_cells("dual", seen=set())
    assert [(c["dataset"], c["encoder"], c["l1_coef"]) for c in got] == [
        ("imagenet", "clip", 0.0)]


def test_dual_scan_rejects_sweep_cells_off_the_grid_centre(tmp_path, monkeypatch):
    """The load-bearing guard: the 65-cell Dual-Gated sweep varies span and
    warm-up at the CHOSEN lambda. Admitting one of those points would
    calibrate lambda against a span effect."""
    import pick_gated_lambda as P
    centre = P.C.DUAL_WARMUP_CENTRE["imagenet"]
    _mk(tmp_path,
        # wrong span (a sweep cell)
        f"t3x_imagenet_clip_dual_gated_n8192k64_l10.05_span0.5_warm{centre}_seed42",
        # wrong warm-up (a sweep cell)
        "t3x_imagenet_clip_dual_gated_n8192k64_l10.05_span0.01_warm900_seed42",
        # wrong seed (a confirm run, never informs selection)
        f"t3x_imagenet_clip_dual_gated_n8192k64_l10.05_span0.01_warm{centre}_seed43",
        # the plain Gated arm must not leak into the dual scan
        "t3x_imagenet_clip_gated_n8192k64_l10.05_seed42")
    monkeypatch.setattr(P.C, "OUTPUT_DIR", str(tmp_path))
    assert P.extra_probe_cells("dual", seen=set()) == []


def test_scan_skips_names_already_in_the_base_grid(tmp_path, monkeypatch):
    """A refinement is a bonus measurement, never a duplicate of a grid point."""
    import pick_gated_lambda as P
    n = "t3x_imagenet_clip_dual_gated_n8192k64_l10_span0.01_warm17500_seed42"
    _mk(tmp_path, n)
    monkeypatch.setattr(P.C, "OUTPUT_DIR", str(tmp_path))
    assert P.extra_probe_cells("dual", seen={n}) == []


def test_baseline_scan_is_unaffected_by_dual_runs(tmp_path, monkeypatch):
    """Both arms scan the same tree; 'gated' must not match 'dual_gated'."""
    import pick_gated_lambda as P
    _mk(tmp_path,
        "t3x_imagenet_clip_gated_n8192k64_l10.17_seed42",
        "t3x_imagenet_clip_dual_gated_n8192k64_l10_span0.01_warm17500_seed42")
    monkeypatch.setattr(P.C, "OUTPUT_DIR", str(tmp_path))
    got = P.extra_probe_cells("baseline", seen=set())
    assert [c["exp_name"] for c in got] == [
        "t3x_imagenet_clip_gated_n8192k64_l10.17_seed42"]


# --------------------------------------------------------------------------
# Converged-L0 fallback: used only when checkpoint selection is untrustworthy
# for an ENTIRE cell. It decides a reported lambda, so its edges are pinned.
# --------------------------------------------------------------------------
def _early(lam, l0_best, l0_final, dead=0.0):
    """A run whose selected checkpoint is epoch 1 (so the filter rejects it)."""
    c = cand(lam, l0_best, dead=dead, epoch=1)
    c["l0_final"] = l0_final
    return c


def test_fallback_selects_on_the_converged_l0_when_all_runs_peak_early():
    """Every lambda selects epoch 1 with a huge L0, but one lambda converges
    into the band."""
    winner, reason = pick([_early(0.01, 3716.0, 1410.8),
                           _early(0.0333, 3232.0, 63.3),
                           _early(0.1, 2325.0, 7.9),
                           _early(0.3, 1088.0, 2.7)])
    assert winner is not None
    assert winner["l1_coef"] == 0.0333
    assert winner["l0_mean"] == pytest.approx(63.3)
    assert "CONVERGED-L0 fallback" in reason
    # It must be visibly marked, never silently substituted.
    assert winner.get("_l0_from") == "trajectory"


def test_fallback_refuses_when_its_best_converged_point_is_out_of_band():
    """Falling back is not the same as lowering the bar: if no lambda
    converges near 64 the cell still needs a wider grid."""
    winner, reason = pick([_early(0.01, 3000.0, 900.0),
                           _early(0.1, 2000.0, 400.0)])
    assert winner is None
    assert "outside the band" in reason


def test_fallback_does_not_fire_when_any_run_converged():
    """A single trustworthy point is enough; the normal path must win, so a
    working cell is never rerouted through the fallback."""
    good = cand(0.1, 60.0, epoch=40)
    good["l0_final"] = 999.0            # would be picked if the fallback ran
    winner, reason = pick([good, _early(1.0, 5.0, 3.0)])
    assert winner is good
    assert "fallback" not in reason
    assert winner["l0_mean"] == pytest.approx(60.0)


def test_fallback_still_rejects_a_dead_branch():
    """MIN_ALIVE_L0 and the dead_frac ceiling apply to the fallback too --
    otherwise it would resurrect exactly the collapses it must report."""
    winner, reason = pick([_early(0.01, 3000.0, 0.2),
                           _early(0.1, 2000.0, 0.05)])
    assert winner is None


def test_fallback_is_unavailable_without_trajectory_data():
    """No trajectory file -> say so and name the script, rather than refusing
    with a reason that points at the wrong fix."""
    winner, reason = pick([cand(0.01, 3000.0, epoch=1),
                           cand(0.1, 2000.0, epoch=1)])
    assert winner is None
    assert "l0_trajectory.py" in reason


def test_fallback_winner_is_recorded_as_resolved_not_unresolved():
    """The fallback appends provenance to the reason, so a cell decided this
    way reads 'in band (CONVERGED-L0 fallback: ...)'. main() classifies on
    that string; an equality check silently demoted a resolved cell back to
    unresolved although a winner was in hand."""
    winner, reason = pick([_early(0.0333, 3232.0, 63.3),
                           _early(0.1, 2325.0, 7.9)])
    assert winner is not None
    assert reason.startswith("in band")
    assert reason != "in band", "provenance must survive in the reason"
