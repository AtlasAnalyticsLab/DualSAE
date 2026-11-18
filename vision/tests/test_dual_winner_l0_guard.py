"""The L0-band constraint and convergence guard of the Dual winner selection
(scripts/main/pick_dual_winners.py).

The sweep's criterion is max val MonoScore. MonoScore rises as a model fires
FEWER features, so for a variant whose L0 is emergent (Dual-Gated) that criterion
could select degenerate sparsity rather than quality; the winner must therefore
lie in the same L0 band the lambda calibration targets. A k-pinned dual (L0 == k
at every setting) is never constrained. All values below are synthetic.
"""
import sys
from pathlib import Path

VISION = Path(__file__).resolve().parents[1]
for p in (str(VISION), str(VISION / "scripts" / "main"), str(VISION / "analysis" / "main")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pick_dual_winners as W  # noqa: E402
import grid_config as C  # noqa: E402
import main_analysis_common  # noqa: E402,F401  (import check of the analysis helpers)


def _rec(name, span=0.01, warm=17500):
    return {"exp_name": name, "span_coef": span, "warmup_steps": warm}


TRAJ = {"in_band": {"l0_final": 60.0},
        "collapsed": {"l0_final": 4.0},
        "too_dense": {"l0_final": 900.0}}


def test_emergent_l0_variant_rejects_a_collapsed_setting():
    assert not W._l0_admissible("dual_gated", _rec("collapsed"), TRAJ)


def test_emergent_l0_variant_rejects_an_over_dense_setting():
    """The band is two-sided."""
    assert not W._l0_admissible("dual_gated", _rec("too_dense"), TRAJ)


def test_emergent_l0_variant_accepts_an_in_band_setting():
    assert W._l0_admissible("dual_gated", _rec("in_band"), TRAJ)


def test_k_pinned_variant_is_never_constrained():
    for name in TRAJ:
        assert W._l0_admissible("dual_matryoshka", _rec(name), TRAJ)


def test_unmeasured_setting_is_admitted_not_silently_dropped():
    """The trajectory file can lag; dropping unmeasured settings would narrow
    the sweep without saying so."""
    assert W._l0_admissible("dual_gated", _rec("no_such_run"), TRAJ)


def test_band_matches_the_lambda_calibration_band():
    lo, hi = C.L0_BAND
    assert lo < C.L0_TARGET < hi
    assert W._l0_admissible("dual_gated", _rec("x"), {"x": {"l0_final": lo}})
    assert W._l0_admissible("dual_gated", _rec("x"), {"x": {"l0_final": hi}})
    assert not W._l0_admissible("dual_gated", _rec("x"), {"x": {"l0_final": lo - 0.1}})


def _s(span, warm, ms, ep, name=None):
    return {"span_coef": span, "warmup_steps": warm, "val_monoscore": ms,
            "best_epoch": ep, "exp_name": name or f"run_s{span}_w{warm}"}


def test_min_best_epoch_frac_matches_the_lambda_picker():
    import pick_gated_lambda as P
    assert W.MIN_BEST_EPOCH_FRAC == P.MIN_BEST_EPOCH_FRAC


def test_a_converged_setting_beats_a_higher_scoring_early_one():
    """A setting whose best epoch is very early did not converge; it must not win
    over a converged setting with a slightly lower validation MonoScore."""
    min_ep = max(1, int(W.MIN_BEST_EPOCH_FRAC * C.EPOCHS))
    early = _s(0.01, 2500, 0.351, 3)
    conv = _s(0.1, 17500, 0.350, 30)
    converged = [r for r in (early, conv) if (r.get("best_epoch") or 0) >= min_ep]
    assert converged == [conv]
    assert max(converged, key=lambda r: r["val_monoscore"]) is conv


def test_fallback_when_every_setting_is_early():
    """The filter never empties a cell -- the caller falls back to the degraded
    field and flags it."""
    min_ep = max(1, int(W.MIN_BEST_EPOCH_FRAC * C.EPOCHS))
    pool = [_s(0.01, 500, 0.47, 1), _s(0.03, 500, 0.46, 2)]
    assert not [r for r in pool if (r.get("best_epoch") or 0) >= min_ep]
    assert max(pool, key=lambda r: r["val_monoscore"])["best_epoch"] == 1


def test_result_path_suffixes_only_non_default_checkpoints():
    import main_analysis_common as A
    r = Path("/results")
    assert A.result_path(r, "/ck/run_a", "val") == r / "run_a__val.json"
    assert A.result_path(r, "/ck/run_a", "val", "best") == r / "run_a__val.json"
    assert A.result_path(r, "/ck/run_a", "val", "final") == r / "run_a__val__final.json"


def _spread(vals):
    return max(vals) / max(min(vals), 1e-9)


def test_seed_spread_threshold_separates_normal_and_split_regimes():
    """Seeds in different regimes (one converged, two dense) make their mean a
    fiction; ordinary seed variation must not be flagged."""
    normal, split = [50.0, 58.0, 41.0], [80.0, 1300.0, 1310.0]
    assert _spread(normal) < W.SEED_SPREAD_MAX < _spread(split)


def test_per_seed_l0_reports_unmeasured_seeds_as_none():
    traj = {"run_seed42": {"l0_best": 60.0}}
    got = W._per_seed_l0("run_seed42", traj)
    assert got[42] == 60.0
    assert got[43] is None and got[44] is None
