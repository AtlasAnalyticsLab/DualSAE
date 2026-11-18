#!/usr/bin/env python3
"""Correctness tests for ``branch_metrics.py``.

CPU-only and tiny by design: every reference here is either a closed form or a
brute-force double loop that is cheap on a handful of synthetic rows, so the
whole file runs in seconds on a login node and needs no checkpoint, no LMDB and
no GPU.

What is pinned down
-------------------
a) the streaming O(N*D*M) MonoScore equals the brute-force pairwise definition,
   for a probe space that is NOT the input (that is the whole point of this
   module: the probe changes per representation while ``z`` does not);
b) the ``r_hat`` SPACE INVERSION -- ``xM_orig + r_hat_orig == x_hat`` holds with
   the scale-only inversion and does NOT hold with the naive
   ``postprocess_output`` one, on a real DualSAE with ``input_unit_norm=True``;
c) both R^2 estimators against a closed form, plus ``R^2@x == 1``;
d) binary purity is invariant to the representation (it is a function of ``z``
   and the labels only) and is not a per-representation metric of the schema;
e) retrieval purity against a hand-built bank whose answer is computed by hand;
f) single-branch baselines yield ``xM IS xhat`` and no ``r_hat`` at all.

Regression tests added after adversarial review (each names the defect it pins):
g) MonoScore's per-probe-space FLOOR (``null_cosine``) is the score of a latent
   with no selectivity, so raw MonoScore is not comparable across probe spaces;
h) per-latent MonoScore CAN be negative, so ``mean_all <= mean_active_code`` is
   not a theorem (it was asserted as one in a docstring, a report note and a
   test);
i) a zero-norm probe row is detected instead of silently biasing the numerator;
j) a capped run is INTERNALLY CONSISTENT -- ``l0_mean`` / ``active_counts``
   describe exactly the rows that ``z``, the banks and the labels describe;
k) a single-branch model's ``xhat`` bank is the SAME OBJECT as its ``xM`` bank,
   so the duplicate metric pass is skipped;
l) the completion predicate refuses a result computed with a different
   ``encode_batch_size`` / ``knn_k`` / ``retrieval_k`` (a BatchTopK cell computed
   at another batch size is well-formed and wrong);
m) ``--reps r_hat`` on a baseline fails fast, before the encode passes.

Run:  cd vision && python -m pytest tests/test_branch_metrics.py -x -q
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from branch_metrics import (  # noqa: E402
    KNN_K,
    MONO_DEN_EPS,
    MONO_DEN_FLOOR,
    MONO_EPS,
    SCHEMA_VERSION,
    TRAINING_BATCH_SIZE,
    R2Accumulator,
    binary_purity_summary,
    encode_split,
    monoscore_streaming,
    resolve_active_reps,
    result_is_complete,
    retrieval_purity,
    summarize_monoscore,
)
from knn_lp_sparse import DenseBank, forward_reps  # noqa: E402
from metrics import compute_binary_purity, compute_multilabel_binary_purity  # noqa: E402
from models import BatchTopKSAE, DualSAE  # noqa: E402

CPU = torch.device("cpu")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _dual_cfg(act_size: int, dict_size: int, input_unit_norm: bool, seed: int = 0) -> dict:
    """Minimal DualSAE cfg.  Mirrors the keys the Table-3 checkpoints carry."""
    return {
        "seed": seed,
        "act_size": act_size,
        "dict_size": dict_size,
        "device": "cpu",
        "dtype": torch.float32,
        "input_unit_norm": input_unit_norm,
        "top_k": 4,
        "sparsity_mode": "topk",
        "l1_coeff": 1e-4,
        "bandwidth": 1e-3,
        "poly_layers": 1,          # nn.Linear(act, act) + GELU -- what all 54 duals use
        "poly_hidden": 512,        # inert at poly_layers=1
        "poly_hidden_dims": None,
    }


def _brute_force_monoscore(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """The definition, straight out of the paper appendix, as a double loop.

        MS_j = sum_{i<l} a_ij a_lj <p_i, p_l> / sum_{i<l} a_ij a_lj

    with ``a`` the per-latent min/max rescaling of ``z``.  O(N^2 M): only usable
    on toy inputs, which is exactly why it is the reference.
    """
    z = z.double()
    p = p.double()
    mins = z.min(dim=0, keepdim=True)[0]
    maxs = z.max(dim=0, keepdim=True)[0]
    a = (z - mins) / (maxs - mins + MONO_EPS)
    n, m = a.shape
    sim = p @ p.T
    out = torch.zeros(m, dtype=torch.float64)
    for j in range(m):
        num = den = 0.0
        for i in range(n):
            for l in range(i + 1, n):
                wgt = float(a[i, j] * a[l, j])
                den += wgt
                num += wgt * float(sim[i, l])
        out[j] = num / (den + MONO_DEN_EPS) if den > MONO_DEN_FLOOR else 0.0
    return out


# --------------------------------------------------------------------------- #
# (a) MonoScore: streaming identity == brute force
# --------------------------------------------------------------------------- #

def test_monoscore_streaming_matches_brute_force_non_x_probe():
    g = torch.Generator().manual_seed(17)
    n, m, d = 13, 6, 4
    # Sparse-ish, non-negative latents with a dead column (index 5) and a
    # constant-valued column (index 4): both must land on exactly 0.0.
    z = torch.rand(n, m, generator=g)
    z[z < 0.4] = 0.0
    z[:, 5] = 0.0
    z[:, 4] = 0.7
    # The probe space is deliberately NOT the model input: it stands in for
    # r_hat / xM, which is the case this module exists for.  Rows must be
    # L2-normalized -- the closed form uses <p_i, p_i> == 1.
    probe_a = F.normalize(torch.randn(n, d, generator=g), p=2, dim=1)
    probe_b = F.normalize(torch.randn(n, d + 3, generator=g), p=2, dim=1)

    # chunk_size=5 does not divide 13, so the ragged final chunk is exercised.
    got = monoscore_streaming(z, {"r_hat": probe_a, "xM": probe_b}, CPU,
                              chunk_size=5, progress=False)

    for name, probe in (("r_hat", probe_a), ("xM", probe_b)):
        ref = _brute_force_monoscore(z, probe)
        assert torch.allclose(got["ms"][name].double(), ref, atol=2e-5), (
            f"{name}: streaming {got['ms'][name]} vs brute force {ref}")
    # A dead latent has den == 0 and must score exactly 0, not NaN.
    assert float(got["ms"]["r_hat"][5]) == 0.0
    assert float(got["den"][5]) == 0.0
    # A constant-valued latent rescales to a == 0 as well (max - min == 0).
    assert float(got["ms"]["r_hat"][4]) == 0.0

    # Chunking must not change the answer beyond fp32 noise.
    whole = monoscore_streaming(z, {"r_hat": probe_a}, CPU, chunk_size=1000, progress=False)
    assert torch.allclose(whole["ms"]["r_hat"], got["ms"]["r_hat"], atol=2e-6)


def test_monoscore_reductions_differ_as_documented():
    g = torch.Generator().manual_seed(3)
    n, m, d = 20, 8, 5
    z = torch.rand(n, m, generator=g)
    z[z < 0.5] = 0.0
    z[:, :3] = 0.0                      # three dead latents
    probe = F.normalize(torch.randn(n, d, generator=g), p=2, dim=1)
    res = monoscore_streaming(z, {"x": probe}, CPU, chunk_size=7, progress=False)
    active_counts = (z > 0).sum(dim=0)
    s = summarize_monoscore(res["ms"]["x"], res["den"], active_counts,
                            res["probe_stats"]["x"])

    assert s["n_latents"] == m
    assert s["n_active_paper"] == int((active_counts > 0).sum())
    assert s["n_active_paper"] == m - 3
    # Dead latents contribute exactly 0, so mean_all is the active SUM spread
    # over more latents.  The inequality therefore holds iff that sum is >= 0 --
    # NOT because per-latent scores are non-negative (this very fixture has four
    # negative ones: an isotropic random probe has a ~0 floor, so roughly half
    # the latents land below it).  Asserting the real precondition keeps the test
    # honest about what it is demonstrating; see
    # test_monoscore_can_be_negative_and_mean_all_can_exceed_the_masked_mean for
    # the case where it flips.
    assert float(res["ms"]["x"].min()) < 0.0, "fixture should contain negative scores"
    assert float(res["ms"]["x"].sum()) > 0.0, "precondition for the inequality"
    assert s["mean_all"] <= s["mean_active_code"] + 1e-12
    assert s["mean_all"] <= s["mean_active_paper"] + 1e-12
    assert s["mean_all"] < s["mean_active_paper"]      # strictly, with 3 dead


# --------------------------------------------------------------------------- #
# (g) MonoScore's per-probe-space floor
# --------------------------------------------------------------------------- #

def test_monoscore_null_cosine_is_the_mean_pairwise_cosine():
    """``null_cosine`` must equal the probe space's mean pairwise cosine.

    That value is the score of a latent carrying NO information, so it is the
    floor MonoScore is measured against.  Without it the four representation
    columns are on four different scales (dense branch outputs have a much
    higher floor than ``x``).
    """
    g = torch.Generator().manual_seed(41)
    n, d = 25, 4
    # Deliberately anisotropic: a common direction plus small noise, which is
    # what a decoder output looks like and what drives the floor up to ~0.95.
    probe = F.normalize(torch.tensor([3.0, 0.0, 0.0, 0.0]) + 0.4 * torch.randn(n, d, generator=g),
                        p=2, dim=1)
    z = torch.rand(n, 3, generator=g)

    res = monoscore_streaming(z, {"p": probe}, CPU, chunk_size=6, progress=False)
    got = res["probe_stats"]["p"]["null_cosine"]

    sim = probe.double() @ probe.double().T
    iu = torch.triu_indices(n, n, offset=1)
    expected = float(sim[iu[0], iu[1]].mean())
    assert got == pytest.approx(expected, abs=1e-5)
    assert expected > 0.8, "the fixture must actually be anisotropic"


def test_monoscore_of_an_uninformative_latent_equals_the_floor():
    """A latent with uniform ``a`` weights scores ``null_cosine``, NOT 0.

    That is precisely why the floor is the right subtrahend: "no selectivity"
    does not map to a score of 0, it maps to the probe space's mean pairwise
    cosine.  A column that is constant-and-nonzero everywhere except one row
    rescales to ``a == 1`` on the rest, i.e. an unweighted mean of pairwise
    cosines over those rows.
    """
    g = torch.Generator().manual_seed(42)
    n, d = 12, 3
    probe = F.normalize(torch.tensor([2.0, 0.0, 0.0]) + 0.5 * torch.randn(n, d, generator=g),
                        p=2, dim=1)
    z = torch.ones(n, 1)
    z[0, 0] = 0.0                                   # min=0, max=1 -> a in {0, 1}

    res = monoscore_streaming(z, {"p": probe}, CPU, chunk_size=5, progress=False)
    floor = res["probe_stats"]["p"]["null_cosine"]

    # a == 1 on rows 1..n-1, so MS is the unweighted mean pairwise cosine over
    # exactly those rows -- the floor restricted to the active subset.
    sub = probe[1:].double()
    sim = sub @ sub.T
    iu = torch.triu_indices(sub.shape[0], sub.shape[0], offset=1)
    expected = float(sim[iu[0], iu[1]].mean())
    assert float(res["ms"]["p"][0]) == pytest.approx(expected, abs=1e-5)
    # ... and it lands near the whole-space floor rather than anywhere near 0.
    assert abs(float(res["ms"]["p"][0]) - floor) < 0.05
    assert floor > 0.5, "the fixture must have a high floor for this to be meaningful"


def test_summarize_monoscore_emits_excess_over_the_floor():
    g = torch.Generator().manual_seed(43)
    z = torch.rand(15, 4, generator=g)
    probe = F.normalize(torch.tensor([1.0, 0.0]) + 0.3 * torch.randn(15, 2, generator=g), p=2, dim=1)
    res = monoscore_streaming(z, {"p": probe}, CPU, chunk_size=15, progress=False)
    s = summarize_monoscore(res["ms"]["p"], res["den"], (z > 0).sum(dim=0), res["probe_stats"]["p"])

    floor = res["probe_stats"]["p"]["null_cosine"]
    assert s["null_cosine"] == pytest.approx(floor)
    for key in ("all", "active_code", "active_paper"):
        assert s[f"excess_{key}"] == pytest.approx(s[f"mean_{key}"] - floor, abs=1e-9)

    # Without probe_stats the excess must be None, never 0.0: a missing floor
    # silently treated as a floor of zero is exactly the confound being fixed.
    bare = summarize_monoscore(res["ms"]["p"], res["den"], (z > 0).sum(dim=0))
    assert bare["null_cosine"] is None
    assert bare["excess_all"] is None


# --------------------------------------------------------------------------- #
# (h) MonoScore is not sign-constrained
# --------------------------------------------------------------------------- #

def test_monoscore_can_be_negative_and_mean_all_can_exceed_the_masked_mean():
    """``mean_all <= mean_active_code`` is NOT a theorem.

    ``num = 0.5*(||sum_i a_i p_i||^2 - sum_i a_i^2)`` is negative whenever a
    latent's co-activating rows point in opposing probe directions.  Dead latents
    contribute exactly 0, so they pull the all-latent mean ABOVE the masked mean
    in that regime.  It holds on the 108 real checkpoints only because every
    probe space there has a strongly positive mean pairwise cosine -- which is
    the same anisotropy that makes the raw scores incomparable across reps.
    """
    # a = [0,1,1,1] over probes +e1, +e1, -e1, +e2  ->  num = -1.0, den = 3.0.
    z = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])  # latent 1 is dead
    probe = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]), p=2, dim=1)

    res = monoscore_streaming(z, {"p": probe}, CPU, chunk_size=4, progress=False)
    ms = res["ms"]["p"]
    assert float(ms[0]) == pytest.approx(-1.0 / 3.0, abs=1e-6)
    assert float(ms[1]) == 0.0                      # dead latent

    s = summarize_monoscore(ms, res["den"], (z > 0).sum(dim=0), res["probe_stats"]["p"])
    # The dead latent's exact 0 drags the all-latent mean UP past the masked one.
    assert s["mean_all"] == pytest.approx(-1.0 / 6.0, abs=1e-6)
    assert s["mean_active_code"] == pytest.approx(-1.0 / 3.0, abs=1e-6)
    assert s["mean_all"] > s["mean_active_code"], "the inequality is not guaranteed"
    # The floor is negative here too, and the excess is still well defined.
    assert s["null_cosine"] < 0.0
    assert s["excess_all"] == pytest.approx(s["mean_all"] - s["null_cosine"], abs=1e-9)


# --------------------------------------------------------------------------- #
# (i) degenerate probe rows are detected, not silently absorbed
# --------------------------------------------------------------------------- #

def test_monoscore_flags_zero_norm_probe_rows():
    """A zero row survives ``F.normalize`` and biases ``num`` downward.

    ``num = 0.5*(q - v)`` subtracts ``sum_i a_ij^2`` assuming ``<p_i,p_i> == 1``.
    A zero row contributes 0 to ``q`` but its full ``a_ij^2`` is still
    subtracted, with no other symptom.  Reachable in principle for ``r_hat`` when
    ``x_std`` is 0 for a sample.  The scores stay bit-compatible with the
    reference implementation; the DETECTION is what this pins down.
    """
    probe = F.normalize(torch.randn(8, 3), p=2, dim=1)
    clean = monoscore_streaming(torch.rand(8, 2), {"p": probe}, CPU, chunk_size=8, progress=False)
    assert clean["probe_stats"]["p"]["n_degenerate_rows"] == 0
    assert clean["probe_stats"]["p"]["min_row_norm"] == pytest.approx(1.0, abs=1e-5)

    dirty = probe.clone()
    dirty[3] = 0.0                                   # what F.normalize does to a zero row
    res = monoscore_streaming(torch.rand(8, 2), {"p": dirty}, CPU, chunk_size=3, progress=False)
    assert res["probe_stats"]["p"]["n_degenerate_rows"] == 1
    assert res["probe_stats"]["p"]["min_row_norm"] == 0.0


def test_monoscore_rejects_a_probe_with_the_wrong_row_count():
    with pytest.raises(ValueError, match="rows"):
        monoscore_streaming(torch.rand(6, 2), {"p": F.normalize(torch.randn(5, 3), p=2, dim=1)},
                            CPU, chunk_size=6, progress=False)


# --------------------------------------------------------------------------- #
# (b) the r_hat space inversion
# --------------------------------------------------------------------------- #

@torch.no_grad()
def test_rhat_space_inversion_with_input_unit_norm():
    torch.manual_seed(5)
    act, dic, batch = 12, 32, 7
    model = DualSAE(_dual_cfg(act, dic, input_unit_norm=True)).eval()
    # A clearly non-zero per-sample mean, otherwise the trap is invisible.
    x = torch.randn(batch, act) + 3.0

    out = forward_reps(model, x, is_dual=True)
    correct_err = float((out["xM"] + out["r_hat"] - out["xhat"]).abs().max())
    assert correct_err < 1e-5, f"scale-only inversion should be exact, got {correct_err:.2e}"

    # The trap: postprocess_output() adds x_mean a SECOND time.
    naive_r_hat = model.postprocess_output(model._r_hat, model._x_mean, model._x_std)
    naive_err = float((out["xM"] + naive_r_hat - out["xhat"]).abs().max())
    assert naive_err > 1e-3, "the naive inversion must be detectably wrong"
    assert naive_err > 100 * max(correct_err, 1e-12)
    # ... and the whole difference is exactly the per-sample mean, broadcast over
    # the feature dim: a pure additive offset, which is why the mistake survives
    # every downstream op that is translation-tolerant.
    assert torch.allclose(naive_r_hat - out["r_hat"],
                          model._x_mean.expand(batch, act), atol=1e-6)
    # NB: on the real (trained) Dual-JumpReLU checkpoints this offset is only a
    # ~12% perturbation and leaves cos(correct, naive) at 0.997-0.999, i.e. the
    # bug is nearly invisible to KNN/LP/retrieval, which L2-normalize rows.  That
    # ratio depends on the trained poly branch's scale and is NOT reproducible
    # with a randomly initialized toy model (here r_hat is tiny next to the mean,
    # so the cosine is far lower), so it is documented rather than asserted.


@torch.no_grad()
def test_rhat_inversion_is_identity_without_input_unit_norm():
    torch.manual_seed(6)
    model = DualSAE(_dual_cfg(10, 24, input_unit_norm=False)).eval()
    x = torch.randn(5, 10) + 2.0
    out = forward_reps(model, x, is_dual=True)
    # _x_mean / _x_std are None, so both inversions collapse to the identity.
    assert model._x_std is None and model._x_mean is None
    assert torch.equal(out["r_hat"], model._r_hat)
    assert torch.allclose(out["xM"] + out["r_hat"], out["xhat"], atol=1e-6)


# --------------------------------------------------------------------------- #
# (c) R^2
# --------------------------------------------------------------------------- #

def test_r2_closed_form_and_identity():
    # col 0: [0,2,0,2] -> mean 1, ss_tot = 4
    # col 1: [0,0,4,4] -> mean 2, ss_tot = 16
    # col 2: constant   -> ss_tot = 0, must be dropped from the per-feature mean
    x = torch.tensor([[0.0, 0.0, 7.0],
                      [2.0, 0.0, 7.0],
                      [0.0, 4.0, 7.0],
                      [2.0, 4.0, 7.0]])
    v = x + torch.tensor([0.5, 0.5, 0.5])          # ss_res = [1, 1, 1]

    acc = R2Accumulator(["x", "v"], CPU)
    for start in (0, 3):                            # ragged batches on purpose
        chunk = slice(start, start + 3)
        acc.update(x[chunk], {"x": x[chunk], "v": v[chunk]})
    res = acc.result()

    # rep 'x' against itself: ss_res == 0 exactly in float64 -> both estimators 1.
    assert res["x"]["r2_perfeat"] == pytest.approx(1.0, abs=1e-12)
    assert res["x"]["r2_global"] == pytest.approx(1.0, abs=1e-12)

    # Per-feature mean over the two dims with variance: mean(1-1/4, 1-1/16).
    assert res["v"]["r2_perfeat"] == pytest.approx((0.75 + 0.9375) / 2, abs=1e-9)
    # Global ratio sums over ALL dims: 1 - (1+1+1)/(4+16+0).
    assert res["v"]["r2_global"] == pytest.approx(1.0 - 3.0 / 20.0, abs=1e-9)
    # The two estimators genuinely disagree -- that is why both are emitted.
    assert res["v"]["r2_perfeat"] != pytest.approx(res["v"]["r2_global"], abs=1e-6)

    assert res["v"]["n_valid_dims"] == 2 and res["v"]["n_dims"] == 3
    assert res["v"]["n_samples"] == 4


def test_r2_of_a_perfect_reconstruction_is_one_even_when_shifted_by_noise():
    """A sanity guard: R^2 must fall when the reconstruction degrades."""
    g = torch.Generator().manual_seed(11)
    x = torch.randn(50, 6, generator=g)
    acc = R2Accumulator(["good", "bad"], CPU)
    acc.update(x, {"good": x + 0.01 * torch.randn(50, 6, generator=g),
                   "bad": torch.zeros_like(x)})
    res = acc.result()
    assert res["good"]["r2_perfeat"] > 0.99
    assert res["bad"]["r2_perfeat"] < 0.2


# --------------------------------------------------------------------------- #
# (d) purity is representation-invariant
# --------------------------------------------------------------------------- #

def test_binary_purity_is_representation_invariant_single_label():
    g = torch.Generator().manual_seed(21)
    z = torch.rand(40, 9, generator=g)
    z[z < 0.6] = 0.0
    z[:, 8] = 0.0                                   # dead latent, excluded
    labels = torch.randint(0, 4, (40,), generator=g)

    ref = compute_binary_purity(z, labels)
    active = (z > 0).any(dim=0)
    expected = float(ref[active].mean())

    got = binary_purity_summary(z, labels, CPU)
    assert got["mean_active"] == pytest.approx(expected, abs=1e-6)
    assert got["n_active_latents"] == int(active.sum())
    assert got["multilabel"] is False

    # The API takes no representation argument: purity CANNOT vary by rep.  If a
    # future refactor adds one, this fails loudly rather than silently producing
    # four identical columns that read as four measurements.
    params = set(inspect.signature(binary_purity_summary).parameters)
    assert params == {"z", "labels", "device"}
    # Calling it while different "representations" are in play changes nothing.
    for _rep in (torch.randn(40, 5), torch.randn(40, 7), torch.zeros(40, 3)):
        assert binary_purity_summary(z, labels, CPU)["mean_active"] == pytest.approx(expected, abs=1e-6)


def test_binary_purity_multilabel_path():
    g = torch.Generator().manual_seed(22)
    z = torch.rand(30, 7, generator=g)
    z[z < 0.5] = 0.0
    labels = (torch.rand(30, 5, generator=g) < 0.3).float()
    ref = compute_multilabel_binary_purity(z, labels)
    active = (z > 0).any(dim=0)
    got = binary_purity_summary(z, labels, CPU)
    assert got["multilabel"] is True
    assert got["mean_active"] == pytest.approx(float(ref[active].mean()), abs=1e-6)


def _complete_payload() -> dict:
    """A minimal payload the completion predicate accepts."""
    rep_block = {"r2_perfeat": 1.0, "r2_global": 1.0, "monoscore": {}, "knn": {}, "lp": {},
                 "retrieval_purity_self": {}}
    return {"schema_version": SCHEMA_VERSION, "status": "ok",
            "encode_batch_size": TRAINING_BATCH_SIZE, "knn_k": KNN_K, "retrieval_k": 50,
            "model_level": {"binary_purity": {"mean_active": 0.1}},
            "reps": {"x": rep_block, "xM": rep_block, "xhat": rep_block, "r_hat": None}}


def test_schema_has_no_per_representation_purity():
    """Purity lives in ``model_level``; a rep block without it is still complete."""
    payload = _complete_payload()
    assert result_is_complete(payload, ["x", "xM", "r_hat", "xhat"], ["self"])
    # A baseline's null r_hat counts as complete -- recomputing it cannot help.
    assert result_is_complete(payload, ["r_hat"], ["self"])
    # A requested retrieval mode that was never written does not.
    assert not result_is_complete(payload, ["x"], ["self", "bank"])
    # An older schema is not the same answer.
    assert not result_is_complete({**payload, "schema_version": 0}, ["x"], ["self"])
    # A crashed cell is not a result.
    assert not result_is_complete({**payload, "status": "error"}, ["x"], ["self"])
    # A capped smoke run is not a result either: the loader is unshuffled, so a
    # cap is a prefix of the split, and a prefix is not the split.  Without this
    # a stray smoke result would mark a real cell done and it would look
    # perfectly well-formed on inspection.
    assert not result_is_complete({**payload, "max_eval_samples": 2048}, ["x"], ["self"])
    assert not result_is_complete({**payload, "max_train_samples": 4096}, ["x"], ["self"])


# --------------------------------------------------------------------------- #
# (l) the completion predicate guards the PROTOCOL constants, not just presence
# --------------------------------------------------------------------------- #

def test_completion_predicate_refuses_a_different_protocol():
    """A well-formed result computed with other constants is not this result.

    The failure it prevents: ``branch_metrics.py --encode-batch-size 512`` writes
    an uncapped ``status: "ok"`` JSON.  BatchTopK takes its top-k over the whole
    batch, so z -- and with it MonoScore, purity, L0 and every representation --
    is wrong for 36 of the 108 cells.  Without this check the submitter reports
    that cell as "already done" forever and the collector aggregates it silently.
    """
    payload = _complete_payload()
    assert result_is_complete(payload, ["x"], ["self"], TRAINING_BATCH_SIZE, KNN_K, 50)

    assert not result_is_complete({**payload, "encode_batch_size": 512}, ["x"], ["self"])
    assert not result_is_complete({**payload, "knn_k": 10}, ["x"], ["self"])
    assert not result_is_complete({**payload, "retrieval_k": 10}, ["x"], ["self"])
    # retrieval_k only constrains a result when a retrieval mode was asked for.
    assert result_is_complete({**payload, "retrieval_k": 10}, ["x"], [])
    # A pre-protocol payload (key absent entirely) is likewise not accepted.
    stripped = {k: v for k, v in payload.items() if k != "encode_batch_size"}
    assert not result_is_complete(stripped, ["x"], ["self"])


# --------------------------------------------------------------------------- #
# (m) --reps r_hat on a baseline fails fast
# --------------------------------------------------------------------------- #

def test_resolve_active_reps_drops_rhat_for_baselines_and_fails_loudly_when_empty():
    # Normal path: r_hat is silently dropped so one submitter command covers all
    # 108 cells; the caller writes null for it.
    assert resolve_active_reps(["x", "xM", "r_hat", "xhat"], is_dual=False) == ["x", "xM", "xhat"]
    assert resolve_active_reps(["x", "xM", "r_hat", "xhat"], is_dual=True) == \
        ["x", "xM", "r_hat", "xhat"]
    # The empty case must raise -- and it must raise BEFORE any encoding, which
    # is why the check lives in a helper the driver calls up front rather than in
    # a summary line 150 lines (and two full-split encodes) later.
    with pytest.raises(SystemExit, match="r_hat"):
        resolve_active_reps(["r_hat"], is_dual=False, model_type="batch_topk")
    assert resolve_active_reps(["r_hat"], is_dual=True) == ["r_hat"]


# --------------------------------------------------------------------------- #
# (e) retrieval purity on a hand-built bank
# --------------------------------------------------------------------------- #

def _unit(deg: float) -> torch.Tensor:
    rad = torch.tensor(deg * torch.pi / 180.0)
    return torch.stack([torch.cos(rad), torch.sin(rad)])


def test_retrieval_purity_self_hand_computed():
    """Six unit vectors in the plane; every top-2 neighbourhood is unambiguous.

    Angles 0,10,20 (cluster A) and 90,100,110 (cluster B); labels
    [0,0,1, 1,1,0].  With k=2 and the query excluded:

        0deg  -> {10,20}  labels [0,1] vs query 0 -> 0.5
        10deg -> {0,20}   labels [0,1] vs query 0 -> 0.5   (tie at cos10, both kept)
        20deg -> {10,0}   labels [0,0] vs query 1 -> 0.0
        90deg -> {100,110}labels [1,0] vs query 1 -> 0.5
        100deg-> {90,110} labels [1,0] vs query 1 -> 0.5   (tie at cos10, both kept)
        110deg-> {100,90} labels [1,1] vs query 0 -> 0.0
        mean = 2.0 / 6
    """
    bank = torch.stack([_unit(a) for a in (0, 10, 20, 90, 100, 110)])
    labels = torch.tensor([0, 0, 1, 1, 1, 0])
    db = DenseBank(F.normalize(bank, p=2, dim=1), labels)

    res = retrieval_purity(db, db, CPU, k=2, exclude_self=True,
                           query_block=4, db_block=4, progress=False)
    assert res["purity"] == pytest.approx(2.0 / 6.0, abs=1e-6)
    assert res["n_scored"] == 6 and res["n_skipped_unlabeled"] == 0
    assert res["exclude_self"] is True and res["k"] == 2

    # Self-exclusion is what makes this non-trivial: without it every query
    # retrieves itself at similarity 1 and purity is inflated.
    incl = retrieval_purity(db, db, CPU, k=2, exclude_self=False,
                            query_block=4, db_block=4, progress=False)
    assert incl["purity"] > res["purity"]

    # Block sizes must not matter (the top-k merge across blocks is exact).
    blocked = retrieval_purity(db, db, CPU, k=2, exclude_self=True,
                               query_block=2, db_block=2, progress=False)
    assert blocked["purity"] == pytest.approx(res["purity"], abs=1e-9)

    # Coherence is the modal-class share: every neighbourhood above is one 0 and
    # one 1 except 20deg ({0,0}) and 110deg ({1,1}), so (4*0.5 + 2*1)/6.
    assert res["coherence"] == pytest.approx((4 * 0.5 + 2 * 1.0) / 6.0, abs=1e-6)


def test_retrieval_purity_bank_mode_and_multilabel():
    """Query the *train* bank; multi-label purity with an unlabeled query."""
    db_x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    db_y = torch.tensor([[1.0, 0.0, 0.0],
                         [1.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, 1.0, 1.0]])
    q_x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    q_y = torch.tensor([[1.0, 0.0, 0.0],      # positive class 0
                        [0.0, 0.0, 0.0]])     # NO positive label -> skipped
    db = DenseBank(db_x, db_y)
    q = DenseBank(q_x, q_y)

    res = retrieval_purity(q, db, CPU, k=2, exclude_self=False,
                           query_block=1, db_block=3, progress=False)
    # Query 0's two neighbours are rows 0 and 1; class-0 support = 2/2 = 1.0 and
    # that is the query's only positive class.
    assert res["purity"] == pytest.approx(1.0, abs=1e-9)
    assert res["n_scored"] == 1
    # Scoring the unlabeled query 0 instead of skipping it would halve the number
    # -- 10.6% of the OIv7 test split is unlabeled, so that bias is not academic.
    assert res["n_skipped_unlabeled"] == 1
    assert res["multilabel"] is True
    # Coherence is defined for every query (max support over ALL classes).
    assert res["coherence"] == pytest.approx(1.0, abs=1e-9)


def test_retrieval_purity_reduces_to_single_label_definition():
    """The multi-label generalization must agree with the single-label formula
    whenever each query has exactly one positive class."""
    g = torch.Generator().manual_seed(31)
    n, c = 40, 5
    x = F.normalize(torch.randn(n, 3, generator=g), p=2, dim=1)
    y_int = torch.randint(0, c, (n,), generator=g)
    y_hot = F.one_hot(y_int, num_classes=c).float()

    single = retrieval_purity(DenseBank(x, y_int), DenseBank(x, y_int), CPU, k=5,
                              exclude_self=True, query_block=8, db_block=16, progress=False)
    multi = retrieval_purity(DenseBank(x, y_hot), DenseBank(x, y_hot), CPU, k=5,
                             exclude_self=True, query_block=8, db_block=16, progress=False)
    assert multi["purity"] == pytest.approx(single["purity"], abs=1e-6)
    assert multi["n_skipped_unlabeled"] == 0


def test_retrieval_purity_handles_a_bank_smaller_than_k():
    x = F.normalize(torch.randn(3, 4), p=2, dim=1)
    bank = DenseBank(x, torch.tensor([0, 1, 1]))
    res = retrieval_purity(bank, bank, CPU, k=50, exclude_self=True, progress=False)
    assert res["k"] == 2                    # clamped to n_db - 1
    assert not torch.isnan(torch.tensor(res["purity"]))


# --------------------------------------------------------------------------- #
# (f) baselines: xM IS xhat, r_hat absent
# --------------------------------------------------------------------------- #

def test_baseline_xM_is_xhat_and_rhat_absent():
    torch.manual_seed(9)
    cfg = {"seed": 0, "act_size": 8, "dict_size": 16, "device": "cpu", "dtype": torch.float32,
           "input_unit_norm": False, "top_k": 3, "l1_coeff": 1e-4}
    model = BatchTopKSAE(cfg).eval()
    out = forward_reps(model, torch.randn(6, 8), is_dual=False)

    # Not merely equal: the SAME object, so no future refactor can make them
    # drift apart silently.
    assert out["xM"] is out["xhat"]
    # r_hat must be absent rather than zeros -- there is no poly branch to read.
    assert "r_hat" not in out
    assert not hasattr(model, "_r_hat")
    # And 'x' is echoed back as the caller's raw tensor, never _x_preprocessed.
    assert out["x"].shape == (6, 8)


def test_dual_forward_exposes_all_four_representations():
    torch.manual_seed(4)
    model = DualSAE(_dual_cfg(9, 20, input_unit_norm=True)).eval()
    out = forward_reps(model, torch.randn(5, 9) + 1.0, is_dual=True)
    assert set(out) == {"x", "z", "xM", "xhat", "r_hat"}
    assert all(out[k].shape == (5, 9) for k in ("x", "xM", "xhat", "r_hat"))
    assert out["z"].shape == (5, 20)


# --------------------------------------------------------------------------- #
# (j)/(k) encode_split: capped runs stay internally consistent, and a
#         single-branch model's xhat bank is the SAME OBJECT as its xM bank
# --------------------------------------------------------------------------- #

def _fake_loader(x: torch.Tensor, labels: torch.Tensor, batch: int):
    """A list of ``(features, labels)`` tuples -- all ``encode_split`` iterates.

    Using a plain list instead of a DataLoader keeps the test CPU-only and
    instant while exercising the identical code path (``unpack_batch`` accepts a
    2-tuple), including the ragged final batch.
    """
    return [(x[i:i + batch], labels[i:i + batch]) for i in range(0, x.shape[0], batch)]


def test_encode_split_capped_run_is_internally_consistent():
    """``l0_mean``/``active_counts`` must describe the SAME rows as z and the banks.

    The defect this pins: the activity counters used to accumulate over whole
    forward batches while the banks, labels and cached ``z`` were truncated to the
    cap, so ``--max-eval-samples 1024 --encode-batch-size 2048`` reported an
    8192-latent activity mask over 2048 rows next to a MonoScore over 1024 -- a
    smoke test that contradicts itself, which is the opposite of what a smoke
    test is for.  The forward still runs on the WHOLE batch on purpose
    (BatchTopK's top-k is batch-level), only the accumulation is truncated.
    """
    torch.manual_seed(77)
    model = BatchTopKSAE({"seed": 0, "act_size": 6, "dict_size": 12, "device": "cpu",
                          "dtype": torch.float32, "input_unit_norm": False, "top_k": 3,
                          "l1_coeff": 1e-4}).eval()
    n, cap, batch = 10, 7, 4                       # cap falls mid-batch on purpose
    x = torch.randn(n, 6)
    labels = torch.randint(0, 3, (n,))

    ev = encode_split(model, _fake_loader(x, labels, batch), CPU, ["x", "xM", "xhat"],
                      is_dual=False, split_name="test", cache_z=True, want_r2=True,
                      want_active_counts=True, max_samples=cap)

    assert ev.n_rows == cap
    assert ev.labels.shape[0] == cap
    assert ev.z.shape[0] == cap
    assert ev.banks["x"].n_rows == cap
    assert ev.r2["x"]["n_samples"] == cap
    # The counters describe exactly those rows, not the 8 rows that were forwarded.
    assert int(ev.active_counts.sum()) == int((ev.z > 0).sum())
    assert ev.l0_mean == pytest.approx(float((ev.z > 0).sum()) / cap, abs=1e-9)

    # Uncapped, everything still lines up (and the ragged last batch is covered).
    full = encode_split(model, _fake_loader(x, labels, batch), CPU, ["x", "xhat"],
                        is_dual=False, split_name="test", cache_z=True,
                        want_active_counts=True)
    assert full.n_rows == n and int(full.active_counts.sum()) == int((full.z > 0).sum())


def test_encode_split_aliases_xhat_to_xM_for_single_branch_models():
    """54 of 108 cells would otherwise compute every metric twice on one tensor.

    ``forward_reps`` returns the SAME object for a baseline's ``xM`` and
    ``xhat``, so materializing both doubles the largest allocation in the job (a
    SigLIP2 ImageNet train bank is 5.9 GB) and reruns KNN + LP + two k=50
    retrievals on identical rows for no extra information.
    """
    torch.manual_seed(78)
    model = BatchTopKSAE({"seed": 0, "act_size": 5, "dict_size": 10, "device": "cpu",
                          "dtype": torch.float32, "input_unit_norm": False, "top_k": 2,
                          "l1_coeff": 1e-4}).eval()
    x, labels = torch.randn(6, 5), torch.randint(0, 2, (6,))

    ev = encode_split(model, _fake_loader(x, labels, 3), CPU, ["x", "xM", "xhat"],
                      is_dual=False, split_name="test", want_r2=True)
    assert ev.canonical["xhat"] == "xM" and ev.canonical["xM"] == "xM"
    # Identity, not equality: an alias cannot drift from its source.
    assert ev.banks["xhat"] is ev.banks["xM"]
    assert ev.banks["x"] is not ev.banks["xM"]
    assert ev.r2["xhat"] == ev.r2["xM"]
    # The structural baseline check still fires.
    assert ev.checks["xM_is_xhat_object"] is True

    # A DUAL model must NOT alias: xM and xhat are genuinely different tensors.
    dual = DualSAE(_dual_cfg(5, 10, input_unit_norm=True)).eval()
    dv = encode_split(dual, _fake_loader(x, labels, 3), CPU, ["xM", "r_hat", "xhat"],
                      is_dual=True, split_name="test")
    assert dv.canonical == {"xM": "xM", "r_hat": "r_hat", "xhat": "xhat"}
    assert dv.banks["xhat"] is not dv.banks["xM"]
    assert dv.checks["decomposition_rel_err"] < 1e-4
