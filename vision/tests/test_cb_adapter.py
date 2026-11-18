#!/usr/bin/env python3
"""Contract tests for cb_adapter.py.

A CB row is post-hoc: it never passes through a trainer, so it can ONLY be
scored offline. That makes this shim the single point where a CB number becomes
comparable to the other models' numbers, and these tests pin the two properties
that comparability rests on:

  1. the branch decomposition is EXACT (`_xM + _r_hat == recons`), so
     branch_metrics' own self-check cannot drift;
  2. the code and reconstruction agree with cb_sae/featbsf/adapters.py's
     `_CBDualAdapted`, which is what the sibling harness scores the same
     checkpoints with.

CPU-only and tiny. Run:  pytest tests/test_cb_adapter.py -q
"""

import sys
from pathlib import Path

import pytest
import torch

SAE_DIR = Path(__file__).resolve().parents[1]
REPO = SAE_DIR.parent
for p in (str(SAE_DIR), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

torch.set_num_threads(2)

D, G, K = 16, 32, 4
N_CONCEPTS, K_CB = 12, 3


def make_cb(use_poly: bool):
    """A tiny CBDualBSF on a b=1 base, the paper-faithful scalar arm."""
    cbd = pytest.importorskip("bsf.cb_dual_bsf")
    from bsf.sae_models import build_sae
    if use_poly:
        # use_poly requires a Dual base (the poly branch is warm-started from
        # the pruned dual's own weights).
        base = build_sae("dual_matryoshka_sae", D, G, K, poly_hidden=8)
    else:
        # deploy_batch_global mirrors models/matryoshka_sae.py's eval gate.
        base = build_sae("batchtopk_sae", D, G, K, deploy_batch_global=True)
    torch.manual_seed(0)
    model = cbd.CBDualBSF(base, n_concepts=N_CONCEPTS, k_cb=K_CB, b_cb=1,
                          use_poly=use_poly)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model.eval()


@pytest.mark.parametrize("use_poly", [False, True])
def test_branch_decomposition_is_exact(use_poly):
    """xM + r_hat == recons to float precision. Computing r_hat as a DIFFERENCE
    of two input-space reconstructions is what guarantees this: inverting the
    poly output directly would re-add the mean and break the identity (the
    exact trap knn_lp_sparse.forward_reps documents for DualSAE)."""
    from cb_adapter import CBRunAdapter
    cb = make_cb(use_poly)
    ad = CBRunAdapter(cb, n_retained=G, n_concepts=N_CONCEPTS).eval()
    x = torch.randn(9, D)
    _, _, recons = ad(x)
    if use_poly:
        assert ad._r_hat is not None
        assert torch.allclose(ad._xM + ad._r_hat, recons, atol=1e-6)
    else:
        # No poly branch: r_hat is UNDEFINED, not zero. Emitting zeros would
        # make a single-branch model look like a dual whose poly does nothing.
        assert ad._r_hat is None
        assert torch.allclose(ad._xM, recons, atol=1e-6)


@pytest.mark.parametrize("use_poly", [False, True])
def test_matches_featbsf_adapter(use_poly):
    """Same acts and same reconstruction as the sibling harness's adapter, so
    a CB checkpoint means one thing no matter which package scores it."""
    fb = pytest.importorskip("featbsf.adapters")
    from cb_adapter import CBRunAdapter
    cb = make_cb(use_poly)
    ours = CBRunAdapter(cb, n_retained=G, n_concepts=N_CONCEPTS).eval()
    # Same construction load_any uses for a cb_dual_bsf payload.
    theirs = fb._CBDualAdapted(
        cb, family="bsf", variant="cb_dual_bsf", is_dual=True,
        n_units=cb.R + cb.n_concepts, d=int(cb.d),
        dims_per_unit=cb.dims_per_unit_vector(), device="cpu")
    x = torch.randn(11, D)
    _, acts_ours, recon_ours = ours(x)
    acts_theirs, recon_theirs = theirs.acts_and_recon(x)
    assert torch.allclose(acts_ours, acts_theirs, atol=1e-6), \
        (acts_ours - acts_theirs).abs().max()
    assert torch.allclose(recon_ours, recon_theirs, atol=1e-6), \
        (recon_ours - recon_theirs).abs().max()


def test_units_are_split_not_pooled():
    """n_units = R + |C|, and the two halves stay separately identifiable.
    Pooling them would hide concept death whenever |C| >> R -- the failure mode
    the CB arm is most exposed to at the paper's k_cb."""
    from cb_adapter import CBRunAdapter
    cb = make_cb(False)
    ad = CBRunAdapter(cb, n_retained=G, n_concepts=N_CONCEPTS).eval()
    x = torch.randn(7, D)
    _, acts, _ = ad(x)
    assert acts.shape == (7, G + N_CONCEPTS)
    assert ad.cfg["dict_size"] == G + N_CONCEPTS
    # Concept half obeys its own top-k budget, independent of the base's gate.
    assert int((acts[:, G:] > 0).sum(dim=1).max()) <= K_CB


def test_concept_gate_is_relu_then_topk():
    """sigma_cb is ReLU THEN top-k: a negative concept logit must never survive
    just because it ranks in the top k_cb by value."""
    from cb_adapter import CBRunAdapter
    cb = make_cb(False)
    ad = CBRunAdapter(cb, n_retained=G, n_concepts=N_CONCEPTS).eval()
    x = torch.randn(64, D)
    _, acts, _ = ad(x)
    assert bool((acts[:, G:] >= 0).all())
