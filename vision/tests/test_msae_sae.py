#!/usr/bin/env python3
"""Correctness tests for models/msae_sae.py (Zaigrajew MSAE port).

Pins: the ported helpers equal the already-vetted bsf/sae_models.py originals
bit-for-bit; the nesting-support subset property; the deploy gate's exact
sparsity; the weighted-mean level loss against a hand computation; and the
buffer-guarded median b_dec init.

CPU-only and tiny (login-node safe).
Run:  pytest tests/test_msae_sae.py -q
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

SAE_DIR = Path(__file__).resolve().parents[1]
if str(SAE_DIR) not in sys.path:
    sys.path.insert(0, str(SAE_DIR))
REPO_ROOT = SAE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.msae_sae import (  # noqa: E402
    MSAE, build_nesting, matryoshka_weights, soft_cap, topk_mask)

torch.set_num_threads(2)

D, N, K = 16, 64, 4
NESTING = [4, 8, 16]


def make_model(weighting="rw", cap=30.0, seed=0):
    return MSAE({
        "model_type": "msae",
        "act_size": D, "dict_size": N, "top_k": K,
        "nesting_list": NESTING,
        "relative_importance": matryoshka_weights(len(NESTING), weighting),
        "soft_cap": cap,
        "l1_coeff": 0.0, "n_batches_to_dead": 3,
        "seed": seed, "device": "cpu", "dtype": torch.float32,
        "input_unit_norm": False,
    })


def test_helpers_equal_bsf_originals():
    """The re-implemented helpers must agree with the vetted bsf/ port."""
    try:
        from bsf.sae_models import (build_nesting as bsf_nesting,
                                    matryoshka_weights as bsf_weights,
                                    soft_cap as bsf_cap)
    except Exception as exc:
        pytest.skip(f"bsf.sae_models unavailable in this venv: {exc}")
    for k, mx in [(64, 256), (4, 16), (32, 256), (256, 256), (3, 20)]:
        assert build_nesting(k, mx) == bsf_nesting(k, mx)
    for n in (1, 2, 3, 5):
        for w in ("rw", "uw"):
            assert matryoshka_weights(n, w) == bsf_weights(n, w)
    x = torch.randn(32, 8) * 40
    assert torch.allclose(soft_cap(x, 30.0), bsf_cap(x, 30.0))
    assert torch.allclose(soft_cap(x, None), bsf_cap(x, None))


def test_official_constructions():
    assert build_nesting(64, 256) == [64, 128, 256]
    # rw: HIGHEST weight on the SMALLEST k (the reference code, not its
    # comment, is authoritative).
    assert matryoshka_weights(3, "rw") == [3.0, 2.0, 1.0]
    assert matryoshka_weights(3, "uw") == [1.0, 1.0, 1.0]


def test_nesting_support_subset():
    """top-k_i support is contained in top-k_{i+1} support for the same code."""
    torch.manual_seed(0)
    a = F.relu(torch.randn(16, N))
    prev = None
    for k in NESTING:
        z = topk_mask(a, k)
        supp = z != 0
        if prev is not None:
            assert bool((prev & ~supp).sum() == 0)
        prev = supp


def test_deploy_gate_sparsity():
    torch.manual_seed(1)
    model = make_model().eval()
    x = torch.randn(8, D)
    _, latents, _ = model(x)
    per_row = (latents != 0).sum(dim=1)
    assert (per_row <= K).all()
    assert per_row.float().mean() > 0  # not degenerate


def test_weighted_mean_loss_hand_computed():
    torch.manual_seed(2)
    model = make_model().eval()
    x = torch.randn(8, D)
    acts, latents, recons = model(x)
    loss, metrics = model.compute_loss(x, acts, latents, recons)

    # Hand computation from the SAME stashed quantities.
    w = matryoshka_weights(len(NESTING), "rw")
    hand_terms = []
    for k in NESTING:
        z = latents if k == K else topk_mask(acts, k)
        x_hat = z @ model.W_dec + model.b_dec
        hand_terms.append(F.mse_loss(x_hat, x))
    hand = sum(wi * t for wi, t in zip(w, hand_terms)) / sum(w)
    assert loss.item() == pytest.approx(hand.item(), rel=1e-6)
    # Per-level diagnostics carry the raw (unweighted) MSEs.
    for k, t in zip(NESTING, hand_terms):
        assert metrics[f"msae_l2_k{k}"] == pytest.approx(t.item(), rel=1e-6)
    # Deploy-level term IS the deployed reconstruction's error (latents reused
    # verbatim, no re-gating drift).
    assert metrics[f"msae_l2_k{K}"] == pytest.approx(
        metrics["l2_loss"], rel=1e-6)


def test_soft_cap_applied():
    """Large pre-acts are bounded by the cap in the model's code path."""
    torch.manual_seed(3)
    model = make_model(cap=1.0).eval()
    x = torch.randn(8, D) * 100
    acts, _, _ = model(x)
    assert acts.max().item() <= 1.0 + 1e-6


def test_b_dec_median_init_idempotent():
    torch.manual_seed(4)
    model = make_model()
    x1, x2 = torch.randn(64, D), torch.randn(64, D) + 7.0
    model.maybe_init_b_dec(x1)
    expected = x1.median(dim=0).values
    assert torch.allclose(model.b_dec.data, expected, atol=1e-6)
    model.maybe_init_b_dec(x2)
    assert torch.allclose(model.b_dec.data, expected, atol=1e-6)


def test_deploy_k_must_match_smallest_level():
    with pytest.raises(ValueError):
        MSAE({"act_size": D, "dict_size": N, "top_k": 8,   # != NESTING[0]=4
              "nesting_list": NESTING, "l1_coeff": 0.0,
              "n_batches_to_dead": 3, "seed": 0, "device": "cpu",
              "dtype": torch.float32, "input_unit_norm": False})
