#!/usr/bin/env python3
"""Correctness + regression tests for the Dual-Matryoshka sparsity modes
('matryoshka', 'msae') added to models/dual_sae.py.

The composition is NOVEL (no official reference); what CAN be pinned is:
a) single-level nesting reduces the multi-level recon term EXACTLY to the
   standard DualSAE l2 term (the design's defining property);
b) the multi-level term matches a hand computation (shared r_hat in every
   level's target, per-element mean reduction, normalized weights);
c) warmup semantics: with poly frozen, gradients reach the mono dictionary
   and NOT the poly branch;
d) LEGACY REGRESSION: the four pre-existing modes gain no new state-dict
   entries (strict-load compatibility with every existing dual checkpoint)
   and maybe_init_b_dec is a no-op for them.

CPU-only and tiny (login-node safe).
Run:  pytest tests/test_dual_matryoshka.py -q
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

SAE_DIR = Path(__file__).resolve().parents[1]
if str(SAE_DIR) not in sys.path:
    sys.path.insert(0, str(SAE_DIR))

from models.dual_sae import DualSAE, topk_sparsify  # noqa: E402

torch.set_num_threads(2)

D, N, K = 16, 32, 4


def make_dual(mode, seed=0, **extra):
    cfg = {
        "act_size": D, "dict_size": N, "top_k": K,
        "l1_coeff": 1e-4, "bandwidth": 1e-3,
        "poly_hidden": 0, "poly_layers": 1, "poly_hidden_dims": None,
        "sparsity_mode": mode,
        "n_batches_to_dead": 3, "seed": seed,
        "device": "cpu", "dtype": torch.float32, "input_unit_norm": False,
    }
    cfg.update(extra)
    return DualSAE(cfg)


# --------------------------------------------------------------------------
# Legacy regression
# --------------------------------------------------------------------------
def test_legacy_modes_state_dict_unchanged():
    """No new buffers/params may appear for the legacy modes — old dual
    checkpoints must keep loading strict=True."""
    for mode in ("topk", "batch_topk", "vanilla"):
        sd = make_dual(mode).state_dict()
        # num_batches_not_active is a plain attribute (not a registered
        # buffer) in BaseAutoencoder — deliberately absent from state dicts.
        expected = {"b_dec", "b_enc", "W_enc", "W_dec",
                    "poly.0.weight", "poly.0.bias"}
        assert set(sd.keys()) == expected, (mode, sorted(sd.keys()))
    sd = make_dual("jump_relu").state_dict()
    assert "jumprelu.log_threshold" in sd
    assert not any("group_sizes" in k or "nesting" in k or
                   "_b_dec_initialized" in k for k in sd)


def test_legacy_modes_l2_is_plain_mse_and_bdec_hook_noop():
    torch.manual_seed(0)
    for mode in ("topk", "batch_topk", "vanilla"):
        model = make_dual(mode).eval()
        b_dec_before = model.b_dec.data.clone()
        model.maybe_init_b_dec(torch.randn(32, D) + 3.0)   # must be a no-op
        assert torch.equal(model.b_dec.data, b_dec_before)
        x = torch.randn(8, D)
        _, acts, recons = model(x)
        _, metrics = model.compute_loss(x, acts, recons, span_coef=0.0)
        expected = (recons.float() - x.float()).pow(2).mean().item()
        assert metrics["l2_loss"] == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------
# Dual-Matryoshka (Bussmann prefix groups)
# --------------------------------------------------------------------------
def test_dual_matryoshka_single_group_reduces_to_dualsae():
    """One group spanning the whole dictionary == plain DualSAE(batch_topk):
    the multi-level recon term must equal the standard l2 exactly."""
    torch.manual_seed(1)
    model = make_dual("matryoshka", group_sizes=[N]).eval()
    x = torch.randn(8, D)
    _, acts, recons = model(x)
    loss, metrics = model.compute_loss(x, acts, recons, span_coef=0.0)
    expected = (recons.float() - x.float()).pow(2).mean().item()
    assert metrics["l2_loss"] == pytest.approx(expected, rel=1e-6)


def test_dual_matryoshka_hand_computed():
    torch.manual_seed(2)
    sizes = [4, 8, 20]
    model = make_dual("matryoshka", group_sizes=sizes).eval()
    x = torch.randn(8, D)
    _, acts, recons = model(x)
    _, metrics = model.compute_loss(x, acts, recons, span_coef=0.0)

    r_hat = model._r_hat
    x_hat = torch.zeros_like(x) + model.b_dec
    hand_terms = []
    for f_g, w_g in zip(torch.split(acts, sizes, dim=1),
                        torch.split(model.W_dec, sizes, dim=0)):
        x_hat = x_hat + f_g @ w_g
        hand_terms.append(((x_hat + r_hat) - x).pow(2).mean())
    hand = torch.stack(hand_terms).mean()          # uniform weights
    assert metrics["l2_loss"] == pytest.approx(hand.item(), rel=1e-6)
    # Last prefix == full dictionary -> its term is the standard l2.
    assert metrics[f"matryoshka_l2_g{len(sizes)-1}"] == pytest.approx(
        (recons.float() - x.float()).pow(2).mean().item(), rel=1e-6)
    # The deploy gate is batch top-k: exactly k*B nonzeros.
    assert int((acts != 0).sum()) == K * x.shape[0]


# --------------------------------------------------------------------------
# Dual-MSAE (sparsity-level nesting)
# --------------------------------------------------------------------------
def test_dual_msae_single_level_reduces_to_dualsae():
    torch.manual_seed(3)
    model = make_dual("msae", nesting_list=[K],
                      relative_importance=[1.0]).eval()
    x = torch.randn(8, D)
    _, acts, recons = model(x)
    _, metrics = model.compute_loss(x, acts, recons, span_coef=0.0)
    expected = (recons.float() - x.float()).pow(2).mean().item()
    assert metrics["l2_loss"] == pytest.approx(expected, rel=1e-6)


def test_dual_msae_hand_computed():
    torch.manual_seed(4)
    nesting = [4, 8, 16]
    weights = [3.0, 2.0, 1.0]
    model = make_dual("msae", nesting_list=nesting,
                      relative_importance=weights).eval()
    x = torch.randn(8, D)
    _, acts, recons = model(x)
    _, metrics = model.compute_loss(x, acts, recons, span_coef=0.0)

    a = model._a_pre_gate
    r_hat = model._r_hat
    hand_terms = []
    for k_l in nesting:
        z = acts if k_l == K else topk_sparsify(a, k_l)
        xM_l = z @ model.W_dec + model.b_dec
        hand_terms.append(((xM_l + r_hat) - x).pow(2).mean())
    hand = sum(w * t for w, t in zip(weights, hand_terms)) / sum(weights)
    assert metrics["l2_loss"] == pytest.approx(hand.item(), rel=1e-6)
    # Deploy-level term (its z is `acts` verbatim) == the standard l2.
    assert metrics[f"msae_l2_k{K}"] == pytest.approx(
        (recons.float() - x.float()).pow(2).mean().item(), rel=1e-6)


def test_dual_msae_deploy_k_must_match_smallest_level():
    with pytest.raises(ValueError):
        make_dual("msae", nesting_list=[8, 16])    # smallest != K=4


# --------------------------------------------------------------------------
# Warmup semantics + b_dec init for the new modes
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode,extra", [
    ("matryoshka", {"group_sizes": [4, 8, 20]}),
    ("msae", {"nesting_list": [4, 8, 16]}),
])
def test_warmup_gradient_flow(mode, extra):
    """Poly frozen (warmup): the multi-level loss must reach W_enc/W_dec and
    not the poly branch — mirroring how legacy duals warm up."""
    torch.manual_seed(5)
    model = make_dual(mode, **extra).train()
    for p in model.poly.parameters():
        p.requires_grad = False
    x = torch.randn(8, D)
    _, acts, recons = model(x)
    loss, _ = model.compute_loss(x, acts, recons, span_coef=0.0)
    loss.backward()
    assert model.W_enc.grad is not None and model.W_enc.grad.abs().sum() > 0
    assert model.W_dec.grad is not None and model.W_dec.grad.abs().sum() > 0
    for p in model.poly.parameters():
        assert p.grad is None


@pytest.mark.parametrize("mode,extra", [
    ("matryoshka", {"group_sizes": [4, 8, 20]}),
    ("msae", {"nesting_list": [4, 8, 16]}),
])
def test_new_modes_b_dec_init_guarded(mode, extra):
    torch.manual_seed(6)
    model = make_dual(mode, **extra)
    x1, x2 = torch.randn(64, D), torch.randn(64, D) + 9.0
    model.maybe_init_b_dec(x1)
    after_first = model.b_dec.data.clone()
    assert not torch.allclose(after_first, torch.zeros(D))  # actually moved
    model.maybe_init_b_dec(x2)
    assert torch.equal(model.b_dec.data, after_first)       # guarded no-op
