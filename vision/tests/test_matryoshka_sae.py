#!/usr/bin/env python3
"""Correctness tests for models/matryoshka_sae.py (Bussmann port).

The load-bearing test is LOSS PARITY against the vendored reference
implementation (saprmarks/dictionary_learning): identical weights +
identical inputs must give the identical Matryoshka reconstruction loss and
the identical AuxK loss. Everything else pins the gate arithmetic, the
train-only threshold EMA, and the buffer-guarded b_dec init.

CPU-only and tiny by design (runs on a login node in seconds).
Run:  pytest tests/test_matryoshka_sae.py -q
"""

import sys
import types
from pathlib import Path

import pytest
import torch

# The vision modules use flat imports (from loss import ...) — put vision/ itself on
# the path, mirroring how the trainers run.
SAE_DIR = Path(__file__).resolve().parents[1]
if str(SAE_DIR) not in sys.path:
    sys.path.insert(0, str(SAE_DIR))

from models.matryoshka_sae import MatryoshkaSAE, geometric_median  # noqa: E402

torch.set_num_threads(2)  # login-node CPU-time ulimit safety

D, N, K = 16, 32, 4
GROUP_SIZES = [4, 8, 20]                      # sums to N
GROUP_FRACTIONS = [4 / 32, 8 / 32, 20 / 32]   # the same split as fractions


def make_model(seed=0, n_batches_to_dead=3):
    return MatryoshkaSAE({
        "model_type": "matryoshka",
        "act_size": D, "dict_size": N, "top_k": K,
        "group_sizes": GROUP_SIZES,
        "l1_coeff": 0.0, "aux_penalty": 1 / 32,
        "n_batches_to_dead": n_batches_to_dead,
        "seed": seed, "device": "cpu", "dtype": torch.float32,
        "input_unit_norm": False,
    })


def _import_vendored():
    """Import the vendored dictionary_learning matryoshka trainer. einops is
    imported at module scope upstream but NOT used on any code path this test
    exercises (SAE ctor, encode, trainer ctor, trainer.loss, geometric
    median) — stub it if the venv lacks it rather than skipping the parity
    test that is the whole point of this file."""
    # Parity tests against the reference implementation (saprmarks/dictionary_learning).
    # Point DICTIONARY_LEARNING_PARENT at a directory that contains a checkout of the
    # `dictionary_learning` package; without it the parity tests are skipped.
    import os
    libs = os.environ.get("DICTIONARY_LEARNING_PARENT")
    if not libs:
        pytest.skip("DICTIONARY_LEARNING_PARENT not set (reference implementation unavailable)")
    if str(libs) not in sys.path:
        sys.path.insert(0, str(libs))
    try:
        import einops  # noqa: F401
    except ImportError:
        sys.modules["einops"] = types.ModuleType("einops")
    try:
        from dictionary_learning.trainers.matryoshka_batch_top_k import (
            MatryoshkaBatchTopKTrainer)
    except Exception as exc:  # missing transitive dep in this venv
        pytest.skip(f"vendored dictionary_learning unavailable: {exc}")
    return MatryoshkaBatchTopKTrainer


def make_vendored_trainer(model):
    trainer_cls = _import_vendored()
    trainer = trainer_cls(
        steps=10, activation_dim=D, dict_size=N, k=K, layer=0,
        lm_name="parity-test", group_fractions=GROUP_FRACTIONS,
        lr=1e-4, warmup_steps=1,  # its lr schedule asserts warmup < steps
        device="cpu", seed=0)
    assert trainer.group_sizes == GROUP_SIZES  # same fraction arithmetic
    # Copy OUR weights into the vendored model (identical param shapes).
    with torch.no_grad():
        trainer.ae.W_enc.data.copy_(model.W_enc.data)
        trainer.ae.W_dec.data.copy_(model.W_dec.data)
        trainer.ae.b_enc.data.copy_(model.b_enc.data)
        trainer.ae.b_dec.data.copy_(model.b_dec.data)
    return trainer


def test_loss_parity_no_dead():
    """Same weights + same batch -> the vendored trainer's l2_loss equals our
    matryoshka reconstruction term exactly (both use the vendored reduction:
    weighted per-prefix sum(-1).mean(), mean over groups)."""
    torch.manual_seed(1)
    model = make_model().eval()  # eval: keep the threshold EMA out of parity
    trainer = make_vendored_trainer(model)
    x = torch.randn(8, D)

    acts, latents, recons = model(x)
    loss, metrics = model.compute_loss(x, acts, latents, recons)
    log = trainer.loss(x, step=0, logging=True)

    assert metrics["matryoshka_loss"] == pytest.approx(
        log.losses["l2_loss"], rel=1e-6)
    # No dead features on either side -> aux is 0 and totals match too.
    assert metrics["auxk_loss"] == 0.0
    assert log.losses["auxk_loss"] == 0.0
    assert loss.item() == pytest.approx(log.losses["loss"], rel=1e-6)
    # And the gated codes themselves agree.
    f_vendored = trainer.ae.encode(x, use_threshold=False)
    assert torch.allclose(latents, f_vendored, atol=1e-6)


def test_auxk_parity_forced_dead():
    """Force the SAME dead set on both sides -> identical normalized AuxK."""
    torch.manual_seed(2)
    model = make_model(n_batches_to_dead=3).eval()
    trainer = make_vendored_trainer(model)
    x = torch.randn(8, D)

    acts, latents, recons = model(x)
    # Mark only latents that did NOT fire in this batch as dead — the
    # vendored loss() zeroes the counter of anything that fired (revival),
    # while our counter was set after the forward; a fired "dead" latent
    # would therefore be dead on our side only and break parity for a
    # reason that never occurs in real training.
    fired = (latents != 0).any(dim=0)
    dead_idx = torch.where(~fired)[0][:10]
    assert len(dead_idx) >= 4, "test needs some unfired latents"
    model.num_batches_not_active[:] = 0
    model.num_batches_not_active[dead_idx] = 3     # >= n_batches_to_dead
    trainer.num_tokens_since_fired[:] = 0
    trainer.num_tokens_since_fired[dead_idx] = 10_000_000

    _, metrics = model.compute_loss(x, acts, latents, recons)
    log = trainer.loss(x, step=0, logging=True)
    assert metrics["auxk_loss"] > 0
    assert metrics["auxk_loss"] == pytest.approx(log.losses["auxk_loss"],
                                                 rel=1e-6)


def test_batch_topk_gate_count():
    torch.manual_seed(3)
    model = make_model().eval()
    x = torch.randn(8, D)
    _, latents, _ = model(x)
    # Batch top-k keeps exactly k*B nonzeros over the whole batch (per-sample
    # counts may vary — that is the point of the batch variant).
    assert int((latents != 0).sum()) == K * x.shape[0]


def test_threshold_ema_train_only():
    torch.manual_seed(4)
    model = make_model()
    x = torch.randn(8, D)
    model.eval()
    for _ in range(3):
        model(x)
    assert model.threshold.item() == -1.0          # eval never touches it
    assert model._step.item() == 0                 # nor the step counter
    model.train()
    model(x)
    assert model._step.item() == 1
    # Still -1: the EMA only starts after the vendored start step (1000).
    assert model.threshold.item() == -1.0
    model._step.fill_(2000)
    model(x)
    assert model.threshold.item() > 0              # first EMA write happened


def test_b_dec_init_idempotent():
    torch.manual_seed(5)
    model = make_model()
    x1, x2 = torch.randn(64, D), torch.randn(64, D) + 5.0
    model.maybe_init_b_dec(x1)
    expected = geometric_median(x1)
    assert torch.allclose(model.b_dec.data, expected, atol=1e-5)
    model.maybe_init_b_dec(x2)                     # guarded no-op
    assert torch.allclose(model.b_dec.data, expected, atol=1e-5)
    assert model._b_dec_initialized.item() == 1


def test_group_sizes_must_sum():
    with pytest.raises(ValueError):
        MatryoshkaSAE({
            "act_size": D, "dict_size": N, "top_k": K,
            "group_sizes": [4, 8, 8],  # sums to 20 != 32
            "l1_coeff": 0.0, "n_batches_to_dead": 3, "seed": 0,
            "device": "cpu", "dtype": torch.float32,
            "input_unit_norm": False,
        })
