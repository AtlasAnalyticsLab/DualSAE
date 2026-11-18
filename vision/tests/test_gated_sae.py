#!/usr/bin/env python3
"""Correctness tests for models/gated_sae.py (Rajamanoharan et al. port).

The load-bearing test is LOSS PARITY against the vendored reference
(saprmarks/dictionary_learning): identical weights + identical input
must give the identical gated loss, term by term. Everything else pins the
two-head arithmetic, the ReLU-not-Heaviside gate used by the L1/aux terms,
the train-only sparsity ramp, and the vendored weight init.

CPU-only and tiny by design (runs on a login node in seconds).
Run:  pytest tests/test_gated_sae.py -q
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

from models.gated_sae import GatedSAE  # noqa: E402

torch.set_num_threads(2)  # login-node CPU-time ulimit safety

D, N = 16, 32
LAMBDA = 0.07


def make_model(seed=0, n_batches_to_dead=3, sparsity_warmup_steps=0):
    return GatedSAE({
        "model_type": "gated",
        "act_size": D, "dict_size": N,
        "l1_coeff": LAMBDA,
        "sparsity_warmup_steps": sparsity_warmup_steps,
        "n_batches_to_dead": n_batches_to_dead,
        "seed": seed, "device": "cpu", "dtype": torch.float32,
        "input_unit_norm": False,
    })


def _import_vendored():
    """Import the vendored GatedAutoEncoder + GatedSAETrainer. einops is
    imported at module scope upstream but is not used on any path this test
    exercises — stub it if the venv lacks it rather than skipping the parity
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
        from dictionary_learning.dictionary import GatedAutoEncoder
        from dictionary_learning.trainers.gdm import GatedSAETrainer
    except Exception as exc:  # missing transitive dep in this venv
        pytest.skip(f"vendored dictionary_learning unavailable: {exc}")
    return GatedAutoEncoder, GatedSAETrainer


def make_vendored(sparsity_warmup_steps=None, steps=10_000):
    _, GatedSAETrainer = _import_vendored()
    return GatedSAETrainer(
        steps=steps, activation_dim=D, dict_size=N, layer=0, lm_name="test",
        lr=1e-4, l1_penalty=LAMBDA, warmup_steps=0,
        sparsity_warmup_steps=sparsity_warmup_steps, device="cpu",
    )


def sync_weights(ours: GatedSAE, trainer) -> None:
    """Copy OUR (randomly initialised, non-zero) weights onto the vendored
    model through the layout map. Biases are deliberately randomised first:
    every reference bias inits to zero, so a parity test run at the default
    init would pass even if b_enc/mag_bias were wired to the wrong head."""
    with torch.no_grad():
        ours.b_enc.copy_(torch.randn(N) * 0.3)      # gate_bias
        ours.mag_bias.copy_(torch.randn(N) * 0.3)
        ours.r_mag.copy_(torch.randn(N) * 0.3)
        ours.b_dec.copy_(torch.randn(D) * 0.3)
        ae = trainer.ae
        ae.encoder.weight.copy_(ours.W_enc.t())     # (U, d) <- (d, U).T
        ae.decoder.weight.copy_(ours.W_dec.t())     # (d, U) <- (U, d).T
        ae.gate_bias.copy_(ours.b_enc)
        ae.mag_bias.copy_(ours.mag_bias)
        ae.r_mag.copy_(ours.r_mag)
        ae.decoder_bias.copy_(ours.b_dec)


def test_forward_parity_with_vendored():
    """latents == ae.encode(x); recons == ae(x); pre_acts == ReLU(pi_gate)."""
    torch.manual_seed(0)
    ours = make_model()
    trainer = make_vendored()
    sync_weights(ours, trainer)
    x = torch.randn(7, D)

    ours.eval()
    pre_acts, latents, recons = ours(x)
    with torch.no_grad():
        f_ref, gate_ref = trainer.ae.encode(x, return_gate=True)
        x_hat_ref = trainer.ae(x)

    assert torch.allclose(latents, f_ref, atol=1e-6), (latents - f_ref).abs().max()
    assert torch.allclose(recons, x_hat_ref, atol=1e-6)
    assert torch.allclose(pre_acts, gate_ref, atol=1e-6)


def test_loss_parity_with_vendored():
    """The whole point of this file: identical total loss AND identical terms."""
    torch.manual_seed(1)
    ours = make_model(sparsity_warmup_steps=0)   # ramp off -> scale == 1.0
    trainer = make_vendored(sparsity_warmup_steps=None)  # -> scale_fn == 1.0
    sync_weights(ours, trainer)
    x = torch.randn(11, D)

    ours.train()
    pre_acts, latents, recons = ours(x)
    loss, metrics = ours.compute_loss(x, pre_acts, latents, recons)
    log = trainer.loss(x, step=0, logging=True)

    assert loss.item() == pytest.approx(log.losses["loss"], abs=1e-6)
    assert metrics["gated_recon_loss"] == pytest.approx(log.losses["mse_loss"], abs=1e-6)
    assert metrics["gated_l1"] == pytest.approx(log.losses["sparsity_loss"], abs=1e-6)
    assert metrics["gated_aux_loss"] == pytest.approx(log.losses["aux_loss"], abs=1e-6)


def test_gradient_parity_with_vendored():
    """Pins the ENTIRE detach structure at once: every parameter's gradient of
    the total loss must match the vendored trainer's. This is strictly stronger
    than spot-checking that L_aux detaches the decoder, and it catches the
    subtler direction too -- b_dec legitimately DOES receive gradient from
    L_aux through the encoder centering (x - b_dec), so a naive
    "assert b_dec grad is zero" test would be wrong in both implementations."""
    torch.manual_seed(2)
    ours = make_model(sparsity_warmup_steps=0)
    trainer = make_vendored(sparsity_warmup_steps=None)
    sync_weights(ours, trainer)
    x = torch.randn(13, D)

    ours.train()
    p_, l_, r_ = ours(x)
    loss, _ = ours.compute_loss(x, p_, l_, r_)
    ours.zero_grad(set_to_none=True)
    loss.backward()

    trainer.ae.zero_grad(set_to_none=True)
    trainer.loss(x, step=0).backward()
    ae = trainer.ae

    # (d,U) vs (U,d): compare through the same transpose the ctor used.
    pairs = [
        (ours.W_enc.grad, ae.encoder.weight.grad.t()),
        (ours.W_dec.grad, ae.decoder.weight.grad.t()),
        (ours.b_enc.grad, ae.gate_bias.grad),
        (ours.mag_bias.grad, ae.mag_bias.grad),
        (ours.r_mag.grad, ae.r_mag.grad),
        (ours.b_dec.grad, ae.decoder_bias.grad),
    ]
    for i, (mine, ref) in enumerate(pairs):
        assert mine is not None and ref is not None, i
        assert torch.allclose(mine, ref, atol=1e-6), (i, (mine - ref).abs().max())


def test_aux_term_does_not_reshape_the_dictionary():
    """The decoder WEIGHT is detached in L_aux (gdm.py:80), so L_aux alone
    contributes no gradient to W_dec -- it can only be satisfied by making
    ReLU(pi_gate) informative, never by moving the atoms."""
    torch.manual_seed(2)
    ours = make_model(sparsity_warmup_steps=0)
    x = torch.randn(9, D)
    ours.train()
    pre_acts, _, _ = ours(x)
    x_pre = ours._x_preprocessed.float()
    l_aux = (x_pre - (pre_acts.float() @ ours.W_dec.detach().float()
                      + ours.b_dec.detach().float())).pow(2).sum(-1).mean()
    g_dec, = torch.autograd.grad(l_aux, [ours.W_dec], allow_unused=True,
                                 retain_graph=True)
    assert g_dec is None or torch.count_nonzero(g_dec) == 0


def test_gate_is_relu_not_binary():
    """L_sparse/L_aux see ReLU(pi_gate), not the 0/1 mask. Scaling the gate
    pre-activation without flipping any sign must change both terms; with a
    binary mask they would be invariant."""
    torch.manual_seed(3)
    ours = make_model(sparsity_warmup_steps=0)
    x = torch.randn(9, D)

    def terms():
        p, l, r = ours(x)
        _, m = ours.compute_loss(x, p, l, r)
        return m["gated_l1"], m["gated_aux_loss"]

    ours.eval()
    before = terms()
    with torch.no_grad():
        # Scale the whole encoder: every pi_gate sign is preserved, so the
        # binary mask is bit-identical while ReLU(pi_gate) is not.
        ours.W_enc.mul_(2.0)
        ours.b_enc.mul_(2.0)
    after = terms()
    assert after[0] != pytest.approx(before[0], rel=1e-3)
    assert after[1] != pytest.approx(before[1], rel=1e-3)


def test_sparsity_warmup_matches_vendored_schedule():
    """_sparsity_scale() == get_sparsity_warmup_fn(steps, warm)(step)."""
    _import_vendored()
    from dictionary_learning.trainers.trainer import get_sparsity_warmup_fn
    warm = 2000
    ref = get_sparsity_warmup_fn(10_000, warm)
    ours = make_model(sparsity_warmup_steps=warm)
    for step in (0, 1, 999, 1999, 2000, 5000):
        ours._step.fill_(step)
        assert ours._sparsity_scale() == pytest.approx(ref(step))
    # ramp disabled -> constant 1.0, matching scale_fn's `if not warmup`
    off = make_model(sparsity_warmup_steps=0)
    off._step.fill_(7)
    assert off._sparsity_scale() == 1.0


def test_step_advances_only_in_training_mode():
    """evaluate() calls compute_loss in eval mode; those calls must not move
    the ramp, or the schedule desynchronizes from the optimizer step count."""
    torch.manual_seed(4)
    ours = make_model(sparsity_warmup_steps=100)
    x = torch.randn(5, D)

    ours.eval()
    for _ in range(3):
        p, l, r = ours(x)
        ours.compute_loss(x, p, l, r)
    assert int(ours._step.item()) == 0

    ours.train()
    for _ in range(3):
        p, l, r = ours(x)
        ours.compute_loss(x, p, l, r)
    assert int(ours._step.item()) == 3


def test_vendored_init_layout():
    """Unit-norm ATOMS (rows of (U,d) W_dec), encoder tied to the transpose,
    all four biases zero — dictionary.py::_reset_parameters."""
    ours = make_model()
    norms = ours.W_dec.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(N), atol=1e-6)
    assert torch.allclose(ours.W_enc, ours.W_dec.t(), atol=1e-6)
    for name in ("b_dec", "b_enc", "r_mag", "mag_bias"):
        assert torch.count_nonzero(getattr(ours, name)) == 0, name


def test_decoder_unit_norm_drift_is_lr_bounded():
    """FLAG G1: project-then-step (ours) instead of project-step-renormalize
    (ConstrainedAdam) leaves the SAVED atoms off unit norm by O(lr*||g||).
    Asserting a magic tolerance would be brittle, so pin the MECHANISM: the
    drift must be small AND must shrink with the learning rate. A regression
    that drops the projection, or normalizes the wrong axis, breaks both."""
    def drift_at(lr):
        torch.manual_seed(5)
        ours = make_model(sparsity_warmup_steps=0)
        opt = torch.optim.Adam(ours.parameters(), lr=lr, betas=(0.0, 0.999))
        x = torch.randn(32, D)
        ours.train()
        p_, l_, r_ = ours(x)
        loss, _ = ours.compute_loss(x, p_, l_, r_)
        opt.zero_grad()
        loss.backward()
        ours.make_decoder_weights_and_grad_unit_norm()
        opt.step()
        return (ours.W_dec.norm(dim=-1) - 1.0).abs().max().item()

    d1 = drift_at(1e-4)
    d2 = drift_at(1e-5)
    assert d1 < 1e-3, d1                      # small in absolute terms
    assert d2 < d1 / 5, (d1, d2)              # and first-order in lr


def test_optimizer_overrides_accepted_by_trainer():
    """The hook must return exactly what train_other._adam_betas expects."""
    import train_other
    ov = train_other._optimizer_overrides(make_model())
    assert ov == {"betas": (0.0, 0.999)}
    assert train_other._adam_betas(ov) == (0.0, 0.999)


def test_metric_keys_the_trainer_reads():
    """train_other.py's epoch loop indexes these four by name."""
    torch.manual_seed(6)
    ours = make_model()
    x = torch.randn(5, D)
    p, l, r = ours(x)
    _, m = ours.compute_loss(x, p, l, r)
    for key in ("num_active", "l2_loss", "num_dead_features", "l1_sparsity"):
        assert key in m, key


def test_state_dict_fingerprint():
    """featbsf/adapters.py and knn_lp_sparse.load_run identify this class by
    the ('r_mag','mag_bias') pair; no other model in the repo registers them."""
    sd = make_model().state_dict()
    assert "r_mag" in sd and "mag_bias" in sd
    assert {"W_enc", "W_dec", "b_enc", "b_dec"} <= set(sd)
