#!/usr/bin/env python3
"""Correctness tests for DualSAE(sparsity_mode='gated') -- the Dual-Gated arm.

Three things are pinned here:
  1. the gated mono branch IS the GatedSAE model (reduction test), so the
     activation pair Gated-SAE <-> Dual-Gated-SAE differs in exactly one
     thing: the poly branch;
  2. adding the mode did not perturb ANY legacy dual checkpoint layout;
  3. the post-warm-up optimizer rebuild picks the new parameters up -- the
     failure mode that silently freezes a head halfway through a 50-epoch run;
  4. the opt-in --dual_train_threshold_post_warmup: off (default) keeps the default freeze of
     jumprelu.log_threshold bit-exact, on keeps it training after warm-up,
     and it never leaks into a default run's serialized config.

CPU-only and tiny. Run:  pytest tests/test_dual_gated.py -q
"""

import sys
from pathlib import Path

import pytest
import torch

SAE_DIR = Path(__file__).resolve().parents[1]
if str(SAE_DIR) not in sys.path:
    sys.path.insert(0, str(SAE_DIR))

from models.dual_sae import DualSAE  # noqa: E402
from models.gated_sae import GatedSAE  # noqa: E402
import train_dual  # noqa: E402

torch.set_num_threads(2)

D, N, K = 16, 32, 4
LAMBDA = 0.07
GROUPS = [4, 8, 20]
NESTING = [4, 8, 16]

LEGACY_MODES = {
    "topk": {}, "batch_topk": {}, "vanilla": {}, "jump_relu": {},
    "matryoshka": {"group_sizes": GROUPS},
    "msae": {"nesting_list": NESTING, "relative_importance": [3.0, 2.0, 1.0],
             "soft_cap": 30.0},
}


def dual_cfg(mode, **extra):
    cfg = {"act_size": D, "dict_size": N, "top_k": K, "l1_coeff": LAMBDA,
           "bandwidth": 1e-3, "n_batches_to_dead": 3, "seed": 0,
           "device": "cpu", "dtype": torch.float32, "input_unit_norm": False,
           "sparsity_mode": mode, "poly_hidden": 8, "poly_layers": 1}
    cfg.update(LEGACY_MODES.get(mode, {}))
    cfg.update(extra)
    return cfg


def gated_cfg(**extra):
    cfg = {"model_type": "gated", "act_size": D, "dict_size": N,
           "l1_coeff": LAMBDA, "sparsity_warmup_steps": 0,
           "n_batches_to_dead": 3, "seed": 0, "device": "cpu",
           "dtype": torch.float32, "input_unit_norm": False}
    cfg.update(extra)
    return cfg


def test_dual_gated_reduces_to_gated_sae():
    """With the poly branch zeroed (r_hat == 0), span/sim/mono off, and
    identical weights, Dual-Gated must be the SAME MODEL as GatedSAE.

    The two losses differ by a single GLOBAL factor D, and by nothing else:
        dual_loss == gated_loss / D
    exactly (at span_coef=0). Adam is invariant to a global loss scale, so
    under this relation the dual's mono branch trains as the reference model at
    the SAME lambda -- which is the whole point of the pair.

    The alternative relation dual_loss == (gated_loss - l1) / D + l1 would
    divide the reconstruction and aux terms by D but not the L1, i.e. the dual
    would behave as the reference at lambda_eff = lambda * D. The two forces on
    the Heaviside gate are L_sparse (closes) and L_aux (opens); dividing only
    the opening force by D closes every gate, and a shut gate is absorbing
    because both terms see ReLU(pi_gate)=0. The physical invariant -- lambda
    means the same thing in both arms -- is pinned separately below.
    """
    torch.manual_seed(0)
    dual = DualSAE(dual_cfg("gated", sparsity_warmup_steps=0))
    gated = GatedSAE(gated_cfg())

    with torch.no_grad():
        # Randomize so a wiring bug cannot hide behind zero-initialised biases.
        for m in (dual,):
            m.b_enc.copy_(torch.randn(N) * 0.3)
            m.mag_bias.copy_(torch.randn(N) * 0.3)
            m.r_mag.copy_(torch.randn(N) * 0.3)
            m.b_dec.copy_(torch.randn(D) * 0.3)
        for name in ("W_enc", "W_dec", "b_enc", "b_dec", "r_mag", "mag_bias"):
            getattr(gated, name).copy_(getattr(dual, name))
        # Zero the poly branch so r_hat == 0 (GELU(0) == 0 for any depth).
        for p_ in dual.poly.parameters():
            p_.zero_()

    x = torch.randn(11, D)
    dual.train(); gated.train()

    d_pre, d_acts, d_rec = dual(x)
    assert torch.allclose(dual._r_hat, torch.zeros_like(dual._r_hat), atol=1e-7)
    d_loss, d_metrics = dual.compute_loss(
        x, d_pre, d_rec, span_coef=0.0, active_only_span=True,
        sim_coef=0.0, mono_coef=0.0)

    g_pre, g_acts, g_rec = gated(x)
    g_loss, g_metrics = gated.compute_loss(x, g_pre, g_acts, g_rec)

    # Same code, same reconstruction.
    assert torch.allclose(d_acts, g_acts, atol=1e-6), (d_acts - g_acts).abs().max()
    assert torch.allclose(d_rec, g_rec, atol=1e-6)
    # Same L1 (shared reduction). Tolerance is RELATIVE: the two paths use
    # different but mathematically identical reductions over a non-negative
    # tensor (torch.linalg.norm(ord=1) vs .abs().sum(-1)), which differ in
    # float32 summation order by ~1e-7 relative on a quantity of order 10.
    assert d_metrics["gated_l1"] == pytest.approx(g_metrics["gated_l1"], rel=1e-6)
    # And the exact loss relation: one global factor D, nothing else.
    assert d_loss.item() == pytest.approx(g_loss.item() / D, rel=1e-6)


def test_lambda_means_the_same_thing_in_both_arms():
    """THE invariant the collapse violated.

    A pair comparison at 'the same lambda' is only meaningful if lambda exerts
    the same relative pressure in both arms. That is a statement about the
    RATIO of the sparsity term to the reconstruction+aux terms -- the quantity
    that actually sets where the gate settles -- and it is invariant to any
    global rescaling of either loss.

    If the dual reduced its reconstruction/aux terms per element but kept the
    L1 summed over units, this ratio would be D times larger in the dual, and
    the dual at lambda would behave as the plain model at lambda*D -- far past
    the point where the identical architecture closes every gate.
    """
    torch.manual_seed(3)
    dual = DualSAE(dual_cfg("gated", sparsity_warmup_steps=0))
    gated = GatedSAE(gated_cfg())
    with torch.no_grad():
        dual.b_enc.copy_(torch.randn(N) * 0.3)
        dual.mag_bias.copy_(torch.randn(N) * 0.3)
        dual.r_mag.copy_(torch.randn(N) * 0.3)
        dual.b_dec.copy_(torch.randn(D) * 0.3)
        for name in ("W_enc", "W_dec", "b_enc", "b_dec", "r_mag", "mag_bias"):
            getattr(gated, name).copy_(getattr(dual, name))
        for p_ in dual.poly.parameters():
            p_.zero_()
    x = torch.randn(13, D)
    dual.train(); gated.train()

    d_pre, _, d_rec = dual(x)
    d_loss, dm = dual.compute_loss(x, d_pre, d_rec, span_coef=0.0,
                                   active_only_span=True, sim_coef=0.0,
                                   mono_coef=0.0)
    g_pre, g_acts, g_rec = gated(x)
    g_loss, gm = gated.compute_loss(x, g_pre, g_acts, g_rec)

    # Sparsity as a FRACTION of the whole objective must match.
    d_frac = (LAMBDA * dm["gated_l1"] / D) / d_loss.item()
    g_frac = (LAMBDA * gm["gated_l1"]) / g_loss.item()
    assert d_frac == pytest.approx(g_frac, rel=1e-6), (
        f"lambda exerts {d_frac / g_frac:.1f}x the relative pressure in the "
        f"dual arm; at D={D} the pre-fix bug gave exactly D")


def test_gated_aux_excludes_the_poly_branch():
    """[GATED-DUAL-AUX]: the auxiliary term must be a constraint on the GATE
    alone. If poly leaked into it, poly could rescue a degenerate gate and the
    term would stop constraining anything -- and its meaning would jump
    discontinuously when warm-up ends and poly unfreezes."""
    torch.manual_seed(1)
    dual = DualSAE(dual_cfg("gated", sparsity_warmup_steps=0))
    x = torch.randn(9, D)

    def aux():
        pre, acts, rec = dual(x)
        _, m = dual.compute_loss(x, pre, rec, span_coef=0.0, sim_coef=0.0,
                                 mono_coef=0.0)
        return m["gated_aux_loss"]

    before = aux()
    with torch.no_grad():
        for p_ in dual.poly.parameters():
            p_.add_(torch.randn_like(p_))
    assert aux() == pytest.approx(before, abs=1e-9)


def test_gated_aux_is_zero_for_every_legacy_mode():
    """The new term is added unconditionally to the total loss, so it must be
    exactly 0.0 for legacy modes or every other dual mode changes."""
    x = torch.randn(7, D)
    for mode in LEGACY_MODES:
        torch.manual_seed(2)
        dual = DualSAE(dual_cfg(mode))
        pre, acts, rec = dual(x)
        loss_a, m = dual.compute_loss(x, pre, rec, span_coef=0.01,
                                      sim_coef=0.0, mono_coef=0.0)
        assert "gated_aux_loss" not in m, mode
        # Recompute the total by hand from the logged pieces: l2 + span + sparsity.
        by_hand = (m["l2_loss"] if mode not in ("matryoshka", "msae")
                   else None)
        if by_hand is not None:
            assert loss_a.item() == pytest.approx(
                by_hand + 0.01 * m["span_loss"] + m["sparsity_penalty"],
                rel=1e-6), mode


def test_legacy_dual_state_dicts_gain_no_keys():
    """Regression: the gated tensors must be registered ONLY in gated mode, or
    every pre-existing dual checkpoint stops strict-loading."""
    gated_keys = {"r_mag", "mag_bias", "sparsity_warmup_steps", "_gated_step"}
    for mode in LEGACY_MODES:
        torch.manual_seed(3)
        model = DualSAE(dual_cfg(mode))
        keys = set(model.state_dict())
        assert not (keys & gated_keys), (mode, keys & gated_keys)
        # ...and a fresh instance strict-loads the saved dict.
        DualSAE(dual_cfg(mode)).load_state_dict(model.state_dict(), strict=True)

    gated = DualSAE(dual_cfg("gated"))
    assert gated_keys <= set(gated.state_dict())


def test_post_warmup_rebuild_covers_the_gated_parameters():
    """THE failure mode this refactor exists to prevent: the rebuild replaces
    the optimizer wholesale, so a parameter missing from it stops training at
    warm-up end with no error and a plausible loss curve."""
    dual = DualSAE(dual_cfg("gated"))
    mono, poly = train_dual.build_post_warmup_param_groups(dual, 1e-4, 1e-5)
    covered = {id(p) for g in mono + poly for p in g["params"]}
    by_id = {id(p): n for n, p in dual.named_parameters()}
    missing = sorted(by_id[i] for i in set(by_id) - covered)
    assert missing == [], missing
    # No parameter may sit in two groups (Adam would step it twice).
    flat = [id(p) for g in mono + poly for p in g["params"]]
    assert len(flat) == len(set(flat))


@pytest.mark.parametrize("mode", sorted(LEGACY_MODES))
def test_post_warmup_rebuild_legacy_coverage_is_declared(mode):
    """Legacy modes keep their EXACT historical coverage. Anything left out
    must be declared in POST_WARMUP_EXCLUDED_PARAMS with a reason, so the
    omission is visible instead of being an accident of an explicit list.

    The rebuild takes an opt-in train_threshold flag. Its DEFAULT must be the
    standard coverage, so this is asserted both
    with the argument omitted (every pre-existing caller) and with an explicit
    train_threshold=False (what train_dual passes for a default run), and the
    two must produce the same groups."""
    dual = DualSAE(dual_cfg(mode))
    mono, poly = train_dual.build_post_warmup_param_groups(dual, 1e-4, 1e-5)
    mono_f, poly_f = train_dual.build_post_warmup_param_groups(
        dual, 1e-4, 1e-5, train_threshold=False)
    as_ids = lambda groups: [([id(p) for p in g["params"]], g["lr"]) for g in groups]
    assert as_ids(mono) == as_ids(mono_f) and as_ids(poly) == as_ids(poly_f), mode
    covered = {id(p) for g in mono + poly for p in g["params"]}
    by_id = {id(p): n for n, p in dual.named_parameters()}
    missing = sorted(by_id[i] for i in set(by_id) - covered)
    for name in missing:
        assert name in train_dual.POST_WARMUP_EXCLUDED_PARAMS, (mode, name)
    if mode == "jump_relu":
        # Pin the known one so a future "cleanup" that starts optimizing it
        # has to do so deliberately (it would change the default recipe).
        assert "jumprelu.log_threshold" in missing


def test_post_warmup_rebuild_opt_in_trains_the_threshold():
    """--dual_train_threshold_post_warmup (train_threshold=True): the JumpReLU
    threshold joins the MONO group at mono_lr (the rate it had in warm-up and
    the baseline's rate), nothing else changes, nothing is missing, and no
    parameter is listed twice."""
    dual = DualSAE(dual_cfg("jump_relu"))
    off_mono, off_poly = train_dual.build_post_warmup_param_groups(dual, 1e-4, 1e-5)
    mono, poly = train_dual.build_post_warmup_param_groups(
        dual, 1e-4, 1e-5, train_threshold=True)
    lt = dual.jumprelu.log_threshold
    assert train_dual.POST_WARMUP_OPT_IN_THRESHOLD_PARAM == "jumprelu.log_threshold"
    # Exactly one extra group, appended LAST to the mono groups, holding only the threshold.
    assert len(mono) == len(off_mono) + 1
    assert [p is lt for p in mono[-1]["params"]] == [True]
    assert mono[-1]["lr"] == 1e-4                     # mono_lr, not poly_lr
    assert all(id(p) != id(lt) for g in poly for p in g["params"])
    covered = {id(p) for g in mono + poly for p in g["params"]}
    by_id = {id(p): n for n, p in dual.named_parameters()}
    assert sorted(by_id[i] for i in set(by_id) - covered) == []
    flat = [id(p) for g in mono + poly for p in g["params"]]
    assert len(flat) == len(set(flat))


@pytest.mark.parametrize("mode", sorted(set(LEGACY_MODES) - {"jump_relu"}) + ["gated"])
def test_post_warmup_rebuild_opt_in_refused_without_a_threshold(mode):
    """A mode with no JumpReLU threshold must REFUSE the opt-in flag rather
    than silently ignore it (a run would otherwise claim a setting it did not
    have)."""
    dual = DualSAE(dual_cfg(mode))
    with pytest.raises(ValueError, match="JumpReLU"):
        train_dual.build_post_warmup_param_groups(dual, 1e-4, 1e-5,
                                                  train_threshold=True)


def _threshold_trace(train_threshold, warm_steps=3, post_steps=4):
    """Mimic train_dual's warm-up -> rebuild -> joint-training sequence on a
    tiny jump_relu DualSAE and return (lt_at_init, lt_at_rebuild, lt_final,
    W_enc_at_rebuild, W_enc_final).

    Same order of operations as train_dual.train_autoencoder: poly frozen and
    Adam(all params) during warm-up; at warm-up end poly unfrozen and a NEW
    Adam built from build_post_warmup_param_groups; each step = forward,
    compute_loss(x, acts, recons), zero_grad, backward, decoder-column
    normalisation, step. bandwidth 0.5 and theta 0.1 put many pre-activations
    inside the straight-through window |x - theta| < bandwidth/2, so the
    threshold gets a non-zero gradient on every step (with a 1e-3 bandwidth a
    64-row batch may give it none, which would make "stays constant" vacuous)."""
    torch.manual_seed(11)
    dual = DualSAE(dual_cfg("jump_relu", bandwidth=0.5, jump_threshold_init=0.1))
    x = torch.randn(64, D)
    lt = dual.jumprelu.log_threshold
    lt0 = lt.detach().clone()
    for p in dual.poly.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(dual.parameters(), lr=1e-2)

    def step():
        pre, acts, rec = dual(x)
        loss, _ = dual.compute_loss(x, acts, rec, span_coef=0.01, sim_coef=0.0,
                                    mono_coef=0.0)
        opt.zero_grad()
        loss.backward()
        dual.normalize_decoder_columns_()
        opt.step()

    for _ in range(warm_steps):
        step()
    for p in dual.poly.parameters():
        p.requires_grad = True
    mono, poly = train_dual.build_post_warmup_param_groups(
        dual, 1e-2, 1e-2, train_threshold=train_threshold)
    opt = torch.optim.Adam(mono + poly)
    lt_w, wenc_w = lt.detach().clone(), dual.W_enc.detach().clone()
    for _ in range(post_steps):
        step()
    return lt0, lt_w, lt.detach().clone(), wenc_w, dual.W_enc.detach().clone()


def test_threshold_frozen_after_warmup_by_default_and_trained_with_opt_in():
    """THE behavioural contract of --dual_train_threshold_post_warmup.

    Default recipe: log_threshold trains during warm-up, then is
    BIT-constant for every post-warm-up step while the rest of the mono branch
    keeps training. Opt-in: it keeps moving after warm-up. Both runs are
    identical up to the rebuild (same seed, same data), so the only cause of
    the difference is the flag."""
    lt0, lt_w, lt_end, w_w, w_end = _threshold_trace(train_threshold=False)
    assert not torch.equal(lt0, lt_w), "threshold must train during warm-up"
    assert torch.equal(lt_w, lt_end), "default: threshold must be bit-constant after warm-up"
    assert not torch.equal(w_w, w_end), "sanity: the mono encoder still trains after warm-up"

    lt0b, lt_wb, lt_endb, _, _ = _threshold_trace(train_threshold=True)
    assert torch.equal(lt0, lt0b) and torch.equal(lt_w, lt_wb), \
        "the flag must not change anything before the rebuild"
    assert not torch.equal(lt_wb, lt_endb), "opt-in: threshold must keep training after warm-up"
    moved = (lt_endb - lt_wb).abs()
    assert float(moved.max()) > 1e-4, float(moved.max())


def test_opt_in_flag_is_serialized_only_when_set():
    """training_config.json / checkpoint 'config' / W&B are all written via
    utils.config_payload: a default run must not gain the new key (every
    pre-existing payload keeps its key set), a flagged run must record it.
    The bool default is compared type-exactly, so only a real False is
    dropped."""
    from utils import config_payload
    off = train_dual.TrainingConfig(train_path="x", sparsity_mode="jump_relu")
    assert off.dual_train_threshold_post_warmup is False
    pay = config_payload(off)
    assert "dual_train_threshold_post_warmup" not in pay
    assert "jump_threshold_init" not in pay                  # the step-0 rule still holds
    assert "dual_train_threshold_post_warmup" in vars(off)   # payload is a copy, not a mutation
    on = train_dual.TrainingConfig(train_path="x", sparsity_mode="jump_relu",
                                   jump_threshold_init=1.0,
                                   dual_train_threshold_post_warmup=True)
    pay = config_payload(on)
    assert pay["dual_train_threshold_post_warmup"] is True
    assert pay["jump_threshold_init"] == 1.0
    # A non-bool value is never mistaken for the unset default (0 == False in Python).
    odd = train_dual.TrainingConfig(train_path="x", dual_train_threshold_post_warmup=0)
    assert "dual_train_threshold_post_warmup" in config_payload(odd)


def test_main_refuses_the_opt_in_flag_outside_dual_jump_relu(monkeypatch, capsys):
    """main.py must reject --dual_train_threshold_post_warmup at argparse
    time for every model / sparsity_mode that has no Dual JumpReLU threshold
    (it would otherwise be accepted and silently ignored)."""
    import main as sae_main
    for extra in (["--model", "jumprelu"], ["--model", "topk"],
                  ["--model", "dual", "--sparsity_mode", "topk"],
                  ["--model", "dual", "--sparsity_mode", "batch_topk"]):
        monkeypatch.setattr(sys, "argv", ["main.py", *extra,
                                          "--dual_train_threshold_post_warmup"])
        with pytest.raises(SystemExit) as ei:
            sae_main.main()
        assert ei.value.code == 2, extra
        assert "--dual_train_threshold_post_warmup applies only" in capsys.readouterr().err


def test_gated_sparsity_ramp_is_train_only():
    """Validation batches must not advance the ramp."""
    dual = DualSAE(dual_cfg("gated", sparsity_warmup_steps=100))
    x = torch.randn(5, D)
    dual.eval()
    for _ in range(3):
        pre, acts, rec = dual(x)
        dual.compute_loss(x, pre, rec, span_coef=0.0, sim_coef=0.0, mono_coef=0.0)
    assert int(dual._gated_step.item()) == 0
    dual.train()
    for _ in range(3):
        pre, acts, rec = dual(x)
        dual.compute_loss(x, pre, rec, span_coef=0.0, sim_coef=0.0, mono_coef=0.0)
    assert int(dual._gated_step.item()) == 3


def test_dead_neuron_reinit_resets_the_magnitude_head():
    """A respawned latent that kept a stale mag_bias would be born saturated
    (or permanently shut) no matter how good its fresh direction is."""
    torch.manual_seed(4)
    dual = DualSAE(dual_cfg("gated"))
    with torch.no_grad():
        dual.r_mag.fill_(3.0)
        dual.mag_bias.fill_(-5.0)
    dead = torch.zeros(N, dtype=torch.bool)
    dead[[1, 7, 30]] = True
    train_dual.reinitialize_dead_neurons(dual, dead, torch.randn(8, D), "cpu")
    assert torch.all(dual.r_mag[dead] == 0.0)
    assert torch.all(dual.mag_bias[dead] == 0.0)
    assert torch.all(dual.r_mag[~dead] == 3.0)      # untouched elsewhere
