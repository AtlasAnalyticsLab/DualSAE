#!/usr/bin/env python3
"""CPU smoke tests of the ablation-grid architectures on synthetic data: the projector options of
the Residual Branch (projector ablation), the standard-SAE arm S, the sequential-training arm F
(--freeze_sparse_from + --snapshot_steps) and the zero-residual warm-up arm Z.

Each test guards a failure that would otherwise be invisible in a run's output: a config key
dropped so the wrong architecture strict-loads, or a "frozen" tensor that is rewritten. Every run
goes through the real trainer (train_dual.train_autoencoder) or the real CLI (main.py) on a tiny
synthetic LMDB, on CPU, in seconds.

Run with:  python -m pytest vision/tests/test_abl_arms.py -q
"""

import json
import sys
from pathlib import Path

import lmdb
import numpy as np
import pytest
import torch
import torch.nn as nn

VISION_DIR = Path(__file__).resolve().parents[1]
if str(VISION_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_DIR))

import main as main_module  # noqa: E402
import train_dual  # noqa: E402
from knn_lp_sparse import forward_reps, load_run  # noqa: E402
from models import BatchTopKSAE, DualSAE, TopK, TopKSAE  # noqa: E402
from train_dual import TrainingConfig, sparse_state_sha256, train_autoencoder  # noqa: E402
from utils import config_payload  # noqa: E402

ACT, DICT, K = 16, 64, 4
BATCH, ROWS = 32, 256          # 8 steps per epoch


def dual_cfg(**extra):
    cfg = dict(act_size=ACT, dict_size=DICT, top_k=K, seed=0, device="cpu", dtype=torch.float32,
               sparsity_mode="topk", n_batches_to_dead=10)
    cfg.update(extra)
    return cfg


def write_lmdb(path: Path, rows: int, seed: int) -> str:
    """A tiny feature store in LMDBFeatureDataset's format. Rows are L2-normalized and carry
    low-rank structure, like the real embeddings, so reconstruction has something to learn."""
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(6, ACT))
    data = rng.normal(size=(rows, 6)) @ basis + 0.05 * rng.normal(size=(rows, ACT))
    data = (data / np.linalg.norm(data, axis=1, keepdims=True)).astype(np.float32)
    env = lmdb.open(str(path), map_size=64 * 1024 * 1024, subdir=True)
    with env.begin(write=True) as txn:
        txn.put(b"__len__", str(rows).encode())
        txn.put(b"__shape__", str((ACT,)).encode())
        for i in range(rows):
            txn.put(str(i).encode(), data[i].tobytes())
            txn.put(f"__label__{i}".encode(), str(i % 5).encode())
    env.close()
    return str(path)


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    root = tmp_path_factory.mktemp("features")
    return {"train": write_lmdb(root / "train.lmdb", ROWS, 1),
            "val": write_lmdb(root / "val.lmdb", 96, 2)}


def train(stores, out: Path, name: str, **overrides):
    """One tiny run through the real trainer. Respawn is forced on (every step, threshold reached
    after one batch), so any path that should skip it is actually exercised."""
    settings = dict(
        train_path=stores["train"], val_path=stores["val"], test_path=None, exp_name=name,
        n_latents=DICT, top_k=K, batch_size=BATCH, num_epochs=2, lr=1e-3, span_coef=0.01,
        dead_steps_threshold=BATCH, dead_check_interval=1, sparsity_mode="topk",
        device="cpu", num_workers=0, seed=0, output_dir=str(out), wandb_project=None,
        skip_extra_eval=True, skip_epoch_checkpoints=True)
    settings.update(overrides)
    config = TrainingConfig(**settings)
    # main.py writes this file before calling the trainer; load_run and the frozen path need it.
    run = out / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "training_config.json").write_text(json.dumps(config_payload(config)))
    model = train_autoencoder(config)
    return model, run


# --------------------------------------------------------------------------- #
# Projector construction
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("extra,expected", [
    ({}, [nn.Linear, nn.GELU]),
    ({"poly_layers": 2, "poly_hidden": 8}, [nn.Linear, nn.GELU, nn.Linear]),
    ({"poly_layers": 3, "poly_hidden": 8}, [nn.Linear, nn.GELU, nn.Linear, nn.GELU, nn.Linear]),
    ({"poly_hidden_dims": [8]}, [nn.Linear, nn.GELU, nn.Linear]),
])
def test_legacy_projector_paths_unchanged(extra, expected):
    """Every reported run used the first path, and earlier ablations the others. Their layer
    structure, including the disagreement about the final GELU, must not move."""
    assert [type(m) for m in DualSAE(dual_cfg(**extra)).poly] == expected


@pytest.mark.parametrize("extra", [{}, {"poly_hidden_dims": [8]}, {"poly_layers": 3, "poly_hidden": 8}])
def test_auto_is_identical_to_no_key(extra):
    """poly_final_act='auto' and a legacy cfg with no key must build the same weights and output."""
    legacy, explicit = DualSAE(dual_cfg(**extra)), DualSAE(dual_cfg(poly_final_act="auto", **extra))
    state_a, state_b = legacy.state_dict(), explicit.state_dict()
    assert state_a.keys() == state_b.keys()
    assert all(torch.equal(state_a[k], state_b[k]) for k in state_a)
    x = torch.randn(4, ACT)
    assert torch.equal(legacy(x)[2], explicit(x)[2])


def test_final_act_changes_only_the_terminal_activation():
    """The override must not disturb layer names or RNG use, so the same seed gives the same weights."""
    plain = DualSAE(dual_cfg())
    linear = DualSAE(dual_cfg(poly_final_act="none"))
    assert [type(m) for m in linear.poly] == [nn.Linear]
    assert torch.equal(plain.poly[0].weight, linear.poly[0].weight)
    bottleneck = DualSAE(dual_cfg(poly_hidden_dims=[8]))
    terminal = DualSAE(dual_cfg(poly_hidden_dims=[8], poly_final_act="gelu"))
    assert [type(m) for m in terminal.poly] == [nn.Linear, nn.GELU, nn.Linear, nn.GELU]
    assert bottleneck.state_dict().keys() == terminal.state_dict().keys()
    assert torch.equal(bottleneck.poly[2].weight, terminal.poly[2].weight)


def test_zero_poly_is_a_standard_sae():
    model = DualSAE(dual_cfg(poly_arch="none"))
    assert sum(p.numel() for p in model.poly.parameters()) == 0
    assert not any(k.startswith("poly.") for k in model.state_dict())
    x = torch.randn(4, ACT)
    _, _, x_hat = model(x)
    assert torch.equal(x_hat, model._xM)
    assert torch.equal(model._r_hat, torch.zeros_like(x))


@pytest.mark.parametrize("key,value", [("poly_arch", "bogus"), ("poly_final_act", "relu")])
def test_unknown_options_are_refused(key, value):
    with pytest.raises(ValueError):
        DualSAE(dual_cfg(**{key: value}))


def test_poly_bypass_forces_zero_without_changing_the_state_dict():
    """Bypass is training state only: strict loading of existing checkpoints depends on that."""
    model = DualSAE(dual_cfg())
    keys = set(model.state_dict())
    model.poly_bypass = True
    x = torch.randn(4, ACT)
    _, _, x_hat = model(x)
    assert torch.equal(x_hat, model._xM)
    assert set(model.state_dict()) == keys


def test_unset_options_stay_out_of_saved_configs():
    """A default run must write the same config keys as before these options existed."""
    payload = config_payload(TrainingConfig(train_path="x"))
    for name in ("poly_arch", "poly_final_act", "poly_warmup_zero", "freeze_sparse_from", "snapshot_steps"):
        assert name not in payload
    payload = config_payload(TrainingConfig(train_path="x", poly_arch="none", snapshot_steps=[3]))
    assert payload["poly_arch"] == "none" and payload["snapshot_steps"] == [3]


# --------------------------------------------------------------------------- #
# Round trip through the checkpoint
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,overrides,terminal", [
    ("r0", {}, nn.GELU),
    ("l1", {"poly_final_act": "none"}, nn.Linear),
    ("b512", {"poly_hidden_dims": [8]}, nn.Linear),
    ("b512g", {"poly_hidden_dims": [8], "poly_final_act": "gelu"}, nn.GELU),
])
def test_architecture_survives_the_checkpoint(stores, tmp_path, name, overrides, terminal):
    """R0/L1 and B512/B512G have identical state dicts, so strict loading cannot catch a dropped
    key; only the rebuilt module and its output can. Both checkpoints are checked."""
    model, run = train(stores, tmp_path, name, **overrides)
    x = torch.randn(8, ACT)
    for checkpoint in ("best", "final"):
        loaded, _ = load_run(run, torch.device("cpu"), checkpoint=checkpoint)
        assert isinstance(list(loaded.poly)[-1], terminal), checkpoint
    # The final checkpoint is the same weights as the returned model, so outputs must match.
    loaded, _ = load_run(run, torch.device("cpu"), checkpoint="final")
    assert torch.allclose(forward_reps(model.eval(), x, True)["xhat"],
                          forward_reps(loaded, x, True)["xhat"])


def test_every_option_survives_the_cli(stores, tmp_path, monkeypatch):
    """main.py is what the grid calls, and it builds TrainingConfig from a long explicit
    keyword list, so a flag can parse correctly and still never arrive."""
    argv = ["main.py", "--model", "dual", "--sparsity_mode", "topk", "--dataset", "imagenet",
            "--extractor", "clip", "--train_path", stores["train"], "--val_path", stores["val"],
            "--test_path", stores["val"], "--n_latents", str(DICT), "--topk_k", str(K),
            "--batch_size", str(BATCH), "--num_epochs", "1", "--lr", "0.001", "--device", "cpu",
            "--num_workers", "0", "--wandb_project", "", "--skip_extra_eval",
            "--skip_epoch_checkpoints", "--output_dir", str(tmp_path), "--exp_name", "cli",
            "--poly_hidden_dims", "8", "--poly_final_act", "gelu"]
    monkeypatch.setattr(sys, "argv", argv)
    main_module.main()
    saved = json.loads((tmp_path / "cli" / "training_config.json").read_text())
    assert saved["poly_final_act"] == "gelu" and saved["poly_hidden_dims"] == [8]
    loaded, _ = load_run(tmp_path / "cli", torch.device("cpu"))
    assert [type(m) for m in loaded.poly] == [nn.Linear, nn.GELU, nn.Linear, nn.GELU]


def test_arm_options_survive_the_cli(stores, tmp_path, monkeypatch):
    """Drives the S, Z and F command lines through main.py in the order the grid runs them."""
    common = ["main.py", "--model", "dual", "--sparsity_mode", "topk", "--dataset", "imagenet",
              "--extractor", "clip", "--train_path", stores["train"], "--val_path", stores["val"],
              "--test_path", stores["val"], "--n_latents", str(DICT), "--topk_k", str(K),
              "--batch_size", str(BATCH), "--num_epochs", "1", "--lr", "0.001", "--device", "cpu",
              "--num_workers", "0", "--wandb_project", "", "--skip_extra_eval",
              "--skip_epoch_checkpoints", "--output_dir", str(tmp_path)]

    monkeypatch.setattr(sys, "argv", common + ["--exp_name", "s", "--poly_arch", "none",
                                               "--warmup_steps", "0", "--span_coef", "0.0"])
    main_module.main()
    assert json.loads((tmp_path / "s" / "training_config.json").read_text())["poly_arch"] == "none"
    loaded, _ = load_run(tmp_path / "s", torch.device("cpu"))
    assert sum(p.numel() for p in loaded.poly.parameters()) == 0

    monkeypatch.setattr(sys, "argv", common + ["--exp_name", "z", "--poly_warmup_zero",
                                               "--warmup_steps", "2"])
    main_module.main()
    assert json.loads((tmp_path / "z" / "training_config.json").read_text())["poly_warmup_zero"] is True

    monkeypatch.setattr(sys, "argv", common + ["--exp_name", "f", "--warmup_steps", "0",
                                               "--freeze_sparse_from", str(tmp_path / "s"),
                                               "--snapshot_steps", "3,5"])
    main_module.main()
    saved = json.loads((tmp_path / "f" / "training_config.json").read_text())
    assert saved["freeze_sparse_from"] == str(tmp_path / "s") and saved["snapshot_steps"] == [3, 5]
    assert (tmp_path / "f__step3" / "best_val_monoscore_model.pt").exists()
    assert (tmp_path / "f__step5" / "best_val_monoscore_model.pt").exists()
    source = torch.load(tmp_path / "s" / "best_val_monoscore_model.pt", map_location="cpu", weights_only=False)
    final = torch.load(tmp_path / "f" / "autoencoder_final.pt", map_location="cpu", weights_only=False)
    assert sparse_state_sha256(final["model_state_dict"]) == sparse_state_sha256(source["model_state_dict"])


def test_cli_refuses_options_on_other_models(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["main.py", "--model", "topk", "--poly_arch", "none",
                                      "--output_dir", str(tmp_path), "--exp_name", "x"])
    with pytest.raises(SystemExit):
        main_module.main()


# --------------------------------------------------------------------------- #
# The S and F arms
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def s_run(stores, tmp_path_factory):
    out = tmp_path_factory.mktemp("s")
    _, run = train(stores, out, "s", poly_arch="none", span_coef=0.0)
    return run


def test_s_arm_runs_full_extra_eval_without_nan(stores, tmp_path):
    """S runs without --skip_extra_eval on the real grid, so the zero r_hat has to pass through
    every diagnostic there. poly_arch='none' must also skip the r_hat probes, or S's dead-latent
    clock would run at a DualSAE's joint-stage rate."""
    calls = []
    original = train_dual.compute_r_hat_monosemanticity
    train_dual.compute_r_hat_monosemanticity = lambda *a, **k: calls.append(1) or original(*a, **k)
    try:
        model, run = train(stores, tmp_path, "s_full", poly_arch="none", span_coef=0.0,
                           skip_extra_eval=False)
    finally:
        train_dual.compute_r_hat_monosemanticity = original
    assert not calls
    assert (run / "autoencoder_final.pt").exists()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    loaded, meta = load_run(run, torch.device("cpu"))
    x = torch.randn(8, ACT)
    reps = forward_reps(loaded, x, bool(meta["is_dual"]))
    assert torch.equal(reps["xhat"], reps["xM"])


def test_f_arm_changes_only_the_projector(stores, tmp_path, s_run):
    """The defining property of the frozen arm. Respawn is forced on every step here, so if the
    gates around normalization and respawn were missing, these tensors would change."""
    source = torch.load(s_run / "best_val_monoscore_model.pt", map_location="cpu", weights_only=False)
    # Same seed and constructor as the trainer's model, so this is the projector before training.
    fresh = DualSAE(dual_cfg())
    model, run = train(stores, tmp_path, "f", freeze_sparse_from=str(s_run), span_coef=0.01)
    final = torch.load(run / "autoencoder_final.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    assert sparse_state_sha256(final) == sparse_state_sha256(source["model_state_dict"])
    assert not torch.equal(final["poly.0.weight"], fresh.poly[0].weight)
    assert json.loads((run / "frozen_sparse.json").read_text())["sparse_sha256"] == \
        sparse_state_sha256(final)
    # F must end up selecting its LAST epoch; a strict comparison on a constant keeps epoch 1.
    assert torch.load(run / "best_val_monoscore_model.pt", map_location="cpu",
                      weights_only=False)["epoch"] == 2


def test_f_arm_optimizer_excludes_sparse_branch(stores, tmp_path, s_run, monkeypatch):
    built = []
    real_adam = torch.optim.Adam
    def spy(params, *a, **k):
        params = list(params)
        built.append((params, k.get("lr"), k.get("eps")))
        return real_adam(params, *a, **k)
    monkeypatch.setattr(train_dual.torch.optim, "Adam", spy)
    model, _ = train(stores, tmp_path, "f_opt", freeze_sparse_from=str(s_run), poly_lr_factor=0.5)
    assert len(built) == 1
    params, lr, eps = built[0]
    assert {id(p) for p in params} == {id(p) for p in model.poly.parameters()}
    assert lr == pytest.approx(1e-3 * 0.5) and eps == 6.25e-10


def test_changed_sparse_branch_blocks_completion(stores, tmp_path, s_run, monkeypatch):
    """If the hash ever differs, the run must not become reportable."""
    real = train_dual.sparse_branch_sha256
    calls = []
    def drifting(model):
        # Called once when the branch is loaded and once when training ends.
        calls.append(1)
        return real(model) if len(calls) == 1 else "different"
    monkeypatch.setattr(train_dual, "sparse_branch_sha256", drifting)
    with pytest.raises(RuntimeError, match="changed during training"):
        train(stores, tmp_path, "f_bad", freeze_sparse_from=str(s_run))
    assert not (tmp_path / "f_bad" / "autoencoder_final.pt").exists()


def test_snapshots_are_loadable_and_exact(stores, tmp_path, s_run):
    """Two epochs at 8 steps each is 16 updates. A snapshot at step 16 is the same model as the
    end of training, and one at step 3 is an earlier, different one."""
    _, run = train(stores, tmp_path, "f_snap", freeze_sparse_from=str(s_run), snapshot_steps=[3, 16])
    final = torch.load(run / "autoencoder_final.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    late = torch.load(tmp_path / "f_snap__step16" / "best_val_monoscore_model.pt",
                      map_location="cpu", weights_only=False)
    early = torch.load(tmp_path / "f_snap__step3" / "best_val_monoscore_model.pt",
                       map_location="cpu", weights_only=False)
    assert late["step"] == 16 and early["step"] == 3
    assert torch.equal(late["model_state_dict"]["poly.0.weight"], final["poly.0.weight"])
    assert not torch.equal(early["model_state_dict"]["poly.0.weight"], final["poly.0.weight"])
    # val_monoscore comes from the frozen source, keeping the evaluator's self-check meaningful.
    source = torch.load(s_run / "best_val_monoscore_model.pt", map_location="cpu", weights_only=False)
    assert early["val_monoscore"] == source["val_monoscore"]
    loaded, meta = load_run(tmp_path / "f_snap__step3", torch.device("cpu"))
    assert meta["is_dual"] and torch.equal(loaded.poly[0].weight, early["model_state_dict"]["poly.0.weight"])
    assert not list(tmp_path.glob(".*.tmp")), "a temporary snapshot directory was left behind"


@pytest.mark.parametrize("overrides,message", [
    ({"poly_arch": "none", "span_coef": 0.01}, "standard SAE"),
    ({"poly_arch": "none", "span_coef": 0.0, "warmup_steps": 2}, "standard SAE"),
    ({"poly_warmup_zero": True}, "poly_warmup_zero"),
    ({"snapshot_steps": [3]}, "snapshot_steps"),
    ({"freeze_sparse_from": "x", "warmup_steps": 2}, "warm-up"),
    ({"freeze_sparse_from": "x", "sparsity_mode": "jump_relu"}, "supports"),
    ({"freeze_sparse_from": "x", "mono_coef": 0.1}, "mono_coef"),
])
def test_mislabelling_combinations_are_refused(stores, tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        train(stores, tmp_path, "bad", **overrides)


def test_frozen_source_must_be_a_finished_matching_sae(stores, tmp_path, s_run):
    # A run that has a real projector is refused, or F would inherit a jointly trained branch.
    _, dual_run = train(stores, tmp_path, "dual_src")
    with pytest.raises(ValueError, match="standard SAE"):
        train(stores, tmp_path, "f1", freeze_sparse_from=str(dual_run))
    # An unfinished source is refused, or F would be built on a different S than the one reported.
    with pytest.raises(FileNotFoundError):
        train(stores, tmp_path, "f2", freeze_sparse_from=str(tmp_path / "missing"))
    # A different activation or seed would make the pair differ in more than joint training.
    with pytest.raises(ValueError, match="does not match"):
        train(stores, tmp_path, "f3", freeze_sparse_from=str(s_run), sparsity_mode="batch_topk")
    with pytest.raises(ValueError, match="does not match"):
        train(stores, tmp_path, "f4", freeze_sparse_from=str(s_run), seed=1)


def test_snapshot_beyond_budget_is_refused(stores, tmp_path, s_run):
    with pytest.raises(ValueError, match="exceed"):
        train(stores, tmp_path, "f5", freeze_sparse_from=str(s_run), snapshot_steps=[17])


# --------------------------------------------------------------------------- #
# The Z arm
# --------------------------------------------------------------------------- #

def test_z_arm_bypasses_only_during_warm_up(stores, tmp_path, monkeypatch):
    """Warm-up covers updates 0..W, so with W=4 the first five training steps see r_hat == 0."""
    seen = []
    real = DualSAE.compute_loss
    def spy(self, *a, **k):
        if self.training:
            seen.append((self.poly_bypass, float(self._r_hat.abs().max())))
        return real(self, *a, **k)
    monkeypatch.setattr(DualSAE, "compute_loss", spy)
    model, run = train(stores, tmp_path, "z", warmup_steps=4, poly_warmup_zero=True)
    assert [b for b, _ in seen] == [True] * 5 + [False] * 11
    assert all(m == 0.0 for b, m in seen if b) and all(m > 0.0 for b, m in seen if not b)
    assert model.poly_bypass is False
    # A saved model has the bypass off, so it must really use its projector when reloaded.
    loaded, _ = load_run(run, torch.device("cpu"))
    x = torch.randn(8, ACT)
    reps = forward_reps(loaded, x, True)
    assert not torch.equal(reps["xhat"], reps["xM"])


def test_z_selection_restarts_after_warm_up(stores, tmp_path):
    """A bypassed checkpoint would load with the bypass off and describe a different model. Here
    warm-up ends in epoch 2, so a checkpoint saved at epoch 1 must not survive as the best."""
    _, run = train(stores, tmp_path, "z2", warmup_steps=9, poly_warmup_zero=True)
    best = torch.load(run / "best_val_monoscore_model.pt", map_location="cpu", weights_only=False)
    assert best["epoch"] == 2
