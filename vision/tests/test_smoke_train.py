#!/usr/bin/env python3
"""Smoke test: train every model type end to end through main.py on tiny synthetic LMDBs.

For each model (TopK, BatchTopK, JumpReLU, vanilla, Gated, Matryoshka, MSAE and DualSAE with
each Sparse-Branch rule) this writes a fake ImageNet-style feature LMDB (random features + class
labels), trains 2 epochs on CPU through ``main.py`` (the Dual runs with a warm-up that ends
inside the run), and checks that the run directory holds the final checkpoint, the
best-validation checkpoint, ``training_config.json`` and a ``best_metrics.json`` with the five
metrics. It then scores one Dual run with ``branch_metrics.py`` (offline evaluator).

No downloads, no GPU. Run from vision/:  python -m pytest tests/test_smoke_train.py -q
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import lmdb
import numpy as np
import pytest

VISION = Path(__file__).resolve().parents[1]
D, N_TRAIN, N_VAL, N_CLASSES = 16, 256, 96, 5

MODELS = [
    ("topk", []), ("batch_topk", []), ("jumprelu", ["--normalize"]), ("vanilla", []),
    ("gated", ["--l1_coef", "0.1", "--max_grad_norm", "1e9", "--lr", "5e-5"]),
    ("matryoshka", []), ("msae", []),
    ("dual", ["--sparsity_mode", "topk"]), ("dual", ["--sparsity_mode", "batch_topk"]),
    ("dual", ["--sparsity_mode", "jump_relu", "--normalize"]),
    ("dual", ["--sparsity_mode", "matryoshka"]),
    ("dual", ["--sparsity_mode", "gated", "--l1_coef", "0.02", "--lr", "5e-5"]),
]


def _write_lmdb(path: Path, n: int, rng: np.random.Generator) -> None:
    """Features in LMDBFeatureDataset's format: raw float32 rows + __label__ keys."""
    centers = rng.standard_normal((N_CLASSES, D)).astype(np.float32)
    env = lmdb.open(str(path), map_size=64 << 20)
    with env.begin(write=True) as txn:
        for i in range(n):
            y = int(rng.integers(N_CLASSES))
            x = centers[y] + 0.3 * rng.standard_normal(D).astype(np.float32)
            txn.put(str(i).encode(), x.astype(np.float32).tobytes())
            txn.put(f"__label__{i}".encode(), str(y).encode())
        txn.put(b"__len__", str(n).encode())
        txn.put(b"__shape__", str((D,)).encode())
    env.close()


@pytest.fixture(scope="module")
def features(tmp_path_factory):
    root = tmp_path_factory.mktemp("features")
    rng = np.random.default_rng(0)
    # "imagenet" in the path keeps the single-label code path (as for the real features).
    tr, va = root / "imagenet_train.lmdb", root / "imagenet_validation.lmdb"
    _write_lmdb(tr, N_TRAIN, rng)
    _write_lmdb(va, N_VAL, rng)
    return tr, va


def _train(features, out_dir: Path, model: str, extra: list, name: str) -> Path:
    tr, va = features
    cmd = [sys.executable, str(VISION / "main.py"), "--model", model, *extra,
           "--train_path", str(tr), "--val_path", str(va), "--test_path", "",
           "--n_latents", "64", "--topk_k", "4", "--batch_size", "32", "--num_epochs", "2",
           "--num_workers", "0", "--device", "cpu", "--seed", "42",
           "--exp_name", name, "--output_dir", str(out_dir)]
    if model == "dual":
        # Warm-up must end inside the run: 256 rows / batch 32 = 8 steps per epoch.
        cmd += ["--warmup_steps", "4", "--dead_check_interval", "100", "--skip_epoch_checkpoints"]
    env = dict(os.environ, OMP_NUM_THREADS="2", WANDB_MODE="disabled")
    res = subprocess.run(cmd, cwd=str(VISION), env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, f"{name} failed:\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}"
    return out_dir / name


@pytest.mark.parametrize("model,extra", MODELS,
                         ids=[m if m != "dual" else f"dual_{e[1]}" for m, e in MODELS])
def test_train_one_model(features, tmp_path, model, extra):
    run = _train(features, tmp_path, model, extra, "run")
    for f in ("autoencoder_final.pt", "best_val_monoscore_model.pt", "training_config.json",
              "best_metrics.json"):
        assert (run / f).exists(), f
    bm = json.loads((run / "best_metrics.json").read_text())
    for key in ("BEST_R2", "BEST_MONOSCORE", "BEST_PURITY", "BEST_ACC_KNN", "BEST_ACC_LP"):
        assert key in bm and np.isfinite(bm[key]), key


def test_offline_evaluator_on_a_dual_run(features, tmp_path):
    run = _train(features, tmp_path / "ck", "dual", ["--sparsity_mode", "batch_topk"], "dual_run")
    out = tmp_path / "eval"
    cmd = [sys.executable, str(VISION / "branch_metrics.py"), "--run-dir", str(run),
           "--eval-split", "val", "--out-dir", str(out), "--device", "cpu", "--num-workers", "0",
           "--encode-batch-size", "32", "--retrieval-k", "5"]
    res = subprocess.run(cmd, cwd=str(VISION), env=dict(os.environ, OMP_NUM_THREADS="2"),
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    d = json.loads((out / "dual_run__val.json").read_text())
    assert d["self_checks"]["all_ok"]
    for rep in ("x", "xM", "r_hat", "xhat"):
        assert rep in d["reps"]
