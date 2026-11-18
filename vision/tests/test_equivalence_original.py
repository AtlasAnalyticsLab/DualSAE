#!/usr/bin/env python3
"""Code-equivalence checks between this release and an original copy of the training code.

Both checks are skipped unless the environment points at the original code; they are a
"same code" check (identical tensors), not a results comparison.

1. ``DUALSAE_ORIGINAL_CORE=<dir>`` (a directory holding the original main.py, trainers and
   models/): every model type is trained for 2 epochs on CPU, from the same seed and the same
   synthetic LMDB, once with the original code and once with this release, and the final
   state dicts must be bit-identical. (The JumpReLU threshold initialization is passed
   explicitly to both, since its default differs between the two code versions.)
2. ``DUALSAE_EQUIV_RUNS=<run_dir>[:<run_dir>...]`` (trained run directories, each with
   training_config.json and best_val_monoscore_model.pt) in addition to the variable above:
   each checkpoint is loaded with ``knn_lp_sparse.load_run`` of both code versions and pushed
   through ``forward_reps`` on the same input; every representation must be bit-identical.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

VISION = Path(__file__).resolve().parents[1]
ORIG = os.environ.get("DUALSAE_ORIGINAL_CORE")
pytestmark = pytest.mark.skipif(not ORIG, reason="DUALSAE_ORIGINAL_CORE not set")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_smoke_train import MODELS, _write_lmdb  # noqa: E402


@pytest.fixture(scope="module")
def features(tmp_path_factory):
    root = tmp_path_factory.mktemp("eqfeat")
    rng = np.random.default_rng(1)
    tr, va = root / "imagenet_train.lmdb", root / "imagenet_validation.lmdb"
    _write_lmdb(tr, 256, rng)
    _write_lmdb(va, 96, rng)
    return tr, va


def _train(core: Path, features, out: Path, model: str, extra: list) -> Path:
    tr, va = features
    cmd = [sys.executable, str(core / "main.py"), "--model", model, *extra,
           "--train_path", str(tr), "--val_path", str(va), "--test_path", "",
           "--n_latents", "64", "--topk_k", "4", "--batch_size", "32", "--num_epochs", "2",
           "--num_workers", "0", "--device", "cpu", "--seed", "7", "--skip_extra_eval",
           "--exp_name", "run", "--output_dir", str(out), "--wandb_project", ""]
    if model == "jumprelu" or "jump_relu" in extra:
        cmd += ["--jump_threshold_init", "1.0"]
    if model == "dual":
        cmd += ["--warmup_steps", "4", "--dead_check_interval", "100"]
    env = dict(os.environ, OMP_NUM_THREADS="1", WANDB_MODE="disabled")
    res = subprocess.run(cmd, cwd=str(core), env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    return out / "run" / "autoencoder_final.pt"


@pytest.mark.parametrize("model,extra", MODELS,
                         ids=[m if m != "dual" else f"dual_{e[1]}" for m, e in MODELS])
def test_training_is_bit_identical(features, tmp_path, model, extra):
    import torch
    a = torch.load(_train(Path(ORIG), features, tmp_path / "orig", model, extra),
                   map_location="cpu", weights_only=False)
    b = torch.load(_train(VISION, features, tmp_path / "rel", model, extra),
                   map_location="cpu", weights_only=False)
    sa, sb = a.get("model_state_dict", a), b.get("model_state_dict", b)
    assert sa.keys() == sb.keys()
    for k in sa:
        if torch.is_tensor(sa[k]):
            assert torch.equal(sa[k], sb[k]), k


_FORWARD = r"""
import sys, torch
core, run, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, core)
from knn_lp_sparse import load_run, forward_reps
model, meta = load_run(__import__("pathlib").Path(run), torch.device("cpu"))
g = torch.Generator().manual_seed(0)
x = torch.randn(64, int(meta["act_size"]), generator=g)
with torch.no_grad():
    reps = forward_reps(model, x, bool(meta.get("is_dual")))
torch.save({k: v for k, v in reps.items() if torch.is_tensor(v)}, out)
"""


@pytest.mark.skipif(not os.environ.get("DUALSAE_EQUIV_RUNS"), reason="DUALSAE_EQUIV_RUNS not set")
def test_published_checkpoints_forward_identically(tmp_path):
    import torch
    runs = [r for r in os.environ["DUALSAE_EQUIV_RUNS"].split(":") if r]
    for i, run in enumerate(runs):
        outs = []
        for tag, core in (("orig", ORIG), ("rel", str(VISION))):
            out = tmp_path / f"{i}_{tag}.pt"
            res = subprocess.run([sys.executable, "-c", _FORWARD, core, run, str(out)], cwd=core,
                                 env=dict(os.environ, OMP_NUM_THREADS="1"),
                                 capture_output=True, text=True, timeout=600)
            assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
            outs.append(torch.load(out))
        assert outs[0].keys() == outs[1].keys(), run
        for k in outs[0]:
            assert torch.equal(outs[0][k], outs[1][k]), (run, k)
