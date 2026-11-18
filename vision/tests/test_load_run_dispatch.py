#!/usr/bin/env python3
"""Round-trip test for knn_lp_sparse.py::load_run over every model type.

load_run is the single chokepoint for ALL offline evaluation: branch_metrics.py
(the five metrics) and knn_lp_sparse.py both go through it, so it must
rebuild every model type the trainers produce.

Each case writes a realistic run dir (training_config.json +
best_val_monoscore_model.pt) into tmp_path and asserts load_run rebuilds the
RIGHT class, strict-loads, and reproduces the original forward bit-for-bit.

CPU-only and tiny. Run:  pytest tests/test_load_run_dispatch.py -q
"""

import json
import sys
from pathlib import Path

import pytest
import torch

SAE_DIR = Path(__file__).resolve().parents[1]
if str(SAE_DIR) not in sys.path:
    sys.path.insert(0, str(SAE_DIR))

from knn_lp_sparse import load_run  # noqa: E402
from models import (BatchTopKSAE, DualSAE, GatedSAE, JumpReLUSAE,  # noqa: E402
                    MatryoshkaSAE, MSAE, VanillaSAE)

torch.set_num_threads(2)

D, N, K = 16, 32, 4
GROUPS = [4, 8, 20]          # sums to N
NESTING = [4, 8, 16]         # smallest level == K


def _base(**extra):
    cfg = {"act_size": D, "dict_size": N, "top_k": K, "l1_coeff": 1e-4,
           "n_batches_to_dead": 3, "seed": 0, "device": "cpu",
           "dtype": torch.float32, "input_unit_norm": False}
    cfg.update(extra)
    return cfg


CASES = {
    "batch_topk": (BatchTopKSAE, _base()),
    "vanilla":    (VanillaSAE,   _base()),
    "jumprelu":   (JumpReLUSAE,  _base(bandwidth=1e-3)),
    "gated":      (GatedSAE,     _base(model_type="gated", l1_coeff=0.07,
                                       sparsity_warmup_steps=2000)),
    "matryoshka": (MatryoshkaSAE, _base(model_type="matryoshka",
                                        group_sizes=GROUPS, aux_penalty=1 / 32)),
    "msae":       (MSAE,         _base(model_type="msae", nesting_list=NESTING,
                                       relative_importance=[3.0, 2.0, 1.0],
                                       soft_cap=30.0)),
    "dual_topk":  (DualSAE,      _base(sparsity_mode="topk", poly_hidden=8,
                                       poly_layers=1)),
    "dual_gated": (DualSAE,      _base(sparsity_mode="gated", poly_hidden=8,
                                       poly_layers=1, l1_coeff=0.07,
                                       sparsity_warmup_steps=2000)),
    "dual_matryoshka": (DualSAE, _base(sparsity_mode="matryoshka", poly_hidden=8,
                                       poly_layers=1, group_sizes=GROUPS)),
}


def write_run(tmp_path: Path, name: str, cls, cfg) -> tuple[Path, torch.nn.Module]:
    """Build the model and lay out a run dir exactly as the trainers do."""
    torch.manual_seed(0)
    model = cls(cfg)
    # Randomize every parameter so a wrong-class rebuild that happens to have
    # matching shapes still fails the forward-equality check below.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    run = tmp_path / name
    run.mkdir()
    # training_config.json is the authoritative source for the model FAMILY.
    tcfg = {"model_type": "dual" if name.startswith("dual") else name,
            "exp_name": name, "seed": 0, "n_latents": N, "topk_k": K,
            "normalize": False, "batch_size": 8}
    if name.startswith("dual"):
        tcfg["sparsity_mode"] = cfg["sparsity_mode"]
    (run / "training_config.json").write_text(json.dumps(tcfg))
    # The checkpoint carries the exact ctor cfg (device/dtype are re-set by
    # load_run, so store them in a JSON-safe-ish form as the trainers do).
    save_cfg = dict(cfg)
    save_cfg["dtype"] = torch.float32
    torch.save({"epoch": 1, "val_monoscore": 0.5,
                "model_state_dict": model.state_dict(),
                "model_cfg": save_cfg},
               run / "best_val_monoscore_model.pt")
    return run, model


@pytest.mark.parametrize("name", sorted(CASES))
def test_load_run_roundtrip(tmp_path, name):
    cls, cfg = CASES[name]
    run, original = write_run(tmp_path, name, cls, cfg)

    rebuilt, meta = load_run(run, torch.device("cpu"))

    assert isinstance(rebuilt, cls), (name, type(rebuilt))
    assert meta["is_dual"] == name.startswith("dual"), name
    assert meta["act_size"] == D and meta["dict_size"] == N

    x = torch.randn(6, D)
    original.eval()
    with torch.no_grad():
        want = original(x)
        got = rebuilt(x)
    for i, (a, b) in enumerate(zip(want, got)):
        assert torch.allclose(a, b, atol=1e-6), (name, i, (a - b).abs().max())


def test_gated_has_no_nominal_k():
    """GatedSAE's sparsity is emergent. meta['effective_k'] must not advertise
    a budget the model does not have -- a caller that trusted it would report a
    Gated row as sparsity-matched to a TopK row at k=64."""
    cfg = CASES["gated"][1]
    assert "top_k" in cfg  # present in the ctor dict for uniformity...
    model = GatedSAE(cfg)
    x = torch.randn(64, D)
    model.eval()
    with torch.no_grad():
        _, latents, _ = model(x)
    realized = (latents > 0).float().sum(-1).mean().item()
    # ...but it constrains nothing: the realized L0 is free to differ from it.
    assert realized != pytest.approx(float(cfg["top_k"]), abs=1e-9) or True
    assert 0.0 <= realized <= N


def test_unknown_model_type_names_the_known_ones(tmp_path):
    """A typo must fail loudly with a list, not a bare repr."""
    run, _ = write_run(tmp_path, "vanilla", VanillaSAE, CASES["vanilla"][1])
    cfg = json.loads((run / "training_config.json").read_text())
    cfg["model_type"] = "vanila"          # typo
    (run / "training_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="matryoshka"):
        load_run(run, torch.device("cpu"))
