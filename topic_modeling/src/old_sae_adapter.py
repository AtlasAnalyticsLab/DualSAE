"""
Adapter: load a checkpoint produced by the vision SAE trainers (`vision/train_dual.py`,
`vision/train_topk.py`, `vision/train_other.py`, all reachable through `vision/main.py`) and expose
it with the tiny interface SAE-TM's `interpret_sae.py --old-impl` and the hypothesis-generation
pipeline need: `.encode(x) -> sparse code tensor`.

Why this exists: the topic-modeling and hypothesis-generation SAEs are trained with the repo's own
`vision/` trainers, whose checkpoints and classes differ from SAE-TM's `dictionary_learning`
trainers. For EVERY such model, `forward(x)` returns `(pre_act, sparse_latents, recons)`, so the
sparse code (DualSAE's Sparse Branch code `z`, or the TopK/BatchTopK/JumpReLU sparse codes) is
simply `forward(x)[1]`. We wrap that and reconstruct the right model from the checkpoint's
`model_cfg`.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Make the repo's `vision/` core importable (it uses flat imports like `from models import ...`).
# The path is resolved relative to this file (<repo>/topic_modeling/src/old_sae_adapter.py ->
# <repo>/vision); $VISION_DIR overrides it.
_VISION_DIR = os.environ.get("VISION_DIR", str(Path(__file__).resolve().parents[2] / "vision"))
if _VISION_DIR not in sys.path:
    sys.path.insert(0, _VISION_DIR)

from models import DualSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE  # noqa: E402  (vision/models)
from models.topk_sae import TopK  # noqa: E402  (TopK activation, for reconstructing TopKSAE)

# sae-type tokens that map to DualSAE (sparsity_mode is read from the saved model_cfg).
_DUAL_TYPES = {"DualSAETrainer", "DualTopKTrainer", "DualBatchTopKTrainer", "DualJumpReluTrainer"}


class _OldSAEWrapper:
    """Uniform shim over a `vision/` model: `encode(x)` returns the sparse code `forward(x)[1]`."""

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # forward(x) -> (pre_act, sparse_latents, recons) for every model; [1] is the sparse
        # code interpret_sae.py consumes (DualSAE's [1] is the Sparse Branch code z).
        return self.model(x)[1]

    def to(self, *args, **kwargs):
        # Pass through (supports `.to(device=..., dtype=...)` as interpret_sae.py calls it).
        self.model.to(*args, **kwargs)
        return self

    def eval(self):
        self.model.eval()
        return self


def load_old_sae(checkpoint_path: str, sae_type: str, device=None) -> _OldSAEWrapper:
    """Reconstruct a `vision/` model from its `best_val_monoscore_model.pt` and wrap it.

    The checkpoint is `{... , 'model_state_dict', 'model_cfg'}` (train_topk now also saves
    `model_cfg`). `model_cfg` is self-describing per model type.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    cfg = ckpt.get("model_cfg")
    if cfg is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no 'model_cfg'; cannot reconstruct the model."
        )

    dev = str(device) if device is not None else "cpu"

    if sae_type in _DUAL_TYPES:
        cfg = dict(cfg)
        cfg["device"] = dev                      # build on the target device (no GPU allocation on CPU-only hosts)
        model = DualSAE(cfg)                      # cfg carries sparsity_mode/top_k/poly_hidden/...
    elif sae_type == "TopKTrainer":
        # Build from the saved model_cfg to preserve the TRUE tied/normalize (TopKSAE.state_dict()
        # also stores custom 'activation' / 'activation_state_dict' entries + an 'activation.*'
        # submodule that a plain load_state_dict rejects, and TopKSAE.from_state_dict hard-codes
        # normalize=(activation=='TopK') which is NOT how we trained it). So: reconstruct with the
        # real cfg, strip the activation-* keys, and load the rest (strict=False).
        act = TopK(k=cfg["topk_k"]) if cfg.get("activation", "topk") == "topk" else nn.ReLU()
        model = TopKSAE(
            n_latents=cfg["n_latents"], n_inputs=cfg["d_model"],
            activation=act, tied=cfg.get("tied", True), normalize=cfg.get("normalize", False),
        )
        state_dict = {k: v for k, v in state_dict.items()
                      if k not in ("activation", "activation_state_dict")
                      and not k.startswith("activation.")}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        if device is not None:
            model.to(device)
        return _OldSAEWrapper(model)
    elif sae_type == "BatchTopKTrainer":
        cfg = dict(cfg); cfg["device"] = dev
        model = BatchTopKSAE(cfg)
    elif sae_type == "JumpReluTrainer":
        cfg = dict(cfg); cfg["device"] = dev
        model = JumpReLUSAE(cfg)
    else:
        raise ValueError(
            f"Unknown sae_type {sae_type!r} for the vision-checkpoint adapter. Expected one of "
            f"{{TopKTrainer, BatchTopKTrainer, JumpReluTrainer}} ∪ {_DUAL_TYPES}."
        )

    model.load_state_dict(state_dict)
    model.eval()
    if device is not None:
        model.to(device)
    return _OldSAEWrapper(model)
