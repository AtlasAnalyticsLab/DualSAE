"""Loading the steered language model (pythia-410m-deduped) from a LOCAL snapshot.

Runtime code never downloads: the snapshot is fetched once by ``prepare_assets.sh`` and loaded
with ``local_files_only=True``, so a cache miss is a clear exception instead of a hanging
network call on an offline node.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import torch

from common import lang_config as config


def resolve_model_path(cli_path: Optional[str]) -> str:
    """Resolve the local pythia snapshot.

    Order: an explicit path, then ``$A16_LM_PATH``, then <LANGDUAL_ROOT>/raw/models/<org>__<name>
    (where prepare_assets.sh puts it), then <LANGDUAL_ROOT>/raw/<basename>, then the bare repo id
    (resolved by from_pretrained(local_files_only=True) out of the HF cache)."""
    if cli_path:
        if not os.path.isdir(cli_path):
            raise FileNotFoundError(f"model path {cli_path!r} is not a directory")
        return cli_path
    env = os.environ.get("A16_LM_PATH")
    if env:
        if not os.path.isdir(env):
            raise FileNotFoundError(f"A16_LM_PATH={env!r} is not a directory")
        return env
    candidates = [config.local_snapshot(config.LM_REPO, "model"),
                  os.path.join(config.RAW_DIR, os.path.basename(config.LM_REPO))]
    for candidate in candidates:
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
            return candidate
    return config.LM_REPO


def load_model(model_path: str, device: torch.device, dtype: torch.dtype):
    """Load the causal LM in eval mode with frozen weights.

    fp32 is used (not bf16): bf16's 8 mantissa bits would inject an error far larger than the
    fp16 storage error of the residual shards, and the same compute dtype is used for the
    activations and for every splice/steering pass."""
    from transformers import AutoModelForCausalLM  # heavy; imported on demand

    # transformers 5.x renamed torch_dtype -> dtype (torch_dtype is still accepted for BC).
    kwargs = dict(local_files_only=True, attn_implementation="sdpa")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError:  # pragma: no cover - older transformers
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, **kwargs)
    model.eval()
    model.to(device)
    # Frozen parameters also prevent an autograd graph if a caller forgets torch.no_grad().
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_blocks(model) -> Sequence[torch.nn.Module]:
    """The transformer blocks (``.gpt_neox.layers`` for GPTNeoXForCausalLM)."""
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError(f"cannot locate transformer blocks on {type(model).__name__}")
