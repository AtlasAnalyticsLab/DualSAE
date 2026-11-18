"""langdual.lmsae — Track B: SAEs on the pythia-410m-deduped residual stream.

Two modules, one handover point:

  ``extract_resid``   runs EleutherAI/pythia-410m-deduped over the packed token stream and
                      captures the residual stream AFTER blocks ``config.LM_LAYERS`` in a
                      SINGLE forward pass (torch forward hooks, early exit past the deepest
                      hooked block).  Writes fp16 memmap shards plus a manifest into
                      ``config.resid_dir(layer, split)``, with one fixed global norm factor per
                      layer FOLDED INTO the stored bytes.

  ``resid_shards``    ``ResidShardDataset``, a drop-in substitute for
                      ``sae/dataset/lmdb_features.LMDBFeatureDataset`` over those shards, so the
                      same ``sae/`` trainers and post-hoc scorers train and score Track B with
                      no modification.  It applies NO L2 normalization -- the single point of
                      the folded global scale.

WHY THE IMPORTS BELOW ARE LAZY: ``extract_resid`` imports ``transformers`` and loads a model,
while ``resid_shards`` needs only numpy and torch.  A plain ``from .extract_resid import ...``
here would make every dataset-only consumer (the trainers, branch_metrics, knn_lp_sparse) pay
the transformers import.  PEP 562 module ``__getattr__`` keeps ``from langdual.lmsae import
ResidShardDataset`` working while importing each submodule only when it is actually touched.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ResidShardDataset",
    "extract_split",
    "verify_split",
    "ResidCapture",
]

# Attribute -> submodule.  Kept explicit (rather than scanning) so a typo raises here instead
# of importing something unexpected.
_LAZY = {
    "ResidShardDataset": "resid_shards",
    "extract_split": "extract_resid",
    "verify_split": "extract_resid",
    "ResidCapture": "extract_resid",
}


def __getattr__(name: str) -> Any:
    """Import the owning submodule on first attribute access (PEP 562)."""
    mod_name = _LAZY.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(f".{mod_name}", __name__)
    value = getattr(mod, name)
    globals()[name] = value  # cache, so subsequent accesses skip __getattr__ entirely
    return value


def __dir__():
    # Makes tab-completion and dir() show the lazy names, which module __getattr__ otherwise hides.
    return sorted(list(globals().keys()) + __all__)
