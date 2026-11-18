"""Rebuild a trained SAE / DualSAE from its run directory (vendored loader).

Copied from the evaluation code of the SAE pipeline (``load_run`` and ``forward_reps``); the
model classes come from the shared core in ``<repo>/vision`` (flat imports, as the trainers use
them).  ``training_config.json`` is the authoritative source for the model family; the
checkpoint's ``model_cfg`` (or, for ``autoencoder_final.pt``, the sibling
``best_val_monoscore_model.pt``'s ``model_cfg``) carries the exact constructor cfg.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

# The shared SAE core lives in <repo>/vision and imports its own modules by bare name
# (``from models import ...``), so its directory goes on sys.path, computed relative to this file.
_VISION = Path(__file__).resolve().parents[2] / "vision"
if str(_VISION) not in sys.path:
    sys.path.insert(0, str(_VISION))

from models import (BatchTopKSAE, DualSAE, GatedSAE, JumpReLUSAE,  # noqa: E402
                    MatryoshkaSAE, MSAE, TopK, TopKSAE, VanillaSAE)

# Which saved checkpoint is read.  The steering pipeline always reads "final"
# (autoencoder_final.pt, the end of training); "best" (best_val_monoscore_model.pt) is kept for
# parity with the evaluation code this loader was copied from.
CHECKPOINT_CHOICES = ("best", "final")


def resolve_checkpoint(run_dir: Path, which: str = "best") -> Path:
    """Map a checkpoint selector to a file, failing loudly rather than falling back silently.

    A silent fallback to the other checkpoint would make two runs' numbers incomparable without
    anything in the output saying so, which is the failure this whole option exists to expose.
    """
    if which not in CHECKPOINT_CHOICES:
        raise ValueError(f"checkpoint must be one of {CHECKPOINT_CHOICES}, got {which!r}")
    name = "best_val_monoscore_model.pt" if which == "best" else "autoencoder_final.pt"
    path = run_dir / name
    if not path.exists():
        raise FileNotFoundError(f"no {name} in {run_dir} (checkpoint={which!r})")
    return path


def load_run(run_dir: Path, device: torch.device,
             checkpoint: str = "best") -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild the trained model from a run directory.

    ``training_config.json`` is the authoritative source for the model family and
    for ``normalize``; the checkpoint's ``model_cfg`` (when present) carries the
    exact constructor cfg.  The TopK baselines predate ``model_cfg`` entirely, so
    they are rebuilt from the training config plus the state dict -- note that
    ``normalize`` genuinely varies across the paper's runs (True for TopK and
    JumpReLU, False for BatchTopK and the TopK/BatchTopK Duals), so it must never
    be hard-coded.
    """
    cfg_path = run_dir / "training_config.json"
    ckpt_path = resolve_checkpoint(run_dir, checkpoint)
    if not cfg_path.exists():
        raise FileNotFoundError(f"no training_config.json in {run_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        # autoencoder_final.pt stores the state dict directly and carries no epoch/val metadata.
        state, ckpt = ckpt, {}
    tcfg = json.loads(cfg_path.read_text())
    model_type = str(tcfg.get("model_type", "topk"))

    meta: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "exp_name": tcfg.get("exp_name", run_dir.name),
        "model_type": model_type,
        "sparsity_mode": tcfg.get("sparsity_mode"),
        "seed": tcfg.get("seed"),
        "n_latents": tcfg.get("n_latents"),
        "topk_k": tcfg.get("topk_k", tcfg.get("top_k")),
        "span_coef": tcfg.get("span_coef"),
        "warmup_steps": tcfg.get("warmup_steps"),
        "lr": tcfg.get("lr"),
        "mono_coef": tcfg.get("mono_coef"),
        "normalize": tcfg.get("normalize"),
        # None when the run predates the flag, which is exactly the "no opinion" value
        # resolve_multilabel expects, so old runs fall through to the legacy behaviour.
        "multilabel": tcfg.get("multilabel"),
        "labels_lmdb": tcfg.get("labels_lmdb"),
        "steps_per_epoch": tcfg.get("steps_per_epoch"),
        "dataset_name": tcfg.get("dataset_name"),
        "extractor_name": tcfg.get("extractor_name"),
        "batch_size": tcfg.get("batch_size"),
        "train_path": tcfg.get("train_path"),
        "val_path": tcfg.get("val_path"),
        "test_path": tcfg.get("test_path"),
        # Recorded so a result JSON always says WHICH checkpoint produced it.  Without this a
        # re-scored number is indistinguishable from the original.
        "checkpoint": checkpoint,
        "checkpoint_path": str(ckpt_path),
        "best_epoch": ckpt.get("epoch"),
        "val_monoscore": ckpt.get("val_monoscore"),
    }

    if model_type == "topk":
        # TopKSAE takes explicit kwargs rather than a cfg dict.  ``activation.k``
        # lives in the state dict (TopK.state_dict injects it), which is a more
        # reliable source than the config for what the model actually ran with.
        n_latents, d_model = state["encoder.weight"].shape
        k = int(state.get("activation.k", tcfg.get("topk_k", 64)))
        activation = TopK(k=k) if tcfg.get("activation", "topk") == "topk" else nn.ReLU()
        model = TopKSAE(
            n_latents=n_latents,
            n_inputs=d_model,
            activation=activation,
            tied=bool(tcfg.get("tied_weights", True)),
            normalize=bool(tcfg.get("normalize", False)),
        )
        # ``activation.k``/``activation.postact_fn`` are metadata, not parameters:
        # they are injected by TopK.state_dict and must be dropped before loading.
        clean = {
            key: value for key, value in state.items()
            if key not in ("activation", "activation_state_dict", "activation.k", "activation.postact_fn")
        }
        model.load_state_dict(clean, strict=True)
        # ``input_unit_norm`` is the DualSAE/BaseAutoencoder spelling of what
        # TopKSAE calls ``normalize``; both mean "per-sample standardize the input
        # before encoding and undo it after decoding".  The TopK baselines predate
        # ``model_cfg`` so the flag only exists in training_config.json as
        # ``normalize``.  Mirroring it into ``meta`` here (additive: no existing
        # key changes) means downstream callers can read one key for all model
        # families instead of a ``.get(..., .get(...))`` cascade that silently
        # defaults to False for every TopK run.
        meta.update(act_size=int(d_model), dict_size=int(n_latents), effective_k=k, is_dual=False,
                    input_unit_norm=bool(tcfg.get("normalize", False)))
    else:
        model_cfg = ckpt.get("model_cfg")
        if not model_cfg:
            # autoencoder_final.pt is saved as a BARE state dict with no model_cfg, while
            # best_val_monoscore_model.pt in the SAME run directory carries one.  The two
            # checkpoints are two points on one training trajectory: identical architecture,
            # different weights.  So borrowing the sibling's model_cfg is exact, not a guess --
            # and strict load_state_dict below would fail loudly if the shapes disagreed.
            # Without this, --checkpoint final is unusable for every Dual run, which is most of
            # the cells this option exists to re-examine.
            sibling = run_dir / "best_val_monoscore_model.pt"
            if sibling.exists() and sibling != ckpt_path:
                sib = torch.load(sibling, map_location="cpu", weights_only=False)
                model_cfg = sib.get("model_cfg") if isinstance(sib, dict) else None
                if model_cfg:
                    meta["model_cfg_source"] = str(sibling)
        if not model_cfg:
            raise ValueError(
                f"{ckpt_path} has no model_cfg and model_type={model_type!r} cannot be "
                "rebuilt from training_config.json alone"
            )
        model_cfg = dict(model_cfg)
        model_cfg["device"] = str(device)
        model_cfg["dtype"] = torch.float32
        if model_type == "dual":
            model = DualSAE(model_cfg)
        elif model_type == "batch_topk":
            model = BatchTopKSAE(model_cfg)
        elif model_type == "jumprelu":
            model = JumpReLUSAE(model_cfg)
        elif model_type == "vanilla":
            model = VanillaSAE(model_cfg)
        elif model_type == "gated":
            # GatedSAE has no top_k: its L0 is emergent from l1_coeff, so
            # `effective_k` below stays None and every caller must read the
            # MEASURED l0_mean instead of assuming the nominal budget.
            model = GatedSAE(model_cfg)
        elif model_type == "matryoshka":
            model = MatryoshkaSAE(model_cfg)
        elif model_type == "msae":
            model = MSAE(model_cfg)
        else:
            raise ValueError(
                f"unknown model_type: {model_type!r} (known: topk, batch_topk, "
                "vanilla, jumprelu, gated, matryoshka, msae, dual)")
        model.load_state_dict(state, strict=True)
        meta.update(
            act_size=int(model_cfg["act_size"]),
            dict_size=int(model_cfg["dict_size"]),
            effective_k=model_cfg.get("top_k"),
            is_dual=(model_type == "dual"),
            input_unit_norm=bool(model_cfg.get("input_unit_norm", False)),
        )

    return model.to(device).eval(), meta


def forward_reps(model: nn.Module, x: torch.Tensor, is_dual: bool) -> Dict[str, torch.Tensor]:
    """Run one forward pass and return every representation we may need.

    Keys returned (all in the ORIGINAL feature space unless stated otherwise):

    ``x``      the raw input batch, echoed back so callers that want the
               reference representation do not have to thread it separately.
               It is deliberately the caller's tensor and **never**
               ``model._x_preprocessed``: the preprocessed vector is
               ``(x - mean)/std`` per sample, and a subsequent row
               L2-normalization does *not* undo that rank-1 mean projection, so
               using it would make the "raw embedding" column silently
               model-dependent (it differs between ``normalize=True`` and
               ``normalize=False`` runs).
    ``z``      the sparse code.
    ``xM``     the mono reconstruction.  DualSAE stores ``_xM`` in the
               *preprocessed* space, so it has to go back through
               ``postprocess_output`` (``xM_p * std + mean``) to live in the same
               space as ``x`` and ``x_hat`` -- a no-op unless ``input_unit_norm``
               is set, which it is for the TopK/JumpReLU baselines and the
               Dual-JumpReLU runs.
    ``xhat``   the full reconstruction, exactly as the model returns it.
    ``r_hat``  DUAL ONLY (absent for single-branch models, where the quantity is
               undefined -- never emit zeros for it).  The PolyBranch output.

    The ``r_hat`` inversion is the one genuine trap here.  DualSAE.forward does::

        x_hat = postprocess(xM_p + r_hat_p) = (xM_p + r_hat_p) * std + mean
              = (xM_p * std + mean) + r_hat_p * std
              = xM_orig                + r_hat_p * std

    so the original-space poly output is ``r_hat_p * std`` -- a SCALE-ONLY
    inversion.  Running ``postprocess_output`` on ``_r_hat`` would add ``x_mean``
    a second time and break the decomposition; that mistake perturbs the signal while barely
    rotating the vector, so it is nearly invisible to any metric that
    L2-normalizes rows and shows up only in R^2 and MonoScore.  Callers should assert ``xM + r_hat == x_hat`` explicitly.

    When ``input_unit_norm`` is False, ``_x_mean``/``_x_std`` are ``None`` and both
    inversions are the identity.
    """
    _, z, x_hat = model(x)
    out = {"x": x, "z": z, "xhat": x_hat}
    if is_dual:
        x_mean = getattr(model, "_x_mean", None)
        x_std = getattr(model, "_x_std", None)
        out["xM"] = model.postprocess_output(model._xM, x_mean, x_std)
        r_hat_p = model._r_hat
        # Scale only -- see the docstring.  ``x_std`` is (B, 1) so it broadcasts.
        out["r_hat"] = r_hat_p if x_std is None else r_hat_p * x_std
    else:
        # Single-branch models have no separate mono reconstruction: x^M IS x_hat.
        # ``r_hat`` is intentionally absent rather than zero -- there is no poly
        # branch to read, not even accidentally (verified: baselines have no
        # ``_xM``/``_r_hat`` attribute at all after forward()).
        out["xM"] = x_hat
    return out


# --------------------------------------------------------------------------- #
# Banks
