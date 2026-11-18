#!/usr/bin/env python3
"""Present a trained CB-SAE (cb_sae/bsf/cb_dual_bsf.py) through the vision forward
contract, so branch_metrics.py scores a CB / CB-DualSAE run with the SAME code
that scores every other model.

Why a shim instead of a port
----------------------------
The CB stages (dissect -> prune -> A/B/C training) already exist, are
LMDB-native, are reference-parity tested, and already rank neurons by
interpretability alone -- which is forced here, because feature space has no
generative model to steer. Re-implementing them on the vision chassis would mean
re-deriving soft-WPMI and the cosine-cubed losses for no gain. What does NOT
transfer is the SCORING: the CB package has its own purity estimator, which
differs from the one used for every other model here. Hence: reuse the trainer,
adapt the model, score it here.

The contract branch_metrics.py / knn_lp_sparse.forward_reps rely on:
    forward(x) -> (pre_acts, latents, recons)      all in INPUT space
    model._xM, model._r_hat                        the dual decomposition
    model.postprocess_output(v, mean, std)         inversion hook
    model.cfg["input_unit_norm"]

Space handling. A BSFBase normalizes internally ((x - mu) * scale) and inverts
with denormalize(), which is affine, so the branch decomposition survives it:
    denormalize(xB + x_cb + poly) - denormalize(xB + x_cb) = poly / scale
i.e. the poly contribution inverts by a SCALE ONLY -- the same asymmetry
DualSAE's own inversion has, and the reason _r_hat is computed here as a
DIFFERENCE of two input-space reconstructions rather than by inverting the poly
output directly. That makes `_xM + _r_hat == recons` exact by construction, so
branch_metrics' self-check cannot drift.

Units. A CB model's code is the concatenation [retained SAE latents ++ concept
units], so n_units = R + |C| and L0 = L0_retained + L0_concept. Those are
different mechanisms with different failure modes (dead dictionary vs dead
concepts), so the caller is expected to report them split -- `meta` carries
`n_retained` and `n_concepts` for exactly that.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

# The CB-SAE code (packages `bsf` and `featbsf`) lives in <repo>/cb_sae; make it importable
# relative to this file, so the imports below keep their original package names.
_CB_ROOT = Path(__file__).resolve().parents[1] / "cb_sae"
if _CB_ROOT.is_dir() and str(_CB_ROOT) not in sys.path:
    sys.path.insert(0, str(_CB_ROOT))


class CBRunAdapter(nn.Module):
    """A CBDualBSF wearing the vision BaseAutoencoder forward contract."""

    def __init__(self, cb, n_retained: int, n_concepts: int):
        super().__init__()
        self.cb = cb
        self.n_retained = int(n_retained)
        self.n_concepts = int(n_concepts)
        # input_unit_norm False: the BSF model does its own normalization and
        # returns INPUT-space tensors, so this class must not apply a second
        # per-sample standardization on top. postprocess_output is therefore
        # the identity (see below) and _x_mean/_x_std stay None, which is the
        # "no inversion needed" value forward_reps already understands.
        self.cfg: Dict[str, Any] = {"input_unit_norm": False,
                                    "dict_size": self.n_retained + self.n_concepts}

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        """Identity. Kept as a method because forward_reps calls it by name on
        every model; returning the input unchanged is correct here precisely
        because forward() already emits input-space tensors."""
        return x_reconstruct

    @torch.no_grad()
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from bsf.models import block_norms
        m = self.cb
        x_norm = m.normalize(x)

        # One manual pass rather than unit_acts() + full_recon(), each of which
        # re-encodes the frozen base. The KNN/LP bank streams >1M rows through
        # here, so the duplicate encode is worth avoiding. Mirrors
        # cb_sae/featbsf/adapters.py::_CBDualAdapted.acts_and_recon.
        _, z, _ = m.base.encode(x_norm)
        xB = m.base.decode(z)
        # SAE-port bases reconstruct decode(z) + b_dec; BSF bases have no
        # b_dec attribute at all.
        b_dec = getattr(m.base, "b_dec", None)
        if b_dec is not None:
            xB = xB + b_dec
        z_cb, _ = m.sigma_cb(m.concept_pre(x_norm))
        x_cb = m.cb_recon(z_cb)

        deploy = xB + x_cb                       # the STEERABLE reconstruction
        full = deploy
        if m.use_poly:
            full = deploy + m.base.poly(x_norm - deploy)

        recons = m.denormalize(full)
        xM = m.denormalize(deploy)
        # Difference of two input-space reconstructions, so xM + r_hat == recons
        # holds EXACTLY regardless of how denormalize is parameterised.
        self._xM = xM
        self._r_hat = (recons - xM) if m.use_poly else None
        self._x_mean = None
        self._x_std = None
        self._x_preprocessed = x_norm

        acts = torch.cat([block_norms(z), block_norms(z_cb)], dim=1).float()
        # pre_acts == latents: a CB model has no meaningful pre-gate dense code
        # that is comparable across its two halves (the retained half is gated
        # by the base's own rule, the concept half by top-k_cb), and no consumer
        # in this suite reads element 0 for a non-dual model.
        return acts, acts, recons


def load_cb_run(run_dir: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild a CB run for offline scoring. Mirrors knn_lp_sparse.load_run's
    return contract so the caller needs no special case beyond dispatch."""
    from bsf.cb_dual_bsf import load_cb_dual

    run_dir = Path(run_dir)
    ckpt = run_dir / "model.pt"
    cfg_path = run_dir / "training_config.json"
    if not ckpt.exists():
        raise FileNotFoundError(f"no model.pt in {run_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"no training_config.json in {run_dir}")
    tcfg = json.loads(cfg_path.read_text())

    cb = load_cb_dual(ckpt, device=str(device)).eval()
    n_concepts = int(tcfg.get("n_concepts", cb.n_concepts))
    n_retained = int(tcfg.get("retain", cb.base.G))
    model = CBRunAdapter(cb, n_retained, n_concepts).to(device).eval()

    meta: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "exp_name": tcfg.get("exp_name", run_dir.name),
        "model_type": tcfg.get("model_type", "cb_sae"),
        "sparsity_mode": tcfg.get("sparsity_mode"),
        "seed": tcfg.get("seed"),
        "n_latents": n_retained + n_concepts,
        "act_size": int(cb.d),
        "dict_size": n_retained + n_concepts,
        # Reported SPLIT, because the two halves fail differently: the retained
        # dictionary can go dead, the concept bank can die from too small a
        # k_cb/|C| active rate. A single pooled number would hide the latter
        # entirely whenever |C| >> R.
        "n_retained": n_retained,
        "n_concepts": n_concepts,
        "k_cb": tcfg.get("k_cb"),
        "cset": tcfg.get("cset"),
        "base_exp_name": tcfg.get("base_exp_name"),
        # A CB model has no single k: the retained half keeps its base's gate
        # (and under a batch-global gate pruning is not even exact), the
        # concept half is top-k_cb. Never advertise a nominal budget.
        "effective_k": None,
        "is_dual": bool(getattr(cb, "use_poly", False)),
        "input_unit_norm": False,
        "multilabel": tcfg.get("multilabel"),
        "labels_lmdb": tcfg.get("labels_lmdb"),
        "dataset_name": tcfg.get("dataset_name"),
        "extractor_name": tcfg.get("extractor_name"),
        "train_path": tcfg.get("train_path"),
        "val_path": tcfg.get("val_path"),
        "test_path": tcfg.get("test_path"),
        "batch_size": tcfg.get("batch_size"),
    }
    return model, meta
