#!/usr/bin/env python3
"""KNN and linear-probe evaluation on the SPARSE CODE of an SAE / DualSAE.

Motivation
----------
KNN and LP accuracy are reported on the *reconstructed* embedding ``x_hat``.
For DualSAE ``x_hat = x^M + r_hat``, so those two columns measure the Sparse
Branch and the Residual Branch together and cannot say what the sparse code
itself retains.  This module runs the *same* KNN/LP protocol on:

  ``z``     the sparse code (dict_size-dim).  For DualSAE this is the MonoBranch
            code -- the poly branch has no latent code at all, it is a dense MLP
            on the residual -- and for the baselines it is their only code.  Both
            sides have the same dict_size (8192) and the same k (64), so the
            comparison is apples-to-apples within an activation pair.
  ``xM``    the mono reconstruction ``x^M = z W_dec + b_dec`` (act_size-dim).
            Shows what the sparse code alone decodes to, i.e. it removes the
            Residual Branch's contribution from the ``x_hat`` number.  For
            non-dual models ``x^M == x_hat``, which also makes it a free
            consistency check.
  ``xhat``  the full reconstruction.  Recomputing it here is a consistency
            check of the loader and the protocol against the trainers'
            in-training evaluation of the same checkpoint.

Protocol parity
---------------
Every number uses exactly the recipe of ``train_dual.py`` /
``train_topk.py`` / ``train_other.py`` so the sparse columns are directly
comparable to the reconstruction columns:

* representation rows are L2-normalized (``_build_reconstruction_bank`` does
  this for both KNN *and* LP, so the sparse path normalizes for both too),
* KNN: k=20, cosine similarity against the full train bank, unweighted majority
  vote -> top-1 accuracy (single-label) or neighbour-mean >= 0.5 -> micro-F1
  (multi-label),
* LP: ``nn.Linear(dim, C)``, AdamW(lr=1e-3, weight_decay=1e-4), 10 epochs,
  batch 1024, CrossEntropy / BCEWithLogits, argmax / sigmoid>=0.5.

Why a dedicated implementation instead of reusing ``_build_reconstruction_bank``
--------------------------------------------------------------------------------
A dense bank of ImageNet-train sparse codes would be 1,281,167 x 8192 x 4 B =
42 TB.  The codes are sparse (measured L0: 64 for TopK/BatchTopK and their Dual
twins, ~27-47 for JumpReLU, ~530-740 for Dual-JumpReLU), so the bank is stored
CSR-style: 0.66 GB for a k=64 model, 5.4 GB for Dual-JumpReLU.  KNN then uses a
sparse-times-dense product whose cost is ``nnz(bank) x n_queries`` rather than
``n_bank x n_queries x dict_size`` -- ~130x fewer FLOPs than a dense GEMM at
L0=64, which is what makes the full 1.28M-row bank affordable at all.

Sparsity caveat (report L0 with every number)
---------------------------------------------
TopK and BatchTopK pin L0=k on both sides of a pair, but JumpReLU does not: the
learned thresholds give the baseline L0~27 and Dual-JumpReLU L0~530 on ImageNet.
A denser code trivially retains more information, so ``l0_mean`` is emitted for
every representation and MUST be shown next to the JumpReLU rows.  Use
``--truncate-topk 64`` to additionally score every model at a matched L0.

BatchTopK caveat
----------------
``BatchTopKSAE`` (and DualSAE with ``sparsity_mode='batch_topk'``) selects the
top ``k * batch_size`` activations across the WHOLE batch, so its code depends
on the encoding batch size.  ``--encode-batch-size`` therefore defaults to 2048,
the training batch size, and is recorded in the output JSON.

Usage
-----
    python knn_lp_sparse.py --run-dir ./checkpoints/<exp> \
        --eval-split val --reps z xM xhat --out-dir ./outputs/sparse_knn_lp
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.lmdb_features import LMDBFeatureDataset
from dataset.resolve import (
    build_feature_dataset,
    is_legacy_multilabel_path,
    resolve_multilabel,
)
from models import (BatchTopKSAE, DualSAE, GatedSAE, JumpReLUSAE, MatryoshkaSAE,
                    MSAE, TopK, TopKSAE, VanillaSAE)
from utils import unpack_batch

# Protocol constants, copied from the trainers' call sites so the sparse numbers
# and the trainers' reconstruction numbers are produced by the same recipe.
KNN_K = 20
LP_EPOCHS = 10
LP_BATCH_SIZE = 1024
LP_LR = 1e-3
LP_WEIGHT_DECAY = 1e-4
LP_SEED = 0  # probe init + epoch permutations; fixed so re-runs are reproducible


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

# Which saved checkpoint a scorer reads.  "best" (best_val_monoscore_model.pt) is the DEFAULT
# and is the checkpoint the trainers evaluate.
#
# Why this is selectable at all: best_val_monoscore_model.pt is chosen by validation MonoScore,
# and MonoScore can be HIGHEST very early, before the dictionary has specialised (in some
# settings the selected checkpoint is epoch 1).  A sweep scored that way can report identical
# numbers for settings whose trained models differ, because the knob being swept has not yet
# acted.  "final" reads autoencoder_final.pt (end of training) so a sweep can be re-scored at
# convergence and the two readings compared.
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
    # CB runs are POST-HOC: they never pass through a trainer, so they carry a
    # cb_dual_bsf `model.pt` instead of best_val_monoscore_model.pt and are
    # rebuilt by their own loader. Dispatch before the checkpoint check below,
    # which would otherwise reject them for a file they are not supposed to
    # have. Everything downstream (forward_reps, branch_metrics) then treats
    # them like any other model.
    if cfg_path.exists():
        _tc = json.loads(cfg_path.read_text())
        if str(_tc.get("model_type", "")) in ("cb_sae", "cb_dual_sae"):
            from cb_adapter import load_cb_run
            return load_cb_run(run_dir, device)

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
        # defaults to False for the TopK baselines.
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
    a second time and break the decomposition; measured on the real Dual-JumpReLU
    checkpoints that mistake is a ~12% perturbation of the signal that barely
    rotates the vector (cosine 0.997-0.999), so it is nearly invisible to any
    metric that L2-normalizes rows (KNN/LP/retrieval) and shows up only in R^2 and
    MonoScore.  Callers should assert ``xM + r_hat == x_hat`` explicitly.

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
# --------------------------------------------------------------------------- #

@dataclass
class SparseBank:
    """CSR-style bank of L2-normalized sparse codes, held on one device.

    ``indptr`` is int64 (nnz can exceed 2^31 for dense JumpReLU codes over the
    1.28M-row ImageNet bank), ``indices`` is int32 (dict_size <= 8192 so column
    ids are tiny), ``values`` is float32.
    """

    indptr: torch.Tensor
    indices: torch.Tensor
    values: torch.Tensor
    n_rows: int
    n_cols: int
    labels: torch.Tensor

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1].item())

    @property
    def l0_mean(self) -> float:
        return self.nnz / max(self.n_rows, 1)

    def to(self, device: torch.device) -> "SparseBank":
        if self.values.device == device:
            return self
        return SparseBank(
            self.indptr.to(device), self.indices.to(device), self.values.to(device),
            self.n_rows, self.n_cols, self.labels,
        )

    def csr_slice(self, start: int, end: int) -> torch.Tensor:
        """Rows [start, end) as a ``torch.sparse_csr_tensor`` for sparse.mm."""
        ptr = self.indptr[start:end + 1]
        base = ptr[0]
        lo, hi = int(base.item()), int(ptr[-1].item())
        return torch.sparse_csr_tensor(
            (ptr - base).to(torch.int64),
            self.indices[lo:hi].to(torch.int64),
            self.values[lo:hi],
            size=(end - start, self.n_cols),
        )

    def gather_dense(self, row_ids: torch.Tensor) -> torch.Tensor:
        """Materialize the given rows as a dense (len(row_ids), n_cols) tensor.

        Fully vectorized: no Python loop over rows, which matters because the LP
        does 10 epochs x ~1250 batches over the ImageNet bank.
        """
        starts = self.indptr[row_ids]
        counts = self.indptr[row_ids + 1] - starts
        total = int(counts.sum().item())
        out = torch.zeros((row_ids.numel(), self.n_cols), dtype=torch.float32, device=self.values.device)
        if total == 0:
            return out
        # flat positions of every nnz belonging to the selected rows
        row_of = torch.repeat_interleave(
            torch.arange(row_ids.numel(), device=row_ids.device), counts
        )
        offset_in_row = torch.arange(total, device=row_ids.device) - torch.repeat_interleave(
            torch.cumsum(counts, 0) - counts, counts
        )
        flat = torch.repeat_interleave(starts, counts) + offset_in_row
        out[row_of, self.indices[flat].to(torch.int64)] = self.values[flat]
        return out


@dataclass
class DenseBank:
    """Dense bank of L2-normalized representations (act_size-dim)."""

    x: torch.Tensor          # (N, D) float32
    labels: torch.Tensor
    n_rows: int = field(init=False)
    n_cols: int = field(init=False)

    def __post_init__(self):
        self.n_rows, self.n_cols = int(self.x.shape[0]), int(self.x.shape[1])

    @property
    def l0_mean(self) -> float:
        # Dense representations are fully occupied by construction; reporting the
        # dimension keeps the l0 column meaningful across sparse and dense reps.
        return float(self.n_cols)

    def to(self, device: torch.device) -> "DenseBank":
        if self.x.device == device:
            return self
        return DenseBank(self.x.to(device), self.labels)

    def gather_dense(self, row_ids: torch.Tensor) -> torch.Tensor:
        return self.x[row_ids]


def _truncate_topk(z: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the k largest entries per row (sparsity-matched control)."""
    if k <= 0 or k >= z.shape[1]:
        return z
    vals, idx = torch.topk(z, k=k, dim=1)
    out = torch.zeros_like(z)
    out.scatter_(1, idx, vals)
    return out


@torch.no_grad()
def build_banks(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    reps: List[str],
    is_dual: bool,
    split_name: str,
    truncate_topk: int = 0,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """One pass over ``loader`` producing every requested representation's bank.

    Doing all representations in a single pass matters: encoding the 1.28M-row
    ImageNet train split is I/O bound, and the trainers' version re-walks it once
    for KNN and again for LP.
    """
    model.eval()
    want_sparse = "z" in reps
    dense_reps = [r for r in reps if r != "z"]

    idx_chunks: List[torch.Tensor] = []
    val_chunks: List[torch.Tensor] = []
    counts_chunks: List[torch.Tensor] = []
    dense_acc: Dict[str, List[torch.Tensor]] = {r: [] for r in dense_reps}
    label_chunks: List[torch.Tensor] = []
    seen = 0

    for batch in tqdm(loader, desc=f"encode {split_name}"):
        features, class_idx, _ = unpack_batch(batch)
        if class_idx is None:
            raise ValueError("evaluation requires labels; open the LMDB with return_class_idx=True")
        x = features.to(device, non_blocking=True).float()
        out = forward_reps(model, x, is_dual)

        if want_sparse:
            z = out["z"].float()
            if truncate_topk:
                z = _truncate_topk(z, truncate_topk)
            # L2-normalize the row (parity with _build_reconstruction_bank, which
            # normalizes for KNN *and* LP), then keep only the strictly positive
            # entries -- every activation here is post-ReLU/post-JumpReLU so a
            # zero carries no information.
            z = F.normalize(z, p=2, dim=1)
            mask = z > 0
            counts = mask.sum(dim=1)
            cols = mask.nonzero(as_tuple=True)[1]
            idx_chunks.append(cols.to(torch.int32).cpu())
            val_chunks.append(z[mask].cpu())
            counts_chunks.append(counts.cpu())

        for rep in dense_reps:
            dense_acc[rep].append(F.normalize(out[rep].float(), p=2, dim=1).cpu())

        if class_idx.ndim == 2:
            label_chunks.append(class_idx.detach().cpu().float())
        else:
            label_chunks.append(class_idx.detach().cpu().long())

        seen += x.shape[0]
        if max_samples is not None and seen >= max_samples:
            break

    if seen == 0:
        raise ValueError(f"no samples encoded for split {split_name}")

    labels = torch.cat(label_chunks, dim=0)
    if max_samples is not None and labels.shape[0] > max_samples:
        labels = labels[:max_samples]
    n_rows = labels.shape[0]

    banks: Dict[str, Any] = {}
    if want_sparse:
        counts = torch.cat(counts_chunks)[:n_rows]
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64)
        torch.cumsum(counts.to(torch.int64), 0, out=indptr[1:])
        nnz = int(indptr[-1].item())
        banks["z"] = SparseBank(
            indptr=indptr,
            indices=torch.cat(idx_chunks)[:nnz],
            values=torch.cat(val_chunks)[:nnz],
            n_rows=n_rows,
            n_cols=int(model.W_enc.shape[1]) if hasattr(model, "W_enc") else int(model.encoder.weight.shape[0]),
            labels=labels,
        )
    for rep in dense_reps:
        banks[rep] = DenseBank(torch.cat(dense_acc[rep], dim=0)[:n_rows], labels)
    return banks


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

class MicroF1:
    """Micro-F1 accumulated globally over all samples and classes.

    The trainers instead average a per-batch micro-F1 weighted by batch size,
    which is a slightly different estimator.  Both are tracked: ``value`` is
    micro-F1 accumulated over all classes and evaluation examples, and
    ``batchwise`` is the trainers' per-batch, sample-weighted estimator, so the
    two can be compared on the same run.
    """

    def __init__(self) -> None:
        self.tp = self.fp = self.fn = 0
        self._bw_sum = 0.0
        self._bw_n = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.bool()
        target = target.bool()
        tp = int((pred & target).sum().item())
        fp = int((pred & ~target).sum().item())
        fn = int((~pred & target).sum().item())
        self.tp += tp
        self.fp += fp
        self.fn += fn
        denom = 2 * tp + fp + fn
        self._bw_sum += (2.0 * tp / denom if denom > 0 else 0.0) * pred.shape[0]
        self._bw_n += pred.shape[0]

    @property
    def value(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return (2.0 * self.tp / denom) if denom > 0 else 0.0

    @property
    def batchwise(self) -> float:
        return self._bw_sum / max(self._bw_n, 1)


def _merge_topk(
    best_vals: Optional[torch.Tensor], best_lbls: Optional[torch.Tensor],
    cand_vals: torch.Tensor, cand_lbls: torch.Tensor, k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep the global top-k across bank chunks (same merge as the trainers)."""
    if best_vals is None:
        return cand_vals, cand_lbls
    merged_vals = torch.cat([best_vals, cand_vals], dim=1)
    merged_lbls = torch.cat([best_lbls, cand_lbls], dim=1)
    keep = torch.topk(merged_vals, k=min(k, merged_vals.shape[1]), dim=1).indices
    vals = torch.gather(merged_vals, 1, keep)
    if merged_lbls.ndim == 2:
        lbls = torch.gather(merged_lbls, 1, keep)
    else:  # multi-label: (B, cand, C)
        lbls = torch.gather(merged_lbls, 1, keep.unsqueeze(-1).expand(-1, -1, merged_lbls.shape[-1]))
    return vals, lbls


@torch.no_grad()
def knn_metric(
    train_bank: Any,
    eval_bank: Any,
    device: torch.device,
    k: int = KNN_K,
    query_batch: int = 1024,
    bank_chunk: int = 262_144,
) -> Dict[str, float]:
    """Cosine KNN of ``eval_bank`` rows against ``train_bank``.

    Rows are already L2-normalized, so the inner product IS the cosine.  For the
    sparse path the similarity is computed as ``bank_chunk_csr @ q^T``, whose cost
    scales with ``nnz(bank)`` instead of ``n_bank * dict_size``.
    """
    sparse = isinstance(train_bank, SparseBank)
    n_train = train_bank.n_rows
    effective_k = min(k, n_train)
    if effective_k <= 0:
        return {"metric": 0.0}

    train_labels = train_bank.labels
    is_multilabel = train_labels.ndim == 2
    num_classes = train_labels.shape[1] if is_multilabel else int(train_labels.max().item()) + 1
    train_labels_dev = train_labels.to(device)

    correct = 0
    total = 0
    f1 = MicroF1()

    for q0 in tqdm(range(0, eval_bank.n_rows, query_batch), desc="knn"):
        q1 = min(q0 + query_batch, eval_bank.n_rows)
        q = eval_bank.gather_dense(torch.arange(q0, q1, device=device))
        y = eval_bank.labels[q0:q1].to(device)
        # cuSPARSE SpMM wants a contiguous dense operand; the transpose is built
        # once per query batch (D x B, a few tens of MB) rather than per chunk.
        qT = q.t().contiguous() if sparse else None

        best_vals = best_lbls = None
        for b0 in range(0, n_train, bank_chunk):
            b1 = min(b0 + bank_chunk, n_train)
            if sparse:
                # (chunk, B).  Reducing along dim=0 and transposing only the
                # (k, B) result avoids materializing a second (B, chunk) copy,
                # which at chunk=262144 would be another ~1 GB per iteration.
                sim = torch.sparse.mm(train_bank.csr_slice(b0, b1), qT)
                kk = min(effective_k, sim.shape[0])
                vals, inds = torch.topk(sim, k=kk, dim=0)
                vals, inds = vals.t().contiguous(), inds.t().contiguous()
            else:
                sim = q @ train_bank.x[b0:b1].t()
                kk = min(effective_k, sim.shape[1])
                vals, inds = torch.topk(sim, k=kk, dim=1)
            best_vals, best_lbls = _merge_topk(
                best_vals, best_lbls, vals, train_labels_dev[b0:b1][inds], effective_k
            )
            del sim

        if is_multilabel:
            pred = best_lbls.float().mean(dim=1) >= 0.5
            f1.update(pred, y.float() >= 0.5)
            total += y.shape[0]
        else:
            votes = F.one_hot(best_lbls, num_classes=num_classes).sum(dim=1)
            correct += int((votes.argmax(dim=1) == y.long()).sum().item())
            total += y.shape[0]

    if is_multilabel:
        return {"metric": f1.value, "micro_f1": f1.value,
                "micro_f1_batchwise": f1.batchwise, "metric_type": "micro_f1"}
    return {"metric": correct / max(total, 1), "accuracy": correct / max(total, 1),
            "metric_type": "accuracy"}


def lp_metric(
    train_bank: Any,
    eval_bank: Any,
    device: torch.device,
    epochs: int = LP_EPOCHS,
    batch_size: int = LP_BATCH_SIZE,
    lr: float = LP_LR,
    weight_decay: float = LP_WEIGHT_DECAY,
) -> Dict[str, float]:
    """Linear probe fitted on the train bank, scored on the eval bank."""
    feat_dim = train_bank.n_cols
    train_y = train_bank.labels
    is_multilabel = train_y.ndim == 2
    num_classes = train_y.shape[1] if is_multilabel else int(train_y.max().item()) + 1

    # Fixed seed: the trainers leave the probe to the ambient RNG, which makes
    # their LP number irreproducible at the 1e-4 level.  Pinning it here costs
    # nothing and makes re-runs byte-stable.
    torch.manual_seed(LP_SEED)
    probe = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()

    n_train = train_bank.n_rows
    generator = torch.Generator().manual_seed(LP_SEED)
    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(n_train, generator=generator)
        running = 0.0
        for start in tqdm(range(0, n_train, batch_size), desc=f"lp epoch {epoch + 1}/{epochs}", leave=False):
            idx = perm[start:start + batch_size]
            xb = train_bank.gather_dense(idx.to(device))
            yb = train_y[idx].to(device)
            logits = probe(xb)
            loss = criterion(logits, yb.float() if is_multilabel else yb.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * xb.shape[0]
        print(f"    lp epoch {epoch + 1}/{epochs}: loss={running / max(n_train, 1):.4f}")

    probe.eval()
    correct = 0
    total = 0
    f1 = MicroF1()
    with torch.no_grad():
        for start in range(0, eval_bank.n_rows, batch_size):
            end = min(start + batch_size, eval_bank.n_rows)
            xb = eval_bank.gather_dense(torch.arange(start, end, device=device))
            yb = eval_bank.labels[start:end].to(device)
            logits = probe(xb)
            if is_multilabel:
                f1.update(torch.sigmoid(logits) >= 0.5, yb.float() >= 0.5)
                total += yb.shape[0]
            else:
                correct += int((logits.argmax(dim=1) == yb.long()).sum().item())
                total += yb.shape[0]

    if is_multilabel:
        return {"metric": f1.value, "micro_f1": f1.value,
                "micro_f1_batchwise": f1.batchwise, "metric_type": "micro_f1"}
    return {"metric": correct / max(total, 1), "accuracy": correct / max(total, 1),
            "metric_type": "accuracy"}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _is_open_images(path: Optional[str]) -> bool:
    """Deprecated shim; branch_metrics re-exports this name.

    Multi-label mode is now resolved by ``dataset.resolve.resolve_multilabel``, which prefers
    the value the RUN recorded in training_config.json over any inspection of the path.  That
    ordering matters: the recorded value provably describes the run that produced the
    checkpoint, whereas a path can be moved, renamed, or merely contain an unlucky substring.
    """
    return is_legacy_multilabel_path(path)


def make_loader(path: str, batch_size: int, num_workers: int, multilabel: bool) -> DataLoader:
    # The factory serves an LMDB of embeddings and a directory of activation shards through one
    # __getitem__ contract, so the residual-stream track needs no separate scorer.
    dataset = build_feature_dataset(
        path, return_class_idx=True, return_multilabel=multilabel, verbose=False
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="checkpoint directory (holds best_val_monoscore_model.pt)")
    p.add_argument("--checkpoint", default="best", choices=("best", "final"),
                   help='which saved checkpoint to score: "best" = best_val_monoscore_model.pt (default; the checkpoint the trainers evaluate), "final" = autoencoder_final.pt (end of training). MonoScore often peaks at epoch 1, so "best" can score a model from before the swept knob has acted; "final" re-scores at convergence.')
    p.add_argument("--eval-split", default="val", choices=["val", "test"],
                   help="split the metrics are reported on; the bank is always the train split")
    p.add_argument("--reps", nargs="+", default=["z", "xM", "xhat"], choices=["z", "xM", "xhat"])
    p.add_argument("--out-dir", default="./outputs/sparse_knn_lp")
    p.add_argument("--encode-batch-size", type=int, default=2048,
                   help="MUST match training (2048) -- BatchTopK's top-k is batch-level")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--knn-k", type=int, default=KNN_K)
    p.add_argument("--knn-query-batch", type=int, default=1024)
    p.add_argument("--knn-bank-chunk", type=int, default=262_144)
    p.add_argument("--truncate-topk", type=int, default=0,
                   help="keep only the top-N code entries per row (sparsity-matched control; 0=off)")
    p.add_argument("--max-train-samples", type=int, default=None, help="smoke-test cap on the bank")
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true", help="recompute even if the output JSON exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{run_dir.name}__{args.eval_split}"
    if args.truncate_topk:
        tag += f"__trunc{args.truncate_topk}"
    out_path = out_dir / f"{tag}.json"
    # Resumability: SLURM jobs can hit the wall clock, so a completed cell is
    # never recomputed unless asked for.  The guard checks which representations
    # the cached file actually holds, not merely that it exists -- a screening
    # pass may have written `z` only, and a later pass asking for xM/xhat must
    # not silently inherit that partial result.
    if out_path.exists() and not args.overwrite:
        try:
            cached = set(json.loads(out_path.read_text()).get("reps", {}))
        except Exception:
            cached = set()
        wanted = set(args.reps)
        if wanted.issubset(cached):
            print(f"[skip] {out_path} already has reps {sorted(wanted)}")
            return
        print(f"[recompute] {out_path} has {sorted(cached)}, need {sorted(wanted - cached)} as well")

    model, meta = load_run(run_dir, device, checkpoint=args.checkpoint)
    is_dual = bool(meta.get("is_dual"))
    # Prefer what the run recorded; fall back to the LMDB marker, then the legacy substring.
    multilabel = resolve_multilabel(meta.get("train_path"), meta.get("multilabel"))

    eval_path = meta.get(f"{args.eval_split}_path")
    if not eval_path:
        raise ValueError(f"run has no {args.eval_split}_path in training_config.json")

    print(f"run={run_dir.name}\n  model={meta['model_type']} mode={meta.get('sparsity_mode')} "
          f"seed={meta['seed']} dict={meta['dict_size']} multilabel={multilabel}")

    t0 = time.time()
    train_loader = make_loader(meta["train_path"], args.encode_batch_size, args.num_workers, multilabel)
    eval_loader = make_loader(eval_path, args.encode_batch_size, args.num_workers, multilabel)

    train_banks = build_banks(model, train_loader, device, args.reps, is_dual, "train",
                              truncate_topk=args.truncate_topk, max_samples=args.max_train_samples)
    eval_banks = build_banks(model, eval_loader, device, args.reps, is_dual, args.eval_split,
                             truncate_topk=args.truncate_topk, max_samples=args.max_eval_samples)
    encode_s = time.time() - t0

    results: Dict[str, Any] = {
        "meta": meta,
        "eval_split": args.eval_split,
        "multilabel": multilabel,
        "knn_k": args.knn_k,
        "encode_batch_size": args.encode_batch_size,
        "truncate_topk": args.truncate_topk,
        "n_train": train_banks[args.reps[0]].n_rows,
        "n_eval": eval_banks[args.reps[0]].n_rows,
        "encode_seconds": encode_s,
        "lp": {"epochs": LP_EPOCHS, "batch_size": LP_BATCH_SIZE, "lr": LP_LR,
               "weight_decay": LP_WEIGHT_DECAY, "seed": LP_SEED},
        "reps": {},
    }

    for rep in args.reps:
        # Move this representation's banks onto the GPU once and reuse them for
        # both KNN and LP.  Done per rep (not up front) so only one bank pair is
        # resident at a time -- the Dual-JumpReLU sparse bank alone is ~5.4 GB.
        tb = train_banks[rep].to(device)
        eb = eval_banks[rep].to(device)
        print(f"\n== rep={rep}  dim={tb.n_cols}  train_L0={tb.l0_mean:.1f}  eval_L0={eb.l0_mean:.1f}")
        t1 = time.time()
        knn = knn_metric(tb, eb, device, k=args.knn_k,
                         query_batch=args.knn_query_batch, bank_chunk=args.knn_bank_chunk)
        t2 = time.time()
        lp = lp_metric(tb, eb, device)
        t3 = time.time()
        results["reps"][rep] = {
            "dim": tb.n_cols,
            "train_l0_mean": tb.l0_mean,
            "eval_l0_mean": eb.l0_mean,
            "knn": knn,
            "lp": lp,
            "knn_seconds": t2 - t1,
            "lp_seconds": t3 - t2,
        }
        print(f"   KNN={knn['metric']:.4f}  LP={lp['metric']:.4f}  "
              f"({t2 - t1:.0f}s / {t3 - t2:.0f}s)")
        # Drop the GPU copies before the next representation is staged.
        del tb, eb
        if device.type == "cuda":
            torch.cuda.empty_cache()

    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.replace(out_path)  # atomic: a killed job never leaves a half-written result
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
