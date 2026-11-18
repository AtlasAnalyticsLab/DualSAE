#!/usr/bin/env python3
"""Evaluator: every reported metric for ONE trained residual-stream SAE cell.

    python -m langdual.lmsae.eval_trackb --tag B_L12_owt_topk_dual_seed42
    python -m langdual.lmsae.eval_trackb --tag abl_warmupB1250_B_L12_owt_topk_dual_seed42

Writes ``results/trackb/<tag>.json`` under the cell's root (the main grid's or the
ablation's).  The cell is a DualSAE or a single-branch baseline trained on pre-extracted fp16
residual shards of ``config.LM_REPO``; every number is computed identically for both arms of a
pair, and nothing is averaged across anything but seeds downstream.

WHAT IS REPORTED, AND WHY EACH NAME IS QUALIFIED
------------------------------------------------
``recon``
    BOTH R^2 estimators, because two are in circulation and they are not interchangeable:
      * ``r2_global_a4``      the POOLED ratio
                              ``1 - sum_i||x_i - xhat_i||^2 / sum_i||x_i - xbar||^2``
                              (``r2_global`` in the vision core's ``branch_metrics.py``).
      * ``r2_perfeat_table3`` the MEAN OF PER-DIMENSION R^2 over the dims with
                              ``ss_tot > 1e-12`` (``compute_r2`` in the vision trainers).
    On residual-stream data (whose per-dimension variances span orders of magnitude) the two
    can separate materially.  ``frac_variance_explained`` is ``dictionary_learning``'s pooled
    estimator (the residual CENTERED per dimension), plus cosine similarity, norm ratio,
    relative reconstruction bias and normalised MSE.

``monoscore``
    MonoScore of the sparse code ``z`` with the L2-normalised RAW residual activation as the
    probe, in three reductions under unambiguous names:
      * ``mean_active_paper_a4``  mean over latents that fired on the scoring set (|A_j| > 0);
      * ``mean_all_table3``       mean over ALL dict_size latents, dead ones contributing 0;
      * ``mean_active_code_ref``  the reference implementation's validity mask ``den > 1e-8``;
    plus ``null_cosine`` (the probe space's mean pairwise cosine, i.e. the score of a latent
    that carries no information) and ``n_active_*``.  A QUANTILE-ROBUST variant (min/max
    replaced by the 0.1st/99.9th per-latent percentiles) is reported ALONGSIDE, never instead.

``sparsity``   L0 (mean nnz per token), dead-latent fraction, active-latent count.
``ce``         the LM-native metrics from ``loss_recovered_hooks``: CE of the clean model, of
               the model with the hooked residual replaced by the SAE reconstruction, and with
               it zero-ablated (delta CE = spliced - clean), ``frac_recovered``, and the mean
               next-token KL(clean || spliced).

WHY THE SCORING SET IS CAPPED
-----------------------------
MonoScore needs the sparse code DENSELY (``branch_metrics.monoscore_streaming`` takes a
materialised ``(N, M)`` ``z``), which is ``8192 * 4 = 32 KB`` per row.  The cap is
``config.TRACKB_SCORE_TOKENS``, recorded in the JSON, and identical for every cell.  It is
rounded DOWN to a whole number of encode batches because BatchTopK selects the top
``k * batch`` over the whole batch, so a short final batch would be scored at a different
sparsity than every other batch.

RESUMABILITY
------------
``result_is_complete``: the schema version, the protocol constants (score-token cap, encode
batch size) and every requested block must match, or the cell is recomputed.  Writes are
atomic (tmp + rename).  ``--overwrite`` forces a recompute.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:  # package import -- the normal path
    from .. import ablate, config, jobs
    from . import loss_recovered_hooks as H
except ImportError:  # direct execution: python langdual/lmsae/eval_trackb.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from langdual import ablate, config, jobs  # type: ignore[no-redef]
    from langdual.lmsae import loss_recovered_hooks as H  # type: ignore[no-redef]

# The vision core is imported by BARE MODULE NAME (`from models import ...`), so its directory
# has to be on sys.path.  Prepending is deliberate: `metrics`/`utils` are common names.
if config.SAE_DIR not in sys.path:
    sys.path.insert(0, config.SAE_DIR)

# ---- reused, never re-implemented ------------------------------------------------------
# The model loader / forward pass and the R^2 / MonoScore accumulators come from the vision
# core, so a language row and a vision row mean the same thing.
from knn_lp_sparse import forward_reps, load_run  # noqa: E402
from branch_metrics import (  # noqa: E402
    MONO_DEN_FLOOR,
    R2_EPS,
    R2Accumulator,
    monoscore_streaming,
    summarize_monoscore,
)


# --------------------------------------------------------------------------- #
# Protocol constants
# --------------------------------------------------------------------------- #

# Bumping this invalidates every cached result, which is intended.
SCHEMA_VERSION = 1

BLOCKS: Tuple[str, ...] = ("recon", "monoscore", "sparsity", "ce")
# recon / monoscore / sparsity all come out of the single pass over the activation shards.
SHARD_BLOCKS = frozenset(("recon", "monoscore", "sparsity"))
# The quantile-robust MonoScore variant's bounds.  0.1 / 99.9 percentiles: aggressive enough
# to drop the one-in-a-thousand outlier tokens that dominate a min-max rescaling, mild enough
# that the ordinary body of the distribution is untouched.
MONO_Q_LO = 0.001
MONO_Q_HI = 0.999
# --------------------------------------------------------------------------- #
# Cell resolution
# --------------------------------------------------------------------------- #


def _resolve_lm_path() -> str:
    """Local directory holding the language model, or the repo id as a last resort.

    Single source of truth is langdual.assets, which is what the download stage actually wrote;
    extract_resid.py resolves the same way.
    """
    import os
    try:
        from langdual import assets as _assets
        cand = _assets.local_snapshot(config.LM_REPO, "model")
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "config.json")):
            return cand
    except Exception:  # pragma: no cover - partial checkout
        pass
    return config.LM_REPO


def cell_layer(cell: config.Cell) -> int:
    """``Cell.key`` is ``"L<layer>"`` for Track B."""
    if cell.track != "B" or not cell.key.startswith("L"):
        raise ValueError(f"{cell.tag} is not a Track B cell")
    return int(cell.key[1:])


# --------------------------------------------------------------------------- #
# Activation shards
# --------------------------------------------------------------------------- #

def _shard_files(manifest: Dict[str, Any], layer: int, split: str,
                 corpus: str = "owt") -> List[Tuple[str, Optional[int]]]:
    """``[(absolute path, n_rows or None), ...]`` in row order, from the manifest.

    ``file`` may be absolute or relative to the shard directory.  The row count is read from
    ``n_rows`` or ``n_tokens`` (``lmsae/extract_resid.py`` writes both) and is optional: it is
    derived from the file size when absent, and cross-checked against the file when present.

    The manifest -- not a directory glob -- is the row ORDER.  It agrees with the glob order
    ``lmsae/resid_shards.ResidShardDataset`` uses (``shard_%05d.npy`` sorts lexicographically
    into index order), and the guard below asserts that agreement instead of assuming it: a
    shard file that exists on disk but is absent from the manifest would mean the scorer and
    the trainer saw different data, which no downstream number would reveal.
    """
    entries = manifest.get("shards")
    if not entries:
        raise ValueError(f"shard manifest for L{layer}/{split} lists no shards")
    declared = manifest.get("n_shards")
    if declared is not None and int(declared) != len(entries):
        raise ValueError(
            f"shard manifest for L{layer}/{split} says n_shards={declared} but lists "
            f"{len(entries)} entries")
    base = config.resid_dir(layer, split, corpus)
    out: List[Tuple[str, Optional[int]]] = []
    for entry in entries:
        if isinstance(entry, str):
            path, rows = entry, None
        elif isinstance(entry, dict):
            path = entry.get("file") or entry.get("path")
            rows = entry.get("n_rows") or entry.get("n_tokens") or entry.get("rows")
            if path is None:
                raise ValueError(f"shard entry {entry!r} has no 'file'")
        else:
            raise ValueError(f"unsupported shard entry type {type(entry).__name__}")
        out.append((path if os.path.isabs(path) else os.path.join(base, path),
                    int(rows) if rows is not None else None))

    # Glob cross-check, applied ONLY when the manifest uses the extraction stage's own
    # ``shard_*.npy`` naming -- the pattern ResidShardDataset globs for.  A manifest that
    # names its shards some other way (a synthetic test fixture) is left alone rather than
    # failed on a pattern it never claimed to follow.
    names = [os.path.basename(p) for p, _ in out]
    if names and all(n.startswith("shard_") and n.endswith(".npy") for n in names):
        on_disk = sorted(os.path.basename(f) for f in glob.glob(os.path.join(base, "shard_*.npy")))
        if on_disk and on_disk != sorted(names):
            raise ValueError(
                f"L{layer}/{split}: the manifest lists {len(names)} shards but the directory "
                f"holds {len(on_disk)} matching files; the scorer would read different rows "
                f"than the trainer did (extra on disk: "
                f"{sorted(set(on_disk) - set(names))[:3]}, missing: "
                f"{sorted(set(names) - set(on_disk))[:3]})")
    return out


def load_shard_rows(layer: int, split: str, n_rows: int,
                    manifest: Optional[Dict[str, Any]] = None,
                    corpus: str = "owt") -> Tuple[torch.Tensor, Dict[str, Any]]:
    """The first ``n_rows`` activation rows of one (layer, split), as a CPU fp16 tensor.

    A CONTIGUOUS PREFIX, deliberately: every cell must be scored on the same tokens or the
    dual-vs-baseline delta picks up sampling noise, and a prefix is the only selection that is
    identical between a fresh run and a resume without storing an index.  The shards are
    written in stream order (which is document order), so the prefix is a coherent sample of
    the corpus rather than a biased slice of one topic.

    Kept in fp16 -- the dtype they were written in, so this is lossless -- and cast per batch
    at encode time; 200k x 1024 fp16 is 400 MB, and doubling that for no reason would be the
    largest avoidable allocation in the job.
    """
    manifest = manifest or H.read_shard_manifest(layer, split, corpus)
    d_model = int(manifest.get("d_model", config.LM_D_MODEL))
    dtype = np.dtype(str(manifest.get("dtype", "float16")))
    itemsize = dtype.itemsize

    chunks: List[torch.Tensor] = []
    got = 0
    for path, declared_rows in _shard_files(manifest, layer, split, corpus):
        if got >= n_rows:
            break
        if not os.path.exists(path):
            raise FileNotFoundError(f"shard listed in the manifest is missing: {path}")
        if path.endswith(".npy"):
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2 or arr.shape[1] != d_model:
                raise ValueError(f"{path}: expected (rows, {d_model}), got {arr.shape}")
        else:
            # Raw memmap: the row count comes from the file size, which also validates any
            # declared count.  A partial final row means a truncated (killed) write.
            nbytes = os.path.getsize(path)
            if nbytes % (d_model * itemsize):
                raise ValueError(
                    f"{path}: size {nbytes} is not a whole number of {d_model}-wide "
                    f"{dtype} rows -- the shard write was interrupted")
            rows = nbytes // (d_model * itemsize)
            arr = np.memmap(path, dtype=dtype, mode="r", shape=(rows, d_model))
        if declared_rows is not None and int(arr.shape[0]) != declared_rows:
            raise ValueError(f"{path}: manifest says {declared_rows} rows, file holds {arr.shape[0]}")
        take = min(n_rows - got, int(arr.shape[0]))
        # np.array (not np.asarray) forces a real copy of exactly the rows we keep, so the
        # returned tensor owns its memory: a torch tensor wrapping a read-only memmap view is
        # both a warning and a live file mapping we would rather not hold open.
        tensor = torch.from_numpy(np.array(arr[:take]))
        # fp16 is the contracted shard dtype and keeping it halves the resident bytes.  A
        # manifest that declares anything else is widened to fp32 instead of being squeezed
        # into fp16, which would be a silent precision loss.
        chunks.append(tensor if tensor.dtype == torch.float16 else tensor.float())
        got += take
        del arr, tensor

    if got < n_rows:
        raise ValueError(
            f"L{layer}/{split} shards hold only {got} rows but {n_rows} were requested; "
            "lower --score-tokens or extract more activations")
    x = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
    info = {"n_rows": int(x.shape[0]), "d_model": int(x.shape[1]),
            "dtype": str(dtype), "n_shards_touched": len(chunks)}
    return x, info


# --------------------------------------------------------------------------- #
# One encode pass over a matrix of activations
# --------------------------------------------------------------------------- #

@dataclass
class Encoded:
    """Everything one pass over an activation matrix produces."""

    z: Optional[torch.Tensor] = None              # (N, M) dense code, CPU float32
    xhat: Optional[torch.Tensor] = None           # (N, D) L2-normalized reconstruction, CPU
    probe: Optional[torch.Tensor] = None          # (N, D) L2-normalized INPUT (MonoScore probe)
    n_rows: int = 0
    l0_mean: float = 0.0
    active_counts: Optional[torch.Tensor] = None  # (M,) nnz per latent
    recon: Dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0


class _ReconAccumulator:
    """Both paper R^2 estimators plus dictionary_learning's pooled frac_variance_explained.

    ``R2Accumulator`` (reused from the vision core's ``branch_metrics.py``) already streams the float64
    per-dimension ``sum_y`` / ``sum_y2`` / ``ss_res`` that ``r2_perfeat`` and ``r2_global``
    need.  The only thing it does not carry is the residual MEAN, which
    ``frac_variance_explained`` needs because ``torch.var`` centers:

        frac_variance_explained = 1 - sum_j Var(x_j - xhat_j) / sum_j Var(x_j)
                                = 1 - sum_j (ss_res_j - (sum_res_j)^2 / n) / sum_j ss_tot_j

    (the ``1/(n-1)`` of the unbiased variance cancels between numerator and denominator).
    So this class composes the reused accumulator and adds ONE extra ``(D,)`` vector, rather
    than duplicating the R^2 arithmetic.

    Deviation from ``dictionary_learning.evaluate``: it computes frac_variance_explained per
    batch and averages the ratios; this is the POOLED value over the whole scoring set, which
    is what the metric's definition asks for and is what the task specifies.
    """

    def __init__(self, device: torch.device) -> None:
        self._r2 = R2Accumulator(["xhat"], device)
        self._sum_res: Optional[torch.Tensor] = None
        self._device = device
        # Streaming accumulators for the four dictionary_learning fidelity diagnostics that
        # R2/FVU alone cannot express. All are sums over rows, so they are exact regardless of
        # batch size -- a per-batch mean-of-means would not be.
        self._n = 0
        self._sum_cos = 0.0          # sum of cos(x, x_hat)
        self._sum_l2ratio = 0.0      # sum of ||x_hat|| / ||x||
        self._sum_xhat_sq = 0.0      # sum of ||x_hat||^2      \ relative reconstruction bias
        self._sum_x_dot_xhat = 0.0   # sum of <x, x_hat>       / = E||x_hat||^2 / E<x,x_hat>
        self._sum_sq_err = 0.0       # sum of ||x - x_hat||^2  \ normalized MSE
        self._sum_sq_x = 0.0         # sum of ||x||^2          /

    def update(self, x: torch.Tensor, x_hat: torch.Tensor) -> None:
        self._r2.update(x, {"xhat": x_hat})
        res = (x.to(torch.float64) - x_hat.to(torch.float64)).sum(dim=0)
        self._sum_res = res if self._sum_res is None else self._sum_res + res

        # float64 throughout: these are sums over up to 200k rows and the squared terms lose
        # significance fast in float32.
        xd, hd = x.to(torch.float64), x_hat.to(torch.float64)
        nx = xd.norm(dim=-1)
        nh = hd.norm(dim=-1)
        dot = (xd * hd).sum(dim=-1)
        eps = torch.finfo(torch.float64).tiny
        self._n += int(xd.shape[0])
        self._sum_cos += float((dot / (nx * nh).clamp_min(eps)).sum().item())
        self._sum_l2ratio += float((nh / nx.clamp_min(eps)).sum().item())
        self._sum_xhat_sq += float((nh ** 2).sum().item())
        self._sum_x_dot_xhat += float(dot.sum().item())
        self._sum_sq_err += float(((xd - hd) ** 2).sum().item())
        self._sum_sq_x += float((xd ** 2).sum().item())

    def result(self) -> Dict[str, Any]:
        base = self._r2.result()["xhat"]
        out: Dict[str, Any] = {
            # Each key names its estimator (pooled vs mean-of-per-dimension), so the two can
            # never be confused downstream.
            "r2_global_a4": base["r2_global"],
            "r2_perfeat_table3": base["r2_perfeat"],
            "n_samples": base["n_samples"],
            "n_dims": base["n_dims"],
            "n_valid_dims": base["n_valid_dims"],
        }
        # --- the four diagnostics R2 cannot express -------------------------------------
        if self._n:
            out["cossim"] = self._sum_cos / self._n
            out["l2_ratio"] = self._sum_l2ratio / self._n
            # Equation 10 of the gated-SAE paper: E||x_hat||^2 / E<x, x_hat>. A value below 1
            # means the reconstruction is systematically SHRUNK toward the origin -- a known
            # failure mode that a high explained-variance score hides completely, because
            # shrinkage costs little variance but changes every downstream inner product.
            denom = self._sum_x_dot_xhat
            out["relative_reconstruction_bias"] = (
                (self._sum_xhat_sq / denom) if abs(denom) > 0 else float("nan"))
            # MSE relative to the energy of the input (NOT to the mean-predictor baseline,
            # which is what FVU already reports). Kept distinct so the two are not confused.
            out["normalized_mse"] = (
                (self._sum_sq_err / self._sum_sq_x) if self._sum_sq_x > 0 else float("nan"))
        sum_y, sum_y2, n = self._r2.sum_y, self._r2.sum_y2, self._r2.n
        if sum_y is None or n < 2 or self._sum_res is None:
            out["frac_variance_explained"] = float("nan")
            return out
        ss_tot = sum_y2 - (sum_y * sum_y) / n
        ss_res = self._r2.ss_res["xhat"] - (self._sum_res * self._sum_res) / n
        tot = float(ss_tot.sum().item())
        out["frac_variance_explained"] = (1.0 - float(ss_res.sum().item()) / tot) if tot > R2_EPS else float("nan")
        return out


@torch.no_grad()
def encode_rows(
    model: torch.nn.Module,
    rows: torch.Tensor,
    device: torch.device,
    is_dual: bool,
    batch_size: int,
    want_z: bool = True,
    want_xhat: bool = True,
    want_probe: bool = False,
    want_recon: bool = False,
    desc: str = "encode",
) -> Encoded:
    """Encode an ``(N, D)`` activation matrix that is ALREADY in shard space.

    ``batch_size`` is a protocol constant, not a throughput knob: BatchTopK (and DualSAE in
    ``sparsity_mode='batch_topk'``) takes the top ``k * batch`` activations over the WHOLE
    batch, so z -- and with it MonoScore, purity and L0 -- depends on it.  Callers pass
    ``config.BATCH_SIZE`` (the training batch) and ``N`` is trimmed to a whole multiple of it
    by the caller, so no batch is short.

    ``forward_reps`` is reused from the vision core's ``knn_lp_sparse.py`` so the dual/baseline handling
    (including the fact that a single-branch model's ``x^M`` IS its ``x_hat``) is the vision
    path's, not a second implementation of it.
    """
    model.eval()
    n = int(rows.shape[0])
    z_chunks: List[torch.Tensor] = []
    xhat_chunks: List[torch.Tensor] = []
    probe_chunks: List[torch.Tensor] = []
    acc = _ReconAccumulator(device) if want_recon else None
    active_counts: Optional[torch.Tensor] = None
    nnz_total = 0
    t0 = time.time()

    for s in range(0, n, batch_size):
        x = rows[s:s + batch_size].to(device, non_blocking=True).float()
        out = forward_reps(model, x, is_dual)
        z = out["z"].float()
        x_hat = out["xhat"].float()

        mask = z > 0
        nnz_total += int(mask.sum().item())
        counts = mask.sum(dim=0)
        active_counts = counts if active_counts is None else active_counts + counts

        if acc is not None:
            acc.update(x, x_hat)
        if want_z:
            z_chunks.append(z.cpu())
        if want_xhat:
            # Same L2 normalization the trainers' bank builder applies for KNN and LP.
            xhat_chunks.append(F.normalize(x_hat, p=2, dim=1).cpu())
        if want_probe:
            # MonoScore's probe is ``F.normalize(x)`` -- the RAW INPUT, exactly as
            # ``loss.compute_monosemanticity_fast`` uses it in the vision trainers (NOT the
            # reconstruction).  Note it is invariant to the shard norm factor: a global
            # positive scale cancels under row normalization, so the probe space (and hence
            # null_cosine) is the same whether or not the factor was folded in.
            probe_chunks.append(F.normalize(x, p=2, dim=1).cpu())
        del out, z, x_hat, mask, x

    if active_counts is None:
        raise ValueError(f"{desc}: nothing to encode (0 rows)")

    return Encoded(
        z=torch.cat(z_chunks, dim=0) if z_chunks else None,
        xhat=torch.cat(xhat_chunks, dim=0) if xhat_chunks else None,
        probe=torch.cat(probe_chunks, dim=0) if probe_chunks else None,
        n_rows=n,
        l0_mean=nnz_total / max(n, 1),
        active_counts=active_counts.cpu(),
        recon=acc.result() if acc is not None else {},
        seconds=time.time() - t0,
    )


# --------------------------------------------------------------------------- #
# MonoScore, exact and quantile-robust
# --------------------------------------------------------------------------- #

def per_latent_quantiles(z: torch.Tensor, q_lo: float, q_hi: float,
                         device: torch.device, col_chunk: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-latent ``(q_lo, q_hi)`` order statistics of a dense ``(N, M)`` code.

    NEAREST-RANK (type-1) quantiles via ``torch.kthvalue``, not ``torch.quantile``: the
    latter interpolates AND refuses inputs above ~16M elements, and a (200k, 8192) code is
    1.6 billion.  At N=200k the two differ by at most one order statistic, which is far below
    anything MonoScore can resolve.

    Returning ATTAINED values matters for how the variant is applied (see
    ``monoscore_blocks``): clamping a column to [lo, hi] leaves its min/max exactly lo/hi.
    """
    n, m = int(z.shape[0]), int(z.shape[1])
    if n < 1:
        raise ValueError("per_latent_quantiles: empty code")
    k_lo = min(max(1, int(math.ceil(q_lo * n))), n)
    k_hi = min(max(1, int(math.ceil(q_hi * n))), n)
    lo = torch.empty(m, dtype=torch.float32)
    hi = torch.empty(m, dtype=torch.float32)
    for c0 in range(0, m, col_chunk):
        c1 = min(c0 + col_chunk, m)
        chunk = z[:, c0:c1].to(device, non_blocking=True).float()
        lo[c0:c1] = torch.kthvalue(chunk, k_lo, dim=0).values.cpu()
        hi[c0:c1] = torch.kthvalue(chunk, k_hi, dim=0).values.cpu()
        del chunk
    return lo, hi


def monoscore_blocks(z: torch.Tensor, probe: torch.Tensor, active_counts: torch.Tensor,
                     device: torch.device, chunk_size: int,
                     q_lo: float = MONO_Q_LO, q_hi: float = MONO_Q_HI) -> Dict[str, Any]:
    """Exact min-max MonoScore plus the quantile-robust variant, both fully labelled.

    *** ORDER MATTERS: the exact value is computed FIRST, because the robust variant clamps
    ``z`` IN PLACE. ***  The clamp is what makes the variant free of a second implementation:
    ``monoscore_streaming``'s pass 1 takes each latent's global min/max, and after clamping a
    column to its own [q_lo, q_hi] those ARE the quantiles (both are attained values), so
    running the same reused function on the clamped code computes exactly
    ``a = (z - q_lo) / (q_hi - q_lo + eps)`` with values outside the range saturating at 0/1.
    Saturation rather than exclusion is deliberate: dropping the outlier rows would change the
    row SET per latent, and MonoScore's closed form assumes one shared row set across latents.

    In-place clamping costs no extra memory, which is the whole reason the variant is
    affordable at all: a second (200k, 8192) float32 code is another 6.4 GB.

    A latent whose ``q_lo == q_hi`` (dead, or firing on fewer than 0.1% of rows) becomes
    constant after clamping, so ``den = 0`` and its score is 0 -- the same convention the
    reference uses for dead latents, and the reason the robust variant's ``n_active_code``
    can be LOWER than the exact one.  Both counts are reported.
    """
    out: Dict[str, Any] = {
        "probe": "l2-normalized raw residual activation (as "
                 "loss.compute_monosemanticity_fast: F.normalize(x), not the reconstruction)",
        "chunk_size": int(chunk_size),
    }

    # ---- exact, min-max: the parity quantity -------------------------------
    mono = monoscore_streaming(z, {"x": probe}, device, chunk_size=chunk_size, progress=False)
    out["minmax"] = _label_monoscore(
        summarize_monoscore(mono["ms"]["x"], mono["den"], active_counts,
                            mono["probe_stats"].get("x")))
    out["minmax"]["variant"] = "exact min-max rescaling (the MonoScore definition)"

    # ---- quantile-robust ---------------------------------------------------
    lo, hi = per_latent_quantiles(z, q_lo, q_hi, device)
    n_degenerate = int((hi <= lo).sum().item())
    for c0 in range(0, int(z.shape[1]), 4096):
        c1 = min(c0 + 4096, int(z.shape[1]))
        # Views of z: clamp_ writes through to the underlying storage.  Bounds broadcast
        # over rows, so this is one clamp per column block, not per latent.
        z[:, c0:c1].clamp_(min=lo[c0:c1].unsqueeze(0), max=hi[c0:c1].unsqueeze(0))
    mono_q = monoscore_streaming(z, {"x": probe}, device, chunk_size=chunk_size, progress=False)
    out["quantile_robust"] = _label_monoscore(
        summarize_monoscore(mono_q["ms"]["x"], mono_q["den"], active_counts,
                            mono_q["probe_stats"].get("x")))
    out["quantile_robust"].update(
        variant=f"min/max replaced by the {q_lo:.3%} / {q_hi:.3%} per-latent quantiles "
                "(clamp-then-rescale, values outside the range saturate at 0/1)",
        q_lo=q_lo, q_hi=q_hi,
        n_latents_collapsed_by_clamp=n_degenerate,
        note=("reported ALONGSIDE the exact min-max value and never instead of it: LM token "
              "activations are much heavier tailed than unit-norm vision embeddings, so a "
              "single outlier token can compress a latent's whole a-vector toward 0"),
    )
    out["z_clamped_in_place"] = True
    return out


def _label_monoscore(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Rename ``summarize_monoscore``'s three reductions to unambiguous names.

    The repo's short names (``mean_all`` / ``mean_active_code`` / ``mean_active_paper``) are
    only unambiguous to someone who has read ``branch_metrics``'s docstring; a Track B JSON is
    read by the report generator and by a human, so the protocol each number belongs to is
    spelled out in the KEY.  The originals are kept alongside so a diff against a vision
    result is still a key-for-key diff.
    """
    out = dict(summary)
    out["mean_active_paper_a4"] = summary["mean_active_paper"]   # latents with |A_j| > 0
    out["mean_all_table3"] = summary["mean_all"]                 # mean over ALL latents (the vision trainers' reduction)
    out["mean_active_code_ref"] = summary["mean_active_code"]    # reference mask den > 1e-8
    out["n_active_latents"] = summary["n_active_paper"]
    out["mono_den_floor"] = MONO_DEN_FLOOR
    return out



# --------------------------------------------------------------------------- #
# Resumability
# --------------------------------------------------------------------------- #

def result_is_complete(payload: Dict[str, Any], blocks: Sequence[str],
                       score_tokens: int, encode_batch_size: int) -> bool:
    """Does this cached payload already answer the requested question?

    The PROTOCOL CONSTANTS are part of the question: a result computed at a different
    score-token cap or a different encode batch size is a well-formed, ``status: "ok"`` file
    that answers a different question (BatchTopK's top-k is batch-level, and MonoScore's
    min/max rescaling moves with the number of rows), so it does not count as complete.
    """
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != "ok":
        return False
    protocol = payload.get("protocol") or {}
    if protocol.get("score_tokens_requested") != score_tokens:
        return False
    if protocol.get("encode_batch_size") != encode_batch_size:
        return False
    for block in blocks:
        value = payload.get(block)
        if not isinstance(value, dict) or not value:
            return False
    return True


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True,
                   help="cell tag: a main-grid tag (B_L12_owt_topk_dual_seed42) or an ablation "
                        "tag (abl_warmupB1250_B_L12_owt_topk_dual_seed42)")
    p.add_argument("--checkpoint", default="best", choices=("best", "final"),
                   help='"best" = best_val_monoscore_model.pt (selected by validation '
                        'MonoScore; the reported evaluations use it), "final" = '
                        'autoencoder_final.pt (end of training)')
    p.add_argument("--run-dir", default=None,
                   help="override the checkpoint directory (default: the registry's run dir)")
    p.add_argument("--out-dir", default=None,
                   help="override the output directory; the file is <run-dir basename>.json")
    p.add_argument("--out-path", default=None, help="exact result JSON to write")
    p.add_argument("--blocks", nargs="+", default=list(BLOCKS), choices=list(BLOCKS),
                   help="which metric families to compute (default: all)")
    p.add_argument("--eval-split", default=config.EVAL_SPLIT,
                   help="activation-shard / token-stream split the metrics use")
    p.add_argument("--score-tokens", type=int, default=config.TRACKB_SCORE_TOKENS,
                   help="cap on the densely-cached scoring set (config.TRACKB_SCORE_TOKENS)")
    p.add_argument("--encode-batch-size", type=int, default=config.BATCH_SIZE,
                   help="MUST match training: BatchTopK's top-k is batch-level")
    p.add_argument("--ce-tokens", type=int, default=262_144,
                   help="token budget for the three CE forward passes (the 200K-token test "
                        "stream is shorter, so by default all of it is used)")
    p.add_argument("--lm-dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                   help="float32 keeps the CE bit-comparable with the reference")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute even if the output JSON is already complete")
    return p.parse_args(argv)


def _resolve_cell(args: argparse.Namespace):
    """(cell, reference main-grid cell, default run dir, default output path) for --tag.

    An ablation cell inherits its identity fields (corpus, layer, activation, variant, seed)
    from the main-grid cell at the same coordinates, but its run directory and result path live
    in the ablation tree, so an ablation evaluation can never overwrite a main-grid result.
    """
    try:
        cell = jobs.cell_from_tag(args.tag)
    except KeyError as exc:
        raise SystemExit(str(exc))
    if isinstance(cell, ablate.AblCell):
        return cell, jobs.reference_cell(cell), Path(cell.run_dir), Path(ablate.trackb_result(cell))
    return cell, cell, Path(cell.run_dir), Path(config.trackb_result(cell))


def _load_lm(device: torch.device, dtype: str) -> Tuple[torch.nn.Module, Any]:
    """Load the frozen LM and its tokenizer, strictly offline.

    Runtime never downloads (``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` are set by
    scripts/env.sh); the weights are the snapshot prepare_assets.sh wrote.  The import is local
    so a ``--blocks recon monoscore sparsity`` run never pays for transformers at all.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # local: see docstring

    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype]
    # Resolve the LOCAL snapshot, never the bare repo id: prepare_assets.sh downloads with
    # local_dir=, so the weights live under config.RAW_DIR and are NOT in HF_HOME's hub cache.
    lm_path = _resolve_lm_path()
    tokenizer = AutoTokenizer.from_pretrained(lm_path, local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(lm_path, dtype=torch_dtype, local_files_only=True)
    return lm.to(device).eval(), tokenizer


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    cell, ref, default_run_dir, default_out = _resolve_cell(args)
    layer = ref.layer
    blocks = [b for b in BLOCKS if b in set(args.blocks)]   # canonical order

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir
    if args.out_path and args.out_dir:
        raise SystemExit("pass --out-path or --out-dir, not both")
    if args.out_path:
        out_path = Path(args.out_path)
    elif args.out_dir:
        out_path = Path(args.out_dir) / f"{run_dir.name}.json"
    else:
        out_path = default_out
    if not run_dir.is_dir():
        raise SystemExit(f"run directory does not exist: {run_dir}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- resumability: a kill must cost only the unfinished cells.
    if out_path.exists() and not args.overwrite:
        try:
            cached = json.loads(out_path.read_text())
        except Exception:
            cached = {}
        if result_is_complete(cached, blocks, args.score_tokens, args.encode_batch_size):
            print(f"[skip] {out_path} already holds blocks {blocks}")
            return
        print(f"[recompute] {out_path} is missing part of blocks={blocks} "
              "(or was computed with different protocol constants)")

    t_start = time.time()
    sae, meta = load_run(run_dir, device, checkpoint=args.checkpoint)
    is_dual = bool(meta.get("is_dual"))
    dict_size = int(meta["dict_size"])
    if bool(meta.get("input_unit_norm", False)) != config.TRACKB_NORMALIZE:
        # Not fatal (the checkpoint is what it is) but it means the cell was trained under a
        # different preprocessing than the registry declares, which changes every number.
        print(f"[WARN] {args.tag}: checkpoint input_unit_norm="
              f"{meta.get('input_unit_norm')} but config.TRACKB_NORMALIZE="
              f"{config.TRACKB_NORMALIZE}")

    manifest = H.read_shard_manifest(layer, args.eval_split, ref.corpus)
    norm_factor, convention = H.norm_factor_from_manifest(manifest)
    print(f"cell={args.tag}\n  model={meta['model_type']} dual={is_dual} dict={dict_size} "
          f"layer={layer} norm_factor={norm_factor:g} ({convention}) blocks={blocks}")

    results: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "cell": {
            "tag": args.tag, "ref_tag": ref.tag, "track": ref.track, "layer": layer,
            "corpus": ref.corpus, "activation": ref.activation, "variant": ref.variant,
            "seed": ref.seed, "model_type": ref.model_type, "lr": ref.lr,
            "normalize": ref.normalize,
            "axis": getattr(cell, "axis", None), "axis_value": getattr(cell, "value", None),
        },
        "meta": meta,
        "blocks_computed": blocks,
        "protocol": {
            "lm": config.LM_REPO,
            "eval_split": args.eval_split,
            "score_tokens_requested": args.score_tokens,
            "encode_batch_size": args.encode_batch_size,
            "norm_factor": norm_factor,
            "norm_convention": convention,
            "ctx_len": config.LM_CTX_LEN,
            "score_cap_note": (
                "the sparse code is cached DENSELY (8192 latents x 4 B = 32 KB per row), so an "
                "uncapped pass would need terabytes; the cap is identical for every cell"),
        },
        "timings": {},
    }

    # ---- one pass over the activation shards: recon + monoscore + sparsity --
    if SHARD_BLOCKS & set(blocks):
        # Trim to whole encode batches (BatchTopK is batch-level).
        n_rows = (args.score_tokens // args.encode_batch_size) * args.encode_batch_size
        if n_rows == 0:
            raise SystemExit(f"--score-tokens {args.score_tokens} is less than one encode batch")
        rows, shard_info = load_shard_rows(layer, args.eval_split, n_rows, manifest,
                                           ref.corpus)
        want_mono = "monoscore" in blocks
        enc = encode_rows(sae, rows, device, is_dual, args.encode_batch_size,
                          want_z=want_mono, want_xhat=False, want_probe=want_mono,
                          want_recon="recon" in blocks, desc=ref.corpus)
        del rows
        results["protocol"]["n_tokens_scored"] = enc.n_rows
        results["protocol"]["shards"] = shard_info
        results["timings"]["encode_seconds"] = enc.seconds
        print(f"  scored {enc.n_rows} tokens  L0={enc.l0_mean:.1f}  ({enc.seconds:.0f}s)")

        if "recon" in blocks:
            results["recon"] = dict(enc.recon, rep="xhat")
            print(f"  R2(pooled)={enc.recon['r2_global_a4']:.4f}  "
                  f"R2(per-feat mean)={enc.recon['r2_perfeat_table3']:.4f}  "
                  f"fracVarExpl={enc.recon['frac_variance_explained']:.4f}")

        if "sparsity" in blocks:
            n_active = int((enc.active_counts > 0).sum().item())
            results["sparsity"] = {
                "l0_mean": enc.l0_mean,
                "dict_size": dict_size,
                "n_active_latents": n_active,
                "dead_fraction": 1.0 - n_active / float(dict_size),
                "n_tokens": enc.n_rows,
                "note": ("L0 and the dead fraction are measured on the capped scoring set; "
                         "JumpReLU pairs are NOT sparsity-matched, so L0 must be shown next "
                         "to every JumpReLU row"),
            }
            print(f"  L0={enc.l0_mean:.1f}  dead={results['sparsity']['dead_fraction']:.3f}")

        if want_mono:
            t0 = time.time()
            results["monoscore"] = monoscore_blocks(
                enc.z, enc.probe, enc.active_counts, device, args.encode_batch_size)
            results["timings"]["monoscore_seconds"] = time.time() - t0
            mm, qr = results["monoscore"]["minmax"], results["monoscore"]["quantile_robust"]
            print(f"  MonoScore  mean(active)={mm['mean_active_paper_a4']:.4f}  "
                  f"mean(all)={mm['mean_all_table3']:.4f}  "
                  f"ref(den>1e-8)={mm['mean_active_code_ref']:.4f}  "
                  f"null={mm['null_cosine']:.4f}")
            print(f"  MonoScore(quantile-robust)  mean(active)={qr['mean_active_paper_a4']:.4f}  "
                  f"mean(all)={qr['mean_all_table3']:.4f}")
        # z was clamped in place by the robust variant and is the biggest allocation here.
        enc.z = None
        enc.probe = None

    # ---- the LM-native metrics ----------------------------------------------
    need_lm = "ce" in blocks
    if need_lm:
        t0 = time.time()
        lm, _tokenizer = _load_lm(device, args.lm_dtype)
        module = H.resolve_layer_module(lm, layer)
        results["timings"]["lm_load_seconds"] = time.time() - t0

        if "ce" in blocks:
            t0 = time.time()
            n_seq = args.ce_tokens // config.LM_CTX_LEN
            batches = H.iter_token_batches(
                # The CELL's corpus stream. Cross-entropy-recovered is meaningless if the
                # model is spliced back over a different corpus' tokens than it was trained on.
                config.lm_token_stream(ref.corpus, args.eval_split),
                batch_sequences=H.DEFAULT_BATCH_SEQUENCES,
                ctx_len=config.LM_CTX_LEN,
                max_sequences=n_seq,
                device=device,
            )
            results["ce"] = H.evaluate_loss_recovered(
                lm, batches, module, H.sae_callable(sae), norm_factor, dict_size,
                convention=convention, device=device, verbose=True)
            results["timings"]["ce_seconds"] = time.time() - t0
            ce = results["ce"]
            print(f"  CE orig={ce['loss_original']:.4f} recon={ce['loss_reconstructed']:.4f} "
                  f"zero={ce['loss_zero']:.4f}  frac_recovered={ce['frac_recovered']:.4f}  "
                  f"frac_alive={ce['frac_alive']:.4f}")

        del lm
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["timings"]["total_seconds"] = time.time() - t_start

    # Atomic: a wall-clock kill never leaves half a result behind for the skip-guard to
    # mistake for a finished cell.
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2, default=str))
    tmp.replace(out_path)
    print(f"\nwrote {out_path}  ({results['timings']['total_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
