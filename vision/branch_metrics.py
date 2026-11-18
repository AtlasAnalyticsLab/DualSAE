#!/usr/bin/env python3
"""Offline evaluator: the five metrics (R^2, MonoScore, binary purity, KNN, LP),
measured on one checkpoint and separately on each branch output.

What it measures
----------------
The five metrics are defined on one representation, the full reconstruction
``x_hat``.  For DualSAE ``x_hat = x^M + r_hat``, so those numbers are a *sum*
of the two branches and cannot say how much of the information lives in the
Sparse Branch versus the dense Residual Branch.  This module measures the same
five metrics, with the same protocol as the trainers' own evaluation, for four
representations::

    x       the raw encoder embedding                        (reference)
    xM      the Sparse Branch reconstruction  x^M            (sparse branch alone)
    r_hat   the Residual Branch output                       (residual branch alone)
    xhat    the full reconstruction  x_hat = x^M + r_hat     (the reported column)

For the single-branch baselines ``x^M IS x_hat`` -- the same tensor, which
doubles as a free consistency check -- and ``r_hat`` is *undefined*: it is
emitted as ``null``, never as zeros.

The space-inversion trap
------------------------
``DualSAE.forward`` computes ``x^M`` and ``r_hat`` in the PREPROCESSED space
(``(x-mean)/std`` per sample, only when ``input_unit_norm`` is set, e.g. for
Dual-JumpReLU).  Getting back to the original space is::

    xM_orig    = xM_p    * x_std + x_mean      (== postprocess_output(_xM, ...))
    r_hat_orig = r_hat_p * x_std               (SCALE ONLY -- no mean!)

because ``x_hat = (xM_p + r_hat_p)*std + mean = xM_orig + r_hat_p*std``.  Running
``postprocess_output`` on ``_r_hat`` would add ``x_mean`` a second time; that
perturbation barely rotates the vector, so it is nearly invisible to
KNN/LP/retrieval (which L2-normalize rows) and only shows up in R^2 and
MonoScore.  The inversion lives in ``knn_lp_sparse.forward_reps`` and the
``self_checks`` block of every output JSON asserts ``xM_orig + r_hat_orig ==
x_hat`` so a bad cell is detectable from the output alone.

Metric definitions
------------------
R^2
    ``r2_perfeat``: PER-FEATURE-DIMENSION R^2 in float64, unweighted mean over
    the dims with ``ss_tot > 1e-12`` -- the estimator of ``train_dual.compute_r2``
    (and of the other trainers), i.e. the reported one.  ``r2_global``: the
    single global ratio ``1 - sum_i||x_i-xhat_i||^2 / sum_i||x_i-xbar||^2``.
    Both are emitted.  ``r2@x`` must come out exactly 1.0 -- it is a self-check.

MonoScore
    Latent-level, streaming O(N*D*M), identical to
    ``loss.compute_monosemanticity_fast``: pass 1 takes the global per-latent
    min/max of ``z`` over the eval split; pass 2 accumulates
    ``u += a.sum(0)``, ``v += (a*a).sum(0)``, ``w += p.T @ a`` with
    ``a = (z-min)/(max-min+1e-8)`` and ``p = normalize(V)``; then
    ``MS = 0.5(q-v) / 0.5(u*u-v)`` where ``q = (w*w).sum(0)``.
    THE ONLY PER-REPRESENTATION QUANTITY IS ``p``.  The latents ``z`` are
    identical for all four representations, so pass 1 runs once and pass 2
    accumulates all four ``w`` matrices in a single sweep.
    Three reductions are emitted:
      * ``mean_all``          - unweighted mean over ALL dict_size latents, dead
                                latents contributing 0.  This is what the
                                trainers report (``best_mono.cpu().numpy().mean()``)
                                and what the collectors read.
      * ``mean_active_code``  - mean over ``den > 1e-8`` (the reference code's own
                                validity mask).
      * ``mean_active_paper`` - mean over latents with ``|A_j| > 0``.
    ``mean_all <= mean_active_code`` holds only when every per-latent score is
    >= 0, which is NOT guaranteed (see ``null_cosine`` below).

    MonoScore is not comparable across probe spaces without its floor.
    ``MS_j = sum_{i<l} a_ij a_lj <p_i,p_l> / sum_{i<l} a_ij a_lj`` is a WEIGHTED
    MEAN OF PAIRWISE COSINES in the probe space.  A latent carrying no
    information at all (uniform ``a``) therefore does not score 0 -- it scores
    the space's own mean pairwise cosine::

        null_cosine = (||sum_i p_i||^2 - N) / (N * (N - 1))

    That floor is a property of the REPRESENTATION, not of the latents.  For the
    reported column the probe is always ``normalize(x)``, identical for every
    model, so the floor cancels in a within-encoder comparison.  Across the
    columns here the probe changes and the floor moves with it (dense branch
    outputs such as ``r_hat`` have a much higher floor than ``x``), so every
    MonoScore reduction is emitted together with ``null_cosine`` and the
    corresponding ``excess_*`` (reduction minus floor).  Compare columns on
    ``excess_*``; the raw value is only comparable WITHIN a fixed representation.

Binary purity
    ``metrics.compute_binary_purity`` / ``compute_multilabel_binary_purity``,
    aggregated exactly as ``train_dual.compute_active_binary_purity`` does
    (concatenate the WHOLE eval split, then mean over latents that activate).
    Binary purity is a function of ``z`` and the labels ONLY, so it is
    IDENTICAL for all four representations; it is emitted ONCE per model, in
    the ``model_level`` block.  For a branch comparison the space-dependent
    ``retrieval_purity`` below is also computed.

Retrieval purity @ k=50
    Bank == query set == the eval split, the query itself excluded (diagonal set
    to -inf), cosine similarity on L2-normalized rows, purity = fraction of the
    50 neighbours sharing the query's class, meaned over queries
    (``retrieval_purity_self``).  A second variant ``retrieval_purity_bank``
    queries the *train* bank instead (no self-exclusion needed, splits are
    disjoint), consistent with the KNN/LP protocol.  Note that the residual
    ``x - x_hat`` is NOT ``r_hat``.

KNN (k=20) / LP
    Reused verbatim from ``knn_lp_sparse`` (``knn_metric`` / ``lp_metric``), whose
    protocol constants match the trainers: rows L2-normalized, cosine KNN with
    unweighted majority vote, probe = ``nn.Linear`` + AdamW(1e-3, wd 1e-4),
    10 epochs, batch 1024, seed 0.  Single-label -> top-1 accuracy; multi-label
    (OpenImagesV7) -> micro-F1, both globally accumulated (``metric``) and
    per-batch sample-weighted as in the trainers (``micro_f1_batchwise``).

NOT these metrics
-----------------
Two functions in ``train_dual.py`` sound like they measure what this module
measures.  Neither is reused, and neither is comparable to the columns here:

``compute_r_hat_monosemanticity``
    Treats the ``r_hat`` *feature dimensions* as if they were latents and uses a
    selectivity/sqrt(D) formula.  It is not MonoScore.

``compute_mono_branch_r2``  <-- the confusable one
    Literally "mono-branch R^2" (logged as ``val_r2_mono``), but it compares
    ``autoencoder._xM`` -- the PREPROCESSED-space mono reconstruction -- against
    the RAW ``x``.  For ``input_unit_norm=True`` runs the two spaces differ by
    the per-sample mean and scale, so its value is not a mono-branch R^2 there;
    for ``input_unit_norm=False`` preprocessing is the identity and the two
    agree.  This module's ``xM`` column is the one in the same space as ``x``.

Sparsity caveat
---------------
The JumpReLU / Gated pairs are NOT sparsity-matched (their L0 is emergent).
``l0_mean`` is emitted with every result and should be shown next to those rows.

Usage
-----
    python branch_metrics.py --run-dir ./checkpoints/<exp> \
        --eval-split val --reps x xM r_hat xhat \
        --out-dir ./outputs/branch_metrics/results
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import compute_binary_purity, compute_multilabel_binary_purity
from utils import unpack_batch

# Everything below is imported rather than re-implemented so the numbers are
# produced by the same code as the sparse-code evaluation.
from knn_lp_sparse import (  # noqa: E402
    KNN_K,
    LP_BATCH_SIZE,
    LP_EPOCHS,
    LP_LR,
    LP_SEED,
    LP_WEIGHT_DECAY,
    DenseBank,
    _is_open_images,
    resolve_multilabel,
    forward_reps,
    knn_metric,
    load_run,
    lp_metric,
    make_loader,
)

# --------------------------------------------------------------------------- #
# Protocol constants
# --------------------------------------------------------------------------- #

# Bumping this invalidates every cached result (the skip-guard compares it
# exactly), which is intended: a schema change means the old files are not
# answering the same question.
#   v1 -> v2: MonoScore blocks gained ``null_cosine`` / ``excess_*`` (without the
#             probe space's no-selectivity floor the four MonoScore columns are
#             not on a common scale) plus the probe row-norm diagnostics; a
#             baseline's ``xhat`` block gained ``aliased_from``.
SCHEMA_VERSION = 2

REPS_ALL: Tuple[str, ...] = ("x", "xM", "r_hat", "xhat")
RETRIEVAL_MODES: Tuple[str, ...] = ("self", "bank")

# MonoScore constants, copied from loss.compute_monosemanticity_fast so the
# numbers are bit-comparable with the trainers' MonoScore.
MONO_EPS = 1e-8        # (z - min) / (max - min + MONO_EPS)
MONO_DEN_FLOOR = 1e-8  # den > this, else MS := 0
MONO_DEN_EPS = 1e-12   # num / (den + MONO_DEN_EPS)

# train_dual.compute_r2's default: dims with ss_tot <= R2_EPS are dropped from the
# per-feature mean (they carry no variance, so their R^2 is undefined).
R2_EPS = 1e-12

# The batch size every vision run trained and evaluated with.  BatchTopK selects
# the top k*B activations over the WHOLE batch, so z -- and therefore MonoScore,
# purity and L0 -- depend on this value.  Changing it silently changes the answer.
TRAINING_BATCH_SIZE = 2048


# --------------------------------------------------------------------------- #
# R^2
# --------------------------------------------------------------------------- #

class R2Accumulator:
    """Streaming float64 R^2 of several representations against the same target.

    Reproduces ``train_dual.compute_r2`` exactly for ``r2_perfeat`` (that is the
    estimator the trainers report): per-feature-dimension
    accumulation in float64, ``ss_tot = sum_y2 - sum_y^2/N``, validity mask
    ``ss_tot > 1e-12``, then an UNWEIGHTED MEAN over the valid dims.

    ``r2_global`` is the single global ratio
    ``1 - sum_i||x_i - v_i||^2 / sum_i||x_i - xbar||^2``.  It falls out of the same
    accumulators because summing the per-dim ``ss_res``/``ss_tot`` over dims is
    exactly the numerator/denominator of the global ratio.  It is summed over ALL
    dims (not just the valid ones): a zero-variance dim contributes ~0 to both
    sums, so masking would be a difference without a distinction, and the global
    formula has no mask.

    The two estimators genuinely differ (the per-feature mean weights every
    dimension equally regardless of its variance), which is why both are emitted
    and neither is silently preferred.
    """

    def __init__(self, reps: Sequence[str], device: torch.device) -> None:
        self.reps = list(reps)
        self.device = device
        self.sum_y: Optional[torch.Tensor] = None   # (D,) float64
        self.sum_y2: Optional[torch.Tensor] = None  # (D,) float64
        self.ss_res: Dict[str, torch.Tensor] = {}   # rep -> (D,) float64
        self.n = 0

    def update(self, x: torch.Tensor, outs: Dict[str, torch.Tensor]) -> None:
        """Accumulate one batch.  ``x`` is the RAW target, ``outs`` the reps."""
        x64 = x.to(torch.float64)
        if self.sum_y is None:
            d = x64.shape[1]
            self.sum_y = torch.zeros(d, dtype=torch.float64, device=self.device)
            self.sum_y2 = torch.zeros(d, dtype=torch.float64, device=self.device)
            self.ss_res = {r: torch.zeros(d, dtype=torch.float64, device=self.device)
                           for r in self.reps}
        self.sum_y += x64.sum(dim=0)
        self.sum_y2 += (x64 * x64).sum(dim=0)
        for rep in self.reps:
            diff = x64 - outs[rep].to(torch.float64)
            self.ss_res[rep] += (diff * diff).sum(dim=0)
        self.n += int(x64.shape[0])

    def result(self) -> Dict[str, Dict[str, float]]:
        if self.sum_y is None or self.n == 0:
            return {rep: {"r2_perfeat": float("nan"), "r2_global": float("nan"),
                          "n_valid_dims": 0, "n_dims": 0, "n_samples": 0}
                    for rep in self.reps}
        ss_tot = self.sum_y2 - (self.sum_y * self.sum_y) / max(self.n, 1)
        valid = ss_tot > R2_EPS
        n_valid = int(valid.sum().item())
        ss_tot_sum = float(ss_tot.sum().item())

        out: Dict[str, Dict[str, float]] = {}
        for rep in self.reps:
            ss_res = self.ss_res[rep]
            if n_valid > 0:
                per = 1.0 - (ss_res[valid] / (ss_tot[valid] + R2_EPS))
                r2_perfeat = float(per.mean().item())
            else:
                # Degenerate split (every dim constant).  train_dual.compute_r2
                # returns 0.0 here; match that rather than inventing a NaN.
                r2_perfeat = 0.0
            r2_global = (1.0 - float(ss_res.sum().item()) / ss_tot_sum) if ss_tot_sum > 0 else 0.0
            out[rep] = {
                "r2_perfeat": r2_perfeat,
                "r2_global": r2_global,
                "n_valid_dims": n_valid,
                "n_dims": int(ss_tot.numel()),
                "n_samples": self.n,
            }
        return out


# --------------------------------------------------------------------------- #
# MonoScore
# --------------------------------------------------------------------------- #

@torch.no_grad()
def monoscore_streaming(
    z: torch.Tensor,
    probes: Dict[str, torch.Tensor],
    device: torch.device,
    chunk_size: int,
    eps: float = MONO_EPS,
    progress: bool = True,
) -> Dict[str, Any]:
    """Streaming O(N*D*M) MonoScore for several probe spaces at once.

    Parameters
    ----------
    z       (N, M) dense activations of the eval split, any device (chunks are
            staged to ``device``).  These are the SAME latents for every probe.
    probes  rep -> (N, D_rep) rows that define the similarity space.  They MUST be
            L2-normalized: the closed form ``num = 0.5*(q - v)`` uses
            ``p_i . p_i == 1``.  The eval banks built here are normalized by
            construction, and re-normalizing a unit row is idempotent.
    chunk_size
            The partition of the rows.  Chunking changes nothing mathematically
            but does change fp32 summation order, so callers pass the training
            batch size (2048) to reproduce the trainers' accumulation exactly.

    Returns
    -------
    ``{"ms": {rep: (M,) cpu}, "den": (M,) cpu, "u": ..., "v": ...,
       "probe_stats": {rep: {...}}}``.
    ``den`` depends only on ``z``, so it is shared across probes and returned
    once.  ``probe_stats`` carries the per-probe-space quantities that make the
    scores comparable / auditable:

    ``null_cosine``
        ``(||sum_i p_i||^2 - N) / (N(N-1))`` -- the space's mean pairwise cosine,
        which is the score a latent gets when its ``a`` weights carry NO
        information (uniform ``a`` reduces the weighted mean of pairwise cosines
        to the unweighted one).  This is the per-representation FLOOR of
        MonoScore.  It costs one extra ``(D,)`` accumulator and one dot product,
        and without it the four MonoScore columns are not on a common scale --
        see the module docstring.
    ``min_row_norm`` / ``n_degenerate_rows``
        The closed form ``num = 0.5*(q - v)`` subtracts ``sum_i a_ij^2`` on the
        assumption ``<p_i, p_i> == 1``.  ``F.normalize`` leaves a zero row as a
        zero row (it does not error), so such a row contributes 0 to ``q`` while
        its full ``a_ij^2`` is still subtracted -- biasing ``num`` DOWNWARD with
        no other symptom.  Reachable in principle for ``r_hat`` when ``x_std``
        is 0 for a sample (a constant feature vector zeroes
        ``r_hat_orig = r_hat_p * x_std``) or for an all-zero embedding.  Not
        expected for the vision features (raw features are unit-norm by
        construction; ``r_hat`` rows are far from zero), so this is
        reported as a diagnostic and surfaced in ``self_checks`` rather than
        raising -- killing a long evaluation array over one degenerate row would cost
        more than it saves, and the flag makes the cell detectable either way.
        The scores themselves stay bit-identical to
        ``loss.compute_monosemanticity_fast`` (folding ``||p_i||^2`` into ``v``
        would fix the bias but break that parity, because a normalized row's
        norm is 1.0 +/- 1ulp rather than exactly 1.0).

    Edge cases
    ----------
    * A latent that never fires has ``min == max == 0`` -> ``a = 0`` -> ``u = v = 0``
      -> ``den = 0`` -> MS := 0, matching the reference.
    * A latent that fires with a constant value has ``max - min = 0`` and is
      likewise mapped to 0 by the ``+eps`` denominator; again matching.
    * ``num`` is NOT sign-constrained: a latent whose co-activating rows point in
      opposing probe directions scores negative (constructed case: ``a=[0,1,1,1]``
      over probes ``+e1,+e1,-e1,+e2`` gives MS = -1/3).  Callers must not assume
      MS >= 0.
    """
    n, m = int(z.shape[0]), int(z.shape[1])
    if n == 0:
        raise ValueError("monoscore_streaming: empty split")
    for rep, p in probes.items():
        if int(p.shape[0]) != n:
            raise ValueError(f"probe {rep!r} has {p.shape[0]} rows but z has {n}")

    steps = range(0, n, chunk_size)

    # ---- pass 1: global per-latent min/max (exactly the reference's running
    # min/max over batches; min/max is exact so chunking cannot perturb it).
    min_vals: Optional[torch.Tensor] = None
    max_vals: Optional[torch.Tensor] = None
    for s in tqdm(steps, desc="monoscore pass1", disable=not progress, leave=False):
        zc = z[s:s + chunk_size].to(device, non_blocking=True).float()
        bmin = zc.min(dim=0, keepdim=True)[0]
        bmax = zc.max(dim=0, keepdim=True)[0]
        if min_vals is None:
            min_vals, max_vals = bmin, bmax
        else:
            min_vals = torch.minimum(min_vals, bmin)
            max_vals = torch.maximum(max_vals, bmax)
        del zc

    # ---- pass 2: fp32 accumulators (u, v shared; one w per probe space).
    u = torch.zeros(m, device=device, dtype=torch.float32)
    v = torch.zeros(m, device=device, dtype=torch.float32)
    w = {rep: torch.zeros(int(p.shape[1]), m, device=device, dtype=torch.float32)
         for rep, p in probes.items()}
    # Per-probe row sum -- the ONLY extra state needed for the null floor.  It is
    # accumulated in the same sweep so the floor costs no extra pass over the
    # data (the alternative, a separate reduction over the bank, would re-touch
    # up to 6 GB per rep).
    p_sum = {rep: torch.zeros(int(p.shape[1]), device=device, dtype=torch.float32)
             for rep, p in probes.items()}
    min_row_norm = {rep: float("inf") for rep in probes}
    n_degenerate = {rep: 0 for rep in probes}

    for s in tqdm(steps, desc="monoscore pass2", disable=not progress, leave=False):
        zc = z[s:s + chunk_size].to(device, non_blocking=True).float()
        a = (zc - min_vals) / (max_vals - min_vals + eps)
        u += a.sum(dim=0)
        v += (a * a).sum(dim=0)
        for rep, p in probes.items():
            pc = p[s:s + chunk_size].to(device, non_blocking=True).float()
            w[rep] += pc.T @ a  # (D, B) @ (B, M) -> (D, M)
            p_sum[rep] += pc.sum(dim=0)
            nrm = pc.norm(dim=1)
            min_row_norm[rep] = min(min_row_norm[rep], float(nrm.min().item()))
            n_degenerate[rep] += int((nrm < 1e-6).sum().item())
            del pc, nrm
        del zc, a

    den = 0.5 * (u * u - v)
    ms: Dict[str, torch.Tensor] = {}
    probe_stats: Dict[str, Dict[str, Any]] = {}
    for rep, wr in w.items():
        q = (wr * wr).sum(dim=0)          # ||sum_i a_ij p_i||^2
        num = 0.5 * (q - v)               # sum_{i<l} a_ij a_lj <p_i, p_l>
        ms[rep] = torch.where(den > MONO_DEN_FLOOR, num / (den + MONO_DEN_EPS),
                              torch.zeros_like(den)).cpu()
        # Mean pairwise cosine of the probe space: sum_{i<l} <p_i,p_l> is
        # 0.5*(||sum p||^2 - sum ||p_i||^2) = 0.5*(||sum p||^2 - N) for unit rows,
        # over N(N-1)/2 pairs.  Undefined for N < 2.
        s_vec = p_sum[rep]
        null_cos = (float((s_vec @ s_vec).item()) - n) / (n * (n - 1)) if n > 1 else float("nan")
        probe_stats[rep] = {
            "null_cosine": null_cos,
            "min_row_norm": min_row_norm[rep],
            "n_degenerate_rows": n_degenerate[rep],
            "n_rows": n,
            "dim": int(probes[rep].shape[1]),
        }
    return {"ms": ms, "den": den.cpu(), "u": u.cpu(), "v": v.cpu(),
            "probe_stats": probe_stats}


def summarize_monoscore(ms: torch.Tensor, den: torch.Tensor,
                        active_counts: torch.Tensor,
                        probe_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Reduce a per-latent MonoScore vector the three ways that are in circulation.

    ``mean_all`` is the trainers' reduction (``train_dual.py`` means over all
    ``dict_size`` latents, dead ones contributing 0), ``mean_active_code`` uses the
    reference implementation's own validity mask (``den > 1e-8``), and
    ``mean_active_paper`` uses the activating-latent mask (``|A_j| > 0``).  Per-latent
    distribution stats are included so a shifted mean can be attributed to the
    body of the distribution rather than to a change in how many latents are dead.

    ``probe_stats`` (from ``monoscore_streaming``) adds the probe space's
    ``null_cosine`` -- the no-selectivity floor -- and one ``excess_*`` per
    reduction.  THE ``excess_*`` VALUES ARE THE CROSS-REPRESENTATION NUMBERS: a
    raw MonoScore below the space's floor is *below chance*, however high it
    looks.  ``mean_*`` remains the reported quantity and the right thing to
    compare WITHIN a fixed
    representation, where the floor is shared and cancels.

    No inequality is assumed between the three reductions: per-latent MonoScore
    can be negative (opposing probe directions among a latent's co-activating
    rows), in which case the dead latents' exact zeros pull ``mean_all`` ABOVE
    the masked means.
    """
    ms = ms.detach().float().cpu()
    den = den.detach().float().cpu()
    active_counts = active_counts.detach().cpu()

    code_mask = den > MONO_DEN_FLOOR
    paper_mask = active_counts > 0

    def _mean(mask: torch.Tensor) -> float:
        return float(ms[mask].mean().item()) if bool(mask.any()) else 0.0

    qs = torch.tensor([0.25, 0.5, 0.75])
    quant = torch.quantile(ms, qs) if ms.numel() else torch.full((3,), float("nan"))
    out = {
        "mean_all": float(ms.mean().item()),
        "mean_active_code": _mean(code_mask),
        "mean_active_paper": _mean(paper_mask),
        "n_latents": int(ms.numel()),
        "n_active_code": int(code_mask.sum().item()),
        "n_active_paper": int(paper_mask.sum().item()),
        "sd_all": float(ms.std(unbiased=True).item()) if ms.numel() > 1 else 0.0,
        "min": float(ms.min().item()),
        "p25": float(quant[0].item()),
        "median": float(quant[1].item()),
        "p75": float(quant[2].item()),
        "max": float(ms.max().item()),
    }
    stats = probe_stats or {}
    null_cos = stats.get("null_cosine")
    out["null_cosine"] = null_cos
    out["probe_min_row_norm"] = stats.get("min_row_norm")
    out["probe_n_degenerate_rows"] = stats.get("n_degenerate_rows")
    # excess = reduction - floor.  None (not 0.0) when the floor is unavailable,
    # so a consumer can never silently treat a missing floor as a floor of zero.
    for key in ("mean_all", "mean_active_code", "mean_active_paper"):
        out[f"excess_{key.replace('mean_', '')}"] = (
            None if null_cos is None or math.isnan(null_cos) else out[key] - null_cos)
    return out


# --------------------------------------------------------------------------- #
# Binary purity (model level -- rep independent)
# --------------------------------------------------------------------------- #

def binary_purity_summary(z: torch.Tensor, labels: torch.Tensor,
                          device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Mean binary purity over the latents that activate somewhere on the split.

    This is ``train_dual.compute_active_binary_purity`` verbatim: the WHOLE eval
    split is concatenated first (never averaged per batch, which would be a
    different estimator), the single-label / multi-label variant is chosen by the
    label rank, and the mean runs over ``(acts > 0).any(dim=0)``.

    *** Rep independence *** -- the value depends on ``z`` and ``labels`` only.
    There is deliberately no representation argument: computing it four times
    would produce four identical numbers and invite a false reading.

    The GPU path can need ~3x the size of ``z`` (the multi-label branch casts the
    activity mask to float for a matmul), so an OOM falls back to CPU rather than
    killing a long evaluation array.
    """
    labels_2d = labels.ndim == 2
    targets = ([device] if device is not None and device.type != "cpu" else []) + [torch.device("cpu")]
    for target in targets:
        zt = yt = None
        try:
            zt = z.to(target)
            yt = labels.to(target)
            purity = (compute_multilabel_binary_purity(zt, yt) if labels_2d
                      else compute_binary_purity(zt, yt))
            active_counts = (zt > 0).sum(dim=0)
            active_mask = active_counts > 0
            mean_active = float(purity[active_mask].mean().item()) if bool(active_mask.any()) else 0.0
            out = {
                "mean_active": mean_active,
                "n_active_latents": int(active_mask.sum().item()),
                "n_latents": int(purity.numel()),
                "multilabel": bool(labels_2d),
                "device": str(target),
                "note": ("binary purity is a function of z and the labels only -- it is "
                         "IDENTICAL for x, xM, r_hat and xhat, so it is reported once "
                         "per model, not once per representation"),
            }
            del zt, yt
            if target.type == "cuda":
                torch.cuda.empty_cache()
            return out
        except RuntimeError as exc:  # pragma: no cover - needs a real GPU to hit
            # torch.cuda.OutOfMemoryError subclasses RuntimeError; matching on the
            # message keeps this working on torch builds that predate the class.
            if "out of memory" not in str(exc).lower():
                raise
            del zt, yt
            if target.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[warn] binary purity OOMed on {target}; retrying on CPU")
    raise RuntimeError("binary purity failed on every device")


# --------------------------------------------------------------------------- #
# Retrieval purity @ k
# --------------------------------------------------------------------------- #

@torch.no_grad()
def retrieval_purity(
    query_bank: DenseBank,
    db_bank: DenseBank,
    device: torch.device,
    k: int = 50,
    exclude_self: bool = False,
    query_block: int = 512,
    db_block: int = 65_536,
    progress: bool = True,
) -> Dict[str, Any]:
    """Label agreement among each query's top-k nearest neighbours.

    Protocol (top-k class-purity retrieval):

    * rows are L2-normalized already (both banks are built that way), so the inner
      product IS the cosine;
    * ``exclude_self=True`` is within-split self-retrieval:
      the query's own row is masked out with ``-inf`` before the top-k, which is
      required because otherwise every query retrieves itself at similarity 1;
    * the global top-k across bank blocks is obtained by merging per-block top-k
      candidates, which is exact;
    * single-label purity = ``mean_n 1[y_n == y_query]``, i.e. the fraction of the
      k neighbours carrying the query's class, then meaned over queries.

    Two deliberate extensions of the plain protocol:

    1. MULTI-LABEL SUPPORT (OpenImagesV7 has none in the original script).  The
       generalization used here is the max per-class support among the k
       neighbours, restricted to the query's own positive classes::

           purity(i) = max_{c : y_i[c]=1} (1/k) sum_{n in N_i} y_n[c]

       This is the same functional shape as
       ``metrics.compute_multilabel_binary_purity`` (max class support over an
       index set) and it REDUCES EXACTLY to the single-label definition when the
       query has one positive class, so the ImageNet and OpenImages columns mean
       the same thing.  Queries with NO positive label are skipped, not scored 0:
       10.57% of the OIv7 test split is unlabeled, and scoring those as 0 would
       depress every representation of every model by ~10% uniformly -- an
       artifact that looks like a result.  ``n_scored`` / ``n_skipped_unlabeled``
       are always reported.
    2. ``coherence`` is additionally emitted: the largest class share among the k
       neighbours *regardless of the query's own label* (single-label: the modal
       class's share).  It is a neighbourhood-homogeneity diagnostic -- it rewards
       a coherently WRONG neighbourhood -- so it must never be called purity.

    Block sizes affect neither result (exact top-k merge) except in the measure-zero
    case of exactly tied similarities.
    """
    n_db = int(db_bank.n_rows)
    n_q = int(query_bank.n_rows)
    if exclude_self and n_q != n_db:
        raise ValueError("exclude_self=True requires the query bank to BE the db bank")
    # One row is unavailable to itself under self-retrieval.
    k_eff = min(k, n_db - (1 if exclude_self else 0))
    if k_eff <= 0:
        return {"purity": float("nan"), "coherence": float("nan"), "k": 0,
                "n_scored": 0, "n_skipped_unlabeled": 0,
                "note": f"bank has {n_db} rows; not enough for k={k}"}

    db_labels = db_bank.labels          # kept on CPU: the (B, k, C) gather is small
    q_labels = query_bank.labels
    multilabel = db_labels.ndim == 2

    purity_sum = 0.0
    coherence_sum = 0.0
    n_scored = 0
    n_skipped = 0

    for q0 in tqdm(range(0, n_q, query_block), desc=f"retrieval@{k}", disable=not progress, leave=False):
        q1 = min(q0 + query_block, n_q)
        q = query_bank.x[q0:q1].to(device, non_blocking=True).float()

        best_vals: Optional[torch.Tensor] = None
        best_inds: Optional[torch.Tensor] = None
        for b0 in range(0, n_db, db_block):
            b1 = min(b0 + db_block, n_db)
            d = db_bank.x[b0:b1].to(device, non_blocking=True).float()
            sim = q @ d.t()                                   # (Bq, Bd)
            if exclude_self:
                # Mask the diagonal of the overlapping sub-block: query row g maps
                # to db column g - b0 when b0 <= g < b1.
                lo, hi = max(q0, b0), min(q1, b1)
                if hi > lo:
                    rows = torch.arange(lo - q0, hi - q0, device=device)
                    cols = torch.arange(lo - b0, hi - b0, device=device)
                    sim[rows, cols] = float("-inf")
            kk = min(k_eff, b1 - b0)
            vals, inds = torch.topk(sim, k=kk, dim=1)
            inds = inds + b0                                   # -> global row ids
            if best_vals is None:
                best_vals, best_inds = vals, inds
            else:
                merged_v = torch.cat([best_vals, vals], dim=1)
                merged_i = torch.cat([best_inds, inds], dim=1)
                keep = torch.topk(merged_v, k=min(k_eff, merged_v.shape[1]), dim=1).indices
                best_vals = torch.gather(merged_v, 1, keep)
                best_inds = torch.gather(merged_i, 1, keep)
            del sim, d

        nn_idx = best_inds.cpu()                               # (Bq, k_eff)
        if multilabel:
            nn_lbl = db_labels[nn_idx].float()                 # (Bq, k, C)
            support = nn_lbl.mean(dim=1)                       # (Bq, C)
            qpos = q_labels[q0:q1].float() > 0.5               # (Bq, C)
            scored = qpos.any(dim=1)
            if bool(scored.any()):
                masked = torch.where(qpos, support, torch.full_like(support, float("-inf")))
                row_purity = masked.max(dim=1).values[scored]
                purity_sum += float(row_purity.sum().item())
                n_scored += int(scored.sum().item())
            n_skipped += int((~scored).sum().item())
            coherence_sum += float(support.max(dim=1).values.sum().item())
        else:
            nn_lbl = db_labels[nn_idx].long()                  # (Bq, k)
            qy = q_labels[q0:q1].long().unsqueeze(1)
            row_purity = (nn_lbl == qy).float().mean(dim=1)
            purity_sum += float(row_purity.sum().item())
            n_scored += int(row_purity.numel())
            # Modal-class share among the neighbours (query-agnostic diagnostic).
            modal = torch.mode(nn_lbl, dim=1).values.unsqueeze(1)
            coherence_sum += float((nn_lbl == modal).float().mean(dim=1).sum().item())

    return {
        "purity": purity_sum / n_scored if n_scored else float("nan"),
        "coherence": coherence_sum / max(n_q, 1),
        "k": int(k_eff),
        "n_queries": n_q,
        "n_bank": n_db,
        "n_scored": n_scored,
        "n_skipped_unlabeled": n_skipped,
        "exclude_self": bool(exclude_self),
        "multilabel": bool(multilabel),
    }


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

@dataclass
class EncodedSplit:
    """Everything one pass over a split produces."""

    banks: Dict[str, DenseBank]                      # rep -> L2-normalized rows (CPU)
    labels: torch.Tensor
    n_rows: int
    l0_mean: float                                   # mean nnz of z per row
    active_counts: Optional[torch.Tensor] = None     # (M,) nnz per latent
    z: Optional[torch.Tensor] = None                 # (N, M) dense code, CPU float32
    r2: Dict[str, Dict[str, float]] = field(default_factory=dict)
    checks: Dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    # rep -> the rep it is an ALIAS OF (identity for a rep computed in its own
    # right).  Single-branch models return the same tensor for ``xM`` and
    # ``xhat``, so one of the two is a pure alias: materializing it twice would
    # double the largest allocation in the job (a SigLIP2 ImageNet train bank is
    # 5.9 GB) and make every downstream metric run twice on identical rows.
    canonical: Dict[str, str] = field(default_factory=dict)


@torch.no_grad()
def encode_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    reps: Sequence[str],
    is_dual: bool,
    split_name: str,
    cache_z: bool = False,
    want_r2: bool = False,
    want_active_counts: bool = False,
    max_samples: Optional[int] = None,
) -> EncodedSplit:
    """One pass over ``loader`` producing every bank plus the streaming statistics.

    Why this is not ``knn_lp_sparse.build_banks``: that function builds banks and
    nothing else.  This pass must ALSO (a) stream float64 R^2 accumulators against
    the raw input, (b) cache ``z`` densely for the two-pass MonoScore and for
    purity, (c) count per-latent activity, and (d) run the branch-decomposition
    self-checks.  Folding all of that into ``build_banks`` would complicate a
    function with 1124 result files depending on it; the bank-building code here
    is deliberately identical in behaviour (``F.normalize(rep, p=2, dim=1)``, same
    label dtypes, same ``max_samples`` prefix semantics).

    The pass is I/O bound on the 1.28M-row ImageNet train split, so every
    representation is produced from a single forward -- re-walking the LMDB per
    representation would cost 4x the dominant term.

    ``max_samples`` semantics: the FORWARD always runs on whole batches (slicing
    the input would change BatchTopK's batch-level top-k and therefore ``z``),
    but every accumulator is truncated to the sample budget in the same place.
    That keeps ``l0_mean`` / ``active_counts`` describing exactly the ``n_rows``
    rows that ``z``, the banks and the labels describe -- otherwise a
    ``--max-eval-samples 1024`` smoke run reports an activity mask over 2048 rows
    next to a MonoScore over 1024, i.e. a smoke test that is internally
    inconsistent.
    """
    model.eval()
    reps = list(reps)
    # Alias map: for a single-branch model xM IS xhat (the same tensor object),
    # so only the canonical one is materialized and every consumer of the alias
    # gets the canonical object back.  ``xM`` is chosen as canonical because it
    # sorts first in REPS_ALL, which keeps the printed order stable.
    canonical: Dict[str, str] = {r: r for r in reps}
    if not is_dual and "xM" in reps and "xhat" in reps:
        canonical["xhat"] = "xM"
    mat_reps = [r for r in reps if canonical[r] == r]

    dense_acc: Dict[str, List[torch.Tensor]] = {r: [] for r in mat_reps}
    label_chunks: List[torch.Tensor] = []
    z_chunks: List[torch.Tensor] = []
    r2_acc = R2Accumulator(mat_reps, device) if want_r2 else None

    nnz_total = 0
    active_counts: Optional[torch.Tensor] = None
    seen = 0
    t0 = time.time()

    # Self-check accumulators (only meaningful on a dual model).
    inv_max_abs = 0.0
    naive_max_abs = 0.0
    xhat_abs_sum = 0.0
    xhat_n = 0
    xM_is_xhat = True

    for batch in tqdm(loader, desc=f"encode {split_name}"):
        features, class_idx, _ = unpack_batch(batch)
        if class_idx is None:
            raise ValueError("evaluation requires labels; open the LMDB with return_class_idx=True")
        x = features.to(device, non_blocking=True).float()
        out = forward_reps(model, x, is_dual)

        # ---- self-checks, per batch, on the tensors we actually use downstream.
        # Run BEFORE any truncation: ``is`` identity survives slicing in neither
        # direction (``t[:n] is t`` is False even for a full-length slice), and
        # the checks are strictly stronger on the whole batch.
        if is_dual:
            recomposed = out["xM"] + out["r_hat"]
            inv_max_abs = max(inv_max_abs, float((recomposed - out["xhat"]).abs().max().item()))
            # The trap: postprocess_output() on _r_hat adds x_mean a SECOND time.
            # Measuring it here means the JSON proves the correct inversion was
            # used, instead of merely asserting it.
            naive_r_hat = model.postprocess_output(model._r_hat, model._x_mean, model._x_std)
            naive_max_abs = max(naive_max_abs,
                                float((out["xM"] + naive_r_hat - out["xhat"]).abs().max().item()))
            del recomposed, naive_r_hat
        else:
            # For single-branch models forward_reps returns the SAME object twice.
            xM_is_xhat = xM_is_xhat and (out["xM"] is out["xhat"])
        xhat_abs_sum += float(out["xhat"].abs().sum().item())
        xhat_n += int(out["xhat"].numel())

        # How many of this batch's rows still fit in the sample budget.  The
        # forward above used the WHOLE batch on purpose (BatchTopK's top-k is
        # batch-level, so a sliced input would produce different z for the rows
        # we keep); only the accumulation is truncated, and it is truncated for
        # every accumulator at once.
        take = int(x.shape[0])
        if max_samples is not None:
            take = min(take, max_samples - seen)
            if take <= 0:
                break

        if r2_acc is not None:
            r2_acc.update(x[:take], {r: out[r][:take] for r in mat_reps})

        z = out["z"][:take].float()
        mask = z > 0
        nnz_total += int(mask.sum().item())
        if want_active_counts:
            counts = mask.sum(dim=0)
            active_counts = counts if active_counts is None else active_counts + counts
        if cache_z:
            z_chunks.append(z.cpu())

        for rep in mat_reps:
            # Same normalization the trainers' bank builder applies, for KNN and
            # LP alike; the MonoScore probe reuses these rows (normalizing an
            # already-unit row is idempotent).
            dense_acc[rep].append(F.normalize(out[rep][:take].float(), p=2, dim=1).cpu())

        ci = class_idx[:take]
        label_chunks.append(ci.detach().cpu().float() if ci.ndim == 2
                            else ci.detach().cpu().long())

        seen += take
        del out, z, mask
        if max_samples is not None and seen >= max_samples:
            break

    if seen == 0:
        raise ValueError(f"no samples encoded for split {split_name}")

    labels = torch.cat(label_chunks, dim=0)
    del label_chunks
    n_rows = int(labels.shape[0])
    # Every accumulator was truncated at the same point, so this is an invariant
    # rather than a hope: if it ever breaks, l0_mean/active_counts would describe
    # a different row set than z and the banks.
    assert n_rows == seen, f"label rows {n_rows} != accumulated rows {seen}"

    banks: Dict[str, DenseBank] = {}
    for rep in mat_reps:
        # Concatenate and immediately drop the chunk list: holding all four chunk
        # lists AND all four concatenated banks is what drives the peak RSS
        # (~27 GiB at ImageNet/SigLIP2), and this halves it.
        banks[rep] = DenseBank(torch.cat(dense_acc[rep], dim=0), labels)
        dense_acc[rep] = []
    del dense_acc
    # Aliases point at the SAME DenseBank object, so `banks["xhat"] is
    # banks["xM"]` for a baseline -- callers use that identity to skip the
    # duplicate metric computation.
    for rep, src in canonical.items():
        if src != rep:
            banks[rep] = banks[src]

    z_cat: Optional[torch.Tensor] = None
    if cache_z:
        z_cat = torch.cat(z_chunks, dim=0)
        del z_chunks

    checks: Dict[str, Any] = {
        "xhat_abs_mean": xhat_abs_sum / max(xhat_n, 1),
        "xM_is_xhat_object": bool(xM_is_xhat) if not is_dual else None,
    }
    if is_dual:
        scale = checks["xhat_abs_mean"] or 1.0
        checks.update(
            decomposition_max_abs_err=inv_max_abs,
            decomposition_rel_err=inv_max_abs / scale,
            naive_postprocess_max_abs_err=naive_max_abs,
            naive_postprocess_rel_err=naive_max_abs / scale,
        )

    r2 = r2_acc.result() if r2_acc is not None else {}
    # Mirror the alias into the R^2 dict so downstream code can index every
    # requested rep without knowing about aliasing.
    for rep, src in canonical.items():
        if src != rep and src in r2:
            r2[rep] = r2[src]

    return EncodedSplit(
        banks=banks,
        labels=labels,
        n_rows=n_rows,
        l0_mean=nnz_total / max(seen, 1),
        active_counts=active_counts,
        z=z_cat,
        r2=r2,
        checks=checks,
        seconds=time.time() - t0,
        canonical=canonical,
    )


# --------------------------------------------------------------------------- #
# Completion predicate (shared with the submitter)
# --------------------------------------------------------------------------- #

def resolve_active_reps(reps: Sequence[str], is_dual: bool, model_type: str = "?") -> List[str]:
    """Which of the requested reps this model can actually produce.

    ``r_hat`` exists only for DualSAE.  Dropping it (rather than erroring) is what
    lets ONE command cover baselines and DualSAEs; the caller records it as
    ``null`` in the JSON so a reader cannot mistake "undefined" for "zero".

    If that leaves nothing -- only reachable via ``--reps r_hat`` on a
    single-branch model -- this raises.  It is a hard error and it must be raised
    HERE, before the two encode passes: the ImageNet train split is 1.28M rows,
    so the alternative is burning GPU-minutes and then dying with an
    ``IndexError`` that names neither the rep nor the model family.  Writing a
    stub result instead would be worse still: it would clobber a complete result
    sitting at the same output path.
    """
    active = [r for r in reps if r != "r_hat" or is_dual]
    if not active:
        raise SystemExit(
            f"--reps selected only 'r_hat', which is undefined for this single-branch "
            f"{model_type} model (there is no poly branch to measure). "
            "Request at least one of x/xM/xhat, or restrict the array to dual runs.")
    return active


def result_is_complete(payload: Dict[str, Any], reps: Sequence[str],
                       retrieval_modes: Sequence[str] = (),
                       encode_batch_size: int = TRAINING_BATCH_SIZE,
                       knn_k: int = KNN_K,
                       retrieval_k: int = 50) -> bool:
    """Does this cached payload already answer the requested question?

    Pure dict logic (no torch), so the submitter can reuse it and the skip-guard
    can never drift from the completion check that decides whether to burn an
    array slot.  Every queued task costs fair-share priority even if it exits in
    seconds, so a false "incomplete" is expensive.

    Rules:
    * the schema version must match exactly -- an older schema is not the same
      answer, even if the keys happen to line up;
    * every requested representation must be present.  ``None`` counts as present
      and complete: that is how a baseline records "r_hat is undefined here", and
      recomputing it would never produce anything else;
    * a present block must carry every metric family we would have written;
    * a CAPPED run (``--max-train-samples`` / ``--max-eval-samples``) is never
      complete.  Those flags exist for smoke tests, the loader is unshuffled so a
      cap is a prefix, and a 2048-row prefix is not the eval split.  Without this
      rule a single smoke run dropped into the results directory would silently
      mark a real cell done -- which is exactly the failure mode that is hardest
      to notice, because the JSON looks entirely well-formed;
    * the PROTOCOL CONSTANTS must match what is being asked for.  A result
      computed at ``--encode-batch-size 512`` is a fully-formed, ``status: "ok"``,
      uncapped file that is nevertheless WRONG for the 36 BatchTopK cells (the
      top-k is taken over the whole batch, so z -- and with it MonoScore, purity,
      L0 and every representation -- depends on the batch size).  ``knn_k`` and
      ``retrieval_k`` are checked for the same reason: a k=10 retrieval answers a
      different question than a k=50 one, and nothing else in the pipeline would
      notice.
    """
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != "ok":
        return False
    if payload.get("max_train_samples") is not None or payload.get("max_eval_samples") is not None:
        return False
    if payload.get("encode_batch_size") != encode_batch_size:
        return False
    if payload.get("knn_k") != knn_k:
        return False
    # retrieval_k only constrains the result when a retrieval mode was requested.
    if retrieval_modes and payload.get("retrieval_k") != retrieval_k:
        return False
    blocks = payload.get("reps")
    if not isinstance(blocks, dict):
        return False
    required = ["r2_perfeat", "r2_global", "monoscore", "knn", "lp"]
    required += [f"retrieval_purity_{m}" for m in retrieval_modes]
    for rep in reps:
        if rep not in blocks:
            return False
        block = blocks[rep]
        if block is None:          # undefined-by-construction, e.g. baseline r_hat
            continue
        if not isinstance(block, dict) or any(key not in block for key in required):
            return False
    return True


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def remap_feature_paths(meta: Dict[str, Any], features_root: str) -> None:
    """Rewrite meta['{train,val,test}_path'] onto ``features_root`` (in place).

    The dataset directory is taken from the run's recorded dataset name (falling back to the
    path itself), the split directory and the LMDB name from the recorded path.
    """
    ds = meta.get("dataset_name")
    for key in ("train_path", "val_path", "test_path"):
        old = meta.get(key)
        if not old:
            continue
        name = ds or ("open_images" if "open_image" in str(old).lower() else "imagenet")
        meta[key] = str(Path(features_root) / name / Path(old).parent.name / Path(old).name)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="checkpoint directory (holds best_val_monoscore_model.pt)")
    p.add_argument("--checkpoint", default="best", choices=("best", "final"),
                   help='which saved checkpoint to score: "best" = best_val_monoscore_model.pt (default; the checkpoint the trainers evaluate), "final" = autoencoder_final.pt (end of training). MonoScore often peaks at epoch 1, so "best" can score a model from before the swept knob has acted; "final" re-scores at convergence.')
    p.add_argument("--eval-split", default="val", choices=["val", "test"],
                   help="split every metric is reported on; the KNN/LP bank is always the train split")
    p.add_argument("--reps", nargs="+", default=list(REPS_ALL), choices=list(REPS_ALL))
    p.add_argument("--out-dir", default="./outputs/branch_metrics/results")
    p.add_argument("--encode-batch-size", type=int, default=TRAINING_BATCH_SIZE,
                   help="MUST match training (2048) -- BatchTopK's top-k is batch-level, so z, "
                        "MonoScore, purity and L0 all depend on this")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--knn-k", type=int, default=KNN_K)
    p.add_argument("--knn-query-batch", type=int, default=1024)
    p.add_argument("--knn-bank-chunk", type=int, default=262_144)
    p.add_argument("--retrieval-k", type=int, default=50, help="neighbours for retrieval purity")
    p.add_argument("--retrieval-modes", nargs="+", default=list(RETRIEVAL_MODES),
                   choices=list(RETRIEVAL_MODES),
                   help="'self' = within-eval-split, self-excluded ; "
                        "'bank' = eval queries against the train bank (consistent with KNN/LP)")
    p.add_argument("--retrieval-query-block", type=int, default=512)
    p.add_argument("--retrieval-db-block", type=int, default=65_536)
    p.add_argument("--max-train-samples", type=int, default=None, help="smoke-test cap on the bank")
    p.add_argument("--max-eval-samples", type=int, default=None, help="smoke-test cap on the eval split")
    p.add_argument("--features-root", default=None,
                   help="remap the run's recorded feature paths onto this root "
                        "(<root>/<dataset>/<split dir>/<lmdb name>)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true", help="recompute even if the output JSON is complete")
    return p.parse_args(argv)


def _fmt(value: Optional[float], nd: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{nd}f}"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    reps = [r for r in REPS_ALL if r in set(args.reps)]  # canonical order
    modes = [m for m in RETRIEVAL_MODES if m in set(args.retrieval_modes)]
    if not reps:
        raise SystemExit("--reps selected nothing")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Same tag scheme as knn_lp_sparse: run_dir basename + eval split.  Run dirs
    # may live in one flat directory and their names need not encode
    # neither dataset nor encoder, but the names ARE unique (a time_<epoch>
    # suffix), so this tag is collision-free.  Collectors must join back to the
    # manifest to recover dataset/encoder.
    # The checkpoint is part of the identity of a result. Scoring one run at
    # both `best` and `final` (which the Gated block does, to show the
    # protocol's checkpoint selection against the converged model) would
    # otherwise write to ONE path -- the second run either overwriting the
    # first or being skipped by the resume guard below, silently reporting one
    # checkpoint's numbers under the other's label.
    #
    # Only NON-default checkpoints are suffixed, so every result already on
    # disk keeps its name and stays valid.
    ckpt_tag = "" if args.checkpoint == "best" else f"__{args.checkpoint}"
    out_path = out_dir / f"{run_dir.name}__{args.eval_split}{ckpt_tag}.json"

    # Resumability: a wall-clock kill must cost only the unfinished cells.
    if out_path.exists() and not args.overwrite:
        try:
            cached = json.loads(out_path.read_text())
        except Exception:
            cached = {}
        # The protocol constants are part of the question, not just of the
        # answer: a result computed with a different encode batch size / k is not
        # this cell's result even though it is well-formed.
        if result_is_complete(cached, reps, modes, args.encode_batch_size,
                              args.knn_k, args.retrieval_k):
            print(f"[skip] {out_path} already holds reps {reps} with modes {modes}")
            return
        print(f"[recompute] {out_path} is missing part of reps={reps} modes={modes} "
              f"(or was computed with a different encode_batch_size/knn_k/retrieval_k)")

    model, meta = load_run(run_dir, device, checkpoint=args.checkpoint)
    if args.features_root:
        # The run's training_config.json records absolute feature paths. When the features
        # have moved, remap them onto <features_root>/<dataset>/<split dir>/<lmdb name> (the
        # layout written by extract_imagenet.py / extract_open_images.py).
        remap_feature_paths(meta, args.features_root)
    is_dual = bool(meta.get("is_dual"))
    # Same precedence as knn_lp_sparse: recorded value, then LMDB marker, then path substring.
    multilabel = resolve_multilabel(meta.get("train_path"), meta.get("multilabel"))
    dict_size = int(meta["dict_size"])

    eval_path = meta.get(f"{args.eval_split}_path")
    if not eval_path:
        raise ValueError(
            f"run has no {args.eval_split}_path in training_config.json "
            f"(ImageNet runs have test_path=None; the manifest asks for eval_split='val' there)")
    train_path = meta.get("train_path")
    if not train_path:
        raise ValueError("run has no train_path in training_config.json; the KNN/LP bank is undefined")

    # r_hat exists only for DualSAE.  Dropping it from the active set (rather than
    # erroring) is what lets one command cover baselines and DualSAEs; the JSON
    # records it as null so a reader cannot mistake "undefined" for "zero".
    r_hat_requested = "r_hat" in reps
    active_reps = resolve_active_reps(reps, is_dual, str(meta.get("model_type")))
    if r_hat_requested and not is_dual:
        print("[note] non-dual model: r_hat is undefined and will be emitted as null")

    if args.encode_batch_size != TRAINING_BATCH_SIZE and str(meta.get("sparsity_mode") or
                                                             meta.get("model_type")).endswith("batch_topk"):
        print(f"[WARN] encode-batch-size={args.encode_batch_size} != training {TRAINING_BATCH_SIZE} on a "
              "BatchTopK model: the top-k is taken over the whole batch, so z (and MonoScore, "
              "purity, L0) will NOT match the trained run's own evaluation")

    print(f"run={run_dir.name}\n  model={meta['model_type']} mode={meta.get('sparsity_mode')} "
          f"seed={meta['seed']} dict={dict_size} dual={is_dual} "
          f"input_unit_norm={meta.get('input_unit_norm')} multilabel={multilabel}")
    print(f"  reps={active_reps} retrieval_modes={modes} eval_split={args.eval_split}")

    t_start = time.time()
    # Eval split first: it is ~25x smaller than the ImageNet train split and runs
    # every self-check, so a broken cell fails in seconds rather than minutes.
    eval_loader = make_loader(eval_path, args.encode_batch_size, args.num_workers, multilabel)
    ev = encode_split(model, eval_loader, device, active_reps, is_dual, args.eval_split,
                      cache_z=True, want_r2=True, want_active_counts=True,
                      max_samples=args.max_eval_samples)
    print(f"  eval: n={ev.n_rows} L0={ev.l0_mean:.1f} ({ev.seconds:.0f}s)")

    train_loader = make_loader(train_path, args.encode_batch_size, args.num_workers, multilabel)
    tr = encode_split(model, train_loader, device, active_reps, is_dual, "train",
                      cache_z=False, want_r2=False, want_active_counts=False,
                      max_samples=args.max_train_samples)
    print(f"  train bank: n={tr.n_rows} L0={tr.l0_mean:.1f} ({tr.seconds:.0f}s)")
    encode_s = time.time() - t_start

    # ---- model-level (rep-independent) metrics -----------------------------
    t0 = time.time()
    purity = binary_purity_summary(ev.z, ev.labels, device)
    purity_s = time.time() - t0
    print(f"  binary purity (rep-independent) = {purity['mean_active']:.4f} "
          f"over {purity['n_active_latents']}/{purity['n_latents']} active latents ({purity_s:.0f}s)")

    # ---- MonoScore: one sweep, four probe spaces ---------------------------
    t0 = time.time()
    # Aliased reps (a baseline's xhat, which IS its xM) share a probe tensor, so
    # they are passed once and the summary is mirrored afterwards.
    probe_reps = [r for r in active_reps if ev.canonical.get(r, r) == r]
    mono = monoscore_streaming(
        ev.z, {rep: ev.banks[rep].x for rep in probe_reps}, device,
        chunk_size=args.encode_batch_size,
    )
    mono_s = time.time() - t0
    mono_summary = {}
    for rep in active_reps:
        src = ev.canonical.get(rep, rep)
        mono_summary[rep] = summarize_monoscore(mono["ms"][src], mono["den"], ev.active_counts,
                                                mono["probe_stats"].get(src))
    for rep in probe_reps:
        s = mono_summary[rep]
        # The floor is printed next to the score on purpose: reading MonoScore
        # across representations without it inverts the branch conclusion.
        print(f"  MonoScore@{rep:5s} mean_all={s['mean_all']:.4f} "
              f"null={_fmt(s['null_cosine'])} excess={_fmt(s['excess_all'])} "
              f"mean_active_code={s['mean_active_code']:.4f}")

    # z is no longer needed; it is the single largest CPU allocation (4.1 GB on
    # the OIv7 test split) and KNN/LP still have to stage 6 GB banks to the GPU.
    ev.z = None

    # ---- self-checks -------------------------------------------------------
    r2_x = ev.r2.get("x", {}).get("r2_perfeat")
    ms_x = mono_summary.get("x", {}).get("mean_all")
    ckpt_ms = meta.get("val_monoscore")
    self_checks: Dict[str, Any] = {
        "r2_x_equals_one": {
            "applicable": "x" in active_reps,
            "r2_perfeat": r2_x,
            "r2_global": ev.r2.get("x", {}).get("r2_global"),
            # ss_res is identically 0 in float64 when the target IS the rep, so
            # this is exact rather than approximate.  A cell that did not request
            # rep 'x' cannot fail a check it never ran.
            "ok": bool("x" not in active_reps or (r2_x is not None and abs(r2_x - 1.0) < 1e-9)),
        },
        "dual_decomposition": {
            "applicable": bool(is_dual and r_hat_requested),
            "input_unit_norm": bool(meta.get("input_unit_norm", False)),
            "max_abs_err": ev.checks.get("decomposition_max_abs_err"),
            "rel_err": ev.checks.get("decomposition_rel_err"),
            "naive_postprocess_max_abs_err": ev.checks.get("naive_postprocess_max_abs_err"),
            "naive_postprocess_rel_err": ev.checks.get("naive_postprocess_rel_err"),
            # Relative rather than a bare atol: |x| ~ 0.02 on these features, so an
            # absolute tolerance is only ~1.5 decades below the trap it guards.
            "ok": bool(not is_dual or not r_hat_requested
                       or (ev.checks.get("decomposition_rel_err", 1.0) < 1e-4)),
            "naive_is_detectably_wrong": bool(
                is_dual and r_hat_requested
                and (ev.checks.get("naive_postprocess_rel_err") or 0.0) > 1e-3),
        },
        "baseline_xM_is_xhat": {
            "applicable": not is_dual,
            # Structural, not numeric: forward_reps returns the same object twice
            # for single-branch models, so this can only be False if that changes.
            "identical_object": ev.checks.get("xM_is_xhat_object"),
            "ok": bool(is_dual or ev.checks.get("xM_is_xhat_object")),
        },
        "monoscore_x_vs_checkpoint": {
            # The checkpoint's val_monoscore was computed on the VAL split with
            # batch 2048; it is a genuine parity gate for ImageNet (eval=val) and
            # not comparable for OpenImages (eval=test).
            "applicable": bool(args.eval_split == "val" and ms_x is not None and ckpt_ms is not None
                               and args.max_eval_samples is None),
            "checkpoint_val_monoscore": ckpt_ms,
            "recomputed_mean_all": ms_x,
            "abs_diff": (abs(ms_x - ckpt_ms) if (ms_x is not None and ckpt_ms is not None) else None),
        },
    }
    # The MonoScore closed form assumes unit-norm probe rows (it subtracts
    # sum_i a_ij^2 for sum_i a_ij^2 <p_i,p_i>).  F.normalize leaves a zero row as
    # a zero row rather than erroring, and such a row biases the numerator DOWN
    # with no other symptom -- so the diagnostic is recorded per cell instead of
    # being assumed away.
    degenerate = {rep: mono_summary[rep].get("probe_n_degenerate_rows") or 0 for rep in active_reps}
    self_checks["probe_rows_unit_norm"] = {
        "applicable": True,
        "min_row_norm": {rep: mono_summary[rep].get("probe_min_row_norm") for rep in active_reps},
        "n_degenerate_rows": degenerate,
        "ok": bool(sum(degenerate.values()) == 0),
        "note": ("a zero-norm probe row would bias MonoScore's numerator downward; "
                 "the raw features are unit-norm by construction and real r_hat row "
                 "norms are 0.20-0.46, so this is a robustness guard, not a live risk"),
    }

    check = self_checks["monoscore_x_vs_checkpoint"]
    check["ok"] = bool((not check["applicable"]) or (check["abs_diff"] is not None and check["abs_diff"] < 1e-3))
    self_checks["all_ok"] = all(v.get("ok", True) for v in self_checks.values() if isinstance(v, dict))
    if not self_checks["all_ok"]:
        print(f"[WARN] self-checks failed: {json.dumps(self_checks, indent=2, default=str)}")

    # ---- per-representation metrics ---------------------------------------
    results: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "meta": meta,
        "eval_split": args.eval_split,
        "multilabel": multilabel,
        "reps_requested": reps,
        "reps_computed": active_reps,
        "retrieval_modes": modes,
        "knn_k": args.knn_k,
        "retrieval_k": args.retrieval_k,
        "encode_batch_size": args.encode_batch_size,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "n_train": tr.n_rows,
        "n_eval": ev.n_rows,
        "lp": {"epochs": LP_EPOCHS, "batch_size": LP_BATCH_SIZE, "lr": LP_LR,
               "weight_decay": LP_WEIGHT_DECAY, "seed": LP_SEED},
        "model_level": {
            "binary_purity": purity,
            "dict_size": dict_size,
            "train_l0_mean": tr.l0_mean,
            "eval_l0_mean": ev.l0_mean,
            "n_active_latents_eval": int((ev.active_counts > 0).sum().item()),
            "monoscore_den_shared": True,
            # den depends only on z, so n_active_code is the same for every probe
            # space; read it off whichever rep was computed.  active_reps cannot
            # be empty here (guarded above), but stay defensive: this is a
            # bookkeeping field, not worth crashing a finished cell over.
            "n_active_code": (mono_summary[active_reps[0]]["n_active_code"]
                              if active_reps else None),
        },
        "self_checks": self_checks,
        "timings": {"encode_seconds": encode_s,
                    "encode_eval_seconds": ev.seconds,
                    "encode_train_seconds": tr.seconds,
                    "binary_purity_seconds": purity_s,
                    "monoscore_seconds": mono_s},
        "reps": {},
    }

    for rep in reps:
        if rep not in active_reps:
            # Undefined by construction (baseline r_hat).  null, never zeros.
            results["reps"][rep] = None
            continue
        src = ev.canonical.get(rep, rep)
        if src != rep:
            # Single-branch model: xhat IS xM (the same tensor object, see
            # forward_reps).  Recomputing KNN + LP + both k=50 retrievals on an
            # identical bank costs ~1/3 of the per-cell GPU time on every baseline
            # cell and adds exactly zero information -- the consistency check it
            # would nominally buy is already structural and free
            # (self_checks.baseline_xM_is_xhat compares object identity).  The
            # block is copied so the JSON schema is unchanged, with an explicit
            # marker so a reader knows the two columns are one measurement.
            results["reps"][rep] = dict(results["reps"][src],
                                        aliased_from=src,
                                        alias_note="single-branch model: x^M and x_hat are the "
                                                   "same tensor, so this is one measurement "
                                                   "reported under two names")
            print(f"\n== rep={rep}  ALIAS of {src} (single-branch model: x^M IS x_hat)")
            continue
        # Stage this representation's banks on the GPU once and reuse them for
        # KNN, LP and both retrieval variants; done per rep so at most one pair is
        # resident (a SigLIP2 ImageNet train bank is 5.9 GB on its own).
        tb = tr.banks[rep].to(device)
        eb = ev.banks[rep].to(device)
        print(f"\n== rep={rep}  dim={tb.n_cols}")
        timings: Dict[str, float] = {}

        t0 = time.time()
        knn = knn_metric(tb, eb, device, k=args.knn_k,
                         query_batch=args.knn_query_batch, bank_chunk=args.knn_bank_chunk)
        timings["knn_seconds"] = time.time() - t0

        t0 = time.time()
        lp = lp_metric(tb, eb, device)
        timings["lp_seconds"] = time.time() - t0

        block: Dict[str, Any] = {
            "dim": tb.n_cols,
            **ev.r2[rep],
            "monoscore": mono_summary[rep],
            "knn": knn,
            "lp": lp,
        }
        for mode in modes:
            t0 = time.time()
            if mode == "self":
                block["retrieval_purity_self"] = retrieval_purity(
                    eb, eb, device, k=args.retrieval_k, exclude_self=True,
                    query_block=args.retrieval_query_block, db_block=args.retrieval_db_block)
            else:
                block["retrieval_purity_bank"] = retrieval_purity(
                    eb, tb, device, k=args.retrieval_k, exclude_self=False,
                    query_block=args.retrieval_query_block, db_block=args.retrieval_db_block)
            timings[f"retrieval_{mode}_seconds"] = time.time() - t0
        block["timings"] = timings
        results["reps"][rep] = block

        print(f"   R2(perfeat)={_fmt(block['r2_perfeat'])} R2(global)={_fmt(block['r2_global'])} "
              f"MS={_fmt(mono_summary[rep]['mean_all'])} KNN={_fmt(knn['metric'])} LP={_fmt(lp['metric'])} "
              + " ".join(f"ret_{m}={_fmt(block[f'retrieval_purity_{m}']['purity'])}" for m in modes))

        del tb, eb
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["timings"]["total_seconds"] = time.time() - t_start

    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.replace(out_path)  # atomic: a wall-clock kill never leaves half a result
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
