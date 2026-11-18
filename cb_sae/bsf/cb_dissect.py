"""CLIP-Dissect interpretability scoring for SAE / CB-SAE neurons
(CB-SAE paper Sec. 3 "Measuring SAE interpretability" + App. B).

Per neuron m, CLIP-Dissect compares the neuron's activation profile over a
probing image set against every concept's CLIP image-text similarity profile
via soft-WPMI (Oikarinen & Weng, ICLR'23). The neuron's interpretability
score I_m = max_c sim(m, c) and its assigned concept c_m = argmax_c.

Faithfulness: soft_wpmi below is a vectorized re-implementation of the
OFFICIAL CLIP-Dissect similarity.py with the official defaults (softmax
temperature a=10, top_k=100, p 0.998 -> 0.97, lam=1); numerical parity with
the official code was unit-tested in the original repository.

Only the four helpers the feature-space CB pipeline imports
(featbsf/cb_dissect.py, featbsf/cb_prune.py) are kept; they are copied
verbatim from the code that produced the reported runs. Probing-set
construction and embedding caching for the feature-space pipeline live in
featbsf/cb_dissect.py and featbsf/cb_text_embeds.py.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_vocab(path: Path) -> list[str]:
    """One concept per line; blank lines dropped, order preserved (concept
    index = line rank, the contract concepts.json relies on)."""
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# soft-WPMI (official CLIP-dissect, vectorized)
# ---------------------------------------------------------------------------
@torch.no_grad()
def soft_wpmi(clip_feats: torch.Tensor, target_feats: torch.Tensor,
              top_k: int = 100, a: float = 10.0, lam: float = 1.0,
              min_prob: float = 1e-7, p_start: float = 0.998,
              p_end: float = 0.97, chunk: int = 256) -> torch.Tensor:
    """(n_img, n_concepts) image-concept sims x (n_img, n_neurons) activations
    -> (n_neurons, n_concepts) soft-WPMI. Chunked over neurons instead of the
    reference's python loop; identical math (parity unit-tested).

    clip_feats here are RAW image-text sims — the softmax over concepts (the
    reference's first line) happens inside, as in the reference."""
    dev = clip_feats.device
    n_img, n_neuron = target_feats.shape[0], target_feats.shape[1]
    top_k = min(top_k, n_img)
    probs = torch.softmax(a * clip_feats.float(), dim=1)          # (I, C)
    inds = torch.topk(target_feats.float(), dim=0, k=top_k)[1]    # (K, N)
    p_in = (p_start - torch.arange(top_k, device=dev) / top_k
            * (p_start - p_end)).unsqueeze(1)                     # (K, 1)
    rows = []
    for s in range(0, n_neuron, chunk):
        sel = probs[inds[:, s:s + chunk]]                         # (K, n, C)
        # log p(d | E_m) = sum_k log(1 + p_k (P[img_k, d] - 1))
        rows.append(torch.log(1 + p_in.unsqueeze(1) * (sel - 1)
                              + min_prob).sum(dim=0))             # (n, C)
    log_p_d_given_e = torch.cat(rows, dim=0)                      # (N, C)
    # p(d): marginal over the NEURON population's evidence sets (the official
    # estimator: logsumexp over neurons minus log N).
    log_p_d = (torch.logsumexp(log_p_d_given_e, dim=0, keepdim=True)
               - torch.log(torch.tensor([float(n_neuron)], device=dev)))
    return log_p_d_given_e - lam * log_p_d


def dissect(img_scores: np.ndarray, img_embeds: np.ndarray,
            text_embeds: np.ndarray, device: str = "cuda",
            a: float = 10.0, img_encoder: str = "", text_encoder: str = "",
            ) -> dict[str, np.ndarray]:
    """Full dissect of one neuron population against one vocab.
    Returns raw wpmi max ('wpmi'), min-max normalized score in [0,1]
    ('score' — the paper treats I as [0,1]^omega; the normalization constant
    is not specified there, so min-max over the scored population is recorded
    as decision D-NORM), and argmax concept index ('concept_idx').

    `a` defaults to CLIP-Dissect's 10.0 (the value used for every reported
    run; featbsf/cb_dissect.py passes it explicitly).
    """
    # Both operands must come from the SAME joint embedding space. Two
    # different encoders of the same width do not raise on a shape check --
    # their inner product silently sits at the 1/sqrt(d) noise floor. Only
    # provenance can catch that, which is why the caller passes the encoder
    # tags.
    assert img_encoder == text_encoder, (
        f"embedding-space mismatch: images from {img_encoder!r} but text from "
        f"{text_encoder!r}. Their inner product is meaningless (same width, "
        f"so nothing else will complain). Rebuild the text embeds with "
        f"the image encoder before dissecting.")
    P = (torch.from_numpy(img_embeds.astype(np.float32)).to(device)
         @ torch.from_numpy(text_embeds.astype(np.float32)).to(device).T)
    acts = torch.from_numpy(img_scores.astype(np.float32)).to(device)
    mi = soft_wpmi(P, acts, a=a)                                  # (N, C)
    best, idx = mi.max(dim=1)
    lo, hi = best.min(), best.max()
    score = (best - lo) / (hi - lo).clamp_min(1e-9)
    return {"wpmi": best.cpu().numpy(), "score": score.cpu().numpy(),
            "concept_idx": idx.cpu().numpy()}
