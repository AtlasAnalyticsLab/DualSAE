"""Cosine-cubed similarity losses of the Concept-Bottleneck SAE (CB-SAE,
Kulkarni et al., arXiv 2512.10805, Eqs. 9-11; the similarity itself follows
Label-free CBM).

Only the two loss functions the CB stage-4 trainer (featbsf/cb_train.py)
imports are kept here; the rest of the original module (an image-generation
substrate CB-SAE featurizer) is not part of this release. Function bodies are
copied verbatim from the code that produced the reported runs.
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Cosine-cubed similarity (paper Eq. 11, following Label-Free CBM)
# ---------------------------------------------------------------------------
def cosine_cubed_similarities(c: torch.Tensor, y: torch.Tensor,
                              center: bool = True,
                              eps: float = 1e-6) -> torch.Tensor:
    """Per-concept cubed-cosine similarity over the batch dimension.

    c: (N, K) concept predictions, y: (N, K) pseudo-GT concept activations.
    Returns (K,) similarities in [-1, 1]; the training loss is their NEGATIVE
    mean (maximize similarity).

    center=True follows the OFFICIAL Label-free-CBM implementation
    (cos_similarity_cubed_single in the Label-free-CBM repository:
    mean-center each concept column over the batch -> cube element-wise ->
    L2-normalize per column -> dot). The paper's Eq. 11 does not show the
    centering, but its text defers to "following [35]" (= LF-CBM), and this
    port follows the official implementation; center=False
    gives the literal Eq. 11 form for the ablation flag. (The paper's
    "normalize before raising to the third power" is numerically inert:
    cosine is scale-invariant and cubing commutes with positive scaling, so
    a pre-cube L2 normalization cannot change the result — it is subsumed by
    the post-cube normalization below.)

    Deviation from the reference: the zero-norm guard (clamp_min(eps)) —
    a constant concept column (e.g. a concept absent from every image in the
    batch) has zero centered norm and would produce NaN; the official single
    variant has no guard because LF-CBM filters such concepts upstream.
    Parity with the reference on non-degenerate inputs is pinned by
    a unit test in the original repository.
    """
    c = c.float()
    y = y.float()
    if center:
        c = c - c.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
    c = c ** 3
    y = y ** 3
    c = c / c.norm(p=2, dim=0, keepdim=True).clamp_min(eps)
    y = y / y.norm(p=2, dim=0, keepdim=True).clamp_min(eps)
    return (c * y).sum(dim=0)


def cosine_cubed_loss(c: torch.Tensor, y: torch.Tensor,
                      center: bool = True) -> torch.Tensor:
    """L_int / L_st (Eqs. 9-10): negative similarity, MEAN over concepts.
    Eq. 11 writes a sum; with Adam the constant |C| factor only rescales the
    raw gradient magnitude (renormalized away by the second moment), and the
    mean keeps the loss scale comparable across concept-set sizes."""
    return -cosine_cubed_similarities(c, y, center=center).mean()
