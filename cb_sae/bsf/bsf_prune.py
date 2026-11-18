"""Block-level pruning for BSF / DualBSF models (CB-SAE Step 3, Eqs. 3-4,
generalized from b=1 neurons to b-dim blocks).

CB-SAE prunes a trained dictionary by DELETING encoder rows / decoder atoms
of the discarded units and keeping everything else byte-identical
(SAEFeaturizer.clone_pruned for the b=1 SAE families). This module is the BSF analogue: `keep` indexes
BLOCKS, and each family's per-block state is sliced while the global state
(normalization, thresholds, gamma, the dual poly branch) is copied whole.

Exactness contract (the block version of sae_models.PRUNE_EXACT — see
bsf/prune_retention.py):

  * batch_blocktopk : EXACT. Deploy gate is `||z_g|| > threshold` with a
                      GLOBAL scalar threshold — each block's decision is
                      independent of every other block, so on identical
                      pre-codes the pruned gate reproduces the kept blocks
                      bit-for-bit. (End-to-end codes agree only to matmul
                      reduction-order rounding ~1e-6: the pruned encoder's
                      (N,d)@(d,R*b) matmul blocks differently in BLAS than
                      the full (N,d)@(d,G*b) one. Only while `inited` is
                      True; an unseeded threshold falls back to per-sample
                      top-l, the inexact case — bsf_prune_exact() checks.)
  * grouplasso      : EXACT, both variants. The JumpReLU gate uses per-block
                      thresholds theta_g (sliced), the paper soft-threshold a
                      scalar theta (copied); neither looks at other blocks.
  * vanilla /
    grassmannian    : INEXACT. The per-sample top-l gate re-ranks over the
                      retained pool: deleting G-R competitors frees slots and
                      PROMOTES blocks the full model did not select. Measure
                      the damage with block_retention_stats() and record it —
                      dissect labels (I_m) computed on the full model dilute
                      in proportion to the promoted fraction.

Duals prune exactly like their plain family: the poly branch is a dense
d->d MLP with no per-block state, so it is copied verbatim — which is also
what gives CB-DualBSF stage 4 its poly warm start for free.

Deliberately a free-function dispatch (NOT methods on the frozen classes in
bsf/models.py): the four families' definitions stay textually untouched, and
this module can be imported from featbsf-land without pulling anything beyond
bsf.models (+ bsf.prune_retention).
"""
from __future__ import annotations

import torch

from bsf.models import (
    BatchBlockTopKBSF,
    BSFBase,
    DualBSFMixin,
    GrassmannianBSF,
    GroupLassoBSF,
    VanillaBSF,
    block_norms,
    build_bsf,
)
from bsf.prune_retention import retention_stats

# Static family -> gate-independence table (duals inherit their family's
# entry). Use bsf_prune_exact(model) for the model-aware answer — it also
# catches the one dynamic case (an unseeded batch_blocktopk threshold).
BSF_PRUNE_EXACT = {
    "batch_blocktopk": True,
    "grouplasso": True,
    "vanilla": False,
    "grassmannian": False,
}


def _is_sae_port(model) -> bool:
    """b=1 SAE-featurizer bases (bsf/sae_models.py), incl. the converted
    vision/main.py DualSAE carriers. Lazy import: sae_models imports from models,
    which bsf_prune sits beside — the established cycle-break idiom."""
    from bsf.sae_models import SAEFeaturizer
    return isinstance(model, SAEFeaturizer)


def _family_of(model: BSFBase) -> str:
    """Plain family name ('vanilla', 'grassmannian', 'grouplasso',
    'batch_blocktopk') for a plain or dual BSF instance; for b=1 SAE-port
    instances, the PLAIN pair-mate's variant name (e.g. 'topk_sae' for a
    dual_topk_sae) — descriptive only, never used to construct a model."""
    if _is_sae_port(model):
        from bsf.sae_models import DualSAEFeaturizer
        if isinstance(model, DualSAEFeaturizer):
            return f"{model.mono_mode.replace('_', '')}_sae"
        return model.variant_name()
    if isinstance(model, DualBSFMixin):
        return model._family
    # Order matters: BatchBlockTopKBSF subclasses VanillaBSF, and
    # (hypothetically) further subclasses should resolve to the most
    # derived family — hence isinstance checks most-derived-first.
    if isinstance(model, BatchBlockTopKBSF):
        return "batch_blocktopk"
    if isinstance(model, GroupLassoBSF):
        return "grouplasso"
    if isinstance(model, GrassmannianBSF):
        return "grassmannian"
    if isinstance(model, VanillaBSF):
        return "vanilla"
    raise TypeError(f"not a BSF family instance: {type(model).__name__}")


def _variant_of(model: BSFBase) -> str:
    """build_bsf/save_bsf variant string ('vanilla', 'dual_grouplasso', ...);
    for SAE-port instances the sae_models registry name (variant_name()),
    which build_from_payload rebuilds — the property load_cb_dual relies on."""
    if _is_sae_port(model):
        return model.variant_name()
    fam = _family_of(model)
    return f"dual_{fam}" if isinstance(model, DualBSFMixin) else fam


def bsf_prune_exact(model: BSFBase) -> bool:
    """Is pruning THIS model code-exact (retained blocks keep bit-identical
    deployment codes)? Family table + the one dynamic exception: a
    batch_blocktopk whose EMA threshold was never seeded deploys the
    per-sample top-l fallback gate, which re-ranks like vanilla."""
    if _is_sae_port(model):
        # The SAE classes carry their own per-gate answer (a per-mode property
        # on the dual port); the EMA-threshold caveat is the same as
        # batch_blocktopk's: an unseeded threshold (0.0) gates dense, and a
        # dense gate is trivially per-neuron independent — no downgrade.
        return bool(model.PRUNE_EXACT)
    fam = _family_of(model)
    if fam == "batch_blocktopk" and not bool(model.inited):
        return False
    return BSF_PRUNE_EXACT[fam]


def _check_keep(keep: torch.Tensor, G: int, l: int) -> torch.Tensor:
    """Validate the ascending-keep contract (a permuted/duplicated keep would
    silently scramble the block-id <-> retained-block mapping recorded in
    prune.json) and return it as a CPU long tensor."""
    keep = torch.as_tensor(keep, dtype=torch.long).cpu()
    if keep.dim() != 1 or keep.numel() == 0:
        raise ValueError(f"keep must be a non-empty 1-D index tensor, "
                         f"got shape {tuple(keep.shape)}")
    if keep.numel() > 1 and not bool((keep[1:] > keep[:-1]).all()):
        raise ValueError("keep must be STRICTLY ascending (the block-id "
                         "contract; sort + dedupe before calling)")
    if int(keep[0]) < 0 or int(keep[-1]) >= G:
        raise ValueError(f"keep indices out of range [0, {G})")
    if keep.numel() < l:
        # A top-l gate cannot select l blocks from fewer than l — torch.topk
        # would raise deep inside encode() at deploy time; fail loudly now.
        raise ValueError(f"R={keep.numel()} retained blocks < l={l}")
    return keep


@torch.no_grad()
def clone_pruned_bsf(model: BSFBase, keep: torch.Tensor) -> BSFBase:
    """Materialize the retained-blocks model: same class, G=R, per-block state
    sliced to `keep`, global state copied whole. Returns an .eval() model on
    the source model's device; the source is untouched.

    SAE-port bases (b=1) delegate to SAEFeaturizer.clone_pruned, which
    implements the identical strictly-ascending-keep contract and — via the
    dual port's _copy_pruned_extra — the same poly warm-start carry this
    function performs for DualBSF bases.

    The clone is constructed through the public build_bsf() with the source's
    own hyperparameters, so `save_bsf(clone)` reloads through the existing
    `load_bsf` with ZERO changes to checkpoint I/O (G is a payload field; all
    sliced tensors are shape-consistent at the smaller G).

    Per-family state map (what is sliced vs copied):
      all families      mu, scale buffers -> copied; l, aux_alpha preserved;
                        step-norm tables (if any) carried over.
      vanilla-family    enc.weight rows: block g owns the CONTIGUOUS rows
      (vanilla, batch_  [g*b, (g+1)*b) (enc output is reshaped (N, G, b) in
      blocktopk,        _pre_code, verified by VanillaBSF.reinit_dead's
      grouplasso)       repeat_interleave layout) -> row-sliced; enc.bias
                        same rows; D[keep].
      grassmannian      B_raw[keep] (the QR frames are derived, so slicing
                        the free parameter is sufficient — retained frames
                        stay exactly orthonormal); log_gamma copied.
      grouplasso        + raw_theta[keep] and the STE bandwidth/inited buffers
                        (default variant) or the scalar log_theta (paper
                        variant). Buffers must be COPIED, not re-seeded: a
                        fresh `inited=False` would let a later training-mode
                        forward re-estimate the bandwidth on pruned norms and
                        silently change the STE window.
      batch_blocktopk   + threshold/inited buffers (global scalar — the very
                        thing that makes this family prune-EXACT).
      dual_*            + poly.state_dict() verbatim (dense d->d MLP, no
                        per-block state — this IS the CB-DualBSF stage-4 poly
                        warm start), lam_span/_poly_hidden via build_bsf,
                        detach_poly/dual_warmup_steps attributes carried for
                        provenance.
    """
    if _is_sae_port(model):
        keep = _check_keep(keep, model.G, model.l)
        return model.clone_pruned(keep)

    fam = _family_of(model)
    variant = _variant_of(model)
    keep = _check_keep(keep, model.G, model.l)
    R = int(keep.numel())
    device = next(model.parameters()).device

    new = build_bsf(
        variant, d=model.d, G=R, b=model.b, l=model.l,
        aux_alpha=model.aux_alpha,
        # GroupLasso knobs (ignored by the other families' build paths).
        coef=getattr(model, "coef", 1e-2),
        gain=getattr(model, "gain", 10.0),
        paper_version=bool(getattr(model, "paper_version", False)),
        # Dual knobs (ignored for plain variants).
        poly_hidden=getattr(model, "_poly_hidden", 0),
        lam_span=getattr(model, "lam_span", 0.01),
    )

    # -- global normalization (identical for every family) --------------------
    new.mu.copy_(model.mu.cpu())
    new.scale.copy_(model.scale.cpu())
    if model.has_step_norm:
        new.set_step_normalization(model.step_ids.tolist(),
                                   model.mu_steps.cpu().clone(),
                                   model.scale_steps.cpu().clone())

    # -- per-family sliced/copied state ---------------------------------------
    if fam == "grassmannian":
        new.B_raw.copy_(model.B_raw[keep].cpu())
        new.log_gamma.copy_(model.log_gamma.cpu())
    else:
        # Free-encoder families: block g's b encoder rows are contiguous.
        b = model.b
        row_idx = (keep.unsqueeze(1) * b
                   + torch.arange(b, dtype=torch.long)).reshape(-1)  # (R*b,)
        new.enc.weight.copy_(model.enc.weight[row_idx].cpu())
        new.enc.bias.copy_(model.enc.bias[row_idx].cpu())
        new.D.copy_(model.D[keep].cpu())
        if fam == "grouplasso":
            if model.paper_version:
                new.log_theta.copy_(model.log_theta.cpu())
            else:
                new.raw_theta.copy_(model.raw_theta[keep].cpu())
                new.bandwidth.copy_(model.bandwidth.cpu())
                new.inited.copy_(model.inited.cpu())
        elif fam == "batch_blocktopk":
            new.threshold.copy_(model.threshold.cpu())
            new.inited.copy_(model.inited.cpu())

    # -- dual poly branch (copied whole; block-count independent) -------------
    if isinstance(model, DualBSFMixin):
        new.poly.load_state_dict(model.poly.state_dict())
        new.detach_poly = bool(getattr(model, "detach_poly", False))
        new.dual_warmup_steps = int(getattr(model, "dual_warmup_steps", 0))

    # Bounded-score gate flag: build_bsf constructs with the default (False), so the
    # source's bounded-scores gate must be carried explicitly — a pruned model
    # must select blocks the way its parent was trained to. (The bounded score
    # divides by per-block encoder norms, which slice cleanly with the blocks:
    # retained blocks' scores are unchanged by pruning.)
    new.bounded_scores = bool(getattr(model, "bounded_scores", False))
    return new.to(device).eval()


@torch.no_grad()
def block_retention_stats(model: BSFBase, x_norm: torch.Tensor,
                          keep: torch.Tensor) -> dict:
    """Prune-fidelity numbers for prune.json, on a batch of NORMALIZED inputs.

    Exact families return the trivial marker (retention 1.0 by gate
    independence — not simulated, stated). Inexact families (per-sample
    top-l) reuse retention_stats() on BLOCK NORMS: block norms are >= 0 and
    the top-l simulation inside retention_stats is exactly block_topk's
    selection rule, so `pre_z=block_norms(...)` + `k=l` measures precisely
    how many of the full model's selected blocks survive the prune vs how
    many the pruned model newly promotes.
    """
    keep = _check_keep(keep, model.G, model.l)
    if bsf_prune_exact(model):
        gn = block_norms(model._pre_code(x_norm))
        return {
            "prune_exact": True,
            "retention": 1.0,
            "promoted_per_token": 0.0,
            # Context stat kept for parity with the inexact branch.
            "dead_in_batch": float(
                (~(gn[:, keep] > 0).any(0)).float().mean()),
            "k": int(model.l), "G": int(model.G),
            "R": int(keep.numel()), "n_rows": int(x_norm.shape[0]),
        }
    pre_z = model._pre_code(x_norm)
    # Rank by the model's ACTUAL selection statistic: raw block norms for the
    # official gate, bounded scores under the bounded-score flag (scores are per-
    # block functions of the block's own weights, so slicing keeps retained
    # blocks' scores unchanged — the simulation stays exact w.r.t. the rule).
    if getattr(model, "bounded_scores", False):
        gn = model._selection_scores(pre_z)                      # (N, G) >= 0
    else:
        gn = block_norms(pre_z)                                  # (N, G) >= 0
    st = retention_stats(gn, keep, model.l)
    st["prune_exact"] = False
    return st
