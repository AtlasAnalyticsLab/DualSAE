"""CB-DualBSF: the three-way combination of BSF (vector-block latents),
DualSAE (poly residual branch + span decorrelation), and CB-SAE (user-defined
concept head on a frozen pruned dictionary).

In this release the frozen base is always a b=1 SAE port (bsf/sae_models.py)
of a converted Matryoshka-SAE (CB-SAE arm, use_poly=False) or
Dual-Matryoshka-SAE (CB-DualSAE arm, use_poly=True); with b=1 and b_cb=1 the
math below is exactly the CB-SAE paper's, plus (for the dual arm) the DualSAE
Residual Branch ("poly"). The block-sparse (b>1) generalization is kept
unchanged because the same class implements both.

Composition (stage 4 of the featbsf CB pipeline; stages 1-3 are the base
conversion + dissect + prune in featbsf/):

    x_norm = base.normalize(x)                      # (x - mu) * scale
    xB     = base.decode(base.encode(x_norm).z)     # FROZEN retained blocks
    c      = E_cb(x_norm).view(N, |C|, b_cb)        # dense concept blocks
    x_cb   = sigma_cb(c) @ D_cb                     # gated concept recon
    r_hat  = poly(x_norm - xB - x_cb)               # trainable dense residual
    x_hat  = xB + x_cb + r_hat                      # training reconstruction
    deploy = xB + x_cb                              # steerable reconstruction

The poly branch is TRAINING-ONLY (the Dual deployment contract, mirrored from
DualBSFMixin): steering/eval of the steerable dictionary uses `deploy`. The
span penalty keeps r_hat out of BOTH steerable dictionaries' subspaces — the
retained blocks' frames AND the concept blocks' frames — so polysemantic
residual content cannot re-enter anything a user might steer.

Concept-head geometry — the `b_cb` knob:
  * b_cb > 1 (block concepts, the genuine 3-way synthesis): each user concept
    owns a b_cb-dim SUBSPACE. Codes are SIGNED and gated by block_topk on
    block norms (BSF Sec. 2.3: coordinate nonlinearities would restrict each
    block to a cone instead of its full subspace — no ReLU). The dense
    activation L_int correlates with the pseudo-GT is the block NORM (>= 0;
    sign-blind by construction — the price of block concepts, isolated by the
    b_cb=1 ablation arm).
  * b_cb == 1 (paper-faithful CB-SAE ablation): exactly CBSAEFeaturizer's
    head — sigma_cb = per-sample top-k_cb of ReLU(c), L_int sees the SIGNED
    dense c. Parity with bsf/cb_sae.py math is pinned by
    a unit test in the original repository.

ADAPTATION vs CB-SAE Eq. 6: the paper subtracts the SAE's pre-decoder bias b
(`c = E_cb(v - b)`). BSF has no b_dec; its centering is the mu subtraction
inside base.normalize(). The concept head therefore reads the already
mu-centered, scaled x_norm directly (E_cb has bias=False, as in the paper).

NOT a BSFBase subclass: units have MIXED widths (R blocks of width b plus
|C| concept blocks of width b_cb), so the uniform (G, b) surface would lie to
every consumer. The class instead exposes exactly the granular pieces the
stage-4 trainer (featbsf/cb_train.py) and the eval adapter
(featbsf/adapters.py `_CBDualAdapted`, vision/cb_adapter.py) drive.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bsf.bsf_prune import _family_of, _is_sae_port, _variant_of
from bsf.models import (
    BSFBase,
    DualBSFMixin,
    GrassmannianBSF,
    block_norms,
    block_topk,
    build_bsf,
)

_EPS = 1e-12


class CBDualBSF(nn.Module):
    """Frozen pruned (Dual)BSF blocks + trainable concept head + (optionally)
    trainable poly branch.

    Trainable parameters: E_cb, D_cb, and (when use_poly) base.poly — nothing
    else. The base's blocks are frozen with requires_grad_(False) AND kept in
    eval mode permanently (train() below re-asserts it), but live in the
    state_dict so a CB-DualBSF checkpoint is self-contained.
    """

    def __init__(self, base: BSFBase, n_concepts: int, k_cb: int,
                 b_cb: int = 4, use_poly: bool = True):
        super().__init__()
        if n_concepts <= 0 or k_cb <= 0 or b_cb <= 0:
            raise ValueError(f"n_concepts={n_concepts}, k_cb={k_cb}, "
                             f"b_cb={b_cb} must all be positive")
        if k_cb > n_concepts:
            raise ValueError(f"k_cb={k_cb} > n_concepts={n_concepts}")
        # A Dual base is any class carrying the warm-startable poly branch:
        # the block-sparse DualBSFMixin families, or the b=1 DualSAEFeaturizer
        # carriers of converted vision/main.py DualSAE checkpoints (the poly
        # module is architecturally identical, and
        # SAEFeaturizer.clone_pruned carries its weights just like
        # clone_pruned_bsf does for the mixin).
        from bsf.sae_models import DualSAEFeaturizer
        if use_poly and not isinstance(base, (DualBSFMixin, DualSAEFeaturizer)):
            raise ValueError("use_poly=True requires a Dual base (the poly "
                             "branch is warm-started from the pruned dual's "
                             "own poly weights)")
        if base.has_step_norm:
            # featbsf bases use global (mu, scale) normalization only; the
            # per-step normalization tables of bsf/models.py are not carried
            # by this class (it is not a BSFBase and has no step-norm surface).
            raise ValueError("CB-DualBSF does not support step-normalized "
                             "bases")
        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()
        if use_poly:
            # Re-enable ONLY the poly branch. Its weights arrive warm-started:
            # clone_pruned_bsf copied the stage-1 dual's poly verbatim.
            self.base.poly.requires_grad_(True)

        self.R = int(base.G)                 # retained block count
        self.b = int(base.b)                 # retained block width
        self.d = int(base.d)
        self.n_concepts = int(n_concepts)
        self.k_cb = int(k_cb)
        self.b_cb = int(b_cb)
        self.use_poly = bool(use_poly)
        # lam_span comes from the base dual when present (DualSAE-paper value
        # carried through the prune); the trainer may override per run.
        self.lam_span = float(getattr(base, "lam_span", 0.01))

        # Concept head. Warm start mirrors CBSAEFeaturizer / VanillaBSF:
        # random decoder frames normalized to unit FROBENIUS norm per concept
        # block (at b_cb=1 this is exactly CB-SAE's unit row norm), encoder
        # tied to the decoder so pre-activations start at the right magnitude
        # — which suppresses dead concepts before L_int shapes the encoder.
        self.E_cb = nn.Linear(self.d, n_concepts * b_cb, bias=False)
        self.D_cb = nn.Parameter(torch.randn(n_concepts, b_cb, self.d))
        with torch.no_grad():
            self.D_cb.div_(self.D_cb.norm(dim=(1, 2), keepdim=True)
                           .clamp_min(_EPS))
            self.E_cb.weight.copy_(self.D_cb.reshape(n_concepts * b_cb,
                                                     self.d))

        self._refresh_base_frames()

    # -- base plumbing --------------------------------------------------------
    def _refresh_base_frames(self) -> None:
        """Pin the frozen base's decoder frames.

        For a Grassmannian base the usable frames are QR(B_raw), recomputed on
        every decoder_atoms() call — but B_raw is FROZEN here, so the QR result
        is a constant. Setting the base's forward-cache slot permanently makes
        every encode()/decode()/span read one precomputed QR instead of one
        per call. The cache is a plain attribute (never serialized), so
        load_cb_dual() re-calls this after load_state_dict fills B_raw.
        MUST be re-called if the base is ever unfrozen (it is not, anywhere).
        """
        if isinstance(self.base, GrassmannianBSF):
            self.base._atoms_cache = self.base._orthonormal_atoms().detach()

    def train(self, mode: bool = True):
        # nn.Module.train() recurses into children; the frozen base must stay
        # in eval mode no matter what (its gates are mode-switched — a
        # training-mode BatchBlockTopK forward would batch-gate and move the
        # EMA threshold buffer). encode() is mode-independent anyway; this is
        # belt-and-braces so no future code path can regress it.
        super().train(mode)
        self.base.eval()
        return self

    def _apply(self, fn, *args, **kwargs):
        # .to()/.cuda()/.float() move parameters and buffers but NOT plain
        # tensor attributes — the pinned Grassmannian frame cache
        # (base._atoms_cache) would be left on the old device/dtype while
        # decoder_atoms() keeps returning it unconditionally, crashing the
        # first cross-device einsum. Recompute the pin from the freshly-moved
        # B_raw after every apply.
        out = super()._apply(fn, *args, **kwargs)
        self._refresh_base_frames()
        return out

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.normalize(x)

    def denormalize(self, x_hat: torch.Tensor) -> torch.Tensor:
        # Delegate: the base owns the (possibly non-affine, e.g. per-sample
        # LN) preprocess, so it owns the only correct inversion too.
        return self.base.denormalize(x_hat)

    @property
    def mu(self) -> torch.Tensor:
        return self.base.mu

    @property
    def scale(self) -> torch.Tensor:
        return self.base.scale

    # -- concept head pieces (the trainer's building blocks) ------------------
    def concept_pre(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Dense signed concept blocks, (N, |C|, b_cb) — no gate (Eq. 6)."""
        return self.E_cb(x_norm).view(-1, self.n_concepts, self.b_cb)

    def concept_dense_acts(self, x_norm: torch.Tensor) -> torch.Tensor:
        """What L_int / L_st correlate with the pseudo-GT y — (N, |C|), dense
        (deliberately k-invariant, CB-SAE App. C.3). b_cb=1 keeps the SIGNED
        scalar (CB-SAE parity); b_cb>1 uses the block norm (>= 0)."""
        c = self.concept_pre(x_norm)
        if self.b_cb == 1:
            return c[..., 0]
        return block_norms(c)

    def sigma_cb(self, c_pre: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decoder-path gate -> (z_cb (N, |C|, b_cb), sel_cb (N, |C|) bool).

        b_cb=1: ReLU then per-sample top-k_cb — bit-identical math to
        CBSAEFeaturizer.sigma_cb (adaptation A7 there). b_cb>1: per-sample
        block top-k_cb on block norms, blocks stay SIGNED (BSF convention).
        """
        if self.b_cb == 1:
            pos = F.relu(c_pre[..., 0])                       # (N, |C|)
            if self.k_cb >= pos.shape[1]:
                z = pos
            else:
                vals, idx = torch.topk(pos, self.k_cb, dim=1)
                z = torch.zeros_like(pos).scatter(1, idx, vals)
            return z.unsqueeze(-1), z > 0
        if self.k_cb >= c_pre.shape[1]:
            # Degenerate budget: every block passes (block_topk would raise).
            sel = block_norms(c_pre) > 0
            return c_pre * sel.unsqueeze(-1), sel
        return block_topk(c_pre, self.k_cb)

    def cb_recon(self, z_cb: torch.Tensor) -> torch.Tensor:
        """Concept branch of the reconstruction: sum_j z_j D_cb[j], in the
        base's NORMALIZED space."""
        return torch.einsum("ncb,cbd->nd", z_cb, self.D_cb)

    # -- reconstructions ------------------------------------------------------
    @torch.no_grad()
    def base_recon(self, x_norm: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """Frozen retained-blocks reconstruction (xB', sel_base). ALWAYS the
        side-effect-free deployment gate (base.encode), so the CB head
        complements the base exactly as it will be deployed."""
        _, z, sel = self.base.encode(x_norm)
        xB = self.base.decode(z)
        # SAE-port bases reconstruct decode(z) + b_dec (the SAE centering
        # bias). Converted bases normally FOLD b_dec into mu (making this a
        # zero no-op); the input_unit_norm (LN) carrier cannot fold — its mu
        # is unused — so the real b_dec must enter here, or xB (and the poly
        # residual below) would be off by the centering bias. BSF bases have
        # no b_dec attribute: unchanged.
        b_dec = getattr(self.base, "b_dec", None)
        if b_dec is not None:
            xB = xB + b_dec
        return xB, sel

    def deployment_recon(self, x_norm: torch.Tensor) -> torch.Tensor:
        """The STEERABLE reconstruction xB' + x_cb — what steering/eval of the
        dictionary sees; the poly branch never contributes here (Dual
        deployment contract)."""
        xB, _ = self.base_recon(x_norm)
        z_cb, _ = self.sigma_cb(self.concept_pre(x_norm))
        return xB + self.cb_recon(z_cb)

    def full_recon(self, x_norm: torch.Tensor) -> dict:
        """Training reconstruction with all parts exposed:
        {xB, sel_base, z_cb, sel_cb, x_cb, r_hat, x_hat}. r_hat is None when
        use_poly is False (then x_hat == the deployment recon)."""
        xB, sel_base = self.base_recon(x_norm)
        c = self.concept_pre(x_norm)
        z_cb, sel_cb = self.sigma_cb(c)
        x_cb = self.cb_recon(z_cb)
        r_hat = None
        x_hat = xB + x_cb
        if self.use_poly:
            # Residual AFTER both steerable branches — the poly mops up only
            # what neither the retained blocks nor the concepts explain. No
            # stop-grad (DualBSF parity): the concept head receives gradient
            # through r as well.
            r_hat = self.base.poly(x_norm - xB - x_cb)
            x_hat = x_hat + r_hat
        return {"xB": xB, "sel_base": sel_base, "z_cb": z_cb,
                "sel_cb": sel_cb, "x_cb": x_cb, "r_hat": r_hat,
                "x_hat": x_hat}

    def span_penalty(self, r_hat: torch.Tensor, sel_base: torch.Tensor,
                     sel_cb: torch.Tensor) -> torch.Tensor:
        """Mixed-width generalization of DualBSFMixin._span_penalty: per
        sample, the mean over ALL ACTIVE steerable blocks — retained (width b)
        and concept (width b_cb) alike — of ||D_g r_hat||^2, batch-averaged.
        With |C|=0 active concepts this reduces bitwise to the mixin formula
        (pinned by a unit test in the original repository)."""
        o_ret = torch.einsum("nd,gbd->ngb", r_hat,
                             self.base.decoder_atoms())          # (N, R, b)
        sq_ret = o_ret.pow(2).sum(-1) * sel_base.to(r_hat.dtype)  # (N, R)
        o_cb = torch.einsum("nd,cbd->ncb", r_hat, self.D_cb)     # (N,|C|,b_cb)
        sq_cb = o_cb.pow(2).sum(-1) * sel_cb.to(r_hat.dtype)     # (N, |C|)
        denom = (sel_base.to(r_hat.dtype).sum(dim=1)
                 + sel_cb.to(r_hat.dtype).sum(dim=1)).clamp_min(1.0)
        return ((sq_ret.sum(dim=1) + sq_cb.sum(dim=1)) / denom).mean()

    # -- eval surface (featbsf adapter contract) ------------------------------
    @torch.no_grad()
    def unit_acts(self, x_norm: torch.Tensor) -> torch.Tensor:
        """(N, R + |C|) per-unit activations of the GATED code: retained-block
        norms ++ concept-block norms. Nonnegative, exactly 0 iff the unit did
        not fire — the invariant every '>0' convention in the metrics
        relies on. (At b_cb=1 the concept norm equals the ReLU-top-k value.)
        """
        _, z, _ = self.base.encode(x_norm)
        z_cb, _ = self.sigma_cb(self.concept_pre(x_norm))
        return torch.cat([block_norms(z), block_norms(z_cb)], dim=1)

    def dims_per_unit_vector(self):
        """Per-unit dictionary widths, [b]*R + [b_cb]*|C| — the honest
        realized-L0-in-dims weighting for mixed-width units."""
        import numpy as np
        return np.asarray([self.b] * self.R
                          + [self.b_cb] * self.n_concepts, dtype=np.int64)

    def forward(self, x: torch.Tensor) -> dict:
        """Eval/test convenience: normalize + full_recon + plain MSE. The
        alternating A/B/C training logic lives in featbsf/cb_train.py, which
        drives the granular pieces so each objective updates exactly its
        parameters."""
        x_norm = self.normalize(x)
        parts = self.full_recon(x_norm)
        parts["recon"] = F.mse_loss(parts["x_hat"], x_norm)
        return parts

    # -- self-description ------------------------------------------------------
    def variant_name(self) -> str:
        return "cb_dual_bsf"

    def extra_config(self) -> dict:
        """Everything load_cb_dual needs to rebuild the bare base + head."""
        base = self.base
        return {
            "base_variant": _variant_of(base),
            "base_family": _family_of(base),
            "base_G": int(base.G), "base_b": int(base.b),
            "base_l": int(base.l), "aux_alpha": float(base.aux_alpha),
            "gl_coef": getattr(base, "coef", None),
            "gl_gain": getattr(base, "gain", None),
            "gl_paper_version": getattr(base, "paper_version", None),
            "poly_hidden": getattr(base, "_poly_hidden", 0),
            "lam_span": getattr(base, "lam_span", 0.01),
            "dual_warmup_steps": getattr(base, "dual_warmup_steps", 0),
            # Bounded-score gate flag: the pruned base must gate at CB deployment
            # exactly as it was trained (bounded scores vs raw norms).
            "base_bounded_scores": bool(getattr(base, "bounded_scores",
                                                False)),
            "n_concepts": self.n_concepts, "k_cb": self.k_cb,
            "b_cb": self.b_cb, "use_poly": self.use_poly,
            # SAE-port bases (b=1 carriers of converted vision/main.py SAEs)
            # carry their OWN self-description — mono mode, soft cap, LN
            # flag, deploy-gate flag: everything build_sae needs to rebuild
            # the exact class before load_state_dict fills the weights.
            # None for BSF-family bases (build_bsf covers them), which also
            # keeps every pre-existing payload byte-compatible.
            "base_sae_cfg": (base.extra_config()
                             if _is_sae_port(base) else None),
        }


# ---------------------------------------------------------------------------
# Checkpoint I/O. Deliberately NOT registered in bsf.models.load_bsf: that
# dispatch chain is frozen (unknown variants fall through to the SAE builder,
# which would misparse this payload). featbsf/adapters.py branches on
# payload["variant"] == "cb_dual_bsf" explicitly instead.
# ---------------------------------------------------------------------------
def save_cb_dual(model: CBDualBSF, path, extra: dict | None = None) -> None:
    """Self-describing checkpoint (save_bsf idiom). state_dict includes the
    frozen base.* weights, so the file is loadable without the pruned parent
    checkpoint present."""
    payload = {
        "variant": "cb_dual_bsf",
        "d": model.d,
        "extra_config": model.extra_config(),
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_cb_dual(path, device: str = "cpu") -> CBDualBSF:
    """Rebuild a save_cb_dual checkpoint: bare base via build_bsf from the
    recorded config, CBDualBSF around it, then load_state_dict fills every
    weight (incl. the frozen base.*). Returned .eval() on `device`."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("variant") != "cb_dual_bsf":
        raise ValueError(f"{path}: not a cb_dual_bsf checkpoint "
                         f"(variant={payload.get('variant')!r})")
    cfg = payload["extra_config"]
    sae_cfg = cfg.get("base_sae_cfg")
    if sae_cfg is not None:
        # SAE-port base (b=1 converted scalar dual): rebuild through
        # build_sae with the base's own recorded config — build_bsf's
        # registry does not know these variants (its VARIANTS lookup would
        # KeyError) and could not reconstruct the mono-mode state anyway.
        from bsf.sae_models import build_sae
        kw = {k: v for k, v in sae_cfg.items()
              if k not in ("n_latent", "k", "sparsity_mode", "mono_mode",
                           "dual", "weighting")}
        base = build_sae(cfg["base_variant"], d=payload["d"],
                         n_latent=cfg["base_G"], k=cfg["base_l"], **kw)
    else:
        base = build_bsf(
            cfg["base_variant"], d=payload["d"], G=cfg["base_G"],
            b=cfg["base_b"], l=cfg["base_l"], aux_alpha=cfg.get("aux_alpha"),
            coef=cfg["gl_coef"] if cfg.get("gl_coef") is not None else 1e-2,
            gain=cfg["gl_gain"] if cfg.get("gl_gain") is not None else 10.0,
            paper_version=bool(cfg.get("gl_paper_version")),
            poly_hidden=cfg.get("poly_hidden", 0),
            lam_span=cfg.get("lam_span", 0.01),
        )
    if isinstance(base, DualBSFMixin):
        base.dual_warmup_steps = cfg.get("dual_warmup_steps", 0)
    # Restore the bounded-gate flag BEFORE wrapping (absent in older
    # payloads -> False = official gate, backward-compatible).
    base.bounded_scores = bool(cfg.get("base_bounded_scores", False))
    model = CBDualBSF(base, cfg["n_concepts"], cfg["k_cb"],
                      b_cb=cfg.get("b_cb", 1),
                      use_poly=bool(cfg.get("use_poly", True)))
    model.load_state_dict(payload["state_dict"])
    # Move FIRST, then re-pin the Grassmannian frame cache: __init__ pinned it
    # from the random-init B_raw, load_state_dict replaced the weights, and
    # .to(device) (via _apply above) recomputes it from the loaded, moved
    # B_raw — so the cache is correct in both value and device. The explicit
    # refresh after .eval() is belt-and-braces for a CPU-target load, where
    # _apply may not run at all.
    model = model.to(device).eval()
    model._refresh_base_frames()
    return model
