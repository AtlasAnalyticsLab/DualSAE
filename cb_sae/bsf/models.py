"""Block-Sparse Featurizers (BSF) — faithful reimplementation of Fel et al.
(arXiv 2606.25234), Eq. 4 + Appendix D. Every design choice below cites the
paper text it implements; documented adaptations are marked with "ADAPTATION".

In this release the module is the shared featurizer base the CB-SAE pipeline
builds on: the b=1 SAE ports (bsf/sae_models.py) subclass BSFBase and are
saved/loaded through save_bsf/load_bsf. The block-sparse families below are
kept unchanged because the checkpoint I/O and pruning dispatch reference them.

Shared skeleton (App. D): a linear encoder producing *signed* block codes (no
ReLU anywhere — coordinate-wise nonlinearities would restrict each block to a
cone instead of its full subspace, Sec. 2.3), a linear decoder x_hat = z @ D,
and inputs scaled so that E||x||_2 = sqrt(d). The variants differ only in how
the code is produced:

    Grassmannian: z_g = gamma * x D_g^T,  z = Pi_l(z_1..z_G),  D_g D_g^T = I_b
    Vanilla:      z_g = (xW + c)_g,       z = Pi_l(z_1..z_G),  ||D_g||_F <= 1
    Group Lasso:  z_g = sh_theta_g((xW + c)_g),                + lambda ||z||_{2,1}

All variants minimize  E||x - zD||^2  (+ Group-Lasso penalty)  + (1/l) * AuxK,
where AuxK asks the l runner-up blocks to explain the residual (App. D
"Losses", the block analogue of Gao et al. 2024).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# Guard for divisions by block norms: blocks with exactly-zero pre-code (dead
# inputs) must map to zero, not NaN.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Block-wise operators (Sec. 2.3). Both act on a block only through ||z_g||_2,
# which is what makes the *subspace* (not any basis of it) the atomic unit.
# ---------------------------------------------------------------------------
def block_norms(pre_z: torch.Tensor) -> torch.Tensor:
    """pre_z: (N, G, b) -> (N, G) block l2 norms."""
    return pre_z.norm(dim=-1)


def block_topk(pre_z: torch.Tensor, l: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pi_l: keep the l blocks of largest ||z_g||_2 per sample, zero the rest.

    Returns (z, sel_mask) with z: (N, G, b), sel_mask: (N, G) bool.
    Ties are broken by torch.topk's deterministic ordering — irrelevant in
    practice for float activations.
    """
    norms = block_norms(pre_z)                                   # (N, G)
    idx = norms.topk(l, dim=1).indices                           # (N, l)
    sel = torch.zeros_like(norms, dtype=torch.bool)
    sel.scatter_(1, idx, True)
    return pre_z * sel.unsqueeze(-1), sel


def block_soft_threshold(pre_z: torch.Tensor, theta: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """sh_theta(u)_g = max(1 - theta_g/||u_g||_2, 0) * u_g — the proximal
    operator of the l2,1 norm (Sec. 2.3). theta: (G,) positive thresholds.

    Blocks whose norm falls below theta_g are zeroed exactly (this is what
    produces genuine block sparsity rather than mere shrinkage).
    """
    norms = block_norms(pre_z)                                   # (N, G)
    scale = (1.0 - theta.unsqueeze(0) / norms.clamp_min(_EPS)).clamp_min(0.0)
    z = pre_z * scale.unsqueeze(-1)
    return z, scale > 0


def runner_up_codes(pre_z: torch.Tensor, sel_mask: torch.Tensor, l: int
                    ) -> torch.Tensor:
    """AuxK codes (App. D "Losses"): the code built from the next l blocks by
    pre-code norm among those NOT selected. These runner-up blocks are asked to
    explain what the selected ones missed, which keeps otherwise-dead blocks
    receiving gradient (the block analogue of Gao et al.'s aux loss).
    """
    norms = block_norms(pre_z)
    # Exclude already-selected blocks from the ranking by sending them to -inf.
    masked = norms.masked_fill(sel_mask, float("-inf"))
    idx = masked.topk(l, dim=1).indices                          # (N, l)
    aux_sel = torch.zeros_like(sel_mask)
    aux_sel.scatter_(1, idx, True)
    # Edge case: if fewer than l blocks remain unselected (tiny G), -inf rows
    # can be picked; their pre-code is real data so this stays harmless.
    return pre_z * aux_sel.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class BSFOutput:
    z: torch.Tensor            # (N, G, b) gated code (signed)
    x_hat: torch.Tensor        # (N, d) reconstruction in NORMALIZED space
    sel_mask: torch.Tensor     # (N, G) bool, blocks that fired
    losses: dict = field(default_factory=dict)   # total / recon / aux / lasso


class BSFBase(nn.Module):
    """Shared skeleton. Subclasses implement `_pre_code` (linear encoder) and
    `_gate` (Pi_l or soft-threshold), plus their constraint projection.

    The module stores a single global `scale` buffer with x_norm = x * scale,
    calibrated so E||x_norm|| = sqrt(d) (paper App. D normalization). forward()
    takes UNnormalized activations and returns losses/reconstructions in the
    normalized space; steering code multiplies edits by 1/scale to go back.
    """

    # How often the trainer must call project_constraints(): the Frobenius-ball
    # clamps (Vanilla/GroupLasso) are nearly free and run EVERY step — block
    # norms feed the top-l selection, so drift above 1 corrupts selection.
    # The Grassmannian overrides this with the paper's 20-step QR cadence.
    constraint_cadence = 1

    def __init__(self, d: int, G: int, b: int, l: int, aux_alpha: float | None = None):
        super().__init__()
        self.d, self.G, self.b, self.l = d, G, b, l
        # AuxK coefficient. Default 0.0 = OFF, matching the official BSF repo,
        # which trains every variant with pure reconstruction MSE (no Gao et al.
        # aux loss). Kept as an opt-in knob: re-enable with aux_alpha>0 (1/l is
        # a natural choice) if dead blocks regress.
        self.aux_alpha = 0.0 if aux_alpha is None else aux_alpha
        # Bounded-score gate option (see VanillaBSF._selection_scores): rank top-l by
        # scale-invariant scores instead of raw block norms. Plain attribute
        # (not a buffer) but PERSISTED via an explicit save_bsf payload key —
        # a model must gate at deployment the way it gated in training.
        self.bounded_scores = False
        # Decoder dictionary, one b-frame per block. Stored (G, b, d) so that
        # D_g = D[g] and decode is a single einsum.
        self.D = nn.Parameter(torch.randn(G, b, d) / math.sqrt(d))
        self.register_buffer("scale", torch.tensor(1.0))
        # Data mean, subtracted before encoding. NECESSARY for real network
        # activations: residual streams can carry most of their energy in the
        # mean vector, and a bias-free tied encoder (z_g = gamma x D_g^T) then
        # selects the same mu-aligned charts for EVERY token (observed: 99.6%
        # dead blocks). The paper states only the ||x||~sqrt(d) scaling, but
        # its covariance-based analyses presuppose centering — this is the
        # standard SAE pre-bias. The steering edit is unaffected: it adds a
        # DIFFERENCE of block contributions, so mu cancels.
        self.register_buffer("mu", torch.zeros(d))

    # -- data normalization --------------------------------------------------
    def set_normalization(self, mu: torch.Tensor, mean_centered_norm: float) -> None:
        """x_norm = (x - mu) * sqrt(d)/E||x - mu||, so E||x_norm|| = sqrt(d)."""
        self.mu.copy_(mu.to(self.mu))
        self.scale.fill_(math.sqrt(self.d) / max(mean_centered_norm, _EPS))

    def set_scale_from_mean_norm(self, mean_norm: float) -> None:
        """Scale-only normalization (mu = 0) — used for the zero-mean toy data
        and kept for backward compatibility (App. C.1's literal statement)."""
        self.scale.fill_(math.sqrt(self.d) / max(mean_norm, _EPS))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) * self.scale

    def denormalize(self, x_hat: torch.Tensor) -> torch.Tensor:
        """Inverse of normalize(): normalized-space recon -> input space.
        THE canonical inversion hook — consumers (adapters, evaluators) should
        call this instead of hand-writing `x_hat / scale + mu`, because
        subclasses with a non-affine preprocess (DualSAEFeaturizer's
        input_unit_norm per-sample LN) override it with the only correct
        inversion for their space. For plain BSFBase this is byte-identical
        to the historical inline expression."""
        return x_hat / self.scale + self.mu

    # -- per-step normalization (diffusion-activation escape hatch; unused by
    # the feature-space CB pipeline, whose bases use one global (mu, scale)) --
    # Activation statistics can vary strongly across sampler steps, so a
    # single global (mu, scale) lets some steps dominate both the loss
    # and the selection geometry.
    # Per-recorded-step (mu_s, scale_s) equalizes the steps; training data is
    # pre-normalized by the shuffle stage (the model's scalar normalize stays
    # identity), while deployment (Steerer/discovery) normalizes raw
    # activations through these tables.
    def set_step_normalization(self, record_steps, mus: torch.Tensor,
                               scales: torch.Tensor) -> None:
        self.register_buffer("step_ids", torch.as_tensor(record_steps,
                                                         dtype=torch.long))
        self.register_buffer("mu_steps", mus.to(torch.float32))       # (S, d)
        self.register_buffer("scale_steps", scales.to(torch.float32)) # (S,)

    @property
    def has_step_norm(self) -> bool:
        return hasattr(self, "mu_steps")

    def normalize_steps(self, x: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """x (n, d) raw + slots (n,) indices into record_steps -> normalized."""
        return (x - self.mu_steps[slots]) * self.scale_steps[slots].unsqueeze(-1)

    def norm_for_ode_step(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(mu, scale) of the NEAREST recorded step — steering visits all ODE
        steps but statistics were estimated on the 8 recorded ones; stats vary
        smoothly along the trajectory so nearest-slot is adequate."""
        slot = int((self.step_ids - step).abs().argmin())
        return self.mu_steps[slot], self.scale_steps[slot]

    # -- encoder pieces (variant-specific) ------------------------------------
    def _pre_code(self, x_norm: torch.Tensor) -> torch.Tensor:  # (N, G, b)
        raise NotImplementedError

    def _gate(self, pre_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _gate_train(self, pre_z: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training-forward gate seam (mirror of bsf/sae_models.py's
        SAEFeaturizer._gate_train): forward()/_dual_core gate through THIS,
        encode() through _gate. Identical for every family except
        BatchBlockTopK, whose batch-global training gate is only meaningful
        inside a training forward (a real batch) and must never leak into
        encode() — the deployment surface every evaluation
        calls, where it would make selection depend on chunk composition and
        mutate the EMA threshold buffer as a side effect."""
        return self._gate(pre_z)

    # -- public API ------------------------------------------------------------
    def encode(self, x_norm: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x_norm: (N, d) NORMALIZED input -> (pre_z, z, sel_mask).

        ALWAYS the per-sample DEPLOYMENT gate (_gate), independent of
        train/eval mode and free of buffer side effects — callers (discovery,
        steering, firing stats) need not remember to .eval() first."""
        pre_z = self._pre_code(x_norm)
        z, sel = self._gate(pre_z)
        return pre_z, z, sel

    # (G, b, d) decoder frames. The default is the raw parameter self.D;
    # variants whose usable decoder is a DERIVED quantity override this
    # (GrassmannianBSF returns QR(B_raw), recomputed every call). ALL decoder
    # reads — decode(), block_contribution(), steering, discovery, the span
    # penalty — must go through this so those variants stay correct.
    def decoder_atoms(self) -> torch.Tensor:
        return self.D

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(N, G, b) -> (N, d): x_hat = sum_g z_g D_g."""
        return torch.einsum("ngb,gbd->nd", z, self.decoder_atoms())

    def block_contribution(self, z_g: torch.Tensor, g: int) -> torch.Tensor:
        """m_g = z_g D_g for a single block. z_g: (N, b) -> (N, d)."""
        return z_g @ self.decoder_atoms()[g]

    def forward(self, x: torch.Tensor, target: torch.Tensor | None = None
                ) -> BSFOutput:
        """Reconstruct x. If `target` is given, reconstruct `target` from the
        (already noised) input x instead — the denoising objective the official
        trainer uses by default (train(snr>0)). Both are normalized identically.
        """
        x_norm = self.normalize(x)
        tgt_norm = x_norm if target is None else self.normalize(target)
        # Gate through the TRAINING seam (== _gate for every family except
        # BatchBlockTopK; see _gate_train). encode() stays deployment-only.
        pre_z = self._pre_code(x_norm)
        z, sel = self._gate_train(pre_z)
        x_hat = self.decode(z)
        recon = F.mse_loss(x_hat, tgt_norm)
        losses = {"recon": recon, "total": recon}

        # AuxK is OFF by default (aux_alpha=0; the official BSF has no aux loss).
        # When enabled, the runner-up blocks reconstruct the DETACHED residual so
        # the auxiliary task cannot degrade the main reconstruction (Gao et al.).
        if self.aux_alpha:
            r = (tgt_norm - x_hat).detach()
            z_aux = runner_up_codes(pre_z, sel, self.l)
            aux = F.mse_loss(self.decode(z_aux), r)
            losses["aux"] = aux
            losses["total"] = recon + self.aux_alpha * aux
        return BSFOutput(z=z, x_hat=x_hat, sel_mask=sel, losses=losses)

    # -- constraints / init (variant-specific) ---------------------------------
    @torch.no_grad()
    def project_constraints(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def calibrate_init(self, x_batch: torch.Tensor) -> None:
        """Variant-specific output-scale calibration on a real data batch, so
        training starts from an unbiased reconstruction (paper App. D)."""
        raise NotImplementedError

    @torch.no_grad()
    def _ls_output_gain(self, x_batch: torch.Tensor) -> float:
        """Least-squares gain c = <x, x_hat>/||x_hat||^2 over a batch: the
        scalar that makes c * x_hat an unbiased (in scale) reconstruction.

        ADAPTATION: the paper calibrates on "the average activation"; we solve
        the same 1-D problem over a batch with per-sample block selection,
        which is strictly more stable when the mean activation is small.
        """
        x_norm = self.normalize(x_batch)
        _, z, _ = self.encode(x_norm)
        x_hat = self.decode(z)
        num = (x_norm * x_hat).sum()
        den = (x_hat * x_hat).sum().clamp_min(_EPS)
        return (num / den).item()


class GrassmannianBSF(BSFBase):
    """Tied encoder z_g = gamma * x D_g^T with row-orthonormal charts
    D_g in St(b, d) (App. D). The single positive scalar gamma compensates the
    energy lost by tying encoder and decoder; it is parameterized through its
    logarithm to keep it positive.

    Orthonormality is enforced by REPARAMETERIZATION, exactly as the official
    repo does it: the free parameter is B_raw (G, d, b), and the usable frames
    are Q = QR(B_raw), recomputed on EVERY forward via decoder_atoms(). So the
    frames are always exactly orthonormal and gradients flow through the QR to
    B_raw — no periodic Stiefel retraction and no off-manifold drift between
    projections. (The paper text suggests projecting every ~20 steps; the
    published code uses QR-in-forward, which we follow for exactness. QR is
    recomputed in both encode and decode, matching the official code; the batched
    QR over (G, d, b) is sub-millisecond.)
    """

    def __init__(self, d, G, b, l, aux_alpha=None):
        super().__init__(d, G, b, l, aux_alpha)
        # The decoder is DERIVED from B_raw via QR, so the raw (G, b, d)
        # parameter created by the base is dropped (it would shadow
        # decoder_atoms() and bloat the checkpoint).
        del self.D
        # Free parameter in R^{d x b} per block; orthonormal COLUMNS at init.
        B = torch.randn(G, d, b)
        B, _ = torch.linalg.qr(B)                                # (G, d, b)
        self.B_raw = nn.Parameter(B)
        self.log_gamma = nn.Parameter(torch.zeros(()))

    @property
    def gamma(self) -> torch.Tensor:
        return self.log_gamma.exp()

    def _orthonormal_atoms(self) -> torch.Tensor:
        """QR(B_raw) as row-orthonormal (G, b, d) frames. QR of B_raw (G, d, b)
        gives orthonormal columns Q (G, d, b); transpose to (G, b, d) so each D_g
        satisfies D_g D_g^T = I_b. No sign-fix: the block projector D_g^T D_g and
        the block norm are invariant to the chart's basis (official omits it too).
        """
        Q, _ = torch.linalg.qr(self.B_raw)                       # (G, d, b)
        return Q.permute(0, 2, 1).reshape(self.G, self.b, self.d)

    def decoder_atoms(self) -> torch.Tensor:
        # Within a single forward() the QR is computed ONCE and reused by both
        # encode and decode via this cache. The official code recomputes it in
        # each (2 QRs/forward), but at our scale (G=4096) that doubled QR cost
        # dominated step time (~2.7x slower). Reusing one cached QR is
        # numerically identical (same B_raw) and routes both einsums' gradients
        # through a single QR backward, so it is faithful AND ~2x cheaper.
        # Outside forward() (steering/discovery/calibration call encode/decode
        # directly) the cache is None and the QR is recomputed on demand.
        cached = getattr(self, "_atoms_cache", None)
        return cached if cached is not None else self._orthonormal_atoms()

    def forward(self, x, target=None):
        # Compute the orthonormal frames once for this forward; encode, decode
        # (and, if enabled, AuxK's extra decode) all read the cache. The finally
        # clears our reference — the autograd graph is held by the returned
        # tensors, so backward still flows through the single QR.
        self._atoms_cache = self._orthonormal_atoms()
        try:
            return super().forward(x, target)
        finally:
            self._atoms_cache = None

    def _pre_code(self, x_norm):
        # z_g = gamma * x D_g^T for every block at once (tied to the QR frames).
        return torch.einsum("nd,gbd->ngb", x_norm,
                            self.decoder_atoms()) * self.gamma

    def _gate(self, pre_z):
        return block_topk(pre_z, self.l)

    @torch.no_grad()
    def project_constraints(self):
        # No-op: orthonormality is enforced in forward via decoder_atoms()'s QR.
        # Kept so the trainer's periodic project_constraints() call stays safe.
        pass

    @torch.no_grad()
    def calibrate_init(self, x_batch):
        # gamma0 = LS gain of the gamma=1 reconstruction: with a tied encoder
        # x_hat = gamma * P(x) x where P sums the selected charts' projectors,
        # so the optimum is the closed-form <x, Px>/||Px||^2 of the plan.
        self.log_gamma.zero_()
        c = self._ls_output_gain(x_batch)
        # Selection used norms scaled by gamma=1; rescaling gamma preserves the
        # top-l ranking (a global positive factor), so this stays consistent.
        self.log_gamma.fill_(math.log(max(c, _EPS)))


class VanillaBSF(BSFBase):
    """Free encoder W and dictionary D, each decoder block renormalized to the
    unit-Frobenius SPHERE ||D_g||_F = 1 every step (official unit_blocks). This
    resolves the z-vs-D scale ambiguity that would otherwise corrupt top-l
    selection. (Earlier we used a shrink-only ball; the published code
    renormalizes to exactly 1.)
    """

    def __init__(self, d, G, b, l, aux_alpha=None):
        super().__init__(d, G, b, l, aux_alpha)
        self.enc = nn.Linear(d, G * b, bias=True)
        with torch.no_grad():
            # ||D_g||_F = 1 exactly at init (on the constraint boundary).
            self.D.div_(self.D.norm(dim=(1, 2), keepdim=True).clamp_min(_EPS))
            # Paper init: W = D^T so pre-code block norms have the correct
            # order of magnitude at initialization, which removes most dead
            # blocks without auxiliary losses. calibrate_init() then applies
            # the data-dependent gain.
            self.enc.weight.copy_(self.D.reshape(self.G * self.b, self.d))
            self.enc.bias.zero_()

    def _pre_code(self, x_norm):
        return self.enc(x_norm).view(-1, self.G, self.b)

    # ---- Bounded selection scores (optional gate) (design-faithful collapse
    # fix — changes the GATE, not the loss; the joint dual objective stays
    # untouched). Motivation: the free-encoder collapse is a rich-get-richer
    # ratchet where incumbents inflate their top-l selection score (the raw
    # pre-code block norm) by growing unbounded encoder rows. Grassmannian is
    # immune because its scores are projections onto orthonormal frames,
    # bounded by ||x||. This transplants that property (Gated-SAE-style
    # score/code decoupling): rank blocks by ||pre_z_g|| / ||[W_g | b_g]||_F —
    # scale-invariant in the block's full affine encoder params and bounded by
    # sqrt(||x||^2+1) (Cauchy-Schwarz on the augmented map [W_g|b_g][x;1]) —
    # while the RECONSTRUCTION code stays the raw pre_z (full expressivity).
    # The bias is inside the denominator on purpose: dividing by ||W_g||_F
    # alone would leave bias growth as a fresh ratchet channel.
    # Default False = official BSF behavior, byte-identical gates; persisted
    # through save_bsf/load_bsf (deployment must gate the way training did).
    def _selection_scores(self, pre_z):
        norms = block_norms(pre_z)
        if not getattr(self, "bounded_scores", False):
            return norms
        W = self.enc.weight.view(self.G, self.b, self.d)
        bias = self.enc.bias.view(self.G, self.b)
        denom = (W.pow(2).sum(dim=(1, 2))
                 + bias.pow(2).sum(dim=1)).sqrt().clamp_min(_EPS)
        return norms / denom.unsqueeze(0)

    def _gate(self, pre_z):
        if not getattr(self, "bounded_scores", False):
            return block_topk(pre_z, self.l)
        # Rank by bounded scores, KEEP the raw code (z magnitudes drive the
        # reconstruction; only the competition is normalized). Scores feed
        # topk indices only — no gradient path, same as block_topk.
        scores = self._selection_scores(pre_z)
        idx = scores.topk(self.l, dim=1).indices
        sel = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)
        return pre_z * sel.unsqueeze(-1), sel

    @torch.no_grad()
    def project_constraints(self):
        # Project each decoder block onto the unit-Frobenius SPHERE (exact unit
        # norm every step), matching the official repo's unit_blocks(): x_hat's
        # scale is carried entirely by the code, removing the z-vs-D ambiguity
        # that would corrupt top-l selection.
        norms = self.D.norm(dim=(1, 2), keepdim=True).clamp_min(_EPS)
        self.D.div_(norms)

    @torch.no_grad()
    def calibrate_init(self, x_batch):
        c = self._ls_output_gain(x_batch)
        # Scaling W scales z (and hence x_hat) linearly; top-l selection is
        # invariant to the global factor.
        self.enc.weight.mul_(c)

    @torch.no_grad()
    def reinit_dead(self, x_norm, dead_mask, scale: float = 0.1):
        """Dead-BLOCK resampling — the block analogue of
        SAEFeaturizer.reinit_dead (bsf/sae_models.py). Re-points each
        dead block's b encoder rows AND its b decoder atoms at random unit data
        directions (0.1 scale, as the SAE does) so the block can start winning
        top-l again. NOTE: no trainer resets the Adam state of the revived rows, so
        revived params carry their stale second-moment estimates (the b=1 SAE
        trainers behave the same way).

        PROJECT EXTENSION (not in the BSF paper — mirrors the DualSAE
        dead-latent resampling that keeps the SAE-duals alive at dead 11-24%
        while the un-revived BSF duals collapse to ~91%). Opt-in only; the default BSF/dual path never calls this.
        Not used by the CB-SAE pipeline.

        x_norm: (M, d) NORMALIZED activation batch (the encoder's input space).
        dead_mask: (G,) bool over BLOCKS. enc.weight is (G*b, d) so a dead
        block owns the contiguous rows [g*b:(g+1)*b]; D is (G, b, d). Returns
        the number of blocks reset."""
        n_dead = int(dead_mask.sum())
        if n_dead == 0:
            return 0
        b, d = self.b, self.d
        # b random data directions per dead block (with replacement) -> the
        # block's b-frame spans a fresh data-aligned subspace.
        # `scale` sets the revived encoder rows' magnitude and hence the
        # block's pre-code norm — i.e. whether it can WIN top-l selection
        # again. 0.1 is the historical SAE-parity default; under hard
        # per-sample top-l with entrenched incumbents a 0.1-scale revival is
        # dead on arrival, so callers may pass a competitive scale instead.
        idx = torch.randint(0, x_norm.shape[0], (n_dead * b,), device=x_norm.device)
        s = F.normalize(x_norm[idx].float(), dim=-1).view(n_dead, b, d) * scale
        # Expand the (G,) block mask to the (G*b,) encoder-row mask; block g's b
        # rows are contiguous and intra-block-ordered (enc output reshaped
        # (N, G, b) in _pre_code), matching s's (n_dead, b, d) layout.
        row_mask = dead_mask.repeat_interleave(b)
        self.enc.weight.data[row_mask] = s.reshape(n_dead * b, d).to(self.enc.weight.dtype)
        self.enc.bias.data[row_mask] = 0.0
        # D is renormalized to unit-Frobenius every step by project_constraints,
        # so only the direction of the reset atoms matters (the 0.1 is cosmetic
        # here, kept for parity with the SAE reinit).
        self.D.data[dead_mask] = s.to(self.D.dtype)
        return n_dead


class _BlockJumpReLU(torch.autograd.Function):
    """Block JumpReLU gate with a straight-through estimator (official repo's
    default Group-Lasso variant). Forward: a hard {0,1} gate 1[||z_g|| > theta_g]
    on the block norms gn (N, G). Backward: a rectangle pseudo-derivative of
    width `bandwidth` centered at each threshold, passing gradient to theta only
    (the encoder learns via the magnitude path z = a * gate, not via the gate).
    """
    @staticmethod
    def forward(ctx, gn, theta, bandwidth):
        ctx.save_for_backward(gn, theta)
        ctx.bw = float(bandwidth)
        return (gn > theta).to(gn.dtype)

    @staticmethod
    def backward(ctx, g):
        gn, theta = ctx.saved_tensors
        K = ((gn - theta).abs() <= ctx.bw / 2).to(gn.dtype) / ctx.bw
        # No gradient to gn; theta gradient is summed over the batch.
        return None, -(g * K).sum(0), None


class GroupLassoBSF(BSFBase):
    """Free linear encoder + block sparsity via a learned threshold. Two variants
    matching the official repo, selected by `paper_version`:

      * paper_version=False (DEFAULT): block JumpReLU with a straight-through
        estimator. Per-block threshold theta_g = softplus(gain * raw_theta_g);
        the gate 1[||z_g|| > theta_g] hard-zeroes sub-threshold blocks; sparsity
        is driven by an L0 penalty coef * E[#active blocks]. The STE `bandwidth`
        is seeded from the first training batch's block-norm std.

      * paper_version=True: the paper's soft-threshold. A single scalar threshold
        theta = exp(log_theta) shrinks every block by the prox of the l2,1 norm,
        sh_theta(u)_g = max(1 - theta/||u_g||, 0) u_g; sparsity is driven by an
        l2,1 penalty coef * E[sum_g ||z_g||].

    `l` is only the AuxK runner-up count / a target-sparsity reference — sparsity
    here is set by (coef, theta), not a hard top-l cap.
    """

    def __init__(self, d, G, b, l, aux_alpha=None, coef: float = 1e-2,
                 gain: float = 10.0, paper_version: bool = False):
        super().__init__(d, G, b, l, aux_alpha)
        self.coef = coef
        self.gain = gain
        self.paper_version = paper_version
        self.enc = nn.Linear(d, G * b, bias=True)
        with torch.no_grad():
            # Unit-Frobenius SPHERE decoder + tied init W = D^T (as the TopK
            # variants). ADAPTATION: official uses a free decoder for group
            # lasso, but with a free decoder the penalty can be evaded by
            # shrinking z and growing D; the unit-norm constraint keeps it
            # well-posed and is consistent with our other variants.
            self.D.div_(self.D.norm(dim=(1, 2), keepdim=True).clamp_min(_EPS))
            self.enc.weight.copy_(self.D.reshape(self.G * self.b, self.d))
            self.enc.bias.zero_()
        if paper_version:
            # Scalar threshold, theta = exp(0) = 1 at init.
            self.log_theta = nn.Parameter(torch.zeros(()))
        else:
            # Per-block raw threshold, theta_g = softplus(gain*0) at init; the
            # STE bandwidth is estimated lazily from the first training batch.
            self.raw_theta = nn.Parameter(torch.zeros(G))
            self.register_buffer("bandwidth", torch.tensor(1.0))
            self.register_buffer("inited", torch.tensor(False))

    @property
    def theta(self) -> torch.Tensor:
        if self.paper_version:
            return self.log_theta.exp()
        return F.softplus(self.gain * self.raw_theta)

    def _pre_code(self, x_norm):
        return self.enc(x_norm).view(-1, self.G, self.b)

    def _maybe_init_bandwidth(self, gn: torch.Tensor) -> None:
        # Seed the STE window from the first training batch's block-norm spread
        # (official: gn.std()*0.5 + 1e-6) so the rectangle overlaps real density.
        if self.training and not bool(self.inited):
            self.bandwidth.fill_(float(gn.detach().std() * 0.5 + 1e-6))
            self.inited.fill_(True)

    def _gate(self, pre_z):
        # Deployment/plain gate (used by encode() for discovery/steering).
        if self.paper_version:
            return block_soft_threshold(pre_z, self.theta)
        gn = block_norms(pre_z)                                  # (N, G)
        self._maybe_init_bandwidth(gn)
        gate = _BlockJumpReLU.apply(gn, self.theta, float(self.bandwidth))
        return pre_z * gate.unsqueeze(-1), gate > 0

    def _gate_train_with_penalty(self, pre_z):
        """Gate + sparsity penalty with the TRAINING gradient path (the base
        encode()/sel_mask is grad-free): the l2,1 term flows through the
        soft-threshold scale, the L0 term through the JumpReLU STE.

        Factored out of forward() so DualGroupLassoBSF reuses the *identical*
        tensor-op sequence (plain and dual can never drift apart on the gate).
        Returns (z, sel, extra) where extra carries the differentiable "lasso"
        penalty plus a detached active-block count for logging.
        """
        if self.paper_version:
            z, sel = block_soft_threshold(pre_z, self.theta)
            # l2,1 penalty: E[sum_g ||z_g||] (per-sample sum, batch mean).
            penalty = self.coef * block_norms(z).sum(dim=1).mean()
            extra = {"lasso": penalty}
        else:
            gn = block_norms(pre_z)
            self._maybe_init_bandwidth(gn)
            gate = _BlockJumpReLU.apply(gn, self.theta, float(self.bandwidth))
            z = pre_z * gate.unsqueeze(-1)
            sel = gate > 0
            # L0 penalty: E[#active blocks], STE gradient to theta via `gate`.
            penalty = self.coef * gate.sum(dim=1).mean()
            extra = {"lasso": penalty}
            # --- STE-stall diagnostics (detached; DO NOT affect the loss). ------
            # The L0 penalty's gradient reaches theta only through the rectangle
            # window K=1[|gn-theta|<=bw/2]: a block instance can push theta ONLY
            # if its norm is within bw/2 of its threshold. The hypothesis for the
            # grouplasso 50x-over-sparsity is that theta stalls once the active
            # block-norm cluster sits far ABOVE the window (K->0), so no coef can
            # sparsify further. These scalars measure that directly; all detached
            # (torch.no_grad + already-computed gn), so training is bit-unchanged.
            with torch.no_grad():
                th = self.theta                     # (G,) per-block thresholds
                bw = self.bandwidth                 # scalar STE window width
                diff = gn - th.unsqueeze(0)         # (N,G) norm minus block threshold
                active = diff > 0
                extra.update({
                    "gl_theta_mean": th.mean(),
                    "gl_bandwidth": bw.clone(),
                    "gl_blocknorm_mean": gn.mean(),
                    # Live fraction that can still move theta: if this -> 0 while
                    # #active >> target, theta is gradient-starved (the stall).
                    "gl_frac_in_window": (diff.abs() <= bw * 0.5).float().mean(),
                    # Mean over-threshold margin of ACTIVE blocks, in bandwidth
                    # units. >>1 => the active cluster is out of the STE window.
                    "gl_active_gap_bw": (diff[active].mean() / bw.clamp_min(_EPS)
                                         if bool(active.any())
                                         else gn.new_zeros(())),
                })
        extra["mean_active_blocks"] = sel.float().sum(dim=1).mean().detach()
        return z, sel, extra

    def forward(self, x, target=None):
        # Own forward so the sparsity penalty keeps its gradient path (the base
        # forward's encode() would drop it); the gate+penalty math lives in
        # _gate_train_with_penalty, shared with DualGroupLassoBSF.
        x_norm = self.normalize(x)
        tgt_norm = x_norm if target is None else self.normalize(target)
        pre_z = self._pre_code(x_norm)
        z, sel, extra = self._gate_train_with_penalty(pre_z)
        x_hat = self.decode(z)
        recon = F.mse_loss(x_hat, tgt_norm)
        total = recon + extra["lasso"]
        losses = {"recon": recon, **extra}
        if self.aux_alpha:
            r = (tgt_norm - x_hat).detach()
            z_aux = runner_up_codes(pre_z, sel, self.l)
            aux = F.mse_loss(self.decode(z_aux), r)
            losses["aux"] = aux
            total = total + self.aux_alpha * aux
        losses["total"] = total
        return BSFOutput(z=z, x_hat=x_hat, sel_mask=sel, losses=losses)

    @torch.no_grad()
    def project_constraints(self):
        # Unit-Frobenius sphere (same as Vanilla).
        norms = self.D.norm(dim=(1, 2), keepdim=True).clamp_min(_EPS)
        self.D.div_(norms)

    @torch.no_grad()
    def calibrate_init(self, x_batch):
        # Scale the encoder by the LS output gain so recon starts unbiased, and
        # scale the thresholds by the same factor so the threshold-to-norm ratio
        # (hence the initial sparsity) is preserved: both gates act on the block
        # norm, which scales linearly with the encoder.
        c = self._ls_output_gain(x_batch)
        self.enc.weight.mul_(c)
        self.enc.bias.mul_(c)
        if self.paper_version:
            self.log_theta.add_(math.log(max(c, _EPS)))
        else:
            # raw_theta s.t. softplus(gain*raw_new) = softplus(gain*raw_old) * c.
            new_theta = self.theta * c                          # (G,)
            inv = torch.where(new_theta > 20, new_theta,        # overflow-safe
                              torch.log(torch.expm1(new_theta.clamp_min(1e-8))))
            self.raw_theta.copy_(inv / self.gain)
            # _ls_output_gain's encode() may have seeded the bandwidth on the
            # pre-scaling norms; reset so the first real training batch re-seeds
            # it on the correctly-scaled block norms.
            self.inited.fill_(False)


# ---------------------------------------------------------------------------
# DualBSF: the DualSAE decomposition (bsf/sae_models.py:DualSAEFeaturizer,
# itself a port of vision/models/dual_sae.py) generalized from b=1 scalar latents
# to b-dim blocks. The BSF block branch stays the PRIMARY, steerable dictionary
# encoding x; a dense residual MLP ("poly") mops up r = x_norm - xB; a span
# penalty keeps the poly output out of the active blocks' subspaces so
# polysemantic residual content cannot re-enter the steered dictionary.
#
# Deployment contract (mirror of the DualSAE one, sae_models.py): the
# duals override ONLY __init__/forward — _pre_code/_gate/encode/decode/
# decoder_atoms/block_contribution/project_constraints/calibrate_init and all
# normalization are inherited verbatim from the plain family, so discovery and
# steering treat a dual_<family> run byte-for-byte like its plain <family>
# (only the learned weights differ). The poly branch is training-only.
# ---------------------------------------------------------------------------
class DualBSFMixin:
    """Poly residual branch + span penalty + two-phase warmup for BSF families.

    Training protocol (matches the canonical DualSAE trainers —
    vision/train_dual.py freezes poly and zeroes the span during warmup, the topic-model
    DualSAE trainer reconstructs with the mono branch only for dual_warmup_steps):
      phase 1 (`dual_warmup_active=True`, set per-step by the trainer):
        x_hat = xB, span = 0, poly never called -> the model optimizes the
        exact plain-BSF objective while the poly params sit frozen.
      phase 2: x_hat = xB + poly(x_norm - xB) with the span penalty on.
    `dual_warmup_active` defaults to False so any forward outside the trainer
    (tests, ad-hoc eval) exercises the full dual path; deployment is unaffected
    either way because discovery/steering only ever call encode().
    """

    def _init_dual(self, poly_hidden: int, lam_span: float) -> None:
        # Called at the END of each dual subclass __init__ (after the family
        # init built the encoder/decoder), mirroring DualSAEFeaturizer's args:
        # poly_hidden <= 0 -> the repo's legacy 1-layer Linear(d,d)+GELU (the
        # setting the trained dual_*_sae checkpoints used); > 0 -> the paper's
        # d->h->d bottleneck MLP.
        self.lam_span = lam_span
        # Stored verbatim (not introspected from the Sequential) so the
        # save/load round-trip through extra_config() is exact.
        self._poly_hidden = int(poly_hidden) if poly_hidden else 0
        d = self.d
        if self._poly_hidden > 0:
            self.poly = nn.Sequential(nn.Linear(d, self._poly_hidden), nn.GELU(),
                                      nn.Linear(self._poly_hidden, d))
        else:
            self.poly = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.dual_warmup_active = False
        # PROJECT EXTENSION ablation knob (default False = paper behavior): when
        # True, the poly input residual is detached in _dual_core so the poly
        # branch's gradient cannot flow back into the mono (block) branch. This
        # severs the shared-MSE co-adaptation that drives the Dual-Vanilla mono
        # collapse. Off by default; not used by the CB-SAE pipeline.
        self.detach_poly = False
        # OPTIONAL REPAIR KNOBS (default 0/off = paper behavior; train-time only,
        # deliberately NOT in extra_config so checkpoints round-trip through the
        # existing schema; not used by the CB-SAE pipeline).
        # Motivation: the free-encoder dual
        # collapse is caused by the poly branch eliminating the residual error
        # that keeps marginal blocks alive under winner-take-all top-l
        # selection — blocks die starting EXACTLY at poly unfreeze, even from a
        # converged 0%-dead warmup-17500 population.
        #   mono_anchor_alpha: adds alpha*MSE(xB, target) in phase 2 so the
        #     mono branch keeps an undiluted reconstruction pressure of its own
        #     (NOT the failed --detach-poly arm: that detached poly's input but
        #     left the loss joint, so mono still saw only post-poly crumbs).
        #   dead_auxk_alpha/_blocks: Gao et al. (2024) AuxK adapted to blocks
        #     and to the dual — reconstruct the MONO residual (x_norm - xB),
        #     not the post-poly residual (which is the starved signal), using
        #     the top-k_aux DEAD blocks. Target detached, selection restricted
        #     to dead blocks => gradient reaches ONLY dead-block params —
        #     FREE-ENCODER FAMILIES ONLY: a tied encoder's shared gain
        #     (Grassmannian log_gamma) rides in pre_z and would receive the
        #     aux gradient too, so the combination is not supported.
        #     NOTE this is NOT the repo's runner-up aux_alpha (bsf/models.py
        #     runner_up_codes), which is known to WORSEN dead fractions.
        self.mono_anchor_alpha = 0.0
        self.dead_auxk_alpha = 0.0
        self.dead_auxk_blocks = 64
        # (G,) bool set by the trainer from its silent-block counter; a plain
        # attribute (not a buffer) so the checkpoint payload is unchanged.
        self.dead_mask = None
        # Provenance only (recorded into checkpoints by extra_config); the
        # trainer overwrites it with the actual --dual-warmup-steps used.
        self.dual_warmup_steps = 0

    def _span_penalty(self, r_hat: torch.Tensor,
                      sel_mask: torch.Tensor) -> torch.Tensor:
        """b-dim generalization of sae_models._span_overlap_penalty
        (active_only=True): per sample, the mean over ACTIVE blocks of
        ||D_g r_hat||^2, averaged over the batch. At b=1 this is numerically
        identical to the DualSAE formula (overlap^2 per active latent /
        #active, batch mean). For the Grassmannian's row-orthonormal frames
        ||D_g r_hat||^2 is exactly the energy of the poly output inside block
        g's subspace; for Vanilla/GroupLasso (unit-Frobenius, rows not
        orthonormal) it is the frame-analysis energy — the natural surrogate.
        """
        # MUST read frames via decoder_atoms(): the Grassmannian's usable
        # decoder is QR(B_raw), not a raw parameter. Inside a dual forward the
        # QR is cached, so this einsum reuses it (no extra QR).
        o = torch.einsum("nd,gbd->ngb", r_hat, self.decoder_atoms())  # (N,G,b)
        sq = o.pow(2).sum(-1) * sel_mask.to(r_hat.dtype)              # (N, G)
        denom = sel_mask.to(r_hat.dtype).sum(dim=1).clamp_min(1.0)    # (N,)
        return (sq.sum(dim=1) / denom).mean()

    def _dual_core(self, x, target=None, gate_fn=None) -> BSFOutput:
        """Shared dual forward body. `gate_fn` lets GroupLasso inject its
        training gate + differentiable sparsity penalty (returned via `extra`);
        the TopK families use the plain grad-free _gate, exactly as their base
        forward does."""
        x_norm = self.normalize(x)
        tgt_norm = x_norm if target is None else self.normalize(target)
        pre_z = self._pre_code(x_norm)
        if gate_fn is None:
            # Training seam, same as BSFBase.forward: duals must run the SAME
            # mode-switched gate as their plain family (BatchBlockTopK's
            # batch-global gate + EMA update at train time).
            z, sel = self._gate_train(pre_z)
            extra = {}
        else:
            z, sel, extra = gate_fn(pre_z)
        xB = self.decode(z)                     # steerable block reconstruction
        if self.dual_warmup_active:
            # Phase 1: plain-BSF objective. poly is not called (its params are
            # frozen by the trainer anyway) and the span is a true zero so the
            # loss/gradients match the plain family exactly during warmup.
            x_hat = xB
            r_hat = None
            span = xB.new_zeros(())
        else:
            # Residual from the (possibly noised) INPUT side — where DualSAE
            # computes it (sae_models.py), and deliberately NOT from
            # tgt_norm: under the snr>0 denoising objective that would leak the
            # clean target into the poly input. No stop-grad (DualSAE parity:
            # the block branch receives gradient through r as well).
            #
            # STABILITY NOTE: with a free-encoder family (vanilla/grouplasso)
            # and an OVERCOMPLETE dictionary the dense poly can simply replace
            # the blocks (the two branches co-adapt to cancel via the shared
            # x_hat MSE, poly_frac >> 1). What prevents this is the warmup:
            # phase 1 trains the blocks ALONE to near-complete reconstruction,
            # so when the poly unfreezes the genuine residual is small and the
            # poly stays small (poly_frac ~0.1, verified on the tied-encoder
            # grassmannian even at warmup=500). On real activation streams plain
            # BSF already reaches R^2~0.99, so post-warmup the residual is the
            # genuinely-polysemantic remainder and the blocks are not
            # outcompeted. The metrics.json `poly_frac` / `heldout_r2_mono`
            # fields exist to CATCH a regression here on real data.
            r = x_norm - xB
            if self.detach_poly:
                # Ablation (--detach-poly): cut the gradient path from the poly
                # branch back into the mono branch. The blocks still receive the
                # full reconstruction gradient through xB, but poly now fits a
                # DETACHED residual, so it can no longer co-adapt with the blocks
                # to cancel via the shared x_hat MSE — the mechanism that lets
                # the free-encoder mono branch shed blocks until they die.
                r = r.detach()
            r_hat = self.poly(r)
            x_hat = xB + r_hat
            span = self._span_penalty(r_hat, sel)
        recon = F.mse_loss(x_hat, tgt_norm)
        lasso = extra.get("lasso")
        total = recon + self.lam_span * span
        if lasso is not None:
            total = total + lasso
        losses = {"recon": recon, "span": span, **extra}
        # Diagnostic: share of reconstruction energy carried by the poly branch
        # (0 during warmup). A healthy dual keeps this small — the blocks, not
        # the dense mop-up, should explain the data.
        losses["poly_frac"] = (
            xB.new_zeros(()) if r_hat is None
            else (r_hat.pow(2).sum() / x_hat.pow(2).sum().clamp_min(_EPS))
            .detach())
        if self.aux_alpha:
            # AuxK on the MONO residual (DualSAEFeaturizer parity, incl. its
            # x_norm-side target): runner-up blocks explain what the selected
            # blocks missed, keeping otherwise-dead blocks — the steerable
            # dictionary — receiving gradient. Off by default (aux_alpha=0).
            z_aux = runner_up_codes(pre_z, sel, self.l)
            aux = F.mse_loss(self.decode(z_aux), (x_norm - xB).detach())
            losses["aux"] = aux
            total = total + self.aux_alpha * aux
        # Optional mono anchor. Phase-2 only: during warmup the recon
        # term already IS mse(xB, tgt) so the anchor would just rescale it and
        # break the exact plain-BSF warmup equivalence.
        if self.mono_anchor_alpha and not self.dual_warmup_active:
            anchor = F.mse_loss(xB, tgt_norm)
            losses["mono_anchor"] = anchor
            total = total + self.mono_anchor_alpha * anchor
        # Optional dead-block AuxK (Gao et al. adapted; see _init_dual).
        # Phase-2 only (no meaningful dead set exists during warmup, and the
        # warmup objective must stay exactly plain-BSF). Skipped silently when
        # the trainer has not provided a dead mask or nothing is dead.
        if self.dead_auxk_alpha and not self.dual_warmup_active:
            dead = self.dead_mask
            if dead is not None and bool(dead.any()):
                k_aux = min(int(self.dead_auxk_blocks), int(dead.sum()))
                # Rank dead blocks by pre-code norm; live blocks are masked to
                # -inf so they can NEVER be selected (k_aux <= #dead keeps the
                # topk inside the finite entries). detach: the ranking is
                # index-only, no gradient should flow through the scores.
                scores = block_norms(pre_z).detach().masked_fill(
                    ~dead.unsqueeze(0), float("-inf"))
                idx = scores.topk(k_aux, dim=1).indices
                sel_dead = torch.zeros(pre_z.shape[0], pre_z.shape[1],
                                       dtype=torch.bool,
                                       device=pre_z.device).scatter_(1, idx, True)
                z_dead = pre_z * sel_dead.unsqueeze(-1)
                # Target = MONO residual, detached: gradient flows only through
                # decode(z_dead) into the encoder rows / decoder atoms of the
                # selected dead blocks — live-model dynamics are untouched.
                aux_dead = F.mse_loss(self.decode(z_dead),
                                      (x_norm - xB).detach())
                losses["aux_dead"] = aux_dead
                total = total + self.dead_auxk_alpha * aux_dead
        losses["total"] = total
        return BSFOutput(z=z, x_hat=x_hat, sel_mask=sel, losses=losses)

    def variant_name(self) -> str:
        return f"dual_{self._family}"

    def extra_config(self) -> dict:
        # Self-describing checkpoint payload (save_bsf/load_bsf round-trip).
        return {"dual": True, "family": self._family,
                "poly_hidden": self._poly_hidden, "lam_span": self.lam_span,
                "dual_warmup_steps": getattr(self, "dual_warmup_steps", 0)}


class DualGrassmannianBSF(GrassmannianBSF, DualBSFMixin):
    _family = "grassmannian"

    def __init__(self, d, G, b, l, aux_alpha=None, poly_hidden: int = 0,
                 lam_span: float = 0.01):
        super().__init__(d, G, b, l, aux_alpha)
        self._init_dual(poly_hidden, lam_span)

    def forward(self, x, target=None):
        # Same single-QR cache pattern as the plain forward (see
        # GrassmannianBSF.forward): encode, decode, AND the span-penalty einsum
        # all read one cached QR; recon+span gradients flow through that single
        # QR backward. Defined here (not on the mixin) because the plain
        # forward's super().forward() would bypass _dual_core under MRO.
        self._atoms_cache = self._orthonormal_atoms()
        try:
            return self._dual_core(x, target)
        finally:
            self._atoms_cache = None


class DualVanillaBSF(VanillaBSF, DualBSFMixin):
    _family = "vanilla"

    def __init__(self, d, G, b, l, aux_alpha=None, poly_hidden: int = 0,
                 lam_span: float = 0.01):
        super().__init__(d, G, b, l, aux_alpha)
        self._init_dual(poly_hidden, lam_span)

    def forward(self, x, target=None):
        return self._dual_core(x, target)


class DualGroupLassoBSF(GroupLassoBSF, DualBSFMixin):
    _family = "grouplasso"

    def __init__(self, d, G, b, l, aux_alpha=None, coef: float = 1e-2,
                 gain: float = 10.0, paper_version: bool = False,
                 poly_hidden: int = 0, lam_span: float = 0.01):
        super().__init__(d, G, b, l, aux_alpha, coef=coef, gain=gain,
                         paper_version=paper_version)
        self._init_dual(poly_hidden, lam_span)

    def forward(self, x, target=None):
        # The training gate carries the sparsity penalty's gradient path
        # (identical tensor ops to the plain forward via the shared helper).
        return self._dual_core(x, target,
                               gate_fn=self._gate_train_with_penalty)


VARIANTS = {
    "grassmannian": GrassmannianBSF,
    "vanilla": VanillaBSF,
    "grouplasso": GroupLassoBSF,
}

# Dual-BSF registry. Named DUAL_BSF_VARIANTS (not DUAL_VARIANTS — that name is
# taken by the DualSAE mono-mode set in bsf/sae_models.py).
DUAL_BSF_VARIANTS = {
    "dual_grassmannian": DualGrassmannianBSF,
    "dual_vanilla": DualVanillaBSF,
    "dual_grouplasso": DualGroupLassoBSF,
}


def build_bsf(variant: str, d: int, G: int, b: int, l: int,
              aux_alpha: float | None = None, coef: float = 1e-2,
              gain: float = 10.0, paper_version: bool = False,
              poly_hidden: int = 0, lam_span: float = 0.01) -> BSFBase:
    if variant == "grouplasso":
        return GroupLassoBSF(d, G, b, l, aux_alpha, coef=coef, gain=gain,
                             paper_version=paper_version)
    if variant in DUAL_BSF_VARIANTS:
        cls = DUAL_BSF_VARIANTS[variant]
        if variant == "dual_grouplasso":
            return cls(d, G, b, l, aux_alpha, coef=coef, gain=gain,
                       paper_version=paper_version,
                       poly_hidden=poly_hidden, lam_span=lam_span)
        return cls(d, G, b, l, aux_alpha,
                   poly_hidden=poly_hidden, lam_span=lam_span)
    return VARIANTS[variant](d, G, b, l, aux_alpha)


# ---------------------------------------------------------------------------
# BatchBlockTopK — the block analogue of BatchTopK-SAE (Bussmann et al. 2024),
# This whole section is ADDITIVE: nothing above it changed, so existing checkpoints (whose
# payloads never reference the new registry keys) keep loading byte-identically.
# ---------------------------------------------------------------------------
def batch_block_topk(pre_z: torch.Tensor, l: int
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """BatchTopK TRAINING gate on BLOCK NORMS: keep the N*l blocks of largest
    ||z_g||_2 across the ENTIRE batch (Bussmann et al. 2024, generalized from
    the b=1 scalar gate bsf/sae_models.py:batch_topk_gate to b-dim blocks).
    The per-sample budget l then holds only ON AVERAGE — "busy" samples borrow
    blocks from sparse ones, which is the method's point.

    Block norms are >= 0 by construction (no ReLU needed, unlike the SAE
    gate's post-ReLU codes), so the same smallest-kept-value '>= tau'
    selection idiom carries over unchanged.

    Only valid where a real batch exists (training); single-sample encoding at
    deployment uses the EMA-threshold gate in BatchBlockTopKBSF._gate instead.
    Returns (z, sel_mask) with z: (N, G, b), sel_mask: (N, G) bool.
    """
    norms = block_norms(pre_z)                                   # (N, G) >= 0
    N, G = norms.shape
    total = min(N * l, N * G)
    if total <= 0 or total >= N * G:
        # Degenerate budget (l >= G, or an empty batch): keep every non-dead
        # block, mirroring batch_topk_gate's fallback.
        sel = norms > 0
        return pre_z * sel.unsqueeze(-1), sel
    tau = torch.topk(norms.reshape(-1), total, sorted=True).values[-1]
    # '>=' includes the smallest kept norm; exact ties at tau can keep
    # marginally more than `total` blocks — harmless (the loss reconstructs
    # from a few extra near-tau blocks) and never observed on float norms.
    sel = norms >= tau
    return pre_z * sel.unsqueeze(-1), sel


class BatchBlockTopKBSF(VanillaBSF):
    """Vanilla-family encoder/decoder with a BATCH-global top-(N*l) gate on
    block norms — the block analogue of BatchTopK-SAE (Bussmann et al. 2024).

    Everything but the gate is inherited from VanillaBSF: free Linear encoder
    with tied W = D^T init, unit-Frobenius-SPHERE decoder projection every
    step, and the LS-gain output calibration. Two gates:

      * _gate (deployment; what encode() ALWAYS uses, in any train/eval
        mode): sel = ||z_g|| > threshold — STRICTLY greater, mirroring
        BatchTopKSAEFeaturizer's deploy gate at b=1 — evaluated per sample
        and free of side effects, so single-token encoding (any
        evaluation) is safe even when a caller forgets .eval().
        Before any training batch has seeded the threshold it falls back to
        the per-sample top-l gate (a 0 threshold would pass every block).
      * _gate_train (training forward, self.training True): batch_block_topk
        keeps the N*l largest block norms batch-wide, and an EMA of the
        smallest kept norm is maintained as the deployment threshold
        (Bussmann et al.'s recommended BatchTopK -> threshold inference
        conversion, exactly what bsf/sae_models.py:BatchTopKSAEFeaturizer
        does at b=1). An eval-mode forward() (mid-training val R2) routes to
        the deploy gate: evaluation must score deployment behavior and must
        never move the threshold buffer.

    `threshold`/`inited` are registered buffers, so save_bsf/load_bsf
    round-trip them through state_dict with NO changes to those functions.
    """

    def __init__(self, d, G, b, l, aux_alpha=None):
        super().__init__(d, G, b, l, aux_alpha)
        # Deployment threshold in NORMALIZED block-norm units. `inited` lets
        # the FIRST training batch set threshold = tau directly: EMA-ing up
        # from the 0 init would drag the estimate low for hundreds of steps
        # and make early-eval gates spuriously dense.
        self.register_buffer("threshold", torch.zeros(()))
        self.register_buffer("inited", torch.tensor(False))
        self._thr_momentum = 0.99                # SAE-port parity (0.99 EMA)

    def _gate(self, pre_z):
        # DEPLOYMENT gate — unconditional (no self.training branch), because
        # encode() is the discovery/steering surface and batch-global
        # selection there would (a) make a token's code depend on which other
        # rows share its chunk and (b) mutate the threshold buffer via the
        # EMA (buffer ops run fine under torch.no_grad) on every encode call.
        if not bool(self.inited):
            # Never-seeded threshold (untrained / just-calibrated model): a 0
            # threshold would pass EVERY nonzero block (dense code); the
            # per-sample top-l gate is the correct l-sparse behavior absent
            # batch statistics. (VanillaBSF._gate honors bounded_scores.)
            return VanillaBSF._gate(self, pre_z)
        # Strictly greater than the EMA threshold, matching
        # BatchTopKSAEFeaturizer._gate's `> threshold` semantics at b=1.
        # Under bounded_scores both this comparison AND the training EMA
        # below run in normalized-score units — they must agree, or the
        # deployment gate would compare apples to a norm-unit threshold.
        sel = self._selection_scores(pre_z) > self.threshold
        return pre_z * sel.unsqueeze(-1), sel

    def _gate_train(self, pre_z):
        # Reached only from forward()/_dual_core. Eval-mode forwards (val R2
        # during training, ad-hoc scoring) still get the deploy gate.
        if not self.training:
            return self._gate(pre_z)
        if getattr(self, "bounded_scores", False):
            # Bounded-score gate: batch-global top-(N*l) on BOUNDED scores; the
            # reconstruction code stays the raw pre_z (see VanillaBSF).
            scores = self._selection_scores(pre_z).detach()
            N, G = scores.shape
            total = min(N * self.l, N * G)
            if 0 < total < N * G:
                tau = torch.topk(scores.reshape(-1), total,
                                 sorted=True).values[-1]
                sel = scores >= tau
            else:
                # Degenerate budget (l >= G or empty batch), mirroring
                # batch_block_topk's fallback; no meaningful tau to EMA.
                sel = scores > 0
                tau = None
            z = pre_z * sel.unsqueeze(-1)
        else:
            z, sel = batch_block_topk(pre_z, self.l)
            # Smallest kept block norm this batch (the official gate's tau).
            kept = z[sel]                                        # (K, b)
            tau = (kept.norm(dim=-1).min().detach()
                   if kept.numel() > 0 else None)
        # Update the deployment threshold from this batch's tau (detached:
        # threshold is a buffer, not a parameter — no gradient flows through
        # the EMA). Units follow the gate: norms (official) or bounded scores.
        if tau is not None:
            if not bool(self.inited):
                self.threshold.fill_(float(tau))
                self.inited.fill_(True)
            else:
                self.threshold.mul_(self._thr_momentum).add_(
                    (1 - self._thr_momentum) * tau)
        return z, sel

    @torch.no_grad()
    def calibrate_init(self, x_batch):
        # VanillaBSF scales the encoder by the LS output gain c; block norms
        # scale linearly with the encoder, so a threshold seeded BEFORE the
        # scaling would be stale by exactly 1/c. _ls_output_gain's encode()
        # now runs the side-effect-free deploy gate (per-sample top-l via the
        # not-inited fallback), so calibration itself no longer seeds the
        # EMA — but a caller may have run a training forward before
        # calibrating, so reset the seed anyway: the first real training
        # batch then re-seeds it on correctly-scaled norms (same idiom as
        # GroupLassoBSF.calibrate_init's bandwidth reset; calibrate_init is
        # an init-time call, so discarding a one-batch EMA costs nothing).
        super().calibrate_init(x_batch)
        self.threshold.zero_()
        self.inited.fill_(False)


class DualBatchBlockTopKBSF(BatchBlockTopKBSF, DualBSFMixin):
    _family = "batch_blocktopk"

    def __init__(self, d, G, b, l, aux_alpha=None, poly_hidden: int = 0,
                 lam_span: float = 0.01):
        super().__init__(d, G, b, l, aux_alpha)
        self._init_dual(poly_hidden, lam_span)

    def forward(self, x, target=None):
        # _dual_core gates through self._gate_train, so the dual runs the SAME
        # mode-switched gate (incl. the EMA threshold update at train time) as
        # its plain family — deployment parity holds by inheritance, like the
        # other duals (encode() stays the side-effect-free deploy gate).
        return self._dual_core(x, target)


# Registered by ASSIGNMENT (not by editing the dict literals above) so the
# original definitions stay textually untouched. build_bsf routes the plain
# variant through its VARIANTS[...] tail (no GroupLasso kwargs leak: that
# branch passes only (d, G, b, l, aux_alpha)) and the dual through the
# DUAL_BSF_VARIANTS branch (poly_hidden/lam_span only); load_bsf dispatches
# 'dual_batch_blocktopk' via the DUAL_BSF_VARIANTS membership check BEFORE the
# SAE fallback, exactly like the existing dual families.
VARIANTS["batch_blocktopk"] = BatchBlockTopKBSF
DUAL_BSF_VARIANTS["dual_batch_blocktopk"] = DualBatchBlockTopKBSF


# ---------------------------------------------------------------------------
# Checkpoint I/O — self-describing files so downstream stages (discovery,
# steering) can rebuild the exact model without re-plumbing hyperparameters.
# ---------------------------------------------------------------------------
def save_bsf(model: BSFBase, path, extra: dict | None = None) -> None:
    # Variant name: BSF variants resolve through the VARIANTS registry; SAE /
    # DualSAE featurizers (bsf/sae_models.py) self-report via variant_name()
    # and carry an extra_config payload so load_bsf can rebuild them without
    # the pipeline knowing which family a run belongs to.
    if hasattr(model, "variant_name"):
        variant = model.variant_name()
    else:
        variant = {v: k for k, v in VARIANTS.items()}[type(model)]
    payload = {
        "variant": variant,
        "d": model.d, "G": model.G, "b": model.b, "l": model.l,
        "aux_alpha": model.aux_alpha,
        # Group-Lasso config (None for the other variants / SAE families).
        "gl_coef": getattr(model, "coef", None),
        "gl_gain": getattr(model, "gain", None),
        "gl_paper_version": getattr(model, "paper_version", None),
        # Bounded-score gate flag: additive key; absent in older payloads -> False on
        # load, keeping every existing checkpoint byte-identical in behavior.
        "bounded_scores": bool(getattr(model, "bounded_scores", False)),
        # Step-norm buffers are registered lazily, so they travel as an
        # explicit payload field rather than relying on state_dict keys the
        # freshly-built model would reject as unexpected.
        "step_norm": ({"record_steps": model.step_ids.tolist(),
                       "mus": model.mu_steps.cpu(),
                       "scales": model.scale_steps.cpu()}
                      if model.has_step_norm else None),
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    # SAE / DualSAE / DualBSF self-description (absent for plain BSF variants
    # -> their payload keys are unchanged, so the frozen save/load round-trip
    # stays identical). "dual" covers both dual families: DualSAE featurizers
    # carry mono_mode, DualBSFs set dual=True in their extra_config.
    if hasattr(model, "extra_config"):
        payload["extra_config"] = model.extra_config()
        payload["dual"] = (getattr(model, "mono_mode", None) is not None
                           or bool(payload["extra_config"].get("dual")))
    torch.save(payload, path)


def load_bsf(path, device="cpu") -> BSFBase:
    """Rebuild a save_bsf checkpoint. Returned models are in EVAL mode:
    load_bsf's consumers are all deployment surfaces (evaluation,
    pruning, reports) and nn.Module's train-mode default previously let
    a mode-dependent gate (BatchBlockTopK) or a lazily-seeded buffer run
    training behavior on them. Resume paths don't use load_bsf (the trainers
    reload ckpt_last.pt state dicts directly), and any future training caller
    must opt in with .train() — the standard PyTorch contract."""
    payload = torch.load(path, map_location=device, weights_only=False)
    # Dual-BSF checkpoints MUST be dispatched before the SAE fallback below:
    # their variant names also start with "dual_", which sae_models.
    # build_from_payload would misinterpret as a DualSAE (b=1) and rebuild the
    # wrong architecture (pinned by test_dual_bsf.py::
    # test_dual_does_not_load_as_dualsae).
    if payload["variant"] in DUAL_BSF_VARIANTS:
        cfg = payload.get("extra_config", {})
        coef = payload.get("gl_coef")
        gain = payload.get("gl_gain")
        model = build_bsf(payload["variant"], payload["d"], payload["G"],
                          payload["b"], payload["l"], payload["aux_alpha"],
                          coef=1e-2 if coef is None else coef,
                          gain=10.0 if gain is None else gain,
                          paper_version=bool(payload.get("gl_paper_version")),
                          poly_hidden=cfg.get("poly_hidden", 0),
                          lam_span=cfg.get("lam_span", 0.01))
        sn = payload.get("step_norm")
        if sn is not None:
            # Register buffers BEFORE load_state_dict so the saved keys match.
            model.set_step_normalization(sn["record_steps"], sn["mus"],
                                         sn["scales"])
        model.load_state_dict(payload["state_dict"])
        # Provenance (not needed to rebuild, but keeps reloaded models
        # self-describing for reports).
        model.dual_warmup_steps = cfg.get("dual_warmup_steps", 0)
        # Bounded-score gate flag (absent in older payloads -> False = official gate).
        model.bounded_scores = bool(payload.get("bounded_scores", False))
        return model.to(device).eval()
    # Dispatch SAE / DualSAE featurizers (variant not in the BSF registry) to
    # their builder. Lazy import breaks the models <-> sae_models cycle.
    if payload["variant"] not in VARIANTS:
        from bsf.sae_models import build_from_payload
        model = build_from_payload(payload)
        sn = payload.get("step_norm")
        if sn is not None:
            model.set_step_normalization(sn["record_steps"], sn["mus"],
                                         sn["scales"])
        model.load_state_dict(payload["state_dict"])
        return model.to(device).eval()
    # Group-Lasso config (absent -> defaults; ignored by the other variants).
    coef = payload.get("gl_coef")
    gain = payload.get("gl_gain")
    paper = payload.get("gl_paper_version")
    model = build_bsf(payload["variant"], payload["d"], payload["G"],
                      payload["b"], payload["l"], payload["aux_alpha"],
                      coef=1e-2 if coef is None else coef,
                      gain=10.0 if gain is None else gain,
                      paper_version=bool(paper))
    sn = payload.get("step_norm")
    if sn is not None:
        # Register buffers BEFORE load_state_dict so the saved keys match.
        model.set_step_normalization(sn["record_steps"], sn["mus"], sn["scales"])
    model.load_state_dict(payload["state_dict"])
    # Bounded-score gate flag (absent in older payloads -> False = official gate).
    model.bounded_scores = bool(payload.get("bounded_scores", False))
    return model.to(device).eval()
