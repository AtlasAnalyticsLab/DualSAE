"""Traditional SAE and DualSAE featurizers exposed through the exact same
interface as the Block-Sparse Featurizers (bsf/models.py:BSFBase).

Why subclass BSFBase instead of writing a parallel pipeline: an SAE is the
special case of a block-sparse featurizer with block dimension b = 1 and one
"block" per dictionary latent (G = n_latent). Under that identification a scalar
SAE feature g is a 1-D block whose code z_g in R^1 is the feature's activation,
and its decoder direction d_g in R^d is one dictionary atom. Every downstream
stage that consumes the BSFBase surface (encode -> (pre_z, z, sel) with z of
shape (N, G, b); the decoder tensor D of shape (G, b, d); the normalization
buffers) therefore works unchanged on an SAE.

In this release the classes serve as the b=1 carriers of converted
Matryoshka-SAE / Dual-Matryoshka-SAE checkpoints (featbsf/convert_plain_sae.py,
featbsf/convert_dualsae.py) on which the CB-SAE pipeline (dissect -> prune ->
concept-bottleneck training) runs. Each plain SAE is paired with the DualSAE
that uses the SAME Sparse-Branch (mono) activation.

Faithfulness: the encoder/decoder/activation math mirrors the repository's
own SAEs (vision/models/). Deviations are marked "ADAPTATION"; every
conversion is parity-checked against the native model before use.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from bsf.models import (BSFBase, BSFOutput, block_topk, block_norms,
                        runner_up_codes)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Sparsity operators. Each returns (z, sel_mask) in the BSF (N, G, 1) / (N, G)
# convention so BSFBase.encode's contract is preserved. For b = 1 the block
# l2-norm equals |z_g|, and because a ReLU precedes every gate z_g >= 0, "top-k
# by block norm" == "top-k by value" == the repo's topk_sparsify — the three
# selection rules coincide, so block_topk is reused verbatim for TopK.
# ---------------------------------------------------------------------------
def batch_topk_gate(pre_z: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """BatchTopK TRAINING gate: keep the B*k largest activations across the whole
    batch (Bussmann et al. 2024), not per-sample. pre_z: (N, G, 1) >= 0.

    Only valid at train time where a real batch exists; at deployment a single
    token is encoded, so BatchTopKSAE.encode uses a calibrated global threshold
    instead (the standard BatchTopK->threshold inference conversion)."""
    N, G, _ = pre_z.shape
    flat = pre_z.reshape(N, G)                                   # (N, G), >= 0
    total = min(N * k, N * G)
    if total <= 0 or total >= N * G:
        sel = flat > 0
        return pre_z * sel.unsqueeze(-1), sel
    thresh_vals, _ = torch.topk(flat.reshape(-1), total, sorted=True)
    tau = thresh_vals[-1]                                        # smallest kept
    sel = flat >= tau
    # A strict '>= tau' can keep marginally more than total on ties; harmless
    # (the loss just reconstructs from a few extra near-tau atoms).
    return pre_z * sel.unsqueeze(-1), sel


# ---------------------------------------------------------------------------
# MSAE (Matryoshka SAE) pieces — Zaigrajew, Baniecki & Biecek, "Interpreting
# CLIP with Hierarchical Sparse Autoencoders", ICML 2025 (WolodjaZ/MSAE).
# ---------------------------------------------------------------------------
def soft_cap(x: torch.Tensor, cap: float | None) -> torch.Tensor:
    """SoftCapping: cap * tanh(x / cap)  (WolodjaZ/MSAE sae.py:18, cap=30.0).

    Bounds latent magnitude so the largest nesting level cannot let a handful
    of atoms blow up and dominate every level's reconstruction.

    ADAPTATION (A-M3): we apply it AFTER our ReLU rather than before. tanh is
    odd and monotone, so on the kept (non-negative) support cap-then-ReLU and
    ReLU-then-cap agree exactly; doing it after preserves _pre_code's
    non-negativity contract, which block_topk relies on (see the note above).

    On our per-step-normalized substrate pre-codes are O(1) — the same fact
    JumpReLUSAEFeaturizer.calibrate_init documents — and
    30*tanh(x/30) = x*(1 - x^2/2700 + ...), so at |x| ~ 1 this is a ~0.04%
    perturbation: faithful to the reference AND numerically near-inert here.
    cap=None disables it.
    """
    return x if cap is None else cap * torch.tanh(x / cap)


def build_nesting(k: int, max_nesting: int) -> list[int]:
    """Official nesting construction (WolodjaZ/MSAE train.py:223-235): start at
    k, double while below max_nesting, then make sure max_nesting is present.

    k=64, max_nesting=256 -> [64, 128, 256].

    These are SPARSITY LEVELS (values of k), NOT dictionary prefixes — the
    single most misread thing about MSAE. One pre-code is pushed through a
    top-k gate at each level, so the levels are nested by construction:
    top-k_i(v) is a subset of top-k_{i+1}(v) for the same v.
    """
    levels, cur = [], int(k)
    while cur < int(max_nesting):
        levels.append(cur)
        cur *= 2
    levels.append(int(max_nesting))
    return levels


def matryoshka_weights(n_levels: int, weighting: str) -> list[float]:
    """Per-level loss weights (WolodjaZ/MSAE train.py:238-243).

    'rw' -> list(reversed(range(1, L+1))) = [3, 2, 1] for L=3, i.e. the HIGHEST
            weight on the SMALLEST k. train.py's inline comment claims the
            opposite ("higher weight for larger k values") but nesting_list is
            sorted ascending and reversed(range(1,L+1)) starts at L, so the
            CODE gives index 0 (smallest k) the largest weight. The code is
            authoritative; the comment is wrong.
    'uw' -> [1.0] * L, uniform.
    """
    if weighting == "rw":
        return [float(w) for w in reversed(range(1, n_levels + 1))]
    if weighting == "uw":
        return [1.0] * n_levels
    raise ValueError(f"unknown matryoshka weighting {weighting!r} (rw|uw)")


class _RectangleFn(autograd.Function):
    """Rectangle window with a straight-through estimator (GDM JumpReLU)."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        g = grad_output.clone()
        g[(x <= -0.5) | (x >= 0.5)] = 0
        return g


class _JumpReLUFn(autograd.Function):
    """JumpReLU: y = x * 1[x > exp(log_theta)] with STE gradients for the
    learnable per-feature threshold (verbatim math from
    vision/models/jumprelu_sae.py, reimplemented here to avoid importing the vision/
    package, whose module import pulls in `loss` and its heavy deps)."""
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(float(bandwidth)))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bw = ctx.saved_tensors
        bandwidth = bw.item()
        threshold = torch.exp(log_threshold)
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * _RectangleFn.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


class _StepFn(autograd.Function):
    """L0 surrogate: 1[x > threshold] with STE gradient on the threshold only."""
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(float(bandwidth)))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bw = ctx.saved_tensors
        bandwidth = bw.item()
        threshold = torch.exp(log_threshold)
        threshold_grad = (
            -(1.0 / bandwidth)
            * _RectangleFn.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return torch.zeros_like(x), threshold_grad, None


# ---------------------------------------------------------------------------
# Shared SAE featurizer skeleton (b = 1 block-sparse featurizer)
# ---------------------------------------------------------------------------
class SAEFeaturizer(BSFBase):
    """A ReLU sparse autoencoder presented as a b=1 block-sparse featurizer.

    Layout mapping to BSFBase:
      - G  = n_latent dictionary features (each a 1-D "block").
      - b  = 1.
      - D  (G, 1, d): the decoder dictionary, one atom per feature. BSFBase.D
            is reused directly, so decode()/block_contribution()/the Steerer's
            (z* - z_g) @ D[g] edit all work unchanged.
      - l  = k_sparsity, the active-feature budget (drives the TopK gate and is
            recorded so discovery/steering see the realized selection).

    Extra parameters beyond BSFBase (an SAE has an explicit encoder + biases):
      - enc: Linear(d -> G), the encoder W_E (+ latent bias b_enc).
      - b_dec (d,): the pre-decoder bias, subtracted before encoding and added
            back on reconstruction (standard SAE centering). Like mu, it CANCELS
            in the steering edit (a difference of block contributions), so it
            never appears in the Steerer.

    Deployment vs training gate: discovery/steering call encode() -> _gate()
    (the per-token DEPLOYMENT gate); forward() calls _gate_train() (the TRAINING
    gate). They differ only for BatchTopK (batch-global at train, threshold at
    deploy); for TopK/JumpReLU they are identical.
    """

    VARIANT_NAME: str = "sae"                 # overridden per subclass
    SPARSITY_MODE: str = "topk"               # topk | batch_topk | jump_relu
    constraint_cadence = 1                    # unit-norm decoder atoms every step
    # True iff the DEPLOYMENT gate decides each neuron independently of the
    # others, which is exactly the condition under which clone_pruned preserves
    # codes bit-for-bit (see clone_pruned's docstring).
    # False for the per-sample top-k gate, where deleting competitors promotes
    # retained neurons into the selection.
    PRUNE_EXACT: bool = False

    def __init__(self, d: int, n_latent: int, k: int, aux_alpha: float | None = 0.0):
        # aux_alpha defaults to 0 for SAEs: the repo's TopK/BatchTopK/DualSAE
        # train with reconstruction only (no Gao et al. AuxK). Kept as a knob so
        # an AuxK ablation can be switched on without touching the pipeline.
        super().__init__(d, G=n_latent, b=1, l=k, aux_alpha=aux_alpha)
        self.k = k
        # Encoder W_E (+ latent bias). Tied init W_E = D^T is the standard SAE
        # warm start (base.py): pre-code norms then have the right magnitude at
        # step 0, which suppresses dead features before any AuxK is involved.
        self.enc = nn.Linear(d, n_latent, bias=True)
        self.register_parameter("b_dec", nn.Parameter(torch.zeros(d)))
        with torch.no_grad():
            self.D.div_(self.D.norm(dim=(1, 2), keepdim=True).clamp_min(_EPS))
            self.enc.weight.copy_(self.D.reshape(n_latent, d))
            self.enc.bias.zero_()

    # -- encoder pieces -------------------------------------------------------
    def _pre_code(self, x_norm: torch.Tensor) -> torch.Tensor:
        """(N, d) normalized -> (N, G, 1) pre-gate code. ReLU makes every code
        non-negative so 'largest block norm' == 'largest activation'."""
        pre = F.relu(self.enc(x_norm - self.b_dec))             # (N, G)
        return pre.unsqueeze(-1)                                 # (N, G, 1)

    def _gate(self, pre_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Deployment gate (single-token). Default = per-sample TopK; BatchTopK
        overrides with a calibrated threshold, JumpReLU with its own gate."""
        return block_topk(pre_z, self.k)

    def _gate_train(self, pre_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Training gate; identical to _gate except for BatchTopK."""
        return self._gate(pre_z)

    # -- reconstruction / loss (no AuxK by default) ---------------------------
    def _reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """x_hat in normalized space = sum_g z_g d_g + b_dec."""
        return self.decode(z) + self.b_dec

    def forward(self, x: torch.Tensor,
                target: torch.Tensor | None = None) -> BSFOutput:
        # `target` mirrors BSFBase.forward (denoising): reconstruct target from
        # x. SAEs are not denoised in our pipeline, so target is None in
        # practice and tgt_norm == x_norm (behavior unchanged).
        x_norm = self.normalize(x)
        tgt_norm = x_norm if target is None else self.normalize(target)
        pre_z = self._pre_code(x_norm)
        z, sel = self._gate_train(pre_z)
        x_hat = self._reconstruct(z)
        recon = F.mse_loss(x_hat, tgt_norm)
        losses = {"recon": recon, "total": recon}
        if self.aux_alpha:                                      # optional AuxK
            from bsf.models import runner_up_codes
            r = (tgt_norm - x_hat).detach()
            z_aux = runner_up_codes(pre_z, sel, self.l)
            aux = F.mse_loss(self.decode(z_aux), r)
            losses["aux"] = aux
            losses["total"] = recon + self.aux_alpha * aux
        return BSFOutput(z=z, x_hat=x_hat, sel_mask=sel, losses=losses)

    # -- constraints / init ---------------------------------------------------
    @torch.no_grad()
    def project_constraints(self) -> None:
        # Unit-norm decoder atoms (||d_g|| = 1), the SAE analogue of the paper's
        # decoder constraint. Applied every step (constraint_cadence = 1), as in
        # the repo's normalize_decoder_columns_().
        norms = self.D.norm(dim=(1, 2), keepdim=True).clamp_min(_EPS)
        self.D.div_(norms)

    @torch.no_grad()
    def calibrate_init(self, x_batch: torch.Tensor) -> None:
        # Standard SAE init: b_dec := mean activation (in normalized space), so
        # the encoder sees centered inputs and reconstruction starts unbiased.
        x_norm = self.normalize(x_batch)
        self.b_dec.copy_(x_norm.mean(0))

    @torch.no_grad()
    def reinit_dead(self, x_norm: torch.Tensor, dead_mask: torch.Tensor) -> int:
        """Dead-latent resampling (E3; port of vision/train_dual.py). Re-point
        each dead latent's encoder row AND decoder atom at a random unit data
        direction (at 0.1 scale) so it can start firing again. Returns #reset.

        x_norm: (M, d) a NORMALIZED activation batch (the space the encoder acts
        in). dead_mask: (G,) bool over latents. enc.weight is (G, d) so dead
        latents are ROWS; D is (G, 1, d)."""
        n_dead = int(dead_mask.sum())
        if n_dead == 0:
            return 0
        # Random data vectors, unit-normalized (with replacement if the batch is
        # smaller than the dead set). randint keeps this device-safe.
        idx = torch.randint(0, x_norm.shape[0], (n_dead,), device=x_norm.device)
        s = F.normalize(x_norm[idx].float(), dim=-1)              # (n_dead, d)
        self.enc.weight.data[dead_mask] = s * 0.1
        self.enc.bias.data[dead_mask] = 0.0
        self.D.data[dead_mask] = s.unsqueeze(1) * 0.1            # (n_dead, 1, d)
        return n_dead

    # -- pruning (CB-SAE Step 3) ---------------------------------------------
    def _clone_ctor(self, n_latent: int) -> "SAEFeaturizer":
        """A fresh, same-class, same-hyperparameter model of a new width.

        Overridden by any subclass with extra constructor arguments. Keeping
        this next to __init__ (rather than dispatching on type() from
        cb_sae.py) is what stops the two from drifting apart: build_sae's tail
        already shows the failure mode, where unlisted kwargs are silently
        dropped and you get a DEFAULT-configured model back."""
        return type(self)(self.d, n_latent, self.k, aux_alpha=self.aux_alpha)

    @torch.no_grad()
    def _copy_pruned_extra(self, dst: "SAEFeaturizer",
                           keep: torch.Tensor) -> None:
        """Copy variant-specific gate state onto the pruned clone.

        Base class has none. Subclasses with gate state MUST override, and must
        be careful whether that state is per-feature (slice it with `keep`) or
        global (copy it whole)."""

    @torch.no_grad()
    def clone_pruned(self, keep: torch.Tensor) -> "SAEFeaturizer":
        """CB-SAE Step 3 (paper Eqs. 3-4): keep the retained neurons only.
        E' = E[keep, :], D' = D[keep], the shared biases unchanged.

        `keep` must be strictly ASCENDING: retained_idx.npy's block-id <->
        original-neuron mapping depends on pruned block j being the j-th
        smallest kept index (featbsf/cb_prune.py), and every consumer of
        retained_idx assumes it.

        CODE EXACTNESS IS NOT UNIVERSAL. It holds iff self.PRUNE_EXACT, i.e.
        iff the deployment gate is per-neuron independent (BatchTopK's global
        threshold, JumpReLU's per-feature thresholds): then for every input,
        pruned.encode(x)[:, j] == full.encode(x)[:, keep[j]] EXACTLY, which is
        what lets CLIP-Dissect labels I_m and steerability scores S_m computed
        on the full model carry over to pruned neurons.

        For a per-sample top-k gate (TopK, MSAE) it does NOT hold: pruning
        deletes G-R competitors, so a retained neuron that ranked k+1-th on
        some token is promoted into the pruned model's top-k. The pruned model
        is still a valid k-sparse model of the retained dictionary — it simply
        computes a different function. Measured on this substrate
        (bsf/prune_retention.py, L14 R=4096 of G=8192): only 40.6% of the full
        top-64 survives the prune, so a top-k gate refills 38 of 64 slots.
        Callers that depend on exactness must check PRUNE_EXACT; cb_prune
        records it (and the measured retention) in prune.json."""
        keep = keep.to(torch.long).cpu()
        assert keep.dim() == 1 and keep.numel() > 0, keep.shape
        assert bool((keep[1:] > keep[:-1]).all()), \
            "keep indices must be strictly ascending"
        assert 0 <= int(keep.min()) and int(keep.max()) < self.G, \
            f"keep out of range for G={self.G}"
        dst = self._clone_ctor(int(keep.numel()))
        dst.enc.weight.copy_(self.enc.weight[keep])
        dst.enc.bias.copy_(self.enc.bias[keep])
        dst.D.copy_(self.D[keep])
        # b_dec / mu / scale are SHARED (per-input, not per-latent), so they are
        # copied whole — slicing them would be a shape error at best and a
        # silent mis-centering at worst.
        dst.b_dec.copy_(self.b_dec)
        dst.mu.copy_(self.mu)
        dst.scale.copy_(self.scale)
        if self.has_step_norm:
            # Registered BEFORE any load_state_dict downstream, matching the
            # ordering load_bsf relies on.
            dst.set_step_normalization(self.step_ids.tolist(),
                                       self.mu_steps.clone(),
                                       self.scale_steps.clone())
        self._copy_pruned_extra(dst, keep)
        return dst.eval()

    # -- self-description for save/load --------------------------------------
    def variant_name(self) -> str:
        return self.VARIANT_NAME

    def extra_config(self) -> dict:
        return {"n_latent": self.G, "k": self.k, "aux_alpha": self.aux_alpha,
                "sparsity_mode": self.SPARSITY_MODE}


class TopKSAEFeaturizer(SAEFeaturizer):
    VARIANT_NAME = "topk_sae"
    SPARSITY_MODE = "topk"
    # Per-sample top-k: pruning promotes retained runners-up into the
    # selection, so codes are NOT preserved. See SAEFeaturizer.clone_pruned.
    PRUNE_EXACT = False


class BatchTopKSAEFeaturizer(SAEFeaturizer):
    """BatchTopK at train time, global-threshold at deploy time.

    The threshold buffer is an EMA of the per-batch smallest kept activation
    (Bussmann et al.'s recommended BatchTopK->threshold conversion); encode()
    (deployment) then gates each token independently with z_g > threshold, which
    is what discovery and steering need since they encode single tokens."""
    VARIANT_NAME = "batchtopk_sae"
    SPARSITY_MODE = "batch_topk"
    # Deployment gate is `pre_z > threshold`, a per-neuron-independent rule, so
    # pruning preserves the retained codes bit-for-bit. Overridden PER INSTANCE
    # when deploy_batch_global is set (see __init__).
    PRUNE_EXACT = True

    def __init__(self, d, n_latent, k, aux_alpha=0.0,
                 deploy_batch_global: bool = False):
        super().__init__(d, n_latent, k, aux_alpha)
        # threshold in NORMALIZED-pre-code units; 0 until first train step.
        self.register_buffer("threshold", torch.zeros(()))
        self._thr_momentum = 0.99
        # deploy_batch_global: gate with batch-global top-k at DEPLOY too,
        # instead of the EMA threshold. Default False, so every existing
        # checkpoint and every existing caller is bit-for-bit unchanged.
        #
        # Needed to carry a vision/models/matryoshka_sae.py checkpoint faithfully:
        # that model's FLAG A1 makes its eval gate a plain batch top-k in ALL
        # modes (the prefix structure lives only in the loss), so a port that
        # deployed the threshold gate would not be the same function as the row
        # it is supposed to sit under. DualSAEFeaturizer already has the same
        # flag for the same reason.
        self.deploy_batch_global = bool(deploy_batch_global)
        if self.deploy_batch_global:
            # Batch-global top-k ranks neurons AGAINST EACH OTHER, so deleting
            # G-R competitors promotes retained runners-up into the budget:
            # pruning is no longer an exact restriction. Set per instance so
            # cb_prune/prune_retention report the truth without any of them
            # needing to know about this flag.
            self.PRUNE_EXACT = False

    def _clone_ctor(self, n_latent):
        # Must carry the flag, or a pruned clone would silently switch gates.
        return type(self)(self.d, n_latent, self.k, aux_alpha=self.aux_alpha,
                          deploy_batch_global=self.deploy_batch_global)

    def extra_config(self):
        cfg = super().extra_config()
        cfg["deploy_batch_global"] = self.deploy_batch_global
        return cfg

    @torch.no_grad()
    def _copy_pruned_extra(self, dst, keep):
        # GLOBAL scalar, shared by every neuron -> copied whole, not sliced.
        # Copying it is what makes the pruned model's gate identical to the
        # full model's, hence the exactness in clone_pruned's docstring.
        dst.threshold.copy_(self.threshold)

    def _gate(self, pre_z):
        if self.deploy_batch_global:
            # Same batch-global gate at deploy as at train -- the vision/ eval
            # semantics (batch_topk_sparsify / matryoshka_sae FLAG A1). NOT
            # single-token safe by construction: the budget is k*N over the
            # batch, so encoding one row alone keeps k of it.
            return batch_topk_gate(pre_z, self.k)
        # Deployment: threshold gate (single-token safe).
        sel = pre_z.squeeze(-1) > self.threshold                # (N, G)
        return pre_z * sel.unsqueeze(-1), sel

    def _gate_train(self, pre_z):
        z, sel = batch_topk_gate(pre_z, self.k)
        # Update the deployment threshold from the smallest kept activation this
        # batch (detached; buffer, not a parameter).
        kept = z[sel]
        if kept.numel() > 0:
            batch_min = kept.min().detach()
            if float(self.threshold) == 0.0:
                self.threshold.fill_(float(batch_min))
            else:
                self.threshold.mul_(self._thr_momentum).add_(
                    (1 - self._thr_momentum) * batch_min)
        return z, sel


class JumpReLUSAEFeaturizer(SAEFeaturizer):
    """JumpReLU gate with per-feature learnable thresholds and an L0 penalty
    (vision/models/jumprelu_sae.py). Same gate at train and deploy."""
    VARIANT_NAME = "jumprelu_sae"
    SPARSITY_MODE = "jump_relu"
    # Each feature has its OWN threshold and is gated independently of the
    # others, so pruning preserves the retained codes bit-for-bit.
    PRUNE_EXACT = True

    def __init__(self, d, n_latent, k, aux_alpha=0.0, l1_coeff: float = 1e-4,
                 bandwidth: float = 1e-3, threshold_init: float = 1e-3):
        super().__init__(d, n_latent, k, aux_alpha)
        self.l1_coeff = l1_coeff
        self.bandwidth = bandwidth
        self.log_threshold = nn.Parameter(
            torch.full((n_latent,), math.log(threshold_init)))

    def _jump(self, pre: torch.Tensor) -> torch.Tensor:
        # pre: (N, G) >= 0 -> gated (N, G).
        return _JumpReLUFn.apply(pre, self.log_threshold, self.bandwidth)

    def _gate(self, pre_z):
        pre = pre_z.squeeze(-1)                                  # (N, G) >= 0
        gated = self._jump(pre)                                  # (N, G)
        sel = gated > 0
        return gated.unsqueeze(-1), sel

    @torch.no_grad()
    def calibrate_init(self, x_batch):
        # Data-scale adaptive JumpReLU init (ADAPTATION vs the repo's fixed
        # bandwidth=threshold_init=1e-3, which was tuned for L2-normalized CLIP
        # features with acts ~1e-2 and VANISHES on our per-step-normalized data
        # with acts ~O(1): the STE rectangle window ±bandwidth/2 then never
        # overlaps any activation, the threshold gets zero gradient, and the SAE
        # collapses to dense, L0~2378, R2~1.0). Fix: seed the threshold at the
        # per-token k-th-largest activation (so L0 starts ~k) and set bandwidth
        # to a fraction of it (so the STE window covers real activation density).
        super().calibrate_init(x_batch)                         # sets b_dec
        pre = torch.relu(self.enc(self.normalize(x_batch) - self.b_dec))
        kth = pre.topk(self.k, dim=1).values[:, -1]             # (N,)
        thr = float(kth.median().clamp_min(1e-4))
        self.log_threshold.fill_(math.log(thr))
        self.bandwidth = max(0.25 * thr, 1e-3)

    def forward(self, x, target=None):
        out = super().forward(x, target)                        # recon (+aux)
        # L0 penalty via the differentiable step surrogate on the gated code.
        pre = self._pre_code(self.normalize(x)).squeeze(-1)
        l0 = _StepFn.apply(pre, self.log_threshold, self.bandwidth).sum(-1).mean()
        lasso = self.l1_coeff * l0
        out.losses["l0"] = l0.detach()
        out.losses["lasso"] = lasso
        out.losses["total"] = out.losses["total"] + lasso
        return out

    def extra_config(self):
        cfg = super().extra_config()
        cfg.update({"l1_coeff": self.l1_coeff, "bandwidth": self.bandwidth})
        return cfg

    def _clone_ctor(self, n_latent):
        # l1_coeff/bandwidth are real ctor args; the base implementation would
        # silently rebuild with the defaults and change the gate's behaviour.
        return type(self)(self.d, n_latent, self.k, aux_alpha=self.aux_alpha,
                          l1_coeff=self.l1_coeff, bandwidth=self.bandwidth)

    @torch.no_grad()
    def _copy_pruned_extra(self, dst, keep):
        # PER-FEATURE Parameter (G,), so it must be SLICED, not copied whole:
        # a whole-tensor copy is a shape mismatch, and reusing the full tensor
        # would gate pruned neuron j by the threshold of original neuron j.
        dst.log_threshold.data.copy_(self.log_threshold.data[keep])


class MatryoshkaSAEFeaturizer(SAEFeaturizer):
    """MSAE — Matryoshka Sparse Autoencoder (Zaigrajew, Baniecki & Biecek,
    "Interpreting CLIP with Hierarchical Sparse Autoencoders", ICML 2025;
    reference implementation WolodjaZ/MSAE sae.py:637 MatryoshkaAutoencoder).

    THE NESTING IS OVER SPARSITY LEVELS, NOT DICTIONARY PREFIXES. One pre-code
    is pushed through L top-k gates at k in nesting_list ([64, 128, 256] here);
    each level is decoded, and the per-level reconstruction losses are combined
    by a weighted mean (train.py:347-360). Because top-k_i(v) is a subset of
    top-k_{i+1}(v) for the same v, the levels are genuinely nested — that is
    the "matryoshka", and it is what forces the most important features into
    the first slots so the model can be deployed at ANY k after training.

    (A BatchTopK-gated Matryoshka is NOT part of this method and is not built
    here — the reference ships activation="TopKReLU" only.)

    ADAPTATION A-M1 (deploy level): the reference's encode() returns the
    LARGEST level (k=256). We deploy `deploy_k`, defaulting to the SMALLEST
    (k=64), so the code is sparsity-matched against the k=64 BatchTopK
    baseline (comparison by activation pair). This is not a departure
    from the method — evaluating an MSAE at a chosen k is the reference's own
    protocol (encode(topk_number=...) at sae.py:711-716, and the whole point
    of score_topk_sae_embeddings.py), and it is how the Pareto frontier the
    paper claims is drawn. RW weighting also puts its largest weight on the
    smallest level, so the deployed level is the best-optimized one.

    ADAPTATION A-M2: per-level loss is plain F.mse_loss rather than the
    reference's normalized MSE. Our shards are already per-step normalized, so
    the per-sample 1/||x||^2 reweighting has little to bite on, it is identical
    across levels (so it cannot change the relative weighting that the
    Matryoshka contribution is about), and plain MSE keeps heldout_r2 and
    final_recon directly comparable to the BatchTopK baseline's 0.9041/0.0938.
    """

    VARIANT_NAME = "matryoshka_sae"            # overridden by the rw/uw pair
    SPARSITY_MODE = "topk"                     # per-sample gate
    WEIGHTING: str = "rw"                      # overridden by the rw/uw pair
    # Per-sample top-k -> pruning is not code-preserving. Measured on this
    # substrate: only 40.6% of the full top-64 survives an R=4096 of G=8192
    # prune (bsf/prune_retention.py). See SAEFeaturizer.clone_pruned.
    PRUNE_EXACT = False

    def __init__(self, d, n_latent, k, aux_alpha=0.0, *,
                 nesting_list: list[int] | None = None,
                 max_nesting: int = 256,
                 deploy_k: int | None = None,
                 relative_importance: list[float] | None = None,
                 latent_soft_cap: float | None = 30.0,
                 decoder_grad_proj: bool = True,
                 bdec_init: str = "median"):
        levels = list(nesting_list) if nesting_list else build_nesting(k, max_nesting)
        assert levels == sorted(levels) and len(set(levels)) == len(levels), \
            f"nesting_list must be strictly ascending, got {levels}"
        assert max(levels) <= n_latent, \
            f"largest nesting level {max(levels)} exceeds n_latent {n_latent}"
        deploy = int(deploy_k) if deploy_k else min(levels)
        assert 1 <= deploy <= n_latent, deploy
        # SAEFeaturizer sets self.k = self.l = deploy, so every downstream
        # consumer (CBSAEFeaturizer's l = sae.k + k_cb, discovery, paper_eval)
        # sees the budget the model is actually DEPLOYED at, not the largest
        # level it was trained against.
        super().__init__(d, n_latent, deploy, aux_alpha)
        self.nesting_list = levels
        self.deploy_k = deploy
        self.relative_importance = (list(relative_importance)
                                    if relative_importance
                                    else matryoshka_weights(len(levels), self.WEIGHTING))
        assert len(self.relative_importance) == len(levels), \
            (len(self.relative_importance), len(levels))
        self.latent_soft_cap = latent_soft_cap
        self.decoder_grad_proj = bool(decoder_grad_proj)
        self.bdec_init = bdec_init

    # -- encoder: ReLU then SoftCapping --------------------------------------
    def _pre_code(self, x_norm):
        return soft_cap(super()._pre_code(x_norm), self.latent_soft_cap)

    # -- gates ----------------------------------------------------------------
    def _level_gate(self, pre_z, k):
        """Top-k at one nesting level. block_topk is reused verbatim: at b=1
        with a ReLU in front, top-k by block norm == top-k by value."""
        return block_topk(pre_z, min(int(k), self.G))

    def _gate(self, pre_z):
        """Deployment gate: the single deploy level. Side-effect free, as
        BSFBase.encode's contract requires."""
        return self._level_gate(pre_z, self.deploy_k)

    def _gate_train(self, pre_z):
        # Same gate; forward() below overrides the loss to cover every level.
        return self._gate(pre_z)

    def set_deploy_k(self, k: int) -> None:
        """Re-point the deployment level. This is MSAE's defining capability —
        one trained model serves every sparsity budget (the reference exposes
        it as encode(topk_number=...)) — and it is what lets a PRUNED MSAE be
        evaluated at the baseline's realized L0 for a sparsity-matched
        comparison, as well as at its own full budget."""
        assert 1 <= int(k) <= self.G, k
        self.deploy_k = int(k)
        self.k = int(k)
        self.l = int(k)

    def level_reconstructions(self, x_norm) -> list[torch.Tensor]:
        """Per-level x_hat, for reporting held-out R^2 at every level — that
        curve IS the Pareto frontier the paper claims, so it is reported per
        level and never averaged."""
        pre_z = self._pre_code(x_norm)
        return [self._reconstruct(self._level_gate(pre_z, k)[0])
                for k in self.nesting_list]

    # -- MSAE loss (train.py:347-360) -----------------------------------------
    def forward(self, x, target=None):
        x_norm = self.normalize(x)
        tgt_norm = x_norm if target is None else self.normalize(target)
        pre_z = self._pre_code(x_norm)

        losses, num = {}, None
        z_deploy = sel_deploy = x_hat_deploy = None
        for k_i, w_i in zip(self.nesting_list, self.relative_importance):
            z_i, sel_i = self._level_gate(pre_z, k_i)
            xh_i = self._reconstruct(z_i)
            r_i = F.mse_loss(xh_i, tgt_norm)
            losses[f"recon_k{k_i}"] = r_i.detach()
            num = r_i * w_i if num is None else num + r_i * w_i
            if k_i == self.deploy_k:
                z_deploy, sel_deploy, x_hat_deploy = z_i, sel_i, xh_i
        # Weighted MEAN (not sum): keeps the loss scale comparable to a
        # single-level SAE's, so the existing LR schedule transfers unchanged.
        total = num / float(sum(self.relative_importance))

        if z_deploy is None:
            # deploy_k need not be one of the trained levels (e.g. a pruned
            # model evaluated at the baseline's realized L0). Gate it here so
            # BSFOutput still describes the DEPLOYED model.
            z_deploy, sel_deploy = self._gate(pre_z)
            x_hat_deploy = self._reconstruct(z_deploy)
            losses[f"recon_k{self.deploy_k}"] = F.mse_loss(x_hat_deploy, tgt_norm).detach()
        # "recon" is the DEPLOY level, so metrics.json / TensorBoard /
        # train.heldout_r2 all report the model as it will actually be used.
        losses["recon"] = losses[f"recon_k{self.deploy_k}"]

        if self.aux_alpha:                       # off by default (faithful)
            from bsf.models import runner_up_codes
            r = (tgt_norm - x_hat_deploy).detach()
            # Runners-up are ranked against the LARGEST level's selection: any
            # latent inside it already receives gradient from that level, so
            # ranking against the deploy selection would double-count.
            _, sel_max = self._level_gate(pre_z, self.nesting_list[-1])
            z_aux = runner_up_codes(pre_z, sel_max, self.l)
            aux = F.mse_loss(self.decode(z_aux), r)
            losses["aux"] = aux
            total = total + self.aux_alpha * aux
        losses["total"] = total
        return BSFOutput(z=z_deploy, x_hat=x_hat_deploy, sel_mask=sel_deploy,
                         losses=losses)

    # -- official training machinery -----------------------------------------
    @torch.no_grad()
    def project_decoder_grads(self) -> None:
        """MSAE's project_grads_decode(), called every step: remove the
        component of each decoder atom's gradient that is parallel to the atom
        itself. project_constraints() re-normalizes D every step anyway
        (constraint_cadence=1), so the radial displacement is already undone —
        this additionally keeps it out of Adam's second-moment accumulation.
        Normalizes defensively so the very first step is correct too."""
        if not self.decoder_grad_proj or self.D.grad is None:
            return
        dn = self.D / self.D.norm(dim=(1, 2), keepdim=True).clamp_min(_EPS)
        self.D.grad.sub_((self.D.grad * dn).sum(dim=(1, 2), keepdim=True) * dn)

    @torch.no_grad()
    def calibrate_init(self, x_batch):
        """MSAE initializes b_dec at the data MEDIAN (our other SAEs use the
        mean). Warm start only, but kept faithful and recorded."""
        if self.bdec_init != "median":
            return super().calibrate_init(x_batch)
        self.b_dec.copy_(self.normalize(x_batch).median(dim=0).values)

    # -- save/load ------------------------------------------------------------
    def _clone_ctor(self, n_latent):
        return type(self)(self.d, n_latent, self.k, aux_alpha=self.aux_alpha,
                          nesting_list=list(self.nesting_list),
                          deploy_k=self.deploy_k,
                          relative_importance=list(self.relative_importance),
                          latent_soft_cap=self.latent_soft_cap,
                          decoder_grad_proj=self.decoder_grad_proj,
                          bdec_init=self.bdec_init)

    def extra_config(self):
        cfg = super().extra_config()
        cfg.update({"nesting_list": list(self.nesting_list),
                    "deploy_k": self.deploy_k,
                    "relative_importance": list(self.relative_importance),
                    "latent_soft_cap": self.latent_soft_cap,
                    "decoder_grad_proj": self.decoder_grad_proj,
                    "bdec_init": self.bdec_init,
                    "weighting": self.WEIGHTING})
        return cfg


class MatryoshkaRWSAEFeaturizer(MatryoshkaSAEFeaturizer):
    """MSAE_RW — reverse weighting [L, ..., 1], largest weight on the smallest
    k (WolodjaZ/MSAE config.py:197-203)."""
    VARIANT_NAME = "matryoshka_rw_sae"
    WEIGHTING = "rw"


class MatryoshkaUWSAEFeaturizer(MatryoshkaSAEFeaturizer):
    """MSAE_UW — uniform weighting [1, ..., 1] (WolodjaZ/MSAE
    config.py:170-176)."""
    VARIANT_NAME = "matryoshka_uw_sae"
    WEIGHTING = "uw"


# ---------------------------------------------------------------------------
# DualSAE featurizers: a mono branch (one of the three SAEs above) + a poly
# residual MLP. Only the MONO branch is a steerable dictionary; encode()/D
# expose the mono features exactly as the plain SAE does, so steering a
# Dual-<act> model is byte-for-byte the same operation as steering its paired
# plain <act> SAE. The poly branch exists only to absorb polysemantic residual
# during training (span penalty), and is unused at discovery/steering time.
# ---------------------------------------------------------------------------
def _span_overlap_penalty(D: torch.Tensor, r_hat: torch.Tensor, z: torch.Tensor,
                          active_only: bool = True) -> torch.Tensor:
    """Penalize overlap of the poly output r_hat with span(decoder) (repo
    vision/models/dual_sae.py). D: (G, 1, d); r_hat: (N, d);
    z: (N, G, 1). overlap = r_hat . d_g for every atom g."""
    W = D.reshape(D.shape[0], D.shape[2])                       # (G, d)
    overlap = r_hat @ W.t()                                     # (N, G)
    if active_only:
        mask = (z.reshape(z.shape[0], z.shape[1]) != 0).float()
        overlap = overlap * mask
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (overlap.pow(2).sum(dim=1, keepdim=True) / denom).mean()
    return overlap.pow(2).mean()


class DualSAEFeaturizer(SAEFeaturizer):
    """DualSAE: mono reconstruction xM = decode(z) + b_dec, residual r = x - xM,
    poly correction r_hat = Poly(r), final x_hat = xM + r_hat, trained with
    recon + lam_span * span_overlap(r_hat, span D). Mirrors the repo DualSAE
    (vision/models/dual_sae.py) with a selectable mono
    activation.

    Extended for the CB pipeline on converted DualSAE checkpoints (the b=1
    carriers produced by featbsf/convert_dualsae.py):
      * three more mono modes — 'vanilla' (no gate, dense ReLU code),
        'matryoshka' (Bussmann prefix objective; its GATE is batch-global
        top-k, all that matters for a frozen CB base) and 'msae' (Zaigrajew
        soft-capped code, per-sample top-k at the deploy level). The
        multi-level LOSSES are deliberately NOT ported: converted bases are
        frozen under the CB head, so only pre-code + deploy gate semantics
        must match vision/models/dual_sae.py — asserted batch-for-batch by the
        converter's parity gate.
      * deploy_batch_global — vision/main.py's DualSAE gates batch-globally at
        EVAL TIME too for batch_topk/matryoshka (batch_topk_sparsify), while
        this port's native batch_topk deploy is the EMA threshold. The
        converter sets this flag so a converted base reproduces the exact
        function the native model was scored with. Batch-size dependence is
        inherited from the source model, not introduced here.
      * input_unit_norm — sae DualSAE's per-sample LN preprocess (jump_relu
        protocol). normalize() becomes the LN and stashes the per-batch stats;
        denormalize() inverts with THOSE stats, so input-space metrics stay
        exact even though the global (mu, scale) affine cannot represent LN.
        The normalized-space CB quantities (holdout r2_*) are then LN-space
        numbers — flagged in the reports.
    """
    constraint_cadence = 1

    def __init__(self, d, n_latent, k, mono_mode: str = "topk",
                 poly_hidden: int = 0, lam_span: float = 0.01,
                 l1_coeff: float = 1e-4, bandwidth: float = 1e-3,
                 threshold_init: float = 1e-3, aux_alpha: float = 0.0,
                 latent_soft_cap: float = 30.0,
                 deploy_batch_global: bool = False,
                 input_unit_norm: bool = False):
        super().__init__(d, n_latent, k, aux_alpha)
        self.mono_mode = mono_mode
        self.latent_soft_cap = latent_soft_cap
        self.deploy_batch_global = deploy_batch_global
        self.input_unit_norm = input_unit_norm
        self.lam_span = lam_span
        self.l1_coeff = l1_coeff
        self.bandwidth = bandwidth
        # Poly residual projector. poly_hidden <= 0 -> the repo's legacy 1-layer
        # Linear(d,d)+GELU; else a d->h->d bottleneck MLP.
        if poly_hidden and poly_hidden > 0:
            self.poly = nn.Sequential(nn.Linear(d, poly_hidden), nn.GELU(),
                                      nn.Linear(poly_hidden, d))
        else:
            self.poly = nn.Sequential(nn.Linear(d, d), nn.GELU())
        # Two-phase warmup switch, mirroring bsf/models.py DualBSFMixin: while
        # True, forward() reconstructs with the MONO branch alone (poly never
        # called, span a true zero) — the exact paired plain-SAE objective the
        # canonical DualSAE trainers use during warmup (vision/train_dual.py
        # freezes poly + zeroes span; the topic-model DualSAE trainer reconstructs
        # mono-only). Default False = joint from step 0; a trainer toggles it
        # per step (converted CB bases are frozen and never toggle it). Plain attribute (not a buffer): it is trainer state, not
        # model state, and must not enter checkpoints.
        self.dual_warmup_active = False
        # Mono-branch activation machinery (BatchTopK threshold / JumpReLU thr).
        if mono_mode == "batch_topk":
            self.register_buffer("threshold", torch.zeros(()))
            self._thr_momentum = 0.99
        elif mono_mode == "jump_relu":
            self.log_threshold = nn.Parameter(
                torch.full((n_latent,), math.log(threshold_init)))

    # -- LN preprocess (input_unit_norm) --------------------------------------
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample LN when input_unit_norm (vision/models/base.py
        verbatim: rowmean/rowstd with the +1e-5 guard), else the BSFBase
        global affine. The stats are stashed so denormalize() can invert the
        SAME batch — exactly how sae DualSAE's postprocess_output works."""
        if not self.input_unit_norm:
            return super().normalize(x)
        self._ln_mean = x.mean(dim=-1, keepdim=True)
        x = x - self._ln_mean
        self._ln_std = x.std(dim=-1, keepdim=True)
        return x / (self._ln_std + 1e-5)

    def denormalize(self, x_hat: torch.Tensor) -> torch.Tensor:
        if not self.input_unit_norm:
            return super().denormalize(x_hat)
        if not hasattr(self, "_ln_mean"):
            raise RuntimeError(
                "denormalize() before normalize(): the LN inversion needs the "
                "per-batch stats of the batch being inverted")
        # BARE _ln_std, not (_ln_std + 1e-5): vision/models/base.py's
        # postprocess_output multiplies by x_std while preprocess divided by
        # (x_std + 1e-5) — an asymmetry the reported metrics were scored
        # under, reproduced verbatim for parity (the ~1e-5-relative round-trip
        # slack is the NATIVE model's own).
        return x_hat * self._ln_std + self._ln_mean

    # -- encoder --------------------------------------------------------------
    def _pre_code(self, x_norm):
        pre = super()._pre_code(x_norm)
        if self.mono_mode == "msae":
            # MSAE soft-caps the (non-negative) code before any gate; on the
            # kept support cap-after-ReLU == cap-before (tanh is odd+monotone).
            pre = soft_cap(pre, self.latent_soft_cap)
        return pre

    # Reuse the paired plain-SAE gate for the mono branch so a Dual-<act> and
    # its plain <act> encode IDENTICALLY (only the learned weights differ).
    def _gate(self, pre_z):
        if self.mono_mode == "batch_topk":
            if self.deploy_batch_global:
                # sae DualSAE eval semantics (batch_topk_sparsify): the same
                # batch-global gate at deploy as at train.
                return batch_topk_gate(pre_z, self.k)
            sel = pre_z.squeeze(-1) > self.threshold
            return pre_z * sel.unsqueeze(-1), sel
        if self.mono_mode == "jump_relu":
            pre = pre_z.squeeze(-1)
            gated = _JumpReLUFn.apply(pre, self.log_threshold, self.bandwidth)
            return gated.unsqueeze(-1), gated > 0
        if self.mono_mode == "vanilla":
            # No gate: the dense ReLU code IS the mono code (sae DualSAE
            # sparsity_mode='vanilla' applies no sparsity operator).
            sel = pre_z.squeeze(-1) > 0
            return pre_z, sel
        if self.mono_mode == "matryoshka":
            # The Bussmann dual's gate is batch-global top-k over the full
            # dictionary (vision/models/dual_sae.py matryoshka branch); the
            # prefix structure lives in the LOSS, which a frozen base never
            # computes.
            return batch_topk_gate(pre_z, self.k)
        # topk and msae: per-sample top-k at the deploy budget.
        return block_topk(pre_z, self.k)

    def _gate_train(self, pre_z):
        if self.mono_mode == "batch_topk":
            z, sel = batch_topk_gate(pre_z, self.k)
            kept = z[sel]
            if kept.numel() > 0:
                bmin = kept.min().detach()
                if float(self.threshold) == 0.0:
                    self.threshold.fill_(float(bmin))
                else:
                    self.threshold.mul_(self._thr_momentum).add_(
                        (1 - self._thr_momentum) * bmin)
            return z, sel
        return self._gate(pre_z)

    @torch.no_grad()
    def calibrate_init(self, x_batch):
        # b_dec via the SAE base; seed the jump_relu mono threshold + bandwidth
        # from data (same data-scale adaptation as JumpReLUSAEFeaturizer).
        super().calibrate_init(x_batch)
        if self.mono_mode == "jump_relu":
            pre = torch.relu(self.enc(self.normalize(x_batch) - self.b_dec))
            kth = pre.topk(self.k, dim=1).values[:, -1]
            thr = float(kth.median().clamp_min(1e-4))
            self.log_threshold.fill_(math.log(thr))
            self.bandwidth = max(0.25 * thr, 1e-3)

    def forward(self, x, target=None):
        x_norm = self.normalize(x)
        tgt_norm = x_norm if target is None else self.normalize(target)
        pre_z = self._pre_code(x_norm)
        z, sel = self._gate_train(pre_z)
        xM = self.decode(z) + self.b_dec                        # mono recon
        if self.dual_warmup_active:
            # Phase-1 warmup (see __init__): mono-only reconstruction. poly is
            # never called (its params sit frozen by the trainer) and span is
            # an exact zero, so loss/gradients match the paired plain SAE
            # during warmup (plus the jump_relu L0 below, which the plain
            # jumprelu SAE also carries). The phase is a pure function of the
            # trainer's step, so resume lands in the right phase automatically.
            x_hat = xM
            span = xM.new_zeros(())
        else:
            r = x_norm - xM
            r_hat = self.poly(r)                                # poly correction
            x_hat = xM + r_hat
            span = _span_overlap_penalty(self.D, r_hat, z, active_only=True)
        recon = F.mse_loss(x_hat, tgt_norm)
        total = recon + self.lam_span * span
        losses = {"recon": recon, "span": span}
        # AuxK on the MONO residual (E3 dead-feature mitigation): runner-up mono
        # latents explain what the selected mono atoms missed, keeping otherwise
        # dead mono dictionary atoms — the steerable branch — receiving gradient.
        # Off by default (aux_alpha 0 -> baseline DualSAE unchanged). Uses the
        # mono residual x_norm - xM (NOT the full x - x_hat): the poly branch is
        # not a steerable dictionary, so AuxK must target the mono reconstruction.
        if self.aux_alpha:
            z_aux = runner_up_codes(pre_z, sel, self.l)
            aux = F.mse_loss(self.decode(z_aux), (x_norm - xM).detach())
            total = total + self.aux_alpha * aux
            losses["aux"] = aux
        # JumpReLU mono branch keeps its own L0 pressure (sae DualSAE parity).
        if self.mono_mode == "jump_relu":
            l0 = _StepFn.apply(pre_z.squeeze(-1), self.log_threshold,
                               self.bandwidth).sum(-1).mean()
            losses["l0"] = l0.detach()
            total = total + self.l1_coeff * l0
        losses["total"] = total
        return BSFOutput(z=z, x_hat=x_hat, sel_mask=sel, losses=losses)

    @property
    def _poly_hidden(self) -> int:
        """Bottleneck width of the poly MLP (0 = legacy Linear(d,d)+GELU) —
        derived from the module so it can never drift from the weights.
        Exposed under the DualBSFMixin attribute name so CBDualBSF's
        extra_config (getattr(base, '_poly_hidden', 0)) records the true
        value for SAE-port bases instead of silently defaulting to 0."""
        return (self.poly[0].out_features
                if isinstance(self.poly[0], nn.Linear)
                and len(self.poly) == 3 else 0)

    # Prune exactness is a PER-MODE property for the dual port: jump_relu
    # (per-feature thresholds) and vanilla (no gate) decide each neuron
    # independently; EMA-threshold batch_topk too. Any top-k gate — per-sample
    # (topk, msae) or batch-global (matryoshka, deploy_batch_global) —
    # re-ranks after competitors are deleted, so codes are not preserved.
    @property
    def PRUNE_EXACT(self) -> bool:  # noqa: N802 (matches the class-attr name)
        if self.mono_mode in ("jump_relu", "vanilla"):
            return True
        if self.mono_mode == "batch_topk" and not self.deploy_batch_global:
            return True
        return False

    def _clone_ctor(self, n_latent):
        # EVERY ctor knob must ride along: the base implementation forwards
        # only (d, n, k, aux_alpha) and would silently rebuild a
        # default-configured dual (topk gate, no LN, no soft cap) — the exact
        # drift failure its own docstring warns about.
        return type(self)(self.d, n_latent, self.k,
                          mono_mode=self.mono_mode,
                          poly_hidden=self._poly_hidden,
                          lam_span=self.lam_span, l1_coeff=self.l1_coeff,
                          bandwidth=self.bandwidth,
                          aux_alpha=self.aux_alpha,
                          latent_soft_cap=self.latent_soft_cap,
                          deploy_batch_global=self.deploy_batch_global,
                          input_unit_norm=self.input_unit_norm)

    @torch.no_grad()
    def _copy_pruned_extra(self, dst, keep):
        # The poly branch is per-INPUT-dim, not per-latent: copied whole. This
        # carry is what makes SAEFeaturizer.clone_pruned the CB stage-4 poly
        # warm start for scalar duals (the exact analogue of
        # clone_pruned_bsf's poly.load_state_dict for DualBSF bases).
        dst.poly.load_state_dict(self.poly.state_dict())
        if self.mono_mode == "batch_topk":
            dst.threshold.copy_(self.threshold)      # global scalar
        elif self.mono_mode == "jump_relu":
            # Per-feature Parameter: SLICED, like JumpReLUSAEFeaturizer's.
            dst.log_threshold.data.copy_(self.log_threshold.data[keep])

    def variant_name(self) -> str:
        return f"dual_{self.mono_mode.replace('_', '')}_sae"

    def extra_config(self):
        return {"n_latent": self.G, "k": self.k, "aux_alpha": self.aux_alpha,
                "dual": True, "mono_mode": self.mono_mode,
                "poly_hidden": self._poly_hidden,
                "lam_span": self.lam_span, "l1_coeff": self.l1_coeff,
                "bandwidth": self.bandwidth,
                # Round-trip the newer extensions; defaults keep every
                # pre-existing payload rebuildable (build_from_payload uses
                # .get with the same defaults).
                "latent_soft_cap": self.latent_soft_cap,
                "deploy_batch_global": self.deploy_batch_global,
                "input_unit_norm": self.input_unit_norm}


# ---------------------------------------------------------------------------
# Registry + builders. run-name activation keys mirror the repo's mode strings.
# ---------------------------------------------------------------------------
PLAIN_VARIANTS = {
    "topk_sae": TopKSAEFeaturizer,
    "batchtopk_sae": BatchTopKSAEFeaturizer,
    "jumprelu_sae": JumpReLUSAEFeaturizer,
    "matryoshka_rw_sae": MatryoshkaRWSAEFeaturizer,
    "matryoshka_uw_sae": MatryoshkaUWSAEFeaturizer,
}
# MSAE variants, which take extra constructor kwargs and therefore need an
# explicit branch in build_sae (the generic tail below forwards aux_alpha only).
MATRYOSHKA_VARIANTS = ("matryoshka_rw_sae", "matryoshka_uw_sae")
# mono-mode string -> dual variant name (self-describing checkpoints).
# vanilla/matryoshka/msae: eval-only carriers for converted DualSAE checkpoints — see the DualSAEFeaturizer
# docstring for what is and is NOT ported.
DUAL_VARIANTS = {"topk", "batch_topk", "jump_relu",
                 "vanilla", "matryoshka", "msae"}


def build_sae(variant: str, d: int, n_latent: int, k: int, **kw) -> SAEFeaturizer:
    """variant in {topk_sae, batchtopk_sae, jumprelu_sae,
                   matryoshka_rw_sae, matryoshka_uw_sae,
                   dual_topk_sae, dual_batchtopk_sae, dual_jumprelu_sae}."""
    if variant in PLAIN_VARIANTS:
        cls = PLAIN_VARIANTS[variant]
        if variant == "jumprelu_sae":
            return cls(d, n_latent, k,
                       aux_alpha=kw.get("aux_alpha", 0.0),
                       l1_coeff=kw.get("l1_coeff", 1e-4),
                       bandwidth=kw.get("bandwidth", 1e-3))
        if variant in MATRYOSHKA_VARIANTS:
            # Explicit, because the generic tail below passes ONLY aux_alpha:
            # rebuilding a checkpoint through it would silently produce a
            # DEFAULT-configured MSAE (default nesting, default deploy level)
            # whose state_dict happens to have matching shapes.
            return cls(d, n_latent, k,
                       aux_alpha=kw.get("aux_alpha", 0.0),
                       nesting_list=kw.get("nesting_list"),
                       max_nesting=kw.get("max_nesting", 256),
                       deploy_k=kw.get("deploy_k"),
                       relative_importance=kw.get("relative_importance"),
                       latent_soft_cap=kw.get("latent_soft_cap", 30.0),
                       decoder_grad_proj=kw.get("decoder_grad_proj", True),
                       bdec_init=kw.get("bdec_init", "median"))
        if variant == "batchtopk_sae":
            # Explicit for the same reason MATRYOSHKA_VARIANTS is: the generic
            # tail below forwards ONLY aux_alpha, so a deploy_batch_global=True
            # request would be silently dropped and you would get a
            # threshold-gated model whose state_dict loads fine and whose codes
            # are simply a different function.
            return cls(d, n_latent, k, aux_alpha=kw.get("aux_alpha", 0.0),
                       deploy_batch_global=kw.get("deploy_batch_global", False))
        return cls(d, n_latent, k, aux_alpha=kw.get("aux_alpha", 0.0))
    if variant.startswith("dual_"):
        mono = {"dual_topk_sae": "topk",
                "dual_batchtopk_sae": "batch_topk",
                "dual_jumprelu_sae": "jump_relu",
                "dual_vanilla_sae": "vanilla",
                "dual_matryoshka_sae": "matryoshka",
                "dual_msae_sae": "msae"}[variant]
        return DualSAEFeaturizer(
            d, n_latent, k, mono_mode=mono,
            poly_hidden=kw.get("poly_hidden", 0),
            lam_span=kw.get("lam_span", 0.01),
            l1_coeff=kw.get("l1_coeff", 1e-4),
            bandwidth=kw.get("bandwidth", 1e-3),
            aux_alpha=kw.get("aux_alpha", 0.0),
            latent_soft_cap=kw.get("latent_soft_cap", 30.0),
            deploy_batch_global=kw.get("deploy_batch_global", False),
            input_unit_norm=kw.get("input_unit_norm", False))
    raise ValueError(f"unknown SAE variant {variant!r}")


# variant-name -> constructor mapping used by load path, tolerant to how the
# name was recorded (plain key or dual mono-mode key).
def build_from_payload(payload: dict) -> SAEFeaturizer:
    v = payload["variant"]
    # (The original module also dispatched two image-generation CB variants
    # here; neither is produced by the feature-space CB pipeline, so they are
    # not part of this release. Feature-space CB checkpoints are rebuilt by
    # bsf/cb_dual_bsf.py:load_cb_dual, which calls this function for the base.)
    if v in ("cb_sae", "cb_dual_feat"):
        raise ValueError(f"variant {v!r} is not part of this release")
    cfg = payload.get("extra_config", {})
    d, G, k = payload["d"], payload["G"], payload["l"]
    if payload.get("dual") or v.startswith("dual_"):
        return DualSAEFeaturizer(
            d, G, k, mono_mode=cfg.get("mono_mode", "topk"),
            poly_hidden=cfg.get("poly_hidden", 0),
            lam_span=cfg.get("lam_span", 0.01),
            l1_coeff=cfg.get("l1_coeff", 1e-4),
            bandwidth=cfg.get("bandwidth", 1e-3),
            aux_alpha=cfg.get("aux_alpha", 0.0),
            latent_soft_cap=cfg.get("latent_soft_cap", 30.0),
            deploy_batch_global=cfg.get("deploy_batch_global", False),
            input_unit_norm=cfg.get("input_unit_norm", False))
    # Drop keys that are already passed positionally / are descriptive-only, so
    # only real build_sae kwargs (aux_alpha, l1_coeff, bandwidth, ...) remain.
    # 'weighting' is descriptive: the MSAE subclass is chosen by `v`, and the
    # realized weights round-trip explicitly via 'relative_importance'.
    kw = {kk: vv for kk, vv in cfg.items()
          if kk not in ("n_latent", "k", "sparsity_mode", "mono_mode", "dual",
                        "weighting")}
    return build_sae(v, d, G, k, **kw)


SAE_VARIANT_NAMES = set(PLAIN_VARIANTS) | {
    "dual_topk_sae", "dual_batchtopk_sae", "dual_jumprelu_sae",
    "dual_vanilla_sae", "dual_matryoshka_sae", "dual_msae_sae"}
