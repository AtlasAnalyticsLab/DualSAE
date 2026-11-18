import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .base import BaseAutoencoder
from .jumprelu_sae import JumpReLU, StepFunction
from .matryoshka_sae import geometric_median
from .msae_sae import matryoshka_weights, soft_cap
from loss import compute_monosemanticity_loss_batch

# Accepted values of the projector-ablation config keys (see _build_poly_projector).
POLY_ARCHS = ("mlp", "none")
POLY_FINAL_ACTS = ("auto", "none", "gelu")


def topk_sparsify(z: torch.Tensor, k: int):
    """
    Keep top-k entries by absolute value per row; zero out the rest.
    z: (B, m)
    """
    if k <= 0 or k >= z.shape[1]:
        return z
    # indices of top-k magnitudes
    vals, idx = torch.topk(z.abs(), k=k, dim=1, largest=True, sorted=False)
    mask = torch.zeros_like(z, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return z * mask


def batch_topk_sparsify(z: torch.Tensor, k_per_sample: int):
    """
    Keep exactly B * k_per_sample entries (by absolute value) across the whole batch.
    z: (B, m)
    """
    if k_per_sample <= 0 or k_per_sample >= z.shape[1]:
        return z

    batch_size, num_features = z.shape
    total_k = min(batch_size * k_per_sample, batch_size * num_features)

    flat_abs = z.abs().reshape(-1)
    _, flat_idx = torch.topk(flat_abs, k=total_k, largest=True, sorted=False)
    flat_mask = torch.zeros_like(flat_abs, dtype=torch.bool)
    flat_mask.scatter_(0, flat_idx, True)
    mask = flat_mask.view(batch_size, num_features)
    return z * mask


def span_overlap_penalty(
    decoder_weight: torch.Tensor,
    r_hat: torch.Tensor,
    z: torch.Tensor = None,
    active_only: bool = False,
) -> torch.Tensor:
    """
    Compute the span overlap penalty.
    
    Discourages r_hat from overlapping with span(W_D).
    The overlap is the projection of r_hat onto the decoder columns.
    
    Args:
        decoder_weight: Decoder weight matrix shape (dict_size, act_size)
        r_hat: Residual reconstruction (B, act_size)
        z: Sparse latents (B, dict_size), used for active_only masking
        active_only: If True, only compute overlap for active latents
    
    Returns:
        penalty: Scalar overlap penalty
    """
    # decoder_weight shape: (dict_size, act_size)
    # r_hat shape: (B, act_size)
    # overlap will be: (B, dict_size)
    overlap = r_hat @ decoder_weight.t()  # (B, dict_size)
    
    if active_only and z is not None:
        mask = (z != 0).float()
        overlap = overlap * mask
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (overlap.pow(2).sum(dim=1, keepdim=True) / denom).mean()
    
    return overlap.pow(2).mean()


def l0_sparsity(z: torch.Tensor) -> torch.Tensor:
    """Compute L0 sparsity (mean number of non-zero activations)."""
    return (z != 0).float().sum(dim=1).mean()



def cosine_similarity_penalty(xM: torch.Tensor, r_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute cosine similarity between mono branch output (xM) and poly branch output (r_hat).
    We want to minimize this to encourage the two branches to learn complementary features.
    
    Args:
        xM: Mono branch reconstruction (B, D)
        r_hat: Poly branch residual reconstruction (B, D)
        eps: Small constant for numerical stability
    
    Returns:
        similarity: Mean cosine similarity (scalar to minimize)
    """
    # Normalize vectors
    xM_norm = F.normalize(xM, p=2, dim=1, eps=eps)
    r_hat_norm = F.normalize(r_hat, p=2, dim=1, eps=eps)
    
    # Compute cosine similarity per sample
    cos_sim = (xM_norm * r_hat_norm).sum(dim=1)
    
    # Return mean absolute similarity (penalize both positive and negative correlation)
    return cos_sim.abs().mean()


class ZeroPoly(nn.Module):
    """A residual branch that predicts nothing, so ``x_hat == xM``.

    This is how a *standard* SAE is expressed with DualSAE's exact sparse-branch
    code (poly_arch='none'). Running it through this class rather than through
    BatchTopKSAE/TopKSAE is the point: those differ from DualSAE's sparse branch
    in tying, per-sample normalization and the decoder constraint, so freezing
    one of them and attaching a residual branch would change more than whether
    the branches train jointly.

    Parameterless on purpose: the state dict then has no ``poly.*`` keys, which
    makes such a checkpoint impossible to strict-load as a model that has a real
    projector (the reverse silent failure of the GELU-only variants below).
    """

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(r)


class DualSAE(BaseAutoencoder):
    """
    Dual-branch Sparse Autoencoder.

    Architecture:
        1. Mono branch: x -> z (sparse) -> xM (monosemantic reconstruction)
        2. Residual: r = x - xM
        3. Poly branch: r -> r_hat (residual reconstruction via MLP)
        4. Final: x_hat = xM + r_hat
    
    The span penalty discourages r_hat from overlapping with span(W_D).
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.poly_hidden = cfg.get("poly_hidden", 512)
        self.poly_layers = cfg.get("poly_layers", 1)
        self.poly_hidden_dims = self._normalize_hidden_dims(cfg.get("poly_hidden_dims", None))
        # Projector-ablation knobs. Both keys are absent from every legacy
        # model_cfg, and these defaults rebuild exactly the legacy module.
        self.poly_arch = cfg.get("poly_arch", "mlp")
        self.poly_final_act = cfg.get("poly_final_act", "auto")
        if self.poly_arch not in POLY_ARCHS:
            raise ValueError(f"poly_arch must be one of {POLY_ARCHS}, got {self.poly_arch!r}")
        if self.poly_final_act not in POLY_FINAL_ACTS:
            raise ValueError(f"poly_final_act must be one of {POLY_FINAL_ACTS}, "
                             f"got {self.poly_final_act!r}")
        if self.poly_arch == "none" and self.poly_final_act != "auto":
            # There is no module for the activation to terminate, so a value
            # here would claim an architecture the run did not have.
            raise ValueError("poly_final_act has no meaning with poly_arch='none'")
        # Forces r_hat to exactly 0 without touching the projector (the Z arm's
        # warm-up). Deliberately a plain attribute, NOT a buffer: it is training
        # state, and adding it to the state dict would break strict loading of
        # every existing dual checkpoint. A loaded model always starts with it
        # off, which is why train_dual never lets a bypassed epoch stay "best".
        self.poly_bypass = False
        self.k_sparsity = cfg.get("top_k", 64)
        self.sparsity_mode = cfg.get("sparsity_mode", "topk")  # topk, batch_topk, vanilla, jump_relu
        self.l1_coeff = cfg.get("l1_coeff", 1e-4)
        self.bandwidth = cfg.get("bandwidth", 0.001)

        self.jumprelu = None
        if self.sparsity_mode == "jump_relu":
            self.jumprelu = JumpReLU(
                feature_size=self.cfg["dict_size"],
                bandwidth=self.bandwidth,
                device=self.cfg.get("device", "cpu"),
                threshold_init=self.cfg.get("jump_threshold_init", 1.0),
            )

        # --- Dual-Matryoshka modes (NOVEL — no official reference exists for
        # a Matryoshka mono branch inside DualSAE; every design choice is
        # flagged in _multilevel_recon_loss). The matryoshka structure enters
        # ONLY through the reconstruction loss: gating, warmup, dead-neuron
        # respawn, span/sim penalties and the optimizer schedule are DualSAE's
        # unchanged (no new parameters, buffers only — so the post-warmup
        # optimizer rebuild in train_dual.py is unaffected).
        # Buffers are registered ONLY for these modes so legacy dual
        # checkpoints keep their exact state-dict layout (strict-load
        # compatibility) — and they double as the state-dict fingerprint
        # cb_sae/featbsf/adapters.py identifies these modes by.
        self.soft_cap_val = None
        if self.sparsity_mode == "gated":
            # --- Dual-Gated (NOVEL composition; the gate itself is the
            # faithful Rajamanoharan et al. 2024 encoder, ported in
            # models/gated_sae.py, but no reference exists for a gated
            # mono branch inside DualSAE). The gate REPLACES the mono ReLU;
            # the poly branch, span penalty and warm-up are untouched.
            #
            # These four names exist in NO legacy dual checkpoint and are
            # registered only in this mode, so every pre-existing dual still
            # load_state_dict(..., strict=True)s -- the same discipline the
            # matryoshka/msae buffers follow below. They double as the
            # state-dict fingerprint for adapters.py / knn_lp_sparse.load_run.
            #
            # b_enc (allocated by BaseAutoencoder and UNUSED by every other
            # dual mode) becomes the gate bias here, exactly as in
            # gated_sae.py FLAG G7. It is already in train_dual.py's
            # post-warm-up optimizer rebuild; r_mag/mag_bias are NOT, and are
            # appended there explicitly -- omitting them would silently freeze
            # the magnitude head at warm-up end.
            #
            # DEVIATION vs GatedSAE [DUAL-GATED-INIT]: the vendored unit-norm
            # decoder init is deliberately NOT applied here. Every dual mode
            # inherits BaseAutoencoder's init followed by
            # normalize_decoder_columns_() at the end of this ctor, and giving
            # the gated pair a different init than its dual siblings would
            # make it the odd row in a table whose other five pairs share the
            # existing asymmetry. Consistency with the table wins; recorded.
            self.r_mag = nn.Parameter(torch.zeros(self.cfg["dict_size"]))
            self.mag_bias = nn.Parameter(torch.zeros(self.cfg["dict_size"]))
            self.register_buffer(
                "sparsity_warmup_steps",
                torch.tensor(int(self.cfg.get("sparsity_warmup_steps", 2000)),
                             dtype=torch.long))
            self.register_buffer("_gated_step",
                                 torch.tensor(0, dtype=torch.long))
        elif self.sparsity_mode == "matryoshka":
            # Bussmann-style dictionary-prefix groups over the mono dictionary
            # (gate = batch top-k, the same as sparsity_mode='batch_topk').
            group_sizes = [int(g) for g in self.cfg["group_sizes"]]
            if sum(group_sizes) != self.cfg["dict_size"]:
                raise ValueError(
                    f"group_sizes {group_sizes} must sum to dict_size "
                    f"{self.cfg['dict_size']}")
            self.register_buffer(
                "group_sizes", torch.tensor(group_sizes, dtype=torch.long))
            # Uniform group weights = the vendored trainer's default.
            n_groups = len(group_sizes)
            self.register_buffer(
                "level_weights",
                torch.full((n_groups,), 1.0 / n_groups, dtype=torch.float32))
            self.register_buffer("_b_dec_initialized",
                                 torch.tensor(0, dtype=torch.long))
        elif self.sparsity_mode == "msae":
            # Zaigrajew-style sparsity-level nesting over the mono code
            # (gate = per-sample top-k at the smallest level, like 'topk').
            nesting = [int(v) for v in self.cfg["nesting_list"]]
            weights = [float(w) for w in self.cfg.get(
                "relative_importance",
                matryoshka_weights(len(nesting), "rw"))]
            if self.k_sparsity != nesting[0]:
                raise ValueError(
                    f"deploy k (top_k={self.k_sparsity}) must equal the "
                    f"smallest nesting level {nesting[0]}")
            self.register_buffer("nesting_levels",
                                 torch.tensor(nesting, dtype=torch.long))
            self.register_buffer("level_weights",
                                 torch.tensor(weights, dtype=torch.float32))
            self.register_buffer("_b_dec_initialized",
                                 torch.tensor(0, dtype=torch.long))
            self.soft_cap_val = self.cfg.get("soft_cap", 30.0)

        # Poly branch (residual projector MLP)
        self.poly = self._build_poly_projector()
        
        # Initialize decoder columns to roughly unit norm
        self.normalize_decoder_columns_()

    def _normalize_hidden_dims(self, raw_dims) -> Optional[List[int]]:
        """Normalize optional hidden-dimension schedule for poly projector."""
        if raw_dims is None:
            return None

        if isinstance(raw_dims, str):
            stripped = raw_dims.strip()
            if not stripped:
                return None
            dims = [int(x.strip()) for x in stripped.split(",") if x.strip()]
        elif isinstance(raw_dims, (list, tuple)):
            dims = [int(x) for x in raw_dims]
        else:
            raise ValueError(f"Unsupported poly_hidden_dims type: {type(raw_dims)}")

        if any(d <= 0 for d in dims):
            raise ValueError(f"poly_hidden_dims values must be positive, got {dims}")
        return dims

    def _build_poly_projector(self) -> nn.Module:
        """Build the residual branch selected by poly_arch and poly_final_act.

        poly_final_act='auto' returns the legacy module untouched. The legacy
        paths DISAGREE about the last layer: the 1-layer path ends in GELU while
        the poly_layers>=2 and poly_hidden_dims paths end in Linear. A naive depth
        sweep would therefore change the output activation along with the depth,
        and that is what the explicit 'none'/'gelu' values let an ablation
        separate.

        The override only adds or drops a parameterless GELU after construction,
        so RNG consumption and ``poly.<i>`` parameter names match the legacy path.
        It also means two architectures can have IDENTICAL state dicts (R0/L1,
        B512/B512G), so a dropped config key would strict-load the wrong model
        without any error. A config round-trip test guards this.
        """
        if self.poly_arch == "none":
            return ZeroPoly()
        projector = self._build_mlp_projector()
        if self.poly_final_act == "auto":
            return projector
        layers = list(projector)
        ends_in_gelu = isinstance(layers[-1], nn.GELU)
        if self.poly_final_act == "none" and ends_in_gelu:
            layers.pop()
        elif self.poly_final_act == "gelu" and not ends_in_gelu:
            layers.append(nn.GELU())
        return nn.Sequential(*layers)

    def _build_mlp_projector(self) -> nn.Sequential:
        """Build residual projector with configurable depth and hidden size."""
        act_size = self.cfg["act_size"]

        # Preferred path: user-provided per-layer widths.
        if self.poly_hidden_dims:
            widths = [act_size] + self.poly_hidden_dims + [act_size]
            layers = []
            for i in range(len(widths) - 1):
                layers.append(nn.Linear(widths[i], widths[i + 1]))
                if i < len(widths) - 2:
                    layers.append(nn.GELU())
            return nn.Sequential(*layers)

        if self.poly_layers < 1:
            raise ValueError(f"poly_layers must be >= 1, got {self.poly_layers}")

        layers = []

        if self.poly_layers == 1:
            # Keep legacy behavior: one linear layer followed by GELU.
            layers.append(nn.Linear(act_size, act_size))
            layers.append(nn.GELU())
            return nn.Sequential(*layers)

        # First layer expands/contracts into poly hidden space.
        layers.append(nn.Linear(act_size, self.poly_hidden))
        layers.append(nn.GELU())

        # Optional intermediate hidden layers.
        for _ in range(self.poly_layers - 2):
            layers.append(nn.Linear(self.poly_hidden, self.poly_hidden))
            layers.append(nn.GELU())

        # Final projection back to residual dimensionality.
        layers.append(nn.Linear(self.poly_hidden, act_size))
        return nn.Sequential(*layers)
    
    def normalize_decoder_columns_(self, eps: float = 1e-8):
        """
        Normalize columns of decoder weight so the span penalty is well-scaled.
        PyTorch Linear weight shape: (out_features=n, in_features=m).
        Columns correspond to latent features.
        """
        with torch.no_grad():
            W = self.W_dec  # (act_size, dict_size)
            col_norms = W.norm(dim=0, keepdim=True).clamp_min(eps)  # (1, dict_size)
            self.W_dec.copy_(W / col_norms)

    @torch.no_grad()
    def maybe_init_b_dec(self, x: torch.Tensor):
        """Data-dependent b_dec init for the Matryoshka-family modes only
        (geometric median for 'matryoshka', per-dim median for 'msae' —
        matching their plain pair-mates). Deliberate NO-OP for every legacy
        mode so pre-existing dual behavior is bit-identical; buffer-guarded so
        it runs exactly once."""
        if self.sparsity_mode not in ("matryoshka", "msae"):
            return
        if int(self._b_dec_initialized.item()):
            return
        x_pre, _, _ = self.preprocess_input(x.to(self.b_dec.device))
        if self.sparsity_mode == "matryoshka":
            init = geometric_median(x_pre.float())
        else:
            init = x_pre.float().median(dim=0).values
        self.b_dec.data.copy_(init.to(self.b_dec.dtype))
        self._b_dec_initialized.fill_(1)

    def _multilevel_recon_loss(self, x: torch.Tensor, acts: torch.Tensor):
        """Dual-Matryoshka reconstruction term (NOVEL composition — flagged).

        Design: every level's target is the FULL embedding x, and every level
        shares the SAME poly output r_hat — so the DualSAE contract ("the
        reconstruction loss scores only the final embedding") holds at every
        level, and the level at the deploy sparsity reduces EXACTLY to the
        standard DualSAE l2 term (a single-level nesting IS plain DualSAE).
        During warmup the trainer freezes poly and zeroes span/sim, but the
        frozen-poly r_hat stays in every level's target — strictly consistent
        with how legacy duals warm up (their l2 also contains the frozen
        r_hat); this is deliberately NOT "plain matryoshka on the mono branch
        during warmup".

        FLAG (reduction): per-element .mean() — the in-house DualSAE
        convention — NOT the vendored Bussmann sum(-1).mean(). The vendored
        reduction would rescale the recon term by ~d and silently re-balance
        it against the span/sim/mono penalties whose coefficients are pinned
        by the paper protocol; keeping the in-house reduction preserves the
        DualSAE loss-scale contract (and the exact single-level reduction
        property above). Within-term structure is unchanged (all levels share
        d, weights are normalized).

        Rejected alternatives: per-level residuals fed to poly (L poly passes
        per step and no well-defined residual for the span penalty);
        matryoshka loss on xM alone (a direct mono-branch reconstruction
        target — exactly the DualSAE contract violation the bounded-gate work
        was done to avoid).
        """
        x = x.float()
        r_hat = self._r_hat.float()
        w = self.level_weights.float()
        level_losses = []
        if self.sparsity_mode == "matryoshka":
            # Cumulative dictionary-prefix mono recon + shared r_hat. `acts`
            # is the deploy (batch-top-k) code, so the LAST prefix equals xM
            # and its term equals the standard l2 exactly.
            sizes = self.group_sizes.tolist()
            w_dec_chunks = torch.split(self.W_dec.float(), sizes, dim=0)
            f_chunks = torch.split(acts.float(), sizes, dim=1)
            xM_prefix = torch.zeros_like(x) + self.b_dec.float()
            for i in range(len(sizes)):
                xM_prefix = xM_prefix + f_chunks[i] @ w_dec_chunks[i]
                rec = self.postprocess_output(
                    xM_prefix + r_hat, self._x_mean, self._x_std)
                level_losses.append((rec.float() - x).pow(2).mean())
        else:  # msae
            # Re-gate the shared pre-code at each nesting level; the deploy
            # level reuses `acts` verbatim (guarantees exact reduction).
            for k_l in self.nesting_levels.tolist():
                z_l = acts.float() if k_l == self.k_sparsity else \
                    topk_sparsify(self._a_pre_gate.float(), int(k_l))
                xM_l = z_l @ self.W_dec.float() + self.b_dec.float()
                rec = self.postprocess_output(
                    xM_l + r_hat, self._x_mean, self._x_std)
                level_losses.append((rec.float() - x).pow(2).mean())
        stacked = torch.stack(level_losses)
        return (w * stacked).sum() / w.sum(), level_losses
    
    def _gated_sparsity_scale(self) -> float:
        """Linear ramp of the gate L1 over sparsity_warmup_steps; 1.0 when
        disabled. Byte-for-byte the vendored get_sparsity_warmup_fn
        (dictionary_learning/trainers/trainer.py). Counted on its OWN buffer,
        separate from DualSAE's poly warm-up: the two schedules are unrelated
        and sharing a counter would couple them."""
        warm = int(self.sparsity_warmup_steps.item())
        if not warm:
            return 1.0
        return min(int(self._gated_step.item()) / warm, 1.0)

    def forward(self, x):
        """
        Forward pass through dual SAE.
        
        Args:
            x: input data (shape: [batch, n_inputs])
        
        Returns:
            acts: Activations (shape: [batch, dict_size])
            latents: Same as acts (shape: [batch, dict_size])
            recons: Final reconstruction x_hat (shape: [batch, n_inputs])
        """
        x_preprocessed, x_mean, x_std = self.preprocess_input(x)
        
        # Mono branch
        x_cent = x_preprocessed - self.b_dec
        latents_pre_act = x_cent @ self.W_enc
        acts = F.relu(latents_pre_act)
        
        # Apply sparsity operator based on mode
        if self.sparsity_mode == "batch_topk":
            acts = batch_topk_sparsify(acts, self.k_sparsity)
        elif self.sparsity_mode == "topk":
            acts = topk_sparsify(acts, self.k_sparsity)
        elif self.sparsity_mode == "vanilla":
            pass
        elif self.sparsity_mode == "jump_relu":
            acts = self.jumprelu(acts)
        elif self.sparsity_mode == "matryoshka":
            # Bussmann training gate: one batch top-k over the FULL
            # dictionary; the prefix structure lives in the loss only.
            self._a_pre_gate = acts
            acts = batch_topk_sparsify(acts, self.k_sparsity)
        elif self.sparsity_mode == "msae":
            # Soft-cap then per-sample top-k at the DEPLOY level (the
            # smallest nesting level); larger levels re-gate _a_pre_gate
            # inside the loss.
            acts = soft_cap(acts, self.soft_cap_val)
            self._a_pre_gate = acts
            acts = topk_sparsify(acts, self.k_sparsity)
        elif self.sparsity_mode == "gated":
            # Two heads over ONE pre-activation: latents_pre_act is already
            # x_cent @ W_enc, so no second matmul. The plain `acts` computed
            # above (a bare ReLU) is DISCARDED for this mode -- the gate, not
            # the ReLU, decides what fires.
            pi_gate = latents_pre_act + self.b_enc
            # ReLU(pi_gate), not the 0/1 mask: this is what the L1 and the
            # auxiliary term see in compute_loss (gdm.py's `return_gate`
            # output). Using the mask instead makes the L1 a gradient-free L0
            # count and the aux term a reconstruction from an all-ones code.
            self._gate_pre = F.relu(pi_gate)
            # exp() keeps the rescaling positive, so the two heads can differ
            # in magnitude but never disagree about a latent's sign.
            pi_mag = torch.exp(self.r_mag) * latents_pre_act + self.mag_bias
            acts = (pi_gate > 0).to(latents_pre_act.dtype) * F.relu(pi_mag)
        else:
            raise ValueError(
                f"Unknown sparsity_mode: {self.sparsity_mode}. "
                "Use one of: topk, batch_topk, vanilla, jump_relu, "
                "matryoshka, msae, gated"
            )
        
        # Monosemantic reconstruction
        xM = acts @ self.W_dec + self.b_dec
        
        # Residual
        r = x_preprocessed - xM  # (B, act_size)
        
        # Poly branch. poly_bypass is False for everything except the Z arm's
        # warm-up, where it removes the frozen random projector from the objective.
        r_hat = torch.zeros_like(r) if self.poly_bypass else self.poly(r)  # (B, act_size)

        # Final reconstruction
        x_reconstruct = xM + r_hat
        
        self.update_inactive_features(acts)
        recons = self.postprocess_output(x_reconstruct, x_mean, x_std)
        
        # Store intermediate values for loss computation
        self._x_preprocessed = x_preprocessed
        self._xM = xM
        self._r = r
        self._r_hat = r_hat
        self._x_mean = x_mean
        self._x_std = x_std
        
        # Return (acts, latents, recons) to match interface
        return latents_pre_act, acts, recons
    
    def compute_loss(
        self,
        x: torch.Tensor,
        acts: torch.Tensor,
        recons: torch.Tensor,
        span_coef: float = 0.01,  # paper default 0.01
        active_only_span: bool = True,
        sim_coef: float = 0.0,
        mono_coef: float = 0.0,
    ):
        """
        Compute loss for dual SAE.
        
        Loss = L2(recons) + span_penalty + similarity_penalty + mono_loss
        
        Args:
            x: Input features (B, D)
            acts: Activations (B, N)
            recons: Final reconstructions x_hat (B, D)
            span_coef: Coefficient for span overlap penalty
            active_only_span: Whether to compute span penalty only for active latents
            sim_coef: Coefficient for similarity penalty between xM and r_hat
            mono_coef: Coefficient for monosemanticity loss
        
        Returns:
            loss: Total loss scalar
            output: Dictionary of metrics for logging
        """
        # Reconstruction loss on final output. Matryoshka-family modes
        # replace ONLY this term with the multi-level composition (span/sim/
        # sparsity/mono terms below are shared, per the pair-rule contract).
        level_losses = None
        if self.sparsity_mode in ("matryoshka", "msae"):
            l2_loss, level_losses = self._multilevel_recon_loss(x, acts)
        else:
            l2_loss = (recons.float() - x.float()).pow(2).mean()

        # Span overlap penalty (primary loss component for dual SAE)
        r_hat = self._r_hat
        # xM = self._xM
        
        # Compute span overlap penalty
        span_loss = span_overlap_penalty(
            self.W_dec,  # (act_size, dict_size)
            r_hat,
            z=acts,
            active_only=active_only_span
        )
               
        # L0 sparsity metric
        l0_norm = l0_sparsity(acts)
        
        # Dead feature detection
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        
        # Mode-specific sparsity penalty on mono-branch activations.
        l1_norm = acts.float().abs().sum(-1).mean()
        l0_penalty = torch.tensor(0.0, device=x.device)
        l1_penalty = torch.tensor(0.0, device=x.device)
        gated_aux = torch.tensor(0.0, device=x.device)
        gated_scale = 1.0
        if self.sparsity_mode == "vanilla":
            l1_penalty = self.l1_coeff * l1_norm
        elif self.sparsity_mode == "gated":
            # Gated sparsity lands on the GATE path, never on the firing
            # magnitude -- that decoupling is the whole point of the
            # architecture (it removes the shrinkage an L1-on-magnitude
            # imposes). `_gate_pre` is ReLU(pi_gate), stashed by forward().
            gate = self._gate_pre.float()
            gated_scale = self._gated_sparsity_scale()
            # [GATED-DUAL-L1SCALE] Divide by act_size so that lambda means the
            # SAME thing here as it does in the plain GatedSAE.
            #
            # This is a FIX for a measured collapse, not a preference. DualSAE
            # reduces reconstruction with a per-element .mean() (l2_loss above,
            # and gated_aux below), whereas the vendored GatedSAE keeps the
            # reference's .pow(2).sum(-1).mean(). The L1, however, is
            # sum-over-units in BOTH. Multiplying the dual's loss by d shows
            # what that asymmetry costs:
            #
            #   d*loss = sum_d(recon) + d*span*span + (lambda*d)*||gate||_1
            #            + sum_d(aux)
            #
            # i.e. the dual at lambda behaved as the vendored model at
            # lambda_eff = lambda * act_size (768 or 1152). The two forces on
            # the Heaviside gate are L_sparse (closes it) and L_aux (opens it),
            # so dividing only the OPENING force by d slammed every gate shut
            # within a few hundred steps -- and a closed gate is absorbing,
            # because both terms see ReLU(pi_gate) = 0.
            #
            # Measured: every dual lambda tested spanned lambda_eff 5.8..272,
            # while the plain model at the SAME architecture goes L0 62.6 at
            # lambda 0.1, 9.1 at 0.3, 1.0 at 1.0 and 0.2 at 3.0. The whole dual
            # sweep sat beyond the point where the plain model is already dead,
            # which is why all 20 runs collapsed and why it looked architectural.
            #
            # Scaling the L1 (rather than switching recon/aux to sum-over-d) is
            # deliberate: span_coef/sim_coef/mono_coef are shared across every
            # dual sparsity_mode and are calibrated against the per-element
            # recon, so rescaling recon here would silently change what
            # span_coef means in gated mode only.
            l1_penalty = (self.l1_coeff
                          * gate.abs().sum(-1).mean() * gated_scale
                          / self.cfg["act_size"])
            # Auxiliary term: the gate path reconstructing the input on its
            # own, through a FROZEN copy of the mono decoder.
            #
            # DESIGN DECISION [GATED-DUAL-AUX] -- no reference exists, so the
            # choice is recorded here. The target is the input and the
            # reconstruction is MONO-ONLY: r_hat is deliberately excluded.
            #   * the term's job is to make ReLU(pi_gate) itself informative;
            #     letting the dense poly MLP into it would allow poly to
            #     rescue a degenerate gate and the term would constrain
            #     nothing;
            #   * poly is FROZEN during warm-up, so a poly-inclusive aux term
            #     would change meaning discontinuously at warm-up end, while a
            #     mono-only one is constant across the whole run -- exactly
            #     like the vendored formulation;
            #   * it preserves the DualSAE contract that only the l2 term
            #     scores the final embedding.
            # Rejected alternative: target x_hat_gate + r_hat.detach().
            #
            # REDUCTION [GATED-DUAL-RED]: per-element .mean(), not the
            # vendored .sum(-1).mean() used in gated_sae.py. DualSAE's l2 term
            # is per-element, so this preserves the reference's 1:1
            # L_recon:L_aux ratio while leaving span_coef's inherited grid on
            # its usual scale. The L1 keeps the vendored
            # sum-over-units-then-mean form (it is an L1 over UNITS, not over
            # feature dims, so the reduction question does not arise there).
            x_pre_g = (self._x_preprocessed
                       if hasattr(self, '_x_preprocessed') else x).float()
            x_hat_gate = (gate @ self.W_dec.detach().float()
                          + self.b_dec.detach().float())
            gated_aux = (x_pre_g - x_hat_gate).pow(2).mean()
            if self.training:
                self._gated_step += 1
        elif self.sparsity_mode == "jump_relu":
            if self.jumprelu is None:
                raise RuntimeError("JumpReLU module is not initialized for jump_relu sparsity mode.")
            l0 = StepFunction.apply(
                acts,
                self.jumprelu.log_threshold,
                self.bandwidth,
            ).sum(dim=-1).mean()
            l0_penalty = self.l1_coeff * l0
        sparsity_penalty = l1_penalty + l0_penalty

        # Total loss. gated_aux is exactly 0.0 for every non-gated mode, so
        # the expression is numerically identical for legacy runs.
        loss = l2_loss + span_coef * span_loss + sparsity_penalty + gated_aux
        
        output = {
            "l2_loss": l2_loss.item(),
            "span_loss": span_loss.item(),
            "num_active": l0_norm.item(),
            "num_dead_features": num_dead_features.item(),
            "l1_sparsity": l1_penalty.item(),
            "l0_sparsity": l0_penalty.item(),
            "sparsity_penalty": sparsity_penalty.item(),
            "l1_norm": l1_norm.item(),
        }
        if self.sparsity_mode == "gated":
            # gate_open_frac is the honest read on whether l1_coeff is in
            # range: it is the gate's density BEFORE the magnitude head's ReLU
            # can close a latent, so a run where it sits far above
            # num_active/dict_size is one whose two heads disagree.
            output["gated_aux_loss"] = gated_aux.item()
            output["gated_l1"] = self._gate_pre.float().abs().sum(-1).mean().item()
            output["sparsity_scale"] = gated_scale
            output["gate_open_frac"] = (self._gate_pre > 0).float().mean().item()
        # Per-level diagnostics for the Matryoshka-family modes (raw,
        # unweighted; the top/deploy level's value equals the standard l2).
        if level_losses is not None:
            if self.sparsity_mode == "matryoshka":
                for i, ll in enumerate(level_losses):
                    output[f"matryoshka_l2_g{i}"] = ll.item()
            else:
                for k_l, ll in zip(self.nesting_levels.tolist(),
                                   level_losses):
                    output[f"msae_l2_k{k_l}"] = ll.item()

        # Monosemanticity loss
        if mono_coef > 0:
            x_for_mono = self._x_preprocessed if hasattr(self, '_x_preprocessed') else x
            mono_loss, mono_metrics = compute_monosemanticity_loss_batch(x_for_mono, acts)
            loss = loss + mono_coef * mono_loss
            output.update(mono_metrics)
        
        return loss, output
