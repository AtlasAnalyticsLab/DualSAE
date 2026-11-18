"""MSAE — Matryoshka SAE with SPARSITY-LEVEL nesting (Zaigrajew, Baniecki &
Biecek, "Interpreting CLIP with Hierarchical Sparse Autoencoders", ICML 2025;
reference code WolodjaZ/MSAE).

Port of the math in cb_sae/bsf/sae_models.py
(MatryoshkaSAEFeaturizer + soft_cap/build_nesting/matryoshka_weights, which is
itself a documented port of the official implementation) onto this package's
BaseAutoencoder conventions, so it trains under train_other.py and
evaluates under cb_sae/featbsf/adapters.py unchanged.

The nesting is over VALUES OF k, NOT dictionary prefixes — the single most
misread thing about MSAE (one pre-code is pushed through a top-k gate at each
level; top-k_i(v) ⊆ top-k_{i+1}(v) by construction). One trained model can be
DEPLOYED at any level; this port deploys at the smallest level (k = the
suite's K, 64) so the evaluator scores exactly the sparsity-matched code and
the activation-pair partner is the plain TopK-SAE.

Faithfulness notes (deviations FLAGGED; A-M2/A-M3 are inherited verbatim from
the cb_sae/bsf/sae_models.py port and documented there at length):
- A-M2: per-level loss is plain per-element MSE (F.mse_loss) instead of the
  reference's normalized MSE — keeps the loss scale comparable with the
  package's TopKSAE under the shared constant-LR/grad-clip protocol.
- A-M3: SoftCapping (cap * tanh(x / cap), cap 30.0) is applied AFTER our ReLU
  rather than before; tanh is odd and monotone so on the kept (non-negative)
  support the two orders agree exactly.
- Weighting: 'rw' (reverse weights [3, 2, 1] for 3 levels — the HIGHEST
  weight on the SMALLEST k; the reference train.py comment claims the
  opposite but its code is authoritative, see matryoshka_weights). rw is the
  right default here because the deployed level IS the smallest k. 'uw'
  (uniform) is accepted and recorded in cfg for completeness.
- No auxiliary loss (the reference default aux_alpha = 0).
- b_dec init: per-dimension MEDIAN of the first batch (MSAE convention),
  buffer-guarded; the reference's decoder-grad projection + per-step renorm
  are provided by train_other.py's make_decoder_weights_and_grad_unit_norm.
- b_enc IS used (relu((x - b_dec) @ W_enc + b_enc)), matching the reference
  SAE and this package's VanillaSAE (TopKSAE-family omits it — existing
  asymmetry preserved on both sides).

cfg keys (beyond BaseAutoencoder's): 'top_k' (deploy level), 'nesting_list'
(the DISTINCTIVE key cb_sae/featbsf/adapters.py fingerprints on),
'relative_importance' (per-level weights), 'soft_cap' (float or None).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAutoencoder
from loss import compute_monosemanticity_loss_batch


def soft_cap(x: torch.Tensor, cap) -> torch.Tensor:
    """SoftCapping: cap * tanh(x / cap) (WolodjaZ/MSAE sae.py, cap=30.0).
    Bounds latent magnitude so the largest nesting level cannot let a handful
    of atoms blow up and dominate every level's reconstruction. cap=None (or
    <=0) disables it. Math identical to cb_sae/bsf/sae_models.py:soft_cap."""
    if cap is None or float(cap) <= 0:
        return x
    cap = float(cap)
    return cap * torch.tanh(x / cap)


def build_nesting(k: int, max_nesting: int) -> list:
    """Official nesting construction (WolodjaZ/MSAE train.py:223-235): start
    at k, double while below max_nesting, then ensure max_nesting is present.
    k=64, max_nesting=256 -> [64, 128, 256]. Identical to
    cb_sae/bsf/sae_models.py:build_nesting."""
    levels, cur = [], int(k)
    while cur < int(max_nesting):
        levels.append(cur)
        cur *= 2
    levels.append(int(max_nesting))
    return levels


def matryoshka_weights(n_levels: int, weighting: str) -> list:
    """Per-level loss weights (WolodjaZ/MSAE train.py:238-243).
    'rw' -> reversed(range(1, L+1)) = [3, 2, 1] for L=3: HIGHEST weight on the
    SMALLEST k (the reference's inline comment claims the opposite; the code
    is authoritative). 'uw' -> uniform. Identical to
    cb_sae/bsf/sae_models.py:matryoshka_weights."""
    if weighting == "rw":
        return [float(w) for w in reversed(range(1, n_levels + 1))]
    if weighting == "uw":
        return [1.0] * n_levels
    raise ValueError(f"unknown matryoshka weighting {weighting!r} (rw|uw)")


def topk_mask(z: torch.Tensor, k: int) -> torch.Tensor:
    """Per-sample top-k on a non-negative code (post-ReLU, so magnitude top-k
    == value top-k). Returns z with everything below the per-row k-th value
    zeroed."""
    if k <= 0 or k >= z.shape[1]:
        return z
    _, idx = torch.topk(z, k=k, dim=1, largest=True, sorted=False)
    mask = torch.zeros_like(z, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return z * mask


class MSAE(BaseAutoencoder):
    """Zaigrajew MSAE on the BaseAutoencoder chassis, deployed at the
    smallest nesting level."""

    def __init__(self, cfg):
        super().__init__(cfg)

        nesting = [int(v) for v in cfg["nesting_list"]]
        if sorted(nesting) != nesting or len(set(nesting)) != len(nesting):
            raise ValueError(f"nesting_list must be strictly increasing: "
                             f"{nesting}")
        weights = [float(w) for w in cfg.get(
            "relative_importance",
            matryoshka_weights(len(nesting), "rw"))]
        if len(weights) != len(nesting):
            raise ValueError(
                f"relative_importance {weights} must match nesting "
                f"{nesting} in length")

        # Deploy at the smallest level == the suite's sparsity budget K. A
        # mismatch would silently break the sparsity-matched comparison.
        self.deploy_k = int(cfg["top_k"])
        if self.deploy_k != nesting[0]:
            raise ValueError(
                f"deploy k (top_k={self.deploy_k}) must equal the smallest "
                f"nesting level {nesting[0]}")
        self.cap = cfg.get("soft_cap", 30.0)

        # Buffers: single source of truth at load time + the state-dict
        # fingerprint cb_sae/featbsf/adapters.py distinguishes this class by.
        self.register_buffer("nesting_levels",
                             torch.tensor(nesting, dtype=torch.long))
        self.register_buffer("level_weights",
                             torch.tensor(weights, dtype=torch.float32))
        self.register_buffer("_b_dec_initialized",
                             torch.tensor(0, dtype=torch.long))

        self.to(cfg["dtype"]).to(cfg["device"])

    @torch.no_grad()
    def maybe_init_b_dec(self, x: torch.Tensor):
        """Set b_dec to the per-dimension median of the first batch (MSAE
        convention; cb_sae/bsf/sae_models.py MSAE_BDEC_INIT='median'). No-op after
        the first call."""
        if int(self._b_dec_initialized.item()):
            return
        x_pre, _, _ = self.preprocess_input(x.to(self.b_dec.device))
        self.b_dec.data.copy_(
            x_pre.float().median(dim=0).values.to(self.b_dec.dtype))
        self._b_dec_initialized.fill_(1)

    def forward(self, x):
        x_preprocessed, x_mean, x_std = self.preprocess_input(x)

        x_cent = x_preprocessed - self.b_dec
        # A-M3: soft-cap AFTER ReLU (exact on the kept support, preserves the
        # non-negativity every downstream consumer relies on).
        acts = soft_cap(F.relu(x_cent @ self.W_enc + self.b_enc), self.cap)

        # Deploy gate: per-sample top-k at the SMALLEST nesting level — the
        # evaluator scores exactly this k=64 code (sparsity-matched to TopK).
        latents = topk_mask(acts, self.deploy_k)
        x_reconstruct = latents @ self.W_dec + self.b_dec

        self.update_inactive_features(latents)
        recons = self.postprocess_output(x_reconstruct, x_mean, x_std)

        self._x_preprocessed = x_preprocessed

        return acts, latents, recons

    def compute_loss(self, x: torch.Tensor, acts: torch.Tensor,
                     latents: torch.Tensor, recons: torch.Tensor,
                     mono_coef: float = 0.0):
        """Weighted mean of per-level reconstruction MSEs: for each nesting
        level k_l, gate the SAME pre-code at k_l, decode, score against the
        preprocessed input (A-M2 plain MSE). The deploy level reuses `latents`
        verbatim so its term is exactly the deployed reconstruction's error.
        """
        x_pre = (self._x_preprocessed if hasattr(self, "_x_preprocessed")
                 else x).float()
        levels = self.nesting_levels.tolist()
        w = self.level_weights.float()

        level_losses = []
        for i, k_l in enumerate(levels):
            z_l = latents if k_l == self.deploy_k else topk_mask(
                acts.float(), int(k_l))
            x_hat_l = z_l.float() @ self.W_dec.float() + self.b_dec.float()
            level_losses.append(F.mse_loss(x_hat_l, x_pre))
        stacked = torch.stack(level_losses)
        loss = (w * stacked).sum() / w.sum()

        l0_norm = (latents > 0).float().sum(-1).mean()
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        output = {
            "num_dead_features": num_dead_features.item(),
            "l1_sparsity": 0.0,
            # In-house convention: deploy-level recon error in original space.
            "l2_loss": (recons.float() - x.float()).pow(2).mean().item(),
            "num_active": l0_norm.item(),
        }
        for k_l, ll in zip(levels, level_losses):
            output[f"msae_l2_k{k_l}"] = ll.item()

        if mono_coef > 0:
            x_for_mono = (self._x_preprocessed
                          if hasattr(self, "_x_preprocessed") else x)
            mono_loss, mono_metrics = compute_monosemanticity_loss_batch(
                x_for_mono, latents)
            loss = loss + mono_coef * mono_loss
            output.update(mono_metrics)

        return loss, output
