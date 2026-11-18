"""Matryoshka SAE (Bussmann-style dictionary-prefix nesting, BatchTopK-gated).

Faithful port of the vendored reference implementation at
dictionary_learning/ (saprmarks/dictionary_learning) trainers/matryoshka_batch_top_k.py
(MatryoshkaBatchTopKSAE + MatryoshkaBatchTopKTrainer loss/update math) onto
this package's BaseAutoencoder conventions (W_enc (d,U), W_dec (U,d), b_enc,
b_dec; forward -> (pre_acts, latents, recons); compute_loss(x, acts, latents,
recons, mono_coef) -> (loss, metrics)) so it trains under train_other.py
and evaluates under cb_sae/featbsf/adapters.py unchanged.

The dictionary is partitioned into contiguous GROUPS; the loss reconstructs x
from each dictionary PREFIX (groups 0..i) and sums the weighted per-prefix
errors, so early groups are forced to carry coarse, self-sufficient structure
(the Matryoshka property). Gating is a single batch top-k over the FULL
dictionary — identical to this package's BatchTopKSAE — so the activation-pair
comparison partner is batch_topk.

Faithfulness notes (every deliberate deviation FLAGGED):
- FLAG A1 (eval gate): the vendored forward gates with an EMA threshold at
  inference (use_threshold=True). We gate with batch top-k in ALL modes so the
  code the evaluator sees is the same kind of code BatchTopKSAE deploys (pair
  comparability, and no dependence on how well the threshold EMA converged).
  The threshold is still maintained and persisted exactly as upstream (EMA
  beta 0.999 from step > 1000) for fidelity / off-suite use.
- Loss reduction kept VENDORED: per-prefix (x - x_hat_i)^2.sum(-1).mean(),
  weighted, then MEAN over groups — about d x the in-house per-element-mean
  loss scale. Adam is invariant to a static loss scale and the vendored
  trainer clips grad-norm at 1.0 exactly as train_other.py does, so training
  dynamics match the reference; the logged 'l2_loss' metric key is still the
  in-house per-element convention (see compute_loss) for log comparability.
- AuxK is the vendored (Bussmann/OpenAI) form — dead-latent top-k decoded
  WITHOUT bias against the detached residual, normalized by the residual's
  variance, coefficient alpha (default 1/32) — NOT the in-house
  BatchTopKSAE.get_auxiliary_loss. Dead = "tokens since fired >= 10M" upstream;
  here the in-house per-batch counter num_batches_not_active >=
  n_batches_to_dead reproduces it exactly at n_batches_to_dead =
  dead_steps_threshold // batch_size (10M tokens / 2048 = 4882 batches; the
  trainer computes this the same way for every model).
- Decoder constraint: train_other.py's make_decoder_weights_and_grad_unit_norm
  (unit rows of W_dec (U,d) + removal of the parallel gradient component) is
  mathematically identical to the vendored set_decoder_norm_to_unit_norm +
  remove_gradient_parallel_to_decoder_directions pair, so no in-model handling
  is needed.

cfg keys (beyond BaseAutoencoder's): 'top_k' (batch top-k per-sample budget),
'group_sizes' (list[int], sums to dict_size — the DISTINCTIVE key
cb_sae/featbsf/adapters.py fingerprints on), 'aux_penalty' (auxk alpha).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAutoencoder
from loss import compute_monosemanticity_loss_batch

# Vendored trainer constants (matryoshka_batch_top_k.py:144-145,161-162).
_THRESHOLD_BETA = 0.999
_THRESHOLD_START_STEP = 1000


def geometric_median(points: torch.Tensor, max_iter: int = 100,
                     tol: float = 1e-5) -> torch.Tensor:
    """Weiszfeld iteration, verbatim from the vendored trainer (:374-388).
    Used to initialize b_dec from the first training batch."""
    guess = points.mean(dim=0)
    prev = torch.zeros_like(guess)
    weights = torch.ones(len(points), device=points.device)
    for _ in range(max_iter):
        prev = guess
        weights = 1 / torch.norm(points - guess, dim=1)
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break
    return guess


class MatryoshkaSAE(BaseAutoencoder):
    """Bussmann Matryoshka SAE on the BaseAutoencoder chassis."""

    def __init__(self, cfg):
        super().__init__(cfg)

        group_sizes = [int(g) for g in cfg["group_sizes"]]
        if sum(group_sizes) != cfg["dict_size"]:
            raise ValueError(
                f"group_sizes {group_sizes} must sum to dict_size "
                f"{cfg['dict_size']}")
        if any(g <= 0 for g in group_sizes):
            raise ValueError(f"group sizes must be positive: {group_sizes}")

        self.aux_alpha = float(cfg.get("aux_penalty", 1 / 32))

        # Buffers are the single source of truth at load time (a checkpoint
        # restores them, so partitioning can never desynchronize from the
        # weights) AND the state-dict fingerprint cb_sae/featbsf/adapters.py needs to
        # distinguish this class from VanillaSAE (identical parameter names).
        # 'k' is a buffer exactly as in the vendored model (:47) so an
        # untyped checkpoint still knows its own gate budget.
        self.register_buffer("k",
                             torch.tensor(int(cfg["top_k"]), dtype=torch.long))
        self.register_buffer("group_sizes",
                             torch.tensor(group_sizes, dtype=torch.long))
        # Group weights: uniform is the vendored trainer's default when
        # group_weights=None (:174-175). (apply_temperature exists upstream
        # but is never invoked by the trainer — no temperature knob here.)
        n_groups = len(group_sizes)
        self.register_buffer(
            "level_weights",
            torch.full((n_groups,), 1.0 / n_groups, dtype=torch.float32))
        # EMA threshold (vendored init -1.0 = "unset") + step counter driving
        # its start-up delay; persisted so a reloaded model keeps its state.
        self.register_buffer("threshold", torch.tensor(-1.0))
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))
        # Guard so the geometric-median b_dec init runs exactly once even if
        # the trainer hook fires again (idempotence documented for tests).
        self.register_buffer("_b_dec_initialized",
                             torch.tensor(0, dtype=torch.long))

        # Vendored init ORDER (:56-65), which differs from BaseAutoencoder's
        # (base inits W_enc then ties W_dec): kaiming W_dec, unit-norm its
        # rows (atoms), W_enc = W_dec.T. Biases stay zero.
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.W_dec)
            self.W_dec.data /= self.W_dec.data.norm(dim=-1, keepdim=True)
            self.W_enc.data.copy_(self.W_dec.data.t())

        self.to(cfg["dtype"]).to(cfg["device"])

    # -- b_dec initialization (vendored update() step-0 behavior, :322-324) --
    @torch.no_grad()
    def maybe_init_b_dec(self, x: torch.Tensor):
        """Set b_dec to the geometric median of the first batch (preprocessed
        space, where encode/decode live). No-op after the first call."""
        if int(self._b_dec_initialized.item()):
            return
        x_pre, _, _ = self.preprocess_input(x.to(self.b_dec.device))
        self.b_dec.data.copy_(geometric_median(x_pre.float())
                              .to(self.b_dec.dtype))
        self._b_dec_initialized.fill_(1)

    @torch.no_grad()
    def _update_threshold(self, gated: torch.Tensor):
        """EMA of the smallest positive gated activation (vendored
        update_threshold, :247-262). Maintained for fidelity/off-suite use;
        the suite's forward never gates with it (FLAG A1 above)."""
        active = gated[gated > 0]
        min_activation = (active.min().detach().float() if active.numel()
                          else torch.tensor(0.0, device=gated.device))
        if self.threshold < 0:
            self.threshold.fill_(float(min_activation))
        else:
            self.threshold.mul_(_THRESHOLD_BETA).add_(
                (1 - _THRESHOLD_BETA) * float(min_activation))

    def forward(self, x):
        x_preprocessed, x_mean, x_std = self.preprocess_input(x)

        # Vendored encode (:68): relu((x - b_dec) @ W_enc + b_enc). Note the
        # b_enc term — faithful to Bussmann even though this package's
        # BatchTopKSAE omits it (existing asymmetry, preserved on both sides).
        x_cent = x_preprocessed - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)

        # Batch top-k over the flattened batch (k * B kept), the vendored
        # use_threshold=False path == this package's BatchTopKSAE gate.
        acts_topk = torch.topk(acts.flatten(),
                               int(self.k.item()) * x.shape[0], dim=-1)
        latents = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )

        # Threshold EMA bookkeeping is a TRAINING-side effect (the vendored
        # trainer updates it inside loss()); eval forwards must not move it.
        if self.training:
            self._step += 1
            if int(self._step.item()) > _THRESHOLD_START_STEP:
                self._update_threshold(latents)

        x_reconstruct = latents @ self.W_dec + self.b_dec

        self.update_inactive_features(latents)
        recons = self.postprocess_output(x_reconstruct, x_mean, x_std)

        # Stash for compute_loss: prefix losses live in preprocessed space.
        self._x_preprocessed = x_preprocessed

        return acts, latents, recons

    def compute_loss(self, x: torch.Tensor, acts: torch.Tensor,
                     latents: torch.Tensor, recons: torch.Tensor,
                     mono_coef: float = 0.0):
        """Vendored Matryoshka loss (trainer.loss, :264-303): cumulative
        dictionary-prefix reconstructions, per-prefix weighted
        (x - x_hat_i)^2.sum(-1).mean(), MEAN over groups, + alpha * AuxK.

        `acts` is the post-ReLU pre-gate code (AuxK reads it), `latents` the
        batch-top-k gated code (the prefix reconstructions read it).
        """
        x_pre = (self._x_preprocessed if hasattr(self, "_x_preprocessed")
                 else x).float()
        sizes = self.group_sizes.tolist()
        w = self.level_weights

        # Cumulative prefix reconstruction: x_hat_0 = b_dec, then add one
        # group's contribution at a time (vendored :273-288).
        x_hat = torch.zeros_like(x_pre) + self.b_dec
        weighted_losses = []
        group_l2 = []
        w_dec_chunks = torch.split(self.W_dec, sizes, dim=0)
        f_chunks = torch.split(latents.float(), sizes, dim=1)
        for i in range(len(sizes)):
            x_hat = x_hat + f_chunks[i] @ w_dec_chunks[i].float()
            raw = (x_pre - x_hat).pow(2).sum(dim=-1).mean()
            group_l2.append(raw)
            weighted_losses.append(w[i].float() * raw)
        # Vendored reduction: mean over the weighted per-group terms.
        matryoshka_loss = torch.stack(weighted_losses).mean()

        # AuxK against the FULL-prefix residual, detached (vendored :302).
        residual = (x_pre - x_hat).detach()
        aux_loss = self._auxk_loss(residual, acts.float())

        loss = matryoshka_loss + self.aux_alpha * aux_loss

        # Metric keys follow the in-house convention every model emits;
        # 'l2_loss' is the DEPLOY-level (full dictionary) recon error in the
        # per-element-mean convention so epoch logs compare across models.
        l0_norm = (latents > 0).float().sum(-1).mean()
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        output = {
            "num_dead_features": num_dead_features.item(),
            "l1_sparsity": 0.0,
            "l2_loss": (recons.float() - x.float()).pow(2).mean().item(),
            "num_active": l0_norm.item(),
            "auxk_loss": aux_loss.item(),
            "matryoshka_loss": matryoshka_loss.item(),
        }
        for i, raw in enumerate(group_l2):
            output[f"matryoshka_l2_g{i}"] = raw.item()

        # Cross-cutting MonoLoss hook, identical to every other model.
        if mono_coef > 0:
            x_for_mono = (self._x_preprocessed
                          if hasattr(self, "_x_preprocessed") else x)
            mono_loss, mono_metrics = compute_monosemanticity_loss_batch(
                x_for_mono, latents)
            loss = loss + mono_coef * mono_loss
            output.update(mono_metrics)

        return loss, output

    def _auxk_loss(self, residual: torch.Tensor, pre_acts: torch.Tensor):
        """Vendored get_auxiliary_loss (:214-245): top k_aux DEAD latents
        (k_aux = min(activation_dim // 2, n_dead) — the INPUT dim, the paper's
        B.1 heuristic), decoded WITHOUT bias, scored against the residual and
        normalized by the residual's variance."""
        dead = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return torch.tensor(0.0, dtype=residual.dtype,
                                device=residual.device)
        k_aux = min(self.cfg["act_size"] // 2, n_dead)
        auxk_latents = torch.where(dead[None], pre_acts,
                                   torch.tensor(-torch.inf,
                                                device=pre_acts.device))
        auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
        auxk_buffer = torch.zeros_like(pre_acts)
        auxk_acts_full = auxk_buffer.scatter_(-1, auxk_indices, auxk_acts)
        x_reconstruct_aux = auxk_acts_full @ self.W_dec.float()
        l2_loss_aux = (residual.float() - x_reconstruct_aux.float()) \
            .pow(2).sum(dim=-1).mean()
        # OpenAI normalization: divide by the residual's centered energy so
        # the aux term is scale-free (vendored :237-240).
        residual_mu = residual.mean(dim=0, keepdim=True) \
            .broadcast_to(residual.shape)
        loss_denom = (residual.float() - residual_mu.float()) \
            .pow(2).sum(dim=-1).mean()
        return (l2_loss_aux / loss_denom).nan_to_num(0.0)
