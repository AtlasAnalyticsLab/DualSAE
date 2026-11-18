"""Gated SAE (Rajamanoharan et al. 2024, "Improving Dictionary Learning with
Gated Sparse Autoencoders", arXiv 2404.16014).

Faithful port of the vendored reference implementation at
dictionary_learning/ (saprmarks/dictionary_learning) dictionary.py:178 (``GatedAutoEncoder``)
+ trainers/gdm.py:13 (``GatedSAETrainer.loss``) onto this package's
BaseAutoencoder conventions (W_enc (d,U), W_dec (U,d), b_enc, b_dec;
forward -> (pre_acts, latents, recons); compute_loss(x, acts, latents, recons,
mono_coef) -> (loss, metrics)) so it trains under train_other.py and
evaluates under knn_lp_sparse.py / cb_sae/featbsf/adapters.py unchanged.

The architecture splits the encoder into two heads sharing ONE weight matrix:
a GATING head decides *which* latents fire (a Heaviside on pi_gate) and a
MAGNITUDE head decides *how much* they fire (a ReLU on a per-latent rescaling
exp(r_mag) of the same pre-activation). Decoupling the two is what removes the
shrinkage bias that an L1 on the firing magnitude imposes on a plain ReLU SAE.
The L1 therefore lands on the GATE path only, and an auxiliary term forces the
gate path to stay a usable autoencoder in its own right (otherwise nothing
would anchor pi_gate's scale, and the L1 could be driven to zero for free).

Sparsity is EMERGENT here: there is no k. The realized L0 is a function of
l1_coeff (lambda), so lambda must be calibrated per (dataset, encoder) against
a target L0 and the achieved L0 reported next to every metric — never assumed.

Faithfulness notes (every deliberate deviation FLAGGED):
- FLAG G1 (decoder constraint, one step out of phase): the reference uses
  ConstrainedAdam (trainer.py), which projects the parallel gradient
  component, STEPS, then renormalizes. train_other.py calls
  BaseAutoencoder.make_decoder_weights_and_grad_unit_norm() which projects and
  renormalizes BEFORE optimizer.step(). The constraint SET is identical (unit
  atoms; our W_dec is (U,d) so per-atom is norm(dim=-1), matching the
  reference's norm(dim=0) on its (d,U) weight), but the saved W_dec rows are
  off unit norm by O(lr*||g||) ~ 1e-6. Consequence: the reference's
  encode(normalize_decoder=True) rescue path is unnecessary and NOT ported.
- FLAG G2 (Adam betas): the reference uses betas=(0.0, 0.999). Requested via
  optimizer_overrides() so the trainer honours it without an if-chain. lr and
  eps come from the suite protocol (--lr, --eps), NOT the reference's 5e-5.
- FLAG G3 (no LR schedule): GatedSAETrainer wraps a LambdaLR warmup/decay.
  Every row in this suite trains at constant LR, so we deliberately do NOT
  request a schedule (optimizer_overrides() could, via 'schedule').
- FLAG G4 (sparsity warm-up IS honoured): get_sparsity_warmup_fn (trainer.py)
  ramps the L1 coefficient linearly over sparsity_warmup_steps (reference
  default 2000). Reproduced by _sparsity_scale() off a persistent _step
  buffer, because train_other.py's compute_loss contract has no step argument.
  Kept because lambda is being calibrated against realized L0 and the ramp is
  what stops the gate collapsing before the magnitude head has any signal.
- FLAG G5 (loss reduction kept VENDORED): all three terms use
  .pow(2).sum(-1).mean() / L1-over-units-then-mean, so their mutual ratios
  (notably L_recon : L_aux = 1 : 1) are exactly the reference's. Adam is
  invariant to a static loss scale. The logged 'l2_loss' metric key is still
  the in-house per-element convention for log comparability (same policy as
  matryoshka_sae.py). NOTE this makes the loss ~d x the in-house scale, so
  train_other.py's grad-norm clip must be effectively disabled for gated rows
  (the reference gated trainer does not clip at all) -- see G6.
- FLAG G6 (gradient clipping): GatedSAETrainer.update does NOT clip.
  train_other.py clips unconditionally at config.max_grad_norm. Launch gated
  cells with a very large --max_grad_norm so the clip is inert, and watch the
  logged 'grad_norm'. This is a LAUNCH-side deviation, recorded here because
  the reason for it lives in this file (G5).
- FLAG G7 (gate_bias is stored as b_enc): the reference's gate_bias occupies
  BaseAutoencoder's existing b_enc parameter, keeping the state dict minimal.
  b_enc therefore means something DIFFERENT here (a gate threshold, compared
  against 0 by a Heaviside) than in JumpReLUSAE/MatryoshkaSAE (a pre-ReLU
  encoder bias). Flagged at the use site too.
- FLAG G8 (no dead-latent handling): the reference gated trainer has no AuxK
  and no resampling, and neither do we. num_batches_not_active is still
  tracked so the dead fraction can be REPORTED; train_other.py does not
  respawn. Expect and print a non-trivial dead fraction.

cfg keys (beyond BaseAutoencoder's): 'l1_coeff' (lambda, the sparsity penalty
on the gate path), 'sparsity_warmup_steps' (int, 0 disables the ramp),
'n_batches_to_dead'. The state-dict fingerprint cb_sae/featbsf/adapters.py and
knn_lp_sparse.load_run identify this class by is the PAIR ('r_mag',
'mag_bias'); the ctor-dict tag is model_cfg['model_type'] == 'gated'.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAutoencoder
from loss import compute_monosemanticity_loss_batch

# Reference default (gdm.py GatedSAETrainer.sparsity_warmup_steps).
_DEFAULT_SPARSITY_WARMUP = 2000


class GatedSAE(BaseAutoencoder):
    """Gated SAE on the BaseAutoencoder chassis."""

    def __init__(self, cfg):
        super().__init__(cfg)

        # lambda. Named l1_coeff to match every other model in this package;
        # the reference calls it l1_penalty (default 1e-1).
        self.l1_coeff = float(cfg.get("l1_coeff", 1e-1))

        # Magnitude head. These two are the ONLY tensors this class adds, and
        # together they are its state-dict fingerprint. exp(r_mag) is the
        # per-latent rescaling that lets the magnitude head diverge from the
        # gate head while sharing W_enc (the reference's weight sharing).
        self.r_mag = nn.Parameter(torch.zeros(self.cfg["dict_size"]))
        self.mag_bias = nn.Parameter(torch.zeros(self.cfg["dict_size"]))

        # Sparsity warm-up state (FLAG G4). Both are persistent buffers so a
        # reloaded checkpoint knows where it is on the ramp -- a resumed run
        # that restarted the ramp at 0 would briefly train an unregularized
        # model and silently change the realized L0 we calibrate on.
        self.register_buffer(
            "sparsity_warmup_steps",
            torch.tensor(int(cfg.get("sparsity_warmup_steps",
                                     _DEFAULT_SPARSITY_WARMUP)),
                         dtype=torch.long))
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

        # Vendored init (_reset_parameters, dictionary.py:199-205) translated
        # to the (U,d) decoder layout: random UNIT-NORM atoms, encoder tied to
        # the decoder transpose, all four biases zero. This replaces
        # BaseAutoencoder's kaiming init (which normalizes W_dec only AFTER
        # tying, leaving W_enc un-normalized) so the gate and magnitude heads
        # start on the same scale the reference gives them.
        with torch.no_grad():
            dec = torch.randn_like(self.W_dec)                  # (U, d)
            self.W_dec.data.copy_(dec / dec.norm(dim=-1, keepdim=True))
            self.W_enc.data.copy_(self.W_dec.data.t())
            self.b_dec.zero_()      # reference decoder_bias
            self.b_enc.zero_()      # reference gate_bias (FLAG G7)

        self.to(cfg["dtype"]).to(cfg["device"])

    # ------------------------------------------------------------------ #
    # Optimizer request (train_other.py::_optimizer_overrides hook)
    # ------------------------------------------------------------------ #
    def optimizer_overrides(self):
        """FLAG G2. The reference builds ConstrainedAdam(..., betas=(0.0,
        0.999)); beta1=0 makes the update a pure RMSProp-style step, which the
        gated objective was tuned with. Only the betas are requested: lr/eps
        stay on the suite protocol, and no schedule is requested (FLAG G3)."""
        return {"betas": (0.0, 0.999)}

    # ------------------------------------------------------------------ #
    # Sparsity warm-up
    # ------------------------------------------------------------------ #
    def _sparsity_scale(self) -> float:
        """min(step / sparsity_warmup_steps, 1.0); 1.0 when the ramp is off.
        Byte-for-byte the reference's get_sparsity_warmup_fn scale_fn."""
        warm = int(self.sparsity_warmup_steps.item())
        if not warm:
            return 1.0
        return min(int(self._step.item()) / warm, 1.0)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, x):
        """Returns (pre_acts, latents, recons).

        pre_acts is ReLU(pi_gate) -- the reference's `return_gate=True` second
        output -- NOT the raw pre-activation. That choice is load-bearing
        twice: compute_loss needs exactly this tensor for both L_sparse and
        L_aux, and cb_sae/featbsf/adapters.py's dense-activation surface expects a
        non-negative dense code (its clamp_min(0) is then a no-op here).
        """
        x_pre, x_mean, x_std = self.preprocess_input(x)

        # Reference: encoder(x - decoder_bias). ONE matmul feeds both heads.
        x_cent = x_pre - self.b_dec
        x_enc = x_cent @ self.W_enc                       # (B, U)

        # Gating head: a hard Heaviside, no straight-through estimator. The
        # gate gets its gradient exclusively through L_sparse and L_aux (both
        # of which see ReLU(pi_gate)); that is the whole point of the
        # architecture and why L_aux cannot be dropped.
        pi_gate = x_enc + self.b_enc                      # FLAG G7
        f_gate = (pi_gate > 0).to(x_enc.dtype)

        # Magnitude head: exp() keeps the rescaling strictly positive, so the
        # two heads can never disagree about a latent's SIGN, only its size.
        pi_mag = torch.exp(self.r_mag) * x_enc + self.mag_bias
        latents = f_gate * F.relu(pi_mag)

        pre_acts = F.relu(pi_gate)

        x_reconstruct = latents @ self.W_dec + self.b_dec

        self.update_inactive_features(latents)
        recons = self.postprocess_output(x_reconstruct, x_mean, x_std)

        # Stash for compute_loss: the decoder lives in the PREPROCESSED space,
        # so every loss term must be scored there (postprocess would undo the
        # per-sample standardization and change the reconstruction error).
        self._x_preprocessed = x_pre
        self._x_hat_pre = x_reconstruct

        return pre_acts, latents, recons

    # ------------------------------------------------------------------ #
    # Loss (gdm.py::GatedSAETrainer.loss)
    # ------------------------------------------------------------------ #
    def compute_loss(self, x, acts, latents, recons, mono_coef=0.0):
        """L = ||x - x_hat||^2 + lambda*scale*||ReLU(pi_gate)||_1
                              + ||x - x_hat_gate||^2

        `acts` is forward()'s element 0 = ReLU(pi_gate) = the reference's
        f_gate. Using the BINARY mask here instead is the classic way to get
        this wrong: it would make L_sparse an L0 count with zero gradient and
        L_aux a reconstruction from an all-ones code.
        """
        x_pre = (self._x_preprocessed if hasattr(self, "_x_preprocessed")
                 else x).float()
        x_hat = (self._x_hat_pre if hasattr(self, "_x_hat_pre")
                 else recons).float()
        gate = acts.float()

        # Auxiliary reconstruction through a FROZEN copy of the decoder
        # (gdm.py:80 detaches BOTH the weight and the bias). Detaching is what
        # makes this term a pure constraint on the gate path: it cannot be
        # satisfied by reshaping the dictionary, only by making ReLU(pi_gate)
        # itself informative about x.
        x_hat_gate = (gate @ self.W_dec.detach().float()
                      + self.b_dec.detach().float())

        # FLAG G5: vendored reductions.
        l_recon = (x_pre - x_hat).pow(2).sum(dim=-1).mean()
        l_sparse = torch.linalg.norm(gate, ord=1, dim=-1).mean()
        l_aux = (x_pre - x_hat_gate).pow(2).sum(dim=-1).mean()

        scale = self._sparsity_scale()
        sparsity_penalty = self.l1_coeff * l_sparse * scale
        loss = l_recon + sparsity_penalty + l_aux

        # Advance the ramp only while training. train_other.py's evaluate()
        # also calls compute_loss (under torch.no_grad, in eval mode); letting
        # those calls advance _step would desynchronize the schedule from the
        # optimizer step count by one per validation batch.
        if self.training:
            self._step += 1

        l0_norm = (latents > 0).float().sum(-1).mean()
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]).sum()

        output = {
            "num_dead_features": num_dead_features.item(),
            # Keys the trainer's epoch logging reads:
            "l1_sparsity": sparsity_penalty.item(),
            "l2_loss": (recons.float() - x.float()).pow(2).mean().item(),
            "num_active": l0_norm.item(),
            # Gated-specific diagnostics. gate_open_frac is the honest read on
            # whether lambda is in range: it is the density of the gate BEFORE
            # the magnitude head's ReLU can close a latent, so a run with
            # gate_open_frac >> L0/U is one whose two heads disagree.
            "gated_recon_loss": l_recon.item(),
            "gated_aux_loss": l_aux.item(),
            "gated_l1": l_sparse.item(),
            "sparsity_scale": scale,
            "gate_open_frac": (gate > 0).float().mean().item(),
        }

        if mono_coef > 0:
            x_for_mono = (self._x_preprocessed
                          if hasattr(self, "_x_preprocessed") else x)
            mono_loss, mono_metrics = compute_monosemanticity_loss_batch(
                x_for_mono, latents)
            loss = loss + mono_coef * mono_loss
            output.update(mono_metrics)

        return loss, output
