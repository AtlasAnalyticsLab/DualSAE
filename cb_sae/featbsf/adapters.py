"""Unified checkpoint loading for the CB-SAE converters.

`load_any(path)` accepts the checkpoints this pipeline consumes and wraps them
in an `Adapted` object exposing

    acts_and_recon(x (N,d) on device) -> (acts (N,U) float32 >= 0,
                                          x_hat (N,d) in the INPUT space)
    mono_recon(x) -> Sparse-Branch-only x_hat in input space (duals; else None)
    .model        -> the rebuilt native model (what the converters parity-check
                     their b=1 ports against)

Two checkpoint families are dispatched on the payload structure:

  * bsf/featbsf format (dict with a 'variant' key, written by
    bsf.models.save_bsf): the b=1 SAE ports and CB checkpoints. Rebuilt via
    bsf.models.load_bsf / bsf.cb_dual_bsf.load_cb_dual. Unit j := block j with
    activation ||z_j||_2 of the GATED code — nonnegative, exactly 0 iff the
    unit did not fire.
  * vision/ trainer format (dict with 'model_state_dict' plus 'config' — final
    checkpoints — or 'model_cfg' — best_val_monoscore checkpoints, translated
    by _cfg_from_model_cfg — written by vision/train_{topk,other,dual}.py):
    rebuilt from the checkpoint's OWN saved config (normalize / input_unit_norm
    included — it genuinely varies across runs and must never be hard-coded).
    Model classes are imported from vision/models through the path shim.

THE SPACE CONTRACT: every quantity is computed in the L2-normalized LMDB
feature space — the space `x` arrives in from featbsf/data.py loaders.
  * vision/ models already return `recons` in that space (their unit-norm
    preprocess is undone by postprocess_output before returning).
  * BSFBase models return x_hat in their internal normalized space
    x_norm = (x - mu) * scale, so the adapter inverts via denormalize().
    (b=1 SAE ports fold b_dec into mu with scale 1.)

The original module also rebuilt three hierarchical SAE families and exposed
two extra activation views for a hierarchy-metric suite; neither is used by
the CB pipeline, so they are not part of this release.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from featbsf._saepath import ensure_sae_on_path


class Adapted:
    """A loaded model + the metadata metrics.compute_all needs.

    All model calls run under torch.no_grad with the model in eval() mode —
    Adapted objects are strictly read-only views of a checkpoint.
    """

    def __init__(self, model, family: str, variant: str, is_dual: bool,
                 n_units: int, d: int, dims_per_unit: int, device: str,
                 seed=None):
        self.model = model
        self.family = family              # 'bsf' | 'sae'
        self.variant = variant
        self.is_dual = is_dual
        self.n_units = n_units
        self.d = d
        self.dims_per_unit = dims_per_unit  # b for BSF blocks, 1 for latents
        self.device = device
        self.seed = seed
        model.eval()

    # -- family-specific implementations injected below ---------------------
    def acts_and_recon(self, x: torch.Tensor):
        raise NotImplementedError

    def mono_recon(self, x: torch.Tensor):
        return None



class _BSFAdapted(Adapted):
    @torch.no_grad()
    def acts_and_recon(self, x):
        from bsf.models import block_norms
        m = self.model
        if self.is_dual:
            # Full dual reconstruction xB + poly(r). dual_warmup_active
            # defaults to False after load, but pin it: a crashed mono_recon
            # call must never leave the flag flipped for later batches.
            m.dual_warmup_active = False
        out = m(x)
        acts = block_norms(out.z)                      # (N, G) >= 0
        # Undo the model-internal normalization (see module docstring) via
        # the model's own inversion hook: identical to x_hat/scale + mu for
        # every BSF base, and the only correct inversion for an
        # input_unit_norm (per-sample LN) SAE-port base, whose stats were
        # stashed by the normalize() call inside forward().
        x_hat = m.denormalize(out.x_hat)
        return acts.float(), x_hat

    @torch.no_grad()
    def mono_recon(self, x):
        if not self.is_dual:
            return None
        m = self.model
        # The warmup path routes x_hat = xB (block branch alone, poly never
        # called) — exactly the mono-only reconstruction (bsf/models.py).
        m.dual_warmup_active = True
        try:
            out = m(x)
            return m.denormalize(out.x_hat)
        finally:
            m.dual_warmup_active = False


class _CBDualAdapted(Adapted):
    """CB-DualBSF checkpoints (bsf/cb_dual_bsf.py). Units are MIXED width —
    R retained blocks of width b ++ |C| concept blocks of width b_cb — so
    dims_per_unit is a per-unit numpy vector (metrics.compute_all handles
    both). acts follow the BSF convention: gated block norms, >= 0, exactly
    0 iff the unit did not fire.

    Recon roles mirror the dual convention:
      * acts_and_recon -> the FULL reconstruction (base + CB + poly) — how
        DualBSF runs are scored today;
      * mono_recon -> the DEPLOYMENT (steerable) reconstruction base + CB,
        the poly-free recon a steering user actually edits — so r2_mono
        reports the steerable dictionary's quality. (For the no-poly arm the
        two coincide by construction.)
    Both are inverted to input space (x_hat/scale + mu), same contract as
    _BSFAdapted.
    """

    @torch.no_grad()
    def acts_and_recon(self, x):
        from bsf.models import block_norms
        m = self.model
        x_norm = m.normalize(x)
        # One manual pass instead of unit_acts + full_recon (each would
        # re-encode the frozen base; the KNN/LP bank streams 1.28M rows
        # through here, so the duplicate encode is worth avoiding).
        _, z, _ = m.base.encode(x_norm)
        xB = m.base.decode(z)
        # SAE-port bases reconstruct decode(z) + b_dec (mirrors
        # CBDualBSF.base_recon; zero for folded converts, load-bearing for
        # the LN carrier). BSF bases have no b_dec attribute: unchanged.
        b_dec = getattr(m.base, "b_dec", None)
        if b_dec is not None:
            xB = xB + b_dec
        z_cb, _ = m.sigma_cb(m.concept_pre(x_norm))
        x_cb = m.cb_recon(z_cb)
        x_hat = xB + x_cb
        if m.use_poly:
            x_hat = x_hat + m.base.poly(x_norm - xB - x_cb)
        acts = torch.cat([block_norms(z), block_norms(z_cb)], dim=1)
        return acts.float(), m.denormalize(x_hat)

    @torch.no_grad()
    def mono_recon(self, x):
        m = self.model
        return m.denormalize(m.deployment_recon(m.normalize(x)))


class _SAEAdapted(Adapted):
    @torch.no_grad()
    def acts_and_recon(self, x):
        # All vision/ models share the (pre_acts, latents, recons) contract with
        # recons already in the input space (LN preprocess undone in decode).
        _, latents, recons = self.model(x)
        # Latents are nonnegative by construction for every variant we score
        # (ReLU precedes TopK/BatchTopK; JumpReLU gates at exp(log_theta) > 0;
        # Vanilla is plain ReLU). clamp_min(0) is a no-op safety net against
        # fp round-off producing -1e-9s that would flip a '>0' activity test.
        return latents.float().clamp_min(0), recons

    @torch.no_grad()
    def mono_recon(self, x):
        if not self.is_dual:
            return None
        m = self.model
        # DualSAE.forward stashes the pre-postprocess mono reconstruction and
        # the preprocessing stats (vision/models/dual_sae.py); replaying
        # postprocess_output on _xM gives the mono-only recon in input space.
        m(x)
        return m.postprocess_output(m._xM, m._x_mean, m._x_std)


# ---------------------------------------------------------------------------
# vision/ format rebuilders (config-faithful: built from the saved config).
# ---------------------------------------------------------------------------
def _cfg_from_model_cfg(mc: dict) -> dict:
    """Translate a best_val checkpoint's 'model_cfg' into the 'config'
    vocabulary _rebuild_sae reads.

    The trainers write TWO checkpoint payload shapes: the FINAL checkpoints
    carry vars(config) under 'config' (TrainingConfig vocabulary), while the
    best_val_monoscore_model.pt payloads (resolved by --which best_val) carry
    only 'model_cfg' — the model constructor dict (train_other.py,
    train_dual.py) or the small self-describing dict train_topk.py
    writes (train_topk.py). Without this translation cfg collapsed
    to {} and every setting silently fell to its default: a BatchTopK best_val
    checkpoint loaded strict=True as a dense VanillaSAE (identical registered
    params), normalize=True runs rebuilt un-normalized, non-default top_k fell
    to 64, and dual sparsity_mode fell to 'topk' — all wrong numbers with no
    crash.
    """
    # --- model_type: honor an EXPLICIT tag first (the matryoshka/msae ctor
    # dicts carry one precisely because their registered parameter names are
    # identical to VanillaSAE's — a key-fingerprint miss would rebuild the
    # wrong class and load strict=True without error). Then fall back to the
    # key fingerprint (legacy model_cfg stores no explicit type). Constructor
    # dicts (train_other/train_dual) carry 'act_size'; the train_topk dict
    # carries 'n_latents'/'d_model' instead.
    if "model_type" in mc:
        model_type = str(mc["model_type"])
    elif "sparsity_mode" in mc:
        model_type = "dual"                     # DualSAE ctor dict (train_dual)
    elif "group_sizes" in mc:
        # MatryoshkaSAE — MUST be tested before the aux-key check below: its
        # ctor dict also carries 'aux_penalty' and would otherwise be routed
        # to batch_topk.
        model_type = "matryoshka"
    elif "nesting_list" in mc:
        model_type = "msae"                     # MSAE ctor dict
    elif "top_k_aux" in mc or "aux_penalty" in mc:
        model_type = "batch_topk"               # BatchTopKSAE ctor dict
    elif "bandwidth" in mc:
        model_type = "jumprelu"                 # JumpReLUSAE ctor dict
    elif "act_size" in mc:
        model_type = "vanilla"                  # VanillaSAE ctor dict
    else:
        model_type = "topk"                     # train_topk self-describing dict
    cfg: dict = {"model_type": model_type}

    # --- key-name translation (ctor vocabulary -> TrainingConfig vocabulary).
    # 'normalize' is train_topk's spelling; 'input_unit_norm' the ctor dicts'.
    if "input_unit_norm" in mc or "normalize" in mc:
        cfg["normalize"] = bool(mc.get("input_unit_norm",
                                       mc.get("normalize", False)))
    # top_k (ctor) / topk_k (train_topk + TrainingConfig): emit both spellings
    # so every branch below finds its native name.
    if "top_k" in mc or "topk_k" in mc:
        k = mc.get("top_k", mc.get("topk_k"))
        cfg["top_k"] = cfg["topk_k"] = int(k)
    if "l1_coeff" in mc:
        cfg["l1_coef"] = float(mc["l1_coeff"])
    if "top_k_aux" in mc:
        cfg["auxk_k"] = int(mc["top_k_aux"])
    if "aux_penalty" in mc:
        cfg["auxk_coef"] = float(mc["aux_penalty"])
    # Same-name pass-throughs (present per model family; absent keys keep the
    # branch defaults, matching the final-checkpoint behavior).
    for key in ("bandwidth", "poly_hidden", "poly_layers", "poly_hidden_dims",
                "sparsity_mode", "seed", "tied", "activation",
                "n_batches_to_dead",
                # Matryoshka-family structure (plain models + dual modes).
                "group_sizes", "nesting_list", "relative_importance",
                "soft_cap",
                "jump_threshold_init"):
        if key in mc:
            cfg[key] = mc[key]
    return cfg


def _rebuild_sae(payload: dict, device: str):
    """(model, variant, is_dual, n_units, d) from a vision/ trainer checkpoint."""
    ensure_sae_on_path()
    sd = payload["model_state_dict"]
    cfg = payload.get("config", {}) or {}
    # best_val checkpoints carry 'model_cfg' instead of 'config' — translate
    # rather than silently rebuilding from all-default settings (see
    # _cfg_from_model_cfg for the failure modes this prevents).
    if not cfg and isinstance(payload.get("model_cfg"), dict):
        cfg = _cfg_from_model_cfg(payload["model_cfg"])

    # Dispatch order: explicit model_type (train_other), poly.* keys (dual),
    # encoder.weight (the OpenAI-style TopKSAE), then W_enc-family fallbacks.
    model_type = cfg.get("model_type")
    if model_type is None:
        if any(k.startswith("poly.") for k in sd):
            model_type = "dual"
        elif "group_sizes" in sd:
            # MatryoshkaSAE registers its group partition as a persistent
            # buffer for exactly this purpose: its PARAMETER names are
            # identical to VanillaSAE's, so without the buffer fingerprint an
            # untyped payload would rebuild as VanillaSAE and load strict=True
            # with no error (the docstring's silent-wrong-metrics hazard).
            model_type = "matryoshka"
        elif "nesting_levels" in sd:
            model_type = "msae"                 # MSAE's buffer fingerprint
        elif "encoder.weight" in sd:
            model_type = "topk"
        elif any(k.endswith("log_threshold") for k in sd):
            # JumpReLUSAE registers the parameter on a SUBMODULE, so the state
            # dict key is 'jumprelu.log_threshold' (jumprelu_sae.py) — an
            # exact-key test for 'log_threshold' can never fire and jumprelu
            # checkpoints would fall through to the vanilla branch (loud but
            # misleading strict-load failure). endswith() catches both the
            # submodule spelling and any flat legacy layout.
            model_type = "jumprelu"
        elif "sparsity_mode" in cfg:
            model_type = "dual"
        else:
            model_type = "vanilla"

    normalize = bool(cfg.get("normalize", False))
    # best_val model_cfg payloads store the already-derived n_batches_to_dead;
    # final-checkpoint configs store the raw dead_steps_threshold/batch_size
    # pair the trainers derive it from (train_other.py).
    n_batches_to_dead = int(cfg.get(
        "n_batches_to_dead",
        cfg.get("dead_steps_threshold", 10_000_000)
        // max(cfg.get("batch_size", 2048), 1)))

    if model_type == "topk":
        from models.topk_sae import ACTIVATIONS_CLASSES, TopK, TopKSAE
        n_latents, d = sd["encoder.weight"].shape
        # The as-trained activation is recorded IN the state dict itself
        # (TopKSAE.state_dict adds 'activation' = class name — 'TopK'/'ReLU' —
        # plus 'activation_state_dict' with the as-trained k); the config's
        # TrainingConfig.activation uses lowercase 'topk'/'relu'. The sd is
        # ground truth, cfg the fallback. Rebuilding a relu-trained TopKSAE
        # with a TopK gate loads strict=True without error (the gate holds no
        # registered params) and silently changes every metric — so honor the
        # recorded activation instead of hardcoding TopK.
        act_sd = sd.get("activation_state_dict") or {}
        act_name = str(sd.get("activation", cfg.get("activation", "topk")))
        # Prefer the as-trained k stored inside the activation state dict over
        # the config value (it is what the model actually gated with).
        k = int(act_sd.get("k", cfg.get("topk_k", 64)))
        if act_name.lower() == "topk":
            # postact_fn is 'ReLU' for every trainer build (TopK's default);
            # honor the recorded name anyway for faithfulness to
            # TopK.from_state_dict (topk_sae.py:249-253).
            post = ACTIVATIONS_CLASSES.get(
                str(act_sd.get("postact_fn", "ReLU")), nn.ReLU)()
            activation = TopK(k=k, postact_fn=post)
        elif act_name.lower() == "relu":
            # train_topk.py builds nn.ReLU() for config.activation
            # == 'relu' (exposed by vision/main.py --activation).
            activation = nn.ReLU()
        elif act_name.lower() == "identity":
            activation = nn.Identity()
        else:
            raise ValueError(f"unknown TopKSAE activation {act_name!r} "
                             "in checkpoint")
        # tied decoder shows up as decoder.linear.weight.
        tied = "decoder.linear.weight" in sd
        model = TopKSAE(n_latents=n_latents, n_inputs=d,
                        activation=activation, tied=tied,
                        # Config-faithful (see module docstring), never a
                        # hard-coded value.
                        normalize=normalize)
        clean = {k_: v for k_, v in sd.items()
                 if k_ not in ("activation", "activation_state_dict",
                               "activation.k", "activation.postact_fn")}
        model.load_state_dict(clean, strict=True)
        return model.to(device), "topk", False, n_latents, d

    if model_type == "dual":
        from models.dual_sae import DualSAE
        d, n_latents = sd["W_enc"].shape
        mode = cfg.get("sparsity_mode", "topk")
        dual_cfg = {
            "act_size": d,
            "dict_size": n_latents,
            # train_dual.py's config field is top_k (fed from main.py
            # --topk_k); tolerate both spellings across trainer eras.
            "top_k": int(cfg.get("top_k", cfg.get("topk_k", 64))),
            "l1_coeff": float(cfg.get("l1_coef", 1e-4)),
            "bandwidth": float(cfg.get("bandwidth", 1e-3)),
            "poly_hidden": int(cfg.get("poly_hidden", 512)),
            "poly_layers": int(cfg.get("poly_layers", 1)),
            "poly_hidden_dims": cfg.get("poly_hidden_dims", ""),
            "sparsity_mode": mode,
            "n_batches_to_dead": n_batches_to_dead,
            "seed": int(cfg.get("seed", 42)),
            "device": device,
            "dtype": torch.float32,
            "input_unit_norm": normalize,
        }
        # Matryoshka-family dual modes need their structure at ctor time (the
        # DualSAE ctor sizes the buffers from it). The state dict's buffers
        # are the ground truth — a checkpoint knows its own partition even if
        # the sidecar config drifted; cfg is the fallback for both.
        if mode == "matryoshka":
            dual_cfg["group_sizes"] = (
                sd["group_sizes"].tolist() if "group_sizes" in sd
                else cfg["group_sizes"])
        elif mode == "msae":
            dual_cfg["nesting_list"] = (
                sd["nesting_levels"].tolist() if "nesting_levels" in sd
                else cfg.get("nesting_list", cfg.get("msae_nesting")))
            if "level_weights" in sd:
                dual_cfg["relative_importance"] = sd["level_weights"].tolist()
            elif "relative_importance" in cfg:
                dual_cfg["relative_importance"] = cfg["relative_importance"]
            # Two vocabularies: best_val ctor dicts say 'soft_cap', final
            # vars(TrainingConfig) payloads say 'msae_soft_cap'. The cap is a
            # plain attribute (no state-dict fingerprint), so missing the
            # trainer spelling would silently rebuild a non-default cap as
            # 30.0 — read both before defaulting.
            dual_cfg["soft_cap"] = cfg.get("soft_cap",
                                           cfg.get("msae_soft_cap", 30.0))
        model = DualSAE(dual_cfg)
        model.load_state_dict(sd, strict=True)
        return model.to(device), f"dual_{mode}", True, n_latents, d

    # The three cfg-dict models share the (act_size, dict_size) inference.
    d, n_latents = sd["W_enc"].shape
    common = {
        "act_size": d, "dict_size": n_latents,
        "n_batches_to_dead": n_batches_to_dead,
        "seed": int(cfg.get("seed", 42)),
        "device": device, "dtype": torch.float32,
        "input_unit_norm": normalize,
    }
    if model_type == "batch_topk":
        from models.batch_topk_sae import BatchTopKSAE
        model = BatchTopKSAE({**common,
                              "top_k": int(cfg.get("topk_k", 64)),
                              "l1_coeff": 0.0,
                              "top_k_aux": int(cfg.get("auxk_k") or 512),
                              "aux_penalty": float(cfg.get("auxk_coef", 1 / 32))})
    elif model_type == "jumprelu":
        from models.jumprelu_sae import JumpReLUSAE
        model = JumpReLUSAE({**common,
                             "l1_coeff": float(cfg.get("l1_coef", 1e-4)),
                             "bandwidth": float(cfg.get("bandwidth", 1e-3))})
    elif model_type == "vanilla":
        from models.vanilla_sae import VanillaSAE
        model = VanillaSAE({**common,
                            "l1_coeff": float(cfg.get("l1_coef", 1e-4))})
    elif model_type == "matryoshka":
        from models.matryoshka_sae import MatryoshkaSAE
        # The buffer in the state dict is the ground truth for the partition
        # (strict load would fail loudly on any mismatch anyway).
        group_sizes = (sd["group_sizes"].tolist() if "group_sizes" in sd
                       else cfg["group_sizes"])
        # The state dict knows its own gate budget ('k' is a buffer, as in
        # the vendored model) — the cfg is only the legacy fallback, so an
        # untyped/config-less checkpoint still rebuilds with the right k.
        top_k = (int(sd["k"].item()) if "k" in sd
                 else int(cfg.get("top_k", cfg.get("topk_k", 64))))
        model = MatryoshkaSAE({**common,
                               "top_k": top_k,
                               "group_sizes": group_sizes,
                               "l1_coeff": 0.0,
                               "aux_penalty": float(
                                   cfg.get("auxk_coef", 1 / 32))})
    elif model_type == "msae":
        from models.msae_sae import MSAE
        nesting = (sd["nesting_levels"].tolist() if "nesting_levels" in sd
                   else cfg.get("nesting_list", cfg.get("msae_nesting")))
        # MSAE deploys at the smallest nesting level BY CONSTRUCTION (the
        # ctor enforces it), so the nesting buffer — not a cfg default — is
        # the authority on top_k for a config-less checkpoint.
        msae_cfg = {**common,
                    "top_k": int(cfg.get("top_k",
                                         cfg.get("topk_k", nesting[0]))),
                    "nesting_list": nesting,
                    # 'soft_cap' = best_val ctor-dict spelling,
                    # 'msae_soft_cap' = final vars(TrainingConfig) spelling;
                    # the cap has no state-dict fingerprint, so read both
                    # before defaulting (a non-default cap would otherwise be
                    # silently rebuilt as 30.0).
                    "soft_cap": cfg.get("soft_cap",
                                        cfg.get("msae_soft_cap", 30.0)),
                    "l1_coeff": 0.0}
        if "level_weights" in sd:
            msae_cfg["relative_importance"] = sd["level_weights"].tolist()
        elif "relative_importance" in cfg:
            msae_cfg["relative_importance"] = cfg["relative_importance"]
        model = MSAE(msae_cfg)
    else:
        raise ValueError(f"unknown sae model_type {model_type!r}")
    model.load_state_dict(sd, strict=True)
    return model.to(device), model_type, False, n_latents, d


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def resolve_ckpt(path: str | Path, which: str = "final") -> Path:
    """Accept a run DIRECTORY or a checkpoint file. For directories, resolve
    by convention: featbsf/bsf runs store model.pt; sae runs store
    autoencoder_final.pt / best_val_monoscore_model.pt (--which best_val)."""
    p = Path(path)
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(p)
    candidates = (["model.pt"] if (p / "model.pt").exists() else
                  ["best_val_monoscore_model.pt" if which == "best_val"
                   else "autoencoder_final.pt"])
    for c in candidates:
        if (p / c).exists():
            return p / c
    raise FileNotFoundError(f"no checkpoint ({which}) under {p}")


def _sidecar_seed(ckpt: Path, payload: dict):
    """Seed provenance: checkpoint config first (final 'config', best_val
    'model_cfg', bsf 'extra'), then the run dir's metrics.json /
    training_config.json sidecars (best effort)."""
    for key in ("config", "model_cfg", "extra"):
        cfg = payload.get(key)
        if isinstance(cfg, dict) and cfg.get("seed") is not None:
            return int(cfg["seed"])
    for name in ("metrics.json", "training_config.json"):
        f = ckpt.parent / name
        if f.exists():
            try:
                j = json.loads(f.read_text())
                if j.get("seed") is not None:
                    return int(j["seed"])
            except (json.JSONDecodeError, OSError):
                pass
    return None


def load_any(path: str | Path, device: str = "cuda",
             which: str = "final") -> Adapted:
    """Load any project checkpoint and wrap it for the unified evaluator."""
    ckpt = resolve_ckpt(path, which)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{ckpt}: unrecognized checkpoint payload type "
                         f"{type(payload)}")

    if payload.get("variant") == "cb_dual_bsf":
        # CB-DualBSF (bsf/cb_dual_bsf.py). MUST dispatch before the generic
        # bsf branch: load_bsf does not know this variant and would fall
        # through to the SAE builder, which misparses the payload.
        from bsf.cb_dual_bsf import load_cb_dual
        model = load_cb_dual(ckpt, device=device)
        return _CBDualAdapted(
            model, family="bsf", variant="cb_dual_bsf",
            # is_dual: r2_mono (the deployment/steerable recon) is reported
            # for the no-poly arm too — there it coincides with r2.
            is_dual=True, n_units=model.R + model.n_concepts,
            d=int(model.d), dims_per_unit=model.dims_per_unit_vector(),
            device=device, seed=_sidecar_seed(ckpt, payload))

    if "variant" in payload:
        # bsf/featbsf family (save_bsf format) — covers BSF, DualBSF, and the
        # b=1 SAE ports; load_bsf dispatches internally.
        from bsf.models import load_bsf
        model = load_bsf(ckpt, device=device)
        variant = payload["variant"]
        is_dual = bool(payload.get("dual")) or variant.startswith("dual_")
        return _BSFAdapted(model, family="bsf", variant=variant,
                           is_dual=is_dual, n_units=int(model.G),
                           d=int(model.d), dims_per_unit=int(model.b),
                           device=device, seed=_sidecar_seed(ckpt, payload))

    if "model_state_dict" in payload:
        model, variant, is_dual, n_units, d = _rebuild_sae(payload, device)
        return _SAEAdapted(model, family="sae", variant=variant,
                           is_dual=is_dual, n_units=n_units, d=d,
                           dims_per_unit=1, device=device,
                           seed=_sidecar_seed(ckpt, payload))

    raise ValueError(f"{ckpt}: neither a save_bsf payload (no 'variant' key) "
                     f"nor a sae trainer payload (no 'model_state_dict')")
