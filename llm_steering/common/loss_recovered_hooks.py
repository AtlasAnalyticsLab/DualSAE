"""Cross-entropy *loss recovered* for residual-stream SAEs, with plain torch hooks.

Vendored from the language pipeline (language/langdual/lmsae/loss_recovered_hooks.py) so the
steering stages use the identical hook, splice and norm-factor code.

WHAT THIS MEASURES
------------------
The LM-native SAE metric: how much of the language model's next-token loss survives when
the hooked activation is replaced by the SAE's reconstruction.  Transliterated from
dictionary_learning's ``evaluation.py`` (``loss_recovered`` plus the
``frac_recovered`` line inside ``evaluate``), so the DEFINITION is the reference one:

    three forward passes -> three next-token cross-entropies, each
        CrossEntropyLoss(logits[:, :-1, :].reshape(-1, V), tokens[:, 1:].reshape(-1))
      1. the unmodified model                       -> loss_original
      2. the hooked activation replaced by SAE(x)   -> loss_reconstructed
      3. the hooked activation zeroed               -> loss_zero

    frac_recovered = (loss_reconstructed - loss_zero) / (loss_original - loss_zero)

``frac_alive`` (the fraction of dictionary latents that fire at least once) is reported
alongside, as ``evaluate`` does.

DELIBERATE DEVIATIONS FROM THE REFERENCE (all four are forced, none is a preference)
-----------------------------------------------------------------------------------
1. NO NNSIGHT.  The reference drives the interventions through ``nnsight``'s tracing
   context; here the three interventions are done with ``register_forward_hook`` so that
   nnsight is not a dependency.
   The only behavioural difference is that pass 2 computes the reconstruction INSIDE the
   hook rather than in a separate trace that first saved ``x``.  That is identical here:
   there is exactly one intervention site, every layer before it is unaffected by the
   intervention, so the ``x`` the hook sees is bit-identical to the ``x`` the reference's
   extra trace would have saved.  (It also removes one of the reference's four forward
   passes, which is a pure saving.)

2. ``ignore_index``.  The reference passes ``ignore_index=model.tokenizer.pad_token_id``.
   Pythia's tokenizer has ``pad_token_id = None`` and ``CrossEntropyLoss(ignore_index=None)``
   raises ``TypeError``, so that line cannot be transliterated literally.  Our Track B token
   streams are PACKED (``config.token_stream``: a flat uint16 array cut into fixed-length
   ``LM_CTX_LEN`` windows) and therefore contain no padding at all, so the question the
   reference's argument answers does not arise.  We pass the torch default ``-100``, which
   masks nothing in a stream that has nothing to mask.  Any caller that DOES build padded
   batches must pass ``ignore_index`` explicitly and set the pad positions to it.

3. FIXED GLOBAL NORM FACTOR instead of a per-batch one.  The reference's
   ``normalize_batch`` recomputes ``scale = sqrt(d) / x.norm(dim=-1).mean()`` per batch and
   undoes it with ``x_hat / scale``.  Our shards were folded ONCE with a fixed global factor
   (see ``config.TRACKB_NORMALIZE`` and the shard manifest), because a per-batch scale makes
   the SAE's input distribution depend on the batch composition.  We therefore read the
   factor out of the shard manifest -- NEVER recompute it -- apply it on the way into the
   SAE and undo it on the way back into the model, which is the same two-sided handling the
   reference performs, with a constant in place of a statistic.

4. CE IS COMPUTED IN FLOAT32.  Identical to the reference when the LM runs in fp32 (our
   default) and strictly more accurate in half precision.  The logits of each pass are also
   reduced to a scalar and freed before the next pass runs, which keeps three
   (B, L, 50304) tensors from being resident at once; the arithmetic is unchanged.

KNOWN LIMITATIONS
-----------------
* ``sae_fn`` receives a 2-D ``(B*L, d_model)`` tensor: the SAE models (vision/models) index
  ``x.shape[0]`` as the batch dimension (``BatchTopKSAE.forward`` takes the top
  ``k * x.shape[0]`` over the FLATTENED activations), so a 3-D input would silently change
  the sparsity.  For a BatchTopK or Dual-BatchTopK cell the batch composition is therefore
  part of the measurement: pass ``batch_sequences * ctx_len == 2048`` (the training batch,
  ``config.BATCH_SIZE``) to reproduce training-time sparsity.  ``DEFAULT_BATCH_SEQUENCES``
  is set so this holds at ``config.LM_CTX_LEN = 128``.
* ``frac_alive`` is measured over the CE token set only (the reference measures it over its
  activation buffer).  It is a diagnostic here.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from common import lang_config as config


# CrossEntropyLoss's own default.  See deviation (2): our streams are packed, so nothing is
# ignored; the constant exists so a caller with padded batches has one knob to turn.
IGNORE_INDEX = -100

# 16 sequences x config.LM_CTX_LEN (128) = 2048 rows = config.BATCH_SIZE.  See the
# BatchTopK limitation above: this is not a throughput choice, it is a protocol constant.
DEFAULT_BATCH_SEQUENCES = max(1, config.BATCH_SIZE // config.LM_CTX_LEN)

# The manifest key the extraction stage is contracted to write.  The fallbacks exist only so
# a manifest written with a different spelling fails LOUDLY-but-usefully (the key actually
# used is printed and recorded) instead of silently defaulting to 1.0, which would look like
# a working run with a 30x wrong activation scale.
_NORM_FACTOR_KEYS = ("norm_factor", "norm_scale", "scale")

# Canonical convention: the shards hold ``raw_activation * norm_factor`` (this is the
# direction the reference scales in -- ``x = x * scale`` before the dictionary, and
# ``x_hat = x_hat / scale`` before writing back).  A manifest may state the other direction
# explicitly with ``"norm_convention": "divide"``; anything else is refused rather than
# guessed, because guessing wrong is a silent factor-of-``norm_factor**2`` error.
_CONVENTIONS = ("multiply", "divide")


# --------------------------------------------------------------------------- #
# Shard manifest: the single source of the fixed global norm factor
# --------------------------------------------------------------------------- #

def shard_manifest_path(layer: int, split: str, corpus: str = "owt") -> str:
    """Where ``config.resid_dir`` keeps the manifest for one (layer, split, corpus).

    The manifest holds the ONLY copy of the norm factor the shards were folded with, and that
    factor is an expectation over its own corpus' tokens. Reading the wrong corpus' manifest
    would rescale a model's input into a space it never saw in training and still produce
    plausible numbers, so the corpus is explicit here rather than inferred.
    """
    return os.path.join(config.resid_dir(layer, split, corpus), "manifest.json")


def read_shard_manifest(layer: int, split: str, corpus: str = "owt") -> Dict[str, Any]:
    """Load and validate the fp16 shard manifest for one (layer, split).

    The manifest is the contract between the extraction stage and every consumer.  The
    fields this module depends on:

        norm_factor      float > 0   the FIXED global scale the shards were folded with
        norm_convention  optional    "multiply" (default) | "divide"
        d_model          int         must equal config.LM_D_MODEL
        shards           list        [{"file": ..., "n_rows": ...}, ...] (order is the
                                     row order; ``file`` may be relative to the shard dir)
        dtype            optional    numpy dtype name of the shard rows (default float16)

    Validation is deliberately eager: a missing/zero/NaN norm factor is a silent 30x
    activation-scale error downstream, and a d_model mismatch means the wrong layer's shards
    were pointed at.  Both are cheap to check once and impossible to notice later.
    """
    path = shard_manifest_path(layer, split, corpus)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no shard manifest at {path}; run the Track B extraction stage for layer "
            f"{layer} / split {split!r} first"
        )
    with open(path, "r") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: manifest must be a JSON object, got {type(manifest).__name__}")

    d_model = int(manifest.get("d_model", config.LM_D_MODEL))
    if d_model != config.LM_D_MODEL:
        raise ValueError(
            f"{path}: d_model={d_model} but config.LM_D_MODEL={config.LM_D_MODEL}; "
            "the manifest describes a different model than the configured one"
        )
    # Cheap identity check -- catches a manifest copied between layer directories, which
    # would otherwise produce a perfectly well-formed but wrong result.
    if "layer" in manifest and int(manifest["layer"]) != int(layer):
        raise ValueError(f"{path}: manifest says layer {manifest['layer']}, asked for {layer}")
    if "split" in manifest and str(manifest["split"]) != str(split):
        raise ValueError(f"{path}: manifest says split {manifest['split']!r}, asked for {split!r}")
    return manifest


def norm_factor_from_manifest(manifest: Dict[str, Any], verbose: bool = True) -> Tuple[float, str]:
    """Return ``(norm_factor, convention)`` -- never recomputed from data.

    The factor is a property of the SHARDS the SAE was trained on.  Recomputing it from
    whatever activations happen to be in hand would give a different number on a different
    token set and would silently put the SAE's input in a space it never saw during
    training, so this function reads and validates only.
    """
    key = next((k for k in _NORM_FACTOR_KEYS if k in manifest), None)
    if key is None:
        raise KeyError(
            "shard manifest has no norm factor (looked for "
            f"{_NORM_FACTOR_KEYS}); it must record the fixed global scale the shards were "
            "folded with, because it cannot be recovered from the shards afterwards"
        )
    if key != _NORM_FACTOR_KEYS[0] and verbose:
        # Visible, not silent: a non-canonical spelling is usable but worth seeing in the log.
        print(f"[loss_recovered] manifest norm factor read from non-canonical key {key!r}")
    factor = float(manifest[key])
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"shard manifest norm factor must be finite and > 0, got {factor!r}")

    convention = str(manifest.get("norm_convention", _CONVENTIONS[0]))
    if convention not in _CONVENTIONS:
        raise ValueError(
            f"shard manifest norm_convention={convention!r} is not one of {_CONVENTIONS}; "
            "refusing to guess the direction of a scale factor"
        )
    return factor, convention


def to_shard_space(x: torch.Tensor, norm_factor: float, convention: str = "multiply") -> torch.Tensor:
    """Model-space activation -> the space the SAE was trained in."""
    return x * norm_factor if convention == "multiply" else x / norm_factor


def from_shard_space(x: torch.Tensor, norm_factor: float, convention: str = "multiply") -> torch.Tensor:
    """Shard-space reconstruction -> model space (the exact inverse of ``to_shard_space``)."""
    return x / norm_factor if convention == "multiply" else x * norm_factor


# --------------------------------------------------------------------------- #
# Locating and hooking the residual stream
# --------------------------------------------------------------------------- #

def resolve_layer_module(model: nn.Module, layer: int) -> nn.Module:
    """The decoder block whose OUTPUT is the residual stream we hook (``io='out'``).

    Pythia is GPTNeoX (``model.gpt_neox.layers``); the other spellings are there so a smoke
    test on a different small causal LM does not need a second function.  Hooking the block
    output means the captured tensor is the residual stream AFTER block ``layer``, which is
    the convention ``config.LM_LAYERS`` (6, 12, 18 of 24) is stated in.
    """
    candidates: List[Any] = []
    for attr_path in (("gpt_neox", "layers"), ("model", "layers"), ("transformer", "h")):
        obj: Any = model
        for attr in attr_path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            candidates.append(obj)
    if not candidates:
        raise AttributeError(
            f"cannot find the decoder block list on {type(model).__name__}; expected one of "
            "model.gpt_neox.layers / model.model.layers / model.transformer.h"
        )
    blocks = candidates[0]
    n = len(blocks)
    if not 0 <= layer < n:
        raise IndexError(f"layer {layer} out of range for a {n}-block model")
    return blocks[layer]


def _split_output(output: Any) -> Tuple[torch.Tensor, Callable[[torch.Tensor], Any]]:
    """Return ``(hidden_states, rebuild)`` for whatever a decoder block returned.

    transformers 5.8's ``GPTNeoXLayer.forward`` returns a BARE TENSOR (verified by reading
    the installed source); versions before 5.x return ``(hidden_states, present, ...)``.
    Both shapes are handled here rather than at every call site, and anything else raises
    instead of being coerced -- a silently mis-spliced residual stream would show up only as
    an implausible loss number.
    """
    if torch.is_tensor(output):
        return output, lambda new: new
    if isinstance(output, tuple):
        if not output or not torch.is_tensor(output[0]):
            raise TypeError("block returned a tuple whose first element is not a tensor")
        rest = output[1:]
        return output[0], lambda new: (new,) + rest
    raise TypeError(
        f"unsupported block output type {type(output).__name__}; expected a tensor (transformers "
        ">= 5) or a tuple (transformers < 5)"
    )


@contextmanager
def splice_activation(module: nn.Module, fn: Optional[Callable[[torch.Tensor], torch.Tensor]]):
    """Temporarily replace ``module``'s output hidden states with ``fn(hidden_states)``.

    ``fn=None`` installs no hook at all (the unmodified pass), so pass 1 is exactly the
    stock model rather than the stock model plus an identity hook.  The hook is always
    removed, including on an exception, so a failed batch cannot leave the LM permanently
    spliced for the next one.
    """
    if fn is None:
        yield
        return

    def _hook(_mod: nn.Module, _args: Any, output: Any) -> Any:
        hidden, rebuild = _split_output(output)
        return rebuild(fn(hidden))

    handle = module.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def capture_activation(module: nn.Module, store: List[torch.Tensor]):
    """Append ``module``'s output hidden states to ``store`` (no modification).

    Used to pull per-token residuals for the labeled corpora; kept here
    because it must agree with ``splice_activation`` about WHICH tensor is the residual
    stream -- if those two ever disagreed, the CE numbers and the label metrics would be
    measured at different sites.
    """
    def _hook(_mod: nn.Module, _args: Any, output: Any) -> None:
        hidden, _ = _split_output(output)
        store.append(hidden.detach())

    handle = module.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


# --------------------------------------------------------------------------- #
# The SAE callable
# --------------------------------------------------------------------------- #

def sae_callable(model: nn.Module) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Wrap an SAE model (DualSAE or any baseline) as ``x -> (x_hat, z)``.

    Convenience only: ``loss_recovered`` takes a PLAIN CALLABLE so that nothing about the
    dual/baseline distinction leaks into the CE code.  Every model family in ``vision/models``
    returns ``(acts, latents, recons)`` from ``forward``, and for DualSAE ``recons`` is
    already the full ``x_hat = x^M + r_hat`` in the input space -- which is the right thing
    to splice back, because that is the model's reconstruction of the residual stream.
    """
    def _fn(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, z, x_hat = model(x)
        return x_hat, z

    return _fn


def _call_sae(sae_fn: Callable[..., Any], x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Normalize the two allowed return shapes of ``sae_fn`` to ``(x_hat, code_or_None)``."""
    out = sae_fn(x)
    if torch.is_tensor(out):
        return out, None
    if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
        code = out[1] if len(out) > 1 and torch.is_tensor(out[1]) else None
        return out[0], code
    raise TypeError(
        "sae_fn must return x_hat, or (x_hat, code); got "
        f"{type(out).__name__}"
    )


# --------------------------------------------------------------------------- #
# Cross-entropy
# --------------------------------------------------------------------------- #

def next_token_ce(logits: torch.Tensor, tokens: torch.Tensor,
                  ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """The reference's loss, verbatim in shape: CE(logits[:, :-1], tokens[:, 1:]).

    float32 is forced (see deviation 4).  A window of length 1 has no next token at all and
    is rejected rather than returning NaN.
    """
    if logits.ndim != 3 or tokens.ndim != 2:
        raise ValueError(f"expected logits (B, L, V) and tokens (B, L), got {tuple(logits.shape)} "
                         f"and {tuple(tokens.shape)}")
    if logits.shape[1] < 2:
        raise ValueError("need at least 2 positions to score a next-token loss")
    vocab = logits.shape[-1]
    return nn.CrossEntropyLoss(ignore_index=ignore_index)(
        logits[:, :-1, :].reshape(-1, vocab).float(),
        tokens[:, 1:].reshape(-1).long(),
    )


@torch.no_grad()
def next_token_kl(logits_clean: torch.Tensor, logits_patched: torch.Tensor,
                  chunk: int = 8) -> float:
    """Mean KL( clean || patched ) over the next-token distributions.

    WHY KL AND NOT ONLY loss-recovered: loss-recovered divides by the headroom that ZERO-ABLATING
    the site creates, which is an arbitrary reference and an enormous one -- measured here at
    7.5-8.7 nats against an original loss of 3.2.  That compresses every usable result into the
    top few percent of the scale: a 33x reduction in added loss shows up as 0.926 -> 0.998.  KL
    has no such denominator; it is the distance between the two predictive distributions, in
    nats, and is the quantity most of the LM-SAE literature now reports alongside loss recovered.

    DIRECTION: KL(clean || patched), i.e. the clean distribution is the reference.  The reverse
    KL is a different number and the two are not interchangeable, so the direction is fixed here
    and named in the output key rather than left to the caller.

    MEMORY: the logits are (B, L, 50304); in float32 that is ~400 MB per tensor at B*L=2048, and
    a naive implementation holds clean log-probs, patched log-probs and their product at once.
    Chunking over the batch keeps the peak at roughly one extra chunk rather than one extra
    full tensor, which is what lets this run in the same allocation as the three forward passes.
    """
    if logits_clean.shape != logits_patched.shape:
        raise ValueError(f"shape mismatch: {tuple(logits_clean.shape)} vs {tuple(logits_patched.shape)}")
    # Only positions 0..L-2 predict a next token, matching next_token_ce's slicing exactly.
    total, n = 0.0, 0
    b = logits_clean.shape[0]
    for i in range(0, b, chunk):
        lc = logits_clean[i:i + chunk, :-1, :].float()
        lp = logits_patched[i:i + chunk, :-1, :].float()
        logp_clean = torch.log_softmax(lc, dim=-1)
        logp_patch = torch.log_softmax(lp, dim=-1)
        # sum_v P_clean * (log P_clean - log P_patch), summed over vocab, meaned over positions.
        kl = (logp_clean.exp() * (logp_clean - logp_patch)).sum(dim=-1)
        total += float(kl.sum().item())
        n += kl.numel()
        del lc, lp, logp_clean, logp_patch, kl
    return total / max(n, 1)


@torch.no_grad()
def loss_recovered(
    model: nn.Module,
    tokens: torch.Tensor,
    module: nn.Module,
    sae_fn: Callable[..., Any],
    norm_factor: float,
    convention: str = "multiply",
    ignore_index: int = IGNORE_INDEX,
    alive_counter: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """The three cross-entropies for ONE batch of packed token windows.

    Parameters
    ----------
    tokens          (B, L) int64, already on the model's device.  PACKED: no padding.
    module          the block whose output is spliced (``resolve_layer_module``).
    sae_fn          plain callable ``x_2d -> x_hat_2d`` or ``-> (x_hat_2d, code_2d)``.
                    It sees the activation in SHARD SPACE and must return a reconstruction
                    in the same space.
    norm_factor     the fixed global factor from the shard manifest, applied on the way in
                    and undone on the way out.
    alive_counter   optional (dict_size,) int64 accumulator; incremented per latent that
                    fires anywhere in this batch (requires ``sae_fn`` to return the code).

    Returns a dict of python floats plus ``n_predicted`` (the number of CE terms), which the
    caller needs to weight batches correctly.
    """
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be (B, L); got {tuple(tokens.shape)}")

    # ---- pass 1: the unmodified model.  No hook is installed at all.
    logits = model(input_ids=tokens, use_cache=False).logits
    loss_original = float(next_token_ce(logits, tokens, ignore_index).item())
    # Held (not deleted) ONLY until pass 2 has produced its logits, because KL needs both
    # distributions at once. Freed immediately afterwards, so the peak is two logit tensors
    # rather than one -- the same peak the reference already pays inside its own tracing.
    logits_clean = logits
    del logits

    # ---- pass 2: splice in the SAE reconstruction.
    # Diagnostics captured from inside the hook, so they describe the tensor the SAE
    # actually saw rather than a re-derivation of it.
    stats: Dict[str, float] = {}

    def _reconstruct(hidden: torch.Tensor) -> torch.Tensor:
        b, length, d = hidden.shape
        flat = hidden.reshape(b * length, d).float()          # SAE models index shape[0] as batch
        scaled = to_shard_space(flat, norm_factor, convention)
        # Mean row norm in shard space: if the shards were folded to mean norm sqrt(d) this
        # lands near sqrt(d_model); a wildly different value means the manifest's factor or
        # its convention does not match these activations.  Recorded, not enforced -- the
        # folding target is the extraction stage's choice, not ours.
        stats["mean_scaled_norm"] = float(scaled.norm(dim=-1).mean().item())
        x_hat, code = _call_sae(sae_fn, scaled)
        if x_hat.shape != scaled.shape:
            raise ValueError(
                f"sae_fn returned {tuple(x_hat.shape)} for an input of {tuple(scaled.shape)}; "
                "the reconstruction must live in the same space as the input"
            )
        if code is not None:
            stats["l0"] = float((code != 0).float().sum(dim=-1).mean().item())
            if alive_counter is not None:
                fired = (code != 0).any(dim=0).to(alive_counter.dtype)
                if fired.numel() != alive_counter.numel():
                    raise ValueError(
                        f"alive_counter has {alive_counter.numel()} slots but the code has "
                        f"{fired.numel()} latents")
                # IN-PLACE on purpose: `alive_counter += ...` inside this closure would make
                # the name local to _reconstruct and shadow the enclosing parameter.
                alive_counter.add_(fired.to(alive_counter.device))
        restored = from_shard_space(x_hat, norm_factor, convention)
        # Back to the LM's own dtype -- the reference's ``.to(model.dtype)``.
        return restored.reshape(b, length, d).to(hidden.dtype)

    with splice_activation(module, _reconstruct):
        logits = model(input_ids=tokens, use_cache=False).logits
    loss_reconstructed = float(next_token_ce(logits, tokens, ignore_index).item())
    kl_clean_vs_reconstructed = next_token_kl(logits_clean, logits)
    del logits, logits_clean

    # ---- pass 3: zero the activation.  ``torch.zeros_like`` keeps dtype/device/layout.
    with splice_activation(module, torch.zeros_like):
        logits = model(input_ids=tokens, use_cache=False).logits
    loss_zero = float(next_token_ce(logits, tokens, ignore_index).item())
    del logits

    out: Dict[str, Any] = {
        "loss_original": loss_original,
        "loss_reconstructed": loss_reconstructed,
        "loss_zero": loss_zero,
        # Mean KL(clean || patched) in nats. Direction is in the key name on purpose.
        "kl_clean_vs_reconstructed": kl_clean_vs_reconstructed,
        "n_predicted": int(tokens.shape[0] * (tokens.shape[1] - 1)),
    }
    out.update(stats)
    return out


def frac_recovered(loss_original: float, loss_reconstructed: float, loss_zero: float) -> float:
    """``(loss_reconstructed - loss_zero) / (loss_original - loss_zero)`` -- the reference line.

    The denominator is the headroom ablating the site creates.  It is ~0 only if zeroing the
    residual stream does not hurt the model at all, which for a mid-stack residual site does
    not happen; NaN is returned rather than a huge number if it ever does, so a degenerate
    cell is visible instead of merely extreme.
    """
    denom = loss_original - loss_zero
    if abs(denom) < 1e-12:
        return float("nan")
    return (loss_reconstructed - loss_zero) / denom


# --------------------------------------------------------------------------- #
# Packed token streams
# --------------------------------------------------------------------------- #

def iter_token_batches(
    path: str,
    batch_sequences: int = DEFAULT_BATCH_SEQUENCES,
    ctx_len: int = config.LM_CTX_LEN,
    max_sequences: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Iterator[torch.Tensor]:
    """Yield ``(batch_sequences, ctx_len)`` int64 windows from a packed uint16 stream.

    The stream (``config.token_stream(split)``) is a flat ``.npy`` array of token ids with
    NO document padding, memory-mapped here so a 64M-token file costs no RAM.  Windows are a
    contiguous, deterministic prefix: the CE estimate must not move between a rerun and a
    resume, and every cell must be scored on the SAME tokens or the comparison is noise.

    Trailing tokens that do not fill a whole window are dropped, and so is a trailing PARTIAL
    batch: for a BatchTopK cell the top-k is taken over the whole batch, so a short final
    batch would be scored at a different sparsity than every other batch.

    uint16 is widened to int64 because ``nn.Embedding`` indices must be long; uint16 is
    enough to hold Pythia's 50304-entry vocabulary, which is why the stream is stored that
    way in the first place.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no packed token stream at {path}; run the Track B token-packing stage first")
    stream = np.load(path, mmap_mode="r")
    if stream.ndim != 1:
        raise ValueError(f"{path}: expected a flat 1-D token stream, got shape {stream.shape}")

    n_windows = int(stream.shape[0]) // int(ctx_len)
    if max_sequences is not None:
        n_windows = min(n_windows, int(max_sequences))
    n_batches = n_windows // int(batch_sequences)      # drop the partial final batch
    if n_batches == 0:
        raise ValueError(
            f"{path}: holds {stream.shape[0]} tokens -> {n_windows} windows of {ctx_len}, "
            f"which is fewer than one batch of {batch_sequences}")

    rows = int(batch_sequences) * int(ctx_len)
    for b in range(n_batches):
        flat = np.asarray(stream[b * rows:(b + 1) * rows], dtype=np.int64)
        batch = torch.from_numpy(flat).view(int(batch_sequences), int(ctx_len))
        yield batch.to(device) if device is not None else batch


# --------------------------------------------------------------------------- #
# Driver over many batches
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_loss_recovered(
    model: nn.Module,
    token_batches: Sequence[torch.Tensor] | Iterator[torch.Tensor],
    module: nn.Module,
    sae_fn: Callable[..., Any],
    norm_factor: float,
    dict_size: int,
    convention: str = "multiply",
    ignore_index: int = IGNORE_INDEX,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Aggregate ``loss_recovered`` over an iterable of token batches.

    Aggregation note (a real, if small, deviation): the reference averages the three losses
    AND ``frac_recovered`` over batches, i.e. it reports a mean of per-batch ratios.  Here the
    three losses are aggregated token-weighted and ``frac_recovered`` is formed from the
    aggregates -- a ratio of means, which is the better-conditioned estimator and is what the
    metric's definition asks for when the batches are a partition of one token set.  With
    equal-length packed windows and the partial batch dropped the two aggregations of the
    LOSSES coincide exactly; only the ratio differs, so BOTH are reported
    (``frac_recovered`` and ``frac_recovered_batchmean``) and neither is silently preferred.

    ``frac_alive`` counts latents that fired at least once anywhere in the CE token set.
    The reference instead accumulates ``sum_i f_ij`` and tests it against 0, which is the
    same predicate for a non-negative code (every activation in this repo passes through
    ReLU/TopK/JumpReLU) but not in general; the sum-based value is reported too, as
    ``frac_alive_ref``, so the parity with ``dictionary_learning`` is checkable.
    """
    alive_counter = torch.zeros(int(dict_size), dtype=torch.int64,
                                device=device or torch.device("cpu"))
    alive_sum = torch.zeros(int(dict_size), dtype=torch.float64,
                            device=device or torch.device("cpu"))

    # The sum-based counterpart needs the code itself, so it is accumulated through a thin
    # wrapper rather than by changing the sae_fn contract.
    def _wrapped(x: torch.Tensor):
        x_hat, code = _call_sae(sae_fn, x)
        if code is not None:
            alive_sum.add_(code.detach().to(alive_sum.dtype).sum(dim=0).to(alive_sum.device))
        return x_hat, code

    # Every key here is accumulated as a TOKEN-WEIGHTED sum and divided by n_predicted below,
    # so a final batch of a different size cannot skew the mean. The KL belongs in this dict for
    # exactly that reason: a plain mean over batches would weight a short tail batch equally.
    totals = {"loss_original": 0.0, "loss_reconstructed": 0.0, "loss_zero": 0.0,
              "kl_clean_vs_reconstructed": 0.0}
    n_predicted = 0
    n_batches = 0
    batch_fracs: List[float] = []
    l0_sum = 0.0
    l0_n = 0
    mean_scaled_norm_sum = 0.0

    for tokens in token_batches:
        if device is not None:
            tokens = tokens.to(device)
        res = loss_recovered(model, tokens, module, _wrapped, norm_factor, convention,
                             ignore_index, alive_counter)
        w = res["n_predicted"]
        for key in totals:
            totals[key] += res[key] * w
        n_predicted += w
        n_batches += 1
        batch_fracs.append(frac_recovered(res["loss_original"], res["loss_reconstructed"],
                                          res["loss_zero"]))
        if "l0" in res:
            l0_sum += res["l0"]
            l0_n += 1
        mean_scaled_norm_sum += res.get("mean_scaled_norm", float("nan"))
        if verbose and (n_batches % 20 == 0 or n_batches == 1):
            print(f"  [ce] batch {n_batches}: orig={res['loss_original']:.4f} "
                  f"recon={res['loss_reconstructed']:.4f} zero={res['loss_zero']:.4f} "
                  f"frac={batch_fracs[-1]:.4f}")

    if n_batches == 0:
        raise ValueError("evaluate_loss_recovered: the token batch iterator was empty")

    means = {k: v / max(n_predicted, 1) for k, v in totals.items()}
    finite = [f for f in batch_fracs if np.isfinite(f)]
    out: Dict[str, Any] = {
        **means,
        "frac_recovered": frac_recovered(means["loss_original"], means["loss_reconstructed"],
                                         means["loss_zero"]),
        "frac_recovered_batchmean": float(np.mean(finite)) if finite else float("nan"),
        "frac_recovered_batch_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "frac_alive": float((alive_counter > 0).sum().item()) / float(dict_size),
        "frac_alive_ref": float((alive_sum != 0).sum().item()) / float(dict_size),
        "n_ce_batches": n_batches,
        # One target is lost per window (position 0 has no predecessor), so the number of
        # CE terms is n_windows * (ctx_len - 1), not n_windows * ctx_len.
        "n_ce_predicted_tokens": n_predicted,
        "dict_size": int(dict_size),
        "norm_factor": float(norm_factor),
        "norm_convention": convention,
        "mean_scaled_norm": mean_scaled_norm_sum / max(n_batches, 1),
        "ignore_index": int(ignore_index),
        "l0_ce_tokens": (l0_sum / l0_n) if l0_n else None,
        "note": ("loss recovered as defined in dictionary_learning/evaluation.py: "
                 "(loss_reconstructed - loss_zero) / (loss_original - loss_zero) over "
                 "next-token CE; hooks replace nnsight; ignore_index=-100 because the token "
                 "stream is packed and has no padding"),
    }
    return out


__all__ = [
    "IGNORE_INDEX",
    "DEFAULT_BATCH_SEQUENCES",
    "shard_manifest_path",
    "read_shard_manifest",
    "norm_factor_from_manifest",
    "to_shard_space",
    "from_shard_space",
    "resolve_layer_module",
    "splice_activation",
    "capture_activation",
    "sae_callable",
    "next_token_ce",
    "loss_recovered",
    "frac_recovered",
    "iter_token_batches",
    "evaluate_loss_recovered",
]
