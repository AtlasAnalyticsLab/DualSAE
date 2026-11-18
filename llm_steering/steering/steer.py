"""Steering primitives shared by calibration (s4/s5), generation (s6) and scoring (s7).

LM (D-49 "fp32 LM"): pythia-410m-deduped from the LOCAL snapshot (common/lm_model.py) in FULL
fp32 -- TF32 matmuls OFF (torch's default, the precision the language pipeline's CE numbers are
scored in).  Why not TF32 like the residual extraction: under TF32 an ulp-level fp32 perturbation
(e.g. the raw->shard->raw round trip) flips TF32 rounding of whole mantissas and comes out as
~1e-3 relative noise; measured on an H100, the round-trip splice then gives
next-token KL 3.6e-6, failing the D-49 precondition (< 1e-6), while in full fp32 it is ~1e-8.  The
SAEs' training residuals were extracted with TF32 and stored in fp16, so their own quantisation
(~5e-4 relative) already exceeds the fp32-vs-TF32 difference.  SAE maths inside the hook also runs
in full fp32 (sae_io._fp32_exact), as every other steering encode does.

The edit hook (D-47, D-49, C30)
  A forward hook on ``lm.gpt_neox.layers[L]`` registered with ``with_kwargs=True``; it reads the
  block OUTPUT (resid_post of block L, the site every Track B SAE was trained on).
  Edit mask = (kwargs['position_ids'] >= 1) AND (the 2-D padding mask, held by the hook object and
  set by the caller before every forward -- the layer's own ``attention_mask`` kwarg is the 4-D
  causal mask, useless for padding).  So the attention sink (position 0 of the real text) and
  left pads are never edited, in prefill, in every KV-cached decode step and in every
  teacher-forced pass (which always passes position_ids = cumsum(mask) - 1 explicitly).
  Modes (per batch row, so one forward can carry up to 16 directions x windows):
    'identity'  x' = x + 0 through the full masking path (the identity-hook precondition test)
    'add'       mode (i):  x' = x + alpha * nf * u               (u unit, shard-space alpha)
    'clamp'     mode (ii'): z'_j = c;  x' = x + nf * (c - z_j) * W_dec[j]   (b_dec cancels; no
                re-sparsification; SAE error and, for Duals, poly output untouched)
    'rerun'     mode (iii), Duals only: x'_s = xM(z') + poly(x_s - xM(z')) + (x_s - xhat_s)
  Rows whose ``active`` flag is False are never edited (fillers, unreachable budgets).

Sampling (D-49; declared adaptation of the seeding mechanism)
  The warper chain of HF generate(do_sample=True, temperature=0.7, top_p=0.9) in the installed
  transformers 5.8 is reproduced: TemperatureLogitsWarper (T=0.7) -> TopKLogitsWarper (k=50,
  min_tokens_to_keep=1) -> TopPLogitsWarper (0.9, min_tokens_to_keep=1) -> categorical draw.
  The top_k=50 step is NOT in D-49's text but IS in generate(): pythia ships no
  generation_config.json, so generate() fills top_k from its global defaults
  (configuration_utils.py:573-582 via utils.py:1720-1722) and builds the TopK warper between
  temperature and top-p (utils.py:1235-1244).  Measured on the unsteered P2 prompts, 11% of steps
  have a >50-token nucleus, where T+top-p alone would draw from a different distribution; steered
  (flatter) distributions differ more.  D-49 names generate(), so its full chain is kept (declared
  in prereg.SPECS['steering']).  The draw uses ONE uniform per (item, continuation,
  step) from a dedicated CPU torch.Generator seeded with the D-49 item seed
  derive_seed(mode, budget, concept, rank, prompt_idx) -- the per-item equivalent of
  "torch.manual_seed(...) immediately before each generate() call" (one call per item that
  returns its n continuations).  Why a per-item generator instead of the global RNG: items must
  be batched for throughput (~10^6 generations), and with torch.multinomial a row's random
  numbers would depend on its position in the batch; per-item streams make every continuation
  independent of batch composition, of the arm and of resumption, so arms share random numbers
  exactly (common random numbers) and a resumed job reproduces an uninterrupted one bit for bit.
  Inverse-CDF sampling with a given uniform is an exact categorical sampler.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from common import lang_config as config
from steering import base
from common import sae_io
from common import loss_recovered_hooks as H

EOS_ID = 0            # pythia: bos = eos = unk = '<|endoftext|>' (id 0); no pad token -> pad = eos


# =========================================================================== LM
def set_full_fp32() -> None:
    """TF32 off for every matmul/conv in this process (module docstring)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def load_lm(device: str):
    """(lm, tokenizer): full-fp32 pythia (TF32 off, see the module docstring), eval mode, frozen."""
    from common import lm_model as xr
    set_full_fp32()
    lm = xr.load_model(xr.resolve_model_path(None), torch.device(device), torch.float32)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(xr.resolve_model_path(None), local_files_only=True)
    tok.pad_token = tok.eos_token          # filler only: every pad is masked (D-49)
    tok.padding_side = "left"
    if tok.eos_token_id != EOS_ID:
        raise RuntimeError(f"unexpected pythia eos id {tok.eos_token_id}")
    return lm, tok


def norm_factor(layer: int) -> Tuple[float, str]:
    """(nf, convention) from the layer's owt TEST shard manifest (raw = shard * nf, 'divide')."""
    man = H.read_shard_manifest(layer, "test", base.CORPUS)
    nf, conv = H.norm_factor_from_manifest(man, verbose=False)
    if conv != "divide":
        raise RuntimeError(f"L{layer}: norm_convention {conv!r}; A16 assumes 'divide'")
    return nf, conv


def owt_windows(start_window: int, n_windows: int, split: str = "test") -> torch.Tensor:
    """Token ids of packed owt windows [start, start+n) (the same stream the shards came from, so
    window w's residual rows are shard rows [w*128, (w+1)*128))."""
    stream = np.load(config.lm_token_stream(base.CORPUS, split), mmap_mode="r")
    lo, hi = start_window * base.CTX_LEN, (start_window + n_windows) * base.CTX_LEN
    if hi > stream.shape[0]:
        raise ValueError(f"windows [{start_window},{start_window + n_windows}) exceed the {split} stream")
    return torch.from_numpy(np.asarray(stream[lo:hi], dtype=np.int64)).view(n_windows, base.CTX_LEN)


# =========================================================================== SAE adapter
def atom_matrix(model: torch.nn.Module) -> torch.Tensor:
    """Decoder atoms as (n_latents, d_model), whatever the family calls them (same logic as the
    language pipeline's qualitative token maps).  TopKSAE has no W_dec: its decoder (tied TiedTranspose or untied nn.Linear) exposes
    ``decoder.weight`` in (d_model, n_latents) layout, which is transposed."""
    if hasattr(model, "W_dec"):
        return model.W_dec.detach().float()
    dec = getattr(model, "decoder", None)
    if dec is not None and hasattr(dec, "weight"):
        return dec.weight.detach().float().t()
    raise AttributeError(f"{type(model).__name__}: no W_dec / decoder.weight")


class SAEAdapter:
    """Family-agnostic view of one loaded Track B SAE for the hooks.

    Track B arms are all normalize=False / input_unit_norm=False (config.TRACKB_NORMALIZE), so the
    reconstruction is AFFINE in the code (decode(z) = z @ W + bias) and a code change of dz on
    latent j moves the reconstruction by exactly dz * W[j].  That is checked, not assumed
    (check_affine), because mode (ii') relies on it."""

    def __init__(self, model: torch.nn.Module, meta: Dict[str, Any], threshold: Optional[float]):
        self.model, self.meta = model, meta
        self.family = meta["family"]
        self.is_dual = bool(meta["is_dual"])
        if meta.get("input_unit_norm") or getattr(model, "normalize", False):
            raise RuntimeError("A16 adapter assumes normalize=False (Track B); got a normalising SAE")
        if self.family == "batch_topk" and threshold is None:
            raise ValueError("BatchTopK arms need the D-44 inference threshold")
        self.threshold = threshold
        self.W = atom_matrix(model)                                  # (F, D)
        self.atom_norm = self.W.norm(dim=1)
        self.U = self.W / self.atom_norm.clamp_min(1e-12)[:, None]   # unit atoms (mode i)

    @torch.no_grad()
    def encode(self, x_s: torch.Tensor) -> torch.Tensor:
        """Per-token inference code (BatchTopK: D-44 frozen threshold; others: own forward)."""
        with sae_io._fp32_exact():
            if self.family == "batch_topk":
                return sae_io.encode_thresholded(self.model, self.meta, x_s, self.threshold,
                                                 fixed_rows=None)
            return self.model(x_s)[1]

    @torch.no_grad()
    def dual_parts(self, x_s: torch.Tensor) -> Dict[str, torch.Tensor]:
        """z, xM, r_hat, xhat of a Dual under the inference encoder (xM + r_hat == xhat)."""
        if not self.is_dual:
            raise ValueError("dual_parts on a baseline arm")
        with sae_io._fp32_exact():
            if self.family == "batch_topk":
                return sae_io.forward_thresholded(self.model, self.meta, x_s, self.threshold,
                                                  fixed_rows=None)
            _, z, xhat = self.model(x_s)
            return {"z": z, "xM": self.model._xM, "r_hat": self.model._r_hat, "xhat": xhat}

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Mono/single-branch reconstruction z -> x (no poly)."""
        with sae_io._fp32_exact():
            if hasattr(self.model, "W_dec"):
                return z @ self.model.W_dec + self.model.b_dec
            return self.model.decoder(z) + self.model.pre_bias

    @torch.no_grad()
    def poly(self, r: torch.Tensor) -> torch.Tensor:
        with sae_io._fp32_exact():
            return self.model.poly(r)

    @torch.no_grad()
    def check_affine(self, x_s: torch.Tensor, latents: Sequence[int] = (), dz: float = 0.37
                     ) -> Dict[str, float]:
        """Mode (ii') moves x by nf * dz * W[j] ONLY IF decode is affine in z.  Measures
        max |decode(z + dz e_j) - decode(z) - dz W[j]| over latent 0, the last latent and the
        given latents, relative to the larger of the output scale max|decode(z)| and the edit
        scale dz * max|W[j]| (fp32 rounding of the decode sum is ~1e-7 of the output; a
        non-affine decoder -- output normalisation, a nonlinearity -- is O(1))."""
        z = self.encode(x_s)
        base_out = self.decode(z)
        worst_abs, worst_rel = 0.0, 0.0
        F_ = int(self.W.shape[0])
        for j in sorted({0, F_ - 1, *[int(v) for v in latents]}):
            z2 = z.clone()
            z2[:, j] += dz
            err = float((self.decode(z2) - base_out - dz * self.W[j]).abs().max())
            scale = max(float(base_out.abs().max()), dz * float(self.W[j].abs().max()), 1e-30)
            worst_abs, worst_rel = max(worst_abs, err), max(worst_rel, err / scale)
        return {"max_abs": worst_abs, "max_rel": worst_rel}


# =========================================================================== the edit hook
class EditHook:
    """See the module docstring.  Usage::

        hook = EditHook(lm, layer, nf, adapter)          # adapter None for mode 'add' only
        with hook.installed():
            hook.configure('add', delta_raw=..., active=...)
            hook.attn2d = attention_mask                   # before EVERY forward
            lm(...)
    """

    def __init__(self, lm, layer: int, nf: float, adapter: Optional[SAEAdapter] = None,
                 token_chunk: int = 16384):
        self.module = lm.gpt_neox.layers[layer]
        self.layer, self.nf, self.adapter = layer, float(nf), adapter
        self.mode: Optional[str] = None
        self.params: Dict[str, torch.Tensor] = {}
        self.attn2d: Optional[torch.Tensor] = None
        self.token_chunk = token_chunk
        self.n_calls = 0
        self.n_edited_tokens = 0
        self._handle = None

    @contextlib.contextmanager
    def installed(self):
        self._handle = self.module.register_forward_hook(self._hook, with_kwargs=True)
        try:
            yield self
        finally:
            self._handle.remove()
            self._handle = None

    def configure(self, mode: Optional[str], **params: torch.Tensor) -> None:
        if mode not in (None, "identity", "add", "clamp", "rerun"):
            raise ValueError(mode)
        if mode in ("clamp", "rerun") and self.adapter is None:
            raise ValueError(f"mode {mode} needs an SAE adapter")
        if mode == "rerun" and not self.adapter.is_dual:
            raise ValueError("mode (iii) exists for Dual arms only")
        self.mode, self.params = mode, params

    def _mask(self, kwargs: Dict[str, Any], B: int, T: int, device) -> torch.Tensor:
        pos = kwargs.get("position_ids")
        if pos is None:
            raise RuntimeError("the layer received no position_ids kwarg; every A16 forward must "
                               "pass explicit position_ids (C30)")
        pos = pos.to(device)
        if pos.dim() == 1:
            pos = pos[None]
        pos = pos.expand(B, T)
        if self.attn2d is None:
            raise RuntimeError("EditHook.attn2d not set before the forward")
        m2 = self.attn2d[:, -T:].to(device).bool()
        if m2.shape != (B, T):
            raise RuntimeError(f"2-D mask {tuple(m2.shape)} does not match hidden {(B, T)}")
        mask = (pos >= 1) & m2
        act = self.params.get("active")
        if act is not None:
            mask = mask & act.to(device).bool()[:, None]
        return mask

    def _hook(self, mod, args, kwargs, output):
        if self.mode is None:
            return None                                    # pass-through (output unchanged)
        hidden, rebuild = H._split_output(output)
        B, T, D = hidden.shape
        self.n_calls += 1
        mask = self._mask(kwargs, B, T, hidden.device)
        new = self.apply(hidden, mask)
        return None if new is None else rebuild(new)

    @torch.no_grad()
    def apply(self, hidden: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
        """The edit itself on raw-space hidden states (B, T, D) with a boolean edit mask (B, T);
        per-row parameters index the batch dimension.  Used by the forward hook AND by WindowKL's
        cached-residual path, so calibration and generation run the same code.  Returns the new
        hidden tensor (None = unchanged)."""
        B, T, D = hidden.shape
        if self.mode == "identity":
            zero = torch.zeros(B, D, device=hidden.device, dtype=hidden.dtype)
            return torch.where(mask[..., None], hidden + zero[:, None, :], hidden)
        if self.mode == "add":
            d = self.params["delta_raw"].to(hidden.device, hidden.dtype)   # (B, D) raw delta/row
            self.n_edited_tokens += int(mask.sum())
            return hidden + mask[..., None].to(hidden.dtype) * d[:, None, :]
        # clamp / rerun: encode only the masked tokens (the only ones edited).
        bi, ti = torch.nonzero(mask, as_tuple=True)
        if bi.numel() == 0:
            return None
        self.n_edited_tokens += int(bi.numel())
        new = hidden.clone()
        j_all, c_all = self.params["j"].to(hidden.device), self.params["c"].to(hidden.device)
        A = self.adapter
        for s in range(0, bi.numel(), self.token_chunk):
            b, t = bi[s:s + self.token_chunk], ti[s:s + self.token_chunk]
            x_s = H.to_shard_space(hidden[b, t].float(), self.nf, "divide")
            j, c = j_all[b], c_all[b].float()
            if self.mode == "clamp":
                z = A.encode(x_s)
                zj = z.gather(1, j[:, None]).squeeze(1)
                d_s = (c - zj)[:, None] * A.W[j]
            else:
                parts = A.dual_parts(x_s)
                zj = parts["z"].gather(1, j[:, None]).squeeze(1)
                xM_new = parts["xM"] + (c - zj)[:, None] * A.W[j]
                x_new = xM_new + A.poly(x_s - xM_new) + (x_s - parts["xhat"])
                d_s = x_new - x_s
            new[b, t] = hidden[b, t] + H.from_shard_space(d_s, self.nf, "divide").to(hidden.dtype)
        return new


# =========================================================================== sampling
def item_uniforms(seed: int, n_cont: int, max_new: int = base.MAX_NEW_TOKENS) -> torch.Tensor:
    """(n_cont, max_new) uniforms of ONE item from its own generator (module docstring)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.rand((n_cont, max_new), generator=g, dtype=torch.float64)


def sample_hf(logits: torch.Tensor, u: torch.Tensor, temperature: float = base.TEMPERATURE,
              top_k: int = base.TOP_K, top_p: float = base.TOP_P) -> torch.Tensor:
    """HF generate()'s sampling warpers, verbatim logic, then an inverse-CDF draw with the given
    uniform per row (module docstring).  logits (B, V); u (B,) in [0, 1).

      TemperatureLogitsWarper   scores / T
      TopKLogitsWarper          k = min(max(top_k, min_tokens_to_keep=1), V); every score STRICTLY
                                below the k-th largest is set to -inf (ties at the k-th value are
                                kept, exactly as HF's ``scores < topk(...)[0][..., -1, None]``)
      TopPLogitsWarper          ascending sort, softmax, cumsum; remove cum <= 1 - top_p but always
                                keep the last (largest) token (min_tokens_to_keep=1); the -inf
                                entries left by top-k have probability 0 and are removed again
      draw                      softmax of the surviving scores in float64, sorted descending,
                                first index whose CDF reaches u * total (an exact categorical
                                sampler for a uniform u; the sort only fixes a canonical order).
    """
    scores = logits.float() / float(temperature)
    if top_k is not None and top_k > 0:
        k = min(max(int(top_k), 1), scores.shape[-1])
        kth = torch.topk(scores, k, dim=-1)[0][..., -1, None]
        scores = scores.masked_fill(scores < kth, -float("inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scores, descending=False, dim=-1)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove_sorted = cum <= (1 - top_p)
        remove_sorted[..., -1:] = False
        remove = remove_sorted.scatter(1, sorted_idx, remove_sorted)
        scores = scores.masked_fill(remove, -float("inf"))
    probs = scores.softmax(dim=-1).double()
    p_sorted, idx_sorted = probs.sort(dim=-1, descending=True)
    cdf = p_sorted.cumsum(dim=-1)
    target = u.to(cdf.device, cdf.dtype)[:, None] * cdf[:, -1:]
    k_idx = torch.searchsorted(cdf, target).clamp_max(cdf.shape[-1] - 1)
    return idx_sorted.gather(1, k_idx).squeeze(1)


# =========================================================================== generation
def left_pad(seqs: Sequence[Sequence[int]], device) -> Tuple[torch.Tensor, torch.Tensor]:
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), EOS_ID, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s):
            ids[i, L - len(s):] = torch.as_tensor(list(s), dtype=torch.long)
            att[i, L - len(s):] = 1
    return ids.to(device), att.to(device)


def positions(att: torch.Tensor) -> torch.Tensor:
    """position_ids = (cumsum(mask) - 1).clamp_min(0): generate()'s rule (generation/utils.py
    708-730); pads and the first real token get 0."""
    return (att.cumsum(-1) - 1).clamp_min(0)


@torch.no_grad()
def generate(lm, prompts: Sequence[Sequence[int]], hook: Optional[EditHook],
             uniforms: Optional[torch.Tensor], max_new: int = base.MAX_NEW_TOKENS,
             use_cache: bool = True, greedy: bool = False) -> torch.Tensor:
    """Decode ``max_new`` tokens for each prompt (rows).  Returns (B, max_new) token ids; after a
    row emits EOS every later token is EOS (HF's pad-after-finish rule, pad = eos) and the row's
    mask keeps growing with ones, as in generate().  ``hook`` (already installed and configured)
    gets the running 2-D mask before every forward.  ``uniforms`` (B, max_new) drives sampling;
    ``greedy`` takes the argmax instead (identity-hook test)."""
    dev = next(lm.parameters()).device
    ids, att = left_pad(prompts, dev)
    B = ids.shape[0]
    if ids.shape[1] + max_new > base.CTX_LEN:
        raise ValueError(f"prompt {ids.shape[1]} + {max_new} new tokens exceeds the SAE window "
                         f"{base.CTX_LEN} (D-49)")
    if hook is not None:
        hook.attn2d = att
    out = lm(input_ids=ids, attention_mask=att, position_ids=positions(att), use_cache=use_cache)
    logits = out.logits[:, -1, :]
    past = out.past_key_values if use_cache else None
    gen = torch.empty((B, max_new), dtype=torch.long, device=dev)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    for t in range(max_new):
        if greedy:
            nxt = logits.float().argmax(dim=-1)
        else:
            nxt = sample_hf(logits, uniforms[:, t])
        nxt = torch.where(finished, torch.full_like(nxt, EOS_ID), nxt)
        gen[:, t] = nxt
        finished |= nxt == EOS_ID
        if t == max_new - 1:
            break
        att = torch.cat([att, torch.ones((B, 1), dtype=att.dtype, device=dev)], dim=1)
        if hook is not None:
            hook.attn2d = att
        if use_cache:
            pos = (att.sum(-1, keepdim=True) - 1)
            out = lm(input_ids=nxt[:, None], attention_mask=att, position_ids=pos,
                     past_key_values=past, use_cache=True)
            past = out.past_key_values
        else:
            full = torch.cat([ids, gen[:, : t + 1]], dim=1)
            out = lm(input_ids=full, attention_mask=att, position_ids=positions(att),
                     use_cache=False)
        logits = out.logits[:, -1, :]
    return gen


def cut_at_eos(gen_row: Sequence[int]) -> Tuple[List[int], bool]:
    """Continuation tokens before the first EOS, and whether an EOS was emitted."""
    out = []
    for t in gen_row:
        if int(t) == EOS_ID:
            return out, True
        out.append(int(t))
    return out, False


# =========================================================================== teacher forcing
@torch.no_grad()
def teacher_forced_logp(lm, prompts: Sequence[Sequence[int]], conts: Sequence[Sequence[int]],
                        hook: Optional[EditHook] = None) -> List[torch.Tensor]:
    """Per row: log-probs (n_cont, V) of the next-token distributions that PREDICT each
    continuation token, from a left-padded teacher-forced pass with explicit position ids (C30).
    Rows with an empty continuation return an empty tensor."""
    dev = next(lm.parameters()).device
    seqs = [list(p) + list(c) for p, c in zip(prompts, conts)]
    ids, att = left_pad(seqs, dev)
    if hook is not None:
        hook.attn2d = att
    logits = lm(input_ids=ids, attention_mask=att, position_ids=positions(att),
                use_cache=False).logits
    L = ids.shape[1]
    out = []
    for i, (p, c) in enumerate(zip(prompts, conts)):
        n = len(c)
        if n == 0:
            out.append(logits.new_zeros((0, logits.shape[-1])))
            continue
        # token at absolute index k is predicted by position k-1; continuation occupies the last
        # n positions of the row, so its predictors are positions [L-n-1, L-1).
        out.append(torch.log_softmax(logits[i, L - n - 1: L - 1].float(), dim=-1))
    return out


# =========================================================================== window KL (C29)
def forward_from_layer(lm, h_L: torch.Tensor, layer: int) -> torch.Tensor:
    """Final hidden states (after final_layer_norm) given the OUTPUT of block ``layer`` for
    unpadded windows: runs blocks layer+1 .. 23 exactly as GPTNeoXModel.forward does (same
    rotary embeddings, same causal mask with attention_mask=None, same per-layer kwargs;
    transformers 5.8 modeling_gpt_neox.py:350-373).  Equality with the full forward is asserted
    by WindowKL.__init__ (bit-exact on the clean windows)."""
    from transformers.masking_utils import create_causal_mask
    m = lm.gpt_neox
    B, T, _ = h_L.shape
    pos = torch.arange(T, device=h_L.device)[None].expand(B, T)
    causal = create_causal_mask(config=m.config, inputs_embeds=h_L, attention_mask=None,
                                past_key_values=None, position_ids=pos)
    pe = m.rotary_emb(h_L, position_ids=pos)
    h = h_L
    for blk in m.layers[layer + 1:]:
        h = blk(h, attention_mask=causal, position_ids=pos, layer_past=None, use_cache=False,
                position_embeddings=pe)
    return m.final_layer_norm(h)


class WindowKL:
    """Teacher-forced next-token KL(clean || edited) on fixed owt test windows (D-48 / C29):
    32 windows, editing and scoring positions 1..127, clean log-probs cached once.

    KL is averaged over windows and positions 1..127 (the distributions at those positions; the
    position-0 distribution cannot change: the sink is never edited and attention is causal).
    Speed: blocks 0..L never see an edit, so their output on the windows is computed ONCE and
    cached; each evaluation applies the edit with EditHook.apply (the code the generation hook
    runs) and runs only blocks L+1..23 (forward_from_layer).  Up to DIRS_PER_PASS directions share
    one pass: rows d*W .. (d+1)*W-1 carry direction d.
    """

    def __init__(self, lm, layer: int, windows: torch.Tensor, chunk_windows: int = 8):
        self.lm = lm
        self.layer = layer
        self.dev = next(lm.parameters()).device
        self.win = windows.to(self.dev)
        self.W, self.T = int(windows.shape[0]), int(windows.shape[1])
        self.chunk = chunk_windows
        store: List[torch.Tensor] = []
        pos = torch.arange(self.T, device=self.dev)[None].expand(self.W, -1)
        with torch.no_grad(), H.capture_activation(lm.gpt_neox.layers[layer], store):
            full = lm.gpt_neox(input_ids=self.win, position_ids=pos, use_cache=False).last_hidden_state
        self.h_L = store[0].detach()                          # (W, T, D) raw block-L output
        with torch.no_grad():
            part = forward_from_layer(lm, self.h_L, layer)
        self.partial_forward_max_abs_diff = float((part - full).abs().max())
        if self.partial_forward_max_abs_diff > 1e-4:
            raise RuntimeError(f"forward_from_layer disagrees with the full forward "
                               f"({self.partial_forward_max_abs_diff:.2e})")
        with torch.no_grad():
            self.clean = [torch.log_softmax(self.lm.embed_out(full[s:s + self.chunk, 1:]).float(), -1)
                          for s in range(0, self.W, self.chunk)]
        self.mask1 = torch.zeros((self.W, self.T), dtype=torch.bool, device=self.dev)
        self.mask1[:, 1:] = True                              # positions >= 1, no padding
        self.n_forward = 0

    def shard_rows(self, nf: float) -> torch.Tensor:
        """Shard-space residual rows of the calibration windows at positions 1..127 (W*127, D)."""
        return H.to_shard_space(self.h_L[:, 1:].reshape(-1, self.h_L.shape[-1]).float(), nf, "divide")

    @torch.no_grad()
    def kl(self, hook: "EditHook", n_dirs: int) -> np.ndarray:
        """KL per direction for the hook's current per-row configuration (n_dirs*W rows)."""
        h = self.h_L.repeat(n_dirs, 1, 1)
        mask = self.mask1.repeat(n_dirs, 1)
        act = hook.params.get("active")
        if act is not None:
            mask = mask & act.to(self.dev).bool()[:, None]
        h_new = hook.apply(h, mask)
        hf = forward_from_layer(self.lm, h if h_new is None else h_new, self.layer)
        self.n_forward += 1
        out = np.zeros(n_dirs)
        for d in range(n_dirs):
            tot = 0.0
            for ci, s in enumerate(range(0, self.W, self.chunk)):
                hs = hf[d * self.W + s: d * self.W + s + self.chunk, 1:]
                lp = torch.log_softmax(self.lm.embed_out(hs).float(), -1)
                lc = self.clean[ci]
                tot += float((lc.exp() * (lc - lp)).sum(-1).sum())
            out[d] = tot / (self.W * (self.T - 1))
        return out


def rows_for(values: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat per-direction values for n_rep consecutive rows each (direction-major)."""
    return values.repeat_interleave(n_rep, dim=0)
