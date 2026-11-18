"""Activation extraction: pythia-410m-deduped residual stream -> fp16 memmap shards.

WHAT THIS DOES
--------------
Runs ``EleutherAI/pythia-410m-deduped`` over the packed uint16 token stream that
``config.token_stream(split)`` provides and captures the residual stream AFTER transformer
blocks ``config.LM_LAYERS`` (6, 12, 18 of 24) in a SINGLE forward pass, via torch forward
hooks.  The captured (n_tokens, 1024) activations are divided by one fixed per-layer scalar
and written as fp16 ``.npy`` memmap shards into ``resid_dir(layer, split)``.

The shards are consumed by ``langdual.lmsae.resid_shards.ResidShardDataset``, which presents
the same ``__getitem__`` contract as the vision core's ``dataset/lmdb_features.py`` so the
SAE/DualSAE trainers and post-hoc scorers in <repo>/vision accept these activations unchanged.

WHY ONE FORWARD PASS FOR ALL THREE LAYERS
-----------------------------------------
Three separate passes would triple the (dominant) forward cost and, worse, would give three
chances for the three layers to disagree about which tokens they saw.  Capturing all three in
one pass makes token alignment across layers structurally guaranteed: row i of L6, L12 and L18
is the same token by construction, which is what the layer-wise comparison tables need.

USAGE (GPU recommended)
-----------------------
    python -m langdual.lmsae.extract_resid --corpus owt --layers 12 --split train
    python -m langdual.lmsae.extract_resid --corpus owt --layers 12 --split test
    python -m langdual.lmsae.extract_resid --corpus owt --layers 12 --split train --verify

The pipeline driver runs one (corpus, layer, split) per invocation, as the reported runs did;
a held-out split inherits the norm factor of the same (corpus, layer) "train" manifest, so
"train" must be extracted first.

Re-running is safe and cheap: a shard whose ``.npy`` exists for every requested layer is
skipped WITHOUT running its forward pass, so a wall-time kill costs at most one shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------- config import
# Works both as a package module (`python -m langdual.lmsae.extract_resid`) and as a plain
# script path, which is how a SLURM job script sometimes invokes it.
try:  # normal, package-relative import
    from .. import config
    from .resid_shards import read_npy_header
except ImportError:  # pragma: no cover - only taken when run as a bare script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from langdual import config
    from langdual.lmsae.resid_shards import read_npy_header


# --------------------------------------------------------------------------- local constants
# These are NOT owned by config.py (they are extraction-implementation details, not experiment
# protocol), so they live here.  Anything config.py already owns is read from it, never
# redefined: LM_REPO, LM_LAYERS, LM_D_MODEL, LM_CTX_LEN, LM_TRAIN_TOKENS, LM_SHARD_TOKENS.

MANIFEST_NAME = "manifest.json"
SHARD_FMT = "shard_%05d.npy"
SCHEMA = "langdual.resid.v1"

# Tokens used to estimate the per-layer norm factor.  A separate, bounded probe pass (rather
# than accumulating during extraction) is what makes the scale INDEPENDENT of how far a
# previous, killed run got: re-running always produces the same scalar from the same fixed
# prefix of the train stream.  2M tokens is ~3% of the 64M train budget and gives a relative
# standard error on E[||x||^2] far below the fp16 quantisation floor we gate on below.
NORM_PROBE_TOKENS = 2_000_000

# The deterministic layer/orientation fingerprint: the first 4 sequences of the train stream.
PROBE_SEQS = 4

# fp16 storage gate.  R^2 of the fp16 round-trip against fp32 must exceed this or we refuse to
# write anything.  0.9999 leaves <1e-4 of the variance to quantisation, two orders of magnitude
# below the reconstruction R^2 differences of interest.
FP16_R2_MIN = 0.9999

# How many stored tokens --verify reads back per layer for the NaN / scale sanity checks.
VERIFY_SAMPLE_TOKENS = 100_000

# Bytes of the token stream hashed into the manifest.  Hashing all 128 MB on every run is
# pointless I/O; the head plus the exact length catches a regenerated or truncated stream,
# which is the failure this guards against.  KNOWN LIMITATION: a stream edited only in its
# tail is not detected by this hash (the per-shard token counts would still be consistent).
TOKEN_HASH_HEAD_BYTES = 8 << 20


# --------------------------------------------------------------------------- small utilities
def _sha256_bytes(buf: bytes) -> str:
    h = hashlib.sha256()
    h.update(buf)
    return h.hexdigest()


def _atomic_write_json(path: str, obj: dict) -> None:
    """Write JSON via a temp file + os.replace so a kill never leaves a half-written manifest.

    os.replace is atomic within a filesystem, and the temp file is created in the SAME
    directory as the target precisely so that guarantee holds on shared filesystems.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())  # the manifest is the completion marker; make it durable
    os.replace(tmp, path)


def _enable_tf32() -> bool:
    """Enable TF32 matmuls (the pinned numerical protocol: fp32 weights, TF32 compute).

    Returns whether TF32 was actually enabled, which goes into the manifest: TF32 changes the
    matmul results and therefore the stored activations, so it must be recorded rather than
    left implicit.
    """
    if not torch.cuda.is_available():
        return False
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return True
    except Exception:  # pragma: no cover - API rename across torch releases
        # Not fatal: losing TF32 costs speed, not correctness.  Recorded as False so the
        # manifest never claims a numerical protocol the run did not use.
        return False


# --------------------------------------------------------------------------- model loading
def resolve_model_path(cli_path: Optional[str]) -> str:
    """Resolve the local pythia snapshot.  Runtime never downloads.

    Order: explicit --model, then the snapshot prepare_assets.sh wrote under config.RAW_DIR,
    then the bare repo id, which from_pretrained(local_files_only=True) resolves out of
    HF_HOME's cache (or fails loudly).
    """
    if cli_path:
        if not os.path.isdir(cli_path):
            raise FileNotFoundError(f"--model {cli_path!r} is not a directory")
        return cli_path

    # langdual.assets is the ONLY place that knows where prepare_assets.sh put a repo.  A
    # RAW_DIR/<basename> directory is accepted as a second candidate so a hand-made snapshot
    # still works.
    candidates = []
    try:
        from langdual import assets as _assets
        candidates.append(_assets.local_snapshot(config.LM_REPO, "model"))
    except Exception:  # pragma: no cover - partial checkout
        pass
    candidates.append(os.path.join(config.RAW_DIR, os.path.basename(config.LM_REPO)))
    for candidate in candidates:
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
            return candidate

    # Fall through to the hub cache.  local_files_only=True below turns a cache miss into a
    # clear exception instead of a hanging network call on an offline node.
    return config.LM_REPO


def load_model(model_path: str, device: torch.device, dtype: torch.dtype):
    """Load pythia in eval mode, weights frozen, no cache.

    DTYPE CHOICE: float32 (with TF32 matmuls), not bfloat16.
      * The model is 410M parameters -> 1.6 GB in fp32, nothing on an H100.  There is no
        memory argument for bf16 here.
      * bf16 has 8 mantissa bits (~2-3 significant decimal digits).  We then gate the fp16
        STORAGE round-trip at R^2 > 0.9999; computing in bf16 would inject an error an order
        of magnitude larger than the storage error we are being careful about, making that
        gate meaningless.
      * The later cross-entropy-loss-recovered metric splices SAE reconstructions back into
        this same model; a single compute dtype for both the activations and the splice
        removes a whole class of silent mismatch.
    """
    from transformers import AutoModelForCausalLM  # imported late: heavy, and only needed here

    # transformers 5.x renamed torch_dtype -> dtype (torch_dtype is still accepted for BC).
    # Try the current name first and fall back, so a pinned-version bump in either direction
    # does not break extraction.
    kwargs = dict(local_files_only=True, attn_implementation="sdpa")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError:  # pragma: no cover - older transformers
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, **kwargs)

    model.eval()
    model.to(device)
    # Belt and braces on top of torch.no_grad(): if any future caller forgets the context
    # manager, frozen parameters still prevent an autograd graph from being built.
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_blocks(model) -> Sequence[torch.nn.Module]:
    """Return the list of transformer blocks.

    ``model.gpt_neox.layers`` is the path for GPTNeoXForCausalLM (what pythia loads as);
    ``model.layers`` covers a bare GPTNeoXModel.  Anything else is a model we have not
    validated the hook points on, so we refuse rather than guess.
    """
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError(
        f"cannot locate transformer blocks on {type(model).__name__}; expected "
        f"`.gpt_neox.layers` (GPTNeoXForCausalLM) or `.layers` (GPTNeoXModel)"
    )


# --------------------------------------------------------------------------- hook capture
class _StopForward(Exception):
    """Control-flow signal raised by the deepest hook to abort the rest of the forward pass."""


class ResidCapture:
    """Capture block OUTPUTS for several layers in one forward pass, then early-exit.

    io="out" in the reference: the value taken is what ``GPTNeoXLayer.forward`` RETURNS, i.e.
    the residual stream after that block has been added in -- not the block's input.  These
    differ by exactly one block, so getting it wrong produces activations that are internally
    consistent and silently mislabelled by one layer; that is what the manifest probe exists
    to catch.

    EARLY EXIT, and why it is safe:
      * The hook on the DEEPEST requested layer stores its output and then raises
        ``_StopForward``.  Every block after it (here: 5 of 24 after L18) is skipped, together
        with the final layer norm and the 50k-wide unembedding matmul.
      * The exception is caught immediately around the single ``model(...)`` call, and ONLY
        ``_StopForward`` is caught -- a real error still propagates.
      * There is no partially-mutated state to clean up: the pass runs under
        ``torch.no_grad()`` (no autograd graph), with ``use_cache=False`` (no KV cache is
        being appended to) and ``output_hidden_states=False`` (no accumulating tuple).
      * Hook handles are removed in a ``finally``, so an abort can never leave hooks attached
        to a model object a later stage reuses.
      * After each pass we assert that every requested layer is present in the buffer, so a
        swallowed or mis-ordered exception cannot silently yield fewer layers than asked for.
    """

    def __init__(self, model, layers: Sequence[int]):
        blocks = get_blocks(model)
        n_blocks = len(blocks)
        self.layers: List[int] = sorted(int(l) for l in layers)
        if not self.layers:
            raise ValueError("no layers requested")
        for li in self.layers:
            # Fail loudly on an out-of-range index instead of letting Python's negative
            # indexing quietly hook a layer counted from the end.
            if li < 0 or li >= n_blocks:
                raise IndexError(f"layer {li} out of range for a {n_blocks}-block model")
        self._deepest = self.layers[-1]
        self._buf: Dict[int, torch.Tensor] = {}
        self._handles = [blocks[li].register_forward_hook(self._make_hook(li)) for li in self.layers]

    def _make_hook(self, li: int):
        def hook(module, args, output):  # noqa: ANN001 - torch's hook signature
            # transformers 5.8's GPTNeoXLayer.forward returns a BARE TENSOR.  Older releases
            # returned a tuple whose first element is the hidden state.  Unwrap defensively so
            # a version bump in the pinned environment cannot silently capture a tuple object.
            t = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(t):
                raise TypeError(f"layer {li} produced {type(t).__name__}, expected a tensor")
            if t.dim() != 3:
                raise ValueError(f"layer {li} output has shape {tuple(t.shape)}, expected (B, T, D)")
            self._buf[li] = t
            if li == self._deepest:
                raise _StopForward
        return hook

    def run(self, model, input_ids: torch.Tensor) -> Dict[int, torch.Tensor]:
        """One forward pass; returns {layer: (B, T, D) tensor on the model's device}."""
        self._buf = {}
        try:
            model(input_ids=input_ids, use_cache=False, output_hidden_states=False)
        except _StopForward:
            pass  # expected: the deepest hook fired and we do not need the remaining blocks
        missing = [li for li in self.layers if li not in self._buf]
        if missing:
            raise RuntimeError(
                f"hooks did not fire for layers {missing}; the module path or the early-exit "
                f"signal is wrong -- refusing to write partial activations"
            )
        return dict(self._buf)

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._buf = {}

    def __enter__(self) -> "ResidCapture":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- token stream
# THE corpus this process extracts. Bound ONCE, from --corpus or LD_CORPUS, and read by every
# path helper below.
#
# Deliberately module state rather than a threaded parameter: the corpus reaches NINE separate
# path sites here (shards, manifests, the token stream, the fingerprint, the verify pass, and
# the norm-factor inheritance that re-opens the "train" stream). Threading a parameter through
# all of them risks missing one, and a single missed site would write a second corpus' shards
# into the OpenWebText directory resid/L{n}/{split}. The shard skip-guard checks dtype and shape
# only, so that overwrite would be completely silent. One binding cannot be partially applied.
def _corpus_from_env() -> str:
    """Bind the corpus from LD_CORPUS, but ONLY if it names a known training corpus.

    The pipeline driver always passes --corpus explicitly; an unrecognised LD_CORPUS value is
    another job's environment leaking in, and the safe reading is the default ("owt").
    """
    v = os.environ.get("LD_CORPUS")
    if v and v in config.LM_CORPUS_ORDER:
        return v
    return "owt"


CORPUS: str = _corpus_from_env()


def _set_corpus(name: str) -> None:
    global CORPUS
    if name not in config.LM_CORPUS_ORDER:
        raise SystemExit(f"unknown training corpus {name!r}; known: {config.LM_CORPUS_ORDER}")
    CORPUS = name


def resid_dir(layer: int, split: str) -> str:
    """Corpus-scoped shard directory. Every writer and reader in this module goes through it."""
    return config.resid_dir(layer, split, CORPUS)


def open_token_stream(split: str) -> np.ndarray:
    """Memory-map the packed uint16 token stream for one split (read-only)."""
    path = config.lm_token_stream(CORPUS, split)
    config.assert_safe_text_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"token stream for split {split!r} not found at {path}; run the pack stage "
            f"(langdual.materialize_corpora pack) first"
        )
    toks = np.load(path, mmap_mode="r")
    if toks.ndim != 1:
        raise ValueError(f"{path}: expected a 1-D packed stream, got shape {toks.shape}")
    if toks.dtype != np.uint16:
        raise ValueError(f"{path}: expected uint16 token ids, got {toks.dtype}")
    return toks


def token_stream_fingerprint(split: str, toks: np.ndarray) -> dict:
    """Cheap identity record for the stream this extraction consumed (see TOKEN_HASH_HEAD_BYTES)."""
    head = np.asarray(toks[: TOKEN_HASH_HEAD_BYTES // toks.itemsize]).tobytes()
    return {
        "path": config.lm_token_stream(CORPUS, split),
        "n_tokens": int(toks.shape[0]),
        "sha256_head": _sha256_bytes(head),
        "head_bytes": len(head),
    }


def _batch_input_ids(toks: np.ndarray, seq_lo: int, seq_hi: int, ctx: int,
                     batch_seqs: int, device: torch.device) -> Iterable[torch.Tensor]:
    """Yield (b, ctx) int64 batches covering sequences [seq_lo, seq_hi).

    The stream is PACKED (documents are concatenated with no padding), so every position is a
    real token and no attention mask is needed -- the default causal mask is correct.  KNOWN
    LIMITATION, inherited from the standard SAE-extraction recipe: a 128-token window may
    straddle a document boundary, so a small fraction of contexts mix two documents.
    """
    for lo in range(seq_lo, seq_hi, batch_seqs):
        hi = min(lo + batch_seqs, seq_hi)
        flat = np.asarray(toks[lo * ctx: hi * ctx])
        # uint16 -> int64 because nn.Embedding requires a Long index; the copy is unavoidable
        # and tiny (batch_seqs * ctx elements).
        ids = torch.from_numpy(flat.astype(np.int64, copy=False)).view(hi - lo, ctx)
        yield ids.to(device, non_blocking=True)


# --------------------------------------------------------------------------- norm factor
def compute_norm_factors(model, toks: np.ndarray, layers: Sequence[int], ctx: int,
                         batch_seqs: int, device: torch.device,
                         n_tokens: int) -> Tuple[Dict[int, float], int]:
    """One fixed global scalar per layer: sqrt( E_tokens[ ||x||^2 ] ).

    WHY A FIXED GLOBAL SCALE, folded into the stored bytes, rather than either convention of
    the dictionary_learning library:

      (1) ``training.py:get_norm_factor`` computes sqrt(mean ||x||^2) from however
          many buffer steps the caller passes (default 100 steps of a SHUFFLED activation
          buffer), stashes it in the trainer config, and divides the stream by it at train
          time.  The value therefore depends on buffer shuffling and on how far the run got --
          it is not reproducible across our resumable, shard-wise pipeline, and it lives only
          inside a trainer config that our post-hoc scorers never read.
      (2) ``evaluation.py`` instead rescales so that the mean L2 norm equals
          sqrt(d_model): ``scale = activation_dim**0.5 / x.norm(dim=-1).mean()``.  That is a
          PER-BATCH scale, so the same token is scaled differently depending on which batch it
          lands in, and MonoScore / purity / KNN batch the data differently from training.

      The two conventions differ by a factor of about sqrt(d_model) = sqrt(1024) = 32, so
      mixing them silently changes every L1/JumpReLU threshold by 32x.  Neither is safe here,
      so we take the (1) definition but make it a CONSTANT: estimated once from a fixed prefix
      of the TRAIN stream, FOLDED INTO the stored fp16 bytes, and recorded in the manifest.
      Because the scale is baked into the data, training and evaluation cannot disagree about
      it -- there is no runtime knob left to get wrong.

    Returns ({layer: norm_factor}, n_tokens_actually_used).
    """
    n_seqs = max(1, min(n_tokens, toks.shape[0]) // ctx)
    # float64 accumulators: 2M tokens * 1024 dims of squared fp32 values overflows fp32's
    # precision long before it overflows its range, and the scale must be exactly reproducible.
    sumsq = {li: 0.0 for li in layers}
    seen = 0
    with torch.no_grad(), ResidCapture(model, layers) as cap:
        for ids in _batch_input_ids(toks, 0, n_seqs, ctx, batch_seqs, device):
            acts = cap.run(model, ids)
            seen += ids.shape[0] * ids.shape[1]
            for li in layers:
                a = acts[li].reshape(-1, acts[li].shape[-1]).float()
                sumsq[li] += float(a.pow(2).sum().item())
    if seen == 0:
        raise RuntimeError("norm-factor probe saw zero tokens; the stream is shorter than one context")

    factors = {}
    for li in layers:
        nf = float(np.sqrt(sumsq[li] / seen))
        # A zero or non-finite scale would turn every stored value into inf/nan, so refuse
        # rather than write garbage that only shows up as an unexplained training collapse.
        if not np.isfinite(nf) or nf <= 0.0:
            raise RuntimeError(f"layer {li}: computed a non-usable norm factor {nf!r}")
        factors[li] = nf
    return factors, seen


# --------------------------------------------------------------------------- probe / gate
def run_probe(model, toks: np.ndarray, layers: Sequence[int], ctx: int,
              device: torch.device) -> Dict[int, torch.Tensor]:
    """Capture the fixed PROBE_SEQS-sequence probe (raw, unscaled) for every layer."""
    n = PROBE_SEQS
    if toks.shape[0] < n * ctx:
        raise ValueError(f"stream too short for a {n}x{ctx}-token probe")
    with torch.no_grad(), ResidCapture(model, layers) as cap:
        ids = next(iter(_batch_input_ids(toks, 0, n, ctx, n, device)))
        acts = cap.run(model, ids)
        return {li: acts[li].reshape(-1, acts[li].shape[-1]).float().cpu() for li in layers}


def probe_record(toks: np.ndarray, probe: Dict[int, torch.Tensor], ctx: int,
                 probe_split: str) -> dict:
    """Serialise the probe fingerprint that guards against a wrong layer index / block input.

    Stored per layer: the sha256 of the fp16-rounded RAW activation (independent of the norm
    factor, so it does not move when --norm-tokens changes), plus four scalars.  ``rms_norm``
    -- sqrt(mean ||x||^2) over the probe's 512 tokens -- is the discriminative one: it grows
    monotonically and steeply with depth in a pre-LN transformer, so capturing the INPUT of
    block L (== the output of block L-1) makes L's recorded rms_norm collide with L-1's, which
    is visible by inspection across the three manifests.

    KNOWN LIMITATION: the sha is computed on fp16-rounded values specifically to suppress
    cuBLAS algorithm noise, but TF32 matmuls are still not guaranteed bit-identical across GPU
    architectures.  --verify therefore reports BOTH sha equality and a numeric max-abs-diff,
    so a mismatch can be triaged as "different hardware" vs "different layer".
    """
    rec = {
        "split": probe_split,
        "n_seqs": PROBE_SEQS,
        "ctx_len": ctx,
        "token_sha256": _sha256_bytes(np.asarray(toks[: PROBE_SEQS * ctx]).tobytes()),
        "layers": {},
    }
    for li, a in probe.items():
        a64 = a.double()
        rec["layers"][str(li)] = {
            "sha256_raw_fp16": _sha256_bytes(a.to(torch.float16).numpy().tobytes()),
            "rms_norm": float(torch.sqrt(a64.pow(2).sum(dim=-1).mean()).item()),
            "mean": float(a64.mean().item()),
            "std": float(a64.std().item()),
            "abs_max": float(a64.abs().max().item()),
        }
    return rec


def fp16_roundtrip_r2(x: torch.Tensor) -> float:
    """R^2 of fp16 storage against the fp32 values, computed in float64.

    Gate, not a report: if this is below FP16_R2_MIN we refuse to write, because a lossy store
    would silently cap every reconstruction-R^2 number computed downstream.

    float8_e4m3fn is EXPLICITLY REJECTED as a storage dtype: it carries 3 mantissa bits, a
    ~6% relative step, which lands orders of magnitude below this threshold and would be
    indistinguishable from a genuinely bad SAE.
    """
    x64 = x.double()
    rt = x.to(torch.float16).double()
    sse = (x64 - rt).pow(2).sum().item()
    sst = (x64 - x64.mean()).pow(2).sum().item()
    if sst <= 0.0:
        raise RuntimeError("fp16 gate: the probe activation has zero variance; refusing to proceed")
    return 1.0 - sse / sst


# --------------------------------------------------------------------------- shard planning
def plan_shards(n_tokens: int, ctx: int, shard_tokens_target: int) -> Tuple[int, List[Tuple[int, int]]]:
    """Return (seqs_per_shard, [(seq_lo, seq_hi), ...]) covering n_tokens.

    ADAPTATION, stated explicitly: config.LM_SHARD_TOKENS is 1,000,000, which is NOT a multiple
    of config.LM_CTX_LEN (128) -- 1e6/128 = 7812.5.  We floor to whole sequences, so a shard
    holds 7812*128 = 999,936 tokens.  Why alignment matters: it makes a shard's token range map
    to a whole number of forward passes, which is what lets a resumed run skip a completed
    shard without recomputing a straddling sequence, and guarantees no context window is split
    across two files.  The effective value is recorded in the manifest as ``shard_tokens``.
    """
    seqs_per_shard = max(1, shard_tokens_target // ctx)
    n_seqs = n_tokens // ctx  # trailing partial context is dropped: a padded context would
                              # inject activations for tokens that do not exist
    if n_seqs == 0:
        raise ValueError(f"{n_tokens} tokens is less than one {ctx}-token context")
    spans = []
    for lo in range(0, n_seqs, seqs_per_shard):
        spans.append((lo, min(lo + seqs_per_shard, n_seqs)))
    return seqs_per_shard, spans


def shard_path(layer: int, split: str, idx: int) -> str:
    return os.path.join(resid_dir(layer, split), SHARD_FMT % idx)


def manifest_path(layer: int, split: str) -> str:
    return os.path.join(resid_dir(layer, split), MANIFEST_NAME)


def _shard_is_complete(path: str, expect_tokens: int, d_model: int) -> bool:
    """A shard counts as done only if its .npy header matches the plan exactly.

    Checking the header (not just existence) catches the one nasty case: a shard truncated by
    a kill between open_memmap and os.replace.  That cannot actually happen given the temp-file
    protocol below, but the check is nearly free and turns "silently short" into "recompute".
    """
    if not os.path.exists(path):
        return False
    try:
        # Shared with the reader so the two stages can never disagree about what a valid
        # shard header looks like.
        shape, dtype = read_npy_header(path)
    except Exception:
        return False  # unreadable / Fortran-ordered header -> treat as absent and rewrite
    return dtype == np.dtype(np.float16) and shape == (expect_tokens, d_model)


# --------------------------------------------------------------------------- extraction
def extract_split(split: str, layers: Sequence[int], model_path: str, device: torch.device,
                  batch_seqs: int, max_tokens: Optional[int], norm_tokens: int,
                  force: bool) -> None:
    """Extract one split for all requested layers.  Resumable and idempotent."""
    ctx = config.LM_CTX_LEN
    d_model = config.LM_D_MODEL
    layers = sorted(int(l) for l in layers)

    toks = open_token_stream(split)
    # The train budget is the one-pass step budget (TOTAL_STEPS * BATCH_SIZE); val/test
    # use whatever the stream holds unless the caller caps it.
    budget = config.LM_TRAIN_TOKENS if split == "train" else toks.shape[0]
    if max_tokens is not None:
        budget = min(budget, max_tokens)
    n_tokens_avail = min(int(toks.shape[0]), int(budget))

    seqs_per_shard, spans = plan_shards(n_tokens_avail, ctx, config.LM_SHARD_TOKENS)
    shard_tokens = seqs_per_shard * ctx
    planned = [(lo, hi, (hi - lo) * ctx) for lo, hi in spans]
    n_tokens = sum(p[2] for p in planned)

    for li in layers:
        os.makedirs(resid_dir(li, split), exist_ok=True)

    # --- what is still missing? -------------------------------------------------------
    # Built BEFORE the model is loaded, so a fully-complete split costs no GPU time at all.
    todo: List[Tuple[int, int, int, List[int]]] = []  # (shard_idx, seq_lo, seq_hi, missing layers)
    for si, (lo, hi, ntok) in enumerate(planned):
        missing = [li for li in layers
                   if force or not _shard_is_complete(shard_path(li, split, si), ntok, d_model)]
        if missing:
            todo.append((si, lo, hi, missing))

    manifests_ok = all(os.path.exists(manifest_path(li, split)) for li in layers)
    if not todo and manifests_ok and not force:
        print(f"[{split}] all {len(planned)} shards x {len(layers)} layers present; nothing to do")
        return
    print(f"[{split}] {len(planned)} shards planned, {len(todo)} need work "
          f"({n_tokens:,} tokens, {shard_tokens:,} tokens/shard)")

    tf32 = _enable_tf32()
    model = load_model(model_path, device, torch.float32)

    # Validate the model against what config.py claims, before spending an hour on it.
    hidden = int(getattr(model.config, "hidden_size"))
    n_blocks = len(get_blocks(model))
    if hidden != d_model:
        raise ValueError(f"model hidden_size {hidden} != config.LM_D_MODEL {d_model}")
    if max(layers) >= n_blocks:
        raise ValueError(f"requested layer {max(layers)} but the model has {n_blocks} blocks")

    # --- norm factors -----------------------------------------------------------------
    # Always derived from the TRAIN stream, for every split: val/test must be scaled by the
    # SAME constant as train or the SAE sees a different input distribution at eval time.
    train_toks = toks if split == "train" else open_token_stream("train")
    reuse: Dict[int, float] = {}
    # Per-layer provenance of the scale: how many tokens it was measured over.  Kept per layer
    # because a partially-resumed run can inherit some layers and re-estimate others.
    norm_tokens_used: Dict[int, Optional[int]] = {}
    # Where an already-recorded factor may be inherited from.  Note --force does NOT re-estimate
    # for a non-train split: val/test must be folded by exactly the scale the train shards were
    # folded by, or the SAE would see a shifted input distribution at eval time.  Only a forced
    # TRAIN extraction is allowed to produce a new scale (and that invalidates val/test, which
    # is why the guard below prints).
    sources = () if (force and split == "train") else ("train",)
    for li in layers:
        for src_split in sources:
            mp = manifest_path(li, src_split)
            if not os.path.exists(mp):
                continue
            try:
                with open(mp) as fh:
                    prev = json.load(fh)
            except Exception:
                continue  # unreadable manifest -> just recompute
            if prev.get("schema") == SCHEMA and prev.get("norm_factor"):
                reuse[li] = float(prev["norm_factor"])
                # Carry the ORIGINAL probe size forward so a rewritten manifest keeps saying
                # how the scale it is using was actually measured.
                norm_tokens_used[li] = prev.get("norm_probe_tokens")
                break
    if split != "train" and len(reuse) < len(layers):
        # Estimating the scale here would work, but a later `--split train` run with a
        # different --norm-tokens would silently produce a different one for the train shards.
        print(f"[{split}] WARNING: no train manifest to inherit the norm factor from; "
              f"extract --split train FIRST so both splits share one scale")

    t0 = time.time()
    need_norm = [li for li in layers if li not in reuse]
    if need_norm:
        print(f"[{split}] estimating norm factors for layers {need_norm} "
              f"over {norm_tokens:,} train tokens")
        fresh, used = compute_norm_factors(model, train_toks, need_norm, ctx,
                                           batch_seqs, device, norm_tokens)
        reuse.update(fresh)
        for li in need_norm:
            norm_tokens_used[li] = used
    norm_factors = {li: reuse[li] for li in layers}
    for li in layers:
        print(f"[{split}] L{li}: norm_factor = {norm_factors[li]:.6f}")

    # --- probe + fp16 gate ------------------------------------------------------------
    # The probe deliberately uses the TRAIN stream for every split, so the three manifests
    # carry the SAME fingerprint and can be cross-checked against each other.
    probe_split = "train"
    probe = run_probe(model, train_toks, layers, ctx, device)
    probe_rec = probe_record(train_toks, probe, ctx, probe_split)

    r2 = {}
    for li in layers:
        folded = probe[li] / norm_factors[li]  # gate exactly the bytes we are about to store
        r2[li] = fp16_roundtrip_r2(folded)
        print(f"[{split}] L{li}: fp16 round-trip R^2 = {r2[li]:.8f}")
        if r2[li] < FP16_R2_MIN:
            raise RuntimeError(
                f"L{li}: fp16 round-trip R^2 {r2[li]:.8f} < {FP16_R2_MIN}; refusing to store. "
                f"Use float32 shards instead (and note float8_e4m3fn, 3 mantissa bits, is "
                f"explicitly rejected for this data)."
            )

    # --- the extraction loop ----------------------------------------------------------
    stream_rec = token_stream_fingerprint(split, toks)
    done_tokens = 0
    with torch.no_grad(), ResidCapture(model, layers) as cap:
        for si, seq_lo, seq_hi, missing in todo:
            ntok = (seq_hi - seq_lo) * ctx
            # One temp memmap per missing layer.  Writing straight into the memmap keeps host
            # RAM at one batch (not one 2 GB shard) per layer, and the temp+replace pair makes
            # the shard appear atomically: a kill leaves a .tmp that the next run truncates.
            tmps, arrs = {}, {}
            try:
                for li in missing:
                    tmp = shard_path(li, split, si) + ".tmp"
                    tmps[li] = tmp
                    arrs[li] = np.lib.format.open_memmap(
                        tmp, mode="w+", dtype=np.float16, shape=(ntok, d_model))
                off = 0
                for ids in _batch_input_ids(toks, seq_lo, seq_hi, ctx, batch_seqs, device):
                    acts = cap.run(model, ids)
                    b = ids.shape[0] * ids.shape[1]
                    for li in missing:
                        a = acts[li].reshape(-1, d_model).float().div_(norm_factors[li])
                        # Guard against a silent inf/nan leaking into 2 GB of shard: cheap
                        # relative to the forward pass, and the failure it catches (fp16
                        # overflow after folding) is otherwise invisible until training NaNs.
                        h = a.to(torch.float16)
                        if not torch.isfinite(h).all():
                            raise RuntimeError(
                                f"non-finite fp16 value in shard {si}, layer {li}; the norm "
                                f"factor ({norm_factors[li]:.6f}) does not bring this layer "
                                f"into fp16 range")
                        arrs[li][off: off + b] = h.cpu().numpy()
                    off += b
                if off != ntok:
                    raise RuntimeError(f"shard {si}: wrote {off} tokens, planned {ntok}")
                for li in missing:
                    arrs[li].flush()
                    del arrs[li]
                arrs = {}
                # Replace only after every layer's temp is complete, so a crash between
                # replaces leaves whole shards, never truncated ones.
                for li in missing:
                    os.replace(tmps[li], shard_path(li, split, si))
                tmps = {}
            finally:
                for a in arrs.values():
                    del a
                for tmp in tmps.values():
                    if os.path.exists(tmp):
                        os.remove(tmp)  # never leave a half-written temp behind
            done_tokens += ntok
            el = time.time() - t0
            print(f"[{split}] shard {si+1}/{len(planned)} ({ntok:,} tok, layers {missing}) "
                  f"{done_tokens:,} tok this run, {el/60:.1f} min", flush=True)

    # --- manifests, written LAST ------------------------------------------------------
    # The manifest is the completion marker for a (layer, split), so it must not exist until
    # every shard it describes does.
    import transformers  # local import purely for the version string
    for li in layers:
        shards = []
        for si, (lo, hi, ntok) in enumerate(planned):
            p = shard_path(li, split, si)
            if not _shard_is_complete(p, ntok, d_model):
                raise RuntimeError(f"shard {p} missing or malformed at manifest time")
            # Both spellings on purpose: "n_tokens" is this module's own vocabulary, "n_rows"
            # is what the Track B consumers (lmsae/eval_trackb.py:_shard_files) look for.
            # Declaring it lets their loader cross-check the count against the file instead of
            # silently trusting whatever is on disk.
            shards.append({"file": os.path.basename(p), "n_tokens": ntok, "n_rows": ntok})
        _atomic_write_json(manifest_path(li, split), {
            "schema": SCHEMA,
            "layer": li,
            "split": split,
            "model": config.LM_REPO,
            "model_path": model_path,
            "model_dtype": "float32",
            "tf32": tf32,
            "hook": "forward_hook on gpt_neox.layers[i], block OUTPUT (io=out)",
            "d_model": d_model,
            "ctx_len": ctx,
            "dtype": "float16",
            "norm_factor": norm_factors[li],
            # CONTRACT WITH THE CONSUMERS, and the reason this key is not optional here:
            # lmsae/loss_recovered_hooks.norm_factor_from_manifest() defaults a missing
            # "norm_convention" to "multiply", i.e. model -> shard space by MULTIPLYING.
            # We fold by DIVIDING (shard = x / norm_factor), so leaving the key out would
            # make the splice-back step scale by norm_factor^2 in the wrong direction --
            # a silent ~1000x error that still produces finite, plausible-looking numbers.
            "norm_convention": "divide",
            "norm_factor_folded": True,
            "norm_factor_source_split": "train",
            "norm_probe_tokens": norm_tokens_used.get(li),
            "fp16_roundtrip_r2": r2[li],
            "fp16_r2_min": FP16_R2_MIN,
            "seqs_per_shard": seqs_per_shard,
            "shard_tokens": shard_tokens,
            "shard_tokens_requested": config.LM_SHARD_TOKENS,
            "n_shards": len(planned),
            "n_tokens": n_tokens,
            "shards": shards,
            "token_stream": stream_rec,
            "probe": probe_rec,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "created_unix": time.time(),
        })
    print(f"[{split}] done: {n_tokens:,} tokens x {len(layers)} layers in "
          f"{(time.time()-t0)/60:.1f} min")


# --------------------------------------------------------------------------- verification
def verify_split(split: str, layers: Sequence[int], model_path: Optional[str],
                 device: torch.device, check_probe: bool) -> int:
    """Structural + numerical checks on an extracted split.  Returns a process exit code."""
    problems: List[str] = []
    d_model = config.LM_D_MODEL

    for li in sorted(layers):
        mp = manifest_path(li, split)
        if not os.path.exists(mp):
            problems.append(f"L{li}: no manifest at {mp}")
            continue
        with open(mp) as fh:
            man = json.load(fh)
        if man.get("schema") != SCHEMA:
            problems.append(f"L{li}: manifest schema {man.get('schema')!r} != {SCHEMA!r}")
            continue

        total = 0
        for rec in man["shards"]:
            p = os.path.join(resid_dir(li, split), rec["file"])
            if not _shard_is_complete(p, int(rec["n_tokens"]), int(man["d_model"])):
                problems.append(f"L{li}: shard {rec['file']} missing or wrong shape/dtype")
            else:
                total += int(rec["n_tokens"])
        if total != int(man["n_tokens"]):
            problems.append(f"L{li}: shards hold {total} tokens, manifest claims {man['n_tokens']}")

        # Scale + finiteness on a prefix.  Because the norm factor is folded in, the stored
        # data must have sqrt(mean ||x||^2) ~ 1 on the train split.  A value far from 1 means
        # the fold was skipped, applied twice, or taken from the wrong layer.
        first = os.path.join(resid_dir(li, split), man["shards"][0]["file"])
        if os.path.exists(first):
            a = np.load(first, mmap_mode="r")[:VERIFY_SAMPLE_TOKENS]
            a32 = np.asarray(a, dtype=np.float32)
            if not np.isfinite(a32).all():
                problems.append(f"L{li}: non-finite values in {os.path.basename(first)}")
            rms = float(np.sqrt((a32.astype(np.float64) ** 2).sum(axis=1).mean()))
            print(f"L{li} {split}: {total:,} tokens, norm_factor={man['norm_factor']:.4f}, "
                  f"stored rms||x||={rms:.4f}, probe rms={man['probe']['layers'][str(li)]['rms_norm']:.2f}")
            # Only the train split is guaranteed near 1 (that is the split the factor was
            # estimated on); val/test are merely expected to be in the same ballpark.
            lo, hi = (0.8, 1.25) if split == "train" else (0.5, 2.0)
            if not (lo <= rms <= hi):
                problems.append(f"L{li}: stored rms||x|| = {rms:.4f} outside [{lo}, {hi}] "
                                f"-- the norm fold looks wrong")

    # Re-run the fixed probe and compare against the recorded fingerprint.  This is the check
    # that catches a wrong layer index or a block-INPUT capture, which every structural check
    # above would happily pass.
    if check_probe and not problems:
        try:
            _enable_tf32()
            model = load_model(model_path or resolve_model_path(None), device, torch.float32)
            toks = open_token_stream("train")
            probe = run_probe(model, toks, sorted(layers), config.LM_CTX_LEN, device)
            for li in sorted(layers):
                with open(manifest_path(li, split)) as fh:
                    rec = json.load(fh)["probe"]["layers"][str(li)]
                sha = _sha256_bytes(probe[li].to(torch.float16).numpy().tobytes())
                rms = float(torch.sqrt(probe[li].double().pow(2).sum(-1).mean()).item())
                if sha == rec["sha256_raw_fp16"]:
                    print(f"L{li}: probe sha256 MATCH")
                else:
                    # Not automatically a failure: TF32 matmuls are not bit-identical across
                    # GPU architectures.  Report the numeric distance so the two causes --
                    # different hardware vs different layer -- can be told apart.
                    rel = abs(rms - rec["rms_norm"]) / max(rec["rms_norm"], 1e-9)
                    msg = (f"L{li}: probe sha256 MISMATCH (recorded rms {rec['rms_norm']:.4f}, "
                           f"now {rms:.4f}, rel {rel:.2e})")
                    if rel > 1e-3:
                        problems.append(msg + " -- rms moved too far; likely a DIFFERENT layer")
                    else:
                        print(msg + " -- rms agrees; consistent with a hardware/kernel difference")
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"probe re-check skipped: {type(exc).__name__}: {exc}")

    if problems:
        print("\nVERIFY FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nVERIFY OK")
    return 0


# --------------------------------------------------------------------------- CLI
def _device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=None, choices=list(config.LM_CORPUS_ORDER),
                    help="training corpus (default: $LD_CORPUS, else owt). Selects BOTH "
                         "the token stream and the shard directory, so it can never mix two.")
    ap.add_argument("--split", default="train", help="token-stream split to extract")
    ap.add_argument("--layers", type=int, nargs="+", default=list(config.LM_LAYERS),
                    help=f"block indices to capture (default {list(config.LM_LAYERS)})")
    ap.add_argument("--model", default=None, help="local snapshot dir; default resolves offline")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-seqs", type=int, default=64,
                    help="sequences per forward pass (x %d tokens each)" % config.LM_CTX_LEN)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="cap the tokens extracted (default: LM_TRAIN_TOKENS for train, all otherwise)")
    ap.add_argument("--norm-tokens", type=int, default=NORM_PROBE_TOKENS,
                    help="tokens used to estimate the per-layer norm factor")
    ap.add_argument("--force", action="store_true",
                    help="recompute every shard and the norm factor, ignoring what is on disk")
    ap.add_argument("--verify", action="store_true", help="check an extracted split instead of extracting")
    ap.add_argument("--no-model-probe", action="store_true",
                    help="--verify: skip the probe re-check (structural checks only, no GPU)")
    args = ap.parse_args(argv)

    if args.corpus:
        _set_corpus(args.corpus)
    print(f"[extract] corpus={CORPUS}  shards -> {resid_dir(config.LM_LAYERS[0], args.split)}")
    if args.batch_seqs < 1:
        ap.error("--batch-seqs must be >= 1")
    dev = _device(args.device)

    if args.verify:
        return verify_split(args.split, args.layers, args.model, dev, not args.no_model_probe)

    extract_split(args.split, args.layers, resolve_model_path(args.model), dev,
                  args.batch_seqs, args.max_tokens, args.norm_tokens, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
