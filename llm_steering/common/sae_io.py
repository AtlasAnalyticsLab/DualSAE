"""Loading SAE arms and encoding rows for the steering stages (D-04, D-10, D-11, D-21, D-44).

load_arm(run_dir, device)
    ``common/sae_load.load_run(run_dir, device, checkpoint='final')`` (D-10: every arm is scored
    at autoencoder_final.pt, step 31,250; strict load; non-TopK types borrow the sibling best_val
    model_cfg, TopK is rebuilt from training_config.json).  Adds the TRUE best-val epoch read
    from best_val_monoscore_model.pt['epoch'] -- load_run's own meta['best_epoch'] is the epoch
    of the LOADED checkpoint, i.e. always 50 at final.  The load is made
    RNG-neutral (BaseAutoencoder.__init__ calls torch.manual_seed; D-11).

encode_rows(model, meta, X, device, perm_seed)       D-21, batch-safe codes as a CSR matrix
    * TopK and JumpReLU act PER ROW, so rows are encoded in chunks in their natural order.
      Mathematically a row's code cannot depend on its neighbours, but numerically it can:
      cuBLAS picks a different fp32 GEMM kernel for a different number of rows, which moves the
      last bits (measured: up to 2e-5 absolute on the normalize=True TopK baseline).  So every
      chunk is PADDED to exactly ``chunk_rows`` rows (wrap-around over the chunk's own rows,
      padded outputs discarded): with one GEMM shape a row's code is BIT-IDENTICAL whichever
      other rows share its chunk, so encoding a subset of rows reproduces the full-set codes
      exactly.
    * BatchTopK (baseline model_type 'batch_topk', or a Dual with sparsity_mode 'batch_topk')
      selects the top k*B activations of a WHOLE batch, so a row's code depends on the batch it
      sits in.  Every BatchTopK row is therefore encoded inside a batch of EXACTLY 2048 rows (the
      training batch size, so the operator sees the batch statistics it was trained under), with
      batch membership fixed by a seeded permutation of the rows, and the codes are un-permuted.
      The permutation depends only on (perm_seed, n_rows) -- never on the arm -- so every arm of
      a comparison sees the same batches (D-21: "both arms identical").
      LAST-BATCH RULE: with n = q*2048 + r, r > 0, the final batch holds the r remaining rows plus
      the first 2048 - r rows of the permutation again (wrap-around).  Those padding rows already
      have their codes from batch 0; their second outputs are DISCARDED.  Because the permutation
      is random, the padding is a random sample of the same rows, so the last batch is
      statistically like every other batch.  With n < 2048 the permutation is tiled until the
      single batch is full and each row keeps its FIRST occurrence (the batch then contains
      repeated rows -- unavoidable, and flagged in encode_policy()).
    Why not plain sequential batching: on a class-sorted row set, sequential batches make each
    BatchTopK batch (nearly) single-class.

estimate_batchtopk_threshold / encode_thresholded       D-44 (Track B steering)
    The declared adaptation of dictionary_learning's BatchTopK inference threshold
    (dictionary_learning, trainers/batch_top_k.py, threshold update): EMA (beta 0.999,
    initialised with the first batch's value) of the per-batch MINIMUM POSITIVE value of the
    model's own batch-top-k code, run post hoc over >= 5,000 train-shard batches of 2048 NON-SINK
    rows (position % ctx_len != 0).  Inference then encodes each token independently as
    acts * (acts > threshold) with acts = ReLU((x_pre - b_dec) @ W_enc), the pre-activation both
    BatchTopK families actually use (vision/models/batch_topk_sae.py and dual_sae.py).
d44_l0_columns                                           D-44 report columns
    The two L0s every steering row prints side by side on the SAME non-sink test rows: the inference
    L0 under the frozen threshold, and the TRAINING-TIME non-sink L0 -- the model's own batch
    top-k over shuffled ALL-ROW 2048 batches (sinks included at their natural 1/128 rate, as the
    trainer saw them), averaged over the non-sink rows only.

All encoding runs in fp32 with TF32 matmuls disabled (the trainers never enable TF32).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from common import lang_config as config
from common import seeds as _seeds

# The shared SAE core (<repo>/vision) imports its modules by bare name (``from models import
# ...``), so its directory must be on sys.path; common/sae_load.py inserts it (relative to this
# file).  Known shadowing risk: it has top-level modules named `models`, `loss`, `utils`,
# `dataset`; code that also needs another project's `utils` must import it by package path.
def _ensure_sae_path() -> None:
    from common import sae_load  # noqa: F401  (inserts <repo>/vision into sys.path)


BATCHTOPK_BATCH = int(config.BATCH_SIZE)      # 2048: the training batch size of every cell
# Per-row families: the FIXED GEMM row count (every chunk is padded to it, see the module
# docstring).  Changing it changes codes in the last bits, so it is part of the encode policy.
DEFAULT_CHUNK = 4096
D44_BETA = 0.999                              # dictionary_learning threshold_beta default
D44_MIN_BATCHES = 5000                        # D-44: first-batch init keeps < 0.7% weight
FINAL_CKPT = "autoencoder_final.pt"
BEST_CKPT = "best_val_monoscore_model.pt"


# =========================================================================== loading
def arm_family(meta: Dict[str, Any]) -> str:
    """'topk' | 'batch_topk' | 'jump_relu' for the six Track A/B model types.

    Baselines are identified by model_type; Duals by their mono-branch sparsity_mode.  Anything
    else (gated, matryoshka, msae, vanilla ...) is not an arm of these apps and is rejected so
    it cannot be encoded under the wrong batching policy."""
    mt = meta.get("model_type")
    if mt == "dual":
        mode = meta.get("sparsity_mode")
        fam = {"topk": "topk", "batch_topk": "batch_topk", "jump_relu": "jump_relu"}.get(mode)
    else:
        fam = {"topk": "topk", "batch_topk": "batch_topk", "jumprelu": "jump_relu"}.get(mt)
    if fam is None:
        raise ValueError(f"unsupported arm: model_type={mt!r} sparsity_mode="
                         f"{meta.get('sparsity_mode')!r}")
    return fam


def is_batchtopk(meta: Dict[str, Any]) -> bool:
    return arm_family(meta) == "batch_topk"


def _read_ckpt_field(path: Path, key: str) -> Any:
    """One scalar from a (large) checkpoint without materialising its tensors: torch.load with
    mmap=True maps the zip archive and only unpickles the small objects touched."""
    obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return obj.get(key) if isinstance(obj, dict) else None


def load_arm(run_dir: Union[str, Path], device: Union[str, torch.device], *,
             reseed: Optional[int] = None) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load one arm at its FINAL checkpoint.  Returns (model.eval(), meta).

    meta additions over load_run's:
      family          'topk' | 'batch_topk' | 'jump_relu'
      best_val_epoch  true best-val epoch (best_val_monoscore_model.pt['epoch'])
      best_epoch      OVERWRITTEN with best_val_epoch, so a D-04 row that prints
                      meta['best_epoch'] is right (load_run put the loaded epoch there)
      loaded_epoch    epoch of autoencoder_final.pt (load_run's original best_epoch)
      final_step      optimizer step of autoencoder_final.pt (31,250 for every cell)
      checkpoint_name 'autoencoder_final.pt'

    RNG: the load runs inside torch.random.fork_rng over the CPU generator AND every visible
    CUDA device's generator (BaseAutoencoder.__init__'s torch.manual_seed reseeds all of them,
    not only the target device), so it leaves torch's global RNG exactly as it found it;
    ``reseed`` (optional) then seeds python/numpy/torch explicitly (D-11).  Apps that load
    several arms should load all of them and then call seeds.seed_everything once.
    Cost note: forking a device's generator initialises the CUDA runtime but creates no context
    on the non-target devices (measured: one 612 MiB context, on the target only, with two GPUs
    visible).  With CUDA available but a CPU target the runtime is initialised too; that is what
    makes torch.manual_seed's CUDA reseed run INSIDE the fork (uninitialised, torch would queue
    it and apply it silently at the next CUDA init, outside any fork).
    """
    from common.sae_load import load_run

    run_dir = Path(run_dir)
    device = torch.device(device)
    fork_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=fork_devices):
        model, meta = load_run(run_dir, device, checkpoint="final")
    model.eval()
    meta = dict(meta)
    meta["family"] = arm_family(meta)
    meta["loaded_epoch"] = meta.get("best_epoch")
    meta["final_step"] = _read_ckpt_field(run_dir / FINAL_CKPT, "step")
    meta["best_val_epoch"] = _read_ckpt_field(run_dir / BEST_CKPT, "epoch")
    meta["best_epoch"] = meta["best_val_epoch"]
    meta["checkpoint_name"] = FINAL_CKPT
    if meta["final_step"] is not None and int(meta["final_step"]) != int(config.TOTAL_STEPS):
        # Not fatal (a reader may want to inspect a short run), but loud: D-10 expects 31,250.
        print(f"[sae_io] WARNING {run_dir.name}: final_step={meta['final_step']} != "
              f"{config.TOTAL_STEPS}", file=sys.stderr)
    if meta["model_type"] == "dual":
        # The trainer's config and the constructed module must agree on the sparsifier; a
        # mismatch would make encode_rows pick the wrong batching policy.
        if getattr(model, "sparsity_mode", None) != meta.get("sparsity_mode"):
            raise RuntimeError(f"{run_dir}: model.sparsity_mode={getattr(model, 'sparsity_mode', None)!r}"
                               f" != training_config sparsity_mode={meta.get('sparsity_mode')!r}")
    if reseed is not None:
        _seeds.seed_everything(int(reseed))
    return model, meta


# =========================================================================== input rows
def normalize_like_lmdb(X: np.ndarray) -> np.ndarray:
    """Track A input normalisation, BIT-identical to LMDBFeatureDataset(normalize_rows=True)
    (vision/dataset/lmdb_features.py): per 1-D float32 row, ``norm = np.linalg.norm(x);
    x = x / (norm + 1e-8)`` when norm > 0, else unchanged.  The trainers used this default; the
    models' own input_unit_norm happens inside forward, so callers must NOT standardise rows
    themselves.

    Why a per-row loop and not one vectorised expression: the loader's 1-D norm is a BLAS dot
    (a different summation order than ``np.linalg.norm(X, axis=1)``'s reduction), and its
    ``norm + 1e-8`` adds a Python float to a float32 scalar (computed in float64 under numpy
    1.26's value-based casting, then used as a float32 divisor).  A vectorised version differed
    from the loader in a fraction of rows (by ~3e-8) -- negligible numerically,
    but enough to flip a borderline BatchTopK selection and to break the bit-exact claims of
    encode_rows.  The loop costs ~5 us per row (0.35 s for 70k rows).
    """
    X = np.asarray(X, dtype=np.float32)
    out = np.array(X, dtype=np.float32, copy=True)
    for r in range(out.shape[0]):
        x = out[r]
        norm = np.linalg.norm(x)          # the loader's exact expression, row by row
        if norm > 0:
            out[r] = x / (norm + 1e-8)
    return out


def read_lmdb_rows(encoder: str, corpus: str, split: str,
                   indices: Optional[Sequence[int]] = None, *, normalize: bool = True,
                   return_paths: bool = False):
    """Rows of a Track A LMDB (float32), in the order of ``indices`` (default: all rows in key
    order).  ``normalize=True`` applies the training-time row normalisation.  With
    ``return_paths`` the ``__path__<i>`` ids ('<corpus>/<split>/<i>') are returned too."""
    import lmdb
    path = config.lmdb_path(encoder, corpus, split)
    config.assert_safe_text_path(path)
    env = lmdb.open(path, readonly=True, lock=False, readahead=True, meminit=False)
    try:
        with env.begin() as txn:
            n = int(txn.get(b"__len__").decode("ascii"))
            shape = tuple(int(s) for s in txn.get(b"__shape__").decode("ascii").strip("()").split(",") if s.strip())
            d = shape[-1]
            idx = np.arange(n) if indices is None else np.asarray(indices, dtype=np.int64)
            if idx.size and (idx.min() < 0 or idx.max() >= n):
                raise IndexError(f"indices outside [0, {n}) for {path}")
            X = np.empty((idx.size, d), dtype=np.float32)
            paths: List[str] = []
            for r, i in enumerate(idx):
                buf = txn.get(str(int(i)).encode("ascii"))
                if buf is None:
                    raise KeyError(f"missing row {i} in {path}")
                X[r] = np.frombuffer(buf, dtype=np.float32, count=d)
                if return_paths:
                    pb = txn.get(f"__path__{int(i)}".encode("ascii"))
                    paths.append(pb.decode("utf-8") if pb is not None else str(int(i)))
    finally:
        env.close()
    if normalize:
        X = normalize_like_lmdb(X)
    return (X, paths) if return_paths else X


def _shard_files(shard_dir: str) -> Tuple[Dict[str, Any], List[Tuple[str, int]]]:
    with open(os.path.join(shard_dir, "manifest.json")) as fh:
        man = json.load(fh)
    files = [(os.path.join(shard_dir, s["file"]), int(s["n_rows"])) for s in man["shards"]]
    return man, files


def nonsink_mask(n_rows: int, ctx_len: int = config.LM_CTX_LEN, start: int = 0) -> np.ndarray:
    """True for rows whose position in their packed 128-token window is >= 1.  Track B windows
    are PACKED with no BOS, so position 0 of every window is the attention sink (||x|| ~ 8 vs a
    non-sink median of 0.16-0.33).  ``start`` is the global row index of the first row."""
    return ((np.arange(start, start + n_rows) % ctx_len) != 0)


def iter_nonsink_batches(shard_dir: str, n_batches: int, seed: int,
                         batch_rows: int = BATCHTOPK_BATCH) -> Iterator[np.ndarray]:
    """Yield ``n_batches`` float32 batches of exactly ``batch_rows`` NON-SINK rows from a Track B
    shard directory (D-44), deterministically from ``seed``.

    Sampling (declared approximation of the trainer's shuffled loader): shards are visited in a
    seeded order; within a shard its non-sink rows are shuffled with a seeded permutation and cut
    into consecutive batches (a shard-sized shuffle buffer).  Each shard holds ~984k non-sink
    rows (~480 batches), so 5,000 batches touch ~11 of the 65 train shards.  A shard is read
    sequentially into RAM when most of it is needed (2 GB fp16), otherwise only the needed rows
    are gathered (sorted, for IO locality) -- same rows either way.
    """
    man, files = _shard_files(shard_dir)
    ctx = int(man.get("ctx_len", config.LM_CTX_LEN))
    g = _seeds.rng(seed)
    order = g.permutation(len(files))
    produced = 0
    for si in order:
        if produced >= n_batches:
            return
        path, n_rows = files[si]
        if n_rows % ctx != 0:
            # Shards are written in whole windows; anything else would misplace every sink.
            raise RuntimeError(f"{path}: n_rows {n_rows} is not a multiple of ctx_len {ctx}")
        ns = np.flatnonzero(nonsink_mask(n_rows, ctx))
        # Per-shard permutation from a per-shard derived seed, so the batches a shard yields do
        # not depend on how many batches earlier shards produced.
        perm = ns[_seeds.rng(_seeds.derive_seed_parts(seed, "shard", int(si))).permutation(ns.size)]
        n_here = min(n_batches - produced, perm.size // batch_rows)
        if n_here <= 0:
            continue
        mm = np.load(path, mmap_mode="r")
        need = perm[: n_here * batch_rows]
        if need.size > 0.1 * n_rows:
            block = np.asarray(mm, dtype=np.float16)          # one sequential read
            get = lambda rows: block[rows]                    # noqa: E731
        else:
            block = None
            get = lambda rows: np.asarray(mm[np.sort(rows)])[np.argsort(np.argsort(rows))]  # noqa: E731
        for b in range(n_here):
            rows = need[b * batch_rows:(b + 1) * batch_rows]
            yield get(rows).astype(np.float32)
            produced += 1
        del mm, block


def read_shard_rows(layer: int, split: str, start: int, stop: int, *, corpus: str = "owt",
                    drop_sinks: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Contiguous rows [start, stop) of a Track B split as float32 (shard space: the global norm
    factor is folded into the bytes, no per-row normalisation), plus their global row indices.
    ``drop_sinks`` removes position-0 rows."""
    man, files = _shard_files(config.resid_dir(layer, split, corpus))
    out, idx, off = [], [], 0
    for path, n_rows in files:
        lo, hi = max(start, off), min(stop, off + n_rows)
        if lo < hi:
            mm = np.load(path, mmap_mode="r")
            out.append(np.asarray(mm[lo - off:hi - off], dtype=np.float32))
            idx.append(np.arange(lo, hi))
        off += n_rows
    X = np.concatenate(out) if out else np.zeros((0, int(man["d_model"])), np.float32)
    ids = np.concatenate(idx) if idx else np.zeros(0, np.int64)
    if drop_sinks:
        keep = (ids % int(man.get("ctx_len", config.LM_CTX_LEN))) != 0
        X, ids = X[keep], ids[keep]
    return X, ids


# =========================================================================== batching core
@contextlib.contextmanager
def _fp32_exact() -> Iterator[None]:
    """Full-precision fp32 matmuls for the duration (restores the caller's setting)."""
    prev = torch.get_float32_matmul_precision()
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def batch_plan(n_rows: int, meta: Dict[str, Any], perm_seed: Optional[int] = None, *,
               perm: Optional[np.ndarray] = None,
               chunk_rows: int = DEFAULT_CHUNK) -> List[Tuple[np.ndarray, int]]:
    """The exact list of (row indices fed to the model, number of leading rows whose outputs are
    KEPT) that encode_rows / forward_reps_rows use.  Exposed so apps can record or audit it.

    BatchTopK: every entry has exactly BATCHTOPK_BATCH indices (see the module docstring for the
    wrap-around rule).  Per-row families: natural order, chunks of ``chunk_rows`` real rows, each
    padded by wrap-around to exactly ``chunk_rows`` indices (one GEMM shape -> bit-exact codes).
    ``perm`` overrides the seeded permutation (diagnostics only, e.g. a sequential order to
    demonstrate that batch composition matters).
    """
    if n_rows <= 0:
        return []
    if not is_batchtopk(meta):
        return _per_row_plan(n_rows, chunk_rows)
    if perm is None:
        if perm_seed is None:
            raise ValueError("BatchTopK encoding needs perm_seed (derive it with "
                             "seeds.derive_seed from the DATASET key, never from the arm)")
        perm = _seeds.rng(perm_seed).permutation(n_rows)
    perm = np.asarray(perm, dtype=np.int64)
    if perm.shape != (n_rows,) or not np.array_equal(np.sort(perm), np.arange(n_rows)):
        raise ValueError("perm must be a permutation of range(n_rows)")
    B = BATCHTOPK_BATCH
    n_batches = -(-n_rows // B)
    order = np.resize(perm, n_batches * B)      # cyclic wrap-around fill (the padding rule)
    plan = []
    for b in range(n_batches):
        idx = order[b * B:(b + 1) * B]
        keep = int(min(B, max(0, n_rows - b * B)))   # positions < n_rows are first occurrences
        plan.append((idx, keep))
    return plan


def _per_row_plan(n_rows: int, chunk_rows: int) -> List[Tuple[np.ndarray, int]]:
    """Natural-order chunks, each padded to exactly ``chunk_rows`` indices by repeating the
    chunk's own rows (np.resize wrap-around); only the first ``keep`` outputs are used."""
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
    plan = []
    for s in range(0, n_rows, chunk_rows):
        real = np.arange(s, min(s + chunk_rows, n_rows))
        plan.append((np.resize(real, chunk_rows), int(real.size)))
    return plan


def encode_policy(meta: Dict[str, Any], n_rows: int, perm_seed: Optional[int],
                  chunk_rows: int = DEFAULT_CHUNK) -> Dict[str, Any]:
    """Provenance record of how encode_rows batched this arm (write it next to every code file)."""
    bt = is_batchtopk(meta)
    return {
        "family": arm_family(meta),
        "policy": "batch_topk_2048_seeded_permutation" if bt else "per_row",
        "batch_rows": BATCHTOPK_BATCH if bt else None,
        "perm_seed": int(perm_seed) if (bt and perm_seed is not None) else None,
        "perm_rng": "numpy PCG64 Generator.permutation" if bt else None,
        "last_batch_rule": "wrap-around padding with the permutation's first rows; padded "
                           "outputs discarded" if bt else None,
        "n_rows": int(n_rows),
        "n_padded": int((-n_rows) % BATCHTOPK_BATCH) if bt else 0,
        "repeated_rows_in_single_batch": bool(bt and n_rows < BATCHTOPK_BATCH),
        "per_row_fixed_chunk": None if bt else int(chunk_rows),
        "dtype": "fp32, TF32 off",
        "numpy": np.__version__, "torch": torch.__version__,
    }


def _run_plan(model: torch.nn.Module, X: Union[np.ndarray, torch.Tensor], device: torch.device,
              plan: List[Tuple[np.ndarray, int]],
              fn: Callable[[torch.Tensor], Dict[str, torch.Tensor]]) -> Iterator[Tuple[np.ndarray, Dict[str, torch.Tensor]]]:
    """Run ``fn`` on each planned batch; yield (original row indices of the KEPT rows, outputs
    restricted to those rows)."""
    Xt = X if isinstance(X, torch.Tensor) else None
    with torch.no_grad(), _fp32_exact():
        for idx, keep in plan:
            if Xt is not None:
                xb = Xt[torch.as_tensor(idx, dtype=torch.long)]
            else:
                xb = torch.from_numpy(np.ascontiguousarray(X[idx]))
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
            out = fn(xb)
            yield idx[:keep], {k: v[:keep] for k, v in out.items()}


def _to_csr(n_rows: int, n_cols: int,
            parts: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> sp.csr_matrix:
    if parts:
        r = np.concatenate([p[0] for p in parts])
        c = np.concatenate([p[1] for p in parts])
        v = np.concatenate([p[2] for p in parts]).astype(np.float32)
    else:
        r = c = np.zeros(0, np.int64)
        v = np.zeros(0, np.float32)
    z = sp.csr_matrix((v, (r, c)), shape=(n_rows, n_cols), dtype=np.float32)
    z.sort_indices()
    return z


def _dense_to_parts(rows: np.ndarray, z: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nz = torch.nonzero(z, as_tuple=True)
    vals = z[nz].float().cpu().numpy()
    return rows[nz[0].cpu().numpy()], nz[1].cpu().numpy(), vals


def encode_rows(model: torch.nn.Module, meta: Dict[str, Any], X: Union[np.ndarray, torch.Tensor],
                device: Union[str, torch.device], perm_seed: Optional[int] = None, *,
                perm: Optional[np.ndarray] = None,
                chunk_rows: int = DEFAULT_CHUNK) -> Tuple[sp.csr_matrix, np.ndarray]:
    """D-21 codes of every row of ``X`` (model-input space: Track A rows normalised with
    normalize_like_lmdb, Track B shard rows as stored).  Returns (z as CSR float32 of shape
    (n, dict_size) in the ORIGINAL row order, per-row L0 as int64).

    ``perm_seed`` is required for BatchTopK arms and ignored otherwise; derive it from the
    DATASET (e.g. seeds.derive_seed('sae_io.encode', 'news20|val+test')) so all arms share it.
    """
    device = torch.device(device)
    n = int(X.shape[0])
    plan = batch_plan(n, meta, perm_seed, perm=perm, chunk_rows=chunk_rows)
    parts = []

    def fn(xb: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"z": model(xb)[1]}

    for rows, out in _run_plan(model, X, device, plan, fn):
        parts.append(_dense_to_parts(rows, out["z"]))
    z = _to_csr(n, int(meta["dict_size"]), parts)
    return z, np.diff(z.indptr).astype(np.int64)


def forward_reps_rows(model: torch.nn.Module, meta: Dict[str, Any],
                      X: Union[np.ndarray, torch.Tensor], device: Union[str, torch.device],
                      perm_seed: Optional[int] = None, *, perm: Optional[np.ndarray] = None,
                      chunk_rows: int = DEFAULT_CHUNK,
                      keys: Sequence[str] = ("xM", "r_hat", "xhat")) -> Dict[str, Any]:
    """sae_load.forward_reps under the SAME batching as encode_rows (so a Dual-BatchTopK
    xM matches its codes).  Returns dense float32 arrays in the original row order, in ORIGINAL
    space (xM via postprocess_output, r_hat scale-only), for each requested key, plus 'z' (CSR)
    and 'l0'.  'r_hat' exists for Duals only and is skipped for baselines (never zero-filled)."""
    _ensure_sae_path()
    from common.sae_load import forward_reps
    device = torch.device(device)
    n = int(X.shape[0])
    plan = batch_plan(n, meta, perm_seed, perm=perm, chunk_rows=chunk_rows)
    want = [k for k in keys if k != "r_hat" or meta["is_dual"]]
    D = int(meta["act_size"])
    dense = {k: np.empty((n, D), dtype=np.float32) for k in want}
    parts = []

    def fn(xb: torch.Tensor) -> Dict[str, torch.Tensor]:
        reps = forward_reps(model, xb, meta["is_dual"])
        return {k: reps[k] for k in ["z"] + want}

    for rows, out in _run_plan(model, X, device, plan, fn):
        parts.append(_dense_to_parts(rows, out["z"]))
        for k in want:
            dense[k][rows] = out[k].float().cpu().numpy()
    z = _to_csr(n, int(meta["dict_size"]), parts)
    dense["z"] = z
    dense["l0"] = np.diff(z.indptr).astype(np.int64)
    return dense


# =========================================================================== D-44 threshold
def batchtopk_preacts(model: torch.nn.Module, meta: Dict[str, Any], x: torch.Tensor) -> torch.Tensor:
    """ReLU((x_pre - b_dec) @ W_enc): the pre-activation BOTH BatchTopK families select from
    (BatchTopKSAE.forward: acts = relu(x_cent @ W_enc); DualSAE.forward: acts =
    relu((x_pre - b_dec) @ W_enc), no b_enc).  x_pre = model.preprocess_input(x), the identity
    for every Track B arm (normalize=False) and for the Track A BatchTopK arms."""
    if not is_batchtopk(meta):
        raise ValueError(f"batchtopk_preacts on a {arm_family(meta)} arm")
    x_pre, _, _ = model.preprocess_input(x)
    return F.relu((x_pre - model.b_dec) @ model.W_enc)


def estimate_batchtopk_threshold(model: torch.nn.Module, meta: Dict[str, Any],
                                 row_iter: Iterator[np.ndarray], n_batches: int,
                                 device: Union[str, torch.device], *, beta: float = D44_BETA,
                                 allow_short: bool = False) -> Dict[str, Any]:
    """D-44 frozen inference threshold for one BatchTopK arm.

    For each of ``n_batches`` batches of exactly 2048 NON-SINK rows from ``row_iter`` (use
    iter_nonsink_batches on the TRAIN shard): run the model's OWN batch-top-k forward (so the
    code is exactly what training produced), take the minimum POSITIVE value of the code (0.0 if
    none, as upstream), and update thr = beta*thr + (1-beta)*min, initialised with the first
    batch's value (upstream's `threshold < 0` sentinel).  Upstream arithmetic is float32 on a
    buffer; here it is float64 (difference ~1e-7 relative, irrelevant at the 0.1% EMA step).

    D-44 requires n_batches >= 5000 (the first batch then keeps beta^4999 < 0.7% weight);
    ``allow_short=True`` is for smoke tests only and is recorded in the output.

    The returned ``nonsink_only_batch_l0_mean`` is the L0 of these NON-SINK-ONLY batches, which
    is k (64) by construction for a batch top-k (k * 2048 slots, every row non-sink).  It is a
    sanity value, NOT the D-44 "training-time non-sink L0" column: that one needs ALL-ROW
    batches (sinks included, as in training) and comes from d44_l0_columns.
    """
    if not is_batchtopk(meta):
        raise ValueError(f"estimate_batchtopk_threshold on a {arm_family(meta)} arm")
    if n_batches < D44_MIN_BATCHES and not allow_short:
        raise ValueError(f"D-44 needs n_batches >= {D44_MIN_BATCHES}, got {n_batches} "
                         f"(allow_short=True only for smoke tests)")
    device = torch.device(device)
    thr: Optional[float] = None
    mins: List[float] = []
    l0s: List[float] = []
    t0 = time.time()
    with torch.no_grad(), _fp32_exact():
        for b, xb in enumerate(row_iter):
            if b >= n_batches:
                break
            if xb.shape[0] != BATCHTOPK_BATCH:
                raise ValueError(f"batch {b} has {xb.shape[0]} rows, need {BATCHTOPK_BATCH}")
            x = torch.as_tensor(xb, dtype=torch.float32).to(device)
            z = model(x)[1]
            pos = z[z > 0]
            m = float(pos.min().item()) if pos.numel() else 0.0
            thr = m if thr is None else beta * thr + (1.0 - beta) * m
            mins.append(m)
            l0s.append(float((z > 0).sum().item()) / BATCHTOPK_BATCH)
    if len(mins) < n_batches:
        raise RuntimeError(f"row_iter ended after {len(mins)} of {n_batches} batches")
    mins_a = np.asarray(mins)
    return {
        "threshold": float(thr),
        "n_batches": len(mins),
        "beta": beta,
        "first_batch_weight": float(beta ** (len(mins) - 1)),
        "batch_min_first": float(mins_a[0]),
        "batch_min_mean": float(mins_a.mean()),
        "batch_min_sd": float(mins_a.std(ddof=1)) if mins_a.size > 1 else float("nan"),
        # k by construction (see the docstring); NOT the D-44 training-time column.
        "nonsink_only_batch_l0_mean": float(np.mean(l0s)),
        "short_run": bool(n_batches < D44_MIN_BATCHES),
        "seconds": round(time.time() - t0, 2),
    }


def _fixed_shape_rows(x: torch.Tensor, fixed_rows: Optional[int],
                     fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Apply a per-row ``fn`` to 2-D ``x`` in chunks of exactly ``fixed_rows`` rows (the last one
    padded by wrap-around), so each row's output is bit-identical whatever else is in ``x``.
    ``fixed_rows=None`` calls ``fn`` once on ``x`` as-is (fastest; equal to fp32 rounding)."""
    if fixed_rows is None:
        return fn(x)
    n = x.shape[0]
    outs = []
    for s in range(0, n, fixed_rows):
        real = x[s:s + fixed_rows]
        k = real.shape[0]
        if k < fixed_rows:
            idx = torch.arange(fixed_rows, device=x.device) % k
            real = real[idx]
        outs.append(fn(real)[:k])
    return torch.cat(outs, 0) if outs else fn(x)


def encode_thresholded(model: torch.nn.Module, meta: Dict[str, Any], x: torch.Tensor,
                       threshold: float, *,
                       fixed_rows: Optional[int] = DEFAULT_CHUNK) -> torch.Tensor:
    """D-44 per-token codes: acts * (acts > threshold), acts = batchtopk_preacts
    (upstream's encode(use_threshold=True), batch_top_k.py:37-40).  ``x``: torch tensor on the
    model's device, shape (N, D) or (..., D) (flattened and restored).

    Every token is independent, and with the default ``fixed_rows`` the GEMM always has the same
    row count, so a token's code is bit-identical regardless of the other tokens in the call.
    ``fixed_rows=None`` skips the padding (e.g. one decode step in a generation hook); results
    then match to fp32 rounding, so only activations within ~1e-6 relative of the threshold can
    flip."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])

    def fn(xx: torch.Tensor) -> torch.Tensor:
        acts = batchtopk_preacts(model, meta, xx)
        return acts * (acts > float(threshold))

    return _fixed_shape_rows(x2, fixed_rows, fn).reshape(*shape[:-1], -1)


def forward_thresholded(model: torch.nn.Module, meta: Dict[str, Any], x: torch.Tensor,
                        threshold: float, *,
                        fixed_rows: Optional[int] = DEFAULT_CHUNK) -> Dict[str, torch.Tensor]:
    """Reconstruction under the D-44 code, mirroring each model's forward with ONLY the sparsifier
    replaced.  Returns z, xhat and, for Duals, xM (original space) and r_hat (scale-only
    inversion, as sae_load.forward_reps), so that xM + r_hat == xhat."""
    x_pre, x_mean, x_std = model.preprocess_input(x)
    z = encode_thresholded(model, meta, x, threshold, fixed_rows=fixed_rows)
    xM_p = z @ model.W_dec + model.b_dec
    if not meta["is_dual"]:
        return {"z": z, "xhat": model.postprocess_output(xM_p, x_mean, x_std)}
    r_hat_p = model.poly(x_pre - xM_p)
    return {"z": z,
            "xM": model.postprocess_output(xM_p, x_mean, x_std),
            "r_hat": r_hat_p if x_std is None else r_hat_p * x_std,
            "xhat": model.postprocess_output(xM_p + r_hat_p, x_mean, x_std)}


def encode_thresholded_rows(model: torch.nn.Module, meta: Dict[str, Any],
                            X: Union[np.ndarray, torch.Tensor], device: Union[str, torch.device],
                            threshold: float, *,
                            chunk_rows: int = DEFAULT_CHUNK) -> Tuple[sp.csr_matrix, np.ndarray]:
    """encode_thresholded over many rows (per-row, so plain chunking is exact); CSR + per-row L0."""
    device = torch.device(device)
    n = int(X.shape[0])
    plan = _per_row_plan(n, chunk_rows)          # fixed GEMM shape, as encode_rows
    parts = []
    for rows, out in _run_plan(model, X, device, plan,
                               lambda xb: {"z": encode_thresholded(model, meta, xb, threshold,
                                                                   fixed_rows=None)}):
        parts.append(_dense_to_parts(rows, out["z"]))
    z = _to_csr(n, int(meta["dict_size"]), parts)
    return z, np.diff(z.indptr).astype(np.int64)


D44_EVAL_WINDOWS = 400        # default test windows for d44_l0_columns (51,200 rows, 50,800 non-sink)


def d44_l0_columns(model: torch.nn.Module, meta: Dict[str, Any], threshold: float, layer: int,
                   device: Union[str, torch.device], perm_seed: int, *, split: str = "test",
                   start_window: int = 0, n_windows: int = D44_EVAL_WINDOWS,
                   corpus: str = "owt") -> Dict[str, Any]:
    """The two D-44 L0 columns of one BatchTopK arm, measured on the SAME non-sink rows.

      inference_nonsink_l0      mean per-token L0 of encode_thresholded (the frozen D-44
                                threshold) over the non-sink rows; D-43 gate needs >= 57.6.
      training_nonsink_l0       "training-time non-sink L0 (all-row 2048 batch top-k)": the
                                model's OWN batch top-k run on 2048-row batches of ALL rows of the
                                window range (sinks included at their natural 1/128 rate, batch
                                membership by a seeded permutation, i.e. shuffled as the trainer's
                                loader shuffles), then the per-row L0 averaged over the NON-SINK
                                rows only.  The sinks absorb a large share of the k*2048 budget
                                (L0 ~2000 each), so this sits below k (about 47-49 measured at L12)
                                by an arm-dependent amount -- exactly the shift D-44 declares.
      training_sink_l0          the same encoding's mean L0 on the sink rows (diagnostic).
      n_rows / n_nonsink / n_sink, windows, perm_seed, split, layer.

    ``perm_seed`` must be derived from the DATASET (e.g. seeds.derive_seed('a16.d44',
    'L12|test|0-400')), never from the arm, so both arms see the same batches (D-21).  With
    n_windows * 128 not a multiple of 2048 the last batch is wrap-padded as encode_rows does.
    """
    if not is_batchtopk(meta):
        raise ValueError(f"d44_l0_columns on a {arm_family(meta)} arm")
    man, _ = _shard_files(config.resid_dir(layer, split, corpus))
    ctx = int(man.get("ctx_len", config.LM_CTX_LEN))
    X, ids = read_shard_rows(layer, split, start_window * ctx, (start_window + n_windows) * ctx,
                             corpus=corpus, drop_sinks=False)
    if X.shape[0] != n_windows * ctx:
        raise ValueError(f"window range {start_window}+{n_windows} exceeds the {split} split")
    ns = (ids % ctx) != 0
    _, l0_bt = encode_rows(model, meta, X, device, perm_seed)          # all-row 2048 batch top-k
    _, l0_thr = encode_thresholded_rows(model, meta, X[ns], device, threshold)
    return {
        "inference_nonsink_l0": float(l0_thr.mean()),
        "training_nonsink_l0": float(l0_bt[ns].mean()),
        "training_sink_l0": float(l0_bt[~ns].mean()) if (~ns).any() else float("nan"),
        "threshold": float(threshold),
        "n_rows": int(X.shape[0]), "n_nonsink": int(ns.sum()), "n_sink": int((~ns).sum()),
        "windows": [int(start_window), int(start_window + n_windows)], "split": split,
        "layer": int(layer), "corpus": corpus, "perm_seed": int(perm_seed),
        "batch_policy": encode_policy(meta, int(X.shape[0]), perm_seed)["policy"],
    }


# =========================================================================== D-04 statistics
def code_stats(z: sp.csr_matrix, dict_size: Optional[int] = None) -> Dict[str, float]:
    """D-04 row statistics from the codes an app actually scores: mean/median/min/max per-row L0,
    live-latent count (latents nonzero on at least one scored row), dead % (100 * (1 - live /
    dict_size)).  'Dead' here is relative to THESE rows, which is what D-04 asks for (the
    training-time dead count uses a step window and is a different quantity)."""
    n, F_ = z.shape
    dict_size = int(dict_size or F_)
    l0 = np.diff(z.indptr)
    live = int(np.unique(z.indices).size) if z.nnz else 0
    return {
        "n_rows": int(n),
        "mean_l0": float(l0.mean()) if n else float("nan"),
        "median_l0": float(np.median(l0)) if n else float("nan"),
        "min_l0": int(l0.min()) if n else 0,
        "max_l0": int(l0.max()) if n else 0,
        "live_latents": live,
        "dead_pct": 100.0 * (1.0 - live / dict_size),
        "dict_size": dict_size,
    }


def row_annotation(meta: Dict[str, Any], stats: Dict[str, float]) -> Dict[str, Any]:
    """The D-04 columns every result row carries: checkpoint name, best_epoch (the TRUE best-val
    epoch), eval L0, dead %, live-latent count."""
    return {"checkpoint": meta.get("checkpoint_name", FINAL_CKPT),
            "best_epoch": meta.get("best_val_epoch"),
            "eval_l0": stats["mean_l0"], "dead_pct": stats["dead_pct"],
            "live_latents": stats["live_latents"]}
