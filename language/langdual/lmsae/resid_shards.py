"""Dataset over the fp16 residual-stream memmap shards, with the LMDBFeatureDataset contract.

``ResidShardDataset`` is a drop-in substitute for the vision core's
``dataset/lmdb_features.LMDBFeatureDataset`` for the shards written by
``langdual.lmsae.extract_resid``.  It returns the SAME tuple cascade, so the core's
``utils.unpack_batch`` -- and therefore its trainers (``train_topk.py``, ``train_other.py``,
``train_dual.py``) and post-hoc scorers -- accept these activations with no change at all.
The core's ``dataset/resolve.build_feature_dataset`` dispatches here for any directory that
carries a ``manifest.json``.

THE ONE DELIBERATE DIFFERENCE: no L2 normalization.
    ``LMDBFeatureDataset.__getitem__`` divides every vector by its own L2 norm.
    That is right for CLIP/SigLIP embeddings, whose norm carries no task signal.  It is WRONG
    for a residual stream:
      * the residual-stream norm grows systematically with depth and varies strongly across
        tokens; it is signal, not nuisance, and per-sample normalization deletes it;
      * the SAE reconstruction has to be spliceable back INTO the model for the
        cross-entropy-loss-recovered metric.  A per-sample rescale is not invertible from the
        reconstruction alone, so a unit-norm-trained SAE cannot be spliced back without
        smuggling the original norm in.
    Scale is instead handled ONCE, globally, at extraction time: a single fixed per-layer
    ``norm_factor`` is folded into the stored fp16 bytes.  Nothing is rescaled at read time,
    which is why training and evaluation cannot disagree about the scale.
    Correspondingly, ``config.TRACKB_NORMALIZE`` is False, so the models' own
    ``input_unit_norm`` path is off for every language cell.

Self-test (CPU, no GPU, no downloads):
    python -m langdual.lmsae.resid_shards --self-test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from torch.utils.data import Dataset

try:  # normal, package-relative import
    from .. import config
except ImportError:  # pragma: no cover - only taken when run as a bare script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from langdual import config


SHARD_GLOB = "shard_*.npy"
MANIFEST_NAME = "manifest.json"

# Returned for every token when the caller asks for labels but the shard directory has none.
# Track B's main tables are unsupervised (R^2, L0, MonoScore, dead latents, CE recovered), and
# the trainers call unpack_batch on a 3-tuple regardless, so a constant keeps the tuple arity
# right without inventing structure.  See the __init__ docstring for the consequences.
PLACEHOLDER_LABEL = 0


def _coerce_multihot_to_class_idx(label_vec: np.ndarray) -> int:
    """Smallest active class index, else argmax.

    Reimplemented (not imported) from the vision core's dataset/lmdb_features.py on purpose:
    importing that module drags in ``lmdb`` and requires the core on sys.path, and this file
    must stay importable in a bare CPU self-test.  The behaviour is identical, deliberately.
    """
    if label_vec.ndim != 1:
        label_vec = label_vec.reshape(-1)
    active = np.flatnonzero(label_vec > 0.5)
    if active.size > 0:
        return int(active[0])
    return int(np.argmax(label_vec))


def read_npy_header(path: str) -> Tuple[Tuple[int, ...], np.dtype]:
    """Read shape/dtype from a .npy header without mapping the data.

    Used in __init__ (and by extract_resid's skip-guard) so the shard index can be built from
    the geometry while opening -- and immediately closing -- only a file handle: nothing
    memmap-shaped survives into a DataLoader fork.

    ``np.lib.format._read_array_header`` is private, so a NumPy that renames it falls back to
    a throwaway memmap, which reads the same header and is closed immediately.  The fallback
    costs one extra mmap/munmap per shard at construction time only.
    """
    try:
        with open(path, "rb") as fh:
            version = np.lib.format.read_magic(fh)
            shape, fortran, dtype = np.lib.format._read_array_header(fh, version)
    except AttributeError:  # pragma: no cover - NumPy private-API rename
        mm = np.load(path, mmap_mode="r")
        shape, fortran, dtype = mm.shape, not mm.flags.c_contiguous, mm.dtype
        del mm
    if fortran:
        raise ValueError(f"{path}: Fortran-ordered shards are not supported")
    return tuple(int(s) for s in shape), dtype


class ResidShardDataset(Dataset):
    """Read-only, fork-safe view over a directory of fp16 (n_tokens, d_model) .npy shards."""

    def __init__(
        self,
        shard_dir: str,
        labels: Optional[Union[str, np.ndarray]] = None,
        return_index: bool = False,
        return_class_idx: bool = False,
        return_multilabel: bool = False,
        return_path: bool = False,
        return_center: bool = False,
        limit: Optional[int] = None,
        placeholder_label: int = PLACEHOLDER_LABEL,
        verbose: bool = True,
    ):
        """
        Args:
            shard_dir: directory written by extract_resid, i.e. config.resid_dir(layer, split).
            labels: optional PER-TOKEN labels, as a path to a .npy or an array.  Either
                1-D integer (single label per token) or 2-D uint8/float multi-hot
                (n_tokens, n_classes) -- the latter is a multi-label
                Track B table needs.  Must be aligned to the SAME packed token order the
                shards were written in; the length is checked against the shard total.
                When None and return_class_idx=True, every item gets `placeholder_label`.
                KNOWN LIMITATION: a constant label makes any label-dependent metric
                (purity, KNN, linear probe) meaningless, so those tables must only be built
                for a shard directory that DOES carry labels. Nothing here can detect a
                caller who ignores that, hence the loud warning when verbose=True.
            return_index / return_class_idx / return_multilabel / return_path: exactly as in
                LMDBFeatureDataset, producing exactly the same tuple cascade.
            return_center: accepted only to keep the signature substitutable; True is
                rejected because a "hospital/center id" is a CAMELYON-17 concept with no
                Track B meaning, and silently returning None would corrupt unpack_batch's
                4-tuple branch.
            limit: keep only the first `limit` tokens.  Used for config.TRACKB_SCORE_TOKENS,
                because branch_metrics.encode_split caches the sparse code DENSELY
                (8192 * 4 bytes per row), so a full split would need terabytes of host RAM.
                A PREFIX (not a random subset) so the cap is deterministic and reproducible
                without carrying a subsample index around.
            verbose: print the same kind of summary LMDBFeatureDataset prints.
        """
        if return_center:
            raise ValueError(
                "ResidShardDataset does not support return_center: 'center' is CAMELYON-17 "
                "metadata with no Track B analogue")

        self.shard_dir = str(shard_dir)
        config.assert_safe_text_path(self.shard_dir)
        self.return_index = return_index
        self.return_class_idx = return_class_idx
        self.return_multilabel = return_multilabel
        self.return_path = return_path
        self.placeholder_label = int(placeholder_label)

        files = sorted(glob.glob(os.path.join(self.shard_dir, SHARD_GLOB)))
        if not files:
            raise FileNotFoundError(f"no {SHARD_GLOB} shards under {self.shard_dir}")
        self.shard_files: List[str] = files

        # --- geometry -----------------------------------------------------------------
        counts: List[int] = []
        dims: List[int] = []
        for f in files:
            shape, dtype = read_npy_header(f)
            if len(shape) != 2:
                raise ValueError(f"{f}: expected a 2-D (n_tokens, d_model) shard, got {shape}")
            if dtype != np.dtype(np.float16):
                # Loud rather than silent: a float8_e4m3fn or float32 shard would still "work"
                # numerically but would mean a different storage protocol than the one the
                # manifest and the extraction gate describe.
                raise ValueError(f"{f}: expected float16 shards, got {dtype}")
            counts.append(shape[0])
            dims.append(shape[1])
        if len(set(dims)) != 1:
            raise ValueError(f"shards disagree about d_model: {sorted(set(dims))}")
        self.feature_dim = int(dims[0])
        self.feat_shape: Tuple[int, ...] = (self.feature_dim,)

        # Exclusive prefix sums; _starts[i] is the global index of shard i's first token, and
        # _starts[-1] is the total.  searchsorted on this is O(log n_shards) per item.
        self._starts = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(np.asarray(counts, dtype=np.int64), out=self._starts[1:])
        total = int(self._starts[-1])

        # --- optional manifest cross-check ---------------------------------------------
        # The manifest is optional (the self-test builds shards without one), but when present
        # a disagreement means the directory was half-rewritten, so refuse rather than train on
        # a mixture of two extractions.
        self.manifest: Optional[dict] = None
        mpath = os.path.join(self.shard_dir, MANIFEST_NAME)
        if os.path.exists(mpath):
            with open(mpath) as fh:
                self.manifest = json.load(fh)
            m_tokens = int(self.manifest.get("n_tokens", total))
            m_dim = int(self.manifest.get("d_model", self.feature_dim))
            if m_dim != self.feature_dim:
                raise ValueError(f"{mpath}: d_model {m_dim} != shard d_model {self.feature_dim}")
            if m_tokens != total:
                raise ValueError(
                    f"{mpath}: manifest claims {m_tokens} tokens but the shards on disk hold "
                    f"{total}; the directory is inconsistent (partial re-extraction?)")

        # --- limit ----------------------------------------------------------------------
        if limit is not None:
            if limit <= 0:
                raise ValueError(f"limit must be positive, got {limit}")
            if limit > total and verbose:
                print(f"Warning: limit={limit} exceeds the {total} tokens available; using all")
            total = min(total, int(limit))
        self.length = total

        # --- labels ---------------------------------------------------------------------
        self.labels: Optional[np.ndarray] = None
        self.labels_path: Optional[str] = None
        if labels is not None:
            if isinstance(labels, (str, os.PathLike)):
                self.labels_path = str(labels)
                # mmap_mode='r': a per-token label array for 64M tokens is 512 MB as int64 and
                # must not be materialised in every DataLoader worker.
                arr = np.load(self.labels_path, mmap_mode="r")
            else:
                arr = np.asarray(labels)
            if arr.ndim not in (1, 2):
                raise ValueError(f"labels must be 1-D or 2-D, got shape {arr.shape}")
            if arr.shape[0] < self.length:
                raise ValueError(
                    f"labels cover {arr.shape[0]} tokens but the dataset has {self.length}; "
                    f"labels must be aligned to the packed token order of the shards")
            self.labels = arr
        self.is_multilabel = self.labels is not None and self.labels.ndim == 2

        if self.return_multilabel and not self.is_multilabel:
            # Asking for multi-hot vectors from scalar (or absent) labels would silently
            # change the collated batch's shape downstream, so it is an error here.
            raise ValueError(
                "return_multilabel=True requires 2-D multi-hot labels; got "
                + ("no labels" if self.labels is None else f"shape {self.labels.shape}"))

        self._mmaps: Optional[List[np.ndarray]] = None  # opened lazily, see _open()

        if verbose:
            print(f"Loading residual-stream shards from: {self.shard_dir}")
            print(f"  {len(self.shard_files)} shards, {self.length} tokens, dim {self.feature_dim}")
            print(f"  Has labels: {self.labels is not None}"
                  + (f" (multi-hot, {self.labels.shape[1]} classes)" if self.is_multilabel else ""))
            if self.manifest is not None:
                print(f"  norm_factor {self.manifest.get('norm_factor')} "
                      f"(folded into the stored bytes; no read-time rescaling)")
            if self.return_class_idx and self.labels is None:
                print(f"  WARNING: no labels supplied; every item returns the constant "
                      f"placeholder {self.placeholder_label}. Label-dependent metrics "
                      f"(purity / KNN / linear probe) are MEANINGLESS on this dataset.")
            print(f"=> Dataset has {self.length} samples.")

    # ------------------------------------------------------------------ plumbing
    def __len__(self) -> int:
        return self.length

    def _open(self) -> List[np.ndarray]:
        """Open the shard memmaps lazily.

        Same fork-safety reason as LMDBFeatureDataset._open_env: a DataLoader worker is forked
        from the parent, and a memmap/handle created before the fork is shared state.  Opening
        on first use inside the worker gives every worker its own mapping.
        """
        if self._mmaps is None:
            self._mmaps = [np.load(f, mmap_mode="r") for f in self.shard_files]
        return self._mmaps

    def _locate(self, idx: int) -> Tuple[int, int]:
        """Global token index -> (shard, offset within shard)."""
        # side='right' minus one gives the largest start <= idx, i.e. the owning shard.
        s = int(np.searchsorted(self._starts, idx, side="right")) - 1
        return s, idx - int(self._starts[s])

    # ------------------------------------------------------------------ item access
    def __getitem__(self, idx):
        if idx < 0:  # support Python's negative indexing; samplers never use it, humans do
            idx += self.length
        if not (0 <= idx < self.length):
            raise IndexError(f"index {idx} out of range for {self.length} tokens")

        mmaps = self._open()
        s, off = self._locate(idx)
        # .astype gives a fresh, writable, C-contiguous float32 row.  fp16 -> float32 happens
        # here (per item) rather than in the model, because the vision-core trainers assume float32
        # inputs and a fp16 batch would silently change every matmul's precision.
        x = np.asarray(mmaps[s][off]).astype(np.float32)

        # NO L2 NORMALIZATION HERE -- see the module docstring.  The global norm_factor is
        # already folded into the stored bytes.

        class_idx = None
        sample_path = None

        if self.return_class_idx:
            if self.labels is None:
                class_idx = self.placeholder_label
            elif self.is_multilabel:
                vec = np.asarray(self.labels[idx])
                if self.return_multilabel:
                    class_idx = vec.astype(np.float32)  # matches LMDBFeatureDataset:280
                else:
                    class_idx = _coerce_multihot_to_class_idx(vec)
            else:
                class_idx = int(self.labels[idx])

        if self.return_path:
            # A residual-stream token has no file path.  str(idx) is exactly what
            # LMDBFeatureDataset falls back to when __path__ keys are absent, so
            # every consumer that prints or groups by "path" behaves identically.
            sample_path = str(idx)

        # The cascade below is byte-for-byte the order LMDBFeatureDataset uses, so
        # unpack_batch resolves the same way for LMDB features and activation shards.
        if self.return_class_idx and self.return_path:
            return x, class_idx, sample_path
        if self.return_class_idx and self.return_index:
            return x, class_idx, idx
        if self.return_class_idx:
            return x, class_idx
        if self.return_path:
            return x, sample_path
        if self.return_index:
            return x, idx
        return x

    # ------------------------------------------------------------------ misc
    def get_feature_dim(self) -> int:
        """Return the feature dimension (config.LM_D_MODEL for the pythia residual stream)."""
        return self.feature_dim

    @property
    def norm_factor(self) -> Optional[float]:
        """The scalar folded into the stored bytes, or None when no manifest is present.

        Exposed so a splice-back-into-the-model step (CE loss recovered) can undo the fold
        exactly, rather than re-deriving a scale that would not match.
        """
        if self.manifest is None:
            return None
        nf = self.manifest.get("norm_factor")
        return float(nf) if nf is not None else None

    @classmethod
    def from_layer_split(cls, layer: int, split: str, **kwargs) -> "ResidShardDataset":
        """Build from the registry's path helper, so no caller hardcodes a shard path."""
        return cls(config.resid_dir(int(layer), str(split)), **kwargs)

    def close(self) -> None:
        # np.memmap releases its mapping on deallocation; dropping the references is the
        # supported way to close it.
        self._mmaps = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter teardown
            pass


# --------------------------------------------------------------------------- self-test
def _load_unpack_batch():
    """Load the vision core's utils.py unpack_batch by file path.

    By file path rather than `from utils import unpack_batch` because `utils` is a dangerously
    generic module name to put on sys.path, and the core is imported by bare module name from
    its own directory.  Falls back to None so the self-test still runs outside the repo.
    """
    import importlib.util
    path = os.path.join(config.SAE_DIR, "utils.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("_sae_utils_for_selftest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.unpack_batch


def _self_test() -> int:
    """Build tiny synthetic shards on CPU and assert the trainers' contract holds."""
    import shutil
    import tempfile
    import torch
    from torch.utils.data import DataLoader

    unpack_batch = _load_unpack_batch()
    rng = np.random.default_rng(0)
    d = 16
    counts = [7, 5, 3]          # uneven shards on purpose: exercises the searchsorted index
    total = sum(counts)
    tmp = tempfile.mkdtemp(prefix="residshard_selftest_")
    failures: List[str] = []
    try:
        # Values are deliberately NOT unit-norm so the "no L2 normalization" assertion below
        # can actually fail if someone reintroduces normalization.
        ref = (rng.standard_normal((total, d)) * 3.0).astype(np.float16)
        o = 0
        for i, c in enumerate(counts):
            np.save(os.path.join(tmp, "shard_%05d.npy" % i), ref[o:o + c])
            o += c

        labels_1d = rng.integers(0, 4, size=total).astype(np.int64)
        np.save(os.path.join(tmp, "labels.npy"), labels_1d)
        labels_mh = (rng.random((total, 6)) > 0.7).astype(np.uint8)

        def check(cond: bool, msg: str) -> None:
            if not cond:
                failures.append(msg)

        # --- plain: features only ------------------------------------------------------
        ds = ResidShardDataset(tmp, verbose=False)
        check(len(ds) == total, f"len {len(ds)} != {total}")
        check(ds.get_feature_dim() == d, "get_feature_dim mismatch")
        x0 = ds[0]
        check(isinstance(x0, np.ndarray) and x0.dtype == np.float32,
              f"item dtype {getattr(x0, 'dtype', None)} != float32")
        check(x0.shape == (d,), f"item shape {x0.shape} != ({d},)")
        # Exact fp16 -> float32 round-trip: the stored value must arrive unchanged.
        check(np.array_equal(x0, ref[0].astype(np.float32)), "item 0 content mismatch")
        check(np.array_equal(np.asarray(ds[total - 1]), ref[-1].astype(np.float32)),
              "last item content mismatch (shard index arithmetic)")
        check(abs(float(np.linalg.norm(x0)) - 1.0) > 1e-3,
              "item appears L2-normalized; per-sample normalization must NOT be applied")
        check(x0.flags.writeable, "returned array must be writable (a memmap view is not)")

        # 2-tuple / 3-tuple arity through a real DataLoader with forked workers.
        dl = DataLoader(ds, batch_size=4, num_workers=2, shuffle=False)
        b = next(iter(dl))
        check(isinstance(b, torch.Tensor) and b.dtype == torch.float32 and tuple(b.shape) == (4, d),
              f"plain batch is {type(b)} {getattr(b, 'shape', None)}")
        if unpack_batch is not None:
            f, c, p = unpack_batch(b)
            check(f.shape == (4, d) and c is None and p is None, "unpack_batch(plain) mismatch")

        # --- labels: scalar ------------------------------------------------------------
        ds2 = ResidShardDataset(tmp, labels=os.path.join(tmp, "labels.npy"),
                                return_class_idx=True, verbose=False)
        item = ds2[3]
        check(isinstance(item, tuple) and len(item) == 2, f"arity {len(item)} != 2 for class_idx")
        check(int(item[1]) == int(labels_1d[3]), "scalar label mismatch")
        b2 = next(iter(DataLoader(ds2, batch_size=4, num_workers=2, shuffle=False)))
        check(isinstance(b2, list) and len(b2) == 2, f"collated arity {len(b2)} != 2")
        if unpack_batch is not None:
            f, c, p = unpack_batch(b2)
            check(f.shape == (4, d) and c.shape == (4,) and c.dtype == torch.int64 and p is None,
                  "unpack_batch(class_idx) mismatch")

        # --- labels + path: the 3-tuple the trainers actually request -------------------
        ds3 = ResidShardDataset(tmp, labels=labels_1d, return_class_idx=True,
                                return_path=True, verbose=False)
        it3 = ds3[5]
        check(len(it3) == 3 and isinstance(it3[2], str), "3-tuple (x, label, path) mismatch")
        b3 = next(iter(DataLoader(ds3, batch_size=4, num_workers=0, shuffle=False)))
        if unpack_batch is not None:
            f, c, p = unpack_batch(b3)
            # torch 2.11's default_collate returns a TUPLE of strings (older releases returned
            # a list), so accept either -- the trainers only iterate/index it.
            check(f.shape == (4, d) and c.shape == (4,) and isinstance(p, (list, tuple))
                  and len(p) == 4 and all(isinstance(q, str) for q in p),
                  "unpack_batch(class_idx+path) mismatch")

        # --- multi-hot labels ----------------------------------------------------------
        ds4 = ResidShardDataset(tmp, labels=labels_mh, return_class_idx=True,
                                return_multilabel=True, verbose=False)
        _, mh = ds4[2]
        check(isinstance(mh, np.ndarray) and mh.dtype == np.float32 and mh.shape == (6,),
              f"multi-hot item is {getattr(mh, 'dtype', None)} {getattr(mh, 'shape', None)}")
        b4 = next(iter(DataLoader(ds4, batch_size=4, num_workers=0, shuffle=False)))
        check(tuple(b4[1].shape) == (4, 6) and b4[1].dtype == torch.float32,
              "collated multi-hot shape/dtype mismatch")
        # Collapsing a multi-hot to a scalar must match lmdb_features' rule.
        ds5 = ResidShardDataset(tmp, labels=labels_mh, return_class_idx=True, verbose=False)
        _, ci = ds5[2]
        check(ci == _coerce_multihot_to_class_idx(labels_mh[2]), "multi-hot collapse mismatch")

        # --- placeholder label ---------------------------------------------------------
        ds6 = ResidShardDataset(tmp, return_class_idx=True, verbose=False)
        check(ds6[9][1] == PLACEHOLDER_LABEL, "placeholder label not returned")

        # --- limit ---------------------------------------------------------------------
        ds7 = ResidShardDataset(tmp, limit=9, verbose=False)
        check(len(ds7) == 9, f"limit gave {len(ds7)} != 9")
        check(np.array_equal(np.asarray(ds7[8]), ref[8].astype(np.float32)), "limit prefix mismatch")
        try:
            ds7[9]
            failures.append("limit did not bound __getitem__")
        except IndexError:
            pass

        # --- rejected configurations ----------------------------------------------------
        for bad, msg in (
            (lambda: ResidShardDataset(tmp, return_center=True, verbose=False), "return_center"),
            (lambda: ResidShardDataset(tmp, return_multilabel=True, verbose=False), "return_multilabel w/o labels"),
            (lambda: ResidShardDataset(tmp, labels=labels_1d[:2], return_class_idx=True, verbose=False),
             "short label array"),
        ):
            try:
                bad()
                failures.append(f"{msg} should have raised")
            except ValueError:
                pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("SELF-TEST FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"SELF-TEST OK (unpack_batch {'checked' if unpack_batch else 'NOT available, skipped'})")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="CPU synthetic-shard contract test")
    ap.add_argument("--describe", metavar="SHARD_DIR", default=None,
                    help="print the geometry of an existing shard directory")
    ap.add_argument("--limit", type=int, default=None, help="--describe: cap the token count")
    args = ap.parse_args(argv)

    if args.describe:
        ds = ResidShardDataset(args.describe, limit=args.limit, verbose=True)
        print(f"first item norm = {float(np.linalg.norm(ds[0])):.4f} (expected ~1 on train, "
              f"since the global norm factor is folded in)")
        return 0
    if args.self_test:
        return _self_test()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
