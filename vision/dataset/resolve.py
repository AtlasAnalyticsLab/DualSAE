"""One implementation of the two decisions every trainer and scorer has to make.

BEFORE: "is this dataset multi-label?" was answered by a path SUBSTRING test
(`"open_image" in path or "openimages" in path`) copy-pasted into four files --
train_dual.py, train_topk.py, train_other.py and knn_lp_sparse.py (which re-exports it to
branch_metrics.py).  That has two failure modes, both silent:

  * a genuinely multi-label corpus whose name is not "openimages" is scored with the
    single-label formulas, which do not raise -- they return publishable-looking wrong
    numbers; and
  * any path that happens to contain the substring is switched into multi-label mode.

AFTER: resolve_multilabel() consults, most specific first, an explicit value recorded by the
caller (persisted into training_config.json, so a re-score can never disagree with the run that
produced the checkpoint), then a self-describing marker inside the LMDB, then the legacy
substring.  Keeping the substring LAST is what makes every existing OpenImages run resolve
exactly as before.

build_feature_dataset() is the second decision: LMDB of pre-extracted vectors, or a directory of
memmap activation shards.  Dispatching on the path shape lets the same trainers serve both the
vision/text embedding tracks and the language-model residual-stream track with no fork.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import lmdb

from .lmdb_features import LMDBFeatureDataset

# The legacy heuristic, kept verbatim so its behaviour is unchanged, now in ONE place.
_LEGACY_MULTILABEL_SUBSTRINGS = ("open_image", "openimages")


def is_legacy_multilabel_path(path: Optional[str]) -> bool:
    """The original OpenImages path-name test.  Last resort only."""
    low = str(path or "").lower()
    return any(token in low for token in _LEGACY_MULTILABEL_SUBSTRINGS)


def read_lmdb_multilabel_marker(path: Optional[str]) -> Optional[bool]:
    """Read ``__multilabel__`` out of an LMDB without constructing a Dataset.

    Returns None when the path is not a readable LMDB or carries no marker, which is the case
    for every LMDB written before this refactor -- hence None means "no opinion", not "False".
    """
    if not path or not os.path.isdir(str(path)):
        return None
    try:
        env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
    except Exception:
        # A missing or corrupt LMDB is the caller's problem to report with context; here we
        # only decline to have an opinion.
        return None
    try:
        with env.begin() as txn:
            marker = txn.get(b"__multilabel__")
    finally:
        env.close()
    if marker is None:
        return None
    return marker.decode("ascii").strip() in ("1", "true", "True")


def resolve_multilabel(path: Optional[str], explicit: Optional[bool] = None) -> bool:
    """Decide whether `path` carries multi-hot labels.

    Args:
        path: the feature store (LMDB directory or shard directory).
        explicit: a value recorded by the caller, e.g. ``training_config.json["multilabel"]``
            or a ``--multilabel`` flag.  Wins over everything, because it is the only source
            that provably describes the run that produced the checkpoint.

    Edge case worth naming: a shard directory (Track B) is never an LMDB, so the marker read
    returns None and the answer falls through to `explicit`, which the Track B caller always
    supplies.
    """
    if explicit is not None:
        return bool(explicit)
    marker = read_lmdb_multilabel_marker(path)
    if marker is not None:
        return marker
    return is_legacy_multilabel_path(path)


def is_shard_dir(path: Optional[str]) -> bool:
    """True for a residual-activation shard directory (Track B), False for an LMDB.

    An LMDB directory holds data.mdb/lock.mdb; a shard directory holds manifest.json.  Testing
    for the manifest rather than against the LMDB files means a malformed directory fails loudly
    downstream instead of being silently routed to the wrong reader.
    """
    return bool(path) and os.path.isfile(os.path.join(str(path), "manifest.json"))


def build_feature_dataset(path: str, **kwargs: Any):
    """Return the right Dataset for `path`, with one __getitem__ contract either way.

    Both classes honour return_index / return_class_idx / return_multilabel / return_path and
    expose get_feature_dim(), so callers -- the three trainers, branch_metrics and
    knn_lp_sparse -- need no branch of their own.

    The ResidShardDataset import is LAZY and local: the vision core must not acquire a hard
    dependency on the language package (``language/langdual``), so a vision-only checkout keeps
    working and only a caller that actually points at a shard directory pays for the import (and
    gets a clear error if the package is absent).
    """
    if is_shard_dir(path):
        try:
            import sys
            # <repo>/vision/dataset/resolve.py -> <repo>/language holds the `langdual` package.
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            for root in (os.path.join(repo_root, "language"), repo_root):
                if root not in sys.path:
                    sys.path.insert(0, root)
            from langdual.lmsae.resid_shards import ResidShardDataset  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - only hit on a partial checkout
            raise ImportError(
                f"{path} looks like an activation-shard directory (it has manifest.json) but "
                f"langdual.lmsae.resid_shards could not be imported: {exc}"
            ) from exc
        # Residual streams must NOT be per-sample normalized; ResidShardDataset never does.
        kwargs.pop("normalize_rows", None)
        return ResidShardDataset(path, **kwargs)
    return LMDBFeatureDataset(path, **kwargs)
