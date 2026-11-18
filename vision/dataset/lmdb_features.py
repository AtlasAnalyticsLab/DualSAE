"""Generic LMDB feature dataset for pre-extracted embeddings."""

import ast
import os
import lmdb
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple, Union, Optional


def _parse_feat_shape(shape_str: str) -> Tuple[int, ...]:
    """Parse LMDB shape metadata safely.

    Supports tuple/list strings like "(1024,)" and legacy values like
    "torch.Size([1024])" without requiring torch to be imported.
    """
    shape_str = shape_str.strip()

    if shape_str.startswith("torch.Size(") and shape_str.endswith(")"):
        shape_literal = shape_str[len("torch.Size("):-1].strip()
    else:
        shape_literal = shape_str

    parsed = ast.literal_eval(shape_literal)

    if isinstance(parsed, int):
        parsed = (parsed,)
    elif isinstance(parsed, list):
        parsed = tuple(parsed)
    elif not isinstance(parsed, tuple):
        raise ValueError(f"Invalid feature shape metadata: {shape_str}")

    try:
        return tuple(int(dim) for dim in parsed)
    except Exception as exc:
        raise ValueError(f"Invalid feature shape metadata: {shape_str}") from exc


def _infer_split_from_path(path: str) -> Optional[str]:
    """Infer dataset split name from LMDB path."""
    lower = path.lower()
    if "train" in lower:
        return "train"
    if "val" in lower or "validation" in lower:
        return "validation"
    if "test" in lower:
        return "test"
    return None


def _load_offline_imagenet_labels(lmdb_path: str, split: Optional[str]):
    """Load offline ImageNet labels from NumPy files when available."""
    if split is None:
        return None

    lower = lmdb_path.lower()
    if "imagenet" not in lower and "ilsvrc" not in lower:
        return None

    split_to_filename = {
        "train": "label_train.npy",
        "validation": "label_validation.npy",
        "test": "label_test.npy",
    }
    label_filename = split_to_filename.get(split)
    if label_filename is None:
        return None

    # Optional fallback for LMDBs written without __label__ keys: a directory with
    # label_<split>.npy files, given by the IMAGENET_LABEL_DIR environment variable. The
    # extractors in this repository always store labels inside the LMDB, so this is unused
    # unless the variable is set.
    label_dir = os.environ.get("IMAGENET_LABEL_DIR")
    if not label_dir:
        return None
    label_path = Path(label_dir) / label_filename
    if not label_path.exists():
        return None

    return np.load(label_path, mmap_mode="r")


def _coerce_scalar_label(value):
    """Convert scalar or sequence-like label values to a single int class id."""
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            raise ValueError("Empty label sequence encountered")
        value = value[0]
    return int(value)


def _coerce_multihot_to_class_idx(label_vec: np.ndarray) -> int:
    """Convert a multi-hot label vector to a deterministic scalar class id.

    For multi-label datasets (e.g. OpenImages), pick the smallest active class index.
    If no label is active, fall back to argmax.
    """
    if label_vec.ndim != 1:
        label_vec = label_vec.reshape(-1)
    active = np.flatnonzero(label_vec > 0.5)
    if active.size > 0:
        return int(active[0])
    return int(np.argmax(label_vec))

class LMDBFeatureDataset(Dataset):
    """Fast LMDB dataset using raw bytes (no pickle overhead) for vision features."""

    def __init__(
        self,
        lmdb_path: str,
        return_index: bool = False,
        return_class_idx: bool = False,
        return_multilabel: bool = False,
        return_path: bool = False,
        return_center: bool = False,
        normalize_rows: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            lmdb_path: Path to the LMDB directory.
            return_index: If True, __getitem__ returns (features, index).
            return_class_idx: If True, __getitem__ returns class index from __label__{idx}.
            return_multilabel: If True and multi-label metadata exists, return multi-hot labels.
            return_path: If True, __getitem__ returns sample path from __path__{idx}.
            return_center: If True, __getitem__ appends a hospital/center id (int) read from
                __center__{idx} as the LAST tuple element. Used by the bias-mitigation
                pipeline (CAMELYON-17). When False, return shapes are unchanged so all
                existing callers (and the shared SAE trainers) are unaffected.
            normalize_rows: If True (the default, and what every vision caller relies on)
                each feature vector is L2-normalized in __getitem__.  Set False for data
                where the norm carries signal and must be preserved -- notably a language
                model's residual stream, where per-sample normalization would both destroy
                that signal and make the reconstruction un-spliceable back into the model.
            verbose: If True, print loading information.
        """
        self.lmdb_path = lmdb_path
        self.return_index = return_index
        self.return_class_idx = return_class_idx
        self.return_multilabel = return_multilabel
        self.return_path = return_path
        self.return_center = return_center
        self.normalize_rows = normalize_rows
        self.env = None  # Opened lazily for fork-safety with DataLoader workers
        self.labels_env = None  # Optional companion label LMDB (OpenImages multi-label)
        self.labels_lmdb_path = None
        self.labels_feat_shape = None
        self.offline_labels = None

        # Read metadata once in main process
        env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        with env.begin() as txn:
            # print all data type in the file for overview
            self.length = int(txn.get(b'__len__').decode('ascii'))
            shape_str = txn.get(b'__shape__').decode('ascii')
            self.feat_shape = _parse_feat_shape(shape_str)
            self.feature_dim = self.feat_shape[-1] if len(self.feat_shape) > 0 else 0
            self.has_labels = txn.get(b'__label__0') is not None
            self.has_paths = txn.get(b'__path__0') is not None
            # Optional per-sample site/center metadata (not used by the paper). Stored as an
            # ASCII int under __center__{idx}, mirroring how __label__ is stored.
            self.has_centers = txn.get(b'__center__0') is not None

            # Self-describing multi-label markers.  Historically multi-label mode was inferred
            # from a SUBSTRING of the path ("open_image"/"openimages"), duplicated across four
            # files -- which no non-OpenImages multi-label corpus can satisfy without a
            # dishonest directory name, and which silently mis-scores anything that trips it by
            # accident.  A writer may now declare the fact inside the LMDB itself:
            #   __multilabel__    b"1" / b"0"
            #   __labels_lmdb__   where the multi-hot vectors live; an ABSOLUTE path (so one
            #                     label store can be shared by several feature LMDBs, e.g. the
            #                     three text encoders of one corpus) or a bare basename, which
            #                     is joined against this LMDB's parent for backward compat.
            marker = txn.get(b'__multilabel__')
            self.declared_multilabel = None if marker is None else marker.decode('ascii').strip() in ('1', 'true', 'True')
            labels_ref = txn.get(b'__labels_lmdb__')
            self.declared_labels_lmdb = None if labels_ref is None else labels_ref.decode('utf-8').strip()
        env.close()

        # Optional companion labels LMDB holding multi-hot vectors.
        #
        # Resolution order, most specific first:
        #   1. __labels_lmdb__ written into this LMDB (absolute path, or basename joined
        #      against the parent directory);
        #   2. the legacy sibling name openimages_labels.lmdb, so every existing OpenImages
        #      run resolves EXACTLY as before and reproduces bit-for-bit.
        lmdb_parent = self.lmdb_path.rstrip('/').rsplit('/', 1)[0]
        if self.declared_labels_lmdb:
            ref = self.declared_labels_lmdb
            # An absolute path is used as-is; a bare name is a sibling, matching the legacy
            # convention.  os.sep rather than a hardcoded '/' keeps this honest on any platform.
            candidate_labels_lmdb = ref if os.path.isabs(ref) or '/' in ref else f"{lmdb_parent}/{ref}"
        else:
            candidate_labels_lmdb = f"{lmdb_parent}/openimages_labels.lmdb"
        self.has_label_lmdb = False
        if self.return_class_idx and not self.has_labels:
            try:
                labels_env = lmdb.open(
                    candidate_labels_lmdb,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                with labels_env.begin() as labels_txn:
                    labels_len_bytes = labels_txn.get(b'__len__')
                    labels_shape_bytes = labels_txn.get(b'__shape__')
                    sample0 = labels_txn.get(b'0')
                    if labels_len_bytes is not None and labels_shape_bytes is not None and sample0 is not None:
                        self.labels_lmdb_path = candidate_labels_lmdb
                        self.labels_feat_shape = _parse_feat_shape(labels_shape_bytes.decode('ascii'))
                        self.has_label_lmdb = True
                labels_env.close()
            except Exception:
                self.has_label_lmdb = False

        if self.return_class_idx:
            split = _infer_split_from_path(lmdb_path)
            self.offline_labels = _load_offline_imagenet_labels(lmdb_path, split)
            if self.offline_labels is not None and verbose:
                print(f"  Loaded {len(self.offline_labels)} labels from offline NumPy file")

        if (
            self.return_class_idx
            and self.declared_multilabel is True
            and not self.return_multilabel
        ):
            raise ValueError(
                f"LMDB at {lmdb_path} declares __multilabel__=1 but was opened with "
                f"return_multilabel=False.  Scoring multi-hot labels with the single-label "
                f"formulas does not raise -- it silently returns wrong numbers -- so this is "
                f"refused here.  Resolve the flag with dataset.resolve_multilabel()."
            )

        if (
            self.return_class_idx
            and not self.has_labels
            and self.offline_labels is None
            and not self.has_label_lmdb
        ):
            raise ValueError(
                f"return_class_idx=True but LMDB at {lmdb_path} has no labels "
                f"and offline NumPy labels could not be loaded."
            )
        if self.return_path and not self.has_paths and verbose:
            print(
                f"Warning: LMDB at {lmdb_path} does not contain sample paths "
                f"(__path__{{idx}} keys). Falling back to index identifiers."
            )
        if self.return_center and not self.has_centers and verbose:
            # Non-fatal: center is optional metadata. Callers receive center=None.
            print(
                f"Warning: LMDB at {lmdb_path} does not contain center ids "
                f"(__center__{{idx}} keys). Returning center=None."
            )

        if verbose:
            print(f"Loading features from: {lmdb_path}")
            print(f"  Loaded {self.length} samples with shape {self.feat_shape}")
            print(
                "  Has labels: "
                f"{self.has_labels or self.offline_labels is not None or self.has_label_lmdb}"
            )
            if self.offline_labels is not None:
                print("  Using offline NumPy labels")
            if self.has_label_lmdb:
                print(f"  Using companion label LMDB: {self.labels_lmdb_path}")
            print(f"=> Dataset has {self.length} samples.")
    
    def __len__(self):
        return self.length
    
    def _open_env(self):
        """Lazily open LMDB environment (fork-safe for DataLoader workers)."""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=True,  # Enable readahead for contiguous batch sampling
                meminit=False
            )
        if self.has_label_lmdb and self.labels_env is None:
            self.labels_env = lmdb.open(
                self.labels_lmdb_path,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
            )

    def __getitem__(self, idx) -> Union[np.ndarray, Tuple[np.ndarray, Union[int, np.ndarray]], Tuple[np.ndarray, Union[int, np.ndarray], int], Tuple[np.ndarray, Union[int, np.ndarray], str]]:
        self._open_env()
        with self.env.begin() as txn:
            data = txn.get(str(idx).encode('ascii'))
            if data is None:
                raise KeyError(f"Missing feature for sample index {idx}")
            
            # Make a copy to ensure the array is writable
            x = np.frombuffer(data, dtype=np.float32).reshape(self.feat_shape).copy()
            class_idx = None
            sample_path = None
            center = None

            if self.return_class_idx:
                # Try LMDB labels first
                if self.has_labels:
                    label_bytes = txn.get(f'__label__{idx}'.encode('ascii'))
                    if label_bytes is not None:
                        class_idx = int(label_bytes.decode('ascii'))
                    else:
                        if self.offline_labels is not None:
                            class_idx = _coerce_scalar_label(self.offline_labels[idx])
                        else:
                            raise KeyError(f"Missing label for sample index {idx} in LMDB")
                elif self.offline_labels is not None:
                    class_idx = _coerce_scalar_label(self.offline_labels[idx])
                # Fall back to companion OpenImages labels LMDB (multi-hot vectors)
                elif self.has_label_lmdb:
                    with self.labels_env.begin() as labels_txn:
                        label_bytes = labels_txn.get(str(idx).encode('ascii'))
                    if label_bytes is None:
                        raise KeyError(f"Missing companion label for sample index {idx}")
                    label_vec = np.frombuffer(label_bytes, dtype=np.float32).reshape(self.labels_feat_shape)
                    if self.return_multilabel:
                        class_idx = label_vec.astype(np.float32).copy()
                    else:
                        class_idx = _coerce_multihot_to_class_idx(label_vec)
                # Fall back to HF labels
                else:
                    raise KeyError(f"No labels available for sample index {idx}")
            
            if self.return_path:
                if self.has_paths:
                    path_bytes = txn.get(f'__path__{idx}'.encode('ascii'))
                    sample_path = path_bytes.decode('utf-8') if path_bytes is not None else str(idx)
                else:
                    sample_path = str(idx)

            if self.return_center and self.has_centers:
                center_bytes = txn.get(f'__center__{idx}'.encode('ascii'))
                center = int(center_bytes.decode('ascii')) if center_bytes is not None else None

        # Normalize feature vector to unit norm.  Unconditional for every vision caller
        # (normalize_rows defaults True, so their behaviour is byte-identical); skipped only
        # when a caller explicitly opted out because the norm carries signal -- see the
        # normalize_rows docstring.
        if self.normalize_rows:
            norm = np.linalg.norm(x)
            if norm > 0:
                x = x / (norm + 1e-8)

        # When centers are requested, append `center` as the LAST element. These
        # branches are only taken when return_center=True, so the return shapes
        # below (the original cascade) are byte-for-byte unchanged for every
        # existing caller, including the shared SAE trainers.
        if self.return_center:
            if self.return_class_idx and self.return_path:
                return x, class_idx, sample_path, center
            if self.return_class_idx:
                return x, class_idx, center
            if self.return_path:
                return x, sample_path, center
            return x, center

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
    
    @property
    def has_openimages_label_lmdb(self) -> bool:
        """Deprecated alias for has_label_lmdb.

        Kept because analysis scripts read this attribute through
        getattr(dataset, "has_openimages_label_lmdb", False).  A plain rename would make that
        getattr silently return the default and disable multi-label handling there without
        raising.
        """
        return self.has_label_lmdb

    def get_feature_dim(self) -> int:
        """Return the feature dimension."""
        return self.feature_dim
    
    def close(self):
        """Close the LMDB environment."""
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
            self.env = None
        if hasattr(self, 'labels_env') and self.labels_env is not None:
            self.labels_env.close()
            self.labels_env = None
    
    def __del__(self):
        self.close()

