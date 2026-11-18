#!/usr/bin/env python3
"""t-SNE of sparse representations: joint t-SNE of the sparse codes z of a DualSAE and a standard
SAE ("t-SNE visualization of sparse representations" figure; Dual-JumpReLU vs. JumpReLU).

This script compares two checkpoints at a time. For each pair it:
- loads a shared LMDB feature dataset,
- computes latent activations for a sampled subset,
- fits one joint t-SNE embedding over both checkpoints in latent space,
- saves a comparison figure and a metadata JSON file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, Subset


_HERE = Path(__file__).resolve().parent
_VISION = _HERE.parents[1]
for _p in (str(_VISION), str(_VISION / "scripts" / "ablations"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import feature_paths  # noqa: E402

from dataset.lmdb_features import LMDBFeatureDataset
from models import BatchTopKSAE, DualSAE, JumpReLUSAE, TopK, TopKSAE, VanillaSAE


# ImageNet-1K validation features per encoder ($FEATURES_ROOT layout, scripts/ablations/feature_paths.py).
ENCODER_LMDB_MAP = {enc: feature_paths.lmdb_path("imagenet", enc, "val") for enc in ("clip", "sigclip", "vit")}


# The 20 ImageNet-1K classes drawn in the figure. This fixed list is what the figure used (it is the
# default whenever --class-names is not given); pass --class-names to choose others, or
# --random-classes to sample --num-classes classes with --random-state.
SCREENSHOT_CLASS_NAMES = [
    "sulphur butterfly",
    "ground beetle",
    "barracouta",
    "schooner",
    "steam locomotive",
    "magnetic compass",
    "slide rule",
    "shield",
    "turnstile",
    "peacock",
    "drake",
    "sorrel",
    "groom",
    "shower cap",
    "pajama",
    "curly-coated retriever",
    "Walker hound",
    "Gila monster",
    "robin",
    "daisy",
]

CLUSTER_LABEL_FONT_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize joint t-SNE embeddings for DualSAE and traditional SAE checkpoints."
        )
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        required=True,
        metavar=("DUAL_CHECKPOINT", "BASELINE_CHECKPOINT"),
        help="Checkpoint pair to compare. Repeat to produce multiple figures.",
    )
    parser.add_argument(
        "--lmdb",
        required=False,
        default="",
        help="Path to the LMDB feature store. If omitted, the encoder-specific default is used.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="sigclip",
        choices=sorted(ENCODER_LMDB_MAP.keys()),
        help="Default encoder key used when --lmdb is not provided.",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for feature encoding.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--num-samples", type=int, default=10000, help="Maximum number of samples to visualize.")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of random classes to sample.")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1000,
        help="Maximum number of random samples to draw from each selected class.",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help=(
            "Explicit class names to plot. Use the short name shown in the plot, quoted if it contains spaces. "
            "When provided, these classes are selected instead of random sampling."
        ),
    )
    parser.add_argument("--random-classes", action="store_true",
                        help="sample --num-classes random classes instead of the fixed figure list")
    parser.add_argument("--output-dir", type=str, default="./outputs/ablations/figures/tsne_sparse_codes",
                        help="Directory for outputs.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")  # 5, 10, 30, 50, 100
    parser.add_argument("--max-iter", type=int, default=1000, help="t-SNE optimization iterations.")
    parser.add_argument("--random-state", type=int, default=0, help="Seed for sampling and t-SNE.") # try multiple seeds for robustness
    parser.add_argument(
        "--superclass-level",
        type=int,
        default=-1,
           help="If >= 0, sample classes from different WordNet superclasses at this level "
               "(0=leaf, 1=parent, etc.). If -1 (default), sample random classes without superclass constraint.",
    )
    parser.add_argument(
        "--same-superclass",
        action="store_true",
        help="When used with --superclass-level >= 0, sample all classes from one shared superclass.",
    )
    parser.add_argument(
        "--color-superclass-level",
        type=int,
        default=1,
        help=(
            "WordNet level used for coloring points (0=leaf class, 1=parent, etc.). "
            "Set to -1 to color by class id instead."
        ),
    )
    return parser.parse_args()


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _coalesce(value: Any, default: Any) -> Any:
    return default if value is None else value


def _int_or_default(value: Any, default: int) -> int:
    return int(_coalesce(value, default))


def _float_or_default(value: Any, default: float) -> float:
    return float(_coalesce(value, default))


def _extract_checkpoint_state(checkpoint_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        if isinstance(ckpt, torch.Tensor):
            raise ValueError(f"Checkpoint is a tensor, not a model dict: {checkpoint_path}")
        raise ValueError(f"Unsupported checkpoint type {type(ckpt)} for {checkpoint_path}")

    state_dict = ckpt.get("model_state_dict")
    if state_dict is None:
        state_dict = ckpt.get("model")
    if state_dict is None:
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
        else:
            raise ValueError(f"Cannot find model weights in checkpoint: {checkpoint_path}")

    if not isinstance(state_dict, dict):
        raise ValueError(f"Model weights are not a dictionary in {checkpoint_path}")

    model_cfg = _as_dict(ckpt.get("model_cfg"))
    if not model_cfg:
        model_cfg = _as_dict(ckpt.get("config"))

    return state_dict, model_cfg, ckpt


def _base_cfg_from_state(
    state_dict: Dict[str, torch.Tensor],
    model_cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    if "W_enc" in state_dict:
        act_size, dict_size = state_dict["W_enc"].shape
    elif "encoder.weight" in state_dict:
        dict_size, act_size = state_dict["encoder.weight"].shape
    else:
        raise ValueError("Unable to infer act/dict size from state dict.")

    batch_size = int(model_cfg.get("batch_size", 1))
    dead_steps_threshold = int(model_cfg.get("dead_steps_threshold", 1000))
    n_batches_to_dead = int(model_cfg.get("n_batches_to_dead", max(dead_steps_threshold // max(batch_size, 1), 1)))

    return {
        "act_size": int(model_cfg.get("act_size", act_size)),
        "dict_size": int(model_cfg.get("dict_size", dict_size)),
        "seed": int(model_cfg.get("seed", 0)),
        "device": str(device),
        "dtype": torch.float32,
        "input_unit_norm": bool(model_cfg.get("normalize", model_cfg.get("input_unit_norm", False))),
        "n_batches_to_dead": n_batches_to_dead,
        "l1_coeff": float(model_cfg.get("l1_coeff", model_cfg.get("l1_coef", 0.0))),
    }


def _load_topk_sae(state_dict: Dict[str, torch.Tensor], device: torch.device) -> torch.nn.Module:
    n_latents, d_model = state_dict["encoder.weight"].shape
    activation_state = state_dict.get("activation_state_dict", {})
    activation = TopK(k=_int_or_default(activation_state.get("k"), 64))
    tied = "decoder.linear.weight" in state_dict

    model = TopKSAE(
        n_latents=n_latents,
        n_inputs=d_model,
        activation=activation,
        tied=tied,
        normalize=True,
    )
    state_dict_clean = {
        key: value
        for key, value in state_dict.items()
        if key not in {"activation", "activation_state_dict", "activation.k", "activation.postact_fn"}
    }
    model.load_state_dict(state_dict_clean, strict=True)
    return model.to(device).eval()


def load_autoencoder(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, str]:
    state_dict, model_cfg_raw, _ = _extract_checkpoint_state(checkpoint_path)

    if "encoder.weight" in state_dict:
        return _load_topk_sae(state_dict, device), "topk_sae"

    base_cfg = _base_cfg_from_state(state_dict, model_cfg_raw, device)

    if any(key.startswith("poly.") for key in state_dict.keys()):
        dual_cfg = dict(base_cfg)
        dual_cfg.update(
            {
                "top_k": _int_or_default(model_cfg_raw.get("top_k", model_cfg_raw.get("k_sparsity")), 64),
                "sparsity_mode": str(model_cfg_raw.get("sparsity_mode", "topk")),
                "poly_hidden": _int_or_default(model_cfg_raw.get("poly_hidden"), 512),
                "poly_layers": _int_or_default(model_cfg_raw.get("poly_layers"), 1),
                "poly_hidden_dims": model_cfg_raw.get("poly_hidden_dims", None),
                "bandwidth": _float_or_default(model_cfg_raw.get("bandwidth"), 0.001),
            }
        )
        model = DualSAE(dual_cfg)
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval(), "dual_sae"

    model_type = str(model_cfg_raw.get("model_type", "")).lower()

    if "jumprelu.log_threshold" in state_dict or model_type == "jumprelu":
        jump_cfg = dict(base_cfg)
        jump_cfg.update({"bandwidth": _float_or_default(model_cfg_raw.get("bandwidth"), 0.001)})
        model = JumpReLUSAE(jump_cfg)
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval(), "jumprelu_sae"

    if model_type in {"batch_topk", "batchtopk", "dual_batch_topk"} or "top_k" in model_cfg_raw or "topk_k" in model_cfg_raw:
        batch_cfg = dict(base_cfg)
        batch_cfg.update(
            {
                "top_k": _int_or_default(model_cfg_raw.get("top_k", model_cfg_raw.get("topk_k")), 64),
                "top_k_aux": _int_or_default(model_cfg_raw.get("top_k_aux", model_cfg_raw.get("auxk_k")), 512),
                "aux_penalty": _float_or_default(model_cfg_raw.get("aux_penalty", model_cfg_raw.get("auxk_coef")), 0.0),
            }
        )
        model = BatchTopKSAE(batch_cfg)
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval(), "batchtopk_sae"

    model = VanillaSAE(dict(base_cfg))
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), "vanilla_sae"


def _model_label(model_type: str) -> str:
    labels = {
        "dual_sae": "DualSAE",
        "batchtopk_sae": "BatchTopKSAE",
        "topk_sae": "TopKSAE",
        "jumprelu_sae": "JumpReLUSAE",
        "vanilla_sae": "VanillaSAE",
    }
    return labels.get(model_type, model_type)


def _sample_indices(num_items: int, max_samples: int, seed: int) -> List[int]:
    if max_samples <= 0 or max_samples >= num_items:
        return list(range(num_items))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_items, generator=generator)[:max_samples].tolist()
    indices.sort()
    return indices


def _dataset_labels(dataset: LMDBFeatureDataset) -> np.ndarray:
    offline_labels = getattr(dataset, "offline_labels", None)
    if offline_labels is not None:
        return np.asarray(offline_labels, dtype=np.int64)

    labels = np.empty(len(dataset), dtype=np.int64)
    if getattr(dataset, "has_labels", False):
        dataset._open_env()
        with dataset.env.begin() as txn:
            for idx in range(len(dataset)):
                label_bytes = txn.get(f"__label__{idx}".encode("ascii"))
                if label_bytes is None:
                    raise KeyError(f"Missing label for sample index {idx} in LMDB")
                labels[idx] = int(label_bytes.decode("ascii"))
        return labels

    if getattr(dataset, "has_openimages_label_lmdb", False):
        dataset._open_env()
        with dataset.labels_env.begin() as labels_txn:
            for idx in range(len(dataset)):
                label_bytes = labels_txn.get(str(idx).encode("ascii"))
                if label_bytes is None:
                    raise KeyError(f"Missing companion label for sample index {idx}")
                label_vec = np.frombuffer(label_bytes, dtype=np.float32).reshape(dataset.labels_feat_shape)
                label_vec = np.ravel(label_vec)
                active = np.flatnonzero(label_vec > 0.5)
                labels[idx] = int(active[0]) if active.size > 0 else int(np.argmax(label_vec))
        return labels

    raise ValueError("Dataset does not expose class labels required for class-aware sampling.")


def _wnid_to_synset(wnid: str):
    """Convert an ImageNet WNID (e.g., 'n01440764') to a WordNet synset."""
    if not re.match(r"^n\d{8}$", wnid):
        raise ValueError(f"Invalid wnid: {wnid}")
    offset = int(wnid[1:])
    return wn.synset_from_pos_and_offset("n", offset)


def _get_superclass_at_level(wnid: str, level: int) -> Optional[str]:
    """Get the superclass name at a given hypernym level for a WNID.
    
    level=0: leaf class (the class itself)
    level=1: immediate parent
    level=2: grandparent, etc.
    """
    if level < 0:
        return None
    try:
        synset = _wnid_to_synset(wnid)
        paths = synset.hypernym_paths()
        if not paths:
            return wnid  # No hypernyms; return the WNID itself
        # Paths are ordered root -> ... -> leaf, so index from the end for level counting.
        path = max(paths, key=len)
        if level >= len(path):
            return str(path[0].name())  # Return root if level is too deep
        target_synset = path[-1 - level]
        return str(target_synset.name())
    except Exception:
        return wnid  # Fallback to WNID on any error


def _load_imagenet_wnids() -> List[str]:
    """Load ImageNet-1K class WNIDs in order (0-999)."""
    try:
        from timm.data.imagenet_info import ImageNetInfo
        info = ImageNetInfo("imagenet-1k")
        return list(info.label_names())
    except Exception:
        raise RuntimeError(
            "Unable to load ImageNet WNIDs. Ensure 'timm' is installed: pip install timm"
        )


def _load_imagenet_class_descriptions() -> List[str]:
    """Load ImageNet-1K class display names in index order (0-999)."""
    try:
        from timm.data.imagenet_info import ImageNetInfo
        info = ImageNetInfo("imagenet-1k")
        return list(info.label_descriptions(detailed=False))
    except Exception:
        return []


def _build_superclass_to_classes(
    wnids: List[str],
    level: int,
) -> Dict[str, List[int]]:
    """Build a mapping from superclass name to list of ImageNet class indices."""
    if not WORDNET_AVAILABLE:
        raise RuntimeError(
            "NLTK WordNet is required for superclass grouping. Install it with: "
            "pip install nltk && python -c 'import nltk; nltk.download(\"wordnet\")'"
        )
    superclass_to_classes: Dict[str, List[int]] = {}
    for class_idx, wnid in enumerate(wnids):
        superclass = _get_superclass_at_level(wnid, level)
        if superclass not in superclass_to_classes:
            superclass_to_classes[superclass] = []
        superclass_to_classes[superclass].append(class_idx)
    return superclass_to_classes


def _class_to_superclass_map(
    class_ids: Sequence[int],
    wnids: Optional[List[str]],
    color_superclass_level: int,
) -> Dict[int, str]:
    class_to_superclass: Dict[int, str] = {}
    use_wordnet = (
        color_superclass_level >= 0
        and wnids is not None
        and WORDNET_AVAILABLE
    )

    for class_id in class_ids:
        class_id_int = int(class_id)
        if use_wordnet and 0 <= class_id_int < len(wnids):
            class_to_superclass[class_id_int] = _get_superclass_at_level(
                wnids[class_id_int],
                color_superclass_level,
            ) or f"class_{class_id_int}"
        else:
            class_to_superclass[class_id_int] = f"class_{class_id_int}"
    return class_to_superclass


def _class_id_to_display_name_map(
    class_ids: Sequence[int],
    class_descriptions: Sequence[str],
) -> Dict[int, str]:
    class_to_name: Dict[int, str] = {}
    for class_id in class_ids:
        class_id_int = int(class_id)
        if 0 <= class_id_int < len(class_descriptions):
            raw_name = str(class_descriptions[class_id_int]).strip()
            short_name = raw_name.split(",", 1)[0].strip()
            class_to_name[class_id_int] = short_name if short_name else raw_name
        else:
            class_to_name[class_id_int] = f"class_{class_id_int}"
    return class_to_name


def _resolve_class_ids_by_name(
    class_names: Sequence[str],
    class_descriptions: Sequence[str],
) -> List[int]:
    short_name_to_ids: Dict[str, List[int]] = {}
    for class_id, raw_name in enumerate(class_descriptions):
        short_name = str(raw_name).strip().split(",", 1)[0].strip()
        if not short_name:
            continue
        short_name_to_ids.setdefault(short_name.lower(), []).append(int(class_id))

    selected_ids: List[int] = []
    for requested_name in class_names:
        lookup = requested_name.strip().lower()
        matches = short_name_to_ids.get(lookup, [])
        if not matches:
            raise ValueError(
                f"Unknown class name '{requested_name}'. Use the short class name shown in the plot."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Class name '{requested_name}' is ambiguous across multiple ImageNet classes."
            )
        selected_ids.append(matches[0])

    return selected_ids


def _sample_indices_from_selected_classes(
    labels: np.ndarray,
    selected_classes: Sequence[int],
    samples_per_class: int,
    seed: int,
) -> Tuple[List[int], List[int], Dict[int, int]]:
    if samples_per_class <= 0:
        raise ValueError("--samples-per-class must be > 0")

    rng = np.random.default_rng(seed)
    sampled_indices: List[int] = []
    sampled_classes: List[int] = []
    class_counts: Dict[int, int] = {}

    for class_id in selected_classes:
        class_idx = np.flatnonzero(labels == int(class_id))
        if class_idx.size == 0:
            continue
        take_n = min(samples_per_class, int(class_idx.size))
        picked = rng.choice(class_idx, size=take_n, replace=False)
        sampled_indices.extend(picked.astype(np.int64).tolist())
        sampled_classes.extend([int(class_id)] * take_n)
        class_counts[int(class_id)] = int(take_n)

    if not sampled_indices:
        raise ValueError("Class selection produced no samples.")

    return sampled_indices, sampled_classes, class_counts


def _sample_indices_by_class(
    labels: np.ndarray,
    num_classes: int,
    samples_per_class: int,
    seed: int,
    superclass_level: int = -1,
    same_superclass: bool = False,
    wnids: Optional[List[str]] = None,
) -> Tuple[List[int], List[int], List[int], Dict[int, int]]:
    if num_classes <= 0:
        raise ValueError("--num-classes must be > 0")
    if samples_per_class <= 0:
        raise ValueError("--samples-per-class must be > 0")

    unique_classes = np.unique(labels)
    if unique_classes.size == 0:
        raise ValueError("No classes found in dataset labels.")

    rng = np.random.default_rng(seed)

    # If superclass_level >= 0, sample either from one shared superclass
    # or across different superclasses.
    if superclass_level >= 0 and wnids is not None:
        superclass_to_classes = _build_superclass_to_classes(wnids, superclass_level)
        available_classes_by_superclass: Dict[str, np.ndarray] = {}
        for superclass, classes in superclass_to_classes.items():
            available = np.array([c for c in classes if c in unique_classes], dtype=np.int64)
            if available.size > 0:
                available_classes_by_superclass[superclass] = available

        valid_superclasses = list(available_classes_by_superclass.keys())

        if not valid_superclasses:
            raise ValueError(
                "No valid superclass found with classes available in the dataset. "
                "Try using --superclass-level -1 (disable superclass grouping)."
            )

        if same_superclass:
            same_superclass_candidates = [
                superclass
                for superclass, classes in available_classes_by_superclass.items()
                if classes.size >= num_classes
            ]
            if not same_superclass_candidates:
                raise ValueError(
                    f"No superclass contains at least {num_classes} classes at level {superclass_level}. "
                    "Try lowering --num-classes or disabling --same-superclass."
                )
            selected_superclass = str(rng.choice(same_superclass_candidates, size=1, replace=False)[0])
            superclass_classes = available_classes_by_superclass[selected_superclass]
            selected_classes = rng.choice(superclass_classes, size=num_classes, replace=False).astype(np.int64)
            print(
                f"Selected {len(selected_classes)} classes from one superclass "
                f"{selected_superclass} (level={superclass_level})."
            )
        else:
            take_superclasses = min(num_classes, len(valid_superclasses))
            selected_superclasses = rng.choice(valid_superclasses, size=take_superclasses, replace=False)

            selected_classes_list: List[int] = []
            for superclass in selected_superclasses.tolist():
                superclass_classes = available_classes_by_superclass[superclass]
                picked_class = int(rng.choice(superclass_classes, size=1, replace=False)[0])
                selected_classes_list.append(picked_class)

            selected_classes = np.asarray(selected_classes_list, dtype=np.int64)
            print(
                f"Selected {len(selected_classes)} classes from {len(selected_superclasses)} different superclasses "
                f"(level={superclass_level})."
            )
    else:
        take_classes = min(num_classes, int(unique_classes.size))
        selected_classes = rng.choice(unique_classes, size=take_classes, replace=False)

    sampled_indices: List[int] = []
    sampled_classes: List[int] = []
    class_counts: Dict[int, int] = {}

    for class_id in selected_classes.tolist():
        class_idx = np.flatnonzero(labels == class_id)
        if class_idx.size == 0:
            continue
        take_n = min(samples_per_class, int(class_idx.size))
        picked = rng.choice(class_idx, size=take_n, replace=False)
        sampled_indices.extend(picked.astype(np.int64).tolist())
        sampled_classes.extend([int(class_id)] * take_n)
        class_counts[int(class_id)] = int(take_n)

    if not sampled_indices:
        raise ValueError("Class-aware sampling produced no samples.")

    return sampled_indices, sampled_classes, [int(c) for c in selected_classes.tolist()], class_counts


@torch.no_grad()
def build_banks(
    model: torch.nn.Module,
    model_type: str,
    dataset: LMDBFeatureDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    sample_indices: Sequence[int],
) -> Dict[str, torch.Tensor]:
    subset = Subset(dataset, sample_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    banks: Dict[str, List[torch.Tensor]] = {"latent": []}

    for features in loader:
        x = features.to(device, non_blocking=True).float()

        # Use the second model output as latent representation for both model families.
        _, latent, _ = model(x)

        banks["latent"].append(latent.detach().cpu().float())

    return {key: torch.cat(value, dim=0) for key, value in banks.items()}


def _l2_normalize_bank(bank: torch.Tensor) -> torch.Tensor:
    return F.normalize(bank.float(), p=2, dim=1)


def _fit_tsne_embedding(vectors: np.ndarray, perplexity: float, max_iter: int, random_state: int) -> np.ndarray:
    if vectors.shape[0] < 3:
        raise ValueError("Need at least 3 samples for t-SNE.")

    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError("scikit-learn is required for t-SNE visualizations. Install it with `pip install scikit-learn`.") from exc

    n_samples, n_features = vectors.shape
    if n_features > 50 and n_samples > 50:
        pca_dims = min(50, n_features, n_samples - 1)
        if pca_dims >= 2:
            vectors = PCA(n_components=pca_dims, random_state=random_state).fit_transform(vectors)

    effective_perplexity = min(float(perplexity), max(2.0, (vectors.shape[0] - 1) / 3.0))
    effective_perplexity = max(2.0, effective_perplexity)

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=max_iter,
        random_state=random_state,
    )
    return tsne.fit_transform(vectors)


def _plot_comparison(
    pair_name: str,
    families: List[Dict[str, Any]],
    sample_class_ids: Sequence[int],
    selected_classes: Sequence[int],
    sample_superclass_ids: Sequence[str],
    class_to_superclass: Dict[int, str],
    class_display_names: Dict[int, str],
    selected_superclasses: Sequence[str],
    color_superclass_level: int,
    output_path: Path,
    perplexity: float,
    max_iter: int,
    random_state: int,
) -> None:
    # Fit one joint latent-space t-SNE over both checkpoints, then split by family.
    sample_count = len(sample_class_ids)
    latent_vectors: List[torch.Tensor] = []
    expected_dim: Optional[int] = None
    for family in families:
        bank = _l2_normalize_bank(family["banks"]["latent"])
        if expected_dim is None:
            expected_dim = int(bank.shape[1])
        elif int(bank.shape[1]) != expected_dim:
            raise ValueError(
                f"Mismatched latent dims across checkpoints: expected {expected_dim}, got {int(bank.shape[1])}."
            )
        latent_vectors.append(bank)

    latent_coords = _fit_tsne_embedding(
        torch.cat(latent_vectors, dim=0).numpy(),
        perplexity,
        max_iter,
        random_state,
    )

    superclass_palette = plt.get_cmap("tab20", max(len(selected_superclasses), 1))
    superclass_to_color = {superclass: superclass_palette(i) for i, superclass in enumerate(selected_superclasses)}
    point_colors = np.array([superclass_to_color[superclass] for superclass in sample_superclass_ids])
    sample_class_ids_np = np.asarray(sample_class_ids, dtype=np.int64)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(families),
        figsize=(5.8 * len(families), 5.0),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    if len(families) == 1:
        axes = np.array([axes])

    for family_idx, family in enumerate(families):
        ax = axes[family_idx]
        start = family_idx * sample_count
        end = start + sample_count
        family_coords = latent_coords[start:end]
        ax.scatter(family_coords[:, 0], family_coords[:, 1], s=15, alpha=0.75, c=point_colors, linewidths=0)
        ax.set_title(family["label"])
        family_x_min, family_x_max = float(family_coords[:, 0].min()), float(family_coords[:, 0].max())
        family_y_min, family_y_max = float(family_coords[:, 1].min()), float(family_coords[:, 1].max())
        ax.set_xlim(family_x_min, family_x_max)
        ax.set_ylim(family_y_min, family_y_max)
        # Show numeric axis ticks (5 ticks evenly spaced) for readability.
        ticks_x = np.linspace(family_x_min, family_x_max, 5)
        ticks_y = np.linspace(family_y_min, family_y_max, 5)
        ax.set_xticks(ticks_x)
        ax.set_yticks(ticks_y)
        ax.tick_params(axis='both', labelsize=10)

        # Label each class near its cluster center in this panel.
        for class_id in selected_classes:
            class_id_int = int(class_id)
            class_mask = sample_class_ids_np == class_id_int
            if not np.any(class_mask):
                continue
            class_coords = family_coords[class_mask]
            center_x = float(np.mean(class_coords[:, 0]))
            center_y = float(np.mean(class_coords[:, 1]))
            class_label = class_display_names.get(class_id_int, f"class_{class_id_int}")
            ax.text(
                center_x,
                center_y,
                class_label,
                fontsize=CLUSTER_LABEL_FONT_SIZE,
                ha="center",
                va="center",
                color="black",
                bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none", "pad": 1.0},
                zorder=5,
            )

    legend_handles: List[Line2D] = []
    if color_superclass_level >= 0:
        # Superclass coloring mode: legend shows only superclass names.
        for superclass in selected_superclasses:
            color = superclass_to_color.get(superclass, (0.3, 0.3, 0.3, 1.0))
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    linestyle='None',
                    markerfacecolor=color,
                    markeredgecolor='none',
                    markersize=5,
                    label=str(superclass),
                )
            )
    else:
        for class_id in selected_classes:
            class_id_int = int(class_id)
            superclass = class_to_superclass.get(class_id_int, f"class_{class_id_int}")
            color = superclass_to_color.get(superclass, (0.3, 0.3, 0.3, 1.0))
            label = class_display_names.get(class_id_int, f"class_{class_id_int}")
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    linestyle='None',
                    markerfacecolor=color,
                    markeredgecolor='none',
                    markersize=5,
                    label=label,
                )
            )
    if legend_handles:
        ncol = max(1, min(6, len(legend_handles)))
        nrows = int(np.ceil(len(legend_handles) / ncol))
        # Push legend lower as rows increase to avoid overlap with scatter panels.
        legend_y = -0.08 - 0.06 * max(0, nrows - 1)
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=ncol,
            fontsize=8,
            frameon=False,
        )

    # No legend: class names are drawn next to their clusters instead.
    fig.suptitle(f"t-SNE comparison: {pair_name}")
    # fig.text(0.5, 0.01, "Joint t-SNE fitted on latent activations from both checkpoints (colored by class).", ha="center")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def visualize_pair(
    dual_checkpoint: str,
    baseline_checkpoint: str,
    dataset: LMDBFeatureDataset,
    device: torch.device,
    output_dir: Path,
    sample_indices: Sequence[int],
    sample_class_ids: Sequence[int],
    selected_classes: Sequence[int],
    sample_superclass_ids: Sequence[str],
    selected_superclasses: Sequence[str],
    class_to_superclass: Dict[int, str],
    class_display_names: Dict[int, str],
    color_superclass_level: int,
    class_counts: Dict[int, int],
    batch_size: int,
    perplexity: float,
    max_iter: int,
    random_state: int,
    num_workers: int,
) -> None:
    print(f"Visualizing pair:\n  Dual checkpoint: {dual_checkpoint}\n  Baseline checkpoint: {baseline_checkpoint}")
    dual_model, dual_type = load_autoencoder(dual_checkpoint, device)
    baseline_model, baseline_type = load_autoencoder(baseline_checkpoint, device)
    print(f"Loaded models:\n  Dual: {dual_type}\n  Baseline: {baseline_type}")
    dual_banks = build_banks(
        dual_model,
        dual_type,
        dataset,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_indices=sample_indices,
    )
    baseline_banks = build_banks(
        baseline_model,
        baseline_type,
        dataset,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_indices=sample_indices,
    )

    pair_name = f"{Path(dual_checkpoint).stem}__vs__{Path(baseline_checkpoint).stem}"
    output_path = output_dir / f"{pair_name}_tsne.png"
    metadata_path = output_dir / f"{pair_name}_tsne.json"

    families = [
        {
            "label": f"Dual: {_model_label(dual_type)}",
            "checkpoint": dual_checkpoint,
            "model_type": dual_type,
            "banks": dual_banks,
        },
        {
            "label": f"Baseline: {_model_label(baseline_type)}",
            "checkpoint": baseline_checkpoint,
            "model_type": baseline_type,
            "banks": baseline_banks,
        },
    ]

    _plot_comparison(
        pair_name=pair_name,
        families=families,
        sample_class_ids=sample_class_ids,
        selected_classes=selected_classes,
        sample_superclass_ids=sample_superclass_ids,
        class_to_superclass=class_to_superclass,
        class_display_names=class_display_names,
        selected_superclasses=selected_superclasses,
        color_superclass_level=color_superclass_level,
        output_path=output_path,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
    )

    metadata = {
        "dual_checkpoint": dual_checkpoint,
        "baseline_checkpoint": baseline_checkpoint,
        "dual_model_type": dual_type,
        "baseline_model_type": baseline_type,
        "num_samples": len(sample_indices),
        "sample_indices": sample_indices,
        "selected_classes": [int(c) for c in selected_classes],
        "class_display_names": {str(k): str(v) for k, v in class_display_names.items()},
        "selected_superclasses": [str(s) for s in selected_superclasses],
        "class_to_superclass": {str(k): str(v) for k, v in class_to_superclass.items()},
        "samples_per_class": {str(k): int(v) for k, v in class_counts.items()},
        "output_path": str(output_path),
        "tsne": {
            "perplexity": perplexity,
            "max_iter": max_iter,
            "random_state": random_state,
        },
        "note": "Side-by-side comparison of latent activations (model second output).",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved t-SNE figure to: {output_path}")
    print(f"Saved metadata to: {metadata_path}")


def main() -> None:
    args = parse_args()

    if not args.pair:
        raise SystemExit("pass at least one --pair DUAL_CHECKPOINT BASELINE_CHECKPOINT")
    if not args.class_names and not args.random_classes:
        args.class_names = SCREENSHOT_CLASS_NAMES

    lmdb_path = args.lmdb or ENCODER_LMDB_MAP[args.encoder]
    if not lmdb_path:
        raise ValueError("No LMDB path available. Provide --lmdb or select a supported --encoder.")

    device = torch.device(args.device)
    dataset = LMDBFeatureDataset(
        lmdb_path,
        return_class_idx=False,
        return_index=False,
        return_path=False,
        verbose=True,
    )

    label_dataset = LMDBFeatureDataset(
        lmdb_path,
        return_class_idx=True,
        return_index=False,
        return_path=False,
        verbose=False,
    )
    all_labels = _dataset_labels(label_dataset)

    class_descriptions = _load_imagenet_class_descriptions()
    
    wnids = None
    if args.superclass_level >= 0 or args.color_superclass_level >= 0:
        print("Loading ImageNet WNIDs for superclass grouping/coloring...")
        wnids = _load_imagenet_wnids()
    
    if args.class_names:
        selected_classes = _resolve_class_ids_by_name(args.class_names, class_descriptions)
        sampled_indices, sampled_class_ids, class_counts = _sample_indices_from_selected_classes(
            labels=all_labels,
            selected_classes=selected_classes,
            samples_per_class=args.samples_per_class,
            seed=args.random_state,
        )
    else:
        sampled_indices, sampled_class_ids, selected_classes, class_counts = _sample_indices_by_class(
            labels=all_labels,
            num_classes=args.num_classes,
            samples_per_class=args.samples_per_class,
            seed=args.random_state,
            superclass_level=args.superclass_level,
            same_superclass=args.same_superclass,
            wnids=wnids,
        )
    class_to_superclass = _class_to_superclass_map(
        class_ids=selected_classes,
        wnids=wnids,
        color_superclass_level=args.color_superclass_level,
    )
    class_display_names = _class_id_to_display_name_map(
        class_ids=selected_classes,
        class_descriptions=class_descriptions,
    )
    sample_superclass_ids = [class_to_superclass[int(class_id)] for class_id in sampled_class_ids]
    selected_superclasses = list(dict.fromkeys(sample_superclass_ids))
    print(
        f"Sampling {len(selected_classes)} classes and {len(sampled_indices)} total points "
        f"(target {args.samples_per_class} per class)."
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dual_checkpoint, baseline_checkpoint in args.pair:
        visualize_pair(
            dual_checkpoint=dual_checkpoint,
            baseline_checkpoint=baseline_checkpoint,
            dataset=dataset,
            device=device,
            output_dir=output_dir,
            sample_indices=sampled_indices,
            sample_class_ids=sampled_class_ids,
            selected_classes=selected_classes,
            sample_superclass_ids=sample_superclass_ids,
            selected_superclasses=selected_superclasses,
            class_to_superclass=class_to_superclass,
            class_display_names=class_display_names,
            color_superclass_level=args.color_superclass_level,
            class_counts=class_counts,
            batch_size=args.batch_size,
            perplexity=args.perplexity,
            max_iter=args.max_iter,
            random_state=args.random_state,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
