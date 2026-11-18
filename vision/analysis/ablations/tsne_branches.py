#!/usr/bin/env python3
"""Branch decomposition t-SNE (panel a of the branch-decomposition figure): joint t-SNE of the
Sparse Branch output x^M, the Residual Branch output r_hat and the reconstruction x_hat.

Given several checkpoints, extract model outputs (xM, rHat, xHat) for a sampled
subset of the LMDB feature dataset, fit a joint t-SNE over all extracted
vectors, and plot them together in a single figure. Each checkpoint gets a
color, and each output type gets a marker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import sys

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parents[1]), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset.lmdb_features import LMDBFeatureDataset  # noqa: E402

# Reuse helpers from the pairwise comparison script.
from tsne_sparse_codes import (  # noqa: E402
    ENCODER_LMDB_MAP,
    _fit_tsne_embedding,
    _l2_normalize_bank,
    _load_imagenet_class_descriptions,
    _class_id_to_display_name_map,
    load_autoencoder,
)


# The plot is downscaled when used as a panel of a composite figure, so keep its text
# large enough to remain legible there.
TITLE_FONT_SIZE = 22
TICK_FONT_SIZE = 20
ANNOTATION_FONT_SIZE = 17


def build_banks_three_outputs(
    model: torch.nn.Module,
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

    banks: Dict[str, List[torch.Tensor]] = {"xM": [], "rHat": [], "xHat": []}

    model = model.to(device).eval()
    with torch.no_grad():
        for features in loader:
            if isinstance(features, (list, tuple)):
                x = features[0]
            else:
                x = features
            x = x.to(device, non_blocking=True).float()
            output = model(x)
            xM = model._xM.detach().cpu().float()
            rHat = model._r_hat.detach().cpu().float()
            # final reconstruction may be returned as last output
            xHat = output[-1].detach().cpu().float()

            banks["xM"].append(xM)
            banks["rHat"].append(rHat)
            banks["xHat"].append(xHat)

    return {k: torch.cat(v, dim=0) for k, v in banks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint t-SNE for multiple checkpoints")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths to include")
    parser.add_argument("--lmdb", default="", help="LMDB path (overrides encoder default)")
    # clip: value used for the figure
    parser.add_argument("--encoder", default="clip", choices=sorted(ENCODER_LMDB_MAP.keys()))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=1000, help="Total samples to draw from dataset")
    parser.add_argument("--class-idx", type=int, default=None, help="Single class index to sample from")
    parser.add_argument("--class-idxs", nargs="+", type=int, default=None, help="Multiple class indices to sample from")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of random classes to sample")
    parser.add_argument("--samples-per-class", type=int, default=None, help="Samples per class (defaults to num-samples / num-classes)")
    parser.add_argument("--output-dir", type=str, default="./outputs/ablations/figures/branch_tsne", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=0)

    args = parser.parse_args()

    lmdb_path = args.lmdb or ENCODER_LMDB_MAP.get(args.encoder, "")

    if not lmdb_path:
        raise ValueError("No LMDB path available. Provide --lmdb or select a supported --encoder.")

    device = torch.device(args.device)

    dataset = LMDBFeatureDataset(lmdb_path, return_class_idx=True, return_index=False, return_path=False, verbose=True)
    num_items = len(dataset)

    def _sample_multi_class_indices(
        label_dataset: LMDBFeatureDataset,
        num_samples: int,
        random_state: int,
        class_idx: Optional[int],
        class_idxs: Optional[Iterable[int]],
        num_classes: int,
        samples_per_class: Optional[int],
    ) -> Tuple[List[int], List[int]]:
        rng = np.random.default_rng(random_state)
        if class_idx is not None:
            class_list = [class_idx]
        elif class_idxs is not None:
            class_list = [int(c) for c in class_idxs]
        else:
            class_list = []

        if class_list:
            num_classes = len(class_list)
        samples_per_class = samples_per_class or max(1, num_samples // max(num_classes, 1))

        if label_dataset.offline_labels is not None:
            labels = np.asarray(label_dataset.offline_labels)
            if not class_list:
                class_list = rng.choice(np.unique(labels), size=num_classes, replace=False).tolist()
            selected = []
            for cls in class_list:
                cls = int(cls)
                indices = np.flatnonzero(labels == cls)
                if indices.size == 0:
                    raise ValueError(f"No samples found for class {cls}")
                take = min(samples_per_class, indices.size)
                selected.extend(rng.choice(indices, size=take, replace=False).tolist())
            rng.shuffle(selected)
            return selected, [int(c) for c in class_list]

        class_to_indices: Dict[int, List[int]] = {}
        seen = set()
        max_tries = max(num_samples * 50, 20000)
        for _ in range(max_tries):
            if class_list and all(len(class_to_indices.get(c, [])) >= samples_per_class for c in class_list):
                break
            if not class_list and len(class_to_indices) >= num_classes and all(
                len(v) >= samples_per_class for v in class_to_indices.values()
            ):
                break
            idx = int(rng.integers(0, num_items))
            if idx in seen:
                continue
            seen.add(idx)
            _, label = label_dataset[idx]
            label = int(label)
            if class_list and label not in class_list:
                continue
            class_to_indices.setdefault(label, [])
            if len(class_to_indices[label]) < samples_per_class:
                class_to_indices[label].append(idx)

        if not class_list:
            class_list = list(class_to_indices.keys())
        selected = []
        for cls in class_list:
            selected.extend(class_to_indices.get(cls, [])[:samples_per_class])

        if not selected:
            raise ValueError("Unable to sample any class from the dataset")
        if len(selected) < samples_per_class * len(class_list):
            print(
                f"Warning: only found {len(selected)} samples across classes {class_list}; "
                f"requested {samples_per_class * len(class_list)}."
            )
        rng.shuffle(selected)
        return selected, [int(c) for c in class_list]

    sample_indices, selected_classes = _sample_multi_class_indices(
        dataset,
        args.num_samples,
        args.random_state,
        args.class_idx,
        args.class_idxs,
        args.num_classes,
        args.samples_per_class,
    )
    print(f"Using class indices {selected_classes} with {len(sample_indices)} samples")

    def _labels_for_indices(
        label_dataset: LMDBFeatureDataset,
        indices: Sequence[int],
    ) -> List[int]:
        if label_dataset.offline_labels is not None:
            labels = np.asarray(label_dataset.offline_labels)
            return [int(labels[i]) for i in indices]

        labels_list: List[int] = []
        for idx in indices:
            _, label = label_dataset[idx]
            labels_list.append(int(label))
        return labels_list

    class_labels = _labels_for_indices(dataset, sample_indices)
    unique_classes = sorted(set(class_labels))
    print(f"Coloring by {len(unique_classes)} classes: {unique_classes}")
    class_descriptions = _load_imagenet_class_descriptions()
    class_display_names = _class_id_to_display_name_map(unique_classes, class_descriptions)

    # For each checkpoint, build banks of xM, rHat, xHat.
    all_vectors: List[np.ndarray] = []
    meta_records: List[Dict] = []
    per_ckpt_counts: List[int] = []

    for ckpt in args.checkpoints:
        print(f"Loading checkpoint: {ckpt}")
        model, model_type = load_autoencoder(ckpt, device)
        banks = build_banks_three_outputs(
            model,
            dataset,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_indices=sample_indices,
        )

        # Normalize each bank and append in order xM, rHat, xHat
        for key in ("xM", "rHat", "xHat"):
            vec = _l2_normalize_bank(banks[key]).numpy()
            all_vectors.append(vec)
            meta_records.append({"checkpoint": ckpt, "model_type": model_type, "output": key, "shape": vec.shape})
            per_ckpt_counts.append(vec.shape[0])

    # Concatenate for joint t-SNE. Expect all vecs to have compatible feature dims; _fit_tsne_embedding will PCA if needed.
    concat = np.concatenate(all_vectors, axis=0)
    coords = _fit_tsne_embedding(concat, perplexity=args.perplexity, max_iter=args.max_iter, random_state=args.random_state)

    # Prepare plotting: class = face color, checkpoint = edge color, output = marker.
    n_ckpt = len(args.checkpoints)
    outputs = ["xM", "rHat", "xHat"]
    OUTPUT_LABEL_MAP = {"xM": r"$x^{M}$", "rHat": r"$\hat{r}$", "xHat": r"$\hat{x}$"}
    class_cmap = plt.get_cmap("tab20", max(len(unique_classes), 1))
    class_to_color = {cls: class_cmap(i) for i, cls in enumerate(unique_classes)}
    ckpt_cmap = plt.get_cmap("tab10", max(n_ckpt, 1))
    ckpt_to_edge = {ckpt: ckpt_cmap(i) for i, ckpt in enumerate(args.checkpoints)}
    markers = {"xM": "o", "rHat": "s", "xHat": "^"}
    point_colors = [class_to_color[label] for label in class_labels]

    fig, ax = plt.subplots(figsize=(6.1, 6))
    start = 0
    pending_labels: List[Tuple[np.ndarray, str]] = []
    centroids_by_class: Dict[int, Dict[str, np.ndarray]] = {}
    for ckpt_idx, ckpt in enumerate(args.checkpoints):
        for out_idx, out_name in enumerate(outputs):
            idx = ckpt_idx * len(outputs) + out_idx
            n_points = per_ckpt_counts[idx]
            end = start + n_points
            subset = coords[start:end]
            m = markers[out_name]
            edge = ckpt_to_edge[ckpt]
            sc = ax.scatter(
                subset[:, 0],
                subset[:, 1],
                s=25,
                alpha=0.7,
                c=point_colors,
                marker=m,
                linewidths=0.5,
                edgecolors=[edge],
            )
            # Collect per-class centroids for later, overlap-aware labeling.
            subset_labels = class_labels
            for cls in unique_classes:
                cls_mask = np.array(subset_labels) == cls
                if not np.any(cls_mask):
                    continue
                cls_points = subset[cls_mask]
                centroid = cls_points.mean(axis=0)
                out_label = OUTPUT_LABEL_MAP.get(out_name, out_name)
                label = f"{class_display_names.get(cls, f'class_{cls}')} - {out_label}"
                pending_labels.append((centroid, label))
                centroids_by_class.setdefault(int(cls), {})[out_name] = centroid
            start = end

    # Leave room for the enlarged centroid labels near the plot boundaries.
    ax.margins(x=0.20, y=0.12)

    # Place labels with simple overlap avoidance in screen space.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed_bboxes = []
    # Only try vertical offsets: above then below the cluster.
    offset_candidates: List[Tuple[int, int]] = []
    for radius in (6, 10, 14, 18, 24, 30, 40):
        offset_candidates.extend([(0, radius), (0, -radius)])

    for centroid, label in pending_labels:
        text_artist = None
        for dx, dy in offset_candidates:
            va = "bottom" if dy > 0 else "top"
            candidate = ax.annotate(
                label,
                xy=(centroid[0], centroid[1]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=ANNOTATION_FONT_SIZE,
                color="black",
                ha="center",
                va=va,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.6),
            )
            bbox = candidate.get_window_extent(renderer=renderer)
            if not any(bbox.overlaps(existing) for existing in placed_bboxes):
                text_artist = candidate
                placed_bboxes.append(bbox)
                break
            candidate.remove()
        if text_artist is None:
            # Fall back to a small upward offset if all overlap.
            candidate = ax.annotate(
                label,
                xy=(centroid[0], centroid[1]),
                xytext=(0, 20),
                textcoords="offset points",
                fontsize=ANNOTATION_FONT_SIZE,
                color="black",
                ha="center",
                va="bottom",
                # bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9),
            )
            placed_bboxes.append(candidate.get_window_extent(renderer=renderer))

    # Draw arrows from xM to xHat for each class that has both centroids.
    for cls, centroids in centroids_by_class.items():
        if "xM" not in centroids or "xHat" not in centroids:
            continue
        start_xy = centroids["xM"]
        end_xy = centroids["xHat"]
        ax.annotate(
            "",
            xy=(end_xy[0], end_xy[1]),
            xytext=(start_xy[0], start_xy[1]),
            arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0, alpha=0.8),
            zorder=1,
        )

    # Build legend entries: classes (face color) + checkpoints (edge color) + outputs (marker)
    class_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=class_to_color[c],
            markeredgecolor="k",
            markersize=8,
            label=class_display_names.get(c, f"class_{c}"),
        )
        for c in unique_classes
    ]
    ckpt_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor=ckpt_to_edge[ckpt],
            markersize=8,
            label=Path(ckpt).stem,
        )
        for ckpt in args.checkpoints
    ]
    output_handles = [
        plt.Line2D([0], [0], marker=markers[o], color="k", linestyle="None", markersize=8, label=OUTPUT_LABEL_MAP.get(o, o))
        for o in outputs
    ]
    handles = output_handles
    # ax.legend(handles=handles, loc='best', fontsize=8)

    ax.set_title(r"Joint t-SNE of $x^{M}$, $\hat{r}$, and $\hat{x}$", fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis="both", which="both", labelsize=TICK_FONT_SIZE)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{'__'.join(Path(p).stem for p in args.checkpoints)}_tsne.png"
    meta_path = out_dir / f"{'__'.join(Path(p).stem for p in args.checkpoints)}_tsne.json"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    metadata = {
        "checkpoints": args.checkpoints,
        "num_samples_per_block": per_ckpt_counts,
        "selected_classes": [int(c) for c in selected_classes],
        "meta_records": meta_records,
        "output_path": str(out_path),
        "tsne": {"perplexity": args.perplexity, "max_iter": args.max_iter, "random_state": args.random_state},
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved joint t-SNE figure to: {out_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
