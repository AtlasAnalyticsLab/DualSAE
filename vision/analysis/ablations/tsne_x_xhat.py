#!/usr/bin/env python3
"""Input-reconstruction alignment: joint t-SNE of the input embeddings x and the reconstructions
x_hat of one SAE checkpoint (supplementary figure; BatchTopK vs. Dual-BatchTopK, one panel each).

Note: the plot window is fixed to [-AXIS_LIMIT, AXIS_LIMIT] in both t-SNE coordinates (as in the
figure), so points of the embedding that fall outside it are not drawn."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parents[1]), str(_HERE)):
	if _p not in sys.path:
		sys.path.insert(0, _p)

from dataset.lmdb_features import LMDBFeatureDataset
from tsne_sparse_codes import (
	ENCODER_LMDB_MAP,
	_fit_tsne_embedding,
	_l2_normalize_bank,
	load_autoencoder,
)


# The plot is downscaled when used as a panel of a composite figure, so keep its text
# large enough to remain legible there.
TITLE_FONT_SIZE = 22
TICK_FONT_SIZE = 20
LEGEND_FONT_SIZE = 28
LEGEND_MARKER_SIZE = 20
AXIS_LIMIT = 5
POINT_SIZE = 400

def _model_display_name(checkpoint: str) -> str:
	"""Return the paper-facing model name encoded in a checkpoint path."""
	run_name = Path(checkpoint).parent.name.lower()
	if run_name.startswith("dual_") and "batch_topk" in run_name:
		return "Dual-BatchTopK SAE"
	if "batch_topk" in run_name:
		return "BatchTopK SAE"
	return Path(checkpoint).stem


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
		idx = int(rng.integers(0, len(label_dataset)))
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


def _labels_for_indices(label_dataset: LMDBFeatureDataset, indices: Sequence[int]) -> List[int]:
	if label_dataset.offline_labels is not None:
		labels = np.asarray(label_dataset.offline_labels)
		return [int(labels[i]) for i in indices]

	labels_list: List[int] = []
	for idx in indices:
		_, label = label_dataset[idx]
		labels_list.append(int(label))
	return labels_list


def _extract_features(batch) -> torch.Tensor:
	if isinstance(batch, (list, tuple)):
		return batch[0]
	return batch


def build_x_xhat_banks(
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

	banks: Dict[str, List[torch.Tensor]] = {"x": [], "xHat": []}
	model = model.to(device).eval()

	with torch.no_grad():
		for features in loader:
			x = _extract_features(features).to(device, non_blocking=True).float()
			outputs = model(x)
			if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
				x_hat = outputs[-1]
			else:
				raise RuntimeError("Model forward did not return reconstruction output.")

			banks["x"].append(x.detach().cpu().float())
			banks["xHat"].append(x_hat.detach().cpu().float())

	return {k: torch.cat(v, dim=0) for k, v in banks.items()}


def main() -> None:
	parser = argparse.ArgumentParser(description="Joint t-SNE of x and x_hat across checkpoints")
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
	parser.add_argument("--samples-per-class", type=int, default=None, help="Samples per class")
	parser.add_argument("--output-dir", type=str, default="./outputs/ablations/figures/recon_alignment", help="Output directory")
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--perplexity", type=float, default=30.0)
	parser.add_argument("--max-iter", type=int, default=1000)
	parser.add_argument("--random-state", type=int, default=0)
	parser.add_argument("--model-name", type=str, default=None, help="Model name shown in the plot title")

	args = parser.parse_args()

	lmdb_path = args.lmdb or ENCODER_LMDB_MAP.get(args.encoder, "")
	if not lmdb_path:
		raise ValueError("No LMDB path available. Provide --lmdb or select a supported --encoder.")

	device = torch.device(args.device)
	dataset = LMDBFeatureDataset(lmdb_path, return_class_idx=True, return_index=False, return_path=False, verbose=True)

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

	class_labels = _labels_for_indices(dataset, sample_indices)
	unique_classes = sorted(set(class_labels))
	print(f"Coloring by {len(unique_classes)} classes: {unique_classes}")

	# Build shared x bank once from the first checkpoint (input features are identical).
	first_model, _ = load_autoencoder(args.checkpoints[0], device)
	base_banks = build_x_xhat_banks(
		first_model,
		dataset,
		device,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		sample_indices=sample_indices,
	)
	x_vectors = _l2_normalize_bank(base_banks["x"]).numpy()

	all_vectors: List[np.ndarray] = [x_vectors]
	meta_records: List[Dict[str, str]] = [
		{"checkpoint": "input", "output": "x", "shape": str(x_vectors.shape)}
	]
	group_sizes: List[int] = [x_vectors.shape[0]]

	for ckpt in args.checkpoints:
		print(f"Loading checkpoint: {ckpt}")
		model, model_type = load_autoencoder(ckpt, device)
		banks = build_x_xhat_banks(
			model,
			dataset,
			device,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			sample_indices=sample_indices,
		)
		x_hat = _l2_normalize_bank(banks["xHat"]).numpy()
		all_vectors.append(x_hat)
		meta_records.append({"checkpoint": ckpt, "model_type": model_type, "output": "xHat", "shape": str(x_hat.shape)})
		group_sizes.append(x_hat.shape[0])

	concat = np.concatenate(all_vectors, axis=0)
	coords = _fit_tsne_embedding(concat, perplexity=args.perplexity, max_iter=args.max_iter, random_state=args.random_state)

	output_cmap = plt.get_cmap("Set2", 2)
	output_to_color = {"x": output_cmap(0), "xHat": output_cmap(1)}
	ckpt_cmap = plt.get_cmap("tab10", max(len(args.checkpoints), 1))
	ckpt_to_edge = {ckpt: ckpt_cmap(i) for i, ckpt in enumerate(args.checkpoints)}
	markers = {"x": "o", "xHat": "^"}

	fig, ax = plt.subplots(figsize=(6.1, 6))

	start = 0
	# Plot x once (no checkpoint edge color).
	end = start + group_sizes[0]
	ax.scatter(
		coords[start:end, 0],
		coords[start:end, 1],
		s=POINT_SIZE,
		alpha=0.8,
		c='#0072B2',
		marker=markers["x"],
		linewidths=0.0,
		edgecolors="none",
	)
	start = end

	# Plot each checkpoint's x_hat.
	for ckpt_idx, ckpt in enumerate(args.checkpoints):
		end = start + group_sizes[ckpt_idx + 1]
		ax.scatter(
			coords[start:end, 0],
			coords[start:end, 1],
			s=POINT_SIZE,
			alpha=0.8,
			c='#E69F00',
			marker=markers["xHat"],
			linewidths=0.0,
			edgecolors=[ckpt_to_edge[ckpt]],
		)
		start = end

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
		plt.Line2D(
			[0],
			[0],
			marker=markers["x"],
			color="w",
			markerfacecolor='#0072B2',
			# markeredgecolor="k",
			linestyle="None",
			markersize=LEGEND_MARKER_SIZE,
			label=r"$x$",
		),
		plt.Line2D(
			[0],
			[0],
			marker=markers["xHat"],
			color="w",
			markerfacecolor='#E69F00',
			# markeredgecolor="k",
			linestyle="None",
			markersize=LEGEND_MARKER_SIZE,
			label=r"$\hat{x}$",
		),
	]
	ax.legend(
		handles=output_handles,
		loc="best",
		fontsize=LEGEND_FONT_SIZE,
		handlelength=1,
		handletextpad=0.6,
		borderpad=0.5,
		labelspacing=0.5,
	)

	model_name = args.model_name or _model_display_name(args.checkpoints[0])
	ax.set_title(rf"$x$ and $\hat{{x}}$ from {model_name}", fontsize=TITLE_FONT_SIZE)
	ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
	ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
	ax.tick_params(axis="both", which="both", labelsize=TICK_FONT_SIZE)

	out_dir = Path(args.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / f"{'__'.join(Path(p).stem for p in args.checkpoints)}_x_xhat_tsne.png"
	meta_path = out_dir / f"{'__'.join(Path(p).stem for p in args.checkpoints)}_x_xhat_tsne.json"
	fig.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close(fig)

	metadata = {
		"checkpoints": args.checkpoints,
		"num_samples_per_group": group_sizes,
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
