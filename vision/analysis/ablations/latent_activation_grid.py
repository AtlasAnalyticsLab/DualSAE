#!/usr/bin/env python3
"""Interpretability of sparse latents: activating images and per-latent MonoScore / purity curves.

Compares a BatchTopK SAE and a Dual-BatchTopK SAE on the same dataset split ("Improved
interpretability of sparse latents" figure):
- per-latent MonoScore and per-latent class purity of every latent of both models, sorted and
  plotted against the normalised latent index (--monoscore_plot_path / --class_purity_plot_path);
  the per-latent arrays are also saved to <output stem>_curves.npz;
- for latents paired by MonoScore rank (rank r of BatchTopK next to rank r of Dual-BatchTopK), a PIL
  figure with the top / mid / bottom --k activating images of each latent.

Images are read from --image_root/<path stored in the LMDB> when given; otherwise the ImageNet-1K
LMDB index is used to fetch the image from the Hugging Face `ILSVRC/imagenet-1k` dataset, which must
be in the local cache (scripts/ablations/prepare_assets.sh; HF_DATASETS_OFFLINE=1 at run time).
"""

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader

import sys
_VISION = Path(__file__).resolve().parents[2]
for _p in (str(_VISION), str(_VISION / "scripts" / "ablations")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset.lmdb_features import LMDBFeatureDataset  # noqa: E402
from models.batch_topk_sae import BatchTopKSAE  # noqa: E402
from models.dual_sae import DualSAE  # noqa: E402
import feature_paths  # noqa: E402


_HF_CACHE: Dict[str, object] = {}

COMPARISON_LEGEND_FONT_SIZE = 14


def resolve_lmdb_path(args: argparse.Namespace) -> str:
    if args.lmdb_path:
        return args.lmdb_path

    split_key = args.split
    lmdb_path = feature_paths.lmdb_path(args.dataset, args.extractor,
                                        {"validation": "val"}.get(split_key, split_key))
    if lmdb_path is None:
        raise ValueError(
            f"No default LMDB path for dataset={args.dataset}, extractor={args.extractor}, split={split_key}. "
            "Pass --lmdb_path explicitly."
        )
    return lmdb_path


def infer_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    if "W_enc" in state_dict:
        act_size, dict_size = state_dict["W_enc"].shape
        return int(act_size), int(dict_size)
    if "encoder.weight" in state_dict:
        dict_size, act_size = state_dict["encoder.weight"].shape
        return int(act_size), int(dict_size)
    raise ValueError("Could not infer (act_size, dict_size) from checkpoint state dict")


def load_batch_topk_model(checkpoint_path: str, device: torch.device) -> BatchTopKSAE:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    cfg = ckpt.get("config", {})

    act_size, dict_size = infer_dims_from_state_dict(state_dict)
    model_cfg = {
        "act_size": int(cfg.get("d_model", act_size)),
        "dict_size": int(cfg.get("n_latents", dict_size)),
        "top_k": int(cfg.get("topk_k", cfg.get("top_k", 64))),
        "l1_coeff": 0.0,
        "top_k_aux": int(cfg.get("auxk_k") or 512),
        "aux_penalty": float(cfg.get("auxk_coef", 0.0)),
        "n_batches_to_dead": max(1, int(cfg.get("dead_steps_threshold", 1000)) // max(1, int(cfg.get("batch_size", 1)))),
        "seed": int(cfg.get("seed", 0)),
        "device": str(device),
        "dtype": torch.float32,
        "input_unit_norm": bool(cfg.get("normalize", False)),
    }

    model = BatchTopKSAE(model_cfg)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def load_dual_model(checkpoint_path: str, device: torch.device) -> DualSAE:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    cfg = ckpt.get("config", {})

    act_size, dict_size = infer_dims_from_state_dict(state_dict)
    model_cfg = {
        "act_size": int(cfg.get("d_model", act_size)),
        "dict_size": int(cfg.get("n_latents", dict_size)),
        "top_k": int(cfg.get("top_k", cfg.get("topk_k", 64))),
        "l1_coeff": float(cfg.get("l1_coef", 1e-4)),
        "bandwidth": float(cfg.get("bandwidth", 0.001)),
        "poly_hidden": int(cfg.get("poly_hidden", 512)),
        "sparsity_mode": str(cfg.get("sparsity_mode", "batch_topk")),
        "n_batches_to_dead": max(1, int(cfg.get("dead_steps_threshold", 1000)) // max(1, int(cfg.get("batch_size", 1)))),
        "seed": int(cfg.get("seed", 0)),
        "device": str(device),
        "dtype": torch.float32,
        "input_unit_norm": bool(cfg.get("normalize", False)),
    }

    model = DualSAE(model_cfg)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def unpack_features_and_paths(batch, running_start: int) -> Tuple[torch.Tensor, List[str]]:
    if isinstance(batch, (tuple, list)):
        features = batch[0]
        maybe_path = batch[-1]

        if isinstance(maybe_path, str):
            paths = [maybe_path]
        elif isinstance(maybe_path, (list, tuple)) and len(maybe_path) > 0 and isinstance(maybe_path[0], str):
            paths = [str(v) for v in maybe_path]
        elif hasattr(maybe_path, "dtype") and str(getattr(maybe_path, "dtype", "")).startswith("<U"):
            paths = [str(v) for v in maybe_path.tolist()]
        else:
            batch_size = features.shape[0]
            paths = [str(i) for i in range(running_start, running_start + batch_size)]
        return features, paths

    features = batch
    batch_size = features.shape[0]
    paths = [str(i) for i in range(running_start, running_start + batch_size)]
    return features, paths


def unpack_features_labels_paths(batch, running_start: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[str]]:
    """Unpack (features, labels, paths) from different dataset return formats."""
    if isinstance(batch, (tuple, list)):
        features = batch[0]
        labels = None
        paths = None

        # Common dataset outputs:
        # (x, y), (x, path), (x, y, path), (x, y, idx)
        if len(batch) >= 2:
            second = batch[1]
            if torch.is_tensor(second):
                labels = second
            elif isinstance(second, np.ndarray) and np.issubdtype(second.dtype, np.integer):
                labels = torch.from_numpy(second)
            elif isinstance(second, (list, tuple)) and len(second) > 0 and isinstance(second[0], (int, np.integer)):
                labels = torch.as_tensor(second, dtype=torch.long)

        if len(batch) >= 3:
            third = batch[2]
            if isinstance(third, str):
                paths = [third]
            elif isinstance(third, (list, tuple)) and len(third) > 0 and isinstance(third[0], str):
                paths = [str(v) for v in third]
            elif hasattr(third, "dtype") and str(getattr(third, "dtype", "")).startswith("<U"):
                paths = [str(v) for v in third.tolist()]

        if paths is None and len(batch) >= 2:
            second = batch[-1]
            if isinstance(second, str):
                paths = [second]
            elif isinstance(second, (list, tuple)) and len(second) > 0 and isinstance(second[0], str):
                paths = [str(v) for v in second]

        if paths is None:
            batch_size = features.shape[0]
            paths = [str(i) for i in range(running_start, running_start + batch_size)]

        return features, labels, paths

    features = batch
    batch_size = features.shape[0]
    paths = [str(i) for i in range(running_start, running_start + batch_size)]
    return features, None, paths


@torch.no_grad()
def choose_latent_idx(model, loader: DataLoader, device: torch.device, max_batches: Optional[int] = None) -> int:
    counts = None
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break

        features, _ = unpack_features_and_paths(batch, 0)
        x = features.to(device, non_blocking=True).float()
        _, latents, _ = model(x)

        batch_counts = (latents > 0).sum(dim=0).detach().cpu().long()
        if counts is None:
            counts = torch.zeros_like(batch_counts)
        counts += batch_counts

    if counts is None:
        raise RuntimeError("No data seen while auto-selecting latent index")

    return int(torch.argmax(counts).item())


@torch.no_grad()
def collect_latent_activations(
    model,
    dataset: LMDBFeatureDataset,
    latent_idx: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], int]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    n_total = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    activations = np.zeros(n_total, dtype=np.float32)
    all_paths: List[str] = []
    all_active_count = 0

    write_pos = 0
    for batch in loader:
        features, paths = unpack_features_and_paths(batch, write_pos)
        x = features.to(device, non_blocking=True).float()

        _, latents, _ = model(x)
        vals = latents[:, latent_idx].detach().cpu().float().numpy()

        batch_size_now = vals.shape[0]
        remaining = n_total - write_pos
        take = min(batch_size_now, remaining)

        if take <= 0:
            break

        activations[write_pos:write_pos + take] = vals[:take]
        all_paths.extend(paths[:take])
        all_active_count += int((vals[:take] > 0).sum())
        write_pos += take

        if write_pos >= n_total:
            break

    return activations, all_paths, all_active_count


def pick_top_mid_bottom_indices(values: np.ndarray, k: int) -> Dict[str, np.ndarray]:
    n = int(values.shape[0])
    if n == 0:
        raise ValueError("No activation values found")

    k = min(k, n)
    order = np.argsort(values)[::-1]

    top_idx = order[:k]

    mid_start = max(0, (n // 2) - (k // 2))
    mid_end = min(n, mid_start + k)
    mid_start = max(0, mid_end - k)
    mid_idx = order[mid_start:mid_end]

    bottom_idx = order[-k:]

    return {
        "top": top_idx,
        "mid": mid_idx,
        "bottom": bottom_idx,
    }


def _infer_hf_split_from_lmdb_path(lmdb_path: str) -> str:
    lower = lmdb_path.lower()
    if "train" in lower:
        return "train"
    if "test" in lower:
        return "test"
    return "validation"


def _get_hf_dataset(split: str):
    if split in _HF_CACHE:
        return _HF_CACHE[split]

    try:
        load_dataset = importlib.import_module("datasets").load_dataset
    except Exception as exc:
        raise RuntimeError(
            "HuggingFace datasets is required for image fallback. "
            "Install it with: pip install datasets"
        ) from exc

    if split == "test":
        split = "validation"
    _HF_CACHE[split] = load_dataset("ILSVRC/imagenet-1k", split=split)
    return _HF_CACHE[split]


def load_image_for_sample(
    raw_path: str,
    index: int,
    tile_size: int,
    lmdb_path: str,
    image_root: Optional[str],
    use_hf_fallback: bool,
) -> Image.Image:
    candidate_paths: List[Path] = []

    if raw_path:
        p = Path(raw_path)
        candidate_paths.append(p)
        if image_root is not None:
            candidate_paths.append(Path(image_root) / raw_path)

    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file():
            try:
                img = Image.open(candidate).convert("RGB")
                return ImageOps.fit(img, (tile_size, tile_size), method=Image.Resampling.BICUBIC)
            except Exception:
                pass

    if use_hf_fallback:
        try:
            split = _infer_hf_split_from_lmdb_path(lmdb_path)
            hf_ds = _get_hf_dataset(split)
            hf_img = hf_ds[int(index)]["image"]
            if not isinstance(hf_img, Image.Image):
                hf_img = Image.fromarray(np.asarray(hf_img))
            hf_img = hf_img.convert("RGB")
            return ImageOps.fit(hf_img, (tile_size, tile_size), method=Image.Resampling.BICUBIC)
        except Exception:
            pass

    # Placeholder tile when image is missing
    tile = Image.new("RGB", (tile_size, tile_size), (235, 235, 235))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, tile_size - 1, tile_size - 1), outline=(180, 180, 180), width=1)
    draw.text((8, 8), f"idx={index}", fill=(60, 60, 60))
    return tile


def draw_section(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    left_w: int,
    tile: int,
    gap: int,
    model_title: str,
    latent_idx: int,
    active_count: int,
    selected: Dict[str, List[Dict]],
    lmdb_path: str,
    image_root: Optional[str],
    use_hf_fallback: bool,
    title_font: ImageFont.ImageFont,
):
    ncols = 10

    title_bbox = draw.textbbox((0, 0), model_title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    grid_w = 10 * tile + 9 * gap
    title_x = x0 + left_w + (grid_w - title_w) // 2
    draw.text((title_x, y0 + 2), model_title, fill=(20, 20, 20), font=title_font)

    row_names = ["Top 10", "Mid 10", "Bottom 10"]
    keys = ["top", "mid", "bottom"]

    for ridx, (row_name, row_key) in enumerate(zip(row_names, keys)):
        row_y = y0 + 20 + ridx * (tile + gap)

        items = selected[row_key]
        for cidx in range(ncols):
            tile_x = x0 + left_w + cidx * (tile + gap)
            if cidx < len(items):
                item = items[cidx]
                tile_img = load_image_for_sample(
                    raw_path=item["path"],
                    index=item["index"],
                    tile_size=tile,
                    lmdb_path=lmdb_path,
                    image_root=image_root,
                    use_hf_fallback=use_hf_fallback,
                )
            else:
                tile_img = Image.new("RGB", (tile, tile), (225, 225, 225))
                tile_draw = ImageDraw.Draw(tile_img)
                tile_draw.rectangle((0, 0, tile - 1, tile - 1), outline=(170, 170, 170), width=1)
            canvas.paste(tile_img, (tile_x, row_y))


def build_figure(
    output_path: str,
    batch_model_payload: Dict,
    dual_model_payload: Dict,
    lmdb_path: str,
    image_root: Optional[str],
    use_hf_fallback: bool,
    tile_size: int,
    tile_gap: int,
    dpi: int,
    font_size: int,
    title_font_size: int,
):
    ncols = 10
    left_w = 50
    margin = 20
    section_gap = 0
    title_h = 26
    row_h = tile_size

    right_label_w = 34
    width = margin * 2 + left_w + ncols * tile_size + (ncols - 1) * tile_gap + right_label_w
    section_h = title_h + 3 * row_h
    height = margin * 2 + section_h * 2 + section_gap

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y_top = margin
    y_bottom = margin + section_h + section_gap

    def load_font(size: int) -> ImageFont.ImageFont:
        try:
            # DejaVu Sans as shipped with matplotlib, so no system font path is assumed.
            from matplotlib import font_manager
            return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
        except Exception:
            return ImageFont.load_default()

    left_font = load_font(font_size)
    title_font = load_font(title_font_size)

    def format_monoscore(value: Optional[float]) -> str:
        return f"{value:.3f}" if value is not None else "n/a"

    top_left_text = [
        f"Dim: {batch_model_payload['latent_idx']}",
        f"Monoscore: {format_monoscore(batch_model_payload.get('monoscore'))}",
    ]
    bottom_left_text = [
        f"Dim: {dual_model_payload['latent_idx']}",
        f"Monoscore: {format_monoscore(dual_model_payload.get('monoscore'))}",
    ]

    def create_rotated_text_multi(texts, font, spacing=5):
        text_bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), "Ag", font=font)
        line_h = text_bbox[3] - text_bbox[1]
        max_w = 0
        for text in texts:
            bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
            max_w = max(max_w, bbox[2] - bbox[0])
        total_h = len(texts) * line_h + spacing * (len(texts) - 1) + 10
        img = Image.new("RGB", (max_w + 20, total_h), (255, 255, 255))
        tmp_draw = ImageDraw.Draw(img)
        y = 5
        for text in texts:
            bbox = tmp_draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            x = (max_w - text_w) // 2 + 10
            tmp_draw.text((x, y), text, fill=(20, 20, 20), font=font)
            y += line_h + spacing
        return img.rotate(90, expand=True)

    def create_rotated_text_single(text, font):
        bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        img = Image.new("RGB", (text_w + 10, text_h + 10), (255, 255, 255))
        ImageDraw.Draw(img).text((5, 5), text, fill=(20, 20, 20), font=font)
        return img.rotate(90, expand=True)

    rotated_left_top = create_rotated_text_multi(top_left_text, left_font, spacing=6)
    rotated_left_bottom = create_rotated_text_multi(bottom_left_text, left_font, spacing=6)
    right_labels = [
        create_rotated_text_single("Top 10", left_font),
        create_rotated_text_single("Mid 10", left_font),
        create_rotated_text_single("Bottom 10", left_font),
    ]

    draw_section(
        canvas=canvas,
        draw=draw,
        x0=margin,
        y0=y_top,
        left_w=left_w,
        tile=tile_size,
        gap=tile_gap,
        model_title=batch_model_payload["title"],
        latent_idx=batch_model_payload["latent_idx"],
        active_count=batch_model_payload["active_count"],
        selected=batch_model_payload["selected"],
        lmdb_path=lmdb_path,
        image_root=image_root,
        use_hf_fallback=use_hf_fallback,
        title_font=title_font,
    )

    # Left-side rotated labels, matching the reference layout.
    canvas.paste(rotated_left_top, (margin, y_top + title_h + (3 * row_h - rotated_left_top.height) // 2))

    draw_section(
        canvas=canvas,
        draw=draw,
        x0=margin,
        y0=y_bottom,
        left_w=left_w,
        tile=tile_size,
        gap=tile_gap,
        model_title=dual_model_payload["title"],
        latent_idx=dual_model_payload["latent_idx"],
        active_count=dual_model_payload["active_count"],
        selected=dual_model_payload["selected"],
        lmdb_path=lmdb_path,
        image_root=image_root,
        use_hf_fallback=use_hf_fallback,
        title_font=title_font,
    )

    canvas.paste(rotated_left_bottom, (margin, y_bottom + title_h + (3 * row_h - rotated_left_bottom.height) // 2))

    right_x = margin + left_w + ncols * tile_size + (ncols - 1) * tile_gap + 0
    row_centers_top = [y_top + title_h + row_h * 0.5 + i * row_h for i in range(3)]
    row_centers_bottom = [y_bottom + title_h + row_h * 0.5 + i * row_h for i in range(3)]
    for label, center_y in zip(right_labels, row_centers_top):
        canvas.paste(label, (right_x, int(center_y - label.height / 2)))
    for label, center_y in zip(right_labels, row_centers_bottom):
        canvas.paste(label, (right_x, int(center_y - label.height / 2)))

    footer = "Activation Strength: High -> Low"
    footer_bbox = draw.textbbox((0, 0), footer, font=left_font)
    footer_w = footer_bbox[2] - footer_bbox[0]
    footer_x = (width - footer_w) // 2
    footer_y = height - margin + 2
    draw.text((footer_x, footer_y), footer, fill=(20, 20, 20), font=left_font)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, dpi=(dpi, dpi))


def to_records(indices: Sequence[int], values: np.ndarray, paths: List[str]) -> List[Dict]:
    out = []
    for idx in indices:
        out.append(
            {
                "index": int(idx),
                "activation": float(values[idx]),
                "path": str(paths[idx]) if idx < len(paths) else str(idx),
            }
        )
    return out


def _is_na_score(value: Optional[float]) -> bool:
    if value is None:
        return True
    try:
        return not np.isfinite(float(value))
    except Exception:
        return True


def prepare_model_payload(
    model,
    dataset: LMDBFeatureDataset,
    latent_idx: int,
    device: torch.device,
    args: argparse.Namespace,
    title: str,
    monoscore: Optional[float] = None,
):
    values, paths, active_count = collect_latent_activations(
        model=model,
        dataset=dataset,
        latent_idx=latent_idx,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )

    # Restrict selection to strictly positive activations only.
    positive_indices = np.flatnonzero(values > 0)
    if positive_indices.size > 0:
        positive_values = values[positive_indices]
        local_picks = pick_top_mid_bottom_indices(positive_values, args.k)
        picks = {
            "top": positive_indices[local_picks["top"]],
            "mid": positive_indices[local_picks["mid"]],
            "bottom": positive_indices[local_picks["bottom"]],
        }
    else:
        empty = np.array([], dtype=np.int64)
        picks = {"top": empty, "mid": empty, "bottom": empty}

    selected = {
        "top": to_records(picks["top"], values, paths),
        "mid": to_records(picks["mid"], values, paths),
        "bottom": to_records(picks["bottom"], values, paths),
    }

    return {
        "title": title,
        "latent_idx": int(latent_idx),
        "monoscore": None if monoscore is None else float(monoscore),
        "active_count": int(active_count),
        "selected": selected,
    }


@torch.no_grad()
def compute_monoscores_all_latents(
    model,
    dataset: LMDBFeatureDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute per-latent monosemanticity scores using the streaming formula."""
    n_total = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    if n_total <= 1:
        raise ValueError("Need at least 2 samples for monosemanticity ranking")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Pass 1: activation min/max for per-latent normalization.
    min_vals = None
    max_vals = None
    processed = 0

    for batch in loader:
        features, _ = unpack_features_and_paths(batch, 0)
        x = features.to(device, non_blocking=True).float()
        _, latents, _ = model(x)

        remain = n_total - processed
        if remain <= 0:
            break
        take = min(latents.shape[0], remain)
        lat = latents[:take]

        bmin = lat.min(dim=0, keepdim=True).values
        bmax = lat.max(dim=0, keepdim=True).values
        if min_vals is None:
            min_vals = bmin
            max_vals = bmax
        else:
            min_vals = torch.minimum(min_vals, bmin)
            max_vals = torch.maximum(max_vals, bmax)

        processed += take
        if processed >= n_total:
            break

    if min_vals is None or max_vals is None:
        raise RuntimeError("Failed to compute activation statistics")

    # Pass 2: stream accumulators.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    u = None
    v = None
    w = None
    processed = 0

    for batch in loader:
        features, _ = unpack_features_and_paths(batch, 0)
        x = features.to(device, non_blocking=True).float()
        _, latents, _ = model(x)

        remain = n_total - processed
        if remain <= 0:
            break
        take = min(latents.shape[0], remain)

        x = x[:take]
        latents = latents[:take]
        latents = (latents - min_vals) / (max_vals - min_vals + eps)

        p = torch.nn.functional.normalize(x, p=2, dim=1, eps=eps)
        a32 = latents.float()
        p32 = p.float()

        if u is None:
            num_latents = a32.shape[1]
            feature_dim = p32.shape[1]
            u = torch.zeros(num_latents, device=device, dtype=torch.float32)
            v = torch.zeros(num_latents, device=device, dtype=torch.float32)
            w = torch.zeros(feature_dim, num_latents, device=device, dtype=torch.float32)

        u += a32.sum(dim=0)
        v += (a32 * a32).sum(dim=0)
        w += p32.T @ a32

        processed += take
        if processed >= n_total:
            break

    if u is None or v is None or w is None:
        raise RuntimeError("Failed to compute monosemanticity accumulators")

    q = (w * w).sum(dim=0)
    num = 0.5 * (q - v)
    den = 0.5 * (u * u - v)
    scores = torch.where(den > 1e-8, num / (den + 1e-12), torch.zeros_like(den))
    return scores.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def compute_class_purity_all_latents(
    model,
    dataset: LMDBFeatureDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """Compute per-latent class purity from positive activations.

    purity[m] = max_c count(y=c and a_m>0) / count(a_m>0)
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    n_total = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    if n_total <= 0:
        raise ValueError("Need at least 1 sample for class purity")

    pos_counts = None
    per_class_counts: Dict[int, torch.Tensor] = {}
    processed = 0

    for batch in loader:
        features, labels, _ = unpack_features_labels_paths(batch, processed)
        if labels is None:
            raise RuntimeError("Class labels are required for class purity, but dataset batch has no labels")

        remain = n_total - processed
        if remain <= 0:
            break

        x = features.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        take = min(x.shape[0], remain)
        x = x[:take]
        labels = labels[:take]

        _, latents, _ = model(x)
        is_pos = (latents > 0)

        if pos_counts is None:
            pos_counts = torch.zeros(is_pos.shape[1], device=device, dtype=torch.float32)

        pos_counts += is_pos.float().sum(dim=0)

        unique_classes = labels.unique()
        for cls in unique_classes.tolist():
            cls_mask = (labels == cls).unsqueeze(1)
            cls_pos = (is_pos & cls_mask).float().sum(dim=0)
            if cls not in per_class_counts:
                per_class_counts[cls] = torch.zeros_like(pos_counts)
            per_class_counts[cls] += cls_pos

        processed += take
        if processed >= n_total:
            break

    if pos_counts is None:
        raise RuntimeError("Failed to compute class purity counts")

    max_class_count = torch.zeros_like(pos_counts)
    for cls_counts in per_class_counts.values():
        max_class_count = torch.maximum(max_class_count, cls_counts)

    purity = torch.where(pos_counts > 0, max_class_count / pos_counts, torch.zeros_like(pos_counts))
    return purity.detach().cpu().numpy().astype(np.float32)


def plot_monoscore_comparison(
    score_batch: np.ndarray,
    score_dual: np.ndarray,
    output_path: str,
):
    """Plot sorted monosemanticity score curves for BatchTopK vs Dual-BatchTopK."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to plot monoscore curves. Install with: pip install matplotlib"
        ) from exc

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9

    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=300)

    curves = [
        ("BatchTopK", np.asarray(score_batch, dtype=np.float32), "#1f77b4"),
        ("Dual-BatchTopK", np.asarray(score_dual, dtype=np.float32), "#d62728"),
    ]

    for label, scores, color in curves:
        sorted_scores = np.sort(scores)[::-1]
        x_normalized = np.linspace(0.0, 1.0, sorted_scores.shape[0])
        ax.plot(
            x_normalized,
            sorted_scores,
            linewidth=1.8,
            label=label,
            color=color,
            alpha=0.95,
        )

    ax.set_xlabel("Normalized Index", fontweight="normal")
    ax.set_ylabel("Monosemanticity Score", fontweight="normal")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
        loc="upper right",
        fontsize=COMPARISON_LEGEND_FONT_SIZE,
    )

    fig.tight_layout(pad=0.35)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved monoscore plot to: {out}")


def plot_class_purity_comparison(
    purity_batch: np.ndarray,
    purity_dual: np.ndarray,
    output_path: str,
):
    """Plot sorted class purity curves for BatchTopK vs Dual-BatchTopK."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to plot class purity curves. Install with: pip install matplotlib"
        ) from exc

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9

    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=300)

    curves = [
        ("BatchTopK", np.asarray(purity_batch, dtype=np.float32), "#1f77b4"),
        ("Dual-BatchTopK", np.asarray(purity_dual, dtype=np.float32), "#d62728"),
    ]

    for label, scores, color in curves:
        sorted_scores = np.sort(scores)[::-1]
        x_normalized = np.linspace(0.0, 1.0, sorted_scores.shape[0])
        ax.plot(
            x_normalized,
            sorted_scores,
            linewidth=1.8,
            label=label,
            color=color,
            alpha=0.95,
        )

    ax.set_xlabel("Normalized Index", fontweight="normal")
    ax.set_ylabel("Class Purity", fontweight="normal")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
        loc="upper right",
        fontsize=COMPARISON_LEGEND_FONT_SIZE,
    )

    fig.tight_layout(pad=0.35)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved class purity plot to: {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot Top/Mid/Bottom 15 images for sparse latent activations")

    parser.add_argument("--batch_topk_ckpt", type=str, required=True, help="Path to BatchTopK checkpoint")
    parser.add_argument("--dual_ckpt", type=str, required=True, help="Path to DualSAE checkpoint")

    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "open_images"])
    parser.add_argument("--extractor", type=str, default="clip", choices=["clip", "vit", "sigclip"])
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--lmdb_path", type=str, default=None, help="Override LMDB path")

    parser.add_argument("--latent_idx", type=int, default=None, help="Sparse latent index to visualize (default: auto-select from BatchTopK)")
    parser.add_argument("--k", type=int, default=10, help="Number of images for each group (top/mid/bottom)")
    parser.add_argument("--num_latents", type=int, default=1, help="Number of rank positions to render; use <=0 for all")

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--image_root", type=str, default=None, help="Optional root to prepend to LMDB sample paths")
    parser.add_argument("--disable_hf_fallback", action="store_true", help="Disable ImageNet HF fallback when local image path is missing")

    parser.add_argument("--tile_size", type=int, default=92, help="Tile size in pixels")
    parser.add_argument("--tile_gap", type=int, default=0, help="Gap between tiles")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI metadata for saved figures")
    parser.add_argument("--font_size", type=int, default=14, help="Font size for labels and footer")
    parser.add_argument("--title_font_size", type=int, default=14, help="Font size for section titles")

    parser.add_argument("--min_active_samples", type=int, default=10, help="Minimum number of active samples required for a dimension")

    parser.add_argument("--monoscore_plot_path", type=str, default=None, help="Optional output path for BatchTopK vs Dual-BatchTopK monoscore plot")
    parser.add_argument("--class_purity_plot_path", type=str, default=None, help="Optional output path for BatchTopK vs Dual-BatchTopK class purity plot")

    parser.add_argument("--output", type=str,
                        default=os.path.join(os.environ.get("ABL_ROOT", "./outputs/ablations"),
                                             "figures", "matched_latents", "latent_grid.png"))
    parser.add_argument("--output_json", type=str, default=None, help="Optional metadata JSON path")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lmdb_path = resolve_lmdb_path(args)

    print(f"Loading dataset from {lmdb_path}")
    dataset = LMDBFeatureDataset(
        lmdb_path,
        return_index=False,
        return_class_idx=True,
        return_path=True,
        verbose=True,
    )

    print(f"Loading BatchTopK model from {args.batch_topk_ckpt}")
    batch_model = load_batch_topk_model(args.batch_topk_ckpt, device)

    print(f"Loading Dual model from {args.dual_ckpt}")
    dual_model = load_dual_model(args.dual_ckpt, device)

    dual_mode = getattr(dual_model, "sparsity_mode", "unknown")
    if dual_mode != "batch_topk":
        print(
            f"Warning: dual model sparsity_mode is '{dual_mode}', not 'batch_topk'. "
            "Continuing anyway."
        )

    score_batch = None
    score_dual = None
    purity_batch = None
    purity_dual = None
    ranked_batch = None
    ranked_dual = None

    if args.latent_idx is None:
        print("Computing per-latent monoscores for BatchTopK...")
        score_batch = compute_monoscores_all_latents(
            model=batch_model,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        )
        print("Computing per-latent monoscores for Dual-BatchTopK...")
        score_dual = compute_monoscores_all_latents(
            model=dual_model,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        )

        print("Computing per-latent class purity for BatchTopK...")
        purity_batch = compute_class_purity_all_latents(
            model=batch_model,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        )
        print("Computing per-latent class purity for Dual-BatchTopK...")
        purity_dual = compute_class_purity_all_latents(
            model=dual_model,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        )

        ranked_batch = np.argsort(score_batch)[::-1].tolist()
        ranked_dual = np.argsort(score_dual)[::-1].tolist()

        num_pairs = min(len(ranked_batch), len(ranked_dual))
        if args.num_latents > 0:
            num_pairs = min(num_pairs, args.num_latents)
    else:
        num_pairs = 1

    output_path = Path(args.output)
    if num_pairs == 1:
        output_dir = output_path.parent
        output_prefix = output_path.stem
    else:
        output_dir = output_path.parent / output_path.stem
        output_prefix = "latent_grid"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_payload = {
        "lmdb_path": lmdb_path,
        "batch_topk_checkpoint": args.batch_topk_ckpt,
        "dual_checkpoint": args.dual_ckpt,
        "ranking_mode": "paired_by_model_rank" if args.latent_idx is None else "single_latent",
        "num_latents_requested": args.num_latents,
        "min_active_samples_required": args.min_active_samples,
        "monoscore_plot": None,
        "class_purity_plot": None,
        "results": [],
    }

    monoscore_counts = {
        "batch_topk": {"na": 0, "not_na": 0},
        "dual_batch_topk": {"na": 0, "not_na": 0},
    }

    if score_batch is not None and score_dual is not None:
        np.savez(output_dir / f"{output_prefix}_curves.npz", score_batch=score_batch,
                 score_dual=score_dual, purity_batch=purity_batch, purity_dual=purity_dual)
        monoscore_plot_path = args.monoscore_plot_path
        if monoscore_plot_path is None:
            monoscore_plot_path = str(output_dir / f"{output_prefix}_monoscore_comparison.pdf")
        plot_monoscore_comparison(
            score_batch=score_batch,
            score_dual=score_dual,
            output_path=monoscore_plot_path,
        )
        run_payload["monoscore_plot"] = monoscore_plot_path

    if purity_batch is not None and purity_dual is not None:
        class_purity_plot_path = args.class_purity_plot_path
        if class_purity_plot_path is None:
            class_purity_plot_path = str(output_dir / f"{output_prefix}_class_purity_comparison.pdf")
        plot_class_purity_comparison(
            purity_batch=purity_batch,
            purity_dual=purity_dual,
            output_path=class_purity_plot_path,
        )
        run_payload["class_purity_plot"] = class_purity_plot_path

    rendered_count = 0
    for rank in range(1, num_pairs + 1):
        if args.latent_idx is not None:
            batch_latent_idx = int(args.latent_idx)
            dual_latent_idx = int(args.latent_idx)
        else:
            batch_latent_idx = int(ranked_batch[rank - 1])
            dual_latent_idx = int(ranked_dual[rank - 1])

        batch_monoscore_value = None if score_batch is None else float(score_batch[batch_latent_idx])
        dual_monoscore_value = None if score_dual is None else float(score_dual[dual_latent_idx])

        if _is_na_score(batch_monoscore_value):
            monoscore_counts["batch_topk"]["na"] += 1
            batch_monoscore_value = None
        else:
            monoscore_counts["batch_topk"]["not_na"] += 1

        if _is_na_score(dual_monoscore_value):
            monoscore_counts["dual_batch_topk"]["na"] += 1
            dual_monoscore_value = None
        else:
            monoscore_counts["dual_batch_topk"]["not_na"] += 1

        print(
            f"Processing rank {rank}/{num_pairs}: "
            f"batch latent {batch_latent_idx}, dual latent {dual_latent_idx}"
        )

        batch_payload = prepare_model_payload(
            model=batch_model,
            dataset=dataset,
            latent_idx=batch_latent_idx,
            device=device,
            args=args,
            title="BatchTopK",
            monoscore=batch_monoscore_value,
        )

        dual_payload = prepare_model_payload(
            model=dual_model,
            dataset=dataset,
            latent_idx=dual_latent_idx,
            device=device,
            args=args,
            title="Dual-BatchTopK",
            monoscore=dual_monoscore_value,
        )

        print(
            "  Monoscore counts so far | "
            f"BatchTopK NA={monoscore_counts['batch_topk']['na']} "
            f"not-NA={monoscore_counts['batch_topk']['not_na']} | "
            f"Dual-BatchTopK NA={monoscore_counts['dual_batch_topk']['na']} "
            f"not-NA={monoscore_counts['dual_batch_topk']['not_na']}"
        )

        # Check minimum active samples threshold
        batch_active = batch_payload["active_count"]
        dual_active = dual_payload["active_count"]
        min_req = args.min_active_samples

        if batch_active < min_req:
            # print(
            #     f"  Skipping: BatchTopK has {batch_active} active samples (required: {min_req})"
            # )
            continue
        if dual_active < min_req:
            print(
                f"  Skipping: Dual has {dual_active} active samples (required: {min_req})"
            )
            continue

        rendered_count += 1
        fig_path = output_dir / (
            f"{output_prefix}_rank_{rendered_count:05d}"
            f"_batch_latent_{batch_latent_idx}"
            f"_dual_latent_{dual_latent_idx}.png"
        )
        meta_path = fig_path.with_suffix(".json")

        print(f"  Rendering figure (batch: {batch_active} active, dual: {dual_active} active)")

        build_figure(
            output_path=str(fig_path),
            batch_model_payload=batch_payload,
            dual_model_payload=dual_payload,
            lmdb_path=lmdb_path,
            image_root=args.image_root,
            use_hf_fallback=not args.disable_hf_fallback,
            tile_size=args.tile_size,
            tile_gap=args.tile_gap,
            dpi=args.dpi,
            font_size=args.font_size,
            title_font_size=args.title_font_size,
        )

        item = {
            "batch_rank": int(rank),
            "dual_rank": int(rank),
            "batch_latent_idx": int(batch_latent_idx),
            "dual_latent_idx": int(dual_latent_idx),
            "batch_active_count": int(batch_active),
            "dual_active_count": int(dual_active),
            "figure": str(fig_path),
            "batch_topk": batch_payload,
            "dual_batch_topk": dual_payload,
        }
        if score_batch is not None:
            item["batch_topk_monoscore"] = batch_monoscore_value
        if score_dual is not None:
            item["dual_monoscore"] = dual_monoscore_value
        if purity_batch is not None:
            item["batch_topk_class_purity"] = float(purity_batch[batch_latent_idx])
        if purity_dual is not None:
            item["dual_class_purity"] = float(purity_dual[dual_latent_idx])

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)

        run_payload["results"].append(item)

    output_json = args.output_json
    if output_json is None:
        output_json = str((output_dir / f"{output_prefix}_summary.json"))

    with open(output_json, "w", encoding="utf-8") as f:
        run_payload["monoscore_counts"] = monoscore_counts
        json.dump(run_payload, f, indent=2)

    print(
        "Final monoscore counts | "
        f"BatchTopK NA={monoscore_counts['batch_topk']['na']} "
        f"not-NA={monoscore_counts['batch_topk']['not_na']} | "
        f"Dual-BatchTopK NA={monoscore_counts['dual_batch_topk']['na']} "
        f"not-NA={monoscore_counts['dual_batch_topk']['not_na']}"
    )

    if rendered_count > 0:
        print(f"\nSaved {rendered_count} figure(s) to: {fig_path}")
    else:
        print("\nSaved 0 figure(s). No dimensions met the rendering criteria.")
    print(f"Saved run summary: {output_json}")
    if rendered_count < num_pairs:
        print(f"Skipped {num_pairs - rendered_count} dimension(s) with <{args.min_active_samples} active samples")


if __name__ == "__main__":
    main()
