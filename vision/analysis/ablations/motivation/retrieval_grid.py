"""Motivation figure, right panel: example retrievals for one query image using the SAE
reconstruction x_hat (top block) and the residual r = x - x_hat (bottom block).

For each query index, the 29 most cosine-similar validation images (query excluded) are retrieved
from the L2-normalised x_hat bank and from the L2-normalised residual bank of one SAE checkpoint,
and drawn as two 3 x 10 blocks with the query (yellow border) in the first tile of each block.
The figure uses ImageNet-1K validation index 1 (--queries 1) with the BatchTopK SAE on CLIP
features, epoch-50 checkpoint (autoencoder_final.pt).

Images: --image-root/<path stored in the LMDB> when given, otherwise the ImageNet-1K LMDB index is
used to fetch the image from the Hugging Face `ILSVRC/imagenet-1k` dataset, which must be in the
local cache (scripts/ablations/prepare_assets.sh).
"""
import os
import sys
import json
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt

_VISION = Path(__file__).resolve().parents[3]
if str(_VISION) not in sys.path:
    sys.path.insert(0, str(_VISION))

from dataset.lmdb_features import LMDBFeatureDataset
from models.batch_topk_sae import BatchTopKSAE
from models.topk_sae import TopKSAE, TopK
from models.jumprelu_sae import JumpReLUSAE
from models.vanilla_sae import VanillaSAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_HF_DATASET_CACHE = {}


def load_autoencoder(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cfg = checkpoint.get("config", {})

    if "topk_k" in cfg and "d_model" in cfg:
        model = BatchTopKSAE({
            "act_size": cfg["d_model"],
            "dict_size": cfg["n_latents"],
            "top_k": cfg["topk_k"],
            "l1_coeff": 0.0,
            "top_k_aux": cfg.get("auxk_k") or 512,
            "aux_penalty": cfg.get("auxk_coef", 0.0),
            "n_batches_to_dead": cfg.get("dead_steps_threshold", 1000) // max(cfg.get("batch_size", 1), 1),
            "seed": cfg.get("seed", 0),
            "device": str(DEVICE),
            "dtype": torch.float32,
            "input_unit_norm": cfg.get("normalize", False),
        })
        model.load_state_dict(state_dict)
        return model.to(DEVICE).eval()

    if "encoder.weight" in state_dict:
        n_latents, d_model = state_dict["encoder.weight"].shape
        activation_state = state_dict.get("activation_state_dict", {})
        activation = TopK(k=activation_state.get("k", 64))
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
        return model.to(DEVICE).eval()

    if "W_enc" in state_dict:
        act_size, dict_size = state_dict["W_enc"].shape
        model_cfg = {
            "act_size": act_size,
            "dict_size": dict_size,
            "l1_coeff": 0.0,
            "n_batches_to_dead": 1000,
            "seed": cfg.get("seed", 0),
            "device": str(DEVICE),
            "dtype": torch.float32,
            "input_unit_norm": True,
        }
        if "jumprelu.log_threshold" in state_dict:
            model_cfg["bandwidth"] = cfg.get("bandwidth", 0.001)
            model = JumpReLUSAE(model_cfg)
        else:
            model = VanillaSAE(model_cfg)
        model.load_state_dict(state_dict, strict=True)
        return model.to(DEVICE).eval()

    raise ValueError(f"Unrecognized checkpoint format: {checkpoint_path}")


def load_feature_dataset(lmdb_path):
    return LMDBFeatureDataset(
        lmdb_path,
        return_index=True,
        return_class_idx=True,
        return_path=True,
        verbose=True,
    )


def unpack_item(item, idx):
    if isinstance(item, tuple):
        if len(item) == 3:
            feat, class_idx, sample_path = item
        elif len(item) == 2:
            feat, class_idx = item
            sample_path = str(idx)
        else:
            feat = item[0]
            class_idx = None
            sample_path = str(idx)
    else:
        feat = item
        class_idx = None
        sample_path = str(idx)

    feat = torch.as_tensor(feat, dtype=torch.float32)
    class_idx = int(class_idx) if class_idx is not None else None
    return feat, class_idx, str(sample_path)


def compute_residual_bank(autoencoder, dataset, batch_size=256):
    residuals = []
    class_ids = []
    sample_paths = []

    autoencoder.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            batch_feats = []
            batch_class_ids = []
            batch_paths = []

            for idx in range(start, end):
                feat, class_idx, sample_path = unpack_item(dataset[idx], idx)
                batch_feats.append(feat)
                batch_class_ids.append(-1 if class_idx is None else class_idx)
                batch_paths.append(sample_path)

            batch_feats = torch.stack(batch_feats, dim=0).to(DEVICE)
            _, _, recons = autoencoder(batch_feats)
            residual = batch_feats - recons
            residual = F.normalize(residual, p=2, dim=-1)

            residuals.append(residual.cpu())
            class_ids.extend(batch_class_ids)
            sample_paths.extend(batch_paths)

    return torch.cat(residuals, dim=0), class_ids, sample_paths


def compute_reconstruction_bank(autoencoder, dataset, batch_size=256):
    reconstructions = []
    class_ids = []
    sample_paths = []

    autoencoder.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            batch_feats = []
            batch_class_ids = []
            batch_paths = []

            for idx in range(start, end):
                feat, class_idx, sample_path = unpack_item(dataset[idx], idx)
                batch_feats.append(feat)
                batch_class_ids.append(-1 if class_idx is None else class_idx)
                batch_paths.append(sample_path)

            batch_feats = torch.stack(batch_feats, dim=0).to(DEVICE)
            _, _, recons = autoencoder(batch_feats)
            recons = F.normalize(recons, p=2, dim=-1)

            reconstructions.append(recons.cpu())
            class_ids.extend(batch_class_ids)
            sample_paths.extend(batch_paths)

    return torch.cat(reconstructions, dim=0), class_ids, sample_paths


def resolve_query_index(dataset, query_index=0, query_path=None):
    if query_path is None:
        return query_index

    query_path = str(query_path)
    for idx in range(len(dataset)):
        _, _, sample_path = unpack_item(dataset[idx], idx)
        if sample_path == query_path or sample_path.endswith(query_path) or query_path in sample_path:
            return idx

    raise ValueError(f"Could not find a sample path matching: {query_path}")


def query_topk_similar_images(residual_bank, query_residual, sample_paths, class_ids=None, topk=20, exclude_query_index=None):
    residual_bank = F.normalize(residual_bank, p=2, dim=-1)
    query_residual = F.normalize(query_residual, p=2, dim=-1)
    similarities = residual_bank @ query_residual

    if exclude_query_index is not None:
        similarities[exclude_query_index] = -float("inf")

    topk = min(topk, similarities.numel())
    values, indices = torch.topk(similarities, k=topk)

    rows = []
    for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        rows.append({
            "rank": rank,
            "index": idx,
            "similarity": score,
            "class_id": None if class_ids is None else class_ids[idx],
            "path": sample_paths[idx],
        })
    return rows


def _infer_hf_split_from_lmdb_path(lmdb_path):
    lmdb_path_lower = str(lmdb_path).lower()
    if "train" in lmdb_path_lower:
        return "train"
    return "validation"


def _get_hf_dataset(split):
    if split not in _HF_DATASET_CACHE:
        try:
            load_dataset = importlib.import_module("datasets").load_dataset
        except Exception as exc:
            raise RuntimeError(
                "datasets package is required for ImageNet fallback rendering. "
                "Install it with: pip install datasets"
            ) from exc
        _HF_DATASET_CACHE[split] = load_dataset("ILSVRC/imagenet-1k", split=split)
    return _HF_DATASET_CACHE[split]


def _get_label_name(class_id, lmdb_path=None):
    if class_id is None or int(class_id) < 0:
        return "unknown"

    if lmdb_path is None:
        return str(class_id)

    try:
        split = _infer_hf_split_from_lmdb_path(lmdb_path)
        hf_ds = _get_hf_dataset(split)
        label_feature = hf_ds.features.get("label")
        if label_feature is not None and hasattr(label_feature, "names"):
            names = label_feature.names
            if 0 <= int(class_id) < len(names):
                return names[int(class_id)]
    except Exception:
        pass

    return str(class_id)


IMAGE_ROOT = None  # set from --image-root in main()


def _try_load_image(row, lmdb_path=None):
    path = row["path"]
    if IMAGE_ROOT is not None:
        path = str(Path(IMAGE_ROOT) / path)
    if Path(path).exists():
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            print(f"Warning: Failed to load image from path: {path}")
            pass

    if lmdb_path is not None:
        # try:
        split = _infer_hf_split_from_lmdb_path(lmdb_path)
        hf_ds = _get_hf_dataset(split)
        img = hf_ds[int(row["index"])]["image"]
        return img.convert("RGB") if hasattr(img, "convert") else img
        # except Exception:
        #     print(f"Warning: Failed to load image from HF dataset for index {row['index']} in split {split}")
        #     pass

    return None


def _center_crop_square(image, tile_size=256):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return ImageOps.fit(
        image,
        (tile_size, tile_size),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )


def show_topk_images(rows, query_row, ncols=5, lmdb_path=None, tile_size=256):
    import matplotlib.gridspec as gridspec
    
    n_results = len(rows)
    n_cols_results = ncols - 1  # Reserve 1 column for query
    n_rows_results = int(np.ceil(n_results / n_cols_results))
    
    fig_w = (tile_size * ncols) / 100.0
    fig_h = (tile_size * n_rows_results) / 100.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    
    gs = gridspec.GridSpec(n_rows_results, ncols, figure=fig, hspace=0, wspace=0)
    
    ax_query = fig.add_subplot(gs[:, 0])
    ax_query.axis("off")
    image_query = _try_load_image(query_row, lmdb_path=lmdb_path)
    if image_query is not None:
        image_query = _center_crop_square(image_query, tile_size=tile_size)
        ax_query.imshow(image_query)
    caption_query = "QUERY\nsim=1.000"
    ax_query.text(
        0.01,
        0.01,
        caption_query,
        transform=ax_query.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 2, "edgecolor": "none"},
    )
    ax_query.set_aspect("equal")
    
    for idx, row in enumerate(rows):
        row_idx = idx // n_cols_results
        col_idx = (idx % n_cols_results) + 1
        
        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.axis("off")
        
        image = _try_load_image(row, lmdb_path=lmdb_path)
        if image is not None:
            image = _center_crop_square(image, tile_size=tile_size)
            ax.imshow(image)
        else:
            ax.set_facecolor("#f2f2f2")
            ax.text(
                0.5,
                0.5,
                f"idx={row['index']}\nclass={row['class_id']}\n{row['path']}",
                ha="center",
                va="center",
                wrap=True,
                fontsize=9,
            )
        
        caption = f"#{row['rank']} | sim={row['similarity']:.3f}"
        ax.text(
            0.01,
            0.01,
            caption,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2, "edgecolor": "none"},
        )
        ax.set_aspect("equal")
    
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    return fig


def show_combined_results(rows_recon, rows_residual, query_row, ncols=10, lmdb_path=None, tile_size=96, max_results=29):
    nrows = 6
    section_gap = max(8, tile_size // 8)
    recon_query_pos = (0, 0)
    residual_query_pos = (3, 0)

    recon_positions = [
        (row, col)
        for row in range(0, 3)
        for col in range(ncols)
        if (row, col) != recon_query_pos
    ]
    residual_positions = [
        (row, col)
        for row in range(3, 6)
        for col in range(ncols)
        if (row, col) != residual_query_pos
    ]

    canvas_height = nrows * tile_size + section_gap
    canvas = Image.new("RGBA", (ncols * tile_size, canvas_height), (255, 255, 255, 255))
    query_border_color = 'yellow'

    def y_offset(row):
        return row * tile_size + (section_gap if row >= 3 else 0)

    def paste_tile(image, row, col, border_color=None, border_width=0):
        tile = Image.new("RGBA", (tile_size, tile_size), (245, 245, 245, 255))
        if image is not None:
            tile.paste(_center_crop_square(image, tile_size=tile_size), (0, 0))

        tile_draw = ImageDraw.Draw(tile)
        if border_color is not None and border_width > 0:
            for offset in range(border_width):
                tile_draw.rectangle(
                    [offset, offset, tile_size - 1 - offset, tile_size - 1 - offset],
                    outline=border_color,
                )

        canvas.alpha_composite(tile, (col * tile_size, y_offset(row)))

    def paste_missing(row, col):
        tile = Image.new("RGBA", (tile_size, tile_size), (242, 242, 242, 255))
        canvas.alpha_composite(tile, (col * tile_size, y_offset(row)))

    query_image = _try_load_image(query_row, lmdb_path=lmdb_path)
    paste_tile(query_image, recon_query_pos[0], recon_query_pos[1], border_color=query_border_color, border_width=4)
    paste_tile(query_image, residual_query_pos[0], residual_query_pos[1], border_color=query_border_color, border_width=4)

    for row_data, (row, col) in zip(rows_recon[:max_results], recon_positions):
        image = _try_load_image(row_data, lmdb_path=lmdb_path)
        if image is None:
            paste_missing(row, col)
        else:
            paste_tile(image, row, col)

    for row_data, (row, col) in zip(rows_residual[:max_results], residual_positions):
        image = _try_load_image(row_data, lmdb_path=lmdb_path)
        if image is None:
            paste_missing(row, col)
        else:
            paste_tile(image, row, col)

    return canvas.convert("RGB")


def main():
    import argparse
    global IMAGE_ROOT
    ap = argparse.ArgumentParser(description="x_hat vs. residual retrieval grid for query images")
    ap.add_argument("--checkpoint", required=True,
                    help="SAE checkpoint (the figure: BatchTopK, CLIP, autoencoder_final.pt)")
    ap.add_argument("--lmdb", required=True, help="ImageNet-1K validation LMDB of the same encoder")
    ap.add_argument("--queries", type=int, nargs="+", default=[1], help="validation indices to query")
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--result-dir", default="./outputs/ablations/figures/motivation")
    args = ap.parse_args()
    IMAGE_ROOT = args.image_root
    RESULT_DIR = Path(args.result_dir)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path, lmdb_path = args.checkpoint, args.lmdb

    # Load model and data.
    autoencoder = load_autoencoder(checkpoint_path)
    dataset = load_feature_dataset(lmdb_path)

    # Build normalized banks: residual = x - recons, reconstruction = recons.
    print("Computing residual bank...")
    residual_bank, class_ids, sample_paths = compute_residual_bank(autoencoder, dataset, batch_size=256)
    print("Computing reconstruction bank...")
    recon_bank, _, _ = compute_reconstruction_bank(autoencoder, dataset, batch_size=256)

    for query_index in args.queries:
        print(f"\n=== Processing query {query_index} ===")
    
        # Select the query sample and compute its embeddings.
        resolved_index = resolve_query_index(dataset, query_index=query_index, query_path=None)
        query_feat, query_class_id, query_sample_path = unpack_item(dataset[resolved_index], resolved_index)
        with torch.no_grad():
            query_feat = query_feat.unsqueeze(0).to(DEVICE)
            _, _, query_recons = autoencoder(query_feat)
            query_residual = F.normalize(query_feat - query_recons, p=2, dim=-1).squeeze(0).cpu()
            query_recons_norm = F.normalize(query_recons, p=2, dim=-1).squeeze(0).cpu()

        query_row = {
            "rank": 0,
            "index": resolved_index,
            "similarity": 1.0,
            "class_id": query_class_id,
            "path": query_sample_path,
            "is_query": True,
        }

        # Retrieve top-29 for reconstructions.
        rows_recon = query_topk_similar_images(
            residual_bank=recon_bank,
            query_residual=query_recons_norm,
            sample_paths=sample_paths,
            class_ids=class_ids,
            topk=29,
            exclude_query_index=resolved_index,
        )

        # Retrieve top-29 for residuals.
        rows_residual = query_topk_similar_images(
            residual_bank=residual_bank,
            query_residual=query_residual,
            sample_paths=sample_paths,
            class_ids=class_ids,
            topk=29,
            exclude_query_index=resolved_index,
        )

        # Save combined results (reconstructions + residuals in one file).
        result_base = RESULT_DIR / f"checkpoint_{Path(checkpoint_path).stem}_query_{resolved_index}"
        figure_path = result_base.with_suffix(".png")
        json_path = result_base.with_suffix(".json")

        fig_combined = show_combined_results(rows_recon, rows_residual, query_row=query_row, ncols=10, lmdb_path=lmdb_path, tile_size=96, max_results=29)
        fig_combined.save(figure_path)
        fig_combined.close()

        payload = {
            "checkpoint_path": checkpoint_path,
            "lmdb_path": lmdb_path,
            "query_index": resolved_index,
            "query_path": query_sample_path,
            "query_class_id": query_class_id,
            "query_row": query_row,
            "reconstruction_results": rows_recon,
            "residual_results": rows_residual,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"Saved query {query_index}: {figure_path}")


if __name__ == "__main__":
    main()
