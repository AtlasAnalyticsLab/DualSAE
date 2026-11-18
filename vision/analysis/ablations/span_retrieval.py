"""Effect of the decorrelation loss on branch separation (main-paper retrieval figure).

For the same query image, compares nearest-neighbour retrievals (cosine, full ImageNet-1K validation
set, query excluded) from the Sparse Branch output x^M and the Residual Branch output r_hat of two
Dual-BatchTopK checkpoints that differ in the decorrelation loss: trained without it
(--checkpoint-no-span, gamma = 0) and with it (--checkpoint-with-span). One 6x10 figure and one JSON
per query index in [--query-start, --query-end].

Images: --image-root/<path stored in the LMDB> when given, otherwise the ImageNet-1K LMDB index is
used to fetch the image from the Hugging Face `ILSVRC/imagenet-1k` dataset, which must be in the
local cache (scripts/ablations/prepare_assets.sh).
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

_VISION = Path(__file__).resolve().parents[2]
if str(_VISION) not in sys.path:
    sys.path.insert(0, str(_VISION))

from dataset.lmdb_features import LMDBFeatureDataset
from models.dual_sae import DualSAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_HF_DATASET_CACHE = {}
IMAGE_ROOT = None  # set from --image-root in main()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare DualSAE retrievals (xM and residual/r_hat) with vs without span loss."
    )
    parser.add_argument("--checkpoint-no-span", type=str, required=True)
    parser.add_argument("--checkpoint-with-span", type=str, required=True)
    parser.add_argument("--lmdb-path", type=str, required=True, help="ImageNet-1K validation LMDB")
    parser.add_argument("--image-root", type=str, default=None,
                        help="directory holding the images under the LMDB's stored paths")
    parser.add_argument("--result-dir", type=str, default="./outputs/ablations/figures/span_retrieval")
    parser.add_argument("--query-start", type=int, default=1)
    parser.add_argument("--query-end", type=int, default=9)
    parser.add_argument("--topk", type=int, default=14, help="Top-k neighbors per 3x5 section (excluding query tile).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=96)
    return parser.parse_args()


def load_dual_autoencoder(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    cfg_raw = checkpoint.get("model_cfg", checkpoint.get("config", {}))
    cfg = dict(cfg_raw) if isinstance(cfg_raw, dict) else {}

    if "W_enc" not in state_dict or "poly.0.weight" not in state_dict:
        raise ValueError(f"Checkpoint does not look like a DualSAE model: {checkpoint_path}")

    act_size, dict_size = state_dict["W_enc"].shape
    model_cfg = {
        "act_size": cfg.get("act_size", cfg.get("d_model", act_size)),
        "dict_size": cfg.get("dict_size", cfg.get("n_latents", dict_size)),
        "top_k": cfg.get("top_k", cfg.get("topk_k", 64)),
        "l1_coeff": cfg.get("l1_coeff", cfg.get("l1_coef", 0.0)),
        "poly_hidden": cfg.get("poly_hidden", 512),
        "poly_layers": cfg.get("poly_layers", 1),
        "poly_hidden_dims": cfg.get("poly_hidden_dims", None),
        "sparsity_mode": cfg.get("sparsity_mode", "topk"),
        "n_batches_to_dead": cfg.get("n_batches_to_dead", 1000),
        "seed": cfg.get("seed", 0),
        "device": str(DEVICE),
        "dtype": torch.float32,
        "input_unit_norm": cfg.get("input_unit_norm", cfg.get("normalize", True)),
        "bandwidth": cfg.get("bandwidth", 0.001),
    }

    model = DualSAE(model_cfg)
    model.load_state_dict(state_dict, strict=True)
    return model.to(DEVICE).eval()


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


def resolve_query_index(dataset, query_index=0, query_path=None):
    if query_path is None:
        return query_index

    query_path = str(query_path)
    for idx in range(len(dataset)):
        _, _, sample_path = unpack_item(dataset[idx], idx)
        if sample_path == query_path or sample_path.endswith(query_path) or query_path in sample_path:
            return idx

    raise ValueError(f"Could not find a sample path matching: {query_path}")


def compute_dual_branch_banks(autoencoder, dataset, batch_size=256):
    x_m_bank = []
    residual_bank = []
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
            _ = autoencoder(batch_feats)
            x_m = F.normalize(autoencoder._xM, p=2, dim=-1)
            residual = F.normalize(autoencoder._r_hat, p=2, dim=-1)

            x_m_bank.append(x_m.cpu())
            residual_bank.append(residual.cpu())
            class_ids.extend(batch_class_ids)
            sample_paths.extend(batch_paths)

    return torch.cat(x_m_bank, dim=0), torch.cat(residual_bank, dim=0), class_ids, sample_paths


def query_topk_similar_images(bank, query_vector, sample_paths, class_ids=None, topk=14, exclude_query_index=None):
    bank = F.normalize(bank, p=2, dim=-1)
    query_vector = F.normalize(query_vector, p=2, dim=-1)
    similarities = bank @ query_vector

    if exclude_query_index is not None:
        similarities[exclude_query_index] = -float("inf")

    topk = min(topk, similarities.numel())
    values, indices = torch.topk(similarities, k=topk)

    rows = []
    for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        rows.append(
            {
                "rank": rank,
                "index": idx,
                "similarity": score,
                "class_id": None if class_ids is None else class_ids[idx],
                "path": sample_paths[idx],
            }
        )
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
                "datasets package is required for ImageNet fallback rendering. Install it with: pip install datasets"
            ) from exc
        _HF_DATASET_CACHE[split] = load_dataset("ILSVRC/imagenet-1k", split=split)
    return _HF_DATASET_CACHE[split]


def _try_load_image(row, lmdb_path=None):
    path = row["path"]
    if IMAGE_ROOT is not None:
        path = str(Path(IMAGE_ROOT) / path)
    if Path(path).exists():
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass

    if lmdb_path is not None:
        split = _infer_hf_split_from_lmdb_path(lmdb_path)
        hf_ds = _get_hf_dataset(split)
        img = hf_ds[int(row["index"])]["image"]
        return img.convert("RGB") if hasattr(img, "convert") else img

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


def build_6x10_comparison_figure(
    query_row,
    rows_recon_no_span,
    rows_residual_no_span,
    rows_recon_with_span,
    rows_residual_with_span,
    lmdb_path=None,
    tile_size=96,
    max_results_per_section=14,
    group_gap_x=10,
    group_gap_y=10,
    caption_height=24,
):
    nrows, ncols = 6, 10
    canvas_w = ncols * tile_size + group_gap_x
    canvas_h = nrows * tile_size + group_gap_y + 2 * caption_height
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    def tile_xy(row, col):
        x_offset = group_gap_x if col >= 5 else 0
        y_offset = group_gap_y if row >= 3 else 0
        y_caption = caption_height if row < 3 else 2 * caption_height
        return col * tile_size + x_offset, row * tile_size + y_offset + y_caption

    sections = {
        "recon_no_span": {
            "start_row": 0,
            "start_col": 0,
            "rows": rows_recon_no_span,
            "border": (235, 180, 0),
            "caption": "Mono reconstruction xᴹ (no span)",
        },
        "residual_no_span": {
            "start_row": 0,
            "start_col": 5,
            "rows": rows_residual_no_span,
            "border": (0, 130, 180),
            "caption": "Residual r̂ (no span)",
        },
        "recon_with_span": {
            "start_row": 3,
            "start_col": 0,
            "rows": rows_recon_with_span,
            "border": (20, 160, 60),
            "caption": "Mono reconstruction xᴹ (with span)",
        },
        "residual_with_span": {
            "start_row": 3,
            "start_col": 5,
            "rows": rows_residual_with_span,
            "border": (210, 70, 70),
            "caption": "Residual r̂ (with span)",
        },
    }

    def paste_tile(image, row, col, border_color=None):
        x0, y0 = tile_xy(row, col)
        if image is None:
            tile = Image.new("RGB", (tile_size, tile_size), (242, 242, 242))
        else:
            tile = _center_crop_square(image, tile_size=tile_size)
        canvas.paste(tile, (x0, y0))

        if border_color is not None:
            for k in range(3):
                draw.rectangle(
                    [x0 + k, y0 + k, x0 + tile_size - 1 - k, y0 + tile_size - 1 - k],
                    outline=border_color,
                )

    query_image = _try_load_image(query_row, lmdb_path=lmdb_path)

    for section in sections.values():
        sr = section["start_row"]
        sc = section["start_col"]
        rows = section["rows"][:max_results_per_section]
        border = section["border"]
        caption = section["caption"]

        cap_x0, cap_y0 = tile_xy(sr, sc)
        cap_w = 5 * tile_size
        cap_h = caption_height - 4
        draw.rectangle([cap_x0, cap_y0 - caption_height, cap_x0 + cap_w, cap_y0 - caption_height + cap_h], fill=(245, 245, 245))
        draw.text((cap_x0 + 8, cap_y0 - caption_height + 4), caption, fill=(30, 30, 30))

        positions = [(r, c) for r in range(sr, sr + 3) for c in range(sc, sc + 5)]
        query_pos = positions[0]
        result_positions = positions[1:]

        paste_tile(query_image, query_pos[0], query_pos[1], border_color=(255, 255, 0))

        for row_data, (r, c) in zip(rows, result_positions):
            image = _try_load_image(row_data, lmdb_path=lmdb_path)
            paste_tile(image, r, c)

        if len(rows) < len(result_positions):
            for r, c in result_positions[len(rows) :]:
                paste_tile(None, r, c)

    return canvas


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    global IMAGE_ROOT
    IMAGE_ROOT = args.image_root

    print(f"Loading no-span model: {args.checkpoint_no_span}")
    model_no_span = load_dual_autoencoder(args.checkpoint_no_span)
    print(f"Loading with-span model: {args.checkpoint_with_span}")
    model_with_span = load_dual_autoencoder(args.checkpoint_with_span)

    print(f"Loading dataset: {args.lmdb_path}")
    dataset = load_feature_dataset(args.lmdb_path)

    print("Building banks for no-span model...")
    x_m_no_span, residual_no_span, class_ids, sample_paths = compute_dual_branch_banks(
        model_no_span,
        dataset,
        batch_size=args.batch_size,
    )

    print("Building banks for with-span model...")
    x_m_with_span, residual_with_span, _, _ = compute_dual_branch_banks(
        model_with_span,
        dataset,
        batch_size=args.batch_size,
    )

    for query_index in range(args.query_start, args.query_end + 1):
        print(f"\n=== Processing query {query_index} ===")
        resolved_index = resolve_query_index(dataset, query_index=query_index, query_path=None)
        _, query_class_id, query_sample_path = unpack_item(dataset[resolved_index], resolved_index)

        query_row = {
            "rank": 0,
            "index": resolved_index,
            "similarity": 1.0,
            "class_id": query_class_id,
            "path": query_sample_path,
            "is_query": True,
        }

        query_xm_no_span = x_m_no_span[resolved_index]
        query_res_no_span = residual_no_span[resolved_index]
        query_xm_with_span = x_m_with_span[resolved_index]
        query_res_with_span = residual_with_span[resolved_index]

        rows_recon_no_span = query_topk_similar_images(
            bank=x_m_no_span,
            query_vector=query_xm_no_span,
            sample_paths=sample_paths,
            class_ids=class_ids,
            topk=args.topk,
            exclude_query_index=resolved_index,
        )
        rows_residual_no_span = query_topk_similar_images(
            bank=residual_no_span,
            query_vector=query_res_no_span,
            sample_paths=sample_paths,
            class_ids=class_ids,
            topk=args.topk,
            exclude_query_index=resolved_index,
        )
        rows_recon_with_span = query_topk_similar_images(
            bank=x_m_with_span,
            query_vector=query_xm_with_span,
            sample_paths=sample_paths,
            class_ids=class_ids,
            topk=args.topk,
            exclude_query_index=resolved_index,
        )
        rows_residual_with_span = query_topk_similar_images(
            bank=residual_with_span,
            query_vector=query_res_with_span,
            sample_paths=sample_paths,
            class_ids=class_ids,
            topk=args.topk,
            exclude_query_index=resolved_index,
        )

        result_base = result_dir / f"compare_span_query_{resolved_index}"
        image_path = result_base.with_suffix(".png")
        json_path = result_base.with_suffix(".json")

        fig = build_6x10_comparison_figure(
            query_row=query_row,
            rows_recon_no_span=rows_recon_no_span,
            rows_residual_no_span=rows_residual_no_span,
            rows_recon_with_span=rows_recon_with_span,
            rows_residual_with_span=rows_residual_with_span,
            lmdb_path=args.lmdb_path,
            tile_size=args.tile_size,
            max_results_per_section=args.topk,
        )
        fig.save(image_path, dpi=(300, 300))

        payload = {
            "checkpoint_no_span": args.checkpoint_no_span,
            "checkpoint_with_span": args.checkpoint_with_span,
            "lmdb_path": args.lmdb_path,
            "query_index": resolved_index,
            "query_path": query_sample_path,
            "query_class_id": query_class_id,
            "query_row": query_row,
            "reconstruction_no_span": rows_recon_no_span,
            "residual_no_span": rows_residual_no_span,
            "reconstruction_with_span": rows_recon_with_span,
            "residual_with_span": rows_residual_with_span,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"Saved query {query_index}: {image_path}")


if __name__ == "__main__":
    main()
