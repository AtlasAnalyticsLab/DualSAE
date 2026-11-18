"""Motivation figure, left panel: distribution of top-50 class purity of nearest-neighbour
retrieval with the SAE reconstruction x_hat vs. the residual r = x - x_hat.

For every validation image, its x_hat (or r) is the query and the 50 most cosine-similar x_hat
(or r) of the OTHER validation images are retrieved; class purity is the fraction of them that
share the query's class. Writes the overlaid purity histogram
(top50_class_purity_overlay_histogram.pdf, the figure panel), the per-sample arrays (.npz), two
diagnostic plots and a summary JSON. The figure uses the BatchTopK SAE on CLIP features of
ImageNet-1K, epoch-50 checkpoint (autoencoder_final.pt).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_VISION = Path(__file__).resolve().parents[3]
if str(_VISION) not in sys.path:
    sys.path.insert(0, str(_VISION))

from dataset.lmdb_features import LMDBFeatureDataset  # noqa: E402
from models.batch_topk_sae import BatchTopKSAE  # noqa: E402
from models.jumprelu_sae import JumpReLUSAE
from models.topk_sae import TopK, TopKSAE
from models.vanilla_sae import VanillaSAE


def load_autoencoder(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cfg = checkpoint.get("config", {})

    if "topk_k" in cfg and "d_model" in cfg:
        model = BatchTopKSAE(
            {
                "act_size": cfg["d_model"],
                "dict_size": cfg["n_latents"],
                "top_k": cfg["topk_k"],
                "l1_coeff": 0.0,
                "top_k_aux": cfg.get("auxk_k") or 512,
                "aux_penalty": cfg.get("auxk_coef", 0.0),
                "n_batches_to_dead": cfg.get("dead_steps_threshold", 1000)
                // max(cfg.get("batch_size", 1), 1),
                "seed": cfg.get("seed", 0),
                "device": str(device),
                "dtype": torch.float32,
                "input_unit_norm": cfg.get("normalize", False),
            }
        )
        model.load_state_dict(state_dict)
        return model.to(device).eval()

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
            if key
            not in {
                "activation",
                "activation_state_dict",
                "activation.k",
                "activation.postact_fn",
            }
        }
        model.load_state_dict(state_dict_clean, strict=True)
        return model.to(device).eval()

    if "W_enc" in state_dict:
        act_size, dict_size = state_dict["W_enc"].shape
        model_cfg = {
            "act_size": act_size,
            "dict_size": dict_size,
            "l1_coeff": 0.0,
            "n_batches_to_dead": 1000,
            "seed": cfg.get("seed", 0),
            "device": str(device),
            "dtype": torch.float32,
            "input_unit_norm": True,
        }
        if "jumprelu.log_threshold" in state_dict:
            model_cfg["bandwidth"] = cfg.get("bandwidth", 0.001)
            model = JumpReLUSAE(model_cfg)
        else:
            model = VanillaSAE(model_cfg)
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval()

    raise ValueError(f"Unrecognized checkpoint format: {checkpoint_path}")


def unpack_item(item, idx: int):
    if isinstance(item, tuple):
        if len(item) == 3:
            feat, class_idx, _ = item
        elif len(item) == 2:
            feat, class_idx = item
        else:
            feat = item[0]
            class_idx = None
    else:
        feat = item
        class_idx = None

    feat = torch.as_tensor(feat, dtype=torch.float32)
    class_idx = int(class_idx) if class_idx is not None else -1
    return feat, class_idx


@torch.no_grad()
def build_recon_and_residual_banks(
    autoencoder,
    dataset,
    device: torch.device,
    batch_size: int,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    recon_list = []
    residual_list = []
    labels = []

    n_total = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)

        batch_feats = []
        batch_labels = []
        for idx in range(start, end):
            feat, class_idx = unpack_item(dataset[idx], idx)
            batch_feats.append(feat)
            batch_labels.append(class_idx)

        x = torch.stack(batch_feats, dim=0).to(device)
        _, _, x_hat = autoencoder(x)

        recon = F.normalize(x_hat.float(), p=2, dim=1)
        residual = F.normalize((x - x_hat).float(), p=2, dim=1)

        recon_list.append(recon.cpu())
        residual_list.append(residual.cpu())
        labels.extend(batch_labels)

        print(f"Built embeddings for {end}/{n_total} samples")

    recon_bank = torch.cat(recon_list, dim=0)
    residual_bank = torch.cat(residual_list, dim=0)
    label_arr = np.asarray(labels, dtype=np.int64)
    return recon_bank, residual_bank, label_arr


@torch.no_grad()
def compute_topk_purity_and_mean_similarity(
    bank: torch.Tensor,
    labels: np.ndarray,
    topk: int,
    device: torch.device,
    query_block_size: int,
    db_block_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = bank.shape[0]
    if topk >= n_samples:
        raise ValueError(f"topk={topk} must be smaller than n_samples={n_samples}")

    bank = F.normalize(bank.float(), p=2, dim=1)
    labels_t = torch.from_numpy(labels).long()

    purity = np.zeros(n_samples, dtype=np.float32)
    mean_similarity = np.zeros(n_samples, dtype=np.float32)

    for q_start in range(0, n_samples, query_block_size):
        q_end = min(q_start + query_block_size, n_samples)
        q_size = q_end - q_start

        q = bank[q_start:q_end].to(device, non_blocking=True)
        best_vals = torch.full((q_size, topk), -float("inf"), device=device)
        best_inds = torch.full((q_size, topk), -1, device=device, dtype=torch.long)

        for d_start in range(0, n_samples, db_block_size):
            d_end = min(d_start + db_block_size, n_samples)
            d = bank[d_start:d_end].to(device, non_blocking=True)

            sims = q @ d.t()

            overlap_mask = (torch.arange(q_start, q_end, device=device) >= d_start) & (
                torch.arange(q_start, q_end, device=device) < d_end
            )
            if overlap_mask.any():
                local_rows = torch.nonzero(overlap_mask, as_tuple=False).squeeze(1)
                global_rows = local_rows + q_start
                local_cols = global_rows - d_start
                sims[local_rows, local_cols] = -float("inf")

            local_k = min(topk, d_end - d_start)
            vals, inds = torch.topk(sims, k=local_k, dim=1)
            inds = inds + d_start

            merged_vals = torch.cat([best_vals, vals], dim=1)
            merged_inds = torch.cat([best_inds, inds], dim=1)
            keep = torch.topk(merged_vals, k=topk, dim=1).indices

            best_vals = torch.gather(merged_vals, 1, keep)
            best_inds = torch.gather(merged_inds, 1, keep)

        row_labels = labels_t[q_start:q_end].to(device).unsqueeze(1)
        nn_labels = labels_t[best_inds.cpu()].to(device)
        row_purity = (nn_labels == row_labels).float().mean(dim=1)

        purity[q_start:q_end] = row_purity.cpu().numpy()
        mean_similarity[q_start:q_end] = best_vals.mean(dim=1).cpu().numpy()

        print(f"Computed top-{topk} purity for {q_end}/{n_samples} samples")

    return purity, mean_similarity


def make_plot(
    recon_purity_sorted: np.ndarray,
    residual_purity_sorted: np.ndarray,
    output_path: Path,
):
    x = np.arange(recon_purity_sorted.shape[0])

    plt.figure(figsize=(6, 3))
    plt.plot(x, recon_purity_sorted, linewidth=1.2, label="Reconstruction top-50 purity")
    plt.plot(x, residual_purity_sorted, linewidth=1.2, label="Residual top-50 purity")
    plt.xlabel("Validation sample index sorted by reconstruction similarity")
    plt.ylabel("Class purity in top-50")
    plt.title("Top-50 Class Purity: Reconstruction vs Residual")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def make_diagnostic_plot(
    recon_similarity_sorted: np.ndarray,
    residual_similarity_sorted: np.ndarray,
    output_path: Path,
):
    """Plot the sorted similarity values to verify sorting is correct."""
    x = np.arange(recon_similarity_sorted.shape[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3))

    ax1.plot(x, recon_similarity_sorted, linewidth=0.8, label="Reconstruction")
    ax1.plot(x, residual_similarity_sorted, linewidth=0.8, label="Residual")
    ax1.set_xlabel("Sample index (sorted by reconstruction similarity)")
    ax1.set_ylabel("Mean top-50 similarity")
    ax1.set_title("Top-50 Mean Similarity (sorted by reconstruction)")
    ax1.grid(alpha=0.25)
    ax1.legend()

    # Plot first differences to verify monotonic growth
    recon_diff = np.diff(recon_similarity_sorted)
    residual_diff = np.diff(residual_similarity_sorted)
    ax2.hist(recon_diff, bins=100, alpha=0.6, label="Reconstruction diff")
    ax2.hist(residual_diff, bins=100, alpha=0.6, label="Residual diff")
    ax2.set_xlabel("Difference between consecutive samples")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of differences (should be mostly ≥0 for recon)")
    ax2.grid(alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def make_purity_overlay_hist_plot(
    recon_purity: np.ndarray,
    residual_purity: np.ndarray,
    output_path: Path,
):
    """Plot overlaid histograms of class purity for reconstruction and residual."""

    fig, ax = plt.subplots(figsize=(4, 3))

    bins = np.linspace(0.0, 1.0, 25)
    recon_weights = np.ones_like(recon_purity, dtype=np.float64) / max(len(recon_purity), 1)
    residual_weights = np.ones_like(residual_purity, dtype=np.float64) / max(len(residual_purity), 1)

    ax.hist(
        recon_purity,
        bins=bins,
        weights=recon_weights,
        alpha=0.6,
        label="Reconstruction",
        color="tab:blue",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.hist(
        residual_purity,
        bins=bins,
        weights=residual_weights,
        alpha=0.6,
        label="Residual",
        color="tab:orange",
        edgecolor="black",
        linewidth=0.4,
    )

    ax.set_xlabel("Average class purity in top-50 retrievals")
    ax.set_ylabel("Probability")
    ax.set_title("Class Purity Distribution")
    ax.grid(alpha=0.25, axis='y')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()



def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and plot top-50 class purity for reconstruction/residual embeddings."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAE checkpoint")
    parser.add_argument("--lmdb", type=str, required=True, help="Path to validation LMDB")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/ablations/figures/motivation",
        help="Directory for outputs",
    )
    parser.add_argument("--topk", type=int, default=50, help="Number of neighbors for purity")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for embedding extraction")
    parser.add_argument("--query-block-size", type=int, default=512, help="Query block size for search")
    parser.add_argument("--db-block-size", type=int, default=4096, help="Database block size for search")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for debugging (default uses full validation set)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device for embedding extraction and similarity search",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    autoencoder = load_autoencoder(args.checkpoint, device)

    print(f"Loading dataset: {args.lmdb}")
    dataset = LMDBFeatureDataset(
        args.lmdb,
        return_index=False,
        return_class_idx=True,
        return_path=False,
        verbose=True,
    )

    print("Building reconstruction and residual banks...")
    recon_bank, residual_bank, labels = build_recon_and_residual_banks(
        autoencoder=autoencoder,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    print("Computing top-k purity for reconstruction bank...")
    recon_purity, recon_mean_similarity = compute_topk_purity_and_mean_similarity(
        bank=recon_bank,
        labels=labels,
        topk=args.topk,
        device=device,
        query_block_size=args.query_block_size,
        db_block_size=args.db_block_size,
    )

    print("Computing top-k purity for residual bank...")
    residual_purity, residual_mean_similarity = compute_topk_purity_and_mean_similarity(
        bank=residual_bank,
        labels=labels,
        topk=args.topk,
        device=device,
        query_block_size=args.query_block_size,
        db_block_size=args.db_block_size,
    )

    # Sort by reconstruction similarity from low to high.
    order = np.argsort(recon_mean_similarity)
    recon_purity_sorted = recon_purity[order]
    residual_purity_sorted = residual_purity[order]
    recon_similarity_sorted = recon_mean_similarity[order]
    residual_similarity_sorted = residual_mean_similarity[order]

    print(f"Verification: recon_similarity_sorted min={recon_similarity_sorted.min():.6f}, max={recon_similarity_sorted.max():.6f}")
    print(f"Is recon_similarity sorted? {np.all(recon_similarity_sorted[:-1] <= recon_similarity_sorted[1:])}")

    figure_path = output_dir / "top50_class_purity_sorted_by_recon_similarity.png"
    make_plot(recon_purity_sorted, residual_purity_sorted, figure_path)

    diagnostic_path = output_dir / "top50_similarity_diagnostic.png"
    make_diagnostic_plot(recon_similarity_sorted, residual_similarity_sorted, diagnostic_path)

    purity_hist_path = output_dir / "top50_class_purity_overlay_histogram.pdf"
    make_purity_overlay_hist_plot(recon_purity, residual_purity, purity_hist_path)

    metrics_path = output_dir / "top50_class_purity_sorted_by_recon_similarity.npz"
    np.savez(
        metrics_path,
        sorted_indices=order,
        recon_topk_purity_sorted=recon_purity_sorted,
        residual_topk_purity_sorted=residual_purity_sorted,
        recon_topk_purity=recon_purity,
        residual_topk_purity=residual_purity,
        recon_topk_mean_similarity_sorted=recon_similarity_sorted,
        residual_topk_mean_similarity_sorted=residual_similarity_sorted,
        recon_topk_mean_similarity=recon_mean_similarity,
        residual_topk_mean_similarity=residual_mean_similarity,
        labels=labels,
    )

    summary = {
        "checkpoint": args.checkpoint,
        "lmdb": args.lmdb,
        "num_samples": int(recon_bank.shape[0]),
        "topk": int(args.topk),
        "device": str(device),
        "figure_path": str(figure_path),
        "diagnostic_path": str(diagnostic_path),
        "purity_hist_path": str(purity_hist_path),
        "metrics_path": str(metrics_path),
        "recon_purity_mean": float(recon_purity.mean()),
        "residual_purity_mean": float(residual_purity.mean()),
        "purity_diff_mean": float((recon_purity - residual_purity).mean()),
    }
    summary_path = output_dir / "top50_class_purity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Figure: {figure_path}")
    print(f"Diagnostic: {diagnostic_path}")
    print(f"Purity histogram: {purity_hist_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
