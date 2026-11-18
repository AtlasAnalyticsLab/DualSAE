import os
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from run_logger import RunLogger
from collect_results import write_best_metrics

from dataset import (
    LMDBFeatureDataset,
    build_feature_dataset,
    is_legacy_multilabel_path,
    resolve_multilabel,
)
from models import TopKSAE, TopK
from loss import compute_monosemanticity_fast, compute_monosemanticity_loss_batch
from mono_loss import compute_monosemanticity_custom
from metrics import compute_binary_purity, compute_multilabel_binary_purity
from utils import PersistentBatcher, resolve_step_budget, unpack_batch


def to_relative_imagenet_path(path_key: str) -> str:
    """Convert absolute ImageNet path to split-relative path (e.g., val/.../img.JPEG)."""
    normalized = str(path_key).replace("\\", "/")
    for split in ("train", "val", "test"):
        marker = f"/{split}/"
        pos = normalized.find(marker)
        if pos >= 0:
            return normalized[pos + 1 :]
    return normalized.lstrip("/")


def _is_open_images_path(path: Optional[str]) -> bool:
    """Deprecated shim.  Kept so any in-tree caller of this name keeps working.

    Multi-label mode is no longer decided by a path substring: see
    ``dataset.resolve.resolve_multilabel``, which consults an explicit recorded value first,
    then a self-describing ``__multilabel__`` marker inside the LMDB, and only then this
    legacy test.  That ordering is what lets a non-OpenImages multi-label corpus work while
    every existing OpenImages run resolves exactly as before.
    """
    return is_legacy_multilabel_path(path)


def _is_multilabel_targets(targets: torch.Tensor) -> bool:
    return isinstance(targets, torch.Tensor) and targets.ndim == 2


def _micro_f1_from_binary_preds(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    pred = pred.bool()
    target = target.bool()
    tp = (pred & target).sum().float()
    fp = (pred & (~target)).sum().float()
    fn = ((~pred) & target).sum().float()
    return float((2.0 * tp / (2.0 * tp + fp + fn + eps)).item())


@torch.no_grad()
def save_intermediate_outputs(autoencoder, dataloader, output_path: Path, device='cuda', split_name: str = 'val'):
    """Save a mapping: path -> (x, z, x_hat) for a full dataloader pass."""
    autoencoder.eval()
    outputs = {}

    for batch in tqdm(dataloader, desc=f"Saving intermediates ({split_name})"):
        features, _, sample_path = unpack_batch(batch)
        x = features.to(device)

        _, z, x_hat = autoencoder(x)

        x_cpu = x.detach().cpu()
        z_cpu = z.detach().cpu()
        x_hat_cpu = x_hat.detach().cpu()

        batch_size = x_cpu.shape[0]
        if sample_path is None:
            raise ValueError(
                "save_intermediate_outputs requires sample paths. "
                "Create LMDB with __path__{idx} metadata and load dataset with return_path=True."
            )

        if isinstance(sample_path, str):
            path_list = [sample_path]
        elif torch.is_tensor(sample_path):
            path_list = [str(v) for v in sample_path.detach().cpu().tolist()]
        elif isinstance(sample_path, np.ndarray):
            path_list = [str(v) for v in sample_path.tolist()]
        else:
            path_list = [str(v) for v in list(sample_path)]

        if len(path_list) != batch_size:
            raise ValueError(
                f"Mismatched batch/path sizes: batch_size={batch_size}, num_paths={len(path_list)}"
            )

        key_list = [to_relative_imagenet_path(path) for path in path_list]

        for i, key in enumerate(key_list):
            outputs[key] = (x_cpu[i], z_cpu[i], x_hat_cpu[i])

    torch.save(outputs, output_path)
    autoencoder.train()
    return output_path


@torch.no_grad()
def _build_reconstruction_bank(
    autoencoder,
    dataloader,
    device='cuda',
    max_samples: Optional[int] = None,
    split_name: str = 'train',
):
    """Build a normalized reconstructed-embedding bank (x_hat, labels)."""
    autoencoder.eval()
    feats = []
    labels = []
    collected = 0

    for batch in tqdm(dataloader, desc=f"Building {split_name} reconstruction bank"):
        features, class_idx, _ = unpack_batch(batch)
        if class_idx is None:
            raise ValueError("Classification evaluation requires class indices in dataloader batches.")

        x = features.to(device)
        class_idx = class_idx.to(device)

        _, _, x_hat = autoencoder(x)
        x_hat = F.normalize(x_hat.float(), p=2, dim=1)

        feats.append(x_hat.detach().cpu())
        if _is_multilabel_targets(class_idx):
            labels.append(class_idx.detach().cpu().float())
        else:
            labels.append(class_idx.detach().cpu().long())
        collected += x_hat.shape[0]

        if max_samples is not None and collected >= max_samples:
            break

    if len(feats) == 0:
        raise ValueError(f"No samples found for split {split_name}.")

    bank_x = torch.cat(feats, dim=0)
    bank_y = torch.cat(labels, dim=0)

    if max_samples is not None and bank_x.shape[0] > max_samples:
        bank_x = bank_x[:max_samples]
        bank_y = bank_y[:max_samples]

    return bank_x, bank_y


@torch.no_grad()
def _compute_knn_accuracy_from_recon(
    autoencoder,
    train_loader,
    eval_loader,
    device='cuda',
    k: int = 20,
    train_bank_max_samples: Optional[int] = None,
    bank_chunk_size: int = 65536,
    split_name: str = 'val',
):
    """Compute KNN metric from reconstructed embeddings x_hat.

    Returns:
        - single-label: top-1 accuracy
        - multi-label: micro-F1
    """
    bank_x_cpu, bank_y_cpu = _build_reconstruction_bank(
        autoencoder,
        train_loader,
        device=device,
        max_samples=train_bank_max_samples,
        split_name='train',
    )

    num_classes = int(bank_y_cpu.max().item()) + 1
    effective_k = min(k, bank_x_cpu.shape[0])
    if effective_k <= 0:
        return 0.0

    is_multilabel = bank_y_cpu.ndim == 2
    if is_multilabel:
        num_classes = bank_y_cpu.shape[1]

    autoencoder.eval()
    correct = 0
    total = 0

    for batch in tqdm(eval_loader, desc=f"KNN eval ({split_name})"):
        features, class_idx, _ = unpack_batch(batch)
        if class_idx is None:
            raise ValueError("KNN evaluation requires class indices in eval batches.")

        x = features.to(device)
        y = class_idx.to(device)
        _, _, x_hat = autoencoder(x)
        q = F.normalize(x_hat.float(), p=2, dim=1)

        top_vals = None
        top_labels = None

        for start in range(0, bank_x_cpu.shape[0], bank_chunk_size):
            end = min(start + bank_chunk_size, bank_x_cpu.shape[0])
            chunk_x = bank_x_cpu[start:end].to(device)
            chunk_y = bank_y_cpu[start:end].to(device)

            sim = q @ chunk_x.t()
            kk = min(effective_k, sim.shape[1])
            vals, inds = torch.topk(sim, k=kk, dim=1)
            lbls = chunk_y[inds]

            if top_vals is None:
                top_vals = vals
                top_labels = lbls
            else:
                merged_vals = torch.cat([top_vals, vals], dim=1)
                merged_labels = torch.cat([top_labels, lbls], dim=1)
                keep_inds = torch.topk(merged_vals, k=effective_k, dim=1).indices
                top_vals = torch.gather(merged_vals, 1, keep_inds)
                if merged_labels.ndim == 2:
                    top_labels = torch.gather(merged_labels, 1, keep_inds)
                else:
                    top_labels = torch.gather(
                        merged_labels,
                        1,
                        keep_inds.unsqueeze(-1).expand(-1, -1, merged_labels.shape[-1]),
                    )

        if is_multilabel:
            scores = top_labels.float().mean(dim=1)
            pred = scores >= 0.5
            target = y.float() >= 0.5
            correct += _micro_f1_from_binary_preds(pred, target) * y.shape[0]
            total += y.shape[0]
        else:
            y = y.long()
            votes = F.one_hot(top_labels, num_classes=num_classes).sum(dim=1)
            pred = votes.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.shape[0]

    return float(correct) / float(max(total, 1))


def _compute_linear_probe_accuracy_from_recon(
    autoencoder,
    train_loader,
    eval_loader,
    device='cuda',
    train_bank_max_samples: Optional[int] = None,
    eval_max_samples: Optional[int] = None,
    lp_epochs: int = 10,
    lp_batch_size: int = 1024,
    lp_lr: float = 1e-3,
    lp_weight_decay: float = 1e-4,
    split_name: str = 'val',
):
    """Train linear probe on reconstructed train embeddings and evaluate metric on split."""
    train_x, train_y = _build_reconstruction_bank(
        autoencoder,
        train_loader,
        device=device,
        max_samples=train_bank_max_samples,
        split_name='train',
    )
    eval_x, eval_y = _build_reconstruction_bank(
        autoencoder,
        eval_loader,
        device=device,
        max_samples=eval_max_samples,
        split_name=split_name,
    )

    feat_dim = train_x.shape[1]
    is_multilabel = train_y.ndim == 2
    num_classes = train_y.shape[1] if is_multilabel else (int(train_y.max().item()) + 1)
    probe = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lp_lr, weight_decay=lp_weight_decay)
    criterion = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()

    train_x = train_x.float()
    eval_x = eval_x.float()

    for _ in range(lp_epochs):
        probe.train()
        perm = torch.randperm(train_x.shape[0])
        for start in range(0, train_x.shape[0], lp_batch_size):
            end = min(start + lp_batch_size, train_x.shape[0])
            idx = perm[start:end]
            xb = train_x[idx].to(device)
            yb = train_y[idx].to(device)

            logits = probe(xb)
            if is_multilabel:
                loss = criterion(logits, yb.float())
            else:
                loss = criterion(logits, yb.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, eval_x.shape[0], lp_batch_size):
            end = min(start + lp_batch_size, eval_x.shape[0])
            xb = eval_x[start:end].to(device)
            yb = eval_y[start:end].to(device)
            logits = probe(xb)
            if is_multilabel:
                pred = torch.sigmoid(logits) >= 0.5
                correct += _micro_f1_from_binary_preds(pred, yb.float() >= 0.5) * yb.shape[0]
                total += yb.shape[0]
            else:
                pred = logits.argmax(dim=1)
                correct += (pred == yb.long()).sum().item()
                total += yb.shape[0]

    return float(correct) / float(max(total, 1))


def evaluate_reconstruction_classification(
    autoencoder,
    train_loader,
    eval_loader,
    device='cuda',
    split_name='val',
):
    """Evaluate KNN and linear probe metrics using reconstructed embeddings x_hat."""
    is_multilabel = bool(getattr(getattr(eval_loader, 'dataset', None), 'return_multilabel', False))
    knn_acc = _compute_knn_accuracy_from_recon(
        autoencoder,
        train_loader,
        eval_loader,
        device=device,
        k=20,
        train_bank_max_samples=None,
        bank_chunk_size=65536,
        split_name=split_name,
    )
    lp_acc = _compute_linear_probe_accuracy_from_recon(
        autoencoder,
        train_loader,
        eval_loader,
        device=device,
        train_bank_max_samples=None,
        eval_max_samples=None,
        lp_epochs=10,
        lp_batch_size=1024,
        lp_lr=1e-3,
        lp_weight_decay=1e-4,
        split_name=split_name,
    )
    print(f"{split_name} KNN metric on reconstructed embeddings: {knn_acc:.4f}")
    print(f"{split_name} LP  metric on reconstructed embeddings: {lp_acc:.4f}")

    metrics = {
        f'{split_name}_knn_acc_recon': knn_acc,
        f'{split_name}_lp_acc_recon': lp_acc,
        f'{split_name}_knn_metric_recon': knn_acc,
        f'{split_name}_lp_metric_recon': lp_acc,
        f'{split_name}_metric_type_recon': 'micro_f1' if is_multilabel else 'accuracy',
    }
    if is_multilabel:
        metrics[f'{split_name}_knn_micro_f1_recon'] = knn_acc
        metrics[f'{split_name}_lp_micro_f1_recon'] = lp_acc
    else:
        metrics[f'{split_name}_knn_accuracy_recon'] = knn_acc
        metrics[f'{split_name}_lp_accuracy_recon'] = lp_acc
    return metrics


def unit_norm_decoder_(autoencoder: TopKSAE) -> None:
    """Unit normalize the decoder weights."""
    if hasattr(autoencoder.decoder, 'weight'):
        autoencoder.decoder.weight.data /= (autoencoder.decoder.weight.data.norm(dim=0, keepdim=True) + 1e-8)


def unit_norm_decoder_grad_adjustment_(autoencoder: TopKSAE) -> None:
    """Project out gradient information parallel to the dictionary vectors."""
    from models import TiedTranspose
    
    # For tied weights, skip (gradient flows through encoder)
    if isinstance(autoencoder.decoder, TiedTranspose):
        return
    
    if not hasattr(autoencoder.decoder, 'weight'):
        return
    
    # Check if gradient exists
    if autoencoder.decoder.weight.grad is None:
        return
    
    # Compute parallel component: <grad, weight> for each feature
    parallel_component = torch.einsum('dn,dn->n', autoencoder.decoder.weight.data, autoencoder.decoder.weight.grad)
    
    # Subtract parallel component from gradient
    autoencoder.decoder.weight.grad -= parallel_component.unsqueeze(0) * autoencoder.decoder.weight.data


def reinitialize_dead_neurons(autoencoder, dead_mask, sample_batch, device):
    """Reinitialize dead neurons with random directions."""
    num_dead = dead_mask.sum().item()
    if num_dead == 0:
        return
    
    with torch.no_grad():
        batch_size = sample_batch.shape[0]
        
        # Sample from batch for reinitialization (with replacement if needed)
        if num_dead <= batch_size:
            # Sample without replacement
            sample_indices = torch.randperm(batch_size)[:num_dead]
        else:
            # Sample with replacement when we need more samples than batch size
            sample_indices = torch.randint(0, batch_size, (num_dead,))
        
        samples = sample_batch[sample_indices].to(device)
        
        # Normalize samples and add small random noise for diversity
        samples = F.normalize(samples, p=2, dim=-1)
        
        # Reinitialize encoder weights
        autoencoder.encoder.weight.data[dead_mask] = samples * 0.1
        
        # Reset latent bias
        autoencoder.latent_bias.data[dead_mask] = 0.0
        
        # Reset decoder weights (transpose of encoder for tied initialization)
        if hasattr(autoencoder.decoder, 'weight'):
            autoencoder.decoder.weight.data[:, dead_mask] = samples.T * 0.1

        # UNTIED decoder only (--no_tied_weights): re-apply the
        # init rule to the revived latents -- unit-norm decoder atom, encoder row = its transpose
        # (the same rule as the untied init in train_autoencoder, dictionary_learning top_k.py
        # :72-78). Why it is needed: with TIED weights the two writes above hit ONE matrix, and
        # the next unit_norm_decoder_ rescales it, so a revived latent reads and writes along
        # the sample direction at norm 1. Untied, unit_norm_decoder_ renormalises only the
        # decoder, so the revived ENCODER row would stay at norm 0.1 for good: the latent's
        # pre-activation would start ~10x below the live latents it must beat for a top-k slot,
        # i.e. the reinit would mostly revive latents straight back into the dead set.
        # `samples` rows are already L2-normalised, so samples.T is the unit-norm atom. Scope:
        # the tied path (TiedTranspose) is untouched. Known limitation shared with the tied
        # arm: Adam moments of revived latents are not reset.
        from models import TiedTranspose
        if (hasattr(autoencoder.decoder, 'weight')
                and not isinstance(autoencoder.decoder, TiedTranspose)):
            autoencoder.decoder.weight.data[:, dead_mask] = samples.T
            autoencoder.encoder.weight.data[dead_mask] = autoencoder.decoder.weight.data[:, dead_mask].T

        # Reset statistics
        autoencoder.stats_last_nonzero[dead_mask] = 0


def _mean_active_binary_purity(acts: torch.Tensor, class_idx: Optional[torch.Tensor]) -> Optional[float]:
    """Return mean binary purity over latents that activate in the current batch."""
    if class_idx is None:
        return None

    if _is_multilabel_targets(class_idx):
        purity = compute_multilabel_binary_purity(acts.detach(), class_idx.detach())
    else:
        purity = compute_binary_purity(acts.detach(), class_idx.detach())
    active_latent_mask = (acts > 0).any(dim=0)
    if active_latent_mask.any():
        return purity[active_latent_mask].mean().item()
    return 0.0


@torch.no_grad()
def evaluate(autoencoder, dataloader, config, split_name='val'):
    """Evaluate autoencoder on a dataset."""
    autoencoder.eval()
    
    total_loss = 0
    total_nmse = 0
    total_active = 0
    num_batches = 0

    acts_all =[]
    class_idx_all = []
    
    dead_steps_threshold = config.dead_steps_threshold // config.batch_size
    
    for batch in dataloader:
        features, class_idx, _ = unpack_batch(batch)
        x = features.to(config.device)
        if class_idx is not None:
            class_idx = class_idx.to(config.device)
        
        # TopKSAE returns (latents_pre_act, latents, recons)
        latents_pre_act, latents, recons = autoencoder(x)

        acts_all.append(latents.detach().cpu())
        if class_idx is not None:
            class_idx_all.append(class_idx.detach().cpu())
        
        # Compute loss using model's compute_loss method
        loss, metrics = autoencoder.compute_loss(
            x, latents_pre_act, latents, recons,
            auxk_coef=config.auxk_coef,
            dead_steps_threshold=dead_steps_threshold,
            auxk_k=config.auxk_k,
            mono_coef=0.0  # Don't use mono loss during evaluation
        )
        
        total_loss += loss.item()
        total_nmse += metrics.get('nmse_loss', metrics.get('l2_loss', 0))
        total_active += metrics['num_active']

        num_batches += 1
    
    autoencoder.train()
    
    avg_loss = total_loss / num_batches
    avg_nmse = total_nmse / num_batches
    avg_active = total_active / num_batches

    acts = torch.cat(acts_all, dim=0)
    if len(class_idx_all) > 0:
        class_idx = torch.cat(class_idx_all, dim=0)
        avg_binary_purity = _mean_active_binary_purity(acts, class_idx)
    else:
        avg_binary_purity = 0.0
    
    return {
        f'{split_name}_loss': avg_loss,
        f'{split_name}_nmse': avg_nmse,
        f'{split_name}_active': avg_active,
        f'{split_name}_binary_purity': avg_binary_purity
    }



@torch.no_grad()
def compute_r2(autoencoder, dataloader, device='cuda', eps: float = 1e-12):
    """Compute R² (uniform average across features) over an entire dataloader.

    Uses a single streaming pass with sufficient statistics to avoid storing all data:
      - sum_y (per feature), sum_y2 (per feature), N (scalar), SS_res (per feature)
      - SS_tot = sum_y2 - (sum_y^2) / N (per feature)
      - R2_feature = 1 - SS_res / SS_tot (masked when SS_tot≈0)
      - R2 = mean over features with SS_tot > 0
    """
    sum_y = None
    sum_y2 = None
    ss_res = None
    total_n = 0

    autoencoder.eval()

    for batch in dataloader:
        features, class_idx, _ = unpack_batch(batch)
        x = features.to(device)
        if class_idx is not None:
            class_idx = class_idx.to(device)

        # TopKSAE returns (latents_pre_act, latents, recons)
        _, _, recons = autoencoder(x)

        # Cast to float64 for stable accumulation
        x64 = x.to(torch.float64)
        r64 = recons.to(torch.float64)

        if sum_y is None:
            D = x64.shape[1]
            sum_y = torch.zeros(D, dtype=torch.float64, device=device)
            sum_y2 = torch.zeros(D, dtype=torch.float64, device=device)
            ss_res = torch.zeros(D, dtype=torch.float64, device=device)

        sum_y += x64.sum(dim=0)
        sum_y2 += (x64 * x64).sum(dim=0)
        ss_res += ((x64 - r64) * (x64 - r64)).sum(dim=0)
        total_n += x64.shape[0]

    if total_n == 0:
        return 0.0

    ss_tot = sum_y2 - (sum_y * sum_y) / max(total_n, 1)
    valid = ss_tot > eps
    if valid.any():
        r2_per_feat = torch.zeros_like(ss_tot)
        r2_per_feat[valid] = 1.0 - (ss_res[valid] / (ss_tot[valid] + eps))
        r2 = r2_per_feat[valid].mean().item()
    else:
        r2 = 0.0

    autoencoder.train()
    return r2



@dataclass
class TrainingConfig:
    # Data (required field first)
    train_path: str

    # Model info
    model_type: str = 'topk'
        
    # Data (optional)
    val_path: Optional[str] = None
    test_path: Optional[str] = None

    # Experiment
    exp_name: str = 'default'

    
    # Model architecture (TopK SAE only)
    n_latents: int = 8192
    d_model: Optional[int] = None  # Will be inferred from data
    activation: str = 'topk'  # 'relu' or 'topk'
    topk_k: int = 64
    tied_weights: bool = True
    normalize: bool = False  # Layer normalization
    
    # Training
    batch_size: int = 4096
    num_epochs: int = 10
    lr: float = 1e-4
    eps: float = 6.25e-10
    
    # Loss coefficients (TopK SAE specific)
    auxk_coef: float = 1/32
    auxk_k: Optional[int] = None  # If None, uses same k as topk_k
    mono_coef: float = 0.0  # Coefficient for monosemanticity loss (1 - mono_score)
    mono_period: int = 1  # Compute mono loss every N steps (1=per batch, >1=full dataset every N steps)
    
    # Dead neuron handling
    dead_steps_threshold: int = 10_000_000  # In tokens/samples
    dead_check_interval: int = 1000  # Check every N steps

    # Strip heavy per-epoch / end-of-run eval (train-set R2/mono, KNN/LP, save-intermediates).
    # Training math + per-epoch val-MonoScore selection are kept.
    skip_extra_eval: bool = False

    # Logging
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_group: Optional[str] = None
    tb_dir: Optional[str] = None        # TensorBoard event dir (real-time offline); None = no TB

    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    seed: int = 42
    # --- language-domain additions (all default to today's behaviour) --------------------
    # Fixed-step-budget training.  None == one full pass over the loader per epoch, i.e. the
    # original behaviour, so every existing vision run is unchanged.  A value makes an "epoch"
    # exactly this many optimizer steps, drawn from ONE persistent loader iterator so a small
    # corpus advances through the data instead of replaying the same prefix (see
    # utils.PersistentBatcher).
    steps_per_epoch: Optional[int] = None
    # Explicit multi-label declaration, persisted into training_config.json so a later re-score
    # can never disagree with the run that produced the checkpoint.  None == infer from the
    # LMDB's __multilabel__ marker, then from the legacy OpenImages path substring.
    multilabel: Optional[bool] = None
    # Absolute path to a companion multi-hot label store, for a multi-label corpus whose label
    # LMDB is shared across several feature LMDBs (e.g. one per text encoder).
    labels_lmdb: Optional[str] = None
    # Free-form provenance so the collector can label rows without parsing paths.
    dataset_name: Optional[str] = None
    extractor_name: Optional[str] = None
    output_dir: str = './checkpoints'


def train_autoencoder(config: TrainingConfig):
    """
    Main training loop for TopK sparse autoencoder.
    
    Args:
        config: Training configuration
        
    Returns:
        autoencoder: Trained TopKSAE model
    """
    
    # Construct full output path: output_dir/exp_name
    config.output_dir = os.path.join(config.output_dir, config.exp_name)
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Initialize wandb
    run = None
    tb = RunLogger(config.tb_dir)  # TensorBoard (real-time offline); no-op when --tb_dir unset
    if config.wandb_project:
        # W&B is optional: imported here, only when a project is requested. Every
        # later wandb.* call is guarded by the same test (or by `run is not None`).
        import wandb
        run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            group=config.wandb_group,
            config=vars(config)
        )
    
    # Load train dataset (LMDB format)
    print(f"Loading train dataset from {config.train_path}...")
    train_dataset = build_feature_dataset(
        config.train_path,
        return_index=False,
        return_path=True,
        verbose=True,
        return_class_idx=True,
        return_multilabel=resolve_multilabel(config.train_path, getattr(config, 'multilabel', None)),
    )
    
    # Infer feature dimension
    d_model = train_dataset.get_feature_dim()
    config.d_model = d_model
    print(f"Feature dimension: {d_model}")
    
    # Create train dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Create separate non-shuffled dataloader for mono loss computation (if needed)
    mono_loader = None
    if config.mono_coef > 0 and config.mono_period > 1:
        print(f"Creating separate DataLoader for periodic mono loss computation...")
        mono_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False  # Include all samples
        )
    
    # Load val dataset if provided
    val_loader = None
    if config.val_path:
        print(f"Loading val dataset from {config.val_path}...")
        val_dataset = build_feature_dataset(
            config.val_path,
            return_index=False,
            return_path=True,
            verbose=False,
            return_class_idx=True,
            return_multilabel=resolve_multilabel(config.val_path, getattr(config, 'multilabel', None)),
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False
        )
        print(f"Val dataset: {len(val_dataset)} samples")
    
    # Load test dataset if provided  
    test_loader = None
    if config.test_path:
        print(f"Loading test dataset from {config.test_path}...")
        test_dataset = build_feature_dataset(
            config.test_path,
            return_index=False,
            return_path=True,
            verbose=False,
            return_class_idx=True,
            return_multilabel=resolve_multilabel(config.test_path, getattr(config, 'multilabel', None)),
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False
        )
        print(f"Test dataset: {len(test_dataset)} samples")
    
    # Create data iterator that cycles infinitely
    def cycle_dataloader(dataloader):
        while True:
            for batch in dataloader:
                try:
                    yield batch
                except Exception as e:
                    print(f"Error in dataloader iteration: {e}")
                    print(batch)
                    continue
    
    data_iter = cycle_dataloader(train_loader)
    
    # Initialize TopK SAE model
    print(f"Initializing TopK SAE: {config.n_latents} latents, {d_model} features")
    
    if config.activation == 'topk':
        activation = TopK(k=config.topk_k)
    else:
        activation = nn.ReLU()
    
    autoencoder = TopKSAE(
        n_latents=config.n_latents,
        n_inputs=d_model,
        activation=activation,
        tied=config.tied_weights,
        normalize=config.normalize
    ).to(config.device)
    
    # Convert dead_steps_threshold to steps
    dead_steps_threshold_steps = config.dead_steps_threshold // config.batch_size
    
    # Initialize decoder to unit norm
    unit_norm_decoder_(autoencoder)

    # UNTIED path only (--no_tied_weights). The reference TopK recipe
    # ties the weights AT INIT and unties them for training: Gao et al. 2024 ("we initialize the
    # encoder to the transpose of the decoder") and dictionary_learning's AutoEncoderTopK
    # (trainers/top_k.py: unit-norm the decoder, then encoder.weight = decoder.weight.T.clone()).
    # Without this, TopKSAE(tied=False)
    # starts from two INDEPENDENT random matrices, so the decoder atom a latent writes is unrelated
    # to the direction it reads -- a known source of early dead latents, and a deviation from the
    # reference we would otherwise be silently comparing against.
    # Scope: the tied path is untouched (it already starts from exactly this state: one matrix,
    # unit-norm rows).
    # Done BEFORE the optimizer is built, and via .data.copy_ so the Parameter objects the
    # optimizer registers are the same ones trained.
    if not config.tied_weights:
        with torch.no_grad():
            autoencoder.encoder.weight.data.copy_(autoencoder.decoder.weight.data.t())
    
    # Optimizer
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=config.lr, eps=config.eps)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Training loop
    print(f"Starting training for {config.num_epochs} epochs...")
    autoencoder.train()
    
    # Keep a sample batch for dead neuron reinitialization
    sample_batch_raw = next(data_iter)
    sample_batch, _, _ = unpack_batch(sample_batch_raw)
    if isinstance(sample_batch, np.ndarray):
        sample_batch = torch.from_numpy(sample_batch)
    
    global_step = 0
    best_val_mono = float('-inf')
    best_val_mono_epoch = None
    best_val_mono_metrics = None
    best_val_intermediates_path = Path(config.output_dir) / 'best_val_monoscore_intermediates.pt'
    best_val_model_path = Path(config.output_dir) / 'best_val_monoscore_model.pt'
    
    # One persistent iterator for the whole run.  With steps_per_epoch=None this is a no-op
    # wrapper yielding exactly `train_loader`; with a value it fixes the optimizer-step budget
    # per epoch while still walking the dataset across epochs.  resolve_step_budget refuses a
    # warm-up that exceeds the total budget, because such a run is a MonoSAE reported under the
    # DualSAE name and it fails silently.
    _batcher = PersistentBatcher(train_loader, getattr(config, 'steps_per_epoch', None))
    _total_steps = resolve_step_budget(
        config.num_epochs,
        getattr(config, 'steps_per_epoch', None),
        int(getattr(config, 'warmup_steps', 0) or 0),
        label=getattr(config, 'exp_name', 'run'),
    )
    if _total_steps is not None:
        print(f"Step budget: {config.num_epochs} epochs x {config.steps_per_epoch} steps "
              f"= {_total_steps} optimizer steps (batch {config.batch_size}); "
              f"warm-up {int(getattr(config, 'warmup_steps', 0) or 0)}")

    for epoch in range(config.num_epochs):
        # Track metrics for the epoch
        epoch_loss = 0.0
        epoch_nmse = 0.0
        epoch_active = 0.0
        epoch_binary_purity = 0.0
        purity_batches = 0
        num_batches = 0
        
        pbar = tqdm(_batcher.epoch(), total=len(_batcher),
                    desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch in pbar:
            features, class_idx, _ = unpack_batch(batch)
            x = features.to(config.device)
            if class_idx is not None:
                class_idx = class_idx.to(config.device)
            
            # Forward pass
            latents_pre_act, latents, recons = autoencoder(x)
            
            # Decide mono_coef for this step
            step_mono_coef = config.mono_coef
            if config.mono_coef > 0 and config.mono_period > 1:
                # Periodic mode: only compute mono loss every N steps
                step_mono_coef = 0.0  # Skip batch-level mono loss
            
            # Compute loss using model's compute_loss method
            loss, metrics = autoencoder.compute_loss(
                x.float(),
                latents_pre_act.float(),
                latents.float(),
                recons.float(),
                auxk_coef=config.auxk_coef,
                dead_steps_threshold=dead_steps_threshold_steps,
                auxk_k=config.auxk_k,
                mono_coef=step_mono_coef
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Periodic full-dataset mono loss via custom autograd (if enabled and it's time)
            if config.mono_coef > 0 and config.mono_period > 1 and (global_step + 1) % config.mono_period == 0:
                print(f"\nComputing full dataset mono loss (custom autograd) at step {global_step + 1}...")
                m = compute_monosemanticity_custom(autoencoder, mono_loader, device=config.device, eps=1e-8, verbose=False)
                active_mask = (m != 0)
                active_count = int(active_mask.sum().item())
                denom = max(active_count, 1)
                # Backprop coef * (1 - mean_active(m))
                scalar = (-config.mono_coef / denom) * m[active_mask].sum()
                scalar.backward()

                if active_count > 0:
                    mono_mean = m[active_mask].mean().item()
                    mono_median = m[active_mask].median().item()
                else:
                    mono_mean = 0.0
                    mono_median = 0.0
                mono_metrics = {
                    'mono_score': mono_mean,
                    'mono_median': mono_median,
                    'mono_active_neurons': active_count,
                }
                metrics.update(mono_metrics)
                print(f"Full dataset mono score: {mono_mean:.4f}")

            # Batch monoscore is logged independently of the training mono objective.
            with torch.no_grad():
                _, batch_mono_metrics = compute_monosemanticity_loss_batch(x.float(), latents.float())
            batch_mono_step = float(batch_mono_metrics['mono_score'])
            
            # Apply decoder constraints for TopKSAE
            unit_norm_decoder_(autoencoder)
            unit_norm_decoder_grad_adjustment_(autoencoder)
            
            # Optimizer step
            optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_nmse += metrics.get('nmse_loss', metrics.get('l2_loss', 0))
            epoch_active += metrics['num_active']

            binary_purity = _mean_active_binary_purity(latents, class_idx)
            if binary_purity is not None:
                epoch_binary_purity += binary_purity
                purity_batches += 1

            num_batches += 1
            
            # Update progress bar
            dead_mask = autoencoder.stats_last_nonzero > dead_steps_threshold_steps
            num_dead = dead_mask.sum().item()
            
            l0 = metrics.get('num_active', 0)
            expected_active = config.topk_k if config.activation == 'topk' else config.n_latents
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4e}",
                'active': f"{l0:.1f}/{expected_active}",
                'dead': num_dead
            })
            
            # Check and reinitialize dead neurons
            if global_step > 0 and global_step % config.dead_check_interval == 0:
                dead_mask = autoencoder.stats_last_nonzero > dead_steps_threshold_steps
                num_dead = dead_mask.sum().item()
                
                if num_dead > 0:
                    print(f"\nReinitializing {num_dead} dead neurons")
                    reinitialize_dead_neurons(autoencoder, dead_mask, sample_batch, config.device)
            
            global_step += 1

            # Evaluate dataset monoscore at each step and track its correlation with batch monoscore.
            # val_mono_mean_step = float('nan')
            # val_mono_median_step = float('nan')
            # if val_loader is not None:
            #     val_mono_step = compute_monosemanticity_fast(autoencoder, val_loader, config.device, 'val', verbose=False)
            #     val_mono_np_step = val_mono_step.cpu().numpy()
            #     val_mono_mean_step = float(np.mean(val_mono_np_step))
            #     val_mono_median_step = float(np.median(val_mono_np_step))

            # if np.isfinite(batch_mono_step) and np.isfinite(val_mono_mean_step):
            #     batch_mono_history.append(batch_mono_step)
            #     dataset_mono_history.append(val_mono_mean_step)

            # mono_corr_batch_vs_dataset_step = float('nan')
            # if len(batch_mono_history) >= 2:
            #     batch_arr = np.asarray(batch_mono_history, dtype=np.float64)
            #     dataset_arr = np.asarray(dataset_mono_history, dtype=np.float64)
            #     if np.std(batch_arr) > 1e-12 and np.std(dataset_arr) > 1e-12:
            #         mono_corr_batch_vs_dataset_step = float(np.corrcoef(batch_arr, dataset_arr)[0, 1])
            #     else:
            #         mono_corr_batch_vs_dataset_step = 0.0

            # Log per-step metrics to W&B and/or TensorBoard (same dict, indexed by global_step).
            step_log = {
                'train_loss_step': loss.item(),
                'train_nmse_step': metrics.get('nmse_loss', metrics.get('l2_loss', 0)),
                'train_active_step': metrics['num_active'],
                'train_binary_purity_step': binary_purity if binary_purity is not None else 0.0,
                'batch_mono_mean_step': batch_mono_step,
            }
            if config.wandb_project:
                wandb.log(step_log, step=global_step)
            tb.log(step_log, step=global_step)
        
        # Epoch-end logging
        avg_loss = epoch_loss / num_batches
        avg_nmse = epoch_nmse / num_batches
        avg_active = epoch_active / num_batches
        avg_binary_purity = epoch_binary_purity / purity_batches if purity_batches > 0 else 0.0
        
        print(
            f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {avg_loss:.4e}, "
            f"Train NMSE: {avg_nmse:.4e}, Train Active: {avg_active:.1f}, "
            f"Train Binary Purity: {avg_binary_purity:.4f}"
        )
        
        # Validation at end of epoch
        if val_loader:
            val_metrics = evaluate(autoencoder, val_loader, config, 'val')
            print(f"Epoch {epoch+1}/{config.num_epochs} - Val Loss: {val_metrics['val_loss']:.4e}, "
                  f"Val NMSE: {val_metrics['val_nmse']:.4e}, "
                f"Val Active: {val_metrics['val_active']:.1f}, "
                f"Val Binary Purity: {val_metrics['val_binary_purity']:.4f}")
        
        # R2 at end of epoch. Train R2 streams the full 1.28M train set (expensive) -> only when
        # not stripping extras. Val R2 (50K) is cheap and used in best-ckpt metrics, so keep it.
        train_r2 = float('nan')
        if not config.skip_extra_eval:
            train_r2 = compute_r2(autoencoder, train_loader, config.device)
            print(f"Epoch {epoch+1}/{config.num_epochs} - Train R2: {train_r2:.6f}")
        if val_loader:
            val_r2 = compute_r2(autoencoder, val_loader, config.device)
            print(f"Epoch {epoch+1}/{config.num_epochs} - Val   R2: {val_r2:.6f}")

        # Train monosemanticity (over 1.28M) is logging-only -> skip when stripping extras.
        # (Val MonoScore for best-ckpt selection is computed below regardless.)
        if not config.skip_extra_eval:
            train_mono = compute_monosemanticity_fast(autoencoder, train_loader, config.device, 'train', verbose=False)
            train_mono_np = train_mono.cpu().numpy()
            train_mono_mean = np.mean(train_mono_np)
            train_mono_median = np.median(train_mono_np)
            print(f"Epoch {epoch+1}/{config.num_epochs} - Train Mono: mean={train_mono_mean:.4f}, median={train_mono_median:.4f}")
        else:
            train_mono_mean = float('nan')
            train_mono_median = float('nan')
        
        epoch_log_data = {
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_nmse': avg_nmse,
            'train_active': avg_active,
            'train_binary_purity': avg_binary_purity,
            'train_r2': train_r2,
            'train_mono_mean': train_mono_mean,
            'train_mono_median': train_mono_median,
        }
        
        if val_loader:
            val_mono = compute_monosemanticity_fast(autoencoder, val_loader, config.device, 'val', verbose=False)
            val_mono_np = val_mono.cpu().numpy()
            val_mono_mean = np.mean(val_mono_np)
            val_mono_median = np.median(val_mono_np)
            print(f"Epoch {epoch+1}/{config.num_epochs} - Val Mono: mean={val_mono_mean:.4f}, median={val_mono_median:.4f}")

            if val_mono_mean > best_val_mono:
                best_val_mono = float(val_mono_mean)
                best_val_mono_epoch = epoch + 1
                best_val_mono_metrics = {
                    'val_mono_mean': float(val_mono_mean),
                    'val_r2': float(val_r2),
                    'val_binary_purity': float(val_metrics['val_binary_purity']),
                }
                # Self-describing checkpoint: store model_cfg so it can be reconstructed without
                # the training config (mirrors train_dual.py / train_other.py).
                topk_model_cfg = {
                    'n_latents': config.n_latents,
                    'd_model': config.d_model,
                    'topk_k': config.topk_k,
                    'tied': config.tied_weights,
                    'normalize': config.normalize,
                    'activation': config.activation,  # 'topk' or 'relu'
                }
                torch.save(
                    {
                        'epoch': best_val_mono_epoch,
                        'val_monoscore': best_val_mono,
                        'model_state_dict': autoencoder.state_dict(),
                        'model_cfg': topk_model_cfg,
                    },
                    best_val_model_path,
                )
                if not config.skip_extra_eval:
                    save_intermediate_outputs(
                        autoencoder,
                        val_loader,
                        best_val_intermediates_path,
                        device=config.device,
                        split_name='val'
                    )
                    print(
                        f"Saved best val monoscore intermediates "
                        f"(epoch={best_val_mono_epoch}, score={best_val_mono:.4f}) "
                        f"to {best_val_intermediates_path}"
                    )
                print(f"Saved best val monoscore model to {best_val_model_path}")
                print(
                    "Best-monoscore val metrics - "
                    f"mono={best_val_mono_metrics['val_mono_mean']:.4f}, "
                    f"r2={best_val_mono_metrics['val_r2']:.6f}, "
                    f"purity={best_val_mono_metrics['val_binary_purity']:.4f}"
                )
            
            epoch_log_data.update({
                **val_metrics,
                'val_r2': val_r2,
                'val_mono_mean': val_mono_mean,
                'val_mono_median': val_mono_median,
            })
        
        if config.wandb_project:
            wandb.log(epoch_log_data)
        tb.log(epoch_log_data, step=epoch)   # TensorBoard: per-epoch scalars (step=epoch)

        print("")

    
    # Save final model
    final_path = Path(config.output_dir) / 'autoencoder_final.pt'
    torch.save({
        'epoch': config.num_epochs,
        'step': global_step,
        'model_state_dict': autoencoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(config),
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

    # When stripping extras, skip the heavy best-model re-evaluation (R2/MonoScore/KNN/LP over
    # 1.28M train). The best_val_monoscore_model.pt is already saved; branch_metrics.py can score it.
    if config.skip_extra_eval:
        # best_metrics.json is written on this path too, with only what this mode measured:
        # the selected epoch's own validation pass (KNN/LP and the test split come from the
        # skipped best-checkpoint re-evaluation, so those keys are absent -- a missing key
        # means "not computed", never zero).
        if best_val_mono_metrics is not None:
            write_best_metrics(config.output_dir, {
                'BEST_MONOSCORE': best_val_mono,
                'BEST_EPOCH': best_val_mono_epoch,
                'BEST_R2': best_val_mono_metrics['val_r2'],
                'BEST_PURITY': best_val_mono_metrics['val_binary_purity'],
                'val_mono_mean': best_val_mono,
                'skip_extra_eval': True,
            })
        train_dataset.close()
        if val_loader:
            val_dataset.close()
        if test_loader:
            test_dataset.close()
        if config.wandb_project:
            wandb.finish()
        tb.close()   # flush + close TB writer on the early (skip_extra_eval) exit — the live path
        return autoencoder

    # EVALUATION ON THE EPOCHS WITH BEST VAL MONOSCORE
    best_ckpt = torch.load(best_val_model_path, map_location=config.device)
    if config.activation == 'topk':
        best_activation = TopK(k=config.topk_k)
    else:
        best_activation = nn.ReLU()
    best_model = TopKSAE(
        n_latents=config.n_latents,
        n_inputs=config.d_model,
        activation=best_activation,
        tied=config.tied_weights,
        normalize=config.normalize,
    ).to(config.device)
    best_model.load_state_dict(best_ckpt['model_state_dict'], strict=False)
    best_model.eval()

    # EVALUATE R2
    best_val_r2 = compute_r2(best_model, val_loader, config.device)

    # EVALUATE MONOSEMANTICITY
    best_mono = compute_monosemanticity_fast(best_model, val_loader, config.device, 'val', verbose=True)
    best_mono_mean = best_mono.cpu().numpy().mean()



    # EVALUATE ACC(KNN) AND ACC(LP)
    val_clf_metrics = evaluate_reconstruction_classification(
        best_model,
        train_loader,
        val_loader,
        device=config.device,
        split_name='val',
    )

    if run is not None:
        run.summary['BEST_MONOSCORE'] = best_mono_mean
        run.summary['BEST_EPOCH'] = best_val_mono_epoch
        run.summary['BEST_R2'] = best_val_r2
        run.summary['BEST_PURITY'] = best_val_mono_metrics['val_binary_purity']
        run.summary['BEST_ACC_KNN'] = val_clf_metrics['val_knn_acc_recon']
        run.summary['BEST_ACC_LP'] = val_clf_metrics['val_lp_acc_recon']
    tb.summary({                                 # TensorBoard: best/* scalars at the selected epoch
        'MONOSCORE': best_mono_mean, 'EPOCH': best_val_mono_epoch, 'R2': best_val_r2,
        'PURITY': best_val_mono_metrics['val_binary_purity'],
        'ACC_KNN': val_clf_metrics['val_knn_acc_recon'], 'ACC_LP': val_clf_metrics['val_lp_acc_recon'],
    }, step=best_val_mono_epoch)

    # Same numbers on disk, keyed exactly like the W&B summary above, so the collectors need
    # no experiment tracker. On multi-label OpenImagesV7 the KNN/LP entries hold micro-F1
    # (the trainers switch metric with the dataset; the key names stay the same). Written
    # before the optional test pass, so an interruption there still leaves the val metrics.
    best_metrics_record = {
        'BEST_MONOSCORE': best_mono_mean,
        'BEST_EPOCH': best_val_mono_epoch,
        'BEST_R2': best_val_r2,
        'BEST_PURITY': best_val_mono_metrics['val_binary_purity'],
        'BEST_ACC_KNN': val_clf_metrics['val_knn_acc_recon'],
        'BEST_ACC_LP': val_clf_metrics['val_lp_acc_recon'],
        'val_mono_mean': best_mono_mean,
        'skip_extra_eval': False,
    }
    write_best_metrics(config.output_dir, best_metrics_record)

    if test_loader is not None:
        print("Evaluating best-val-monoscore model on test split...")
        best_test_r2 = compute_r2(best_model, test_loader, config.device)
        best_test_mono = compute_monosemanticity_fast(best_model, test_loader, config.device, 'test', verbose=True)
        best_test_mono_mean = float(best_test_mono.cpu().numpy().mean())

        test_clf_metrics = evaluate_reconstruction_classification(
            best_model,
            train_loader,
            test_loader,
            device=config.device,
            split_name='test',
        )

        print(
            f"Best-epoch test metrics - mono={best_test_mono_mean:.4f}, "
            f"r2={best_test_r2:.6f}, "
            f"knn={test_clf_metrics['test_knn_acc_recon']:.4f}, "
            f"lp={test_clf_metrics['test_lp_acc_recon']:.4f}"
        )

        if run is not None:
            run.summary['BEST_TEST_MONOSCORE'] = best_test_mono_mean
            run.summary['BEST_TEST_R2'] = best_test_r2
            run.summary['BEST_TEST_ACC_KNN'] = test_clf_metrics['test_knn_acc_recon']
            run.summary['BEST_TEST_ACC_LP'] = test_clf_metrics['test_lp_acc_recon']
            wandb.log({
                'best_test_mono_mean': best_test_mono_mean,
                'best_test_r2': best_test_r2,
                'best_test_knn_metric_recon': test_clf_metrics.get('test_knn_metric_recon', test_clf_metrics['test_knn_acc_recon']),
                'best_test_lp_metric_recon': test_clf_metrics.get('test_lp_metric_recon', test_clf_metrics['test_lp_acc_recon']),
                'best_test_knn_acc_recon': test_clf_metrics['test_knn_acc_recon'],
                'best_test_lp_acc_recon': test_clf_metrics['test_lp_acc_recon'],
            })
        tb.log({                                 # TensorBoard: test metrics (use final epoch as step)
            'best_test_mono_mean': best_test_mono_mean, 'best_test_r2': best_test_r2,
            'best_test_knn_acc_recon': test_clf_metrics['test_knn_acc_recon'],
            'best_test_lp_acc_recon': test_clf_metrics['test_lp_acc_recon'],
        }, step=config.num_epochs)
        # Re-write the metrics file with the test columns appended.
        best_metrics_record.update({
            'BEST_TEST_MONOSCORE': best_test_mono_mean,
            'BEST_TEST_R2': best_test_r2,
            'BEST_TEST_ACC_KNN': test_clf_metrics['test_knn_acc_recon'],
            'BEST_TEST_ACC_LP': test_clf_metrics['test_lp_acc_recon'],
        })
        write_best_metrics(config.output_dir, best_metrics_record)

    # # Final evaluation on all splits
    # print("\n" + "="*60)
    # print("FINAL EVALUATION")
    # print("="*60)
    
    # # Dictionary to collect all final results
    # final_results = {
    #     'config': vars(config),
    #     'final_epoch': config.num_epochs,
    #     'final_step': global_step,
    # }
    # if best_val_mono_epoch is not None:
    #     final_results.update({
    #         'best_val_monoscore': best_val_mono,
    #         'best_val_monoscore_epoch': best_val_mono_epoch,
    #         'best_val_monoscore_intermediates_path': str(best_val_intermediates_path),
    #         'best_val_monoscore_model_path': str(best_val_model_path),
    #     })
    #     if best_val_mono_metrics is not None:
    #         final_results.update({
    #             'best_val_monoscore_val_mono_mean': best_val_mono_metrics['val_mono_mean'],
    #             'best_val_monoscore_val_r2': best_val_mono_metrics['val_r2'],
    #             'best_val_monoscore_val_binary_purity': best_val_mono_metrics['val_binary_purity'],
    #         })
    #         if run is not None:
    #             run.summary['best_val_monoscore'] = best_val_mono_metrics['val_mono_mean']
    #             run.summary['best_val_monoscore_epoch'] = best_val_mono_epoch
    #             run.summary['best_val_monoscore_val_r2'] = best_val_mono_metrics['val_r2']
    #             run.summary['best_val_monoscore_val_binary_purity'] = best_val_mono_metrics['val_binary_purity']
    
    # # Train split
    # print("Evaluating on train split...")
    # train_metrics = evaluate(autoencoder, train_loader, config, 'train')
    # print(f"Train - Loss: {train_metrics['train_loss']:.2e}, "
    #       f"NMSE: {train_metrics['train_nmse']:.2e}, "
    #         f"Active: {train_metrics['train_active']:.1f}, "
    #         f"Binary Purity: {train_metrics['train_binary_purity']:.4f}")
    # final_results.update(train_metrics)
    
    # # Val split
    # if val_loader:
    #     print("Evaluating on val split...")
    #     val_metrics = evaluate(autoencoder, val_loader, config, 'val')
    #     print(f"Val   - Loss: {val_metrics['val_loss']:.2e}, "
    #           f"NMSE: {val_metrics['val_nmse']:.2e}, "
    #           f"Active: {val_metrics['val_active']:.1f}, "
    #           f"Binary Purity: {val_metrics['val_binary_purity']:.4f}")
    #     final_results.update(val_metrics)
        
    #     if config.wandb_project:
    #         wandb.log({**val_metrics})
    
    # # Test split
    # if test_loader:
    #     print("Evaluating on test split...")
    #     test_metrics = evaluate(autoencoder, test_loader, config, 'test')
    #     print(f"Test  - Loss: {test_metrics['test_loss']:.2e}, "
    #           f"NMSE: {test_metrics['test_nmse']:.2e}, "
    #           f"Active: {test_metrics['test_active']:.1f}, "
    #           f"Binary Purity: {test_metrics['test_binary_purity']:.4f}")
    #     final_results.update(test_metrics)
        
    #     if config.wandb_project:
    #         wandb.log({**test_metrics})
    
    # # Compute R2 scores for all splits
    # print("\n" + "="*60)
    # print("R2 SCORES")
    # print("="*60)
    
    # print("Computing train R2 score...")
    # train_r2_final = compute_r2(autoencoder, train_loader, config.device)
    # print(f"Train - R2: {train_r2_final:.6f}")
    # final_results['train_r2'] = train_r2_final
    
    # if config.wandb_project:
    #     wandb.log({'final_train_r2': train_r2_final})
    
    # if val_loader:
    #     print("Computing val R2 score...")
    #     val_r2_final = compute_r2(autoencoder, val_loader, config.device)
    #     print(f"Val   - R2: {val_r2_final:.6f}")
    #     final_results['val_r2'] = val_r2_final
        
    #     if config.wandb_project:
    #         wandb.log({'final_val_r2': val_r2_final})
    
    # if test_loader:
    #     print("Computing test R2 score...")
    #     test_r2_final = compute_r2(autoencoder, test_loader, config.device)
    #     print(f"Test  - R2: {test_r2_final:.6f}")
    #     final_results['test_r2'] = test_r2_final
        
    #     if config.wandb_project:
    #         wandb.log({'final_test_r2': test_r2_final})
    
    # # Compute monosemanticity scores
    # print("\n" + "="*60)
    # print("MONOSEMANTICITY EVALUATION")
    # print("="*60)
    
    # print("Computing train monosemanticity score...")
    # train_mono = compute_monosemanticity_fast(autoencoder, train_loader, config.device, 'train', verbose=True)

    # # Compute statistics: median, mean, 90th percentile
    # train_mono_np = train_mono.cpu().numpy()
    # train_mono_mean = np.mean(train_mono_np)
    # train_mono_median = np.median(train_mono_np)
    # train_mono_p90 = np.percentile(train_mono_np, 90)
    
    # print(f"Train - Monosemanticity: mean={train_mono_mean:.4f}, median={train_mono_median:.4f}, 90th={train_mono_p90:.4f}")
    # final_results['train_monosemanticity_mean'] = float(train_mono_mean)
    # final_results['train_monosemanticity_median'] = float(train_mono_median)
    # final_results['train_monosemanticity_p90'] = float(train_mono_p90)
    
    # if config.wandb_project:
    #     wandb.log({
    #         'train_monosemanticity_mean': train_mono_mean,
    #         'train_monosemanticity_median': train_mono_median,
    #         'train_monosemanticity_p90': train_mono_p90
    #     })
    
    # if val_loader:
    #     print("Computing val monosemanticity score (fast)...")
    #     val_mono = compute_monosemanticity_fast(autoencoder, val_loader, config.device, 'val', verbose=True)
        
    #     # Compute statistics: median, mean, 90th percentile
    #     val_mono_np = val_mono.cpu().numpy()
    #     val_mono_mean = np.mean(val_mono_np)
    #     val_mono_median = np.median(val_mono_np)
    #     val_mono_p90 = np.percentile(val_mono_np, 90)
        
    #     print(f"Val   - Monosemanticity: mean={val_mono_mean:.4f}, median={val_mono_median:.4f}, 90th={val_mono_p90:.4f}")
    #     final_results['val_monosemanticity_mean'] = float(val_mono_mean)
    #     final_results['val_monosemanticity_median'] = float(val_mono_median)
    #     final_results['val_monosemanticity_p90'] = float(val_mono_p90)
        
    #     if config.wandb_project:
    #         wandb.log({
    #             'val_monosemanticity_mean': val_mono_mean,
    #             'val_monosemanticity_median': val_mono_median,
    #             'val_monosemanticity_p90': val_mono_p90
    #         })
    
    # if test_loader:
    #     print("Computing test monosemanticity score (fast)...")
    #     test_mono = compute_monosemanticity_fast(autoencoder, test_loader, config.device, 'test', verbose=True)
        
    #     # Compute statistics: median, mean, 90th percentile
    #     test_mono_np = test_mono.cpu().numpy()
    #     test_mono_mean = np.mean(test_mono_np)
    #     test_mono_median = np.median(test_mono_np)
    #     test_mono_p90 = np.percentile(test_mono_np, 90)
        
    #     print(f"Test  - Monosemanticity: mean={test_mono_mean:.4f}, median={test_mono_median:.4f}, 90th={test_mono_p90:.4f}")
    #     final_results['test_monosemanticity_mean'] = float(test_mono_mean)
    #     final_results['test_monosemanticity_median'] = float(test_mono_median)
    #     final_results['test_monosemanticity_p90'] = float(test_mono_p90)
        
    #     if config.wandb_project:
    #         wandb.log({
    #             'test_monosemanticity_mean': test_mono_mean,
    #             'test_monosemanticity_median': test_mono_median,
    #             'test_monosemanticity_p90': test_mono_p90
    #         })

    # # Downstream classification on reconstructed embeddings from best-monoscore epoch.
    # if best_val_mono_epoch is not None and val_loader is not None and best_val_model_path.exists():
    #     print("\n" + "="*60)
    #     print("RECONSTRUCTED-EMBEDDING CLASSIFICATION (BEST MONOSCORE EPOCH)")
    #     print("="*60)

    #     best_ckpt = torch.load(best_val_model_path, map_location=config.device)
    #     if config.activation == 'topk':
    #         best_activation = TopK(k=config.topk_k)
    #     else:
    #         best_activation = nn.ReLU()
    #     best_model = TopKSAE(
    #         n_latents=config.n_latents,
    #         n_inputs=config.d_model,
    #         activation=best_activation,
    #         tied=config.tied_weights,
    #         normalize=config.normalize,
    #     ).to(config.device)
    #     best_model.load_state_dict(best_ckpt['model_state_dict'], strict=False)
    #     best_model.eval()

    #     train_eval_loader = DataLoader(
    #         train_dataset,
    #         batch_size=config.batch_size,
    #         shuffle=False,
    #         num_workers=config.num_workers,
    #         pin_memory=True,
    #         drop_last=False,
    #     )

    #     val_clf_metrics = evaluate_reconstruction_classification(
    #         best_model,
    #         train_eval_loader,
    #         val_loader,
    #         device=config.device,
    #         split_name='val',
    #     )
    #     final_results.update({
    #         'val_knn_acc_recon_best_mono': val_clf_metrics['val_knn_acc_recon'],
    #         'val_lp_acc_recon_best_mono': val_clf_metrics['val_lp_acc_recon'],
    #     })
    #     if run is not None:
    #         run.summary['val_knn_acc_recon_best_mono'] = val_clf_metrics['val_knn_acc_recon']
    #         run.summary['val_lp_acc_recon_best_mono'] = val_clf_metrics['val_lp_acc_recon']

    #     if test_loader is not None:
    #         test_clf_metrics = evaluate_reconstruction_classification(
    #             best_model,
    #             train_eval_loader,
    #             test_loader,
    #             device=config.device,
    #             split_name='test',
    #         )
    #         final_results.update({
    #             'test_knn_acc_recon_best_mono': test_clf_metrics['test_knn_acc_recon'],
    #             'test_lp_acc_recon_best_mono': test_clf_metrics['test_lp_acc_recon'],
    #         })
    #         if run is not None:
    #             run.summary['test_knn_acc_recon_best_mono'] = test_clf_metrics['test_knn_acc_recon']
    #             run.summary['test_lp_acc_recon_best_mono'] = test_clf_metrics['test_lp_acc_recon']
    
    # print("="*60)
    
    # # Save all results to JSON file
    # results_path = Path(config.output_dir) / 'results.json'
    # print(f"\nSaving results to {results_path}")
    # with open(results_path, 'w') as f:
    #     json.dump(final_results, f, indent=2)
    # print(f"Results saved successfully!")
    
    # Close datasets
    train_dataset.close()
    if val_loader:
        val_dataset.close()
    if test_loader:
        test_dataset.close()

    if config.wandb_project:
        wandb.finish()
    tb.close()   # flush + close TB writer on the full-eval exit

    return autoencoder


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train a TopK SAE")
    parser.add_argument('--dataset_config', type=str, required=True,
                        help='JSON config with train_path/val_path/test_path (LMDB dirs)')
    parser.add_argument('--exp_name', type=str, default='topk_sae')
    parser.add_argument('--n_latents', type=int, default=8192)
    parser.add_argument('--topk_k', type=int, default=64)
    parser.add_argument('--activation', type=str, default='topk', choices=['topk', 'relu'])
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eps', type=float, default=6.25e-10)
    parser.add_argument('--auxk_coef', type=float, default=1/32)
    parser.add_argument('--mono_coef', type=float, default=0.0)
    parser.add_argument('--dead_steps_threshold', type=int, default=10_000_000)
    parser.add_argument('--dead_check_interval', type=int, default=1000)
    parser.add_argument('--skip_extra_eval', action='store_true', default=False,
                        help='Skip heavy per-epoch/end eval; keep training + val-MonoScore selection.')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--tb_dir', type=str, default=None,
                       help='TensorBoard event dir (real-time offline). No-op if unset.')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    with open(args.dataset_config, 'r') as f:
        config_dict = json.load(f)

    config = TrainingConfig(
        train_path=config_dict.get('train_path'),
        val_path=config_dict.get('val_path'),
        test_path=config_dict.get('test_path'),
        exp_name=args.exp_name,
        n_latents=args.n_latents,
        topk_k=args.topk_k,
        activation=args.activation,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        eps=args.eps,
        auxk_coef=args.auxk_coef,
        mono_coef=args.mono_coef,
        dead_steps_threshold=args.dead_steps_threshold,
        dead_check_interval=args.dead_check_interval,
        skip_extra_eval=args.skip_extra_eval,
        num_workers=args.num_workers,
        seed=args.seed,
        wandb_project=args.wandb_project,
        tb_dir=args.tb_dir,
        output_dir=args.output_dir,
    )

    train_autoencoder(config)
