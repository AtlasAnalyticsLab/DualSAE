import math
import numbers
import os
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
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
from models import BatchTopKSAE, VanillaSAE, JumpReLUSAE, GatedSAE, MatryoshkaSAE, MSAE
from models.msae_sae import build_nesting, matryoshka_weights
from loss import compute_monosemanticity_fast, compute_monosemanticity_loss_batch
from mono_loss import compute_monosemanticity_custom
from metrics import compute_binary_purity, compute_multilabel_binary_purity
from utils import PersistentBatcher, config_payload, resolve_step_budget, unpack_batch


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
    """Compute KNN classification metric using reconstructed embeddings x_hat.

    Returns:
        - single-label: top-1 accuracy
        - multi-label: micro-F1 with thresholded averaged neighbor label scores
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
    """Train linear probe on reconstructed train embeddings and evaluate on split.

    Returns:
        - single-label: top-1 accuracy
        - multi-label: micro-F1
    """
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
    # Metric naming is kept backward-compatible for existing logging code.
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
    total_l2 = 0
    total_active = 0
    total_binary_purity = 0
    purity_batches = 0
    num_batches = 0

    acts_all =[]
    class_idx_all = []
    
    for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
        features, class_idx, _ = unpack_batch(batch)
        x = features.to(config.device)
        if class_idx is not None:
            class_idx = class_idx.to(config.device)
      
        # All models return (acts/latents_pre_act, latents, recons)
        latents_pre_act, latents, recons = autoencoder(x)
        
        # Compute loss using model's compute_loss method
        loss, metrics = autoencoder.compute_loss(
            x.float(),
            latents_pre_act.float(),
            latents.float(),
            recons.float(),
            mono_coef=0.0
        )
        
        total_loss += loss.item()
        total_l2 += metrics.get('l2_loss', 0)
        total_active += metrics['num_active']

        acts_all.append(latents.detach().cpu())
        if class_idx is not None:
            class_idx_all.append(class_idx.detach().cpu())

        num_batches += 1
    
    autoencoder.train()
    
    avg_loss = total_loss / num_batches
    avg_l2 = total_l2 / num_batches
    avg_active = total_active / num_batches
    
    acts = torch.cat(acts_all, dim=0)
    if len(class_idx_all) > 0:
        class_idx = torch.cat(class_idx_all, dim=0)
        avg_binary_purity = _mean_active_binary_purity(acts, class_idx)
    else:
        avg_binary_purity = 0.0
    
    return {
        f'{split_name}_loss': avg_loss,
        f'{split_name}_l2': avg_l2,
        f'{split_name}_active': avg_active,
        f'{split_name}_binary_purity': avg_binary_purity
    }


@torch.no_grad()
def compute_r2(autoencoder, dataloader, device='cuda', eps: float = 1e-12, split_name='data'):
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

    for batch in tqdm(dataloader, desc=f"Computing R2 {split_name}"):
        features, _, _ = unpack_batch(batch)
        x = features.to(device)

        # All models return (acts/latents_pre_act, latents, recons)
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
    model_type: str  # 'batch_topk', 'vanilla', or 'jumprelu'
        
    # Data (optional)
    val_path: Optional[str] = None
    test_path: Optional[str] = None

    # Experiment
    exp_name: str = 'default'
    
    # Model architecture
    model_type: str = 'vanilla'  # 'batch_topk', 'vanilla', or 'jumprelu'
    n_latents: int = 8192
    d_model: Optional[int] = None  # Will be inferred from data
    topk_k: int = 64  # For batch_topk
    normalize: bool = False  # Layer normalization
    
    # Training
    batch_size: int = 4096
    num_epochs: int = 10
    lr: float = 1e-4
    eps: float = 6.25e-10
    max_grad_norm: float = 1.0
    
    # Loss coefficients
    auxk_coef: float = 1/32  # For batch_topk auxiliary loss
    auxk_k: Optional[int] = None  # For batch_topk (if None, uses 512)
    l1_coef: float = 0.0001  # Coefficient for L1/L0 sparsity penalty
    bandwidth: float = 0.001  # Bandwidth for JumpReLU activation
    # JumpReLU threshold init theta (model_type 'jumprelu' only). None == leave the model default
    # (theta = 1.0) in charge AND add no key to model_cfg. When None it is also dropped from
    # training_config.json / the final ckpt 'config' / W&B (utils.config_payload,
    # OPT_IN_NONE_CONFIG_FIELDS).
    jump_threshold_init: Optional[float] = None
    mono_coef: float = 0.0  # Coefficient for monosemanticity loss
    mono_period: int = 1  # Compute mono loss every N steps (1=per batch)
    
    # Dead neuron handling (tracked by models)
    dead_steps_threshold: int = 10_000_000  # In tokens/samples

    # Gated-SAE knob (model_type 'gated' only; inert otherwise). Linear ramp
    # of the gate-path L1 over this many optimizer steps; 0 disables the ramp.
    # Reference default (dictionary_learning GatedSAETrainer) is 2000.
    gated_sparsity_warmup: int = 2000

    # Matryoshka-family knobs (model_type 'matryoshka' / 'msae' only; None
    # values fall back to the official constructions at model-build time).
    group_fractions: Optional[list] = None   # Bussmann dictionary-group fractions
    msae_nesting: Optional[list] = None      # MSAE sparsity levels (None -> build_nesting)
    msae_weighting: str = 'rw'               # MSAE per-level weights: rw|uw
    msae_soft_cap: float = 30.0              # MSAE SoftCapping (<=0 disables)

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


def _matryoshka_group_sizes(config: TrainingConfig) -> list:
    """Bussmann group sizes from fractions — the vendored trainer's exact
    arithmetic (matryoshka_batch_top_k.py:168-172): int(f*dict_size) for all
    but the last fraction, remainder in the last group. The vendored code has
    NO default fractions; [1/32,1/16,1/8,1/4,1/2+1/32] is the upstream
    dictionary_learning config convention, mirrored in main.py's flag default."""
    from math import isclose
    fractions = config.group_fractions or [1/32, 1/16, 1/8, 1/4, 1/2 + 1/32]
    if not isclose(sum(fractions), 1.0):
        raise ValueError(f"group_fractions must sum to 1.0: {fractions}")
    sizes = [int(f * config.n_latents) for f in fractions[:-1]]
    sizes.append(config.n_latents - sum(sizes))
    return sizes


def _msae_nesting_and_weights(config: TrainingConfig):
    """MSAE nesting levels + per-level weights: official constructions unless
    overridden (build_nesting(k, 256) doubles k up to 256 -> [64,128,256])."""
    nesting = config.msae_nesting or build_nesting(config.topk_k, 256)
    weights = matryoshka_weights(len(nesting), config.msae_weighting)
    return nesting, weights


def _optimizer_overrides(model) -> Dict[str, Any]:
    """Per-MODEL optimizer/schedule requests, or {} when the model has none.

    WHY a hook on the model instead of an `if config.model_type == ...` chain
    in this trainer: the knowledge "this architecture is trained with
    Adam(beta1=0, ...) [and a schedule]" belongs to the model, not to the
    launcher. Consequences that an if-chain would get wrong:
      * every consumer that builds an optimizer for the model (this trainer,
        a notebook, a future trainer) gets the same treatment automatically —
        an if-chain has to be copied and drifts;
      * a model_type string typo in an if-chain silently trains with the
        DEFAULT optimizer and nothing fails, while a model that simply has no
        hook is a documented no-op;
      * the override travels with the class, so it cannot get out of sync with
        the architecture it belongs to.
    Contract: `model.optimizer_overrides()` returns a dict that may contain
      'betas'       -> (beta1, beta2) for torch.optim.Adam
      'schedule'    -> an LR-schedule spec (see _build_lr_schedule)
      'warmup_frac' -> schedule parameters may also be given FLAT next to
      'min_lr_frac'    'schedule' instead of nested in a dict spec; they are
                       folded into the spec below (the nested spelling wins if
                       both are present, since it is the more explicit one).
    Models without the attribute are COMPLETELY unaffected: no hook, no call,
    no behaviour change (the default Adam(lr, eps) of every existing row).
    Anything other than a dict/None is a programming error and raises — a
    silently ignored override would train the wrong optimizer for 50 epochs.
    """
    hook = getattr(model, "optimizer_overrides", None)
    if not callable(hook):
        return {}
    overrides = hook()
    if overrides is None:
        return {}
    if not isinstance(overrides, dict):
        raise TypeError(
            f"{type(model).__name__}.optimizer_overrides() must return a dict "
            f"or None, got {type(overrides).__name__}")
    overrides = dict(overrides)
    # Fold flat schedule parameters into the schedule spec so both spellings
    # mean the same thing:  {'schedule': 'cosine', 'warmup_frac': 0.1}  ==
    # {'schedule': {'name': 'cosine', 'warmup_frac': 0.1}}. Without this the
    # flat form would be rejected as an unknown key and a model asking for the
    # paper's warmup could not be trained at all.
    flat_schedule_keys = {"warmup_frac", "min_lr_frac"} & set(overrides)
    if flat_schedule_keys:
        spec = overrides.get("schedule")
        spec = {"name": spec} if isinstance(spec, str) else dict(spec or {})
        for key in flat_schedule_keys:
            # pop() first (the flat key must not survive into the validated
            # set below); setdefault keeps an explicit nested value if the
            # model gave both spellings.
            spec.setdefault(key, overrides.pop(key))
        # A parameterised spec with NO name is the silent-ignore this function
        # exists to prevent: _build_lr_schedule reads name = '' and returns
        # None, so a model asking for {'betas': ..., 'warmup_frac': 0.1} (a
        # spelling the docstring above invites) would train 50 epochs at a
        # constant LR with no error anywhere. Unknown top-level keys already
        # raise; this closes the same hole for the schedule parameters.
        if not str(spec.get("name", "")).strip():
            raise ValueError(
                f"{type(model).__name__}.optimizer_overrides() gave schedule "
                f"parameter(s) {sorted(flat_schedule_keys)} without a "
                f"'schedule' name; pass e.g. 'schedule': 'cosine'")
        overrides["schedule"] = spec
    unknown = set(overrides) - {"betas", "schedule"}
    if unknown:
        # Loud, not ignored: a model asking for something this trainer cannot
        # honour (weight decay, a second param group, ...) must not train as if
        # the request had been granted.
        raise ValueError(
            f"{type(model).__name__}.optimizer_overrides() returned unsupported "
            f"key(s) {sorted(unknown)}; this trainer honours 'betas' and "
            f"'schedule' only")
    return dict(overrides)


def _adam_betas(overrides: Dict[str, Any]) -> tuple:
    """Adam betas from an override dict, defaulting to torch's (0.9, 0.999).

    Validated here rather than at the torch call so a malformed request names
    the model contract in the error instead of surfacing as an opaque
    'betas[0] must be in [0,1)' from deep inside Adam.
    """
    betas = overrides.get("betas")
    if betas is None:
        return (0.9, 0.999)
    values = tuple(float(b) for b in betas)
    if len(values) != 2 or any(not (0.0 <= b < 1.0) for b in values):
        raise ValueError(
            f"optimizer_overrides()['betas'] must be two floats in [0, 1), got {betas!r}")
    return values


def _build_lr_schedule(optimizer, schedule: Any, total_steps: int):
    """LambdaLR for an optimizer-override 'schedule' spec, or None.

    Accepted specs (kept deliberately small — an unrecognised one RAISES, so a
    model asking for a schedule this trainer does not implement can never train
    silently at a constant LR):
      None / 'none' / 'constant'            -> None (today's behaviour)
      'cosine' | 'cosine_warmup'            -> cosine decay to 0 after a linear
                                               warmup over 10 % of training
      {'name': 'cosine', 'warmup_frac': f,  -> the same with an explicit warmup
       'min_lr_frac': m}                       fraction / LR floor
    The multiplier is relative to the optimizer's base LR (LambdaLR semantics),
    so --lr keeps its meaning as the PEAK learning rate.

    Edge cases:
      * total_steps <= 0 (an empty dataloader): no schedule, so a degenerate
        run fails on the data, not on a division by zero here;
      * warmup_frac <= 0 disables the warmup (step 0 starts at the peak LR);
      * warmup_steps >= total_steps (a tiny smoke run): the cosine arm is
        clamped to progress 1.0 rather than dividing by a non-positive span;
      * the warmup ramp is (step+1)/warmup_steps so step 0 is never a
        zero-LR no-op step.
    """
    if schedule is None:
        return None
    if isinstance(schedule, str):
        spec: Dict[str, Any] = {"name": schedule}
    elif isinstance(schedule, dict):
        spec = dict(schedule)
    else:
        raise TypeError(
            f"optimizer_overrides()['schedule'] must be a str or dict, got "
            f"{type(schedule).__name__}")
    name = str(spec.get("name", "")).strip().lower()
    if name in ("", "none", "constant"):
        return None
    if name not in ("cosine", "cosine_warmup", "warmup_cosine"):
        raise ValueError(
            f"unsupported LR schedule {name!r}; this trainer implements "
            f"'cosine' (linear warmup + cosine decay) and 'constant'")
    if total_steps <= 0:
        return None
    warmup_frac = float(spec.get("warmup_frac", 0.1))
    min_lr_frac = float(spec.get("min_lr_frac", 0.0))
    if not (0.0 <= warmup_frac < 1.0):
        raise ValueError(f"schedule warmup_frac must be in [0, 1), got {warmup_frac}")
    if not (0.0 <= min_lr_frac <= 1.0):
        raise ValueError(f"schedule min_lr_frac must be in [0, 1], got {min_lr_frac}")
    warmup_steps = int(round(warmup_frac * total_steps)) if warmup_frac > 0 else 0
    decay_steps = max(total_steps - warmup_steps, 1)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(max(float(step - warmup_steps) / float(decay_steps), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_frac + (1.0 - min_lr_frac) * cosine

    print(f"LR schedule: cosine decay over {total_steps} steps after a linear "
          f"warmup of {warmup_steps} steps (min LR factor {min_lr_frac})")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_autoencoder(config: TrainingConfig):
    """
    Main training loop for sparse autoencoder (BatchTopK, Vanilla, JumpReLU,
    Gated, Matryoshka, MSAE).

    Args:
        config: Training configuration

    Returns:
        autoencoder: Trained autoencoder model
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
            config=config_payload(config)   # vars(config) minus unset opt-in fields
        )
    
    # Load train dataset (LMDB format)
    print(f"Loading train dataset from {config.train_path}...")
    train_is_open_images = resolve_multilabel(config.train_path, getattr(config, 'multilabel', None))
    train_dataset = build_feature_dataset(config.train_path,
                                       return_index=False,
                                       return_path=True,
                                       verbose=True,
                                       return_class_idx=True,
                                       return_multilabel=train_is_open_images)
    
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
    
    # Create separate dataloader for mono loss computation (if needed)
    mono_loader = None
    if config.mono_coef > 0 and config.mono_period > 1:
        print(f"Creating separate DataLoader for periodic mono loss computation...")
        mono_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=config.num_workers > 0,
            prefetch_factor=4 if config.num_workers > 0 else None
        )
    
    # Load val dataset if provided
    val_loader = None
    if config.val_path:
        print(f"Loading val dataset from {config.val_path}...")
        val_is_open_images = resolve_multilabel(config.val_path, getattr(config, 'multilabel', None))
        val_dataset = build_feature_dataset(config.val_path, return_index=False, return_path=True,
                                         verbose=False, return_class_idx=True,
                                         return_multilabel=val_is_open_images)
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
        test_is_open_images = resolve_multilabel(config.test_path, getattr(config, 'multilabel', None))
        test_dataset = build_feature_dataset(
            config.test_path,
            return_index=False,
            return_path=True,
            verbose=False,
            return_class_idx=True,
            return_multilabel=test_is_open_images,
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
    
    # Initialize model
    print(f"Initializing {config.model_type} autoencoder: {config.n_latents} latents, {d_model} features")
    # Defence in depth behind main.py's argparse check: a threshold init on any model but the
    # plain JumpReLU SAE would be silently ignored by the ctor branches below, so refuse it
    # instead.
    if config.jump_threshold_init is not None and config.model_type != 'jumprelu':
        raise ValueError(f"jump_threshold_init={config.jump_threshold_init} is only read by "
                         f"model_type='jumprelu', not {config.model_type!r}")
    
    if config.model_type == 'batch_topk':
        # BatchTopKSAE uses a config dictionary
        model_cfg = {
            'act_size': d_model,
            'dict_size': config.n_latents,
            'top_k': config.topk_k,
            'l1_coeff': 0.0,  # BatchTopK doesn't use L1
            'top_k_aux': config.auxk_k if config.auxk_k else 512,
            'aux_penalty': config.auxk_coef,
            'n_batches_to_dead': config.dead_steps_threshold // config.batch_size,
            'seed': config.seed,
            'device': config.device,
            'dtype': torch.float32,
            'input_unit_norm': config.normalize,
        }
        autoencoder = BatchTopKSAE(model_cfg)
    elif config.model_type == 'vanilla':
        # VanillaSAE uses a config dictionary
        model_cfg = {
            'act_size': d_model,
            'dict_size': config.n_latents,
            'l1_coeff': config.l1_coef,
            'n_batches_to_dead': config.dead_steps_threshold // config.batch_size,
            'seed': config.seed,
            'device': config.device,
            'dtype': torch.float32,
            'input_unit_norm': config.normalize,
        }
        autoencoder = VanillaSAE(model_cfg)
    elif config.model_type == 'jumprelu':
        # JumpReLUSAE uses a config dictionary
        model_cfg = {
            'act_size': d_model,
            'dict_size': config.n_latents,
            'l1_coeff': config.l1_coef,
            'bandwidth': config.bandwidth,
            'n_batches_to_dead': config.dead_steps_threshold // config.batch_size,
            'seed': config.seed,
            'device': config.device,
            'dtype': torch.float32,
            'input_unit_norm': config.normalize,
        }
        # Added ONLY when set: an absent key makes JumpReLUSAE fall back to its default
        # (theta = 1.0). When present it travels inside model_cfg, so knn_lp_sparse.load_run
        # rebuilds the model with the same init (irrelevant after load_state_dict).
        if config.jump_threshold_init is not None:
            model_cfg['jump_threshold_init'] = float(config.jump_threshold_init)
        autoencoder = JumpReLUSAE(model_cfg)
    elif config.model_type == 'gated':
        # GatedSAE (Rajamanoharan et al. 2024). 'model_type' is stored
        # explicitly like matryoshka/msae: the ctor dict is otherwise a
        # superset of VanillaSAE's, so a key-only fingerprint could rebuild
        # the wrong class from a checkpoint. NOTE l1_coeff here is the
        # penalty on the GATE path and drives the EMERGENT L0 -- there is no
        # top_k for this model, so the realized L0 must be measured.
        model_cfg = {
            'model_type': 'gated',
            'act_size': d_model,
            'dict_size': config.n_latents,
            'l1_coeff': config.l1_coef,
            'sparsity_warmup_steps': config.gated_sparsity_warmup,
            'n_batches_to_dead': config.dead_steps_threshold // config.batch_size,
            'seed': config.seed,
            'device': config.device,
            'dtype': torch.float32,
            'input_unit_norm': config.normalize,
        }
        autoencoder = GatedSAE(model_cfg)
    elif config.model_type == 'matryoshka':
        # Bussmann dictionary-prefix Matryoshka. 'model_type' is stored
        # EXPLICITLY in the ctor dict: MatryoshkaSAE registers the same
        # parameter names as VanillaSAE, and cb_sae/featbsf/adapters.py's key-
        # fingerprint fallback would otherwise be the only guard against a
        # silent wrong-class rebuild (its docstring documents that hazard).
        model_cfg = {
            'model_type': 'matryoshka',
            'act_size': d_model,
            'dict_size': config.n_latents,
            'top_k': config.topk_k,
            'group_sizes': _matryoshka_group_sizes(config),
            'l1_coeff': 0.0,  # no L1 term in the Bussmann loss
            'aux_penalty': config.auxk_coef,
            'n_batches_to_dead': config.dead_steps_threshold // config.batch_size,
            'seed': config.seed,
            'device': config.device,
            'dtype': torch.float32,
            'input_unit_norm': config.normalize,
        }
        autoencoder = MatryoshkaSAE(model_cfg)
    elif config.model_type == 'msae':
        # Zaigrajew sparsity-level-nesting MSAE (deployed at k = topk_k).
        nesting, weights = _msae_nesting_and_weights(config)
        model_cfg = {
            'model_type': 'msae',
            'act_size': d_model,
            'dict_size': config.n_latents,
            'top_k': config.topk_k,
            'nesting_list': nesting,
            'relative_importance': weights,
            'soft_cap': config.msae_soft_cap,
            'l1_coeff': 0.0,  # no L1 term in the MSAE loss
            'n_batches_to_dead': config.dead_steps_threshold // config.batch_size,
            'seed': config.seed,
            'device': config.device,
            'dtype': torch.float32,
            'input_unit_norm': config.normalize,
        }
        autoencoder = MSAE(model_cfg)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}. Use 'batch_topk', 'vanilla', "
                         f"'jumprelu', 'gated', 'matryoshka', or 'msae'")

    # Optimizer (+ optional LR schedule), with per-MODEL overrides.
    # Models that do not define optimizer_overrides() are untouched: betas fall
    # back to torch's (0.9, 0.999) and lr_scheduler stays None, i.e. the exact
    # Adam(lr, eps) + constant LR. Among the shipped models only GatedSAE
    # asks for anything (Adam betas (0.0, 0.999), its reference recipe).
    # total_train_steps: the train loader is built above with drop_last=True,
    # so len(train_loader) is EXACTLY the number of optimizer steps per epoch.
    opt_overrides = _optimizer_overrides(autoencoder)
    total_train_steps = config.num_epochs * len(train_loader)
    adam_betas = _adam_betas(opt_overrides)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=config.lr, eps=config.eps,
                                 betas=adam_betas)
    lr_scheduler = _build_lr_schedule(optimizer, opt_overrides.get('schedule'),
                                      total_train_steps)
    if opt_overrides:
        print(f"Optimizer overrides from {type(autoencoder).__name__}: "
              f"betas={adam_betas}, schedule={opt_overrides.get('schedule')!r}")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Training loop
    print(f"Starting training for {config.num_epochs} epochs...")
    autoencoder.train()
    
    global_step = 0
    best_val_mono = float('-inf')
    best_val_mono_epoch = None
    best_val_mono_metrics = None
    best_val_intermediates_path = Path(config.output_dir) / 'best_val_monoscore_intermediates.pt'
    best_val_model_path = Path(config.output_dir) / 'best_val_monoscore_model.pt'
    batch_mono_history = []
    dataset_mono_history = []
    
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
        epoch_l2 = 0.0
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

            # Data-dependent b_dec init on the very first batch (Matryoshka
            # geometric median / MSAE per-dim median). hasattr-gated: legacy
            # models define no hook and are untouched; the hook itself is
            # buffer-guarded so it fires at most once per run.
            if global_step == 0 and hasattr(autoencoder, 'maybe_init_b_dec'):
                autoencoder.maybe_init_b_dec(x)

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
                mono_coef=step_mono_coef
            )
            
            # Backward pass
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), config.max_grad_norm)
            
            # Apply decoder constraints (all models use same method from BaseAutoencoder)
            autoencoder.make_decoder_weights_and_grad_unit_norm()
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # LR schedule advances ONCE PER OPTIMIZER STEP (not per epoch):
            # _build_lr_schedule sizes warmup/decay in optimizer steps. None
            # for every model without a 'schedule' override, so the existing
            # rows keep a constant --lr exactly as before.
            if lr_scheduler is not None:
                lr_scheduler.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_l2 += metrics.get('l2_loss', 0)
            epoch_active += metrics['num_active']

            binary_purity = _mean_active_binary_purity(latents, class_idx)
            if binary_purity is not None:
                epoch_binary_purity += binary_purity
                purity_batches += 1

            num_batches += 1
            
            # Update progress bar
            num_dead = (autoencoder.num_batches_not_active > autoencoder.cfg["n_batches_to_dead"]).sum().item()
            l0 = metrics.get('num_active', 0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4e}",
                'active': f"{l0:.1f}/{config.n_latents}",
                'dead': num_dead
            })
            
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
                'train_l2_step': metrics.get('l2_loss', 0),
                'train_active_step': metrics['num_active'],
                'train_binary_purity_step': binary_purity if binary_purity is not None else 0.0,
                'batch_mono_mean_step': batch_mono_step,
            }
            # Forward the MODEL's own scalar metrics (e.g. GatedSAE's loss
            # terms) to the loggers, so everything a model reports beyond the
            # five keys above is recorded. Only real scalars are forwarded
            # (run_logger drops non-Real values anyway); bools are excluded
            # because they are flags, not measurements; keys already in
            # step_log keep the trainer's own (differently named) version.
            step_log.update({
                k: v for k, v in metrics.items()
                if isinstance(v, numbers.Real) and not isinstance(v, bool)
                and k not in step_log
            })
            # The LR is only logged when a schedule actually moves it: adding
            # the key unconditionally would change the metric set of every
            # existing row (and every TB/W&B dashboard built on it) for no
            # information — a constant --lr is already in the saved config.
            if lr_scheduler is not None:
                step_log['lr_step'] = float(optimizer.param_groups[0]['lr'])
            if config.wandb_project:
                wandb.log(step_log, step=global_step)
            tb.log(step_log, step=global_step)
        
        # Epoch-end logging
        avg_loss = epoch_loss / num_batches
        avg_l2 = epoch_l2 / num_batches
        avg_active = epoch_active / num_batches
        avg_binary_purity = epoch_binary_purity / purity_batches if purity_batches > 0 else 0.0

        print(
            f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {avg_loss:.4e}, "
            f"Train L2: {avg_l2:.4e}, Train Active: {avg_active:.1f}, "
            f"Train Binary Purity: {avg_binary_purity:.4f}"
        )

        # Validation at end of epoch
        if val_loader:
            val_metrics = evaluate(autoencoder, val_loader, config, 'val')
            print(f"Epoch {epoch+1}/{config.num_epochs} - Val Loss: {val_metrics['val_loss']:.4e}, "
                  f"Val L2: {val_metrics['val_l2']:.4e}, "
                f"Val Active: {val_metrics['val_active']:.1f}, "
                f"Val Binary Purity: {val_metrics['val_binary_purity']:.4f}")
        
        # R2 at end of epoch. Train R2 streams the full 1.28M train set (expensive) -> only when
        # not stripping extras. Val R2 (50K) is cheap and used in best-ckpt metrics, so keep it.
        train_r2 = float('nan')
        if not config.skip_extra_eval:
            train_r2 = compute_r2(autoencoder, train_loader, config.device, split_name='train')
            print(f"Epoch {epoch+1}/{config.num_epochs} - Train R2: {train_r2:.6f}")
        if val_loader:
            val_r2 = compute_r2(autoencoder, val_loader, config.device, split_name='val')
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
            'train_l2': avg_l2,
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
                torch.save(
                    {
                        'epoch': best_val_mono_epoch,
                        'val_monoscore': best_val_mono,
                        'model_state_dict': autoencoder.state_dict(),
                        'model_cfg': model_cfg,
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
    final_payload = {
        'epoch': config.num_epochs,
        'step': global_step,
        'model_state_dict': autoencoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # vars(config) minus unset opt-in fields (utils.config_payload): unchanged key set for
        # every run that does not pass --jump_threshold_init.
        'config': config_payload(config),
    }
    # Structural self-report of the trained model, when the class defines a
    # describe() method (none of the shipped models does). Purely additive:
    # every consumer reads 'model_state_dict'/'config' by key.
    describe = getattr(autoencoder, 'describe', None)
    if callable(describe):
        try:
            final_payload['model_describe'] = describe()
        except Exception as exc:                 # never lose a 50-epoch run
            print(f"[warn] describe() failed, not recorded: {exc}")
    torch.save(final_payload, final_path)
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
    best_model_cfg = best_ckpt.get('model_cfg', model_cfg)
    if config.model_type == 'batch_topk':
        best_model = BatchTopKSAE(best_model_cfg)
    elif config.model_type == 'vanilla':
        best_model = VanillaSAE(best_model_cfg)
    elif config.model_type == 'jumprelu':
        best_model = JumpReLUSAE(best_model_cfg)
    elif config.model_type == 'gated':
        best_model = GatedSAE(best_model_cfg)
    elif config.model_type == 'matryoshka':
        best_model = MatryoshkaSAE(best_model_cfg)
    elif config.model_type == 'msae':
        best_model = MSAE(best_model_cfg)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    best_model.load_state_dict(best_ckpt['model_state_dict'])
    best_model.eval()


    # EVALUATE R2
    best_val_r2 = compute_r2(best_model, val_loader, config.device, split_name='val')

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
        best_test_r2 = compute_r2(best_model, test_loader, config.device, split_name='test')
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
    #       f"L2: {train_metrics['train_l2']:.2e}, "
    #         f"Active: {train_metrics['train_active']:.1f}, "
    #         f"Binary Purity: {train_metrics['train_binary_purity']:.4f}")
    # final_results.update(train_metrics)
    
    # # Val split
    # if val_loader:
    #     print("Evaluating on val split...")
    #     val_metrics = evaluate(autoencoder, val_loader, config, 'val')
    #     print(f"Val   - Loss: {val_metrics['val_loss']:.2e}, "
    #           f"L2: {val_metrics['val_l2']:.2e}, "
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
    #           f"L2: {test_metrics['test_l2']:.2e}, "
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
    # train_r2_final = compute_r2(autoencoder, train_loader, config.device, split_name='train')
    # print(f"Train - R2: {train_r2_final:.6f}")
    # final_results['train_r2'] = train_r2_final

    # if config.wandb_project:
    #     wandb.log({'final_train_r2': train_r2_final})

    # if val_loader:
    #     print("Computing val R2 score...")
    #     val_r2_final = compute_r2(autoencoder, val_loader, config.device, split_name='val')
    #     print(f"Val   - R2: {val_r2_final:.6f}")
    #     final_results['val_r2'] = val_r2_final

    #     if config.wandb_project:
    #         wandb.log({'final_val_r2': val_r2_final})

    # if test_loader:
    #     print("Computing test R2 score...")
    #     test_r2_final = compute_r2(autoencoder, test_loader, config.device, split_name='test')
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
    #     best_model_cfg = best_ckpt.get('model_cfg', model_cfg)
    #     if config.model_type == 'batch_topk':
    #         best_model = BatchTopKSAE(best_model_cfg)
    #     elif config.model_type == 'vanilla':
    #         best_model = VanillaSAE(best_model_cfg)
    #     elif config.model_type == 'jumprelu':
    #         best_model = JumpReLUSAE(best_model_cfg)
    #     else:
    #         raise ValueError(f"Unknown model type: {config.model_type}")
    #     best_model = best_model.to(config.device)
    #     best_model.load_state_dict(best_ckpt['model_state_dict'])
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

    parser = argparse.ArgumentParser(description="Train a BatchTopK / JumpReLU / Vanilla SAE")
    parser.add_argument('--dataset_config', type=str, required=True,
                        help='JSON config with train_path/val_path/test_path (LMDB dirs)')
    parser.add_argument('--exp_name', type=str, default='sae')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['batch_topk', 'jumprelu', 'vanilla'])
    parser.add_argument('--n_latents', type=int, default=8192)
    parser.add_argument('--topk_k', type=int, default=64, help='k for batch_topk')
    parser.add_argument('--l1_coef', type=float, default=1e-4, help='L1/L0 penalty (vanilla/jumprelu)')
    parser.add_argument('--bandwidth', type=float, default=1e-3, help='JumpReLU bandwidth')
    parser.add_argument('--auxk_coef', type=float, default=1/32, help='AuxK penalty (batch_topk)')
    parser.add_argument('--auxk_k', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eps', type=float, default=6.25e-10)
    parser.add_argument('--mono_coef', type=float, default=0.0)
    parser.add_argument('--dead_steps_threshold', type=int, default=10_000_000)
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
        model_type=args.model_type,
        n_latents=args.n_latents,
        topk_k=args.topk_k,
        l1_coef=args.l1_coef,
        bandwidth=args.bandwidth,
        auxk_coef=args.auxk_coef,
        auxk_k=args.auxk_k,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        eps=args.eps,
        mono_coef=args.mono_coef,
        dead_steps_threshold=args.dead_steps_threshold,
        skip_extra_eval=args.skip_extra_eval,
        num_workers=args.num_workers,
        seed=args.seed,
        wandb_project=args.wandb_project,
        tb_dir=args.tb_dir,
        output_dir=args.output_dir,
    )

    train_autoencoder(config)

