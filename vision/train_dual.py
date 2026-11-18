import hashlib
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional
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
from models import DualSAE
from loss import compute_monosemanticity_fast
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

# Mono-branch parameters that the post-warm-up optimizer rebuild must pick up,
# beyond the four every DualSAE has. Keyed by attribute name and applied
# hasattr-gated, so a mode that does not define one is unaffected.
POST_WARMUP_EXTRA_MONO_PARAMS = (
    "r_mag",      # sparsity_mode='gated': magnitude-head rescaling
    "mag_bias",   # sparsity_mode='gated': magnitude-head bias
)

# Parameters DELIBERATELY left out of the rebuild, with the reason. They are
# declared here so the omission is visible and testable rather than an accident
# of an explicit list.
POST_WARMUP_EXCLUDED_PARAMS = {
    # The JumpReLU threshold of a Dual-JumpReLU model trains during warm-up only
    # and is frozen from the end of warm-up onwards (default recipe). The
    # threshold of the plain JumpReLU baseline trains for all steps.
    "jumprelu.log_threshold":
        "frozen after warm-up in the default Dual-JumpReLU recipe",
}
# The one opt-in exception to the table above. When
# TrainingConfig.dual_train_threshold_post_warmup is True (main.py
# --dual_train_threshold_post_warmup), build_post_warmup_param_groups puts this parameter into the
# MONO group, so the Dual's threshold keeps training after warm-up as the JumpReLU baseline's
# does. The default (False) leaves the table above in force.
POST_WARMUP_OPT_IN_THRESHOLD_PARAM = "jumprelu.log_threshold"


def build_post_warmup_param_groups(autoencoder, mono_lr, poly_lr, train_threshold=False):
    """Param groups for the optimizer rebuilt when the poly warm-up ends.

    WHY this is a function rather than an inline list: the rebuild REPLACES
    the optimizer, so any parameter missing from it silently stops training
    at warm-up end -- with no error, no warning, and a plausible-looking loss
    curve for the remaining epochs. Extracting it makes the coverage
    testable (tests/test_dual_gated.py) instead of relying on whoever adds
    the next sparsity mode remembering this call site.

    train_threshold (default False == the default recipe): when True, the
    JumpReLU threshold (POST_WARMUP_OPT_IN_THRESHOLD_PARAM) joins the MONO
    group instead of being frozen (POST_WARMUP_EXCLUDED_PARAMS).  Only
    meaningful for sparsity_mode='jump_relu'; any other mode has no threshold
    and is REFUSED (a silently ignored flag would claim a setting the run did
    not have).

    Returns (mono_params, poly_params), each a list of Adam param-group dicts.
    """
    mono_params = [
        {'params': [autoencoder.W_enc], 'lr': mono_lr},
        {'params': [autoencoder.W_dec], 'lr': mono_lr},
    ]
    if hasattr(autoencoder, 'b_enc'):
        mono_params.append({'params': [autoencoder.b_enc], 'lr': mono_lr})
    if hasattr(autoencoder, 'b_dec'):
        mono_params.append({'params': [autoencoder.b_dec], 'lr': mono_lr})
    for name in POST_WARMUP_EXTRA_MONO_PARAMS:
        if hasattr(autoencoder, name):
            mono_params.append({'params': [getattr(autoencoder, name)],
                                'lr': mono_lr})
    if train_threshold:
        # Opt-in (--dual_train_threshold_post_warmup). `jumprelu` is
        # None on every non-JumpReLU DualSAE (dual_sae.py), so this refuses
        # the flag there rather than dropping it.
        if getattr(autoencoder, 'jumprelu', None) is None:
            raise ValueError(
                "train_threshold=True (--dual_train_threshold_post_warmup) needs a "
                "JumpReLU mono branch (sparsity_mode='jump_relu'); this DualSAE has "
                f"sparsity_mode={getattr(autoencoder, 'sparsity_mode', None)!r}")
        # WHY the MONO group and mono_lr: the threshold is a mono-branch
        # parameter (it gates the mono codes), and mono_lr == config.lr is
        # exactly the rate it trained at during warm-up (the warm-up
        # optimizer is Adam(autoencoder.parameters(), lr=config.lr)) and the
        # rate the JumpReLU BASELINE trains it at for all steps
        # (train_other: Adam(autoencoder.parameters(), lr=config.lr)).  So the
        # only remaining within-pair difference in how theta is optimised is
        # the Adam-moment reset at the rebuild, which this rebuild applies to
        # every mono parameter alike (known limitation, declared).
        # poly_lr_factor deliberately does NOT apply: it scales the POLY
        # branch only.
        mono_params.append({'params': [autoencoder.jumprelu.log_threshold],
                            'lr': mono_lr})
    poly_params = [{'params': list(autoencoder.poly.parameters()),
                    'lr': poly_lr}]
    return mono_params, poly_params


# The sparse branch of a TopK/BatchTopK DualSAE, in a fixed order so its hash is stable. The
# modes that add sparse parameters (gated, jump_relu) are refused by the frozen path, so this
# list is complete for everything that can be frozen.
FROZEN_SPARSE_PARAMS = ("W_enc", "W_dec", "b_enc", "b_dec")
FREEZABLE_SPARSITY_MODES = ("topk", "batch_topk")
# Must match between the frozen run and the run training a residual on top of it, or the pair
# would differ in more than whether the two branches train jointly.
FROZEN_SOURCE_MATCH_KEYS = ("act_size", "dict_size", "top_k", "sparsity_mode", "seed",
                            "input_unit_norm")


def validate_ablation_options(config) -> None:
    """Refuse the projector/frozen-branch option combinations that would mislabel a run.

    main.py checks the same rules against its arguments; this repeats them for direct callers.
    Everything here raises before any data loads, because each of these runs would otherwise
    complete normally and only be wrong in what its name claims it measured.
    """
    warm = config.warmup_steps > 0 or config.warmup_mono_target > 0
    if config.poly_arch == 'none':
        # A warm-up would have nothing to warm up towards, and with r_hat == 0 a span or
        # similarity weight has no effect, so accepting one would record a setting with no meaning.
        if warm or config.span_coef != 0 or config.sim_coef != 0:
            raise ValueError("poly_arch='none' is a standard SAE: it needs warmup_steps=0, "
                             f"span_coef=0 and sim_coef=0 (got warmup_steps={config.warmup_steps}, "
                             f"warmup_mono_target={config.warmup_mono_target}, "
                             f"span_coef={config.span_coef}, sim_coef={config.sim_coef})")
    if config.poly_warmup_zero:
        if config.warmup_steps <= 0 or config.poly_arch == 'none':
            raise ValueError("poly_warmup_zero needs warmup_steps > 0 and a real projector: it "
                             "only changes what the frozen projector contributes during warm-up")
    if config.freeze_sparse_from:
        if warm:
            raise ValueError("freeze_sparse_from cannot combine with a warm-up: there is no "
                             "sparse training for it to precede")
        if config.sparsity_mode not in FREEZABLE_SPARSITY_MODES:
            raise ValueError(f"freeze_sparse_from supports {FREEZABLE_SPARSITY_MODES}, not "
                             f"{config.sparsity_mode!r}: other modes have sparse parameters "
                             f"outside {FROZEN_SPARSE_PARAMS} that would keep training")
        if config.poly_arch == 'none':
            raise ValueError("freeze_sparse_from needs a real projector to train")
        if config.poly_warmup_zero:
            raise ValueError("freeze_sparse_from and poly_warmup_zero are different arms")
        if config.mono_coef != 0:
            raise ValueError("freeze_sparse_from needs mono_coef=0: the monosemanticity loss acts "
                             "on the sparse codes, which are frozen")
    if config.snapshot_steps and not config.freeze_sparse_from:
        # Only the frozen arm has a meaningful "same number of residual updates" to match. Other
        # arms keep the per-epoch selection, so a snapshot would be an unselected model.
        raise ValueError("snapshot_steps is only supported together with freeze_sparse_from")


def sparse_state_sha256(state: dict) -> str:
    """Hash the exact bytes of a sparse branch, given a state dict or any name->tensor mapping.

    Public because the collector compares a frozen arm's checkpoint with its source this way.
    """
    digest = hashlib.sha256()
    for name in FROZEN_SPARSE_PARAMS:
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def sparse_branch_sha256(model) -> str:
    """The same digest for a live model, to prove a frozen branch stayed frozen."""
    return sparse_state_sha256({name: getattr(model, name) for name in FROZEN_SPARSE_PARAMS})


def load_frozen_sparse_branch(autoencoder, config, model_cfg) -> dict:
    """Copy a finished poly_arch='none' run's sparse branch into ``autoencoder`` and freeze it.

    The tensors are copied into the model the trainer already built instead of constructing a
    second DualSAE: the constructor reseeds torch, so a second one would reset the global RNG
    between the projector's init and the data loading.

    Returns provenance for frozen_sparse.json and for the snapshots.
    """
    source_dir = Path(config.freeze_sparse_from)
    ckpt_path = source_dir / 'best_val_monoscore_model.pt'
    # A half-trained source would yield a plausible-looking F arm built on a different S than
    # the one reported, so completion is required, not just the best checkpoint's presence.
    if not (source_dir / 'autoencoder_final.pt').exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"freeze_sparse_from needs a FINISHED run; {source_dir} is missing "
                                "autoencoder_final.pt or best_val_monoscore_model.pt")
    source = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    source_cfg = source['model_cfg']
    if source_cfg.get('poly_arch') != 'none':
        raise ValueError(f"{source_dir} has poly_arch={source_cfg.get('poly_arch')!r}: the frozen "
                         "arm must start from a standard SAE (poly_arch='none'), or it would "
                         "inherit a sparse branch already shaped by a residual branch")
    mismatched = {key: (source_cfg.get(key), model_cfg.get(key))
                  for key in FROZEN_SOURCE_MATCH_KEYS
                  if source_cfg.get(key) != model_cfg.get(key)}
    if mismatched:
        raise ValueError(f"{source_dir} does not match this run: {mismatched}")
    # Same features too: the model_cfg has no dataset, and two encoders can share a dimension.
    source_tcfg = json.loads((source_dir / 'training_config.json').read_text())
    if source_tcfg.get('train_path') != config.train_path:
        raise ValueError(f"{source_dir} trained on {source_tcfg.get('train_path')}, "
                         f"this run uses {config.train_path}")
    state = source['model_state_dict']
    with torch.no_grad():
        for name in FROZEN_SPARSE_PARAMS:
            param = getattr(autoencoder, name)
            param.copy_(state[name].to(param.device, dtype=param.dtype))
            param.requires_grad_(False)
    sha = sparse_branch_sha256(autoencoder)
    # Comparing against the stored bytes catches a device or dtype conversion that changed values.
    if sha != sparse_state_sha256(state):
        raise RuntimeError(f"sparse branch changed while loading from {ckpt_path}")
    return {
        'source_run': str(source_dir),
        'source_best_epoch': source.get('epoch'),
        'source_val_monoscore': source.get('val_monoscore'),
        'sparse_sha256': sha,
    }


def write_residual_snapshot(autoencoder, config, model_cfg, step: int, epoch: int,
                            frozen_meta: dict) -> Path:
    """Save the frozen arm at an exact residual-update count, as a loadable run directory.

    A DualSAE scored at its best epoch has had fewer residual updates than one scored at the end
    of training, so a fair comparison needs the frozen arm at that same count. That count usually
    falls mid-epoch, which is why this runs from the training loop instead of reusing the
    epoch-end checkpoint.

    The layout (training_config.json + best_val_monoscore_model.pt) is what
    knn_lp_sparse.load_run reads, so evaluation code needs no special case. val_monoscore comes
    from the frozen source because z cannot change, which keeps branch_metrics'
    monoscore_x_vs_checkpoint self-check meaningful for snapshots too.
    """
    run_dir = Path(config.output_dir)
    target = run_dir.parent / f"{run_dir.name}__step{step}"
    # Built under a temporary name, then renamed, so a wall-clock kill can never leave a
    # snapshot that looks complete to a later evaluation pass.
    temporary = run_dir.parent / f".{target.name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    payload = config_payload(config)
    payload.update(exp_name=target.name, output_dir=str(target),
                   snapshot_of=run_dir.name, snapshot_step=int(step))
    (temporary / 'training_config.json').write_text(json.dumps(payload, indent=2))
    torch.save(
        {
            'epoch': epoch + 1,
            'step': int(step),
            'val_monoscore': frozen_meta['source_val_monoscore'],
            'model_state_dict': autoencoder.state_dict(),
            'model_cfg': model_cfg,
        },
        temporary / 'best_val_monoscore_model.pt',
    )
    # Only reachable when a timed-out run is restarted; its earlier snapshot came from a run
    # that is being replaced, so keeping it would mix two different trajectories.
    if target.exists():
        shutil.rmtree(target)
    os.replace(temporary, target)
    return target


def reinitialize_dead_neurons(autoencoder, dead_mask, sample_batch, device):
    """Reinitialize dead neurons with random directions from data."""
    num_dead = dead_mask.sum().item()
    if num_dead == 0:
        return

    with torch.no_grad():
        batch_size = sample_batch.shape[0]

        # Sample vectors from current data batch.
        if num_dead <= batch_size:
            sample_indices = torch.randperm(batch_size)[:num_dead]
        else:
            sample_indices = torch.randint(0, batch_size, (num_dead,))

        samples = sample_batch[sample_indices].to(device)
        samples = F.normalize(samples, p=2, dim=-1)

        # W_enc shape: (act_size, dict_size) -> update dead columns.
        autoencoder.W_enc.data[:, dead_mask] = samples.t() * 0.1

        # W_dec shape: (dict_size, act_size) -> update dead rows.
        autoencoder.W_dec.data[dead_mask] = samples * 0.1

        # Reset optional encoder bias slot for these features.
        if hasattr(autoencoder, "b_enc"):
            autoencoder.b_enc.data[dead_mask] = 0.0

        # sparsity_mode='gated' adds a magnitude head (exp(r_mag), mag_bias).
        # A respawned latent that kept a stale mag_bias would be born
        # saturated (or permanently shut) regardless of its fresh direction,
        # so both are reset to their init value. hasattr-gated: every other
        # mode is untouched.
        for _name in ("r_mag", "mag_bias"):
            if hasattr(autoencoder, _name):
                getattr(autoencoder, _name).data[dead_mask] = 0.0

        autoencoder.num_batches_not_active[dead_mask] = 0

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
def _build_reconstruction_bank(
    autoencoder,
    dataloader,
    device='cuda',
    max_samples: Optional[int] = None,
    split_name: str = 'train',
):
    """Build a normalized reconstructed-embedding bank (x_hat, labels) from a dataloader."""
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
    """Compute KNN metric using reconstructed embeddings x_hat.

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

@torch.no_grad()
def save_intermediate_outputs(autoencoder, dataloader, output_path: Path, device='cuda', split_name: str = 'val', include_r_hat: bool = True):
    """Save a mapping: path -> (x, z, xM, r_hat, x_hat) for a full dataloader pass.
    
    Args:
        autoencoder: DualSAE model
        dataloader: DataLoader for the dataset
        output_path: Path to save outputs to
        device: Device to run on
        split_name: Name of the split for logging
        include_r_hat: Whether to include r_hat in the outputs (requires dual-branch model)
    """
    autoencoder.eval()
    outputs = {}
    running_idx = 0

    for batch in tqdm(dataloader, desc=f"Saving intermediates ({split_name})"):
        features, _, sample_path = unpack_batch(batch)
        x = features.to(device)

        _, z, x_hat = autoencoder(x)
        xM = autoencoder._xM
        r_hat = autoencoder._r_hat if include_r_hat and hasattr(autoencoder, '_r_hat') else None

        x_cpu = x.detach().cpu()
        z_cpu = z.detach().cpu()
        xM_cpu = xM.detach().cpu()
        x_hat_cpu = x_hat.detach().cpu()
        r_hat_cpu = r_hat.detach().cpu() if r_hat is not None else None

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
            if r_hat_cpu is not None:
                outputs[key] = (x_cpu[i], z_cpu[i], xM_cpu[i], r_hat_cpu[i], x_hat_cpu[i])
            else:
                outputs[key] = (x_cpu[i], z_cpu[i], xM_cpu[i], x_hat_cpu[i])

        running_idx += batch_size

    torch.save(outputs, output_path)
    autoencoder.train()
    return output_path

@torch.no_grad()
def evaluate(autoencoder, dataloader, config, split_name='val'):
    """Evaluate autoencoder on a dataset."""
    autoencoder.eval()
    total_loss = 0
    total_l2 = 0
    total_span = 0
    total_sim = 0
    total_r_rhat_cosine = 0
    total_samples = 0
    total_active = 0
    num_batches = 0

    acts_all =[]
    class_idx_all = []
    
    for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
        features, class_idx, _ = unpack_batch(batch)
        x = features.to(config.device)
        if class_idx is not None:
            class_idx = class_idx.to(config.device)
        
        # DualSAE returns (latents_pre_act, latents, recons)
        latents_pre_act, acts, recons = autoencoder(x)

        acts_all.append(acts.detach().cpu())
        if class_idx is not None:
            class_idx_all.append(class_idx.detach().cpu())
        
        # Compute loss using model's compute_loss method
        loss, metrics = autoencoder.compute_loss(
            x.float(),
            acts.float(),
            recons.float(),
            span_coef=config.span_coef,
            active_only_span=config.active_only_span,
            sim_coef=0.0,  # Don't use similarity loss during evaluation
            mono_coef=0.0  # Don't use mono loss during evaluation
        )

        # Residual alignment metric: cosine similarity between true residual r and predicted residual r_hat.
        r = autoencoder._r
        r_hat = autoencoder._r_hat
        batch_r_rhat_cosine = F.cosine_similarity(
            r.float(),
            r_hat.float(),
            dim=1,
            eps=1e-8,
        )
        
        total_loss += loss.item()
        total_l2 += metrics.get('l2_loss', 0)
        total_span += metrics.get('span_loss', 0)
        total_sim += metrics.get('sim_loss', 0)
        total_r_rhat_cosine += batch_r_rhat_cosine.sum().item()
        total_samples += batch_r_rhat_cosine.numel()
        total_active += metrics.get('num_active', 0)
        num_batches += 1
    
    autoencoder.train()
    
    avg_loss = total_loss / num_batches
    avg_l2 = total_l2 / num_batches
    avg_span = total_span / num_batches
    avg_sim = total_sim / num_batches
    avg_r_rhat_cosine = total_r_rhat_cosine / max(total_samples, 1)
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
        f'{split_name}_span': avg_span,
        f'{split_name}_sim': avg_sim,
        f'{split_name}_r_rhat_cosine': avg_r_rhat_cosine,
        f'{split_name}_active': avg_active,
        f'{split_name}_binary_purity': avg_binary_purity,
    }


@torch.no_grad()
def compute_active_binary_purity(autoencoder, dataloader, device='cuda', split_name='data'):
    """Compute mean active-latent binary purity over a dataloader."""
    autoencoder.eval()

    acts_all = []
    class_idx_all = []

    for batch in tqdm(dataloader, desc=f"Computing binary purity {split_name}"):
        features, class_idx, _ = unpack_batch(batch)
        if class_idx is None:
            continue

        x = features.to(device)
        class_idx = class_idx.to(device)
        _, acts, _ = autoencoder(x)

        acts_all.append(acts.detach().cpu())
        class_idx_all.append(class_idx.detach().cpu())

    autoencoder.train()

    if len(acts_all) == 0 or len(class_idx_all) == 0:
        return 0.0

    acts = torch.cat(acts_all, dim=0)
    class_idx = torch.cat(class_idx_all, dim=0)
    purity = _mean_active_binary_purity(acts, class_idx)
    if purity is None:
        return 0.0
    return float(purity)


@torch.no_grad()
def compute_r_hat_monosemanticity(autoencoder, dataloader, device='cuda', eps: float = 1e-8, split_name='train', verbose: bool = False):
    """
    Compute monosemanticity for r_hat (poly branch output).
    Similar to compute_monosemanticity_fast but for the residual reconstruction.
    
    Args:
        autoencoder: DualSAE model
        dataloader: DataLoader for the dataset
        device: Device to run on
        eps: Small constant for numerical stability
        split_name: Name of the split for logging
        verbose: If True, print progress messages
    
    Returns:
        monosemanticity: (D,) tensor of scores per feature dimension of r_hat
    """
    autoencoder.eval()
    
    dataset = dataloader.dataset
    
    # Pass 1: r_hat min/max per dimension for scaling
    if verbose:
        print(f"First pass: computing r_hat statistics ({split_name})...")
    min_vals = None
    max_vals = None
    feature_dim = None
    
    for batch in tqdm(dataloader, desc=f"Computing r_hat stats for {split_name}", disable=not verbose):
        features, class_idx, _ = unpack_batch(batch)
        x = features.to(device)
        if class_idx is not None:
            class_idx = class_idx.to(device)
        
        # Forward pass to compute r_hat
        _, _, _ = autoencoder(x)
        r_hat = autoencoder._r_hat  # Access stored r_hat
        
        if min_vals is None:
            feature_dim = r_hat.shape[1]
            min_vals = r_hat.min(dim=0, keepdim=True)[0]
            max_vals = r_hat.max(dim=0, keepdim=True)[0]
        else:
            batch_min = r_hat.min(dim=0, keepdim=True)[0]
            batch_max = r_hat.max(dim=0, keepdim=True)[0]
            min_vals = torch.min(min_vals, batch_min)
            max_vals = torch.max(max_vals, batch_max)
    
    dataset_size = len(dataset)
    if verbose:
        print(f"Dataset size: {dataset_size}")
        print(f"Feature dimension: {feature_dim}")
    
    # Pass 2: stream accumulators
    if verbose:
        print(f"Second pass: streaming accumulators for r_hat...")
    # Accumulators in FP32 for numerical stability
    u = torch.zeros(feature_dim, device=device, dtype=torch.float32)
    v = torch.zeros(feature_dim, device=device, dtype=torch.float32)
    w = torch.zeros(feature_dim, feature_dim, device=device, dtype=torch.float32)
    
    for batch in tqdm(dataloader, desc=f"Accumulating r_hat for {split_name}", disable=not verbose):
        features, class_idx, _ = unpack_batch(batch)
        x = features.to(device)
        if class_idx is not None:
            class_idx = class_idx.to(device)
        
        # Forward pass
        _, _, _ = autoencoder(x)
        r_hat = autoencoder._r_hat
        
        # Normalize r_hat per dimension to [0, 1]
        r_hat_norm = (r_hat - min_vals) / (max_vals - min_vals + eps)
        a = r_hat_norm
        
        # Original input features (for computing selectivity)
        x_preprocessed = autoencoder._x_preprocessed
        x_float = x_preprocessed.to(torch.float32)
        
        # Accumulate
        # u: sum of activations per feature
        u += a.sum(dim=0)
        # v: sum of squared activations per feature
        v += (a * a).sum(dim=0)
        # w: weighted sum of inputs per feature (D x D)
        # For each feature dimension, sum inputs weighted by that dimension's activation
        w += x_float.t() @ a  # (D, N) @ (N, D) = (D, D)
    
    # Finalize
    u_mean = u / dataset_size  # Mean activation per feature
    v_mean = v / dataset_size  # Mean squared activation per feature
    w_mean = w / dataset_size  # Mean weighted input per feature
    
    # Variance of activations: var = E[a²] - E[a]²
    var_a = v_mean - u_mean * u_mean
    var_a = var_a.clamp(min=eps)
    
    # For each feature i: selectivity_i = sqrt((w_mean[:, i]² ).sum()) / sqrt(var_a[i])
    # Monosemanticity = selectivity / sqrt(D)
    selectivity = torch.sqrt((w_mean * w_mean).sum(dim=0)) / torch.sqrt(var_a)
    monosemanticity = selectivity / np.sqrt(feature_dim)
    
    # Clamp to [0, 1] (though it should be naturally bounded)
    monosemanticity = monosemanticity.clamp(0, 1)
    
    autoencoder.train()
    return monosemanticity


@torch.no_grad()
def compute_r2(autoencoder, dataloader, device='cuda', eps: float = 1e-12, split_name='data'):
    """Compute R² (uniform average across features) over an entire dataloader."""
    sum_y = None
    sum_y2 = None
    ss_res = None
    total_n = 0

    autoencoder.eval()

    for batch in tqdm(dataloader, desc=f"Computing R2 {split_name}"):
        features, class_idx, _ = unpack_batch(batch)
        x = features.to(device)
        if class_idx is not None:
            class_idx = class_idx.to(device)

        # DualSAE returns (latents_pre_act, latents, recons)
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


@torch.no_grad()
def compute_mono_branch_r2(autoencoder, dataloader, device='cuda', eps: float = 1e-12, split_name='data'):
    """Compute R2 between mono-branch output and input over an entire dataloader."""
    sum_y = None
    sum_y2 = None
    ss_res = None
    total_n = 0

    autoencoder.eval()

    for batch in tqdm(dataloader, desc=f"Computing Mono-Branch R2 {split_name}"):
        features, class_idx, _ = unpack_batch(batch)
        x = features.to(device)
        if class_idx is not None:
            class_idx = class_idx.to(device)

        # Run forward once to populate intermediate mono-branch tensors.
        _, _, _ = autoencoder(x)

        # xM = autoencoder._xM
        # x_mean = getattr(autoencoder, '_x_mean', None)
        # x_std = getattr(autoencoder, '_x_std', None)
        # mono_recons = autoencoder.postprocess_output(xM, x_mean, x_std)
        mono_recons = autoencoder._xM

        x64 = x.to(torch.float64)
        m64 = mono_recons.to(torch.float64)

        if sum_y is None:
            d_model = x64.shape[1]
            sum_y = torch.zeros(d_model, dtype=torch.float64, device=device)
            sum_y2 = torch.zeros(d_model, dtype=torch.float64, device=device)
            ss_res = torch.zeros(d_model, dtype=torch.float64, device=device)

        sum_y += x64.sum(dim=0)
        sum_y2 += (x64 * x64).sum(dim=0)
        ss_res += ((x64 - m64) * (x64 - m64)).sum(dim=0)
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
    model_type: str = 'dual'  # 'dual' for DualSAE, 'mono' for MonoSAE (ablation)
        
    # Data (optional)
    val_path: Optional[str] = None
    test_path: Optional[str] = None

    # Experiment
    exp_name: str = 'default'
    
    # Model architecture
    n_latents: int = 8192
    d_model: Optional[int] = None  # Will be inferred from data
    poly_hidden: int = 512  # Hidden dimension for polynomial MLP
    poly_layers: int = 1  # Number of linear layers in poly projector
    poly_hidden_dims: Optional[List[int]] = None  # Explicit hidden widths, e.g. [1024, 512, 256]
    # Residual-branch ablation options. All five are opt-in: at these defaults the run is the
    # standard recipe, and utils.config_payload drops them from the saved configs.
    # 'none' removes the residual branch (x_hat = xM), i.e. a standard SAE on this sparse branch.
    poly_arch: str = 'mlp'
    # Output activation of the projector; 'auto' keeps each legacy path's own behaviour (see
    # DualSAE._build_poly_projector for why those paths disagree).
    poly_final_act: str = 'auto'
    # Hold r_hat at exactly 0 during warm-up, so warm-up is a plain SAE objective instead of
    # including the frozen random projector. The joint stage is unchanged.
    poly_warmup_zero: bool = False
    # Path to a finished poly_arch='none' run: freeze its sparse branch and train only the
    # projector here, for the same number of steps budgeted to a normal run.
    freeze_sparse_from: Optional[str] = None
    # Optimizer steps at which the frozen arm also saves a loadable <exp>__step<T>/ directory.
    snapshot_steps: Optional[List[int]] = None
    top_k: int = 64  # Sparsity for monosemantic branch
    sparsity_mode: str = "topk"  # "topk", "batch_topk", "vanilla", or "jump_relu"
    l1_coef: float = 1e-4  # L1 coefficient used when sparsity_mode="vanilla"
    bandwidth: float = 0.001  # JumpReLU bandwidth used when sparsity_mode="jump_relu"
    # JumpReLU threshold init theta (sparsity_mode 'jump_relu' only). None == the model default
    # (theta = 1.0) and NO model_cfg key; a None is also dropped from training_config.json /
    # checkpoint 'config' / W&B (utils.config_payload, OPT_IN_NONE_CONFIG_FIELDS).
    # This sets only the INIT: jumprelu.log_threshold still leaves the optimizer at warm-up end
    # (POST_WARMUP_EXCLUDED_PARAMS) unless dual_train_threshold_post_warmup below is True.
    jump_threshold_init: Optional[float] = None
    # Keep training jumprelu.log_threshold after warm-up (sparsity_mode 'jump_relu' only; opt-in,
    # the default recipe freezes it -- see POST_WARMUP_EXCLUDED_PARAMS). With the threshold frozen,
    # nothing counters post-warm-up drift of the encoder, so the Dual-JumpReLU code can become
    # denser than its baseline's, which trains its threshold for all steps. False (default) is
    # DROPPED from training_config.json / checkpoint 'config' / W&B (utils.config_payload,
    # OPT_IN_DEFAULT_CONFIG_FIELDS). Not part of model_cfg: it is a training behaviour, not an
    # architecture (a loaded model is identical).
    dual_train_threshold_post_warmup: bool = False
    normalize: bool = False  # Layer normalization
    
    # Training
    batch_size: int = 4096
    num_epochs: int = 10
    lr: float = 1e-4
    eps: float = 6.25e-10
    max_grad_norm: float = 1.0
    
    # Loss coefficients
    span_coef: float = 0.01  # Coefficient for span overlap penalty (paper default 0.01)
    active_only_span: bool = True  # Only apply span penalty to active latents
    sim_coef: float = 0.0  # Coefficient for similarity penalty between xM and r_hat
    mono_coef: float = 0.0  # Coefficient for monosemanticity loss
    mono_period: int = 1  # Compute mono loss every N steps (1=per batch)
    
    # Warm start (train mono branch first, then enable poly)
    warmup_steps: int = 0  # Train mono branch alone for this many steps (0=disabled)
    warmup_mono_target: float = 0.0  # If >0, train until mono score reaches this value
    warmup_check_interval: int = 500  # Check mono score every N steps during warmup
    poly_lr_factor: float = 1.0  # Learning rate multiplier for poly branch after warmup (e.g., 0.1 = 10% of base lr)
    
    # Dead neuron handling (tracked by models)
    dead_steps_threshold: int = 10_000_000  # In tokens/samples
    dead_check_interval: int = 1000  # Check every N steps

    # Dual-Gated knob (sparsity_mode 'gated' only; inert otherwise). Linear
    # ramp of the gate-path L1 over this many optimizer steps; 0 disables.
    gated_sparsity_warmup: int = 2000

    # Escape hatch for the warm-up budget guard below: allows a warm-up that
    # exceeds the step budget (the run is then a plain SAE).
    allow_degenerate_warmup: bool = False

    # Dual-Matryoshka knobs (sparsity_mode 'matryoshka' / 'msae' only; None
    # values fall back to the official constructions at model-build time).
    group_fractions: Optional[list] = None   # Bussmann dictionary-group fractions
    msae_nesting: Optional[list] = None      # MSAE sparsity levels (None -> build_nesting)
    msae_weighting: str = 'rw'               # MSAE per-level weights: rw|uw
    msae_soft_cap: float = 30.0              # MSAE SoftCapping (<=0 disables)

    # Strip heavy per-epoch / end-of-run eval (train-set R2/mono, r_hat-mono, KNN/LP,
    # save-intermediates). Training math + per-epoch val-MonoScore selection are kept.
    skip_extra_eval: bool = False

    # Skip the ~150MB-per-epoch checkpoints under epoch_checkpoints/ (optimizer
    # state included). Nothing in this repository loads them. Default False
    # keeps writing them.
    skip_epoch_checkpoints: bool = False

    # Only epochs >= select_from_epoch (1-based) may become best_val_monoscore_model.pt.
    # 0 keeps the default rule (any epoch). Useful for runs whose best val MonoScore would
    # otherwise fall inside warm-up or the first epochs.
    select_from_epoch: int = 0

    # Logging
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_group: Optional[str] = None
    tb_dir: Optional[str] = None        # TensorBoard event dir (real-time offline); None = no TB

    # Checkpoint & embedding saving
    save_val_embeds: bool = False  # Save validation embeddings periodically
    save_embeds_interval: int = 1  # Save embeddings every N epochs (1=every epoch)
    
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
    Main training loop for DualSAE (dual-branch sparse autoencoder).
    
    Args:
        config: Training configuration
        
    Returns:
        autoencoder: Trained DualSAE model
    """
    
    # Before any data loads: these combinations would otherwise run to completion mislabelled.
    validate_ablation_options(config)

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
    train_dataset = build_feature_dataset(
        config.train_path,
        return_path=True,
        return_class_idx=True,
        return_multilabel=resolve_multilabel(config.train_path, getattr(config, 'multilabel', None)),
        verbose=True,
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
            return_path=True,
            return_class_idx=True,
            return_multilabel=resolve_multilabel(config.val_path, getattr(config, 'multilabel', None)),
            verbose=False,
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
            return_path=True,
            return_class_idx=True,
            return_multilabel=resolve_multilabel(config.test_path, getattr(config, 'multilabel', None)),
            verbose=False,
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
    
    # Initialize DualSAE model
    print(f"Initializing DualSAE: {config.n_latents} latents, {d_model} features")
    print(f"Sparsity mode: {config.sparsity_mode}")
    
    model_cfg = {
        'act_size': d_model,
        'dict_size': config.n_latents,
        'top_k': config.top_k,
        'l1_coeff': config.l1_coef,
        'bandwidth': config.bandwidth,
        'poly_hidden': config.poly_hidden,
        'poly_layers': config.poly_layers,
        'poly_hidden_dims': config.poly_hidden_dims,
        'sparsity_mode': config.sparsity_mode,
        'n_batches_to_dead': config.dead_steps_threshold // config.batch_size,
        'seed': config.seed,
        'device': config.device,
        'dtype': torch.float32,
        'input_unit_norm': config.normalize,
    }
    # Forwarded only when non-default, so legacy payloads stay byte-identical. Unlike the keys
    # below, these can change the architecture WITHOUT changing the state dict (a GELU has no
    # parameters), so model_cfg is the only record of which projector was trained.
    if config.poly_arch != 'mlp':
        model_cfg['poly_arch'] = config.poly_arch
    if config.poly_final_act != 'auto':
        model_cfg['poly_final_act'] = config.poly_final_act
    # JumpReLU threshold init: forwarded ONLY when explicitly set, so the saved 'model_cfg' (the
    # only thing downstream rebuilders see) carries no key for the default -- the same
    # conditional-key discipline as the matryoshka/gated keys below.
    # Any other sparsity_mode has no threshold to initialise, so a value there is refused rather
    # than silently dropped (main.py already rejects it at the CLI; this guards direct callers).
    if config.jump_threshold_init is not None:
        if config.sparsity_mode != 'jump_relu':
            raise ValueError(f"jump_threshold_init={config.jump_threshold_init} is only read by "
                             f"sparsity_mode='jump_relu', not {config.sparsity_mode!r}")
        model_cfg['jump_threshold_init'] = float(config.jump_threshold_init)
    # --dual_train_threshold_post_warmup guard (defence in depth behind main.py's check):
    # the flag only has a parameter to act on in jump_relu mode.  Refused BEFORE training starts,
    # not at the warm-up rebuild thousands of steps in.  Deliberately NOT added to model_cfg: it
    # changes how the model is trained, not what is built, so best_val 'model_cfg' payloads and
    # every rebuilder stay unchanged.  Note: with no warm-up (warmup_steps == 0 and no mono
    # target) the optimizer is never rebuilt and the threshold trains for all steps anyway, so
    # the flag is satisfied trivially there (not refused).
    if config.dual_train_threshold_post_warmup and config.sparsity_mode != 'jump_relu':
        raise ValueError("dual_train_threshold_post_warmup=True is only meaningful for "
                         f"sparsity_mode='jump_relu', not {config.sparsity_mode!r}")
    # Matryoshka-family mono branches need their structure in the ctor dict
    # (and thereby in the best_val 'model_cfg' payload, which is all the
    # downstream rebuilder sees). Added CONDITIONALLY so legacy-mode payloads
    # stay byte-identical to pre-change runs.
    if config.sparsity_mode == 'gated':
        model_cfg['sparsity_warmup_steps'] = config.gated_sparsity_warmup
    elif config.sparsity_mode == 'matryoshka':
        from math import isclose
        fractions = config.group_fractions or [1/32, 1/16, 1/8, 1/4, 1/2 + 1/32]
        if not isclose(sum(fractions), 1.0):
            raise ValueError(f"group_fractions must sum to 1.0: {fractions}")
        # Vendored trainer arithmetic: int-floor all but the last group,
        # remainder in the last (matryoshka_batch_top_k.py:168-172).
        sizes = [int(f * config.n_latents) for f in fractions[:-1]]
        sizes.append(config.n_latents - sum(sizes))
        model_cfg['group_sizes'] = sizes
    elif config.sparsity_mode == 'msae':
        from models.msae_sae import build_nesting, matryoshka_weights
        nesting = config.msae_nesting or build_nesting(config.top_k, 256)
        model_cfg['nesting_list'] = nesting
        model_cfg['relative_importance'] = matryoshka_weights(
            len(nesting), config.msae_weighting)
        model_cfg['soft_cap'] = config.msae_soft_cap
    
    autoencoder = DualSAE(model_cfg).to(config.device)
    
    # Warm start: freeze poly branch if warmup is enabled
    warmup_active = config.warmup_steps > 0 or config.warmup_mono_target > 0
    if warmup_active:
        print(f"\n{'='*60}")
        print("WARM START ENABLED")
        print(f"{'='*60}")
        if config.warmup_steps > 0:
            print(f"Training mono branch only for {config.warmup_steps} steps")
        if config.warmup_mono_target > 0:
            print(f"Training mono branch until mono score reaches {config.warmup_mono_target:.4f}")
            print(f"Checking mono score every {config.warmup_check_interval} steps")
        print(f"{'='*60}\n")
        
        # Freeze poly branch
        for param in autoencoder.poly.parameters():
            param.requires_grad = False
        print("Poly branch frozen (mono-only training)")
        if config.poly_warmup_zero:
            autoencoder.poly_bypass = True
            print("Residual output held at 0 during warm-up (poly_warmup_zero)")

    # Create output directory (also needed below for the frozen arm's provenance file)
    os.makedirs(config.output_dir, exist_ok=True)

    # The frozen arm: load the standard SAE's sparse branch before the optimizer exists, so the
    # optimizer can only ever see the projector.
    frozen_sparse = bool(config.freeze_sparse_from)
    frozen_meta = None
    snapshot_steps = set(config.snapshot_steps or [])
    if frozen_sparse:
        frozen_meta = load_frozen_sparse_branch(autoencoder, config, model_cfg)
        (Path(config.output_dir) / 'frozen_sparse.json').write_text(json.dumps(frozen_meta, indent=2))
        print(f"Sparse branch frozen from {frozen_meta['source_run']} "
              f"(best epoch {frozen_meta['source_best_epoch']}, "
              f"sha256 {frozen_meta['sparse_sha256'][:12]})")

    # Optimizer. The frozen arm uses the same Adam settings and residual learning rate that a
    # DualSAE's rebuilt optimizer gives its projector when warm-up ends.
    if frozen_sparse:
        optimizer = torch.optim.Adam(autoencoder.poly.parameters(),
                                     lr=config.lr * config.poly_lr_factor, eps=config.eps)
    else:
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=config.lr, eps=config.eps)


    # Training loop
    print(f"Starting training for {config.num_epochs} epochs...")
    autoencoder.train()

    # Convert threshold from samples to training steps.
    dead_steps_threshold_steps = config.dead_steps_threshold // config.batch_size

    # Keep one sample batch for dead-neuron reinitialization.
    def cycle_dataloader(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    data_iter = cycle_dataloader(train_loader)
    sample_batch = next(data_iter)
    sample_batch, _, _ = unpack_batch(sample_batch)
    
    global_step = 0
    warmup_complete = not warmup_active
    best_val_mono = float('-inf')
    best_val_mono_epoch = None
    best_val_mono_metrics = None
    best_val_intermediates_path = Path(config.output_dir) / 'best_val_monoscore_intermediates.pt'
    best_val_model_path = Path(config.output_dir) / 'best_val_monoscore_model.pt'
    epoch_checkpoint_dir = Path(config.output_dir) / 'epoch_checkpoints'
    epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create embeddings directory if needed
    embeds_dir = None
    if config.save_val_embeds:
        embeds_dir = Path(config.output_dir) / 'val_embeddings'
        embeds_dir.mkdir(parents=True, exist_ok=True)
        print(f"Validation embeddings will be saved to {embeds_dir}")
    
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

    # WARM-UP BUDGET GUARD. resolve_step_budget above only fires when
    # --steps_per_epoch is set; when it is not (the default, used by all vision
    # runs), a warm-up longer than the whole run would otherwise pass
    # silently -- the poly branch never unfreezes and span_coef is pinned at
    # 0, so the job trains a plain SAE under the DualSAE name.
    #
    # Example: OpenImagesV7 has 507,444 train rows: at batch 2048 with
    # drop_last that is 247 steps/epoch, so a 50-epoch run is 12,350 steps and
    # a warm-up of 17500 steps never ends.
    #
    # train_loader is built with drop_last=True, so len() is the exact
    # per-epoch step count.
    #
    # The budget must be the one the loop will ACTUALLY run. With --steps_per_epoch set, the
    # PersistentBatcher runs num_epochs x steps_per_epoch optimizer steps regardless of dataset
    # size (that is the point of a fixed step budget), and resolve_step_budget has already
    # refused a warm-up >= that total. Using num_epochs x len(train_loader) in that case would
    # under-count the budget for small corpora (e.g. 4 true steps/epoch -> 200 "steps" against
    # a real fixed budget) and wrongly refuse valid runs. Only the unset case falls back to the
    # dataset length.
    _warm = int(getattr(config, 'warmup_steps', 0) or 0)
    _budget = _total_steps if _total_steps is not None else config.num_epochs * len(train_loader)
    if _warm >= _budget and not getattr(config, 'allow_degenerate_warmup', False):
        raise ValueError(
            f"[{getattr(config, 'exp_name', 'run')}] warmup_steps={_warm} >= the entire "
            f"training budget of {_budget} steps ({config.num_epochs} epochs x "
            f"{len(train_loader)} steps/epoch, batch {config.batch_size}). The poly branch "
            f"would never unfreeze and span_coef would stay 0 for the whole run: this "
            f"trains a MonoSAE, not a DualSAE. Lower --warmup_steps, or pass "
            f"--allow_degenerate_warmup to run it on purpose."
        )
    # A snapshot past the end of training would simply never fire, and the comparison needing it
    # would only come up short at report time, after the run had been paid for.
    if snapshot_steps and max(snapshot_steps) > _budget:
        raise ValueError(f"snapshot_steps {sorted(snapshot_steps)} exceed the training budget of "
                         f"{_budget} steps")

    for epoch in range(config.num_epochs):
        # Track metrics for the epoch
        epoch_loss = 0.0
        epoch_l2 = 0.0
        epoch_span = 0.0
        epoch_sim = 0.0
        epoch_active = 0.0
        epoch_mono_loss = 0.0
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

            # Data-dependent b_dec init on the very first batch — a NO-OP for
            # every legacy sparsity mode (only the Matryoshka-family modes
            # implement a non-trivial init, buffer-guarded to fire once).
            if global_step == 0 and hasattr(autoencoder, 'maybe_init_b_dec'):
                autoencoder.maybe_init_b_dec(x)

            # Forward pass
            latents_pre_act, acts, recons = autoencoder(x)
            
            # Decide mono_coef for this step
            step_mono_coef = config.mono_coef
            if config.mono_coef > 0 and config.mono_period > 1:
                # Periodic mode: only compute mono loss every N steps
                step_mono_coef = 0.0  # Skip batch-level mono loss
            
            # Compute loss using model's compute_loss method
            # During warmup, disable span loss (poly branch is frozen)
            current_span_coef = 0.0 if not warmup_complete else config.span_coef
            current_sim_coef = 0.0 if not warmup_complete else config.sim_coef
            
            loss, metrics = autoencoder.compute_loss(
                x.float(),
                acts.float(),
                recons.float(),
                span_coef=current_span_coef,
                sim_coef=current_sim_coef,
                active_only_span=config.active_only_span,
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
            
            # Decoder weight normalization for DualSAE. Skipped for the frozen arm because it
            # writes W_dec directly: requires_grad=False would not protect it.
            if not frozen_sparse:
                autoencoder.normalize_decoder_columns_()

            # Optimizer step
            optimizer.step()

            # global_step is 0-based here, so this step is update number global_step + 1.
            if (global_step + 1) in snapshot_steps:
                snapshot_dir = write_residual_snapshot(autoencoder, config, model_cfg,
                                                       global_step + 1, epoch, frozen_meta)
                print(f"\nSaved residual snapshot at step {global_step + 1} to {snapshot_dir}")


            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_l2 += metrics.get('l2_loss', 0)
            epoch_span += metrics.get('span_loss', 0)
            epoch_sim += metrics.get('sim_loss', 0)
            epoch_active += metrics.get('num_active', 0)
            epoch_mono_loss += metrics.get('mono_loss_val', 0)

            # Calculate binary purity based on single label or multi-label class_idx 
            binary_purity = _mean_active_binary_purity(acts, class_idx)
            if binary_purity is not None:
                epoch_binary_purity += binary_purity
                purity_batches += 1

            num_batches += 1
            
            # Update progress bar
            num_dead = metrics.get('num_dead_features', 0)
            l0 = metrics.get('num_active', 0)
            expected_active = config.top_k
            
            postfix_dict = {
                'loss': f"{loss.item():.4e}",
                'active': f"{l0:.1f}/{expected_active}",
                'dead': int(num_dead)
            }
            if warmup_active and not warmup_complete:
                postfix_dict['warmup'] = 'mono-only'
            pbar.set_postfix(postfix_dict)

            # Check warmup completion conditions
            if not warmup_complete:
                should_check_mono = (
                    config.warmup_mono_target > 0 and 
                    global_step > 0 and 
                    global_step % config.warmup_check_interval == 0
                )
                
                # Check if warmup period is complete
                warmup_steps_done = config.warmup_steps > 0 and global_step >= config.warmup_steps
                warmup_mono_achieved = False
                
                if should_check_mono:
                    # Compute current mono score
                    print(f"\nChecking mono score at step {global_step}...")
                    current_mono = compute_monosemanticity_fast(
                        autoencoder, train_loader, config.device, 'train', verbose=False
                    )
                    current_mono_mean = current_mono.cpu().numpy().mean()
                    print(f"Current mono score: {current_mono_mean:.4f} (target: {config.warmup_mono_target:.4f})")
                    
                    if current_mono_mean >= config.warmup_mono_target:
                        warmup_mono_achieved = True
                
                if warmup_steps_done or warmup_mono_achieved:
                    warmup_complete = True
                    print(f"\n{'='*60}")
                    print("WARM START COMPLETE - ENABLING POLY BRANCH")
                    print(f"{'='*60}")
                    if warmup_steps_done:
                        print(f"Completed {global_step} warmup steps")
                    if warmup_mono_achieved:
                        print(f"Achieved mono score target: {current_mono_mean:.4f} >= {config.warmup_mono_target:.4f}")
                    
                    # Unfreeze poly branch
                    for param in autoencoder.poly.parameters():
                        param.requires_grad = True
                    print("Poly branch unfrozen (full dual training enabled)")

                    if config.poly_warmup_zero:
                        autoencoder.poly_bypass = False
                        # A checkpoint saved during warm-up was scored with r_hat forced to 0,
                        # but a loaded model has the bypass off, so it would describe a different
                        # model from the one that earned its score. Restart selection here. The
                        # budget guard ensures at least one epoch-end follows warm-up.
                        best_val_mono = float('-inf')
                        best_val_mono_epoch = None
                        best_val_mono_metrics = None
                        print("Residual bypass cleared; best-checkpoint selection restarted")


                    # Create optimizer with differential learning rates
                    mono_lr = config.lr
                    poly_lr = config.lr * config.poly_lr_factor
                    # train_threshold: False == the default recipe (jumprelu.log_threshold
                    # frozen from here on, POST_WARMUP_EXCLUDED_PARAMS); True only under the
                    # opt-in --dual_train_threshold_post_warmup.
                    mono_params, poly_params = build_post_warmup_param_groups(
                        autoencoder, mono_lr, poly_lr,
                        train_threshold=config.dual_train_threshold_post_warmup)
                    optimizer = torch.optim.Adam(mono_params + poly_params, eps=config.eps)
                    if config.dual_train_threshold_post_warmup:
                        # Printed so the training log itself records the deviation.
                        print("jumprelu.log_threshold stays in the optimizer after warm-up "
                              "(--dual_train_threshold_post_warmup; declared deviation, "
                              f"lr={mono_lr:.2e})")
                    
                    print(f"Differential learning rates: mono_lr={mono_lr:.2e} (factor={config.poly_lr_factor}), poly_lr={poly_lr:.2e}")
                    print(f"{'='*60}\n")

            # Check and reinitialize dead neurons periodically. Skipped for the frozen arm because
            # respawning overwrites W_enc/W_dec/b_enc directly.
            if (not frozen_sparse and global_step > 0
                    and global_step % config.dead_check_interval == 0):
                dead_mask = autoencoder.num_batches_not_active > dead_steps_threshold_steps
                num_dead = dead_mask.sum().item()
                if num_dead > 0:
                    print(f"\nReinitializing {num_dead} dead neurons")
                    reinitialize_dead_neurons(autoencoder, dead_mask, sample_batch, config.device)
            
            global_step += 1
        
        # Epoch-end logging
        avg_loss = epoch_loss / num_batches
        avg_l2 = epoch_l2 / num_batches
        avg_span = epoch_span / num_batches
        avg_sim = epoch_sim / num_batches
        avg_active = epoch_active / num_batches
        avg_mono_loss = epoch_mono_loss / num_batches
        avg_binary_purity = epoch_binary_purity / purity_batches if purity_batches > 0 else 0.0
        
        warmup_status = " [WARMUP: mono-only]" if warmup_active and not warmup_complete else ""
        print(f"Epoch {epoch+1}/{config.num_epochs}{warmup_status} - Train Loss: {avg_loss:.4e}, Train L2: {avg_l2:.4e}, "
              f"Train Span: {avg_span:.4e}, Train Sim: {avg_sim:.4e}, Train Active: {avg_active:.1f}, "
              f"Train Binary Purity: {avg_binary_purity:.4f}")
        
        # Validation at end of epoch
        if val_loader:
            val_metrics = evaluate(autoencoder, val_loader, config, 'val')
            print(f"Epoch {epoch+1}/{config.num_epochs} - Val Loss: {val_metrics['val_loss']:.4e}, Val L2: {val_metrics['val_l2']:.4e}, "
                f"Val Span: {val_metrics['val_span']:.4e}, Val Sim: {val_metrics['val_sim']:.4e}, "
                f"Val Cos(r,r_hat): {val_metrics['val_r_rhat_cosine']:.4f}, "
                f"Val Active: {val_metrics['val_active']:.1f}, Val Binary Purity: {val_metrics['val_binary_purity']:.4f}")
        
        # Epoch-end R2. Train-set passes stream the full 1.28M (expensive) -> gate. Val R2 (50K)
        # is cheap and used in best-ckpt metrics, so keep it.
        train_r2 = float('nan')
        train_r2_mono = float('nan')
        if not config.skip_extra_eval:
            train_r2 = compute_r2(autoencoder, train_loader, config.device, split_name='train')
            print(f"Epoch {epoch+1}/{config.num_epochs} - Train R2: {train_r2:.6f}")
            train_r2_mono = compute_mono_branch_r2(autoencoder, train_loader, config.device, split_name='train')
            print(f"Epoch {epoch+1}/{config.num_epochs} - Train Mono-Branch R2: {train_r2_mono:.6f}")
        if val_loader:
            val_r2 = compute_r2(autoencoder, val_loader, config.device, split_name='val')
            print(f"Epoch {epoch+1}/{config.num_epochs} - Val   R2: {val_r2:.6f}")
            val_r2_mono = compute_mono_branch_r2(autoencoder, val_loader, config.device, split_name='val')
            print(f"Epoch {epoch+1}/{config.num_epochs} - Val   Mono-Branch R2: {val_r2_mono:.6f}")

        # Train sparse MonoScore (over 1.28M) is logging-only -> gate. (Val MonoScore for
        # best-ckpt selection is computed in the val block below.)
        train_mono_mean = float('nan')
        train_mono_median = float('nan')
        if not config.skip_extra_eval:
            train_mono = compute_monosemanticity_fast(autoencoder, train_loader, config.device, 'train', verbose=False)
            train_mono_np = train_mono.cpu().numpy()
            train_mono_mean = np.mean(train_mono_np)
            train_mono_median = np.median(train_mono_np)
            print(f"Epoch {epoch+1}/{config.num_epochs} - Train Mono (sparse): mean={train_mono_mean:.4f}, median={train_mono_median:.4f}")

        # r_hat (poly-branch) monosemanticity is a diagnostic over the full train set -> gate.
        # Also skipped with no residual branch. Beyond being meaningless there, every forward
        # pass here advances the dead-latent counter, and a standard SAE has warmup_complete=True
        # from step 0. Left on, its respawn clock would run at a DualSAE's joint-stage rate
        # instead of the warm-up rate, a difference unrelated to what the arm is measuring.
        train_mono_rhat_mean = 0.0
        train_mono_rhat_median = 0.0
        if warmup_complete and not config.skip_extra_eval and config.poly_arch != 'none':
            train_mono_rhat = compute_r_hat_monosemanticity(autoencoder, train_loader, config.device, eps=1e-8, split_name='train', verbose=False)
            train_mono_rhat_np = train_mono_rhat.cpu().numpy()
            train_mono_rhat_mean = np.mean(train_mono_rhat_np)
            train_mono_rhat_median = np.median(train_mono_rhat_np)
            print(f"Epoch {epoch+1}/{config.num_epochs} - Train Mono (r_hat): mean={train_mono_rhat_mean:.4f}, median={train_mono_rhat_median:.4f}")
        
        # Save validation embeddings periodically
        if config.save_val_embeds and val_loader and (epoch + 1) % config.save_embeds_interval == 0:
            embeds_path = embeds_dir / f'val_embeddings_epoch_{epoch+1:03d}.pt'
            print(f"Saving validation embeddings to {embeds_path}...")
            save_intermediate_outputs(
                autoencoder,
                val_loader,
                embeds_path,
                device=config.device,
                split_name=f'val_epoch_{epoch+1}',
                include_r_hat=True
            )
            print(f"Saved {embeds_path}")
        
        epoch_log_data = {
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_l2': avg_l2,
            'train_span': avg_span,
            'train_sim': avg_sim,
            'train_active': avg_active,
            'train_binary_purity': avg_binary_purity,
            'train_r2': train_r2,
            'train_r2_mono': train_r2_mono,
            'train_mono_mean': train_mono_mean,
            'train_mono_rhat_mean': train_mono_rhat_mean,
            'train_mono_loss': avg_mono_loss,
            # 'train_mono_median': train_mono_median,
            # 'train_mono_rhat_median': train_mono_rhat_median,
        }
        
        if val_loader:
            val_mono = compute_monosemanticity_fast(autoencoder, val_loader, config.device, 'val', verbose=False)
            val_mono_np = val_mono.cpu().numpy()
            val_mono_mean = np.mean(val_mono_np)
            val_mono_median = np.median(val_mono_np)
            print(f"Epoch {epoch+1}/{config.num_epochs} - Val Mono (sparse): mean={val_mono_mean:.4f}, median={val_mono_median:.4f}")

            # With a frozen z, val MonoScore cannot improve, so a strict comparison would keep
            # epoch 1 and report an almost untrained projector. Always keep the latest epoch.
            if frozen_sparse or (epoch + 1 >= config.select_from_epoch
                                 and val_mono_mean > best_val_mono):
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

            # Compute monosemanticity for r_hat (poly branch output) — diagnostic only -> gate.
            # Same no-residual exclusion as the train-set pass above, for the same reason.
            if warmup_complete and not config.skip_extra_eval and config.poly_arch != 'none':
                val_mono_rhat = compute_r_hat_monosemanticity(autoencoder, val_loader, config.device, eps=1e-8, split_name='val', verbose=False)
                val_mono_rhat_np = val_mono_rhat.cpu().numpy()
                val_mono_rhat_mean = np.mean(val_mono_rhat_np)
                val_mono_rhat_median = np.median(val_mono_rhat_np)
                print(f"Epoch {epoch+1}/{config.num_epochs} - Val Mono (r_hat): mean={val_mono_rhat_mean:.4f}, median={val_mono_rhat_median:.4f}")
            else:
                val_mono_rhat_mean = 0.0
                val_mono_rhat_median = 0.0
            
            epoch_log_data.update({
                **val_metrics,
                'val_r2': val_r2,
                'val_r2_mono': val_r2_mono,
                'val_mono_mean': val_mono_mean,
                'val_mono_rhat_mean': val_mono_rhat_mean,
                'val_mono_loss': avg_mono_loss,
                'val_mono_median': val_mono_median,
                'val_mono_rhat_median': val_mono_rhat_median,
            })
        
        if config.wandb_project:
            wandb.log(epoch_log_data)
        tb.log(epoch_log_data, step=epoch)   # TensorBoard: per-epoch scalars (step=epoch)

        # Per-epoch checkpoints (~150MB each with optimizer state) are opt-out
        # via --skip_epoch_checkpoints: nothing in this repository loads them.
        if not config.skip_epoch_checkpoints:
            epoch_checkpoint_path = epoch_checkpoint_dir / f'epoch_{epoch + 1:03d}.pt'
            torch.save(
                {
                    'epoch': epoch + 1,
                    'step': global_step,
                    'model_state_dict': autoencoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config_payload(config),   # unset opt-in fields dropped
                },
                epoch_checkpoint_path,
            )
            print(f"Saved epoch checkpoint to {epoch_checkpoint_path}")
        
        print("")
    

    # The frozen arm's whole claim is that z never changed. Checked before autoencoder_final.pt
    # exists, so a violation leaves the run incomplete instead of reportable.
    if frozen_sparse and sparse_branch_sha256(autoencoder) != frozen_meta['sparse_sha256']:
        raise RuntimeError(f"[{config.exp_name}] the frozen sparse branch changed during training; "
                           "this run must not be reported")

    # Save final model
    final_path = Path(config.output_dir) / 'autoencoder_final.pt'
    if best_val_mono_epoch is not None:
        print(
            f"Best validation monoscore epoch: {best_val_mono_epoch} "
            f"(score={best_val_mono:.4f}); intermediates: {best_val_intermediates_path}"
        )
    torch.save({
        'epoch': config.num_epochs,
        'step': global_step,
        'model_state_dict': autoencoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # vars(config) minus unset opt-in fields (utils.config_payload): unchanged key set for
        # every run that does not pass --jump_threshold_init.
        'config': config_payload(config),
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
        if config.wandb_project:
            wandb.finish()
        tb.close()   # flush + close TB writer on the early (skip_extra_eval) exit — the live path
        return autoencoder

    # EVALUATION ON THE EPOCHS WITH BEST VAL MONOSCORE
    best_ckpt = torch.load(best_val_model_path, map_location=config.device)
    best_model_cfg = best_ckpt.get('model_cfg', model_cfg)
    best_model = DualSAE(best_model_cfg).to(config.device)
    best_model.load_state_dict(best_ckpt['model_state_dict'])
    best_model.eval()


    # EVALUATE R2
    best_val_r2 = compute_r2(best_model, val_loader, config.device, split_name='val')

    # EVALUATE MONOSEMANTICITY
    best_mono = compute_monosemanticity_fast(best_model, val_loader, config.device, 'val', verbose=True)
    best_mono_mean = best_mono.cpu().numpy().mean()
    best_val_purity = compute_active_binary_purity(best_model, val_loader, config.device, split_name='val')

    # EVALUATE ACC(KNN) AND ACC(LP)
    val_clf_metrics = evaluate_reconstruction_classification(
        best_model,
        train_loader,
        val_loader,
        device=config.device,
        split_name='val',
    )
    print(
        f"Best-epoch val metrics - mono={best_mono_mean:.4f}, "
        f"r2={best_val_r2:.6f}, "
        f"purity={best_val_purity:.4f}, "
        f"knn={val_clf_metrics['val_knn_acc_recon']:.4f}, "
        f"lp={val_clf_metrics['val_lp_acc_recon']:.4f}"
    )

    if run is not None:
        run.summary['BEST_MONOSCORE'] = best_mono_mean
        run.summary['BEST_EPOCH'] = best_val_mono_epoch
        run.summary['BEST_R2'] = best_val_r2
        run.summary['BEST_PURITY'] = best_val_purity
        run.summary['BEST_ACC_KNN'] = val_clf_metrics['val_knn_acc_recon']
        run.summary['BEST_ACC_LP'] = val_clf_metrics['val_lp_acc_recon']
    tb.summary({                                 # TensorBoard: best/* scalars at the selected epoch
        'MONOSCORE': best_mono_mean, 'EPOCH': best_val_mono_epoch, 'R2': best_val_r2,
        'PURITY': best_val_purity, 'ACC_KNN': val_clf_metrics['val_knn_acc_recon'],
        'ACC_LP': val_clf_metrics['val_lp_acc_recon'],
    }, step=best_val_mono_epoch)

    # Same numbers on disk, keyed exactly like the W&B summary above, so the collectors need
    # no experiment tracker. On multi-label OpenImagesV7 the KNN/LP entries hold micro-F1
    # (the trainers switch metric with the dataset; the key names stay the same). Written
    # before the optional test pass, so an interruption there still leaves the val metrics.
    best_metrics_record = {
        'BEST_MONOSCORE': best_mono_mean,
        'BEST_EPOCH': best_val_mono_epoch,
        'BEST_R2': best_val_r2,
        'BEST_PURITY': best_val_purity,
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
        best_test_purity = compute_active_binary_purity(best_model, test_loader, config.device, split_name='test')

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
            f"purity={best_test_purity:.4f}, "
            f"knn={test_clf_metrics['test_knn_acc_recon']:.4f}, "
            f"lp={test_clf_metrics['test_lp_acc_recon']:.4f}"
        )

        if run is not None:
            run.summary['BEST_TEST_MONOSCORE'] = best_test_mono_mean
            run.summary['BEST_TEST_R2'] = best_test_r2
            run.summary['BEST_TEST_PURITY'] = best_test_purity
            run.summary['BEST_TEST_ACC_KNN'] = test_clf_metrics['test_knn_acc_recon']
            run.summary['BEST_TEST_ACC_LP'] = test_clf_metrics['test_lp_acc_recon']
            wandb.log({
                'best_test_mono_mean': best_test_mono_mean,
                'best_test_r2': best_test_r2,
                'best_test_purity': best_test_purity,
                'best_test_knn_metric_recon': test_clf_metrics.get('test_knn_metric_recon', test_clf_metrics['test_knn_acc_recon']),
                'best_test_lp_metric_recon': test_clf_metrics.get('test_lp_metric_recon', test_clf_metrics['test_lp_acc_recon']),
                'best_test_knn_acc_recon': test_clf_metrics['test_knn_acc_recon'],
                'best_test_lp_acc_recon': test_clf_metrics['test_lp_acc_recon'],
            })
        tb.log({                                 # TensorBoard: test metrics (use final epoch as step)
            'best_test_mono_mean': best_test_mono_mean, 'best_test_r2': best_test_r2,
            'best_test_purity': best_test_purity,
            'best_test_knn_acc_recon': test_clf_metrics['test_knn_acc_recon'],
            'best_test_lp_acc_recon': test_clf_metrics['test_lp_acc_recon'],
        }, step=config.num_epochs)
        # Re-write the metrics file with the test columns appended.
        best_metrics_record.update({
            'BEST_TEST_MONOSCORE': best_test_mono_mean,
            'BEST_TEST_R2': best_test_r2,
            'BEST_TEST_PURITY': best_test_purity,
            'BEST_TEST_ACC_KNN': test_clf_metrics['test_knn_acc_recon'],
            'BEST_TEST_ACC_LP': test_clf_metrics['test_lp_acc_recon'],
        })
        write_best_metrics(config.output_dir, best_metrics_record)

    if config.wandb_project:
        wandb.finish()
    tb.close()   # flush + close TB writer on the full-eval exit

    return autoencoder


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a DualSAE")
    parser.add_argument('--dataset_config', type=str, required=True,
                       help='Path to JSON config file with dataset paths')
    parser.add_argument('--exp_name', type=str, default='dual_sae',
                       help='Experiment name')
    parser.add_argument('--n_latents', type=int, default=8192,
                       help='Number of latent dimensions')
    parser.add_argument('--top_k', type=int, default=64,
                       help='Sparsity for monosemantic branch')
    parser.add_argument('--sparsity_mode', type=str, default='topk',
                       choices=['topk', 'batch_topk', 'vanilla', 'jump_relu'],
                       help='Sparsity mode: topk, batch_topk, vanilla, or jump_relu')
    parser.add_argument('--l1_coef', type=float, default=1e-4,
                       help='L1 coefficient (used when sparsity_mode=vanilla)')
    parser.add_argument('--bandwidth', type=float, default=0.001,
                       help='JumpReLU bandwidth (used when sparsity_mode=jump_relu)')
    parser.add_argument('--poly_hidden', type=int, default=512,
                       help='Hidden dimension for polynomial MLPk')
    parser.add_argument('--poly_layers', type=int, default=1,
                       help='Number of linear layers in polynomial projector MLP')
    parser.add_argument('--poly_hidden_dims', type=str, default='',
                       help='Comma-separated hidden widths for projector MLP, e.g. 1024,512,256 (overrides poly_layers/poly_hidden)')
    parser.add_argument('--batch_size', type=int, default=2048,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--span_coef', type=float, default=0.01,
                       help='Coefficient for span overlap penalty; paper default 0.01')
    parser.add_argument('--sim_coef', type=float, default=0.0,
                       help='Coefficient for similarity penalty between xM and r_hat')
    parser.add_argument('--active_only_span', action='store_true', default=True,
                       help='Only apply span penalty to active latents')
    parser.add_argument('--mono_coef', type=float, default=0.0,
                       help='Coefficient for monosemanticity loss')
    parser.add_argument('--warmup_steps', type=int, default=0,
                       help='Train mono branch alone for this many steps (0=disabled)')
    parser.add_argument('--warmup_mono_target', type=float, default=0.0,
                       help='Train mono branch until mono score reaches this value (0=disabled)')
    parser.add_argument('--warmup_check_interval', type=int, default=500,
                       help='Check mono score every N steps during warmup')
    parser.add_argument('--dead_steps_threshold', type=int, default=10_000_000,
                       help='Steps threshold for considering a neuron dead')
    parser.add_argument('--dead_check_interval', type=int, default=1000,
                       help='Check for dead neurons every N steps')
    parser.add_argument('--save_val_embeds', action='store_true', default=False,
                       help='Save validation embeddings (x, z, xM, r_hat, x_hat) during training')
    parser.add_argument('--save_embeds_interval', type=int, default=1,
                       help='Save validation embeddings every N epochs (1=every epoch)')
    parser.add_argument('--skip_extra_eval', action='store_true', default=False,
                       help='Skip heavy per-epoch/end eval; keep training + val-MonoScore selection.')
    parser.add_argument('--skip_epoch_checkpoints', action='store_true', default=False,
                       help='Do not write the ~150MB per-epoch checkpoints (never read by anything).')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--eps', type=float, default=6.25e-10,
                       help='Adam epsilon')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='Wandb project name')
    parser.add_argument('--tb_dir', type=str, default=None,
                       help='TensorBoard event dir (real-time offline). No-op if unset.')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Output directory')

    args = parser.parse_args()
    
    # Load dataset paths
    with open(args.dataset_config, 'r') as f:
        config_dict = json.load(f)
    
    parsed_poly_hidden_dims = None
    if args.poly_hidden_dims.strip():
        parsed_poly_hidden_dims = [int(x.strip()) for x in args.poly_hidden_dims.split(',') if x.strip()]

    config = TrainingConfig(
        train_path=config_dict.get('train_path'),
        val_path=config_dict.get('val_path'),
        test_path=config_dict.get('test_path'),
        exp_name=args.exp_name,
        n_latents=args.n_latents,
        top_k=args.top_k,
        sparsity_mode=args.sparsity_mode,
        l1_coef=args.l1_coef,
        bandwidth=args.bandwidth,
        poly_hidden=args.poly_hidden,
        poly_layers=args.poly_layers,
        poly_hidden_dims=parsed_poly_hidden_dims,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        span_coef=args.span_coef,
        sim_coef=args.sim_coef,
        active_only_span=args.active_only_span,
        mono_coef=args.mono_coef,
        warmup_steps=args.warmup_steps,
        warmup_mono_target=args.warmup_mono_target,
        warmup_check_interval=args.warmup_check_interval,
        dead_steps_threshold=args.dead_steps_threshold,
        dead_check_interval=args.dead_check_interval,
        save_val_embeds=args.save_val_embeds,
        save_embeds_interval=args.save_embeds_interval,
        skip_extra_eval=args.skip_extra_eval,
        skip_epoch_checkpoints=args.skip_epoch_checkpoints,
        seed=args.seed,
        eps=args.eps,
        num_workers=args.num_workers,
        wandb_project=args.wandb_project,
        tb_dir=args.tb_dir,
        output_dir=args.output_dir,
    )

    train_autoencoder(config)
