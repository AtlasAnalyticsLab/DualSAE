import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple, Dict
from utils import unpack_batch
from typing import List, Optional, Sequence

def compute_monosemanticity_loss_batch(x: torch.Tensor, latents: torch.Tensor, 
                                       eps: float = 1e-8) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute monosemanticity loss on a batch using 1 - mono_score.
    
    Args:
        x: Input embeddings (B, D) - will be normalized to unit norm
        latents: Neuron activations (B, M) - should be non-negative
        eps: Small constant for numerical stability
    
    Returns:
        loss: 1 - mean(monosemanticity), range [0, 1]
        metrics: Dictionary with monosemanticity statistics
    """
    # Normalize embeddings to unit norm for cosine similarity
    x_norm = F.normalize(x, p=2, dim=1, eps=eps)  # (B, D)
    
    # Normalize activations per neuron to [0, 1] within batch
    min_acts = latents.min(dim=0, keepdim=True)[0]  # (1, M)
    max_acts = latents.max(dim=0, keepdim=True)[0]  # (1, M)
    a = (latents - min_acts) / (max_acts - min_acts + eps)  # (B, M) this is a_tilde in the paper
    
    # Fast monosemanticity via streaming formula (same as compute_monosemanticity_fast)
    # V = X^T @ A, where rows are weighted embeddings per neuron
    w = x_norm.T @ a  # (D, M)
    
    # Numerator: Σ_{i<j} a_i·a_j·cos(e_i,e_j) = 0.5(||Σ a_i·e_i||² - Σ a_i²)
    q = (w * w).sum(dim=0)  # (M,)
    v = (a * a).sum(dim=0)  # (M,)
    num = 0.5 * (q - v)
    
    # Denominator: Σ_{i<j} a_i·a_j = 0.5((Σ a_i)² - Σ a_i²)
    u = a.sum(dim=0)  # (M,)
    den = 0.5 * (u * u - v)
    
    # Monosemanticity score per neuron
    mono_scores = torch.where(
        den > eps,
        num / (den + eps),
        torch.zeros_like(den)
    )  # (M,)
    
    # Only consider active neurons
    active_mask = den > eps
    num_active = active_mask.sum().item()
    
    if num_active > 0:
        mono_mean = mono_scores[active_mask].mean()
        mono_median = mono_scores[active_mask].median()
    else:
        mono_mean = torch.tensor(0.0, device=x.device)
        mono_median = torch.tensor(0.0, device=x.device)
    
    # Loss: 1 - mono_score (minimize to maximize monosemanticity)
    loss = 1.0 - mono_mean
    
    metrics = {
        'mono_loss_val': loss.item(),
        'mono_score': mono_mean.item(),
        'mono_median': mono_median.item(),
        'mono_active_neurons': num_active,
    }
    
    return loss, metrics


def compute_monosemanticity_loss_batch_ref(x: torch.Tensor,
                                           latents: torch.Tensor,
                                           eps: float = 1e-8,
                                           pair_batch_size: int = 1024) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Reference implementation of monosemanticity loss on a single batch via explicit pairwise accumulation.
    This is O(B^2·M) and intended for validation against the fast formula.
    
    Args:
        x: Input embeddings (B, D) - unnormalized; cosine similarity uses unit-normalized vectors
        latents: Neuron activations (B, M) - will be scaled to [0, 1] across the batch per neuron
        eps: Small constant for numerical stability
        pair_batch_size: Chunk size for the inner j-loop to control memory usage
    
    Returns:
        loss: 1 - mean(monosemanticity over active neurons), range [0, 1]
        metrics: Dictionary with monosemanticity statistics
    """
    B = x.shape[0]
    device = x.device
    
    # Normalize embeddings to unit norm for cosine similarity
    x_norm = F.normalize(x, p=2, dim=1, eps=eps)  # (B, D)
    
    # Normalize activations per neuron to [0, 1] within the batch
    min_acts = latents.min(dim=0, keepdim=True)[0]  # (1, M)
    max_acts = latents.max(dim=0, keepdim=True)[0]  # (1, M)
    a = (latents - min_acts) / (max_acts - min_acts + eps)  # (B, M)
    
    M = a.shape[1]
    weighted_cosine_sum = torch.zeros(M, device=device, dtype=torch.float32)
    weight_sum = torch.zeros(M, device=device, dtype=torch.float32)
    
    for i in range(B-1):
        # Preload row i
        emb_i = x_norm[i]                 # (D,)
        act_i = a[i]                      # (M,)
        
        for j_start in range(i + 1, B, pair_batch_size):
            j_end = min(j_start + pair_batch_size, B)
            
            emb_j = x_norm[j_start:j_end]         # (chunk, D)
            act_j = a[j_start:j_end]              # (chunk, M)
            
            # Cosine similarities between i and j chunk
            cos_ij = F.cosine_similarity(
                emb_i.unsqueeze(0).expand(j_end - j_start, -1),  # (chunk, D)
                emb_j,
                dim=1
            ).to(torch.float32)  # (chunk,)
            
            # Weights per neuron for each pair (i, j): a_i * a_j
            w_ij = act_i.unsqueeze(0) * act_j  # (chunk, M)
            
            # Accumulate weighted cosine similarities and weights
            weighted_cosine_sum += (w_ij * cos_ij.unsqueeze(1)).sum(dim=0)  # (M,)
            weight_sum += w_ij.sum(dim=0)  # (M,)
    
    # Per-neuron monosemanticity
    mono_scores = torch.where(
        weight_sum > eps,
        weighted_cosine_sum / (weight_sum + eps),
        torch.zeros_like(weight_sum)
    )  # (M,)
    
    # Consider only active neurons
    active_mask = weight_sum > eps
    num_active = active_mask.sum().item()
    
    if num_active > 0:
        mono_mean = mono_scores[active_mask].mean()
        mono_median = mono_scores[active_mask].median()
    else:
        mono_mean = torch.tensor(0.0, device=device)
        mono_median = torch.tensor(0.0, device=device)
    
    loss = 1.0 - mono_mean
    metrics = {
        'mono_loss_val': loss.item(),
        'mono_score': mono_mean.item(),
        'mono_median': mono_median.item(),
        'mono_active_neurons': num_active,
    }
    return loss, metrics


@torch.no_grad()
def compute_monosemanticity_ref(autoencoder, dataloader, device='cuda', split_name='val', 
                                pair_batch_size: int = 128, eps: float = 1e-8, verbose: bool = False):
    """
    Reference MonoScore implementation using explicit nested loops over all pairs.
    
    
    Args:
        autoencoder: The autoencoder model
        dataloader: DataLoader for the dataset
        device: Device to run on
        split_name: Name of the split for logging
        pair_batch_size: Batch size for j loop (memory control)
        eps: Small constant for numerical stability
        verbose: If True, print progress messages
    
    Returns:
        monosemanticity: (M,) tensor of scores per neuron
    """
    autoencoder.eval()
    dataset = dataloader.dataset
    num_images = len(dataset)
    
    # Pass 1: Collect all embeddings and activations with normalization, also collect min and max values for activations
    if verbose:
        print(f"First pass: collecting all embeddings and activations ({split_name})...")
    all_embeddings = []
    all_activations = []
    min_vals = None
    max_vals = None
    num_neurons = None
    
    for batch in tqdm(dataloader, desc=f"Collecting data for {split_name}", disable=not verbose):
        # if not isinstance(batch, torch.Tensor):
        #     batch = torch.from_numpy(batch)
        x, _, _ = unpack_batch(batch)
        x = x.to(device)

        _, latents, _ = autoencoder(x)
        
        if min_vals is None:
            num_neurons = latents.shape[1]
            min_vals = latents.min(dim=0, keepdim=True)[0]
            max_vals = latents.max(dim=0, keepdim=True)[0]
        else:
            batch_min = latents.min(dim=0, keepdim=True)[0]
            batch_max = latents.max(dim=0, keepdim=True)[0]
            min_vals = torch.min(min_vals, batch_min)
            max_vals = torch.max(max_vals, batch_max)
        
        all_embeddings.append(x)
        all_activations.append(latents)
    
    embeddings = torch.cat(all_embeddings, dim=0)  # (N, D)
    activations = torch.cat(all_activations, dim=0)  # (N, M)
    
    # Normalize activations per neuron to [0, 1]
    activations = (activations - min_vals) / (max_vals - min_vals + eps)
    
    if verbose:
        print(f"Dataset size: {num_images}")
        print(f"Number of neurons: {num_neurons}")
    
    # Pass 2: Nested loop over all pairs i < j
    if verbose:
        print(f"Second pass: computing pairwise monosemanticity (ref {split_name})...")
    weighted_cosine_similarity_sum = torch.zeros(num_neurons, device=device)
    weight_sum = torch.zeros(num_neurons, device=device)
    
    for i in tqdm(range(num_images), desc="Processing image pairs", disable=not verbose):
        for j_start in range(i + 1, num_images, pair_batch_size):  # Process in batches
            j_end = min(j_start + pair_batch_size, num_images)
            
            embeddings_i = embeddings[i]  # (D,)
            embeddings_j = embeddings[j_start:j_end]  # (batch, D)
            activations_i = activations[i]  # (M,)
            activations_j = activations[j_start:j_end]  # (batch, M)
            
            cosine_similarities = F.cosine_similarity(
                embeddings_i.unsqueeze(0).expand(j_end - j_start, -1),  # (batch, D)
                embeddings_j,
                dim=1
            )  # (batch,)
            
            # Compute weights and weighted similarities
            weights = activations_i.unsqueeze(0) * activations_j  # (batch, M)
            weighted_cosine_similarities = weights * cosine_similarities.unsqueeze(1)  # (batch, M)
            
            # Accumulate
            weighted_cosine_similarity_sum += weighted_cosine_similarities.sum(dim=0)  # (M,)
            weight_sum += weights.sum(dim=0)  # (M,)
    
    # Compute final monosemanticity scores
    monosemanticity = torch.where(
        weight_sum > eps,
        weighted_cosine_similarity_sum / weight_sum,
        torch.zeros_like(weight_sum)  
    )
    
    autoencoder.train()
    return monosemanticity


@torch.no_grad()
def compute_monosemanticity_fast(autoencoder, dataloader, device='cuda', split_name='val', eps: float = 1e-8, verbose: bool = False):
    """
    Fast O(N·D·M) monosemanticity in single pass (no pairwise loops).

    Args:
        autoencoder: The autoencoder model
        dataloader: DataLoader for the dataset
        device: Device to run on
        split_name: Name of the split for logging
        eps: Small constant for numerical stability
        verbose: If True, print progress messages

    Returns:
        monosemanticity: (M,) tensor of scores per neuron
    """
    autoencoder.eval()

    dataset = dataloader.dataset

    # Pass 1: activation min/max per neuron for scaling
    if verbose:
        print(f"First pass: computing activation statistics (fast {split_name})...")
    min_vals = None
    max_vals = None
    num_neurons = None
    feature_dim = None

    for batch in tqdm(dataloader, desc=f"Computing stats for {split_name}", disable=not verbose):
        # if not isinstance(batch, torch.Tensor):
        #     batch = torch.from_numpy(batch)
        x, _, _ = unpack_batch(batch)
        x = x.to(device)

        _, latents, _ = autoencoder(x)

        if min_vals is None:
            num_neurons = latents.shape[1]
            feature_dim = x.shape[1]
            min_vals = latents.min(dim=0, keepdim=True)[0]
            max_vals = latents.max(dim=0, keepdim=True)[0]
        else:
            batch_min = latents.min(dim=0, keepdim=True)[0]
            batch_max = latents.max(dim=0, keepdim=True)[0]
            min_vals = torch.min(min_vals, batch_min)
            max_vals = torch.max(max_vals, batch_max)

    dataset_size = len(dataset)
    if verbose:
        print(f"Dataset size: {dataset_size}")
        print(f"Number of neurons: {num_neurons}")

    # Pass 2: stream accumulators
    if verbose:
        print(f"Second pass: streaming accumulators...")
    # Accumulators in FP32 for numerical stability
    u = torch.zeros(num_neurons, device=device, dtype=torch.float32)
    v = torch.zeros(num_neurons, device=device, dtype=torch.float32)
    w = torch.zeros(feature_dim, num_neurons, device=device, dtype=torch.float32)

    for batch in tqdm(dataloader, desc=f"Accumulating {split_name}", disable=not verbose):
        # if not isinstance(batch, torch.Tensor):
        #     batch = torch.from_numpy(batch)
        x, _, _ = unpack_batch(batch)
        x = x.to(device)

        _, latents, _ = autoencoder(x)

        # Normalize activations per neuron to [0, 1]
        latents = (latents - min_vals) / (max_vals - min_vals + eps)
        a = latents

        # Unit-normalize embeddings to match cosine similarity
        p = F.normalize(x, p=2, dim=1, eps=1e-8)

        # Cast to fp32 for accumulators
        a32 = a.float()
        p32 = p.float()

        # Update accumulators
        u += a32.sum(dim=0)
        v += (a32 * a32).sum(dim=0)
        w += p32.T @ a32  # (D, B) @ (B, M) -> (D, M)

    # Compute per-neuron weighted sums
    q = (w * w).sum(dim=0)  # (M,)
    num = 0.5 * (q - v)  # Σ_{i<j} a_i a_j cos 
    den = 0.5 * (u * u - v)      # Σ_{i<j} a_i a_j

    # Final monosemanticity scores
    monosemanticity = torch.where(
        den > 1e-8,
        num / (den + 1e-12),
        torch.zeros_like(den)
    )
    autoencoder.train()
    return monosemanticity


# def compute_monosemanticity_loss_dataset_fast(
#     autoencoder,
#     dataloader,
#     device: str = "cuda",
#     eps: float = 1e-8,
#     detach_stats: bool = True,
# ) -> Tuple[torch.Tensor, Dict[str, float]]:
#     """
#     Fast dataset-level monosemanticity loss, usable as a training loss.

#     Important:
#     - This is differentiable w.r.t. the activations and therefore the model.
#     - By default, global min/max stats are detached. That is usually what you want.
#     - Exact end-to-end gradients through dataset-wide min/max across the full loader
#       are not practical because you'd need to retain the computation graph for the
#       entire dataset.

#     Returns:
#         loss: scalar tensor
#         metrics: logging dict
#     """

#     was_training = autoencoder.training
#     autoencoder.train()

#     # -------------------------
#     # Pass 1: global activation stats
#     # -------------------------
#     min_vals = None
#     max_vals = None
#     feature_dim = None
#     num_neurons = None

#     for batch in dataloader:
#         x, _, _ = unpack_batch(batch)
#         x = x.to(device)

#         _, latents, _ = autoencoder(x)

#         if min_vals is None:
#             num_neurons = latents.shape[1]
#             feature_dim = x.shape[1]
#             min_vals = latents.min(dim=0, keepdim=True).values
#             max_vals = latents.max(dim=0, keepdim=True).values
#         else:
#             min_vals = torch.minimum(min_vals, latents.min(dim=0, keepdim=True).values)
#             max_vals = torch.maximum(max_vals, latents.max(dim=0, keepdim=True).values)

#     if detach_stats:
#         min_vals = min_vals.detach()
#         max_vals = max_vals.detach()

#     # -------------------------
#     # Pass 2: differentiable accumulation
#     # -------------------------
#     u = None
#     v = None
#     w = None

#     for batch in dataloader:
#         x, _, _ = unpack_batch(batch)
#         x = x.to(device)

#         _, latents, _ = autoencoder(x)

#         # Global [0,1] normalization per neuron
#         a = (latents - min_vals) / (max_vals - min_vals + eps)

#         # Optional safety clamp
#         a = a.clamp(min=0.0, max=1.0)

#         # Unit-normalize embeddings for cosine similarity
#         p = F.normalize(x, p=2, dim=1, eps=eps)

#         # Keep same dtype as model graph
#         batch_u = a.sum(dim=0)                  # (M,)
#         batch_v = (a * a).sum(dim=0)            # (M,)
#         batch_w = p.T @ a                       # (D, M)

#         if u is None:
#             u = batch_u
#             v = batch_v
#             w = batch_w
#         else:
#             u = u + batch_u
#             v = v + batch_v
#             w = w + batch_w

#     # Fast formula
#     q = (w * w).sum(dim=0)                     # (M,)
#     num = 0.5 * (q - v)
#     den = 0.5 * (u * u - v)

#     mono_scores = torch.where(
#         den > eps,
#         num / (den + eps),
#         torch.zeros_like(den)
#     )

#     active_mask = den > eps

#     if active_mask.any():
#         mono_mean = mono_scores[active_mask].mean()
#         mono_median = mono_scores[active_mask].median()
#     else:
#         mono_mean = torch.zeros((), device=device, dtype=mono_scores.dtype)
#         mono_median = torch.zeros((), device=device, dtype=mono_scores.dtype)

#     loss = 1.0 - mono_mean

#     metrics = {
#         "mono_loss_val": float(loss.detach().cpu()),
#         "mono_score": float(mono_mean.detach().cpu()),
#         "mono_median": float(mono_median.detach().cpu()),
#         "mono_active_neurons": int(active_mask.sum().detach().cpu()),
#     }

#     if not was_training:
#         autoencoder.eval()

#     return loss, metrics


# def _extract_x(batch):
#     """
#     Handles common dataloader outputs:
#     - Tensor
#     - (x, ...)
#     - [x, ...]
#     """
#     if isinstance(batch, torch.Tensor):
#         return batch
#     if isinstance(batch, (tuple, list)):
#         return batch[0]
#     raise TypeError(f"Unsupported batch type: {type(batch)}")


# def _zeros_like_params(params: Sequence[torch.Tensor]) -> List[Optional[torch.Tensor]]:
#     out = []
#     for p in params:
#         if p.requires_grad:
#             out.append(torch.zeros_like(p))
#         else:
#             out.append(None)
#     return out


# class DatasetMonosemanticityLossFunction(Function):
#     """
#     Custom autograd for exact dataset-level fast monosemanticity loss,
#     with detached global min/max stats and streamed recomputation in backward.

#     Forward computes:
#         a_i = (latent_i - min_vals) / (max_vals - min_vals + eps)
#         p_i = normalize(x_i)

#         u_m = sum_i a_im
#         v_m = sum_i a_im^2
#         w_m = sum_i a_im p_i

#         q_m   = ||w_m||^2
#         num_m = 0.5 * (q_m - v_m)
#         den_m = 0.5 * (u_m^2 - v_m)
#         s_m   = num_m / (den_m + eps), for active neurons

#         loss = 1 - mean_m s_m over active neurons

#     Backward recomputes batches and uses the closed-form gradient wrt a_im:
#         d num_m / d a_im = <p_i, w_m> - a_im
#         d den_m / d a_im = u_m - a_im

#         d s_m / d a_im =
#             [ (dnum)*(den+eps) - num*(dden) ] / (den+eps)^2

#     Then:
#         a_im = (latent_im - min_m) / (range_m + eps)
#         d a_im / d latent_im = 1 / (range_m + eps)

#     Important:
#     - min/max are detached stats, not differentiated through.
#     - gradients are propagated to model parameters through latents only.
#     - this does NOT propagate gradient into x unless x itself is inside the model.
#     """

#     @staticmethod
#     def forward(
#         ctx,
#         *args,
#     ):
#         """
#         Expected args layout:
#             model, dataloader, device, eps, verbose, *params

#         where:
#             model: nn.Module (non-tensor)
#             dataloader: iterable over batches (non-tensor)
#             device: str or torch.device (non-tensor)
#             eps: float (non-tensor)
#             verbose: bool (non-tensor)
#             params: model parameters as tensor inputs so backward can return grads
#         """
#         model = args[0]
#         dataloader = args[1]
#         device = args[2]
#         eps = args[3]
#         verbose = args[4]
#         params = list(args[5:])

#         if not isinstance(model, nn.Module):
#             raise TypeError("First argument must be the model.")
#         if eps <= 0:
#             raise ValueError("eps must be > 0")

#         # Save non-tensor references for backward
#         ctx.model = model
#         ctx.dataloader = dataloader
#         ctx.device = device
#         ctx.eps = float(eps)
#         ctx.verbose = bool(verbose)
#         ctx.num_params = len(params)

#         # Forward itself does not need a graph; backward recomputes.
#         was_training = model.training
#         model.eval()

#         # --------------------------------------------------
#         # Pass 1: global min/max per neuron (detached stats)
#         # --------------------------------------------------
#         min_vals = None
#         max_vals = None
#         num_neurons = None
#         feature_dim = None
#         score_dtype = None

#         with torch.no_grad():
#             for batch in dataloader:
#                 x = _extract_x(batch).to(device)
#                 _, latents, _ = model(x)

#                 if min_vals is None:
#                     num_neurons = latents.shape[1]
#                     feature_dim = x.shape[1]
#                     score_dtype = latents.dtype
#                     min_vals = latents.min(dim=0, keepdim=True).values.detach()
#                     max_vals = latents.max(dim=0, keepdim=True).values.detach()
#                 else:
#                     batch_min = latents.min(dim=0, keepdim=True).values.detach()
#                     batch_max = latents.max(dim=0, keepdim=True).values.detach()
#                     min_vals = torch.minimum(min_vals, batch_min)
#                     max_vals = torch.maximum(max_vals, batch_max)

#             if min_vals is None:
#                 raise ValueError("Dataloader appears to be empty.")

#         # --------------------------------------------------
#         # Pass 2: global accumulators u, v, w
#         # --------------------------------------------------
#         u = torch.zeros(num_neurons, device=device, dtype=torch.float32)
#         v = torch.zeros(num_neurons, device=device, dtype=torch.float32)
#         w = torch.zeros(feature_dim, num_neurons, device=device, dtype=torch.float32)

#         with torch.no_grad():
#             for batch in dataloader:
#                 x = _extract_x(batch).to(device)
#                 _, latents, _ = model(x)

#                 a = (latents - min_vals) / (max_vals - min_vals + eps)
#                 p = F.normalize(x, p=2, dim=1, eps=eps)

#                 a32 = a.float()
#                 p32 = p.float()

#                 u += a32.sum(dim=0)
#                 v += (a32 * a32).sum(dim=0)
#                 w += p32.T @ a32

#         q = (w * w).sum(dim=0)                      # (M,)
#         num = 0.5 * (q - v)                        # (M,)
#         den = 0.5 * (u * u - v)                    # (M,)
#         active_mask = den > eps                    # (M,)
#         num_active = int(active_mask.sum().item())

#         if num_active > 0:
#             mono_scores = torch.where(
#                 active_mask,
#                 num / (den + eps),
#                 torch.zeros_like(den),
#             )
#             mono_mean = mono_scores[active_mask].mean()
#         else:
#             mono_scores = torch.zeros_like(den)
#             mono_mean = torch.zeros((), device=device, dtype=torch.float32)

#         loss = 1.0 - mono_mean
#         loss_out = loss.to(dtype=score_dtype)

#         # Save only cheap tensors needed by backward
#         ctx.save_for_backward(
#             min_vals.detach(),
#             max_vals.detach(),
#             u.detach(),
#             v.detach(),
#             w.detach(),
#             num.detach(),
#             den.detach(),
#             active_mask.detach(),
#         )

#         if was_training:
#             model.train()

#         return loss_out

#     @staticmethod
#     def backward(ctx, grad_output):
#         model = ctx.model
#         dataloader = ctx.dataloader
#         device = ctx.device
#         eps = ctx.eps
#         num_params = ctx.num_params

#         (
#             min_vals,
#             max_vals,
#             u,
#             v,
#             w,
#             num,
#             den,
#             active_mask,
#         ) = ctx.saved_tensors

#         params = [p for p in model.parameters()]
#         if len(params) != num_params:
#             raise RuntimeError(
#                 f"Parameter count changed between forward and backward: "
#                 f"{num_params} -> {len(params)}"
#             )

#         active_count = int(active_mask.sum().item())
#         if active_count == 0:
#             grads = [torch.zeros_like(p) if p.requires_grad else None for p in params]
#             # Return None for non-tensor inputs: model, dataloader, device, eps, verbose
#             return (None, None, None, None, None, *grads)

#         # dL/ds_m for active neurons only, since loss = 1 - mean(s_active)
#         coeff = torch.zeros_like(den, dtype=torch.float32, device=device)
#         coeff[active_mask] = -1.0 / float(active_count)

#         denom_safe = den + eps                      # (M,)
#         range_safe = (max_vals - min_vals + eps)   # (1, M)

#         # accumulate parameter grads batch-by-batch
#         param_grads = _zeros_like_params(params)

#         was_training = model.training
#         model.eval()

#         # We are inside custom backward. Re-enable grad for recomputation.
#         with torch.enable_grad():
#             for batch in dataloader:
#                 x = _extract_x(batch).to(device)
#                 x = x.detach()  # x is treated as constant dataset input here

#                 # recompute latents with graph
#                 _, latents, _ = model(x)

#                 # normalized activations and normalized embeddings
#                 a = (latents - min_vals) / range_safe                # (B, M)
#                 p = F.normalize(x, p=2, dim=1, eps=eps)              # (B, D)

#                 # Compute dnum/da and dden/da for each sample in batch
#                 # dnum/da = <p_i, w_m> - a_im
#                 # dden/da = u_m - a_im
#                 # Shapes:
#                 #   p @ w -> (B, M)
#                 pw = p.float() @ w                                   # (B, M)
#                 a32 = a.float()

#                 dnum_da = pw - a32                                   # (B, M)
#                 dden_da = u.unsqueeze(0) - a32                       # (B, M)

#                 # ds/da = [dnum*(den+eps) - num*dden] / (den+eps)^2
#                 ds_da = (
#                     dnum_da * denom_safe.unsqueeze(0)
#                     - num.unsqueeze(0) * dden_da
#                 ) / (denom_safe.unsqueeze(0) ** 2)

#                 # inactive neurons contribute zero
#                 ds_da = ds_da * active_mask.unsqueeze(0).float()

#                 # dL/da
#                 dloss_da = ds_da * coeff.unsqueeze(0)                # (B, M)

#                 # chain through a = (latents - min) / range
#                 dloss_dlatents = dloss_da / range_safe.float()       # (B, M)
#                 dloss_dlatents = dloss_dlatents.to(dtype=latents.dtype)

#                 # include upstream scalar grad
#                 dloss_dlatents = dloss_dlatents * grad_output.to(latents.dtype)

#                 batch_param_grads = torch.autograd.grad(
#                     outputs=latents,
#                     inputs=params,
#                     grad_outputs=dloss_dlatents,
#                     retain_graph=False,
#                     create_graph=False,
#                     allow_unused=True,
#                 )

#                 for i, g in enumerate(batch_param_grads):
#                     if g is None:
#                         continue
#                     if param_grads[i] is None:
#                         param_grads[i] = g.detach()
#                     else:
#                         param_grads[i] = param_grads[i] + g.detach()

#         if was_training:
#             model.train()

#         return (None, None, None, None, None, *param_grads)