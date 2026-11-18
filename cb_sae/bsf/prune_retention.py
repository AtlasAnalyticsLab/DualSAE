"""Per-sample top-k prune fidelity (CB-SAE Step 3 bookkeeping).

Only `retention_stats` is kept: bsf/bsf_prune.py uses it to measure how much
of a full model's per-sample selection survives a prune. Copied verbatim
from the code that produced the reported runs.
"""
from __future__ import annotations

import torch


def retention_stats(pre_z: torch.Tensor, retained: torch.Tensor, k: int,
                    ) -> dict[str, float]:
    """Per-sample-top-k prune fidelity from a batch of pre-gate codes.

    pre_z:    (N, G) or (N, G, 1) NON-NEGATIVE pre-gate activations.
    retained: (R,) ascending long tensor of kept neuron ids.
    k:        the active-latent budget of the deployment gate.

    Simulating the prune from pre_z is exact and far cheaper than materializing
    a pruned model: pruning only deletes columns, and a top-k gate over the
    retained columns of pre_z IS the pruned model's gate (the encoder rows and
    decoder atoms are copied verbatim, so pre_z restricted to `retained` is
    literally the pruned model's pre-code).

    Returns
      retention   mean_t |top_k(full)_t & retained| / k
                  -- the share of the full model's selection that even
                  survives the prune. 1 - retention is the fraction of slots
                  the pruned model must refill.
      preservation
                  |S_full & retained & S_pruned| / |S_full & retained|
                  -- of the selections that survived, how many the pruned
                  model still makes. Ties aside this is 1.0 by construction
                  (a top-k member of the full set is still top-k in a subset
                  containing it), so it is a CONSISTENCY CHECK: a value below
                  1 means the simulation or the ascending-keep contract broke.
      promoted_per_token
                  k * (1 - retention) -- neurons the pruned model recruits
                  that the full model did not select. This is the number that
                  actually dilutes I_m / S_m.
      dead_in_batch
                  fraction of retained neurons that never fire in this batch;
                  context for reading the others.
    """
    if pre_z.dim() == 3:
        pre_z = pre_z.squeeze(-1)
    N, G = pre_z.shape
    k_eff = min(k, G)
    # Full-model selection: top-k over ALL G columns.
    full_idx = pre_z.topk(k_eff, dim=1).indices                     # (N, k)
    sel_full = torch.zeros(N, G, dtype=torch.bool, device=pre_z.device)
    sel_full.scatter_(1, full_idx, True)

    keep_mask = torch.zeros(G, dtype=torch.bool, device=pre_z.device)
    keep_mask[retained.to(pre_z.device)] = True

    # Survivors: selected by the full model AND kept by the prune.
    surv = sel_full & keep_mask                                     # (N, G)
    n_surv = surv.sum(1).float()                                    # (N,)

    # Pruned-model selection: top-k over the retained columns only.
    sub = pre_z[:, keep_mask]                                       # (N, R)
    R = sub.shape[1]
    k_sub = min(k_eff, R)
    sub_idx = sub.topk(k_sub, dim=1).indices                        # (N, k)
    sel_sub = torch.zeros(N, R, dtype=torch.bool, device=pre_z.device)
    sel_sub.scatter_(1, sub_idx, True)
    # Lift back to full-width ids so the two selections are comparable.
    sel_pruned = torch.zeros(N, G, dtype=torch.bool, device=pre_z.device)
    sel_pruned[:, keep_mask] = sel_sub

    kept_and_kept = (surv & sel_pruned).sum(1).float()
    retention = float((n_surv / k_eff).mean())
    # Guard the denominator: a token whose entire top-k was discarded has
    # nothing to preserve and must not contribute a 0/0.
    nz = n_surv > 0
    preservation = float((kept_and_kept[nz] / n_surv[nz]).mean()) if nz.any() else float("nan")
    return {
        "retention": retention,
        "preservation": preservation,
        "promoted_per_token": float(k_eff * (1.0 - retention)),
        "dead_in_batch": float((~(pre_z[:, keep_mask] > 0).any(0)).float().mean()),
        "k": int(k_eff),
        "G": int(G),
        "R": int(R),
        "n_rows": int(N),
    }


