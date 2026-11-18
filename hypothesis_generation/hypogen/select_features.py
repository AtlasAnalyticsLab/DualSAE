"""Stage 2 — per-class one-vs-rest feature selection on the train activations.

For each class c we pick the K latents most predictive of "is this image class c?", for each method
in config.SELECTION_METHODS and each K in config.K_GRID. Definitions match HypotheSAEs.select_neurons:
  correlation       Pearson corr(latent, one-vs-rest target); top-K by |corr|.
  separation_score  E[y | latent in top-N activations] - E[y | latent == 0]; top-K by |sep|.
  lasso             L1-logistic coef on a balanced subsample (all pos + sampled neg); top-K by |coef|.
correlation/separation are vectorized over ALL 1000 classes at once (the library's per-latent loops are
far too slow here); lasso is one fit per class on a balanced subsample (no alpha search) to stay tractable.

Outputs (under config.SELECT_DIR/<cell>/), resumable:
  <method>_K<k>.json   {class_to_latents, class_to_scores, method, K, dict_size}
  unique_latents.json  union over classes of the INTERP_UNION config's selections, ordered by aggregate
                       predictiveness (defines + bounds the set of latents Stage 3 interprets)

Run: python -m hypogen.select_features --encoder clip --activation topk --variant dual --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from hypogen import config


def _select_dir(cell: config.Cell) -> str:
    return os.path.join(config.SELECT_DIR, cell.encoder, cell.activation, cell.variant, f"seed{cell.seed}")


def _atomic_write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def correlation_matrix(A: np.ndarray, labels: np.ndarray, C: int) -> np.ndarray:
    """corr[c, j] = Pearson correlation between latent j and the one-vs-rest target for class c. (C, M).
    Single-label labels (N,) and multi-hot labels (N, C) both supported (the per-class binary target y_c is
    'label == c' or 'multi-hot column c'); the Pearson formula is identical, only P(y_c=1) and the grouped
    positive sums G[c] = Σ_{y_c=1} a_j differ."""
    n, M = A.shape
    mean = A.mean(0)
    std = A.std(0)
    if labels.ndim == 2:                  # multi-hot (N, C): P(y_c) = column mean, G = Y^T A (vectorized)
        Y = (labels > 0).astype(np.float64)
        p = Y.mean(0)                                     # (C,) P(y_c = 1)
        G = Y.T @ A                                       # (C, M) positive sums Σ_{y_c=1} a_j
    else:                                  # single-label (N,): bincount path (avoids forming an (N,C) one-hot)
        counts = np.bincount(labels, minlength=C).astype(np.float64)
        p = counts / n                       # P(y_c = 1)
        # Grouped positive sums G[c] = sum over rows with label c (so E[a_j y_c] = G[c,j]/n).
        G = np.zeros((C, M), dtype=np.float64)
        for c in range(C):
            rows = A[labels == c]
            if rows.size:
                G[c] = rows.sum(0)
    std_y = np.sqrt(np.clip(p * (1 - p), 1e-12, None))
    cov = G / n - np.outer(p, mean)                       # Cov(a_j, y_c)
    denom = np.outer(std_y, std)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-12)
    corr[:, std <= 1e-12] = 0.0                           # dead latents -> 0
    return corr


def separation_matrix(A: np.ndarray, labels: np.ndarray, C: int, n_top: int) -> np.ndarray:
    """sep[c, j] = E[y_c | latent j in top-N activations] - E[y_c | latent j == 0]. (C, M).
    Multi-label aware: with multi-hot labels (N, C) the class fractions over the top / zero rows are column
    means of the multi-hot matrix instead of bincounts of a single-label vector."""
    n, M = A.shape
    sep = np.zeros((C, M), dtype=np.float64)
    n_top = min(n_top, n)
    multilabel = labels.ndim == 2
    Yb = (labels > 0).astype(np.float64) if multilabel else None   # (N, C) for column-mean of class fractions
    for j in range(M):
        col = A[:, j]
        if not np.any(col > 0):
            continue                                       # dead latent -> 0
        top_idx = np.argpartition(-col, n_top - 1)[:n_top]
        zero_mask = col == 0
        n_zero = int(zero_mask.sum())
        if n_zero == 0:
            continue
        if multilabel:                                     # class fractions = multi-hot column means
            top_mean = Yb[top_idx].mean(0)
            zero_mean = Yb[zero_mask].mean(0)
        else:
            top_mean = np.bincount(labels[top_idx], minlength=C).astype(np.float64) / n_top
            zero_mean = np.bincount(labels[zero_mask], minlength=C).astype(np.float64) / n_zero
        sep[:, j] = top_mean - zero_mean
    return sep


def lasso_select(A: np.ndarray, labels: np.ndarray, C: int, ks: list, rng, classes=None) -> dict:
    """Per class, L1-logistic on a balanced subsample; return {K: {c: ([latents], [coefs])}}.
    `classes` (optional) restricts the fit to a subset/iterable of class ids (default: all range(C))."""
    out = {k: {} for k in ks}
    kmax = max(ks)
    multilabel = labels.ndim == 2
    for c in (range(C) if classes is None else classes):
        # one-vs-rest positive mask: multi-hot column c, or single-label == c.
        pos_mask = (labels[:, c] > 0) if multilabel else (labels == c)
        pos = np.where(pos_mask)[0]
        neg_all = np.where(~pos_mask)[0]
        if len(pos) == 0:
            for k in ks:
                out[k][c] = ([], [])
            continue
        n_neg = min(config.LASSO_NEG_PER_CLASS, len(neg_all))
        neg = rng.choice(neg_all, size=n_neg, replace=False)
        idx = np.concatenate([pos, neg])
        Xc = StandardScaler().fit_transform(A[idx].astype(np.float64))
        yc = pos_mask[idx].astype(int)
        # L1-logistic (matches HypotheSAEs select_neurons_lasso). sklearn 1.8 deprecated `penalty=`
        # (FutureWarning; removed in 1.10 -> migrate to l1_ratio=1) but liblinear+penalty='l1' still
        # produces correctly sparse coefficients here; silence the noisy per-fit warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LogisticRegression(penalty="l1", solver="liblinear", C=config.LASSO_C, max_iter=1000)
            model.fit(Xc, yc)
        coef = model.coef_.ravel()
        order = np.argsort(-np.abs(coef))[:kmax]
        for k in ks:
            sel = order[:k]
            out[k][c] = (sel.tolist(), coef[sel].tolist())
    return out


def _topk_from_matrix(mat: np.ndarray, ks: list) -> dict:
    """For a (C, M) score matrix, return {K: {c: ([latents], [scores])}} by |score| descending."""
    C = mat.shape[0]
    out = {k: {} for k in ks}
    kmax = max(ks)
    order_all = np.argsort(-np.abs(mat), axis=1)[:, :kmax]
    for c in range(C):
        for k in ks:
            sel = order_all[c, :k]
            out[k][c] = (sel.tolist(), mat[c, sel].tolist())
    return out


def run(cell: config.Cell) -> None:
    out_dir = _select_dir(cell)
    expected = [os.path.join(out_dir, f"{m}_K{k}.json") for m in config.SELECTION_METHODS for k in config.K_GRID]
    expected.append(os.path.join(out_dir, "unique_latents.json"))
    if all(os.path.isfile(p) for p in expected):
        print(f"[{cell.tag}] selection already complete.")
        return

    acts_path = config.acts_path(cell, "train")
    labels = config.load_labels("train")   # (N,) int single-label OR (N, C) float multi-hot (OpenImages)
    A = np.asarray(np.load(acts_path, mmap_mode="r"), dtype=np.float32)  # materialize (~5GB) for column ops
    assert A.shape[0] == labels.shape[0], f"row mismatch acts {A.shape[0]} vs labels {labels.shape[0]}"
    C = config.NUM_CLASSES
    M = A.shape[1]
    print(f"[{cell.tag}] train acts {A.shape}; selecting for {C} classes, K={config.K_GRID}")

    selections = {}   # method -> {K: {c: (latents, scores)}}
    if "correlation" in config.SELECTION_METHODS:
        selections["correlation"] = _topk_from_matrix(correlation_matrix(A, labels, C), config.K_GRID)
        print(f"[{cell.tag}] correlation done")
    if "separation_score" in config.SELECTION_METHODS:
        selections["separation_score"] = _topk_from_matrix(
            separation_matrix(A, labels, C, config.SEPARATION_N_TOP), config.K_GRID)
        print(f"[{cell.tag}] separation_score done")
    if "lasso" in config.SELECTION_METHODS:
        rng = np.random.default_rng(1000 + cell.seed)
        selections["lasso"] = lasso_select(A, labels, C, config.K_GRID, rng)
        print(f"[{cell.tag}] lasso done")

    for method, by_k in selections.items():
        for k, by_c in by_k.items():
            _atomic_write_json(
                {"method": method, "K": k, "dict_size": M,
                 "class_to_latents": {str(c): v[0] for c, v in by_c.items()},
                 "class_to_scores": {str(c): v[1] for c, v in by_c.items()}},
                os.path.join(out_dir, f"{method}_K{k}.json"))

    # unique_latents: union over classes of the INTERP_UNION config, ordered by aggregate predictiveness
    # (max |score| across the classes that selected the latent) so an optional cap keeps the best ones.
    iu = selections[config.INTERP_UNION_METHOD][config.INTERP_UNION_K]
    # selmethod mode and OpenImages: faithful scores only the FAITHFUL_CLASSES subset, and latents selected
    # ONLY by non-scored classes are never used by faithful -> restricting the interpret union to those
    # classes is LOSSLESS for the faithful metric and bounds the VLM cost. (ImageNet main mode interprets
    # the union over all 1000 classes.)
    if config.MODE == "selmethod" or config.DATASET == "openimages":
        fcls = set(config.faithful_classes())
        iu = {c: v for c, v in iu.items() if c in fcls}
    agg = {}
    for c, (lats, scores) in iu.items():
        for lat, sc in zip(lats, scores):
            agg[lat] = max(agg.get(lat, 0.0), abs(sc))
    ordered = sorted(agg, key=lambda l: -agg[l])
    _atomic_write_json(
        {"interp_method": config.INTERP_UNION_METHOD, "interp_K": config.INTERP_UNION_K,
         "dict_size": M, "n_unique": len(ordered), "latents": ordered},
        os.path.join(out_dir, "unique_latents.json"))
    print(f"[{cell.tag}] unique latents to interpret: {len(ordered)}")


def main():
    ap = argparse.ArgumentParser(description="Stage 2: per-class feature selection.")
    ap.add_argument("--encoder", required=True, choices=config.ENCODERS)
    ap.add_argument("--activation", required=True, choices=[p.name for p in config.ACTIVATION_PAIRS])
    ap.add_argument("--variant", required=True, choices=config.ALLOWED_VARIANTS)
    ap.add_argument("--seed", required=True, type=int, choices=config.SEEDS)
    args = ap.parse_args()
    run(config.Cell(args.encoder, args.activation, args.variant, args.seed))


if __name__ == "__main__":
    main()
