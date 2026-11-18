#!/usr/bin/env python3
"""Correctness tests for the sparse KNN/LP path in ``knn_lp_sparse.py``.

The whole point of the sparse path is that it must produce the SAME numbers a
naive dense implementation would, just without the 42 TB dense bank.  These
tests pin that equivalence down on synthetic data where the dense reference is
cheap to compute:

* ``gather_dense`` reconstructs the exact dense matrix the CSR encodes,
* ``knn_metric`` on a SparseBank == ``knn_metric`` on the equivalent DenseBank,
  for single-label and multi-label targets, and across a chunk boundary (the
  top-k merge across bank chunks is the easiest thing to get wrong),
* ``MicroF1`` matches a brute-force TP/FP/FN count.

Run:  cd sae && python tests/test_knn_lp_sparse.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knn_lp_sparse import DenseBank, MicroF1, SparseBank, knn_metric, lp_metric  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_sparse_matrix(n: int, d: int, k: int, seed: int) -> torch.Tensor:
    """Dense (n, d) matrix with exactly k positive entries per row, L2-normalized."""
    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(n, d)
    for i in range(n):
        cols = torch.randperm(d, generator=g)[:k]
        z[i, cols] = torch.rand(k, generator=g) + 0.1  # strictly positive
    return F.normalize(z, p=2, dim=1)


def to_sparse_bank(dense: torch.Tensor, labels: torch.Tensor) -> SparseBank:
    mask = dense > 0
    counts = mask.sum(dim=1)
    indptr = torch.zeros(dense.shape[0] + 1, dtype=torch.int64)
    torch.cumsum(counts.to(torch.int64), 0, out=indptr[1:])
    return SparseBank(
        indptr=indptr,
        indices=mask.nonzero(as_tuple=True)[1].to(torch.int32),
        values=dense[mask].clone(),
        n_rows=dense.shape[0],
        n_cols=dense.shape[1],
        labels=labels,
    )


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        raise AssertionError(name)


def test_gather_dense() -> None:
    dense = make_sparse_matrix(37, 64, 5, seed=1)
    bank = to_sparse_bank(dense, torch.zeros(37, dtype=torch.long)).to(DEVICE)
    rows = torch.tensor([0, 5, 36, 12, 12], device=DEVICE)
    got = bank.gather_dense(rows).cpu()
    check("gather_dense reproduces the dense rows",
          torch.allclose(got, dense[rows.cpu()], atol=1e-6),
          f"max|diff|={float((got - dense[rows.cpu()]).abs().max()):.2e}")
    # A row with no active entries must come back as all zeros, not garbage.
    dense2 = dense.clone()
    dense2[3] = 0.0
    bank2 = to_sparse_bank(dense2, torch.zeros(37, dtype=torch.long)).to(DEVICE)
    check("gather_dense handles empty rows",
          bool((bank2.gather_dense(torch.tensor([3], device=DEVICE)) == 0).all()))


def test_knn_sparse_equals_dense_singlelabel() -> None:
    n_train, n_eval, d, k = 500, 120, 128, 6
    train = make_sparse_matrix(n_train, d, k, seed=2)
    ev = make_sparse_matrix(n_eval, d, k, seed=3)
    g = torch.Generator().manual_seed(4)
    ytr = torch.randint(0, 7, (n_train,), generator=g)
    yev = torch.randint(0, 7, (n_eval,), generator=g)

    dense_res = knn_metric(DenseBank(train, ytr).to(DEVICE), DenseBank(ev, yev).to(DEVICE),
                           DEVICE, k=20, query_batch=32, bank_chunk=97)
    sparse_res = knn_metric(to_sparse_bank(train, ytr).to(DEVICE), to_sparse_bank(ev, yev).to(DEVICE),
                            DEVICE, k=20, query_batch=32, bank_chunk=97)
    check("KNN single-label: sparse == dense",
          abs(dense_res["metric"] - sparse_res["metric"]) < 1e-9,
          f"dense={dense_res['metric']:.6f} sparse={sparse_res['metric']:.6f}")
    check("KNN single-label reports accuracy", dense_res["metric_type"] == "accuracy")

    # bank_chunk=97 is deliberately not a divisor of 500, so the cross-chunk
    # top-k merge is exercised; a single chunk must give the identical answer.
    one_chunk = knn_metric(to_sparse_bank(train, ytr).to(DEVICE), to_sparse_bank(ev, yev).to(DEVICE),
                           DEVICE, k=20, query_batch=1000, bank_chunk=10_000)
    check("KNN chunked == unchunked",
          abs(one_chunk["metric"] - sparse_res["metric"]) < 1e-9,
          f"chunked={sparse_res['metric']:.6f} single={one_chunk['metric']:.6f}")


def test_knn_sparse_equals_dense_multilabel() -> None:
    n_train, n_eval, d, k, c = 400, 90, 96, 5, 11
    train = make_sparse_matrix(n_train, d, k, seed=5)
    ev = make_sparse_matrix(n_eval, d, k, seed=6)
    g = torch.Generator().manual_seed(7)
    ytr = (torch.rand(n_train, c, generator=g) < 0.2).float()
    yev = (torch.rand(n_eval, c, generator=g) < 0.2).float()

    dense_res = knn_metric(DenseBank(train, ytr).to(DEVICE), DenseBank(ev, yev).to(DEVICE),
                           DEVICE, k=20, query_batch=16, bank_chunk=73)
    sparse_res = knn_metric(to_sparse_bank(train, ytr).to(DEVICE), to_sparse_bank(ev, yev).to(DEVICE),
                            DEVICE, k=20, query_batch=16, bank_chunk=73)
    check("KNN multi-label: sparse == dense",
          abs(dense_res["metric"] - sparse_res["metric"]) < 1e-9,
          f"dense={dense_res['metric']:.6f} sparse={sparse_res['metric']:.6f}")
    check("KNN multi-label reports micro_f1", dense_res["metric_type"] == "micro_f1")
    check("global and batchwise micro-F1 both present",
          "micro_f1_batchwise" in sparse_res and sparse_res["micro_f1_batchwise"] > 0)


def test_lp_sparse_equals_dense() -> None:
    """The probe sees identical inputs, so with a pinned seed it must converge identically."""
    n_train, n_eval, d, k = 600, 150, 64, 4
    train = make_sparse_matrix(n_train, d, k, seed=8)
    ev = make_sparse_matrix(n_eval, d, k, seed=9)
    g = torch.Generator().manual_seed(10)
    ytr = torch.randint(0, 5, (n_train,), generator=g)
    yev = torch.randint(0, 5, (n_eval,), generator=g)

    dense_res = lp_metric(DenseBank(train, ytr).to(DEVICE), DenseBank(ev, yev).to(DEVICE),
                          DEVICE, epochs=2, batch_size=64)
    sparse_res = lp_metric(to_sparse_bank(train, ytr).to(DEVICE), to_sparse_bank(ev, yev).to(DEVICE),
                           DEVICE, epochs=2, batch_size=64)
    check("LP: sparse == dense",
          abs(dense_res["metric"] - sparse_res["metric"]) < 1e-9,
          f"dense={dense_res['metric']:.6f} sparse={sparse_res['metric']:.6f}")


def test_micro_f1() -> None:
    g = torch.Generator().manual_seed(11)
    pred = torch.rand(200, 13, generator=g) < 0.3
    target = torch.rand(200, 13, generator=g) < 0.25
    f1 = MicroF1()
    for s in range(0, 200, 37):  # ragged batches on purpose
        f1.update(pred[s:s + 37], target[s:s + 37])
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    fn = int((~pred & target).sum())
    ref = 2.0 * tp / (2 * tp + fp + fn)
    check("MicroF1 global matches brute force", abs(f1.value - ref) < 1e-12,
          f"got={f1.value:.8f} ref={ref:.8f}")
    check("MicroF1 batchwise differs from global (documented estimator gap)",
          abs(f1.batchwise - ref) > 0, f"batchwise={f1.batchwise:.6f} global={ref:.6f}")


if __name__ == "__main__":
    print(f"device: {DEVICE}\n")
    test_gather_dense()
    test_knn_sparse_equals_dense_singlelabel()
    test_knn_sparse_equals_dense_multilabel()
    test_lp_sparse_equals_dense()
    test_micro_f1()
    print("\nall tests passed")
