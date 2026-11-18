#!/usr/bin/env python3
"""Collect the five metrics of every run of the main comparison into one long CSV.

Rows of the output (one per dataset x encoder x method x metric, seeds averaged):

    dataset, encoder, method, metric, split, source, checkpoint, mean, sd, n_seeds, seeds,
    l0_mean, dead_frac

``mean`` / ``sd`` are the mean and sample standard deviation over the seeds found (seeds are the
only averaged dimension); ``l0_mean`` / ``dead_frac`` come from the offline evaluator when its
result exists (they are diagnostics, shown next to emergent-sparsity rows).

Which runs
----------
* the cells of ``configs/main_cells.json`` (methods topk, dual_topk, batch_topk, dual_batch_topk,
  jumprelu, dual_jump_relu, matryoshka, dual_matryoshka, gated, dual_gated), found under
  ``--ckpt_dir`` (in-training results) / in ``--eval_dir`` (offline results);
* the parameter-matched baselines (``pmatch_*`` runs under ``--pmatch_dir``; methods
  ``topk_pmatch`` / ``batch_topk_pmatch`` / ``jumprelu_pmatch``);
* the CB-SAE runs (``model_type`` cb_sae / cb_dual_sae), found in ``--eval_dir`` by the
  metadata the evaluator records. The arm with ``retain == --cb_retain`` and
  ``k_cb == --cb_k`` is reported as ``cb_sae`` / ``cb_dual_sae``; other arms get a suffix
  (e.g. ``cb_sae_R2048_k64``).

Where each number comes from (``--source``)
-------------------------------------------
* ``as_reported`` (default): the TopK / BatchTopK / JumpReLU pairs and the parameter-matched
  arms from the trainers' own best-checkpoint evaluation (``best_metrics.json``), the
  Matryoshka / Gated pairs and the CB rows from the offline evaluator (branch_metrics.py);
* ``offline``: every row from the offline evaluator (CB rows are offline-only in any case);
* ``in_training``: every trained run from best_metrics.json.

Splits (main_analysis_common.SPLIT_BY_METRIC): ImageNet-1K val for every column; OpenImagesV7
test for R^2 / MonoScore / KNN / LP and val for binary purity (the baseline trainers do not
compute a test-split purity, so val is the split on which purity exists for every model).

Usage
-----
    python analysis/main/collect_main.py --ckpt_dir ./checkpoints/main \\
        --pmatch_dir ./checkpoints/param_matched --eval_dir ./outputs/main/branch_metrics \\
        --out ./outputs/main/main_long.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_analysis_common as A  # noqa: E402

VISION = Path(__file__).resolve().parents[2]
CELLS = VISION / "configs" / "main_cells.json"
_PMATCH = re.compile(r"^pmatch_(imagenet|open_images)_(clip|vit|sigclip)_(topk|batch_topk|jumprelu)_n(\d+)_seed(\d+)$")


def _runs_from_cells() -> list:
    return [dict(dataset=c["dataset"], encoder=c["encoder"], method=c["method"], seed=int(c["seed"]),
                 exp_name=c["exp_name"]) for c in json.loads(CELLS.read_text())["cells"]]


def _runs_from_pmatch(pmatch_dir: Path | None, eval_dir: Path) -> list:
    names = set()
    if pmatch_dir and pmatch_dir.is_dir():
        names |= {p.name for p in pmatch_dir.iterdir() if _PMATCH.match(p.name)}
    names |= {p.name.split("__")[0] for p in eval_dir.glob("pmatch_*__*.json")} if eval_dir.is_dir() else set()
    out = []
    for n in sorted(names):
        m = _PMATCH.match(n)
        if m:
            out.append(dict(dataset=m.group(1), encoder=m.group(2), method=f"{m.group(3)}_pmatch",
                            seed=int(m.group(5)), exp_name=n))
    return out


def _runs_from_cb(eval_dir: Path, retain: int, k_cb: int) -> list:
    """CB runs, identified from the metadata branch_metrics records in each result."""
    out, seen = [], set()
    if not eval_dir.is_dir():
        return out
    for p in sorted(eval_dir.glob("*__*.json")):
        try:
            meta = json.loads(p.read_text()).get("meta", {})
        except Exception:
            continue
        mt = meta.get("model_type")
        if mt not in ("cb_sae", "cb_dual_sae"):
            continue
        name = Path(meta.get("run_dir", p.name.split("__")[0])).name
        if name in seen:
            continue
        seen.add(name)
        r, k = meta.get("n_retained"), meta.get("k_cb")
        suffix = "" if (r == retain and k == k_cb) else f"_R{r}_k{k}"
        out.append(dict(dataset=meta.get("dataset_name"), encoder=meta.get("extractor_name"),
                        method=f"{mt}{suffix}", seed=int(meta.get("seed")), exp_name=name))
    return out


def _value(run: dict, metric: str, split: str, source: str, dirs: dict, checkpoint: str):
    """(value or None, offline diagnostics dict or None) of one run/metric."""
    offline = None
    off_path = A.result_path(dirs["eval"], run["exp_name"], split, checkpoint)
    if off_path.exists():
        offline = A.read_offline(off_path)
    if source == "offline":
        return (offline[metric] if offline else None), offline
    base = dirs["pmatch"] if run["method"].endswith("_pmatch") else dirs["ckpt"]
    bm = Path(base) / run["exp_name"] / "best_metrics.json" if base else None
    if bm is None or not bm.exists():
        return None, offline
    rec = json.loads(bm.read_text())
    return rec.get(A.in_training_key(metric, split)), offline


def collect(runs: list, policy: str, dirs: dict, checkpoint: str = "best") -> list:
    vals = defaultdict(dict)          # (ds, enc, method, metric) -> {seed: value}
    diag = defaultdict(lambda: defaultdict(dict))
    srcs = {}
    for run in runs:
        source = A.source_for(run["method"], policy) if run["method"] in A.KNOWN_METHODS or \
            run["method"].startswith(("cb_sae", "cb_dual_sae")) else policy
        if run["method"].startswith(("cb_sae", "cb_dual_sae")):
            source = "offline"
        for metric in A.METRICS:
            split = A.SPLIT_BY_METRIC[run["dataset"]][metric]
            v, off = _value(run, metric, split, source, dirs, checkpoint)
            key = (run["dataset"], run["encoder"], run["method"], metric)
            srcs[key] = (split, source)
            if v is not None:
                vals[key][run["seed"]] = float(v)
            if off is not None:
                diag[key[:3]]["l0"][run["seed"]] = off["_l0"]
                diag[key[:3]]["dead"][run["seed"]] = off["_dead"]
    rows = []
    for key in sorted(srcs):
        seeds = vals.get(key, {})
        m, sd, n = A.mean_sd(seeds.values())
        d = diag.get(key[:3], {})
        l0 = A.mean_sd(d.get("l0", {}).values())[0] if d else None
        dead = A.mean_sd(d.get("dead", {}).values())[0] if d else None
        split, source = srcs[key]
        rows.append(dict(dataset=key[0], encoder=key[1], method=key[2], metric=key[3], split=split,
                         source=source, checkpoint=checkpoint,
                         mean="" if m is None else f"{m:.6f}", sd="" if sd is None else f"{sd:.6f}",
                         n_seeds=n, seeds=" ".join(str(s) for s in sorted(seeds)),
                         l0_mean="" if l0 is None else f"{l0:.2f}",
                         dead_frac="" if dead is None else f"{dead:.4f}"))
    return rows


def merge_cb_long(rows: list, cb_path: Path) -> list:
    """Replace the CB rows by those of the CB pipeline's long CSV (same column meaning:
    dataset, encoder, method, metric, split, mean, sd, n_seeds, l0_mean, dead_frac)."""
    with open(cb_path, newline="", encoding="utf-8") as fh:
        cb = [r for r in csv.DictReader(fh) if r["method"].startswith(("cb_sae", "cb_dual_sae"))]
    keep = [r for r in rows if not r["method"].startswith(("cb_sae", "cb_dual_sae"))]
    for r in cb:
        keep.append(dict(dataset=r["dataset"], encoder=r["encoder"], method=r["method"],
                         metric=r["metric"], split=r.get("split", ""), source="offline",
                         checkpoint=r.get("checkpoint", "best"), mean=r.get("mean", ""),
                         sd=r.get("sd", ""), n_seeds=r.get("n_seeds", ""), seeds=r.get("seeds", ""),
                         l0_mean=r.get("l0_mean", ""), dead_frac=r.get("dead_frac", "")))
    print(f"merged {len(cb)} CB rows from {cb_path}")
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_dir", default="./checkpoints/main")
    ap.add_argument("--pmatch_dir", default="./checkpoints/param_matched")
    ap.add_argument("--eval_dir", default="./outputs/main/branch_metrics")
    ap.add_argument("--source", default="as_reported", choices=A.SOURCES)
    ap.add_argument("--checkpoint", default="best", choices=["best", "final"],
                    help="offline results of which checkpoint to read (default best)")
    ap.add_argument("--cb_retain", type=int, default=1024)
    ap.add_argument("--cb_k", type=int, default=5)
    ap.add_argument("--cb_long", default="./outputs/cb_sae/cb_long.csv",
                    help="long CSV of the CB pipeline (cb_sae/); when it exists, its CB / CB-DUAL "
                         "rows replace the ones collected from --eval_dir")
    ap.add_argument("--out", default="./outputs/main/main_long.csv")
    args = ap.parse_args()
    dirs = {"ckpt": Path(args.ckpt_dir), "pmatch": Path(args.pmatch_dir), "eval": Path(args.eval_dir)}
    runs = (_runs_from_cells() + _runs_from_pmatch(dirs["pmatch"], dirs["eval"])
            + _runs_from_cb(dirs["eval"], args.cb_retain, args.cb_k))
    rows = collect(runs, args.source, dirs, args.checkpoint)
    cb_path = Path(args.cb_long)
    if args.cb_long and cb_path.exists():
        rows = merge_cb_long(rows, cb_path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        w.writeheader()
        w.writerows(rows)
    n_filled = sum(1 for r in rows if r["n_seeds"])
    print(f"{len(rows)} rows ({n_filled} with data) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
