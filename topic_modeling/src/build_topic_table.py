#!/usr/bin/env python
"""Build the topic-modeling results tables from the SAE-TM evaluation outputs.

For every dataset, encoder, activation pair, DualSAE configuration (run x warm-up, all of them, as
listed in configs/runs.json), topic count K and metric, the builder reads the per-seed evaluation
files of the DualSAE cell and of the activation-matched baseline cell (configs/runs.json:
baseline_run) and reports

    baseline mean/sd, DualSAE mean/sd, Delta = mean(DualSAE) - mean(baseline), #seeds

Seeds are the only dimension that is averaged. Every DualSAE configuration is its own row: the
builder never picks a configuration per cell.

Metrics (SAE-TM's three default metrics; the per-file scalar is computed as below):
  diversity  : diversity.jsonl  result.diversity                       (WMD over top-20 words)
  coherence  : coherence.jsonl  mean over topics of mean(scores)       (LLM rating, 0-100)
  intruder   : intruder.jsonl   100 * mean over topics of accuracy     (LLM intruder detection, %)

Optional checkpoint metadata (--with-epochs, default on when checkpoints exist): the epoch of each
seed's best_val_monoscore_model.pt and whether that epoch ends inside the Sparse-Branch-only warm-up
(`pre_residual_seeds`), using steps_per_epoch = floor(n_train / batch_size).

Outputs (under --out-dir, default <TM_OUT>/tables):
  topic_table_long.csv                      one row per (dataset, metric, enc, act, config, K)
  topic_table_<dataset>_<metric>.md         Delta tables, rows = activation/encoder/config, cols = K
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tm_driver as D  # noqa: E402

PAIRS = [("TopK", "TopKTrainer", "DualTopKTrainer"),
         ("BatchTopK", "BatchTopKTrainer", "DualBatchTopKTrainer"),
         ("JumpReLU", "JumpReluTrainer", "DualJumpReluTrainer")]
METRICS = ["diversity", "coherence", "intruder"]
# n_train rows of the SAE training split / batch size 2048, floor (the loaders drop the last batch).
DEFAULT_STEPS_PER_EPOCH = {"imagenet": 1281167 // 2048, "openimages": 507444 // 2048}


def metric_value(path: str, name: str):
    """Per-file scalar for one evaluation jsonl (None if unreadable/empty)."""
    try:
        with open(path) as f:
            r = json.loads(f.readline())["result"]
    except Exception:
        return None
    if name == "diversity":
        return float(r["diversity"])
    if name == "coherence":
        v = [mean(t["scores"]) for t in r.values() if isinstance(t, dict) and t.get("scores")]
        return mean(v) if v else None
    if name == "intruder":
        v = [float(x) for x in r.values()]
        return 100.0 * mean(v) if v else None
    raise ValueError(name)


def seed_values(run: str, run_cfg: dict, enc: str, trainer: str, warmup, k: int, name: str, seeds):
    out = {}
    for s in seeds:
        d = D.stage_dir(run, run_cfg, enc, "evaluation", trainer, s, warmup) / f"K{k}" / f"{name}.jsonl"
        v = metric_value(str(d), name) if d.is_file() else None
        if v is not None and not math.isnan(v):
            out[s] = v
    return out


def best_epochs(run, run_cfg, enc, trainer, warmup, seeds):
    import torch  # only needed for the optional checkpoint metadata
    eps = {}
    for s in seeds:
        p = D.stage_dir(run, run_cfg, enc, "trained_models", trainer, s, warmup) / "best_val_monoscore_model.pt"
        if p.is_file():
            try:
                ck = torch.load(str(p), map_location="cpu", mmap=True, weights_only=False)
                eps[s] = int(ck.get("epoch"))
            except Exception:
                pass
    return eps


def _ms(vals):
    v = list(vals)
    return (mean(v), pstdev(v)) if v else (None, None)


def build(reg, out_dir: Path, with_epochs: bool, steps_per_epoch: dict):
    rows = []
    seeds = reg["seeds"]
    for dataset, base_run in reg["baseline_run"].items():
        base_cfg = reg["runs"][base_run]
        dual_cfgs = [(run, cfg, w) for run, cfg in reg["runs"].items() if cfg["dataset"] == dataset
                     for w in cfg["warmups"]]
        for act, base_tr, dual_tr in PAIRS:
            for enc in reg["encoders"]:
                base_eps = best_epochs(base_run, base_cfg, enc, base_tr, None, seeds) if with_epochs else {}
                for run, cfg, w in dual_cfgs:
                    if dual_tr not in cfg["trainers"]:
                        continue
                    dual_eps = best_epochs(run, cfg, enc, dual_tr, w, seeds) if with_epochs else {}
                    spe = steps_per_epoch.get(dataset)
                    pre_res = (sum(1 for e in dual_eps.values() if spe and e * spe <= w)
                               if dual_eps else "")
                    for k in reg["topic_counts"]:
                        for name in METRICS:
                            b = seed_values(base_run, base_cfg, enc, base_tr, None, k, name, seeds)
                            d = seed_values(run, cfg, enc, dual_tr, w, k, name, seeds)
                            bm, bs = _ms(b.values())
                            dm, ds = _ms(d.values())
                            rows.append({
                                "dataset": dataset, "metric": name, "encoder": enc, "activation": act,
                                "dual_run": run, "span_coef": cfg["span_coef"], "warmup": w,
                                "warmup_ge_budget": w in cfg.get("degenerate_warmups", []),
                                "K": k,
                                "baseline_mean": bm, "baseline_sd": bs, "baseline_n": len(b),
                                "dual_mean": dm, "dual_sd": ds, "dual_n": len(d),
                                "delta": (dm - bm) if (dm is not None and bm is not None) else None,
                                "baseline_best_epochs": ";".join(str(base_eps[s]) for s in sorted(base_eps)),
                                "dual_best_epochs": ";".join(str(dual_eps[s]) for s in sorted(dual_eps)),
                                "pre_residual_seeds": pre_res,
                            })
    out_dir.mkdir(parents=True, exist_ok=True)
    long_csv = out_dir / "topic_table_long.csv"
    with open(long_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {long_csv} ({len(rows)} rows)")

    fmt = {"diversity": "{:+.2f}", "coherence": "{:+.2f}", "intruder": "{:+.2f}"}
    for dataset in reg["baseline_run"]:
        for name in METRICS:
            sel = [r for r in rows if r["dataset"] == dataset and r["metric"] == name]
            if not sel:
                continue
            keys = []
            for r in sel:
                key = (r["activation"], r["encoder"], r["dual_run"], r["warmup"])
                if key not in keys:
                    keys.append(key)
            ks = reg["topic_counts"]
            lines = [f"# {dataset} — {name}: Delta = DualSAE − activation-matched baseline "
                     f"(mean over seeds; n = seeds with outputs)", "",
                     "| activation | encoder | Dual run | span_coef | warm-up | " +
                     " | ".join(f"K={k}" for k in ks) + " | n (base/dual) |",
                     "|---|---|---|---|---|" + "---|" * len(ks) + "---|"]
            for key in keys:
                cells, ns = [], set()
                for k in ks:
                    r = next((x for x in sel if (x["activation"], x["encoder"], x["dual_run"], x["warmup"]) == key
                              and x["K"] == k), None)
                    cells.append(fmt[name].format(r["delta"]) if r and r["delta"] is not None else "n/a")
                    if r:
                        ns.add(f"{r['baseline_n']}/{r['dual_n']}")
                r0 = next(x for x in sel if (x["activation"], x["encoder"], x["dual_run"], x["warmup"]) == key)
                wtxt = f"{key[3]}" + (" (>= budget)" if r0["warmup_ge_budget"] else "")
                lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {r0['span_coef']} | {wtxt} | " +
                             " | ".join(cells) + f" | {','.join(sorted(ns))} |")
            md = out_dir / f"topic_table_{dataset}_{name}.md"
            md.write_text("\n".join(lines) + "\n")
            print(f"wrote {md}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None, help="default <TM_OUT>/tables")
    ap.add_argument("--with-epochs", action=argparse.BooleanOptionalAction, default=True,
                    help="read best-checkpoint epochs (needs torch + the checkpoints; skipped if absent)")
    ap.add_argument("--steps-per-epoch", default=None,
                    help="override, e.g. 'imagenet=625,openimages=247' (default: floor(n_train/2048))")
    a = ap.parse_args(argv)
    reg = D.load_registry()
    spe = dict(DEFAULT_STEPS_PER_EPOCH)
    if a.steps_per_epoch:
        for kv in a.steps_per_epoch.split(","):
            k, v = kv.split("=")
            spe[k.strip()] = int(v)
    out = Path(a.out_dir) if a.out_dir else Path(D.paths()["tm_out"]) / "tables"
    build(reg, out, a.with_epochs, spe)


if __name__ == "__main__":
    main()
