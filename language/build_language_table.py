#!/usr/bin/env python3
"""Build the language-model evaluation table from the evaluator's result JSONs.

Reads every ``results/trackb/<tag>.json`` under the main-grid root and the ablation root
(written by ``langdual.lmsae.eval_trackb``), groups cells that differ only by seed, and writes
one row per (corpus, block, activation, variant, training configuration) with the mean and
sample standard deviation over seeds of:

    dCE   ce.loss_reconstructed - ce.loss_original           (nats; spliced minus clean)
    KL    ce.kl_clean_vs_reconstructed                       (mean next-token KL(clean||spliced))
    R2    recon.r2_global_a4                                 (pooled R^2 of x_hat against x)
    MS    monoscore.minmax.mean_active_paper_a4              (MonoScore, mean over latents that
                                                              fire on the scoring set; use
                                                              --ms-reduction all for the mean
                                                              over all latents)

Nothing is chosen or filtered by outcome: every configuration that has results gets its own
row, labelled with the configuration it was trained with.  The provenance columns are read
from each run's own ``training_config.json`` (warm-up, learning rate, decorrelation
coefficient gamma = span_coef, JumpReLU threshold init, tied decoder) and from the result JSON
(checkpoint scored, its epoch, L0, dead-latent fraction), so the source of every row is
visible.  Seeds are the only dimension averaged over.

    python build_language_table.py                       # block 12, both corpora
    python build_language_table.py --layers 6 12 18 --out-dir <dir>

Outputs (default <LANGDUAL_ROOT>/reports/): language_table.md, language_table.csv,
language_table.tex, language_table_cells.csv (one row per cell, before averaging).
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
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from langdual import ablate, config, jobs  # noqa: E402

PAIR_LABEL = {a: (b, d) for a, b, d in config.ACTIVATION_PAIRS}
CORPUS_LABEL = {"owt": "OpenWebText", "ghcode": "GitHub code"}
METRICS = ("dCE", "KL", "R2", "MS")


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _training_config(run_dir: str) -> Dict[str, Any]:
    return _read_json(os.path.join(run_dir, "training_config.json")) or {}


def cell_row(path: str, ms_reduction: str) -> Optional[Dict[str, Any]]:
    """One per-cell record from a result JSON (None if it lacks a required block)."""
    d = _read_json(path)
    if not d or d.get("status") != "ok" or not all(k in d for k in ("ce", "recon", "monoscore")):
        return None
    tag = Path(path).stem            # the file is named after the run directory = cell tag
    meta, cell = d.get("meta", {}), d.get("cell", {})
    run_dir = meta.get("run_dir") or ""
    # Resolve the run directory through the registry when the recorded one is not on this
    # machine (results copied from elsewhere): main-grid and ablation tags are both known.
    if not os.path.isdir(run_dir):
        run_dir = (os.path.join(ablate.RUNS_DIR, tag) if tag.startswith("abl_")
                   else os.path.join(config.RUNS_DIR, tag))
    tc = _training_config(run_dir)
    ce, rc, ms = d["ce"], d["recon"], d["monoscore"]["minmax"]
    ms_key = "mean_active_paper_a4" if ms_reduction == "active" else "mean_all_table3"
    is_dual = (tc.get("model_type") or meta.get("model_type")) == "dual"
    row = {
        "tag": tag,
        "grid": "ablation" if tag.startswith("abl_") else "main",
        "axis": getattr(jobs.cell_from_tag(tag), "axis", ""),
        "corpus": cell.get("corpus"),
        "layer": int(cell.get("layer")),
        "activation": cell.get("activation"),
        "variant": cell.get("variant"),
        "seed": int(cell.get("seed")),
        # ---- training configuration, from the run's own training_config.json ----
        "warmup_steps": tc.get("warmup_steps") if is_dual else "",
        "lr": tc.get("lr", meta.get("lr")),
        "span_coef": tc.get("span_coef") if is_dual else "",
        # Runs trained before the explicit flag existed did not record it; for those, the
        # value is the model-code default of that time (see README, Implementation notes).
        "jump_threshold_init": (tc.get("jump_threshold_init", "not recorded")
                                if cell.get("activation") == "jump_relu" else ""),
        "tied_weights": tc.get("tied_weights", "") if (cell.get("activation") == "topk"
                                                       and not is_dual) else "",
        "num_steps": (tc.get("num_epochs") or 0) * (tc.get("steps_per_epoch") or 0) or "",
        # ---- what was scored ----
        "checkpoint": meta.get("checkpoint", "best"),
        "best_epoch": meta.get("best_epoch"),
        "n_tokens_scored": d.get("protocol", {}).get("n_tokens_scored"),
        "L0": d.get("sparsity", {}).get("l0_mean"),
        "dead_fraction": d.get("sparsity", {}).get("dead_fraction"),
        # ---- metrics ----
        "dCE": ce["loss_reconstructed"] - ce["loss_original"],
        "KL": ce.get("kl_clean_vs_reconstructed", float("nan")),
        "R2": rc["r2_global_a4"],
        "MS": ms[ms_key],
    }
    return row


def config_label(r: Dict[str, Any]) -> str:
    """Human-readable training configuration of a row (what distinguishes it)."""
    parts = []
    if r["variant"] == "dual":
        parts.append(f"warm-up {r['warmup_steps']}")
        if r["span_coef"] not in ("", None) and float(r["span_coef"]) != config.SPAN_COEF:
            parts.append(f"gamma {r['span_coef']}")
    if r["activation"] == "topk" and r["variant"] == "baseline" and r["tied_weights"] is False:
        parts.append("untied decoder")
    return ", ".join(parts) if parts else "-"


def _mean_sd(vals: List[float]) -> Tuple[float, float]:
    """Mean and SAMPLE standard deviation (ddof=1); NaN when undefined.  Written out rather
    than statistics.stdev, which raises on NaN inputs (e.g. a result without a KL entry)."""
    if not vals:
        return float("nan"), float("nan")
    mu = sum(vals) / len(vals)
    if len(vals) < 2:
        return mu, float("nan")
    return mu, math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def group(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse seeds; everything else (incl. the configuration) stays a separate row."""
    keys = ("corpus", "layer", "activation", "variant", "grid", "axis", "warmup_steps", "lr",
            "span_coef", "jump_threshold_init", "tied_weights", "num_steps", "checkpoint")
    buckets: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(tuple(str(r[k]) for k in keys), []).append(r)
    out = []
    for _, rs in buckets.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        g = {k: rs[0][k] for k in keys}
        g["config"] = config_label(rs[0])
        g["seeds"] = "|".join(str(r["seed"]) for r in rs)
        g["n_seeds"] = len(rs)
        g["tags"] = " ".join(r["tag"] for r in rs)
        g["best_epochs"] = "|".join(str(r["best_epoch"]) for r in rs)
        for m in METRICS + ("L0", "dead_fraction"):
            vals = [float(r[m]) for r in rs if r[m] is not None]
            g[f"{m}_mean"], g[f"{m}_sd"] = _mean_sd(vals)
        out.append(g)
    order = {c: i for i, c in enumerate(config.LM_CORPUS_ORDER)}
    act = {a: i for i, a in enumerate(config.ACTIVATIONS)}
    out.sort(key=lambda g: (order.get(g["corpus"], 9), g["layer"], act.get(g["activation"], 9),
                            g["variant"] != "baseline", g["grid"] != "main", g["axis"],
                            int(g["warmup_steps"]) if str(g["warmup_steps"]).isdigit() else -1))
    return out


def _fmt(m: float, s: float) -> Tuple[str, str]:
    return ("nan" if math.isnan(m) else f"{m:.4f}", "n/a" if math.isnan(s) else f"{s:.4f}")


def method_name(g: Dict[str, Any]) -> str:
    base, dual = PAIR_LABEL[g["activation"]]
    return dual if g["variant"] == "dual" else base


def write_markdown(groups: List[Dict[str, Any]], path: str, ms_reduction: str) -> None:
    lines = [
        "# Language-model evaluation (Pythia-410m-deduped residual stream)",
        "",
        "Mean ± sample sd over seeds (n given).  dCE = CE(spliced) − CE(clean) in nats; "
        "KL = mean next-token KL(clean ‖ spliced); R² = pooled R² of x̂; "
        f"MS = MonoScore ({'mean over latents that fire' if ms_reduction == 'active' else 'mean over all latents'}).  "
        "Configuration columns are read from each run's training_config.json.",
        "",
        "| corpus | block | method | configuration | n | dCE | KL | R² | MS | L0 | dead | lr | "
        "JumpReLU θ init | checkpoint (epochs) | cells |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for g in groups:
        cols = []
        for m in METRICS:
            mu, sd = _fmt(g[f"{m}_mean"], g[f"{m}_sd"])
            cols.append(f"{mu} ± {sd}")
        lines.append(
            f"| {CORPUS_LABEL.get(g['corpus'], g['corpus'])} | {g['layer']} | {method_name(g)} | "
            f"{g['config']} | {g['n_seeds']} | " + " | ".join(cols) + " | "
            f"{g['L0_mean']:.1f} | {g['dead_fraction_mean']:.3f} | {g['lr']} | "
            f"{g['jump_threshold_init'] or '-'} | {g['checkpoint']} ({g['best_epochs']}) | "
            f"{g['tags'].split()[0].rsplit('_seed', 1)[0]}_seed{{{g['seeds'].replace('|', ',')}}} |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_latex(groups: List[Dict[str, Any]], path: str) -> None:
    rows = []
    last = None
    for g in groups:
        corpus = CORPUS_LABEL.get(g["corpus"], g["corpus"]) if g["corpus"] != last else ""
        last = g["corpus"]
        cells = []
        for m in METRICS:
            mu, sd = _fmt(g[f"{m}_mean"], g[f"{m}_sd"])
            cells.append(f"{mu}\\textsubscript{{{sd}}}")
        cfg = g["config"].replace("gamma", "$\\gamma$")
        rows.append(f"{corpus} & {g['layer']} & {method_name(g)} & {cfg} & " + " & ".join(cells)
                    + " \\\\")
    body = "\n".join(rows)
    tex = ("% Generated by build_language_table.py from the evaluation result JSONs.\n"
           "\\begin{tabular}{llllcccc}\n\\toprule\n"
           "Corpus & Block & Method & Configuration & $\\Delta$CE & KL & $R^2$ & MS \\\\\n"
           "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")
    with open(path, "w") as fh:
        fh.write(tex)


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, nargs="+", default=[12])
    ap.add_argument("--corpora", nargs="+", default=list(config.LM_CORPUS_ORDER))
    ap.add_argument("--include", choices=("all", "main", "ablation"), default="all",
                    help="which result trees to read (default: both)")
    ap.add_argument("--ms-reduction", choices=("active", "all"), default="active")
    ap.add_argument("--out-dir", default=config.REPORT_DIR)
    a = ap.parse_args()

    files = []
    if a.include in ("all", "main"):
        files += glob.glob(os.path.join(config.RESULT_DIR, "trackb", "*.json"))
    if a.include in ("all", "ablation"):
        files += glob.glob(os.path.join(ablate.RESULT_DIR, "trackb", "*.json"))
    # Only cells this component's registry defines are tabulated (a result tree may hold
    # other files); the count of skipped files is printed, never silently dropped.
    known, unknown = [], 0
    for f in sorted(files):
        try:
            jobs.cell_from_tag(Path(f).stem)
            known.append(f)
        except KeyError:
            unknown += 1
    if unknown:
        print(f"[note] {unknown} result file(s) are not cells of this registry and are skipped")
    rows = [r for r in (cell_row(f, a.ms_reduction) for f in known) if r]
    rows = [r for r in rows if r["layer"] in a.layers and r["corpus"] in a.corpora]
    if not rows:
        print("no result JSONs matched; run the eval stage first "
              f"(searched {config.RESULT_DIR} and {ablate.RESULT_DIR})", file=sys.stderr)
        return 1
    groups = group(rows)
    os.makedirs(a.out_dir, exist_ok=True)
    write_csv(rows, os.path.join(a.out_dir, "language_table_cells.csv"))
    write_csv(groups, os.path.join(a.out_dir, "language_table.csv"))
    write_markdown(groups, os.path.join(a.out_dir, "language_table.md"), a.ms_reduction)
    write_latex(groups, os.path.join(a.out_dir, "language_table.tex"))
    print(f"{len(rows)} cells -> {len(groups)} rows; wrote language_table.{{md,csv,tex}} "
          f"and language_table_cells.csv to {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
