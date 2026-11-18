"""The quantitative steering table, built from the s9 outputs (no hand-typed numbers).

Inputs:  $A16_ROOT/s9/per_seed.csv (one row per SAE seed x cell x metric) and
         $A16_ROOT/s9/tests.csv (or tests_strict.csv with --gate strict): the Welch test,
         BH q-value and verdict word of every confirmatory contrast.
Output:  a LaTeX table (``--out``, default $A16_ROOT/s9/steering_table.tex), the same table as
         markdown (.md) and a JSON summary of the counts the caption states (.json).

What the table shows (defaults = the confirmatory family of PROTOCOL 4.1): mode (i) direction
add, the 'full' latent selection, the primary KL budget P1 (s4 record), the two co-primary
metrics (AxBench harmonic mean ``hm`` and bge classifier ``p_concept``), one column group per
(activation, layer) setting and one column per arm.  A cell is mean +- sample sd (ddof=1) over
seeds.  Every column is labelled from the ``arm`` field of the rows it is computed from
(ARM_LABEL maps the arm id to its display name); nothing is placed by position.

Settings: a (layer, activation) pair enters the table only when its contrast is in the tested
family (it passed the health gates).  Pairs removed by a gate are listed in the caption with
the label that removed them (e.g. "collapsed", "not sparsity-matched ...").

Markers, computed from the tests file (the contrast ``dual-baseline`` by default):
  bold        the higher mean of the two arms in each (setting, concept, metric) cell;
  $^\\dagger$  the arm with the LOWER mean, when the test is significant after BH (q < 0.10);
  $^\\ast$     both arms, when the test is excluded ("not assessable": seed-outlier rule or < 2
              seeds).

    python -m steering.make_table                      # primary gate (the headline)
    python -m steering.make_table --gate strict        # the sensitivity analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from common import stats
from steering import base, concepts, s6_generate

ARM_LABEL = {"baseline": "Standard", "dual": "DualSAE", "dual_g0": r"DualSAE ($\gamma{=}0$)"}
ACT_LABEL = {"topk": "TK", "batch_topk": "BTK", "jump_relu": "JR"}
ACT_ORDER = ("topk", "batch_topk", "jump_relu")
METRIC_LABEL = {"hm": r"AxBench score (0--2) $\uparrow$",
                "p_concept": r"$P(\text{concept})$ (0--1) $\uparrow$"}
CORPUS_LABEL = {"amazon": "AMZ", "agnews": "AGN", "goemotions": "GoE"}
CONCEPT_LABEL = {"amazon_positive": "Positive", "amazon_negative": "Negative",
                 "agnews_world": "World", "agnews_sports": "Sports",
                 "agnews_business": "Business", "agnews_scitech": "Sci/Tech",
                 "goemotions_joy": "Joy", "goemotions_anger": "Anger",
                 "goemotions_sadness": "Sadness", "goemotions_fear": "Fear"}
NOT_ASSESSABLE = "not assessable"


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(x: str) -> float:
    return float(x) if x not in ("", None) else float("nan")


def _same_budget(a: str, b: float) -> bool:
    return a not in ("", None) and abs(float(a) - b) < 1e-9


def seed_values(rows: List[Dict[str, str]], mode: str, dirset: str, budget: float,
                metrics: Tuple[str, ...], gate: str) -> Dict[Tuple, Dict[int, float]]:
    """{(layer, activation, arm, concept, metric): {seed: value}} for the selected cell type,
    without the rows the chosen gate version removed (gate_removed_<gate> non-empty)."""
    col = f"gate_removed_{gate}"
    out: Dict[Tuple, Dict[int, float]] = defaultdict(dict)
    for r in rows:
        if (r["mode"] != mode or r["dirset"] != dirset or r["metric"] not in metrics
                or not _same_budget(r["budget"], budget) or r.get(col)):
            continue
        v = _f(r["value"])
        if not math.isfinite(v):
            continue
        out[(int(r["layer"]), r["activation"], r["arm"], r["concept"], r["metric"])][int(r["seed"])] = v
    return out


def family_tests(rows: List[Dict[str, str]], contrast: str, mode: str, dirset: str,
                 budget: float, metrics: Tuple[str, ...]) -> Dict[Tuple, Dict[str, str]]:
    """{(layer, activation, concept, metric): test row} of one contrast, family cells only."""
    out = {}
    for r in rows:
        if (r["contrast"] == contrast and r["family"] == "True" and r["mode"] == mode
                and r["dirset"] == dirset and r["metric"] in metrics
                and _same_budget(r["budget"], budget)):
            out[(int(r["layer"]), r["activation"], r["concept"], r["metric"])] = r
    return out


def is_tested(t: Dict[str, str]) -> bool:
    """A family test that reached BH (it has a BH q-value and is not 'not assessable')."""
    return t.get("q", "") not in ("", "nan") and t.get("verdict") != NOT_ASSESSABLE


def build(per_seed: str, tests: str, gate: str, contrast: Tuple[str, str], mode: str,
          dirset: str, budget: float, metrics: Tuple[str, ...]) -> Dict[str, Any]:
    vals = seed_values(_read_csv(per_seed), mode, dirset, budget, metrics, gate)
    fam = family_tests(_read_csv(tests), f"{contrast[0]}-{contrast[1]}", mode, dirset, budget,
                       metrics)
    # Settings in the table: (layer, activation) pairs with at least one family test that is
    # NOT removed by a gate label (collapsed / not sparsity-matched / pending).  'not assessable'
    # tests stay in their setting (they are marked, not removed).
    settings, removed = set(), defaultdict(set)
    for (layer, act, _c, _m), t in fam.items():
        lab = t.get("gate_label") or ""
        if t.get("verdict") == NOT_ASSESSABLE or is_tested(t):
            settings.add((layer, act))
        elif lab:
            removed[(layer, act)].add(lab)
    settings = sorted(settings, key=lambda s: (ACT_ORDER.index(s[1]), s[0]))
    removed = {k: sorted(v) for k, v in removed.items() if k not in settings}
    arms_shown = (contrast[1], contrast[0])          # comparator first, then the Dual arm
    cells = {}
    counts = defaultdict(int)
    for (layer, act) in settings:
        for metric in metrics:
            for c in concepts.CONCEPTS:
                cell = {}
                for arm in arms_shown:
                    sv = vals.get((layer, act, arm, c.cid, metric), {})
                    x = np.array([sv[s] for s in sorted(sv)], dtype=np.float64)
                    cell[arm] = {"arm": arm, "n": int(x.size),
                                 "mean": float(x.mean()) if x.size else float("nan"),
                                 "sd": stats.sample_sd(x) if x.size >= 2 else float("nan"),
                                 "seeds": sorted(sv)}
                t = fam.get((layer, act, c.cid, metric))
                cell["test"] = t
                m_a, m_b = cell[arms_shown[1]]["mean"], cell[arms_shown[0]]["mean"]
                lower = arms_shown[1] if m_a < m_b else arms_shown[0] if m_b < m_a else None
                cell["lower"] = lower
                cell["significant"] = bool(t is not None and is_tested(t)
                                           and float(t["q"]) < stats.Q_THRESHOLD)
                cell["excluded"] = bool(t is not None and t.get("verdict") == NOT_ASSESSABLE)
                cells[(layer, act, metric, c.cid)] = cell
                counts["comparisons"] += 1
                if lower is not None:
                    counts[f"lower_mean_{lower}"] += 1
                if t is not None:
                    v = (t.get("verdict") or "").split(" (")[0]
                    counts[f"verdict: {v}"] += 1
    n_tested_all = 0
    for r in _read_csv(tests):
        if r["family"] == "True" and is_tested(r):
            n_tested_all += 1
    return {"settings": settings, "removed": removed, "cells": cells, "arms": arms_shown,
            "counts": dict(counts), "n_tested_family_all_contrasts": n_tested_all,
            "gate": gate, "contrast": contrast, "mode": mode, "dirset": dirset,
            "budget": budget, "metrics": metrics}


def _fmt_cell(cell: Dict[str, Any], arm: str, latex: bool) -> str:
    a = cell[arm]
    if not a["n"]:
        return "--"
    mean, sd = f"{a['mean']:.3f}", (f"{a['sd']:.3f}" if math.isfinite(a["sd"]) else "--")
    other = [x for x in cell if x in ARM_LABEL and x != arm][0]
    higher = math.isfinite(cell[other]["mean"]) and a["mean"] > cell[other]["mean"]
    dag = cell["significant"] and cell["lower"] == arm
    ast = cell["excluded"]
    if latex:
        s = (r"\textbf{" + mean + "}" if higher else mean) + r"\textsubscript{$\pm$" + sd + "}"
        return s + (r"$^\dagger$" if dag else "") + (r"$^\ast$" if ast else "")
    s = (f"**{mean}**" if higher else mean) + f" ± {sd}"
    return s + (" †" if dag else "") + (" *" if ast else "")


def caption(res: Dict[str, Any]) -> str:
    comp, dual = res["arms"]
    c = res["counts"]
    parts = [f"Steering at a next-token KL budget of {res['budget']:g} nats (mode "
             f"({res['mode']}), '{res['dirset']}' selection, {res['gate']} gate). "
             f"Columns: {ARM_LABEL[comp]} = arm '{comp}', {ARM_LABEL[dual]} = arm '{dual}'. "
             "Mean $\\pm$ sd over seeds; bold = higher mean; "
             f"$^\\dagger$ = lower mean in a test significant after BH ($q<{stats.Q_THRESHOLD:g}$); "
             "$^\\ast$ = test excluded (not assessable).",
             f"{ARM_LABEL[dual]} has the lower mean in {c.get('lower_mean_' + dual, 0)} and "
             f"{ARM_LABEL[comp]} in {c.get('lower_mean_' + comp, 0)} of {c.get('comparisons', 0)} "
             "comparisons."]
    verd = {k[len("verdict: "):]: v for k, v in c.items() if k.startswith("verdict: ")}
    if verd:
        parts.append("Verdicts: " + "; ".join(f"{k}: {v}" for k, v in sorted(verd.items())) + "; "
                     f"BH over {res['n_tested_family_all_contrasts']} tested family tests "
                     "(all contrasts).")
    if res["removed"]:
        parts.append("Not shown: " + "; ".join(
            f"block {l} {ACT_LABEL[a]} ({', '.join(labs)})"
            for (l, a), labs in sorted(res["removed"].items(),
                                       key=lambda kv: (ACT_ORDER.index(kv[0][1]), kv[0][0]))) + ".")
    return " ".join(parts)


def to_latex(res: Dict[str, Any]) -> str:
    comp, dual = res["arms"]
    st = res["settings"]
    ncol = 2 + 2 * len(st)
    lines = [r"% generated by steering/make_table.py from s9/per_seed.csv and s9/tests.csv",
             r"\begin{table*}[t]", r"\centering",
             r"\caption{" + caption(res) + "}", r"\label{tab:text_steering_quant}",
             r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{@{}ll" + "c" * (2 * len(st)) + "@{}}", r"\toprule",
             " & & " + " & ".join(r"\multicolumn{2}{c}{Block " + f"{l}, {ACT_LABEL[a]}" + "}"
                                  for l, a in st) + r" \\",
             "".join(r"\cmidrule(lr){" + f"{3 + 2 * i}-{4 + 2 * i}" + "}" for i in range(len(st))),
             "Data & Concept & " + " & ".join(f"{ARM_LABEL[comp]} & {ARM_LABEL[dual]}" for _ in st)
             + r" \\", r"\midrule"]
    for mi, metric in enumerate(res["metrics"]):
        if mi:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{" + str(ncol) + r"}{@{}l}{\textit{" + METRIC_LABEL.get(metric, metric)
                     + r"}} \\")
        prev = None
        for c in concepts.CONCEPTS:
            corpus = CORPUS_LABEL.get(c.corpus, c.corpus)
            if prev is not None and corpus != prev:
                lines.append(r"\cmidrule(lr){1-" + str(ncol) + "}")
            n_in = sum(1 for x in concepts.CONCEPTS if x.corpus == c.corpus)
            first = corpus != prev
            prev = corpus
            head = (r"\multirow{" + str(n_in) + r"}{*}{" + corpus + "}" if first else "")
            row = [head, CONCEPT_LABEL.get(c.cid, c.cid)]
            for (l, a) in st:
                cell = res["cells"][(l, a, metric, c.cid)]
                row += [_fmt_cell(cell, comp, True), _fmt_cell(cell, dual, True)]
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def to_markdown(res: Dict[str, Any]) -> str:
    comp, dual = res["arms"]
    st = res["settings"]
    hdr = ["metric", "data", "concept"] + [f"B{l} {ACT_LABEL[a]} {ARM_LABEL[arm]}"
                                           for l, a in st for arm in (comp, dual)]
    out = [caption(res), "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for metric in res["metrics"]:
        for c in concepts.CONCEPTS:
            row = [metric, CORPUS_LABEL.get(c.corpus, c.corpus), CONCEPT_LABEL.get(c.cid, c.cid)]
            for (l, a) in st:
                cell = res["cells"][(l, a, metric, c.cid)]
                row += [_fmt_cell(cell, comp, False), _fmt_cell(cell, dual, False)]
            out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", choices=("primary", "strict"), default="primary")
    ap.add_argument("--contrast", default="dual-baseline", choices=("dual-baseline", "dual-dual_g0"))
    ap.add_argument("--mode", default="i")
    ap.add_argument("--dirset", default="full")
    ap.add_argument("--budget", type=float, default=None, help="default: P1 (s4 record)")
    ap.add_argument("--metrics", nargs="+", default=["hm", "p_concept"])
    ap.add_argument("--per-seed", default=None, help="default: $A16_ROOT/s9/per_seed.csv")
    ap.add_argument("--tests", default=None,
                    help="default: $A16_ROOT/s9/tests.csv (tests_strict.csv with --gate strict)")
    ap.add_argument("--out", default=None, help="output stem (default $A16_ROOT/s9/steering_table)")
    a = ap.parse_args()
    budget = a.budget
    if budget is None:
        ok, budget = s6_generate.primary_budget()
        if not ok or budget is None:
            raise SystemExit("P1 is not decided (run s4); pass --budget explicitly")
    per_seed = a.per_seed or base.p("s9", "per_seed.csv")
    tests = a.tests or base.p("s9", "tests.csv" if a.gate == "primary" else "tests_strict.csv")
    res = build(per_seed, tests, a.gate, tuple(a.contrast.split("-", 1)), a.mode, a.dirset,
                budget, tuple(a.metrics))
    stem = a.out or base.p("s9", "steering_table" + ("" if a.gate == "primary" else "_strict"))
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    base.atomic_write_bytes(stem + ".tex", to_latex(res).encode())
    base.atomic_write_bytes(stem + ".md", to_markdown(res).encode())
    summary = {k: res[k] for k in ("gate", "contrast", "mode", "dirset", "budget", "metrics",
                                   "counts", "n_tested_family_all_contrasts")}
    summary["settings"] = [list(s) for s in res["settings"]]
    summary["removed"] = {f"L{l}_{a}": v for (l, a), v in res["removed"].items()}
    base.atomic_write_json(stem + ".json", summary)
    print(caption(res))
    print(f"-> {stem}.tex / .md / .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
