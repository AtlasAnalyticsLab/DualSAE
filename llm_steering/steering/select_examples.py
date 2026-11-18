"""Qualitative steering example pool: items where one arm clearly steers and the other clearly
does not, under the SAME prompt, continuation seed, concept and latent rank.

The pool is built from the stage outputs (no model is run):
  s6/<unit>/<mode>_<dirset>_b<budget>.jsonl          the generated continuation (field 'text')
  s7/judge/<unit>/<same file>                        phi-4 AxBench ratings
  s7/metrics/<unit>/<same file>                      bge classifier P(concept)
  s6/null/null.jsonl                                 the unsteered continuation of each prompt

Items are matched across the two arms of a pair (Dual vs baseline of one activation, layer and
seed) on (concept, rank, prompt_idx, cont): the same concept, the latent with the same AUROC
rank in each SAE's selection, the same prompt and the same continuation index -- hence the same
sampling random numbers (common random numbers across arms, D-49).

Selection rule (``--direction`` names the arm that must WIN; both directions are supported):
  winner concept rating == 2  AND  loser concept rating == 0
  AND  P_winner(concept) - P_loser(concept) >= 0.5
  AND  winner fluency rating >= 1
Only (layer, activation) pairs that pass the primary D-43 health gate are used (a pair whose
SAEs collapsed is not steered as a pair).  The defaults are mode (i), the 'full' selection and
the primary KL budget P1 read from s4/p1_primary_budget.json.

Outputs (``--out``, default $A16_ROOT/s9/examples_<direction>_wins.{json,md}): the matched items
with both arms' FULL continuations, the unsteered continuation, the ratings and P(concept) of
both arms and both latents; the markdown prints every text verbatim (no truncation).  The pool
size of BOTH directions is printed, so a reader always sees how often each arm wins under the
same rule.

    python -m steering.select_examples --direction dual
    python -m steering.select_examples --direction baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from common import arms
from steering import base, gates43, s6_generate

Key = Tuple[str, int, int, int]          # (concept, rank, prompt_idx, cont)


def _load_rows(path: str) -> Dict[Key, Dict[str, Any]]:
    out: Dict[Key, Dict[str, Any]] = {}
    if not os.path.isfile(path):
        return out
    for row in base.read_jsonl(path):
        out[(row.get("concept"), row.get("rank"), row.get("prompt_idx"), row.get("cont"))] = row
    return out


def unit_items(unit: str, fname: str) -> Dict[Key, Dict[str, Any]]:
    """Generation + judge + metric rows of one unit and shard, merged per item."""
    gen = _load_rows(os.path.join(base.s6_dir(unit), fname))
    jud = _load_rows(os.path.join(base.s7j_dir(unit), fname))
    met = _load_rows(os.path.join(base.s7m_dir(unit), fname))
    out = {}
    for k, g in gen.items():
        if g.get("status") != "ok" or k not in jud or k not in met:
            continue
        c = k[0]
        out[k] = {"text": g.get("text"), "prompt": g.get("prompt"), "latent": g.get("latent"),
                  "concept_rating": (jud[k].get("concept_rating") or {}).get(c),
                  "fluency": jud[k].get("fluency_rating"),
                  "instruction": jud[k].get("instruction_rating"),
                  "hm": (jud[k].get("hm") or {}).get(c),
                  "p": (met[k].get("p_concept") or {}).get(c)}
    return out


def null_texts() -> Dict[Tuple[int, int], str]:
    """Unsteered continuation per (prompt_idx, cont).  The unsteered model does not depend on
    the layer; the null row is generated once, at the pilot layer."""
    path = os.path.join(base.s6_dir("null"), "null.jsonl")
    out = {}
    if os.path.isfile(path):
        for row in base.read_jsonl(path):
            if int(row.get("layer", base.PILOT_LAYER)) == base.PILOT_LAYER:
                out[(row["prompt_idx"], row["cont"])] = row.get("text")
    return out


def is_win(w: Dict[str, Any], l: Dict[str, Any]) -> bool:
    """The selection rule (module docstring); a missing value never qualifies."""
    vals = (w["concept_rating"], l["concept_rating"], w["p"], l["p"], w["fluency"])
    if any(v is None for v in vals):
        return False
    return (w["concept_rating"] == 2 and l["concept_rating"] == 0
            and w["p"] - l["p"] >= 0.5 and w["fluency"] >= 1)


def steered_pairs(gate: str = gates43.PRIMARY):
    """(layer, activation, seed, dual unit, baseline unit) for every pair passing the gate."""
    for layer in base.LAYERS:
        for act in arms.ACTIVATIONS:
            v = gates43.pair_verdict(layer, act, strict=(gate == gates43.STRICT))
            if v["state"] != "pass":
                continue
            for s in arms.SEEDS:
                d = arms.resolve_arm("B", layer, None, act, "dual", s)
                b = arms.resolve_arm("B", layer, None, act, "baseline", s)
                yield layer, act, s, d.tag, b.tag


def build_pool(direction: str, mode: str, dirset: str, budget: float) -> List[Dict[str, Any]]:
    fname = f"{mode}_{dirset}_b{budget:g}.jsonl"
    nulls = null_texts()
    pool = []
    for layer, act, seed, dual_u, base_u in steered_pairs():
        D, B = unit_items(dual_u, fname), unit_items(base_u, fname)
        for k in sorted(set(D) & set(B)):
            d, b = D[k], B[k]
            w, l = (d, b) if direction == "dual" else (b, d)
            if not is_win(w, l):
                continue
            concept, rank, pidx, cont = k
            pool.append({"layer": str(layer), "act": act, "seed": seed, "concept": concept,
                         "rank": rank, "prompt_idx": pidx, "cont": cont, "prompt": d["prompt"],
                         "winner": direction, "dual_unit": dual_u, "baseline_unit": base_u,
                         "dual": d["text"], "base": b["text"], "unsteered": nulls.get((pidx, cont)),
                         "cb": b["concept_rating"], "cd": d["concept_rating"],
                         "pb": b["p"], "pd": d["p"], "fb": b["fluency"], "fd": d["fluency"],
                         "ib": b["instruction"], "id": d["instruction"],
                         "hmb": b["hm"], "hmd": d["hm"],
                         "lat_b": b["latent"], "lat_d": d["latent"]})
    return pool


def write_md(path: str, pool: List[Dict[str, Any]], direction: str, n_other: int) -> None:
    other = "baseline" if direction == "dual" else "dual"
    lines = [f"# Steering examples: clear {direction} wins ({len(pool)} items; clear {other} "
             f"wins under the mirrored rule: {n_other})", "",
             "Rule: winner concept rating 2, loser 0, P(concept) gap >= 0.5, winner fluency >= 1. "
             "Ratings shown as [concept, instruction, fluency, harmonic mean, P(concept)]. "
             "Texts are printed verbatim (the continuation after the prompt).", ""]
    for x in pool:
        lines += [f"## {x['concept']} | block {x['layer']} {x['act']} seed {x['seed']} | rank "
                  f"{x['rank']} | prompt {x['prompt_idx']} cont {x['cont']}",
                  f"PROMPT: {x['prompt']!r}",
                  f"UNSTEERED: {x['unsteered']!r}",
                  f"BASELINE (latent {x['lat_b']}) [{x['cb']}, {x['ib']}, {x['fb']}, {x['hmb']}, "
                  f"{x['pb']:.3f}]: {x['base']!r}",
                  f"DUAL (latent {x['lat_d']}) [{x['cd']}, {x['id']}, {x['fd']}, {x['hmd']}, "
                  f"{x['pd']:.3f}]: {x['dual']!r}", ""]
    base.atomic_write_bytes(path, "\n".join(lines).encode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--direction", choices=("dual", "baseline"), required=True,
                    help="which arm must win (the other direction's pool size is printed too)")
    ap.add_argument("--mode", default="i")
    ap.add_argument("--dirset", default="full")
    ap.add_argument("--budget", type=float, default=None, help="default: P1 (s4 record)")
    ap.add_argument("--out", default=None, help="output path stem (.json and .md are written)")
    a = ap.parse_args()
    budget = a.budget
    if budget is None:
        ok, budget = s6_generate.primary_budget()
        if not ok or budget is None:
            raise SystemExit("P1 is not decided (run s4); pass --budget explicitly")
    pool = build_pool(a.direction, a.mode, a.dirset, budget)
    other = build_pool("baseline" if a.direction == "dual" else "dual", a.mode, a.dirset, budget)
    stem = a.out or base.p("s9", f"examples_{a.direction}_wins")
    base.atomic_write_json(stem + ".json", pool)
    write_md(stem + ".md", pool, a.direction, len(other))
    print(f"clear {a.direction} wins: {len(pool)}; clear "
          f"{'baseline' if a.direction == 'dual' else 'dual'} wins (mirrored rule): {len(other)}"
          f" -> {stem}.json / .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
