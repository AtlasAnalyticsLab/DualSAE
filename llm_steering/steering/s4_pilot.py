"""s4: the arm-free L12 pilot that fixes the PRIMARY KL budget (D-48, C17;
PROTOCOL P1, R6, R7).

  s4a (GPU, one unit):  calibrate DiffMean and the 5 random unit directions of each of the 10
        concepts at L12 in mode (i) at every budget (the s5 arm-free procedure, written to the s5
        'armfree_L12' record, so s5 never repeats it), then generate the first 20 prompts (R6) x 1
        continuation per (direction, budget) (+ the unsteered 20 x 1 null, a reference only).
        Same generation code and item seeds as s6 (s6_generate.run_cells).
  s4j (GPU, one unit):  phi-4 CONCEPT_RELEVANCE rating of every pilot continuation
        (python -m steering.s7_judge --pilot).
  s4p (CPU, seconds): the P1 rule, fixed in PROTOCOL: P1 = the smallest budget at which
        DiffMean's mean concept score >= 0.5 (0-2 scale) while the random directions' mean concept
        score <= 0.2; "none" if no budget qualifies (PROTOCOL 4.6).  Declared reading: "mean concept
        score" pools every (concept, prompt) item of the direction family at that budget
        (random: 5 directions x 10 concepts x 20 prompts); unreachable directions contribute no
        items (their count is printed).  Writes base.p1_json() (the record for PROTOCOL P1) and prints
        the P1 table.  PROTOCOL R7: the L12 value applies to all three layers.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import os
import sys
from typing import Any, Dict, List

import numpy as np

from steering import base, calibrate, concepts, s6_generate

PILOT_UNIT = f"armfree_L{base.PILOT_LAYER}"


def pilot_cells() -> List[Dict[str, Any]]:
    cells = []
    for ds in ("diffmean", "random"):
        for b in base.BUDGETS:
            cells.append({"name": f"i_{ds}_b{b:g}", "mode": "i", "dirset": ds, "budget": b,
                          "headline": False, "n_prompts": base.N_PROMPTS_OTHER,
                          "n_cont": base.N_CONT_OTHER})
    return cells


def null_cell() -> Dict[str, Any]:
    return {"name": "null_pilot", "mode": "null", "dirset": "null", "budget": 0.0,
            "headline": False, "n_prompts": base.N_PROMPTS_OTHER, "n_cont": base.N_CONT_OTHER}


def gen_done() -> bool:
    d = base.p("s4", "gen")
    return all(os.path.isfile(os.path.join(d, c["name"] + ".jsonl")) for c in pilot_cells() + [null_cell()])


def run_generate(device: str, n_prompts_cap: int = 0, budgets=None, cells=None) -> Dict[str, Any]:
    if not calibrate.is_done(PILOT_UNIT):
        calibrate.run_unit(PILOT_UNIT, device, budgets=budgets or base.BUDGETS)
    out = base.p("s4", "gen")
    cs = [c for c in pilot_cells() if (not cells or c["name"] in cells)]
    s1 = s6_generate.run_cells(PILOT_UNIT, cs, out, device, n_prompts_cap=n_prompts_cap)
    s2 = s6_generate.run_cells("null", [null_cell()], out, device, n_prompts_cap=n_prompts_cap,
                               skip_identity=True)
    summ = {"pilot": s1, "null": s2}
    base.atomic_write_json(base.p("s4", "gen", "_summary.json"), summ)
    return summ


def p1_rule(judge_dir: str = None) -> Dict[str, Any]:
    judge_dir = judge_dir or base.p("s4", "judge")
    table = []
    per_concept: Dict[str, Dict[str, Any]] = {}
    for b in base.BUDGETS:
        row = {"budget": b}
        for ds in ("diffmean", "random"):
            path = os.path.join(judge_dir, f"i_{ds}_b{b:g}.jsonl")
            if not os.path.isfile(path):
                if base.IS_SMOKE:              # a smoke judges a subset of budgets only
                    rows = []
                else:
                    raise FileNotFoundError(f"{path} missing: judge the pilot first (s7_judge --pilot)")
            else:
                rows = base.read_jsonl(path)
            ok = [r for r in rows if r["status"] == "ok"]
            sc = [r["concept_rating"][r["concept"]] for r in ok]
            row[f"{ds}_mean"] = float(np.mean(sc)) if sc else float("nan")
            row[f"{ds}_n_items"] = len(sc)
            row[f"{ds}_n_unreachable_rows"] = len(rows) - len(ok)
            for c in concepts.CONCEPTS:
                v = [r["concept_rating"][r["concept"]] for r in ok if r["concept"] == c.cid]
                per_concept.setdefault(f"{b:g}", {}).setdefault(ds, {})[c.cid] = (
                    float(np.mean(v)) if v else None)
        row["qualifies"] = bool(np.isfinite(row["diffmean_mean"]) and row["diffmean_mean"] >= 0.5
                                and np.isfinite(row["random_mean"]) and row["random_mean"] <= 0.2)
        table.append(row)
    q = [r["budget"] for r in table if r["qualifies"]]
    primary = min(q) if q else None
    return {"primary_budget": primary if primary is not None else "none", "table": table,
            "per_concept_means": per_concept, "rule": "smallest budget with DiffMean mean concept "
            "score >= 0.5 and random mean concept score <= 0.2 (PROTOCOL P1)",
            "layer": base.PILOT_LAYER, "applies_to_layers": list(base.LAYERS),
            "prompts_sha256": base.PROMPTS_SHA256,
            "concept_registry_sha256": concepts.concept_registry_sha256(),
            "date": datetime.date.today().isoformat(), "smoke": base.IS_SMOKE}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate", action="store_true", help="s4a")
    ap.add_argument("--rule", action="store_true", help="s4p: apply the P1 rule, write the P1 JSON")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-prompts-cap", type=int, default=0, help="smoke only")
    ap.add_argument("--budgets", type=float, nargs="*", default=None, help="smoke only")
    ap.add_argument("--cells", nargs="*", default=None, help="smoke only")
    ap.add_argument("--job", default=os.environ.get("A16_JOB_ID", "local"),
                    help="job id / log recorded in the P1 JSON")
    a = ap.parse_args()
    if (a.n_prompts_cap or a.budgets or a.cells) and not base.IS_SMOKE:
        raise SystemExit("--n-prompts-cap / --budgets / --cells are smoke options")
    if a.generate or a.rule:
        # P2 (D-49: before any steered generation, including this pilot): the pinned prompt file.
        base.check_prompts()
    if a.generate:
        if gen_done() and not a.cells:
            print("[s4a] pilot generation already complete")
            return 0
        s = run_generate(a.device or base.device_default(), a.n_prompts_cap, a.budgets, a.cells)
        print(f"[s4a] {s['pilot']['cells']}")
    if a.rule:
        if os.path.isfile(base.p1_json()) and not base.IS_SMOKE:
            print(f"[s4p] {base.p1_json()} exists; a decided P1 is never re-written (PROTOCOL 6)")
            print(open(base.p1_json()).read())
            return 0
        rec = p1_rule()
        rec["job"] = a.job
        base.atomic_write_json(base.p1_json(), rec)
        print("| Budget (nats) | DiffMean mean concept score | Random mean concept score | Qualifies |")
        print("|---|---|---|---|")
        for r in rec["table"]:
            print(f"| {r['budget']:g} | {r['diffmean_mean']:.3f} (n={r['diffmean_n_items']}) | "
                  f"{r['random_mean']:.3f} (n={r['random_n_items']}) | {'yes' if r['qualifies'] else 'no'} |")
        print(f"P1 = {rec['primary_budget']}  -> {base.p1_json()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
