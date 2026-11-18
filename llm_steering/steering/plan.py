"""Completion-aware planner for the steering cascade.

Reads each stage's on-disk outputs with predicates that MIRROR the stages' own skip-guards and
classifies every unit as
  done      its output exists (never re-run),
  todo      runnable now (all inputs on disk),
  blocked   waiting on an input or HELD (the JumpReLU pairs, see base.held_reason),
  excluded  never runs (the SAE failed the D-43 gate: "not calibrated, generated or judged").
Many-to-one stages (s1 per layer, s4a/s4j, s9) are 'todo' only when EVERY input is present, so
run_all.sh can run the cascade in waves without dependency chains (D-51).  Steps that are not
per-unit stages are listed as MANUAL: s3 (prompt file check, P2), s4p (P1 rule), s8x
(hand-label export).
Rules encoded here (each mirrors the stage's own refusal):
  * D-43 / D-18 exclusion (gates43.excluded_reason): a failing SAE, and every TopK SAE of a layer
    whose tkuntied baseline failed, is 'excluded' from s5/s6/s7j (never calibrated, generated or
    judged).
  * s5 armfree_L<PILOT_LAYER> is owned by s4a (the pilot calibrates it); it is 'done' when that
    record exists and otherwise 'blocked', never 'todo', so the two can never race.
  * s7m is merged into s6 (D-51): an s6 unit is done only with generation AND metrics; s6 needs
    the s7c classifier first.  s7j needs only the generation (live-matched shards excluded).

CLI
  python -m steering.plan --summary
  python -m steering.plan --plan          # stage|array|n_todo|n_done|n_total|n_blocked|n_excluded
  python -m steering.plan --stage s0      # compressed incomplete indices
  python -m steering.plan --why s5        # blocked reasons, aggregated
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

from steering import base, jobs

STAGE_ORDER = ("s0", "s1", "s2", "s4a", "s4j", "s5", "s6", "s7c", "s7j", "s8j", "s9")


@functools.lru_cache(maxsize=None)
def _refs():
    return {r.tag: r for r in base.sae_refs()}


@functools.lru_cache(maxsize=None)
def _pair(layer: int, act: str):
    from steering import gates43
    return gates43.pair_verdict(layer, act)


def _sae_gate(tag: str) -> str:
    r = _refs()[tag]
    return _pair(int(r.key[1:]), r.activation)["sae"][tag]["status"]


def _sae_excluded(tag: str) -> Optional[str]:
    from steering import gates43
    r = _refs()[tag]
    return gates43.excluded_reason(_pair(int(r.key[1:]), r.activation), tag)


@functools.lru_cache(maxsize=None)
def _prereg(stage: str) -> Optional[str]:
    # The original pipeline also kept stages blocked until their protocol entries were recorded
    # in a pre-registration file; this release has no such guard (see README).
    return None


def _s0_done(tag):
    from steering import s0_rescore
    return s0_rescore.is_done(tag)


def _p1() -> Tuple[bool, Optional[float]]:
    from steering import s6_generate
    return s6_generate.primary_budget()


def _prompts_ok() -> Optional[str]:
    try:
        base.check_prompts()
        return None
    except Exception as e:  # noqa: BLE001 - reason string for the operator
        return f"prompt file: {e}"


def classify_unit(kind: str, unit: str) -> Tuple[str, str]:
    """(state, reason) for one unit; state in done | todo | blocked | excluded."""
    from steering import calibrate, s1_select, s2_cancel, s4_pilot, s6_generate, s7_judge, s7_metrics
    refs = _refs()
    ref = refs.get(unit)
    if ref is not None:
        why = base.held_reason(ref)
        if why:
            return "blocked", why
    if kind == "s0":
        return ("done", "") if _s0_done(unit) else ("todo", "")
    if kind == "s1":
        layer = int(unit[1:])
        todo, skipped = s1_select.layer_sae_plan(layer)
        # Waiting = an SAE that WILL become selectable (its s0 record or a trained pair partner's
        # record is still to come).  Held/untrained arms and D-43/D-18-excluded SAEs never block.
        waiting = [t for t, w in skipped.items() if w in ("no s0 record", "D-43 gate pending")]
        if waiting:
            return "blocked", f"s0/D-43 not complete for {len(waiting)} SAEs of L{layer}"
        if s1_select.is_done(layer):
            return "done", ""
        p = _prereg("s1")
        return ("blocked", f"PROTOCOL: {p}") if p else ("todo", "")
    if kind == "s2":
        if s2_cancel.is_done(unit):
            return "done", ""
        return ("todo", "") if _s0_done(unit) else ("blocked", "no s0 record")
    if kind == "s4a":
        if s4_pilot.gen_done():
            return "done", ""
        p = _prompts_ok() or _prereg("s4a")
        if p:
            return "blocked", f"PROTOCOL/prompts: {p}"
        af = base.read_json(base.s1_armfree(base.PILOT_LAYER).replace(".npz", ".json"))
        return ("todo", "") if af and af.get("complete") else ("blocked", "s1 L12 arm-free directions missing")
    if kind == "s4j":
        if s7_judge.unit_done("pilot"):
            return "done", ""
        return ("todo", "") if s4_pilot.gen_done() else ("blocked", "pilot generation (s4a) not done")
    if kind in ("s5", "s6", "s7j"):
        if ref is not None:
            if not _s0_done(unit):
                return "blocked", "no s0 record"
            g = _sae_gate(unit)
            ex = _sae_excluded(unit)
            if ex:
                return "excluded", ex
            if g == "pending":
                return "blocked", "D-43 gate pending (pair baseline records missing)"
        if kind == "s5":
            if calibrate.is_done(unit):
                return "done", ""
            if unit == f"armfree_L{base.PILOT_LAYER}":
                return "blocked", "owned by s4a (the pilot calibrates armfree_L12)"
            if ref is not None:
                if not s1_select.load_selection(unit):
                    return "blocked", "no s1 selection"
            else:
                layer = int(unit.split("L")[1])
                af = base.read_json(base.s1_armfree(layer).replace(".npz", ".json"))
                if not (af and af.get("complete")):
                    return "blocked", "s1 arm-free missing"
            p = _prereg("s5")
            return ("blocked", f"PROTOCOL: {p}") if p else ("todo", "")
        if kind == "s6":
            if s6_generate.unit_done(unit):
                return "done", ""
            ok, _ = _p1()
            if not ok:
                return "blocked", "P1 not decided (run s4a, s4j, then s4p)"
            p = _prompts_ok() or _prereg("s6")
            if p:
                return "blocked", f"PROTOCOL/prompts: {p}"
            if unit != "null" and not calibrate.is_done(unit):
                return "blocked", "s5 calibration missing"
            if not s7_metrics.clf_done():
                return "blocked", "s7c classifier not trained (s6 runs the merged metrics)"
            return "todo", ""
        if kind == "s7j":
            if s6_generate.generation_done(unit) and s7_judge.unit_done(unit):
                return "done", ""
            return ("todo", "") if s6_generate.generation_done(unit) else ("blocked", "s6 generation not done")
    if kind == "s7c":
        if s7_metrics.clf_done():
            return "done", ""
        p = _prereg("s7c")
        return ("blocked", f"PROTOCOL: {p}") if p else ("todo", "")
    if kind == "s8j":
        if os.path.isfile(base.p("s8", "second_judge.json")):
            return "done", ""
        sm = base.read_json(base.SECOND_JUDGE_ENV_RECORD)
        if not (sm and sm.get("passed")):
            return "blocked", "InternVL environment smoke not passed (optional stage)"
        return ("todo", "") if os.path.isfile(base.p("s8", "handlabel_sheet.csv")) else (
            "blocked", "hand-label sheet not exported (s8 --export)")
    if kind == "s9":
        # Many-to-one: every generation(+metrics) and judge unit that CAN exist must be done.
        # Units that never run (excluded by D-43/D-18) or are HELD do not hold the report back.
        missing = 0
        for k in ("s6", "s7j"):
            for u in base.gen_units():
                st, why = classify_unit(k, u)
                if st in ("done", "excluded") or why.startswith("held") or why.startswith("arm not trained"):
                    continue
                missing += 1
        if missing:
            return "blocked", f"{missing} generation/judge units not done (many-to-one: waits for all)"
        p = _prereg("s9")
        if p:
            return "blocked", f"PROTOCOL: {p}"
        from steering import s9_report
        d = base.read_json(base.p("s9", "_done.json"))
        return ("done", "") if d and d.get("inputs") == s9_report.report_inputs() else ("todo", "")
    raise KeyError(kind)


def classify(kind: str) -> Dict[str, List]:
    out = {"done": [], "todo": [], "blocked": [], "excluded": [], "reasons": Counter()}
    for i, u in enumerate(jobs.units(kind)):
        st, why = classify_unit(kind, u)
        out[st].append(i)
        if st in ("blocked", "excluded"):
            out["reasons"][why] += 1
    return out


def compress(idxs: List[int]) -> str:
    if not idxs:
        return ""
    items = sorted(idxs)
    parts, start, prev = [], items[0], items[0]
    for i in items[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = i
    parts.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(parts)


def manual_steps() -> List[str]:
    out = []
    p = _prompts_ok()
    out.append(f"s3  prompt file (P2): {'OK sha256 ' + str(base.PROMPTS_SHA256) if not p else p}")
    ok, prim = _p1()
    out.append(f"s4p P1 rule: {'decided, primary = ' + str(prim if prim is not None else 'none') if ok else 'NOT decided'}"
               f"  (python -m steering.s4_pilot --rule, after s4j)")
    out.append(f"s8x hand-label export: {'done' if os.path.isfile(base.p('s8', 'handlabel_sheet.csv')) else 'not yet'}"
               f"  (python -m steering.s8_handlabel --export, after the headline s7j units)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--stage", choices=STAGE_ORDER)
    ap.add_argument("--why", choices=STAGE_ORDER)
    a = ap.parse_args()
    if a.plan:
        for k in STAGE_ORDER:
            c = classify(k)
            print("|".join([k, compress(c["todo"]), str(len(c["todo"])), str(len(c["done"])),
                            str(len(jobs.units(k))), str(len(c["blocked"])), str(len(c["excluded"]))]))
        return 0
    if a.stage:
        print(compress(classify(a.stage)["todo"]))
        return 0
    if a.why:
        for why, n in classify(a.why)["reasons"].most_common():
            print(f"{n:4d}  {why}")
        return 0
    print(f"A16 root: {base.ROOT}{'  (SMOKE)' if base.IS_SMOKE else ''}")
    for k in STAGE_ORDER:
        c = classify(k)
        tail = f"  todo: {compress(c['todo'])}" if c["todo"] else ""
        print(f"{k:4s} done {len(c['done']):3d}/{len(jobs.units(k)):<3d} todo {len(c['todo']):3d} "
              f"blocked {len(c['blocked']):3d} excluded {len(c['excluded']):3d}{tail}")
        for why, n in c["reasons"].most_common(3):
            print(f"         {n:3d} x {why}")
    print("MANUAL (between waves):")
    for m in manual_steps():
        print("  " + m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
