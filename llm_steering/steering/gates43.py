"""D-43 Track B health gate verdicts (PROTOCOL 1.5 row "D-43 Track B health", rule R3), computed from
the s0 records.  Pure functions of the on-disk records, so the planner, every worker and the
report reach the same verdict.

Two gate versions exist (PROTOCOL 1.5): rule (b) is a reported diagnostic in the PRIMARY gate
and a removal rule only in the STRICT gate.
  primary  removal by (a), (c) and (D-44) only; (b) is computed and printed on every record but
           never removes an SAE.  This is the gate the cascade RUNS under (selection,
           calibration, generation, judging) and whose verdict words are the report's headline.
  strict   the gate exactly as pre-registered: (a), (b), (c) and (D-44) all remove.  Reported
           beside every steering result as a sensitivity analysis (s9_report), never used to
           decide what runs.  ``--strict`` gives these verdicts.
Why it is safe to run only the primary gate: the strict failure set is a SUPERSET of the primary
one (strict = primary + one more removal rule; the relative-CE check (c) reads the baseline's
CE, not its gate status, so pending states are identical), hence every SAE the strict analysis
keeps is also generated and judged under the primary gate.  _selftest() checks this invariant.

Per SAE (arm x seed at one layer), all at positions >= 1 (D-43):
  (a)   top-k arms (topk, batch_topk; baseline, dual and dual_g0): final val Active >= 57.6
        (0.9 k) in train_stdout.log.
  (b)   final val reconstruction loss <= 1.5 x the run's own minimum over epochs.
        primary: DIAGNOSTIC (printed, stored in every gate record, never removes);
        strict:  removal gate (as pre-registered).
  (c)   CE recovered (primary: splice at positions >= 1, inference encoder) >= 0.80, and for
        dual / dual_g0 arms also >= (pair baseline - 0.10).  Declared reading of "pair
        baseline": the SEED MEAN of the same layer's baseline arm (D-01: seeds are the only
        axis averaged); the seed-matched baseline value is printed beside it.
        Baseline seeds are of three kinds: measured (s0 record), PENDING (the arm is trained and
        runnable but its s0 record is not written yet -> the Dual's check (c) is 'pending' and
        waits), and UNTRAINED (the arm is not trained or is held, e.g. the Track B JumpReLU cells
        after the failed D-15 pilot -> it can never be measured under the current state, so it
        is left out: the relative check uses the measured seeds and records how many; with no
        measured seed at all the check is 'not assessable' and the SAE keeps descriptive rows).
        Why: a pending that can never resolve would block the layer's s1 selection -- and with
        it the pilot, P1 and every generation unit -- forever.
  (D-44) BatchTopK arms: inference L0 on non-sink test rows >= 57.6.
Per (layer, activation pair): 'pass' iff all 9 SAEs pass; 'collapsed' as soon as any evaluated SAE
fails; otherwise 'pending'.  Label: "baseline collapsed" when the pair is TopK and any tkuntied
baseline seed fails (D-18 "a layer that fails is reported as 'baseline collapsed'";
PROTOCOL 1.5), else "collapsed".  "Fails" always means fails THIS VERSION's removal rules: a
tkuntied seed failing only (b) collapses its layer (and triggers D-18) under the strict gate but
not under the primary gate.  A collapsed pair leaves the confirmatory family; the FAILING SAEs
are not calibrated, generated or judged (their rows read "collapsed"); the passing SAEs of a
collapsed pair still run and get descriptive per-seed rows -- EXCEPT under D-18: when a TopK
baseline seed fails at a layer, "its TopK pair is not steered", so all 9 TopK SAEs of that layer
are excluded from selection, calibration, generation and judging (sae_excluded(), primary gate).
JumpReLU pairs are additionally subject to D-15 (common/gates.py); held pairs are 'pending'.

CLI
  python -m steering.gates43                # primary verdicts + (b) diagnostics
  python -m steering.gates43 --strict       # strict pre-registered verdicts (sensitivity)
  python -m steering.gates43 --compare      # one row per pair: primary vs strict
  python -m steering.gates43 --write        # + per-SAE gate JSONs (both versions)
  python -m steering.gates43 --selftest     # synthetic records: both gate versions + D-18 + superset
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from steering import base
from common import arms

TOPK_FAMILIES = ("topk", "batch_topk")

# Gate versions.  PRIMARY is the default everywhere (planner, workers, s8 headline pool, the s9
# headline); STRICT is only ever requested explicitly (CLI --strict, the s9 sensitivity tables).
PRIMARY, STRICT = "primary", "strict"
GATE_VERSIONS = (PRIMARY, STRICT)
# Rule identifiers.  They are stamped into every gate JSON and into the s9 completion record
# (s9_report.report_inputs), so an output produced under a different rule is recognisably stale
# and the planner re-runs the report instead of trusting it.
RULE_ID = {
    PRIMARY: "D-43 primary gate: removal by (a), (c), D-44; (b) is a reported diagnostic",
    STRICT: "D-43 as pre-registered: removal by (a), (b), (c), D-44 (sensitivity analysis)",
}
# The exact wording every strict-gate table of the report carries.
STRICT_LABEL = "sensitivity: strict pre-registered D-43 gate incl. rule (b)"
B_DIAG_SUFFIX = " -- reported diagnostic, not a removal gate"


def _is_strict(strict: bool) -> str:
    return STRICT if strict else PRIMARY


def sae_checks(rec: Dict[str, Any], baseline_ce: Optional[Dict[int, Any]] = None,
               strict: bool = False) -> Dict[str, Any]:
    """Every D-43 check of one s0 record.  ``baseline_ce``: {seed: primary CE | None (pending) |
    'untrained'} of the pair's baseline arm, needed for Dual/g0 arms only (see the module
    docstring for the three kinds).

    ``strict=False`` (the primary gate): rule (b) is evaluated and stored under
    ``b_loss_ratio`` / ``diagnostics`` but never enters ``checks`` / ``reasons``, so it cannot
    change ``status``.  ``strict=True`` reproduces the pre-registered gate exactly: (b) is a
    check and its failure a reason, in the original order (a), (b), (c), (c rel), (D-44), so the
    reason strings follow the pre-registered rule order."""
    out: Dict[str, Any] = {"checks": {}, "reasons": [], "pending": [], "notes": [],
                           "diagnostics": [], "gate": _is_strict(strict)}
    tl = rec["train_log"]
    if rec["activation"] in TOPK_FAMILIES:
        act = tl.get("final_active")
        ok = act is not None and act >= base.GATE_ACTIVE_MIN
        out["checks"]["a_active"] = ok
        if not ok:
            out["reasons"].append(f"(a) final val Active {act} < {base.GATE_ACTIVE_MIN}")
    # (b) is ALWAYS computed (it is reported on every record in both versions); only its role
    # differs.  recon_ratio can be inf when the logged minimum is 0 (s0_rescore guards the
    # division); inf > 1.5 is a (b) failure, which is the conservative reading.
    ok_b = tl["recon_ratio"] <= base.GATE_LOSS_RATIO
    b_text = (f"(b) final {tl['recon_key']} {tl['final_recon']:.3e} > 1.5 x min "
              f"{tl['min_recon']:.3e} (ratio {tl['recon_ratio']:.2f})")
    out["b_loss_ratio"] = {"ok": bool(ok_b), "ratio": float(tl["recon_ratio"]),
                           "final": tl["final_recon"], "min": tl["min_recon"],
                           "min_epoch": tl.get("min_recon_epoch"), "key": tl["recon_key"],
                           "threshold": base.GATE_LOSS_RATIO,
                           "role": "removal gate" if strict else "diagnostic"}
    if strict:
        out["checks"]["b_loss_ratio"] = ok_b
        if not ok_b:
            out["reasons"].append(b_text)
    elif not ok_b:
        # Primary gate: the failure is surfaced (printed by the CLI, stored in the gate JSON,
        # shown in the s9 gate table) but it is not a reason, so it cannot remove the SAE.
        out["diagnostics"].append(b_text + B_DIAG_SUFFIX)
    ce = rec["ce"]["frac_recovered_primary"]
    ok = ce >= base.GATE_CE_MIN
    out["checks"]["c_ce_abs"] = ok
    if not ok:
        out["reasons"].append(f"(c) CE recovered {ce:.4f} < {base.GATE_CE_MIN}")
    if rec["arm"] in ("dual", "dual_g0"):
        bce = baseline_ce or {}
        measured = [float(v) for v in bce.values() if isinstance(v, (int, float))]
        pending = [s for s, v in bce.items() if v is None]
        untrained = sorted(s for s, v in bce.items() if v == "untrained")
        if pending or not bce:
            # A trained baseline seed has no record yet: wait for it (s0 of that seed is todo).
            out["checks"]["c_ce_rel"] = None
            out["pending"].append("(c) pair baseline CE not measured for every trained seed")
        elif not measured:
            out["checks"]["c_ce_rel"] = None
            out["notes"].append("(c) relative check not assessable: no trained pair-baseline seed "
                                f"(untrained seeds {untrained}); descriptive rows only")
        else:
            ref = float(np.mean(measured)) - base.GATE_CE_DUAL_MARGIN
            ok = ce >= ref
            out["checks"]["c_ce_rel"] = ok
            out["baseline_ce_seedmean"] = float(np.mean(measured))
            out["baseline_ce_n_seeds"] = len(measured)
            if untrained:
                out["notes"].append(f"(c) pair baseline = mean of {len(measured)} trained seed(s); "
                                    f"seeds {untrained} untrained/held")
            if not ok:
                out["reasons"].append(f"(c) CE {ce:.4f} < pair baseline seed-mean "
                                      f"{np.mean(measured):.4f} - 0.10")
    if rec["activation"] == "batch_topk":
        l0 = rec["d44"]["inference_nonsink_l0"]
        ok = l0 >= base.GATE_ACTIVE_MIN
        out["checks"]["d44_l0"] = ok
        if not ok:
            out["reasons"].append(f"(D-44) inference non-sink L0 {l0:.1f} < {base.GATE_ACTIVE_MIN}")
    out["status"] = "fail" if out["reasons"] else ("pending" if out["pending"] else "pass")
    return out


def _untrained(ref) -> bool:
    """True when this arm cannot get an s0 record under the current state: not trained, or held
    (base.held_reason covers both: the JumpReLU hold and arms whose status is not 'ready')."""
    return base.held_reason(ref) is not None


# One entry per SAE of a pair: (arm, seed, tag, complete s0 record | None, untrained?).  Keeping
# the disk reads (_entries) apart from the verdict logic (verdict_from_entries) lets the selftest
# drive the exact production logic with synthetic records.
Entry = Tuple[str, int, Optional[str], Optional[Dict[str, Any]], bool]


def _entries(layer: int, activation: str) -> List[Entry]:
    out: List[Entry] = []
    for arm in arms.ARMS:
        for s in arms.SEEDS:
            ref = arms.resolve_arm("B", layer, None, activation, arm, s)
            rec = base.read_json(base.s0_record(ref.tag)) if ref.tag else None
            rec = rec if (rec and rec.get("complete")) else None
            # 'untrained' only matters for a baseline seed WITHOUT a record (the relative-CE
            # check skips it instead of waiting for it); computed for every entry for simplicity.
            out.append((arm, s, ref.tag, rec, _untrained(ref) if rec is None else False))
    return out


def verdict_from_entries(layer: int, activation: str, entries: List[Entry],
                         strict: bool = False) -> Dict[str, Any]:
    """Gate state of one (layer, activation) pair and of each of its 9 SAEs under one gate
    version (pure: no disk access)."""
    base_ce: Dict[int, Any] = {}
    for arm, s, _tag, rec, untrained in entries:
        if arm != "baseline":
            continue
        if rec:
            base_ce[s] = rec["ce"]["frac_recovered_primary"]
        else:
            base_ce[s] = "untrained" if untrained else None
    per: Dict[str, Dict[str, Any]] = {}
    for arm, s, tag, rec, _untr in entries:
        if rec is None:
            per[tag] = {"status": "pending", "reasons": [], "pending": ["no s0 record"],
                        "notes": [], "diagnostics": [], "arm": arm, "seed": s}
            continue
        v = sae_checks(rec, base_ce if arm != "baseline" else None, strict=strict)
        v.update(arm=arm, seed=s)
        per[tag] = v
    fails = [t for t, v in per.items() if v["status"] == "fail"]
    # D-18 reads the failures of THIS version: under the primary gate a tkuntied seed failing only
    # (b) is not a failure, so it neither labels the layer 'baseline collapsed' nor stops the
    # layer's TopK pair from being steered; under the strict gate it does both.
    base_failed = [t for t in fails if per[t]["arm"] == "baseline"]
    if fails:
        state = "collapsed"
        label = "baseline collapsed" if (activation == "topk" and base_failed) else "collapsed"
    elif all(v["status"] == "pass" for v in per.values()):
        state, label = "pass", None
    else:
        state, label = "pending", None
    # D-18: a failing tkuntied seed means the layer's TopK pair is not steered at all.
    d18 = bool(activation == "topk" and base_failed)
    # Tags whose (b) diagnostic fails, whatever the version (in the strict version they are also
    # among 'failing'); the report and the CLI print them beside the verdict.
    b_flagged = [t for t, v in per.items() if v.get("b_loss_ratio") and not v["b_loss_ratio"]["ok"]]
    gate = _is_strict(strict)
    return {"layer": layer, "activation": activation, "state": state, "label": label,
            "failing": fails, "sae": per, "d18_not_steered": d18, "b_flagged": b_flagged,
            "gate": gate, "rule": RULE_ID[gate]}


def pair_verdict(layer: int, activation: str, strict: bool = False) -> Dict[str, Any]:
    """Gate state of one (layer, activation) pair and of each of its 9 SAEs.  Primary gate unless
    ``strict`` (the pre-registered gate, for the sensitivity analysis only)."""
    return verdict_from_entries(layer, activation, _entries(layer, activation), strict=strict)


def _ref_of(tag: str):
    for ref in base.sae_refs():
        if ref.tag == tag:
            return ref
    raise KeyError(tag)


def sae_status(tag: str, strict: bool = False) -> str:
    """'pass' | 'fail' | 'pending' for one SAE tag (its own checks + the relative CE check)."""
    ref = _ref_of(tag)
    return pair_verdict(int(ref.key[1:]), ref.activation, strict=strict)["sae"][tag]["status"]


def excluded_reason(verdict: Dict[str, Any], tag: str) -> Optional[str]:
    """Why an SAE must NOT be selected, calibrated, generated or judged (None = it may run),
    given its pair's verdict: its own D-43 failure, or D-18 (TopK baseline collapsed at the
    layer -> the whole TopK pair is not steered).  Works for either version's verdict; the
    cascade only ever passes PRIMARY verdicts (see the module docstring)."""
    if verdict["sae"][tag]["status"] == "fail":
        return "D-43 gate failed (collapsed): not calibrated, generated or judged"
    if verdict.get("d18_not_steered"):
        return (f"D-18: TopK baseline collapsed at L{verdict['layer']}; its TopK pair is not "
                f"steered")
    return None


def sae_excluded(tag: str) -> Optional[str]:
    """Exclusion under the PRIMARY gate -- the gate the cascade runs under.  Deliberately has no
    strict switch: the strict gate is a reporting sensitivity analysis and must never stop a unit
    (everything it keeps is a subset of what the primary gate runs)."""
    ref = _ref_of(tag)
    return excluded_reason(pair_verdict(int(ref.key[1:]), ref.activation), tag)


def all_verdicts(strict: bool = False) -> Dict[Tuple[int, str], Dict[str, Any]]:
    return {(l, a): pair_verdict(l, a, strict=strict) for l in base.LAYERS for a in arms.ACTIVATIONS}


def verdict_digest() -> Dict[str, List[Any]]:
    """{'L<layer>_<act>': [primary state, primary label, strict state, strict label]} -- the
    compact fingerprint s9_report.report_inputs stores, so a report built under other verdicts
    (a different rule, or records that changed a pair's state) is re-run by the planner."""
    p, s = all_verdicts(strict=False), all_verdicts(strict=True)
    return {f"L{l}_{a}": [p[(l, a)]["state"], p[(l, a)]["label"], s[(l, a)]["state"],
                          s[(l, a)]["label"]] for (l, a) in p}


def gate_json_path(tag: str) -> str:
    return base.p("s0", "gate", f"{tag}.json")


def write_all() -> Dict[str, int]:
    """One gate JSON per SAE with an s0 record (its checks, status and the pair's state/label),
    plus the combined gates/d43.json (primary) and gates/d43_strict.json (sensitivity).
    Idempotent; called by s1 at the start of every layer unit and runnable standalone (seconds).
    A pending verdict is written as 'pending' and rewritten once the pair's records are
    complete.  Every file is rewritten atomically on every call and carries the rule id, so a
    JSON written under another rule is replaced, never read as current.  The top-level keys are
    the PRIMARY gate; the strict verdict sits under 'strict'."""
    n = {"pass": 0, "fail": 0, "pending": 0}
    allv = all_verdicts(strict=False)
    alls = all_verdicts(strict=True)
    for (layer, act), v in allv.items():
        vs = alls[(layer, act)]
        for tag, sv in v["sae"].items():
            if sv.get("pending") == ["no s0 record"]:
                continue
            ss = vs["sae"][tag]
            base.atomic_write_json(gate_json_path(tag), {
                "tag": tag, "layer": layer, "activation": act, "arm": sv["arm"], "seed": sv["seed"],
                "gate": PRIMARY, "gate_rule": RULE_ID[PRIMARY],
                "status": sv["status"], "checks": sv.get("checks", {}), "reasons": sv.get("reasons", []),
                "pending": sv.get("pending", []), "notes": sv.get("notes", []),
                "diagnostics": sv.get("diagnostics", []), "b_loss_ratio": sv.get("b_loss_ratio"),
                "baseline_ce_seedmean": sv.get("baseline_ce_seedmean"),
                "baseline_ce_n_seeds": sv.get("baseline_ce_n_seeds"),
                "excluded": excluded_reason(v, tag),
                "pair_state": v["state"], "pair_label": v["label"],
                "d18_not_steered": v.get("d18_not_steered", False),
                "strict": {"gate_rule": RULE_ID[STRICT], "label": STRICT_LABEL,
                           "status": ss["status"], "checks": ss.get("checks", {}),
                           "reasons": ss.get("reasons", []),
                           "excluded_if_strict": excluded_reason(vs, tag),
                           "pair_state": vs["state"], "pair_label": vs["label"],
                           "d18_not_steered": vs.get("d18_not_steered", False)}})
            n[sv["status"]] += 1
    base.atomic_write_json(base.p("gates", "d43.json"), {f"L{l}_{a}": r for (l, a), r in allv.items()})
    base.atomic_write_json(base.p("gates", "d43_strict.json"),
                           {f"L{l}_{a}": r for (l, a), r in alls.items()})
    return n


# ----------------------------------------------------------------------------- CLI printing
def _b_ratio_line(r: Dict[str, Any]) -> Optional[str]:
    """One line with the (b) loss/min ratio of every SAE of the pair (arm-major, seeds in
    arms.SEEDS order), '-' for SAEs without a record.  Printed in the primary mode only (it is
    the diagnostic that replaced the removal rule); None when the pair has no record at all."""
    by_arm: Dict[str, List[str]] = {a: [] for a in arms.ARMS}
    any_rec = False
    for sv in r["sae"].values():
        b = sv.get("b_loss_ratio")
        any_rec = any_rec or b is not None
        by_arm[sv["arm"]].append((sv["seed"], "-" if b is None else
                                  f"{b['ratio']:.2f}{'' if b['ok'] else '*'}"))
    if not any_rec:
        return None
    parts = [f"{a} " + " ".join(x for _, x in sorted(by_arm[a])) for a in arms.ARMS]
    return (f"        (b) diagnostic, final/min val loss per seed {list(arms.SEEDS)} "
            f"(* = > {base.GATE_LOSS_RATIO}): " + " | ".join(parts))


def print_verdicts(v: Dict[Tuple[int, str], Dict[str, Any]], strict: bool) -> None:
    """The verdict table.  In strict mode it lists the pre-registered (strict) verdicts; the
    primary mode adds the (b) diagnostic lines."""
    print(f"{'layer':5s} {'pair':10s} {'state':10s} {'label':18s} failing")
    for (l, act), r in v.items():
        print(f"L{l:<4d} {act:10s} {r['state']:10s} {str(r['label'] or '-'):18s} "
              f"{', '.join(r['failing']) or '-'}")
        for tag in r["failing"]:
            print(f"        {tag}: {'; '.join(r['sae'][tag]['reasons'])}")
        if not strict:
            line = _b_ratio_line(r)
            if line:
                print(line)
            for tag, sv in r["sae"].items():
                for d in sv.get("diagnostics", []):
                    print(f"        diag {tag}: {d}")
        for tag, sv in r["sae"].items():
            for note in sv.get("notes", []):
                print(f"        note {tag}: {note}")
        if r.get("d18_not_steered"):
            print(f"        D-18: TopK baseline collapsed at L{l}; all 9 TopK SAEs of the layer are not steered")


def print_compare() -> None:
    """One row per pair: the primary verdict (headline) beside the strict one (sensitivity)."""
    p, s = all_verdicts(strict=False), all_verdicts(strict=True)

    def cell(r):
        return r["state"] + (f" ({r['label']})" if r["label"] else "") + (" D-18" if r["d18_not_steered"] else "")
    print(f"{'layer':5s} {'pair':10s} {'primary (headline)':30s} {'strict (sensitivity)':30s} "
          f"(b)-flagged SAEs")
    for key in p:
        l, act = key
        print(f"L{l:<4d} {act:10s} {cell(p[key]):30s} {cell(s[key]):30s} "
              f"{len(p[key]['b_flagged'])}")
    print(f"strict = {STRICT_LABEL}; (b)-flagged = SAEs whose final/min val loss exceeds "
          f"{base.GATE_LOSS_RATIO} (a removal reason under strict, a diagnostic under primary)")


# ----------------------------------------------------------------------------- selftest
def _fake_rec(arm: str, act: str, *, active=64.0, ratio=1.0, ce=0.95, l0=64.0) -> Dict[str, Any]:
    """A minimal complete s0 record carrying exactly the fields sae_checks reads."""
    return {"complete": True, "arm": arm, "activation": act,
            "train_log": {"final_active": active, "recon_ratio": ratio, "recon_key": "Val L2",
                          "final_recon": 1e-6 * ratio, "min_recon": 1e-6, "min_recon_epoch": 10},
            "ce": {"frac_recovered_primary": ce}, "d44": {"inference_nonsink_l0": l0}}


def _fake_entries(act: str, tweak: Optional[Tuple[str, int, Dict[str, Any]]] = None,
                  missing: Optional[Tuple[str, int]] = None) -> List[Entry]:
    out: List[Entry] = []
    for arm in arms.ARMS:
        for s in arms.SEEDS:
            kw = tweak[2] if (tweak and (arm, s) == tweak[:2]) else {}
            rec = None if missing == (arm, s) else _fake_rec(arm, act, **kw)
            out.append((arm, s, f"{act}_{arm}_seed{s}", rec, False))
    return out


def _selftest() -> int:
    """Synthetic pairs through the production verdict logic (verdict_from_entries), both gates:
    (b)-only failures of a Dual, a g0 and a TopK/BatchTopK baseline seed; an (a) failure of a
    TopK baseline; a pending baseline record; the strict-superset invariant; the diagnostics."""
    bad: List[str] = []
    B = base.GATE_LOSS_RATIO + 0.5                       # a (b) failure, every other rule fine
    cases = {
        # name: (activation, tweak, missing, expected primary (state, label, d18),
        #                                        expected strict (state, label, d18))
        "all pass": ("topk", None, None, ("pass", None, False), ("pass", None, False)),
        "dual (b) only": ("topk", ("dual", 42, {"ratio": B}), None,
                          ("pass", None, False), ("collapsed", "collapsed", False)),
        "g0 (b) only, BatchTopK": ("batch_topk", ("dual_g0", 44, {"ratio": B}), None,
                                   ("pass", None, False), ("collapsed", "collapsed", False)),
        "tkuntied (b) only": ("topk", ("baseline", 43, {"ratio": B}), None,
                              ("pass", None, False), ("collapsed", "baseline collapsed", True)),
        "BatchTopK baseline (b) only": ("batch_topk", ("baseline", 43, {"ratio": B}), None,
                                        ("pass", None, False), ("collapsed", "collapsed", False)),
        "tkuntied (a) fails": ("topk", ("baseline", 42, {"active": 50.0}), None,
                               ("collapsed", "baseline collapsed", True),
                               ("collapsed", "baseline collapsed", True)),
        "dual D-44 fails": ("batch_topk", ("dual", 43, {"l0": 40.0}), None,
                            ("collapsed", "collapsed", False), ("collapsed", "collapsed", False)),
        "baseline record pending": ("topk", None, ("baseline", 44),
                                    ("pending", None, False), ("pending", None, False)),
    }
    for name, (act, tweak, missing, want_p, want_s) in cases.items():
        ent = _fake_entries(act, tweak, missing)
        vp = verdict_from_entries(6, act, ent, strict=False)
        vs = verdict_from_entries(6, act, ent, strict=True)
        for gate, v, want in ((PRIMARY, vp, want_p), (STRICT, vs, want_s)):
            got = (v["state"], v["label"], v["d18_not_steered"])
            if got != want:
                bad.append(f"{name} [{gate}]: {got} != {want}")
            if v["gate"] != gate or v["rule"] != RULE_ID[gate]:
                bad.append(f"{name} [{gate}]: gate/rule not stamped")
        # Superset invariant: strict failures include every primary failure, and whatever the
        # strict gate would let run the primary gate lets run too.
        if not set(vp["failing"]) <= set(vs["failing"]):
            bad.append(f"{name}: primary failures not a subset of strict failures")
        for tag in vp["sae"]:
            if excluded_reason(vs, tag) is None and excluded_reason(vp, tag) is not None:
                bad.append(f"{name}: {tag} runs under strict but is excluded under primary")
        # (b) is reported in both versions; a removal reason only in the strict one.
        if tweak and "ratio" in tweak[2]:
            tag = f"{act}_{tweak[0]}_seed{tweak[1]}"
            sp, ss = vp["sae"][tag], vs["sae"][tag]
            if not (sp["diagnostics"] and sp["diagnostics"][0].endswith(B_DIAG_SUFFIX)
                    and not sp["reasons"] and "b_loss_ratio" not in sp["checks"]):
                bad.append(f"{name}: primary did not demote (b) to a diagnostic: {sp}")
            if not (ss["reasons"] and ss["reasons"][0].startswith("(b)") and not ss["diagnostics"]
                    and ss["checks"].get("b_loss_ratio") is False):
                bad.append(f"{name}: strict did not keep (b) as a removal rule: {ss}")
            if vp["b_flagged"] != [tag] or vs["b_flagged"] != [tag]:
                bad.append(f"{name}: b_flagged {vp['b_flagged']} / {vs['b_flagged']} != [{tag}]")
            # D-18 exclusion follows the version: strict excludes all 9 TopK SAEs, primary none.
            if tweak[0] == "baseline" and act == "topk":
                if any(excluded_reason(vp, t) for t in vp["sae"]):
                    bad.append(f"{name}: primary excluded an SAE for a (b)-only tkuntied seed")
                if not all(excluded_reason(vs, t) for t in vs["sae"]):
                    bad.append(f"{name}: strict D-18 did not exclude the whole TopK pair")
    for b in bad:
        print("  FAIL", b)
    print(f"GATES43 SELFTEST {'OK' if not bad else 'FAILED'} ({len(cases)} synthetic pairs x 2 gates)")
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true",
                    help="the pre-registered gate incl. rule (b) (sensitivity analysis)")
    ap.add_argument("--compare", action="store_true", help="primary vs strict, one row per pair")
    ap.add_argument("--write", action="store_true",
                    help="also write s0/gate/<tag>.json per SAE (both versions) and "
                         "gates/d43.json + gates/d43_strict.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    # The version banner goes to stderr so that --strict's stdout lists only the verdicts.
    gate = _is_strict(a.strict)
    print(f"[gates43] gate: {gate} -- {RULE_ID[gate]}", file=sys.stderr)
    if a.compare:
        print_compare()
    else:
        print_verdicts(all_verdicts(strict=a.strict), strict=a.strict)
    if a.write:
        print("wrote per-SAE gate JSONs:", write_all())
    return 0


if __name__ == "__main__":
    sys.exit(main())
