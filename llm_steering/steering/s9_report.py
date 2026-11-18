"""s9: the A16 report (D-01, D-03, D-04, D-43..D-50; PROTOCOL section 4, rules R1-R7).

Aggregation (D-01: seeds are the ONLY axis averaged; everything else stays a row/column):
  item    one continuation (prompt x continuation) of one selected latent.
  latent  metric averaged over its items (the per-latent score, PROTOCOL 4.3).
  seed    per (SAE, mode, dirset, budget, concept): the MEAN over the concept's REACHABLE selected
          latents (never best-of-3; R5: an unreachable latent contributes nothing, the count is
          printed; no reachable latent -> the seed is missing for that concept).
  arm     mean +- sample sd (ddof=1) over seeds; Welch per contrast over seeds (stats.welch).
Metrics: hm (official AxBench harmonic mean; co-primary), p_concept (bge classifier; co-primary),
concept_score / instruction / fluency ratings, hm_nofail (items with any parse failure dropped: the
D-50 sensitivity row), ppl, distinct2, kl_cont.
Confirmatory family (PROTOCOL 4.1): mode (i), 'full' selection, the P1 budget, metrics
{hm, p_concept}, layers 6/12/18 x 10 concepts x pairs {topk, batch_topk, jump_relu} x contrasts
{Dual-baseline, Dual-g0}.  Removed before BH (PROTOCOL 1.5, R3, R5): a (layer, pair) failing D-43
("collapsed" / "baseline collapsed"), a JumpReLU contrast failing D-15 ("not sparsity-matched"),
contrasts using an outlier-flagged arm (stats.flag_outliers, sd_ref = pooled within-arm seed sd
over every cell of that metric at that layer; "not assessable"), and tests with < 2 seeds in an
arm (R5, "not assessable").  The FULL PROTOCOL 4.1 grid (3 layers x 10 concepts x 2 metrics x 3
pairs x 2 contrasts = 360) is always listed: a family cell without data carries a label and is
not a test -- Track B JumpReLU "not sparsity-matched (D-15 pilot failed; cells held)" while its
cells are held/untrained (PROTOCOL 1.5 step 1), a collapsed pair its D-43 label, anything else
"pending (no data)".  So "N tests listed" is the family size and "tested" what BH actually saw.
BH over the remaining tests; verdict words from stats.verdict_from with the PROTOCOL SESOIs; win
counts print n_tests and 0.05 x n_tests.
Attribution (PROTOCOL 1.6, verbatim table) per (layer, concept, metric, pair).
Everything else is exploratory: delta, 95% CI and raw p only.  The live-matched row carries
judge-free metrics only (D-46; its shards are never judged).
Labels (D-47, PROTOCOL 4.6): mode (iii) tables of a (layer, pair) with a bypass-regime Dual read
"bypass diagnostic (not a property of DualSAE)"; P1 = "none" prints the PROTOCOL 4.6 sentence.
Kappa (D-50, PROTOCOL 4.7): the InternVL second judge vs phi-4 (s8/second_judge.json) and, once
a filled s8/handlabel_sheet_filled.csv exists, human vs phi-4 / InternVL per rating kind.
Scale: every (arm, cell) lookup goes through one dict index built once per gate version (the
per-seed table has ~10^5 rows at production size; a linear scan per lookup was quadratic).
D-43 gate versions (PROTOCOL 1.5): rule (b) is a reported diagnostic in the primary gate, not a
removal gate.  Every confirmatory output is therefore produced under TWO gate versions:
  primary  gates43 primary gate ((a), (c), D-44 remove).  Its verdict words are the HEADLINE.
  strict   the pre-registered gate incl. (b) -- the sensitivity analysis.  Every strict table is
           labelled gates43.STRICT_LABEL ('sensitivity: strict pre-registered D-43 gate incl.
           rule (b)') and sits directly under its primary twin.
Each version is computed from the per-seed rows of the SAEs THAT version keeps (gate_filter,
gated_versions): what that gate would have reported had the cascade run under it.  The cascade
runs under the primary gate, so on a consistent root the primary filter drops nothing; the strict
filter drops the rows of the SAEs only the strict gate removes (a (b)-only failure, or -- D-18 --
all 9 TopK SAEs of a layer whose tkuntied seed fails only (b)).  The pre-registered protocol never
generates those SAEs, so their seeds must not reach the strict twin's arm columns (mean +- sd
over seeds), the Welch numbers printed beside its gate labels, or its outlier sd_ref -- which is
pooled over every pair of the layer, so a removed seed of one pair would otherwise move the
outlier flag of ANOTHER pair's tested contrast.  raw_tests (Welch + outlier flag) therefore runs
once on the primary rows and a second time on the strict rows ONLY when the strict filter drops
an SAE the primary one keeps; otherwise both versions share the one pass.  A strict-twin arm cell
that lost seeds names them and the reason (D-43 / D-18) instead of silently showing a smaller n.
BH runs within the family separately per version (each version removes its own gated tests
before BH, PROTOCOL 1.5, so m differs and so do the q-values).  Confirmatory outputs doubled: the
family count / win-count line, the attribution table, and the CONFIRMATORY (P1 budget) results
tables; the D-43 table shows both verdicts and the (b) ratio as a diagnostic column.
Exploratory tables are produced once, from the primary rows with the primary gate's labels.
Outputs: s9/per_seed.csv (every generated row, plus gate_removed_primary / gate_removed_strict:
the reason a version dropped the row, empty when kept), s9/tests.csv (primary),
s9/tests_strict.csv (sensitivity), s9/report.md (tables only).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from steering import base, concepts, gates43, s6_generate
from common import arms, stats

METRICS = ("hm", "p_concept", "concept_score", "instruction", "fluency", "hm_nofail", "ppl",
           "distinct2", "kl_cont")
PRIMARY = ("hm", "p_concept")
SESOI_KEY = {"hm": "harmonic_mean", "p_concept": "p_concept"}
JUDGE_METRICS = ("hm", "concept_score", "instruction", "fluency", "hm_nofail")
HELD_JR_LABEL = "not sparsity-matched (D-15 pilot failed; cells held)"
P1_NONE_SENTENCE = ("Steering headline infeasible at budgets {0.05, 0.1, 0.2, 0.4, 0.8, 1.6} nats "
                    "(PROTOCOL 4.6): the confirmatory family is empty and every steering result "
                    "is exploratory.")
BYPASS_SUFFIX = " -- bypass diagnostic (not a property of DualSAE)"


# ----------------------------------------------------------------------------- item rows
def _merge_unit(unit: str) -> List[Dict[str, Any]]:
    """Judge + metric rows of one generation unit, merged per item."""
    out = []
    for gpath in sorted(glob.glob(os.path.join(base.s6_dir(unit), "*.jsonl"))):
        name = os.path.basename(gpath)
        jp = os.path.join(base.s7j_dir(unit), name)
        mp = os.path.join(base.s7m_dir(unit), name)
        jr = {(_k(r)): r for r in base.read_jsonl(jp)} if os.path.isfile(jp) else {}
        mr = {(_k(r)): r for r in base.read_jsonl(mp)} if os.path.isfile(mp) else {}
        for g in base.read_jsonl(gpath):
            k = _k(g)
            j, m = jr.get(k, {}), mr.get(k, {})
            cids = [c.cid for c in concepts.CONCEPTS] if g["mode"] == "null" else [g["concept"]]
            for cid in cids:
                row = {kk: g.get(kk) for kk in ("unit", "layer", "tag", "activation", "arm", "seed",
                                                  "mode", "dirset", "budget", "headline", "rank",
                                                  "latent", "prompt_idx", "cont", "status")}
                row["concept"] = cid
                v: Dict[str, float] = {}
                if g["status"] == "ok":
                    if j.get("concept_rating") is not None:
                        v["concept_score"] = j["concept_rating"].get(cid, np.nan)
                    if j.get("hm") is not None:
                        v["hm"] = j["hm"][cid]
                        v["instruction"] = j["instruction_rating"]
                        v["fluency"] = j["fluency_rating"]
                        st = j.get("judge_status", {})
                        keys = [f"concept:{cid}", "instruction", "fluency"]
                        v["hm_nofail"] = v["hm"] if all(st.get(x) == "ok" for x in keys) else np.nan
                    if m.get("p_concept") is not None:
                        v["p_concept"] = m["p_concept"].get(cid, np.nan)
                        for x in ("ppl", "distinct2", "kl_cont"):
                            v[x] = m.get(x, np.nan)
                row.update(v)
                out.append(row)
    return out


def _k(r: Dict[str, Any]) -> Tuple:
    return (r["mode"], r["dirset"], float(r["budget"]), r["concept"], r["rank"], r["prompt_idx"], r["cont"])


def _num(x) -> float:
    """float(x), or NaN for None / non-numeric values (a missing rating or metric)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _mean(v) -> float:
    a = np.asarray([_num(x) for x in v], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def per_seed_table() -> List[Dict[str, Any]]:
    """One row per (SAE or arm-free unit, mode, dirset, budget, concept, metric)."""
    rows_out: List[Dict[str, Any]] = []
    s0 = {}
    for u in base.gen_units():
        items = _merge_unit(u) if os.path.isdir(base.s6_dir(u)) else []
        if not items:
            continue
        lat: Dict[Tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        reach: Dict[Tuple, Dict[int, bool]] = defaultdict(dict)
        info: Dict[Tuple, Dict[str, Any]] = {}
        for it in items:
            key = (it["mode"], it["dirset"], float(it["budget"]), it["concept"])
            reach[key][it["rank"]] = reach[key].get(it["rank"], False) or it["status"] == "ok"
            info[key] = it
            if it["status"] != "ok":
                continue
            for mtr in METRICS:
                if mtr in it:
                    lat[key + (it["rank"],)][mtr].append(it[mtr])
        for key, rk in reach.items():
            it = info[key]
            ranks_ok = [r for r, ok in rk.items() if ok]
            for mtr in METRICS:
                if key[1] == "livematched" and mtr in JUDGE_METRICS:
                    continue          # never judged (D-46): no judge-metric rows at all
                vals = [_mean(lat[key + (r,)][mtr]) for r in ranks_ok if lat[key + (r,)].get(mtr)]
                vals = [x for x in vals if np.isfinite(x)]
                rows_out.append({"unit": u, "layer": it["layer"], "tag": it.get("tag"),
                                 "activation": it.get("activation"), "arm": it.get("arm"),
                                 "seed": it.get("seed"), "mode": key[0], "dirset": key[1],
                                 "budget": key[2], "concept": key[3], "metric": mtr,
                                 "value": float(np.mean(vals)) if vals else float("nan"),
                                 "n_latents_reachable": len(ranks_ok),
                                 "n_latents_unreachable": len(rk) - len(ranks_ok)})
    # D-04 annotations on every SAE row.
    for r in rows_out:
        if r["tag"]:
            rec = s0.get(r["tag"]) or base.read_json(base.s0_record(r["tag"]))
            s0[r["tag"]] = rec
            if rec:
                r.update({f"d04_{k}": v for k, v in rec["d04"].items()})
    return rows_out


# ----------------------------------------------------------------------------- tests
def _bkey(b) -> float:
    """Budget as a dict key (the same float whatever JSON/CSV round trip produced it)."""
    return round(float(b), 9)


Index = Dict[Tuple, Dict[Any, float]]


def index_ps(ps: List[Dict[str, Any]]) -> Index:
    """(layer, activation, arm, mode, dirset, budget, concept, metric) -> {seed: finite value},
    built ONCE; every lookup below is O(1)."""
    idx: Index = defaultdict(dict)
    for r in ps:
        if r["arm"] in arms.ARMS and np.isfinite(r["value"]):
            idx[(r["layer"], r["activation"], r["arm"], r["mode"], r["dirset"], _bkey(r["budget"]),
                 r["concept"], r["metric"])][r["seed"]] = r["value"]
    return idx


def _vals(idx: Index, layer, act, arm, mode, dirset, budget, concept, metric) -> Dict[Any, float]:
    return idx.get((layer, act, arm, mode, dirset, _bkey(budget), concept, metric), {})


# ----------------------------------------------------------------------------- gate filter
# (layer, activation, arm, seed): the SAE identity index_ps keys its seeds by.  Keying the filter
# the same way (rather than by tag) makes it apply to exactly the values the tables and Welch
# read, and lets the selftest drive it with tag-free synthetic rows.
SeedKey = Tuple[Any, str, str, Any]


def gate_removed_seeds(gates: Dict[Tuple[int, str], Dict[str, Any]]) -> Dict[SeedKey, str]:
    """{(layer, activation, arm, seed): 'D-43' | 'D-18'} of every SAE one gate version keeps out of
    selection, calibration, generation and judging.  The decision is gates43.excluded_reason (the
    single source of truth the cascade also uses: the SAE's own D-43 failure, or D-18 -- all 9
    TopK SAEs of a layer whose tkuntied seed fails); the short reason only labels table cells.
    A pending SAE is not removed (it may still run), exactly as in the cascade.  Pair verdicts
    without per-SAE entries (the selftest's pair-level gates) remove nothing."""
    out: Dict[SeedKey, str] = {}
    for (layer, act), v in gates.items():
        for tag, sv in (v.get("sae") or {}).items():
            if gates43.excluded_reason(v, tag):
                out[(layer, act, sv["arm"], sv["seed"])] = "D-43" if sv.get("status") == "fail" else "D-18"
    return out


def gate_filter(ps: List[Dict[str, Any]], gates: Dict[Tuple[int, str], Dict[str, Any]]
                ) -> Tuple[List[Dict[str, Any]], Dict[SeedKey, str]]:
    """The per-seed rows ONE gate version reports: every row except the SAE-arm rows of an SAE
    that version removes (arm-free / null reference rows are never gated).  Returns (kept rows,
    {seed key: reason} of the removed SAEs that actually HAD rows -- an SAE that was never
    generated is not 'dropped' from anything, so it is not listed).  Rows are neither copied nor
    modified: both versions may share row objects."""
    rm = gate_removed_seeds(gates)
    kept: List[Dict[str, Any]] = []
    hit: Dict[SeedKey, str] = {}
    for r in ps:
        k = (r["layer"], r["activation"], r["arm"], r["seed"])
        if r["arm"] in arms.ARMS and k in rm:
            hit[k] = rm[k]
        else:
            kept.append(r)
    return kept, hit


def family_cells(primary: Optional[float]) -> List[Tuple]:
    """The full PROTOCOL 4.1 grid as (layer, act, mode, dirset, budget, concept, metric) cells."""
    if primary is None:
        return []
    return [(layer, act, "i", "full", _bkey(primary), c.cid, metric)
            for layer in base.LAYERS for act in arms.ACTIVATIONS for c in concepts.CONCEPTS
            for metric in PRIMARY]


def build_tests(ps: List[Dict[str, Any]], primary: Optional[float],
                gates: Optional[Dict[Tuple[int, str], Dict[str, Any]]] = None,
                jr_status=None, idx: Optional[Index] = None, strict: bool = False
                ) -> List[Dict[str, Any]]:
    """Every contrast test under ONE D-43 gate version (primary unless ``strict``).  ``gates`` /
    ``jr_status`` are injectable for the selftest (defaults: the on-disk D-43 verdicts of that
    version and arms.contrast_ref for the D-15 state of JumpReLU pairs).  ``idx`` indexes ``ps``.
    main() goes through gated_versions() instead (one Welch pass shared by both versions when
    they keep the same rows); the selftest checks that both entry points agree."""
    gates = gates if gates is not None else gates43.all_verdicts(strict=strict)
    # Only the rows of the SAEs this version keeps (module docstring).  A caller's ``idx`` indexes
    # the UNFILTERED rows, so it is reused only when the filter dropped nothing.
    kept, removed = gate_filter(ps, gates)
    return apply_gates(raw_tests(kept, primary, jr_status=jr_status, idx=None if removed else idx),
                       gates, gates43.STRICT if strict else gates43.PRIMARY)


def raw_tests(ps: List[Dict[str, Any]], primary: Optional[float], jr_status=None,
              idx: Optional[Index] = None) -> List[Dict[str, Any]]:
    """Every contrast test of ``ps`` WITHOUT the D-43 label: family membership, Welch, the outlier
    flag and every non-D-43 label.  ``ps`` must already be ONE gate version's rows (gate_filter).
    Given the rows, the D-43 label is the only input that differs between the primary and the
    strict gate, so on rows both versions keep this runs once and apply_gates() finishes it per
    version (gated_versions).

    Label precedence is the pre-amendment chain (first non-None wins): D-15 / held JumpReLU ->
    D-43 -> D-17 -> no data -> outlier flag -> degenerate Welch.  The part BEFORE D-43 is stored
    in '_pre_gl', the first label AFTER it in '_post_gl'; apply_gates() rebuilds the chain as
    _pre_gl or <D-43 label of the version> or _post_gl, which equals the original chain."""
    idx = idx if idx is not None else index_ps(ps)
    if jr_status is None:
        def jr_status(layer, con):
            cr = arms.contrast_ref("B", layer, None, "jump_relu", con)
            return cr.status, cr.gate_label
    ok17 = arms.dual_g0_contrast_precondition("B")[0]
    tests = []
    cells = sorted({(r["layer"], r["activation"], r["mode"], r["dirset"], _bkey(r["budget"]),
                     r["concept"], r["metric"]) for r in ps if r["arm"] in arms.ARMS}
                   | set(family_cells(primary)))
    # Outlier flag (PROTOCOL 1.5): sd_ref(metric, layer) = pooled within-arm seed sd over every
    # (cell, arm) of that metric at that layer.  Declared reading of "all cells of that metric":
    # the cells of ONE (mode, selection, budget) -- concepts x pairs -- so the confirmatory
    # population (mode (i), full, P1) is never pooled with the noisier exploratory modes/budgets.
    flags: Dict[Tuple, List[int]] = {}
    groups: Dict[Tuple, List[Tuple]] = defaultdict(list)
    for c in cells:
        groups[(c[6], c[0], c[2], c[3], c[4])].append(c)
    for gkey, gcells in groups.items():
        pop = {}
        for c in gcells:
            for arm in arms.ARMS:
                v = _vals(idx, *c[:2], arm, *c[2:])
                if len(v) >= 2:
                    pop[(c, arm)] = v
        if pop:
            _, fl = stats.flag_outliers(pop)
            flags.update(fl)
    for c in cells:
        layer, act, mode, dirset, budget, concept, metric = c
        fam = (primary is not None and mode == "i" and dirset == "full"
               and abs(budget - primary) < 1e-9 and metric in PRIMARY)
        for con in arms.CONTRASTS:
            a = _vals(idx, layer, act, con[0], mode, dirset, budget, concept, metric)
            b = _vals(idx, layer, act, con[1], mode, dirset, budget, concept, metric)
            if not a and not b and not fam:
                continue
            pre = None
            if act == "jump_relu":
                # D-15 first: the JumpReLU pairs are held (base.held_reason); until every cell is
                # trained and D-15 is evaluated on records, the contrast is held.
                st, lab = jr_status(layer, con)
                if st == "gated":
                    pre = lab
                elif st != "testable":
                    pre = HELD_JR_LABEL
            # (the D-43 label goes here -- apply_gates, per gate version)
            post = None
            if con == ("dual", "dual_g0") and ok17 is not True:
                # D-17 code-drift gate (PROTOCOL 1.5 precondition of every Dual-g0 contrast)
                post = post or ("D-17 drift gate failed" if ok17 is False else "D-17 drift gate pending")
            if not a and not b:
                post = post or "pending (no data)"
            if (c, con[0]) in flags or (c, con[1]) in flags:
                post = post or "not assessable"
            w = stats.welch(list(a.values()), list(b.values())) if (a and b) else None
            if w is None or w["degenerate"]:
                post = post or "not assessable"
            tests.append({"layer": layer, "activation": act, "mode": mode, "dirset": dirset,
                          "budget": budget, "concept": concept, "metric": metric,
                          "contrast": f"{con[0]}-{con[1]}", "family": fam,
                          "_pre_gl": pre, "_post_gl": post,
                          "n_dual": len(a), "n_comp": len(b), **(w or {})})
    return tests


def d43_label(g: Dict[str, Any]) -> Optional[str]:
    """The D-43 label one pair verdict puts on its tests: the collapse label, 'gate pending', or
    None for a passing pair."""
    if g["state"] == "collapsed":
        return g["label"]
    if g["state"] == "pending":
        return "gate pending"
    return None


def apply_gates(raw: List[Dict[str, Any]], gates: Dict[Tuple[int, str], Dict[str, Any]],
                gate_version: str) -> List[Dict[str, Any]]:
    """raw_tests() + one D-43 gate version -> final tests: gate labels, BH over THIS version's
    remaining family tests (so the primary and the strict family are corrected separately, each
    with its own m), verdict words / exploratory rows.  ``raw`` is not modified (each test is a
    shallow copy without the private '_' keys), so it can be applied once per version."""
    tests = []
    for r in raw:
        t = {k: v for k, v in r.items() if not k.startswith("_")}
        t["gate_label"] = r["_pre_gl"] or d43_label(gates[(r["layer"], r["activation"])]) or r["_post_gl"]
        t["gate"] = gate_version
        tests.append(t)
    fam_tests = [t for t in tests if t["family"] and not t["gate_label"]]
    q = stats.bh([t["p_bh"] for t in fam_tests]) if fam_tests else []
    for t, qq in zip(fam_tests, q):
        t["q"] = float(qq)
    for t in tests:
        if t["family"]:
            comp = t["contrast"].split("-")[1]
            if t["gate_label"]:
                t["verdict"] = t["gate_label"]
            else:
                t["verdict"] = stats.verdict_from(t, t["q"], stats.SESOI[SESOI_KEY[t["metric"]]],
                                                  comparator=comp)
        else:
            t["verdict"] = None
            t["exploratory"] = stats.exploratory_row(t) if "delta" in t and np.isfinite(t.get("p", np.nan)) else None
    return tests


def gated_versions(ps: List[Dict[str, Any]], primary: Optional[float],
                   gp: Dict[Tuple[int, str], Dict[str, Any]],
                   gs: Dict[Tuple[int, str], Dict[str, Any]], jr_status=None
                   ) -> Dict[str, Dict[str, Any]]:
    """Both D-43 gate versions from one per-seed table (module docstring).  Per version
    (gates43.PRIMARY / gates43.STRICT): 'ps' the rows it keeps, 'idx' their index, 'removed'
    {seed key: 'D-43' | 'D-18'} of the SAEs whose generated rows it dropped, 'tests' its final
    tests (gate labels, BH within that version's family, verdicts).  main() and the selftest both
    go through here.

    raw_tests (Welch + outlier flag) runs once on the primary rows.  The strict version reuses
    that pass when it drops exactly the same SAEs (identical rows -> identical Welch and sd_ref),
    and otherwise gets its own pass on its own rows.  The removed-key SETS are compared, not the
    row counts, so the reuse stays exact even if the strict-superset invariant (gates43) broke."""
    ps_p, rm_p = gate_filter(ps, gp)
    ps_s, rm_s = gate_filter(ps, gs)
    idx_p = index_ps(ps_p)
    raw_p = raw_tests(ps_p, primary, jr_status=jr_status, idx=idx_p)
    if set(rm_s) == set(rm_p):
        idx_s, raw_s = idx_p, raw_p            # same rows: share the Welch pass (apply_gates copies)
    else:
        idx_s = index_ps(ps_s)
        raw_s = raw_tests(ps_s, primary, jr_status=jr_status, idx=idx_s)
    return {gates43.PRIMARY: {"ps": ps_p, "idx": idx_p, "removed": rm_p,
                              "tests": apply_gates(raw_p, gp, gates43.PRIMARY)},
            gates43.STRICT: {"ps": ps_s, "idx": idx_s, "removed": rm_s,
                             "tests": apply_gates(raw_s, gs, gates43.STRICT)}}


ATTR_ATTRIBUTED = "The Dual advantage is attributed to the decorrelation loss."
ATTR_JOINT = "Joint dense-residual training effect, not attributed to the decorrelation loss."
ATTR_NOT_ASSESSABLE = "Dual advantage; attribution not assessable."


def attribution_statement(db: Optional[Dict[str, Any]], dg: Optional[Dict[str, Any]]) -> Optional[str]:
    """PROTOCOL 1.6, row by row:
      Dual better | Dual better                    -> attributed
      Dual better | q >= 0.10 (unlabelled)         -> joint effect
      Dual better | removed by a gate, or labelled -> not assessable (also when no Dual-g0 test
                                                      exists at all: nothing to attribute with)
      any other combination                        -> no sentence (e.g. Dual-g0 'Dual worse',
                                                      q < 0.10)."""
    if not db or db.get("gate_label") or not str(db.get("verdict") or "").startswith("Dual better"):
        return None
    if dg is None or dg.get("gate_label"):
        return ATTR_NOT_ASSESSABLE
    if str(dg.get("verdict") or "").startswith("Dual better"):
        return ATTR_ATTRIBUTED
    if dg.get("q", float("nan")) >= stats.Q_THRESHOLD:
        return ATTR_JOINT
    return None


def attribution(tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """PROTOCOL 1.6 per (layer, concept, metric, pair) of the family."""
    out = []
    fam: Dict[Tuple, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for t in tests:
        if t["family"]:
            fam[(t["layer"], t["activation"], t["concept"], t["metric"])][t["contrast"]] = t
    for k in sorted(fam):
        db, dg = fam[k].get("dual-baseline"), fam[k].get("dual-dual_g0")
        out.append({"layer": k[0], "activation": k[1], "concept": k[2], "metric": k[3],
                    "dual_baseline": db["verdict"] if db else None,
                    "dual_g0": dg["verdict"] if dg else None,
                    "statement": attribution_statement(db, dg)})
    return out


def report_inputs() -> Dict[str, Any]:
    """What the report was built from (the planner re-runs s9 when this changes).  The gate rule
    ids and the compact verdict digest of BOTH gate versions are part of it, so a report built
    under another D-43 rule or other pair verdicts is stale by construction and never counted as
    done."""
    # 'strict_rows' marks how the sensitivity twin is computed: a report written before the
    # strict twin used only the strict-kept rows lacks the key and is therefore re-run.
    return {"gate_rules": dict(gates43.RULE_ID), "gate_verdicts": gates43.verdict_digest(),
            "strict_rows": "rows of the SAEs the strict gate keeps (s9 gate_filter)",
            "s7j_units_done": sum(os.path.isfile(os.path.join(base.s7j_dir(u), "_summary.json"))
                                  for u in base.gen_units()),
            "s7m_units_done": sum(os.path.isfile(os.path.join(base.s7m_dir(u), "_summary.json"))
                                  for u in base.gen_units()),
            "second_judge": os.path.isfile(base.p("s8", "second_judge.json")),
            "handlabel_filled": os.path.isfile(base.p("s8", "handlabel_sheet_filled.csv"))}


# ----------------------------------------------------------------------------- tables
def _fmt(x, d=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def md_table(header: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _sae_gate_cell(v: Dict[str, Any]) -> str:
    """'<status> (<reasons>)' of one SAE under one gate version."""
    return v.get("status", "?") + (" (" + "; ".join(v["reasons"]) + ")" if v.get("reasons") else "")


def _pair_gate_cell(g: Dict[str, Any]) -> str:
    return (f"**{g['state']}**" + (f" ({g['label']})" if g["label"] else "")
            + (" D-18: TopK pair not steered" if g.get("d18_not_steered") else ""))


def gate_section() -> str:
    """Per-SAE D-43 rows with BOTH verdicts: the primary gate (headline; decides what ran) and
    the strict pre-registered gate (sensitivity).  The (b) final/min loss ratio is a diagnostic
    column ('*' = above the 1.5 threshold, a removal reason only under the strict gate)."""
    rows = []
    for layer in base.LAYERS:
        for act in arms.ACTIVATIONS:
            g = gates43.pair_verdict(layer, act)
            gs = gates43.pair_verdict(layer, act, strict=True)
            for arm in arms.ARMS:
                for s in arms.SEEDS:
                    ref = arms.resolve_arm("B", layer, None, act, arm, s)
                    rec = base.read_json(base.s0_record(ref.tag)) if ref.tag else None
                    v = g["sae"].get(ref.tag, {})
                    vs = gs["sae"].get(ref.tag, {})
                    if not rec:
                        rows.append([f"L{layer}", act, arm, s, "-", "-", "-", "-", "-", "-", "-",
                                     "-", "-", "-", "-", "-", v.get("status", "pending"),
                                     vs.get("status", "pending")])
                        continue
                    ro, ce, tl = rec["rows"], rec["ce"], rec["train_log"]
                    b_flag = "" if tl["recon_ratio"] <= base.GATE_LOSS_RATIO else "*"
                    rows.append([f"L{layer}", act, arm, s, rec["best_epoch"], _fmt(ro["eval_l0"], 1),
                                 _fmt(rec.get("d44", {}).get("training_nonsink_l0"), 1),
                                 _fmt(ro["dead_pct"], 1), ro["live_latents"],
                                 _fmt(ce["frac_recovered_primary"], 4),
                                 _fmt(ce["frac_recovered_inference_all"], 4),
                                 _fmt(tl.get("final_active"), 1), _fmt(tl["recon_ratio"], 2) + b_flag,
                                 _fmt(ro["fvu_nonsink"], 4), _fmt(ro["fvu_sink"], 3),
                                 f"{ro['n_sink_latents']}/{_fmt(ro['excluded_frac_of_live'], 2)}",
                                 _sae_gate_cell(v), _sae_gate_cell(vs)])
            rows.append([f"L{layer}", act, "PAIR", "", "", "", "", "", "", "", "", "", "", "", "", "",
                         _pair_gate_cell(g), _pair_gate_cell(gs)])
    hdr = ["layer", "pair", "arm", "seed", "best_epoch", "L0 (inference, pos>=1)",
           "L0 (training-time non-sink)", "dead %", "live", "CE rec. pos>=1 (primary)",
           "CE rec. all pos", "val Active",
           f"loss/min ((b) diagnostic; * = > {base.GATE_LOSS_RATIO})", "FVU non-sink", "FVU sink",
           "sink latents / excluded frac of live", "D-43 (primary gate; headline)",
           f"D-43 strict ({gates43.STRICT_LABEL})"]
    return md_table(hdr, rows)


def gate_pair_table(gp: Dict[Tuple[int, str], Dict[str, Any]],
                    gs: Dict[Tuple[int, str], Dict[str, Any]]) -> str:
    """One row per (layer, pair): the primary verdict beside the strict one."""
    rows = [[f"L{l}", a, _pair_gate_cell(gp[(l, a)]), _pair_gate_cell(gs[(l, a)]),
             len(gp[(l, a)].get("b_flagged", []))] for (l, a) in gp]
    return md_table(["layer", "pair", "primary gate (headline)", gates43.STRICT_LABEL,
                     f"SAEs with (b) ratio > {base.GATE_LOSS_RATIO}"], rows)


def dual_diag_section() -> str:
    rows = []
    for ref in base.dual_refs():
        rec = base.read_json(base.s0_record(ref.tag))
        c2 = base.read_json(base.s2_record(ref.tag))
        if not rec:
            continue
        ro = rec["rows"]
        s = (c2 or {}).get("summary", {}).get("alpha1m", {})
        rows.append([ref.key, ref.activation, ref.arm, ref.seed, _fmt(ro.get("xM_fvu_nonsink"), 3),
                     _fmt(ro.get("poly_identity_ratio"), 4),
                     _fmt(s.get("cancel_ratio", {}).get("median")), _fmt(s.get("net_along", {}).get("median")),
                     _fmt(s.get("cos_poly_atom", {}).get("median")),
                     "%.3e" % ro.get("span_penalty_test_nonsink", float("nan")),
                     "bypass regime" if ro.get("bypass_regime") else "-"])
    return md_table(["layer", "pair", "arm", "seed", "xM FVU", "||poly(r)-r||/||r||",
                     "cancel ratio (median, alpha=m)", "net along atom (median)",
                     "cos(d_poly, W_j) (median)", "span penalty (test)", "label"], rows)


def selection_section() -> str:
    rows = []
    for layer in base.LAYERS:
        for act in arms.ACTIVATIONS:
            for c in concepts.CONCEPTS:
                cells = []
                for arm in arms.ARMS:
                    vals = defaultdict(list)
                    for s in arms.SEEDS:
                        ref = arms.resolve_arm("B", layer, None, act, arm, s)
                        sel = base.read_json(base.s1_select(ref.tag)) if ref.tag else None
                        e = (sel or {}).get("concepts", {}).get(c.cid)
                        if e:
                            for k in ("test_auroc_best", "test_auroc_mean"):
                                vals[k].append(e[k])
                            lm = e.get("livematched", {})
                            if lm.get("status") == "ok" and lm["test_auroc"]:
                                vals["lm"].append(float(np.mean(lm["test_auroc"])))
                    cells.append(" / ".join(stats.fmt_mean_sd(vals[k]) if vals[k] else "-"
                                            for k in ("test_auroc_best", "test_auroc_mean", "lm")))
                if any(x != "- / - / -" for x in cells):
                    rows.append([f"L{layer}", act, c.cid] + cells)
    return md_table(["layer", "pair", "concept"] + [f"{a}: best / mean / live-matched mean test AUROC" for a in arms.ARMS], rows)


def bypass_pairs() -> set:
    """(layer, activation) pairs with at least one bypass-regime Dual/g0 SAE (s0 rows, C20)."""
    out = set()
    for ref in base.dual_refs():
        rec = base.read_json(base.s0_record(ref.tag))
        if rec and rec.get("rows", {}).get("bypass_regime"):
            out.add((int(ref.key[1:]), ref.activation))
    return out


def _tindex(tests: List[Dict[str, Any]]) -> Dict[Tuple, Dict[str, Any]]:
    return {(t["layer"], t["activation"], t["mode"], t["dirset"], _bkey(t["budget"]), t["concept"],
             t["metric"], t["contrast"]): t for t in tests}


def _combo_rows(idx: Index, tindex: Dict[Tuple, Dict[str, Any]], layer, act, mode, dirset,
                b, ref_idx: Optional[Index] = None,
                removed: Optional[Dict[SeedKey, str]] = None) -> List[List[Any]]:
    """The rows of one (layer, pair, mode, selection, budget) results table: arm mean +- sd over
    seeds from ``idx`` (one gate version's rows), then each contrast's verdict (family) or
    exploratory row from ``tindex`` (the same version's tests).
    ``ref_idx`` (strict twin only) is the primary version's index: an arm cell whose seeds the
    primary table reports but this version dropped names those seeds and the reason from
    ``removed`` ('D-43' own failure, 'D-18' TopK layer not steered), so the twin never shows a
    silently smaller n -- and an arm whose every seed was dropped still gets a cell (and its row
    stays) instead of vanishing as '-'."""
    rows = []
    for c in concepts.CONCEPTS:
        for metric in METRICS:
            cells = []
            for arm in arms.ARMS:
                v = _vals(idx, layer, act, arm, mode, dirset, b, c.cid, metric)
                cell = stats.fmt_mean_sd(list(v.values())) if v else "-"
                if ref_idx is not None:
                    # Both indexes come from the same rows minus the gated SAEs, so the set
                    # difference is exactly the seeds this version dropped for this cell.
                    gone = sorted(set(_vals(ref_idx, layer, act, arm, mode, dirset, b, c.cid, metric))
                                  - set(v))
                    if gone:
                        why = "/".join(sorted({(removed or {}).get((layer, act, arm, s), "gate")
                                               for s in gone}))
                        note = (f"seed{'s' if len(gone) > 1 else ''} {', '.join(map(str, gone))} "
                                f"removed by the strict gate ({why})")
                        cell = f"{cell}; {note}" if v else note
                cells.append(cell)
            tt = []
            for con in ("dual-baseline", "dual-dual_g0"):
                t = tindex.get((layer, act, mode, dirset, b, c.cid, metric, con))
                if t is None:
                    tt.append("-")
                elif t.get("verdict"):
                    tt.append(f"{t['verdict']}; delta={_fmt(t.get('delta'))} "
                              f"[{_fmt(t.get('ci_lo'))}, {_fmt(t.get('ci_hi'))}] p={stats.fmt_p(t.get('p', float('nan')))} "
                              f"q={stats.fmt_p(t.get('q', float('nan')))} MDE80={_fmt(t.get('mde80'))}")
                elif t.get("exploratory"):
                    tt.append(t["exploratory"] + (f" ({t['gate_label']})" if t["gate_label"] else ""))
                else:
                    tt.append(t.get("gate_label") or "-")
            if any(x != "-" for x in cells):
                rows.append([c.cid, metric] + cells + tt)
    return rows


RESULTS_HDR = ["concept", "metric"] + list(arms.ARMS) + ["Dual - baseline", "Dual - dual_g0"]


def results_tables(ps: List[Dict[str, Any]], tests: List[Dict[str, Any]], primary,
                   idx: Optional[Index] = None, bypass: Optional[set] = None,
                   tests_strict: Optional[List[Dict[str, Any]]] = None,
                   idx_strict: Optional[Index] = None,
                   removed_strict: Optional[Dict[SeedKey, str]] = None) -> str:
    """One table per (layer, pair, mode, selection, budget).  ``ps`` / ``idx`` / ``tests`` are
    the primary gate's rows, index and tests (headline).  With ``tests_strict``, every
    CONFIRMATORY table is followed by its twin under the strict pre-registered gate, headed with
    gates43.STRICT_LABEL: its arm columns come from ``idx_strict`` (the rows the strict gate keeps;
    default ``idx``, i.e. a strict gate that drops nothing), dropped seeds are named in the cell
    with their reason from ``removed_strict``, and its gate labels, BH q and verdicts are that
    version's.  Exploratory tables are produced once (primary rows and gate labels)."""
    idx = idx if idx is not None else index_ps(ps)
    idx_s = idx_strict if idx_strict is not None else idx
    bypass = bypass if bypass is not None else bypass_pairs()
    out = []
    tindex = _tindex(tests)
    sindex = _tindex(tests_strict) if tests_strict is not None else None
    combos = sorted({(r["layer"], r["activation"], r["mode"], r["dirset"], _bkey(r["budget"])) for r in ps
                     if r["arm"] in arms.ARMS})
    for (layer, act, mode, dirset, b) in combos:
        conf = primary is not None and mode == "i" and dirset == "full" and abs(b - primary) < 1e-9
        role = "CONFIRMATORY (P1 budget; primary D-43 gate, headline)" if conf else "exploratory"
        label = BYPASS_SUFFIX if (mode == "iii" and (layer, act) in bypass) else ""
        head = f"L{layer} / {act} / mode {base.mode_label(mode)} / {dirset} / budget {b:g} nats"
        out.append(f"\n#### {head} -- {role}{label}\n")
        out.append(md_table(RESULTS_HDR, _combo_rows(idx, tindex, layer, act, mode, dirset, b)))
        if conf and sindex is not None:
            out.append(f"\n#### {head} -- CONFIRMATORY (P1 budget) -- {gates43.STRICT_LABEL}{label}\n")
            out.append(md_table(RESULTS_HDR, _combo_rows(idx_s, sindex, layer, act, mode, dirset, b,
                                                         ref_idx=idx, removed=removed_strict)))
    return "\n".join(out)


def judge_rates_section() -> str:
    """D-50: parse-failure, truncation (finish 'length' after the retry) and retry rates per
    layer x pair x arm x mode x selection x budget (judgement-weighted over seeds)."""
    acc: Dict[Tuple, List[float]] = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    refs = {r.tag: r for r in base.sae_refs()}
    for u in base.gen_units():
        s = base.read_json(os.path.join(base.s7j_dir(u), "_summary.json"))
        if not s:
            continue
        r = refs.get(u)
        who = (r.key, r.activation, r.arm) if r else (u, "-", "armfree" if u.startswith("armfree") else "null")
        for shard, d in s.get("per_shard", {}).items():
            n = d.get("n_judgements") or 0
            if not n:
                continue
            k = who + (shard.replace(".jsonl", ""),)
            a = acc[k]
            a[0] += n
            a[1] += n * (d["parse_failure_rate"] or 0)
            a[2] += n * (d["truncation_rate"] or 0)
            a[3] += n * (d["retried_rate"] or 0)
    rows = [[*k, a[0], _fmt(a[1] / a[0], 4), _fmt(a[2] / a[0], 4), _fmt(a[3] / a[0], 4)]
            for k, a in sorted(acc.items())]
    return md_table(["layer/unit", "pair", "arm", "cell (mode_selection_budget)", "n judgements",
                     "parse failure rate", "truncation rate", "retried rate"], rows)


def calibration_section() -> str:
    """s5 reachability per layer x pair x arm x mode x budget (unreachable directions, summed over
    seeds; PROTOCOL R5 prints the unreachable count)."""
    acc: Dict[Tuple, List[int]] = defaultdict(lambda: [0, 0])
    refs = {r.tag: r for r in base.sae_refs()}
    for u in base.gen_units()[:-1]:
        rec = base.read_json(base.s5_record(u))
        if not rec:
            continue
        r = refs.get(u)
        who = (r.key, r.activation, r.arm) if r else (u, "-", "armfree")
        for d in rec["directions"].values():
            for b, e in d["budgets"].items():
                a = acc[who + (d["mode"], b)]
                a[0] += 1
                a[1] += e["status"] != "ok"
    rows = [[*k, a[0], a[1]] for k, a in sorted(acc.items())]
    return md_table(["layer/unit", "pair", "arm", "mode", "budget", "directions (sum over seeds)",
                     "unreachable"], rows)


def reference_tables(ps: List[Dict[str, Any]]) -> str:
    rows = []
    for r in sorted((r for r in ps if r["arm"] in ("armfree", "null") and r["metric"] in ("hm", "p_concept", "concept_score")),
                    key=lambda r: (r["layer"] or 0, r["dirset"], r["budget"], r["concept"], r["metric"])):
        rows.append([f"L{r['layer']}" if r["arm"] == "armfree" else "-", r["dirset"], f"{r['budget']:g}",
                     r["concept"], r["metric"], _fmt(r["value"]), r["n_latents_reachable"]])
    return md_table(["layer", "direction", "budget", "concept", "metric", "value",
                     "n directions reachable"], rows)


def kappa_section() -> str:
    """D-50 / PROTOCOL 4.7: second-judge and hand-label agreement (exploratory)."""
    from steering import s8_handlabel as S8
    rows = []
    sj = base.read_json(base.p("s8", "second_judge.json"))
    if sj:
        for k, v in sj.get("kappa_vs_phi4", {}).items():
            rows.append(["InternVL3.5-8B vs phi-4", k, v.get("n", sj.get("n_items")),
                         _fmt(v.get("cohen_kappa")), _fmt(v.get("quadratic_weighted_kappa")),
                         _fmt(v.get("exact_agreement")), _fmt(v.get("internvl_parse_failure_rate"), 4)])
    hk = S8.handlabel_kappa()
    if hk:
        for k in S8.KINDS:
            for who in ("phi4", "internvl"):
                v = hk.get(f"{k}_vs_{who}")
                if v:
                    rows.append([f"human vs {'phi-4' if who == 'phi4' else 'InternVL3.5-8B'}", k,
                                 v["n"], _fmt(v["cohen_kappa"]), _fmt(v["quadratic_weighted_kappa"]),
                                 _fmt(v["exact_agreement"]), "-"])
    if not rows:
        return ("Not available yet: s8j (InternVL second judge) has not run and "
                "s8/handlabel_sheet_filled.csv has not been returned.")
    note = "" if hk else "\n\nHand-label kappa: waiting for s8/handlabel_sheet_filled.csv."
    return md_table(["raters", "rating", "n items", "Cohen kappa", "quadratic-weighted kappa",
                     "exact agreement", "parse failure rate"], rows) + note


def _attr_table(attr: List[Dict[str, Any]]) -> str:
    return md_table(["layer", "pair", "concept", "metric", "Dual-baseline", "Dual-g0", "statement"],
                    [[a["layer"], a["activation"], a["concept"], a["metric"], a["dual_baseline"],
                      a["dual_g0"], a["statement"] or "-"] for a in attr])


def _family_counts(tests: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """(tests listed, tests that reached BH, 'Dual better' verdicts) of one version's family."""
    fam = [t for t in tests if t["family"]]
    tested = [t for t in fam if not t["gate_label"]]
    n_win = sum(1 for t in tested if str(t["verdict"]).startswith("Dual better"))
    return len(fam), len(tested), n_win


def _test_key(t: Dict[str, Any]) -> Tuple:
    return (t["layer"], t["activation"], t["mode"], t["dirset"], _bkey(t["budget"]), t["concept"],
            t["metric"], t["contrast"])


def verdict_transitions(tests: List[Dict[str, Any]],
                        tests_strict: List[Dict[str, Any]]) -> Dict[Tuple[str, str], int]:
    """{(primary verdict, strict verdict): n} over the family tests whose verdict differs between
    the two gate versions (the verdict word up to its first ' (' so MDE80 values do not split
    rows)."""
    sidx = {_test_key(t): t for t in tests_strict if t["family"]}
    out: Dict[Tuple[str, str], int] = defaultdict(int)
    for t in tests:
        if not t["family"]:
            continue
        a = str(t["verdict"]).split(" (")[0]
        b = str(sidx[_test_key(t)]["verdict"]).split(" (")[0]
        if str(t["verdict"]) != str(sidx[_test_key(t)]["verdict"]):
            out[(a, b)] += 1
    return dict(out)


def family_summary(tests: List[Dict[str, Any]], tests_strict: List[Dict[str, Any]]) -> List[str]:
    """The report's headline bullets: the primary-gate family first (its verdict words are the
    headline), then the strict-gate family labelled as the sensitivity analysis."""
    n, nt, nw = _family_counts(tests)
    ns, nts, nws = _family_counts(tests_strict)
    n_changed = sum(verdict_transitions(tests, tests_strict).values())
    return [
        f"- D-43 gate (PROTOCOL 1.5): headline = primary gate, "
        f"{gates43.RULE_ID[gates43.PRIMARY]}. Every confirmatory table is followed by its twin "
        f"under the strict pre-registered gate ({gates43.RULE_ID[gates43.STRICT]}), labelled "
        f"'{gates43.STRICT_LABEL}'; BH runs within the family separately for each version.",
        f"- **Confirmatory family, primary D-43 gate (headline)**: {n} tests listed, {nt} tested "
        f"after gates; " + stats.win_count_line(nw, nt, "'Dual better' verdicts"),
        f"- Confirmatory family, {gates43.STRICT_LABEL}: {ns} tests listed, {nts} tested after "
        f"gates; " + stats.win_count_line(nws, nts, "'Dual better' verdicts")
        + f"; {n_changed} family tests have a different verdict than under the primary gate",
    ]


def _seed_list(rm: Dict[SeedKey, str]) -> str:
    """'L6 topk dual_g0 seed 42 (D-43), ...' for report bullets and warnings."""
    return ", ".join(f"L{k[0]} {k[1]} {k[2]} seed {k[3]} ({rm[k]})" for k in sorted(rm))


def removed_lines(removed_p: Dict[SeedKey, str], removed_s: Dict[SeedKey, str]) -> List[str]:
    """Report bullets naming the SAEs whose GENERATED rows a gate version dropped (module
    docstring): the strict-only ones, which the sensitivity analysis leaves out of its arm
    columns, Welch numbers and outlier sd_ref; and -- expected empty -- any the primary gate drops,
    i.e. generated data of an SAE the cascade's own gate now removes (outputs left over from
    before an s0 record changed), flagged as a warning because the cascade should not have run
    them."""
    only_s = {k: v for k, v in removed_s.items() if k not in removed_p}
    out = ["- Strict-gate sensitivity is computed from the seeds the strict gate keeps: "
           + (f"{len(only_s)} generated SAE(s) removed by it but not by the primary gate are left "
              f"out of its arm columns, Welch numbers and outlier sd_ref: {_seed_list(only_s)}"
              if only_s else "it drops no generated SAE the primary gate keeps")]
    if removed_p:
        out.append(f"- **WARNING**: generated rows of {len(removed_p)} SAE(s) that the primary gate "
                   f"removes were left out of every table (stale outputs from before an s0 record "
                   f"changed?): {_seed_list(removed_p)}")
    return out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    import csv
    if not rows:
        base.atomic_write_bytes(path, b"")
        return
    keys = sorted({k for r in rows for k in r})
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
    base.atomic_write_bytes(path, buf.getvalue().encode())


def _selftest() -> int:
    """Synthetic 3-seed data through build_tests: the full 360-test grid, family membership, BH,
    verdict words, the D-43 gate removal, the held-JumpReLU and no-data labels, the outlier flag,
    R5 ('not assessable' with < 2 seeds) and every row of the PROTOCOL 1.6 attribution table.
    Then the D-43 amendment's sensitivity analysis (_selftest_strict): the same data under a
    strict gate that additionally collapses three pairs and drops the rows of the SAEs it
    removes."""
    rng = np.random.default_rng(0)
    ps = []
    for layer in base.LAYERS:
        for act in ("topk", "batch_topk"):
            for arm in arms.ARMS:
                for s in arms.SEEDS:
                    for c in concepts.CONCEPTS[:3]:
                        for metric in METRICS:
                            shift = 0.5 if (arm == "dual" and c.cid == "amazon_positive") else 0.0
                            v = 1.0 + shift + 0.01 * rng.standard_normal()
                            if (layer, act, arm, s, c.cid, metric) == (18, "topk", "baseline", 44,
                                                                       "agnews_world", "hm"):
                                v = 50.0                         # an outlier seed
                            ps.append({"layer": layer, "activation": act, "arm": arm, "seed": s,
                                       "mode": "i", "dirset": "full", "budget": 0.2,
                                       "concept": c.cid, "metric": metric, "value": v})
    # one arm with a single seed -> R5
    ps = [r for r in ps if not (r["layer"] == 6 and r["activation"] == "topk" and r["arm"] == "dual_g0"
                                and r["seed"] != 42 and r["concept"] == "amazon_negative")]
    gates = {(l, a): {"state": "pass", "label": None} for l in base.LAYERS for a in arms.ACTIVATIONS}
    gates[(12, "batch_topk")] = {"state": "collapsed", "label": "collapsed"}
    tests = build_tests(ps, 0.2, gates=gates, jr_status=lambda l, c: ("pending", None))
    fam = [t for t in tests if t["family"]]
    data_cids = {c.cid for c in concepts.CONCEPTS[:3]}
    bad = []
    if len(fam) != 3 * 10 * 2 * 3 * 2:
        bad.append(f"family listed {len(fam)} != 360 (the full PROTOCOL 4.1 grid)")
    if not all(t["verdict"] == HELD_JR_LABEL for t in fam if t["activation"] == "jump_relu"):
        bad.append("held JumpReLU family tests not labelled")
    if not all(t["verdict"] == "collapsed" for t in fam if t["layer"] == 12 and t["activation"] == "batch_topk"):
        bad.append("D-43 collapsed pair not removed")
    nod = [t for t in fam if t["activation"] != "jump_relu" and t["concept"] not in data_cids
           and not (t["layer"] == 12 and t["activation"] == "batch_topk")]
    if not nod or not all(t["verdict"] == "pending (no data)" for t in nod):
        bad.append("family cells without data not labelled 'pending (no data)'")
    t_out = [t for t in fam if t["layer"] == 18 and t["activation"] == "topk" and t["concept"] == "agnews_world"
             and t["metric"] == "hm" and t["contrast"] == "dual-baseline"]
    if not (t_out and t_out[0]["verdict"] == "not assessable"):
        bad.append(f"outlier seed not flagged: {t_out[0]['verdict'] if t_out else None}")
    t_r5 = [t for t in fam if t["layer"] == 6 and t["activation"] == "topk" and t["concept"] == "amazon_negative"
            and t["contrast"] == "dual-dual_g0"]
    if not all(t["verdict"] == "not assessable" for t in t_r5):
        bad.append("R5 (<2 seeds) not applied")
    wins = [t for t in fam if t["concept"] == "amazon_positive" and not t["gate_label"]]
    if not wins or not all(t["verdict"].startswith("Dual better") for t in wins):
        bad.append(f"true +0.5 effects not detected: {[t['verdict'] for t in wins][:3]}")
    # Null cells (no true effect): BH controls the FDR, not the FWER, so a few can come out
    # significant; more than a quarter would mean the test machinery is broken.
    nulls = [t for t in fam if t["concept"] != "amazon_positive" and not t["gate_label"]]
    fp = sum(t["verdict"].startswith("Dual") for t in nulls)
    if not nulls or fp > 0.25 * len(nulls):
        bad.append(f"{fp}/{len(nulls)} null cells significant")
    tested = [t for t in fam if not t["gate_label"]]
    if len(tested) != sum(1 for t in fam if t["concept"] in data_cids and t["activation"] != "jump_relu"
                          and not (t["layer"] == 12 and t["activation"] == "batch_topk")
                          and t["verdict"] != "not assessable"):
        bad.append("tested count does not equal the data-bearing, ungated family cells")
    words_ok = all(not any(w in str(t["verdict"]) for w in stats.FORBIDDEN_WORDS) for t in fam)
    if not words_ok:
        bad.append("forbidden verdict words")
    # PROTOCOL 1.6, row by row (synthetic verdict records).
    better = {"verdict": "Dual better than baseline", "gate_label": None, "q": 0.01}
    cases = [
        (better, {"verdict": "Dual better than dual_g0", "gate_label": None, "q": 0.02}, ATTR_ATTRIBUTED),
        (better, {"verdict": "inconclusive (MDE80 = 0.2)", "gate_label": None, "q": 0.4}, ATTR_JOINT),
        (better, {"verdict": "equivalent", "gate_label": None, "q": 0.6}, ATTR_JOINT),
        (better, {"verdict": "collapsed", "gate_label": "collapsed"}, ATTR_NOT_ASSESSABLE),
        (better, None, ATTR_NOT_ASSESSABLE),
        (better, {"verdict": "Dual worse than dual_g0", "gate_label": None, "q": 0.03}, None),
        ({"verdict": "inconclusive (MDE80 = 0.2)", "gate_label": None, "q": 0.5},
         {"verdict": "Dual better than dual_g0", "gate_label": None, "q": 0.01}, None),
        ({"verdict": "Dual worse than baseline", "gate_label": None, "q": 0.01}, None, None),
    ]
    for i, (db, dg, want) in enumerate(cases):
        got = attribution_statement(db, dg)
        if got != want:
            bad.append(f"attribution case {i}: {got!r} != {want!r}")
    n_strict = _selftest_strict(ps, gates, tests, bad)
    for b in bad:
        print("  FAIL", b)
    print(f"S9 SELFTEST {'OK' if not bad else 'FAILED'} ({len(fam)} family tests listed, "
          f"{len(tested)} tested under the primary gate, {n_strict} under the strict gate; "
          f"verdicts: {sorted(set(str(t['verdict']).split(' (')[0] for t in fam))})")
    return 0 if not bad else 1


def _sig(t: Dict[str, Any]) -> Tuple:
    """Comparable signature of one test (NaN-safe: q is absent or NaN for untested cells)."""
    q = t.get("q")
    return _test_key(t) + (t["gate_label"], str(t.get("verdict")), str(t.get("exploratory")),
                           None if q is None or not np.isfinite(q) else round(q, 12))


def _fake_pair(layer: int, act: str, fails: Tuple[Tuple[str, int], ...] = (),
               label: str = "collapsed", d18: bool = False) -> Dict[str, Any]:
    """A synthetic pair verdict WITH per-SAE entries (the fields gate_removed_seeds and
    gates43.excluded_reason read): the (arm, seed)s in ``fails`` fail, every other SAE passes."""
    sae = {f"L{layer}_{act}_{arm}_s{s}": {"status": "fail" if (arm, s) in fails else "pass",
                                          "arm": arm, "seed": s}
           for arm in arms.ARMS for s in arms.SEEDS}
    return {"layer": layer, "activation": act, "state": "collapsed" if fails else "pass",
            "label": label if fails else None, "sae": sae, "d18_not_steered": d18}


def _parse_md_tables(md: str) -> Dict[str, List[List[str]]]:
    """{'#### heading': [body-row cells]} of rendered results tables (selftest only)."""
    out: Dict[str, List[List[str]]] = {}
    head = None
    for ln in md.splitlines():
        if ln.startswith("#### "):
            head = ln
            out[head] = []
        elif head and ln.startswith("| ") and not ln.startswith("| concept |"):
            out[head].append([c.strip() for c in ln.strip().strip("|").split(" | ")])
    return out


def _selftest_strict(ps: List[Dict[str, Any]], gates: Dict[Tuple[int, str], Dict[str, Any]],
                     tests_build: List[Dict[str, Any]], bad: List[str]) -> int:
    """D-43 amendment: primary vs strict gate on the selftest data.
    Gates.  Primary = the _selftest gates, with per-SAE all-pass entries for L6 and L12 TopK (so
    gate_filter has entries to read, and must drop nothing).  Strict = primary plus one collapse
    per mechanism:
      L6 TopK       dual_g0 seed 42 fails (the shape of a (b)-only failure): its rows dropped;
      L12 TopK      tkuntied baseline seed 43 fails -> 'baseline collapsed' + D-18: all 9 dropped;
      L18 BatchTopK pair-level collapse without per-SAE entries: no rows dropped.
    Checks: exactly those rows are dropped (none by the primary gate); gated_versions agrees with
    build_tests for both versions; the shared-Welch path (strict drops nothing more) reuses the
    primary pass without leaking the strict labels into it; tests are stamped and carry no private
    keys; the extra collapses are labelled only in the strict version; BH is computed within each
    version's family (q == stats.bh over THAT version's tested p-values, and differs from the
    primary q for shared tests because m differs); the strict tested set is a strict subset of the
    primary one; Welch numbers are identical wherever no seed was dropped and recomputed on the
    kept seeds where one was; the outlier flag works in the strict pass, and its sd_ref pool
    excludes dropped seeds (a planted value in a dropped SAE must not mask an outlier of another
    pair); the report renders a labelled strict twin after every confirmatory table whose arm
    columns name the dropped seeds and otherwise equal the primary ones; the headline bullets put
    the primary family first and name the dropped SAEs.  Returns the number of strict tests that
    reached BH."""
    jr = lambda l, c: ("pending", None)  # noqa: E731 - held JumpReLU, as in _selftest
    P, S = gates43.PRIMARY, gates43.STRICT
    gates_p = dict(gates)
    gates_p[(6, "topk")] = _fake_pair(6, "topk")
    gates_p[(12, "topk")] = _fake_pair(12, "topk")
    gates_s = dict(gates_p)
    gates_s[(6, "topk")] = _fake_pair(6, "topk", fails=(("dual_g0", 42),))
    gates_s[(12, "topk")] = _fake_pair(12, "topk", fails=(("baseline", 43),),
                                       label="baseline collapsed", d18=True)
    gates_s[(18, "batch_topk")] = {"state": "collapsed", "label": "collapsed"}
    extra = {(6, "topk"): "collapsed", (12, "topk"): "baseline collapsed",
             (18, "batch_topk"): "collapsed"}
    gv = gated_versions(ps, 0.2, gates_p, gates_s, jr_status=jr)
    vp, vs = gv[P], gv[S]
    tp, ts = vp["tests"], vs["tests"]

    # Which SAEs each version drops (only SAEs that have rows count).
    want_rm = {(12, "topk", arm, s): ("D-43" if (arm, s) == ("baseline", 43) else "D-18")
               for arm in arms.ARMS for s in arms.SEEDS}
    want_rm[(6, "topk", "dual_g0", 42)] = "D-43"
    if vp["removed"]:
        bad.append(f"primary gate dropped rows: {sorted(vp['removed'])}")
    if vs["removed"] != want_rm:
        bad.append(f"strict gate dropped {sorted(vs['removed'].items())}, expected "
                   f"{sorted(want_rm.items())}")
    if len(vs["ps"]) >= len(vp["ps"]) or len(vp["ps"]) != len(ps):
        bad.append(f"row counts: all {len(ps)}, primary {len(vp['ps'])}, strict {len(vs['ps'])}")

    # Entry points agree; nothing leaks between versions.
    if [_sig(t) for t in tp] != [_sig(t) for t in tests_build]:
        bad.append("gated_versions primary != build_tests (primary)")
    ts_build = build_tests(ps, 0.2, gates=gates_s, jr_status=jr, strict=True)
    if [_sig(t) for t in ts] != [_sig(t) for t in ts_build]:
        bad.append("gated_versions strict != build_tests(strict=True)")
    # Shared-Welch path: a strict gate that drops nothing more (pair-level collapse only) reuses
    # the primary index/raw pass, and applying its labels must not alter the primary tests.
    gates_s0 = dict(gates_p)
    gates_s0[(18, "batch_topk")] = {"state": "collapsed", "label": "collapsed"}
    gv0 = gated_versions(ps, 0.2, gates_p, gates_s0, jr_status=jr)
    if gv0[S]["idx"] is not gv0[P]["idx"]:
        bad.append("strict gate dropping nothing more did not reuse the primary Welch pass")
    if [_sig(t) for t in gv0[P]["tests"]] != [_sig(t) for t in tests_build]:
        bad.append("applying the strict gate on the shared pass changed the primary tests (raw mutated)")
    if not all(t["verdict"] == "collapsed" for t in gv0[S]["tests"]
               if t["family"] and (t["layer"], t["activation"]) == (18, "batch_topk")):
        bad.append("shared pass: strict pair-level collapse not applied")
    if not (all(t["gate"] == P for t in tp) and all(t["gate"] == S for t in ts)):
        bad.append("tests not stamped with their gate version")
    if any(k.startswith("_") for t in tp + ts for k in t):
        bad.append("private raw keys leaked into the final tests")

    fam_p = {_test_key(t): t for t in tp if t["family"]}
    fam_s = {_test_key(t): t for t in ts if t["family"]}
    if set(fam_p) != set(fam_s):
        bad.append("the two versions list different family tests")
    if not all(t["verdict"] == extra[(k[0], k[1])] for k, t in fam_s.items() if (k[0], k[1]) in extra):
        bad.append("strict gate: extra collapsed pairs not removed with their label")
    if any(t["gate_label"] in ("collapsed", "baseline collapsed") for k, t in fam_p.items()
           if (k[0], k[1]) in extra):
        bad.append("primary gate: pairs that collapse only under strict were removed")
    if not any(not t["gate_label"] for k, t in fam_p.items() if (k[0], k[1]) in extra):
        bad.append("primary gate: no tested cell in the strict-only collapsed pairs (test is vacuous)")
    # BH within each version's family, separately.
    for name, tv in (("primary", tp), ("strict", ts)):
        tested = [t for t in tv if t["family"] and not t["gate_label"]]
        want = stats.bh([t["p_bh"] for t in tested])
        got = np.asarray([t["q"] for t in tested])
        if not (len(tested) and np.allclose(got, want, equal_nan=True)):
            bad.append(f"{name}: q is not BH over that version's own family")
    tested_p = {k for k, t in fam_p.items() if not t["gate_label"]}
    tested_s = {k for k, t in fam_s.items() if not t["gate_label"]}
    if not (tested_s < tested_p):
        bad.append("strict tested set is not a strict subset of the primary tested set")
    shared = sorted(tested_s)
    if not any(abs(fam_p[k]["q"] - fam_s[k]["q"]) > 1e-12 for k in shared):
        bad.append("BH q identical in both versions for every shared test (m not separate?)")

    # Welch: identical wherever neither arm of the contrast lost a seed; recomputed on the kept
    # seeds where one did (L6 TopK: dual_g0 lost seed 42; L12 TopK: every seed of every arm).
    def lost(k) -> bool:
        return (k[0], k[1]) == (12, "topk") or ((k[0], k[1]) == (6, "topk") and k[7] == "dual-dual_g0")
    welch_keys = ("delta", "p", "ci_lo", "ci_hi", "mde80", "p_bh", "n_dual", "n_comp")
    n_lost = 0
    for k, t in fam_p.items():
        u = fam_s[k]
        if lost(k):
            n_lost += 1
            if (k[0], k[1]) == (12, "topk"):
                ok = u["n_dual"] == 0 and u["n_comp"] == 0
            else:
                ok = u["n_dual"] == t["n_dual"] and u["n_comp"] == max(t["n_comp"] - 1, 0)
            if not ok:
                bad.append(f"strict Welch at {k} not recomputed on the kept seeds: n "
                           f"{t['n_dual']}/{t['n_comp']} -> {u['n_dual']}/{u['n_comp']}")
                break
            continue
        for w in welch_keys:
            a, b = t.get(w), u.get(w)
            if not ((a is None and b is None) or (a is not None and b is not None
                                                   and (a == b or (np.isnan(a) and np.isnan(b))))):
                bad.append(f"Welch field {w} differs between versions at {k} (no seed dropped)")
                break
    if not n_lost:
        bad.append("no family test lost a seed under the strict gate (check is vacuous)")
    # Outlier flag in the strict pass (L18 TopK keeps every seed in both versions).
    k_out = (18, "topk", "i", "full", _bkey(0.2), "agnews_world", "hm", "dual-baseline")
    if fam_s[k_out]["verdict"] != "not assessable":
        bad.append(f"strict pass lost the outlier flag at {k_out}: {fam_s[k_out]['verdict']}")
    # sd_ref pool: plant a huge value in the dropped SAE (L6 TopK dual_g0 seed 42) and a moderate
    # outlier in ANOTHER pair of the same layer.  Primary pool (with the huge value): sd_ref is
    # inflated and the moderate outlier passes; strict pool (dropped SAE left out): it is flagged.
    ps_o = [dict(r) for r in ps]
    for r in ps_o:
        if (r["concept"], r["metric"]) != ("agnews_world", "hm") or r["layer"] != 6:
            continue
        if (r["activation"], r["arm"], r["seed"]) == ("topk", "dual_g0", 42):
            r["value"] = 50.0
        elif (r["activation"], r["arm"], r["seed"]) == ("batch_topk", "baseline", 44):
            r["value"] = r["value"] + 0.3
    gvo = gated_versions(ps_o, 0.2, gates_p, gates_s, jr_status=jr)
    k_pool = (6, "batch_topk", "i", "full", _bkey(0.2), "agnews_world", "hm", "dual-baseline")
    vo_p = {_test_key(t): t for t in gvo[P]["tests"]}[k_pool]
    vo_s = {_test_key(t): t for t in gvo[S]["tests"]}[k_pool]
    if not (vo_p["gate_label"] is None and vo_s["verdict"] == "not assessable"):
        bad.append(f"strict outlier sd_ref still pools the dropped SAE: primary "
                   f"{vo_p['gate_label']!r}, strict {vo_s['verdict']!r} (want None / not assessable)")

    # Report rendering.
    md = results_tables(vp["ps"], tp, 0.2, idx=vp["idx"], bypass=set(), tests_strict=ts,
                        idx_strict=vs["idx"], removed_strict=vs["removed"])
    tabs = _parse_md_tables(md)
    conf = [h for h in tabs if "CONFIRMATORY" in h]
    n_combo = len({(r["layer"], r["activation"]) for r in ps})
    if len(conf) != 2 * n_combo or any(
            ("primary D-43 gate, headline" in conf[i]) != (i % 2 == 0)
            or (gates43.STRICT_LABEL in conf[i]) != (i % 2 == 1) for i in range(len(conf))):
        bad.append(f"results tables: {len(conf)} confirmatory headings, expected {2 * n_combo} "
                   f"alternating primary/strict")
    note = "removed by the strict gate"
    for i in range(0, len(conf) - 1, 2):
        hp, hs = conf[i], conf[i + 1]
        rp, rs = tabs[hp], tabs[hs]
        if any(note in c for r in rp for c in r):
            bad.append(f"primary table names strict-dropped seeds: {hp}")
        if not rs or len(rs) != len(rp) or [r[:2] for r in rs] != [r[:2] for r in rp]:
            bad.append(f"strict twin rows differ from the primary rows: {hs}")
            continue
        if hp.startswith("#### L6 / topk /"):
            # dual_g0 column (index 4) names seed 42 in every row, with n=2 where seeds remain
            # (amazon_negative had only seed 42 -> nothing remains); baseline / dual untouched.
            if not all("seed 42 removed by the strict gate (D-43)" in r[4]
                       and ("(n=2)" in r[4] or (r[0] == "amazon_negative" and "+-" not in r[4]))
                       for r in rs) or [r[2:4] for r in rs] != [r[2:4] for r in rp]:
                bad.append("strict twin L6 TopK: dual_g0 seed 42 not dropped/named, or other arms changed")
        elif hp.startswith("#### L12 / topk /"):
            # D-18: every arm empty; baseline names its own D-43 seed and the D-18 ones.
            if not all(all("+-" not in c for c in r[2:5]) and "seeds 42, 43, 44 removed by the "
                       "strict gate (D-18/D-43)" in r[2] and "(D-18)" in r[3] and "(D-18)" in r[4]
                       for r in rs):
                bad.append("strict twin L12 TopK: D-18 layer not emptied/named")
        elif [r[2:5] for r in rs] != [r[2:5] for r in rp]:
            bad.append(f"strict twin arm columns differ although nothing was dropped: {hs}")
    summ = family_summary(tp, ts)
    if not ("primary D-43 gate (headline)" in summ[1] and gates43.STRICT_LABEL in summ[2]):
        bad.append("headline bullets: primary family is not the headline / strict not labelled")
    rl = removed_lines(vp["removed"], vs["removed"])
    if len(rl) != 1 or "10 generated SAE(s)" not in rl[0] or "L6 topk dual_g0 seed 42 (D-43)" not in rl[0]:
        bad.append(f"removed-SAE bullet wrong: {rl}")
    rl2 = removed_lines({(6, "topk", "dual", 42): "D-43"}, {(6, "topk", "dual", 42): "D-43"})
    if len(rl2) != 2 or "WARNING" not in rl2[1] or "drops no generated SAE" not in rl2[0]:
        bad.append(f"primary-dropped warning bullet wrong: {rl2}")
    trans = verdict_transitions(tp, ts)
    n_extra = sum(1 for k, t in fam_p.items() if (k[0], k[1]) in extra and t["verdict"] != extra[(k[0], k[1])])
    if sum(n for (a, b), n in trans.items() if b in ("collapsed", "baseline collapsed")) != n_extra:
        bad.append(f"verdict transitions to a collapse label {trans} != {n_extra} extra-collapsed tests")
    return len(tested_s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    ok, primary = s6_generate.primary_budget()
    p1 = base.read_json(base.p1_json()) or {}
    ps = per_seed_table()
    # One per-seed table, two gate versions (module docstring): each version is computed from the
    # rows of the SAEs IT keeps; the primary gate's tests are the headline, the strict
    # pre-registered gate's the sensitivity analysis.  BH runs inside apply_gates, i.e.
    # separately per version.
    gp, gs = gates43.all_verdicts(strict=False), gates43.all_verdicts(strict=True)
    gv = gated_versions(ps, primary, gp, gs)
    vp, vs = gv[gates43.PRIMARY], gv[gates43.STRICT]
    ps_p, idx, tests, tests_s = vp["ps"], vp["idx"], vp["tests"], vs["tests"]
    if vp["removed"]:
        print(f"[s9] WARNING: generated rows of {len(vp['removed'])} SAE(s) the primary gate "
              f"removes are left out of every table: {_seed_list(vp['removed'])}", file=sys.stderr)
    # per_seed.csv keeps EVERY generated row (the raw data); these two columns record which gate
    # version dropped a row and why, so either version can be re-derived from the CSV alone.
    for r in ps:
        k = (r["layer"], r["activation"], r["arm"], r["seed"])
        r["gate_removed_primary"] = vp["removed"].get(k)
        r["gate_removed_strict"] = vs["removed"].get(k)
    attr, attr_s = attribution(tests), attribution(tests_s)
    write_csv(base.p("s9", "per_seed.csv"), ps)
    write_csv(base.p("s9", "tests.csv"), tests)
    write_csv(base.p("s9", "tests_strict.csv"), tests_s)
    fam_tested = [t for t in tests if t["family"] and not t["gate_label"]]
    trans = verdict_transitions(tests, tests_s)
    p1_none = ok and primary is None
    md = [f"# A16 steering report ({'SMOKE' if base.IS_SMOKE else 'production'})", "",
          f"- Primary budget P1: {p1.get('primary_budget', 'not decided')} (PROTOCOL P1; applies to all layers, R7)",
          *([f"- **{P1_NONE_SENTENCE}**"] if p1_none else []),
          f"- Prompt file sha256 (P2): {base.PROMPTS_SHA256}",
          f"- Concept registry sha256: {concepts.concept_registry_sha256()}",
          *family_summary(tests, tests_s),
          *removed_lines(vp["removed"], vs["removed"]),
          "", "## D-43 health gate (final checkpoints, positions >= 1)", "",
          "Pair verdicts under both gate versions (the primary gate decided what was selected, "
          "calibrated, generated and judged):", "", gate_pair_table(gp, gs), "", gate_section(),
          "", f"### Family verdict changes, primary -> strict ({gates43.STRICT_LABEL})", "",
          (md_table(["primary verdict (headline)", "strict verdict (sensitivity)", "n family tests"],
                    [[a, b, n] for (a, b), n in sorted(trans.items())]) if trans else
           "No family test changes verdict between the two gate versions."),
          "", "## Dual branch diagnostics (C20, D-47)", "", dual_diag_section(),
          "", "## Selection: held-out detection AUROC of the 3 selected latents (seed mean +- sd)", "",
          selection_section(),
          "", "## Attribution (PROTOCOL 1.6) -- primary D-43 gate (headline)", "", _attr_table(attr),
          "", f"### Attribution (PROTOCOL 1.6) -- {gates43.STRICT_LABEL}", "", _attr_table(attr_s),
          "", "## Results per layer x pair x mode x budget (arms: mean +- sd over seeds)", "",
          results_tables(ps_p, tests, primary, idx=idx, tests_strict=tests_s,
                         idx_strict=vs["idx"], removed_strict=vs["removed"]),
          "", "## Arm-free references (DiffMean, random, unsteered null)", "", reference_tables(ps),
          "", "## Judge parse-failure / truncation rates (D-50)", "", judge_rates_section(),
          "", "## KL-calibration reachability (C29, PROTOCOL R5)", "", calibration_section(),
          "", "## Judge agreement: second judge and hand labels (D-50, PROTOCOL 4.7; exploratory)", "",
          kappa_section()]
    base.atomic_write_bytes(base.p("s9", "report.md"), "\n".join(md).encode())
    base.atomic_write_json(base.p("s9", "_done.json"), {"inputs": report_inputs()})
    fam_tested_s = [t for t in tests_s if t["family"] and not t["gate_label"]]
    n_only_s = sum(1 for k in vs["removed"] if k not in vp["removed"])
    print(f"[s9] {len(ps)} per-seed rows, {len(tests)} tests ({len(fam_tested)} confirmatory tested "
          f"under the primary gate, {len(fam_tested_s)} under the strict gate; {n_only_s} generated "
          f"SAE(s) dropped by the strict gate only) -> {base.p('s9', 'report.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
