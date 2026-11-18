"""The D-15 JumpReLU sparsity-match gate, evaluated once per run (PROTOCOL.md, D-15 and 1.5).

An arm's status 'ready' only means its files exist.  A JumpReLU DualSAE whose threshold does
not keep pace with the baseline's can end up many times denser than its baseline, and a Dual
vs baseline steering contrast between codes of very different sparsity is not interpretable.
So every JumpReLU arm is measured once, one record per run, and arms.contrast_ref() reads the
pair decision from those records.

Run gate (per JumpReLU run; every arm and seed):
    pass  iff  final log_threshold mean > ln(init) - 1   AND   eval L0 in [16, 512]
    * log_threshold mean: arithmetic mean over latents of ``jumprelu.log_threshold`` in
      autoencoder_final.pt (the final checkpoint is scored); init: training_config.json
      ``jump_threshold_init`` (a run without it is refused).
    * eval L0: mean per-row L0 of the run's final-checkpoint codes on the canonical gate rows:
      the run's layer, OpenWebText TEST split, the first ``D15_TRACKB_WINDOWS`` = 400 packed
      128-token windows, NON-SINK rows only (positions >= 1; D-19): 50,800 rows.  JumpReLU is
      per-row, so batching cannot change it.

Pair rules (PROTOCOL.md 1.5 and R3), per layer, over seeds 42-44:
    * An ARM fails when any of its seeds fails the run gate.
    * Arm L0 = the seed-mean eval L0 (D-01: seeds are the only axis averaged).  "L0s differ by
      more than 2x": max(L0_a, L0_b) / min(L0_a, L0_b) > 2 on these seed means; the per-seed
      worst ratio is recorded as a diagnostic, not used.
    * Dual-baseline contrast removed ("not sparsity-matched") when baseline or dual fails, or
      ratio(dual, baseline) > 2   ("the whole pair").
    * Dual-g0 contrast removed when the whole pair is removed, or dual_g0 fails, or
      ratio(dual, dual_g0) > 2.
    * A definitive failure is reported as soon as ANY evaluated record shows it; otherwise a
      contrast with an unevaluated run is 'pending' (never 'pass' by default).

Records: one JSON per run at ``<root>/d15/<tag>.json`` (root: ``$A16_D15_ROOT``, default
``$A16_ROOT/gates_d15``), written atomically.  A record is STALE -- treated as absent -- when
autoencoder_final.pt's size or mtime changed (a retrain) or its rows spec differs, so a
decision can never outlive the checkpoint it measured.

CLI (one GPU; seconds per run):
    python -m common.gates --evaluate [--device cuda:0]
    python -m common.gates --report
    python -m common.gates --selftest          # pair logic on synthetic records (CPU)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from common import arms as _arms

# ----------------------------------------------------------------------------- fixed thresholds
D15_L0_MIN = 16.0                 # eval L0 must lie in the CLOSED interval [16, 512]
D15_L0_MAX = 512.0
D15_LOG_THETA_MARGIN = 1.0        # final log_threshold mean must exceed ln(init) - 1
D15_MAX_L0_RATIO = 2.0            # pair L0s may differ by at most 2x
D15_LABEL = "not sparsity-matched"
# The rows spec string is stored in every record (a change of definition invalidates them).
D15_ROWS_SPEC = "d15-rows-v1: A=own corpus test all rows; B=owt test windows [0,400) non-sink"
D15_TRACKB_WINDOWS = 400
RECORD_ROOT = os.environ.get("A16_D15_ROOT", os.path.join(
    os.environ.get("A16_ROOT", os.path.join("outputs", "llm_steering")), "gates_d15"))

# Contrast keys, as in arms.CONTRASTS.
DUAL_BASELINE = ("dual", "baseline")
DUAL_G0 = ("dual", "dual_g0")


# ----------------------------------------------------------------------------- run level
def d15_run_verdict(log_threshold_mean: float, init: float, eval_l0: float
                    ) -> Tuple[bool, List[str]]:
    """The D-15 run gate on measured quantities.  Returns (passed, reasons for failure).

    NaN inputs fail (a quantity that could not be measured never passes a gate)."""
    reasons: List[str] = []
    if not (init > 0 and math.isfinite(init)):
        raise ValueError(f"threshold init must be a positive finite number, got {init!r}")
    floor = math.log(init) - D15_LOG_THETA_MARGIN
    if not (log_threshold_mean > floor):          # NaN compares False -> fails
        reasons.append(f"final log_threshold mean {log_threshold_mean:.3f} <= ln(init) - 1 = "
                       f"{floor:.3f} (init {init:g})")
    if not (D15_L0_MIN <= eval_l0 <= D15_L0_MAX):
        reasons.append(f"eval L0 {eval_l0:.1f} outside [{D15_L0_MIN:g}, {D15_L0_MAX:g}]")
    return (not reasons), reasons


def _ckpt_fingerprint(run_dir: Path) -> Dict[str, int]:
    st = (Path(run_dir) / _arms.FINAL_CKPT).stat()
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def threshold_init(run_dir: Path) -> float:
    """``jump_threshold_init`` from training_config.json.  Refuses a run without it: guessing a
    default would gate a run against the wrong floor."""
    with open(Path(run_dir) / "training_config.json") as fh:
        cfg = json.load(fh)
    v = cfg.get("jump_threshold_init")
    if v is None:
        raise ValueError(f"{run_dir}: training_config.json has no jump_threshold_init")
    return float(v)


def final_log_threshold(run_dir: Path) -> Dict[str, float]:
    """Mean (and spread) of the final checkpoint's per-latent log_threshold.  Reads only that
    tensor (torch.load with mmap maps the archive; no model is built)."""
    import torch
    obj = torch.load(Path(run_dir) / _arms.FINAL_CKPT, map_location="cpu", mmap=True,
                     weights_only=False)
    sd = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    keys = [k for k in sd if k.endswith("log_threshold")]
    if len(keys) != 1:
        raise ValueError(f"{run_dir}: expected exactly one *log_threshold tensor, found {keys}")
    t = sd[keys[0]].detach().float()
    return {"mean": float(t.mean()), "min": float(t.min()), "max": float(t.max()),
            "n_latents": int(t.numel()), "key": keys[0]}


def _gate_rows(ref: "_arms.ArmRef"):
    """The canonical D-15 rows of one arm (module docstring) and their description."""
    from common import sae_io
    layer = int(ref.key[1:])
    X, _ = sae_io.read_shard_rows(layer, "test", 0, D15_TRACKB_WINDOWS * 128, corpus=ref.corpus,
                                  drop_sinks=True)
    return X, (f"B L{layer}/{ref.corpus} test windows [0, {D15_TRACKB_WINDOWS}), non-sink, "
               f"{X.shape[0]} rows")


def measure_d15_run(ref: "_arms.ArmRef", device: str) -> Dict[str, Any]:
    """Measure one JumpReLU arm and return its D-15 record (not written)."""
    import numpy as np
    from common import sae_io
    if ref.activation != "jump_relu":
        raise ValueError(f"D-15 applies to JumpReLU arms only, got {ref.activation}")
    if ref.status != "ready":
        raise _arms.ArmPending(f"{ref.tag}: status {ref.status}, cannot be gated yet")
    t0 = time.time()
    fp_before = _ckpt_fingerprint(ref.run_dir)
    init = threshold_init(ref.run_dir)
    lt = final_log_threshold(ref.run_dir)
    model, meta = sae_io.load_arm(ref.run_dir, device)
    if meta["family"] != "jump_relu":
        raise RuntimeError(f"{ref.tag}: registry says jump_relu, checkpoint says {meta['family']}")
    X, rows_desc = _gate_rows(ref)
    z, l0 = sae_io.encode_rows(model, meta, X, device)
    st = sae_io.code_stats(z, meta["dict_size"])
    del model
    if _ckpt_fingerprint(ref.run_dir) != fp_before:
        # A retrain replaced the checkpoint while we measured: nothing here is trustworthy.
        raise RuntimeError(f"{ref.tag}: autoencoder_final.pt changed during measurement")
    passed, reasons = d15_run_verdict(lt["mean"], init, float(np.mean(l0)))
    return {
        "tag": ref.tag, "run_dir": str(ref.run_dir), "substrate": ref.substrate, "key": ref.key,
        "corpus": ref.corpus, "activation": ref.activation, "arm": ref.arm, "seed": ref.seed,
        "init": init, "ln_init": math.log(init),
        "log_threshold_mean": lt["mean"], "log_threshold_min": lt["min"],
        "log_threshold_max": lt["max"], "log_threshold_drift": lt["mean"] - math.log(init),
        "n_latents": lt["n_latents"],
        "eval_l0": float(np.mean(l0)), "eval_l0_median": float(np.median(l0)),
        "live_latents": st["live_latents"], "dead_pct": st["dead_pct"],
        "eval_rows": rows_desc, "n_rows": int(X.shape[0]), "rows_spec": D15_ROWS_SPEC,
        "passed": bool(passed), "reasons": reasons,
        "ckpt_fingerprint": fp_before, "checkpoint": _arms.FINAL_CKPT,
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 2),
    }


# ----------------------------------------------------------------------------- records
def _record_dir(root: Optional[str]) -> Path:
    return Path(root or RECORD_ROOT) / "d15"


def record_path(tag: str, root: Optional[str] = None) -> Path:
    return _record_dir(root) / f"{tag}.json"


def write_record(rec: Mapping[str, Any], root: Optional[str] = None) -> Path:
    """Atomic write: a killed job leaves either the old record or the new one, never a torn
    file (temp file in the same directory, fsync, os.replace)."""
    d = _record_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{rec['tag']}.json"
    fd, tmp = tempfile.mkstemp(prefix=f".{rec['tag']}.", suffix=".tmp", dir=str(d))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(rec, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o640)              # mkstemp creates 0600
        os.replace(tmp, out)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return out


def load_record(ref: "_arms.ArmRef", root: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The run's D-15 record, or None when absent, unreadable or STALE (the checkpoint changed
    since it was measured, or the row definition changed)."""
    if ref.tag is None or ref.run_dir is None:
        return None
    p = record_path(ref.tag, root)
    if not p.is_file():
        return None
    try:
        with open(p) as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return None
    if rec.get("rows_spec") != D15_ROWS_SPEC:
        return None
    try:
        if rec.get("ckpt_fingerprint") != _ckpt_fingerprint(ref.run_dir):
            return None
    except FileNotFoundError:
        return None
    return rec


# ----------------------------------------------------------------------------- pair level
def _ratio(a: float, b: float) -> float:
    lo, hi = min(a, b), max(a, b)
    if lo <= 0:
        return math.inf
    return hi / lo


def d15_pair_from_records(records: Mapping[str, Mapping[int, Optional[Mapping[str, Any]]]]
                          ) -> Dict[str, Any]:
    """Pure pair logic (module docstring) on {arm: {seed: record or None}}; arms may be absent
    (treated as unevaluated).  Returns

      arm_failed   {arm: True | False | None}   None = no failure seen but a seed is unevaluated
      arm_l0       {arm: seed-mean eval L0 or None (any seed unevaluated)}
      ratio_dual_baseline, ratio_dual_g0, worst_seed_ratio_dual_baseline / _dual_g0
      contrasts    {DUAL_BASELINE: (state, reasons), DUAL_G0: (state, reasons)},
                   state in 'pass' | 'fail' | 'pending'
    """
    seeds = list(_arms.SEEDS)
    arm_failed: Dict[str, Optional[bool]] = {}
    arm_l0: Dict[str, Optional[float]] = {}
    fail_reasons: Dict[str, List[str]] = {}
    for arm in _arms.ARMS:
        recs = records.get(arm, {})
        got = [recs.get(s) for s in seeds]
        bad = [(s, r) for s, r in zip(seeds, got) if r is not None and not r["passed"]]
        fail_reasons[arm] = [f"{arm} seed {s}: " + "; ".join(r["reasons"]) for s, r in bad]
        if bad:
            arm_failed[arm] = True
        elif any(r is None for r in got):
            arm_failed[arm] = None
        else:
            arm_failed[arm] = False
        arm_l0[arm] = (sum(r["eval_l0"] for r in got) / len(got)
                       if all(r is not None for r in got) else None)

    def ratio(a: str, b: str) -> Optional[float]:
        if arm_l0[a] is None or arm_l0[b] is None:
            return None
        return _ratio(arm_l0[a], arm_l0[b])

    def worst_seed_ratio(a: str, b: str) -> Optional[float]:
        ra, rb = records.get(a, {}), records.get(b, {})
        vals = [_ratio(ra[s]["eval_l0"], rb[s]["eval_l0"]) for s in seeds
                if ra.get(s) is not None and rb.get(s) is not None]
        return max(vals) if vals else None

    r_db, r_dg = ratio("dual", "baseline"), ratio("dual", "dual_g0")

    # Dual-baseline ("the whole pair").
    reasons_pair = fail_reasons["baseline"] + fail_reasons["dual"]
    if r_db is not None and r_db > D15_MAX_L0_RATIO:
        reasons_pair.append(f"dual/baseline L0 ratio {r_db:.2f} > {D15_MAX_L0_RATIO:g} "
                            f"(seed means {arm_l0['dual']:.1f} vs {arm_l0['baseline']:.1f})")
    if reasons_pair:
        st_db = ("fail", reasons_pair)
    elif arm_failed["baseline"] is None or arm_failed["dual"] is None:
        st_db = ("pending", ["D-15 not evaluated for every baseline/dual seed"])
    else:
        st_db = ("pass", [])

    # Dual-g0: the whole pair, plus dual_g0's own run gate and the dual/dual_g0 L0 match.
    reasons_g0 = list(reasons_pair) + fail_reasons["dual_g0"]
    if r_dg is not None and r_dg > D15_MAX_L0_RATIO:
        reasons_g0.append(f"dual/dual_g0 L0 ratio {r_dg:.2f} > {D15_MAX_L0_RATIO:g} "
                          f"(seed means {arm_l0['dual']:.1f} vs {arm_l0['dual_g0']:.1f})")
    if reasons_g0:
        st_dg = ("fail", reasons_g0)
    elif None in (arm_failed["baseline"], arm_failed["dual"], arm_failed["dual_g0"]):
        st_dg = ("pending", ["D-15 not evaluated for every baseline/dual/dual_g0 seed"])
    else:
        st_dg = ("pass", [])

    return {"arm_failed": arm_failed, "arm_l0": arm_l0,
            "ratio_dual_baseline": r_db, "ratio_dual_g0": r_dg,
            "worst_seed_ratio_dual_baseline": worst_seed_ratio("dual", "baseline"),
            "worst_seed_ratio_dual_g0": worst_seed_ratio("dual", "dual_g0"),
            "contrasts": {DUAL_BASELINE: st_db, DUAL_G0: st_dg}}


def d15_pair(substrate: str, key, corpus: Optional[str], *, root: Optional[str] = None
             ) -> Dict[str, Any]:
    """d15_pair_from_records on the fresh on-disk records of one JumpReLU pair."""
    records: Dict[str, Dict[int, Optional[Dict[str, Any]]]] = {}
    for arm in _arms.ARMS:
        records[arm] = {}
        for s in _arms.SEEDS:
            ref = _arms.resolve_arm(substrate, key, corpus, "jump_relu", arm, s)
            records[arm][s] = load_record(ref, root)
    out = d15_pair_from_records(records)
    out["records"] = records
    return out


def d15_contrast_state(substrate: str, key, corpus: Optional[str], contrast: Tuple[str, str], *,
                       root: Optional[str] = None) -> Tuple[str, Optional[str], List[str]]:
    """(state, label, reasons) of one JumpReLU contrast: state 'pass' | 'fail' | 'pending';
    label = "not sparsity-matched" iff state == 'fail'."""
    contrast = tuple(contrast)
    if contrast not in (DUAL_BASELINE, DUAL_G0):
        raise KeyError(f"contrast {contrast} not in {(DUAL_BASELINE, DUAL_G0)}")
    state, reasons = d15_pair(substrate, key, corpus, root=root)["contrasts"][contrast]
    return state, (D15_LABEL if state == "fail" else None), list(reasons)


# ----------------------------------------------------------------------------- CLI
def _evaluate(substrate: Optional[str], device: str, force: bool, root: Optional[str]) -> int:
    n_new = n_skip = n_wait = 0
    for ref in _arms.iter_arms(substrate, activations=["jump_relu"]):
        if ref.status != "ready":
            n_wait += 1
            continue
        if not force and load_record(ref, root) is not None:
            n_skip += 1
            continue
        rec = measure_d15_run(ref, device)
        write_record(rec, root)
        n_new += 1
        print(f"  {'PASS' if rec['passed'] else 'FAIL'} {ref.tag}: log-theta mean "
              f"{rec['log_threshold_mean']:.3f} (floor {rec['ln_init'] - 1:.3f}), eval L0 "
              f"{rec['eval_l0']:.1f} on {rec['n_rows']} rows  {'; '.join(rec['reasons'])}",
              flush=True)
    print(f"D-15 evaluate: {n_new} measured, {n_skip} fresh records kept, {n_wait} arms not ready")
    return 0


def _report(root: Optional[str]) -> int:
    print(f"D-15 records under {_record_dir(root)}  (L0 = mean eval L0 over the EVALUATED seeds, "
          f"n/3 in brackets; the gate itself uses all 3)")
    print(f"{'key':5s} {'L0 base':>12s} {'L0 dual':>12s} {'L0 g0':>12s} {'dual-base':>10s} "
          f"{'dual-g0':>10s}")
    for k in (f"L{n}" for n in _arms.TRACK_B_LAYERS):
        p = d15_pair("B", k, _arms.TRACK_B_CORPUS, root=root)

        def cell(arm: str) -> str:
            got = [r["eval_l0"] for r in p["records"][arm].values() if r is not None]
            return f"{sum(got) / len(got):.1f} ({len(got)}/3)" if got else "-"

        print(f"{k:5s} {cell('baseline'):>12s} {cell('dual'):>12s} {cell('dual_g0'):>12s} "
              f"{p['contrasts'][DUAL_BASELINE][0]:>10s} {p['contrasts'][DUAL_G0][0]:>10s}")
    return 0


def _selftest(device: str) -> int:
    failures: List[str] = []

    def check(cond: bool, msg: str) -> None:
        print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
        if not cond:
            failures.append(msg)

    # 1. Run gate thresholds (closed L0 interval, strict log-theta floor).
    ln1 = math.log(1.0)
    check(d15_run_verdict(0.3, 1.0, 120.0)[0], "healthy run passes (log-theta 0.3, L0 120)")
    check(not d15_run_verdict(ln1 - 1.0, 1.0, 100.0)[0], "log-theta mean == ln(init) - 1 fails (strict >)")
    check(d15_run_verdict(0.0, 1.0, 16.0)[0] and d15_run_verdict(0.0, 1.0, 512.0)[0],
          "L0 exactly 16 and 512 pass (closed interval)")
    ok, why = d15_run_verdict(-5.21, 0.00518, 1219.4)
    check(not ok and len(why) == 1 and "eval L0" in why[0], f"L0 1219 fails: {why}")
    check(not d15_run_verdict(float("nan"), 1.0, 100.0)[0], "NaN log-theta never passes")

    # 2. Pair logic on synthetic records.
    def rec(l0: float, passed: bool = True) -> Dict[str, Any]:
        return {"eval_l0": l0, "passed": passed, "reasons": [] if passed else ["synthetic"]}

    S = list(_arms.SEEDS)
    ok_pair = {a: {s: rec(100.0) for s in S} for a in _arms.ARMS}
    r = d15_pair_from_records(ok_pair)
    check(r["contrasts"][DUAL_BASELINE][0] == "pass" and r["contrasts"][DUAL_G0][0] == "pass",
          "healthy pair: both contrasts pass")
    p2 = {a: dict(v) for a, v in ok_pair.items()}
    p2["dual_g0"] = {s: rec(250.0) for s in S}
    r = d15_pair_from_records(p2)
    check(r["contrasts"][DUAL_BASELINE][0] == "pass" and r["contrasts"][DUAL_G0][0] == "fail",
          f"dual/dual_g0 ratio 2.5 removes ONLY the Dual-g0 contrast ({r['ratio_dual_g0']:.2f})")
    p3 = {a: dict(v) for a, v in ok_pair.items()}
    p3["baseline"] = {s: rec(40.0) for s in S}
    r = d15_pair_from_records(p3)
    check(r["contrasts"][DUAL_BASELINE][0] == "fail" and r["contrasts"][DUAL_G0][0] == "fail",
          "dual/baseline ratio 2.5 removes the whole pair (both contrasts)")
    p4 = {a: dict(v) for a, v in ok_pair.items()}
    p4["dual_g0"] = {42: rec(100.0, passed=False), 43: rec(100.0), 44: rec(100.0)}
    r = d15_pair_from_records(p4)
    check(r["contrasts"][DUAL_BASELINE][0] == "pass" and r["contrasts"][DUAL_G0][0] == "fail",
          "one failing dual_g0 seed removes only the Dual-g0 contrast")
    p5 = {"baseline": {42: rec(27.5)}, "dual": {42: rec(1219.4, passed=False)},
          "dual_g0": {42: rec(1531.9, passed=False)}}
    r = d15_pair_from_records(p5)
    check(r["contrasts"][DUAL_BASELINE][0] == "fail" and r["arm_l0"]["dual"] is None,
          "a failure is definitive even while other seeds are unevaluated")
    p6 = {"baseline": {42: rec(100.0)}, "dual": {42: rec(110.0)}}
    r = d15_pair_from_records(p6)
    check(r["contrasts"][DUAL_BASELINE][0] == "pending" and r["contrasts"][DUAL_G0][0] == "pending",
          "no failure yet but unevaluated seeds -> pending (never pass by default)")
    # Every arm evaluated at seed 42 only, all healthy: 'pending' for both contrasts -- a single
    # seed can never make a pair pass (all three seeds are required).
    p7 = {a: {42: rec(v)} for a, v in (("baseline", 26.3), ("dual", 40.0), ("dual_g0", 45.0))}
    r = d15_pair_from_records(p7)
    check(r["contrasts"][DUAL_BASELINE][0] == "pending" and r["contrasts"][DUAL_G0][0] == "pending",
          "healthy seed-42-only records (all 3 arms) -> both contrasts pending, never pass")
    check(d15_pair_from_records({a: {s: rec(0.0) for s in S} for a in _arms.ARMS})
          ["contrasts"][DUAL_BASELINE][0] == "fail", "L0 0 fails (and a 0 L0 ratio is infinite)")

    # 3. Record round trip and staleness on a synthetic record (no checkpoint needed).
    with tempfile.TemporaryDirectory() as td:
        rc = {"tag": "synthetic_tag", "rows_spec": D15_ROWS_SPEC, "eval_l0": 50.0}
        write_record(rc, td)
        with open(record_path("synthetic_tag", td)) as fh:
            check(json.load(fh)["eval_l0"] == 50.0, "atomic record write round-trips")
        check(not list(Path(td, "d15").glob(".*.tmp")), "no temp files left by atomic writes")
    print("GATES SELFTEST", "OK" if not failures else f"FAILED ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaluate", action="store_true",
                    help="measure every ready JumpReLU arm without a fresh record")
    ap.add_argument("--report", action="store_true", help="per-layer D-15 table")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true", help="re-measure even with a fresh record")
    ap.add_argument("--root", default=None, help=f"record root (default {RECORD_ROOT})")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest(a.device))
    if a.evaluate:
        sys.exit(_evaluate("B", a.device, a.force, a.root))
    if a.report:
        sys.exit(_report(a.root))
    ap.print_help()
