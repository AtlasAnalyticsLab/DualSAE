"""s5: KL-budget calibration per SAE x direction x mode x budget (D-48, C29).

Procedure (C29, pre-registered):
  * 32 owt TEST windows (base.CALIB_START_WINDOW.., disjoint from the s0/s2 rows), editing and
    scoring positions 1..127, clean log-probs cached; teacher-forced next-token KL(clean || edited)
    averaged over windows and positions (steer.WindowKL, which applies the edit with the SAME
    EditHook code the generation hook runs).
  * parameter offset s >= 0 per direction:
      mode (i)            alpha = s                    (shard units; KL(0) = 0)
      modes (ii')/(iii)   c = c0 + s, c0 = the latent's 99th-percentile POSITIVE activation on the
                          calibration windows (the monotone branch: KL is not monotone in c near
                          the natural activation, where the edit vanishes; C29).  A latent that
                          never fires there has c0 = 0 (recorded).
    log-grid s = CALIB_GRID x scale, scale = m (the layer's median non-sink token norm, shard
    space) for (i) and m / ||W_dec[j]|| for (ii')/(iii) so that a grid step moves the residual by
    the same amount for every latent; the cap is s <= 1e3 x scale ("alpha and c capped at 1e3 x
    the median non-sink token norm").  The scan stops early once every direction of the group is
    above the largest budget.
  * per budget (ascending), the tightest bracket among ALL points evaluated so far ("reusing
    brackets across budgets"; first crossing in s), then 8 bisection steps (geometric midpoint;
    halving while the lower end is s = 0).  The reported parameter is the evaluated bracket end
    closest to the budget in |log(KL / budget)|.
  * 'unreachable_cap' when KL at the cap stays below the budget (mode (iii) cancellation can do
    this), 'unreachable_branch_start' when KL at c0 already exceeds the budget (ii'/iii); both are
    recorded, never looped.
  * up to 16 directions per forward pass (rows d*32 .. d*32+31 carry direction d).

Units (jobs.py kind 's5'): one per SAE (arms.iter_arms order) and one arm-free unit per layer.
  SAE unit: mode (i) for the D-46 selected latents AND the live-matched latents (deduplicated by
  latent), mode (ii') for the selected latents, mode (iii) for the selected latents of Dual arms.
  Arm-free unit: DiffMean (10) and the 5 random directions per concept (50), mode (i).
  armfree_L<PILOT_LAYER> is OWNED by the s4a pilot (it calibrates it before generating); the
  planner never schedules it as an s5 task, so the pilot and the headline read one record.
Every budget is calibrated for every direction (the primary budget P1 is not needed here, so s5
runs while the pilot is being judged).  Declared: the live-matched latents are therefore also
calibrated at all 6 budgets although only P1 is generated (D-46 names the primary budget only);
it costs ~2 groups of passes per SAE and keeps s5 independent of P1.
Resumability inside a unit: after every 16-direction group the results so far are written
atomically to s5/<unit>.partial.json; a restarted task skips every direction already there
(directions are calibrated independently of their group mates).  A COMPLETE record that lacks
directions the current selection needs (the live-matched refresh of s1) is extended with just
those directions; existing entries are never recomputed (is_done checks coverage).
Precondition (mode ii'): the SAE decoder must be affine in z (SAEAdapter.check_affine); a
relative violation above AFFINE_MAX_REL aborts the unit instead of calibrating silently.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from steering import base, concepts, gates43, s0_rescore, s1_select, steer
from common import sae_io

AFFINE_MAX_REL = 1e-4        # fp32 decode rounding is ~1e-7 relative; non-affine is O(1)


@dataclass
class Dir:
    key: str                    # unique within the unit, e.g. 'i|j6733' or 'i|diffmean|agnews_world'
    mode: str                   # 'i' | 'ii' | 'iii'
    vec: Optional[np.ndarray] = None     # unit vector (mode i)
    latent: Optional[int] = None         # latent index (SAE directions)
    c0: float = 0.0             # branch start (ii'/iii)
    scale: float = 1.0          # grid unit for s
    meta: Dict[str, Any] = field(default_factory=dict)


def _configure(hook: steer.EditHook, dirs: Sequence[Dir], s_vals: np.ndarray, active: np.ndarray,
               nf: float, n_rep: int, device) -> None:
    """Per-row hook parameters for evaluating direction d at offset s_vals[d]."""
    mode = dirs[0].mode
    act = torch.as_tensor(np.repeat(active, n_rep), device=device)
    if mode == "i":
        U = torch.as_tensor(np.stack([d.vec for d in dirs]), device=device, dtype=torch.float32)
        delta = U * torch.as_tensor(s_vals, device=device, dtype=torch.float32)[:, None] * nf
        hook.configure("add", delta_raw=steer.rows_for(delta, n_rep), active=act)
    else:
        j = torch.as_tensor([d.latent for d in dirs], device=device, dtype=torch.long)
        c = torch.as_tensor([d.c0 + s for d, s in zip(dirs, s_vals)], device=device, dtype=torch.float32)
        hook.configure("clamp" if mode == "ii" else "rerun", j=steer.rows_for(j, n_rep),
                       c=steer.rows_for(c, n_rep), active=act)


def calibrate_group(wkl: steer.WindowKL, hook: steer.EditHook, dirs: List[Dir], nf: float,
                    budgets: Sequence[float] = base.BUDGETS) -> Dict[str, Dict[str, Any]]:
    """Run the C29 search for <= DIRS_PER_PASS directions of ONE mode."""
    n = len(dirs)
    if n == 0:
        return {}
    if n > base.DIRS_PER_PASS or len({d.mode for d in dirs}) != 1:
        raise ValueError("a group holds <= 16 directions of one mode")
    dev = wkl.dev
    W = wkl.W
    evals: List[List[Tuple[float, float]]] = [[] for _ in range(n)]
    ones = np.ones(n, bool)

    def eval_at(s_vals: np.ndarray, active: np.ndarray) -> np.ndarray:
        _configure(hook, dirs, s_vals, active, nf, W, dev)
        return wkl.kl(hook, n)

    if dirs[0].mode == "i":
        for d in range(n):
            evals[d].append((0.0, 0.0))                   # alpha = 0 is the clean model
    else:
        kl0 = eval_at(np.zeros(n), ones)
        for d in range(n):
            evals[d].append((0.0, float(kl0[d])))
    cap = np.array([base.CAP_MULT * d.scale for d in dirs])
    top_b = max(budgets)
    curve = np.full((n, len(base.CALIB_GRID)), np.nan)
    for k, g in enumerate(base.CALIB_GRID):
        s_vals = np.array([g * d.scale for d in dirs])
        klk = eval_at(s_vals, ones)
        curve[:, k] = klk
        for d in range(n):
            evals[d].append((float(s_vals[d]), float(klk[d])))
        if all(max(kv for _, kv in evals[d]) >= top_b for d in range(n)):
            break
    out: Dict[str, Dict[str, Any]] = {d.key: {"mode": d.mode, "latent": d.latent, "c0": d.c0,
                                              "scale": d.scale, "budgets": {}, **d.meta}
                                      for d in dirs}
    for d in range(n):
        out[dirs[d].key]["grid_kl"] = [None if not np.isfinite(v) else float(v) for v in curve[d]]
    for b in sorted(budgets):
        lo = np.zeros(n)
        hi = np.zeros(n)
        klo = np.zeros(n)
        khi = np.zeros(n)
        status = ["ok"] * n
        for d in range(n):
            pts = sorted(evals[d])
            idx = next((i for i, (_, kv) in enumerate(pts) if kv >= b), None)
            if idx is None:
                status[d] = "unreachable_cap"
            elif idx == 0:
                status[d] = "unreachable_branch_start" if dirs[d].mode != "i" else "ok"
                lo[d], klo[d], hi[d], khi[d] = 0.0, 0.0, pts[0][0], pts[0][1]
            else:
                (lo[d], klo[d]), (hi[d], khi[d]) = pts[idx - 1], pts[idx]
        active = np.array([st == "ok" for st in status])
        for _ in range(base.N_BISECT):
            if not active.any():
                break
            mid = np.where(lo > 0, np.sqrt(np.maximum(lo, 1e-300) * np.maximum(hi, 1e-300)), hi / 2.0)
            km = eval_at(np.where(active, mid, 0.0), active)
            for d in range(n):
                if not active[d]:
                    continue
                evals[d].append((float(mid[d]), float(km[d])))
                if km[d] >= b:
                    hi[d], khi[d] = mid[d], km[d]
                else:
                    lo[d], klo[d] = mid[d], km[d]
        for d in range(n):
            key = dirs[d].key
            if status[d] != "ok":
                out[key]["budgets"][f"{b:g}"] = {"status": status[d]}
                continue
            cand = [(hi[d], khi[d])] + ([(lo[d], klo[d])] if klo[d] > 0 else [])
            s_best, k_best = min(cand, key=lambda t: abs(math.log(t[1] / b)))
            param = s_best if dirs[d].mode == "i" else dirs[d].c0 + s_best
            out[key]["budgets"][f"{b:g}"] = {
                "status": "ok", "s": float(s_best), "param": float(param), "kl": float(k_best),
                "rel_err": float(k_best / b - 1.0), "bracket": [float(lo[d]), float(hi[d])]}
    for d in range(n):
        out[dirs[d].key]["n_evals"] = len(evals[d])
    return out


# ----------------------------------------------------------------------------- units
def armfree_dirs(layer: int, m: float) -> List[Dir]:
    af = s1_select.load_armfree(layer)
    dirs = []
    for ci, cid in enumerate(af["concept_ids"]):
        cid = str(cid)
        dirs.append(Dir(f"i|diffmean|{cid}", "i", vec=af["diffmean"][ci].astype(np.float32),
                        scale=m, meta={"dirset": "diffmean", "concept": cid, "rank": 0}))
        for r in range(af["random"].shape[1]):
            dirs.append(Dir(f"i|random|{cid}|r{r}", "i", vec=af["random"][ci, r].astype(np.float32),
                            scale=m, meta={"dirset": "random", "concept": cid, "rank": r}))
    return dirs


def _uses(sel: Dict[str, Any], is_dual: bool, modes: Sequence[str]
          ) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    """(mode, latent) -> the (dirset, concept, rank) slots that use it: every selected latent
    (full + live-matched) in mode (i); the full selection in (ii') and, for Duals, (iii)."""
    uses: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for cid, e in sel["concepts"].items():
        for r, j in enumerate(e["latents"]):
            for mode in modes:
                if mode == "iii" and not is_dual:
                    continue
                uses.setdefault((mode, int(j)), []).append({"dirset": "full", "concept": cid, "rank": r})
        lm = e.get("livematched", {})
        if lm.get("status") == "ok" and "i" in modes:
            for r, j in enumerate(lm["latents"]):
                uses.setdefault(("i", int(j)), []).append({"dirset": "livematched", "concept": cid, "rank": r})
    return uses


def needed_keys(unit: str) -> Optional[List[str]]:
    """Direction keys the unit's CURRENT inputs require (None when the inputs are missing).
    Cheap (no model load): the planner's coverage check."""
    if unit.startswith("armfree_L"):
        meta = base.read_json(base.s1_armfree(int(unit.split("L")[1])).replace(".npz", ".json"))
        if not meta:
            return None
        return ([f"i|diffmean|{c}" for c in meta["concept_ids"]]
                + [f"i|random|{c}|r{r}" for c in meta["concept_ids"] for r in range(meta["n_random"])])
    sel = s1_select.load_selection(unit)
    if not sel:
        return None
    ref = next(r for r in base.sae_refs() if r.tag == unit)
    return [f"{m}|j{j}" for (m, j) in _uses(sel, ref.arm in ("dual", "dual_g0"), base.MODES_ALL)]


def sae_dirs(sel: Dict[str, Any], A: steer.SAEAdapter, m: float, c0: Dict[int, float],
             is_dual: bool, modes: Sequence[str]) -> List[Dir]:
    """Deduplicated directions of one SAE (see _uses); meta lists which (dirset, concept, rank)
    slots use each direction."""
    uses = _uses(sel, is_dual, modes)
    dirs = []
    for (mode, j), u in sorted(uses.items()):
        if mode == "i":
            dirs.append(Dir(f"i|j{j}", "i", vec=A.U[j].cpu().numpy(), latent=j, scale=m,
                            meta={"uses": u, "atom_norm": float(A.atom_norm[j])}))
        else:
            dirs.append(Dir(f"{mode}|j{j}", mode, latent=j, c0=float(c0.get(j, 0.0)),
                            scale=m / float(A.atom_norm[j]),
                            meta={"uses": u, "atom_norm": float(A.atom_norm[j]),
                                  "c0_never_active": bool(c0.get(j, 0.0) == 0.0)}))
    return dirs


def partial_path(unit: str) -> str:
    return base.s5_record(unit).replace(".json", ".partial.json")


def p99_positive(A: steer.SAEAdapter, x_s: torch.Tensor, latents: Sequence[int]) -> Dict[int, float]:
    z = A.encode(x_s)
    out = {}
    for j in sorted(set(int(v) for v in latents)):
        v = z[:, j]
        v = v[v > 0]
        out[j] = float(torch.quantile(v.double(), 0.99)) if v.numel() else 0.0
    return out


def run_unit(unit: str, device: str, *, modes: Sequence[str] = base.MODES_ALL,
             max_dirs: int = 0, budgets: Sequence[float] = base.BUDGETS,
             lm=None, reuse: bool = True) -> Dict[str, Any]:
    """Calibrate every direction of one unit.  ``reuse`` (default) keeps the entries of a
    previous complete record / partial checkpoint and calibrates only the missing directions;
    reuse=False (the CLI's --force) recomputes everything from scratch."""
    t0 = time.time()
    refs = {r.tag: r for r in base.sae_refs()}
    if unit.startswith("armfree_L"):
        layer, ref = int(unit.split("L")[1]), None
    else:
        ref = refs[unit]
        layer = int(ref.key[1:])
    lm = lm or steer.load_lm(device)[0]
    nf, _ = steer.norm_factor(layer)
    wkl = steer.WindowKL(lm, layer, steer.owt_windows(base.CALIB_START_WINDOW, base.CALIB_WINDOWS))
    x_s = wkl.shard_rows(nf)
    m = float(x_s.norm(dim=1).median())
    # Earlier work on this unit: a complete record (extended when the selection grew) or the
    # partial checkpoint of a killed task.  Existing direction entries are reused verbatim.
    prev = base.read_json(base.s5_record(unit)) if reuse else None
    part = base.read_json(partial_path(unit)) if reuse else None
    results: Dict[str, Any] = {}
    n_forward_prev = 0
    for old in (part, prev):
        if old and old.get("budgets") == list(budgets):
            results.update(old.get("directions", {}))
            n_forward_prev = max(n_forward_prev, int(old.get("n_forward", 0)))
    rec: Dict[str, Any] = {"unit": unit, "layer": layer, "nf": nf, "median_nonsink_norm_calib": m,
                           "calib_windows": [base.CALIB_START_WINDOW,
                                             base.CALIB_START_WINDOW + base.CALIB_WINDOWS],
                           "partial_forward_max_abs_diff": wkl.partial_forward_max_abs_diff,
                           "budgets": list(budgets), "smoke": base.IS_SMOKE}
    if ref is None:
        dirs = armfree_dirs(layer, m)
        adapter = None
    else:
        model, meta = sae_io.load_arm(ref.run_dir, device)
        thr = base.read_json(base.s0_threshold(ref.tag))["threshold"] if meta["family"] == "batch_topk" else None
        adapter = steer.SAEAdapter(model, meta, thr)
        sel = s1_select.load_selection(ref.tag)
        all_lat = [j for e in sel["concepts"].values() for j in e["latents"]]
        aff = adapter.check_affine(x_s[:256], latents=all_lat[:3])
        rec["affine_check_max_abs"], rec["affine_check_max_rel"] = aff["max_abs"], aff["max_rel"]
        if aff["max_rel"] > AFFINE_MAX_REL:
            # Mode (ii') assumes x moves by exactly nf * dz * W[j]; a non-affine decoder would be
            # calibrated and generated with the wrong edit, silently.
            raise RuntimeError(f"{unit}: decoder not affine in z (relative error "
                               f"{aff['max_rel']:.2e} > {AFFINE_MAX_REL:g}); mode (ii') invalid")
        c0 = p99_positive(adapter, x_s, all_lat)
        dirs = sae_dirs(sel, adapter, m, c0, adapter.is_dual, modes)
        rec.update(tag=ref.tag, activation=ref.activation, arm=ref.arm, seed=ref.seed)
    if max_dirs:
        if not base.IS_SMOKE:
            raise RuntimeError("max_dirs is a smoke option")
        keep, per_mode = [], {}
        for d in dirs:
            if per_mode.get(d.mode, 0) < max_dirs:
                keep.append(d)
                per_mode[d.mode] = per_mode.get(d.mode, 0) + 1
        dirs = keep
        rec["max_dirs"] = int(max_dirs)
    todo = [d for d in dirs if d.key not in results]
    rec["n_directions_reused"] = sum(d.key in results for d in dirs)
    hook = steer.EditHook(lm, layer, nf, adapter)
    for mode in ("i", "ii", "iii"):
        md = [d for d in todo if d.mode == mode]
        for g in range(0, len(md), base.DIRS_PER_PASS):
            results.update(calibrate_group(wkl, hook, md[g:g + base.DIRS_PER_PASS], nf, budgets))
            # Per-group checkpoint: a timeout loses at most one group (~1 min), not the unit.
            base.atomic_write_json(partial_path(unit), dict(
                rec, directions=results, n_forward=n_forward_prev + wkl.n_forward, complete=False))
    rec["directions"] = results
    rec["n_directions"] = len(results)
    rec["n_forward"] = n_forward_prev + wkl.n_forward
    n_unreach = sum(1 for v in results.values() for b in v["budgets"].values() if b["status"] != "ok")
    rec["n_unreachable"] = n_unreach
    rec["seconds"] = round(time.time() - t0, 1) + float((prev or {}).get("seconds", 0.0))
    if prev and prev.get("complete") and todo:
        rec["incremental_additions"] = sorted(d.key for d in todo)
    rec["complete"] = True
    base.atomic_write_json(base.s5_record(unit), rec)
    if os.path.isfile(partial_path(unit)):
        os.unlink(partial_path(unit))
    return rec


def is_done(unit: str) -> bool:
    """Complete record that covers every direction the unit's current inputs need (a smoke
    record made with --max-dirs covers a subset on purpose and counts as complete)."""
    r = base.read_json(base.s5_record(unit))
    if not (r and r.get("complete")):
        return False
    if r.get("max_dirs"):
        return True                  # a smoke record calibrated a subset on purpose
    need = needed_keys(unit)
    return need is not None and all(k in r.get("directions", {}) for k in need)


def lookup(rec: Dict[str, Any], key: str, budget: float) -> Dict[str, Any]:
    """The calibrated entry of one direction at one budget ({'status': ...})."""
    d = rec["directions"].get(key)
    if d is None:
        return {"status": "not_calibrated"}
    return d["budgets"].get(f"{budget:g}", {"status": "not_calibrated"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, help="jobs.py s5 index (SAE units, then arm-free L6/12/18)")
    ap.add_argument("--unit", help="SAE tag or armfree_L<n>")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-dirs", type=int, default=0, help="smoke only: directions per mode")
    ap.add_argument("--budgets", type=float, nargs="*", default=list(base.BUDGETS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    unit = a.unit if a.unit else base.gen_units()[a.index]
    if unit == f"armfree_L{base.PILOT_LAYER}" and not base.IS_SMOKE and not a.force:
        # Owned by the s4a pilot (module docstring): never calibrated twice.
        print(f"[s5] {unit}: owned by s4a (the pilot calibrates it); not run as an s5 task")
        return 3
    if not unit.startswith("armfree_L"):
        ref = next(r for r in base.sae_refs() if r.tag == unit)
        why = base.held_reason(ref)
        if why:
            print(f"[s5] {unit}: not runnable -- {why}")
            return 3
    if is_done(unit) and not a.force:
        print(f"[s5] {unit}: already done")
        return 0
    rec = run_unit(unit, a.device or base.device_default(), max_dirs=a.max_dirs, budgets=a.budgets,
                   reuse=not a.force)
    print(f"[s5] {unit}: {rec['n_directions']} directions, {rec['n_forward']} passes, "
          f"{rec['n_unreachable']} unreachable (dir x budget), {rec['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
