"""s6: steered generation (D-49, C19, C30) + the judge-free metrics (s7m, merged; D-51) for
one generation unit (jobs.py kind 's6').

Units (base.gen_units()): one per SAE, one arm-free unit per layer (DiffMean + random), 'null'.
Cells per unit (a cell = one shard file, written atomically once complete -> skip-guard):
  SAE unit       mode (i)  'full'        every budget   (primary budget: 60 prompts x 2, else 20 x 1)
                 mode (i)  'livematched' primary budget only (60 x 2; D-46 secondary row), only
                                         once s1 has the live-matched selection (status ok)
                 mode (ii') 'full'       every budget (20 x 1)
                 mode (iii) 'full'       every budget, Dual arms only (20 x 1)
  arm-free unit  mode (i)  'diffmean', 'random' (5 per concept)   every budget, 20 x 1
  null           unsteered continuation of every prompt (60 x 2), no hook
Headline items (60 x 2) = D-49 "headline cells (mode (i), primary budget, all arms, including the
live-matched row)": mode (i) at P1 for the SAE arms' 'full' and 'livematched' selections, plus
the null row (the reference every headline item is read against).  Declared reading: DiffMean
and random directions are REFERENCES, not arms (PROTOCOL 1.1 arms = baseline / dual / dual_g0), so
they stay 20 x 1 at every budget -- which makes armfree_L<PILOT_LAYER> exactly the pilot's cell
plan: its shards are COPIED from the pilot (same calibration record, same item seeds, same
prompts), so the P1 decision and the reported reference rows are the same generations.
The primary budget comes from the P1 JSON written by s4p (PROTOCOL P1).  If P1 = "none" every cell
is 20 x 1 (PROTOCOL 4.6: the A16 family is empty and everything is exploratory).  The prompts are
the pinned P2 file (base.check_prompts refuses any other file); the 20-prompt set = its first 20
lines (R6).
Every edit follows D-47 via steer.EditHook: positions with position_ids >= 1 AND 2-D mask 1, KV
cache, left padding with pad = eos, fp32 LM, 16 + 48 <= 128 positions.  A direction whose budget
is unreachable (s5) contributes a row with status 'unreachable' and no text (PROTOCOL R5).
Seeding (D-49 key, per item): derive_seed_parts('a16.gen', mode, budget, concept, rank,
prompt_idx) -> the item's own generator -> uniforms for its n continuations (steer module
docstring).  The key has no arm and no layer, so every arm (and DiffMean/random/null) of the same
(mode, budget, concept, rank, prompt) draws the same random numbers.
Hard precondition: identity_test.run() for the unit's layer at job start; a failure aborts.
Merged s7m (D-51 "per-cell 1:1 stages are merged into one task"): after its shards are
generated the same task scores them with s7_metrics.unit_metrics, reusing the loaded pythia, so
the unit is DONE only when every shard has both its generation and its metric file.  The
classifier (s7c) must exist first; the planner holds s6 until it does.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from steering import base, calibrate, concepts, identity_test, s1_select, steer
from common import sae_io, seeds as _seeds


# ----------------------------------------------------------------------------- P1
def primary_budget() -> Tuple[bool, Optional[float]]:
    """(P1 decided?, primary budget or None for 'none')."""
    r = base.read_json(base.p1_json())
    if not r or "primary_budget" not in r:
        return False, None
    return True, (None if r["primary_budget"] in (None, "none") else float(r["primary_budget"]))


def item_seed(mode: str, budget: float, concept: str, rank: int, prompt_idx: int) -> int:
    return _seeds.derive_seed_parts("a16.gen", mode, float(budget), concept, int(rank), int(prompt_idx))


# ----------------------------------------------------------------------------- cell plan
def livematched_ready(unit: str) -> bool:
    """The unit's s1 selection carries the live-matched latents (n_match was computable)."""
    from steering import s1_select
    sel = s1_select.load_selection(unit)
    return bool(sel) and any(e.get("livematched", {}).get("status") == "ok"
                             for e in sel["concepts"].values())


def cell_plan(unit: str, primary: Optional[float]) -> List[Dict[str, Any]]:
    """Every shard of a unit: name, mode, dirset, budget, n_prompts, n_cont, headline flag."""
    cells = []

    def add(mode, dirset, b):
        # 60 x 2 ONLY for mode (i) at P1 on the SAE arms' selections (module docstring).
        head = (primary is not None and mode == "i" and dirset in ("full", "livematched")
                and abs(b - primary) < 1e-12)
        cells.append({"name": f"{mode}_{dirset}_b{b:g}", "mode": mode, "dirset": dirset,
                      "budget": b, "headline": head,
                      "n_prompts": base.N_PROMPTS_HEADLINE if head else base.N_PROMPTS_OTHER,
                      "n_cont": base.N_CONT_HEADLINE if head else base.N_CONT_OTHER})
    if unit == "null":
        cells.append({"name": "null", "mode": "null", "dirset": "null", "budget": 0.0,
                      "headline": True, "n_prompts": base.N_PROMPTS_HEADLINE,
                      "n_cont": base.N_CONT_HEADLINE})
        return cells
    if unit.startswith("armfree_L"):
        for ds in ("diffmean", "random"):
            for b in base.BUDGETS:
                add("i", ds, b)
        return cells
    ref = next(r for r in base.sae_refs() if r.tag == unit)
    for b in base.BUDGETS:
        add("i", "full", b)
    if primary is not None and livematched_ready(unit):
        add("i", "livematched", primary)
    for b in base.BUDGETS:
        add("ii", "full", b)
    if ref.arm in ("dual", "dual_g0"):
        for b in base.BUDGETS:
            add("iii", "full", b)
    return cells


def shard_path(unit: str, cell_name: str) -> str:
    return os.path.join(base.s6_dir(unit), f"{cell_name}.jsonl")


def generation_done(unit: str) -> bool:
    """Every shard of the unit's current cell plan exists (the s7j input condition)."""
    ok, prim = primary_budget()
    if not ok:
        return False
    return all(os.path.isfile(shard_path(unit, c["name"])) for c in cell_plan(unit, prim))


def unit_done(unit: str) -> bool:
    """The merged s6 task is done: generation AND the judge-free metrics of every shard."""
    from steering import s7_metrics
    return generation_done(unit) and s7_metrics.unit_done(unit)


# ----------------------------------------------------------------------------- slots
def cell_slots(unit: str, cell: Dict[str, Any], cal: Optional[Dict[str, Any]],
               sel: Optional[Dict[str, Any]], af_meta: Optional[List[str]]) -> List[Dict[str, Any]]:
    """The (concept, rank) direction slots of a cell with their calibrated parameter."""
    slots = []
    b = cell["budget"]
    if cell["mode"] == "null":
        return [{"concept": "-", "rank": 0, "dir_key": None, "latent": None,
                 "cal": {"status": "ok", "param": 0.0, "kl": 0.0}}]
    if cell["dirset"] in ("diffmean", "random"):
        for cid in af_meta:
            ranks = [0] if cell["dirset"] == "diffmean" else list(range(base.N_RANDOM))
            for r in ranks:
                key = f"i|diffmean|{cid}" if cell["dirset"] == "diffmean" else f"i|random|{cid}|r{r}"
                slots.append({"concept": cid, "rank": r, "dir_key": key, "latent": None,
                              "cal": calibrate.lookup(cal, key, b)})
        return slots
    for cid, e in sel["concepts"].items():
        if cell["dirset"] == "livematched":
            lm = e.get("livematched", {})
            lat = lm.get("latents", []) if lm.get("status") == "ok" else []
        else:
            lat = e["latents"]
        for r, j in enumerate(lat):
            key = f"{cell['mode']}|j{j}"
            slots.append({"concept": cid, "rank": r, "dir_key": key, "latent": int(j),
                          "cal": calibrate.lookup(cal, key, b)})
    return slots


# ----------------------------------------------------------------------------- run
@torch.no_grad()
def run_cells(unit: str, cells: List[Dict[str, Any]], out_dir: str, device: str, *,
              n_prompts_cap: int = 0, skip_identity: bool = False, lm=None, tok=None
              ) -> Dict[str, Any]:
    """Generate every cell of ``cells`` for ``unit`` into ``out_dir``/<cell>.jsonl (skip-guard per
    shard).  Shared by s6 (production plan) and the s4 pilot (its own 20 x 1 plan)."""
    t0 = time.time()
    prompts_txt = base.check_prompts()
    if lm is None:
        lm, tok = steer.load_lm(device)
    refs = {r.tag: r for r in base.sae_refs()}
    adapter, sel, cal, af_cids, meta_row = None, None, None, None, {}
    if unit == "null":
        layer = base.PILOT_LAYER                  # no hook is used; identity test still run
        meta_row = {"arm": "null"}
    elif unit.startswith("armfree_L"):
        layer = int(unit.split("L")[1])
        cal = base.read_json(base.s5_record(unit))
        af_cids = [str(c) for c in s1_select.load_armfree(layer)["concept_ids"]]
        meta_row = {"arm": "armfree"}
    else:
        ref = refs[unit]
        layer = int(ref.key[1:])
        model, meta = sae_io.load_arm(ref.run_dir, device)
        thr = base.read_json(base.s0_threshold(ref.tag))["threshold"] if meta["family"] == "batch_topk" else None
        adapter = steer.SAEAdapter(model, meta, thr)
        sel = s1_select.load_selection(ref.tag)
        cal = base.read_json(base.s5_record(unit))
        meta_row = {"tag": ref.tag, "activation": ref.activation, "arm": ref.arm, "seed": ref.seed}
    if unit != "null" and not (cal and cal.get("complete")):
        raise RuntimeError(f"{unit}: s5 calibration missing or incomplete")
    ident = None
    if not skip_identity:
        ident = identity_test.run([layer], base.N_PROMPTS_OTHER, device)
        if not ident["passed"]:
            raise RuntimeError(f"identity-hook precondition FAILED at L{layer}: {ident}")
    nf, _ = steer.norm_factor(layer)
    hook = steer.EditHook(lm, layer, nf, adapter)
    prompt_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts_txt]
    summary: Dict[str, Any] = {"unit": unit, "layer": layer, "out_dir": out_dir,
                               "prompts_sha256": base.PROMPTS_SHA256, "cells": {},
                               "identity_test": ident, "smoke": base.IS_SMOKE}
    for cell in cells:
        path = os.path.join(out_dir, f"{cell['name']}.jsonl")
        if os.path.isfile(path):
            summary["cells"][cell["name"]] = "exists"
            continue
        tc = time.time()
        n_p = cell["n_prompts"] if not n_prompts_cap else min(cell["n_prompts"], n_prompts_cap)
        slots = cell_slots(unit, cell, cal, sel, af_cids)
        vecmap = _vec_lookup(layer, cell, adapter)
        rows: List[Dict[str, Any]] = []
        todo: List[Tuple[Dict[str, Any], int, int, int]] = []     # (slot, prompt, cont, seed)
        U: List[torch.Tensor] = []
        for sl in slots:
            for pi in range(n_p):
                seed = item_seed(cell["mode"], cell["budget"], sl["concept"], sl["rank"], pi)
                if sl["cal"].get("status") != "ok":
                    rows.append(_row(unit, layer, meta_row, cell, sl, pi, None, seed,
                                     prompts_txt[pi], status=sl["cal"].get("status")))
                    continue
                u = steer.item_uniforms(seed, cell["n_cont"])
                for ci in range(cell["n_cont"]):
                    todo.append((sl, pi, ci, seed))
                    U.append(u[ci])
        for s in range(0, len(todo), base.GEN_BATCH_ROWS):
            batch = todo[s:s + base.GEN_BATCH_ROWS]
            uni = torch.stack(U[s:s + base.GEN_BATCH_ROWS]).to(device)
            pr = [prompt_ids[pi] for (_, pi, _, _) in batch]
            if cell["mode"] == "null":
                gen = steer.generate(lm, pr, None, uni)
            else:
                _configure_rows(hook, cell["mode"], batch, nf, device, vecmap)
                with hook.installed():
                    gen = steer.generate(lm, pr, hook, uni)
            for (sl, pi, ci, seed), g in zip(batch, gen.tolist()):
                rows.append(_row(unit, layer, meta_row, cell, sl, pi, ci, seed, prompts_txt[pi],
                                 gen=g, tok=tok))
        rows.sort(key=lambda r: (r["concept"], r["rank"], r["prompt_idx"],
                                 r["cont"] if r["cont"] is not None else -1))
        base.atomic_write_jsonl(path, rows)
        summary["cells"][cell["name"]] = {"rows": len(rows), "generated": len(todo),
                                          "unreachable_rows": sum(r["status"] != "ok" for r in rows),
                                          "seconds": round(time.time() - tc, 1)}
        base.log(f"s6 {unit} {cell['name']}: {len(todo)} generations ({time.time() - tc:.1f}s)")
    summary["seconds"] = round(time.time() - t0, 1)
    return summary


def _reuse_pilot_shards(unit: str, cells: List[Dict[str, Any]]) -> Dict[str, str]:
    """armfree_L<PILOT_LAYER>: copy the pilot's 20 x 1 shards (module docstring).  A pilot shard is
    reused only if the pilot ran on the pinned prompts and has exactly the rows this cell needs
    (a smoke pilot with capped prompts is never reused)."""
    from steering import s1_select
    out: Dict[str, str] = {}
    if unit != f"armfree_L{base.PILOT_LAYER}":
        return out
    summ = base.read_json(base.p("s4", "gen", "_summary.json")) or {}
    if (summ.get("pilot") or {}).get("prompts_sha256") != base.PROMPTS_SHA256:
        return out
    n_cid = len(s1_select.load_armfree(base.PILOT_LAYER)["concept_ids"])
    for c in cells:
        dst = shard_path(unit, c["name"])
        src = os.path.join(base.p("s4", "gen"), c["name"] + ".jsonl")
        if os.path.isfile(dst) or not os.path.isfile(src) or c["headline"]:
            continue
        n_slots = n_cid * (1 if c["dirset"] == "diffmean" else base.N_RANDOM)
        rows = base.read_jsonl(src)
        if len(rows) != n_slots * c["n_prompts"] * c["n_cont"]:
            continue
        with open(src, "rb") as fh:
            base.atomic_write_bytes(dst, fh.read())
        out[c["name"]] = f"copied from the pilot ({src})"
    return out


@torch.no_grad()
def run_unit(unit: str, device: str, *, n_prompts_cap: int = 0,
             cells_only: Optional[List[str]] = None, skip_identity: bool = False,
             metrics: bool = True) -> Dict[str, Any]:
    ok, primary = primary_budget()
    if not ok:
        raise RuntimeError(f"P1 not decided: {base.p1_json()} missing (run s4 + s4p first)")
    cells = [c for c in cell_plan(unit, primary) if not cells_only or c["name"] in cells_only]
    reused = _reuse_pilot_shards(unit, cells)
    lm, tok = steer.load_lm(device)
    summary = run_cells(unit, cells, base.s6_dir(unit), device, n_prompts_cap=n_prompts_cap,
                        skip_identity=skip_identity, lm=lm, tok=tok)
    summary["primary_budget"] = primary
    summary["reused_from_pilot"] = reused
    base.atomic_write_json(os.path.join(base.s6_dir(unit), "_summary.json"), summary)
    if metrics:
        from steering import s7_metrics
        if s7_metrics.clf_done():
            # Merged s7m: same process, same pythia (full fp32), shards without a metric file only.
            summary["metrics"] = s7_metrics.unit_metrics(unit, device, lm=lm, tok=tok)
        else:
            base.log(f"s6 {unit}: s7c classifier not trained yet -- metrics left for a later task")
    return summary


def _vec_lookup(layer: int, cell: Dict[str, Any], adapter) -> Dict[str, Any]:
    """dir_key -> unit vector for arm-free mode (i) cells; the adapter for SAE cells."""
    if cell["mode"] != "i" or cell["dirset"] not in ("diffmean", "random"):
        return {"__adapter__": adapter}
    af = s1_select.load_armfree(layer)
    out: Dict[str, Any] = {}
    for ci, cid in enumerate(af["concept_ids"]):
        out[f"i|diffmean|{cid}"] = af["diffmean"][ci]
        for r in range(af["random"].shape[1]):
            out[f"i|random|{cid}|r{r}"] = af["random"][ci, r]
    return out


def _configure_rows(hook: steer.EditHook, mode: str, batch, nf: float, device,
                    vecmap: Dict[str, Any]) -> None:
    """Per-row hook parameters of one generation batch (rows may mix concepts and ranks)."""
    if mode == "i":
        vecs = []
        for (sl, _, _, _) in batch:
            if sl["latent"] is not None:
                vecs.append(vecmap["__adapter__"].U[sl["latent"]].float().to(device))
            else:
                vecs.append(torch.as_tensor(vecmap[sl["dir_key"]], dtype=torch.float32, device=device))
        V = torch.stack(vecs)
        alpha = torch.as_tensor([sl["cal"]["param"] for (sl, _, _, _) in batch], device=device,
                                dtype=torch.float32)
        hook.configure("add", delta_raw=V * alpha[:, None] * nf)
    else:
        j = torch.as_tensor([sl["latent"] for (sl, _, _, _) in batch], device=device, dtype=torch.long)
        c = torch.as_tensor([sl["cal"]["param"] for (sl, _, _, _) in batch], device=device,
                            dtype=torch.float32)
        hook.configure("clamp" if mode == "ii" else "rerun", j=j, c=c)


def _row(unit, layer, meta_row, cell, sl, pi, ci, seed, prompt, *, gen=None, tok=None,
         status: str = "ok") -> Dict[str, Any]:
    r = {"unit": unit, "layer": layer, **meta_row, "mode": cell["mode"], "dirset": cell["dirset"],
         "budget": cell["budget"], "headline": cell["headline"], "concept": sl["concept"],
         "rank": sl["rank"], "latent": sl["latent"], "dir_key": sl["dir_key"],
         "param": sl["cal"].get("param"), "calib_kl": sl["cal"].get("kl"),
         "prompt_idx": pi, "cont": ci, "item_seed": seed, "prompt": prompt, "status": status}
    if gen is not None:
        ids, eos = steer.cut_at_eos(gen)
        r.update(gen_ids=ids, hit_eos=eos, n_tokens=len(ids),
                 text=tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))
    else:
        r.update(gen_ids=None, hit_eos=None, n_tokens=None, text=None)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, help="jobs.py s6 index (base.gen_units() order)")
    ap.add_argument("--unit")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-prompts-cap", type=int, default=0, help="smoke only")
    ap.add_argument("--cells", nargs="*", default=None, help="smoke only: subset of cell names")
    a = ap.parse_args()
    unit = a.unit if a.unit else base.gen_units()[a.index]
    if (a.n_prompts_cap or a.cells) and not base.IS_SMOKE:
        raise SystemExit("--n-prompts-cap / --cells are smoke options")
    from steering import gates43
    if unit in {r.tag for r in base.sae_refs()} and not base.IS_SMOKE:
        why = gates43.sae_excluded(unit)
        if why:                      # D-43 / D-18: never generated (defence in depth)
            print(f"[s6] {unit}: excluded -- {why}")
            return 3
    if unit_done(unit) and not a.cells:
        print(f"[s6] {unit}: already done")
        return 0
    s = run_unit(unit, a.device or base.device_default(), n_prompts_cap=a.n_prompts_cap,
                 cells_only=a.cells)
    print(f"[s6] {unit}: {s['cells']} ({s['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
