"""s7 (judge): phi-4 with the OFFICIAL AxBench judge templates (D-50, C5, C30).

A separate process from generation (one vLLM engine per process; common/llm_backend.TextLLM with
the pinned phi-4 snapshot, in-process engine, length guard, per-task JSONL cache).  Per generated
continuation, three single-item prompts, each the verbatim official template
(axbench/evaluators/prompt_templates.py @ 41c8332, sha256-pinned by llm_backend):
  concept      CONCEPT_RELEVANCE.format(concept=<pre-registered description>, sentence=<continuation>)
  instruction  INSTRUCTION_RELEVANCE.format(instruction=<neutral prompt>, sentence=<continuation>)
  fluency      FLUENCY.format(sentence=<continuation>)
sent as one user turn (AxBench chat_completion sends {"role": "user", "content": prompt}), at T=0
with max_tokens 768; a completion with finish_reason 'length' is re-asked ONCE at 1536
(TextLLM.generate_retry_length).  Parsing = the official lm_judge._get_ratings_from_completions
executed from the pinned file (DEFAULT_RATING 0.0 on failure/out-of-range, no clamping);
aggregate = the official harmonic mean (0 if any rating is 0).  Arm-blinded: the judge sees only
the concept description, the prompt and the continuation.
The "sentence" is the continuation only (AxBench judges the steered generation, not the prompt);
it is the text before the first EOS.  Declared adaptations (D-50): phi-4 replaces AxBench's
judge; pythia is a base model, so the neutral prompt plays the instruction's role.
Null rows (unsteered) are scored against EVERY concept (their concept/hm per concept).
Rows the calibration could not reach (status != ok) are carried through unjudged (PROTOCOL R5).

Outputs per generation shard: <s7/judge/<unit>/<cell>.jsonl> (atomic), one line per generation
row with ratings, per-prompt parse status ('ok' | 'no_marker' | 'unparsable' | 'out_of_range'),
finish reasons and retry flags; plus _summary.json with parse-failure and truncation rates.  The
raw completions go to a sidecar (base.s7j_completions; written BEFORE the rating file, which is the
skip-guard) so the report reads slim rating rows (~0.9 KB instead of ~2.7 KB per item).
``--pilot`` judges the s4 pilot shards with the concept prompt only (the P1 rule uses only the
concept rating; PROTOCOL P1 record).
NOT judged: the live-matched shards ('*_livematched_*').  D-46 fixes the live-matched control to
judge-free metrics only (classifier, KL, perplexity, distinct-2), so the full judge is not
spent on it; judged_shards() is the ONE predicate the worker, unit_done and the
planner share, so a unit is complete without those shards.
Cache: one phi-4 cache directory per unit (base.s7_judge_cache).  armfree_L<PILOT_LAYER> also
preloads the pilot's cache: its shards ARE the pilot's generations (s6 copies them), so every
concept prompt is a cache hit and only the instruction/fluency prompts are sent.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from steering import base, concepts
from common import llm_backend as LB

KINDS = ("concept", "instruction", "fluency")


def _prompts_for_row(row: Dict[str, Any], tpl: Dict[str, str], concept_only: bool
                     ) -> List[Tuple[str, str, str]]:
    """[(kind, concept_id, prompt)] of one generation row."""
    text = row["text"] if row["text"] is not None else ""
    cids = [c.cid for c in concepts.CONCEPTS] if row["mode"] == "null" else [row["concept"]]
    out = []
    for cid in cids:
        desc = concepts.CONCEPT_BY_ID[cid].description
        out.append(("concept", cid, tpl["concept"].format(concept=desc, sentence=text)))
    if not concept_only:
        out.append(("instruction", "-", tpl["instruction"].format(instruction=row["prompt"], sentence=text)))
        out.append(("fluency", "-", tpl["fluency"].format(sentence=text)))
    return out


def judged_shards(src_dir: str) -> List[str]:
    """The generation shards of a unit that the judge scores (every one but the live-matched)."""
    return sorted(p for p in glob.glob(os.path.join(src_dir, "*.jsonl"))
                  if "_livematched_" not in os.path.basename(p))


def judge_shards(shards: List[str], out_dir: str, llm: "LB.TextLLM", concept_only: bool = False,
                 comp_dir: Optional[str] = None) -> Dict[str, Any]:
    tpl = LB.load_axbench_templates()
    todo_shards = [s for s in shards if not os.path.isfile(os.path.join(out_dir, os.path.basename(s)))]
    all_rows: List[Tuple[str, List[Dict[str, Any]]]] = [(s, base.read_jsonl(s)) for s in todo_shards]
    prompts: List[str] = []
    where: List[Tuple[int, int, str, str]] = []                 # (shard idx, row idx, kind, cid)
    for si, (_, rows) in enumerate(all_rows):
        for ri, row in enumerate(rows):
            if row["status"] != "ok":
                continue
            for kind, cid, p in _prompts_for_row(row, tpl, concept_only):
                prompts.append(p)
                where.append((si, ri, kind, cid))
    t0 = time.time()
    outs = llm.generate_retry_length(prompts, max_tokens=base.JUDGE_MAX_TOKENS,
                                     retry_max_tokens=base.JUDGE_RETRY_TOKENS,
                                     temperature=0.0) if prompts else []
    info = llm.last_info if prompts else []
    judged: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for (si, ri, kind, cid), (text, fr, retried), inf in zip(where, outs, info or [{}] * len(outs)):
        d = judged.setdefault((si, ri), {"concept": {}, "status": {}, "finish": {}, "retried": {}})
        rating = LB.parse_axbench_rating(text)
        st = LB.axbench_rating_status(text)
        if text is None:
            st = "skipped_overlength"
        k = f"concept:{cid}" if kind == "concept" else kind
        if kind == "concept":
            d["concept"][cid] = rating
        else:
            d[kind] = rating
        d["status"][k], d["finish"][k], d["retried"][k] = st, fr, bool(retried)
        d.setdefault("completions", {})[k] = text
    stats = {"n_prompts": len(prompts), "seconds": round(time.time() - t0, 1)}
    for si, (spath, rows) in enumerate(all_rows):
        out_rows = []
        for ri, row in enumerate(rows):
            r = {k: row.get(k) for k in ("unit", "layer", "tag", "activation", "arm", "seed", "mode",
                                          "dirset", "budget", "headline", "concept", "rank",
                                          "latent", "dir_key", "param", "calib_kl", "prompt_idx",
                                          "cont", "item_seed", "status", "n_tokens", "hit_eos")}
            j = judged.get((si, ri))
            if j is not None:
                r["concept_rating"] = j["concept"]                      # {cid: rating}
                if not concept_only:
                    r["instruction_rating"] = j["instruction"]
                    r["fluency_rating"] = j["fluency"]
                    r["hm"] = {cid: LB.axbench_harmonic_mean([cr, j["instruction"], j["fluency"]])
                               for cid, cr in j["concept"].items()}
                r["judge_status"] = j["status"]
                r["judge_finish"] = j["finish"]
                r["judge_retried"] = j["retried"]
            out_rows.append(r)
        if comp_dir:
            base.atomic_write_jsonl(os.path.join(comp_dir, os.path.basename(spath)), [
                {"concept": row.get("concept"), "rank": row.get("rank"),
                 "prompt_idx": row.get("prompt_idx"), "cont": row.get("cont"),
                 "judge_completions": judged[(si, ri)]["completions"]}
                for ri, row in enumerate(rows) if (si, ri) in judged])
        # The rating file is the skip-guard, so it is written LAST.
        base.atomic_write_jsonl(os.path.join(out_dir, os.path.basename(spath)), out_rows)
    return stats


def summarize(out_dir: str) -> Dict[str, Any]:
    """Parse-failure and truncation rates per shard (D-50 reporting)."""
    summ: Dict[str, Any] = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "*.jsonl"))):
        n = fail = trunc = retried = 0
        for r in base.read_jsonl(p):
            for k, st in (r.get("judge_status") or {}).items():
                n += 1
                fail += st != "ok"
                trunc += (r["judge_finish"][k] == "length")
                retried += bool(r["judge_retried"][k])
        summ[os.path.basename(p)] = {"n_judgements": n, "parse_failure_rate": fail / n if n else None,
                                     "truncation_rate": trunc / n if n else None,
                                     "retried_rate": retried / n if n else None}
    return summ


def make_llm(unit: str, util: Optional[float] = None) -> "LB.TextLLM":
    if util is None:
        # vLLM's GPU memory fraction (a capacity knob; it does not change any answer).  0.9 on a
        # dedicated GPU is the value used for the reported runs; A16_GPU_UTIL overrides it (e.g.
        # on a shared GPU: LB.suggest_gpu_memory_utilization()).
        util = float(os.environ["A16_GPU_UTIL"]) if os.environ.get("A16_GPU_UTIL") else 0.9
    llm = LB.TextLLM(max_model_len=base.JUDGE_MAX_MODEL_LEN, gpu_memory_utilization=util,
                     cache_dir=base.s7_judge_cache(unit), seed=0)
    if unit == f"armfree_L{base.PILOT_LAYER}" and llm.cache is not None:
        # Read-only preload of the pilot's records (module docstring).  _JsonlCache._load is a
        # private helper of the shared layer, used here without modifying it: it only fills the
        # in-memory map (first record wins); this task still appends to its own file only.
        for pth in sorted(glob.glob(os.path.join(base.s7_judge_cache("pilot"), "cache.*.jsonl"))):
            llm.cache._load(pth)
    return llm


def _dirs(unit: str) -> Tuple[str, str, str]:
    """(generation dir, rating dir, completions dir) of a unit ('pilot' = the s4 pilot)."""
    if unit == "pilot":
        return base.p("s4", "gen"), base.p("s4", "judge"), base.p("s4", "judge_completions")
    return base.s6_dir(unit), base.s7j_dir(unit), os.path.dirname(base.s7j_completions(unit, "x"))


def run_unit(unit: str, util: Optional[float] = None, max_prompts: int = 0,
             concept_only: bool = False) -> Dict[str, Any]:
    src_dir, out_dir, comp_dir = _dirs(unit)
    if unit == "pilot":
        concept_only = True
    shards = judged_shards(src_dir)
    if not shards:
        raise RuntimeError(f"no generation shards in {src_dir}")
    if max_prompts:
        # Smoke: keep a prefix of each shard so that at most max_prompts judge prompts are sent.
        if not base.IS_SMOKE:
            raise RuntimeError("max_prompts is a smoke option")
        tmp = base.p("s7", "_smoke_subset", unit)
        per = max(1, max_prompts // (len(shards) * (1 if concept_only else 3)))
        sub = []
        for s in shards:
            rows = [r for r in base.read_jsonl(s) if r["status"] == "ok" and r["mode"] != "null"][:per]
            if rows:
                p = os.path.join(tmp, os.path.basename(s))
                base.atomic_write_jsonl(p, rows)
                sub.append(p)
        shards = sub
    llm = make_llm(unit, util)
    try:
        st = judge_shards(shards, out_dir, llm, concept_only=concept_only, comp_dir=comp_dir)
    finally:
        llm.close()
    summ = {"unit": unit, "judge": llm.describe(), "llm_stats": {k: v for k, v in llm.stats().items()
                                                                 if k != "skipped"},
            "templates_sha256": LB.AXBENCH_TEMPLATES_SHA256, "lm_judge_sha256": LB.AXBENCH_LM_JUDGE_SHA256,
            "concept_registry_sha256": concepts.concept_registry_sha256(),
            "concept_only": concept_only, "per_shard": summarize(out_dir), **st}
    base.atomic_write_json(os.path.join(out_dir, "_summary.json"), summ)
    return summ


def unit_done(unit: str) -> bool:
    """Every judged shard (judged_shards: live-matched excluded) has its rating file."""
    src_dir, out_dir, _ = _dirs(unit)
    shards = judged_shards(src_dir)
    return bool(shards) and all(os.path.isfile(os.path.join(out_dir, os.path.basename(s))) for s in shards)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, help="jobs.py s7j index (base.gen_units() order)")
    ap.add_argument("--unit", help="generation unit, or 'pilot' for the s4 pilot")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--util", type=float, default=None)
    ap.add_argument("--max-prompts", type=int, default=0, help="smoke only")
    a = ap.parse_args()
    unit = "pilot" if a.pilot else (a.unit or base.gen_units()[a.index])
    if unit_done(unit) and not a.max_prompts:
        print(f"[s7j] {unit}: already judged")
        return 0
    if unit != "pilot" and not base.IS_SMOKE:
        # Defence in depth (the planner already requires both): never judge a partial unit or an
        # SAE that D-43 / D-18 excludes.
        from steering import gates43, s6_generate
        if not s6_generate.generation_done(unit):
            print(f"[s7j] {unit}: generation incomplete -- not judging a partial unit")
            return 3
        if unit in {r.tag for r in base.sae_refs()} and gates43.sae_excluded(unit):
            print(f"[s7j] {unit}: excluded -- {gates43.sae_excluded(unit)}")
            return 3
    s = run_unit(unit, a.util, a.max_prompts)
    print(f"[s7j] {unit}: {s['n_prompts']} judge prompts in {s['seconds']}s; engine load "
          f"{s['llm_stats'].get('engine_load_s')}s; per shard: {s['per_shard']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
