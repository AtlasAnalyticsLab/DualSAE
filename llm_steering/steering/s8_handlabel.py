"""s8: the 200-item hand-label sheet and the optional InternVL3.5-8B second judge (D-50).

--export (CPU, seconds; ONCE, after every headline s7 judge unit exists):
    200 HEADLINE items (mode (i), primary budget, 'full' selection, SAE arms of every D-43-passing
    layer x pair), sampled with a fixed seed and stratified by arm (dual / baseline / dual_g0 as
    equal as the pool allows), then by concept.  Two files:
      handlabel_sheet.csv  ARM-BLINDED: item_id, concept description, prompt, continuation, and
                           empty columns concept / instruction / fluency (0-2, the AxBench rubric
                           wording is in the header note).
      handlabel_key.csv    item_id -> unit, layer, activation, arm, seed, concept, latent, rank,
                           prompt_idx, cont and the phi-4 ratings (for kappa; never shown to the
                           labeller).
    Reproducible and write-once: item ids are derived from the item's identity with a D-22 seed
    (never random), the sample is seeded, the export REFUSES to overwrite an existing sheet or key
    (--force to override) and REFUSES while any headline judge file of a passing pair is missing
    (an early export would sample a partial pool).  Why: once a labeller starts filling the sheet, a
    re-export must not silently pair the filled sheet with a new key.
--second-judge-smoke (one GPU, <= 15 prompts): InternVL3.5-8B TEXT-ONLY through vLLM
    (InternVLChatModel is registered in vLLM 0.20; trust_remote_code for its config; no image
    inputs) with the same verbatim AxBench templates, T=0, 768/1536 tokens and the official parser.
    Passes when every completion parses ('Rating:' present, value in 0-2).  It is an ENVIRONMENT
    check; its record lives at base.SECOND_JUDGE_ENV_RECORD, which the planner and s8j read.
--second-judge (GPU job, one unit): the same judge on the 200 sheet items (600 prompts); Cohen's
    kappa (unweighted and quadratic-weighted) against phi-4 per rating kind, and the production
    parse-failure rates, written to s8/second_judge.json.  s9 reports it, and the hand-label kappa
    (human vs phi-4 and vs InternVL) once handlabel_sheet_filled.csv exists next to the sheet.
Optional: not part of the reported steering table; the InternVL snapshot is fetched by
prepare_assets.sh only with --with-second-judge.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from steering import base, concepts, s6_generate
from common import llm_backend as LB, seeds as _seeds

from common import lang_config as _lc  # noqa: E402

INTERNVL_REPO = "OpenGVLab/InternVL3_5-8B"
INTERNVL_PATH = os.environ.get("A16_SECOND_JUDGE_PATH", _lc.local_snapshot(INTERNVL_REPO, "model"))
N_ITEMS = 200
SHEET_NOTE = ("Rate each continuation 0-2 on three scales (AxBench rubric): concept = is the "
              "concept clearly incorporated (0 not at all, 1 somewhat/awkwardly, 2 fully); "
              "instruction = is the continuation related to the prompt; fluency = is it fluent "
              "(0 not fluent/repetitive, 1 some errors, 2 fluent).")


def s8_path(name: str) -> str:
    return base.p("s8", name)


# The InternVL smoke checks that THIS environment can run the second judge (module docstring).
SMOKE_RECORD = base.SECOND_JUDGE_ENV_RECORD


def item_id(r: Dict[str, Any]) -> str:
    """Reproducible, arm-blind id: a D-22 seed of the item's identity, as 8 hex digits.  The key
    file maps it back; the hash does not reveal the arm to the labeller."""
    ident = "|".join(str(r.get(k)) for k in ("unit", "concept", "rank", "prompt_idx", "cont"))
    return f"{_seeds.derive_seed('a16.handlabel.id', ident):08x}"


def missing_headline_units(primary: float) -> List[str]:
    """Gate-passing SAE units whose headline judge file does not exist yet."""
    from steering import gates43
    miss = []
    for (layer, act), v in gates43.all_verdicts().items():
        if v["state"] != "pass":
            continue
        for tag in v["sae"]:
            if not os.path.isfile(os.path.join(base.s7j_dir(tag), f"i_full_b{primary:g}.jsonl")):
                miss.append(tag)
    return miss


def headline_pool() -> List[Dict[str, Any]]:
    ok, primary = s6_generate.primary_budget()
    if not ok or primary is None:
        raise RuntimeError("no primary budget (P1 missing or 'none'): no headline items exist")
    from steering import gates43
    passing = {k for k, v in gates43.all_verdicts().items() if v["state"] == "pass"}
    pool = []
    for ref in base.sae_refs():
        if (int(ref.key[1:]), ref.activation) not in passing and not base.IS_SMOKE:
            continue                  # the headline population = D-43-passing pairs only
        jd = base.s7j_dir(ref.tag)
        path = os.path.join(jd, f"i_full_b{primary:g}.jsonl")
        gpath = os.path.join(base.s6_dir(ref.tag), f"i_full_b{primary:g}.jsonl")
        if not (os.path.isfile(path) and os.path.isfile(gpath)):
            continue
        gen = {(r["concept"], r["rank"], r["prompt_idx"], r["cont"]): r for r in base.read_jsonl(gpath)}
        for r in base.read_jsonl(path):
            if r["status"] != "ok":
                continue
            g = gen[(r["concept"], r["rank"], r["prompt_idx"], r["cont"])]
            pool.append({**r, "prompt": g["prompt"], "text": g["text"]})
    return pool


def export(n_items: int = N_ITEMS, force: bool = False) -> Dict[str, Any]:
    ok, primary = s6_generate.primary_budget()
    if not ok or primary is None:
        raise RuntimeError("no primary budget (P1 missing or 'none'): no headline items exist")
    if not force:
        for name in ("handlabel_sheet.csv", "handlabel_key.csv"):
            if os.path.isfile(s8_path(name)):
                raise RuntimeError(f"{s8_path(name)} exists: the export is write-once (a re-export "
                                   f"would break the sheet -> key link); pass --force to replace it")
        miss = missing_headline_units(primary) if not base.IS_SMOKE else []
        if miss:
            raise RuntimeError(f"{len(miss)} headline judge units missing (e.g. {miss[:3]}): "
                               f"exporting now would sample a partial pool")
    pool = headline_pool()
    rng = _seeds.rng(_seeds.derive_seed("a16.handlabel", "sheet"))
    by_arm: Dict[str, List[int]] = {}
    for i, r in enumerate(pool):
        by_arm.setdefault(r["arm"], []).append(i)
    per_arm = n_items // max(1, len(by_arm))
    chosen: List[int] = []
    for arm, idx in sorted(by_arm.items()):
        # within an arm, round-robin over concepts so every concept is represented
        byc: Dict[str, List[int]] = {}
        for i in rng.permutation(idx):
            byc.setdefault(pool[i]["concept"], []).append(int(i))
        take: List[int] = []
        while len(take) < min(per_arm, len(idx)):
            for c in sorted(byc):
                if byc[c] and len(take) < per_arm:
                    take.append(byc[c].pop())
        chosen += take
    rest = [i for i in rng.permutation(len(pool)) if i not in set(chosen)]
    chosen += [int(i) for i in rest[: max(0, n_items - len(chosen))]]
    chosen = [chosen[i] for i in rng.permutation(len(chosen))]
    os.makedirs(base.p("s8"), exist_ok=True)
    sheet, key = [], []
    for i in chosen:
        r = pool[i]
        iid = item_id(r)
        sheet.append({"item_id": iid, "concept": concepts.CONCEPT_BY_ID[r["concept"]].description,
                      "prompt": r["prompt"], "continuation": r["text"], "concept_rating": "",
                      "instruction_rating": "", "fluency_rating": ""})
        key.append({"item_id": iid, **{k: r.get(k) for k in ("unit", "layer", "activation", "arm", "seed",
                                                             "concept", "latent", "rank", "prompt_idx", "cont")},
                    "phi4_concept": r["concept_rating"][r["concept"]],
                    "phi4_instruction": r["instruction_rating"], "phi4_fluency": r["fluency_rating"]})
    if len({x["item_id"] for x in sheet}) != len(sheet):
        raise RuntimeError("item id collision in the hand-label sample")
    import io
    for name, rows in (("handlabel_key.csv", key), ("handlabel_sheet.csv", sheet)):
        buf = io.StringIO()
        if name.startswith("handlabel_sheet"):
            buf.write("# " + SHEET_NOTE + "\n")
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        base.atomic_write_bytes(s8_path(name), buf.getvalue().encode("utf-8"))
    return {"n_items": len(sheet), "pool": len(pool), "by_arm": {a: len(v) for a, v in by_arm.items()}}


def _internvl(util: float) -> "LB.TextLLM":
    llm = LB.TextLLM(model_path=INTERNVL_PATH, max_model_len=base.JUDGE_MAX_MODEL_LEN,
                     gpu_memory_utilization=util, cache_dir=base.p("s8", "cache"), seed=0,
                     engine_kwargs={"trust_remote_code": True,
                                    "limit_mm_per_prompt": {"image": 0, "video": 0}})
    # TextLLM's lazy tokenizer calls AutoTokenizer without trust_remote_code; for this checkpoint
    # (auto_map config) transformers then ASKS interactively, which a batch job cannot answer.
    # The tokenizer itself is the built-in Qwen2Tokenizer (tokenizer_config.json), so it is
    # pre-loaded here with trust_remote_code=False (no remote code runs on the tokenizer side;
    # vLLM's own engine gets trust_remote_code=True for the model config only).
    from transformers import AutoTokenizer
    llm._tok = AutoTokenizer.from_pretrained(INTERNVL_PATH, local_files_only=True,
                                             trust_remote_code=False)
    return llm


def _judge(llm, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    tpl = LB.load_axbench_templates()
    prompts = []
    for it in items:
        prompts += [tpl["concept"].format(concept=it["concept"], sentence=it["continuation"]),
                    tpl["instruction"].format(instruction=it["prompt"], sentence=it["continuation"]),
                    tpl["fluency"].format(sentence=it["continuation"])]
    outs = llm.generate_retry_length(prompts, max_tokens=base.JUDGE_MAX_TOKENS,
                                     retry_max_tokens=base.JUDGE_RETRY_TOKENS, temperature=0.0)
    res = []
    for k, it in enumerate(items):
        o = outs[3 * k: 3 * k + 3]
        res.append({"item_id": it.get("item_id"),
                    **{f"{n}_rating": LB.parse_axbench_rating(t) for n, (t, _, _) in zip(KINDS, o)},
                    **{f"{n}_status": LB.axbench_rating_status(t) for n, (t, _, _) in zip(KINDS, o)}})
    return res


KINDS = ("concept", "instruction", "fluency")


def second_judge_smoke(util: float, n_items: int = 5) -> Dict[str, Any]:
    items = []
    for p in sorted(glob.glob(os.path.join(base.p("s6"), "*", "*.jsonl"))):
        for r in base.read_jsonl(p):
            if r["status"] == "ok" and r["mode"] != "null":
                items.append({"concept": concepts.CONCEPT_BY_ID[r["concept"]].description,
                              "prompt": r["prompt"], "continuation": r["text"]})
            if len(items) >= n_items:
                break
        if len(items) >= n_items:
            break
    llm = _internvl(util)
    try:
        res = _judge(llm, items)
        st = llm.stats()
    finally:
        llm.close()
    ok = all(r[f"{k}_status"] == "ok" for r in res for k in KINDS)
    rec = {"passed": bool(ok and res), "n_items": len(res), "n_prompts": 3 * len(res), "results": res,
           "engine": llm.describe(), "engine_load_s": st.get("engine_load_s"),
           "model_path": INTERNVL_PATH, "smoke": base.IS_SMOKE}
    base.atomic_write_json(SMOKE_RECORD, rec)
    return rec


def kappa_block(a: List[float], b: List[float]) -> Dict[str, Any]:
    """Cohen's kappa (unweighted and quadratic-weighted) and exact agreement of two 0-2 raters
    over the same items.  Ratings are rounded to the 0/1/2 grid (the rubric's scale).  Items where
    either rating is missing are dropped (their count is returned)."""
    from sklearn.metrics import cohen_kappa_score
    pairs = [(int(round(x)), int(round(y))) for x, y in zip(a, b)
             if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    n_drop = len(a) - len(pairs)
    if len(pairs) < 2:
        return {"n": len(pairs), "n_dropped": n_drop, "cohen_kappa": float("nan"),
                "quadratic_weighted_kappa": float("nan"), "exact_agreement": float("nan")}
    ai, bi = [p[0] for p in pairs], [p[1] for p in pairs]
    with np.errstate(invalid="ignore", divide="ignore"):
        k0 = cohen_kappa_score(ai, bi, labels=[0, 1, 2])
        kq = cohen_kappa_score(ai, bi, labels=[0, 1, 2], weights="quadratic")
    return {"n": len(pairs), "n_dropped": n_drop, "cohen_kappa": float(k0),
            "quadratic_weighted_kappa": float(kq),
            "exact_agreement": float(np.mean(np.array(ai) == np.array(bi)))}


def handlabel_kappa() -> Optional[Dict[str, Any]]:
    """Human vs phi-4 (and vs InternVL when s8j ran) kappa per rating kind, from the filled sheet
    joined to the key by item_id.  None while handlabel_sheet_filled.csv does not exist."""
    fpath = s8_path("handlabel_sheet_filled.csv")
    if not (os.path.isfile(fpath) and os.path.isfile(s8_path("handlabel_key.csv"))):
        return None
    with open(fpath, newline="") as fh:
        filled = list(csv.DictReader(l for l in fh if not l.startswith("#")))
    with open(s8_path("handlabel_key.csv"), newline="") as fh:
        key = {r["item_id"]: r for r in csv.DictReader(fh)}
    sj = base.read_json(s8_path("second_judge.json")) or {}
    ivl = {r["item_id"]: r for r in sj.get("results", [])}

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    out: Dict[str, Any] = {"n_filled_rows": len(filled),
                           "n_unmatched_ids": sum(r.get("item_id") not in key for r in filled)}
    rows = [r for r in filled if r.get("item_id") in key]
    for k in KINDS:
        hum = [num(r.get(f"{k}_rating")) for r in rows]
        out[f"{k}_vs_phi4"] = kappa_block(hum, [num(key[r["item_id"]][f"phi4_{k}"]) for r in rows])
        if ivl:
            out[f"{k}_vs_internvl"] = kappa_block(hum, [num((ivl.get(r["item_id"]) or {}).get(f"{k}_rating"))
                                                       for r in rows])
    return out


def second_judge(util: float) -> Dict[str, Any]:
    sm = base.read_json(SMOKE_RECORD)
    if not (sm and sm.get("passed")):
        raise RuntimeError("InternVL second judge smoke has not passed: not running it (D-50)")
    with open(s8_path("handlabel_sheet.csv")) as fh:
        items = list(csv.DictReader(l for l in fh if not l.startswith("#")))
    with open(s8_path("handlabel_key.csv")) as fh:
        key = {r["item_id"]: r for r in csv.DictReader(fh)}
    llm = _internvl(util)
    try:
        res = _judge(llm, items)
    finally:
        llm.close()
    kappa = {}
    for k in KINDS:
        a = [float(key[r["item_id"]][f"phi4_{k}"]) for r in res]
        b = [r[f"{k}_rating"] for r in res]
        kappa[k] = dict(kappa_block(a, b),
                        internvl_parse_failure_rate=float(np.mean([r[f"{k}_status"] != "ok" for r in res])))
    rec = {"n_items": len(res), "kappa_vs_phi4": kappa, "results": res, "model_path": INTERNVL_PATH,
           "env_record": SMOKE_RECORD}
    base.atomic_write_json(s8_path("second_judge.json"), rec)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--second-judge-smoke", action="store_true")
    ap.add_argument("--second-judge", action="store_true")
    ap.add_argument("--util", type=float, default=None)
    ap.add_argument("--n-items", type=int, default=N_ITEMS)
    ap.add_argument("--force", action="store_true", help="replace an existing sheet + key")
    a = ap.parse_args()
    util = a.util if a.util is not None else (
        float(os.environ["A16_GPU_UTIL"]) if os.environ.get("A16_GPU_UTIL") else 0.9)
    if a.export:
        print(export(a.n_items, force=a.force))
    if a.second_judge_smoke:
        r = second_judge_smoke(util)
        print(f"InternVL smoke passed={r['passed']} on {r['n_prompts']} prompts; engine load "
              f"{r['engine_load_s']}s; {[{k: v for k, v in x.items() if k.endswith('rating')} for x in r['results']]}")
    if a.second_judge:
        r = second_judge(util)
        print(r["kappa_vs_phi4"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
