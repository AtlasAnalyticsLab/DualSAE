#!/usr/bin/env python
"""
LLM judge used to select the images shown in the image-topic figures. For each candidate image it
asks an LLM (microsoft/phi-4 by default) which of two topics better matches the image's caption:
the dominant topic under the DualSAE model or the dominant topic under the activation-matched
baseline. The figure (visualize_topics.py --figures img2topic / img2topic-all) then shows only
images whose verdict is `dual`.

Selection rule (exactly as implemented below):
- Candidate pool per cell = the lift-based "confident" round-robin selection the figure uses
  (visualize_topics.select_images on the Dual model's topic matrix), with --n-candidates images.
- Each image is judged TWICE, once with the Dual topic shown as "Topic A" and once with the
  baseline topic shown as "Topic A". The prompt contains the image caption (documents.jsonl) and
  each topic's top-10 WORDS (not the LLM topic labels). Answer 'A' / 'B' / 'TIE', temperature 0,
  max 4 new tokens; anything that does not start with A or B counts as TIE.
- verdict = `dual` if Dual wins in both orders, `base` if the baseline wins in both orders, else
  `tie`.
- Output: img2topic_judgments.csv in the Dual topics cell dir (row_idx, dual_topic, base_topic,
  verdict). Resumable: cells with an existing CSV are skipped (--force overrides).

Offline vLLM (same pattern as annotate_topics.py; ONE model load for all cells in a run):
  python judge_img2topic.py --encoder clip --activation TopK            # one cell
  python judge_img2topic.py --encoders "clip siglip vit" --activations "TopK BatchTopK JumpReLU"
"""

import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # MUST precede any vllm import
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import csv
import argparse

from viz_common import ACT2TRAINER, topics_cell_dir, load_documents
from visualize_topics import load_cell, dominant_topic, select_images

JUDGE_TMPL = (
    "An image is described as follows:\n\"{caption}\"\n\n"
    "Two topic models each assigned this image a topic, described by its top words:\n"
    "Topic A: {words_a}\n"
    "Topic B: {words_b}\n\n"
    "Which topic more accurately and specifically matches the image description? "
    "Answer with exactly one of: 'A', 'B', or 'TIE'. "
    "Answer 'A' or 'B' only if that topic is CLEARLY more accurate and specific; otherwise answer 'TIE'."
)


def parse_choice(text):
    """Raw completion → 'A' / 'B' / 'TIE' (ambiguous → TIE, conservative)."""
    t = (text or "").strip().upper()
    tok = t.split()[0].strip(".:'\"") if t.split() else ""
    return tok if tok in ("A", "B") else "TIE"


def combine_orders(choice_dual_first, choice_base_first):
    """Verdict from BOTH orderings, to cancel the judge's position bias: a model only gets a win
    if it wins as A AND as B. Everything else (splits, double-ties) is a tie."""
    dual_wins = (choice_dual_first == "A") and (choice_base_first == "B")
    base_wins = (choice_dual_first == "B") and (choice_base_first == "A")
    if dual_wins:
        return "dual"
    if base_wins:
        return "base"
    return "tie"


def collect_cell(results_dir, enc, act, seed, k, docs, n_candidates, judge_words, rng_seed):
    """Candidate rows + BOTH-ORDER prompts for one enc×act cell. Returns (out_csv_path, items, prompts);
    items[i] = (row_idx, dual_topic, base_topic); prompts[2i] = dual-as-A, prompts[2i+1] = base-as-A."""
    dual_tr, base_tr = ACT2TRAINER[act]
    tt_d, w_d, _ = load_cell(results_dir, enc, dual_tr, seed, k, "")
    tt_b, w_b, _ = load_cell(results_dir, enc, base_tr, seed, k, "")
    dom_d = dominant_topic(tt_d)[0]
    dom_b = dominant_topic(tt_b)[0]
    pool = select_images(tt_d, n_candidates, "confident", rng_seed)

    items, prompts = [], []
    for d in pool:
        td, tb = int(dom_d[d]), int(dom_b[d])
        cap = " ".join(docs[d][1].split())[:800]
        wd = ", ".join(w_d[td][:judge_words])
        wb = ", ".join(w_b[tb][:judge_words])
        items.append((d, td, tb))
        prompts.append(JUDGE_TMPL.format(caption=cap, words_a=wd, words_b=wb))   # dual as A
        prompts.append(JUDGE_TMPL.format(caption=cap, words_a=wb, words_b=wd))   # base as A
    out_csv = os.path.join(topics_cell_dir(results_dir, enc, dual_tr, seed, k, ""),
                           "img2topic_judgments.csv")
    return out_csv, items, prompts


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results-dir", default="./outputs/topic_modeling/runs/imagenet_paired")
    p.add_argument("--encoder", default=None, help="single-cell shortcut")
    p.add_argument("--activation", default=None, choices=list(ACT2TRAINER) + [None])
    p.add_argument("--encoders", default="clip siglip vit")
    p.add_argument("--activations", default="TopK BatchTopK JumpReLU")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--documents", default="./outputs/topic_modeling/imagenet_val/data/documents.jsonl")
    p.add_argument("--n-candidates", type=int, default=400)
    p.add_argument("--judge-words", type=int, default=10, help="topic words shown to the judge per side")
    p.add_argument("--model", default="microsoft/phi-4")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    encs = [args.encoder] if args.encoder else args.encoders.split()
    acts = [args.activation] if args.activation else args.activations.split()
    docs = load_documents(args.documents)

    # collect all cells' prompts first → ONE PHI-4 load for everything
    cells, all_prompts = [], []
    for enc in encs:
        for act in acts:
            try:
                out_csv, items, prompts = collect_cell(args.results_dir, enc, act, args.seed, args.k,
                                                       docs, args.n_candidates, args.judge_words,
                                                       args.rng_seed)
            except FileNotFoundError as e:
                print(f"[skip] {enc}/{act}: {e}")
                continue
            if os.path.isfile(out_csv) and not args.force:
                print(f"[skip] already judged: {out_csv}")
                continue
            cells.append((enc, act, out_csv, items, len(all_prompts)))
            all_prompts.extend(prompts)
    if not cells:
        print("Nothing to judge."); return
    print(f"Judging {len(all_prompts)} images across {len(cells)} cells with {args.model} …")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    formatted = [tok.apply_chat_template([{"role": "user", "content": pr}], tokenize=False,
                                         add_generation_prompt=True) for pr in all_prompts]
    llm = LLM(model=args.model, trust_remote_code=True, dtype="auto",
              tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.9)
    outs = llm.generate(formatted, SamplingParams(max_tokens=4, temperature=0.0))

    for enc, act, out_csv, items, off in cells:
        counts = {"dual": 0, "base": 0, "tie": 0}
        with open(out_csv, "w", newline="") as g:
            w = csv.DictWriter(g, fieldnames=["row_idx", "dual_topic", "base_topic", "verdict"])
            w.writeheader()
            for i, (d, td, tb) in enumerate(items):
                v = combine_orders(parse_choice(outs[off + 2 * i].outputs[0].text),
                                   parse_choice(outs[off + 2 * i + 1].outputs[0].text))
                counts[v] += 1
                w.writerow({"row_idx": d, "dual_topic": td, "base_topic": tb, "verdict": v})
        n = len(items)
        print(f"  {enc}/{act}: dual {counts['dual']} ({counts['dual']/n:.0%})  "
              f"base {counts['base']} ({counts['base']/n:.0%})  tie {counts['tie']} ({counts['tie']/n:.0%})"
              f"  → {out_csv}")


if __name__ == "__main__":
    main()
