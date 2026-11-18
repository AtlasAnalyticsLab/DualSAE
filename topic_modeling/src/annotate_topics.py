#!/usr/bin/env python
"""
LLM topic/feature annotation used for the image-topic figures: for each topic (or feature), an LLM
(microsoft/phi-4 by default) (a) classifies it as ABSTRACT vs CONCRETE and (b) summarizes its
top-20 words into a short label. The image-topic figures title each image with the label of its
dominant topic.

Offline (local weights, no downloads). Same vLLM pattern as SAE-TM's evaluation.py: in-process
engine (VLLM_ENABLE_V1_MULTIPROCESSING=0 set BEFORE importing vllm), chat-templated prompts, a
single LLM load, batched llm.generate(). Classification uses temperature 0; labels use
temperature 0.2 / top_p 0.9. One invocation annotates ALL requested cells with one model load.
Resumable (skips cells already annotated).

Outputs (beside the source files):
  topic cells   : <topics_cell>/clusters_annotated.csv     (clusters.csv cols + llm_classification + llm_summary)
  feature cells : <interpreted_cell>/features_annotated.csv (feature_id, feature_ratio, top_words,
                                                             llm_classification, llm_summary)

Usage:
  python annotate_topics.py --topics-cells <dir_with_clusters.csv> [<dir> ...]
  python annotate_topics.py --features-cells <interpreted_dir> [...] --bow-cache <bow_meta.json>
"""

import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # MUST precede any vllm import
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import csv
import argparse

import numpy as np

CLASSIFY_TMPL = (
    "You are labeling topics from a vision topic model. A topic is described by its top words.\n"
    "Classify the topic as exactly one of:\n"
    "- concrete: a nameable object, animal, plant, food, vehicle, structure, place, or scene.\n"
    "- abstract: a visual property or style such as mood, color, lighting, composition, layout, geometry, "
    "perspective, texture, focus, or photographic style.\n"
    "Top words: {words}\n"
    "Answer with exactly one word: 'concrete' or 'abstract'."
)
LABEL_TMPL = (
    "Summarize the following vision topic as a short, human-readable label of 1 to 3 words in Title Case "
    "(for example: 'Delicate Plant', 'Human Interaction', 'Lush Landscape', 'Motorcycle Helmet'). "
    "Give only the label, with no explanation or quotation marks.\n"
    "Top words: {words}\n"
    "Label:"
)


def parse_classification(text):
    t = (text or "").strip().lower()
    if "abstract" in t and "concrete" not in t:
        return "abstract"
    if "concrete" in t:
        return "concrete"
    return "concrete"                              # ambiguous → keep (inclusive; avoids over-filtering)


def parse_label(text):
    lab = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    lab = lab.strip().strip("\"'").strip().rstrip(".")
    lab = " ".join(lab.split())[:40]
    return lab.title() if lab else "Topic"


# ── Item collection (topics / features) ─────────────────────────────────────
def collect_topic_items(cell, topwords):
    """[(cell, id, words_str, base_row_dict)] for each topic in cell/clusters.csv (join preserved)."""
    path = os.path.join(cell, "clusters.csv")
    items = []
    with open(path) as f:
        for row in csv.DictReader(f):
            words = [w.strip() for w in row["cluster_words"].split(",") if w.strip()][:topwords]
            items.append((cell, int(row["cluster_id"]), ", ".join(words), row))
    return items


def collect_feature_items(cell, topwords, bow_cache, max_features):
    """[(cell, feat_id, words_str, {feature_ratio})] for a capped candidate pool of features in an
    interpreted cell. Reuses visualize_features loaders for the per-feature theta + top emitted words."""
    from scipy.sparse import load_npz
    tt = load_npz(os.path.join(cell, "theta_csr.npz")).tocsc()
    N = tt.shape[0]
    support = np.asarray(tt.getnnz(axis=0)).ravel()          # #images a feature activates on
    # candidate pool: features with real support, top by support, capped
    cand = [f for f in range(tt.shape[1]) if support[f] > 0]
    cand.sort(key=lambda f: -support[f])
    cand = cand[:max_features]
    # load B (per-feature word-emission) + BoW vocab directly for the top emitted words per feature
    import torch, json
    B = torch.load(os.path.join(cell, "word_emission_probabilities.pt"), map_location="cpu")["B"]
    vocab = json.load(open(bow_cache)).get("vocab", {}) if bow_cache else {}
    itos = (vocab.get("id2token") or vocab.get("itos") or vocab.get("words") or vocab.get("id2word")
            or (list(vocab.get("stoi", {}).keys()) if isinstance(vocab.get("stoi"), dict) else None))
    if itos is None or len(itos) != B.shape[1]:
        raise SystemExit(f"features mode needs a BoW vocab matching B (V={B.shape[1]}); pass --bow-cache. "
                         f"got itos={None if itos is None else len(itos)}")
    top = torch.topk(B[cand], k=topwords, dim=1).indices.numpy()
    items = []
    for j, f in enumerate(cand):
        words = ", ".join(itos[v] for v in top[j])
        items.append((cell, int(f), words, {"feature_ratio": float(support[f] / N)}))
    return items


# ── Writers ──────────────────────────────────────────────────────────────────
def write_topics_csv(cell, rows, ann):
    """clusters_annotated.csv = clusters.csv columns + llm_classification + llm_summary."""
    src = os.path.join(cell, "clusters.csv")
    with open(src) as f:
        fieldnames = csv.DictReader(f).fieldnames
    out_fields = list(fieldnames) + ["llm_classification", "llm_summary"]
    out = os.path.join(cell, "clusters_annotated.csv")
    with open(out, "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=out_fields); w.writeheader()
        for row in rows:
            cid = int(row["cluster_id"]); cls, lab = ann[(cell, cid)]
            w.writerow({**row, "llm_classification": cls, "llm_summary": lab})
    print(f"  wrote {out}")


def write_features_csv(cell, items, ann):
    out = os.path.join(cell, "features_annotated.csv")
    with open(out, "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=["feature_id", "feature_ratio", "top_words",
                                          "llm_classification", "llm_summary"])
        w.writeheader()
        for (c, fid, words, extra) in items:
            cls, lab = ann[(cell, fid)]
            w.writerow({"feature_id": fid, "feature_ratio": f"{extra['feature_ratio']:.6f}",
                        "top_words": words, "llm_classification": cls, "llm_summary": lab})
    print(f"  wrote {out}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # BOTH kinds in ONE run so vLLM loads exactly once (a second in-process engine init in the same
    # process is not supported reliably).
    p.add_argument("--topics-cells", nargs="*", default=[], help="dirs containing clusters.csv")
    p.add_argument("--features-cells", nargs="*", default=[], help="interpreted dirs (theta_csr.npz + word emissions)")
    p.add_argument("--model", default="microsoft/phi-4")
    p.add_argument("--topwords", type=int, default=20)
    p.add_argument("--bow-cache", default="./outputs/topic_modeling/imagenet_val/bow/bow_meta.json",
                   help="features cells: BoW meta json holding the vocab (idx->word)")
    p.add_argument("--max-features", type=int, default=400, help="features cells: candidate pool cap per cell")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    # per_cell[cell] = (kind, items); collect only cells that still need annotation
    per_cell, items = {}, []
    for c in args.topics_cells:
        if not args.force and os.path.isfile(os.path.join(c, "clusters_annotated.csv")):
            print(f"[skip] already annotated: {c}/clusters_annotated.csv"); continue
        it = collect_topic_items(c, args.topwords); per_cell[c] = ("topics", it); items += it
    for c in args.features_cells:
        if not args.force and os.path.isfile(os.path.join(c, "features_annotated.csv")):
            print(f"[skip] already annotated: {c}/features_annotated.csv"); continue
        it = collect_feature_items(c, args.topwords, args.bow_cache, args.max_features)
        per_cell[c] = ("features", it); items += it
    if not items:
        print("Nothing to annotate."); return
    print(f"Annotating {len(items)} items across {len(per_cell)} cells with {args.model} …")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    def fmt(prompts):
        return [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                        add_generation_prompt=True) for p in prompts]

    classify_prompts = fmt([CLASSIFY_TMPL.format(words=w) for (_, _, w, _) in items])
    label_prompts = fmt([LABEL_TMPL.format(words=w) for (_, _, w, _) in items])

    llm = LLM(model=args.model, trust_remote_code=True, dtype="auto",
              tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.9)
    cls_out = llm.generate(classify_prompts, SamplingParams(max_tokens=6, temperature=0.0))
    lab_out = llm.generate(label_prompts, SamplingParams(max_tokens=12, temperature=0.2, top_p=0.9))

    ann = {}
    for (c, cid, _, _), co, lo in zip(items, cls_out, lab_out):
        ann[(c, cid)] = (parse_classification(co.outputs[0].text), parse_label(lo.outputs[0].text))

    for c, (kind, it) in per_cell.items():
        if kind == "topics":
            write_topics_csv(c, [x[3] for x in it], ann)
        else:
            write_features_csv(c, it, ann)
    n_abs = sum(1 for v in ann.values() if v[0] == "abstract")
    print(f"Done. {n_abs}/{len(ann)} classified abstract, {len(ann)-n_abs} concrete.")


if __name__ == "__main__":
    main()
