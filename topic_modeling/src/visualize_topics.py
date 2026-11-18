#!/usr/bin/env python
"""
Qualitative topic-modeling figures for SAE-TM (CPU-only — no GPU, no SAE re-encode).

Produces two figures, each comparing a DualSAE against its matched baseline SAE (same activation):
  1. img2topic : a grid of images, each labeled with its associated topic's top words under
                 DualSAE vs the baseline (the same physical image for both — aligned).
  2. topic2img : two stacked panels (DualSAE / baseline); each shows a few salient topics, with
                 the topic's top words + its most representative exemplar images.

Everything is read from artifacts already saved by sae_to_topic.py (per encoder×trainer×seed×K):
  <results>/<enc>/topics/<trainer>[/warmup<W>]/seed<S>/K<k>/
      theta_topic_csr.npz   scipy CSR [N_images, K]; theta_topic[d,t] = image d's aggregated
                            activation over the SAE features assigned to topic t.
      clusters.csv          cols cluster_id,cluster_size,cluster_prob,cluster_words,cluster_ratio
                            (rows are NOT in cluster_id order → index by the cluster_id column).
  <documents.jsonl>         row d = {"document": caption, "id": rel_path}; row order == theta_topic
                            row order (both built from the same caption-aligned val embeddings).
  <image_root>/<rel_path>   the raw val image.

NOTE on topic ids: topic column t of theta_topic corresponds to cluster_id == t (from the one-hot
feature→cluster matrix M in sae_to_topic.py). top_words.txt line order may differ, so we read the
words from clusters.csv keyed by cluster_id — never by line number.

The image-topic figures (one row per encoder/activation, K=100, seed 42):
  python visualize_topics.py --figures img2topic-all --results-dir <TM_OUT>/runs/imagenet_paired \
         --documents <TM_OUT>/imagenet_val/data/documents.jsonl --image-root <imagenet val dir>
Single cells:
  python visualize_topics.py --figures img2topic --encoder siglip --activation TopK --k 100
  python visualize_topics.py --results-dir <TM_OUT>/runs/imagenet_warmup_sweep \
         --warmup 7500 ...            # sweep trees hold Dual only → baseline panels are skipped
"""

import os
import csv
import json
import argparse

import numpy as np
from scipy.sparse import load_npz

import matplotlib
matplotlib.use("Agg")                       # headless / offline
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Shared primitives (paths, documents/images/labels, generic column purity + exemplars, grid renderer)
# live in viz_common so the qualitative scripts share ONE implementation.
from viz_common import (ACT2TRAINER, topics_cell_dir, eval_dir, load_documents, load_image,
                        load_labels, wrap_words, exemplars, column_purity, fig_exemplar_grid,
                        load_topic_labels, load_cluster_ratio)


# ── Loading ──────────────────────────────────────────────────────────────────
def load_cell(results_dir, enc, trainer, seed, k, warmup):
    """Return (theta_topic CSC [N,K], id2words {cluster_id: [words]}, salience {cluster_id: cluster_prob}).
    We return CSC (not CSR) because every downstream use here is column-oriented — per-topic exemplars,
    per-topic purity, and per-topic image-set matching all slice theta_topic[:, t]. CSC makes those
    O(nnz_col); the few row-wise ops (dominant_topic argmax) still work on CSC."""
    d = topics_cell_dir(results_dir, enc, trainer, seed, k, warmup)
    tt_path = os.path.join(d, "theta_topic_csr.npz")
    cl_path = os.path.join(d, "clusters.csv")
    if not (os.path.isfile(tt_path) and os.path.isfile(cl_path)):
        raise FileNotFoundError(f"Missing topic artifacts for cell: {d}")
    tt = load_npz(tt_path).tocsc()
    id2words, salience = {}, {}
    with open(cl_path) as f:
        for row in csv.DictReader(f):
            cid = int(row["cluster_id"])
            id2words[cid] = [w.strip() for w in row["cluster_words"].split(",") if w.strip()]
            salience[cid] = float(row["cluster_prob"])
    if tt.shape[1] != len(id2words):
        raise ValueError(f"K mismatch: theta_topic has {tt.shape[1]} cols but clusters.csv has "
                         f"{len(id2words)} ids in {d}")
    return tt, id2words, salience


def dominant_topic(tt):
    """Per-row dominant topic by LIFT (activation / the topic's global mean) and that lift score
    (0 for all-zero rows). Raw argmax is SIZE-BIASED: theta_topic sums member-feature activations,
    so the largest/loudest cluster out-sums every other topic on (nearly) every image in dense cells —
    for JumpReLU literally ONE topic won the raw argmax on 100% of the 50k images. Lift instead asks
    'which topic is most enriched in THIS image relative to its base rate'."""
    gmean = np.asarray(tt.mean(axis=0)).ravel()
    ttn = tt.multiply(1.0 / np.maximum(gmean, 1e-12)[None, :]).tocsr()
    dom = np.asarray(ttn.argmax(axis=1)).ravel()
    mx = np.asarray(ttn.max(axis=1).todense()).ravel()
    return dom, mx


# ── Figure 1: image -> topic (Dual vs baseline, aligned on the same images) ──
def select_images(tt_dual, n, mode, rng_seed, restrict=None):
    """Pick n image rows. 'confident' = high max-topic score + diverse dominant topics.
    `restrict` (set of row ids) limits the pool — used by --judged-only to select among the images
    the PHI-4 judge scored as clear DualSAE wins."""
    dom, mx = dominant_topic(tt_dual)
    valid = np.where(mx > 0)[0]
    if restrict is not None:
        valid = np.array([d for d in valid if int(d) in restrict], dtype=int)
        if valid.size == 0:
            return []
    if mode == "random":
        rng = np.random.default_rng(rng_seed)
        return rng.choice(valid, size=min(n, len(valid)), replace=False).tolist()
    # confident + diverse: ROUND-ROBIN over dominant topics by descending score — round 1 takes each
    # topic's best image, round 2 each topic's 2nd-best, etc. (A plain "top up with the highest scores"
    # floods large n with the single loudest topic — e.g. dozens of near-identical text/branding images.)
    order = valid[np.argsort(-mx[valid])]
    groups = {}                                          # topic -> [images, score-desc]; insertion order
    for d in order:                                      #   = order of each topic's best image
        groups.setdefault(int(dom[d]), []).append(int(d))
    picked = []
    while len(picked) < n and groups:
        for t in list(groups.keys()):
            picked.append(groups[t].pop(0))
            if not groups[t]:
                del groups[t]
            if len(picked) >= n:
                break
    return picked[:n]


def _wrap_label(txt, width=17):
    """Wrap a topic label to the cell width — FULL words shown (no ellipsizing), long labels
    just take a second (rarely third) caption line."""
    import textwrap
    return textwrap.wrap(" ".join(txt.split()), width=width) or [""]


def fig_img2topic(tt_dual, w_dual, tt_base, w_base, docs, image_root,
                  idxs, dual_name, base_name, topwords, show_caption, title,
                  labels_d=None, labels_b=None, img_cols=10):
    """COMPACT grid: each cell = an image thumbnail (touching its row neighbours) with its dominant-topic
    captions directly beneath — bold RED = DualSAE topic, bold BLUE = baseline topic, FULL topic words
    (long labels wrap onto extra lines rather than being trimmed). Captions use the PHI-4 llm_summary when
    available (labels_d/labels_b), else the raw top word(s). show_caption is ignored (no room)."""
    has_base = tt_base is not None
    n = len(idxs)
    cols = max(1, img_cols)
    rows = (n + cols - 1) // cols

    def topic_label(t, words, labels):
        if labels is not None and t in labels:
            return labels[t]
        return ", ".join(words[t][:max(topwords, 1)])

    # lift-based dominant topics (same definition select_images uses — captions must match selection)
    dom_d = dominant_topic(tt_dual)[0]
    dom_b = dominant_topic(tt_base)[0] if has_base else None

    # pre-wrap every caption so row spacing can fit the WORST cell (full words, no trimming)
    dual_caps = [_wrap_label(topic_label(int(dom_d[d]), w_dual, labels_d)) for d in idxs]
    base_caps = [_wrap_label(topic_label(int(dom_b[d]), w_base, labels_b)) for d in idxs] if has_base else None
    # fixed caption rows: red (dual) block reserves the figure-wide max red height, so the blue
    # (baseline) labels start at the SAME y in every cell — horizontally aligned across the row
    max_d = max(len(dc) for dc in dual_caps)
    max_b = max((len(bc) for bc in base_caps)) if has_base else 0

    FS = 7.2                                       # caption font size (bold)
    LH = 0.145                                     # caption line height at FS, in axis-height fractions
    gap_frac = 0.10 + LH * (max_d + max_b)         # inter-row gap sized to the aligned caption rows
    title_lines = title.count("\n") + 2            # +1 legend line appended below, +1 base line
    top_in = 0.30 + 0.22 * title_lines             # title block height in inches
    fig_h = 1.15 * rows * (1.0 + gap_frac) + top_in
    fig = plt.figure(figsize=(1.15 * cols + 0.2, fig_h))
    # wspace=0 → NO gap between images in a row; captions wrap to the cell width so they don't collide.
    # top pinned just under the title block (default margin scales with height → huge blank band).
    gs = GridSpec(rows, cols, figure=fig, wspace=0.0, hspace=gap_frac,
                  top=1.0 - top_in / fig_h, bottom=0.25 / fig_h, left=0.005, right=0.995)

    for j, d in enumerate(idxs):
        r, c = divmod(j, cols)
        ax = fig.add_subplot(gs[r, c]); ax.axis("off")
        ax.imshow(np.asarray(load_image(image_root, docs[d][0])))
        ax.text(0.5, -0.06, "\n".join(dual_caps[j]), transform=ax.transAxes, ha="center", va="top",
                fontsize=FS, fontweight="bold", color="#c1121f")
        if has_base:
            y_base = -0.06 - LH * max_d - 0.02     # fixed offset (max red height) → aligned blue row
            ax.text(0.5, y_base, "\n".join(base_caps[j]), transform=ax.transAxes, ha="center", va="top",
                    fontsize=FS, fontweight="bold", color="#1f4e9c")
    legend = (f"bold red = DualSAE topic ({dual_name})"
              + (f",  bold blue = baseline topic ({base_name})" if has_base else ""))
    fig.suptitle(f"{title}\n{legend}", fontsize=11, y=0.998)
    return fig


# ── Figure 2: topic -> exemplar images (two stacked panels) ──────────────────
def top_topics(salience, n):
    return [t for t, _ in sorted(salience.items(), key=lambda kv: -kv[1])[:n]]


def match_topics(anchor_words, other_words, anchor_topics, topn=30):
    """Greedily pair each anchor topic with the OTHER model's most word-overlapping topic (unique),
    so a row shows the SAME theme on both sides. Overlap = |shared words| among each topic's top-N."""
    used, out = set(), []
    for at in anchor_topics:
        aw = set(anchor_words[at][:topn])
        best, best_ov = None, -1
        for ot, ow in other_words.items():
            if ot in used:
                continue
            ov = len(aw & set(ow[:topn]))
            if ov > best_ov:
                best_ov, best = ov, ot
        if best is not None:
            used.add(best)
        out.append(best)
    return out


def aligned_topics(anchor_words, other_words, n, topn=30):
    """Anchor topic ids ranked by how well they ALIGN with the other model (desc): each anchor's
    score = its best top-word overlap with any 'other' topic. Picking these gives rows that are
    genuine same-theme pairs across models — best for a side-by-side comparison."""
    scored = []
    for at, aw in anchor_words.items():
        aws = set(aw[:topn])
        best = max((len(aws & set(ow[:topn])) for ow in other_words.values()), default=0)
        scored.append((best, at))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [at for _, at in scored[:n]]


def match_topics_by_images(tt_a, tt_b, anchor_topics, m):
    """Content-based analogue of match_topics: greedily pair each anchor topic (model A) with the
    UNIQUE model-B topic whose top-m exemplar IMAGE set overlaps most (Jaccard). Use when 'same theme'
    should mean 'built from the same images' rather than 'same top words'."""
    a_sets = {at: set(exemplars(tt_a, at, m)) for at in anchor_topics}
    b_sets = {bt: set(exemplars(tt_b, bt, m)) for bt in range(tt_b.shape[1])}
    used, out = set(), []
    for at in anchor_topics:
        aset = a_sets[at]
        best, best_j = None, -1.0
        for bt, bset in b_sets.items():
            if bt in used:
                continue
            u = len(aset | bset)
            jac = (len(aset & bset) / u) if u else 0.0
            if jac > best_j:
                best_j, best = jac, bt
        if best is not None:
            used.add(best)
        out.append(best)
    return out


def _exemplar_jaccard(tt_a, a, tt_b, b, m):
    A = set(exemplars(tt_a, a, m)); B = set(exemplars(tt_b, b, m))
    u = len(A | B)
    return (len(A & B) / u) if u else 0.0


# ── Per-topic LLM scores (coherence / intruder) already saved by evaluation.py ─
def load_topic_scores(results_dir, enc, trainer, seed, k, warmup, name):
    """Per-topic score dict {topic_id:int -> score:float or None} from evaluation/<name>.jsonl,
    mirroring build_topic_table.py's parse. Higher = better for BOTH:
      coherence -> mean of the LLM 'scores' list (0-100);
      intruder  -> value * 100 (intruder-detection rate, %).
    Raises FileNotFoundError (with the path) if the eval file is absent for this cell."""
    fp = os.path.join(eval_dir(results_dir, enc, trainer, seed, k, warmup), f"{name}.jsonl")
    if not os.path.isfile(fp):
        raise FileNotFoundError(
            f"Missing {name}.jsonl for --topic-select {name}gap (run evaluation for this cell first): {fp}")
    with open(fp) as f:
        r = json.loads(f.readline())["result"]
    out = {}
    for tid, val in r.items():
        t = int(tid)
        if name == "coherence":
            sc = val.get("scores") if isinstance(val, dict) else None
            out[t] = float(np.mean(sc)) if sc else None
        else:                                        # intruder: scalar rate in [0,1]
            out[t] = 100.0 * float(val)
    return out


# ── Topic-row selection: pick MATCHED pairs that MAXIMISE the Dual↔baseline difference ───────────
def _zscore(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)


def _candidate_anchors(tt):
    """Dual topic ids with at least one nonzero exemplar (skip empty/degenerate topics)."""
    nnz = np.asarray(tt.getnnz(axis=0)).ravel()
    return [int(t) for t in np.where(nnz > 0)[0]]


def _best_word_match(w_a, w_b, at, topn=30):
    """Baseline topic with the MOST top-word overlap with Dual topic `at` (non-unique). Returns
    (bt, overlap). Non-unique (best per anchor) beats greedy-unique here: greedy-unique forces late
    anchors onto leftover baseline topics, creating spurious off-theme pairs; best-per-anchor always
    gives each Dual topic its genuine same-theme counterpart (dedupe happens at final selection)."""
    aw = set(w_a[at][:topn])
    best, bo = None, -1
    for bt, ow in w_b.items():
        ov = len(aw & set(ow[:topn]))
        if ov > bo:
            bo, best = ov, bt
    return best, bo


def _best_image_match(tt_a, a, tt_b, m):
    """Baseline topic whose top-m exemplar image set overlaps most (Jaccard) with Dual topic `a`."""
    A = set(exemplars(tt_a, a, m))
    best, bj = None, -1.0
    for bt in range(tt_b.shape[1]):
        B = set(exemplars(tt_b, bt, m))
        u = len(A | B)
        j = (len(A & B) / u) if u else 0.0
        if j > bj:
            bj, best = j, bt
    return best, bj


def rank_topic_pairs(blocks, n_topics, mode, match, m, purity_m,
                     y=None, pair_scores=None, min_match=0.0, verbose=False, allowed=None):
    """Return row_topics = [[t_dual, t_base], ...] (or [[t_dual]] with no baseline) — the top-n_topics
    rows under `mode`. The point: earlier defaults ('aligned'/'salience') pick broad topics both models
    nail, so the two sides look identical. The GAP modes instead match each Dual topic to its SAME-theme
    baseline topic, then rank the pairs by how much the two models DIFFER, surfacing the rows where the
    difference is largest. match='words' pairs by top-word overlap (fixes the THEME, lets image sets /
    purity diverge — the difference we want to show); 'images' by exemplar-set Jaccard. `min_match` gates
    out weak (off-theme) pairings so every shown row is a fair same-theme comparison; final selection
    dedupes baseline topics so one baseline topic can't fill several rows."""
    nblk = len(blocks)
    tt_d, w_d = blocks[0][1], blocks[0][2]
    # `allowed` (from the concrete + coverage filter) restricts which Dual anchor topics may be shown.
    ok = (lambda t: allowed is None or t in allowed)
    cand_anchors = [t for t in _candidate_anchors(tt_d) if ok(t)]

    if nblk < 2:                                     # no baseline to gap against
        if mode == "puritygap" and y is not None:    # show Dual's own cleanest topics
            anchors = sorted(cand_anchors, key=lambda t: -column_purity(tt_d, t, y, purity_m)[0])
            return [[t] for t in anchors[:n_topics]]
        return [[t] for t in top_topics(blocks[0][3], n_topics) if ok(t)][:n_topics]

    tt_b, w_b = blocks[1][1], blocks[1][2]

    # Back-compat modes: pick anchors by their own criterion, then match to a same-theme baseline topic.
    if mode in ("aligned", "salience"):
        ranked = aligned_topics(w_d, w_b, len(w_d)) if mode == "aligned" else top_topics(blocks[0][3], len(w_d))
        anchor = [t for t in ranked if ok(t)][:n_topics]
        matched = (match_topics(w_d, w_b, anchor) if match == "words"
                   else match_topics_by_images(tt_d, tt_b, anchor, m))
        return [[a, bt] for a, bt in zip(anchor, matched)]

    # Gap modes: best same-theme baseline match per Dual topic, gate by match quality, score, rank, dedupe.
    cand = []                                        # (a, bt, match_quality)
    for a in cand_anchors:
        bt, q = (_best_word_match(w_d, w_b, a) if match == "words"
                 else _best_image_match(tt_d, a, tt_b, m))
        if bt is not None and q >= min_match:
            cand.append((a, bt, q))
    if not cand:
        return []

    if mode == "puritygap":
        scores = [column_purity(tt_d, a, y, purity_m)[0] - column_purity(tt_b, bt, y, purity_m)[0]
                  for a, bt, _ in cand]
    elif mode in ("coherencegap", "intrudergap"):
        ds, bs = pair_scores
        scores = [(ds.get(a) - bs.get(bt)) if (ds.get(a) is not None and bs.get(bt) is not None)
                  else -np.inf for a, bt, _ in cand]      # missing per-topic score -> never selected
    elif mode == "divergence":                        # same theme, most-disjoint image sets
        scores = [1.0 - _exemplar_jaccard(tt_d, a, tt_b, bt, m) for a, bt, _ in cand]
    elif mode == "combined":                          # both purer AND more divergent (z-normalised sum)
        gaps = np.array([column_purity(tt_d, a, y, purity_m)[0] - column_purity(tt_b, bt, y, purity_m)[0]
                         for a, bt, _ in cand])
        divs = np.array([1.0 - _exemplar_jaccard(tt_d, a, tt_b, bt, m) for a, bt, _ in cand])
        scores = list(_zscore(gaps) + _zscore(divs))
    else:
        raise ValueError(f"unknown topic-select mode: {mode}")

    order = sorted(range(len(cand)), key=lambda i: -scores[i])
    out, used_b = [], set()
    for i in order:
        a, bt, q = cand[i]
        if not np.isfinite(scores[i]) or bt in used_b:  # dedupe baseline; drop unscorable pairs
            continue
        used_b.add(bt)
        out.append([a, bt])
        if verbose:
            print(f"    row {len(out)}: Dual t{a} vs base t{bt}  score={scores[i]:+.3f}  match_q={q:g}")
        if len(out) >= n_topics:
            break
    return out


def select_images_puritygap(tt_dual, tt_base, y, n, purity_m):
    """img2topic image selection that maximises the Dual↔baseline difference: pick images whose Dual
    dominant-topic purity most EXCEEDS their baseline dominant-topic purity (diversified — one per
    distinct Dual dominant topic). Falls back to fewer if not enough valid images."""
    dom_d, mx_d = dominant_topic(tt_dual)
    dom_b, mx_b = dominant_topic(tt_base)
    pur_d = np.array([column_purity(tt_dual, t, y, purity_m)[0] for t in range(tt_dual.shape[1])])
    pur_b = np.array([column_purity(tt_base, t, y, purity_m)[0] for t in range(tt_base.shape[1])])
    valid = np.where((mx_d > 0) & (mx_b > 0))[0]
    if valid.size == 0:
        return []
    gap = pur_d[dom_d[valid]] - pur_b[dom_b[valid]]
    order = valid[np.argsort(-gap)]
    seen, picked = set(), []
    for d in order:
        t = int(dom_d[d])
        if t in seen:
            continue
        seen.add(t); picked.append(int(d))
        if len(picked) >= n:
            break
    return picked[:n]


# ── Combined img2topic: ONE figure, one row per (encoder × activation) setting ───────────────────
def render_img2topic_all(args):
    """Single figure with one row per setting: [setting label | top --img-cols judged Dual-win images],
    each image captioned with the aligned red (DualSAE) / blue (baseline) dominant-topic labels — the
    per-setting img2topic showcases condensed into one cross-setting overview."""
    encs = args.all_encoders.split()
    acts = args.all_activations.split()
    docs = load_documents(args.documents)
    n = args.img_cols

    rows_data = []
    for enc in encs:
        for act in acts:
            dual_tr, base_tr = ACT2TRAINER[act]
            try:
                tt_d, w_d, _ = load_cell(args.results_dir, enc, dual_tr, args.seed, args.k, "")
                tt_b, w_b, _ = load_cell(args.results_dir, enc, base_tr, args.seed, args.k, "")
            except FileNotFoundError as e:
                print(f"[skip] {enc}/{act}: {e}"); continue
            # judged Dual-wins only (this figure IS the judged showcase)
            jpath = os.path.join(topics_cell_dir(args.results_dir, enc, dual_tr, args.seed, args.k, ""),
                                 "img2topic_judgments.csv")
            wins, nj = [], 0
            if os.path.isfile(jpath):
                with open(jpath) as jf:
                    for row in csv.DictReader(jf):
                        nj += 1
                        if row["verdict"] == "dual":
                            wins.append(int(row["row_idx"]))
            if not wins:
                print(f"[skip] {enc}/{act}: no judgments/wins ({jpath})"); continue
            idxs = select_images(tt_d, n, "confident", args.rng_seed, restrict=set(wins))
            lab_d = lab_b = None
            if args.use_llm_labels:
                ld = load_topic_labels(topics_cell_dir(args.results_dir, enc, dual_tr, args.seed, args.k, ""))
                lab_d = {t: s for t, (c, s) in ld.items() if s} or None
                lb = load_topic_labels(topics_cell_dir(args.results_dir, enc, base_tr, args.seed, args.k, ""))
                lab_b = {t: s for t, (c, s) in lb.items() if s} or None
            rows_data.append(dict(enc=enc, act=act, idxs=idxs, w_d=w_d, w_b=w_b,
                                  dom_d=dominant_topic(tt_d)[0], dom_b=dominant_topic(tt_b)[0],
                                  lab_d=lab_d, lab_b=lab_b, wins=len(wins), nj=nj))
    if not rows_data:
        raise SystemExit("img2topic-all: no settings available (run the judge first).")

    def topic_label(t, words, labels):
        return labels[t] if (labels is not None and t in labels) else ", ".join(words[t][:max(args.topwords, 1)])

    for rd in rows_data:
        rd["dcaps"] = [_wrap_label(topic_label(int(rd["dom_d"][d]), rd["w_d"], rd["lab_d"])) for d in rd["idxs"]]
        rd["bcaps"] = [_wrap_label(topic_label(int(rd["dom_b"][d]), rd["w_b"], rd["lab_b"])) for d in rd["idxs"]]
    # figure-wide caption rows (same offsets in every cell → labels align across ALL rows/settings)
    max_d = max(len(c) for rd in rows_data for c in rd["dcaps"])
    max_b = max(len(c) for rd in rows_data for c in rd["bcaps"])
    FS, LH = 7.2, 0.105                            # line height = the bare 7.2pt line box (no leading)
    gap_frac = 0.05 + LH * (max_d + max_b)         # minimal inter-row gap: caption stack + small margin
    nset = len(rows_data)
    labw = 0.30                                    # narrow label column (labels are rotated vertical)
    top_in = 0.30 + 0.22 * 2
    fig_h = 1.15 * nset * (1.0 + gap_frac) + top_in
    fig = plt.figure(figsize=(1.15 * (n + labw) + 0.2, fig_h))
    gs = GridSpec(nset, n + 1, figure=fig, wspace=0.0, hspace=gap_frac,
                  width_ratios=[labw] + [1.0] * n,
                  top=1.0 - top_in / fig_h, bottom=0.25 / fig_h, left=0.005, right=0.995)
    ENC_NAME = {"clip": "CLIP", "siglip": "SigLIP", "vit": "ViT"}   # official model capitalization
    for r, rd in enumerate(rows_data):
        axl = fig.add_subplot(gs[r, 0]); axl.axis("off")
        axl.text(0.4, 0.5, f"{ENC_NAME.get(rd['enc'], rd['enc'])} / {rd['act']}",
                 ha="center", va="center", fontsize=9.5, fontweight="bold", rotation=90)
        for j, d in enumerate(rd["idxs"]):
            ax = fig.add_subplot(gs[r, 1 + j]); ax.axis("off")
            ax.imshow(np.asarray(load_image(args.image_root, docs[d][0])))
            ax.text(0.5, -0.04, "\n".join(rd["dcaps"][j]), transform=ax.transAxes, ha="center",
                    va="top", fontsize=FS, fontweight="bold", color="#c1121f")
            ax.text(0.5, -0.04 - LH * max_d - 0.005, "\n".join(rd["bcaps"][j]), transform=ax.transAxes,
                    ha="center", va="top", fontsize=FS, fontweight="bold", color="#1f4e9c")
    fig.suptitle(f"Image → topic — PHI-4-judged clear DualSAE wins (seed{args.seed}, K={args.k})\n"
                 f"bold red = DualSAE topic,  bold blue = baseline topic", fontsize=11, y=0.998)
    out_dir = args.out_dir or os.path.join(args.results_dir, "qualitative")
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for ext in ("png", "pdf"):
        fp = os.path.join(out_dir, f"img2topic_all_K{args.k}.{ext}")
        fig.savefig(fp, dpi=150, bbox_inches="tight"); saved.append(fp)
    plt.close(fig)
    print("Saved:")
    for fp in saved:
        print(f"  {fp}")


# ── Cell scan: objectively find the cell where Dual's purity advantage is clearest ──────────────
def scan_cells(args, dual_tr, base_tr, y, purity_m, min_match):
    """Rank cells (encoder × K [× seed]) by the mean top-n_topics purity gap (Dual − baseline), so we
    pick the cell where DualSAE's monosemanticity advantage reads clearest rather than guessing.
    Read-only; reuses rank_topic_pairs('puritygap'). Missing cells (or row-count != len(y)) are skipped."""
    encs = args.scan_encoders.split() if args.scan_encoders else [args.encoder]
    ks = [int(x) for x in args.scan_ks.split()] if args.scan_ks else [args.k]
    seeds = [int(x) for x in str(args.scan_seeds).split()] if args.scan_seeds else [args.seed]
    if base_tr is None:
        raise SystemExit("--scan needs a baseline (purity gap is Dual − baseline); don't pass --baseline none.")
    rows = []
    for enc in encs:
        for k in ks:
            gaps = []
            for sd in seeds:
                try:
                    tt_d, w_d, sal_d = load_cell(args.results_dir, enc, dual_tr, sd, k, args.warmup)
                    tt_b, w_b, sal_b = load_cell(args.results_dir, enc, base_tr, sd, k, args.warmup)
                except FileNotFoundError:
                    continue
                if tt_d.shape[0] != len(y) or tt_b.shape[0] != len(y):
                    print(f"  [skip] {enc} K{k} seed{sd}: theta rows != labels "
                          f"({tt_d.shape[0]}/{tt_b.shape[0]} vs {len(y)})")
                    continue
                blocks = [("d", tt_d, w_d, sal_d), ("b", tt_b, w_b, sal_b)]
                rt = rank_topic_pairs(blocks, args.n_topics, "puritygap", args.match,
                                      args.m_exemplars, purity_m, y=y, min_match=min_match)
                g = [column_purity(tt_d, p[0], y, purity_m)[0] - column_purity(tt_b, p[1], y, purity_m)[0]
                     for p in rt if len(p) > 1 and p[1] is not None]
                if g:
                    gaps.append(float(np.mean(g)))
            if gaps:
                rows.append((enc, k, float(np.mean(gaps)), len(gaps)))
    rows.sort(key=lambda r: -r[2])
    print(f"\nCell scan — mean top-{args.n_topics} purity gap (Dual − baseline), higher = clearer Dual win:")
    print(f"  {'encoder':8s} {'K':>5s} {'mean_Δpurity':>13s} {'nseed':>6s}")
    for enc, k, g, n in rows:
        print(f"  {enc:8s} {k:5d} {g:13.3f} {n:6d}")
    if not rows:
        print("  (no cells found — check --results-dir / --scan-encoders / --scan-ks / --warmup)")
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────
TOPIC_SELECT_MODES = ["puritygap", "coherencegap", "intrudergap", "divergence", "combined",
                      "aligned", "salience"]
NEEDS_LABELS = {"puritygap", "combined"}             # modes that require ground-truth classes


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results-dir", default="./outputs/topic_modeling/runs/imagenet_paired")
    p.add_argument("--encoder", default="clip")
    p.add_argument("--activation", default="TopK", choices=list(ACT2TRAINER),
                   help="derives the Dual+baseline trainer pair; override with --dual/--baseline")
    p.add_argument("--dual", default=None, help="explicit dual trainer (overrides --activation)")
    p.add_argument("--baseline", default=None, help="explicit baseline trainer ('' or 'none' to skip)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--warmup", default="", help="warmup<W> segment (ablation tree only; '' for paired run)")
    p.add_argument("--documents", default="./outputs/topic_modeling/imagenet_val/data/documents.jsonl")
    p.add_argument("--image-root", default="./data/imagenet/val")
    p.add_argument("--class-index", default="./assets/imagenet_class_index.json",
                   help="imagenet_class_index.json for ground-truth purity (synset->class)")
    p.add_argument("--out-dir", default=None, help="default <results-dir>/qualitative")
    p.add_argument("--n-images", type=int, default=100, help="images in the img2topic grid")
    p.add_argument("--img-cols", type=int, default=10, help="img2topic grid columns (rows = n-images/cols)")
    p.add_argument("--n-topics", type=int, default=6)
    p.add_argument("--m-exemplars", type=int, default=20, help="exemplar images per topic (topic2img)")
    p.add_argument("--ex-cols", type=int, default=10, help="exemplars per row; images wrap into ceil(m/ex_cols) rows")
    p.add_argument("--purity-m", type=int, default=0,
                   help="exemplars used to compute the majority class for ranking (0 = same as --m-exemplars; "
                        "a larger window gives a more stable purity gap while the shown grid stays --m-exemplars)")
    p.add_argument("--topwords", type=int, default=1)
    p.add_argument("--select", default="confident", choices=["confident", "random", "puritygap"],
                   help="img2topic image selection ('puritygap' = images where Dual dominant-topic purity "
                        "most exceeds the baseline's)")
    p.add_argument("--topic-select", default="puritygap", choices=TOPIC_SELECT_MODES,
                   help="topic2img rows. GAP modes (default 'puritygap') match each Dual topic to its "
                        "same-theme baseline topic then rank by how much they DIFFER — surfacing the rows "
                        "where DualSAE wins. 'coherencegap'/'intrudergap' use the on-disk per-topic LLM "
                        "scores; 'divergence' = same theme, most-disjoint image sets; 'combined' = purer AND "
                        "more divergent; 'aligned'/'salience' are the old agreement-biased picks.")
    p.add_argument("--match", default="words", choices=["words", "images"],
                   help="how a Dual topic is paired with its baseline counterpart: 'words'=top-word overlap "
                        "(fixes the THEME, lets purity diverge — best for gap modes); 'images'=exemplar-set Jaccard")
    p.add_argument("--min-match", type=float, default=-1.0,
                   help="minimum match quality to keep a pair (word-overlap count if --match words, Jaccard if "
                        "images); -1 = auto (words:5, images:0). Gates out off-theme pairings so every row is fair.")
    p.add_argument("--mark-intruders", action=argparse.BooleanOptionalAction, default=True,
                   help="red-border off-majority-class exemplars in topic2img")
    p.add_argument("--concrete-only", action=argparse.BooleanOptionalAction, default=True,
                   help="keep only topics PHI-4 classified 'concrete' (needs clusters_annotated.csv; "
                        "no-op if unannotated)")
    p.add_argument("--max-coverage", type=float, default=0.30,
                   help="drop topics active in > this fraction of images (cluster_ratio)")
    p.add_argument("--use-llm-labels", action=argparse.BooleanOptionalAction, default=True,
                   help="title cells with the PHI-4 llm_summary label instead of raw top-words")
    p.add_argument("--judged-only", action=argparse.BooleanOptionalAction, default=True,
                   help="img2topic: show only images the PHI-4 judge scored as clear DualSAE wins "
                        "(needs img2topic_judgments.csv from judge_img2topic.py; no-op if absent)")
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--show-caption", action="store_true")
    p.add_argument("--figures", default="both", choices=["both", "img2topic", "topic2img", "img2topic-all"])
    p.add_argument("--all-encoders", default="clip siglip vit", help="img2topic-all: encoders (one row each)")
    p.add_argument("--all-activations", default="TopK BatchTopK JumpReLU",
                   help="img2topic-all: activations (one row each per encoder)")
    p.add_argument("--scan", action="store_true",
                   help="don't render; print cells ranked by mean purity gap (Dual − baseline) then exit")
    p.add_argument("--scan-encoders", default="", help="space-separated encoders for --scan (default: --encoder)")
    p.add_argument("--scan-ks", default="", help="space-separated Ks for --scan (default: --k)")
    p.add_argument("--scan-seeds", default="", help="space-separated seeds for --scan (default: --seed)")
    args = p.parse_args()

    if args.figures == "img2topic-all":            # combined one-row-per-setting figure, then done
        render_img2topic_all(args)
        return

    dual_tr = args.dual or ACT2TRAINER[args.activation][0]
    base_tr = args.baseline if args.baseline is not None else ACT2TRAINER[args.activation][1]
    if base_tr in ("", "none", "None"):
        base_tr = None
    purity_m = args.purity_m if args.purity_m > 0 else args.m_exemplars
    # auto match-quality floor: word-overlap needs a few shared words to be same-theme; image Jaccard
    # is ~0 even for same-theme pairs (top-m sets barely overlap across models) so don't gate it.
    min_match = args.min_match if args.min_match >= 0 else (5.0 if args.match == "words" else 0.0)

    # Ground-truth ImageNet classes per row (row order == theta_topic rows). Needed for purity/combined
    # topic selection, for the purity annotation + intruder borders, and for --scan. Load once; degrade
    # gracefully to no-labels if the class index is absent (only label-free modes remain usable).
    docs = load_documents(args.documents)
    y = idx2name = None
    if os.path.isfile(args.class_index):
        y, idx2name = load_labels(docs, args.class_index)
    else:
        print(f"  [warn] class index not found ({args.class_index}); purity/intruder annotation disabled.")

    if args.scan:
        if y is None:
            raise SystemExit(f"--scan needs ground-truth labels; class index missing: {args.class_index}")
        scan_cells(args, dual_tr, base_tr, y, purity_m, min_match)
        return

    if args.topic_select in NEEDS_LABELS and y is None:
        raise SystemExit(f"--topic-select {args.topic_select} needs ground-truth labels; "
                         f"class index missing: {args.class_index}")

    out_dir = args.out_dir or os.path.join(args.results_dir, "qualitative")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.encoder}_{args.activation}_K{args.k}" + (f"_w{args.warmup}" if args.warmup else "")

    print(f"Loading cell: {args.encoder} {dual_tr} vs {base_tr} seed{args.seed} K{args.k} "
          f"warmup='{args.warmup}'")
    tt_dual, w_dual, sal_dual = load_cell(args.results_dir, args.encoder, dual_tr, args.seed, args.k, args.warmup)
    if base_tr is not None:
        try:
            tt_base, w_base, sal_base = load_cell(args.results_dir, args.encoder, base_tr, args.seed, args.k, args.warmup)
        except FileNotFoundError as e:
            print(f"  [warn] baseline cell missing ({e}); rendering DualSAE only.")
            tt_base = w_base = sal_base = None; base_tr = None
    else:
        tt_base = w_base = sal_base = None

    if len(docs) != tt_dual.shape[0]:
        raise ValueError(f"documents ({len(docs)}) != theta_topic rows ({tt_dual.shape[0]}); "
                         f"row alignment broken — check --documents matches this encoder's val set.")

    saved = []
    if args.figures in ("both", "img2topic"):
        # PHI-4 judge filter: keep only images where the judge scored the Dual topic CLEARLY better
        # than the baseline's (img2topic_judgments.csv from judge_img2topic.py). Top up with ties if
        # there are fewer wins than --n-images; fall back to unfiltered if the cell isn't judged.
        restrict, judge_note = None, ""
        if args.judged_only:
            jpath = os.path.join(topics_cell_dir(args.results_dir, args.encoder, dual_tr,
                                                 args.seed, args.k, args.warmup),
                                 "img2topic_judgments.csv")
            if os.path.isfile(jpath):
                by_verdict = {"dual": [], "base": [], "tie": []}
                with open(jpath) as jf:
                    for row in csv.DictReader(jf):
                        by_verdict[row["verdict"]].append(int(row["row_idx"]))
                nj = sum(len(v) for v in by_verdict.values())
                wins = by_verdict["dual"]
                restrict = set(wins)                       # WINS ONLY — no tie top-up (would dilute
                if len(wins) < args.n_images:              # exactly the contrast this figure shows)
                    print(f"  [note] {len(wins)} clear Dual wins < --n-images {args.n_images}; "
                          f"the grid will show all {len(wins)} wins.")
                judge_note = (f"PHI-4 judge (both-order, debiased) on {nj} candidates: DualSAE clearly "
                              f"better {len(wins)/nj:.0%}, baseline {len(by_verdict['base'])/nj:.0%}, "
                              f"tie {len(by_verdict['tie'])/nj:.0%} — showing only Dual-win samples")
                print(f"  {judge_note}")
            else:
                print(f"  [warn] no judgments ({jpath}); rendering unfiltered "
                      f"(run judge_img2topic.py first).")
        if args.select == "puritygap" and tt_base is not None and y is not None:
            idxs = select_images_puritygap(tt_dual, tt_base, y, args.n_images, purity_m)
        else:
            idxs = select_images(tt_dual, args.n_images,
                                 "confident" if args.select == "puritygap" else args.select,
                                 args.rng_seed, restrict=restrict)
        # PHI-4 labels for the dominant-topic cells (label-only convention; falls back to raw words)
        labels_d = labels_b = None
        if args.use_llm_labels:
            ld = load_topic_labels(topics_cell_dir(args.results_dir, args.encoder, dual_tr,
                                                   args.seed, args.k, args.warmup))
            labels_d = {t: s for t, (c, s) in ld.items() if s} or None
            if base_tr is not None:
                lb = load_topic_labels(topics_cell_dir(args.results_dir, args.encoder, base_tr,
                                                       args.seed, args.k, args.warmup))
                labels_b = {t: s for t, (c, s) in lb.items() if s} or None
        i2t_title = f"Image → topic  ({args.encoder}, {args.activation}, K={args.k})"
        if judge_note:
            i2t_title += f"\n{judge_note}"
        fig = fig_img2topic(tt_dual, w_dual, tt_base, w_base, docs, args.image_root, idxs,
                            dual_tr, base_tr or "-", args.topwords, args.show_caption,
                            labels_d=labels_d, labels_b=labels_b, img_cols=args.img_cols,
                            title=i2t_title)
        for ext in ("png", "pdf"):
            fp = os.path.join(out_dir, f"img2topic_{tag}.{ext}")
            fig.savefig(fp, dpi=150, bbox_inches="tight"); saved.append(fp)
        plt.close(fig)

    if args.figures in ("both", "topic2img"):
        blocks = [(f"DualSAE — {dual_tr}", tt_dual, w_dual, sal_dual)]
        if tt_base is not None:
            blocks.append((f"Baseline — {base_tr}", tt_base, w_base, sal_base))

        # PHI-4 annotations (abstract/concrete + label) + coverage: restrict Dual anchors to concrete +
        # cluster_ratio ≤ max-coverage, and (label-only) title cells with the LLM summary. Missing
        # annotation → concrete filter is a no-op (all kept); coverage still applies (always on disk).
        dcell = topics_cell_dir(args.results_dir, args.encoder, dual_tr, args.seed, args.k, args.warmup)
        lab_d = load_topic_labels(dcell); ratio_d = load_cluster_ratio(dcell)
        concrete_ok = {t for t in range(tt_dual.shape[1])
                       if not args.concrete_only or lab_d.get(t, ("concrete", ""))[0] == "concrete"}
        allowed = {t for t in concrete_ok if ratio_d.get(t, 0.0) <= args.max_coverage}
        if not allowed and concrete_ok:
            # Coverage filter is degenerate in this cell (e.g. JumpReLU: every topic is active in ~all
            # images, cluster_ratio≈1.0 for all 100 topics) → the >30% rule carries no information here.
            # Fall back to the concrete-only filter rather than emptying the figure.
            print(f"  [warn] coverage filter (<= {args.max_coverage}) removes ALL topics "
                  f"(min ratio {min(ratio_d.values()):.2f}); falling back to concrete-only.")
            allowed = concrete_ok
        lab_b = {}
        if tt_base is not None:
            lab_b = load_topic_labels(topics_cell_dir(args.results_dir, args.encoder, base_tr,
                                                      args.seed, args.k, args.warmup))
        labels_arg = None
        if args.use_llm_labels and (lab_d or lab_b):
            labels_arg = [{t: s for t, (c, s) in lab_d.items() if s}]
            if tt_base is not None:
                labels_arg.append({t: s for t, (c, s) in lab_b.items() if s})

        # for the LLM-score gap modes, load both models' per-topic coherence/intruder from disk
        pair_scores = None
        if args.topic_select in ("coherencegap", "intrudergap") and tt_base is not None:
            name = args.topic_select[:-3]                        # 'coherencegap'->'coherence', etc.
            ds = load_topic_scores(args.results_dir, args.encoder, dual_tr, args.seed, args.k, args.warmup, name)
            bs = load_topic_scores(args.results_dir, args.encoder, base_tr, args.seed, args.k, args.warmup, name)
            pair_scores = (ds, bs)

        print(f"Selecting {args.n_topics} topic rows via '{args.topic_select}' (match={args.match}, "
              f"min_match={min_match:g}, concrete_only={args.concrete_only}, "
              f"max_coverage={args.max_coverage}, {len(allowed)} allowed anchors):")
        row_topics = rank_topic_pairs(blocks, args.n_topics, args.topic_select, args.match,
                                      args.m_exemplars, purity_m, y=y, pair_scores=pair_scores,
                                      min_match=min_match, verbose=True, allowed=allowed)
        if not row_topics:
            raise SystemExit(f"No topic pairs passed the '{args.topic_select}' selection "
                             f"(min_match={min_match:g}, concrete_only={args.concrete_only}, "
                             f"max_coverage={args.max_coverage}); relax the filters or --topic-select.")
        fig = fig_exemplar_grid(blocks, docs, args.image_root, row_topics, args.m_exemplars,
                                args.topwords, y=y, idx2name=idx2name, purity_m=purity_m,
                                mark_intruders=args.mark_intruders, ex_cols=args.ex_cols, col_kind="topic",
                                labels=labels_arg,
                                title=f"Topic → exemplar images  ({args.encoder}, {args.activation}, "
                                      f"K={args.k}; select={args.topic_select})")
        for ext in ("png", "pdf"):
            fp = os.path.join(out_dir, f"topic2img_{tag}_{args.topic_select}.{ext}")
            fig.savefig(fp, dpi=150, bbox_inches="tight"); saved.append(fp)
        plt.close(fig)

    print("Saved:")
    for fp in saved:
        print(f"  {fp}")


if __name__ == "__main__":
    main()
