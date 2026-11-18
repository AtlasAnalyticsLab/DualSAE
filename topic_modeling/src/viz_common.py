#!/usr/bin/env python
"""
Shared helpers for the SAE-TM qualitative figures (CPU-only), used by visualize_topics.py
(topic-level exemplar grids: topic2img / img2topic) and judge_img2topic.py.

The core idea reused everywhere: theta_topic (CSC [N, K]) and theta_csr (CSC [N, n_features]) are the
same kind of object — a sparse [images × columns] activation matrix. A "column" is a topic in one case
and an individual feature in the other, but exemplar selection (`exemplars`) and majority-class purity
(`column_purity`) are identical, and both render through the same grid (`fig_exemplar_grid`).
"""

import os
import csv
import json

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")                       # headless / offline
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# activation token -> (dual trainer, baseline trainer); each Dual is paired with its same-activation baseline.
ACT2TRAINER = {
    "TopK":      ("DualTopKTrainer", "TopKTrainer"),
    "BatchTopK": ("DualBatchTopKTrainer", "BatchTopKTrainer"),
    "JumpReLU":  ("DualJumpReluTrainer", "JumpReluTrainer"),
}


# ── Cell path helpers ────────────────────────────────────────────────────────
def topics_cell_dir(results_dir, enc, trainer, seed, k, warmup=""):
    """Per-(topic-count K) topic artifacts dir. warmup segment only for the ablation tree."""
    seg = f"/warmup{warmup}" if warmup else ""
    return os.path.join(results_dir, enc, "topics", f"{trainer}{seg}", f"seed{seed}", f"K{k}")


def interpreted_dir(results_dir, enc, trainer, seed):
    """Per-FEATURE (pre-clustering) artifacts dir: theta_csr.npz, word_emission_probabilities.pt,
    feature_probabilities.pt. Note: at seed level, NOT per-K (features are the same across topic K)."""
    return os.path.join(results_dir, enc, "interpreted", f"{trainer}", f"seed{seed}")


def trained_dir(results_dir, enc, trainer, seed):
    """Trained-model dir: best_val_monoscore_model.pt, autoencoder_final.pt, dataset_config_seed<S>.json."""
    return os.path.join(results_dir, enc, "trained_models", f"{trainer}", f"seed{seed}")


def eval_dir(results_dir, enc, trainer, seed, k, warmup=""):
    """evaluation/ sibling of the topics/ cell dir (same trainer[/warmup]/seed/K layout)."""
    seg = f"/warmup{warmup}" if warmup else ""
    return os.path.join(results_dir, enc, "evaluation", f"{trainer}{seg}", f"seed{seed}", f"K{k}")


# ── Documents / images / labels ──────────────────────────────────────────────
def load_documents(path):
    """List of (rel_path, caption) in row order (== theta_topic / theta_csr row order)."""
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            recs.append((o["id"], o.get("document", "")))
    return recs


def load_image(image_root, rel_path, size=256):
    """Square RGB thumbnail; gray placeholder if the file is missing/unreadable."""
    try:
        im = Image.open(os.path.join(image_root, rel_path)).convert("RGB")
        return im.resize((size, size))
    except Exception:
        return Image.new("RGB", (size, size), (210, 210, 210))


def load_labels(docs, class_index_path):
    """Ground-truth ImageNet class per row, in row order.
    docs = load_documents(...) output [(rel_path, caption)]; rel_path = '<synset>/..._val_*.JPEG'.
    class_index_path -> imagenet_class_index.json = {"0": ["n01440764","tench"], ...}.
    Returns (y int32[N], idx2name {class_idx: human_name}); unknown synset -> label -1."""
    ci = json.load(open(class_index_path))
    syn2idx = {v[0]: int(k) for k, v in ci.items()}
    idx2name = {int(k): v[1] for k, v in ci.items()}
    y = np.fromiter((syn2idx.get(rel.split("/")[0], -1) for rel, _ in docs),
                    dtype=np.int32, count=len(docs))
    return y, idx2name


# ── Text helper ──────────────────────────────────────────────────────────────
def wrap_words(words, n_words, per_line=4):
    ws = words[:n_words]
    return "\n".join(", ".join(ws[i:i + per_line]) for i in range(0, len(ws), per_line))


# ── Generic column (topic OR feature) exemplars + purity ─────────────────────
def exemplars(mat, col, m):
    """Top-m image rows for column `col` of a sparse [N, C] matrix, by descending activation (>0).
    Works identically for a topic column (theta_topic) or a feature column (theta_csr)."""
    v = mat[:, col].toarray().ravel()
    order = np.argsort(-v)
    return [int(i) for i in order[:m] if v[i] > 0]


# ── PHI-4 topic/feature annotations (abstract/concrete + label) from annotate_topics.py ──────────
def load_topic_labels(cell_dir):
    """{cluster_id:int -> (llm_classification, llm_summary)} from clusters_annotated.csv; {} if absent."""
    fp = os.path.join(cell_dir, "clusters_annotated.csv")
    out = {}
    if os.path.isfile(fp):
        with open(fp) as f:
            for row in csv.DictReader(f):
                out[int(row["cluster_id"])] = (row.get("llm_classification", "concrete"),
                                               row.get("llm_summary", ""))
    return out


def load_cluster_ratio(cell_dir):
    """{cluster_id:int -> cluster_ratio:float} (fraction of images the topic is active in) from clusters.csv."""
    out = {}
    with open(os.path.join(cell_dir, "clusters.csv")) as f:
        for row in csv.DictReader(f):
            out[int(row["cluster_id"])] = float(row["cluster_ratio"])
    return out


def load_feature_labels(interpreted_dir):
    """{feature_id:int -> (llm_classification, llm_summary, feature_ratio)} from features_annotated.csv; {} if absent."""
    fp = os.path.join(interpreted_dir, "features_annotated.csv")
    out = {}
    if os.path.isfile(fp):
        with open(fp) as f:
            for row in csv.DictReader(f):
                out[int(row["feature_id"])] = (row.get("llm_classification", "concrete"),
                                               row.get("llm_summary", ""), float(row.get("feature_ratio", 0.0)))
    return out


def random_exemplars(mat, col, m, rng):
    """Random sample of up to m image rows with NONZERO activation for column `col` (vs the top-m of
    `exemplars`). Random active members reveal whether a topic is GENUINELY coherent, not just cherry-
    picked by its most-confident images. `rng` = np.random.Generator (deterministic)."""
    v = mat[:, col].toarray().ravel()
    active = np.where(v > 0)[0]
    if active.size == 0:
        return []
    k = min(m, active.size)
    return [int(i) for i in rng.choice(active, size=k, replace=False)]


def column_purity(mat, col, y, m):
    """Majority-class purity of column `col`'s top-m exemplars: fraction sharing the single most common
    ImageNet class. Returns (purity, maj_idx, maj_count, n). Empty/label-less column -> (0,-1,0,n).
    Unknown-label (-1) exemplars count against purity (never form the majority)."""
    ex = exemplars(mat, col, m)
    if not ex:
        return 0.0, -1, 0, 0
    labs = y[ex]
    known = labs[labs >= 0]
    if known.size == 0:
        return 0.0, -1, 0, len(ex)
    vals, counts = np.unique(known, return_counts=True)
    j = int(np.argmax(counts))
    maj_idx, maj_count = int(vals[j]), int(counts[j])
    return maj_count / len(ex), maj_idx, maj_count, len(ex)


# ── Generic matched side-by-side exemplar grid ───────────────────────────────
def fig_exemplar_grid(blocks, docs, image_root, row_cols, m, topwords, title,
                      y=None, idx2name=None, purity_m=None, mark_intruders=True, ex_cols=10,
                      col_kind="topic", labels=None):
    """Render SIDE-BY-SIDE matched columns. blocks = list of (label, mat, id2words, salience); each block
    is a model shown as a column-group [words | m exemplars]. `row_cols[r][b]` = the column id (topic or
    feature) for row r, block b — so with 2 blocks a row is a MATCHED pair (same theme/concept) for direct
    comparison. `col_kind` just labels the cell ("topic"/"feature").

    If `y` (per-row ImageNet class) is given, each side is annotated with its majority class + purity over
    the DISPLAYED m exemplars, and every off-majority ("intruder") exemplar is outlined in red — so a purer
    grid vs a mixed grid reads at a glance even when the theme is identical."""
    nblk = len(blocks)
    ntop = len(row_cols)
    ex_rows = (m + ex_cols - 1) // ex_cols        # image sub-rows per column (e.g. m=20, ex_cols=10 → 2)
    per = 1 + ex_cols                             # cols per model block: 1 words + ex_cols images
    ncol = per * nblk
    wcol = 1.9                                    # words col: fits '[topic t]', word, class name, count
    width_ratios = ([wcol] + [1.0] * ex_cols) * nblk
    nrows = 1 + ntop * ex_rows                    # 1 header row + ex_rows image sub-rows per column
    pm = purity_m or m

    fig = plt.figure(figsize=(1.0 + (wcol + 1.0 * ex_cols) * nblk, 1.0 + 1.15 * ntop * ex_rows))
    gs = GridSpec(nrows, ncol, figure=fig, height_ratios=[0.3] + [1.0] * (ntop * ex_rows),
                  width_ratios=width_ratios, wspace=0.03, hspace=0.06)

    # one bold header per model, spanning that model's column-group
    for b, (label, _, _, _) in enumerate(blocks):
        axh = fig.add_subplot(gs[0, b * per:(b + 1) * per]); axh.axis("off")
        axh.text(0.5, 0.3, label, fontsize=12, fontweight="bold", va="center", ha="center")

    # each column occupies ex_rows sub-rows; its m exemplars wrap into an ex_rows×ex_cols grid, with the
    # words cell spanning those sub-rows. Rows are matched themes/concepts across models.
    for r in range(ntop):
        r0 = 1 + r * ex_rows
        for b, (label, mat, id2words, salience) in enumerate(blocks):
            c0 = b * per
            t = row_cols[r][b] if b < len(row_cols[r]) else None
            if t is None:
                continue
            ex = exemplars(mat, t, m)                            # the m images actually shown
            is_maj = [True] * len(ex)
            # LLM label (if provided for this block) replaces the raw "[col t] words" title
            if labels is not None and b < len(labels) and t in labels[b]:
                label_txt = labels[b][t]
            else:
                label_txt = f"[{col_kind} {t}]\n" + wrap_words(id2words[t], topwords, per_line=2)
            if y is not None:
                _, maj_idx, _, _ = column_purity(mat, t, y, pm)  # majority over the (wider) purity window
                is_maj = [int(l) == maj_idx and maj_idx >= 0 for l in y[ex]] if ex else []
                shown = int(sum(is_maj))
                if maj_idx >= 0:
                    maj_name = (idx2name or {}).get(maj_idx, str(maj_idx))
                    label_txt += f"\n{maj_name}\n{shown}/{len(ex)} ({shown / max(len(ex), 1):.0%})"
                else:
                    label_txt += "\n(mixed)"
            axw = fig.add_subplot(gs[r0:r0 + ex_rows, c0]); axw.axis("off")
            axw.text(0.04, 0.5, label_txt, va="center", ha="left", fontsize=9)
            for j in range(m):
                rr, cc = divmod(j, ex_cols)
                axi = fig.add_subplot(gs[r0 + rr, c0 + 1 + cc])
                axi.set_xticks([]); axi.set_yticks([])                # keep spines (frame) but no ticks
                if j < len(ex):
                    axi.imshow(np.asarray(load_image(image_root, docs[ex[j]][0])))
                    intruder = mark_intruders and (y is not None) and not is_maj[j]
                    for s in axi.spines.values():                     # red frame = off-majority-class image
                        s.set_visible(True)
                        s.set_edgecolor("red" if intruder else "0.85")
                        s.set_linewidth(2.4 if intruder else 0.4)
                else:
                    axi.axis("off")                                  # empty slot (fewer than m exemplars)
    fig.suptitle(title, fontsize=12, y=0.998)
    return fig
