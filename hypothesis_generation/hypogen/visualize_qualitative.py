"""Qualitative per-class hypothesis figures (concept / contrast layouts) for one cell.

The paper's class-level hypothesis figures are built by encoder_figs.py, which reuses this module's
row renderer (_draw_rows).

For a class A, show example images of A next to the natural-language hypotheses that DualSAE produced for
the latents the downstream classifier relies on — i.e. "why the model predicts class A". Two layouts:
  * concept-centric : rows = a class's top selected latents; each row = hypothesis text + example class-A
                      images that most activate that latent.
  * contrast        : the same class shown for baseline vs DualSAE (top vs bottom), to compare the concepts
                      each model offers.

Everything is read from artifacts already on disk (no recompute, no GPU):
  selected/<enc>/<act>/<var>/seed<S>/<method>_K<k>.json   (class_to_latents + class_to_scores)
  interpreted/<enc>/<act>/<var>/seed<S>/hypotheses.json    ({latent: {hypothesis, top_image_rel}})
  acts/<enc>/<act>/<var>/seed<S>/val_acts.npy              (N x dict_size; pick class-A images per latent)
  embeddings/{val_manifest.json, val_labels.npy}           (row -> image file + class)
Run (CPU):
  HG_DATASET=imagenet   python -m hypogen.visualize_qualitative            # auto-pick best-#Sig cell
  HG_DATASET=openimages python -m hypogen.visualize_qualitative --n-classes 4
  ... --images global   # tiles = each latent's global top-activators (any class) instead of class-A images
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from hypogen import config
from hypogen import class_names as cn


# --------------------------------------------------------------------------- artifact path helpers
def _select_path(enc, act, var, seed, method, k):
    return os.path.join(config.SELECT_DIR, enc, act, var, f"seed{seed}", f"{method}_K{k}.json")


def _interp_path(enc, act, var, seed):
    return os.path.join(config.INTERP_DIR, enc, act, var, f"seed{seed}", "hypotheses.json")


def _faithful_glob(enc, act, var):
    return os.path.join(config.EVAL_DIR, enc, act, var, "seed*", "faithful.csv")


# --------------------------------------------------------------------------- auto-pick best cell / classes
def _cell_nsig(enc, act, var):
    """Seed-averaged macro #Significant for a cell, or None if no faithful results."""
    vals = []
    for fp in glob.glob(_faithful_glob(enc, act, var)):
        df = pd.read_csv(fp)
        if not df.empty:
            vals.append(df["n_significant"].mean())
    return float(np.mean(vals)) if vals else None


def autopick_cell():
    """(enc, act) with the largest Dual−baseline #Significant advantage among COMPLETE cells."""
    best, best_delta = None, -1e9
    for enc in config.ENCODERS:
        for act in [p.name for p in config.ACTIVATION_PAIRS]:
            d, b = _cell_nsig(enc, act, "dual"), _cell_nsig(enc, act, "baseline")
            if d is None or b is None:
                continue
            if d - b > best_delta:
                best_delta, best = d - b, (enc, act)
    if best is None:
        raise SystemExit("autopick: no complete cells with faithful.csv found for this dataset.")
    print(f"[viz] auto-picked best-#Sig cell: {best[0]}/{best[1]} (Δ#Sig={best_delta:+.2f})")
    return best


def autopick_classes(enc, act, seed, n_classes, exclude=()):
    """Faithful classes the Dual cell explains best (highest per-class #Significant), seed-averaged.
    n_classes <= 0 -> ALL faithful classes (still #Sig-ranked); `exclude` drops already-shown classes."""
    fcls = set(config.faithful_classes())
    rows = []
    for fp in glob.glob(_faithful_glob(enc, act, "dual")):
        df = pd.read_csv(fp)
        rows.append(df[["class", "n_significant", "auroc"]])
    if not rows:
        raise SystemExit("autopick_classes: no dual faithful.csv.")
    allc = pd.concat(rows, ignore_index=True)
    allc = allc[allc["class"].isin(fcls)]
    agg = allc.groupby("class")[["n_significant", "auroc"]].mean().sort_values(
        ["n_significant", "auroc"], ascending=False)
    ranked = [int(c) for c in agg.index if int(c) not in set(exclude)]
    chosen = ranked if n_classes <= 0 else ranked[:n_classes]
    print(f"[viz] auto-picked classes (top #Sig): {len(chosen)}"
          + (f" {chosen}" if len(chosen) <= 12 else " classes"))
    return chosen


# --------------------------------------------------------------------------- data assembly
def _load_cell(enc, act, var, seed, method, k):
    """Return (class_to_latents, class_to_scores, latent->hypothesis, latent->global top_image_rel)."""
    with open(_select_path(enc, act, var, seed, method, k)) as f:
        sel = json.load(f)
    with open(_interp_path(enc, act, var, seed)) as f:
        interp = json.load(f)
    hyp = {int(kk): v.get("hypothesis", "") for kk, v in interp.items()}
    # global top-activating images (TRAIN split; the exact evidence the VLM saw when writing the hypothesis)
    top_rel = {int(kk): v.get("top_image_rel", []) for kk, v in interp.items()}
    return sel["class_to_latents"], sel["class_to_scores"], hyp, top_rel


def _global_examples(classes, latents_per_class, top_rel, n_images):
    """Global mode: each latent's stored top-activating images (any class, TRAIN split) — zero compute.
    Same {(class, latent): [rel_path,...]} shape as _example_images so the rest of the pipeline is shared."""
    out = {}
    for c in classes:
        for lat in latents_per_class.get(str(c), []):
            out[(c, int(lat))] = top_rel.get(int(lat), [])[:n_images]
    return out


def _example_images(enc, act, var, seed, classes, latents_per_class, n_images, manifest, labels):
    """For each (class, latent) pick the n_images class-A val images with the highest activation on that
    latent. Returns {(class, latent): [rel_path,...]}. Reads only the needed latent columns of val_acts."""
    needed = sorted({lat for c in classes for lat in latents_per_class.get(str(c), [])})
    out = {}
    if not needed:
        return out
    acts = np.load(config.acts_path(config.Cell(enc, act, var, seed), "val"), mmap_mode="r")
    sub = np.asarray(acts[:, needed], dtype=np.float32)        # one strided read -> (Nval, len(needed))
    col_of = {lat: i for i, lat in enumerate(needed)}
    for c in classes:
        rows = np.where(config.onevsrest(labels, c))[0]        # this class's val image rows
        if rows.size == 0:
            continue
        for lat in latents_per_class.get(str(c), []):
            colv = sub[rows, col_of[lat]]
            order = rows[np.argsort(-colv)][:n_images]         # class-A images, strongest activation first
            out[(c, lat)] = [manifest[i]["rel_path"] for i in order]
    return out


def _load_tile(rel_path, root, size):
    """PIL square RGB tile; gray placeholder on failure (mirrors visualize_latent_activation_grid)."""
    try:
        img = Image.open(os.path.join(root, rel_path)).convert("RGB")
        return ImageOps.fit(img, (size, size), Image.BICUBIC)
    except Exception:
        return Image.new("RGB", (size, size), (200, 200, 200))


# --------------------------------------------------------------------------- figure rendering
def _draw_rows(axes_rows, rows, img_root, n_images, tile=224, wrap=16):
    """Render latent rows into a (len(rows) x (1+n_images)) axes grid. Each row in `rows` is
    (latent, score, hypothesis, [rel_paths]). Col 0 is a SLIM label column holding the hypothesis
    rotated vertically (latent id / coefficient intentionally not shown); cols 1.. are image tiles."""
    for r, (lat, score, hyp, rels) in enumerate(rows):
        tax = axes_rows[r][0]
        tax.axis("off")
        tax.set_xlim(0, 1); tax.set_ylim(0, 1)
        txt = textwrap.fill(hyp or "(uninterpreted)", width=wrap)
        # vertical hypothesis label spanning the row height; counter-evidence rows (negative coef)
        # are dark red so the pos/neg split stays visible without printing the coefficient.
        tax.text(0.5, 0.5, txt, rotation=90, va="center", ha="center", fontsize=9.5, weight="bold",
                 color="black" if score > 0 else "#b02020")
        for j in range(n_images):
            ax = axes_rows[r][1 + j]
            ax.axis("off")
            if j < len(rels):
                # aspect="auto" fills the cell exactly (tiles are pre-squared, axes ~square, so any
                # distortion is sub-percent) -> no seams between images in a row
                ax.imshow(_load_tile(rels[j], img_root, tile), aspect="auto")


def _figure(rows, title, out_base, n_images):
    """Concept-centric figure: len(rows) x (1 slim vertical-label col + n_images image cols)."""
    nr = max(len(rows), 1)
    h = 1.55 * nr + 0.5
    fig, axes = plt.subplots(nr, 1 + n_images, figsize=(1.6 + 1.7 * n_images, h), squeeze=False,
                             gridspec_kw={"width_ratios": [0.9] + [1] * n_images})
    # seamless grid: images in a row touch (wspace~0), slim row gaps; fixed ~0.5in title band on top
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=1 - 0.5 / h,
                        wspace=0.015, hspace=0.06)
    _draw_rows(axes, rows, *_FIG_CTX["img_root_n"])
    fig.suptitle(title, fontsize=14, weight="bold", y=0.995)
    _save(fig, out_base)


def _figure_contrast(base_rows, dual_rows, title, out_base, n_images,
                     base_label="Baseline SAE", dual_label="DualSAE"):
    """Contrast figure: baseline block (top) vs DualSAE block (bottom), same class. Section labels
    are activation-specific (e.g. 'TopK SAE' vs 'Dual-TopK SAE') via base_label/dual_label."""
    img_root, _ = _FIG_CTX["img_root_n"]
    nb, nd = max(len(base_rows), 1), max(len(dual_rows), 1)
    fig = plt.figure(figsize=(1.6 + 1.7 * n_images, 1.55 * (nb + nd) + 1.0))
    # dedicated title band on top, then baseline + dual blocks — so the main title never collides with the
    # per-block section labels.
    tband, mid, bot = fig.subfigures(3, 1, height_ratios=[0.4, nb, nd], hspace=0.02)
    tband.text(0.5, 0.5, title, ha="center", va="center", fontsize=14, weight="bold")
    # center the section labels over the IMAGE strip (3rd of 5 tiles), not the whole canvas — the slim
    # vertical-label column would otherwise pull them visually off-balance to the left.
    x_img = (0.9 + n_images / 2) / (0.9 + n_images)
    for sub, rows, lbl, color in ((mid, base_rows, base_label, "#b00020"),
                                  (bot, dual_rows, dual_label, "#0050b0")):
        sub.suptitle(lbl, x=x_img, fontsize=13, weight="bold", color=color)
        nrows = max(len(rows), 1)
        axes = sub.subplots(nrows, 1 + n_images, squeeze=False,
                            gridspec_kw={"width_ratios": [0.9] + [1] * n_images})
        # seamless grid inside each block; small fixed top margin keeps the section label clear of row 1
        sub.subplots_adjust(left=0.005, right=0.995, bottom=0.015, top=1 - 0.42 / (1.55 * nrows),
                            wspace=0.015, hspace=0.06)
        _draw_rows(axes, rows, img_root, n_images)
    _save(fig, out_base)


def _save(fig, out_base):
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fmts = _FIG_CTX.get("formats", ("png", "pdf"))
    for ext in fmts:
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out_base}.{{{','.join(fmts)}}}")


_FIG_CTX = {}  # set in main: {"img_root_n": (img_root, n_images)}


def _rows_for(class_id, cls2lat, cls2sc, hyp, examples, n_latents, min_coef=0.5):
    """Assemble the per-latent rows for one class+variant: the top-n_latents POSITIVE-coef latents
    (evidence FOR the class) followed by the top-n_latents NEGATIVE-coef latents (counter-evidence),
    keeping only |coef| > min_coef. The selection list is already |coef|-sorted, so the first qualifying
    entries per sign are the strongest; if fewer than n_latents qualify, the row is simply omitted
    (classes can legitimately show fewer rows)."""
    pairs = list(zip(cls2lat.get(str(class_id), []), cls2sc.get(str(class_id), [])))
    pos = [(l, s) for l, s in pairs if s > min_coef][:n_latents]
    neg = [(l, s) for l, s in pairs if s < -min_coef][:n_latents]
    rows = []
    for lat, sc in pos + neg:
        rows.append((lat, float(sc), hyp.get(int(lat), ""), examples.get((class_id, int(lat)), [])))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Qualitative DualSAE hypothesis figures.")
    ap.add_argument("--encoder", default=None, choices=config.ENCODERS)
    ap.add_argument("--activation", default=None, choices=[p.name for p in config.ACTIVATION_PAIRS])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", default=config.PRIMARY_SELECTION_METHOD)
    ap.add_argument("--k", type=int, default=config.PRIMARY_K)
    ap.add_argument("--classes", type=int, nargs="*", default=None, help="class ids; default auto-pick")
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--n-latents", type=int, default=3,
                    help="max latents PER SIGN shown: top-N positive-coef + top-N negative-coef rows")
    ap.add_argument("--min-coef", type=float, default=0.5,
                    help="only show latents with |coef| above this; classes with fewer qualifying "
                         "latents simply show fewer rows")
    ap.add_argument("--n-images", type=int, default=5)
    ap.add_argument("--images", default="global", choices=["class", "global"],
                    help="example tiles: 'class' = class-A val images ranked by activation on the latent; "
                         "'global' (default) = the latent's stored top-activating images from ANY class "
                         "(train split, the exact evidence the VLM saw) — shows the latent's true meaning, "
                         "so polysemantic latents become visible")
    ap.add_argument("--figures", default="both", choices=["both", "contrast", "concept"],
                    help="which layouts to render")
    ap.add_argument("--formats", default="png,pdf",
                    help="comma-separated savefig formats (e.g. 'pdf' for slide-only galleries)")
    ap.add_argument("--exclude-classes", type=int, nargs="*", default=[],
                    help="class ids auto-pick must skip (e.g. already-shown showcase classes)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    enc, act = (a.encoder, a.activation) if (a.encoder and a.activation) else autopick_cell()
    # global top_image_rel paths come from the TRAIN split (interpret_latents samples train acts), so the
    # image root must match the mode; class mode ranks val images.
    _FIG_CTX["img_root_n"] = (config.IMAGE_ROOTS["train" if a.images == "global" else "val"], a.n_images)
    _FIG_CTX["formats"] = tuple(x.strip() for x in a.formats.split(",") if x.strip())
    # separate default out dir per mode so global figures never clobber the class-restricted ones
    suffix = "_global" if a.images == "global" else ""
    out_dir = a.out or os.path.join(config.OUT_ROOT, "qualitative", f"{enc}_{act}{suffix}")
    os.makedirs(out_dir, exist_ok=True)

    classes = a.classes or autopick_classes(enc, act, a.seed, a.n_classes, exclude=a.exclude_classes)

    # shared val manifest (SAE-independent); idx->wnid for ImageNet readable names. Labels/acts are only
    # needed in class mode (global mode reuses the stored top_image_rel — zero compute).
    with open(config.manifest_path("val")) as f:
        manifest = json.load(f)
    idx2wnid = cn.build_idx_to_wnid(manifest) if config.DATASET == "imagenet" else None

    # selections + hypotheses for both variants (dual = our method; baseline = contrast), needed everywhere
    cells = {var: _load_cell(enc, act, var, a.seed, a.method, a.k) for var in ("baseline", "dual")}

    labels = config.load_labels("val") if a.images == "class" else None
    variants = {}
    for var, (c2l, c2s, hyp, top_rel) in cells.items():
        if a.images == "global":
            ex = _global_examples(classes, c2l, top_rel, a.n_images)
        else:
            ex = _example_images(enc, act, var, a.seed, classes, c2l, a.n_images, manifest, labels)
        variants[var] = (c2l, c2s, hyp, ex)

    def _done(base):  # per-figure skip-guard -> re-runs after an interruption are cheap
        return all(os.path.exists(f"{base}.{e}") for e in _FIG_CTX["formats"])

    index = [f"# Qualitative DualSAE figures — {config.DATASET} {enc}/{act} (seed {a.seed})\n",
             f"Method={a.method} K={a.k}; top-{a.n_latents} pos + top-{a.n_latents} neg latents "
             f"(|coef|>{a.min_coef}), n_images={a.n_images}, images={a.images}, figures={a.figures}"
             + ".\n"]
    # activation-specific model names for titles/section labels (paper naming: TopK SAE vs Dual-TopK SAE)
    act_nm = {"topk": "TopK", "batch_topk": "BatchTopK", "jump_relu": "JumpReLU"}.get(act, act)
    for c in classes:
        name = cn.readable(config.DATASET, c, idx2wnid)
        safe = "".join(ch if ch.isalnum() else "_" for ch in name)[:40]
        c2l, c2s, hyp, ex = variants["dual"]
        drows = _rows_for(c, c2l, c2s, hyp, ex, a.n_latents, a.min_coef)
        parts = []
        # concept-centric (DualSAE)
        if a.figures in ("both", "concept"):
            base = os.path.join(out_dir, f"concept_{c}_{safe}")
            if not _done(base):
                _figure(drows, f"{name}  —  why Dual-{act_nm} SAE predicts it   ({enc})",
                        base, a.n_images)
            parts.append(f"concept_{c}_{safe}")
        # contrast (baseline vs dual)
        if a.figures in ("both", "contrast"):
            base = os.path.join(out_dir, f"contrast_{c}_{safe}")
            if not _done(base):
                bc2l, bc2s, bhyp, bex = variants["baseline"]
                brows = _rows_for(c, bc2l, bc2s, bhyp, bex, a.n_latents, a.min_coef)
                _figure_contrast(brows, drows,
                                 f"{name}  —  {act_nm} SAE vs Dual-{act_nm} SAE   ({enc})",
                                 base, a.n_images,
                                 base_label=act_nm, dual_label=f"Dual-{act_nm}")
            parts.append(f"contrast_{c}_{safe}")
        index.append(f"- **{name}** (class {c}): " + " · ".join(parts))

    with open(os.path.join(out_dir, "index.md"), "w") as f:
        f.write("\n".join(index) + "\n")
    print(f"[viz] done -> {out_dir} ({len(classes)} classes)")


if __name__ == "__main__":
    main()
