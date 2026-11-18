"""Per-encoder composite qualitative figures (ImageNet): one figure per encoder, stacking the three
activation pairs (TopK, BatchTopK, JumpReLU) — for each, the baseline block vs the Dual block, top-2
POSITIVE-coef latents each (|coef| > 0.5 gate), global top-activator tiles, vertical hypothesis labels,
section headers without the "SAE" suffix and centered over the image strip.

Selection rules used to build the figure (as implemented below):
  * selection rule per (encoder, activation) cell (--selectors): 'auto' (default) = the rule among
    {lasso, correlation, separation_score} with the largest seed-averaged Dual-minus-baseline
    #Significant on the faithful evaluation (ties -> larger Delta AUROC); or an explicit
    'enc/act=rule' list. Paths are resolved directly (main tree for lasso, selmethod_<rule>/ tree for
    correlation / separation_score), so no HG_MODE env juggling is needed.
  * class per section: among the faithful classes, the class with the largest seed-averaged per-class
    #Significant advantage (Dual - baseline) that has >= 2 qualifying positive latents in BOTH variants
    (falls back to >= 1), unless overridden with --override enc/act=class_id.
  * rows: the top-2 positive-coefficient selected latents per variant (|coef| > 0.5 gate for lasso,
    > 0 for the other rules); tiles = each latent's global top-activating training images.

CPU only (~200 image tiles per figure). Output: <HG_OUT>/figures/encoder_<enc>.pdf.
Run: python -m hypogen.encoder_figs [--out DIR] [--n-latents 2] [--override vit/jump_relu=243]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hypogen import config                       # imported in default (imagenet, main) mode
from hypogen import class_names as cn
from hypogen.visualize_qualitative import _draw_rows

ROOT = config.OUT_ROOT                 # imagenet output tree (this figure is ImageNet-only)
ACTS = ["topk", "batch_topk", "jump_relu"]
ACT_NM = {"topk": "TopK", "batch_topk": "BatchTopK", "jump_relu": "JumpReLU"}
RULES = ["lasso", "correlation", "separation_score"]


def _macro_seed_mean(sel, enc, act, var):
    """(mean over seeds of macro #Significant, same for AUROC) for one variant under one rule, or None."""
    fps = glob.glob(os.path.join(_tree(sel), "eval", enc, act, var, "seed*", "faithful.csv"))
    if not fps:
        return None
    per = [pd.read_csv(fp)[["n_significant", "auroc"]].mean() for fp in fps]
    return float(np.mean([p["n_significant"] for p in per])), float(np.mean([p["auroc"] for p in per]))


def auto_selector(enc, act):
    """Rule with the largest Dual-minus-baseline #Significant (tie -> larger Delta AUROC)."""
    best, key = None, None
    for sel in RULES:
        d, b = _macro_seed_mean(sel, enc, act, "dual"), _macro_seed_mean(sel, enc, act, "baseline")
        if d is None or b is None:
            continue
        k = (round(d[0] - b[0], 10), round(d[1] - b[1], 10))
        if key is None or k > key:
            best, key = sel, k
    if best is None:
        raise SystemExit(f"no faithful results for {enc}/{act} under any rule")
    return best


def _tree(sel):
    # lasso lives in the main output tree; the non-lasso rules in their selmethod_<rule> subtree
    return ROOT if sel == "lasso" else os.path.join(ROOT, f"selmethod_{sel}")


def _load_variant(enc, act, var, sel, seed):
    """(class_to_latents, class_to_scores, latent->hypothesis, latent->top_image_rel) for one cell."""
    base = _tree(sel)
    with open(os.path.join(base, "selected", enc, act, var, f"seed{seed}", f"{sel}_K16.json")) as f:
        s = json.load(f)
    with open(os.path.join(base, "interpreted", enc, act, var, f"seed{seed}", "hypotheses.json")) as f:
        interp = json.load(f)
    hyp = {int(k): v.get("hypothesis", "") for k, v in interp.items()}
    rel = {int(k): v.get("top_image_rel", []) for k, v in interp.items()}
    return s["class_to_latents"], s["class_to_scores"], hyp, rel


def _delta_nsig(enc, act, sel):
    """Per-class seed-averaged #Significant delta (dual - baseline) under this selection rule."""
    per = {}
    for var in ("baseline", "dual"):
        fps = glob.glob(os.path.join(_tree(sel), "eval", enc, act, var, "seed*", "faithful.csv"))
        df = pd.concat([pd.read_csv(fp) for fp in fps])
        per[var] = df.groupby("class")["n_significant"].mean()
    return (per["dual"] - per["baseline"]).dropna()


def _qualifying(c2l, c2s, cls, n, min_coef):
    """Top-n positive-coef latents above the gate for one class (list of (latent, coef))."""
    pairs = zip(c2l.get(str(cls), []), c2s.get(str(cls), []))
    return [(l, s) for l, s in pairs if s > min_coef][:n]


def _pick_class(enc, act, sel, cells, n, min_coef):
    """Highest-delta faithful class where both variants can fill n qualifying positive rows."""
    delta = _delta_nsig(enc, act, sel).sort_values(ascending=False)
    fcls = set(config.faithful_classes())
    ranked = [int(c) for c in delta.index if int(c) in fcls]
    for need in (n, 1):                                        # relax to 1 row only if nothing fills n
        for c in ranked:
            if all(len(_qualifying(cells[v][0], cells[v][1], c, n, min_coef)) >= need
                   for v in ("baseline", "dual")):
                return c
    return ranked[0]


def main():
    ap = argparse.ArgumentParser(description="Per-encoder 3-activation composite figures (ImageNet).")
    ap.add_argument("--out", default=os.path.join(config.OUT_ROOT, "figures"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-latents", type=int, default=2, help="positive rows per block")
    ap.add_argument("--min-coef", type=float, default=0.5)
    ap.add_argument("--n-images", type=int, default=5)
    # manual class override per section, e.g. --override clip/jump_relu=4 vit/jump_relu=243 — used when
    # the auto-picked (highest-delta) class has visually off-topic Dual evidence tiles: the ranking only
    # sees #Significant deltas, not tile content, so a human pass may swap in a lower-delta class whose
    # hypotheses/tiles are on-topic for both variants.
    ap.add_argument("--override", nargs="*", default=[], metavar="ENC/ACT=CLS")
    ap.add_argument("--selectors", nargs="*", default=["auto"], metavar="ENC/ACT=RULE",
                    help="'auto' (rule with the largest Dual-baseline #Significant per cell) or explicit "
                         "enc/act=rule entries (unlisted cells fall back to auto)")
    a = ap.parse_args()
    selectors = {}
    for sv in a.selectors:
        if sv == "auto":
            continue
        key, _, rule = sv.partition("=")
        enc_, _, act_ = key.partition("/")
        if rule not in RULES:
            ap.error(f"unknown selection rule {rule!r} in --selectors {sv}")
        selectors[(enc_, act_)] = rule
    overrides = {}
    for ov in a.override:
        key, _, cls = ov.partition("=")
        enc_, _, act_ = key.partition("/")
        overrides[(enc_, act_)] = int(cls)
    os.makedirs(a.out, exist_ok=True)

    with open(config.manifest_path("val")) as f:
        idx2wnid = cn.build_idx_to_wnid(json.load(f))
    img_root = config.IMAGE_ROOTS["train"]                    # global top_image_rel are train-split paths
    x_img = (0.9 + a.n_images / 2) / (0.9 + a.n_images)       # center of the image strip (3rd of 5 tiles)

    for enc in config.ENCODERS:
        # one subfigure triple per activation: [class band, baseline block, dual block]
        n = a.n_latents
        ratios = [0.32, n + 0.42, n + 0.42] * len(ACTS)
        fig = plt.figure(figsize=(1.6 + 1.7 * a.n_images, 1.45 * sum(ratios)))
        subs = fig.subfigures(len(ratios), 1, height_ratios=ratios, hspace=0.03)

        for ai, act in enumerate(ACTS):
            sel = selectors.get((enc, act)) or auto_selector(enc, act)
            # the |coef|>min_coef gate is calibrated to LASSO logistic-coefficient scale; correlation /
            # separation scores live on different (smaller) scales, so for them "positive" is the gate —
            # their selection lists are already strength-sorted, top-2 positive = strongest evidence.
            gate = a.min_coef if sel == "lasso" else 0.0
            cells = {v: _load_variant(enc, act, v, sel, a.seed) for v in ("baseline", "dual")}
            cls = overrides.get((enc, act)) or _pick_class(enc, act, sel, cells, n, gate)
            name = cn.readable("imagenet", cls, idx2wnid)
            band, bsub, dsub = subs[3 * ai], subs[3 * ai + 1], subs[3 * ai + 2]
            band.text(x_img, 0.35, name, ha="center", va="center", fontsize=12,
                      style="italic", color="dimgray")
            for sub, var, lbl, color in ((bsub, "baseline", ACT_NM[act], "#b00020"),
                                         (dsub, "dual", f"Dual-{ACT_NM[act]}", "#0050b0")):
                c2l, c2s, hyp, rel = cells[var]
                rows = [(l, float(s), hyp.get(int(l), ""), rel.get(int(l), [])[:a.n_images])
                        for l, s in _qualifying(c2l, c2s, cls, n, gate)]
                nrows = max(len(rows), 1)
                sub.suptitle(lbl, x=x_img, fontsize=13, weight="bold", color=color)
                axes = sub.subplots(nrows, 1 + a.n_images, squeeze=False,
                                    gridspec_kw={"width_ratios": [0.9] + [1] * a.n_images})
                sub.subplots_adjust(left=0.005, right=0.995, bottom=0.015,
                                    top=1 - 0.42 / (1.45 * nrows), wspace=0.015, hspace=0.06)
                for ax in axes.flat:                 # blank any placeholder axes (defensive: 0 rows)
                    ax.axis("off")
                _draw_rows(axes, rows, img_root, a.n_images)
            print(f"[encfig] {enc}/{act} (sel={sel}) -> class {cls} '{name}'")

        out = os.path.join(a.out, f"encoder_{enc}.pdf")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[encfig] wrote {out}")


if __name__ == "__main__":
    main()
