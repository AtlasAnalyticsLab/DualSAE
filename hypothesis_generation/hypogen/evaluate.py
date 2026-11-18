"""Stage 4 — evaluate selected/interpreted latents two ways (Dual vs baseline, by activation pair).

proxy  (CPU, no VLM): for each (method, K, class) fit a logistic on the K selected latents using TRAIN
       activations and score held-out VAL activations -> AUROC/AUPRC. Measures feature-selection quality;
       runs for ALL methods x K x 1000 classes (cheap).

faithful (GPU, VLM): the paper's metric, on the PRIMARY config (lasso, K=PRIMARY_K) and a bounded set of
       FAITHFUL_CLASSES classes. Per class: its hypotheses (Stage-3 interpretations of its selected
       latents) are annotated by the VLM on held-out VAL images (class positives + a shared negative
       pool); HypotheSAEs.score_hypotheses regresses class membership on the annotation matrix ->
       AUROC/AUPRC/R2 + #significant (Bonferroni). Annotations are cached globally by (hypothesis,image)
       so reuse across classes is free, and the cache makes the (expensive) job resumable.

Run proxy   (CPU): python -m hypogen.evaluate --stage proxy    --encoder clip --activation topk --variant dual --seed 42
Run faithful (GPU): python -m hypogen.evaluate --stage faithful --encoder clip --activation topk --variant dual --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from hypogen import config


def _eval_dir(cell: config.Cell) -> str:
    return os.path.join(config.EVAL_DIR, cell.encoder, cell.activation, cell.variant, f"seed{cell.seed}")


def _atomic_write(obj, path: str, is_json: bool = True) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    if is_json:
        with open(tmp, "w") as f:
            json.dump(obj, f)
    else:
        obj.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _load_selection(cell: config.Cell, method: str, k: int) -> Dict[str, List[int]]:
    p = os.path.join(config.SELECT_DIR, cell.encoder, cell.activation, cell.variant,
                     f"seed{cell.seed}", f"{method}_K{k}.json")
    with open(p) as f:
        return json.load(f)["class_to_latents"]


# --------------------------------------------------------------------------------------------- proxy
def run_proxy(cell: config.Cell) -> None:
    out_csv = os.path.join(_eval_dir(cell), "proxy.csv")
    if os.path.isfile(out_csv):
        print(f"[{cell.tag}] proxy already done: {out_csv}")
        return
    ytr = config.load_labels("train")   # (N,) int OR (N, C) multi-hot (OpenImages)
    yva = config.load_labels("val")
    Atr = np.asarray(np.load(config.acts_path(cell, "train"), mmap_mode="r"), dtype=np.float32)
    Ava = np.asarray(np.load(config.acts_path(cell, "val"), mmap_mode="r"), dtype=np.float32)
    print(f"[{cell.tag}] proxy: train {Atr.shape}, val {Ava.shape}")

    rows = []
    for method in config.SELECTION_METHODS:
        for k in config.K_GRID:
            sel = _load_selection(cell, method, k)
            for c in range(config.NUM_CLASSES):
                cols = sel.get(str(c), [])
                if not cols:
                    continue
                ytr_c = config.onevsrest(ytr, c).astype(int)   # one-vs-rest target (single- or multi-label)
                yva_c = config.onevsrest(yva, c).astype(int)
                if yva_c.sum() == 0 or ytr_c.sum() == 0:
                    continue
                Xtr, Xva = Atr[:, cols], Ava[:, cols]
                clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                clf.fit(Xtr, ytr_c)
                p = clf.predict_proba(Xva)[:, 1]
                rows.append({"encoder": cell.encoder, "activation": cell.activation,
                             "variant": cell.variant, "seed": cell.seed, "method": method, "K": k,
                             "class": c, "auroc": roc_auc_score(yva_c, p),
                             "auprc": average_precision_score(yva_c, p),
                             "n_pos_val": int(yva_c.sum()), "n_pos_train": int(ytr_c.sum())})
    _atomic_write(pd.DataFrame(rows), out_csv, is_json=False)
    print(f"[{cell.tag}] proxy: wrote {len(rows)} rows -> {out_csv}")


# ------------------------------------------------------------------------------------------ faithful
def _choose_faithful_classes(yva: np.ndarray) -> List[int]:
    present = config.present_classes(yva)   # multi-label aware (column-sum>0) or np.unique (single-label)
    rng = np.random.default_rng(config.FAITHFUL_CLASS_SEED)
    n = min(config.FAITHFUL_CLASSES, len(present))
    if n < len(present):
        return sorted(rng.choice(present, size=n, replace=False).tolist())
    return present


def run_faithful(cell: config.Cell, method: str = None) -> None:
    method = method or config.PRIMARY_SELECTION_METHOD
    k = config.PRIMARY_K
    # A non-primary --method writes method-TAGGED outputs (faithful_<method>.csv
    # + faithful_annotations_<method>.json) so they never clobber the primary lasso faithful.csv / cache.
    suffix = "" if method == config.PRIMARY_SELECTION_METHOD else f"_{method}"
    out_csv = os.path.join(_eval_dir(cell), f"faithful{suffix}.csv")
    cache_path = os.path.join(_eval_dir(cell), f"faithful_annotations{suffix}.json")
    if os.path.isfile(out_csv):
        print(f"[{cell.tag}] faithful ({method}) already done: {out_csv}")
        return

    # Inputs: selection (which latents per class), interpretations (hypothesis per latent), val labels+images.
    sel = _load_selection(cell, method, k)
    hyp_path = os.path.join(config.INTERP_DIR, cell.encoder, cell.activation, cell.variant,
                            f"seed{cell.seed}", "hypotheses.json")
    with open(hyp_path) as f:
        latent_hyp = {int(kk): v.get("hypothesis") for kk, v in json.load(f).items()}
    yva = config.load_labels("val")   # (N,) int OR (N, C) multi-hot (OpenImages)
    with open(config.manifest_path("val")) as f:
        val_rel = [m["rel_path"] for m in json.load(f)]
    val_root = config.IMAGE_ROOTS["val"]

    classes = _choose_faithful_classes(yva)
    print(f"[{cell.tag}] faithful: {len(classes)} classes (of {config.NUM_CLASSES}); primary={method} K={k}")

    # Shared negative pool (same val images for every class -> maximal annotation cache reuse).
    rng = np.random.default_rng(config.FAITHFUL_CLASS_SEED + 1)
    neg_pool = rng.choice(len(val_rel), size=min(config.FAITHFUL_NEG_PER_CLASS, len(val_rel)), replace=False)

    # Per class: hypotheses (unique, non-empty) + held-out image indices (positives + neg pool, minus true c).
    per_class = {}
    uniq_hyps: List[str] = []
    hyp_id = {}
    needed = set()  # (hyp_id, val_idx)
    for c in classes:
        lats = [int(x) for x in sel.get(str(c), [])]
        hyps = []
        for lat in lats:
            h = latent_hyp.get(lat)
            if h and h not in hyp_id:
                hyp_id[h] = len(uniq_hyps); uniq_hyps.append(h)
            if h:
                hyps.append(h)
        hyps = list(dict.fromkeys(hyps))  # dedup, keep order
        pos = np.where(config.onevsrest(yva, c))[0]                   # images positive for class c
        # Cap positives (multi-label common classes have 1000s -> bounds annotation + balances vs the ~200
        # negatives). Deterministic per-class subsample so reruns reuse the annotation cache. No-op for
        # ImageNet (pos < cap).
        if len(pos) > config.FAITHFUL_POS_PER_CLASS:
            pos = np.sort(np.random.default_rng(config.FAITHFUL_CLASS_SEED + 2 + c).choice(
                pos, size=config.FAITHFUL_POS_PER_CLASS, replace=False))
        negs = neg_pool[~config.onevsrest(yva[neg_pool], c)]          # pool images NOT positive for c
        held = np.concatenate([pos, negs])
        per_class[c] = {"hyps": hyps, "held": held}
        for h in hyps:
            for i in held:
                needed.add((hyp_id[h], int(i)))

    # Annotation cache (resumable).
    cache = {}
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            blob = json.load(f)
        if blob.get("hyps") == uniq_hyps:        # only reuse if the hypothesis list is unchanged
            cache = blob.get("ann", {})
    todo = [(hid, vi) for (hid, vi) in needed if f"{hid}|{vi}" not in cache]
    print(f"[{cell.tag}] faithful: {len(needed)} (hyp,img) pairs; {len(todo)} to annotate ({len(cache)} cached)")

    if todo:
        from PIL import Image
        from hypogen.vlm_backend import VLM
        vlm = VLM(seed=cell.seed)
        BATCH = 4096
        for s in range(0, len(todo), BATCH):
            batch = todo[s:s + BATCH]
            items = [(Image.open(os.path.join(val_root, val_rel[vi])).convert("RGB"), uniq_hyps[hid])
                     for (hid, vi) in batch]
            res = vlm.annotate(items)
            for (hid, vi), a in zip(batch, res):
                cache[f"{hid}|{vi}"] = int(bool(a))
            _atomic_write({"hyps": uniq_hyps, "ann": cache}, cache_path)
            print(f"[{cell.tag}] faithful: annotated {min(s + BATCH, len(todo))}/{len(todo)}", flush=True)

    # Score each class with HypotheSAEs.score_hypotheses (OLS/Logit + Bonferroni significance).
    from hypothesaes.evaluation import score_hypotheses
    rows = []
    for c in classes:
        pc = per_class[c]
        held = pc["held"]
        y_true = config.onevsrest(yva[held], c).astype(int)   # single- or multi-label one-vs-rest
        if not pc["hyps"] or y_true.sum() == 0 or (y_true == 0).sum() == 0:
            continue
        hyp_ann = {}
        for h in pc["hyps"]:
            hyp_ann[h] = np.array([cache[f"{hyp_id[h]}|{int(i)}"] for i in held], dtype=int)
        try:
            metrics, _ = score_hypotheses(hyp_ann, y_true, classification=True,
                                          corrected_pval_threshold=config.CORRECTED_PVAL_THRESHOLD)
        except Exception as e:
            print(f"[{cell.tag}] class {c}: score_hypotheses failed: {e}")
            continue
        nsig, ntot, _ = metrics.get("Significant", (0, len(hyp_ann), 0.0))
        rows.append({"encoder": cell.encoder, "activation": cell.activation, "variant": cell.variant,
                     "seed": cell.seed, "method": method, "K": k, "class": c,
                     "n_hypotheses": len(hyp_ann), "auroc": metrics.get("auroc", np.nan),
                     "auprc": metrics.get("auprc", np.nan), "r2": metrics.get("r2", np.nan),
                     "n_significant": nsig, "n_pos": int(y_true.sum()), "n_held": int(len(held))})
    _atomic_write(pd.DataFrame(rows), out_csv, is_json=False)
    print(f"[{cell.tag}] faithful: wrote {len(rows)} class rows -> {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Stage 4: proxy + faithful evaluation.")
    ap.add_argument("--stage", required=True, choices=["proxy", "faithful"])
    ap.add_argument("--encoder", required=True, choices=config.ENCODERS)
    ap.add_argument("--activation", required=True, choices=[p.name for p in config.ACTIVATION_PAIRS])
    ap.add_argument("--variant", required=True, choices=config.ALLOWED_VARIANTS)
    ap.add_argument("--seed", required=True, type=int, choices=config.SEEDS)
    ap.add_argument("--method", default=None, choices=config.SELECTION_METHODS,
                    help="selection method for faithful (default: PRIMARY_SELECTION_METHOD=lasso). "
                         "Non-primary methods write method-tagged outputs. The release scores the "
                         "other rules with HG_MODE=selmethod instead (outputs under selmethod_<rule>/).")
    args = ap.parse_args()
    cell = config.Cell(args.encoder, args.activation, args.variant, args.seed)
    if args.stage == "proxy":
        run_proxy(cell)
    else:
        run_faithful(cell, method=args.method)


if __name__ == "__main__":
    main()
