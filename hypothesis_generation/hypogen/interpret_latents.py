"""Stage 3 — interpret each selected latent with the VLM, from its top/zero-activating images.

For each unique selected latent of a cell (Stage 2's unique_latents.json), gather the N_HIGH most
strongly-activating TRAIN images and N_LOW zero-activating TRAIN images (the image analog of
HypotheSAEs.sample_top_zero), show both sets to the VLM proposer, and record the one-line visual
hypothesis. Optionally (--with-fidelity) score the hypothesis on held-out VAL high/zero images:
the VLM annotates each, and we compute precision/recall/F1/correlation vs the binarized activation
(matches HypotheSAEs.NeuronInterpreter._compute_metrics).

Incremental + resumable: hypotheses.json is merged and flushed every config.INTERP_FLUSH_EVERY latents,
so a re-run after a timeout continues from where it stopped.

Run (GPU): python -m hypogen.interpret_latents --encoder clip --activation topk --variant dual --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

from hypogen import config


def _interp_dir(cell: config.Cell) -> str:
    return os.path.join(config.INTERP_DIR, cell.encoder, cell.activation, cell.variant, f"seed{cell.seed}")


def _hyp_path(cell: config.Cell) -> str:
    return os.path.join(_interp_dir(cell), "hypotheses.json")


def _atomic_write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def _load_manifest(split: str) -> List[str]:
    with open(config.manifest_path(split)) as f:
        return [m["rel_path"] for m in json.load(f)]


def sample_high_zero(col: np.ndarray, n_high: int, n_low: int, rng) -> Tuple[np.ndarray, np.ndarray]:
    """Indices of the n_high strongest (>0) activations and n_low zero-activation rows (HypotheSAEs-style)."""
    pos = np.where(col > 0)[0]
    if len(pos) <= n_high:
        high = pos[np.argsort(-col[pos])]
    else:
        top = np.argpartition(-col, n_high)[:n_high]
        high = top[np.argsort(-col[top])]
    zero = np.where(col == 0)[0]
    if len(zero) >= n_low:
        low = rng.choice(zero, size=n_low, replace=False)
    elif len(zero) > 0:
        low = zero
    else:  # no exact zeros (dense activation): use the lowest-magnitude rows instead
        low = np.argsort(col)[:n_low]
    return high, low


def _open_images(rel_paths: List[str], root: str) -> List[Image.Image]:
    return [Image.open(os.path.join(root, rp)).convert("RGB") for rp in rel_paths]


def _fidelity_metrics(ann: np.ndarray, is_high: np.ndarray, acts: np.ndarray) -> dict:
    """precision/recall/F1 of VLM annotations vs binarized activation, + corr (matches HypotheSAEs)."""
    ann = np.asarray(ann, dtype=bool)
    lab = np.asarray(is_high, dtype=bool)
    if not (lab.any() and (~lab).any()):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "correlation": 0.0}
    tp = int((ann & lab).sum()); fp = int((ann & ~lab).sum()); fn = int((~ann & lab).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    corr = float(np.corrcoef(acts, ann.astype(float))[0, 1]) if len(np.unique(ann)) > 1 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "correlation": corr}


def run(cell: config.Cell, with_fidelity: bool, vlm_batch: int, max_latents, device: str) -> None:
    out_path = _hyp_path(cell)
    uniq_path = os.path.join(config.SELECT_DIR, cell.encoder, cell.activation, cell.variant,
                             f"seed{cell.seed}", "unique_latents.json")
    if not os.path.isfile(uniq_path):
        raise FileNotFoundError(f"Missing Stage-2 unique_latents: {uniq_path}")
    with open(uniq_path) as f:
        latents = json.load(f)["latents"]
    if max_latents is None:
        max_latents = config.MAX_LATENTS_PER_CELL
    if max_latents is not None and len(latents) > max_latents:
        print(f"[{cell.tag}] capping interpreted latents {len(latents)} -> {max_latents} (by predictiveness)")
        latents = latents[:max_latents]

    done = {}
    if os.path.isfile(out_path):
        with open(out_path) as f:
            done = json.load(f)
    todo = [l for l in latents if str(l) not in done or (with_fidelity and "fidelity" not in done[str(l)])]
    if not todo:
        print(f"[{cell.tag}] interpretation already complete ({len(done)} latents).")
        return
    print(f"[{cell.tag}] interpreting {len(todo)} latents ({len(done)} cached); fidelity={with_fidelity}")

    train_acts = np.asarray(np.load(config.acts_path(cell, "train"), mmap_mode="r"), dtype=np.float32)
    train_rel = _load_manifest("train")
    train_root = config.IMAGE_ROOTS["train"]
    if with_fidelity:
        val_acts = np.asarray(np.load(config.acts_path(cell, "val"), mmap_mode="r"), dtype=np.float32)
        val_rel = _load_manifest("val")
        val_root = config.IMAGE_ROOTS["val"]
    rng = np.random.default_rng(7 + cell.seed)

    # Build the VLM once.
    from hypogen.vlm_backend import VLM
    vlm = VLM(seed=cell.seed)

    # Process in chunks to bound memory (each chunk: load images -> interpret -> optional fidelity).
    for start in range(0, len(todo), vlm_batch):
        chunk = todo[start:start + vlm_batch]
        items, meta = [], []
        for lat in chunk:
            col = train_acts[:, lat]
            hi, lo = sample_high_zero(col, config.N_HIGH_INTERP, config.N_LOW_INTERP, rng)
            pos_imgs = _open_images([train_rel[i] for i in hi], train_root)
            neg_imgs = _open_images([train_rel[i] for i in lo], train_root)
            items.append((pos_imgs, neg_imgs))
            meta.append({"latent": int(lat), "top_image_rel": [train_rel[i] for i in hi]})
        hyps = vlm.interpret(items)
        for m, hyp in zip(meta, hyps):
            done[str(m["latent"])] = {"latent": m["latent"], "hypothesis": hyp,
                                      "top_image_rel": m["top_image_rel"]}

        if with_fidelity:
            # Build annotation items for this chunk's hypotheses on held-out VAL high/zero images.
            ann_items, ann_index = [], []  # ann_index: (latent, is_high_bool, act_value)
            for lat in chunk:
                hyp = done[str(lat)]["hypothesis"]
                if not hyp:
                    continue
                col = val_acts[:, lat]
                hi, lo = sample_high_zero(col, config.N_FIDELITY // 2, config.N_FIDELITY // 2, rng)
                for i in hi:
                    ann_items.append((Image.open(os.path.join(val_root, val_rel[i])).convert("RGB"), hyp))
                    ann_index.append((lat, True, float(col[i])))
                for i in lo:
                    ann_items.append((Image.open(os.path.join(val_root, val_rel[i])).convert("RGB"), hyp))
                    ann_index.append((lat, False, float(col[i])))
            anns = vlm.annotate(ann_items)
            # Group annotations back per latent and compute fidelity.
            per = {}
            for (lat, is_high, av), a in zip(ann_index, anns):
                per.setdefault(lat, {"ann": [], "is_high": [], "act": []})
                per[lat]["ann"].append(a); per[lat]["is_high"].append(is_high); per[lat]["act"].append(av)
            for lat, d in per.items():
                done[str(lat)]["fidelity"] = _fidelity_metrics(
                    np.array(d["ann"]), np.array(d["is_high"]), np.array(d["act"]))

        if (start // vlm_batch) % max(1, config.INTERP_FLUSH_EVERY // max(vlm_batch, 1)) == 0:
            _atomic_write_json(done, out_path)
        print(f"[{cell.tag}] {min(start + vlm_batch, len(todo))}/{len(todo)} interpreted", flush=True)

    _atomic_write_json(done, out_path)
    n_valid = sum(1 for v in done.values() if v.get("hypothesis"))
    print(f"[{cell.tag}] done: {len(done)} latents ({n_valid} with a hypothesis) -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Stage 3: VLM interpretation of selected latents.")
    ap.add_argument("--encoder", required=True, choices=config.ENCODERS)
    ap.add_argument("--activation", required=True, choices=[p.name for p in config.ACTIVATION_PAIRS])
    ap.add_argument("--variant", required=True, choices=config.ALLOWED_VARIANTS)
    ap.add_argument("--seed", required=True, type=int, choices=config.SEEDS)
    ap.add_argument("--with-fidelity", action="store_true", help="Also compute per-latent fidelity (extra VLM calls).")
    ap.add_argument("--vlm-batch", type=int, default=64, help="Latents per VLM batch (memory/throughput).")
    ap.add_argument("--max-latents", type=int, default=None, help="Cap interpreted latents (overrides config).")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run(config.Cell(args.encoder, args.activation, args.variant, args.seed),
        args.with_fidelity, args.vlm_batch, args.max_latents, args.device)


if __name__ == "__main__":
    main()
