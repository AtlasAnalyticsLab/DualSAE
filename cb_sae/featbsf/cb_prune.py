"""CB stage 3: prune the base model's latents by interpretability, select the
concept set, and precompute the holdout pseudo-GT.

Per (base run, concept set, R) — cheap (CPU-capable, ~minutes):
  1. RANK latents by I_m from the Broden dissect (stage 2). ADAPTATION vs the
     CB-SAE paper's I+S: feature space has no generative model, so there is
     no steerability score — this is the paper's own interpretability-only
     ranking arm, recorded in prune.json.
     Dead-policy rank-last: a never-firing unit's dissect label is an
     artifact of a degenerate top-100 evidence set, so dead units are hard-
     excluded from the retained set whenever enough live units exist.
  2. MATERIALIZE the retained-latents model via clone_pruned_bsf ->
     CKPT_DIR/pruned_<run>_R<R>/model.pt (once per base run — the ranking is
     concept-set independent; concurrent cells derive identical content from
     the same dissect artifacts and the write is atomic, so the loser's rename
     is a no-op overwrite).
  3. FIDELITY: gate-independent families state retention=1.0; families whose
     deployment gate ranks units against each other get block_retention_stats
     on validation rows (a per-sample top-k simulation; for a batch-global
     top-k base it is a proxy, not the exact batch-global gate).
  4. CONCEPT SET: C = C_user \\ C_rsae when the set's subtract_rsae is True
     (vlg — the paper's rule for a large object vocabulary): the concepts
     already assigned to retained latents are removed.
  5. y_val: supervisor image embeds (probe split) x concept text embeds ->
     (N_probe, |C|) fp16 raw cosines, for the stage-4 holdout metrics. Plus
     the supervisor-quality gates: the noise-floor canary (a cross-space
     product sits at 1/sqrt(d)), the effective rank of y (the true cap on
     independent knobs), and the cosine-cubed eps-guard margin.

Artifacts under cb_config.prune_dir(run, cset, R)/ :
  retained_idx.npy / discarded_idx.npy, concepts.json, y_val.npy,
  prune.json, DONE (or DEGENERATE.json when n_live < R)
plus CKPT_DIR/pruned_<run>_R<R>/model.pt.

concepts.json records the vocabulary by FILE NAME (resolved against
CB_VOCAB_DIR by featbsf/cb_data.load_concept_embeds) plus its sha256, so the
artifact tree can be moved between machines.
"""
from __future__ import annotations

import argparse
import json
import re
import os
import time
from pathlib import Path

import numpy as np
import torch

from bsf.bsf_prune import (block_retention_stats, bsf_prune_exact,
                           clone_pruned_bsf)
from bsf.cb_dissect import _sha16, load_vocab
from bsf.models import load_bsf, save_bsf
from featbsf import cb_config as CB
from featbsf import config as C
from featbsf import data as D
from featbsf.cb_data import load_probe_img_embeds, load_text_embeds


def _atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def effective_rank(y: np.ndarray) -> float:
    """Participation ratio of y's singular values — how many INDEPENDENT
    concept directions the supervisor actually provides; L_int cannot shape
    more axes than this."""
    s = np.linalg.svd(y - y.mean(0, keepdims=True), compute_uv=False)
    p = s ** 2
    p = p / max(p.sum(), 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def eps_guard(y: np.ndarray, batch: int, eps: float = 1e-6,
              draws: int = 10, seed: int = 0) -> dict:
    """How close cosine_cubed's clamp_min(eps) column guard comes to tripping
    at the training batch size — a column under it trains on ~no gradient,
    invisibly."""
    rng = np.random.default_rng(seed)
    yt = torch.from_numpy(y.astype(np.float32))
    mins = []
    for _ in range(draws):
        b = yt[rng.choice(len(y), min(batch, len(y)), replace=False)]
        nrm = ((b - b.mean(0, keepdim=True)) ** 3).norm(p=2, dim=0)
        mins.append(float(nrm.min()))
    return {"eps": eps, "min_column_norm": float(min(mins)),
            "margin_x": float(min(mins) / eps)}


@torch.no_grad()
def _fidelity(model, keep: np.ndarray, dataset: str, encoder: str,
              device: str, rows: int = 16384) -> dict:
    """block_retention_stats over up to `rows` validation rows (row-weighted
    mean across chunks, exact for per-token statistics)."""
    loader = D.build_loader(dataset, encoder, CB.probe_split(dataset),
                            batch_size=8192, labels="none", shuffle=False,
                            num_workers=2)
    keep_t = torch.from_numpy(keep).long()
    acc: dict = {}
    seen = 0
    for x in loader:
        x = x.to(device, torch.float32)
        st = block_retention_stats(model, model.normalize(x), keep_t)
        w = st["n_rows"]
        for k in ("retention", "promoted_per_token", "dead_in_batch"):
            acc[k] = acc.get(k, 0.0) + st[k] * w
        # 'preservation' only exists on the inexact branch.
        if "preservation" in st:
            acc["preservation"] = acc.get("preservation", 0.0) \
                + st["preservation"] * w
        seen += w
        if seen >= rows:
            break
    out = {k: v / seen for k, v in acc.items()}
    out.update(prune_exact=bsf_prune_exact(model), n_rows=seen,
               k=int(model.l), G=int(model.G), R=int(len(keep)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="base (Dual)BSF run name")
    ap.add_argument("--cset", required=True, choices=list(CB.CONCEPT_SETS))
    ap.add_argument("--retain", type=int, default=CB.CB_RETAIN)
    ap.add_argument("--rank", choices=("interp", "random"), default="interp",
                    help="ablation control: 'random' replaces the I_m ranking "
                         "with a seeded uniform score (dead blocks still "
                         "hard-excluded) — tests whether the interpretability "
                         "ranking matters at all")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-degenerate", action="store_true",
                    help="proceed even when n_live < retain (default: refuse "
                         "and withhold DONE so the chain never spends GPU on "
                         "a degenerate cell)")
    args = ap.parse_args()

    out_dir = CB.prune_dir(args.run, args.cset, args.retain, args.rank)
    if (out_dir / "DONE").exists():
        print(f"[skip] prune complete for {args.run}__{args.cset}")
        return
    CB.ensure_cb_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ddir = CB.dissect_dir(args.run)
    assert (ddir / "DONE").exists(), \
        f"dissect incomplete for {args.run} — run featbsf.cb_dissect first"
    meta = json.loads((ddir / "meta.json").read_text())
    dataset, encoder = meta["dataset"], meta["encoder"]

    # ---- 1. rank by I_m (interp-only), dead rank-last ----------------------
    dbroden = np.load(ddir / f"dissect_{CB.DISSECT_VOCAB.stem}.npz")
    I = dbroden["score"].astype(np.float32)                # [0,1] minmax'd
    dead = np.load(ddir / "dead_mask.npy")
    G = len(I)
    if args.rank == "random":
        # Seeded by the BASE run's seed: the control is reproducible and its
        # randomness is independent across seeds (the axis we average over).
        m_seed = re.search(r"_seed(\d+)", args.run)
        rng = np.random.default_rng(int(m_seed.group(1)) if m_seed else 0)
        combo = rng.random(G).astype(np.float32)
    else:
        combo = I.copy()
    n_live = int((~dead).sum())
    dead_excluded = 0
    if n_live >= args.retain:
        # -inf, not a penalty: a hard guarantee no dead block enters the
        # retained set (see the module docstring).
        combo = np.where(dead, -np.inf, combo)
        dead_excluded = int(dead.sum())
    elif not args.allow_degenerate:
        # Check n_live >= retain BEFORE spending the chain — a base with too
        # few live units would carry every one of its CB arms (training +
        # evaluation) through as a guaranteed-degenerate model mislabeled
        # with its nominal R. Write the evidence, but NOT the DONE guard, so the driver never
        # advances this cell; --allow-degenerate overrides deliberately.
        _atomic_json(out_dir / "DEGENERATE.json", {
            "run": args.run, "cset": args.cset, "n_live": n_live,
            "retain": args.retain, "dead_total": int(dead.sum()),
        })
        raise SystemExit(
            f"DEGENERATE: only {n_live} live blocks < R={args.retain} for "
            f"{args.run} — refusing to advance this cell (DONE not written; "
            f"see DEGENERATE.json). Reduce --retain, or re-run with "
            f"--allow-degenerate to force.")
    else:
        print(f"  WARNING: only {n_live} live blocks < R={args.retain} — "
              f"{args.retain - n_live} retained blocks will never fire "
              f"(--allow-degenerate: proceeding; flagged in prune.json).")
    order = np.argsort(-combo, kind="stable")
    keep = np.sort(order[:args.retain]).astype(np.int64)
    drop = np.sort(order[args.retain:]).astype(np.int64)
    np.save(out_dir / "retained_idx.npy", keep)
    np.save(out_dir / "discarded_idx.npy", drop)

    # ---- 2. materialize the pruned model (once per base run) ---------------
    model = load_bsf(C.CKPT_DIR / args.run / "model.pt", args.device)
    pruned_run = CB.pruned_run_name(args.run, args.retain, args.rank)
    pdir = C.CKPT_DIR / pruned_run
    pdir.mkdir(parents=True, exist_ok=True)
    if not (pdir / "model.pt").exists():
        pruned = clone_pruned_bsf(model, torch.from_numpy(keep))
        tmp = pdir / f"model.tmp{os.getpid()}.pt"
        save_bsf(pruned.cpu(), tmp,
                 extra={"parent_run": args.run,
                        "dataset": dataset, "extractor": encoder,
                        "retained_idx": keep.tolist(),
                        "rank_by": args.rank})
        tmp.replace(pdir / "model.pt")
        print(f"[write] {pdir}/model.pt (R={args.retain})")
    else:
        print(f"[cache] {pdir}/model.pt")

    # ---- 3. prune fidelity --------------------------------------------------
    fidelity = _fidelity(model, keep, dataset, encoder, args.device)
    print(f"  [fidelity] exact={fidelity['prune_exact']} "
          f"retention={fidelity['retention']:.4f} "
          f"promoted/sample={fidelity['promoted_per_token']:.2f} "
          f"of l={fidelity['k']}")

    # ---- 4. concept set -----------------------------------------------------
    cs = CB.CONCEPT_SETS[args.cset]
    user_vocab = load_vocab(cs["file"])
    duser = np.load(ddir / f"dissect_{cs['file'].stem}.npz")
    assert int(duser["concept_idx"].max()) < len(user_vocab), \
        "dissect/user-vocab mismatch — stale dissect artifacts?"
    rsae_idx = set(int(i) for i in duser["concept_idx"][keep])
    drop_set = rsae_idx if cs["subtract_rsae"] else set()
    cb_concepts = [c for j, c in enumerate(user_vocab) if j not in drop_set]
    cb_src_idx = [j for j in range(len(user_vocab)) if j not in drop_set]
    _atomic_json(out_dir / "concepts.json", {
        "concepts": cb_concepts,
        "user_vocab_idx": cb_src_idx,
        # File name only (portable); cb_config.resolve_vocab locates it.
        "source": cs["file"].name, "source_sha": _sha16(cs["file"]),
        "n_user": len(user_vocab), "n_rsae": len(rsae_idx),
        "n_cb": len(cb_concepts), "k_cb": cs["k_cb"],
        "subtract_rsae": bool(cs["subtract_rsae"]),
    })
    print(f"concept set {args.cset}: |C_user|={len(user_vocab)} "
          f"|C_rsae|={len(rsae_idx)}"
          f"{'' if cs['subtract_rsae'] else ' (KEPT)'} "
          f"-> |C|={len(cb_concepts)}  k_cb={cs['k_cb']} "
          f"({cs['k_cb'] / max(len(cb_concepts), 1):.2%} active rate)")

    # ---- 5. y_val + supervisor-quality gates -------------------------------
    img_embeds, _ = load_probe_img_embeds(dataset)
    text = load_text_embeds(cs["file"])[np.asarray(cb_src_idx)]
    y = img_embeds @ text.T                                   # (N_probe, |C|)
    sd = float(y.std())
    noise_floor = 1.0 / np.sqrt(text.shape[1])
    np.save(out_dir / "y_val.npy", y.astype(np.float16))
    erank = effective_rank(y)
    guard = eps_guard(y, batch=CB.CB_BATCH)
    # Cross-space canary. A raw sd threshold (the bsf-chain's 2/sqrt(d)) is
    # NOT sufficient here: CLIP-L/14@768's genuine image-text cosines have
    # per-column sd ~0.034, i.e. AT the 1/sqrt(768) scale, while passing the
    # zero-shot gate at 78.9% top-1 (measured on the pilot). The decisive
    # signature is STRUCTURE: a genuine supervisor's y is strongly low-rank
    # (measured effective rank 59 of |C|=4505), whereas an independent
    # cross-space product is near-isotropic (effective rank ~ min(N, |C|)).
    # So warn only when the sd is at the floor AND the rank shows no
    # structure — both stats are recorded in prune.json either way, and the
    # preflight zero-shot gate (ALIGNMENT_OK.json) remains the hard check.
    if sd < 2.0 * noise_floor and erank > 0.5 * min(y.shape):
        print(f"  WARNING: y_val sd={sd:.4f} at the {noise_floor:.4f} noise "
              f"floor AND effective_rank={erank:.0f} ~ isotropic — supervisor "
              f"image/text spaces likely mismatch. Check ALIGNMENT_OK.json.")
    print(f"y_val {y.shape}: sd={sd:.4f} effective_rank={erank:.1f} "
          f"eps margin={guard['margin_x']:.1e}x")

    _atomic_json(out_dir / "prune.json", {
        "run": args.run, "cset": args.cset, "dataset": dataset,
        "encoder": encoder, "G": G, "retain": args.retain,
        "rank_by": args.rank,                      # no S_m in feature space
        "normalization": "minmax-over-population (D-NORM)",
        "pruned_run": pruned_run,
        "prune_fidelity": fidelity,
        "I_mean": float(I.mean()),
        "I_mean_kept": float(I[keep].mean()),
        "I_mean_dropped": float(I[drop].mean()),
        "dead_total": int(dead.sum()), "n_live": n_live,
        "dead_excluded": dead_excluded,
        "dead_kept_frac": float(dead[keep].mean()),
        "y_val_sd": sd, "y_noise_floor": noise_floor,
        "effective_rank": erank, "eps_guard": guard,
        "supervisor": CB.SUPERVISOR_TAG,
        "wall_seconds": round(time.time() - t0, 1),
    })
    (out_dir / "DONE").touch()
    print(f"DONE prune {args.run}__{args.cset} "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
