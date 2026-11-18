"""CB stage 2: dissect a converted base's latents against the supervisor
space — per-latent interpretability I_m + concept assignments.

Per base run (GPU, minutes):
  1. PROFILES: stream the probing corpus (the dataset's evaluation split, rows
     of the run's OWN substrate encoder) through the frozen model's DEPLOYMENT
     gate; profile of latent g = its gated code magnitude per image ->
     block_scores.npy (N_probe, G) fp16. One row = one image (the LMDBs are
     image-level features), so no per-token max-pooling is needed.
     (b=1 bases: a "block" is one latent; block_norms(z) = |z|.)
  2. DEAD MASK: latents that never fire on the probing set (feeds the prune
     stage's dead-policy rank-last and the retain check of the driver).
  3. DISSECT: CLIP-Dissect soft-WPMI (bsf.cb_dissect) of the profiles against
       * the Broden vocab  -> I_m + Broden label   (the ranking vocab)
       * every concept set's user vocab -> the C_rsae assignments
     Image side = the supervisor probe-split embeds (the clip LMDB rows,
     dumped by cb_text_embeds --img-embeds); text side = the bare-string
     vocab embeds. dissect() hard-asserts both provenance tags match.

Artifacts under cb_config.dissect_dir(run)/ :
  block_scores.npy      (N_probe, G) fp16 gated code magnitudes
  dead_mask.npy         (G,) bool — never fired on the probing set
  dissect_<vocab>.npz   {wpmi, score, concept_idx} per vocab (cb_prune schema)
  meta.json, DONE       provenance + the skip-guard

ADAPTATION vs the CB-SAE paper: there is NO steerability score S_m — feature
space has no generative model to steer — so pruning ranks by
interpretability alone (the paper's interpretability-only ranking arm).
Recorded in prune.json.

Batch-size note: the probe pass encodes in batches of --batch (8192 for the
reported runs). A batch-global top-k base gates each batch jointly, so the
profiles depend (mildly) on this batch size.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from bsf.cb_dissect import _sha16, dissect
from bsf.models import block_norms, load_bsf
from featbsf import cb_config as CB
from featbsf import config as C
from featbsf import data as D
from featbsf.cb_data import load_probe_img_embeds, load_text_embeds


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(f".tmp{os.getpid()}.npy")
    np.save(tmp, arr)
    tmp.replace(path)


@torch.no_grad()
def block_profiles(model, dataset: str, encoder: str, device: str,
                   batch: int = 8192, num_workers: int = 4) -> np.ndarray:
    """(N_probe, G) fp16 gated block norms over the probing split, through the
    side-effect-free deployment gate (model.encode)."""
    loader = D.build_loader(dataset, encoder, CB.probe_split(dataset),
                            batch_size=batch, labels="none", shuffle=False,
                            num_workers=num_workers)
    rows = []
    for x in loader:
        x = x.to(device, torch.float32)
        _, z, _ = model.encode(model.normalize(x))
        rows.append(block_norms(z).to("cpu", torch.float16))
    return torch.cat(rows).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True,
                    help="base (Dual)BSF run name (CKPT_DIR/<run>/model.pt)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    out_dir = CB.dissect_dir(args.run)
    if (out_dir / "DONE").exists():
        print(f"[skip] dissect complete for {args.run}")
        return
    # The dissect joins block profiles against the supervisor image embeds by
    # ROW INDEX; that join is only proven sound by the preflight gate. Without
    # this check a not-yet-verified store would produce well-formed, silently
    # cross-space dissect artifacts behind a DONE guard —
    # enforced for EVERY encoder for uniformity, not just the paired ones.
    # (alignment gate re-checked per dataset after meta is read, below)
    _ALIGN_HINT = (
        f"missing — run 'python -m featbsf.cb_text_embeds "
        f"--check-alignment --dataset <dataset>' before dissecting")
    CB.ensure_cb_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ckpt = C.CKPT_DIR / args.run / "model.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    # Dataset/encoder come from the checkpoint's own provenance (the converters
    # record them in extra) — a flag could disagree with the weights.
    dataset = payload["extra"]["dataset"]
    encoder = payload["extra"]["extractor"]
    assert CB.alignment_ok_path(dataset).exists(), (
        f"{CB.alignment_ok_path(dataset)} " + _ALIGN_HINT)
    model = load_bsf(ckpt, args.device)   # eval mode, deployment gates

    # ---- 1. block profiles (cached: the dissect passes below are cheap and
    # re-runnable against a finished profile) --------------------------------
    scores_p = out_dir / "block_scores.npy"
    if scores_p.exists():
        img_scores = np.load(scores_p)
        print(f"[cache] {scores_p.name} {img_scores.shape}")
    else:
        img_scores = block_profiles(model, dataset, encoder, args.device,
                                    args.batch, args.num_workers)
        _atomic_save(scores_p, img_scores)
        print(f"[write] {scores_p.name} {img_scores.shape} "
              f"({time.time() - t0:.0f}s)")

    # ---- 2. dead mask -------------------------------------------------------
    dead = ~(img_scores.astype(np.float32) > 0).any(axis=0)          # (G,)
    _atomic_save(out_dir / "dead_mask.npy", dead)
    print(f"dead blocks: {int(dead.sum())}/{len(dead)} "
          f"({float(dead.mean()):.4f})")

    # ---- 3. dissect vs every vocab -----------------------------------------
    img_embeds, _ = load_probe_img_embeds(dataset)
    assert img_embeds.shape[0] == img_scores.shape[0], (
        f"probe corpus mismatch: {img_embeds.shape[0]} supervisor embeds vs "
        f"{img_scores.shape[0]} profile rows — different splits?")
    vocabs = [CB.DISSECT_VOCAB] + [cs["file"]
                                   for cs in CB.CONCEPT_SETS.values()]
    for vocab in vocabs:
        out_npz = out_dir / f"dissect_{vocab.stem}.npz"
        if out_npz.exists():
            print(f"[cache] {out_npz.name}")
            continue
        text = load_text_embeds(vocab)
        res = dissect(img_scores.astype(np.float32), img_embeds, text,
                      device=args.device, a=10.0,
                      # Both tags are the supervisor: the image side is the
                      # clip LMDB rows, the text side the sidecar-verified
                      # cache — load_text_embeds already asserted it.
                      img_encoder=CB.SUPERVISOR_TAG,
                      text_encoder=CB.SUPERVISOR_TAG)
        tmp = out_npz.with_suffix(f".tmp{os.getpid()}.npz")
        np.savez(tmp, **res, vocab_sha=_sha16(vocab))
        tmp.replace(out_npz)
        print(f"[write] {out_npz.name}  I mean={res['score'].mean():.3f}")

    meta = {
        "run": args.run, "dataset": dataset, "encoder": encoder,
        "probe_split": CB.probe_split(dataset), "n_probe": int(img_scores.shape[0]),
        "G": int(img_scores.shape[1]), "supervisor": CB.SUPERVISOR_TAG,
        "dead_frac": float(dead.mean()),
        "vocabs": [v.stem for v in vocabs],
        "wall_seconds": round(time.time() - t0, 1),
    }
    tmp = (out_dir / "meta.json").with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(out_dir / "meta.json")
    (out_dir / "DONE").touch()
    print(f"DONE dissect {args.run} ({meta['wall_seconds']}s)")


if __name__ == "__main__":
    main()
