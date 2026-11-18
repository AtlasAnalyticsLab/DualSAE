"""Paired LMDB loading + on-the-fly pseudo-GT for the CB-DualBSF stage-4
trainer and the CB evaluators.

The CB objectives (L_int / L_st) need, for every training row, the supervisor
image embedding of the SAME image the substrate feature came from. The clip
LMDB rows ARE those embeddings (extracted with the supervisor checkpoint and
L2-normalized by the loader), and the three encoder LMDBs are row-aligned
(gated by cb_text_embeds --check-alignment). So:

  * substrate == clip : one dataset, batches are (x, x) — zero extra IO.
  * substrate in {vit, sigclip}: two datasets read by SHARED index.

y_pseudo is computed on the fly as y = x_clip @ T_cb^T (raw cosines, the
LF-CBM concept-activation convention — no softmax): the concept text matrix
T_cb is ~|C| x 768 fp32 (< 15 MB), so a matmul per batch is far cheaper than
materializing a 1.28M x |C| y for every run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from featbsf import cb_config as CB
from featbsf import config as C
from featbsf._saepath import ensure_sae_on_path

ensure_sae_on_path()
from dataset.lmdb_features import LMDBFeatureDataset  # noqa: E402


class PairedLMDBFeatures(Dataset):
    """(x_substrate, x_clip) by shared index. Both sides go through
    LMDBFeatureDataset (lazy env open -> DataLoader-worker fork-safe;
    L2-normalized per row, which on the clip side is exactly the unit image
    embedding the supervisor products expect)."""

    def __init__(self, dataset: str, encoder: str, split: str):
        from featbsf.data import _resolve_lmdb
        self.sub = LMDBFeatureDataset(_resolve_lmdb(dataset, encoder, split),
                                      verbose=False)
        self.same = (encoder == "clip")
        if self.same:
            self.clip = self.sub
        else:
            self.clip = LMDBFeatureDataset(
                _resolve_lmdb(dataset, "clip", split), verbose=False)
            if len(self.sub) != len(self.clip):
                raise RuntimeError(
                    f"paired stores disagree on length: {encoder} "
                    f"{len(self.sub)} vs clip {len(self.clip)} ({split})")

    def __len__(self) -> int:
        return len(self.sub)

    def __getitem__(self, idx: int):
        x = self.sub[idx]
        # Same store: return the SAME array twice (collate stacks it into two
        # views; no second LMDB read, no copy beyond collate's own).
        return (x, x) if self.same else (x, self.clip[idx])


def build_paired_loader(dataset: str, encoder: str, split: str,
                        batch_size: int, shuffle: bool = False,
                        num_workers: int = 4, drop_last: bool = False,
                        generator: torch.Generator | None = None,
                        require_alignment: bool = True) -> DataLoader:
    """featbsf.data.build_loader conventions (pin_memory, epoch-seeded
    generator support) over the paired dataset.

    require_alignment: cross-encoder pairing REFUSES to run before the
    preflight gate has passed (cb_text_embeds --check-alignment) — an
    unverified index join would produce well-shaped, silently-wrong y.
    The clip substrate pairs a store with itself, so the gate is moot there.
    Tests pass False with synthetic stores.
    """
    if require_alignment and encoder != "clip" \
            and not CB.alignment_ok_path(dataset).exists():
        raise RuntimeError(
            f"{CB.alignment_ok_path(dataset)} missing — run "
            f"'python -m featbsf.cb_text_embeds --check-alignment "
            f"--dataset {dataset}' before training cross-encoder CB runs")
    ds = PairedLMDBFeatures(dataset, encoder, split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=drop_last, generator=generator)


def make_y(x_clip: torch.Tensor, T_cb: torch.Tensor) -> torch.Tensor:
    """Pseudo-GT concept activations for a batch: raw cosines (both operands
    are unit-normalized), fp32. T_cb: (|C|, 768) on x_clip's device."""
    return x_clip.float() @ T_cb.T


# ---------------------------------------------------------------------------
# Supervisor-embedding loading with provenance checks.
# ---------------------------------------------------------------------------
def load_text_embeds(vocab: Path) -> np.ndarray:
    """Full-vocab text embeddings (V, 768) fp32, asserting the sidecar's
    encoder tag AND the vocab file's current sha — a stale cache after a
    vocab edit would silently mislabel every concept."""
    import hashlib
    p = CB.text_embed_path(vocab)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — run cb_sae/prepare_assets.sh (it calls "
            f"'python -m featbsf.cb_text_embeds --embeds') first")
    side = json.loads(p.with_suffix(".json").read_text())
    if side.get("encoder") != CB.SUPERVISOR_TAG:
        raise RuntimeError(f"{p}: encoder {side.get('encoder')!r} != "
                           f"supervisor {CB.SUPERVISOR_TAG!r}")
    sha = hashlib.sha256(vocab.read_bytes()).hexdigest()[:16]
    if side.get("sha16") != sha:
        raise RuntimeError(f"{p}: cached for vocab sha {side.get('sha16')} "
                           f"but {vocab} is now {sha} — rebuild --embeds")
    emb = np.load(p).astype(np.float32)
    if side.get("n") != emb.shape[0]:
        raise RuntimeError(f"{p}: sidecar n={side.get('n')} != rows "
                           f"{emb.shape[0]}")
    return emb


def load_concept_embeds(pdir: Path) -> tuple[torch.Tensor, dict]:
    """(T_cb (|C|, 768) fp32 tensor, concepts.json dict) for a prune dir —
    the full user vocab's embeddings restricted to the selected CB concepts
    via the recorded user_vocab_idx (the on-the-fly y contract).

    The vocabulary is located through cb_config.resolve_vocab (a file name
    relative to CB_VOCAB_DIR, or a legacy absolute path that still exists).
    Its sha256 is checked against the sha recorded in concepts.json here, and
    against the embedding-cache sidecar in load_text_embeds, so a moved or
    edited vocabulary can never be silently paired with stale embeddings."""
    import hashlib
    concepts = json.loads((pdir / "concepts.json").read_text())
    vocab = CB.resolve_vocab(concepts["source"])
    sha = hashlib.sha256(vocab.read_bytes()).hexdigest()[:16]
    if concepts.get("source_sha") not in (None, sha):
        raise RuntimeError(f"{pdir}/concepts.json was built from vocab sha "
                           f"{concepts['source_sha']} but {vocab} is {sha}")
    emb = load_text_embeds(vocab)
    T = torch.from_numpy(emb[np.asarray(concepts["user_vocab_idx"],
                                        dtype=np.int64)])
    assert T.shape[0] == concepts["n_cb"], (T.shape, concepts["n_cb"])
    return T, concepts


def load_probe_img_embeds(dataset: str = CB.CB_DATASET
                          ) -> tuple[np.ndarray, np.ndarray]:
    """(img_embeds (N, 768) fp32, labels (N,)) of the probing split, with the
    provenance assert."""
    p = CB.img_embed_path(dataset=dataset)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — run 'python -m featbsf.cb_text_embeds "
            f"--img-embeds' first")
    side = json.loads(p.with_suffix(".json").read_text())
    if side.get("encoder") != CB.SUPERVISOR_TAG:
        raise RuntimeError(f"{p}: encoder {side.get('encoder')!r} != "
                           f"supervisor {CB.SUPERVISOR_TAG!r}")
    img = np.load(p).astype(np.float32)
    lab_p = p.with_name(p.stem + "_labels.npy")
    # OIv7 dumps have no scalar labels (multilabel dataset; the labels file
    # only ever served the imagenet zero-shot gate) — consumers use embeds
    # only (`img_embeds, _ = ...`).
    lab = np.load(lab_p) if lab_p.exists() else None
    return img, lab
