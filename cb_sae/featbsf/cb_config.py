"""Configuration for the feature-space CB-SAE / CB-DualSAE pipeline.

Extends featbsf/config.py (imported as the base protocol) with the CB stages'
constants, the concept sets, and the artifact naming every stage agrees on.

Pipeline stages (each a `python -m featbsf.<module>` CLI; driven per cell by
cb_sae/scripts/cb_pipeline.py):
  stage 0  cb_text_embeds     supervisor text / probe-image embeddings + gate
  stage 1  convert_*          base SAE checkpoint -> b=1 featurizer
  stage 2  cb_dissect         one cell per BASE run
  stage 3  cb_prune           one cell per (base run, concept set, R)
  stage 4  cb_train           one cell per CB arm (R, k_cb, b_cb, poly or not)

Like featbsf/config.py, this module imports ONLY the stdlib + featbsf.config.
"""
from __future__ import annotations

import os
from pathlib import Path

from featbsf import config as C

# ---------------------------------------------------------------------------
# Paths (all under the CB work directory).
# ---------------------------------------------------------------------------
CB_ROOT = C.WORK_DIR / "cb"
CB_EMBED_DIR = CB_ROOT / "embeds"        # supervisor text/image embeds + gates
CB_DISSECT_DIR = CB_ROOT / "dissect"     # <base_run>/ per-latent I_m artifacts
CB_PRUNE_DIR = CB_ROOT / "prune"         # <base_run>__<cset>[_R<R>]/ prune artifacts

_CB_DIRS = (CB_ROOT, CB_EMBED_DIR, CB_DISSECT_DIR, CB_PRUNE_DIR)


def ensure_cb_dirs() -> None:
    C.ensure_dirs()
    for d in _CB_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Supervisor space. ONE joint image-text space supervises every substrate
# (clip / vit / sigclip features), mirroring the CB-SAE paper's pattern where
# the supervisor (a CLIP model) is a different network from the substrate:
#   * image side = the clip LMDB rows themselves. They were extracted with
#     open_clip encode_image of EXACTLY this checkpoint (the vision/ feature
#     extraction), and LMDBFeatureDataset L2-normalizes at load — i.e. the
#     loader hands us unit image embeddings for free, for the SAME images the
#     vit/sigclip substrates index.
#   * text side = bare-string encode_text of the SAME checkpoint
#     (featbsf/cb_text_embeds.py; bare strings per the CLIP-Dissect / LF-CBM
#     protocol, NOT the 80-template zero-shot ensemble).
# SUPERVISOR_TAG is the provenance string stamped on BOTH sides; the dissect
# call hard-asserts equality (two same-width embeddings from different
# encoders multiply into silent noise).
# ---------------------------------------------------------------------------
OPEN_CLIP_MODEL = "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
SUPERVISOR_TAG = f"open_clip:{OPEN_CLIP_MODEL}"
SUPERVISOR_DIM = 768                     # == FEATURE_DIM['clip']
_CACHE_TAG = "clipL14dc"                 # short filename token for the caches

# Probing corpus for dissect + I_m + the C_rsae assignment + y_val: the
# evaluation split of each dataset (ImageNet-1K validation, 50k rows;
# OpenImagesV7 'valid'), 1 row = 1 image, index-aligned across the three
# encoder LMDBs — the alignment gate in cb_text_embeds.py verifies this.
#
# PER-ENCODER CAVEAT (do not compare concept-alignment numbers across
# encoders): for the clip substrate the supervisor image embeds ARE the
# model's own input rows, so y = x @ T^T is an exact affine function of the
# input and L_int is closed-form solvable — on clip, the concept-alignment
# numbers (l_int/l_st) measure optimizer convergence more than concept
# learning. On vit/sigclip E_cb must bridge a genuinely different
# representation. Reconstruction metrics (R2/KNN/LP) are unaffected either
# way (they score the substrate, not the supervisor).
CB_DATASET = "imagenet"
PROBE_SPLIT = "validation"
# The CB pipeline is dataset-threaded via the base checkpoint's own
# provenance; these lookups replace the two constants above at every
# dataset-sensitive site (the constants stay as the defaults).
PROBE_SPLITS = {"imagenet": "validation", "open_images": "valid"}


def probe_split(dataset: str = CB_DATASET) -> str:
    return PROBE_SPLITS[dataset]

# ---------------------------------------------------------------------------
# CB protocol constants (CLI defaults of the stage scripts).
#
# NOTE: CB_RETAIN, CB_B_MAIN and the concept-set k_cb below are the stage
# scripts' DEFAULTS. The CB-SAE / CB-DualSAE runs of the main comparison pass
# --retain 1024 --b-cb 1 --k-cb 5 explicitly (cb_sae/scripts/cb_pipeline.py);
# keep that in mind when calling the stages by hand.
# ---------------------------------------------------------------------------
CB_RETAIN = 256          # default R (naming: a non-default R gets an '_R<R>' tag)
CB_B_MAIN = 4            # default concept block width (1 = the paper-faithful scalar CB)
B_CB_ARMS = (CB_B_MAIN, 1)
CB_EPOCHS = 25           # ~15.6k steps at BATCH=2048 over 1.28M rows (ImageNet-1K)
CB_WARMUP = 2500         # CB-first phase: Residual Branch frozen + span off for the
                         # first 2500 steps so the sparse concept head captures the
                         # concept-aligned residual before the dense Residual Branch
                         # can swallow it (mirrors the DualSAE warm-up's role).
CB_LR = C.LR             # 1e-4 constant
CB_BATCH = C.BATCH       # 2048
CB_LAM_SPAN = C.LAM_SPAN  # 0.01 span coefficient (spans retained latents + D_cb)
CB_CKPT_EVERY = 2000

# ---------------------------------------------------------------------------
# Concept sets. The concept vocabularies are NOT shipped (their upstream
# repositories carry no license); cb_sae/prepare_assets.sh downloads them at a
# pinned commit into CB_VOCAB_DIR and verifies their sha256.
#
# k_cb sizing rule used for the default: keep the active rate k_cb/|C| in the
# 1-3% band — below it most concepts never win a decoder slot, get no
# gradient, and die. vlg: ~4000-4500 concepts after the C_rsae subtraction ->
# default k_cb=64 (~1.4%). The CB-SAE paper's own value is k_cb=5; the CB
# rows of the main comparison use k_cb=5 (passed explicitly), and the
# pipeline driver also runs k_cb=64 as a flag-selectable arm.
# ---------------------------------------------------------------------------
VOCAB_DIR = Path(os.environ.get("CB_VOCAB_DIR",
                                "./assets/cb_sae/concepts")).resolve()
CONCEPT_SETS: dict[str, dict] = {
    # C_user: the VLG-CBM ImageNet-filtered concept set (also used for
    # OpenImagesV7 — CB-SAE's concept set is user-specified by construction).
    "vlg": {"file": VOCAB_DIR / "vlgcbm_imagenet_filtered.txt",
            "subtract_rsae": True, "k_cb": 64},
}
CSET_ORDER = ("vlg",)                    # frozen iteration order
# I_m ranking vocabulary: the Broden label set used by CLIP-Dissect.
DISSECT_VOCAB = VOCAB_DIR / "broden_labels_clean.txt"


def resolve_vocab(source: str | Path) -> Path:
    """Locate a concept vocabulary recorded in an artifact (concepts.json).

    New artifacts record only the FILE NAME (resolved against VOCAB_DIR, so a
    prune directory stays valid after the tree is moved). Older artifacts
    recorded an absolute path; it is used when it still exists, otherwise the
    same file name is looked up in VOCAB_DIR. Integrity is checked by the
    caller against the recorded sha256, never by the path.
    """
    p = Path(source)
    if p.is_absolute() and p.exists():
        return p
    return VOCAB_DIR / p.name


# ---------------------------------------------------------------------------
# Embed cache paths (encoder-keyed filenames; each .npy has a .json sidecar
# recording the encoder tag + vocab sha, and consumers assert the tag).
# ---------------------------------------------------------------------------
def text_embed_path(vocab: Path) -> Path:
    return CB_EMBED_DIR / f"text_{_CACHE_TAG}_{vocab.stem}.npy"


def img_embed_path(split: str | None = None,
                   dataset: str = CB_DATASET) -> Path:
    split = split or PROBE_SPLITS[dataset]
    return CB_EMBED_DIR / f"img_{_CACHE_TAG}_{dataset}_{split}.npy"


def alignment_ok_path(dataset: str = CB_DATASET) -> Path:
    name = ("ALIGNMENT_OK.json" if dataset == "imagenet"
            else f"ALIGNMENT_OK_{dataset}.json")
    return CB_EMBED_DIR / name


ALIGNMENT_OK = alignment_ok_path()


# ---------------------------------------------------------------------------
# Artifact naming (shared by every stage; changing these renames every
# downstream directory).
# ---------------------------------------------------------------------------
def pruned_run_name(base_run: str, retain: int = CB_RETAIN,
                    rank: str = "interp") -> str:
    """Checkpoint dir of the materialized retained-latents model (stage 3).
    Concept-set independent: the ranking uses only the Broden I_m, so one
    pruned checkpoint serves every concept set. rank='random' (the
    prune-ranking control) gets a '_rand' tag; default names are unchanged."""
    tag = "" if rank == "interp" else "_rand"
    return f"pruned_{base_run}_R{retain}{tag}"


def prune_dir(base_run: str, cset: str, retain: int = CB_RETAIN,
              rank: str = "interp") -> Path:
    """Per-(base run, concept set) prune artifacts: concepts.json, y_val.npy,
    prune.json, DONE. retained_idx.npy lives here too (duplicated per cset —
    a few KB — so each prune dir is self-contained).

    A non-default retain gets an '_R{v}' dir so the R-dependent artifacts
    (retained_idx, prune.json, DONE) of different radii never collide."""
    tag = "" if retain == CB_RETAIN else f"_R{retain}"
    rtag = "" if rank == "interp" else "_rand"
    return CB_PRUNE_DIR / f"{base_run}__{cset}{tag}{rtag}"


def dissect_dir(base_run: str) -> Path:
    return CB_DISSECT_DIR / base_run


def cb_run_name(base_run: str, cset: str, b_cb: int, use_poly: bool,
                retain: int = CB_RETAIN, k_cb: int | None = None,
                rank: str = "interp") -> str:
    """Stage-4 checkpoint dir name. k_cb defaults to a pure function of cset
    (the CONCEPT_SETS table) and is then NOT in the name; an explicit override
    gets a '_kcb{v}' tag, so override runs never collide with default ones.
    (The 'cbdual_' prefix is historical and also used by the no-poly arm,
    which carries a '_nopoly' suffix.)"""
    poly_tag = "" if use_poly else "_nopoly"
    kcb_tag = "" if k_cb is None else f"_kcb{k_cb}"
    rtag = "" if rank == "interp" else "_rand"
    return (f"cbdual_{base_run}_R{retain}{rtag}_C{cset}_bcb{b_cb}"
            f"{kcb_tag}{poly_tag}")
