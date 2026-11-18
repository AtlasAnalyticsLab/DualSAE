"""Central configuration for the hypothesis-generation pipeline: checkpoint registry, encoder models,
output layout, and the experiment grid (encoders x activation-pairs x variants x seeds).

WHY a single registry: every stage iterates the SAME set of "cells" (one trained SAE =
encoder x activation x variant x seed). Keeping the enumeration + path resolution here means
the stages can never disagree about which checkpoint / embedding / output a cell maps to.

All SAE checkpoints are written by the repository's vision/ trainers (driven by
topic_modeling/src/tm_driver.py, which trains the SAEs shared by the topic-modeling and
hypothesis-generation studies) and load through `old_sae_adapter.load_old_sae(path, sae_type,
device)`. dict_size / k are read per-checkpoint from `model_cfg`.

Environment variables (all optional; relative defaults):
  HG_DATASET      imagenet | openimages                         (default imagenet)
  HG_MODE         main | selmethod                              (default main)
  HG_SEL_METHOD   correlation | separation_score (selmethod)    (default correlation)
  HG_OUT          output root                                   (default ./outputs/hypothesis_generation)
  TM_OUT          topic-modeling output root holding runs/<run>/<enc>/trained_models/...
                                                                (default ./outputs/topic_modeling)
  IMAGENET_TRAIN_DIR / IMAGENET_VAL_DIR                         (default ./data/imagenet/{train,val})
  OPENIMAGES_TEST_DIR / OPENIMAGES_VALID_DIR                    (default ./data/open_image_v7/{test,valid})
  FAITHFUL_CLASSES, FAITHFUL_POS_PER_CLASS                      (default 100, 200)
  HG_ENCODERS     comma list restricting the encoder axis       (default clip,siglip,vit)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

# --- make the shared helpers importable regardless of CWD -------------------------------------
# The stages import `old_sae_adapter` and `prepare_topic_data` (topic_modeling/src), which in turn
# import the repository's vision/ core. Paths are resolved relative to this file:
#   <repo>/hypothesis_generation/hypogen/config.py -> <repo>/topic_modeling/src
REPO_DIR = Path(__file__).resolve().parents[2]
_TM_SRC = os.environ.get("TM_SRC_DIR", str(REPO_DIR / "topic_modeling" / "src"))
if _TM_SRC not in sys.path:
    sys.path.insert(0, _TM_SRC)

# ---------------------------------------------------------------------------------------------
# Run mode + dataset (ENV-DRIVEN).
#   HG_MODE = "main"      : selection with all three rules, interpretation + faithful scoring with
#                           the primary rule (lasso).
#             "selmethod" : the whole faithful pipeline (selection, interpretation union, faithful)
#                           with HG_SEL_METHOD as the rule; outputs under selmethod_<rule>/.
#   HG_DATASET = "imagenet" | "openimages"
# ---------------------------------------------------------------------------------------------
MODE = os.environ.get("HG_MODE", "main")
if MODE not in ("main", "selmethod"):
    raise ValueError(f"HG_MODE must be 'main' or 'selmethod', got {MODE!r}")
DATASET = os.environ.get("HG_DATASET", "imagenet")
SEL_METHOD = os.environ.get("HG_SEL_METHOD", "correlation")

_OUT_BASE = os.environ.get("HG_OUT", "./outputs/hypothesis_generation")
_TM_RUNS = os.path.join(os.environ.get("TM_OUT", "./outputs/topic_modeling"), "runs")

# Per-dataset paths. NOTE: the split KEY is always "train"/"val"; only the on-disk path differs.
_DATASET_CFG = {
    "imagenet": dict(
        num_classes=1000,
        image_roots={"train": os.environ.get("IMAGENET_TRAIN_DIR", "./data/imagenet/train"),
                     "val": os.environ.get("IMAGENET_VAL_DIR", "./data/imagenet/val")},
        out_root=_OUT_BASE,
    ),
    "openimages": dict(
        # 601 = OpenImages V7 boxable classes (oidv7-class-descriptions-boxable); the pre-extracted
        # label LMDB stores a 601-d multi-hot float32 vector per image.
        num_classes=601,
        # IMPORTANT role->split mapping for OpenImages: the "train" ROLE (selection + interpretation image
        # sourcing) is served by the TEST split, and the "val" ROLE (faithful eval) by the VALID split. WHY:
        # the index->image-id ordering of the pre-extracted feature LMDBs is reconstructed from each split's
        # localized-narratives file (see build_oi_data.py / oi_verify_order.py), which is available for the
        # valid and test splits; test is disjoint from valid, so it fills the train role with no
        # interpretation/faithful leakage.
        image_roots={"train": os.environ.get("OPENIMAGES_TEST_DIR", "./data/open_image_v7/test"),
                     "val": os.environ.get("OPENIMAGES_VALID_DIR", "./data/open_image_v7/valid")},
        out_root=os.path.join(_OUT_BASE, "openimages"),
    ),
}
_DCFG = _DATASET_CFG[DATASET]
NUM_CLASSES = _DCFG["num_classes"]
# OpenImages is MULTI-LABEL (601-d multi-hot per image); ImageNet is single-label. Stages branch on this:
# labels load as (N, C) float multi-hot vs (N,) int, and the one-vs-rest target for class c is the c-th
# multi-hot column instead of `labels == c`. See onevsrest()/load_labels() below.
MULTILABEL = (DATASET == "openimages")
# On-disk raw images in ImageFolder layout (<root>/<class>/*) for ImageNet, flat for OpenImages.
IMAGE_ROOTS = _DCFG["image_roots"]

# Per-encoder embedding model — MUST match what the SAEs were trained on (same ids as
# topic_modeling/src/prepare_topic_data.py:ENCODER_MODELS, whose build_encoder() is reused to
# reproduce the training embeddings, then L2-normalized).
ENCODER_MODELS = {
    "clip": "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",  # open_clip, 768-d
    "siglip": "google/siglip2-so400m-patch14-384",               # transformers, 1152-d
    "vit": "google/vit-large-patch16-224",                       # transformers, 1024-d (CLS)
}
# The ImageNet validation feature LMDB per encoder — used ONLY by smoke.py to verify that the
# re-extracted embeddings match the SAE training distribution (max cosine ~ 1.0).
_FEAT_IMAGENET = os.environ.get("FEATURES_IMAGENET", "./features/imagenet")
_LMDB_VERIFY = {
    ("clip", "val"): os.path.join(_FEAT_IMAGENET, "validation", "clip_image_features_CLIP-ViT-L-14-DataComp.lmdb"),
    ("siglip", "val"): os.path.join(_FEAT_IMAGENET, "validation", "siglip2_image_features_siglip2-so400m-p14-384.lmdb"),
    ("vit", "val"): os.path.join(_FEAT_IMAGENET, "validation", "vit_features_vit-large-p16-224.lmdb"),
}

# ---------------------------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------------------------
ENCODERS = ["clip", "siglip", "vit"]
# Optional restriction of the encoder axis (e.g. HG_ENCODERS=clip for a partial run or the smoke test).
if os.environ.get("HG_ENCODERS"):
    ENCODERS = [e for e in ENCODERS if e in os.environ["HG_ENCODERS"].split(",")]
SEEDS = [42, 43, 44]


# Activation pairs: each compares a baseline trainer to its Dual counterpart (same activation).
@dataclass(frozen=True)
class ActivationPair:
    name: str             # short key used in paths / tables
    baseline_trainer: str # sae_type passed to load_old_sae for the baseline checkpoint
    dual_trainer: str     # sae_type passed to load_old_sae for the Dual checkpoint


ACTIVATION_PAIRS = [
    ActivationPair("topk", "TopKTrainer", "DualTopKTrainer"),
    ActivationPair("batch_topk", "BatchTopKTrainer", "DualBatchTopKTrainer"),
    ActivationPair("jump_relu", "JumpReluTrainer", "DualJumpReluTrainer"),
]

VARIANTS = ["baseline", "dual"]
ALLOWED_VARIANTS = VARIANTS

# Checkpoint roots = topic-modeling run trees (topic_modeling/configs/runs.json):
#   ImageNet-1K : Dual from run 'imagenet_paired_span001', baselines from run 'imagenet_paired'.
#   OpenImagesV7: baselines AND Dual from run 'openimages_paired'.
# HG_DUAL_ROOT / HG_BASE_ROOT override them (each must contain <enc>/trained_models/<Trainer>/seed<S>/).
if DATASET == "openimages":
    _DUAL_ROOT = _BASE_ROOT = os.path.join(_TM_RUNS, "openimages_paired")
else:
    _DUAL_ROOT = os.path.join(_TM_RUNS, "imagenet_paired_span001")
    _BASE_ROOT = os.path.join(_TM_RUNS, "imagenet_paired")
_DUAL_ROOT = os.environ.get("HG_DUAL_ROOT", _DUAL_ROOT)
_BASE_ROOT = os.environ.get("HG_BASE_ROOT", _BASE_ROOT)
CKPT_FILENAME = "best_val_monoscore_model.pt"

# ---------------------------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------------------------
OUT_ROOT = _DCFG["out_root"]
# selmethod mode routes selection / interpretation / evaluation to selmethod_<rule>/ but REUSES the
# main run's embeddings and dumped activations (same SAE checkpoints).
_STAGE_ROOT = os.path.join(OUT_ROOT, f"selmethod_{SEL_METHOD}") if MODE == "selmethod" else OUT_ROOT
EMB_DIR = os.path.join(OUT_ROOT, "embeddings")        # Stage 1a: per (encoder, split)
ACTS_DIR = os.path.join(OUT_ROOT, "acts")             # Stage 1b: per cell
SELECT_DIR = os.path.join(_STAGE_ROOT, "selected")    # Stage 2
INTERP_DIR = os.path.join(_STAGE_ROOT, "interpreted") # Stage 3
EVAL_DIR = os.path.join(_STAGE_ROOT, "eval")          # Stage 4
REPORT_DIR = os.path.join(_STAGE_ROOT, "reports")     # Stage 5

# ---------------------------------------------------------------------------------------------
# Pipeline hyper-parameters
# ---------------------------------------------------------------------------------------------
# Stage 1a: stratified train subsample used for selection + interpretation image sourcing. Fixed
# image set across all cells so comparisons are on identical images. Val uses all images.
TRAIN_SUBSAMPLE_PER_CLASS = 300
SUBSAMPLE_SEED = 12345

# Stage 2: feature selection
K_GRID = [8, 16]                                           # n_select per class, reported separately
SELECTION_METHODS = ["lasso", "correlation", "separation_score"]
PRIMARY_SELECTION_METHOD = "lasso"                        # method used for the (VLM-heavy) faithful eval
PRIMARY_K = 16
# correlation/separation are computed vectorized over ALL classes at once (a per-latent
# pearsonr/separation loop is far too slow at 1000 classes x 8192 latents). Definitions follow
# HypotheSAEs select_neurons_{correlation,separation_score} at the pinned commit.
SEPARATION_N_TOP = 100        # top-activating rows per latent for the separation score
# lasso: per class, one L1-logistic fit on a balanced subsample (all positives + sampled negatives),
# then top-K by |coef|. Single fit/class (no alpha binary search) keeps 1000 classes tractable.
LASSO_C = 1.0
LASSO_NEG_PER_CLASS = 2000
# Latents to interpret (Stage 3) = union over classes of the PRIMARY config's selections (the set the
# faithful eval needs). Proxy eval needs no interpretation, so it can use all methods x K for free.
INTERP_UNION_METHOD = PRIMARY_SELECTION_METHOD
INTERP_UNION_K = PRIMARY_K

# selmethod mode: run the WHOLE faithful pipeline with SEL_METHOD as the primary/interpret-union method,
# at K=16. Outputs route to selmethod_<SEL_METHOD>/.
if MODE == "selmethod":
    SELECTION_METHODS = [SEL_METHOD]
    K_GRID = [16]
    PRIMARY_K = 16
    PRIMARY_SELECTION_METHOD = SEL_METHOD
    INTERP_UNION_METHOD = SEL_METHOD
    INTERP_UNION_K = 16

# Stage 3: interpretation sampling (image analog of HypotheSAEs sample_top_zero)
N_HIGH_INTERP = 8        # top-activating images shown to the VLM proposer
N_LOW_INTERP = 8         # zero/low-activating images shown to the VLM proposer
# Stage 3b: per-latent fidelity scoring (held-out high/zero VAL images annotated by the VLM). Optional
# diagnostic, OFF by default (adds N_FIDELITY annotations per latent); enable with --with-fidelity.
N_FIDELITY = 40          # 20 high + 20 zero
# Cap on #latents interpreted per cell (ordered by aggregate predictiveness); None = no cap.
MAX_LATENTS_PER_CELL = None
# Flush the interpretation cache to disk every this many latents (resumable after interruption).
INTERP_FLUSH_EVERY = 200

# Stage 4 faithful (VLM annotation -> score_hypotheses). The faithful metric at all 1000 classes is a
# ~10^8 VLM-call budget, so it is bounded by FAITHFUL_CLASSES (proxy + selection still run at all
# classes). Per evaluated class: its val positives + FAITHFUL_NEG_PER_CLASS sampled negatives;
# annotations are cached globally by (hypothesis, image) so reuse across classes is free.
FAITHFUL_CLASSES = int(os.environ.get("FAITHFUL_CLASSES", "100"))  # classes for faithful metric (1000=all)
FAITHFUL_NEG_PER_CLASS = 200      # sampled val negatives per evaluated class
# Cap on POSITIVES per faithful class. Needed for MULTI-LABEL OpenImages: common classes (e.g. "Person")
# have THOUSANDS of val positives, which (a) blow up the per-cell VLM annotation budget and (b) let a few
# mega-classes dominate. Capping to ~200 pos (seeded subsample) matches the ~200 negatives and keeps cost
# bounded. No-op for ImageNet (<=50 pos/class < cap).
FAITHFUL_POS_PER_CLASS = int(os.environ.get("FAITHFUL_POS_PER_CLASS", "200"))
FAITHFUL_CLASS_SEED = 7           # seed for choosing which classes + negatives + positives are evaluated
CORRECTED_PVAL_THRESHOLD = 0.1    # Bonferroni family-wise threshold (HypotheSAEs default)

# VLM (InternVL2.5-8B via vLLM, offline)
VLM_MODEL = os.environ.get("HG_VLM_MODEL", "OpenGVLab/InternVL2_5-8B")
VLM_MAX_IMAGES_PER_PROMPT = max(N_HIGH_INTERP + N_LOW_INTERP, 1)
# With images downsized to one 448px tile each (max_dynamic_patch=1 in vlm_backend), ~256 image tokens
# per image; 16 images ~ 4k tokens + text -> 12288 leaves comfortable headroom.
VLM_MAX_MODEL_LEN = 12288


# ---------------------------------------------------------------------------------------------
# Cell enumeration + path resolution
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    """One trained SAE in the comparison grid."""
    encoder: str
    activation: str       # ActivationPair.name
    variant: str          # "baseline" | "dual"
    seed: int

    @property
    def sae_type(self) -> str:
        pair = ACTIVATION_PAIR_BY_NAME[self.activation]
        return pair.dual_trainer if self.variant == "dual" else pair.baseline_trainer

    @property
    def ckpt_path(self) -> str:
        root = _DUAL_ROOT if self.variant == "dual" else _BASE_ROOT
        return os.path.join(root, self.encoder, "trained_models", self.sae_type,
                            f"seed{self.seed}", CKPT_FILENAME)

    @property
    def tag(self) -> str:
        return f"{self.encoder}/{self.activation}/{self.variant}/seed{self.seed}"

    @property
    def acts_dir(self) -> str:
        return os.path.join(ACTS_DIR, self.encoder, self.activation, self.variant, f"seed{self.seed}")


ACTIVATION_PAIR_BY_NAME = {p.name: p for p in ACTIVATION_PAIRS}


def iter_cells(encoders: List[str] = None, activations: List[str] = None,
               variants: List[str] = None, seeds: List[int] = None) -> Iterator[Cell]:
    """Yield every Cell in the (optionally filtered) grid. Filters let stages run a slice."""
    encoders = encoders or ENCODERS
    activations = activations or [p.name for p in ACTIVATION_PAIRS]
    variants = variants or VARIANTS
    seeds = seeds or SEEDS
    for enc in encoders:
        for act in activations:
            for var in variants:
                for seed in seeds:
                    yield Cell(enc, act, var, seed)


def embeddings_path(encoder: str, split: str) -> str:
    """fp16 embeddings .npy for (encoder, split) — Stage 1a output, shared across that encoder's cells."""
    return os.path.join(EMB_DIR, encoder, f"{split}_emb.npy")


def manifest_path(split: str) -> str:
    """JSON image manifest for a split: ordered list of (rel_path, label). Encoder-independent."""
    return os.path.join(EMB_DIR, f"{split}_manifest.json")


def labels_path(split: str) -> str:
    """Class labels .npy aligned to the embeddings/manifest rows for a split."""
    return os.path.join(EMB_DIR, f"{split}_labels.npy")


def acts_path(cell: Cell, split: str) -> str:
    """Dense fp16 activations .npy (n_rows, dict_size) for a cell+split — Stage 1b output."""
    return os.path.join(cell.acts_dir, f"{split}_acts.npy")


def acts_meta_path(cell: Cell, split: str) -> str:
    return os.path.join(cell.acts_dir, f"{split}_acts.meta.json")


def load_labels(split: str):
    """Load a split's labels in the dataset's native shape: ImageNet -> (N,) int64 single-label;
    OpenImages -> (N, NUM_CLASSES) float32 multi-hot. Every stage loads labels through here so the
    single- vs multi-label distinction lives in ONE place."""
    import numpy as np
    arr = np.load(labels_path(split))
    return arr.astype(np.float32) if MULTILABEL else arr.astype(np.int64)


def onevsrest(labels, c: int):
    """Boolean one-vs-rest target mask (N,) 'is row positive for class c'. Multi-label -> column c is set;
    single-label -> label equals c. The ONLY place stages should test class membership."""
    return (labels[:, c] > 0) if MULTILABEL else (labels == c)


def present_classes(labels) -> List[int]:
    """Class ids with >=1 positive in `labels` (multi-hot column sum > 0, or unique single-labels)."""
    import numpy as np
    if MULTILABEL:
        return sorted(int(c) for c in np.where(labels.sum(axis=0) > 0)[0])
    return sorted(int(c) for c in np.unique(labels))


def faithful_classes() -> List[int]:
    """The class subset the faithful metric scores (deterministic: present val classes sampled with
    FAITHFUL_CLASS_SEED, capped at FAITHFUL_CLASSES). Mirrors evaluate._choose_faithful_classes but
    loads val labels itself, so non-eval stages (e.g. select restricting the interpret union to these
    classes) can reproduce the exact set without importing evaluate. Multi-label aware."""
    import numpy as np
    present = present_classes(load_labels("val"))
    n = min(FAITHFUL_CLASSES, len(present))
    if n < len(present):
        rng = np.random.default_rng(FAITHFUL_CLASS_SEED)
        return sorted(rng.choice(present, size=n, replace=False).tolist())
    return present
