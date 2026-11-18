"""Paths and protocol constants of the language data tree that the steering pipeline reads.

The steering stages consume artefacts produced by the language pipeline (``language/`` of this
repository): trained SAE run directories, OpenWebText residual shards at the steered blocks,
the packed OpenWebText token stream, the materialised AG News / GoEmotions corpora and their
bge-large document-embedding LMDBs, plus local model snapshots.  This module is the ONE place
that knows where each of them lives, so every stage resolves paths identically.  It is a
subset of the language pipeline's own configuration with the same on-disk layout:

    $LANGDUAL_ROOT/
        raw/models/<org>__<name>/          local model snapshots (pythia, bge)
        raw/datasets/<org>__<name>/        raw dataset snapshots
        corpora/<corpus>/<split>.jsonl     canonical documents (+ <split>_labels.npy)
        lmdb/<encoder>/<corpus>/<split>.lmdb   document embeddings (Track A protocol)
        resid/L<n>/<split>/                fp16 residual shards + manifest.json (OpenWebText)
        tokens/<split>.u16.npy             packed uint16 OpenWebText token stream
        runs/<tag>/                        trained SAEs of the main language grid
        ablation/runs/<tag>/               trained SAEs of the language ablation grid

Every root is an environment variable shared with the language pipeline, with a default
inside this repository, so nothing here is tied to a particular machine:

    LANGDUAL_ROOT      data root of the language pipeline
                       (default <repo>/language/outputs/language)
    LANGDUAL_ABL_ROOT  root of the language ablation grid (default $LANGDUAL_ROOT/ablation);
                       its run directories are $LANGDUAL_ABL_ROOT/runs/<tag>
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

# --------------------------------------------------------------------------- roots
# <repo>/language/outputs/language: where the language pipeline writes when run from language/.
_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "language" / "outputs" / "language"
ROOT = os.path.abspath(os.environ.get("LANGDUAL_ROOT", str(_DEFAULT_ROOT)))

RAW_DIR = os.path.join(ROOT, "raw")            # local snapshots (prepare_assets.sh)
CORPUS_DIR = os.path.join(ROOT, "corpora")     # canonical jsonl + labels npy
LMDB_DIR = os.path.join(ROOT, "lmdb")          # document-embedding LMDBs
SHARD_DIR = os.path.join(ROOT, "resid")        # fp16 memmap activation shards
TOKENS_DIR = os.path.join(ROOT, "tokens")      # packed uint16 token streams
RUNS_DIR = os.path.join(ROOT, "runs")          # main-grid run directories
ABL_ROOT = os.path.abspath(os.environ.get("LANGDUAL_ABL_ROOT", os.path.join(ROOT, "ablation")))
ABL_RUNS_DIR = os.path.join(ABL_ROOT, "runs")  # ablation-grid run directories

# --------------------------------------------------------------------------- protocol
# Constants of the language SAE grid that the steering stages rely on (they are properties of
# the trained checkpoints, not knobs of this pipeline).
N_LATENTS = 8192
TOP_K = 64
BATCH_SIZE = 2048
SEEDS = (42, 43, 44)
STEPS_PER_EPOCH = 625
NUM_EPOCHS = 50
TOTAL_STEPS = STEPS_PER_EPOCH * NUM_EPOCHS      # 31,250 optimizer steps per SAE
ACTIVATIONS = ("topk", "batch_topk", "jump_relu")
# The residual-stream SAEs are trained WITHOUT per-sample input standardisation; inputs are
# divided by one global norm factor per (layer, corpus) instead (see the shard manifest).
TRACKB_NORMALIZE = False


@dataclass(frozen=True)
class TextEncoder:
    name: str
    repo: str
    dim: int
    max_seq_length: int
    prefix: str               # BGE's published prefix is query-side only, so documents get ""


TEXT_ENCODERS: Dict[str, TextEncoder] = {
    "bge": TextEncoder("bge", "BAAI/bge-large-en-v1.5", 1024, 512, ""),
}

# fp32 weights + TF32 matmuls: the encode protocol of the document-embedding LMDBs that the
# concept classifier is trained on, so continuations are embedded exactly like those rows.
ENCODE_DTYPE = "fp32-tf32"

# --------------------------------------------------------------------------- language model
LM_REPO = "EleutherAI/pythia-410m-deduped"
LM_LAYERS = (6, 12, 18)         # of 24 blocks
# Hidden size of the model whose residual shards are read.  Overridable ONLY so the synthetic
# smoke test can use a tiny stand-in model; real runs use pythia-410m-deduped (1024).
LM_D_MODEL = int(os.environ.get("LANG_LM_D_MODEL", "1024"))
LM_CTX_LEN = 128                # context length of the packed windows the SAEs were trained on


# --------------------------------------------------------------------------- paths
def local_snapshot(repo: str, repo_type: str = "model") -> str:
    """Local snapshot directory of a hub repo id: <RAW_DIR>/<models|datasets>/<org>__<name>.

    Pure path arithmetic ('/' -> '__' keeps one flat level per type), shared with the
    language pipeline's downloader so both resolve the same directory."""
    if repo_type not in ("model", "dataset"):
        raise ValueError(f"repo_type must be 'model' or 'dataset', got {repo_type!r}")
    sub = "models" if repo_type == "model" else "datasets"
    return os.path.join(RAW_DIR, sub, repo.replace("/", "__"))


def lmdb_path(encoder: str, corpus: str, split: str) -> str:
    """Document-embedding LMDB of one (encoder, corpus, split)."""
    return os.path.join(LMDB_DIR, encoder, corpus, f"{split}.lmdb")


def corpus_jsonl(corpus: str, split: str) -> str:
    return os.path.join(CORPUS_DIR, corpus, f"{split}.jsonl")


def corpus_labels_npy(corpus: str, split: str) -> str:
    return os.path.join(CORPUS_DIR, corpus, f"{split}_labels.npy")


def resid_dir(layer: int, split: str, corpus: str = "owt") -> str:
    """fp16 memmap shard directory of one (layer, split, corpus).

    OpenWebText ("owt") has no corpus segment in the path; any other corpus is namespaced by
    directory (never by split name, because a non-train split inherits its norm factor from the
    manifest of the split literally named "train")."""
    if corpus == "owt":
        return os.path.join(SHARD_DIR, f"L{layer}", split)
    return os.path.join(SHARD_DIR, corpus, f"L{layer}", split)


def token_stream(split: str) -> str:
    """Packed uint16 token stream (no padding), cut into LM_CTX_LEN windows by its readers."""
    return os.path.join(TOKENS_DIR, f"{split}.u16.npy")


def lm_token_stream(corpus: str, split: str) -> str:
    """Packed token stream of one residual-stream training corpus ("owt" = the bare name)."""
    return token_stream(split if corpus == "owt" else f"{corpus}_{split}")


_FORBIDDEN = ("imagenet", "ilsvrc", "open_image", "openimages")


def assert_safe_text_path(path: str) -> None:
    """Refuse a text artefact path that would trip the vision label fallbacks.

    The shared LMDB reader (vision/dataset/lmdb_features.py) switches to ImageNet label files
    for any path containing 'imagenet'/'ilsvrc' and to multi-label mode for 'open_image(s)';
    either would silently corrupt a text corpus's labels, so it is asserted here."""
    low = str(path).lower()
    for bad in _FORBIDDEN:
        if bad in low:
            raise ValueError(f"text artefact path must not contain {bad!r} (it would trigger "
                             f"the vision label fallback in the LMDB reader): {path}")
