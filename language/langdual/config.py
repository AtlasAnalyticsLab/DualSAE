"""Central registry for the language-model (Pythia residual-stream) experiments.

WHY a single registry: every stage (pack / extract / train / evaluate / aggregate) iterates the
SAME set of cells.  Keeping the enumeration AND the path resolution here means no two stages can
disagree about which checkpoint, which activation shard directory or which result JSON a cell
maps to.

Every cell is one SAE trained on the residual stream of ``EleutherAI/pythia-410m-deduped`` at
one block, on one training corpus, with one activation, one variant (standard SAE or DualSAE)
and one seed.  The on-disk layout below (directory names and cell tags) is a contract: the
LLM-steering component of this repository locates SAE checkpoints through it.

Run ``python -m langdual.config --audit`` to assert the cell counts and the path injectivity.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# --------------------------------------------------------------------------- roots
# One root holds every artefact of this component (downloads, token streams, activation shards,
# runs, results, reports).  The default, <repo>/language/outputs/language, is derived from this
# file's location (not from the working directory), so every stage -- including the training
# worker, which changes directory into the vision core, and the LLM-steering component, which
# reads the same tree -- resolves the same root.  Set LANGDUAL_ROOT to put the data elsewhere.
_DEFAULT_ROOT = str(Path(__file__).resolve().parents[1] / "outputs" / "language")
ROOT = os.path.abspath(os.environ.get("LANGDUAL_ROOT", _DEFAULT_ROOT))

RAW_DIR = os.path.join(ROOT, "raw")            # snapshot_download lands here (prepare_assets.sh)
CORPUS_DIR = os.path.join(ROOT, "corpora")     # text-embedding stage: canonical jsonl + labels npy
LMDB_DIR = os.path.join(ROOT, "lmdb")          # text-embedding stage: document-embedding LMDBs
SHARD_DIR = os.path.join(ROOT, "resid")        # fp16 memmap activation shards
TOKENS_DIR = os.path.join(ROOT, "tokens")      # packed uint16 token streams
RUNS_DIR = os.path.join(ROOT, "runs")          # one directory per trained main-grid cell
RESULT_DIR = os.path.join(ROOT, "results")     # evaluation JSONs (results/trackb/<tag>.json)
REPORT_DIR = os.path.join(ROOT, "reports")     # tables and qualitative pages
LOG_DIR = os.path.join(ROOT, "logs")

TB_ROOT = os.path.abspath(os.environ.get("LANGDUAL_TB_ROOT", os.path.join(ROOT, "tensorboard")))

# The SAE/DualSAE core lives in <repo>/vision and imports its own modules by BARE name
# (`from models import ...`), so it must be on sys.path (and is the working directory of the
# training worker).  Derived from this file's location, never hard-coded.
SAE_DIR = str(Path(__file__).resolve().parents[2] / "vision")

# --------------------------------------------------------------------------- protocol
N_LATENTS = 8192
TOP_K = 64
BATCH_SIZE = 2048
ADAM_EPS = 6.25e-10
BANDWIDTH = 1e-3
# Passed as --l1_coef to every cell.  It is the sparsity coefficient of the JumpReLU objective
# (both the JumpReLU baseline and a Dual-JumpReLU Sparse Branch); the TopK/BatchTopK families
# do not read it.
L1_COEF = 1e-4
POLY_LAYERS = 1          # Residual Branch = Linear(D,D)+GELU; --poly_hidden is inert on this path
SPAN_COEF = 0.01         # decorrelation coefficient gamma (code name: span_coef)
SEEDS = (42, 43, 44)

# Step budget.  Fixed in STEPS and expressed as --num_epochs 50 --steps_per_epoch 625, which
# also gives 50 validation-MonoScore selection points per cell.  31,250 x 2,048 rows is exactly
# one pass over the 64M-token training stream (LM_TRAIN_TOKENS below).
STEPS_PER_EPOCH = 625
NUM_EPOCHS = 50
TOTAL_STEPS = STEPS_PER_EPOCH * NUM_EPOCHS      # 31,250
WARMUP_STEPS = 2500      # DualSAE warm-up (Residual Branch frozen) for the main grid

# AuxK and gradient clipping are NOT applied uniformly by the trainers in <repo>/vision:
#   * train_dual.py (every Dual arm) has no auxiliary-TopK loss and does not clip gradients.
#   * gradient clipping is applied only in train_other.py (BatchTopK / JumpReLU baselines).
#   * dead-latent reinitialisation exists only in train_dual.py and train_topk.py, so
#     --dead_check_interval is not passed to the BatchTopK and JumpReLU baselines.
AUXK_COEF = 1.0 / 32.0          # baselines only, and only where the trainer implements it
MAX_GRAD_NORM = 1.0             # reaches only the train_other.py baselines
DEAD_STEPS_THRESHOLD = 10_000_000
DEAD_CHECK_INTERVAL = 100       # DualSAE + TopK only

ACTIVATIONS = ("topk", "batch_topk", "jump_relu")
VARIANTS = ("baseline", "dual")

# The comparisons are always within an activation pair: a standard SAE is paired only with the
# DualSAE whose Sparse Branch uses the same activation.
ACTIVATION_PAIRS = (
    ("topk", "TopK", "Dual-TopK"),
    ("batch_topk", "BatchTopK", "Dual-BatchTopK"),
    ("jump_relu", "JumpReLU", "Dual-JumpReLU"),
)

# Learning rate per ACTIVATION (both arms of a pair always share it): JumpReLU runs used 5e-5,
# TopK and BatchTopK 1e-4.
LR: Dict[str, float] = {"topk": 1e-4, "batch_topk": 1e-4, "jump_relu": 5e-5}

# JumpReLU threshold initialisation (theta, stored by the model as log(theta)) used by every
# reported language JumpReLU cell, baseline and Dual alike.  Passed EXPLICITLY on the command
# line because the vision core's default is different.  With this init and the residual-stream
# activation scale, the learned thresholds stay close to 1e-3 (see README, Implementation notes).
JUMP_THRESHOLD_INIT = "0.001"

# The residual stream is not unit-norm data, so per-sample input standardisation
# (models/base.py input_unit_norm) is OFF for every cell: it would destroy the norm signal and
# make the reconstruction un-spliceable back into the model.  A single fixed global scale per
# (corpus, layer) is folded into the shards instead (lmsae/extract_resid.py).
TRACKB_NORMALIZE = False


# --------------------------------------------------------------------------- model / corpora
LM_REPO = "EleutherAI/pythia-410m-deduped"
LM_LAYERS = (6, 12, 18)         # of 24; hidden size 1024
LM_D_MODEL = 1024
LM_CTX_LEN = 128                # tokens per packed context
LM_TRAIN_TOKENS = 64_000_000    # == TOTAL_STEPS * BATCH_SIZE, i.e. exactly one pass
LM_SHARD_TOKENS = 1_000_000     # tokens per fp16 memmap shard (rounded down to whole contexts)


@dataclass(frozen=True)
class LMCorpus:
    """One TRAINING corpus: the text the SAE's activations are drawn from."""
    name: str                 # SHORT key; it goes into Cell.tag
    repo: str
    shard_template: str       # repo-relative parquet path, formatted with {idx}
    total_shards: int
    text_column: str          # OpenWebText calls it "text"; the code corpus calls it "code"
    note: str = ""


# "owt" keeps the bare legacy layout (resid/L<n>/<split>, tokens/<split>.u16.npy) and is first;
# any other corpus is namespaced (resid/<corpus>/L<n>/<split>, tokens/<corpus>_<split>.u16.npy).
# Corpus is the OUTERMOST loop of iter_cells_b so adding a corpus APPENDS cells.
LM_CORPORA: Dict[str, LMCorpus] = {
    c.name: c for c in (
        LMCorpus("owt", "Skylion007/openwebtext",
                 "plain_text/train-{idx:05d}-of-00080.parquet", 80, "text",
                 note="general web text"),
        # Mixed-language GitHub source code (the Pile, Pythia's training data, contains a GitHub
        # subset).  The shards are not sorted by programming language, so a prefix of a shard
        # is a mixed sample.
        LMCorpus("ghcode", "codeparrot/github-code-clean",
                 "data/train-{idx:05d}-of-00880.parquet", 880, "code",
                 note="mixed-language GitHub source code"),
    )
}
LM_CORPUS_ORDER: Tuple[str, ...] = ("owt", "ghcode")

# MonoScore caches the sparse code DENSELY (8192 * 4 B = 32 KB per row), so the scoring set is
# capped; the cap is identical for every cell.
TRACKB_SCORE_TOKENS = 200_000


# --------------------------------------------------------------------------- text-embedding stage
# Labelled corpora and their bge-large document embeddings.  No SAE in this component is trained
# on them: they are produced here because the LLM-steering component reads them (the materialised
# AG News / GoEmotions corpora and their bge-large-en-v1.5 document-embedding LMDBs).
@dataclass(frozen=True)
class Corpus:
    name: str
    repo: str
    config: Optional[str]
    n_train_raw: int
    n_val_carve: int          # 0 when the corpus ships its own validation split
    n_test: int
    n_classes: int
    multilabel: bool
    text_fields: Tuple[str, ...]
    label_field: str

    @property
    def n_train(self) -> int:
        return self.n_train_raw - self.n_val_carve

    @property
    def n_val(self) -> int:
        # goemotions ships validation; the others carve it out of train.
        return self.n_val_carve if self.n_val_carve else 5426

    def steps_per_true_epoch(self, batch_size: int = BATCH_SIZE) -> int:
        """Steps in ONE pass over train (drop_last=True, hence floor())."""
        return self.n_train // batch_size


CORPORA: Dict[str, Corpus] = {
    c.name: c for c in (
        Corpus("agnews", "fancyzhx/ag_news", None,
               120_000, 7_600, 7_600, 4, False, ("text",), "label"),
        # GoEmotions 'simplified' is the MULTI-LABEL config and ships a validation split, so
        # nothing is carved from its train.
        Corpus("goemotions", "google-research-datasets/go_emotions", "simplified",
               43_410, 0, 5_427, 28, True, ("text",), "labels"),
    )
}
CORPUS_ORDER = ("agnews", "goemotions")

# The stratified validation carve is deterministic (this seed) and written to
# split_indices.json, which later runs reuse.
SPLIT_SEED = 42


@dataclass(frozen=True)
class TextEncoder:
    name: str
    repo: str
    dim: int
    max_seq_length: int
    prefix: str               # BGE's published prefix is QUERY-side only: documents take none

    @property
    def already_l2_normalized(self) -> bool:
        """The sentence-transformers module chain ends in a Normalize module, so the stored
        embeddings are unit-norm by construction."""
        return True


TEXT_ENCODERS: Dict[str, TextEncoder] = {
    "bge": TextEncoder("bge", "BAAI/bge-large-en-v1.5", 1024, 512, ""),
}
ENCODER_ORDER = ("bge",)

# fp32 weights + TF32 matmuls; recorded in each LMDB because it changes the embeddings.
ENCODE_DTYPE = "fp32-tf32"


# --------------------------------------------------------------------------- cells
@dataclass(frozen=True)
class Cell:
    """One trained SAE of the main grid.  `key` is the layer tag, e.g. "L12"."""
    track: str                # always "B" (the tag prefix is kept for layout compatibility)
    key: str                  # "L6" | "L12" | "L18"
    corpus: str               # a key of LM_CORPORA
    activation: str
    variant: str
    seed: int

    @property
    def tag(self) -> str:
        """Globally unique, flat cell name, e.g. ``B_L12_owt_topk_dual_seed42``.

        MUST be unique across the whole component: the evaluator names its output file from the
        run directory's basename alone, so a nested layout with a repeated leaf would let cells
        overwrite each other's results.
        """
        return f"{self.track}_{self.key}_{self.corpus}_{self.activation}_{self.variant}_seed{self.seed}"

    @property
    def layer(self) -> int:
        return int(self.key[1:])

    @property
    def run_dir(self) -> str:
        return os.path.join(RUNS_DIR, self.tag)

    @property
    def model_type(self) -> str:
        """--model for <repo>/vision/main.py."""
        if self.variant == "dual":
            return "dual"
        return {"topk": "topk", "batch_topk": "batch_topk", "jump_relu": "jumprelu"}[self.activation]

    @property
    def normalize(self) -> bool:
        return TRACKB_NORMALIZE

    @property
    def lr(self) -> float:
        return LR[self.activation]

    @property
    def multilabel(self) -> bool:
        return False

    @property
    def tb_dir(self) -> str:
        return os.path.join(TB_ROOT, f"track{self.track}", self.tag)

    @property
    def done_sentinel(self) -> str:
        return os.path.join(self.run_dir, "CELL_DONE.json")


def iter_cells_b() -> Iterator[Cell]:
    """corpora x 3 layers x 3 activations x 2 variants x 3 seeds = 108 cells.

    Corpus is the OUTERMOST loop, then layer, activation, variant, seed.
    """
    for corpus in LM_CORPUS_ORDER:
        for layer in LM_LAYERS:
            for act in ACTIVATIONS:
                for var in VARIANTS:
                    for seed in SEEDS:
                        yield Cell("B", f"L{layer}", corpus, act, var, seed)


def iter_cells() -> Iterator[Cell]:
    yield from iter_cells_b()


# --------------------------------------------------------------------------- paths
def resid_dir(layer: int, split: str, corpus: str = "owt") -> str:
    """fp16 memmap shard directory for one (layer, split, corpus).

    "owt" uses the bare layout resid/L<n>/<split>; other corpora resid/<corpus>/L<n>/<split>.
    Namespacing by DIRECTORY rather than by split name is deliberate: extract_resid inherits a
    held-out split's norm factor from the manifest of the split literally named "train", so a
    prefixed split name would silently fold a new corpus by OpenWebText's scale.
    """
    if corpus == "owt":
        return os.path.join(SHARD_DIR, f"L{layer}", split)
    return os.path.join(SHARD_DIR, corpus, f"L{layer}", split)


def token_stream(split: str) -> str:
    """Packed uint16 token stream (no padding): tokens/<split>.u16.npy."""
    return os.path.join(TOKENS_DIR, f"{split}.u16.npy")


def lm_token_stream(corpus: str, split: str) -> str:
    """Packed token stream for one training corpus ("owt" -> bare name, else <corpus>_<split>)."""
    return token_stream(split if corpus == "owt" else f"{corpus}_{split}")


def lmdb_path(encoder: str, corpus: str, split: str) -> str:
    """Document-embedding LMDB: lmdb/<encoder>/<corpus>/<split>.lmdb."""
    return os.path.join(LMDB_DIR, encoder, corpus, f"{split}.lmdb")


def labels_lmdb_path(corpus: str, split: str) -> str:
    """Companion multi-hot label store of a MULTI-LABEL corpus (shared by every encoder)."""
    return os.path.join(LMDB_DIR, "_labels", corpus, f"{split}_labels.lmdb")


def corpus_jsonl(corpus: str, split: str) -> str:
    return os.path.join(CORPUS_DIR, corpus, f"{split}.jsonl")


def corpus_labels_npy(corpus: str, split: str) -> str:
    return os.path.join(CORPUS_DIR, corpus, f"{split}_labels.npy")


def split_indices_json(corpus: str) -> str:
    return os.path.join(CORPUS_DIR, corpus, "split_indices.json")


def trackb_result(cell: Cell) -> str:
    return os.path.join(RESULT_DIR, "trackb", f"{cell.tag}.json")


_FORBIDDEN = ("imagenet", "ilsvrc", "open_image", "openimages")


def assert_safe_text_path(path: str) -> None:
    """Refuse any artefact path that would trip the vision label fallbacks.

    The vision core's LMDB dataset (dataset/lmdb_features.py) selects ImageNet label files for
    any path containing 'imagenet'/'ilsvrc' and MULTI-LABEL mode for any path containing
    'open_image'/'openimages'.  The activation-shard reader used here does not go through that
    code, but a ROOT under such a directory would still mislabel any LMDB-path consumer, so the
    condition is asserted at write time rather than trusted.
    """
    low = str(path).lower()
    for bad in _FORBIDDEN:
        if bad in low:
            raise ValueError(
                f"artefact path must not contain {bad!r} (it would trigger the vision label "
                f"fallback in dataset/lmdb_features.py): {path}")


EVAL_SPLIT = "test"   # val is used only for best-checkpoint selection


# --------------------------------------------------------------------------- audit
def _audit() -> int:
    b = list(iter_cells_b())
    problems: List[str] = []
    expect_b = len(LM_CORPUS_ORDER) * len(LM_LAYERS) * len(ACTIVATIONS) * len(VARIANTS) * len(SEEDS)
    if len(b) != expect_b:
        problems.append(f"{len(b)} cells, expected {expect_b}")
    if LM_CORPUS_ORDER[0] != "owt" or set(LM_CORPUS_ORDER) != set(LM_CORPORA):
        problems.append("LM_CORPUS_ORDER must start with 'owt' and match LM_CORPORA")
    tags = [c.tag for c in b]
    if len(set(tags)) != len(tags):
        problems.append("tag is not injective")
    if len({c.run_dir for c in b}) != len(b):
        problems.append("run_dir is not injective")
    assert_safe_text_path(ROOT)
    print(f"cells         : {len(b)}")
    print(f"step budget   : {TOTAL_STEPS} steps @ batch {BATCH_SIZE}, warm-up {WARMUP_STEPS}")
    for act in ACTIVATIONS:
        print(f"  {act:11s} lr={LR[act]:g}")
    print(f"root          : {ROOT}")
    if problems:
        print("\nFAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nAUDIT OK")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", action="store_true", help="assert cell counts and path injectivity")
    ap.parse_args()
    raise SystemExit(_audit())
