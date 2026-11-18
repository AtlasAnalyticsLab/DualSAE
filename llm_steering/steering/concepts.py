"""The 10 pre-registered steering concepts (D-45, PROTOCOL 4.2) and their concept data.

Concept data (D-45 / C31): per concept class, 2,000 TRAIN documents for latent selection and 1,000
TEST documents for held-out detection AUROC, sampled with a fixed seed (after the amazon shuffle),
pythia-tokenised and truncated to 128 tokens (the SAE training window).

Per corpus ("concept vs rest" is always within the concept's own corpus):
  amazon      canrager/amazon_reviews_mcauley_1and5 parquet (read directly with pyarrow; the
              ``datasets`` package is not used).  Both splits are CLASS-SORTED
              (train: 54 rating runs, first 10k rows all 5-star), so each split is first permuted
              with a fixed seed and the per-class samples are the first N rows of each class in
              that permuted order.  positive = rating 5.0, negative = rating 1.0 (the only two
              ratings present).  "rest" of positive = negative and vice versa.
  agnews      corpora/agnews/{train,test}.jsonl + labels (0 World, 1 Sports, 2 Business,
              3 Sci/Tech; 28,100 / 1,900 per class).  rest = the other three classes.
  goemotions  corpora/goemotions (multi-label, 28 classes).  Only SINGLE-LABEL rows are used.
              This is a DECLARED READING (D-45 names the four emotions and nothing about
              multi-label rows): a multi-label row (e.g. joy + sadness) has no single concept
              label for "concept vs rest", and ~15% of rows are multi-label.  Declared reading of
              "rest": the other three concept emotions PLUS an 'other' pool of single-label rows
              of the remaining 24 classes (incl. neutral), sampled like a concept class.  Without
              the pool, "joy vs rest" would be joy vs {anger, sadness, fear} only, i.e. a valence
              detector.  Counts are capped by availability (declared): single-label train rows are
              joy 853 / anger 1,025 / sadness 817 / fear 430 (< 2,000) and test rows 93 / 131 /
              102 / 65 (< 1,000); every count used is recorded in the npz.

Judge concept descriptions (the AxBench ``{concept}`` slot, D-50) are fixed here, before any run,
in the style of AxBench's concept strings ("references to ...").  They are part of the
pre-registered protocol (concept_registry_sha256() is recorded in every s7 output).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common import lang_config as config
from steering import base
from common import seeds as _seeds

# Amazon reviews (positive = 5-star, negative = 1-star), pinned revision of the hub dataset;
# prepare_assets.sh downloads it to <LANGDUAL_ROOT>/raw/datasets/<org>__<name>/data.
AMAZON_REPO = "canrager/amazon_reviews_mcauley_1and5"
AMAZON_REVISION = "2dc1eabedc91e72f3dfc8c3476857ea7aabe52a0"
AMAZON_DIR = os.environ.get("A16_AMAZON_DIR",
                            os.path.join(config.local_snapshot(AMAZON_REPO, "dataset"), "data"))
N_TRAIN_PER_CLASS = 2000
N_TEST_PER_CLASS = 1000
GOEMO_NAMES = ("admiration amusement anger annoyance approval caring confusion curiosity desire "
               "disappointment disapproval disgust embarrassment excitement fear gratitude grief "
               "joy love nervousness optimism pride realization relief remorse sadness surprise "
               "neutral").split()


@dataclass(frozen=True)
class Concept:
    cid: str             # stable id used in every file and seed key
    corpus: str          # amazon | agnews | goemotions
    cls: str             # class name inside the corpus (the npz 'classes' entry)
    label: str           # PROTOCOL 4.2 wording
    description: str     # AxBench {concept} string given to the judge (pre-registered)


CONCEPTS: Tuple[Concept, ...] = (
    Concept("amazon_positive", "amazon", "positive", "amazon positive (5-star)",
            "positive sentiment about a product, such as praise, satisfaction or a recommendation"),
    Concept("amazon_negative", "amazon", "negative", "amazon negative (1-star)",
            "negative sentiment about a product, such as complaints, disappointment or a warning "
            "not to buy it"),
    Concept("agnews_world", "agnews", "World", "agnews World",
            "world news and international affairs, such as governments, diplomacy, elections and "
            "armed conflicts"),
    Concept("agnews_sports", "agnews", "Sports", "agnews Sports",
            "sports, such as games, teams, athletes, matches and tournaments"),
    Concept("agnews_business", "agnews", "Business", "agnews Business",
            "business and the economy, such as companies, markets, stocks, earnings and trade"),
    Concept("agnews_scitech", "agnews", "Sci/Tech", "agnews Sci/Tech",
            "science and technology, such as computers, software, the internet, research and "
            "space exploration"),
    Concept("goemotions_joy", "goemotions", "joy", "goemotions joy",
            "expressions of joy or happiness"),
    Concept("goemotions_anger", "goemotions", "anger", "goemotions anger",
            "expressions of anger or irritation"),
    Concept("goemotions_sadness", "goemotions", "sadness", "goemotions sadness",
            "expressions of sadness or grief"),
    Concept("goemotions_fear", "goemotions", "fear", "goemotions fear",
            "expressions of fear or worry"),
)
CONCEPT_BY_ID: Dict[str, Concept] = {c.cid: c for c in CONCEPTS}
CORPORA: Tuple[str, ...] = ("amazon", "agnews", "goemotions")
# Classes stored per corpus (index = the npz 'cls' code).  goemotions carries the 'other' pool.
CORPUS_CLASSES: Dict[str, Tuple[str, ...]] = {
    "amazon": ("positive", "negative"),
    "agnews": ("World", "Sports", "Business", "Sci/Tech"),
    "goemotions": ("joy", "anger", "sadness", "fear", "other"),
}

# Words that may not appear in a neutral prompt (D-49 "free of concept words"): the class names,
# every content word of the judge descriptions, and obvious synonyms.  Checked case-insensitively
# on word boundaries by s3_prompts.py.
BANNED_PROMPT_WORDS: Tuple[str, ...] = tuple(sorted(set("""
positive negative praise satisfaction satisfied recommend recommendation complaint complaints
disappointment disappointed warning buy bought purchase product products review reviews star stars
rating amazon world international affairs government governments diplomacy election elections
war conflict conflicts military politics political president sport sports game games team teams
athlete athletes match matches tournament tournaments score scored win won player players
business economy economic company companies market markets stock stocks earnings trade finance
financial money price science scientific technology tech computer computers software internet
research space rocket joy joyful happy happiness glad delighted anger angry irritation irritated
furious mad sad sadness grief grieve cry crying fear afraid scared worry worried anxious anxiety
terrified love hate great terrible awful wonderful excellent bad good best worst news
""".split())))


def concept_registry_sha256() -> str:
    """Fingerprint of the concept registry (ids, classes, judge descriptions)."""
    blob = json.dumps([c.__dict__ for c in CONCEPTS], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


# ----------------------------------------------------------------------------- source readers
def _amazon_split(split: str) -> Tuple[List[str], np.ndarray]:
    """(texts, ratings) of one amazon split, in file order.  Needs pyarrow."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - environment problem, explained loudly
        raise ImportError("pyarrow is required to read the amazon parquet "
                          "(pip install pyarrow)") from e
    path = os.path.join(AMAZON_DIR, f"{split}-00000-of-00001.parquet")
    t = pq.read_table(path, columns=["text", "rating"])
    return t.column("text").to_pylist(), np.asarray(t.column("rating").to_numpy(), dtype=np.float64)


def _jsonl_texts(corpus: str, split: str) -> List[str]:
    with open(config.corpus_jsonl(corpus, split)) as fh:
        return [json.loads(line)["text"] for line in fh if line.strip()]


def _sample_class(idx: np.ndarray, n: int, key: str) -> np.ndarray:
    """``n`` rows of ``idx`` (or all when fewer), in a seeded random order (D-22 seed)."""
    perm = _seeds.rng(_seeds.derive_seed("a16.concepts", key)).permutation(idx.size)
    return idx[perm[: min(n, idx.size)]]


def class_rows(corpus: str, split: str, n_per_class: int) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """(texts, class codes, source row indices) of the sampled rows of one corpus split.

    Rows are ordered class by class; within a class in the seeded order.  Empty texts are
    skipped BEFORE sampling (they contribute no tokens)."""
    classes = CORPUS_CLASSES[corpus]
    texts_out: List[str] = []
    cls_out: List[int] = []
    src_out: List[int] = []
    if corpus == "amazon":
        texts, rating = _amazon_split(split)
        # The whole split is permuted first (class-sorted source), then each class takes its
        # first n rows in permuted order: "sampled with a fixed seed after the amazon shuffle".
        order = _seeds.rng(_seeds.derive_seed("a16.amazon.shuffle", split)).permutation(len(texts))
        for ci, (name, r) in enumerate((("positive", 5.0), ("negative", 1.0))):
            rows = order[(rating[order] == r) & np.array([bool(texts[i] and texts[i].strip())
                                                         for i in order])]
            rows = rows[:n_per_class]
            texts_out += [texts[i] for i in rows]
            cls_out += [ci] * rows.size
            src_out += rows.tolist()
        return texts_out, np.asarray(cls_out, np.int8), np.asarray(src_out, np.int64)
    texts = _jsonl_texts(corpus, split)
    y = np.load(config.corpus_labels_npy(corpus, split))
    nonempty = np.array([bool(t and t.strip()) for t in texts])
    if corpus == "agnews":
        for ci in range(len(classes)):
            idx = np.flatnonzero((y == ci) & nonempty)
            rows = _sample_class(idx, n_per_class, f"agnews|{split}|{ci}")
            texts_out += [texts[i] for i in rows]
            cls_out += [ci] * rows.size
            src_out += rows.tolist()
    elif corpus == "goemotions":
        single = (y.sum(axis=1) == 1) & nonempty
        for ci, name in enumerate(classes):
            if name == "other":
                others = [GOEMO_NAMES.index(n) for n in GOEMO_NAMES
                          if n not in classes[:-1]]
                idx = np.flatnonzero(single & (y[:, others].sum(axis=1) == 1))
            else:
                idx = np.flatnonzero(single & (y[:, GOEMO_NAMES.index(name)] == 1))
            rows = _sample_class(idx, n_per_class, f"goemotions|{split}|{name}")
            texts_out += [texts[i] for i in rows]
            cls_out += [ci] * rows.size
            src_out += rows.tolist()
    else:
        raise KeyError(corpus)
    return texts_out, np.asarray(cls_out, np.int8), np.asarray(src_out, np.int64)


def load_tokenizer():
    from transformers import AutoTokenizer
    from common.lm_model import resolve_model_path
    tok = AutoTokenizer.from_pretrained(resolve_model_path(None), local_files_only=True)
    return tok


def build_concept_docs(corpus: str, n_train: int = N_TRAIN_PER_CLASS,
                       n_test: int = N_TEST_PER_CLASS, tok=None) -> Dict[str, np.ndarray]:
    """Tokenise the sampled train+test documents of one corpus (<= 128 tokens, no special tokens:
    Track B windows are packed without BOS, so position 0 is the sink whatever the token)."""
    tok = tok or load_tokenizer()
    ids_all, lens, cls_all, split_all, src_all = [], [], [], [], []
    n_short = 0
    for si, (split, n) in enumerate((("train", n_train), ("test", n_test))):
        texts, cls, src = class_rows(corpus, split, n)
        enc = tok(texts, add_special_tokens=False, truncation=True, max_length=base.CTX_LEN)["input_ids"]
        for t, c, s in zip(enc, cls, src):
            if len(t) < 2:
                n_short += 1          # no position >= 1 to pool over
                continue
            row = np.zeros(base.CTX_LEN, np.uint16)
            row[: len(t)] = t
            ids_all.append(row)
            lens.append(len(t))
            cls_all.append(c)
            split_all.append(si)
            src_all.append(s)
    out = {"ids": np.stack(ids_all), "lens": np.asarray(lens, np.int16),
           "cls": np.asarray(cls_all, np.int8), "split": np.asarray(split_all, np.int8),
           "src": np.asarray(src_all, np.int64),
           "classes": np.asarray(CORPUS_CLASSES[corpus]),
           "n_dropped_short": np.asarray(n_short),
           "n_train_per_class": np.asarray(n_train), "n_test_per_class": np.asarray(n_test)}
    return out


def ensure_concept_docs(corpus: str, n_train: int = N_TRAIN_PER_CLASS,
                        n_test: int = N_TEST_PER_CLASS) -> Dict[str, np.ndarray]:
    """Load the tokenised docs of one corpus, building them once (atomic write) when absent.

    Concurrent s1 tasks may race to build the same file: the content is deterministic and
    the write is atomic, so the loser simply replaces an identical file.  A file built with other
    per-class caps (e.g. a smoke) is rebuilt, never silently reused."""
    path = base.s1_docs(corpus)
    if os.path.isfile(path):
        d = dict(np.load(path, allow_pickle=False))
        if int(d["n_train_per_class"]) == n_train and int(d["n_test_per_class"]) == n_test:
            return d
    d = build_concept_docs(corpus, n_train, n_test)
    base.atomic_save_npz(path, **d)
    return d


def concept_masks(docs: Dict[str, np.ndarray], concept: Concept, split: int) -> Tuple[np.ndarray, np.ndarray]:
    """(positive mask, negative mask) over the docs of the concept's corpus for one split:
    positives = the concept class, negatives = every other class of the corpus (incl. 'other')."""
    ci = list(docs["classes"]).index(concept.cls)
    in_split = docs["split"] == split
    return in_split & (docs["cls"] == ci), in_split & (docs["cls"] != ci)


def classifier_rows(corpus: str, split: str, n_per_class_amazon: int = 10000):
    """Rows for the bge concept classifier (D-50).  agnews / goemotions: every row of the split's
    Track A bge LMDB (goemotions: single-label rows only, labels as class names); amazon: the
    first ``n_per_class_amazon`` rows of each class in the shuffled order (texts to be encoded).
    Returns (X or None, texts or None, class-name array)."""
    from common import sae_io
    if corpus == "amazon":
        texts, cls, _ = class_rows("amazon", split, n_per_class_amazon)
        names = np.asarray(CORPUS_CLASSES["amazon"])[cls]
        return None, texts, names
    y = np.load(config.corpus_labels_npy(corpus, split))
    if corpus == "agnews":
        X = sae_io.read_lmdb_rows("bge", "agnews", split)          # L2-normalised on read
        return X, None, np.asarray(CORPUS_CLASSES["agnews"])[y]
    single = np.flatnonzero(y.sum(axis=1) == 1)
    X = sae_io.read_lmdb_rows("bge", "goemotions", split, indices=single)
    lab = np.asarray(GOEMO_NAMES)[y[single].argmax(axis=1)]
    return X, None, lab
