"""Text-embedding stage: frozen sentence encoder (bge-large-en-v1.5) -> document-embedding LMDBs.

One invocation handles ONE (encoder, corpus) pair and writes every requested split, e.g.

    python -m langdual.extract_text_embeddings --encoder bge --corpus agnews
    python -m langdual.extract_text_embeddings --encoder bge --corpus agnews --verify

The outputs are read by the LLM-steering component (concept classifier and concept data); no
SAE in this component is trained on them.

The LMDB schema is the one the vision core's ``LMDBFeatureDataset.__init__`` / ``__getitem__``
read (a text corpus never triggers its ImageNet / OpenImages label fallbacks, which are keyed on
path substrings):

    b"__len__"            ASCII int, number of rows
    b"__shape__"          a string _parse_feat_shape accepts, e.g. "(768,)"
    b"<idx>"              raw float32 bytes, C-contiguous, exactly 4*D bytes
    b"__label__<idx>"     ASCII int class id -- SINGLE-LABEL corpora only, because
                          ``has_labels`` is probed as ``txn.get(b'__label__0') is not None``
                          and a present scalar label would shadow the companion multi-hot store
    b"__path__<idx>"      UTF-8 document id, so retrieval / qualitative output is inspectable

plus two self-describing markers (read by the vision core's ``dataset/lmdb_features.py``):

    b"__multilabel__"     "1"/"0"
    b"__labels_lmdb__"    ABSOLUTE path of the companion multi-hot store (multi-label only)

The absolute path matters: the companion store is written ONCE per (corpus, split) under
``config.labels_lmdb_path`` and shared by every encoder, so it is NOT a sibling of the feature
LMDB.

Storage convention: embeddings are stored in the encoder's NATIVE scale with
``normalize_embeddings=False``, exactly as the vision features were stored, because
``LMDBFeatureDataset.__getitem__`` L2-normalizes on read.  Normalizing at write time as well
would be a harmless no-op numerically but would destroy the ability to inspect raw norms.

Known limitation (declared, not accidental): documents longer than the encoder's
``max_seq_length`` are truncated by the sentence-transformers Transformer module.  The fraction
of truncated documents is measured per split and stored in the LMDB as provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:  # normal case: run as `python -m langdual.extract_text_embeddings`
    from langdual import config
except ImportError:  # run as a bare script from inside langdual/
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from langdual import config


# --------------------------------------------------------------------------- split registry
# The three splits every labelled corpus exposes.  "val" is selection-only; config.EVAL_SPLIT
# pins "test" as the reported split.
SPLITS: Tuple[str, ...] = ("train", "val", "test")

# Expected row counts, straight off the Corpus dataclass.  Used as a WARNING, never a hard
# assert: config.Corpus.n_val returns a hardcoded 5426 for the corpora that ship no validation
# split, and the real carve can land one row either side of it depending on how the stratifier
# rounds.  A silent size drift is worth flagging; it is not worth refusing to extract over.
_EXPECTED_N = {
    "train": lambda c: c.n_train,
    "val": lambda c: c.n_val,
    "test": lambda c: c.n_test,
}

# Commit granularity.  One transaction per this many documents; large enough that LMDB B-tree
# overhead and fsync cost are amortized, small enough that a wall-time kill loses at most a few
# seconds of GPU work.  8192 x 1024 dims x 4 bytes = 33 MB per transaction, which is fine.
DEFAULT_COMMIT_EVERY = 8192


# --------------------------------------------------------------------------- local snapshots
def resolve_snapshot(repo: str) -> str:
    """Absolute path of the LOCAL snapshot directory for a hub repo id.

    Runtime never downloads, so the encoder is loaded from disk.  We prefer
    ``langdual.assets.local_snapshot`` (where prepare_assets.sh puts it), and fall back to the
    standard HF cache layout under $HF_HOME.

    Raises FileNotFoundError with an actionable message rather than letting
    sentence-transformers attempt a hub fetch that will hang or fail opaquely offline.
    """
    # Preferred: ask the asset registry.  Imported lazily so that --verify / --self-test work
    # in an environment where assets.py does not exist.
    try:
        from langdual import assets  # type: ignore

        snap = assets.local_snapshot(repo)
        if snap and os.path.isdir(snap):
            return os.path.abspath(snap)
    except Exception:
        # Deliberately broad: a missing module, a renamed helper or an un-downloaded repo all
        # mean "fall through to the cache scan", and the cache scan raises a better error.
        pass

    # Fallback: $HF_HOME/hub/models--<org>--<name>/snapshots/<rev>.  If several revisions are
    # cached we take the most recently modified one, which is the one snapshot_download just
    # wrote.  Pinning a revision is the asset registry's job, not ours.
    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    repo_dir = os.path.join(hf_home, "hub", "models--" + repo.replace("/", "--"), "snapshots")
    if os.path.isdir(repo_dir):
        revs = [os.path.join(repo_dir, r) for r in os.listdir(repo_dir)]
        revs = [r for r in revs if os.path.isdir(r)]
        if revs:
            return os.path.abspath(max(revs, key=os.path.getmtime))

    raise FileNotFoundError(
        f"no local snapshot for {repo!r}; run prepare_assets.sh first (runtime stages never "
        f"download)."
    )


def load_encoder(enc: config.TextEncoder, device: str):
    """Load a frozen SentenceTransformer from disk and pin the protocol knobs.

    Pins, in order of how badly getting them wrong would corrupt the embeddings:
      * ``local_files_only=True`` -- never touch the hub, even if one is reachable.
      * ``max_seq_length`` from config.TEXT_ENCODERS, because the checkpoint's own
        sentence_bert_config.json and its tokenizer can disagree, and the truncation rate is
        recorded in the LMDB.
      * fp32 weights + TF32 matmuls == config.ENCODE_DTYPE ("fp32-tf32").  Left as a default
        this would silently vary with
        the torch version, and the embeddings -- hence every metric -- would move with it.
    """
    from sentence_transformers import SentenceTransformer  # heavy; imported on demand
    import torch

    # TF32 is an ampere+ matmul mode: fp32 storage/accumulate, reduced-precision multiply.
    # Enabled explicitly so the recorded ENCODE_DTYPE tag is truthful on every node.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    snapshot = resolve_snapshot(enc.repo)
    model = SentenceTransformer(snapshot, device=device, local_files_only=True)
    model.max_seq_length = int(enc.max_seq_length)
    model.eval()

    # Validate the contract dim rather than trusting it: a wrong dim here would produce LMDBs
    # whose __shape__ disagrees with the bytes, which surfaces much later as a reshape error
    # inside the SAE dataloader.
    got = int(model.get_sentence_embedding_dimension())
    if got != enc.dim:
        raise ValueError(
            f"encoder {enc.name} ({enc.repo}) has dim {got}, config.TEXT_ENCODERS says {enc.dim}"
        )
    return model


# --------------------------------------------------------------------------- corpus reading
def document_text(corpus: config.Corpus, rec: dict) -> str:
    """Assemble one document string from a canonical jsonl record.

    Handles both shapes the corpus-prep stage may produce: separate fields and an
    already-flattened "text" field.  Fields are joined with ". " in the declared order.
    """
    parts: List[str] = []
    for field in corpus.text_fields:
        v = rec.get(field)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    if parts:
        # Avoid ".." when a title already ends in punctuation.
        out = parts[0]
        for p in parts[1:]:
            out = out + (" " if out[-1] in ".!?" else ". ") + p
        return out

    # Flattened fallback: the prep stage may have merged text_fields into "text" already.
    v = rec.get("text")
    if isinstance(v, str):
        return v.strip()

    raise KeyError(
        f"{corpus.name}: record has none of text_fields={corpus.text_fields} nor 'text'; "
        f"keys present = {sorted(rec.keys())}"
    )


def document_id(corpus: str, split: str, idx: int, rec: dict) -> str:
    """Stable, human-readable document id stored under __path__<idx>.

    Prefers an id the corpus carries so an embedding can be traced back to the source row;
    otherwise synthesises one that is still unique.
    """
    for field in ("id", "doc_id", "uid", "guid"):
        v = rec.get(field)
        if v is not None and str(v) != "":
            return f"{corpus}/{split}/{v}"
    return f"{corpus}/{split}/{idx}"


def iter_documents(corpus: config.Corpus, split: str, start: int = 0) -> Iterator[Tuple[int, str, str]]:
    """Stream (idx, text, doc_id) from the canonical jsonl, skipping the first `start` rows.

    Streaming rather than loading: on resume only the tail is needed.  Skipped lines are not json-parsed, only counted, so a resume near the
    end costs one cheap pass over the file instead of a full parse.
    """
    path = config.corpus_jsonl(corpus.name, split)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"corpus jsonl missing: {path} -- run `materialize_corpora corpora` first"
        )
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if idx < start:
                continue
            line = line.strip()
            if not line:
                # A blank LINE is a corrupt file, not an empty DOCUMENT: an empty document is a
                # valid json record with an empty text field.  Refuse, because silently
                # shifting every subsequent index would misalign features against labels.
                raise ValueError(f"{path}: blank line at row {idx}; file is corrupt")
            rec = json.loads(line)
            yield idx, document_text(corpus, rec), document_id(corpus.name, split, idx, rec)


def count_documents(corpus: config.Corpus, split: str) -> int:
    """Number of rows in the canonical jsonl (the authoritative N for this split)."""
    path = config.corpus_jsonl(corpus.name, split)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"corpus jsonl missing: {path}")
    n = 0
    with open(path, "r", encoding="utf-8") as fh:
        for _ in fh:
            n += 1
    return n


def load_labels(corpus: config.Corpus, split: str, n_expected: int) -> np.ndarray:
    """Labels for one split: (N,) int64 single-label, or (N, n_classes) float32 multi-hot.

    The ``.npy`` written by the corpus-prep stage is authoritative.  The jsonl fallback exists
    because labels are load-bearing for every downstream metric (purity, kNN, linear probe) and
    a missing-npy failure three hours into a GPU job is far more expensive than 15 lines here.
    """
    npy = config.corpus_labels_npy(corpus.name, split)
    if os.path.isfile(npy):
        arr = np.load(npy)
    else:
        # Fallback: re-derive from the jsonl label_field.
        vals = []
        with open(config.corpus_jsonl(corpus.name, split), "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    vals.append(json.loads(line).get(corpus.label_field))
        if any(v is None for v in vals):
            raise FileNotFoundError(
                f"{npy} missing and jsonl has no usable {corpus.label_field!r} field"
            )
        if corpus.multilabel:
            # jsonl holds a list of active class ids; densify to multi-hot.
            arr = np.zeros((len(vals), corpus.n_classes), dtype=np.float32)
            for i, v in enumerate(vals):
                for c in (v if isinstance(v, (list, tuple)) else [v]):
                    arr[i, int(c)] = 1.0
        else:
            arr = np.asarray([int(v) for v in vals], dtype=np.int64)

    # ---- validation.  These checks are why the labels are loaded up front rather than lazily.
    if corpus.multilabel:
        if arr.ndim != 2 or arr.shape[1] != corpus.n_classes:
            raise ValueError(
                f"{corpus.name}/{split}: multi-hot labels must be (N, {corpus.n_classes}), got {arr.shape}"
            )
        arr = arr.astype(np.float32, copy=False)
        if not np.all((arr == 0.0) | (arr == 1.0)):
            raise ValueError(f"{corpus.name}/{split}: multi-hot labels are not binary")
    else:
        arr = arr.reshape(-1).astype(np.int64, copy=False)
        lo, hi = int(arr.min()), int(arr.max())
        if lo < 0 or hi >= corpus.n_classes:
            raise ValueError(
                f"{corpus.name}/{split}: class ids out of range [0,{corpus.n_classes}) -> [{lo},{hi}]"
            )
    if arr.shape[0] != n_expected:
        raise ValueError(
            f"{corpus.name}/{split}: {arr.shape[0]} labels vs {n_expected} documents -- "
            f"labels and jsonl are out of sync"
        )
    return arr


# --------------------------------------------------------------------------- LMDB plumbing
def _map_size(n_rows: int, n_floats: int) -> int:
    """Generous LMDB map size: 2.5x the float payload plus 1 GiB.

    map_size is reserved VIRTUAL address space, so over-sizing costs nothing on a 64-bit host,
    whereas hitting MDB_MAP_FULL mid-run means reopening with a bigger map and redoing the
    transaction.  The 2.5x factor covers B-tree overhead plus the per-row __label__/__path__
    keys, mirroring the vision extractor's own sizing rule.
    """
    payload = int(n_rows) * int(n_floats) * 4
    return int(payload * 2.5) + (1 << 30)


def _read_progress(env) -> int:
    """Number of contiguous rows already committed (0 when the LMDB is fresh)."""
    with env.begin() as txn:
        raw = txn.get(b"__progress__")
    return int(raw.decode("ascii")) if raw is not None else 0


def _is_complete(path: str, n_expected: int) -> bool:
    """True iff `path` holds a FINISHED feature LMDB of the expected length.

    Completeness is defined by ``__len__`` existing, because ``__len__`` is written only in the
    final transaction.  That makes a half-written LMDB *unopenable* by LMDBFeatureDataset
    (which would crash on ``None.decode``) instead of silently training on a truncated split --
    a loud failure is the correct behaviour for a partial extraction.
    """
    import lmdb

    if not os.path.isdir(path):
        return False
    try:
        env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
    except lmdb.Error:
        return False
    try:
        with env.begin() as txn:
            raw_len = txn.get(b"__len__")
            raw_prog = txn.get(b"__progress__")
        if raw_len is None or raw_prog is None:
            return False
        return int(raw_len.decode("ascii")) == n_expected == int(raw_prog.decode("ascii"))
    finally:
        env.close()


def write_labels_lmdb(corpus: config.Corpus, split: str, labels: np.ndarray) -> str:
    """Write the companion multi-hot label store for a MULTI-LABEL corpus.

    Format is dictated by the consumer's companion branch (``lmdb_features.py`` ~L270-282):
    key ``str(idx)``, value raw float32 multi-hot bytes, plus its own ``__len__`` and
    ``__shape__``.

    Written once per (corpus, split) and shared by every encoder, so concurrent encoder jobs
    can race for it.  The race is resolved by building into a pid-unique temporary
    directory and ``os.rename``-ing it into place: rename of a directory is atomic on the same
    filesystem and fails if the destination already exists, so the loser simply discards its
    copy.  Never partially visible, never corrupted by two writers.
    """
    import lmdb

    dest = config.labels_lmdb_path(corpus.name, split)
    config.assert_safe_text_path(dest)

    if _is_complete(dest, labels.shape[0]):
        return dest  # skip-guard: another encoder job (or an earlier run) already wrote it

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = f"{dest}.tmp{os.getpid()}"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)  # leftover from a killed run; its content is unverifiable

    n, c = labels.shape
    env = lmdb.open(tmp, map_size=_map_size(n, c), subdir=True, meminit=False)
    try:
        # Small enough (goemotions: 43k x 28 floats = 4.9 MB) for a single transaction, so the
        # whole store is atomic by construction and cannot be resumed part-way.  It still
        # carries a __progress__ marker so that _is_complete() -- the ONE definition of
        # "finished" in this module -- applies to both stores unchanged; without it the
        # skip-guard below rejects a perfectly good store and the second encoder job for the
        # same corpus dies on the rename.
        with env.begin(write=True) as txn:
            for i in range(n):
                row = np.ascontiguousarray(labels[i], dtype=np.float32)
                txn.put(str(i).encode("ascii"), row.tobytes())
            txn.put(b"__len__", str(n).encode("ascii"))
            txn.put(b"__shape__", str((int(c),)).encode("ascii"))
            txn.put(b"__progress__", str(n).encode("ascii"))
            txn.put(b"__corpus__", corpus.name.encode("utf-8"))
            txn.put(b"__split__", split.encode("utf-8"))
    finally:
        env.close()

    try:
        os.rename(tmp, dest)
    except OSError:
        # Lost the race (or the destination appeared between the check and here).  Keep the
        # winner; ours is byte-identical because the labels npy is deterministic.
        shutil.rmtree(tmp, ignore_errors=True)
        if not _is_complete(dest, n):
            raise
    return dest


# --------------------------------------------------------------------------- extraction
def extract_split(
    encoder_name: str,
    corpus_name: str,
    split: str,
    batch_size: int = 256,
    commit_every: int = DEFAULT_COMMIT_EVERY,
    device: Optional[str] = None,
    model=None,
) -> Dict[str, object]:
    """Encode one (encoder, corpus, split) into its LMDB.  Resumable and idempotent.

    Returns a summary dict (also printed by main()).  `model` may be passed in so a single
    invocation reuses one loaded encoder across all three splits.
    """
    import lmdb
    import torch

    enc = config.TEXT_ENCODERS[encoder_name]
    corpus = config.CORPORA[corpus_name]
    out = config.lmdb_path(encoder_name, corpus_name, split)
    config.assert_safe_text_path(out)

    n_docs = count_documents(corpus, split)
    exp = _EXPECTED_N[split](corpus)
    if n_docs != exp:
        # Warn, do not fail -- see the _EXPECTED_N comment.
        print(f"  [warn] {corpus_name}/{split}: {n_docs} rows on disk, config expects ~{exp}")

    if _is_complete(out, n_docs):
        print(f"  [skip] {out} already complete ({n_docs} rows)")
        return {"status": "skip", "path": out, "n": n_docs}

    labels = load_labels(corpus, split, n_docs)

    # Multi-label corpora keep their labels in a companion store, written before the feature
    # LMDB so that __labels_lmdb__ never points at something that does not exist yet.
    labels_lmdb: Optional[str] = None
    if corpus.multilabel:
        labels_lmdb = write_labels_lmdb(corpus, split, labels)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    env = lmdb.open(out, map_size=_map_size(n_docs, enc.dim), subdir=True, meminit=False)
    try:
        start = _read_progress(env)
        if start > n_docs:
            raise ValueError(
                f"{out}: __progress__={start} exceeds {n_docs} documents -- the corpus changed "
                f"under a partial extraction; delete the LMDB and re-extract"
            )
        if start:
            print(f"  [resume] {out} at row {start}/{n_docs}")

        if model is None:
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = load_encoder(enc, device)
        tokenizer = getattr(model, "tokenizer", None)

        # Truncation / empty-document tallies survive a resume by living in the LMDB, committed
        # in the same transaction as __progress__.
        with env.begin() as txn:
            n_trunc = int((txn.get(b"__trunc_count__") or b"0").decode("ascii"))
            n_empty = int((txn.get(b"__empty_count__") or b"0").decode("ascii"))

        t0 = time.time()
        buf_text: List[str] = []
        buf_id: List[str] = []
        buf_start = start
        done = start

        def flush() -> None:
            """Encode the buffered chunk and commit it as ONE transaction."""
            nonlocal n_trunc, n_empty, done, buf_text, buf_id, buf_start
            if not buf_text:
                return
            m = len(buf_text)

            # Truncation accounting, measured on the SAME strings that are about to be encoded
            # (prefix included, because the prefix consumes real tokens).  add_special_tokens
            # matches what the ST Transformer module does, truncation is off so we see the true
            # length.  Best-effort: a tokenizer that refuses the call must not abort extraction.
            if tokenizer is not None and n_trunc >= 0:
                try:
                    ids = tokenizer(
                        buf_text, add_special_tokens=True, truncation=False, padding=False
                    )["input_ids"]
                    n_trunc += sum(1 for s in ids if len(s) > enc.max_seq_length)
                except Exception:
                    n_trunc = -1  # sentinel: "not measurable", reported as such

            # normalize_embeddings=False: store the encoder's NATIVE output. LMDBFeatureDataset
            # L2-normalizes on read.
            emb = model.encode(
                buf_text,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            emb = np.ascontiguousarray(emb, dtype=np.float32)
            if emb.shape != (m, enc.dim):
                raise ValueError(f"encoder returned {emb.shape}, expected {(m, enc.dim)}")
            if not np.isfinite(emb).all():
                raise ValueError(
                    f"{out}: non-finite embedding in rows [{buf_start},{buf_start + m})"
                )

            with env.begin(write=True) as txn:
                for j in range(m):
                    i = buf_start + j
                    txn.put(str(i).encode("ascii"), emb[j].tobytes())
                    txn.put(f"__path__{i}".encode("ascii"), buf_id[j].encode("utf-8"))
                    # Scalar labels ONLY for single-label corpora: the presence of __label__0
                    # is exactly how the consumer decides not to consult the companion store.
                    if not corpus.multilabel:
                        txn.put(f"__label__{i}".encode("ascii"), str(int(labels[i])).encode("ascii"))
                txn.put(b"__trunc_count__", str(n_trunc).encode("ascii"))
                txn.put(b"__empty_count__", str(n_empty).encode("ascii"))
                # __progress__ LAST.  The transaction is atomic so ordering is not strictly
                # required today, but it makes the invariant "a committed feature always has a
                # progress marker" hold even if this loop is ever split across transactions.
                txn.put(b"__progress__", str(buf_start + m).encode("ascii"))

            done = buf_start + m
            buf_text, buf_id, buf_start = [], [], done
            rate = (done - start) / max(time.time() - t0, 1e-6)
            print(f"    {done}/{n_docs}  ({rate:.0f} docs/s)", flush=True)

        for idx, text, doc_id in iter_documents(corpus, split, start=start):
            if not text:
                # Empty documents can exist.  Kept as empty
                # strings so index alignment with the labels is preserved; the encoder still
                # returns a well-defined [CLS]/[SEP] embedding.  Counted as provenance.
                n_empty += 1
            buf_text.append(enc.prefix + text)  # documents take no prefix for BGE
            buf_id.append(doc_id)
            if len(buf_text) >= commit_every:
                flush()
        flush()

        if done != n_docs:
            raise ValueError(f"{out}: wrote {done} rows but the jsonl has {n_docs}")

        # ---- finalize: metadata + provenance in one last transaction.  __len__ appearing is
        # what makes the LMDB "complete" for every skip-guard and for the consumer.
        trunc_frac = (n_trunc / n_docs) if (n_docs and n_trunc >= 0) else float("nan")
        with env.begin(write=True) as txn:
            txn.put(b"__len__", str(n_docs).encode("ascii"))
            txn.put(b"__shape__", str((int(enc.dim),)).encode("ascii"))
            txn.put(b"__multilabel__", b"1" if corpus.multilabel else b"0")
            if labels_lmdb is not None:
                # ABSOLUTE path: the companion store is shared across encoders and is therefore
                # not a sibling directory.
                txn.put(b"__labels_lmdb__", os.path.abspath(labels_lmdb).encode("utf-8"))
            txn.put(b"__encoder__", enc.name.encode("utf-8"))
            txn.put(b"__encoder_repo__", enc.repo.encode("utf-8"))
            txn.put(b"__encode_dtype__", config.ENCODE_DTYPE.encode("utf-8"))
            txn.put(b"__prefix__", enc.prefix.encode("utf-8"))
            txn.put(b"__max_seq_length__", str(int(enc.max_seq_length)).encode("ascii"))
            txn.put(b"__truncated_fraction__", repr(trunc_frac).encode("ascii"))
            txn.put(b"__corpus__", corpus.name.encode("utf-8"))
            txn.put(b"__split__", split.encode("utf-8"))
            txn.put(b"__n_classes__", str(int(corpus.n_classes)).encode("ascii"))
    finally:
        env.close()

    print(
        f"  [done] {out}: {n_docs} x {enc.dim}  truncated={trunc_frac:.4f}  empty={n_empty}"
    )
    return {
        "status": "ok",
        "path": out,
        "n": n_docs,
        "dim": enc.dim,
        "truncated_fraction": trunc_frac,
        "empty": n_empty,
        "labels_lmdb": labels_lmdb,
    }


# --------------------------------------------------------------------------- verification
def verify_split(encoder_name: str, corpus_name: str, split: str) -> Dict[str, object]:
    """Walk 0..N-1 and assert every key the consumer will ask for is present and well-formed.

    Checks, in the order they would bite:
      * ``__len__`` / ``__shape__`` exist and agree with the config dim and the jsonl row count
      * every ``<idx>`` holds exactly 4*D bytes and is finite
      * every ``__path__<idx>`` decodes as UTF-8
      * single-label: every ``__label__<idx>`` is an int in [0, n_classes)
      * multi-label: NO ``__label__`` keys, ``__labels_lmdb__`` points at a complete companion
        store of the right length and per-row byte size
    Raises AssertionError on the first violation; returns a summary otherwise.
    """
    import lmdb

    enc = config.TEXT_ENCODERS[encoder_name]
    corpus = config.CORPORA[corpus_name]
    path = config.lmdb_path(encoder_name, corpus_name, split)
    config.assert_safe_text_path(path)
    assert os.path.isdir(path), f"missing LMDB: {path}"

    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin() as txn:
            raw_len = txn.get(b"__len__")
            assert raw_len is not None, f"{path}: no __len__ (extraction never finished)"
            n = int(raw_len.decode("ascii"))
            shape = txn.get(b"__shape__").decode("ascii")
            assert shape == str((int(enc.dim),)), f"{path}: __shape__={shape}, want ({enc.dim},)"
            nbytes = 4 * enc.dim

            ml = txn.get(b"__multilabel__")
            assert ml is not None, f"{path}: no __multilabel__ marker"
            assert (ml == b"1") == corpus.multilabel, f"{path}: __multilabel__ disagrees with config"

            n_jsonl = count_documents(corpus, split)
            assert n == n_jsonl, f"{path}: __len__={n} but jsonl has {n_jsonl} rows"
            assert int(txn.get(b"__progress__").decode("ascii")) == n

            for i in range(n):
                v = txn.get(str(i).encode("ascii"))
                assert v is not None, f"{path}: missing feature row {i}"
                assert len(v) == nbytes, f"{path}: row {i} has {len(v)} bytes, want {nbytes}"
                p = txn.get(f"__path__{i}".encode("ascii"))
                assert p is not None, f"{path}: missing __path__{i}"
                p.decode("utf-8")  # raises UnicodeDecodeError if the id was mangled
                if corpus.multilabel:
                    assert txn.get(f"__label__{i}".encode("ascii")) is None, (
                        f"{path}: scalar __label__{i} present on a MULTI-LABEL corpus -- it "
                        f"would shadow the companion store in LMDBFeatureDataset"
                    )
                else:
                    lb = txn.get(f"__label__{i}".encode("ascii"))
                    assert lb is not None, f"{path}: missing __label__{i}"
                    c = int(lb.decode("ascii"))
                    assert 0 <= c < corpus.n_classes, f"{path}: label {c} out of range at {i}"

            # Finiteness is checked on a stride rather than on all N rows: decoding 560k
            # vectors twice is pure wall-clock, and a non-finite embedding is already rejected
            # at write time.  The stride is the cheap second line of defence.
            for i in range(0, n, max(1, n // 1000)):
                x = np.frombuffer(txn.get(str(i).encode("ascii")), dtype=np.float32)
                assert np.isfinite(x).all(), f"{path}: non-finite row {i}"

            ll = txn.get(b"__labels_lmdb__")
    finally:
        env.close()

    if corpus.multilabel:
        assert ll is not None, f"{path}: multi-label corpus without __labels_lmdb__"
        lpath = ll.decode("utf-8")
        assert os.path.isabs(lpath), f"{path}: __labels_lmdb__ must be absolute, got {lpath}"
        assert lpath == config.labels_lmdb_path(corpus_name, split), (
            f"{path}: __labels_lmdb__={lpath} does not match config.labels_lmdb_path"
        )
        lenv = lmdb.open(lpath, readonly=True, lock=False, readahead=False, meminit=False)
        try:
            with lenv.begin() as ltxn:
                assert int(ltxn.get(b"__len__").decode("ascii")) == n
                lshape = ltxn.get(b"__shape__").decode("ascii")
                assert lshape == str((int(corpus.n_classes),)), f"{lpath}: __shape__={lshape}"
                want = 4 * corpus.n_classes
                for i in range(n):
                    v = ltxn.get(str(i).encode("ascii"))
                    assert v is not None, f"{lpath}: missing label row {i}"
                    assert len(v) == want, f"{lpath}: row {i} has {len(v)} bytes, want {want}"
        finally:
            lenv.close()
    else:
        assert ll is None, f"{path}: single-label corpus must not declare __labels_lmdb__"

    print(f"  [verify OK] {path}: {n} rows x {enc.dim}")
    return {"status": "ok", "path": path, "n": n, "dim": enc.dim}


# --------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encoder", choices=config.ENCODER_ORDER, help="text encoder")
    ap.add_argument("--corpus", choices=config.CORPUS_ORDER, help="labelled corpus")
    ap.add_argument(
        "--splits", default=",".join(SPLITS),
        help=f"comma-separated subset of {SPLITS} (default: all three)",
    )
    ap.add_argument("--batch-size", type=int, default=256, help="encoder forward batch size")
    ap.add_argument(
        "--commit-every", type=int, default=DEFAULT_COMMIT_EVERY,
        help="documents per LMDB transaction (the resume granularity)",
    )
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--verify", action="store_true", help="verify existing LMDBs, write nothing")
    args = ap.parse_args(argv)

    if not args.encoder or not args.corpus:
        ap.error("--encoder and --corpus are required")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    bad = [s for s in splits if s not in SPLITS]
    if bad:
        ap.error(f"unknown split(s) {bad}; choose from {SPLITS}")

    if args.verify:
        for split in splits:
            verify_split(args.encoder, args.corpus, split)
        return 0

    if args.batch_size < 1 or args.commit_every < args.batch_size:
        ap.error("--commit-every must be >= --batch-size >= 1")

    # Load the encoder ONCE for all requested splits.
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"encoder={args.encoder} corpus={args.corpus} device={device} dtype={config.ENCODE_DTYPE}")
    model = load_encoder(config.TEXT_ENCODERS[args.encoder], device)

    for split in splits:
        extract_split(
            args.encoder, args.corpus, split,
            batch_size=args.batch_size, commit_every=args.commit_every,
            device=device, model=model,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
