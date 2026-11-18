"""Pack the training corpora into the token streams the residual-stream extraction reads.

For each training corpus (config.LM_CORPORA: OpenWebText and GitHub code) this writes three
packed uint16 token streams under config.TOKENS_DIR:

  train   exactly config.LM_TRAIN_TOKENS (64M) tokens = one pass of the SAE step budget;
  val     config.TRACKB_SCORE_TOKENS (200K) tokens, used only for best-checkpoint selection;
  test    another 200K tokens, used for evaluation.

Packing rules: documents are concatenated in raw shard order (sorted shard paths, so the order
is reproducible), each followed by ONE end-of-text token, with NO padding (Pythia has no pad
token, and aliasing pad to end-of-text would mask every genuine end-of-text target in the CE
evaluation).  Empty documents are skipped.  The final document of a stream is truncated so the
length is exact.  The val slice starts strictly after the last document the train stream
touched, and test strictly after val, so the three streams share no token.

The raw parquet shards are read with pyarrow directly (no `datasets` dependency).

Usage
-----
    python -m langdual.materialize_corpora pack --lm-corpus owt
    python -m langdual.materialize_corpora pack --lm-corpus ghcode

Every stream is a no-op when its output already exists and agrees with `config` (skip-guard),
and every file is written through a temp file + `os.replace`, so a killed job leaves no
half-written artefact behind.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import sys
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from langdual import config


def _ensure_pyarrow():
    """Import pyarrow.parquet, with an actionable error if it is not installed."""
    try:
        import pyarrow.parquet as pq  # noqa: F401
        return pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to read the corpus parquet shards "
                          "(pip install -r language/requirements.txt)") from exc


# Bumped whenever the on-disk layout changes in a way that invalidates an existing
# materialisation; the skip-guard refuses to trust a meta.json written by an older version.
FORMAT_VERSION = 1


_HASH_CHUNK = 1 << 20          # 1 MiB; hashing a 200 MB parquet costs well under a second


_PARQUET_BATCH = 2048          # rows per arrow batch: bounded memory, negligible per-batch cost


class MaterializeError(RuntimeError):
    """Raised for any condition that would silently poison a downstream metric."""


# --------------------------------------------------------------------------- atomic IO helpers
def _fsync_dir(path: str) -> None:
    """fsync a directory so a rename is durable.

    On a shared parallel filesystem a rename that is not followed by a directory fsync can be
    lost if the node dies; the cost is microseconds, so we always pay it.  EINVAL/EPERM on
    filesystems that refuse O_RDONLY directory fsync is non-fatal and swallowed.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


@contextlib.contextmanager
def _atomic_text(path: str):
    """Yield a text handle whose content only becomes visible at `path` on clean exit.

    The temp file lives in the SAME directory, because os.replace is only atomic within one
    filesystem.  On any exception the temp file is removed, so a killed job leaves no half file
    and the skip-guard cannot mistake it for a finished output.
    """
    config.assert_safe_text_path(path)          # every written path is checked, no exceptions
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".tmp.{os.path.basename(path)}.{os.getpid()}")
    f = open(tmp, "w", encoding="utf-8", newline="\n")
    try:
        yield f
        f.flush()
        os.fsync(f.fileno())
        f.close()
        os.replace(tmp, path)
        _fsync_dir(d)
    except BaseException:
        try:
            f.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        raise


def _save_npy_atomic(path: str, arr: np.ndarray) -> None:
    """np.save through a temp file + os.replace (same directory, same rules as _atomic_text)."""
    config.assert_safe_text_path(path)
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".tmp.{os.path.basename(path)}.{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            # allow_pickle=False: these arrays are plain numeric data and must stay loadable by
            # a reader that (correctly) refuses pickled npy files.
            np.save(fh, arr, allow_pickle=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_dir(d)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _write_json_atomic(path: str, obj: Any) -> None:
    with _atomic_text(path) as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, ensure_ascii=False)
        fh.write("\n")


def _read_json(path: str) -> Optional[dict]:
    """Read a JSON file, returning None if it is missing or truncated.

    A truncated file should be impossible given the atomic writer, but treating a corrupt
    meta.json as "not done" rather than crashing makes a re-run self-healing.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(x: Any) -> Any:
    """Recursively convert numpy scalars/arrays so json.dump cannot choke on them."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


# --------------------------------------------------------------------------- raw discovery
_SPLIT_TOKENS = {
    "validation": "val", "valid": "val", "val": "val", "dev": "val",
    "test": "test", "eval": "test",
    "train": "train",
}


_DATA_EXTS = (".parquet", ".jsonl", ".jsonl.gz", ".json.gz", ".json")


# Metadata / bookkeeping directories that snapshot_download leaves behind and that must never
# be mistaken for data shards.
_SKIP_DIRS = {".git", ".cache", ".huggingface", "__pycache__"}


# Basenames that carry a data extension but are metadata, never shards.
_NON_DATA_BASENAMES = {
    "dataset_infos.json", "dataset_info.json", "config.json", "state.json",
    "readme.json", "index.json", ".langdual_snapshot.json",
}


def _tokens(component: str) -> List[str]:
    """Split one path component into lowercase alphanumeric tokens."""
    out, cur = [], []
    for ch in component.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def _classify_split(rel_path: str) -> Optional[str]:
    """Infer the split of a raw shard from its relative path.

    Scans from the BASENAME outward: `data/train/test-00000.parquet` is a test shard, and
    `plain_text/partial-train/0000.parquet` (which has no split token in its basename) is a
    train shard.  A single component that names two different splits is ambiguous and raises
    rather than guessing, because a mis-assigned shard would silently corrupt every split.
    """
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    for comp in reversed(parts):                       # basename first, then parents
        found = {_SPLIT_TOKENS[t] for t in _tokens(comp) if t in _SPLIT_TOKENS}
        if len(found) > 1:
            raise MaterializeError(
                f"ambiguous split in raw path component {comp!r} of {rel_path!r}: {sorted(found)}"
            )
        if found:
            return found.pop()
    return None


def _snapshot_dir(repo: str) -> str:
    """Resolve a dataset repo id to the directory prepare_assets.sh downloaded it into.

    Delegates to ``langdual.assets`` rather than re-deriving the layout, so a change to the
    download layout can never leave this module looking in the wrong place.  Falls back to the
    older RAW_DIR/<name> convention only if assets cannot be imported, which keeps the module
    usable standalone in a test fixture.
    """
    try:
        from langdual import assets as _assets
        return _assets.local_snapshot(repo, "dataset")
    except Exception:  # pragma: no cover - only on a partial checkout
        return os.path.join(config.RAW_DIR, repo.split("/")[-1])


def discover_source_files(raw_dir: str) -> Dict[str, List[str]]:
    """Map split -> sorted list of absolute shard paths under a raw snapshot directory.

    Sorting is what makes the row order FIXED: shard `train-00000-of-00004` always precedes
    `train-00001-of-00004`, so the concatenated raw row order is reproducible across machines
    and across re-runs.  Shards whose split cannot be inferred are returned under the key
    ``"_unclassified"`` so the caller can report them instead of silently dropping data.
    """
    if not os.path.isdir(raw_dir):
        raise MaterializeError(
            f"raw snapshot directory does not exist: {raw_dir}\n"
            "Run prepare_assets.sh first (it is the only stage with network access)."
        )
    by_split: Dict[str, List[str]] = {}
    for dirpath, dirnames, filenames in os.walk(raw_dir):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            low = fn.lower()
            if not low.endswith(_DATA_EXTS):
                continue
            # Skip bookkeeping files that share a data extension.  ".json" is in _DATA_EXTS
            # because a couple of corpora ship plain-json splits, which also catches every
            # sidecar a downloader leaves behind: prepare_assets.sh writes a
            # ".langdual_snapshot.json" completion sentinel, and the hub adds dataset_infos.json
            # and friends.  Left unfiltered these land in "_unclassified" and abort the corpus
            # with "could not infer the split", which is a confusing way to report a non-issue.
            if fn.startswith(".") or low in _NON_DATA_BASENAMES:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, raw_dir)
            split = _classify_split(rel) or "_unclassified"
            by_split.setdefault(split, []).append(full)
    for k in by_split:
        by_split[k].sort()
    return by_split


# --------------------------------------------------------------------------- raw readers
def _iter_parquet_rows(path: str, columns: Sequence[str]) -> Iterator[Dict[str, Any]]:
    """Stream one parquet file row by row, materialising at most _PARQUET_BATCH rows at a time.

    `to_pydict()` converts an arrow list column into a python list, which
    is exactly what the multi-hot builder wants; no pandas, no `datasets`.
    """
    pq = _ensure_pyarrow()
    pf = pq.ParquetFile(path)
    have = set(pf.schema_arrow.names)
    missing = [c for c in columns if c not in have]
    if missing:
        raise MaterializeError(
            f"{path}: missing column(s) {missing}; the file has {sorted(have)}"
        )
    for batch in pf.iter_batches(batch_size=_PARQUET_BATCH, columns=list(columns)):
        cols = batch.to_pydict()
        for i in range(batch.num_rows):
            yield {c: cols[c][i] for c in columns}


def _open_maybe_gzip(path: str):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _iter_jsonl_rows(path: str, columns: Sequence[str]) -> Iterator[Dict[str, Any]]:
    """Stream a jsonl file with stdlib json.

    jsonl is one object per line by definition (embedded newlines are escaped), so a line-wise
    reader is both correct and the lowest-memory option.  Blank lines are tolerated because
    some exporters leave a trailing one; anything else that fails to parse raises with the line
    number, since a silently skipped row would shift every subsequent index.
    """
    with _open_maybe_gzip(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MaterializeError(f"{path}:{lineno}: malformed JSON line: {exc}") from exc
            missing = [c for c in columns if c not in obj]
            if missing:
                raise MaterializeError(
                    f"{path}:{lineno}: missing field(s) {missing}; has {sorted(obj)}"
                )
            yield {c: obj[c] for c in columns}


def _iter_rows(files: Sequence[str], columns: Sequence[str]) -> Iterator[Dict[str, Any]]:
    """Concatenate the rows of several shards in the given (already sorted) order."""
    for path in files:
        low = path.lower()
        if low.endswith(".parquet"):
            yield from _iter_parquet_rows(path, columns)
        else:
            yield from _iter_jsonl_rows(path, columns)


# --------------------------------------------------------------------------- Track B: tokens
def token_labels_path(split: str) -> str:
    """Companion per-token label array for a labeled token stream.

    Derived FROM config.token_stream so the two can never drift; config owns the stream path and
    has no helper for the labels, so this is the one place that names it.
    """
    ts = config.token_stream(split)
    suffix = ".u16.npy"
    if not ts.endswith(suffix):
        raise MaterializeError(f"config.token_stream({split!r}) no longer ends in {suffix!r}: {ts}")
    return ts[: -len(suffix)] + ".labels.npy"


def token_docid_path(split: str) -> str:
    """Per-token document id (uint32).

    Not decoration: without it nothing downstream can drop the end-of-text tokens, pool a
    document's tokens, or tell that a 128-token context straddles two documents.  4 B/token.
    """
    ts = config.token_stream(split)
    return ts[: -len(".u16.npy")] + ".docid.npy"


def token_meta_path(split: str) -> str:
    ts = config.token_stream(split)
    return ts[: -len(".u16.npy")] + ".meta.json"


def _smallest_safe_uint_dtype(max_id: int) -> np.dtype:
    """Pick the narrowest unsigned integer type that can hold every token id.

    CHECKED, not assumed: Pythia's padded vocabulary is 50304 (config.vocab_size) and the
    GPT-NeoX tokenizer's largest id is below that.  uint16 spans 0..65535, and 50304 <= 65535,
    so uint16 is safe and is what config.token_stream's `.u16.npy` name promises.  If a model
    ever pushed the vocabulary past 65535 this would return uint32 -- and the caller refuses to
    write it under the `.u16` name rather than lie about the dtype in the filename.
    """
    if max_id <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if max_id <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    raise MaterializeError(f"token id {max_id} does not fit in uint32")


def load_pythia_tokenizer(tokenizer_dir: Optional[str] = None):
    """Load the Pythia tokenizer from the local HF snapshot, strictly offline.

    Returns (tokenizer, eos_id, dtype, max_id).  Raises with an actionable message if the
    snapshot is missing (runtime stages never download).
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise MaterializeError("transformers is required for the `pack` subcommand") from exc

    # prepare_assets.sh downloads with local_dir=, so the snapshot lives under RAW_DIR and
    # is NOT in HF_HOME's hub cache -- the bare repo id would not resolve offline.  Ask
    # langdual.assets where it actually is, and keep the repo id only as a last-resort fallback
    # for a machine where the model happens to be cached the usual way.
    if tokenizer_dir:
        src = tokenizer_dir
    else:
        try:
            from langdual import assets as _assets
            src = _assets.local_snapshot(config.LM_REPO, "model")
            if not os.path.isdir(src):
                src = config.LM_REPO
        except Exception:  # pragma: no cover - partial checkout
            src = config.LM_REPO
    try:
        tok = AutoTokenizer.from_pretrained(src, local_files_only=True)
    except Exception as exc:
        raise MaterializeError(
            f"could not load the tokenizer for {src!r} offline "
            f"(HF_HOME={os.environ.get('HF_HOME')}).  Run prepare_assets.sh first; runtime "
            "stages never download."
        ) from exc

    eos = tok.eos_token_id
    if eos is None:
        raise MaterializeError(f"{src}: tokenizer has no eos_token_id; cannot pack a stream")
    # Pythia's pad_token_id is None.  That is exactly why the packing below uses NO padding: the
    # reference CE-recovered implementation masks targets with ignore_index=pad_token_id, and
    # aliasing pad to <|endoftext|> would silently ignore every genuine end-of-text target.
    pad = tok.pad_token_id

    # len(tok) covers the base vocab plus any added special tokens; the model's padded
    # vocab_size can be larger still, so take the max of what we can see.
    max_id = max(int(len(tok)) - 1, int(eos))
    dtype = _smallest_safe_uint_dtype(max_id)
    return tok, int(eos), dtype, max_id, pad


def _encode_batch(tok, texts: Sequence[str]) -> List[List[int]]:
    """Tokenize a batch with no special tokens.

    add_special_tokens=False is deliberate: the ONLY separator in the packed stream is the
    tokenizer's own end-of-text, appended once per document by the packer.
    """
    return tok(list(texts), add_special_tokens=False)["input_ids"]


def _pack_documents(doc_iter: Iterable[Tuple[int, str]], encode: Callable[[Sequence[str]], List[List[int]]],
                    eos_id: int, dtype: np.dtype, budget: int, batch_size: int = 512,
                    with_docid: bool = False):
    """Pack documents into a flat token array of EXACTLY `budget` tokens (or fewer if exhausted).

    Contract:
      * every document contributes its tokens followed by ONE end-of-text token;
      * there is NO padding anywhere (see load_pythia_tokenizer for why);
      * the final document is truncated mid-way if that is what it takes to hit `budget`
        exactly, because config.LM_TRAIN_TOKENS is defined as TOTAL_STEPS * BATCH_SIZE, i.e.
        exactly one pass -- an off-by-a-few-hundred stream would silently shorten the last step.

    Returns (tokens, docids_or_None, n_docs_consumed, n_docs_fully_packed, next_doc_id).

    `n_docs_consumed` is a COUNT (it includes the partially-truncated final document);
    `next_doc_id` is the id one past the last document touched.  They are NOT the same number
    whenever the iterator skips documents -- empty documents are skipped here -- and it is
    `next_doc_id` that a caller must use to resume, otherwise a held-out slice would silently
    overlap the train stream by however many empty documents were skipped.
    """
    if budget <= 0:
        raise MaterializeError(f"token budget must be positive, got {budget}")
    out = np.empty(budget, dtype=dtype)
    docid = np.empty(budget, dtype=np.uint32) if with_docid else None
    n = 0
    n_consumed = 0
    n_full = 0
    next_id = 0

    buf_ids: List[int] = []
    buf_txt: List[str] = []

    def flush() -> bool:
        """Encode and append one buffered batch; return True when the budget is full."""
        nonlocal n, n_consumed, n_full, next_id
        if not buf_txt:
            return False
        for did, ids in zip(buf_ids, encode(buf_txt)):
            seq = list(ids)
            seq.append(eos_id)                       # the tokenizer's own document separator
            n_consumed += 1
            next_id = did + 1                        # resume point: one past this document
            take = min(len(seq), budget - n)
            out[n:n + take] = np.asarray(seq[:take], dtype=dtype)
            if docid is not None:
                docid[n:n + take] = np.uint32(did)
            n += take
            if take == len(seq):
                n_full += 1
            if n >= budget:
                return True
        buf_ids.clear()
        buf_txt.clear()
        return False

    for did, text in doc_iter:
        # An empty document would contribute only an end-of-text token, i.e. pure noise for the
        # residual-stream statistics, so it is skipped and not counted as consumed.
        if not text:
            continue
        buf_ids.append(did)
        buf_txt.append(text)
        if len(buf_txt) >= batch_size:
            if flush():
                break
    else:
        flush()

    buf_ids.clear()
    buf_txt.clear()
    return out[:n], (docid[:n] if docid is not None else None), n_consumed, n_full, next_id


def _owt_doc_iter(files: Sequence[str], skip_docs: int = 0,
                  column: str = "text") -> Iterator[Tuple[int, str]]:
    """Yield (doc_index, text) over a Track B corpus' shards, optionally skipping a prefix.

    ``column`` is the corpus' text field: OpenWebText calls it "text", the GitHub code corpus
    calls it "code".  It comes from config.LM_CORPORA so there is one definition of it.

    Skipping by count (rather than re-tokenizing) is what makes the held-out eval slice
    resumable: the train stream's meta records how many documents it consumed, so the eval pass
    starts strictly after them with no overlap and no second tokenization of the train text.
    """
    i = 0
    for row in _iter_rows(files, [column]):
        t = row.get(column)
        t = "" if t is None else str(t)
        if i >= skip_docs:
            yield i, t
        i += 1


def _stream_done(split: str, expect: Dict[str, Any]) -> bool:
    """Skip-guard for one packed stream: meta present, current, and matching the request."""
    meta = _read_json(token_meta_path(split))
    if not meta or meta.get("format_version") != FORMAT_VERSION:
        return False
    if not os.path.exists(config.token_stream(split)):
        return False
    for k, v in expect.items():
        if meta.get(k) != v:
            return False
    if meta.get("has_labels"):
        if not (os.path.exists(token_labels_path(split)) and os.path.exists(token_docid_path(split))):
            return False
    return True


def pack_owt(raw_dir: Optional[str] = None, train_tokens: Optional[int] = None,
             eval_tokens: Optional[int] = None, tokenizer_dir: Optional[str] = None,
             force: bool = False, verbose: bool = True,
             corpus: str = "owt") -> Dict[str, Any]:
    """Pack the Track B training stream plus a strictly held-out evaluation slice.

    The eval slice is taken from documents that come AFTER every document the train stream
    consumed, so no token appears in both.  Its default size is config.TRACKB_SCORE_TOKENS,
    which is the hard cap the dense-code Track B metrics can hold in host RAM -- a larger eval
    stream would only cost residual-extraction time that nothing can use.
    """
    spec = config.LM_CORPORA[corpus]
    raw = raw_dir or _snapshot_dir(spec.repo)
    n_train = int(train_tokens or config.LM_TRAIN_TOKENS)
    n_eval = int(eval_tokens or config.TRACKB_SCORE_TOKENS)
    column = spec.text_column
    # The recorded source name of the OpenWebText streams is "openwebtext" (other corpora record
    # their repo id); _stream_done compares it, so it must stay stable across executions.
    source = "openwebtext" if corpus == "owt" else spec.repo

    # Composite split names for every corpus but owt, matching config.lm_token_stream, so the
    # derived .meta.json path follows automatically and owt keeps its bare filenames.
    def _sp(base: str) -> str:
        return base if corpus == "owt" else f"{corpus}_{base}"

    OWT_TRAIN_SPLIT, OWT_VAL_SPLIT, OWT_TEST_SPLIT = _sp("train"), _sp("val"), _sp("test")

    expect_train = {"source": source, "n_tokens": n_train, "lm_repo": config.LM_REPO}
    expect_eval = {"source": source, "n_tokens": n_eval, "lm_repo": config.LM_REPO}
    train_ok = _stream_done(OWT_TRAIN_SPLIT, expect_train)
    eval_ok = _stream_done(OWT_VAL_SPLIT, expect_eval)
    test_ok = _stream_done(OWT_TEST_SPLIT, expect_eval)
    if train_ok and eval_ok and test_ok and not force:
        if verbose:
            print(f"[skip] {corpus}: {OWT_TRAIN_SPLIT} ({n_train} tok) and {OWT_VAL_SPLIT} "
                  f"({n_eval} tok) already packed")
        return {"train": _read_json(token_meta_path(OWT_TRAIN_SPLIT)),
                "val": _read_json(token_meta_path(OWT_VAL_SPLIT)),
                "test": _read_json(token_meta_path(OWT_TEST_SPLIT))}

    by_split = discover_source_files(raw)
    # OpenWebText ships a single unlabeled corpus; whichever split name its shards happen to
    # carry, every shard is train data for us.  Concatenating in sorted path order keeps the
    # document order fixed.
    files = sorted(sum((v for v in by_split.values()), []))
    if not files:
        raise MaterializeError(f"no {corpus} shards under {raw}")

    tok, eos, dtype, max_id, pad = load_pythia_tokenizer(tokenizer_dir)
    if dtype != np.dtype(np.uint16):
        raise MaterializeError(
            f"largest token id {max_id} needs {dtype}, but config.token_stream() names the file "
            "'.u16.npy'; change the registry before writing a wider dtype"
        )
    if verbose:
        print(f"[{corpus}] tokenizer max id {max_id} -> {dtype} (uint16 holds 0..65535), "
              f"eos={eos}, pad={pad}")

    encode = lambda txts: _encode_batch(tok, txts)
    results: Dict[str, Any] = {}
    t0 = time.time()

    # ---- train --------------------------------------------------------------------------
    train_meta = _read_json(token_meta_path(OWT_TRAIN_SPLIT)) if train_ok else None
    if train_ok and not force:
        if verbose:
            print(f"[skip] {corpus}/{OWT_TRAIN_SPLIT} already packed; resuming at document "
                  f"{train_meta['next_doc_index']}")
        results["train"] = train_meta
    else:
        toks, _, n_docs, n_full, next_doc = _pack_documents(
            _owt_doc_iter(files, column=column), encode, eos, dtype, n_train, with_docid=False)
        if toks.shape[0] < n_train:
            raise MaterializeError(
                f"{corpus} ran out after {toks.shape[0]} tokens, needed {n_train}; the raw "
                f"snapshot under {raw} is incomplete"
            )
        _save_npy_atomic(config.token_stream(OWT_TRAIN_SPLIT), toks)
        train_meta = {
            "format_version": FORMAT_VERSION, "source": source, "split": OWT_TRAIN_SPLIT,
            "lm_repo": config.LM_REPO, "n_tokens": int(toks.shape[0]), "dtype": str(dtype),
            "eos_token_id": eos, "pad_token_id": pad, "has_labels": False,
            "n_docs_consumed": n_docs, "n_docs_fully_packed": n_full,
            # THE resume point for the held-out slice: one past the last document touched.
            # Not the same as n_docs_consumed, which does not count skipped empty documents.
            "next_doc_index": next_doc,
            "raw_dir": raw, "n_source_files": len(files),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "packing": ("documents concatenated in raw shard order, one end-of-text token after "
                        "each, NO padding; the final document is truncated so the length is "
                        "exactly n_tokens (= config.TOTAL_STEPS * config.BATCH_SIZE for train)."),
            "pad_note": ("Pythia has pad_token_id=None.  The CE-recovered reference masks with "
                         "ignore_index=pad_token_id; aliasing pad to end-of-text would ignore "
                         "every genuine end-of-text target, so no padding is used at all."),
        }
        _write_json_atomic(token_meta_path(OWT_TRAIN_SPLIT), _jsonable(train_meta))
        results["train"] = train_meta
        if verbose:
            print(f"[{corpus}] {OWT_TRAIN_SPLIT}: {toks.shape[0]} tokens from {n_docs} documents "
                  f"({time.time() - t0:.0f}s)")

    # ---- held-out eval slices ------------------------------------------------------------
    # Two of them, val then test, each starting strictly after the previous one ended, so the
    # three streams are pairwise disjoint by construction rather than by assumption.
    skip = int(train_meta["next_doc_index"])
    for _split in (OWT_VAL_SPLIT, OWT_TEST_SPLIT):
      _done = _stream_done(_split, expect_eval)
      if _done and not force:
        results[_split] = _read_json(token_meta_path(_split))
        skip = int(results[_split].get("next_doc_index", skip))
      else:
        toks, _, n_docs, n_full, next_doc = _pack_documents(
            _owt_doc_iter(files, skip_docs=skip, column=column), encode, eos, dtype, n_eval,
            with_docid=False)
        if toks.shape[0] < n_eval:
            raise MaterializeError(
                f"{corpus} ran out after {toks.shape[0]} held-out tokens, needed {n_eval}"
            )
        _save_npy_atomic(config.token_stream(_split), toks)
        eval_meta = {
            "format_version": FORMAT_VERSION, "source": source, "split": _split,
            "lm_repo": config.LM_REPO, "n_tokens": int(toks.shape[0]), "dtype": str(dtype),
            "eos_token_id": eos, "pad_token_id": pad, "has_labels": False,
            "n_docs_consumed": n_docs, "n_docs_fully_packed": n_full,
            "next_doc_index": next_doc,
            "held_out_after_doc": skip, "raw_dir": raw,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "packing": ("same rules as the train stream; starts strictly after the last document "
                        "the train stream touched, so the two share no token."),
            "size_note": ("defaults to config.TRACKB_SCORE_TOKENS, the cap the dense-code Track B "
                          "metrics can hold in host RAM."),
        }
        _write_json_atomic(token_meta_path(_split), _jsonable(eval_meta))
        results[_split] = eval_meta
        skip = int(next_doc)
        if verbose:
            print(f"[{corpus}] {OWT_VAL_SPLIT}: {toks.shape[0]} held-out tokens "
                  f"(skipped {skip} train documents)")
    return results


# --------------------------------------------------------------------------- text-embedding stage
# Labelled corpora (AG News, GoEmotions) -> canonical jsonl + labels npy + split indices.

# An English subword tokenizer emits roughly 1.3 tokens per whitespace word, so a document with
# more than max_seq_length / 1.3 words is almost certainly truncated by the encoder.  Used only
# to report a truncation proxy in meta.json -- the real per-encoder tokenizer is not run here,
# because loading three tokenizers to produce a diagnostic would make this stage need the model
# snapshots it otherwise does not touch.
WORDS_PER_TOKEN = 0.75


# --------------------------------------------------------------------------- constants
# Canonical split names on disk.  The raw GoEmotions split is called "validation"; we rename it
# to "val" so every corpus has exactly the same three filenames and no downstream stage needs a
# per-corpus special case.
SPLITS: Tuple[str, ...] = ("train", "val", "test")


def _require_split(by_split: Dict[str, List[str]], split: str, raw_dir: str) -> List[str]:
    files = by_split.get(split)
    if not files:
        found = {k: [os.path.relpath(p, raw_dir) for p in v[:4]] for k, v in by_split.items()}
        raise MaterializeError(
            f"no {split!r} shards found under {raw_dir}.  Discovered: {json.dumps(found, indent=2)}"
        )
    return files


def _count_rows(files: Sequence[str]) -> List[int]:
    """Per-shard row counts.

    Parquet stores the count in its footer, so this is O(1) per file; jsonl needs a line pass,
    which is cheap for the corpus sizes used here.
    """
    pq = _ensure_pyarrow()
    counts = []
    for path in files:
        if path.lower().endswith(".parquet"):
            counts.append(int(pq.ParquetFile(path).metadata.num_rows))
        else:
            n = 0
            with _open_maybe_gzip(path) as fh:
                for line in fh:
                    if line.strip():
                        n += 1
            counts.append(n)
    return counts


# --------------------------------------------------------------------------- text and labels
def _assemble_text(row: Dict[str, Any], text_fields: Sequence[str]) -> str:
    """Join a row's text fields into the single string the encoder will see.

    Multiple fields are joined as `field1 + ". " + field2` without special-casing a field that
    already ends in punctuation (the shipped corpora have a single text field).  Empty/None
    parts are dropped rather than contributing a bare ". " separator.  A NULL parquet value becomes "" (counted as an empty document in meta.json)
    rather than crashing, because a single null must not abort a 560k-row materialisation.
    """
    parts = []
    for f in text_fields:
        v = row.get(f)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            parts.append(s)
    return ". ".join(parts)


def _coerce_single_label(value: Any, n_classes: int, where: str) -> int:
    """Validate and convert one single-label value.

    A label outside [0, n_classes) means the schema assumption is wrong (wrong config, wrong
    column), which would poison every purity/KNN/LP number downstream, so it raises with the
    offending value rather than clipping.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        # Some exporters store the label as a string digit; accept that but nothing looser.
        try:
            value = int(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise MaterializeError(f"{where}: label {value!r} is not an integer") from exc
    v = int(value)
    if not (0 <= v < n_classes):
        raise MaterializeError(
            f"{where}: label {v} out of range [0, {n_classes}) -- the corpus schema in "
            f"config.CORPORA disagrees with the raw file"
        )
    return v


def _coerce_multi_label(value: Any, n_classes: int, where: str) -> List[int]:
    """Validate a list-of-class-ids value (GoEmotions 'simplified')."""
    if value is None:
        return []
    if isinstance(value, (int, np.integer)):      # tolerate a degenerate single-id encoding
        value = [value]
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise MaterializeError(f"{where}: multi-label value {value!r} is not a list")
    out = []
    for v in value:
        out.append(_coerce_single_label(v, n_classes, where))
    return out


def _length_stats(char_lens: np.ndarray, word_lens: np.ndarray) -> Dict[str, Any]:
    """Truncation-relevant length distribution for one split.

    Reports both character and whitespace-word percentiles, plus, per text encoder, the
    fraction of documents that will almost certainly be truncated at that encoder's
    max_seq_length.  The word budget uses WORDS_PER_TOKEN as a proxy -- the exact figure needs
    the encoder's own tokenizer, which this stage deliberately does not load.  Limitation: the
    proxy under-counts truncation for code-like or non-English text, so treat it as a lower
    bound.
    """
    def pct(a: np.ndarray) -> Dict[str, float]:
        if a.size == 0:
            return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        q = np.percentile(a, [50, 90, 95, 99])
        return {
            "mean": float(a.mean()), "p50": float(q[0]), "p90": float(q[1]),
            "p95": float(q[2]), "p99": float(q[3]), "max": float(a.max()),
        }

    trunc = {}
    for enc_name in config.ENCODER_ORDER:
        enc = config.TEXT_ENCODERS[enc_name]
        budget_words = int(enc.max_seq_length * WORDS_PER_TOKEN)
        frac = float((word_lens > budget_words).mean()) if word_lens.size else 0.0
        trunc[enc_name] = {
            "max_seq_length": enc.max_seq_length,
            "word_budget_proxy": budget_words,
            "frac_docs_likely_truncated": frac,
        }
    return {
        "chars": pct(char_lens),
        "words": pct(word_lens),
        "truncation_proxy": trunc,
        "words_per_token_assumed": WORDS_PER_TOKEN,
    }


# --------------------------------------------------------------------------- stratified carve
def stratified_carve(labels: np.ndarray, n_val: int, n_classes: int, seed: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministically carve `n_val` validation rows out of a single-label train split.

    Allocation is largest-remainder (Hamilton) apportionment on the class frequencies, so the
    validation split matches the train class distribution as closely as an integer split can
    AND sums to exactly n_val -- a plain round() would over- or under-shoot and change the row
    count the audit checks.

    Determinism: `np.random.RandomState` (legacy Mersenne Twister) is used rather than
    `default_rng`, because numpy documents a permanent stream-compatibility guarantee for
    RandomState and explicitly does NOT give one for Generator.  The classes are visited in
    ascending id order so the RNG consumption order is fixed too.  Belt and braces: the caller
    persists the resulting indices in split_indices.json and reuses them on later runs, so even
    a hypothetical RNG change cannot move a document between train and val.

    Edge cases:
      * every class keeps at least one training row (quota is clamped to count - 1), and any
        deficit that clamping creates is redistributed to the classes with the most spare
        capacity, largest first;
      * a class with zero rows gets zero quota;
      * n_val > n_train - n_classes is impossible to satisfy under the clamp and raises.

    Returns (train_idx, val_idx), both sorted ascending, so the emitted jsonl preserves the raw
    row order minus the carved rows.
    """
    n = int(labels.shape[0])
    if not (0 < n_val < n):
        raise MaterializeError(f"n_val={n_val} is not strictly inside (0, {n})")

    counts = np.bincount(labels, minlength=n_classes).astype(np.int64)
    nonzero = int((counts > 0).sum())
    if n_val > n - nonzero:
        raise MaterializeError(
            f"cannot carve {n_val} validation rows from {n} train rows across {nonzero} "
            f"non-empty classes while leaving >=1 train row per class"
        )

    # ---- Hamilton apportionment ------------------------------------------------------------
    exact = counts.astype(np.float64) * (n_val / float(n))
    quota = np.floor(exact).astype(np.int64)
    remainder = exact - quota
    deficit = n_val - int(quota.sum())
    if deficit > 0:
        # Ties broken by ascending class id (np.lexsort is stable w.r.t. the secondary key),
        # which keeps the allocation reproducible when several classes share a remainder.
        order = np.lexsort((np.arange(n_classes), -remainder))
        for c in order[:deficit]:
            quota[c] += 1

    # ---- clamp so no class loses its entire training presence -------------------------------
    cap = np.maximum(counts - 1, 0)
    over = quota - np.minimum(quota, cap)
    quota = np.minimum(quota, cap)
    short = int(over.sum())
    if short:
        spare = cap - quota
        # Largest spare capacity first; ascending class id breaks ties.
        for c in np.lexsort((np.arange(n_classes), -spare)):
            if short <= 0:
                break
            take = min(short, int(spare[c]))
            quota[c] += take
            short -= take
        if short > 0:  # unreachable given the n_val guard above, kept as a loud invariant
            raise MaterializeError("stratified carve could not place all validation rows")

    # ---- sample within each class ----------------------------------------------------------
    rs = np.random.RandomState(seed)
    picks: List[np.ndarray] = []
    for c in range(n_classes):
        q = int(quota[c])
        if q <= 0:
            continue
        idx_c = np.flatnonzero(labels == c)
        picks.append(rs.choice(idx_c, size=q, replace=False))

    val_idx = np.sort(np.concatenate(picks)) if picks else np.zeros(0, dtype=np.int64)
    val_idx = val_idx.astype(np.int64)
    mask = np.ones(n, dtype=bool)
    mask[val_idx] = False
    train_idx = np.flatnonzero(mask).astype(np.int64)
    if val_idx.shape[0] != n_val:
        raise MaterializeError(f"carve produced {val_idx.shape[0]} val rows, expected {n_val}")
    return train_idx, val_idx


# --------------------------------------------------------------------------- corpus driver
def corpus_meta_path(corpus: str, out_dir: Optional[str] = None) -> str:
    base = out_dir or os.path.join(config.CORPUS_DIR, corpus)
    return os.path.join(base, "meta.json")


def _corpus_out_dir(corpus: str, out_dir: Optional[str]) -> str:
    return out_dir or os.path.join(config.CORPUS_DIR, corpus)


def _split_paths(corpus: str, split: str, out_dir: Optional[str]) -> Tuple[str, str]:
    """(jsonl, labels npy) for one split, honouring an out_dir override used by the self-test."""
    if out_dir is None:
        return config.corpus_jsonl(corpus, split), config.corpus_labels_npy(corpus, split)
    return os.path.join(out_dir, f"{split}.jsonl"), os.path.join(out_dir, f"{split}_labels.npy")


def is_corpus_done(corpus: str, out_dir: Optional[str] = None) -> bool:
    """Skip-guard: a corpus is done when meta.json is current AND every file it names exists.

    Keyed on meta.json because it is written LAST, so its presence proves the jsonl/npy pair for
    every split was already renamed into place.  The file-existence re-check costs three stats
    and catches a partially deleted tree.
    """
    d = _corpus_out_dir(corpus, out_dir)
    meta = _read_json(os.path.join(d, "meta.json"))
    if not meta or meta.get("format_version") != FORMAT_VERSION:
        return False
    if meta.get("corpus") != corpus:
        return False
    for split in SPLITS:
        jl, npy = _split_paths(corpus, split, out_dir)
        if not (os.path.exists(jl) and os.path.exists(npy)):
            return False
    return os.path.exists(os.path.join(d, "split_indices.json"))


def _load_or_make_split_indices(corpus: str, spec, labels_raw: np.ndarray,
                                out_dir: Optional[str]) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return (train_idx, val_idx, record) for the train carve, reusing a persisted carve.

    Reuse-if-present is the durable determinism guarantee: once split_indices.json exists, the
    train/val membership is fixed on disk and is identical for every encoder and every SAE seed
    regardless of the numpy version that later re-runs this stage.  A persisted file whose row
    count disagrees with the current raw data is refused rather than silently reused, since that
    means the raw snapshot changed underneath us.
    """
    path = (config.split_indices_json(corpus) if out_dir is None
            else os.path.join(out_dir, "split_indices.json"))
    n = int(labels_raw.shape[0])

    existing = _read_json(path)
    if existing and existing.get("n_train_raw") == n and existing.get("seed") == config.SPLIT_SEED:
        tr = np.asarray(existing["train"], dtype=np.int64)
        va = np.asarray(existing["val"], dtype=np.int64)
        if tr.shape[0] + va.shape[0] == n:
            return tr, va, existing
        raise MaterializeError(
            f"{path} is inconsistent: {tr.shape[0]} + {va.shape[0]} != {n} raw train rows"
        )
    if existing and existing.get("n_train_raw") not in (None, n):
        raise MaterializeError(
            f"{path} was built for {existing.get('n_train_raw')} raw train rows but the raw "
            f"snapshot now has {n}; delete it deliberately if the change is intended"
        )

    if spec.n_val_carve:
        if spec.multilabel:
            # Proper multi-label stratification needs iterative stratification (Sechidis et al.);
            # no multi-label corpus here needs a carve (GoEmotions ships one), so we
            # refuse loudly rather than ship a wrong approximation.
            raise NotImplementedError(
                f"{corpus} is multi-label and asks for a carve; iterative stratification is not "
                "implemented because no corpus in config.CORPORA needs it"
            )
        tr, va = stratified_carve(labels_raw, spec.n_val_carve, spec.n_classes, config.SPLIT_SEED)
        method = "stratified-largest-remainder"
    else:
        # Shipped validation: train is used whole.  Recorded uniformly so downstream code has one
        # shape to read.
        tr = np.arange(n, dtype=np.int64)
        va = np.zeros(0, dtype=np.int64)
        method = "shipped"

    record = {
        "corpus": corpus,
        "seed": config.SPLIT_SEED,
        "method": method,
        "n_train_raw": n,
        "n_train": int(tr.shape[0]),
        "n_val_carved": int(va.shape[0]),
        # Full index lists, which buy exact reproducibility and
        # makes the carve auditable with `jq`.
        "train": tr.tolist(),
        "val": va.tolist(),
    }
    _write_json_atomic(path, record)
    return tr, va, record


def _emit_split(corpus: str, split: str, files: Sequence[str], spec,
                keep: Optional[np.ndarray], out_dir: Optional[str]) -> Dict[str, Any]:
    """Stream `files`, keep the rows in `keep` (None = all), write jsonl + labels npy.

    `keep` is a SORTED array of raw row indices; membership is tested with a boolean mask so the
    cost is O(1) per row and the emitted order is the raw order (which is what makes the carve's
    "raw order minus the val rows" contract hold).

    The jsonl "id" field is the POSITION in this file, i.e. `id == line number == row of
    <split>_labels.npy == the LMDB key the extractor will write`.  Provenance (which raw rows a
    split came from) lives in split_indices.json, deliberately not in the id, so that no
    downstream stage can accidentally use a non-contiguous id as an array index.
    """
    columns = list(spec.text_fields) + [spec.label_field]
    jl_path, npy_path = _split_paths(corpus, split, out_dir)

    mask: Optional[np.ndarray] = None
    if keep is not None:
        total = int(keep.max()) + 1 if keep.size else 0
        mask = np.zeros(max(total, 1), dtype=bool)
        mask[keep] = True

    labels_out: List[Any] = []
    char_lens: List[int] = []
    word_lens: List[int] = []
    n_empty = 0
    n_written = 0

    with _atomic_text(jl_path) as fh:
        for raw_i, row in enumerate(_iter_rows(files, columns)):
            if mask is not None and not (raw_i < mask.shape[0] and mask[raw_i]):
                continue
            text = _assemble_text(row, spec.text_fields)
            if not text:
                n_empty += 1
            where = f"{corpus}/{split} raw row {raw_i}"
            if spec.multilabel:
                labels_out.append(_coerce_multi_label(row[spec.label_field], spec.n_classes, where))
            else:
                labels_out.append(_coerce_single_label(row[spec.label_field], spec.n_classes, where))
            # ensure_ascii=False keeps UTF-8 as UTF-8 (smaller files, greppable); json.dumps
            # still escapes the newlines that would otherwise break the one-object-per-line rule.
            fh.write(json.dumps({"id": n_written, "text": text}, ensure_ascii=False))
            fh.write("\n")
            char_lens.append(len(text))
            word_lens.append(len(text.split()))
            n_written += 1

    if keep is not None and n_written != int(keep.shape[0]):
        raise MaterializeError(
            f"{corpus}/{split}: wrote {n_written} rows but the index list selects "
            f"{int(keep.shape[0])} -- the raw shard order or row count changed"
        )

    # ---- labels array ----------------------------------------------------------------------
    if spec.multilabel:
        # uint8 multi-hot: 28 classes x 1 byte is 28 B/row, and uint8 is what a downstream
        # BCE target or a Jaccard purity wants without a further cast.
        lab = np.zeros((n_written, spec.n_classes), dtype=np.uint8)
        for i, ids in enumerate(labels_out):
            if ids:
                lab[i, np.asarray(ids, dtype=np.int64)] = 1
        label_summary = {
            "dtype": "uint8", "shape": [int(lab.shape[0]), int(lab.shape[1])],
            "per_class_count": [int(v) for v in lab.sum(axis=0)],
            "mean_labels_per_doc": float(lab.sum(axis=1).mean()) if n_written else 0.0,
            "n_docs_with_no_label": int((lab.sum(axis=1) == 0).sum()),
        }
    else:
        lab = np.asarray(labels_out, dtype=np.int64)
        label_summary = {
            "dtype": "int64", "shape": [int(lab.shape[0])],
            "per_class_count": [int(v) for v in np.bincount(lab, minlength=spec.n_classes)],
        }
    _save_npy_atomic(npy_path, lab)

    stats = _length_stats(np.asarray(char_lens, dtype=np.int64),
                          np.asarray(word_lens, dtype=np.int64))
    return {
        "n": n_written,
        "jsonl": jl_path,
        "labels_npy": npy_path,
        "n_empty_text": n_empty,
        "labels": label_summary,
        "text_length": stats,
    }


def materialize_corpus(corpus: str, raw_dir: Optional[str] = None,
                       out_dir: Optional[str] = None, force: bool = False,
                       strict_counts: bool = True, verbose: bool = True) -> Dict[str, Any]:
    """Materialise one labelled corpus into jsonl + labels npy + split_indices + meta.

    `strict_counts=False` is for the CPU self-test ONLY; it is recorded in meta.json so a
    downstream reader can tell that the declared row counts were not enforced.
    """
    if corpus not in config.CORPORA:
        raise MaterializeError(f"unknown corpus {corpus!r}; known: {list(config.CORPORA)}")
    spec = config.CORPORA[corpus]
    d = _corpus_out_dir(corpus, out_dir)
    config.assert_safe_text_path(d)
    meta_path = os.path.join(d, "meta.json")

    if not force and is_corpus_done(corpus, out_dir):
        if verbose:
            print(f"[skip] {corpus}: meta.json v{FORMAT_VERSION} present and complete")
        return _read_json(meta_path) or {}

    # assets.local_snapshot is the ONLY place that knows where prepare_lang_assets.sh put a
    # repo; deriving the path here independently is how the two halves drifted apart the first
    # time (this file assumed RAW_DIR/<corpus key>, the downloader wrote
    # RAW_DIR/datasets/<org>__<name>, and every corpus failed at discover_source_files).
    raw = raw_dir or _snapshot_dir(config.CORPORA[corpus].repo)
    by_split = discover_source_files(raw)
    t0 = time.time()

    train_files = _require_split(by_split, "train", raw)
    test_files = _require_split(by_split, "test", raw)
    val_files = by_split.get("val") or []
    if spec.n_val_carve == 0 and not val_files:
        raise MaterializeError(
            f"{corpus} declares n_val_carve=0 (i.e. it ships a validation split) but no "
            f"validation shard was found under {raw}"
        )
    if spec.n_val_carve > 0 and val_files:
        raise MaterializeError(
            f"{corpus} declares a carve of {spec.n_val_carve} rows but the raw snapshot already "
            f"ships a validation split ({[os.path.basename(p) for p in val_files]}); "
            "config.CORPORA and the snapshot disagree"
        )
    if by_split.get("_unclassified"):
        raise MaterializeError(
            f"{corpus}: could not infer the split of "
            f"{[os.path.relpath(p, raw) for p in by_split['_unclassified']]}"
        )

    # ---- pass 1: labels only, to compute the carve ------------------------------------------
    # Reading one narrow column is far cheaper than a full scan, and it lets the carve be decided
    # before a single byte of output is written.
    lab_col = [spec.label_field]
    if spec.multilabel:
        raw_train_labels = np.zeros(0, dtype=np.int64)          # unused: no carve for multi-label
        n_train_raw = sum(_count_rows(train_files))
    else:
        vals = [_coerce_single_label(r[spec.label_field], spec.n_classes, f"{corpus}/train row {i}")
                for i, r in enumerate(_iter_rows(train_files, lab_col))]
        raw_train_labels = np.asarray(vals, dtype=np.int64)
        n_train_raw = int(raw_train_labels.shape[0])

    if strict_counts and n_train_raw != spec.n_train_raw:
        raise MaterializeError(
            f"{corpus}: raw train has {n_train_raw} rows, config.CORPORA declares "
            f"{spec.n_train_raw}.  A silent size mismatch would poison every downstream metric."
        )

    train_idx, val_idx, split_record = _load_or_make_split_indices(
        corpus, spec, raw_train_labels if not spec.multilabel else np.zeros(n_train_raw, np.int64),
        out_dir,
    )

    # ---- pass 2: emit -----------------------------------------------------------------------
    out: Dict[str, Any] = {}
    if spec.n_val_carve:
        out["train"] = _emit_split(corpus, "train", train_files, spec, train_idx, out_dir)
        out["val"] = _emit_split(corpus, "val", train_files, spec, val_idx, out_dir)
        val_source = "carved"
    else:
        out["train"] = _emit_split(corpus, "train", train_files, spec, None, out_dir)
        out["val"] = _emit_split(corpus, "val", val_files, spec, None, out_dir)
        val_source = "shipped"
    out["test"] = _emit_split(corpus, "test", test_files, spec, None, out_dir)

    # ---- row-count validation ---------------------------------------------------------------
    expected = {"train": spec.n_train, "val": spec.n_val, "test": spec.n_test}
    problems = [f"{s}: got {out[s]['n']}, config declares {expected[s]}"
                for s in SPLITS if out[s]["n"] != expected[s]]
    if problems:
        msg = f"{corpus}: row-count mismatch -- " + "; ".join(problems)
        if strict_counts:
            raise MaterializeError(
                msg + "\nRefusing to declare this corpus done: a silent size mismatch would "
                "poison every downstream metric."
            )
        print(f"[WARN] {msg}  (strict_counts=False)", file=sys.stderr)

    # ---- provenance -------------------------------------------------------------------------
    src = []
    for split, files in (("train", train_files), ("val", val_files), ("test", test_files)):
        for p, n in zip(files, _count_rows(files)):
            src.append({
                "split": split, "path": os.path.relpath(p, raw), "rows": int(n),
                "bytes": int(os.path.getsize(p)), "sha256": _sha256_file(p),
            })

    meta = {
        "format_version": FORMAT_VERSION,
        "corpus": corpus,
        "repo": spec.repo,
        "hf_config": spec.config,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "multilabel": spec.multilabel,
        "n_classes": spec.n_classes,
        "text_fields": list(spec.text_fields),
        "text_assembly": (". ".join(f"<{f}>" for f in spec.text_fields)),
        "label_field": spec.label_field,
        "strict_counts": bool(strict_counts),
        "val_source": val_source,
        "val_note": ("GoEmotions ships its own validation split; it is used verbatim and nothing "
                     "is carved from train."
                     if val_source == "shipped" else
                     f"stratified carve of {spec.n_val_carve} rows from train with "
                     f"config.SPLIT_SEED={config.SPLIT_SEED}; a validation split is MANDATORY "
                     "because the sae/ trainers only write best_val_monoscore_model.pt inside "
                     "their `if val_loader:` branch."),
        "split_seed": config.SPLIT_SEED,
        "split_method": split_record["method"],
        "expected_counts": expected,
        "splits": out,
        "source_files": src,
        "raw_dir": raw,
    }
    _write_json_atomic(meta_path, _jsonable(meta))
    if verbose:
        print(f"[done] {corpus}: " + "  ".join(f"{s}={out[s]['n']}" for s in SPLITS)
              + f"  ({meta['elapsed_s']}s)")
    return meta



def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpora", help="materialise the labelled corpora (text-embedding stage)")
    c.add_argument("--corpus", default="all", choices=list(config.CORPUS_ORDER) + ["all"])
    c.add_argument("--raw-dir", default=None,
                   help="override the raw snapshot dir written by prepare_assets.sh")
    c.add_argument("--out-dir", default=None,
                   help=f"override the output dir (default {config.CORPUS_DIR}/<corpus>)")
    c.add_argument("--force", action="store_true", help="ignore the skip-guard and rebuild")
    c.add_argument("--no-strict-counts", action="store_true",
                   help="DEBUG ONLY: warn instead of raising on a row-count mismatch; the "
                        "relaxation is recorded in meta.json")
    p = sub.add_parser("pack", help="pack one training corpus into train/val/test token streams")
    p.add_argument("--lm-corpus", default="owt", choices=list(config.LM_CORPUS_ORDER))
    p.add_argument("--raw-dir", default=None,
                   help="override the snapshot directory written by prepare_assets.sh")
    p.add_argument("--tokenizer-dir", default=None,
                   help="override the local tokenizer snapshot (default: the pinned Pythia one)")
    p.add_argument("--train-tokens", type=int, default=None,
                   help=f"default config.LM_TRAIN_TOKENS ({config.LM_TRAIN_TOKENS:,})")
    p.add_argument("--eval-tokens", type=int, default=None,
                   help=f"default config.TRACKB_SCORE_TOKENS ({config.TRACKB_SCORE_TOKENS:,})")
    p.add_argument("--force", action="store_true", help="ignore the skip-guard and repack")
    a = ap.parse_args(argv)
    if a.cmd == "corpora":
        names = list(config.CORPUS_ORDER) if a.corpus == "all" else [a.corpus]
        for name in names:
            materialize_corpus(name, raw_dir=a.raw_dir, out_dir=a.out_dir,
                               force=a.force, strict_counts=not a.no_strict_counts)
        return 0
    pack_owt(raw_dir=a.raw_dir, train_tokens=a.train_tokens, eval_tokens=a.eval_tokens,
             tokenizer_dir=a.tokenizer_dir, force=a.force, corpus=a.lm_corpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
