"""CB stage 0 (preflight): supervisor embeddings + the LMDB row alignment
gate. Run once per machine, before any CB stage (cb_sae/prepare_assets.sh
runs --embeds; the drivers run --img-embeds / --check-alignment).

Three independent subcommands (run in this order):

  1. --embeds          Text embeddings of the concept vocabularies + the
                       ImageNet class names, via open_clip encode_text of the
                       SAME checkpoint that produced the clip LMDB features
                       (cb_config.OPEN_CLIP_MODEL). BARE strings for the
                       vocabs (CLIP-Dissect / LF-CBM protocol); the class
                       names get one "a photo of a {c}." template (zero-shot
                       sanity only, never used for concepts).
                       Needs open_clip_torch and the model weights (the only
                       step that downloads anything; afterwards it also runs
                       with HF_HUB_OFFLINE=1 from the local HF cache).
  2. --img-embeds      Dump the probe-split clip rows (via the featbsf loader,
                       whose L2-normalization makes them unit image
                       embeddings) + their labels (ImageNet-1K only).
  3. --check-alignment The gate everything downstream refuses to run without:
                       ImageNet-1K:
                       (a) FULL per-index __label__ agreement across the
                           three encoder LMDBs, train + validation, read RAW
                           from each LMDB (bypassing any offline label-file
                           fallback, which would compare a file with itself);
                       (b) zero-shot top-1 of the dumped image embeds against
                           the class-name text embeds — the same-space test
                           (a wrong text checkpoint scores ~0.1%, the right
                           one >=~70%; threshold 0.55).
                       OpenImagesV7 (multi-label): row counts + a
                       representational-similarity row-correspondence test.
                       Writes ALIGNMENT_OK[_<dataset>].json on success.

Rationale (why labels, why zero-shot): the ImageNet-1K LMDBs store NO
__path__ keys, so per-index image identity cannot be checked directly.
Labels prove class-level alignment; single-pipeline provenance (all three
stores written by one extraction over one dataset order) covers the
within-class permutation residual. The zero-shot test separately proves the
TEXT side lives in the image features' space (same-width cross-space products
fail silently).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from featbsf import cb_config as CB
from featbsf import config as C


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load_vocab(path: Path) -> list[str]:
    """One concept per line, blanks dropped, order preserved (concept index =
    line rank — the concepts.json contract). Same as bsf/cb_dissect.load_vocab
    (re-implemented so this module does not import the bsf package)."""
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _classnames() -> list[str]:
    """The 1000 ImageNet class names in canonical class-index order (matches
    the HF imagenet-1k labels the LMDBs store), from open_clip's own
    zero-shot metadata (the same list the original run read from a copy of
    that file; --embeds already requires open_clip)."""
    from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES
    names = list(IMAGENET_CLASSNAMES)
    assert len(names) == 1000, len(names)
    return names


def _write_sidecar(npy_path: Path, meta: dict) -> None:
    side = npy_path.with_suffix(".json")
    tmp = side.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(side)


def _atomic_np_save(path: Path, arr: np.ndarray) -> None:
    """np.save via tmp+rename — a killed run (e.g. a CPU-time limit) must
    never leave a torn .npy behind a skip-guard."""
    tmp = path.with_suffix(f".tmp{os.getpid()}.npy")
    np.save(tmp, arr)
    tmp.replace(path)


def _cache_complete(npy_path: Path, *extra: Path) -> bool:
    """A cache is complete only when the npy(s) AND the sidecar exist — the
    sidecar is written LAST, so its presence is the completion marker. Guarding
    on the npy alone wedges permanently if the run dies between the two
    writes."""
    return (npy_path.exists() and npy_path.with_suffix(".json").exists()
            and all(p.exists() for p in extra))


def read_sidecar(npy_path: Path) -> dict:
    return json.loads(npy_path.with_suffix(".json").read_text())


# ---------------------------------------------------------------------------
# --embeds (needs open_clip)
# ---------------------------------------------------------------------------
def build_text_embeds(device: str) -> None:
    import torch
    import open_clip

    # Bound threads: on a shared CPU host unbounded MKL threading can burn a
    # CPU-time limit many times faster than wall time.
    torch.set_num_threads(min(8, os.cpu_count() or 8))

    model, _, _ = open_clip.create_model_and_transforms(CB.OPEN_CLIP_MODEL)
    tokenizer = open_clip.get_tokenizer(CB.OPEN_CLIP_MODEL)
    model = model.to(device).eval()

    @torch.no_grad()
    def encode(strings: list[str]) -> np.ndarray:
        outs = []
        for s in range(0, len(strings), 256):
            toks = tokenizer(strings[s:s + 256]).to(device)
            e = model.encode_text(toks).float()
            e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            outs.append(e.cpu().numpy().astype(np.float16))
        return np.concatenate(outs, axis=0)

    CB.ensure_cb_dirs()
    # Concept vocabularies: BARE strings (protocol) — the dissect vocab plus
    # every concept set's user vocab.
    vocabs = [CB.DISSECT_VOCAB] + [cs["file"] for cs in
                                   CB.CONCEPT_SETS.values()]
    for vocab in vocabs:
        out = CB.text_embed_path(vocab)
        if _cache_complete(out):
            print(f"[skip] {out.name}")
            continue
        strings = _load_vocab(vocab)
        emb = encode(strings)
        assert emb.shape == (len(strings), CB.SUPERVISOR_DIM), emb.shape
        _atomic_np_save(out, emb)
        _write_sidecar(out, {"encoder": CB.SUPERVISOR_TAG,
                             # File name only (portable); sha16 pins content.
                             "vocab": vocab.name, "sha16": _sha16(vocab),
                             "n": len(strings), "bare_strings": True})
        print(f"[write] {out.name}  ({len(strings)} x {CB.SUPERVISOR_DIM})")

    # Class names for the zero-shot sanity gate ONLY. One template (the
    # 80-template ensemble is overkill for a pass/fail gate and 80x the CPU).
    out = CB.CB_EMBED_DIR / "text_clipL14dc_imagenet_classnames.npy"
    if not _cache_complete(out):
        names = _classnames()
        emb = encode([f"a photo of a {n}." for n in names])
        _atomic_np_save(out, emb)
        _write_sidecar(out, {"encoder": CB.SUPERVISOR_TAG,
                             "vocab": "IMAGENET_CLASSNAMES",
                             "template": "a photo of a {c}.", "n": len(names)})
        print(f"[write] {out.name}")
    else:
        print(f"[skip] {out.name}")


CLASSNAME_EMBEDS = CB.CB_EMBED_DIR / "text_clipL14dc_imagenet_classnames.npy"


# ---------------------------------------------------------------------------
# --img-embeds
# ---------------------------------------------------------------------------
def build_img_embeds(dataset: str = CB.CB_DATASET) -> None:
    import torch  # noqa: F401  (featbsf.data pulls it in)
    from featbsf import data as D

    CB.ensure_cb_dirs()
    out = CB.img_embed_path(dataset=dataset)
    lab_out = out.with_name(out.stem + "_labels.npy")
    # OIv7 is multilabel: scalar labels only ever served the imagenet
    # zero-shot gate, so its dump is embeds-only.
    with_labels = dataset == "imagenet"
    if _cache_complete(out, lab_out) if with_labels else \
            (out.exists() and out.with_suffix(".json").exists()):
        print(f"[skip] {out.name}")
        return
    loader = D.build_loader(dataset, "clip", CB.probe_split(dataset),
                            batch_size=8192,
                            labels="class" if with_labels else "none",
                            shuffle=False, num_workers=4)
    feats, labels = [], []
    for batch in loader:
        if with_labels:
            x, y = batch
            labels.append(y.numpy().astype(np.int64))
        else:
            x = batch
        feats.append(x.numpy().astype(np.float16))
    emb = np.concatenate(feats, axis=0)
    _atomic_np_save(out, emb)
    if with_labels:
        _atomic_np_save(lab_out, np.concatenate(labels, axis=0))
    _write_sidecar(out, {"encoder": CB.SUPERVISOR_TAG,
                         "dataset": dataset,
                         "split": CB.probe_split(dataset),
                         "n": int(emb.shape[0]),
                         "l2_normalized_by_loader": True})
    print(f"[write] {out.name}  {emb.shape}"
          + (" + labels" if with_labels else " (no labels: multilabel)"))


# ---------------------------------------------------------------------------
# --check-alignment
# ---------------------------------------------------------------------------
def _raw_labels(lmdb_path: str, n: int) -> np.ndarray:
    """__label__{i} for i in [0, n), read RAW from the store. Missing keys are
    a hard failure — the whole point is to NOT fall back to the shared offline
    npy (which would compare a file with itself)."""
    import lmdb
    out = np.empty(n, dtype=np.int64)
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False,
                    meminit=False)
    with env.begin(write=False) as txn:
        for i in range(n):
            raw = txn.get(f"__label__{i}".encode())
            if raw is None:
                env.close()
                raise RuntimeError(
                    f"{lmdb_path}: no __label__{i} key — this store cannot "
                    f"support the raw label-agreement gate")
            out[i] = int(raw.decode())
    env.close()
    return out


def _raw_label_hashes_env(env, path_for_err: str, n: int) -> "np.ndarray":
    """Format-agnostic row-alignment probe: 64-bit hash of the RAW __label__
    payload bytes per index, read from an ALREADY-OPEN environment (python-
    lmdb caches environments per path; re-opening a closed one in the same
    process raises 'operate on closed object'). Exact byte equality across
    encoder stores is a STRONGER guarantee than the decoded-int comparison
    (works for OIv7's multihot payloads unchanged)."""
    import hashlib
    out = np.empty(n, dtype=np.int64)
    with env.begin(write=False) as txn:
        for i in range(n):
            raw = txn.get(f"__label__{i}".encode())
            if raw is None:
                raise RuntimeError(
                    f"{path_for_err}: no __label__{i} key — this store "
                    f"cannot support the raw label-agreement gate")
            out[i] = int.from_bytes(
                hashlib.blake2b(raw, digest_size=8).digest(), "little",
                signed=True)
    return out


def _check_alignment_oiv7(report: dict, t0: float) -> None:
    """OIv7 alignment gate. The three encoder feature LMDBs share ONE
    companion openimages_labels.lmdb per split, so per-encoder label
    comparison is vacuous — what must hold is that the FEATURE stores are
    row-aligned to the same image order. Checks:
      (a) row counts equal across the 3 feature stores AND the labels store;
      (b) representational-similarity gate: for 4096 sampled rows, the
          pairwise cosine-similarity structure of clip rows must correlate
          with vit/sigclip rows' (same images => shared neighborhood
          structure); a row-shuffled null calibrates the threshold. This is
          a content-based row-correspondence test needing no shared labels.
    Zero-shot is skipped (multilabel; supervisor same-space was proven on
    imagenet with the identical checkpoint)."""
    import lmdb as _l
    rng = np.random.default_rng(0)
    for split in ("train", CB.probe_split("open_images")):
        paths = {enc: C.LMDB_PATHS["open_images"][enc][split]
                 for enc in C.ENCODERS}
        counts = {}
        envs = {}
        for enc, p_ in paths.items():
            envs[enc] = _l.open(p_, readonly=True, lock=False,
                                readahead=False, meminit=False)
            with envs[enc].begin() as txn:
                counts[enc] = int(txn.get(b"__len__").decode())
        lab_p = str(Path(paths["clip"]).parent / "openimages_labels.lmdb")
        lab_env = _l.open(lab_p, readonly=True, lock=False)
        with lab_env.begin() as txn:
            raw = txn.get(b"__len__")
            counts["labels"] = (int(raw.decode()) if raw is not None
                                else txn.stat()["entries"])
        if len({counts[e] for e in C.ENCODERS}) != 1:
            raise SystemExit(f"row-count mismatch on {split}: {counts}")
        n = counts["clip"]
        report["splits"][split] = counts
        idx = np.sort(rng.choice(n, size=min(4096, n), replace=False))
        feats = {}
        for enc in C.ENCODERS:
            d_enc = C.FEATURE_DIM[enc]
            rows = np.empty((len(idx), d_enc), dtype=np.float32)
            with envs[enc].begin() as txn:
                for j, i in enumerate(idx):
                    rows[j] = np.frombuffer(
                        txn.get(str(int(i)).encode()), dtype=np.float32)
            rows /= np.linalg.norm(rows, axis=1, keepdims=True) + 1e-8
            feats[enc] = rows
            envs[enc].close()
        lab_env.close()
        iu = np.triu_indices(len(idx), k=1)
        s_clip = (feats["clip"] @ feats["clip"].T)[iu]
        for enc in ("vit", "sigclip"):
            s_e = (feats[enc] @ feats[enc].T)[iu]
            r = float(np.corrcoef(s_clip, s_e)[0, 1])
            perm = rng.permutation(len(idx))
            s_null = (feats[enc][perm] @ feats[enc][perm].T)[iu]
            r0 = float(np.corrcoef(s_clip, s_null)[0, 1])
            report["label_agreement"][f"rsa_clip-vs-{enc}/{split}"] = \
                {"corr": r, "shuffled_null": r0}
            print(f"  {split:>8} RSA clip vs {enc}: corr {r:.4f} "
                  f"(shuffled null {r0:.4f})")
            if not (r > 0.3 and r > 10 * abs(r0)):
                raise SystemExit(
                    f"RSA alignment gate FAILED for {enc}/{split}: corr "
                    f"{r:.4f} vs null {r0:.4f} — the feature stores do not "
                    f"share a row order. Do not proceed.")
    report["zero_shot_top1"] = None
    report["zero_shot_note"] = "skipped: multilabel; RSA gate used instead"
    report["wall_seconds"] = round(time.time() - t0, 1)
    ok = CB.alignment_ok_path("open_images")
    tmp = ok.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(ok)
    print(f"[write] {ok}")


def check_alignment(dataset: str = CB.CB_DATASET) -> None:
    import lmdb  # noqa: F401  (fail early with a clear error if absent)

    CB.ensure_cb_dirs()
    t0 = time.time()
    report: dict = {"supervisor": CB.SUPERVISOR_TAG,
                    "label_agreement": {}, "splits": {}}

    if dataset == "open_images":
        _check_alignment_oiv7(report, t0)
        return

    # (a) full per-index label agreement, both splits, all encoder pairs.
    for split in ("train", CB.probe_split(dataset)):
        paths = {enc: C.LMDB_PATHS[dataset][enc][split]
                 for enc in C.ENCODERS}
        # ONE env per path per split (count + label reads share it — see
        # _raw_label_hashes_env's caching note).
        import lmdb as _l
        envs = {enc: _l.open(p, readonly=True, lock=False, readahead=False,
                             meminit=False) for enc, p in paths.items()}
        counts = {}
        for enc, env in envs.items():
            with env.begin() as txn:
                counts[enc] = int(txn.get(b"__len__").decode())
        if len(set(counts.values())) != 1:
            raise SystemExit(f"row-count mismatch on {split}: {counts}")
        n = counts["clip"]
        report["splits"][split] = n

        def reader(enc):
            if dataset == "imagenet":
                return _raw_labels(paths[enc], n)
            return _raw_label_hashes_env(envs[enc], paths[enc], n)

        ref = reader("clip")
        for enc in C.ENCODERS:
            if enc == "clip":
                continue
            lab = reader(enc)
            agree = float((lab == ref).mean())
            report["label_agreement"][f"clip-vs-{enc}/{split}"] = agree
            print(f"  {split:>10} clip vs {enc}: agreement {agree:.6f} "
                  f"over {n} rows")
            if agree < 1.0:
                raise SystemExit(
                    f"LABEL MISMATCH clip vs {enc} on {split} "
                    f"(agreement {agree:.6f}) — the LMDBs are NOT row-aligned; "
                    f"cross-encoder y_pseudo would be wrong. Do not proceed.")
        for env in envs.values():
            env.close()

    # (b) zero-shot same-space gate (imagenet only: OIv7 is multilabel and
    # its row alignment rests on the raw-byte agreement above, which is the
    # stronger check; the supervisor same-space property was proven once on
    # imagenet and the checkpoint is identical).
    if dataset != "imagenet":
        report["zero_shot_top1"] = None
        report["zero_shot_note"] = "skipped: multilabel dataset"
        report["wall_seconds"] = round(time.time() - t0, 1)
        ok = CB.alignment_ok_path(dataset)
        tmp = ok.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(ok)
        print(f"[write] {ok}")
        return
    img_p = CB.img_embed_path()
    if not img_p.exists():
        raise SystemExit("run --img-embeds first (validation image embeds "
                         "missing)")
    side = read_sidecar(img_p)
    assert side["encoder"] == CB.SUPERVISOR_TAG, side
    if not CLASSNAME_EMBEDS.exists():
        raise SystemExit("run --embeds first (class-name text embeds missing)")
    tside = read_sidecar(CLASSNAME_EMBEDS)
    assert tside["encoder"] == CB.SUPERVISOR_TAG, tside
    img = np.load(img_p).astype(np.float32)
    lab = np.load(img_p.with_name(img_p.stem + "_labels.npy"))
    txt = np.load(CLASSNAME_EMBEDS).astype(np.float32)
    pred = (img @ txt.T).argmax(axis=1)
    top1 = float((pred == lab).mean())
    report["zero_shot_top1"] = top1
    print(f"  zero-shot top-1 (1 template): {top1:.4f}")
    if top1 < 0.55:
        raise SystemExit(
            f"zero-shot top-1 {top1:.4f} < 0.55 — the text embeddings are NOT "
            f"in the image features' space (wrong checkpoint?). Do not "
            f"proceed; y_pseudo would be noise.")

    report["wall_seconds"] = round(time.time() - t0, 1)
    ok = CB.alignment_ok_path(dataset)
    tmp = ok.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(ok)
    print(f"[write] {ok}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--embeds", action="store_true",
                    help="build vocab + class-name text embeds (needs "
                         "open_clip)")
    ap.add_argument("--img-embeds", action="store_true",
                    help="dump probe-split clip image embeds")
    ap.add_argument("--check-alignment", action="store_true",
                    help="label-agreement + zero-shot gates -> "
                         "ALIGNMENT_OK.json")
    ap.add_argument("--device", default=None,
                    help="for --embeds; default: cuda if available else cpu")
    ap.add_argument("--dataset", default=CB.CB_DATASET,
                    choices=list(CB.PROBE_SPLITS),
                    help="OIv7 generalization: which dataset's probe split "
                         "to dump/check")
    args = ap.parse_args()
    if not (args.embeds or args.img_embeds or args.check_alignment):
        ap.error("pick at least one of --embeds / --img-embeds / "
                 "--check-alignment")

    if args.embeds:
        import torch
        dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        build_text_embeds(dev)
    if args.img_embeds:
        build_img_embeds(args.dataset)
    if args.check_alignment:
        check_alignment(args.dataset)


if __name__ == "__main__":
    main()
