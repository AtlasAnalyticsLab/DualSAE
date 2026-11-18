#!/usr/bin/env python
"""Offline end-to-end smoke test of the topic-modeling pipeline on synthetic data.

No downloads, no GPU, no real models: the vision features, captions, images, word vectors and the
LLM tokenizer are generated here, and vLLM is replaced by tests/stubs/vllm (canned answers). What is
exercised is the real code path:
  tm_driver train   -> vision/main.py (TopK baseline + Dual-TopK, driver flags, tiny sizes)
  tm_driver interpret / topics -> patched SAE-TM interpret_sae.py --old-impl, sae_to_topic.py
  tm_driver evaluate -> patched SAE-TM evaluation.py (diversity, coherence, intruder)
  build_topic_table.py, annotate_topics.py, judge_img2topic.py, visualize_topics.py (img2topic-all)

Requires the patched SAE-TM checkout (setup_saetm.sh) at $SAE_TM_DIR.
Run: SAE_TM_DIR=<checkout> python topic_modeling/tests/test_smoke.py [--workdir DIR]
"""

import argparse
import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TM_DIR = Path(__file__).resolve().parents[1]
REPO = TM_DIR.parent
SRC = TM_DIR / "src"
STUBS = TM_DIR / "tests" / "stubs"
sys.path.insert(0, str(SRC))

ACT, ROWS_TR, ROWS_VA, N_DOCS = 16, 256, 96, 96
VOCAB = ("dog cat bird fish tree grass water sky car road house window red blue green "
         "metal wood stone light shadow").split()


def write_lmdb(path: Path, rows: int, seed: int):
    """Tiny feature store in the vision LMDBFeatureDataset format (L2-normalised low-rank rows)."""
    import lmdb
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(rows, 6)) @ rng.normal(size=(6, ACT)) + 0.05 * rng.normal(size=(rows, ACT))
    data = (data / np.linalg.norm(data, axis=1, keepdims=True)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=64 * 1024 * 1024, subdir=True)
    with env.begin(write=True) as txn:
        txn.put(b"__len__", str(rows).encode())
        txn.put(b"__shape__", str((ACT,)).encode())
        for i in range(rows):
            txn.put(str(i).encode(), data[i].tobytes())
            txn.put(f"__label__{i}".encode(), str(i % 5).encode())
    env.close()


def make_caption_side(side: Path, img_root: Path):
    """documents.jsonl, caption-aligned embeddings (emb_clip/emb_000.pt), BoW in SAE-TM's S2 format,
    and one small image per document."""
    import torch
    import scipy.sparse as sp
    from PIL import Image
    rng = np.random.default_rng(0)
    (side / "data").mkdir(parents=True, exist_ok=True)
    (side / "emb_clip").mkdir(parents=True, exist_ok=True)
    (side / "bow").mkdir(parents=True, exist_ok=True)
    rows, cols, vals, ids = [], [], [], []
    with open(side / "data" / "documents.jsonl", "w") as f:
        for d in range(N_DOCS):
            words = list(rng.choice(VOCAB, size=8))
            rel = f"n0000000{d % 5}/img_{d:04d}.png"
            ids.append(rel)
            f.write(json.dumps({"document": " ".join(words), "id": rel}) + "\n")
            for w in set(words):
                rows.append(d); cols.append(VOCAB.index(w)); vals.append(float(words.count(w)))
            p = img_root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((rng.random((24, 24, 3)) * 255).astype(np.uint8)).save(p)
    emb = rng.normal(size=(N_DOCS, ACT)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    torch.save(torch.from_numpy(emb), side / "emb_clip" / "emb_000.pt")
    bow = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(N_DOCS, len(VOCAB)))
    sp.save_npz(side / "bow" / "bow_bow.npz", bow)
    meta = {"ids": ids, "num_docs": N_DOCS, "vocab_size": len(VOCAB),
            "sparse_matrix_file": str((side / "bow" / "bow_bow.npz").resolve()),
            "vocab": {"id2token": VOCAB, "doc_freq": [1] * len(VOCAB), "term_freq": [1] * len(VOCAB),
                      "num_docs": N_DOCS}}
    (side / "bow" / "bow_meta.json").write_text(json.dumps(meta))


def make_word_vectors(assets: Path):
    """Tiny word2vec: binary .gz for sae_to_topic.py and embeddings.np.npy/vocabulary.json for
    evaluation.py (same layout prepare_assets.sh builds)."""
    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(len(VOCAB), 8)).astype(np.float32)
    gz = assets / "gensim-data" / "w2v.bin.gz"
    gz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz, "wb") as f:
        f.write(f"{len(VOCAB)} 8\n".encode())
        for w, v in zip(VOCAB, vecs):
            f.write(w.encode() + b" " + v.tobytes() + b"\n")
    (assets / "w2v").mkdir(parents=True, exist_ok=True)
    np.save(assets / "w2v" / "embeddings.np.npy", vecs)
    (assets / "w2v" / "vocabulary.json").write_text(json.dumps(VOCAB))
    return gz


def make_tokenizer(path: Path):
    """Local chat tokenizer standing in for the judge model's (only apply_chat_template is used)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    tok = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")
    fast.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    fast.save_pretrained(str(path))


def run(cmd, env, cwd=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], env=env, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-6000:])
        raise SystemExit(f"FAILED: {' '.join(str(c) for c in cmd)}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    saetm = os.environ.get("SAE_TM_DIR")
    if not saetm or not Path(saetm, ".dualsae_patch_applied").is_file():
        raise SystemExit("set SAE_TM_DIR to a checkout prepared by setup_saetm.sh")
    work = Path(tempfile.mkdtemp(prefix="tm_smoke_", dir=a.workdir))
    tm_out, assets = work / "out", work / "assets"
    feats = work / "features"
    write_lmdb(feats / "train" / "clip_image_features_CLIP-ViT-L-14-DataComp.lmdb", ROWS_TR, 1)
    write_lmdb(feats / "validation" / "clip_image_features_CLIP-ViT-L-14-DataComp.lmdb", ROWS_VA, 2)
    make_caption_side(tm_out / "imagenet_val", work / "images")
    gz = make_word_vectors(assets)
    make_tokenizer(assets / "judge_tokenizer")

    # Registry: the real one, restricted to one tiny cell pair and two topic counts.
    reg = json.loads((TM_DIR / "configs" / "runs.json").read_text())
    reg["seeds"] = [42]
    reg["topic_counts"] = [3, 4]
    reg["runs"] = {"imagenet_paired": dict(reg["runs"]["imagenet_paired"],
                                           trainers=["TopKTrainer", "DualTopKTrainer"], warmups=[8])}
    reg["baseline_run"] = {"imagenet": "imagenet_paired"}
    (work / "runs.json").write_text(json.dumps(reg))

    env = dict(os.environ)
    env.update(TM_OUT=str(tm_out), TM_REGISTRY=str(work / "runs.json"), FEATURES_IMAGENET=str(feats),
               W2V_PATH=str(gz), SAE_TM_W2V_DIR=str(assets / "w2v"), JUDGE_MODEL=str(assets / "judge_tokenizer"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               WANDB_MODE="disabled", MPLBACKEND="Agg",
               PYTHONPATH=os.pathsep.join([str(STUBS), str(SRC), env.get("PYTHONPATH", "")]))
    py = sys.executable

    # 1) training through the driver's exact argv, with only the sizes shrunk for the smoke test.
    import tm_driver as D
    os.environ.update({k: env[k] for k in ("TM_OUT", "TM_REGISTRY", "FEATURES_IMAGENET")})
    regd = D.load_registry(work / "runs.json")
    shrink = {"--n_latents": "64", "--batch_size": "32", "--num_epochs": "2", "--topk_k": "4",
              "--num_workers": "0", "--warmup_steps": "8"}
    for tr in ("TopKTrainer", "DualTopKTrainer"):
        w = 8 if D.is_dual(tr) else None
        argv = D.train_argv(regd, "imagenet_paired", "clip", tr, 42, w)
        argv = [shrink.get(argv[i - 1], x) if i > 0 else x for i, x in enumerate(argv)] + ["--device", "cpu"]
        Path(D.stage_dir("imagenet_paired", regd["runs"]["imagenet_paired"], "clip", "trained_models",
                         tr, 42, w)).parent.mkdir(parents=True, exist_ok=True)
        run([py] + argv, env)
    # 2) interpret + topics through the driver (training already done -> skipped by its guard)
    for tr in ("TopKTrainer", "DualTopKTrainer"):
        run([py, SRC / "tm_driver.py", "cell", "--run", "imagenet_paired", "--encoder", "clip",
             "--trainer", tr, "--seed", "42"], env)
    # 3) the three SAE-TM metrics (vLLM stubbed)
    run([py, SRC / "tm_driver.py", "evaluate", "--run", "imagenet_paired", "--encoders", "clip"], env)
    ev = tm_out / "runs" / "imagenet_paired" / "clip" / "evaluation"
    for tr in ("TopKTrainer", "DualTopKTrainer"):
        for k in (3, 4):
            for m in ("diversity", "coherence", "intruder"):
                assert (ev / tr / "seed42" / f"K{k}" / f"{m}.jsonl").is_file(), (tr, k, m)
    # 4) table builder
    run([py, SRC / "build_topic_table.py"], env)
    rows = list(csv.DictReader(open(tm_out / "tables" / "topic_table_long.csv")))
    got = [r for r in rows if r["encoder"] == "clip" and r["activation"] == "TopK" and r["delta"]]
    assert len(got) == 2 * 3, f"expected 6 filled TopK/clip rows (2 K x 3 metrics), got {len(got)}"
    # 5) figure pipeline: labels, judge, render (seed 42, K = 4)
    cells = [tm_out / "runs" / "imagenet_paired" / "clip" / "topics" / tr / "seed42" / "K4"
             for tr in ("TopKTrainer", "DualTopKTrainer")]
    run([py, SRC / "annotate_topics.py", "--model", env["JUDGE_MODEL"], "--topics-cells"] + cells, env)
    docs = tm_out / "imagenet_val" / "data" / "documents.jsonl"
    rd = tm_out / "runs" / "imagenet_paired"
    run([py, SRC / "judge_img2topic.py", "--model", env["JUDGE_MODEL"], "--results-dir", rd, "--documents", docs,
         "--seed", "42", "--k", "4", "--n-candidates", "40", "--encoders", "clip", "--activations", "TopK"], env)
    run([py, SRC / "visualize_topics.py", "--figures", "img2topic-all", "--results-dir", rd, "--documents", docs,
         "--image-root", work / "images", "--seed", "42", "--k", "4", "--img-cols", "3",
         "--all-encoders", "clip", "--all-activations", "TopK", "--out-dir", tm_out / "figures"], env)
    assert (tm_out / "figures" / "img2topic_all_K4.pdf").is_file()
    print(f"SMOKE OK (outputs in {work})")


if __name__ == "__main__":
    main()
