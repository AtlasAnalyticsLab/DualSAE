#!/usr/bin/env python
"""Offline end-to-end smoke test of the hypothesis-generation pipeline on synthetic data.

No downloads, no GPU, no real models: tiny SAEs are trained here with the repository's vision/
trainers (through topic_modeling's driver flags, sizes shrunk), the ImageNet-style images,
embeddings, manifests and labels are synthetic, and vLLM is replaced by tests/stubs/vllm (canned
VLM answers). HypotheSAEs must be installed (prepare_assets.sh); it is used as is.

Exercised: dump_activations -> select_features (3 rules) -> evaluate proxy -> interpret_latents ->
evaluate faithful (HG_MODE=main), one selmethod cell (HG_MODE=selmethod, correlation),
build_results, encoder_figs (HG_ENCODERS=clip).

Run: python hypothesis_generation/tests/test_smoke.py [--workdir DIR]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HG_DIR = Path(__file__).resolve().parents[1]
REPO = HG_DIR.parent
TM_SRC = REPO / "topic_modeling" / "src"
STUBS = HG_DIR / "tests" / "stubs"
ACT, N_CLASSES, N_TR, N_VA = 16, 4, 20, 10
WNIDS = [f"n0000000{c}" for c in range(N_CLASSES)]
TRAINERS = ["TopKTrainer", "BatchTopKTrainer", "JumpReluTrainer",
            "DualTopKTrainer", "DualBatchTopKTrainer", "DualJumpReluTrainer"]
ACTS = [("topk", "TopKTrainer", "DualTopKTrainer"), ("batch_topk", "BatchTopKTrainer", "DualBatchTopKTrainer"),
        ("jump_relu", "JumpReluTrainer", "DualJumpReluTrainer")]


def write_lmdb(path: Path, rows: int, seed: int):
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


def make_split(img_root: Path, emb_dir: Path, split: str, per_class: int, seed: int):
    """ImageFolder images + the Stage-1a files (manifest, labels, embeddings) for one split.
    Embeddings carry a class-dependent direction so the selection stage has signal to find."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    centers = np.random.default_rng(99).normal(size=(N_CLASSES, ACT))
    manifest, embs = [], []
    for c, wnid in enumerate(WNIDS):
        (img_root / wnid).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            rel = f"{wnid}/{split}_{i:03d}.png"
            Image.fromarray((rng.random((20, 20, 3)) * 255).astype(np.uint8)).save(img_root / rel)
            manifest.append({"rel_path": rel, "label": c})
            embs.append(centers[c] + 0.5 * rng.normal(size=ACT))
    e = np.asarray(embs, dtype=np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    (emb_dir / "clip").mkdir(parents=True, exist_ok=True)
    (emb_dir / f"{split}_manifest.json").write_text(json.dumps(manifest))
    np.save(emb_dir / f"{split}_labels.npy", np.array([m["label"] for m in manifest], dtype=np.int16))
    np.save(emb_dir / "clip" / f"{split}_emb.npy", e.astype(np.float16))


def run(cmd, env):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-6000:])
        raise SystemExit(f"FAILED: {' '.join(str(c) for c in cmd)}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="hg_smoke_", dir=a.workdir))
    feats, tm_out, hg_out = work / "features", work / "tm_out", work / "hg_out"
    write_lmdb(feats / "train" / "clip_image_features_CLIP-ViT-L-14-DataComp.lmdb", 256, 1)
    write_lmdb(feats / "validation" / "clip_image_features_CLIP-ViT-L-14-DataComp.lmdb", 96, 2)

    env = dict(os.environ)
    env.update(TM_OUT=str(tm_out), HG_OUT=str(hg_out), FEATURES_IMAGENET=str(feats),
               IMAGENET_TRAIN_DIR=str(work / "img_train"), IMAGENET_VAL_DIR=str(work / "img_val"),
               HG_DATASET="imagenet", HG_MODE="main", HG_ENCODERS="clip", FAITHFUL_CLASSES="3",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               WANDB_MODE="disabled", MPLBACKEND="Agg",
               PYTHONPATH=os.pathsep.join([str(STUBS), str(HG_DIR), str(TM_SRC), env.get("PYTHONPATH", "")]))
    py = sys.executable

    # 1) tiny SAEs with the exact training flags of the reported runs (sizes shrunk)
    sys.path.insert(0, str(TM_SRC))
    os.environ.update({k: env[k] for k in ("TM_OUT", "FEATURES_IMAGENET")})
    import tm_driver as D
    reg = D.load_registry()
    shrink = {"--n_latents": "64", "--batch_size": "32", "--num_epochs": "2", "--topk_k": "4",
              "--num_workers": "0", "--warmup_steps": "8"}
    for tr in TRAINERS:
        run_name = "imagenet_paired_span001" if D.is_dual(tr) else "imagenet_paired"
        w = 17500 if D.is_dual(tr) else None
        argv = D.train_argv(reg, run_name, "clip", tr, 42, w)
        argv = [shrink.get(argv[i - 1], x) if i > 0 else x for i, x in enumerate(argv)] + ["--device", "cpu"]
        D.stage_dir(run_name, reg["runs"][run_name], "clip", "trained_models", tr, 42, w).parent.mkdir(
            parents=True, exist_ok=True)
        run([py] + argv, env)

    # 2) Stage 1a files (synthetic), then Stages 1b-4 per cell
    make_split(work / "img_train", hg_out / "embeddings", "train", N_TR, 3)
    make_split(work / "img_val", hg_out / "embeddings", "val", N_VA, 4)
    for act, _, _ in ACTS:
        for var in ("baseline", "dual"):
            cell = ["--encoder", "clip", "--activation", act, "--variant", var, "--seed", "42"]
            for split in ("train", "val"):
                run([py, "-m", "hypogen.dump_activations"] + cell + ["--split", split, "--device", "cpu"], env)
            run([py, "-m", "hypogen.select_features"] + cell, env)
            run([py, "-m", "hypogen.evaluate", "--stage", "proxy"] + cell, env)
            run([py, "-m", "hypogen.interpret_latents"] + cell + ["--vlm-batch", "8", "--device", "cpu"], env)
            run([py, "-m", "hypogen.evaluate", "--stage", "faithful"] + cell, env)
    # one selmethod cell (correlation rule; seeds its hypotheses from the lasso run as 02_cells.sh does)
    sm = dict(env, HG_MODE="selmethod", HG_SEL_METHOD="correlation")
    for var in ("baseline", "dual"):
        cell = ["--encoder", "clip", "--activation", "topk", "--variant", var, "--seed", "42"]
        main_hyp = hg_out / "interpreted" / "clip" / "topk" / var / "seed42" / "hypotheses.json"
        sm_int = hg_out / "selmethod_correlation" / "interpreted" / "clip" / "topk" / var / "seed42"
        sm_int.mkdir(parents=True, exist_ok=True)
        (sm_int / "hypotheses.json").write_text(main_hyp.read_text())
        run([py, "-m", "hypogen.select_features"] + cell, sm)
        run([py, "-m", "hypogen.interpret_latents"] + cell + ["--vlm-batch", "8", "--device", "cpu"], sm)
        run([py, "-m", "hypogen.evaluate", "--stage", "faithful"] + cell, sm)

    # 3) tables + figure
    run([py, "-m", "hypogen.build_results"], env)
    import pandas as pd
    t = pd.read_csv(hg_out / "tables" / "hypogen_long.csv")
    got = t[(t.dataset == "imagenet") & (t.encoder == "clip")]
    assert set(got.rule) == {"lasso", "correlation"}, set(got.rule)
    assert got[got.rule == "lasso"].delta_auroc.notna().sum() == 3, got
    run([py, "-m", "hypogen.encoder_figs", "--seed", "42", "--out", hg_out / "figures"], env)
    assert (hg_out / "figures" / "encoder_clip.pdf").is_file()
    print(f"SMOKE OK (outputs in {work})")


if __name__ == "__main__":
    main()
