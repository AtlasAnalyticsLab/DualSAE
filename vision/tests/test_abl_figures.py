#!/usr/bin/env python3
"""Smoke test of the qualitative-figure scripts on synthetic data (CPU, no downloads).

Tiny models are trained through main.py on a synthetic LMDB (16-d features, 5 classes, image
paths stored in the LMDB), local PNG tiles stand in for the dataset images (--image_root), and
every figure script then runs end to end: motivation (top-50 purity + retrieval grid), matched
latents (activating-image grid + per-latent curves), t-SNE of sparse codes, branch t-SNE,
input-reconstruction alignment and the decorrelation retrieval figure.

Run with:  python -m pytest vision/tests/test_abl_figures.py -q
"""

import os
import subprocess
import sys
from pathlib import Path

import lmdb
import numpy as np
import pytest
from PIL import Image

VISION_DIR = Path(__file__).resolve().parents[1]
ANA = VISION_DIR / "analysis" / "ablations"
ACT, ROWS_TRAIN, ROWS_VAL = 16, 256, 120


def write_lmdb(path: Path, rows: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(6, ACT))
    labels = np.arange(rows) % 5
    data = rng.normal(size=(rows, 6)) @ basis + 2.0 * np.eye(5, ACT)[labels] + 0.05 * rng.normal(size=(rows, ACT))
    data = (data / np.linalg.norm(data, axis=1, keepdims=True)).astype(np.float32)
    env = lmdb.open(str(path), map_size=64 * 1024 * 1024, subdir=True)
    with env.begin(write=True) as txn:
        txn.put(b"__len__", str(rows).encode())
        txn.put(b"__shape__", str((ACT,)).encode())
        for i in range(rows):
            txn.put(str(i).encode(), data[i].tobytes())
            txn.put(f"__label__{i}".encode(), str(int(labels[i])).encode())
            txn.put(f"__path__{i}".encode(), f"{i}.png".encode())
    env.close()
    return str(path)


def run(args, **kw):
    env = {**os.environ, "MPLBACKEND": "Agg", "WANDB_MODE": "disabled", "OMP_NUM_THREADS": "2"}
    res = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, env=env, **kw)
    assert res.returncode == 0, (res.stdout[-1500:], res.stderr[-3000:])
    return res.stdout


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("figs")
    train = write_lmdb(root / "train.lmdb", ROWS_TRAIN, 1)
    val = write_lmdb(root / "val.lmdb", ROWS_VAL, 2)
    images = root / "images"
    images.mkdir()
    for i in range(ROWS_VAL):
        Image.new("RGB", (8, 8), ((37 * i) % 255, (91 * i) % 255, 120)).save(images / f"{i}.png")
    out = root / "runs"
    common = ["--dataset", "imagenet", "--extractor", "clip", "--train_path", train, "--val_path", val,
              "--n_latents", "64", "--topk_k", "4", "--batch_size", "32", "--num_epochs", "2",
              "--lr", "0.001", "--device", "cpu", "--num_workers", "0", "--output_dir", out, "--seed", "0"]
    runs = {
        "btk": ["--model", "batch_topk"],
        "jr": ["--model", "jumprelu", "--normalize", "--l1_coef", "0.001", "--jump_threshold_init", "1.0"],
        "dual_btk0": ["--model", "dual", "--sparsity_mode", "batch_topk", "--span_coef", "0.0",
                      "--warmup_steps", "4", "--skip_epoch_checkpoints"],
        "dual_btk": ["--model", "dual", "--sparsity_mode", "batch_topk", "--span_coef", "2.5",
                     "--warmup_steps", "4", "--skip_epoch_checkpoints"],
        "dual_tk": ["--model", "dual", "--sparsity_mode", "topk", "--warmup_steps", "4"],  # keeps epoch snapshots
        "dual_jr": ["--model", "dual", "--sparsity_mode", "jump_relu", "--normalize", "--l1_coef", "0.001",
                    "--jump_threshold_init", "1.0", "--warmup_steps", "4", "--skip_epoch_checkpoints"],
    }
    for name, extra in runs.items():
        run([VISION_DIR / "main.py", *common, *extra, "--exp_name", name], cwd=VISION_DIR)
    return {"root": root, "val": val, "images": images, "runs": out}


def best(world, name):
    return world["runs"] / name / "best_val_monoscore_model.pt"


def test_motivation(world):
    out = world["root"] / "motivation"
    final = world["runs"] / "btk" / "autoencoder_final.pt"
    run([ANA / "motivation" / "top50_class_purity.py", "--checkpoint", final, "--lmdb", world["val"],
         "--topk", "5", "--device", "cpu", "--output-dir", out])
    assert (out / "top50_class_purity_overlay_histogram.pdf").exists()
    run([ANA / "motivation" / "retrieval_grid.py", "--checkpoint", final, "--lmdb", world["val"],
         "--queries", "1", "--image-root", world["images"], "--result-dir", out])
    assert (out / "checkpoint_autoencoder_final_query_1.png").exists()


def test_matched_latents(world):
    out = world["root"] / "matched"
    run([ANA / "latent_activation_grid.py", "--lmdb_path", world["val"], "--batch_topk_ckpt", best(world, "btk"),
         "--dual_ckpt", best(world, "dual_btk"), "--num_latents", "2", "--min_active_samples", "1", "--k", "2",
         "--image_root", world["images"], "--disable_hf_fallback", "--output", out / "latent_grid.png",
         "--monoscore_plot_path", out / "ms.pdf", "--class_purity_plot_path", out / "purity.pdf"])
    assert (out / "ms.pdf").exists() and (out / "purity.pdf").exists()
    assert list(out.rglob("*_curves.npz"))


def test_tsne_sparse_codes(world):
    out = world["root"] / "tsne_sparse"
    run([ANA / "tsne_sparse_codes.py", "--pair", best(world, "dual_jr"), best(world, "jr"), "--lmdb", world["val"],
         "--random-classes", "--num-classes", "3", "--samples-per-class", "10", "--color-superclass-level", "-1",
         "--perplexity", "5", "--device", "cpu", "--output-dir", out])
    assert list(out.glob("*_tsne.png"))


def test_branch_tsne(world):
    out = world["root"] / "branch_tsne"
    snap = world["runs"] / "dual_tk" / "epoch_checkpoints" / "epoch_002.pt"
    assert snap.exists()
    run([ANA / "tsne_branches.py", "--checkpoints", snap, "--lmdb", world["val"], "--num-classes", "3",
         "--num-samples", "30", "--perplexity", "5", "--device", "cpu", "--output-dir", out])
    assert list(out.glob("*_tsne.png"))


def test_recon_alignment(world):
    out = world["root"] / "recon"
    run([ANA / "tsne_x_xhat.py", "--checkpoints", best(world, "btk"), "--lmdb", world["val"],
         "--num-samples", "20", "--perplexity", "5", "--device", "cpu", "--model-name", "BatchTopK SAE",
         "--output-dir", out])
    assert list(out.glob("*_x_xhat_tsne.png"))


def test_span_retrieval(world):
    out = world["root"] / "span"
    run([ANA / "span_retrieval.py", "--checkpoint-no-span", best(world, "dual_btk0"),
         "--checkpoint-with-span", best(world, "dual_btk"), "--lmdb-path", world["val"],
         "--image-root", world["images"], "--query-start", "1", "--query-end", "1", "--result-dir", out])
    assert (out / "compare_span_query_1.png").exists()
