#!/usr/bin/env python3
"""Decorrelation-loss ablation (supplementary "Decorrelation loss" table): train DualSAE without
L_dcor (gamma = span_coef = 0) and evaluate it, for TopK, BatchTopK and JumpReLU Sparse Branches,
CLIP / SigLIP2 / ViT, seeds 42-44.

Two protocols, as run for the reported table:

* ImageNet-1K: the ablated arm replays each main-table DualSAE run (per-cell settings from
  ../configs/main_dual_cells.json: gamma, warm-up, lr, normalisation, dead-latent check interval)
  with only --span_coef set to 0. The reference arm is the main-table run itself (its
  branch_metrics results, see ../branch_metrics/run_branch_metrics.py).
* OpenImagesV7: a matched pair trained here, gamma in {0, 0.01}, both at warm-up 2500 and lr 1e-4,
  differing in gamma only. (These runs used the trainer's JumpReLU threshold init 0.001, which is
  passed explicitly below.)

Stages:  train   train every outstanding run (resumable: finished runs are skipped)
         eval    score every finished run with branch_metrics.py (resumable)
         print   print the training commands

    python run_dcor.py --stage train [--only-dataset open_images] [--index i | --shard i/n]
    python run_dcor.py --stage eval
Then build the table with analysis/ablations/make_dcor_table.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ABL_DIR = _HERE.parent
VISION_DIR = ABL_DIR.parents[1]
sys.path.insert(0, str(ABL_DIR))
import feature_paths  # noqa: E402

ROOT = Path(os.environ.get("ABL_ROOT", "./outputs/ablations")).resolve() / "dcor"
ENCODERS = ("clip", "sigclip", "vit")
ACTIVATIONS = ("topk", "batch_topk", "jump_relu")
SEEDS = (42, 43, 44)
OI_WARMUP, OI_LR, OI_SPANS = 2500, 1e-4, (0.0, 0.01)


def main_cells():
    payload = json.loads((ABL_DIR / "configs" / "main_dual_cells.json").read_text())
    return payload["common"], payload["jump_relu_threshold_init"], payload["cells"]


def imagenet_runs():
    """span 0 replays of the main-table ImageNet-1K DualSAE runs."""
    common, theta, cells = main_cells()
    runs = []
    for c in cells:
        if c["dataset"] != "imagenet":
            continue
        exp = f"dcor_imagenet_{c['encoder']}_{c['activation']}_span0_seed{c['seed']}"
        argv = ["--model", "dual", "--sparsity_mode", c["activation"],
                "--dataset", "imagenet", "--extractor", c["encoder"],
                *feature_paths.path_args("imagenet", c["encoder"]),
                "--seed", str(c["seed"]), "--n_latents", str(common["n_latents"]),
                "--topk_k", str(common["top_k"]), "--batch_size", str(common["batch_size"]),
                "--num_epochs", str(common["num_epochs"]), "--lr", repr(c["lr"]),
                "--eps", repr(common["eps"]), "--l1_coef", repr(common["l1_coef"]),
                "--bandwidth", repr(common["bandwidth"]), "--mono_coef", repr(common["mono_coef"]),
                "--mono_period", str(common["mono_period"]), "--sim_coef", repr(common["sim_coef"]),
                # The variable under study; every other setting is the main-table run's.
                "--span_coef", "0.0",
                "--warmup_steps", str(c["warmup_steps"]),
                "--warmup_check_interval", str(common["warmup_check_interval"]),
                "--poly_lr_factor", repr(common["poly_lr_factor"]),
                "--poly_hidden", str(common["poly_hidden"]),
                "--dead_steps_threshold", str(common["dead_steps_threshold"]),
                "--dead_check_interval", str(c["dead_check_interval"]),
                "--num_workers", str(common["num_workers"])]
        if c["normalize"]:
            argv.append("--normalize")
        if c["activation"] == "jump_relu":
            # Threshold init theta used by the main-table JumpReLU runs.
            argv += ["--jump_threshold_init", repr(theta)]
        runs.append({"dataset": "imagenet", "encoder": c["encoder"], "activation": c["activation"],
                     "seed": c["seed"], "span": 0.0, "exp_name": exp, "argv": argv})
    return runs


def openimages_runs():
    """Matched pair gamma in {0, 0.01} at warm-up 2500 (arguments as used for the reported runs)."""
    runs = []
    for enc in ENCODERS:
        for act in ACTIVATIONS:
            for span in OI_SPANS:
                for seed in SEEDS:
                    exp = f"dcor_open_images_{enc}_{act}_warm{OI_WARMUP}_span{span:g}_seed{seed}"
                    argv = ["--model", "dual", "--sparsity_mode", act, "--dataset", "open_images",
                            "--extractor", enc, *feature_paths.path_args("open_images", enc),
                            "--seed", str(seed), "--lr", f"{OI_LR:g}", "--span_coef", f"{span:g}",
                            "--warmup_steps", str(OI_WARMUP), "--mono_coef", "0.0",
                            "--n_latents", "8192", "--topk_k", "64", "--batch_size", "2048",
                            "--num_epochs", "50", "--poly_hidden", "512", "--poly_layers", "1",
                            "--poly_lr_factor", "1",
                            # value used for the reported runs (main.py default is 1000)
                            "--dead_check_interval", "100"]
                    if act == "jump_relu":
                        argv += ["--normalize", "--l1_coef", "0.0001", "--bandwidth", "0.001",
                                 # value used for the reported runs (theta = 1e-3)
                                 "--jump_threshold_init", "0.001"]
                    runs.append({"dataset": "open_images", "encoder": enc, "activation": act,
                                 "seed": seed, "span": span, "exp_name": exp, "argv": argv})
    return runs


def all_runs(args):
    runs = imagenet_runs() + openimages_runs()
    out = []
    for r in runs:
        if args.only_dataset and r["dataset"] != args.only_dataset:
            continue
        if args.only_activation and r["activation"] != args.only_activation:
            continue
        out.append(r)
    return out


def is_complete(run_dir: Path) -> bool:
    return (run_dir / "autoencoder_final.pt").exists() and (run_dir / "best_val_monoscore_model.pt").exists()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["train", "eval", "print"], required=True)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--only-dataset", choices=["imagenet", "open_images"], default=None)
    ap.add_argument("--only-activation", choices=ACTIVATIONS, default=None)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()
    ckpt_dir = args.root.resolve() / "checkpoints"
    env = dict(os.environ)
    env.setdefault("WANDB_MODE", "disabled")

    if args.stage == "eval":
        manifest = args.root / "dcor_runs.json"
        subprocess.run([sys.executable, str(ABL_DIR / "make_run_manifest.py"), "--checkpoints",
                        str(ckpt_dir), "--pattern", "dcor_*", "--out", str(manifest)], check=True)
        cmd = [sys.executable, str(ABL_DIR / "branch_metrics" / "run_branch_metrics.py"),
               "--manifest", str(manifest), "--out-dir", str(args.root / "branch_metrics" / "results"),
               "--num-workers", str(args.num_workers)]
        if args.index is not None:
            cmd += ["--index", str(args.index)]
        if args.shard:
            cmd += ["--shard", args.shard]
        return subprocess.run(cmd, env=env).returncode

    runs = all_runs(args)
    if args.index is not None:
        runs = runs[args.index:args.index + 1]   # index into the full, stable list
    todo = [r for r in runs if not is_complete(ckpt_dir / r["exp_name"])]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        todo = todo[i::n]
    print(f"{len(todo)} run(s) to train")
    for r in todo:
        cmd = [sys.executable, "main.py", *r["argv"], "--exp_name", r["exp_name"],
               "--output_dir", str(ckpt_dir), "--tb_dir", str(args.root.resolve() / "tensorboard" / r["exp_name"]),
               "--skip_existing", "--skip_epoch_checkpoints"]
        print("$ " + " ".join(cmd), flush=True)
        if args.stage == "print":
            continue
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(cmd, cwd=VISION_DIR, env=env)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
