#!/usr/bin/env python
"""Plain, resumable driver for the SAE-as-a-topic-model study.

One *cell* = (run, encoder, trainer, seed[, warm-up]). For each cell the driver runs, in order:

  1. train   : `vision/main.py` with the exact hyper-parameters of the reported run
               -> <TM_OUT>/runs/<run>/<enc>/trained_models/<Trainer>[/warmup<W>]/seed<S>/best_val_monoscore_model.pt
  2. interpret: SAE-TM `interpret_sae.py --old-impl` on the caption-aligned validation embeddings
               -> .../interpreted/<Trainer>[/warmup<W>]/seed<S>/word_emission_probabilities.pt
  3. topics  : SAE-TM `sae_to_topic.py` for every K in configs/runs.json:topic_counts
               -> .../topics/<Trainer>[/warmup<W>]/seed<S>/K<K>/top_words.txt
and, as a separate command, SAE-TM `evaluation.py` (diversity, coherence, intruder; all three
default SAE-TM metrics) over a run's topics tree -> .../evaluation/... (mirrors topics/).

Every stage is skipped when its output already exists, so re-running a command continues an
interrupted grid. Nothing is downloaded at runtime: run prepare_assets.sh and setup_saetm.sh
first. All locations come from environment variables with relative defaults (see `paths()`).

Usage:
  python topic_modeling/src/tm_driver.py list     --run imagenet_paired
  python topic_modeling/src/tm_driver.py cell     --run imagenet_paired --encoder clip --trainer DualTopKTrainer --seed 42
  python topic_modeling/src/tm_driver.py grid     --run imagenet_warmup_sweep [--encoders clip] [--seeds 42]
  python topic_modeling/src/tm_driver.py evaluate --run imagenet_paired [--encoders clip] [--seed 42]
Add --dry-run to print the commands instead of running them.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

TM_DIR = Path(__file__).resolve().parents[1]          # <repo>/topic_modeling
REPO_DIR = TM_DIR.parent                              # <repo>
# $TM_REGISTRY points at an alternative registry (used by the smoke test); default configs/runs.json.
REGISTRY_PATH = Path(os.environ.get("TM_REGISTRY", str(TM_DIR / "configs" / "runs.json")))

DUAL_MODE = {"DualTopKTrainer": "topk", "DualBatchTopKTrainer": "batch_topk",
             "DualJumpReluTrainer": "jump_relu"}
JUMPRELU_TRAINERS = {"JumpReluTrainer", "DualJumpReluTrainer"}


# ── configuration ────────────────────────────────────────────────────────────
def load_registry(path: Path = REGISTRY_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def paths() -> Dict[str, str]:
    """All filesystem locations, from environment variables with relative defaults."""
    return {
        "tm_out": os.environ.get("TM_OUT", "./outputs/topic_modeling"),
        "saetm": os.environ.get("SAE_TM_DIR", str(TM_DIR / "third_party" / "SAE-TM")),
        "vision": os.environ.get("VISION_DIR", str(REPO_DIR / "vision")),
        "w2v_gz": os.environ.get(
            "W2V_PATH", "./assets/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz"),
        "w2v_dir": os.environ.get("SAE_TM_W2V_DIR", "./assets/w2v"),
        "judge": os.environ.get("JUDGE_MODEL", "microsoft/phi-4"),
    }


def is_dual(trainer: str) -> bool:
    return trainer in DUAL_MODE


def cell_seg(run_cfg: dict, trainer: str, warmup: Optional[int]) -> str:
    """Path segment below <stage>/: '<Trainer>' or '<Trainer>/warmup<W>' (warm-up sweep runs only)."""
    if run_cfg["warmup_layer"] and is_dual(trainer):
        return f"{trainer}/warmup{warmup}"
    return trainer


def run_root(run: str) -> Path:
    return Path(paths()["tm_out"]) / "runs" / run


def stage_dir(run: str, run_cfg: dict, enc: str, stage: str, trainer: str, seed: int,
              warmup: Optional[int]) -> Path:
    return run_root(run) / enc / stage / cell_seg(run_cfg, trainer, warmup) / f"seed{seed}"


def caption_side(reg: dict, dataset: str) -> Path:
    return Path(paths()["tm_out"]) / reg["datasets"][dataset]["caption_side"]


def iter_cells(reg: dict, run: str, encoders=None, trainers=None, seeds=None, warmups=None):
    cfg = reg["runs"][run]
    for enc in (encoders or list(reg["encoders"])):
        for tr in (trainers or cfg["trainers"]):
            if tr not in cfg["trainers"]:
                continue
            for seed in (seeds or reg["seeds"]):
                if is_dual(tr):
                    for w in (warmups or cfg["warmups"]):
                        if w in cfg["warmups"]:
                            yield enc, tr, seed, w
                else:
                    yield enc, tr, seed, None


# ── stage commands ───────────────────────────────────────────────────────────
def train_argv(reg: dict, run: str, enc: str, trainer: str, seed: int, warmup: Optional[int]) -> List[str]:
    """`vision/main.py` argv reproducing the reported run's training configuration.

    The reported runs called the standalone trainer CLIs (train_topk.py / train_other.py /
    train_dual.py) with the flags below; `main.py` builds the identical TrainingConfig when the
    extra flags marked "value used for the reported runs" are passed (checked by
    tests/test_train_config.py).
    """
    cfg = reg["runs"][run]
    ds = reg["datasets"][cfg["dataset"]]
    feat_root = Path(os.environ.get(ds["features_env"], ds["features_default"]))
    lmdb = reg["encoders"][enc]["lmdb"]
    models_parent = stage_dir(run, cfg, enc, "trained_models", trainer, seed, warmup).parent
    argv = [
        str(Path(paths()["vision"]) / "main.py"),
        "--train_path", str(feat_root / ds["train_split"] / lmdb),
        "--val_path", str(feat_root / ds["val_split"] / lmdb),   # val split: best-epoch MonoScore selection only
        "--dataset", ds["main_dataset_flag"], "--extractor", reg["encoders"][enc]["extractor_flag"],
        "--n_latents", "8192", "--batch_size", "2048", "--num_epochs", "50",
        "--lr", "1e-4", "--eps", "6.25e-10", "--seed", str(seed), "--num_workers", "8",
        "--skip_extra_eval",
        "--output_dir", str(models_parent), "--exp_name", f"seed{seed}",
        "--wandb_project", "",                                   # no W&B logging
        # Multi-label targets were inferred from the dataset path for the reported runs; stated
        # explicitly here so the result does not depend on how the user names directories.
        "--multilabel" if ds["multilabel"] else "--no_multilabel",
        "--mono_coef", "0",                                      # no MonoLoss term for any SAE
    ]
    if trainer == "TopKTrainer":
        argv += ["--model", "topk", "--topk_k", "64", "--dead_check_interval", "100",
                 # value used for the reported runs: the standalone TopK trainer did not
                 # standardise inputs (main.py's 'auto' would turn it on for --model topk).
                 "--force_normalize", "off"]
    elif trainer == "BatchTopKTrainer":
        argv += ["--model", "batch_topk", "--topk_k", "64"]
    elif trainer == "JumpReluTrainer":
        argv += ["--model", "jumprelu", "--l1_coef", "1e-4", "--bandwidth", "1e-3",
                 # value used for the reported runs (theta initialised at 0.001)
                 "--jump_threshold_init", "0.001"]
    elif is_dual(trainer):
        argv += ["--model", "dual", "--sparsity_mode", DUAL_MODE[trainer], "--topk_k", "64",
                 "--warmup_steps", str(warmup), "--poly_hidden", "512",
                 "--warmup_check_interval", "500", "--dead_check_interval", "100",
                 "--span_coef", str(cfg["span_coef"]),
                 # IO only: the per-epoch snapshots under epoch_checkpoints/ are never read.
                 "--skip_epoch_checkpoints"]
        if trainer == "DualJumpReluTrainer":
            argv += ["--bandwidth", "1e-3",
                     "--jump_threshold_init", "0.001"]           # value used for the reported runs
        if warmup in cfg.get("degenerate_warmups", []):
            # The reported run had warm-up >= its total step budget (the Residual Branch never
            # trains). Reproduced as it was; see configs/runs.json and the README.
            argv += ["--allow_degenerate_warmup"]
    else:
        raise ValueError(f"unknown trainer {trainer}")
    return argv


def interpret_argv(reg, run, enc, trainer, seed, warmup) -> List[str]:
    cfg = reg["runs"][run]
    side = caption_side(reg, cfg["dataset"])
    emb_dir = side / f"emb_{enc}"
    shards = sorted(str(p.resolve()) for p in emb_dir.glob("*.pt")
                    if p.name.startswith("emb_")) or [str((emb_dir / "emb_000.pt").resolve())]
    best = stage_dir(run, cfg, enc, "trained_models", trainer, seed, warmup) / "best_val_monoscore_model.pt"
    out = stage_dir(run, cfg, enc, "interpreted", trainer, seed, warmup)
    return (["interpret_sae.py", "--old-impl", "--checkpoint", str(best.resolve()), "--sae-type", trainer,
             "--embeddings"] + shards +
            ["--bow-dataset", str((side / "bow" / "bow_meta.json").resolve()),
             "--save-path", str(out.resolve()), "--num-epochs", "50", "--batch-size", "1024",
             "--save-full-theta"])


def topics_argv(reg, run, enc, trainer, seed, warmup, k) -> List[str]:
    cfg = reg["runs"][run]
    side = caption_side(reg, cfg["dataset"])
    interp = stage_dir(run, cfg, enc, "interpreted", trainer, seed, warmup)
    out = stage_dir(run, cfg, enc, "topics", trainer, seed, warmup) / f"K{k}"
    return ["sae_to_topic.py", "--sae-results-path", str(interp.resolve()),
            "--vocab-path", str((side / "bow" / "bow_meta.json").resolve()),
            "--num-clusters", str(k), "--save-path", str(out.resolve())]


# ── execution ────────────────────────────────────────────────────────────────
def _env() -> dict:
    env = dict(os.environ)
    p = paths()
    # interpret_sae.py --old-impl imports old_sae_adapter (this directory), which imports vision/.
    env["PYTHONPATH"] = os.pathsep.join([str(TM_DIR / "src"), p["vision"], env.get("PYTHONPATH", "")])
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    env["W2V_PATH"] = p["w2v_gz"]
    env["SAE_TM_W2V_DIR"] = p["w2v_dir"]
    return env


def _run(argv: List[str], cwd: Optional[str], dry: bool) -> None:
    cmd = [sys.executable] + argv
    print(("[dry-run] " if dry else "[run] ") + (f"(cd {cwd}) " if cwd else "") + shlex.join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, cwd=cwd, env=_env(), check=True)


def run_cell(reg, run, enc, trainer, seed, warmup, dry=False, stages=("train", "interpret", "topics")):
    cfg = reg["runs"][run]
    tag = f"{run}/{enc}/{cell_seg(cfg, trainer, warmup)}/seed{seed}"
    best = stage_dir(run, cfg, enc, "trained_models", trainer, seed, warmup) / "best_val_monoscore_model.pt"
    interp = stage_dir(run, cfg, enc, "interpreted", trainer, seed, warmup)
    saetm = paths()["saetm"]
    if "train" in stages:
        if best.is_file():
            print(f"[skip] train {tag}: {best} exists")
        else:
            if not dry:
                best.parent.parent.mkdir(parents=True, exist_ok=True)
            _run(train_argv(reg, run, enc, trainer, seed, warmup), None, dry)
            if not dry and not best.is_file():
                raise RuntimeError(f"training did not produce {best}")
    if "interpret" in stages:
        if (interp / "word_emission_probabilities.pt").is_file():
            print(f"[skip] interpret {tag}")
        else:
            _run(interpret_argv(reg, run, enc, trainer, seed, warmup), saetm, dry)
    if "topics" in stages:
        for k in reg["topic_counts"]:
            out = stage_dir(run, cfg, enc, "topics", trainer, seed, warmup) / f"K{k}"
            if (out / "top_words.txt").is_file():
                print(f"[skip] topics {tag} K={k}")
                continue
            _run(topics_argv(reg, run, enc, trainer, seed, warmup, k), saetm, dry)


def run_evaluate(reg, run, encoders=None, trainer=None, seed=None, dry=False):
    """SAE-TM evaluation.py (diversity + coherence + intruder) over a run's topics tree, one encoder
    at a time (one LLM load per call). evaluation.py itself skips files whose outputs exist."""
    p = paths()
    for enc in (encoders or list(reg["encoders"])):
        sub = f"/{trainer}" if trainer else ""
        topics = run_root(run) / enc / f"topics{sub}"
        outs = run_root(run) / enc / f"evaluation{sub}"
        if not topics.is_dir() and not dry:
            print(f"[skip] no topics for {run}/{enc}{sub}")
            continue
        argv = ["evaluation.py", "--topics-root", str(topics.resolve()), "--outputs-root", str(outs.resolve()),
                "--model", p["judge"]]
        if seed is not None:
            argv += ["--path-filter", f"/seed{seed}/"]
        _run(argv, p["saetm"], dry)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "cell", "grid", "evaluate", "train-argv"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--encoder"); ap.add_argument("--encoders", nargs="*")
    ap.add_argument("--trainer"); ap.add_argument("--trainers", nargs="*")
    ap.add_argument("--seed", type=int); ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--warmup", type=int); ap.add_argument("--warmups", nargs="*", type=int)
    ap.add_argument("--stages", nargs="*", default=["train", "interpret", "topics"],
                    choices=["train", "interpret", "topics"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    reg = load_registry()
    if a.run not in reg["runs"]:
        ap.error(f"unknown run {a.run!r}; known: {', '.join(reg['runs'])}")
    cfg = reg["runs"][a.run]
    encs = [a.encoder] if a.encoder else a.encoders
    trs = [a.trainer] if a.trainer else a.trainers
    seeds = [a.seed] if a.seed is not None else a.seeds
    ws = [a.warmup] if a.warmup is not None else a.warmups

    if a.command == "list":
        for enc, tr, seed, w in iter_cells(reg, a.run, encs, trs, seeds, ws):
            best = stage_dir(a.run, cfg, enc, "trained_models", tr, seed, w) / "best_val_monoscore_model.pt"
            print(f"{enc}\t{cell_seg(cfg, tr, w)}\tseed{seed}\ttrained={best.is_file()}")
    elif a.command == "train-argv":
        if not (a.encoder and a.trainer and a.seed is not None):
            ap.error("train-argv needs --encoder --trainer --seed")
        w = a.warmup if a.warmup is not None else (cfg["warmups"][0] if is_dual(a.trainer) else None)
        print(shlex.join(train_argv(reg, a.run, a.encoder, a.trainer, a.seed, w)))
    elif a.command == "cell":
        if not (a.encoder and a.trainer and a.seed is not None):
            ap.error("cell needs --encoder --trainer --seed")
        w = a.warmup if a.warmup is not None else (cfg["warmups"][0] if is_dual(a.trainer) else None)
        run_cell(reg, a.run, a.encoder, a.trainer, a.seed, w, a.dry_run, a.stages)
    elif a.command == "grid":
        for enc, tr, seed, w in iter_cells(reg, a.run, encs, trs, seeds, ws):
            run_cell(reg, a.run, enc, tr, seed, w, a.dry_run, a.stages)
    elif a.command == "evaluate":
        run_evaluate(reg, a.run, encs, a.trainer, a.seed, a.dry_run)


if __name__ == "__main__":
    main()
