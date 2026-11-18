#!/usr/bin/env python
"""Code-equivalence check for SAE training (no GPU, no data).

The reported topic-modeling / hypothesis-generation SAEs were trained by calling the standalone
trainer CLIs (`vision/train_topk.py`, `vision/train_other.py`, `vision/train_dual.py`) with the
argv reconstructed below. The release drives `vision/main.py` instead (tm_driver.train_argv).
This test builds the TrainingConfig both ways (training itself is stubbed out) and asserts they
are identical, except for the fields listed in ALLOWED, each of which is checked against the value
the reported runs used.

If a standalone CLI is no longer present in vision/, that comparison is skipped and only the
explicit expected values below are checked.

Run: python topic_modeling/tests/test_train_config.py
"""

import dataclasses
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

TM_DIR = Path(__file__).resolve().parents[1]
VISION = Path(os.environ.get("VISION_DIR", TM_DIR.parent / "vision"))
sys.path.insert(0, str(TM_DIR / "src"))
sys.path.insert(0, str(VISION))

import tm_driver  # noqa: E402

# Fields that are metadata / IO only, or that the release passes explicitly (checked below).
IGNORE = {"exp_name", "output_dir", "train_path", "val_path", "test_path", "wandb_project",
          "wandb_name", "wandb_group", "dataset_name", "extractor_name", "tb_dir",
          # inert outside matryoshka / tree / forest model types (main.py passes its flag defaults)
          "group_fractions", "tree_sizes", "tree_k", "forest_sizes"}
# Fields where main.py is given an explicit value; the expected value is what the reported run had.
EXPECTED_EXPLICIT = {
    "JumpReluTrainer": {"jump_threshold_init": 0.001},
    "DualJumpReluTrainer": {"jump_threshold_init": 0.001},
}

COMMON = ["--n_latents", "8192", "--batch_size", "2048", "--num_epochs", "50", "--lr", "1e-4",
          "--eps", "6.25e-10", "--seed", "42", "--num_workers", "8", "--skip_extra_eval"]
# argv of the reported runs (run order: train_<x>.py --dataset_config <json> ...)
ORIGINAL = {
    "TopKTrainer": ("train_topk", COMMON + ["--topk_k", "64", "--dead_check_interval", "100", "--mono_coef", "0"]),
    "BatchTopKTrainer": ("train_other", COMMON + ["--model_type", "batch_topk", "--topk_k", "64", "--mono_coef", "0"]),
    "JumpReluTrainer": ("train_other", COMMON + ["--model_type", "jumprelu", "--l1_coef", "1e-4",
                                                 "--bandwidth", "1e-3", "--mono_coef", "0"]),
}
for tr, mode in tm_driver.DUAL_MODE.items():
    extra = ["--bandwidth", "1e-3"] if mode == "jump_relu" else []
    ORIGINAL[tr] = ("train_dual", COMMON + ["--sparsity_mode", mode, "--top_k", "64"] + extra + [
        "--warmup_steps", "17500", "--poly_hidden", "512", "--warmup_check_interval", "500",
        "--dead_check_interval", "100", "--span_coef", "0.01", "--mono_coef", "0"])

_captured = {}


def _stub_trainers():
    for name in ("train_topk", "train_other", "train_dual"):
        m = importlib.import_module(name)
        m.train_autoencoder = lambda cfg: _captured.__setitem__("cfg", cfg)


def _standalone_config(script, argv, dscfg):
    src = (VISION / f"{script}.py").read_text()
    marker = "if __name__ == '__main__':"
    if marker not in src:
        return None
    mod = importlib.import_module(script)
    code = src[src.index(marker):].replace(marker, "if True:", 1)
    sys.argv = [script, "--dataset_config", dscfg] + argv
    _captured.clear()
    g = dict(vars(mod))
    exec(compile(code, f"{script}.py[__main__]", "exec"), g)
    return _captured["cfg"]


def _main_config(argv):
    import main as vision_main
    sys.argv = argv
    _captured.clear()
    vision_main.main()
    return _captured["cfg"]


def main():
    tmp = tempfile.mkdtemp(prefix="tm_cfg_")
    os.environ["TM_OUT"] = tmp
    os.environ.setdefault("FEATURES_IMAGENET", os.path.join(tmp, "features"))
    dscfg = os.path.join(tmp, "ds.json")
    reg = tm_driver.load_registry()
    _stub_trainers()
    failures = []
    for trainer, (script, orig_argv) in ORIGINAL.items():
        run = "imagenet_paired_span001" if tm_driver.is_dual(trainer) else "imagenet_paired"
        w = 17500 if tm_driver.is_dual(trainer) else None
        drv = tm_driver.train_argv(reg, run, "clip", trainer, 42, w)
        new = _main_config(drv)
        with open(dscfg, "w") as f:
            json.dump({"train_path": new.train_path, "val_path": new.val_path}, f)
        old = _standalone_config(script, orig_argv, dscfg)
        dn = dataclasses.asdict(new)
        for k, v in EXPECTED_EXPLICIT.get(trainer, {}).items():
            if dn.get(k) != v:
                failures.append(f"{trainer}: {k}={dn.get(k)!r}, expected {v!r}")
        if dn.get("multilabel") is not False:
            failures.append(f"{trainer}: multilabel={dn.get('multilabel')!r}, expected False (ImageNet)")
        if old is None:
            print(f"[{trainer}] standalone CLI not present; checked explicit values only")
            continue
        do = dataclasses.asdict(old)
        skip = IGNORE | set(EXPECTED_EXPLICIT.get(trainer, {})) | {"multilabel", "skip_epoch_checkpoints"}
        diffs = [(k, do.get(k, "<absent>"), dn.get(k, "<absent>")) for k in sorted(set(do) | set(dn))
                 if k not in skip and do.get(k, "<absent>") != dn.get(k, "<absent>")]
        print(f"[{trainer}] differing fields: {diffs if diffs else 'none'}")
        failures += [f"{trainer}: {d}" for d in diffs]
    if failures:
        print("FAIL\n  " + "\n  ".join(map(str, failures)))
        sys.exit(1)
    print("OK: vision/main.py driver configs == the reported runs' trainer configs")


if __name__ == "__main__":
    main()
