"""Pinned download descriptors for the language experiments.

Only ``prepare_assets.sh`` (the ONE network step of this component) downloads anything; every
other stage resolves files through ``local_snapshot`` and loads them with
``local_files_only=True``.  Keeping the descriptors in one module means the download step and
the offline loaders can never disagree about where a snapshot lives.

Revisions are pinned to the commits the reported runs were built from.

CLI:
    python -m langdual.assets --audit    # what is present / missing on disk
    python -m langdual.assets --json     # machine-readable manifest (read by prepare_assets.sh)
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langdual import config

# =============================================================================== layout
# Snapshots land under config.RAW_DIR in a flat tree:
#   <RAW_DIR>/models/<org>__<name>      e.g. .../models/EleutherAI__pythia-410m-deduped
#   <RAW_DIR>/datasets/<org>__<name>    e.g. .../datasets/Skylion007__openwebtext
# `local_dir=` snapshots (plain files, no symlink farm) rather than the shared HF blob cache.
MODELS_SUBDIR = "models"
DATASETS_SUBDIR = "datasets"

# Written into every completed snapshot directory by prepare_assets.sh: both the skip-guard
# and the provenance record (resolved commit sha, patterns).
SENTINEL_NAME = ".langdual_snapshot.json"


def local_snapshot(repo: str, repo_type: str = "model") -> str:
    """Absolute local directory for a repo id.  Pure path arithmetic, no I/O."""
    if repo_type not in ("model", "dataset"):
        raise ValueError(f"repo_type must be 'model' or 'dataset', got {repo_type!r}")
    sub = MODELS_SUBDIR if repo_type == "model" else DATASETS_SUBDIR
    return os.path.join(config.RAW_DIR, sub, repo.replace("/", "__"))


# =============================================================================== patterns
# huggingface_hub filters repo-relative paths with fnmatch, whose '*' DOES cross '/'.
MODEL_ALLOW_PATTERNS: Tuple[str, ...] = (
    "*.json", "*.txt", "*.safetensors", "tokenizer*", "1_Pooling/*", "2_Normalize/*",
    "modules.json",
)
# Only safetensors weights are loaded; every other weight format is skipped.
MODEL_IGNORE_PATTERNS: Tuple[str, ...] = (
    "*.bin", "*.h5", "*.msgpack", "*.ot", "*.onnx", "*.tflite", "*.xml",
    "onnx/*", "openvino/*", "tf_model*", "flax_model*",
)


@dataclass(frozen=True)
class ModelAsset:
    key: str
    repo: str
    revision: str                       # commit sha the reported runs used
    required_globs: Tuple[str, ...]
    repo_type: str = "model"
    allow_patterns: Tuple[str, ...] = MODEL_ALLOW_PATTERNS
    ignore_patterns: Tuple[str, ...] = MODEL_IGNORE_PATTERNS

    @property
    def local_dir(self) -> str:
        return local_snapshot(self.repo, self.repo_type)

    @property
    def manifest_key(self) -> str:
        return f"model:{self.key}"


@dataclass(frozen=True)
class DatasetAsset:
    """One dataset repo, with the EXACT repo-relative files needed (read directly with
    pyarrow by materialize_corpora.py)."""
    key: str
    repo: str
    revision: str
    files: Tuple[str, ...]
    repo_type: str = "dataset"
    note: str = ""

    @property
    def allow_patterns(self) -> Tuple[str, ...]:
        return self.files

    @property
    def ignore_patterns(self) -> Tuple[str, ...]:
        return ()

    @property
    def required_globs(self) -> Tuple[str, ...]:
        return self.files

    @property
    def local_dir(self) -> str:
        return local_snapshot(self.repo, self.repo_type)

    @property
    def manifest_key(self) -> str:
        return f"dataset:{self.key}"

    def local_files(self) -> List[str]:
        return [os.path.join(self.local_dir, rel) for rel in self.files]


# =============================================================================== descriptors
LM_ASSET = ModelAsset("pythia", config.LM_REPO, "c4fc8d586d62df497f1f9b69d66d3ca419992d3e",
                      ("config.json", "tokenizer.json", "*.safetensors"))

# Two parquet shards per training corpus: the packer needs config.LM_TRAIN_TOKENS plus two
# 200K-token held-out slices, and two shards give ample headroom for both corpora.
N_LM_SHARDS = 2
_REVISIONS = {
    "owt": "79d93d786212f7344586290adb811d4ae6a1762c",
    "ghcode": "c48d40f9e70f0196f8236901ee35807f7d6c44c0",
}
LM_CORPUS_ASSETS: Dict[str, DatasetAsset] = {
    name: DatasetAsset(
        name, config.LM_CORPORA[name].repo, _REVISIONS[name],
        tuple(config.LM_CORPORA[name].shard_template.format(idx=i) for i in range(N_LM_SHARDS)),
        note=f"{N_LM_SHARDS}/{config.LM_CORPORA[name].total_shards} shards; text column "
             f"'{config.LM_CORPORA[name].text_column}'")
    for name in config.LM_CORPUS_ORDER
}


# Text-embedding stage (outputs read by the LLM-steering component): the bge-large encoder and
# the two labelled corpora it embeds.
TEXT_ENCODER_ASSETS: Dict[str, ModelAsset] = {
    "bge": ModelAsset("bge", config.TEXT_ENCODERS["bge"].repo,
                      "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
                      ("config.json", "modules.json", "*.safetensors", "1_Pooling/config.json")),
}
TEXT_CORPUS_ASSETS: Dict[str, DatasetAsset] = {
    "agnews": DatasetAsset("agnews", config.CORPORA["agnews"].repo,
                           "eb185aade064a813bc0b7f42de02595523103ca4",
                           ("data/train-00000-of-00001.parquet",
                            "data/test-00000-of-00001.parquet"),
                           note="columns: text, label"),
    "goemotions": DatasetAsset("goemotions", config.CORPORA["goemotions"].repo,
                               "add492243ff905527e67aeb8b80c082af02207c3",
                               ("simplified/train-00000-of-00001.parquet",
                                "simplified/validation-00000-of-00001.parquet",
                                "simplified/test-00000-of-00001.parquet"),
                               note="columns: text, labels (list[int], 28 classes), id"),
}


def manifest() -> List[object]:
    """Every descriptor, models first, in a stable order."""
    return ([LM_ASSET] + [TEXT_ENCODER_ASSETS[k] for k in config.ENCODER_ORDER]
            + [LM_CORPUS_ASSETS[k] for k in config.LM_CORPUS_ORDER]
            + [TEXT_CORPUS_ASSETS[k] for k in config.CORPUS_ORDER])


def manifest_json() -> List[Dict[str, object]]:
    return [{
        "key": a.manifest_key, "repo": a.repo, "repo_type": a.repo_type,
        "revision": a.revision, "local_dir": a.local_dir,
        "allow_patterns": list(a.allow_patterns), "ignore_patterns": list(a.ignore_patterns),
        "required_globs": list(a.required_globs),
    } for a in manifest()]


def lm_corpus_files(corpus: str) -> List[str]:
    return LM_CORPUS_ASSETS[corpus].local_files()


# =============================================================================== status
def read_sentinel(asset) -> Optional[Dict[str, object]]:
    """Parse the completion sentinel; a truncated/garbage file reads as absent."""
    try:
        with open(os.path.join(asset.local_dir, SENTINEL_NAME), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _glob_satisfied(root: str, pattern: str) -> bool:
    """True when at least one non-empty file under `root` matches `pattern` (recursively for a
    bare '*.ext' pattern)."""
    if any(os.path.isfile(p) and os.path.getsize(p) > 0
           for p in glob.glob(os.path.join(root, pattern))):
        return True
    if "/" not in pattern and any(ch in pattern for ch in "*?["):
        for dirpath, _d, filenames in os.walk(root):
            for fn in filenames:
                if fnmatch.fnmatch(fn, pattern) and os.path.getsize(os.path.join(dirpath, fn)) > 0:
                    return True
    return False


def snapshot_status(asset) -> Tuple[bool, str]:
    """(complete?, reason).  Complete only when the sentinel exists, records the same patterns
    and revision as the descriptor, and every required file is present and non-empty."""
    root = asset.local_dir
    if not os.path.isdir(root):
        return False, "snapshot directory does not exist"
    sent = read_sentinel(asset)
    if sent is None:
        return False, f"no (or unreadable) {SENTINEL_NAME}"
    if list(sent.get("allow_patterns", [])) != list(asset.allow_patterns):
        return False, "allow_patterns changed since download"
    if sent.get("revision") != asset.revision and sent.get("commit_sha") != asset.revision:
        return False, f"revision differs ({sent.get('commit_sha')!r} != {asset.revision!r})"
    for pattern in asset.required_globs:
        if not _glob_satisfied(root, pattern):
            return False, f"required file missing or empty: {pattern}"
    return True, "ok"


def _audit() -> int:
    print(f"RAW_DIR: {config.RAW_DIR}\n")
    n_missing = 0
    for a in manifest():
        ok, reason = snapshot_status(a)
        n_missing += 0 if ok else 1
        print(f"{a.manifest_key:16s} {'PRESENT' if ok else 'MISSING':8s} {a.repo}@{a.revision[:10]} :: {reason}")
    return 0 if n_missing == 0 else 1


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--audit", action="store_true")
    g.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.json:
        print(json.dumps(manifest_json(), indent=2))
        return 0
    return _audit()


if __name__ == "__main__":
    raise SystemExit(_main())
