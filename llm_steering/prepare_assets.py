#!/usr/bin/env python3
"""Every internet download of the steering pipeline (run ONCE on a machine with internet).

Everything lands under <LANGDUAL_ROOT>/raw in the flat layout the runtime code resolves
(common/lang_config.local_snapshot):

  raw/models/EleutherAI__pythia-410m-deduped    steered LM (also used by the language pipeline)
  raw/models/BAAI__bge-large-en-v1.5            concept-classifier encoder (also language/)
  raw/models/microsoft__phi-4                   AxBench judge (pinned revision)
  raw/datasets/canrager__amazon_reviews_mcauley_1and5/data/{train,test}-00000-of-00001.parquet
  raw/datasets/fancyzhx__ag_news/data/{train,test}-00000-of-00001.parquet          (language/)
  raw/datasets/google-research-datasets__go_emotions/simplified/*.parquet          (language/)
  raw/axbench/axbench/evaluators/{lm_judge.py,prompt_templates.py} + LICENSE      (pinned commit)
  raw/models/OpenGVLab__InternVL3_5-8B          optional second judge (--with-second-judge)

The revisions below are the ones the reported runs used (resolved commit SHAs).  Every
snapshot gets a sentinel file recording the repo, revision and patterns, which is also the
skip-guard: a re-run costs one stat() per asset.  The two AxBench files are sha256-checked
against the pins in common/llm_backend.py.

AG News and GoEmotions are only RAW snapshots here: the steering stages read the language
pipeline's materialised corpora (corpora/<corpus>/*.jsonl) and its bge document-embedding LMDBs,
so run the language pipeline's corpus materialisation and Track A bge embedding steps (and its
residual extraction at blocks 6/12/18 on OpenWebText) before the steering stages.

    python prepare_assets.py [--with-second-judge] [--only NAME ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import lang_config as config  # noqa: E402

SENTINEL = ".snapshot.json"
# Model files to fetch: every load path uses safetensors; tokenizers need json/txt/model files.
MODEL_ALLOW = ["*.json", "*.txt", "*.safetensors", "tokenizer*", "*.model", "1_Pooling/*",
               "2_Normalize/*", "modules.json"]
MODEL_IGNORE = ["*.bin", "*.h5", "*.msgpack", "*.ot", "*.onnx", "*.tflite", "*.xml", "onnx/*",
                "openvino/*", "tf_model*", "flax_model*"]

# name -> (repo, repo_type, revision, allow_patterns, ignore_patterns)
SNAPSHOTS = {
    "pythia": ("EleutherAI/pythia-410m-deduped", "model",
               "c4fc8d586d62df497f1f9b69d66d3ca419992d3e", MODEL_ALLOW, MODEL_IGNORE),
    "bge": ("BAAI/bge-large-en-v1.5", "model",
            "d4aa6901d3a41ba39fb536a557fa166f842b0e09", MODEL_ALLOW, MODEL_IGNORE),
    "phi4": ("microsoft/phi-4", "model",
             "932b33c0ec9ca189badeb22480721a8de9d0e006", MODEL_ALLOW, MODEL_IGNORE),
    "amazon": ("canrager/amazon_reviews_mcauley_1and5", "dataset",
               "2dc1eabedc91e72f3dfc8c3476857ea7aabe52a0",
               ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"], []),
    "agnews": ("fancyzhx/ag_news", "dataset", "eb185aade064a813bc0b7f42de02595523103ca4",
               ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"], []),
    "goemotions": ("google-research-datasets/go_emotions", "dataset",
                   "add492243ff905527e67aeb8b80c082af02207c3",
                   ["simplified/train-00000-of-00001.parquet",
                    "simplified/validation-00000-of-00001.parquet",
                    "simplified/test-00000-of-00001.parquet"], []),
}
# Optional second judge (s8j, not part of the steering table).  Its revision was not pinned in
# the reported runs; the resolved commit is recorded in the sentinel.
SECOND_JUDGE = ("OpenGVLab/InternVL3_5-8B", "model", "main", None, MODEL_IGNORE)

AXBENCH_REPO = "stanfordnlp/axbench"
AXBENCH_FILES = ("axbench/evaluators/lm_judge.py", "axbench/evaluators/prompt_templates.py",
                 "LICENSE")


def snapshot(name: str, repo: str, repo_type: str, revision: str, allow, ignore) -> None:
    from huggingface_hub import snapshot_download
    dest = config.local_snapshot(repo, repo_type)
    sent = os.path.join(dest, SENTINEL)
    want = {"repo": repo, "repo_type": repo_type, "revision": revision,
            "allow_patterns": allow, "ignore_patterns": ignore}
    if os.path.isfile(sent):
        have = json.load(open(sent))
        if {k: have.get(k) for k in want} == want:
            print(f"[skip] {name}: {dest}")
            return
    print(f"[get ] {name}: {repo}@{revision} -> {dest}", flush=True)
    path = snapshot_download(repo_id=repo, repo_type=repo_type, revision=revision,
                             local_dir=dest, allow_patterns=allow, ignore_patterns=ignore or None)
    info = dict(want)
    try:
        from huggingface_hub import HfApi
        info["commit_sha"] = HfApi().repo_info(repo, repo_type=repo_type, revision=revision).sha
    except Exception as e:  # provenance only
        info["commit_sha"] = f"unresolved ({type(e).__name__})"
    with open(os.path.join(path, SENTINEL), "w") as fh:
        json.dump(info, fh, indent=1)


def axbench() -> None:
    from common import llm_backend as LB   # pins: commit + sha256 of the two judge files
    root = Path(LB.AXBENCH_DIR)
    pins = {"axbench/evaluators/lm_judge.py": LB.AXBENCH_LM_JUDGE_SHA256,
            "axbench/evaluators/prompt_templates.py": LB.AXBENCH_TEMPLATES_SHA256}
    for rel in AXBENCH_FILES:
        out = root / rel
        if out.is_file() and (rel not in pins or
                              hashlib.sha256(out.read_bytes()).hexdigest() == pins[rel]):
            print(f"[skip] axbench {rel}")
            continue
        url = (f"https://raw.githubusercontent.com/{AXBENCH_REPO}/{LB.AXBENCH_COMMIT}/{rel}")
        print(f"[get ] {url}", flush=True)
        data = urllib.request.urlopen(url, timeout=60).read()
        if rel in pins and hashlib.sha256(data).hexdigest() != pins[rel]:
            raise SystemExit(f"{rel}: sha256 mismatch against the pinned judge file")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-second-judge", action="store_true",
                    help="also fetch InternVL3.5-8B (optional s8j second judge)")
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"subset of {list(SNAPSHOTS) + ['axbench', 'second_judge']}")
    a = ap.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise SystemExit("HF_HUB_OFFLINE=1 is set: this script needs internet")
    names = a.only or list(SNAPSHOTS) + ["axbench"] + (["second_judge"] if a.with_second_judge else [])
    for n in names:
        if n == "axbench":
            axbench()
        elif n == "second_judge":
            repo, rt, rev, allow, ign = SECOND_JUDGE
            snapshot(n, repo, rt, rev, allow, ign)
        else:
            snapshot(n, *SNAPSHOTS[n])
    print(f"assets under {config.RAW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
