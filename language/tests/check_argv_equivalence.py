#!/usr/bin/env python3
"""Code-equivalence check: the training argv this component builds vs argv recorded in logs.

The training worker prints ``--- python main.py <argv>`` before it trains.  Given log files
from an earlier execution of the pipeline, this script re-derives the argv of every logged cell
with ``LD_DRY_RUN=1 train_cell.sh <tag>`` (no GPU, nothing written) and compares them word for
word.

Words that are expected to differ are reported separately, never silently ignored:
  * ``--jump_threshold_init <v>`` -- appended by this component for JumpReLU cells; a log from a
    code version that relied on the model default of the time does not carry it, so its value
    must equal that default (pass ``--implicit-threshold-init``);
  * ``--num_workers <n>`` -- throughput only (the sampler order is drawn in the main process).

    python tests/check_argv_equivalence.py --logs 'LOGDIR/*.out' --root <LANGDUAL_ROOT> \
        --tb-root <tensorboard root of the logged runs> [--tag-regex '_L12_']

Exit status 0 iff every compared cell matches up to the reported, expected differences.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
LANG_DIR = HERE.parent

ARGV_RE = re.compile(r"^--- python main\.py (.*)$")


def recorded_argv(log_globs: List[str]) -> Dict[str, List[str]]:
    """tag -> argv words, from the LAST '--- python main.py' line naming that tag."""
    out: Dict[str, List[str]] = {}
    files = sorted({f for g in log_globs for f in glob.glob(g)}, key=os.path.getmtime)
    for f in files:
        with open(f, errors="replace") as fh:
            for line in fh:
                m = ARGV_RE.match(line.rstrip("\n"))
                if not m:
                    continue
                # The worker echoes "${ARGS[*]}" (space-joined, unquoted): an empty value such as
                # --wandb_project "" disappears.  Split on whitespace and normalise the same way
                # below.
                words = m.group(1).split()
                if "--exp_name" in words:
                    out[words[words.index("--exp_name") + 1]] = words
    return out


def release_argv(tag: str, env: Dict[str, str]) -> List[str]:
    res = subprocess.run(["bash", str(LANG_DIR / "train_cell.sh"), tag], env=env,
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{tag}: dry run failed:\n{res.stderr}")
    lines = res.stdout.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("--- LD_DRY_RUN argv"))
    words = [shlex.split(l)[0] if l not in ("''",) else "" for l in lines[i + 2:]]
    return [w for w in words if w != ""]          # mirror the logged "${ARGS[*]}" form


def pairs(words: List[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Split an argv into (flag, value) pairs and bare flags, order-insensitive."""
    kv, bare = [], []
    i = 0
    while i < len(words):
        w = words[i]
        if w.startswith("--") and i + 1 < len(words) and not words[i + 1].startswith("--"):
            kv.append((w, words[i + 1]))
            i += 2
        else:
            bare.append(w)
            i += 1
    return kv, bare


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", required=True, help="glob(s) of worker stdout logs")
    ap.add_argument("--root", required=True, help="LANGDUAL_ROOT of the logged runs")
    ap.add_argument("--tb-root", required=True, help="tensorboard root of the logged runs")
    ap.add_argument("--tag-regex", default="", help="only compare tags matching this regex")
    ap.add_argument("--implicit-threshold-init", default="0.001",
                    help="model-default threshold init of the code version that wrote the logs")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    rec = recorded_argv(a.logs)
    env = dict(os.environ, LD_DRY_RUN="1", LANGDUAL_ROOT=a.root, LANGDUAL_TB_ROOT=a.tb_root,
               PYTHON=a.python, PYTHONDONTWRITEBYTECODE="1")
    env.pop("LD_ENV_READY", None)
    env.pop("SLURM_JOB_ID", None)

    # Imported AFTER the environment is fixed so the registry resolves the same roots.
    os.environ.update(LANGDUAL_ROOT=a.root, LANGDUAL_TB_ROOT=a.tb_root)
    sys.path.insert(0, str(LANG_DIR))
    from langdual import jobs  # noqa: E402

    n_ok = n_bad = n_skip = 0
    for tag in sorted(rec):
        if a.tag_regex and not re.search(a.tag_regex, tag):
            continue
        try:
            jobs.cell_from_tag(tag)
        except KeyError:
            n_skip += 1
            continue
        old_kv, old_bare = pairs(rec[tag])
        new_kv, new_bare = pairs(release_argv(tag, env))
        expected = []
        new_d = dict(new_kv)
        old_d = dict(old_kv)
        if "--jump_threshold_init" in new_d and "--jump_threshold_init" not in old_d:
            if new_d["--jump_threshold_init"] != a.implicit_threshold_init:
                print(f"MISMATCH {tag}: explicit --jump_threshold_init "
                      f"{new_d['--jump_threshold_init']} != implicit {a.implicit_threshold_init}")
                n_bad += 1
                continue
            expected.append("--jump_threshold_init (explicit = implicit default)")
            new_d.pop("--jump_threshold_init")
        for d in (new_d, old_d):
            d.pop("--num_workers", None)
        if new_d == old_d and sorted(new_bare) == sorted(old_bare):
            n_ok += 1
            print(f"OK       {tag}" + (f"   [{'; '.join(expected)}]" if expected else ""))
        else:
            n_bad += 1
            diff = sorted(set(new_d.items()) ^ set(old_d.items()))
            print(f"MISMATCH {tag}: {diff} bare: {sorted(set(new_bare) ^ set(old_bare))}")
    print(f"\n{n_ok} identical, {n_bad} mismatched, {n_skip} logged tags not in this registry")
    return 0 if n_bad == 0 and n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
