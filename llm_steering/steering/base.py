"""Constants, output paths, unit lists and atomic-IO helpers shared by every steering stage.

Nothing here touches a GPU.  Every other stage imports its paths from here so that the planner
(submit_plan.py) and the stage drivers can never disagree about where a unit's output lives.

Output root: ``$A16_ROOT`` (default ./outputs/llm_steering, relative to the working directory).
``A16_SMOKE=1`` marks a smoke-test root: it enables the reduced-size smoke options of the stages
and the grid restriction of common/arms.py, and must never be set for real runs.
One input is deliberately NOT under the output root: the neutral prompt file (PROTOCOL.md P2),
which ships with this repository at data/neutral_prompts.txt and is pinned by its sha256.  The
P1 JSON is a RESULT of the pilot and so lives under the root.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------- roots
ROOT = os.path.abspath(os.environ.get("A16_ROOT", os.path.join("outputs", "llm_steering")))
IS_SMOKE = os.environ.get("A16_SMOKE", "0") == "1"
LOG_DIR = os.path.join(ROOT, "logs")
PKG_DIR = Path(__file__).resolve().parents[1]          # <repo>/llm_steering

# PROTOCOL.md P2: the neutral prompt file, shipped with the repository and pinned by sha256:
# every generation stage refuses to run when the file on disk does not hash to PROMPTS_SHA256,
# so the prompts cannot change silently (D-49).  s3_prompts.py --check re-validates it.
PROMPTS_PATH = os.environ.get("A16_PROMPTS", str(PKG_DIR / "data" / "neutral_prompts.txt"))
PROMPTS_SHA256: Optional[str] = "1851bf6910d4967b8c7cc0468a45d2d4222c38eee145f4b30b941cdd73e67d80"  # P2

# ----------------------------------------------------------------------------- protocol constants
def _layers() -> Tuple[int, ...]:
    # The steered blocks.  A16_SMOKE=1 with A16_LAYERS restricts them (smoke test only).
    from common import arms
    return tuple(arms.grid_layers())


LAYERS: Tuple[int, ...] = _layers()                   # (6, 12, 18): D-43
CORPUS = "owt"
CTX_LEN = 128                                          # SAE training window (LM_CTX_LEN)
BUDGETS: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)   # D-48 KL budgets, nats
PILOT_LAYER = 12                                       # D-48 / PROTOCOL R7
N_RANDOM = 5                                           # random unit directions per concept (D-48 pilot)
TOP_LATENTS = 3                                        # D-46 top-3 per concept
SINK_FIRE_FRAC = 0.5                                   # D-46: fires at position 0 in > 50% of windows
ALWAYS_ON_RATE = 0.3                                   # D-46: fire rate > 0.3
GATE_ACTIVE_MIN = 0.9 * 64                             # D-43(a) / D-44: 57.6
GATE_LOSS_RATIO = 1.5                                  # D-43(b)
GATE_CE_MIN = 0.80                                     # D-43(c)
GATE_CE_DUAL_MARGIN = 0.10                             # D-43(c): Dual/g0 >= pair baseline - 0.10
GATE_WINDOWS = 400                                     # s0 rows: test windows [0, 400), as gates.D15
CE_TOKENS = 262_144                                    # eval_trackb --ce-tokens default (2048 windows)
CE_BATCH_SEQ = 16                                      # 16 x 128 = 2048 rows (the training batch)
D44_BATCHES = 5000                                     # D-44 minimum
# D-48 / C29 calibration
CALIB_WINDOWS = 32
CALIB_START_WINDOW = 1000          # owt test windows [1000, 1032): disjoint from the s0/s2 rows
N_BISECT = 8
CAP_MULT = 1e3                     # alpha, c capped at 1e3 x median non-sink token norm
DIRS_PER_PASS = 16
# Log-grid in units of the layer's median non-sink token norm m (shard space).  Mode (i): alpha =
# g*m.  Modes (ii')/(iii): c = c0 + g*m/||W_j|| with c0 the latent's p99 positive activation,
# plus g = 0 (c = c0 itself, the start of the monotone branch).
CALIB_GRID: Tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0,
                                  3.0, 5.0, 8.0, 13.0, 20.0, 50.0, 150.0, 400.0, 1000.0)
# D-49 generation
MAX_NEW_TOKENS = 48
TEMPERATURE = 0.7
TOP_P = 0.9
# HF generate(do_sample=True, temperature=0.7, top_p=0.9) ALSO applies top_k=50: pythia ships no
# generation_config.json, so GenerationConfig falls back to _get_default_generation_params()
# (transformers 5.8 generation/configuration_utils.py:573-582, applied to None fields at
# generation/utils.py:1720-1722) and _get_logits_processor builds Temperature -> TopK(50) -> TopP
# (utils.py:1235-1244).  D-49 specifies generate(), so steer.sample_hf reproduces that chain.
TOP_K = 50
N_PROMPTS_HEADLINE, N_CONT_HEADLINE = 60, 2
N_PROMPTS_OTHER, N_CONT_OTHER = 20, 1
MAX_PROMPT_TOKENS = 16
GEN_BATCH_ROWS = 512
# D-50 judge
JUDGE_MAX_TOKENS = 768
JUDGE_RETRY_TOKENS = 1536
JUDGE_MAX_MODEL_LEN = 4096
MODES_ALL: Tuple[str, ...] = ("i", "ii", "iii")        # "ii" is mode (ii') in D-47


def mode_label(mode: str) -> str:
    return {"i": "(i) direction-add", "ii": "(ii') error-preserving clamp",
            "iii": "(iii) Dual full re-run", "null": "unsteered"}[mode]


# ----------------------------------------------------------------------------- stage paths
def p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def s0_record(tag: str) -> str:
    return p("s0", "rescore", f"{tag}.json")


def s0_threshold(tag: str) -> str:
    return p("s0", "threshold", f"{tag}.json")


def s1_docs(corpus: str) -> str:
    return p("s1", "concept_docs", f"{corpus}.npz")


def s1_armfree(layer: int) -> str:
    return p("s1", f"L{layer}", "armfree.npz")


def s1_select(tag: str) -> str:
    return p("s1", "select", f"{tag}.json")


def s1_auroc(tag: str) -> str:
    return p("s1", "select", f"{tag}.auroc.npz")


def s2_record(tag: str) -> str:
    return p("s2", f"{tag}.json")


def s5_record(unit: str) -> str:
    return p("s5", f"{unit}.json")


def s6_dir(unit: str) -> str:
    return p("s6", unit)


def s7j_dir(unit: str) -> str:
    return p("s7", "judge", unit)


def s7m_dir(unit: str) -> str:
    return p("s7", "metrics", unit)


def s7_judge_cache(unit: str) -> str:
    """Per-unit phi-4 cache directory.  One directory per generation unit (and 'pilot') because
    llm_backend's cache loads EVERY cache.*.jsonl of its directory at start-up: a shared
    directory would make each of the 85 judge tasks parse every other task's records (~2.7M
    records, ~2 GB by the last wave) for no hits.  The only useful cross-unit hits are the pilot's
    concept prompts inside armfree_L<PILOT_LAYER>, which s7_judge preloads explicitly."""
    return p("s7", "judge_cache", unit)


def s7j_completions(unit: str, cell_file: str) -> str:
    """Raw judge completions of one shard, kept OUT of the rating file so s9 reads slim rows."""
    return p("s7", "judge_completions", unit, cell_file)


def s7_clf(corpus: str) -> str:
    return p("s7", "clf", f"{corpus}.npz")


def p1_json() -> str:
    return p("s4", "p1_primary_budget.json")


def identity_record() -> str:
    return p("s6", "_identity_test.json")


# The InternVL3.5 second-judge ENVIRONMENT check (s8 --second-judge-smoke) writes this record;
# the planner and the s8j stage read it, and s8j additionally records its own parse-failure
# rates on the production prompts.
SECOND_JUDGE_ENV_RECORD = os.environ.get("A16_SECOND_JUDGE_ENV",
                                         os.path.join(ROOT, "_smoke", "internvl_env_ok.json"))


# ----------------------------------------------------------------------------- units
def sae_refs():
    """Every arm of the registry, in arms.iter_arms order (layer -> activation -> arm -> seed).
    THIS ORDER IS THE UNIT-INDEX CONTRACT of the per-SAE stages (s0, s5, s6, s7)."""
    from common import arms
    return list(arms.iter_arms("B"))


def dual_refs():
    return [r for r in sae_refs() if r.arm in ("dual", "dual_g0")]


def gen_units() -> List[str]:
    """Generation-unit names (s5 uses the first 84, s6/s7 all 85): one per SAE tag, then
    'armfree_L<n>' per layer, then 'null'."""
    return [r.tag for r in sae_refs()] + [f"armfree_L{l}" for l in LAYERS] + ["null"]


def held_reason(ref) -> Optional[str]:
    """Why an arm must not be run at all right now (None = runnable if its inputs exist).

    JumpReLU: in the reported runs the JumpReLU pairs were not steered -- their DualSAE was not
    sparsity-matched to the baseline (D-15) -- so every JumpReLU unit is HELD by default.  Every
    stage is implemented for jump_relu; ``A16_ALLOW_JUMPRELU=1`` lifts the hold for arms whose
    files exist (the D-15 gate in common/gates.py then decides whether a pair is tested)."""
    if ref.activation == "jump_relu" and os.environ.get("A16_ALLOW_JUMPRELU", "0") != "1":
        return "held: JumpReLU pairs are not steered by default (D-15; A16_ALLOW_JUMPRELU=1 lifts it)"
    if ref.status != "ready":
        return f"arm not trained yet (status {ref.status})"
    return None


# ----------------------------------------------------------------------------- atomic IO
def _tmp_in(path: str) -> Tuple[int, str]:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    return tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", suffix=".tmp", dir=d)


def atomic_write_bytes(path: str, data: bytes) -> None:
    """tmp file in the same directory + fsync + os.replace: a killed job leaves either the old
    file or the new one, never a torn file that a skip-guard would mistake for a finished unit."""
    fd, tmp = _tmp_in(path)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o640)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def atomic_write_json(path: str, obj: Any) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=1, sort_keys=True,
                                        default=_json_default).encode("utf-8"))


def atomic_write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    buf = [json.dumps(r, ensure_ascii=False, separators=(",", ":"), default=_json_default)
           for r in rows]
    atomic_write_bytes(path, ("\n".join(buf) + ("\n" if buf else "")).encode("utf-8"))
    return len(buf)


def atomic_save_npz(path: str, **arrays: np.ndarray) -> None:
    fd, tmp = _tmp_in(path)
    os.close(fd)
    try:
        # np.savez appends '.npz' unless the name already ends with it; the tmp name does not,
        # so write through a file object to keep the exact tmp path.
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o640)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_json(path: str) -> Optional[Dict[str, Any]]:
    """The JSON at ``path`` or None when absent/unreadable (a torn file cannot exist because every
    writer is atomic, but a hand-edited one could)."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_prompts(require_pin: bool = True) -> List[str]:
    """The 60 neutral prompts, after verifying the file's sha256 against the P2 pin.

    ``require_pin=False`` is for s3 itself (before the pin exists) and for smoke tests run before
    the pin was recorded; generation stages always call it with the default."""
    if not os.path.isfile(PROMPTS_PATH):
        raise FileNotFoundError(f"{PROMPTS_PATH} missing: run `python -m steering."
                                f"s3_prompts --write` first")
    got = sha256_file(PROMPTS_PATH)
    if PROMPTS_SHA256 is None:
        if require_pin:
            raise RuntimeError("PROMPTS_SHA256 is not pinned in a16/base.py (P2 not recorded)")
    elif got != PROMPTS_SHA256:
        raise RuntimeError(f"{PROMPTS_PATH} sha256 {got} != pinned P2 {PROMPTS_SHA256}; the "
                           f"pre-registered prompt file changed -- refusing to generate")
    with open(PROMPTS_PATH, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    lines = [ln for ln in lines if ln.strip()]
    if len(lines) != N_PROMPTS_HEADLINE:
        raise RuntimeError(f"{PROMPTS_PATH}: {len(lines)} prompts, expected {N_PROMPTS_HEADLINE}")
    return lines


def log(msg: str) -> None:
    print(f"[a16 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def device_default() -> str:
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"
