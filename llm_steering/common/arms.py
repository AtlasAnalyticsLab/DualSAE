"""The registry of steered SAE arms: which run directory is "the TopK baseline at block 12,
seed 43" (etc.).

Every stage asks THIS module for run directories; no stage builds a tag string itself.  The
tags are the run-directory names written by the language pipeline (``language/``), unchanged:

  activation  arm       run directory                                                   grid
  ----------  --------  --------------------------------------------------------------  --------
  topk        baseline  abl_tkuntied_B_L<n>_owt_topk_baseline_seed<S>                    ablation
              dual      abl_warmupB1250_B_L<n>_owt_topk_dual_seed<S>                     ablation
              dual_g0   abl_dg0B1250_B_L<n>_owt_topk_dual_seed<S>                        ablation
  batch_topk  baseline  B_L<n>_owt_batch_topk_baseline_seed<S>                           main
              dual      abl_warmupB1250_B_L<n>_owt_batch_topk_dual_seed<S>               ablation
              dual_g0   abl_dg0B1250_B_L<n>_owt_batch_topk_dual_seed<S>                  ablation
  jump_relu   baseline  abl_jrfixB_B_L<n>_owt_jump_relu_baseline_seed<S>                 ablation
              dual      abl_jrfixBpw_B_L<n>_owt_jump_relu_dual_seed<S>                   ablation
              dual_g0   abl_jrfixBpwg0_B_L<n>_owt_jump_relu_dual_seed<S>                 ablation

with <n> in 6/12/18 (Pythia block whose output residual the SAE was trained on), <S> in
42/43/44, main-grid runs under ``lang_config.RUNS_DIR`` and ablation-grid runs under
``lang_config.ABL_RUNS_DIR``.  What the arms are:
  * baseline  the traditional SAE.  The TopK baseline is the UNTIED-weight TopK SAE
              (``--no_tied_weights``); the BatchTopK baseline is the main-grid one.
  * dual      DualSAE with a 1,250-step warm-up (span_coef = gamma = 0.01).
  * dual_g0   the same DualSAE with the decorrelation coefficient gamma (span_coef) = 0.
  * JumpReLU arms use threshold init ``--jump_threshold_init`` measured per layer, and the Dual
    arms keep training their threshold after warm-up (``--dual_train_threshold_post_warmup``).
    They are HELD by default (base.held_reason; ``A16_ALLOW_JUMPRELU=1`` lifts the hold).

Arm status:
  'ready'    autoencoder_final.pt AND CELL_DONE.json are present (every arm is scored at its
             final checkpoint; CELL_DONE.json is the trainer's completion sentinel).  'ready'
             says only that the FILES exist -- not that the arm passed a health gate.
  'pending'  not ready.

Contrasts (both arms of a (Dual, comparator) pair for every seed): ``contrast_ref`` adds the
JumpReLU sparsity-match state (D-15, common/gates.py).  The steering health gate (D-43) and the
outlier flag are applied by the stages on their own outputs and are not represented here.

    python -m common.arms --summary
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

from common import lang_config as config

# ----------------------------------------------------------------------------- the grid
ARMS: Tuple[str, ...] = ("baseline", "dual", "dual_g0")
ACTIVATIONS: Tuple[str, ...] = tuple(config.ACTIVATIONS)          # topk, batch_topk, jump_relu
SEEDS: Tuple[int, ...] = tuple(config.SEEDS)                      # 42, 43, 44
TRACK_B_LAYERS: Tuple[int, ...] = tuple(config.LM_LAYERS)          # 6, 12, 18
TRACK_B_CORPUS = "owt"
# Tested contrasts (Dual vs its comparator); dual_g0 vs baseline is never a tested contrast.
CONTRASTS: Tuple[Tuple[str, str], ...] = (("dual", "baseline"), ("dual", "dual_g0"))

FINAL_CKPT = "autoencoder_final.pt"
DONE_SENTINEL = "CELL_DONE.json"

# (activation, arm) -> (grid, tag prefix, variant).  grid 'main' -> RUNS_DIR, 'abl' ->
# ABL_RUNS_DIR.  The variant is the trainer's model variant ('baseline' or 'dual'); dual_g0 is a
# 'dual' run with span_coef 0.
_TAGS = {
    ("topk", "baseline"): ("abl", "abl_tkuntied_", "baseline"),
    ("topk", "dual"): ("abl", "abl_warmupB1250_", "dual"),
    ("topk", "dual_g0"): ("abl", "abl_dg0B1250_", "dual"),
    ("batch_topk", "baseline"): ("main", "", "baseline"),
    ("batch_topk", "dual"): ("abl", "abl_warmupB1250_", "dual"),
    ("batch_topk", "dual_g0"): ("abl", "abl_dg0B1250_", "dual"),
    ("jump_relu", "baseline"): ("abl", "abl_jrfixB_", "baseline"),
    ("jump_relu", "dual"): ("abl", "abl_jrfixBpw_", "dual"),
    ("jump_relu", "dual_g0"): ("abl", "abl_jrfixBpwg0_", "dual"),
}


class ArmPending(RuntimeError):
    """Raised when a run directory is required but the arm is not available."""


@dataclass(frozen=True)
class ArmRef:
    """One resolved arm."""
    substrate: str          # always 'B' here (residual-stream SAEs)
    key: str                # 'L<n>'
    corpus: str
    activation: str
    arm: str
    seed: int
    tag: Optional[str]
    run_dir: Optional[Path]
    source: str             # 'main' or 'abl'
    reused: bool = False
    blocked: Optional[str] = None

    @property
    def status(self) -> str:
        return _status(self)


def normalize_key(substrate: str, key: Union[str, int]) -> str:
    """'L<n>' from 12, '12' or 'L12' (only Track B, the residual-stream SAEs, exists here)."""
    if substrate != "B":
        raise KeyError(f"substrate must be 'B' (residual-stream SAEs), got {substrate!r}")
    s = str(key)
    n = int(s[1:]) if s.startswith("L") else int(s)
    if n not in TRACK_B_LAYERS:
        raise KeyError(f"layer {n} not in {TRACK_B_LAYERS}")
    return f"L{n}"


@lru_cache(maxsize=None)
def resolve_arm(substrate: str, key: Union[str, int], corpus: Optional[str], activation: str,
                arm: str, seed: int) -> ArmRef:
    """Map one arm coordinate to its run directory (table in the module docstring)."""
    key = normalize_key(substrate, key)
    corpus = TRACK_B_CORPUS if corpus is None else corpus
    if corpus != TRACK_B_CORPUS:
        raise KeyError(f"steered arms exist only on {TRACK_B_CORPUS!r}, got {corpus!r}")
    if activation not in ACTIVATIONS:
        raise KeyError(f"activation {activation!r} not in {ACTIVATIONS}")
    if arm not in ARMS:
        raise KeyError(f"arm {arm!r} not in {ARMS}")
    if int(seed) not in SEEDS:
        raise KeyError(f"seed {seed!r} not in {SEEDS}")
    seed = int(seed)
    grid, prefix, variant = _TAGS[(activation, arm)]
    tag = f"{prefix}B_{key}_{corpus}_{activation}_{variant}_seed{seed}"
    root = config.RUNS_DIR if grid == "main" else config.ABL_RUNS_DIR
    return ArmRef("B", key, corpus, activation, arm, seed, tag, Path(root) / tag, grid)


def _status(ref: ArmRef) -> str:
    if ref.run_dir is None:
        return "pending"
    if (ref.run_dir / FINAL_CKPT).is_file() and (ref.run_dir / DONE_SENTINEL).is_file():
        return "ready"
    return "pending"


def arm_run_dir(substrate: str, key: Union[str, int], corpus: Optional[str], activation: str,
                arm: str, seed: int) -> Path:
    return resolve_arm(substrate, key, corpus, activation, arm, seed).run_dir


def arm_status(substrate: str, key: Union[str, int], corpus: Optional[str], activation: str,
               arm: str, seed: int) -> str:
    return _status(resolve_arm(substrate, key, corpus, activation, arm, seed))


def _smoke_filter(name: str, default: Tuple) -> Tuple:
    """Smoke-test restriction of the grid (A16_SMOKE=1 only): A16_LAYERS / A16_ACTIVATIONS as
    comma lists.  Production runs always use the full grid."""
    if os.environ.get("A16_SMOKE", "0") != "1" or not os.environ.get(name):
        return default
    vals = [v.strip() for v in os.environ[name].split(",") if v.strip()]
    return tuple(int(v) for v in vals) if name == "A16_LAYERS" else tuple(vals)


def grid_layers() -> Tuple[int, ...]:
    return _smoke_filter("A16_LAYERS", TRACK_B_LAYERS)


def grid_activations() -> Tuple[str, ...]:
    return _smoke_filter("A16_ACTIVATIONS", ACTIVATIONS)


def iter_arms(substrate: Optional[str] = "B", *, keys: Optional[Sequence] = None,
              corpora: Optional[Sequence[str]] = None,
              activations: Optional[Sequence[str]] = None, arms: Optional[Sequence[str]] = None,
              seeds: Optional[Sequence[int]] = None) -> Iterator[ArmRef]:
    """Every arm (3 layers x 3 activations x 3 arms x 3 seeds = 81), optionally filtered.
    Order: layer -> activation -> arm -> seed, stable across calls (the per-SAE stages use the
    position as their unit index)."""
    if substrate not in (None, "B"):
        return
    ks = (tuple(f"L{n}" for n in grid_layers()) if keys is None
          else tuple(normalize_key("B", k) for k in keys))
    cs = (TRACK_B_CORPUS,) if corpora is None else tuple(corpora)
    for k in ks:
        for c in cs:
            for act in (grid_activations() if activations is None else activations):
                for arm in (ARMS if arms is None else arms):
                    for s in (SEEDS if seeds is None else seeds):
                        yield resolve_arm("B", k, c, act, arm, s)


def dual_g0_contrast_precondition(substrate: str) -> Tuple[Optional[bool], str]:
    """Whether Dual-vs-dual_g0 contrasts may be tested.  All arms here come from one code
    version of the trainer, so the precondition always holds."""
    return True, "dual and dual_g0 arms trained by the same code"


@dataclass(frozen=True)
class ContrastRef:
    """One (Dual, comparator) contrast with both arms for every seed.

    status: 'pending' (an arm not ready, or the JumpReLU sparsity match not evaluated),
    'gated' (D-15 removed it: per-seed rows only, labelled ``gate_label``), 'testable'."""
    substrate: str
    key: str
    corpus: str
    activation: str
    contrast: Tuple[str, str]
    dual: Tuple[ArmRef, ...]
    comparator: Tuple[ArmRef, ...]
    d15_state: str                      # 'n/a' | 'pass' | 'pending' | 'fail'
    gate_label: Optional[str]           # "not sparsity-matched" when D-15 failed
    reasons: Tuple[str, ...]

    @property
    def arms_status(self) -> str:
        st = [r.status for r in self.dual + self.comparator]
        return "ready" if all(s == "ready" for s in st) else "pending"

    @property
    def status(self) -> str:
        if self.arms_status == "pending" or self.d15_state == "pending":
            return "pending"
        if self.d15_state == "fail":
            return "gated"
        return "testable"


def contrast_ref(substrate: str, key: Union[str, int], corpus: Optional[str], activation: str,
                 contrast: Tuple[str, str], *, gates_root: Optional[str] = None) -> ContrastRef:
    contrast = tuple(contrast)
    if contrast not in CONTRASTS:
        raise KeyError(f"contrast {contrast} not in {CONTRASTS}")
    dual = tuple(resolve_arm(substrate, key, corpus, activation, contrast[0], s) for s in SEEDS)
    comp = tuple(resolve_arm(substrate, key, corpus, activation, contrast[1], s) for s in SEEDS)
    reasons: List[str] = []
    d15, label = "n/a", None
    if activation == "jump_relu":
        if all(r.status == "ready" for r in dual + comp):
            from common import gates      # lazy: gates imports this module
            d15, label, why15 = gates.d15_contrast_state(substrate, dual[0].key, dual[0].corpus,
                                                         contrast, root=gates_root)
            reasons.extend(why15)
        else:
            d15 = "pending"
            reasons.append("D-15 needs every arm of the pair trained first")
    return ContrastRef(substrate, dual[0].key, dual[0].corpus, activation, contrast, dual, comp,
                       d15, label, tuple(reasons))


def _summary() -> int:
    refs = list(iter_arms("B"))
    print(f"main-grid runs:     {config.RUNS_DIR}")
    print(f"ablation-grid runs: {config.ABL_RUNS_DIR}")
    print("status:", dict(Counter(r.status for r in refs)))
    for r in refs:
        print(f"  {r.key:4s} {r.activation:10s} {r.arm:8s} seed {r.seed}  {r.status:7s} {r.tag}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", action="store_true")
    ap.parse_args()
    return _summary()


if __name__ == "__main__":
    sys.exit(main())
