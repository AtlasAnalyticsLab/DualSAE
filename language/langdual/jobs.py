"""Cell -> worker-environment decoder for the language experiments.

Every stage that needs to know *how* a cell is trained or evaluated asks this module, so the
training worker (train_cell.sh), the evaluation driver and the listing used by the sequential
drivers can never disagree about a cell's command line.

    python -m langdual.jobs --list main [--layers 12] [--corpora owt ghcode] [--activations ...]
    python -m langdual.jobs --list ablation [--axes warmupB] [--layers 12] [--scored-only]
    python -m langdual.jobs --tag B_L12_owt_topk_dual_seed42 --shell
    python -m langdual.jobs --tag abl_warmupB1250_B_L12_owt_topk_dual_seed42 --shell
    python -m langdual.jobs --selftest

``--shell`` prints KEY=value lines that the worker consumes with ``set -a; eval "$(...)"``.
The block is laid out in three sections and the ORDER IS THE MECHANISM:

  1. the protocol constants and roots (LD_N_LATENTS, LD_WARMUP_STEPS, LD_SPAN_COEF, ...);
  2. the per-cell identity and paths (LD_TAG, LD_RUN_DIR, data stores, --model, --lr, ...);
  3. for an ablation cell, the axis overrides LAST.

Because bash applies the assignments of one ``eval`` in order, an ablation axis overwrites the
inherited default it names and nothing else.  The opt-in knobs (LD_TIED_WEIGHTS, ...) are always
emitted, empty when unused, so a stray ``export`` in the caller's shell can never leak into a
cell.
"""

from __future__ import annotations

import argparse
import os
import shlex
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from langdual import ablate, config

# The three split names.  train: SAE training data; val: ONLY the trainers' best-epoch-by-
# validation-MonoScore selection; test: evaluation (config.EVAL_SPLIT).
SPLITS: Tuple[str, ...] = ("train", "val", "test")

# Token budget per split: train = exactly one pass (TOTAL_STEPS * BATCH_SIZE rows); val/test =
# the dense-code scoring cap.
SPLIT_TOKENS: Dict[str, int] = {
    "train": config.LM_TRAIN_TOKENS,
    "val": config.TRACKB_SCORE_TOKENS,
    "test": config.TRACKB_SCORE_TOKENS,
}

# The shard directory's completion marker.  Not a free choice: the vision core's
# dataset/resolve.py decides "LMDB or activation shards?" by testing for exactly this file, so
# the extractor writes it LAST and atomically.
SHARD_MANIFEST = "manifest.json"

AnyCell = Union[config.Cell, ablate.AblCell]


# --------------------------------------------------------------------------- cell lookup
def main_cells(layers: Optional[Sequence[int]] = None, corpora: Optional[Sequence[str]] = None,
               activations: Optional[Sequence[str]] = None,
               variants: Optional[Sequence[str]] = None,
               seeds: Optional[Sequence[int]] = None) -> List[config.Cell]:
    """Main-grid cells in registry order, optionally filtered."""
    return [c for c in config.iter_cells_b()
            if (not layers or c.layer in layers) and (not corpora or c.corpus in corpora)
            and (not activations or c.activation in activations)
            and (not variants or c.variant in variants) and (not seeds or c.seed in seeds)]


def ablation_cells(axes: Optional[Sequence[str]] = None, layers: Optional[Sequence[int]] = None,
                   activations: Optional[Sequence[str]] = None,
                   values: Optional[Sequence[str]] = None,
                   seeds: Optional[Sequence[int]] = None,
                   scored_only: bool = False) -> List[ablate.AblCell]:
    """Ablation cells in registry order, optionally filtered."""
    return [c for c in ablate.iter_cells(tuple(axes) if axes else None)
            if (not layers or c.layer in layers)
            and (not activations or c.activation in activations)
            and (not values or c.value in values) and (not seeds or c.seed in seeds)
            and (not scored_only or c.scored)]


def cell_from_tag(tag: str) -> AnyCell:
    for c in config.iter_cells_b():
        if c.tag == tag:
            return c
    for c in ablate.iter_cells():
        if c.tag == tag:
            return c
    raise KeyError(f"{tag!r} is neither a main-grid nor an ablation cell tag")


def reference_cell(cell: ablate.AblCell) -> config.Cell:
    """The main-grid cell at the same coordinates: the source of every knob the axis does not
    name (feature stores, --model, --lr, ...).  Its run_dir / result path point into the main
    grid and are never written by an ablation worker."""
    return config.Cell(cell.track, cell.key, cell.corpus, cell.activation, cell.variant,
                       cell.seed)


def data_paths(cell: config.Cell) -> Tuple[str, str, str]:
    """(train, val, test) activation-shard directories of the cell's OWN corpus and layer."""
    paths = tuple(config.resid_dir(cell.layer, s, cell.corpus) for s in SPLITS)
    for p in paths:
        config.assert_safe_text_path(p)
    return paths  # type: ignore[return-value]


# --------------------------------------------------------------------------- shell emission
def _s(value) -> str:
    """One value as the string bash will hold.  bool -> 0/1, sequence -> space-joined,
    None -> '' (an absent value, which the worker tests with -n)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return " ".join(str(x) for x in value)
    return str(value)


def _q(value) -> str:
    """One shell-quoted word."""
    return shlex.quote(_s(value))


# Opt-in knobs, emitted EMPTY ("flag not passed") unless a cell sets them.
OPT_IN_KNOBS: Tuple[str, ...] = ("LD_JUMP_THRESHOLD_INIT", "LD_TIED_WEIGHTS")


def _common_env(abl: bool = False) -> List[Tuple[str, object]]:
    """Protocol constants every worker needs, straight out of the registry."""
    rows: List[Tuple[str, object]] = [
        ("LD_ROOT", ablate.ROOT if abl else config.ROOT),
        ("LD_SAE_DIR", config.SAE_DIR),
        ("LD_RUNS_DIR", ablate.RUNS_DIR if abl else config.RUNS_DIR),
        ("LD_RESULT_DIR", ablate.RESULT_DIR if abl else config.RESULT_DIR),
        ("LD_LOG_DIR", ablate.LOG_DIR if abl else config.LOG_DIR),
        ("LD_TB_ROOT", ablate.TB_ROOT if abl else config.TB_ROOT),
        ("LD_N_LATENTS", config.N_LATENTS),
        ("LD_TOP_K", config.TOP_K),
        ("LD_BATCH_SIZE", config.BATCH_SIZE),
        ("LD_NUM_EPOCHS", config.NUM_EPOCHS),
        ("LD_STEPS_PER_EPOCH", config.STEPS_PER_EPOCH),
        ("LD_TOTAL_STEPS", config.TOTAL_STEPS),
        ("LD_WARMUP_STEPS", config.WARMUP_STEPS),
        ("LD_EPS", config.ADAM_EPS),
        ("LD_L1_COEF", config.L1_COEF),
        ("LD_BANDWIDTH", config.BANDWIDTH),
        ("LD_SPAN_COEF", config.SPAN_COEF),
        ("LD_POLY_LAYERS", config.POLY_LAYERS),
        ("LD_AUXK_COEF", config.AUXK_COEF),
        ("LD_MAX_GRAD_NORM", config.MAX_GRAD_NORM),
        ("LD_DEAD_STEPS_THRESHOLD", config.DEAD_STEPS_THRESHOLD),
        ("LD_DEAD_CHECK_INTERVAL", config.DEAD_CHECK_INTERVAL),
        ("LD_EVAL_SPLIT", config.EVAL_SPLIT),
    ]
    rows += [(k, "") for k in OPT_IN_KNOBS]
    return rows


def _cell_env(cell: config.Cell) -> List[Tuple[str, object]]:
    """Identity, data stores and training arguments of one main-grid cell.

    --model, --sparsity_mode, --normalize, --lr and the JumpReLU threshold init are DERIVED here
    (from the registry), never in bash, so the two arms of a pair cannot be trained differently.
    """
    train_p, val_p, test_p = data_paths(cell)
    is_dual = cell.variant == "dual"
    return [
        ("LD_TRACK", cell.track),
        ("LD_KEY", cell.key),
        ("LD_CORPUS", cell.corpus),
        ("LD_ACT", cell.activation),
        ("LD_VAR", cell.variant),
        ("LD_SEED", cell.seed),
        ("LD_TAG", cell.tag),
        ("LD_RUN_DIR", cell.run_dir),
        ("LD_DONE_SENTINEL", cell.done_sentinel),
        ("LD_TB_DIR", cell.tb_dir),
        # --model; 'dual' for every DualSAE arm (the JumpReLU baseline's model name is
        # 'jumprelu', without the underscore of the activation key).
        ("LD_MODEL", cell.model_type),
        ("LD_IS_DUAL", is_dual),
        # --sparsity_mode selects the Sparse Branch activation of a DualSAE; empty for a baseline.
        ("LD_SPARSITY_MODE", cell.activation if is_dual else ""),
        ("LD_NORMALIZE", cell.normalize),
        ("LD_LR", cell.lr),
        ("LD_MULTILABEL", cell.multilabel),
        ("LD_TRAIN_PATH", train_p),
        ("LD_VAL_PATH", val_p),
        ("LD_TEST_PATH", test_p),
        ("LD_TRACKB_RESULT", config.trackb_result(cell)),
        ("LD_TRACKB_OUT_DIR", os.path.dirname(config.trackb_result(cell))),
        # Must equal the training batch size: BatchTopK takes its top-k over the WHOLE batch.
        ("LD_ENCODE_BATCH_SIZE", config.BATCH_SIZE),
        ("LD_LAYER", cell.layer),
        # The threshold init every reported language JumpReLU cell (baseline and Dual) used;
        # passed explicitly because the vision core's default differs.  Empty for TopK/BatchTopK.
        ("LD_JUMP_THRESHOLD_INIT",
         config.JUMP_THRESHOLD_INIT if cell.activation == "jump_relu" else ""),
    ]


# Keys of _cell_env that must point into the ABLATION tree for an ablation cell.
_REDIRECT_CELL_KEYS: Tuple[str, ...] = (
    "LD_TAG", "LD_RUN_DIR", "LD_DONE_SENTINEL", "LD_TB_DIR", "LD_TRACKB_RESULT",
    "LD_TRACKB_OUT_DIR",
)


def _abl_cell_env(cell: ablate.AblCell) -> List[Tuple[str, object]]:
    """Inherit the reference cell's env, redirect its outputs, then apply the axis LAST."""
    replacements: Dict[str, object] = {
        "LD_TAG": cell.tag,
        "LD_RUN_DIR": cell.run_dir,
        "LD_DONE_SENTINEL": cell.done_sentinel,
        "LD_TB_DIR": cell.tb_dir,
        "LD_TRACKB_RESULT": ablate.trackb_result(cell),
        "LD_TRACKB_OUT_DIR": os.path.dirname(ablate.trackb_result(cell)),
    }
    rows: List[Tuple[str, object]] = []
    seen = set()
    for key, value in _cell_env(reference_cell(cell)):
        seen.add(key)
        rows.append((key, replacements.get(key, value)))
    missing = sorted(set(_REDIRECT_CELL_KEYS) - seen)
    if missing:   # a rename in _cell_env must fail loudly, never write into the main grid
        raise RuntimeError(f"_cell_env no longer emits {missing}; update _REDIRECT_CELL_KEYS")
    axis = ablate.AXES_BY_NAME[cell.axis]
    rows += [
        ("LD_AXIS", cell.axis),
        ("LD_AXIS_ENV", axis.env),
        ("LD_AXIS_VALUE", cell.value),
        ("LD_AXIS_DEFAULT", axis.default),
        ("LD_REF_TAG", reference_cell(cell).tag),
    ]
    rows += list(cell.overrides.items())
    return rows


def shell_env(cell: AnyCell) -> List[Tuple[str, object]]:
    if isinstance(cell, ablate.AblCell):
        return _common_env(abl=True) + _abl_cell_env(cell)
    return _common_env() + _cell_env(cell)


def folded_env(cell: AnyCell) -> Dict[str, str]:
    """The env as bash would see it after ``eval`` (later assignment wins), values unquoted."""
    out: Dict[str, str] = {}
    for k, v in shell_env(cell):
        out[k] = _s(v)
    return out


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    problems: List[str] = []
    mains = main_cells()
    abls = ablation_cells()
    if len(mains) != 108:
        problems.append(f"main grid has {len(mains)} cells, expected 108")
    tags = [c.tag for c in mains] + [c.tag for c in abls]
    if len(set(tags)) != len(tags):
        problems.append("tags are not unique across main + ablation")
    for c in mains + abls:
        env = folded_env(c)
        jr = env.get("LD_JUMP_THRESHOLD_INIT", "")
        if (c.activation == "jump_relu") != bool(jr):
            problems.append(f"{c.tag}: LD_JUMP_THRESHOLD_INIT={jr!r}")
        if isinstance(c, ablate.AblCell):
            if not env["LD_RUN_DIR"].startswith(ablate.RUNS_DIR):
                problems.append(f"{c.tag}: run dir outside the ablation tree")
            ax = ablate.AXES_BY_NAME[c.axis]
            if env[ax.env] != c.value:
                problems.append(f"{c.tag}: {ax.env}={env[ax.env]!r}, expected {c.value!r}")
        elif env["LD_TIED_WEIGHTS"] != "":
            problems.append(f"{c.tag}: main cell sets LD_TIED_WEIGHTS")
    print(f"main grid cells : {len(mains)}")
    for ax in ablate.AXES:
        print(f"ablation {ax.name:9s}: {len(list(ablate.iter_axis_cells(ax)))}")
    if problems:
        print("\nFAILED:")
        for p in problems[:20]:
            print("  -", p)
        return 1
    print("\nJOBS SELFTEST OK")
    return 0


def _ints(xs: Optional[Iterable[str]]) -> Optional[List[int]]:
    return [int(x) for x in xs] if xs else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", help="emit/print one cell")
    ap.add_argument("--shell", action="store_true", help="KEY=value lines for `eval`")
    ap.add_argument("--list", choices=("main", "ablation"), help="print matching cell tags")
    ap.add_argument("--layers", nargs="+")
    ap.add_argument("--corpora", nargs="+", choices=list(config.LM_CORPUS_ORDER))
    ap.add_argument("--activations", nargs="+", choices=list(config.ACTIVATIONS))
    ap.add_argument("--variants", nargs="+", choices=list(config.VARIANTS))
    ap.add_argument("--seeds", nargs="+")
    ap.add_argument("--axes", nargs="+", choices=[a.name for a in ablate.AXES])
    ap.add_argument("--values", nargs="+", help="ablation axis values to keep (e.g. 1250)")
    ap.add_argument("--scored-only", action="store_true",
                    help="ablation: only cells of axes evaluated by this component")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(selftest())
    if a.list == "main":
        for c in main_cells(_ints(a.layers), a.corpora, a.activations, a.variants,
                            _ints(a.seeds)):
            print(c.tag)
        return
    if a.list == "ablation":
        for c in ablation_cells(a.axes, _ints(a.layers), a.activations, a.values,
                                _ints(a.seeds), a.scored_only):
            print(c.tag)
        return
    if not a.tag:
        raise SystemExit("pass --tag, --list or --selftest")
    cell = cell_from_tag(a.tag)
    if a.shell:
        for key, value in shell_env(cell):
            print(f"{key}={_q(value)}")
        return
    print(cell)


if __name__ == "__main__":
    main()
