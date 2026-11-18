#!/usr/bin/env python3
"""Single source of truth for the Matryoshka / Gated grid of the main comparison.

Defines, for the MatryoshkaSAE and GatedSAE pairs and their DualSAE twins
(Dual-Matryoshka, Dual-Gated), every training cell: the baseline recipes, the
Gated lambda calibration grid, the Dual (gamma = span_coef, warm-up) sweep and
the 3-seed confirmation grid. CB-SAE / CB-DualSAE are built post hoc on the
Matryoshka pair (see cb_sae/).

Nothing here runs anything. ``stage_cells.py`` turns a stage into command lines
and the pickers (``pick_gated_lambda.py``, ``pick_dual_winners.py``) import the
same constants, so the grid is defined exactly once. A cell's index in a grid
list is stable (lists are append-only and unresolved cells are emitted as
placeholders), so an index can be used as a job-array id.

Baseline recipes
----------------
Matryoshka uses the BatchTopK recipe (it IS BatchTopK-gated; the prefix
structure lives in the loss): lr 1e-4, normalize=False, l1 1e-4, bandwidth
1e-3, grad clip 1.0, AuxK 1/32. Gated uses the reference recipe (lr 5e-5, no
input normalization, no grad clip) with the suite's batch/epochs/eps; its
lambda is CALIBRATED per (dataset, encoder) to a realized L0 near 64 (it has
no k).

Two protocol asymmetries to keep in mind when reading results:
  * train_other.py has no dead-latent respawn, so the BASELINE arm of every
    pair gets none, while its DUAL twin respawns dead latents every
    ``--dead_check_interval`` steps (train_dual.py). Report dead fractions for
    both arms.
  * GatedSAE keeps the reference's sum-over-feature-dims loss reduction, so a
    1.0 grad-norm clip would fire every step; gated baselines therefore run
    with the clip effectively disabled (GATED_PROTOCOL['max_grad_norm']),
    matching the reference trainer, which does not clip.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Shared protocol (identical for every cell)
# ---------------------------------------------------------------------------
ENCODERS = ("clip", "vit", "sigclip")
DATASETS = ("imagenet", "open_images")
SEEDS = (42, 43, 44)
SWEEP_SEED = 42          # sweeps run on ONE seed; 43/44 never inform selection

N_LATENTS = 8192
TOPK_K = 64
EPOCHS = 50
BATCH = 2048
EPS = 6.25e-10

# Paths (environment overridable; relative to the directory the tools are run from).
OUTPUT_DIR = os.path.abspath(os.environ.get("MAIN_CKPT_DIR", "./checkpoints/main"))
TB_ROOT = os.path.abspath(os.environ.get("MAIN_TB_DIR", "./outputs/main/tensorboard"))
# Picker outputs (lambda / winner JSONs), L0 probes and trajectories.
STATE_DIR = os.path.abspath(os.environ.get("MAIN_STATE_DIR", "./outputs/main/selection"))
# branch_metrics.py results of the grid's runs.
EVAL_RESULTS = os.path.abspath(os.environ.get("MAIN_EVAL_DIR", "./outputs/main/branch_metrics"))

# Exact per-epoch step counts (train_loader uses drop_last=True).
TRAIN_ROWS = {"imagenet": 1_281_167, "open_images": 507_444}
STEPS_PER_EPOCH = {ds: n // BATCH for ds, n in TRAIN_ROWS.items()}
TOTAL_STEPS = {ds: s * EPOCHS for ds, s in STEPS_PER_EPOCH.items()}
# -> imagenet 625 x 50 = 31_250 ; open_images 247 x 50 = 12_350

# ---------------------------------------------------------------------------
# Baseline recipes (the values the reported runs used)
# ---------------------------------------------------------------------------
MATRYOSHKA_PROTOCOL = dict(
    lr=1e-4, normalize=False, l1_coef=1e-4, bandwidth=1e-3,
    max_grad_norm=1.0, auxk_coef=1 / 32,
    # Upstream dictionary_learning convention [1/32,1/16,1/8,1/4,1/2+1/32].
    group_fractions="0.03125,0.0625,0.125,0.25,0.53125",
)
GATED_PROTOCOL = dict(
    lr=5e-5,             # the reference default
    normalize=False,     # the reference does no per-sample standardization
    # FLAG G6: the reference gated trainer does not clip, and our loss carries
    # the vendored ~d x scale, so a 1.0 clip would fire on every step and
    # silently replace the optimizer. Effectively disabled, still logged.
    max_grad_norm=1e9,
    gated_sparsity_warmup=2000,
)

# ---------------------------------------------------------------------------
# Gated lambda calibration. Gated has NO k: L0 is emergent from l1_coef, so
# lambda is chosen per (dataset, encoder) to land the realized L0 near the
# suite's 64. Half-decade log spacing over 3.5 decades, bracketing the
# reference default 1e-1.
# ---------------------------------------------------------------------------
# Endpoints span 3.5 decades around the reference default 1e-1;
# pick_gated_lambda.py reports "NOT BRACKETED" and names the direction if the
# grid does not cross the target, and bisects (log-log) when it does.
GATED_LAMBDA_GRID = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0)
L0_TARGET = 64.0
L0_BAND = (48.0, 85.0)      # "near 64" = within x1.33, in log space
MAX_DEAD_FRAC = 0.90        # reject "L0=64 because only 300 latents are alive"

# ---------------------------------------------------------------------------
# DualSAE span / warm-up grids.
#
# Shape: a PLUS/CROSS per cell -- a span (gamma) arm at a fixed warm-up, plus
# a warm-up arm at span 0.01, sharing the cross point. Span arms are
# LOG-spaced. The warm-up centre is per dataset; on OpenImagesV7 every warm-up
# value must stay below the 12,350-step budget, otherwise the Residual Branch
# never unfreezes and the run is a plain SAE (asserted in _check_grids()).
# ---------------------------------------------------------------------------
DUAL_WARMUP_CENTRE = {"imagenet": 17500, "open_images": 3000}
DUAL_SPAN_CENTRE = 0.01

# Span arm, per encoder. The useful range of gamma depends on the encoder (it
# scales with the decoder-overlap magnitude), so ViT gets a range 2-3 decades
# higher than CLIP / SigLIP2.
DUAL_SPAN_ARM = {
    "clip":    (0.0, 0.001, 0.003, 0.01, 0.03, 0.1),
    "sigclip": (0.0, 0.001, 0.003, 0.01, 0.03, 0.1),
    "vit":     (0.0, 0.01, 0.07, 0.5, 1.5, 3.0, 5.0),
}
# Warm-up arm, per dataset, at span == DUAL_SPAN_CENTRE. Every value must be
# < TOTAL_STEPS[dataset] or it is the degenerate plain-SAE case above; asserted
# in _check_grids().
DUAL_WARMUP_ARM = {
    "imagenet": (2500, 10000, 17500),
    "open_images": (100, 200, 500, 1000, 2000, 3000, 6000, 9000),
}

DUAL_VARIANTS = ("dual_matryoshka", "dual_gated")

def _bracket_lambda(ds: str, enc: str, br) -> float | None:
    """The probe lambda for one bracket, or None when the cell has none.

    Shared by BOTH arms so the refinement rule cannot drift between them:
    the picker supplies a log-log interpolated 'probe' (it has the measured
    L0s, this module does not), and a bare [lo, hi] falls back to the
    GEOMETRIC midpoint because L0 responds to lambda multiplicatively -- an
    arithmetic midpoint sits far closer to the sparse end and overshoots.
    """
    import math
    if br is None:
        return None
    if isinstance(br, dict):
        return float(br.get("probe") or math.sqrt(br["lo"] * br["hi"]))
    if (isinstance(br, (list, tuple)) and len(br) == 2
            and all(isinstance(v, (int, float)) for v in br)):
        return math.sqrt(float(br[0]) * float(br[1]))
    # Say WHAT is wrong. A serialisation slip here previously surfaced as
    # "could not convert string to float: 'lo'" from inside sqrt, which says
    # nothing about the bracket that produced it.
    raise ValueError(
        f"{ds}/{enc}: malformed bracket {br!r}; expected "
        f"{{'lo','hi','probe'}} or a two-number [lo, hi]")


def dual_gated_lambda_bisect_grid(brackets: dict) -> list[dict]:
    """One extra seed-42 Dual-Gated cell per bracketed cell, at the grid centre.

    The dual arm needs this for exactly the reason the baseline does: its
    4-point multiplier grid can straddle L0=64 without landing in the band, and
    without a refinement stage those cells would never resolve.

    Same (span, warm-up) centre and same seed as the base lambda check, so the
    new point is comparable to the four it refines; only lambda moves.
    """
    brackets = _by_cell(brackets)
    out = []
    # Full-size grid with placeholders, same contract as every other grid here:
    # emitting only the bracketed cells would change the list's shape as cells
    # resolve, and the array index IS the cell.
    for ds, enc, _ in _cells(seeds=(SWEEP_SEED,)):
        lam = _bracket_lambda(ds, enc, brackets.get((ds, enc)))
        warm = DUAL_WARMUP_CENTRE[ds]
        out.append({"kind": "dual", "model": "dual", "variant": "dual_gated",
                    "dataset": ds, "encoder": enc, "seed": SWEEP_SEED,
                    "span_coef": DUAL_SPAN_CENTRE, "warmup_steps": warm,
                    "l1_coef": lam, "uncalibrated": lam is None,
                    "exp_name": (dual_name(ds, enc, "dual_gated",
                                           DUAL_SPAN_CENTRE, warm,
                                           SWEEP_SEED, lam)
                                 if lam is not None
                                 else f"UNBRACKETED_{ds}_{enc}")})
    return out


def gated_lambda_bisect_grid(brackets: dict) -> list[dict]:
    """One extra seed-42 cell per cell that BRACKETS 64 without landing in band.

    `brackets` maps 'dataset/encoder' -> (lambda_low, lambda_high), the two
    straddling CONVERGED grid points. The new point is their geometric mean,
    because L0 responds to lambda multiplicatively -- an arithmetic midpoint
    would sit far closer to the sparse end than the dense one and typically
    overshoot in one round.

    Bisection only makes sense on the converged part of the curve: a run that
    peaked at epoch 1 has an L0 that reflects how little it trained, not its
    lambda (see pick_gated_lambda.MIN_BEST_EPOCH_FRAC), so those points are
    excluded from the bracket before it gets here.
    """
    import math
    brackets = _by_cell(brackets)
    # ALWAYS one cell per (dataset, encoder), in the canonical order, with
    # non-bracketed cells as placeholders. Emitting only the bracketed ones
    # would change the grid's shape as cells resolve -- inserting a new cell
    # ahead of an existing one shifts every later array index, so task 1 in a
    # running submission would come to mean a different cell. Same contract as
    # gated_baseline_grid.
    out = []
    for ds, enc, _ in _cells(seeds=(SWEEP_SEED,)):
        lam = _bracket_lambda(ds, enc, brackets.get((ds, enc)))
        out.append({"kind": "baseline", "model": "gated", "dataset": ds,
                    "encoder": enc, "seed": SWEEP_SEED, "l1_coef": lam,
                    "uncalibrated": lam is None,
                    "exp_name": (baseline_name(ds, enc, "gated", SWEEP_SEED, lam)
                                 if lam is not None
                                 else f"UNBRACKETED_{ds}_{enc}")})
    return out


# ---------------------------------------------------------------------------
# Dual-Gated lambda CHECK. Dual-Gated does not inherit the baseline's lambda.
#
# The baseline's lambda is not transferable: the same lambda can keep the
# Gated baseline alive and close every gate of the Dual's Sparse Branch.
# f_gate = (pi_gate > 0) is a hard Heaviside, so L_recon contributes NO
# gradient to the gate path; the gate is moved only by L_sparse (closes it)
# and L_aux (opens it), and both see ReLU(pi_gate), which is identically zero
# once the gate shuts. The only remaining route -- the shared W_enc, still moved
# by L_recon -- is exactly what the Residual Branch removes pressure from by
# reconstructing x on its own. Dual decoder atoms are also not unit-norm (see
# DualSAE.normalize_decoder_columns_), so an L1 on the gate is differently
# scaled on the two arms.
#
# So: a SMALL bracket around the baseline's winner, run on the dual itself at
# the grid centre, before the (span, warm-up) sweep. 6 cells x 4 lambdas.
DUAL_GATED_LAMBDA_MULTIPLIERS = (0.1, 1 / 3, 1.0, 3.0)


def dual_gated_lambda_grid(baseline_lambda_by_cell: dict) -> list[dict]:
    """Seed-42 lambda check for Dual-Gated at the (span, warm-up) grid centre."""
    baseline_lambda_by_cell = _by_cell(baseline_lambda_by_cell)
    out = []
    for ds, enc, _ in _cells(seeds=(SWEEP_SEED,)):
        if (ds, enc) not in baseline_lambda_by_cell:
            continue          # same per-cell skip policy as gated_baseline_grid
        base = baseline_lambda_by_cell[(ds, enc)]
        warm = DUAL_WARMUP_CENTRE[ds]
        for mult in DUAL_GATED_LAMBDA_MULTIPLIERS:
            lam = base * mult
            out.append({"kind": "dual", "model": "dual",
                        "variant": "dual_gated", "dataset": ds, "encoder": enc,
                        "seed": SWEEP_SEED, "span_coef": DUAL_SPAN_CENTRE,
                        "warmup_steps": warm, "l1_coef": lam,
                        "exp_name": dual_name(ds, enc, "dual_gated",
                                              DUAL_SPAN_CENTRE, warm,
                                              SWEEP_SEED, lam)})
    return out

# The lambda=0 CEILING probe for the Dual-Gated arm.
#
# The 4-point multiplier bracket above assumes the dual's right lambda is
# within ~1 decade of the baseline's. When every point of a cell sits below the
# L0 target (or the branch is collapsed at every point), there is no upper
# endpoint to interpolate against. One run at lambda = 0 answers that: it is
# the ARCHITECTURAL CEILING (with the L1 term off nothing closes the gate, so
# no lambda > 0 can give a higher L0). If the ceiling is below 64, no lambda
# reaches the target; otherwise lambda=0 and the smallest grid point straddle
# the target and the log-interpolation applies.
#
# lambda = 0 exactly, not a small epsilon: the point is to remove the sparsity
# term, not to shrink it. It is a bracket endpoint and a diagnostic only --
# never a reported row.
DUAL_GATED_LAMBDA_FLOOR = 0.0


def dual_gated_lambda_floor_grid(dual_choice: dict) -> list[dict]:
    """lambda=0 ceiling probes, ONLY for cells that provably need one.

    Takes the DUAL picker's own output (`dual_gated_lambda_choice.json`), not
    the baseline lambda table, and admits a cell only when all three hold:

      * its 4-point base grid is COMPLETE (`missing` empty) -- probing a cell
        mid-grid would spend GPU on a question the outstanding runs may still
        answer;
      * it is not already resolved (`chosen`);
      * every measured point sits BELOW the L0 target, so no upper bracket
        endpoint exists anywhere in the grid.

    The third condition is read off the measurements, not off the picker's
    free-text reason. It covers both "all lambdas too sparse" and "collapsed at
    every lambda", which mean the same thing operationally: there is nothing
    above 64 to interpolate against.

    Returns [] when the dual choice file is absent or the dual grid is still
    filling, so this is safe to call every round from the driver.
    """
    table = (dual_choice or {}).get("table") or []
    chosen = (dual_choice or {}).get("chosen") or {}
    out = []
    for row in table:
        cell = row.get("cell", "")
        if cell in chosen or row.get("missing"):
            continue
        pts = row.get("grid") or []
        if not pts or max(p["l0_mean"] for p in pts) >= L0_TARGET:
            continue          # a bracket exists (or will): bisect, do not extend
        ds, _, enc = cell.partition("/")
        if ds not in DATASETS or enc not in ENCODERS:
            continue
        warm = DUAL_WARMUP_CENTRE[ds]
        out.append({"kind": "dual", "model": "dual", "variant": "dual_gated",
                    "dataset": ds, "encoder": enc, "seed": SWEEP_SEED,
                    "span_coef": DUAL_SPAN_CENTRE, "warmup_steps": warm,
                    "l1_coef": DUAL_GATED_LAMBDA_FLOOR,
                    "exp_name": dual_name(ds, enc, "dual_gated",
                                          DUAL_SPAN_CENTRE, warm, SWEEP_SEED,
                                          DUAL_GATED_LAMBDA_FLOOR)})
    return out


# ---------------------------------------------------------------------------
# Run naming. Never main.py's auto `..._time_<unix>` form: it encodes neither
# dataset nor encoder and defeats --skip_existing. Every name below is a pure
# function of the cell ("t3x" = this grid).
# ---------------------------------------------------------------------------
PREFIX = "t3x"


def _g(value: float) -> str:
    """Compact, round-trippable float tag: 0.005 -> '0.005', 5.0 -> '5'."""
    return f"{value:g}"


def baseline_name(dataset: str, encoder: str, method: str, seed: int,
                  lam: float | None = None) -> str:
    tag = f"_l1{_g(lam)}" if lam is not None else ""
    return (f"{PREFIX}_{dataset}_{encoder}_{method}_"
            f"n{N_LATENTS}k{TOPK_K}{tag}_seed{seed}")


def dual_name(dataset: str, encoder: str, variant: str, span: float,
              warmup: int, seed: int, lam: float | None = None) -> str:
    tag = f"_l1{_g(lam)}" if lam is not None else ""
    return (f"{PREFIX}_{dataset}_{encoder}_{variant}_n{N_LATENTS}k{TOPK_K}"
            f"{tag}_span{_g(span)}_warm{warmup}_seed{seed}")


# ---------------------------------------------------------------------------
# Grid enumerators. A cell's INDEX in the returned list is its array task id.
# ---------------------------------------------------------------------------
def _by_cell(mapping: dict) -> dict:
    """Normalize a per-cell mapping to (dataset, encoder) tuple keys.

    The same data is written two ways here: the picker JSONs use
    "dataset/encoder" strings (readable on disk), while the grid functions
    index by tuple. Accepting both here means a caller that hands a JSON's
    `chosen` dict straight through gets the right answer instead of a
    KeyError -- and, worse in the other direction, cannot silently miss a cell.
    """
    out = {}
    for k, v in (mapping or {}).items():
        if isinstance(k, str):
            ds, _, enc = k.partition("/")
            out[(ds, enc)] = v
        else:
            out[tuple(k)] = v
    return out


def _cells(datasets=DATASETS, encoders=ENCODERS, seeds=SEEDS):
    for ds in datasets:
        for enc in encoders:
            for s in seeds:
                yield ds, enc, s


def matryoshka_baseline_grid() -> list[dict]:
    """18 cells: 2 datasets x 3 encoders x 3 seeds."""
    return [{"kind": "baseline", "model": "matryoshka", "dataset": ds,
             "encoder": enc, "seed": s,
             "exp_name": baseline_name(ds, enc, "matryoshka", s)}
            for ds, enc, s in _cells()]


def gated_lambda_grid() -> list[dict]:
    """48 seed-42 cells: 2 datasets x 3 encoders x 8 lambdas.

    Full 50 epochs, not a shortened proxy: the quantity being calibrated is the
    realized L0 OF THE REPORTED CHECKPOINT (selected by val MonoScore over all
    50 epochs), and the gate biases are still moving at lr 5e-5 well past epoch
    10. The winner's run is also REUSED verbatim as the seed-42 member of the
    reported 3-seed baseline (same exp_name, same config), so a shortened sweep
    would forfeit that reuse and cost more overall.
    """
    return [{"kind": "baseline", "model": "gated", "dataset": ds,
             "encoder": enc, "seed": SWEEP_SEED, "l1_coef": lam,
             "exp_name": baseline_name(ds, enc, "gated", SWEEP_SEED, lam)}
            for ds, enc, _ in _cells(seeds=(SWEEP_SEED,))
            for lam in GATED_LAMBDA_GRID]


def gated_baseline_grid(lambda_by_cell: dict) -> list[dict]:
    """The 18 reported Gated cells at each cell's calibrated lambda.

    lambda_by_cell maps (dataset, encoder) -> lambda. Seed-42 entries reuse the
    sweep's exp_name exactly, so a completion-aware submitter sees them as
    already done and queues only seeds 43/44 (12 real runs).
    """
    lambda_by_cell = _by_cell(lambda_by_cell)
    # ALWAYS the full 18 cells, in the canonical order. A cell whose lambda is
    # not calibrated yet is emitted with l1_coef=None and skipped at SELECTION
    # time by the submitter -- it is not omitted from the grid.
    #
    # That distinction is the array contract. Omitting unresolved cells would
    # make the grid grow from 6 to 18 as calibration lands, shifting every
    # index: array task 3 would mean a different cell than it did in the
    # submission that is still running. Emitting a placeholder keeps index ==
    # cell for the life of the grid.
    out = []
    for ds, enc, s in _cells():
        lam = lambda_by_cell.get((ds, enc))
        out.append({"kind": "baseline", "model": "gated", "dataset": ds,
                    "encoder": enc, "seed": s, "l1_coef": lam,
                    "uncalibrated": lam is None,
                    "exp_name": (baseline_name(ds, enc, "gated", s, lam)
                                 if lam is not None
                                 else f"UNCALIBRATED_{ds}_{enc}_seed{s}")})
    return out


def dual_settings(dataset: str, encoder: str) -> list[tuple[float, int]]:
    """The plus/cross of (span, warmup) for one cell, cross point deduplicated."""
    warm_c = DUAL_WARMUP_CENTRE[dataset]
    seen, out = set(), []
    for span in DUAL_SPAN_ARM[encoder]:
        key = (span, warm_c)
        if key not in seen:
            seen.add(key)
            out.append(key)
    for warm in DUAL_WARMUP_ARM[dataset]:
        key = (DUAL_SPAN_CENTRE, warm)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def dual_sweep_grid(variant: str, lambda_by_cell: dict | None = None) -> list[dict]:
    """Seed-42 (span, warmup) sweep for one dual variant."""
    lambda_by_cell = _by_cell(lambda_by_cell)
    out = []
    for ds, enc, _ in _cells(seeds=(SWEEP_SEED,)):
        lam = (lambda_by_cell or {}).get((ds, enc)) if variant == "dual_gated" else None
        for span, warm in dual_settings(ds, enc):
            # Only the gated variant needs a calibrated lambda; every other
            # dual takes its l1_coef from its own protocol, so l1_coef=None is
            # normal there and must NOT read as "uncalibrated".
            uncal = (variant == "dual_gated" and lam is None)
            out.append({"kind": "dual", "model": "dual", "variant": variant,
                        "dataset": ds, "encoder": enc, "seed": SWEEP_SEED,
                        "span_coef": span, "warmup_steps": warm,
                        "l1_coef": lam, "uncalibrated": uncal,
                        "exp_name": (f"UNCALIBRATED_{ds}_{enc}_span{_g(span)}"
                                     f"_warm{warm}" if uncal else
                                     dual_name(ds, enc, variant, span, warm,
                                               SWEEP_SEED, lam))})
    return out


def dual_confirm_grid(variant: str, winners: dict,
                      lambda_by_cell: dict | None = None) -> list[dict]:
    """Seeds 43/44 at each cell's winning (span, warmup). winners maps
    (dataset, encoder) -> (span, warmup), or the pickers' "dataset/encoder"
    string form; both are accepted."""
    winners = _by_cell(winners)
    lambda_by_cell = _by_cell(lambda_by_cell)
    # ALWAYS the full 18 cells, in canonical order, with unresolved ones as
    # placeholders -- the same array-contract rule as gated_baseline_grid.
    # Sweeps resolve one cell at a time, so omitting the unresolved ones would
    # shift every later index as they land.
    out = []
    for ds, enc, s in _cells():
        w = winners.get((ds, enc))
        lam = (lambda_by_cell or {}).get((ds, enc)) if variant == "dual_gated" else None
        needs_lam = (variant == "dual_gated")
        if w is None or (needs_lam and lam is None):
            out.append({"kind": "dual", "model": "dual", "variant": variant,
                        "dataset": ds, "encoder": enc, "seed": s,
                        "span_coef": None, "warmup_steps": None,
                        "l1_coef": lam, "uncalibrated": True,
                        "exp_name": f"UNRESOLVED_{variant}_{ds}_{enc}_seed{s}"})
            continue
        # The picker records a dict; a caller may pass a bare pair.
        span, warm = ((w["span_coef"], w["warmup_steps"])
                      if isinstance(w, dict) else tuple(w))
        out.append({"kind": "dual", "model": "dual", "variant": variant,
                    "dataset": ds, "encoder": enc, "seed": s,
                    "span_coef": span, "warmup_steps": warm, "l1_coef": lam,
                    "uncalibrated": False,
                    # Carried from the picker: this cell is scored at BOTH the
                    # protocol checkpoint and convergence (see the eval
                    # manifest). Training is identical either way -- the flag
                    # only affects how many times the finished run is read.
                    "report_both_checkpoints": bool(
                        isinstance(w, dict)
                        and w.get("report_both_checkpoints")),
                    "exp_name": dual_name(ds, enc, variant, span, warm, s, lam)})
    return out


# ---------------------------------------------------------------------------
# Invariants. Imported at module load so a bad edit fails immediately rather
# than after a long training run.
# ---------------------------------------------------------------------------
def _check_grids() -> None:
    for ds, arm in DUAL_WARMUP_ARM.items():
        budget = TOTAL_STEPS[ds]
        bad = [w for w in arm if w >= budget]
        assert not bad, (
            f"{ds}: warm-up values {bad} >= the total budget of {budget} steps; "
            f"the poly branch would never unfreeze")
        assert DUAL_WARMUP_CENTRE[ds] < budget, ds
    # Names must be unique per grid, or two cells silently share a directory.
    for name, grid in (("matryoshka", matryoshka_baseline_grid()),
                       ("gated_lambda", gated_lambda_grid())):
        names = [c["exp_name"] for c in grid]
        assert len(names) == len(set(names)), f"{name}: duplicate exp_name"


_check_grids()


if __name__ == "__main__":
    print(f"steps/epoch {STEPS_PER_EPOCH} -> total {TOTAL_STEPS}")
    print(f"matryoshka baseline cells : {len(matryoshka_baseline_grid())}")
    print(f"gated lambda sweep cells  : {len(gated_lambda_grid())}")
    for v in DUAL_VARIANTS:
        g = dual_sweep_grid(v)
        print(f"{v} sweep cells        : {len(g)}")
    for ds in DATASETS:
        for enc in ENCODERS:
            print(f"  settings/cell {ds:12} {enc:8} = {len(dual_settings(ds, enc))}")
