#!/usr/bin/env python3
"""Single source of truth for the DualSAE ablation grid (vision, TopK / BatchTopK Sparse Branch).

One grid covers four paper items, because they share the same retrained reference arm:

* Joint vs. sequential training (Residual analysis figure, supplementary table): arms S, F, R0.
* Projector of the Residual Branch (supplementary table): arms R0, B256, B512, D3, D4.
* Coefficient gamma of the decorrelation loss (supplementary tables): arms G0 ... G10 (+ R0).
* Expansion ratio (main figure + supplementary figures): arms E<r> and their plain-SAE twins SE<r>.

Nothing here runs anything: run_grid.py, eval_rows.py and the collectors import this module.

Every cell replays a main-table run
-----------------------------------
Each (dataset, encoder, activation, seed) cell copies the training settings of the matching
main-table DualSAE run (`../configs/main_dual_cells.json`, taken from those runs'
training_config.json) and overrides only what its arm is about. In particular the per-cell
gamma (`--span_coef`), warm-up length and `--dead_check_interval` are the values those runs used,
which are NOT main.py's argparse defaults; _published_replay() asserts the shared settings so a
config that does not fit the replay stops the grid instead of producing runs that silently differ.

Arms
----
    S     standard SAE running DualSAE's exact Sparse Branch code (no Residual Branch)
    F     S frozen, a fresh default projector trained alone (sequential training); gamma of the cell
    Z     normal two-stage training with r_hat held at 0 during warm-up
    R0    the main-table configuration retrained: Linear(A,A)+GELU projector (joint training)
    L1    Linear(A,A)
    B<h>  Linear(A,h)+GELU+Linear(h,A)
    B512G B512 plus a terminal GELU, separating the output activation from depth
    D3,D4 A->512->512->A and A->512->512->512->A
    G<g>  R0 with gamma (= span_coef, weight of L_dcor) set to g
    E<r>  R0 with r x encoder-width latents;  SE<r> the matching S arm
    K<k>  R0 with TopK k changed;            SK<k> the matching S arm

The architecture arms use their own projector in both training stages (literal two-stage schedule),
so the frozen random projector they warm up against differs between arms. The collector reports
each arm's warm-up-end R^2 so this can be read directly.

Run names are deterministic (`abl_<dataset>_<encoder>_<activation>_<arm>_seed<s>`): a time-stamped
name would defeat --skip_existing and lets two runs launched in the same second share a directory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))           # scripts/ablations (feature_paths)
import feature_paths  # noqa: E402

# Output root of the whole ablation suite. ABL_ROOT (default ./outputs/ablations) is resolved at
# import time; set_root() re-points every derived directory (used by tests and --root flags).
ROOT = Path(os.environ.get("ABL_ROOT", "./outputs/ablations")).resolve() / "grid"
OUTPUT_DIR = ROOT / "checkpoints"
TB_ROOT = ROOT / "tensorboard"
LOG_DIR = ROOT / "logs"
MANIFEST_DIR = ROOT / "manifests"
BRANCH_METRICS_RESULTS = ROOT / "branch_metrics" / "results"

CELLS_JSON = _HERE.parent / "configs" / "main_dual_cells.json"
PREFIX = "abl"

DATASETS = ("imagenet", "open_images")
ENCODERS = ("clip", "sigclip", "vit")
ACTIVATIONS = ("topk", "batch_topk")
SEEDS = (42, 43, 44)
DUAL_METHOD = {"topk": "dual_topk", "batch_topk": "dual_batch_topk"}
BASELINE_METHOD = {"topk": "topk", "batch_topk": "batch_topk"}

EPOCHS = 50
BATCH = 2048
TRAIN_ROWS = {"imagenet": 1_281_167, "open_images": 507_444}
# train_dual builds its loader with drop_last=True, so an epoch is exactly rows // batch steps.
STEPS_PER_EPOCH = {ds: rows // BATCH for ds, rows in TRAIN_ROWS.items()}
TOTAL_STEPS = {ds: steps * EPOCHS for ds, steps in STEPS_PER_EPOCH.items()}


def set_root(root: Path) -> None:
    """Point every output directory of the grid at `root` (an ablation-suite grid directory)."""
    global ROOT, OUTPUT_DIR, TB_ROOT, LOG_DIR, MANIFEST_DIR, BRANCH_METRICS_RESULTS
    ROOT = Path(root).resolve()
    OUTPUT_DIR = ROOT / "checkpoints"
    TB_ROOT = ROOT / "tensorboard"
    LOG_DIR = ROOT / "logs"
    MANIFEST_DIR = ROOT / "manifests"
    BRANCH_METRICS_RESULTS = ROOT / "branch_metrics" / "results"


# Arguments each architecture arm adds to the replayed main-table command.
ARCH_ARMS: Dict[str, List[str]] = {
    "R0": [],
    "L1": ["--poly_final_act", "none"],
    "B64": ["--poly_hidden_dims", "64"],
    "B256": ["--poly_hidden_dims", "256"],
    "B512": ["--poly_hidden_dims", "512"],
    "B2048": ["--poly_hidden_dims", "2048"],
    "B512G": ["--poly_hidden_dims", "512", "--poly_final_act", "gelu"],
    "D3": ["--poly_hidden_dims", "512,512"],
    "D4": ["--poly_hidden_dims", "512,512,512"],
}
# Decorrelation-coefficient ablation. Each arm replays the main-table config and changes ONLY gamma
# (--span_coef, the weight on L_dcor in L = L_rec + lambda*L_sparse + gamma*L_dcor). gamma is the
# only live loss coefficient for TopK/BatchTopK: L_sparse is 0 by construction there
# (dual_sae.compute_loss adds an L1/L0 term only for vanilla/gated/jump_relu), and sim_coef is
# accepted by the CLI but never added to the loss. A shared ABSOLUTE log grid is used rather than
# multiples of each cell's main-table gamma, because that gamma differs between cells (see
# main_dual_cells.json); a relative grid would put the cells on different gamma values.
# Names are filesystem-safe and parse back to the value via SPAN_ARMS, never via the string.
SPAN_ARMS: Dict[str, float] = {
    "G0": 0.0,
    "G1e-3": 1e-3,
    "G1e-2": 1e-2,
    "G1e-1": 1e-1,
    "G1": 1.0,
    "G10": 10.0,
}


def span_alias(arm: str, published_span: float) -> Optional[str]:
    """"R0" when a gamma arm would exactly repeat the main-table config, else None.

    Where the cell's main-table gamma is exactly that arm's gamma (e.g. 0.01), the G arm would be R0
    bit for bit (same argv except exp_name), so build_grid skips it and the collectors read R0 in
    its place, as scale_alias does for ViT x8. Exact float equality on purpose: per-seed gammas such
    as 0.005 vs 0.006 must NOT alias.
    """
    if arm in SPAN_ARMS and SPAN_ARMS[arm] == float(published_span):
        return "R0"
    return None


# Expansion-ratio and sparsity-level ablation, one factor at a time:
#   E<r>  dictionary = r x encoder width, k = 64
#   K<k>  k changed, dictionary = 8192
# Each Dual arm has a matched plain-SAE arm ("S" + name) that applies arm S's changes (no Residual
# Branch) at the same dictionary size and k, so the Dual - SAE gap can be read at every point.
# The expansion ratio is the true ratio to the encoder width, so 8192 latents is x10.67 for CLIP,
# x7.11 for SigLIP2 and x8 for ViT. Every other hyperparameter, gamma included, stays at the
# cell's main-table value (which was chosen at 8192 latents / k = 64).
ACT_DIM = {"clip": 768, "sigclip": 1152, "vit": 1024}
EXPANSIONS = (4, 8, 16, 32)
K_VALUES = (16, 32, 128, 256)
SCALE_DUAL_ARMS: Dict[str, Dict[str, int]] = {
    **{f"E{r}": {"expansion": r} for r in EXPANSIONS},
    **{f"K{k}": {"top_k": k} for k in K_VALUES},
}
SCALE_SAE_ARMS: Dict[str, str] = {f"S{arm}": arm for arm in SCALE_DUAL_ARMS}  # SAE arm -> Dual arm
SCALE_ARMS = tuple(SCALE_DUAL_ARMS) + tuple(SCALE_SAE_ARMS)

# Two x16 points whose best-validation-MonoScore checkpoint fell before the Residual Branch was
# unfrozen (Dual) or in the first epoch (SAE) were retrained with checkpoint selection restricted
# to later epochs (--select_from_epoch). Each is a separate run with a "_sel<E>" suffix; the
# expansion data builder substitutes it for the original point. 29 = the first epoch after the
# 17,500-step warm-up of that cell (17,500 / 625 = 28 epochs); 10 = epoch 10 onward.
SELECTION_RERUNS: List[dict] = [
    {"dataset": "imagenet", "encoder": "sigclip", "activation": "batch_topk", "arm": "E16",
     "seed": 42, "select_from_epoch": 29},
    {"dataset": "imagenet", "encoder": "vit", "activation": "batch_topk", "arm": "SE16",
     "seed": 42, "select_from_epoch": 10},
]

# Trained without depending on another arm, so they can all run in any order.
INDEPENDENT_ARMS = ("S", "Z") + tuple(ARCH_ARMS)
# The joint-vs-sequential and projector arms.
CORE_ARMS = INDEPENDENT_ARMS + ("F",)
# SPAN_ARMS and SCALE_ARMS are in ALL_ARMS (so build_grid accepts them and the eval stage scores
# them) but NOT in INDEPENDENT_ARMS, the default of `run_grid.py --stage train`.
ALL_ARMS = CORE_ARMS + tuple(SPAN_ARMS) + SCALE_ARMS


def is_sae_arm(arm: str) -> bool:
    """Arms with no Residual Branch: S itself and its scale-ablation counterparts."""
    return arm == "S" or arm in SCALE_SAE_ARMS


def scale_setting(encoder: str, arm: str, published_n: int = 8192,
                  published_k: int = 64) -> Dict[str, int]:
    """(n_latents, top_k) a scale arm trains at. Dual and SAE arms of one setting share it."""
    spec = SCALE_DUAL_ARMS[SCALE_SAE_ARMS.get(arm, arm)]
    n = spec["expansion"] * ACT_DIM[encoder] if "expansion" in spec else published_n
    return {"n_latents": n, "top_k": spec.get("top_k", published_k)}


def scale_alias(encoder: str, arm: str) -> Optional[str]:
    """The already-trained arm a scale arm would duplicate, or None.

    ViT x8 is exactly 8192 latents at k=64, i.e. the main-table config, so E8/SE8 on ViT are R0/S.
    build_grid skips them and the collectors read R0/S in their place.
    """
    s = scale_setting(encoder, arm)
    if (s["n_latents"], s["top_k"]) == (8192, 64):
        return "S" if is_sae_arm(arm) else "R0"
    return None


def exp_name(dataset: str, encoder: str, activation: str, arm: str, seed: int) -> str:
    return f"{PREFIX}_{dataset}_{encoder}_{activation}_{arm}_seed{seed}"


def run_dir(dataset: str, encoder: str, activation: str, arm: str, seed: int) -> Path:
    return OUTPUT_DIR / exp_name(dataset, encoder, activation, arm, seed)


def _cells_payload() -> dict:
    return json.loads(CELLS_JSON.read_text())


def load_published_rows() -> List[dict]:
    """The main-table Dual cells this grid replays (TopK / BatchTopK Sparse Branch), one per seed.

    A row carries its training settings under "config" (shared settings merged with the per-cell
    ones), standing in for the run's training_config.json.
    """
    payload = _cells_payload()
    rows = []
    for cell in payload["cells"]:
        if cell["activation"] not in DUAL_METHOD:
            continue
        cfg = dict(payload["common"])
        cfg.update({k: cell[k] for k in ("lr", "normalize", "span_coef", "warmup_steps",
                                         "dead_check_interval")})
        cfg.update({"model_type": "dual", "sparsity_mode": cell["activation"], "seed": cell["seed"]})
        rows.append({"dataset": cell["dataset"], "encoder": cell["encoder"],
                     "method": DUAL_METHOD[cell["activation"]], "seed": cell["seed"],
                     "eval_split": cell["eval_split"], "config": cfg})
    if len(rows) != len(DATASETS) * len(ENCODERS) * len(ACTIVATIONS) * len(SEEDS):
        raise ValueError(f"expected 36 TopK/BatchTopK main-table cells, found {len(rows)}")
    return rows


def published_config(row: dict) -> dict:
    """Training settings of one main-table cell (what its training_config.json recorded)."""
    return row["config"]


def activation_of(method: str) -> str:
    return method[len("dual_"):] if method.startswith("dual_") else method


def _published_replay(row: dict) -> dict:
    """Read one main-table Dual's settings, refusing anything this replay cannot reproduce."""
    tcfg = published_config(row)
    expected = {
        "model_type": "dual",
        "sparsity_mode": activation_of(row["method"]),
        "normalize": False,
        "mono_coef": 0.0,
        "sim_coef": 0.0,
        "n_latents": 8192,
        "top_k": 64,
        "batch_size": BATCH,
        "num_epochs": EPOCHS,
        "active_only_span": True,
        "warmup_mono_target": 0.0,
        "seed": row["seed"],
    }
    wrong = {k: (tcfg.get(k), v) for k, v in expected.items() if tcfg.get(k) != v}
    if wrong:
        raise ValueError(f"{exp_name(row['dataset'], row['encoder'], activation_of(row['method']), 'R0', row['seed'])}: "
                         f"main-table config differs from the replay: {wrong}")
    if tcfg.get("poly_layers", 1) != 1 or tcfg.get("poly_hidden_dims"):
        raise ValueError("expected the main-table 1-layer projector")
    return tcfg


def base_argv(row: dict, tcfg: dict, exp: str) -> List[str]:
    """The replayed main-table command, shared by every arm of a cell.

    model_type is implied by --model, top_k is spelled --topk_k, and active_only_span cannot be
    turned off (asserted True above). max_grad_norm is not forwarded for dual runs and train_dual
    never clips, so it has no effect to replay. --poly_hidden is recorded by the main-table runs but
    is not read when poly_layers = 1 (the projector is Linear(A,A)+GELU); it is passed so the
    training_config.json of a replay matches the original.
    """
    return [
        "--model", "dual",
        "--sparsity_mode", tcfg["sparsity_mode"],
        "--dataset", row["dataset"],
        "--extractor", row["encoder"],
        *feature_paths.path_args(row["dataset"], row["encoder"]),
        "--seed", str(row["seed"]),
        "--n_latents", str(tcfg["n_latents"]),
        "--topk_k", str(tcfg["top_k"]),
        "--batch_size", str(tcfg["batch_size"]),
        "--num_epochs", str(tcfg["num_epochs"]),
        "--lr", repr(tcfg["lr"]),
        "--eps", repr(tcfg["eps"]),
        "--l1_coef", repr(tcfg["l1_coef"]),
        "--bandwidth", repr(tcfg["bandwidth"]),
        "--mono_coef", repr(tcfg["mono_coef"]),
        "--mono_period", str(tcfg["mono_period"]),
        "--sim_coef", repr(tcfg["sim_coef"]),
        # Per-cell values used by the main-table runs (not main.py defaults): gamma, warm-up and
        # the dead-latent check interval.
        "--span_coef", repr(tcfg["span_coef"]),
        "--warmup_steps", str(tcfg["warmup_steps"]),
        "--warmup_check_interval", str(tcfg["warmup_check_interval"]),
        "--poly_lr_factor", repr(tcfg["poly_lr_factor"]),
        "--poly_hidden", str(tcfg["poly_hidden"]),
        "--dead_steps_threshold", str(tcfg["dead_steps_threshold"]),
        "--dead_check_interval", str(tcfg["dead_check_interval"]),
        "--num_workers", str(tcfg["num_workers"]),
        "--exp_name", exp,
        "--output_dir", str(OUTPUT_DIR),
        "--tb_dir", str(TB_ROOT / exp),
        "--skip_existing",
        # Write-only (per-epoch snapshots, ~150 MB each); nothing in this suite reads them.
        "--skip_epoch_checkpoints",
        # --skip_extra_eval is deliberately absent: evaluation forwards advance the dead-latent
        # counter (dual_sae.update_inactive_features), so that flag would change the run.
    ]


def set_arg(argv: List[str], flag: str, value: str) -> None:
    """Replace a flag's value in place. Raises if absent, so a typo cannot add a dead argument."""
    argv[argv.index(flag) + 1] = value


def poly_updates_at_epoch(dataset: str, warmup_steps: int, epoch: int) -> int:
    """Projector updates a Dual has made by the end of ``epoch`` (1-based).

    train_dual flips warm-up off at global_step == warmup_steps, after that step's optimizer
    update, so warm-up takes warmup_steps + 1 updates. A result <= 0 means that epoch ended
    before the projector had started training.
    """
    return epoch * STEPS_PER_EPOCH[dataset] - warmup_steps - 1


def read_best_epoch(run: Path) -> int:
    """The epoch a finished run selected, read without materializing the weights."""
    import torch  # local: the rest of this module is usable without torch
    ckpt = torch.load(run / "best_val_monoscore_model.pt", map_location="cpu",
                      weights_only=False, mmap=True)
    return int(ckpt["epoch"])


def run_is_complete(run: Path) -> bool:
    """Both files. autoencoder_final.pt marks the end of training, while the best checkpoint is
    rewritten on every improving epoch, so its presence alone proves nothing about completion."""
    return (run / "autoencoder_final.pt").exists() and (run / "best_val_monoscore_model.pt").exists()


def snapshot_plan(row: dict, tcfg: dict) -> Optional[Dict[str, int]]:
    """F's snapshot steps, or None until the Z and R0 runs it must match have finished.

    F (sequential training) is compared with a Dual at the same number of projector updates, and
    that number depends on which epoch the Dual actually selected, so it cannot be known before
    Z and R0 finish. Roles: Z_best / D_best = the selected epoch of Z / R0, dual_final = epoch 50.
    """
    ds, enc, act, seed = row["dataset"], row["encoder"], activation_of(row["method"]), row["seed"]
    z_run, d_run = run_dir(ds, enc, act, "Z", seed), run_dir(ds, enc, act, "R0", seed)
    if not (run_is_complete(z_run) and run_is_complete(d_run)):
        return None
    warm = int(tcfg["warmup_steps"])
    plan = {
        "Z_best": poly_updates_at_epoch(ds, warm, read_best_epoch(z_run)),
        "D_best": poly_updates_at_epoch(ds, warm, read_best_epoch(d_run)),
        "dual_final": poly_updates_at_epoch(ds, warm, EPOCHS),
    }
    # A selected epoch that ended before the projector started has no matching F update count.
    # Dropped here (and flagged by the collector) rather than clamped to 1, which would present an
    # untrained projector as a matched comparison.
    return {role: step for role, step in plan.items() if step > 0}


def build_grid(arms: Optional[List[str]] = None) -> List[dict]:
    """Every training cell, in a stable order. A cell's list position is its manifest index."""
    arms = list(arms) if arms is not None else list(ALL_ARMS)
    unknown = set(arms) - set(ALL_ARMS)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    rows = {(r["dataset"], r["encoder"], activation_of(r["method"]), r["seed"]): r
            for r in load_published_rows()}
    grid = []
    for key in sorted(rows):
        ds, enc, act, seed = key
        row = rows[key]
        tcfg = _published_replay(row)
        for arm in arms:
            if arm in SCALE_ARMS and scale_alias(enc, arm):
                continue
            if span_alias(arm, tcfg["span_coef"]):
                continue
            cell = {"kind": arm, "arm": arm, "dataset": ds, "encoder": enc, "activation": act,
                    "seed": seed, "exp_name": exp_name(ds, enc, act, arm, seed),
                    "published_span": tcfg["span_coef"], "published_warmup": tcfg["warmup_steps"]}
            argv = base_argv(row, tcfg, cell["exp_name"])
            if arm in ARCH_ARMS:
                argv += ARCH_ARMS[arm]
            elif arm in SPAN_ARMS:
                # Only gamma changes; warm-up stays at the cell's value, so L_dcor switches on at
                # the same step as in R0 and the arms differ in weight alone. repr() keeps the
                # exact float on the command line.
                set_arg(argv, "--span_coef", repr(SPAN_ARMS[arm]))
                cell["span_coef"] = SPAN_ARMS[arm]
            elif arm == "Z":
                argv += ["--poly_warmup_zero"]
            elif is_sae_arm(arm):
                # No Residual Branch, so no warm-up to run and nothing for gamma to act on.
                set_arg(argv, "--warmup_steps", "0")
                set_arg(argv, "--span_coef", "0.0")
                argv += ["--poly_arch", "none"]
            if arm in SCALE_ARMS:
                setting = scale_setting(enc, arm, tcfg["n_latents"], tcfg["top_k"])
                set_arg(argv, "--n_latents", str(setting["n_latents"]))
                set_arg(argv, "--topk_k", str(setting["top_k"]))
                cell.update(setting)
            elif arm == "F":
                plan = snapshot_plan(row, tcfg)
                cell["snapshot_plan"] = plan
                set_arg(argv, "--warmup_steps", "0")
                argv += ["--freeze_sparse_from", str(run_dir(ds, enc, act, "S", seed)),
                         # Safe here because nothing is respawned while frozen, so the evaluation
                         # forwards that matter for the other arms have no effect. It removes the
                         # train-set passes that would only re-measure an unchanged sparse branch.
                         "--skip_extra_eval"]
                if plan:
                    argv += ["--snapshot_steps",
                             ",".join(str(s) for s in sorted(set(plan.values())))]
            cell["argv"] = argv
            grid.append(cell)
    return grid


def build_selection_reruns() -> List[dict]:
    """The two retrained x16 expansion points (see SELECTION_RERUNS), as grid cells."""
    by_key = {(c["dataset"], c["encoder"], c["activation"], c["arm"], c["seed"]): c
              for c in build_grid([r["arm"] for r in SELECTION_RERUNS])}
    out = []
    for spec in SELECTION_RERUNS:
        base = by_key[(spec["dataset"], spec["encoder"], spec["activation"], spec["arm"], spec["seed"])]
        cell = json.loads(json.dumps(base))  # deep copy
        sel = spec["select_from_epoch"]
        cell["rerun_of"] = base["exp_name"]
        cell["exp_name"] = f"{base['exp_name']}_sel{sel}"
        cell["select_from_epoch"] = sel
        argv = cell["argv"]
        set_arg(argv, "--exp_name", cell["exp_name"])
        set_arg(argv, "--tb_dir", str(TB_ROOT / cell["exp_name"]))
        argv += ["--select_from_epoch", str(sel)]
        out.append(cell)
    return out


def cell_is_complete(cell: dict) -> bool:
    """Finished training, plus for F every snapshot it was asked for."""
    run = OUTPUT_DIR / cell["exp_name"]
    if not run_is_complete(run):
        return False
    for step in (cell.get("snapshot_plan") or {}).values():
        snap = run.parent / f"{run.name}__step{step}"
        if not (snap / "best_val_monoscore_model.pt").exists():
            return False
    return True


def f_prerequisites_met(cell: dict) -> bool:
    """F needs its S to freeze and the finished Z and R0 runs that fix its snapshot steps."""
    ds, enc, act, seed = cell["dataset"], cell["encoder"], cell["activation"], cell["seed"]
    return (run_is_complete(run_dir(ds, enc, act, "S", seed))
            and run_is_complete(run_dir(ds, enc, act, "Z", seed))
            and run_is_complete(run_dir(ds, enc, act, "R0", seed)))
