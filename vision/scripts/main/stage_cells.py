#!/usr/bin/env python3
"""Turn one stage of the Matryoshka / Gated grid into training commands, and run them.

Stages (run in this order; each later stage reads the picker output of an earlier one):

  matryoshka                18 cells  MatryoshkaSAE baseline (3 seeds)
  gated-lambda              48 cells  seed-42 lambda calibration of GatedSAE (it has no k)
  gated-lambda-bisect        6 cells  log-log refinement of cells that bracket L0=64
  gated                     18 cells  GatedSAE at the calibrated lambda (the seed-42 cells reuse the
                                      calibration run's exp_name, so only seeds 43/44 are new)
  dual-gated-lambda         24 cells  seed-42 lambda check of Dual-Gated (x0.1, x1/3, x1, x3 the
                                      baseline lambda) at the (gamma, warm-up) grid centre
  dual-gated-lambda-bisect   6 cells  refinement of the Dual-Gated lambda
  dual-gated-lambda-floor  <=6 cells  lambda=0 ceiling probe for cells with no upper bracket
  dual-sweep                65 cells  per variant: seed-42 (gamma = span_coef, warm-up) plus/cross
  dual-confirm              18 cells  per variant: seeds 43/44 at the winning setting

Between stages run the evaluation of the new runs (``scripts/main/eval_runs.sh`` or
``l0_probe.py`` for an L0 only), ``l0_trajectory.py`` and the picker:

    gated-lambda[-bisect]       -> pick_gated_lambda.py               -> gated_lambda_choice.json
    dual-gated-lambda[-bisect]  -> pick_gated_lambda.py --arm dual    -> dual_gated_lambda_choice.json
    dual-sweep                  -> pick_dual_winners.py --variant V   -> V_winners.json

The picker JSONs live in ``$MAIN_STATE_DIR`` (default ./outputs/main/selection). Runs go to
``$MAIN_CKPT_DIR`` (default ./checkpoints/main), TensorBoard events (needed by l0_trajectory.py)
to ``$MAIN_TB_DIR``.

Usage
-----
    python scripts/main/stage_cells.py --stage matryoshka                 # list the cells
    python scripts/main/stage_cells.py --stage matryoshka --print         # one command per line
    python scripts/main/stage_cells.py --stage matryoshka --run           # train missing cells
    python scripts/main/stage_cells.py --stage dual-sweep --variant dual_gated --run

``--run`` trains the incomplete cells one after another (sequential and resumable: a cell is
complete when both ``autoencoder_final.pt`` and ``best_val_monoscore_model.pt`` exist; main.py
also gets ``--skip_existing``). ``--print`` emits the per-cell commands for a scheduler.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
VISION = _HERE.parents[1]
for _p in (str(_HERE), str(VISION)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import grid_config as C  # noqa: E402

STAGES = ("matryoshka", "gated-lambda", "gated-lambda-bisect", "gated", "dual-gated-lambda",
          "dual-gated-lambda-bisect", "dual-gated-lambda-floor", "dual-sweep", "dual-confirm")


# --------------------------------------------------------------------------- #
# argv construction -- the ONE place a cell becomes a command line
# --------------------------------------------------------------------------- #
def build_argv(cell: dict, output_dir: str = C.OUTPUT_DIR, tb_root: str = C.TB_ROOT) -> list:
    """main.py arguments of one cell (values the reported runs used)."""
    ds, enc, exp = cell["dataset"], cell["encoder"], cell["exp_name"]
    argv = [
        "--model", cell["model"], "--dataset", ds, "--extractor", enc,
        "--n_latents", str(C.N_LATENTS), "--topk_k", str(C.TOPK_K),
        "--batch_size", str(C.BATCH), "--num_epochs", str(C.EPOCHS),
        "--eps", str(C.EPS), "--seed", str(cell["seed"]),
        "--exp_name", exp, "--output_dir", output_dir,
        # TensorBoard events are what l0_trajectory.py reads for selection.
        "--tb_dir", f"{tb_root}/{exp}",
        "--skip_existing",
        # Per-epoch dual checkpoints (~150MB each) are never read.
        "--skip_epoch_checkpoints",
        "--mono_coef", "0.0",
    ]
    if cell["model"] == "matryoshka":
        p = C.MATRYOSHKA_PROTOCOL
        argv += ["--lr", str(p["lr"]), "--l1_coef", str(p["l1_coef"]),
                 "--bandwidth", str(p["bandwidth"]),
                 "--max_grad_norm", str(p["max_grad_norm"]),
                 "--auxk_coef", str(p["auxk_coef"]),
                 "--group_fractions", p["group_fractions"]]
        # normalize=False is main.py's default; the recipe is expressed by OMITTING --normalize.
        assert p["normalize"] is False
    elif cell["model"] == "gated":
        p = C.GATED_PROTOCOL
        # lambda IS the sparsity knob for this architecture -- there is no k. A command without
        # it would silently train at main.py's unrelated default, so fail loudly.
        if cell.get("l1_coef") is None:
            raise ValueError(f"{cell['exp_name']}: gated cell has no l1_coef; "
                             f"run the lambda calibration stage first")
        argv += ["--lr", str(p["lr"]),
                 "--l1_coef", str(cell["l1_coef"]),
                 "--max_grad_norm", str(p["max_grad_norm"]),
                 "--gated_sparsity_warmup", str(p["gated_sparsity_warmup"])]
        assert p["normalize"] is False
    elif cell["model"] == "dual":
        variant = cell["variant"]
        mode = {"dual_matryoshka": "matryoshka", "dual_gated": "gated"}[variant]
        argv += ["--sparsity_mode", mode,
                 "--span_coef", str(cell["span_coef"]),
                 "--warmup_steps", str(cell["warmup_steps"]),
                 "--poly_layers", "1", "--poly_hidden", "512",
                 "--poly_lr_factor", "1.0",
                 # The duals get dead-latent respawn (train_dual reads this; train_other has no
                 # such knob, so the baselines do not).
                 "--dead_check_interval", "100"]
        if mode == "matryoshka":
            p = C.MATRYOSHKA_PROTOCOL
            argv += ["--lr", str(p["lr"]), "--l1_coef", str(p["l1_coef"]),
                     "--bandwidth", str(p["bandwidth"]),
                     "--max_grad_norm", str(p["max_grad_norm"]),
                     "--group_fractions", p["group_fractions"]]
        else:
            p = C.GATED_PROTOCOL
            if cell.get("l1_coef") is None:
                raise ValueError(
                    f"{cell['exp_name']}: dual_gated cell has no l1_coef. Dual-Gated's lambda is "
                    f"calibrated on the DUAL arm -- run the dual-gated-lambda stage first.")
            # --max_grad_norm is passed for symmetry with the baseline; train_dual never clips.
            argv += ["--lr", str(p["lr"]), "--l1_coef", str(cell["l1_coef"]),
                     "--max_grad_norm", str(p["max_grad_norm"]),
                     "--gated_sparsity_warmup", str(p["gated_sparsity_warmup"])]
    else:
        raise ValueError(f"unknown model {cell['model']!r}")
    return argv


def is_complete(exp_name: str, output_dir: str = C.OUTPUT_DIR) -> bool:
    """Both files: autoencoder_final.pt is written only at the very end, while
    best_val_monoscore_model.pt is rewritten on every improving epoch."""
    d = Path(output_dir) / exp_name
    return (d / "autoencoder_final.pt").exists() and (d / "best_val_monoscore_model.pt").exists()


def _load_json(path: Path, what: str) -> dict:
    if not path.exists():
        sys.exit(f"missing {what}: {path}")
    return json.loads(path.read_text())


def _chosen_lambdas(path: Path, what: str) -> dict:
    raw = _load_json(path, what)
    return {(k.split("/")[0], k.split("/")[1]): v["l1_coef"] for k, v in raw["chosen"].items()}


def build_grid(stage: str, variant: str | None, state_dir: Path) -> list:
    """The cells of one stage (placeholders for unresolved cells are kept, then skipped)."""
    base_lambda_json = state_dir / "gated_lambda_choice.json"
    dual_lambda_json = state_dir / "dual_gated_lambda_choice.json"
    if stage == "matryoshka":
        return C.matryoshka_baseline_grid()
    if stage == "gated-lambda":
        return C.gated_lambda_grid()
    if stage == "gated-lambda-bisect":
        br = _load_json(base_lambda_json, "lambda choice (run pick_gated_lambda.py)").get("brackets") or {}
        return C.gated_lambda_bisect_grid(br)
    if stage == "dual-gated-lambda-bisect":
        br = _load_json(dual_lambda_json, "dual lambda choice (pick_gated_lambda.py --arm dual)").get("brackets") or {}
        return C.dual_gated_lambda_bisect_grid(br)
    if stage == "dual-gated-lambda-floor":
        raw = json.loads(dual_lambda_json.read_text()) if dual_lambda_json.exists() else {}
        return C.dual_gated_lambda_floor_grid(raw)
    if stage == "gated":
        return C.gated_baseline_grid(_chosen_lambdas(base_lambda_json, "lambda choice"))
    if stage == "dual-gated-lambda":
        return C.dual_gated_lambda_grid(_chosen_lambdas(base_lambda_json, "lambda choice"))
    if not variant:
        sys.exit("--variant is required for the dual stages")
    # Dual-Gated's lambda is calibrated on the DUAL arm, never inherited from the baseline.
    lam = _chosen_lambdas(dual_lambda_json, "dual lambda choice") if variant == "dual_gated" else None
    if stage == "dual-sweep":
        return C.dual_sweep_grid(variant, lam)
    if stage == "dual-confirm":
        raw = _load_json(state_dir / f"{variant}_winners.json",
                         f"winners (run pick_dual_winners.py --variant {variant})")
        if not raw.get("chosen"):
            sys.exit("the winners file resolves 0 cells; refusing to confirm a partial sweep")
        return C.dual_confirm_grid(variant, raw["chosen"], lam)
    raise ValueError(stage)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--variant", choices=C.DUAL_VARIANTS, default=None)
    ap.add_argument("--state_dir", default=C.STATE_DIR, help="picker JSONs (default $MAIN_STATE_DIR)")
    ap.add_argument("--output_dir", default=C.OUTPUT_DIR, help="runs (default $MAIN_CKPT_DIR)")
    ap.add_argument("--tb_root", default=C.TB_ROOT, help="TensorBoard root (default $MAIN_TB_DIR)")
    ap.add_argument("--only_dataset", choices=C.DATASETS, default=None)
    ap.add_argument("--only_encoder", choices=C.ENCODERS, default=None)
    ap.add_argument("--python", default=sys.executable)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--print", action="store_true", help="print one command per incomplete cell")
    mode.add_argument("--run", action="store_true", help="train the incomplete cells sequentially")
    args = ap.parse_args()
    if args.stage.startswith("dual-gated-lambda"):
        args.variant = "dual_gated"

    grid = build_grid(args.stage, args.variant, Path(args.state_dir))
    todo = []
    for idx, cell in enumerate(grid):
        if cell.get("uncalibrated"):
            continue  # placeholder: its lambda / winner is not resolved yet
        if args.only_dataset and cell["dataset"] != args.only_dataset:
            continue
        if args.only_encoder and cell["encoder"] != args.only_encoder:
            continue
        done = is_complete(cell["exp_name"], args.output_dir)
        if not (args.print or args.run):
            print(f"[{idx:3d}] {'done' if done else 'todo'}  {cell['exp_name']}")
        if not done:
            todo.append(cell)
    if not (args.print or args.run):
        print(f"{len(todo)} incomplete of {len(grid)} cells")
        return 0
    for cell in todo:
        cmd = [args.python, str(VISION / "main.py")] + build_argv(cell, args.output_dir, args.tb_root)
        if args.print:
            print(" ".join(shlex.quote(c) for c in cmd))
            continue
        print(f"[train] {cell['exp_name']}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(VISION))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
