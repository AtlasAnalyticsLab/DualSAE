#!/usr/bin/env python3
"""Train the runs of the main SAE-vs-DualSAE comparison from configs/main_cells.json.

Each cell of ``configs/main_cells.json`` holds the exact ``main.py`` arguments of one reported
run (dataset x encoder x method x seed). This driver adds only the run name, the output
directory and the feature root, and trains the selected cells one after another. It is
resumable: a cell whose ``autoencoder_final.pt`` and ``best_val_monoscore_model.pt`` both exist
is skipped (and ``--skip_existing`` is passed to main.py as a second guard).

``--sweep`` instead trains the (gamma, warm-up, lr) settings of ``configs/original_dual_sweep.json``
for the Dual-TopK / Dual-BatchTopK / Dual-JumpReLU cells (every other argument as in the cell's
reported run). The Matryoshka / Gated selection procedure has its own driver,
``stage_cells.py``.

Examples
--------
    python scripts/main/train_cells.py --list
    python scripts/main/train_cells.py --dataset imagenet --encoder clip --method dual_topk
    python scripts/main/train_cells.py --group matryoshka_gated --print     # commands for a scheduler
    python scripts/main/train_cells.py --sweep --dataset open_images --method dual_batch_topk --seed 42

Environment: FEATURES_ROOT (./features) and MAIN_CKPT_DIR (./checkpoints/main), or the flags.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

VISION = Path(__file__).resolve().parents[2]
CELLS = VISION / "configs" / "main_cells.json"
SWEEP = VISION / "configs" / "original_dual_sweep.json"
# Optimizer steps of a 50-epoch run at batch 2048 (drop_last): 625 or 247 steps per epoch.
TOTAL_STEPS = {"imagenet": 625 * 50, "open_images": 247 * 50}


def _g(v) -> str:
    return f"{float(v):g}"


def _set_flag(args: list, flag: str, value) -> list:
    """Replace (or append) `flag value` in an argument list."""
    out = list(args)
    if flag in out:
        out[out.index(flag) + 1] = str(value)
    else:
        out += [flag, str(value)]
    return out


def reported_cells() -> list:
    return json.loads(CELLS.read_text())["cells"]


def sweep_cells(seeds=(42, 43, 44)) -> list:
    """The original Dual sweep, built on top of each cell's reported arguments."""
    base = {(c["dataset"], c["encoder"], c["method"], c["seed"]): c for c in reported_cells()}
    out = []
    for cell in json.loads(SWEEP.read_text())["cells"]:
        for st in cell["settings"]:
            for seed in seeds:
                b = base[(cell["dataset"], cell["encoder"], cell["method"], seed)]
                a = _set_flag(b["args"], "--span_coef", st["span_coef"])
                a = _set_flag(a, "--warmup_steps", st["warmup_steps"])
                a = _set_flag(a, "--lr", st["lr"])
                # Settings whose warm-up exceeds the step budget were trained too (the Residual
                # Branch then never unfreezes); main.py refuses them without this flag.
                if int(st["warmup_steps"]) >= TOTAL_STEPS[cell["dataset"]]:
                    a = a + ["--allow_degenerate_warmup"]
                name = (f"sweep_{cell['dataset']}_{cell['encoder']}_{cell['method']}"
                        f"_span{_g(st['span_coef'])}_warm{st['warmup_steps']}_lr{_g(st['lr'])}_seed{seed}")
                out.append(dict(b, exp_name=name, args=a))
    return out


def is_complete(output_dir: Path, exp_name: str) -> bool:
    d = output_dir / exp_name
    return (d / "autoencoder_final.pt").exists() and (d / "best_val_monoscore_model.pt").exists()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=["topk_batchtopk_jumprelu", "matryoshka_gated"])
    ap.add_argument("--dataset", choices=["imagenet", "open_images"])
    ap.add_argument("--encoder", choices=["clip", "vit", "sigclip"])
    ap.add_argument("--method")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--sweep", action="store_true", help="train configs/original_dual_sweep.json instead")
    ap.add_argument("--output_dir", default=os.environ.get("MAIN_CKPT_DIR", "./checkpoints/main"))
    ap.add_argument("--features_root", default=os.environ.get("FEATURES_ROOT", "./features"))
    ap.add_argument("--tb_root", default=None, help="also write TensorBoard events (<tb_root>/<exp_name>)")
    ap.add_argument("--wandb_project", default=None, help="also log to W&B (optional)")
    ap.add_argument("--python", default=sys.executable)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--print", action="store_true", help="print one command per incomplete cell")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    features_root = str(Path(args.features_root).resolve())
    cells = sweep_cells() if args.sweep else reported_cells()
    sel = [c for c in cells
           if (args.group is None or c.get("group") == args.group)
           and (args.dataset is None or c["dataset"] == args.dataset)
           and (args.encoder is None or c["encoder"] == args.encoder)
           and (args.method is None or c["method"] == args.method)
           and (args.seed is None or c["seed"] == args.seed)]
    n_done = 0
    for c in sel:
        done = is_complete(output_dir, c["exp_name"])
        n_done += done
        if args.list:
            print(f"{'done' if done else 'todo'}  {c['exp_name']}")
            continue
        if done:
            continue
        cmd = [args.python, str(VISION / "main.py"), *c["args"],
               "--exp_name", c["exp_name"], "--output_dir", str(output_dir),
               "--features_root", features_root, "--skip_existing"]
        # Per-epoch Dual checkpoints are never read; skipping them changes only disk usage.
        if "dual" in c["method"] and "--skip_epoch_checkpoints" not in cmd:
            cmd.append("--skip_epoch_checkpoints")
        if args.tb_root and "--tb_dir" not in cmd:
            cmd += ["--tb_dir", str(Path(args.tb_root).resolve() / c["exp_name"])]
        if args.wandb_project:
            cmd += ["--wandb_project", args.wandb_project]
        if args.print:
            print(" ".join(shlex.quote(x) for x in cmd))
            continue
        print(f"[train] {c['exp_name']}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(VISION))
    if args.list:
        print(f"{n_done}/{len(sel)} complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
