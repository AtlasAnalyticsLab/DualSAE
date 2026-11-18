#!/usr/bin/env python3
"""Evaluation rows of the ablation grid: what to score with branch_metrics.py.

One row is one (run directory, split, checkpoint). OpenImagesV7 is scored on test and val,
because purity is read from val and the other metrics from test (the main-table convention);
ImageNet-1K is scored on val only. Only finished runs are emitted, because scoring a
half-trained checkpoint would put a number in a table that no reported model ever produced.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation_grid as C  # noqa: E402

SPLITS = {"imagenet": ("val",), "open_images": ("test", "val")}
REPS_DUAL = "x xM r_hat xhat"
# A zero r_hat bank has no direction to normalize, so requesting it for an SAE arm would fail the
# probe_rows_unit_norm self-check. SAE arms have no Residual Branch to measure in any case.
REPS_NO_RESIDUAL = "x xM xhat"


def _row(cell: dict, run: Path, split: str, checkpoint: str, role: str) -> dict:
    return {
        "run_dir": str(run), "eval_split": split, "checkpoint": checkpoint,
        "dataset": cell["dataset"], "encoder": cell["encoder"],
        "activation": cell["activation"], "arm": cell["arm"], "seed": cell["seed"],
        # F has several readings of one run; this says which comparison each one serves.
        "role": role,
        "reps": REPS_NO_RESIDUAL if C.is_sae_arm(cell["arm"]) else REPS_DUAL,
    }


def rows_for_cell(cell: dict) -> List[dict]:
    """All evaluations of one finished training cell."""
    run = C.OUTPUT_DIR / cell["exp_name"]
    if not C.run_is_complete(run):
        return []
    ds = cell["dataset"]
    rows = []
    if cell["arm"] == "F":
        # F's "best" is its last epoch by construction, so "final" would repeat it. Each snapshot
        # is a directory of its own, loadable by the same tools.
        readings: Dict[str, Path] = {"end_of_training": run}
        for role, step in (cell.get("snapshot_plan") or {}).items():
            readings[f"at_{role}"] = run.parent / f"{run.name}__step{step}"
        # Roles that share a step share a directory; the work is deduplicated by the driver, but
        # every role name is kept because the collector looks readings up by role.
        for role, path in readings.items():
            if not (path / "best_val_monoscore_model.pt").exists():
                continue
            for split in SPLITS[ds]:
                rows.append(_row(cell, path, split, "best", role))
    else:
        # Each arm picks its own best epoch, so the converged model is scored too. A run that
        # selected its last epoch has identical best and final weights, so one reading is enough.
        checkpoints = ["best"]
        if C.read_best_epoch(run) != C.EPOCHS:
            checkpoints.append("final")
        for checkpoint in checkpoints:
            for split in SPLITS[ds]:
                rows.append(_row(cell, run, split, checkpoint, checkpoint))
    return rows
