#!/usr/bin/env python3
"""Write the selection files of the Matryoshka / Gated grid for the REPORTED settings.

The selection procedure (``stage_cells.py`` + ``pick_gated_lambda.py`` / ``pick_dual_winners.py``)
writes four JSON files into ``$MAIN_STATE_DIR`` (default ./outputs/main/selection):

    gated_lambda_choice.json        {"chosen": {"<dataset>/<encoder>": {"l1_coef": ...}}}
    dual_gated_lambda_choice.json   same, for Dual-Gated
    dual_matryoshka_winners.json    {"chosen": {"<dataset>/<encoder>": {"span_coef", "warmup_steps", "exp_name"}}}
    dual_gated_winners.json         same, for Dual-Gated

When the runs are trained directly from ``configs/main_cells.json`` (without re-running the
selection), this script writes the same four files from the settings of the reported cells, so
that downstream steps that read them -- ``stage_cells.py --stage dual-confirm`` and the CB-SAE
pipeline in ``cb_sae/`` (which reads ``dual_matryoshka_winners.json``) -- work unchanged. Only
hyperparameters are written (no metric values). Existing files are left alone unless
``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

VISION = Path(__file__).resolve().parents[2]
CELLS = VISION / "configs" / "main_cells.json"


def _arg(args: list, flag: str):
    return args[args.index(flag) + 1] if flag in args else None


def build() -> dict:
    out = {"gated_lambda_choice.json": {}, "dual_gated_lambda_choice.json": {},
           "dual_matryoshka_winners.json": {}, "dual_gated_winners.json": {}}
    for c in json.loads(CELLS.read_text())["cells"]:
        key, a = f"{c['dataset']}/{c['encoder']}", c["args"]
        # Seed 43 carries the lambda exactly as the choice file stated it (the seed-42 run of a
        # Gated cell is the calibration run itself, whose lambda may have more digits).
        if c["seed"] == 43 and c["method"] in ("gated", "dual_gated"):
            name = "gated_lambda_choice.json" if c["method"] == "gated" else "dual_gated_lambda_choice.json"
            out[name][key] = {"l1_coef": float(_arg(a, "--l1_coef"))}
        # Selection runs on seed 42; seeds 43/44 share its (gamma, warm-up).
        if c["seed"] == 42 and c["method"] in ("dual_matryoshka", "dual_gated"):
            out[f"{c['method']}_winners.json"][key] = {
                "span_coef": float(_arg(a, "--span_coef")),
                "warmup_steps": int(_arg(a, "--warmup_steps")),
                "exp_name": c["exp_name"]}
    return {name: {"source": "configs/main_cells.json (reported settings)", "chosen": chosen}
            for name, chosen in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state_dir", default=os.environ.get("MAIN_STATE_DIR", "./outputs/main/selection"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    d = Path(args.state_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, payload in build().items():
        p = d / name
        if p.exists() and not args.overwrite:
            print(f"[keep] {p}")
            continue
        p.write_text(json.dumps(payload, indent=1))
        print(f"[write] {p} ({len(payload['chosen'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
