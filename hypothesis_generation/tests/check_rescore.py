#!/usr/bin/env python
"""Code-equivalence check: re-score one cell's faithful evaluation from an EXISTING output tree.

Copies the cell's Stage-2 selection, Stage-3 hypotheses and the cached VLM annotations from
--reference-root (an existing $HG_OUT produced earlier) into a fresh temporary $HG_OUT, runs
`hypogen.evaluate --stage faithful` (no VLM is loaded because every (hypothesis, image) pair is
already cached) and compares the new faithful.csv with the reference one row by row.

Usage:
  python hypothesis_generation/tests/check_rescore.py --reference-root <old HG_OUT> \
      --dataset imagenet --rule correlation --encoder vit --activation jump_relu --variant dual --seed 42
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HG_DIR = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-root", required=True, help="existing HG_OUT (imagenet root)")
    ap.add_argument("--dataset", default="imagenet", choices=["imagenet", "openimages"])
    ap.add_argument("--rule", default="lasso", choices=["lasso", "correlation", "separation_score"])
    ap.add_argument("--encoder", default="vit")
    ap.add_argument("--activation", default="jump_relu")
    ap.add_argument("--variant", default="dual")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--workdir", default=None, help="parent dir for the temporary HG_OUT")
    a = ap.parse_args()

    ref = Path(a.reference_root) / ("openimages" if a.dataset == "openimages" else "")
    tmp_base = Path(tempfile.mkdtemp(prefix="hg_rescore_", dir=a.workdir))
    new = tmp_base / ("openimages" if a.dataset == "openimages" else "")
    stage = "" if a.rule == "lasso" else f"selmethod_{a.rule}"
    cell = Path(a.encoder) / a.activation / a.variant / f"seed{a.seed}"

    # Inputs of the faithful stage (copied, never linked, so the reference tree is never written).
    (new / "embeddings").mkdir(parents=True, exist_ok=True)
    for f in ("val_manifest.json", "val_labels.npy"):
        shutil.copy2(ref / "embeddings" / f, new / "embeddings" / f)
    for sub in ("selected", "interpreted"):
        shutil.copytree(ref / stage / sub / cell, new / stage / sub / cell)
    (new / stage / "eval" / cell).mkdir(parents=True, exist_ok=True)
    shutil.copy2(ref / stage / "eval" / cell / "faithful_annotations.json",
                 new / stage / "eval" / cell / "faithful_annotations.json")

    env = dict(os.environ, HG_OUT=str(tmp_base), HG_DATASET=a.dataset,
               HG_MODE="main" if a.rule == "lasso" else "selmethod", HG_SEL_METHOD=a.rule)
    env["PYTHONPATH"] = os.pathsep.join([str(HG_DIR), env.get("PYTHONPATH", "")])
    subprocess.run([sys.executable, "-m", "hypogen.evaluate", "--stage", "faithful",
                    "--encoder", a.encoder, "--activation", a.activation, "--variant", a.variant,
                    "--seed", a.seed], env=env, check=True)

    old_df = pd.read_csv(ref / stage / "eval" / cell / "faithful.csv")
    new_df = pd.read_csv(new / stage / "eval" / cell / "faithful.csv")
    key = ["class"]
    m = old_df.merge(new_df, on=key, suffixes=("_ref", "_new"), how="outer", indicator=True)
    missing = int((m["_merge"] != "both").sum())
    cols = ["n_hypotheses", "auroc", "auprc", "n_significant", "n_pos", "n_held"]
    diffs = {}
    for c in cols:
        x, y = m[f"{c}_ref"], m[f"{c}_new"]
        bad = ~((x == y) | (x.isna() & y.isna()) | ((x - y).abs() <= 1e-12))
        diffs[c] = int(bad.sum())
    print(f"[check_rescore] {a.dataset}/{a.rule}/{cell}: {len(old_df)} ref rows, {len(new_df)} new rows, "
          f"{missing} unmatched; differing values per column: {diffs}")
    ok = missing == 0 and all(v == 0 for v in diffs.values())
    print("[check_rescore] IDENTICAL" if ok else "[check_rescore] MISMATCH")
    shutil.rmtree(tmp_base, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
