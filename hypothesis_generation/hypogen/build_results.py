"""Stage 5 — build the hypothesis-generation results tables from the per-cell evaluation CSVs.

For each dataset (imagenet, openimages), selection rule (lasso, correlation, separation_score — each
reported SEPARATELY; nothing is chosen per cell), encoder and activation pair:

  per (variant, seed): macro metric = mean over the faithful classes in faithful.csv (K = 16)
  -> mean +/- sd over seeds (seeds are the only averaged dimension)
  -> baseline vs DualSAE side by side, Delta = DualSAE - baseline

Metrics: auroc and n_significant (#Significant: hypotheses significant after Bonferroni correction,
family-wise alpha = 0.1), plus auprc. Where each rule's files live:
  lasso            : <root>/eval/<enc>/<act>/<variant>/seed<S>/faithful.csv              (HG_MODE=main)
  correlation, separation_score
                   : <root>/selmethod_<rule>/eval/<enc>/<act>/<variant>/seed<S>/faithful.csv
with <root> = $HG_OUT for imagenet and $HG_OUT/openimages for openimages.

Outputs (default <HG_OUT>/tables): hypogen_long.csv and hypogen_<dataset>_<rule>.md.
Run: python -m hypogen.build_results [--out DIR]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

RULES = ["lasso", "correlation", "separation_score"]
ACTS = ["topk", "batch_topk", "jump_relu"]
ENCODERS = ["clip", "siglip", "vit"]
METRICS = ["auroc", "n_significant", "auprc"]
K = 16


def dataset_root(dataset: str) -> str:
    base = os.environ.get("HG_OUT", "./outputs/hypothesis_generation")
    return base if dataset == "imagenet" else os.path.join(base, "openimages")


def rule_eval_dir(dataset: str, rule: str) -> str:
    root = dataset_root(dataset)
    return os.path.join(root, "eval") if rule == "lasso" else os.path.join(root, f"selmethod_{rule}", "eval")


def load_rule(dataset: str, rule: str) -> pd.DataFrame:
    """Per (encoder, activation, variant, seed) macro averages over classes for one rule."""
    frames = []
    pattern = os.path.join(rule_eval_dir(dataset, rule), "*", "*", "*", "seed*", "faithful.csv")
    for fp in sorted(glob.glob(pattern)):
        if os.path.getsize(fp) == 0:
            continue
        df = pd.read_csv(fp)
        if df.empty:
            continue
        frames.append(df[df["K"] == K])
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    keys = ["encoder", "activation", "variant", "seed"]
    per_seed = df.groupby(keys)[METRICS].mean()
    per_seed["n_classes"] = df.groupby(keys)["class"].nunique()
    return per_seed.reset_index()


def build(out_dir: str) -> pd.DataFrame:
    rows = []
    for dataset in ("imagenet", "openimages"):
        for rule in RULES:
            ps = load_rule(dataset, rule)
            if ps.empty:
                print(f"[build_results] no faithful results for {dataset}/{rule}")
                continue
            for enc in ENCODERS:
                for act in ACTS:
                    rec = {"dataset": dataset, "rule": rule, "encoder": enc, "activation": act}
                    ok = True
                    for var in ("baseline", "dual"):
                        sub = ps[(ps.encoder == enc) & (ps.activation == act) & (ps.variant == var)]
                        rec[f"{var}_n_seeds"] = len(sub)
                        rec[f"{var}_n_classes_min"] = int(sub["n_classes"].min()) if len(sub) else 0
                        for m in METRICS:
                            rec[f"{var}_{m}_mean"] = float(sub[m].mean()) if len(sub) else np.nan
                            rec[f"{var}_{m}_sd"] = float(sub[m].std(ddof=0)) if len(sub) else np.nan
                        ok = ok and len(sub) > 0
                    for m in METRICS:
                        rec[f"delta_{m}"] = (rec[f"dual_{m}_mean"] - rec[f"baseline_{m}_mean"]) if ok else np.nan
                    rows.append(rec)
    os.makedirs(out_dir, exist_ok=True)
    long = pd.DataFrame(rows)
    long_path = os.path.join(out_dir, "hypogen_long.csv")
    long.to_csv(long_path, index=False)
    print(f"[build_results] wrote {long_path} ({len(long)} rows)")
    if long.empty:
        return long

    def ms(m, s, nd):
        return "n/a" if pd.isna(m) else f"{m:.{nd}f}±{s:.{nd}f}"

    def dl(v, nd):
        return "n/a" if pd.isna(v) else f"{v:+.{nd}f}"

    for (dataset, rule), g in long.groupby(["dataset", "rule"], sort=False):
        lines = [f"# Hypothesis generation — {dataset}, selection rule: {rule} (K={K})", "",
                 "Mean ± sd over seeds of the per-cell macro average over faithful classes; "
                 "Δ = DualSAE − activation-matched baseline.", "",
                 "| encoder | activation | AUROC baseline | AUROC DualSAE | Δ AUROC | "
                 "#Sig baseline | #Sig DualSAE | Δ #Sig | seeds (b/d) |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(
                f"| {r.encoder} | {r.activation} | {ms(r.baseline_auroc_mean, r.baseline_auroc_sd, 3)} | "
                f"{ms(r.dual_auroc_mean, r.dual_auroc_sd, 3)} | {dl(r.delta_auroc, 3)} | "
                f"{ms(r.baseline_n_significant_mean, r.baseline_n_significant_sd, 2)} | "
                f"{ms(r.dual_n_significant_mean, r.dual_n_significant_sd, 2)} | "
                f"{dl(r.delta_n_significant, 2)} | {r.baseline_n_seeds}/{r.dual_n_seeds} |")
        p = os.path.join(out_dir, f"hypogen_{dataset}_{rule}.md")
        with open(p, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[build_results] wrote {p}")
    return long


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="default <HG_OUT>/tables")
    a = ap.parse_args()
    out = a.out or os.path.join(os.environ.get("HG_OUT", "./outputs/hypothesis_generation"), "tables")
    build(out)


if __name__ == "__main__":
    main()
