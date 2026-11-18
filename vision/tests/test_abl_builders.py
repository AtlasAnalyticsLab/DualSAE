#!/usr/bin/env python3
"""Smoke test of every ablation table / figure builder on synthetic evaluation outputs.

A fake ablation grid (run folders with checkpoint stubs) and fake branch_metrics.py result files
(same JSON schema, random values) are written to a temporary directory; then every collector,
table builder and plot script runs on them end to end. This checks that the builders read what
the evaluators write and produce their outputs; it says nothing about any real result.

Run with:  python -m pytest vision/tests/test_abl_builders.py -q
"""

import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

VISION_DIR = Path(__file__).resolve().parents[1]
ABL = VISION_DIR / "scripts" / "ablations"
ANA = VISION_DIR / "analysis" / "ablations"
for p in (str(VISION_DIR), str(ABL), str(ABL / "grid"), str(ANA)):
    if p not in sys.path:
        sys.path.insert(0, p)

import ablation_grid as C  # noqa: E402

RNG = random.Random(0)
SPARSE = {k: torch.zeros(2, 2) for k in ("W_enc", "W_dec", "b_enc", "b_dec")}


def fake_result(path: Path, reps, frozen=None):
    """One branch_metrics.py result file with random metric values (schema of the evaluator)."""
    blocks = {}
    for rep in reps:
        blocks[rep] = {"r2_perfeat": RNG.uniform(0.5, 1.0), "r2_global": RNG.uniform(0.5, 1.0),
                       "monoscore": {"mean_all": RNG.uniform(0.2, 0.7)},
                       "knn": {"metric": RNG.uniform(0.4, 0.9)}, "lp": {"metric": RNG.uniform(0.4, 0.9)}}
    level = {"eval_l0_mean": 64.0, "n_active_latents_eval": RNG.randint(6000, 8192), "dict_size": 8192,
             "binary_purity": {"mean_active": RNG.uniform(0.1, 0.6)}}
    if frozen:  # the frozen arm F shares every z-dependent value with its S source
        blocks["xM"].update({k: frozen["reps"]["xM"][k] for k in ("r2_perfeat", "r2_global")})
        blocks["x"]["monoscore"] = frozen["reps"]["x"]["monoscore"]
        level = frozen["model_level"]
    payload = {"self_checks": {"all_ok": True}, "meta": {"best_epoch": 40}, "reps": blocks,
               "model_level": level}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


def finish(run: Path, best_epoch: int):
    run.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": best_epoch, "model_state_dict": SPARSE}, run / "best_val_monoscore_model.pt")
    torch.save({"epoch": 50, "model_state_dict": SPARSE}, run / "autoencoder_final.pt")


def score(run: Path, dataset: str, reps, results: Path, checkpoints=("best",), frozen=None):
    out = {}
    splits = ("val",) if dataset == "imagenet" else ("test", "val")
    for ck in checkpoints:
        for split in splits:
            tag = "" if ck == "best" else f"__{ck}"
            out[(split, ck)] = fake_result(results / f"{run.name}__{split}{tag}.json", reps,
                                           (frozen or {}).get((split, ck)))
    return out


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("abl")
    C.set_root(root / "grid")
    results = C.BRANCH_METRICS_RESULTS
    dual_reps, sae_reps = ("x", "xM", "r_hat", "xhat"), ("x", "xM", "xhat")
    s_scores = {}
    arms = list(C.INDEPENDENT_ARMS) + list(C.SPAN_ARMS)
    for cell in C.build_grid(arms):
        run = C.OUTPUT_DIR / cell["exp_name"]
        finish(run, 40)
        sc = score(run, cell["dataset"], sae_reps if C.is_sae_arm(cell["arm"]) else dual_reps, results,
                   ("best", "final"))
        if cell["arm"] == "S":
            s_scores[(cell["dataset"], cell["encoder"], cell["activation"], cell["seed"])] = sc
    for cell in C.build_grid(["F"]):
        run = C.OUTPUT_DIR / cell["exp_name"]
        finish(run, 50)
        frozen = s_scores[(cell["dataset"], cell["encoder"], cell["activation"], cell["seed"])]
        score(run, cell["dataset"], dual_reps, results, frozen=frozen)
        for step in cell["snapshot_plan"].values():
            snap = run.parent / f"{run.name}__step{step}"
            finish(snap, 50)
            score(snap, cell["dataset"], dual_reps, results, frozen=frozen)
    scale = [c for c in C.build_grid([a for a in C.SCALE_ARMS if a.startswith(("E", "SE"))]) if c["seed"] == 42]
    for cell in scale + C.build_selection_reruns():
        run = C.OUTPUT_DIR / cell["exp_name"]
        finish(run, 45)
        score(run, cell["dataset"], sae_reps if C.is_sae_arm(cell["arm"]) else dual_reps, results)

    # main-table runs (TopK / BatchTopK / JumpReLU pairs) and their evaluation outputs
    main_rows, main_results = [], root / "main_bm"
    for ds in C.DATASETS:
        for enc in C.ENCODERS:
            for method in ("topk", "batch_topk", "jumprelu", "dual_topk", "dual_batch_topk", "dual_jump_relu"):
                for seed in C.SEEDS:
                    run = root / "main" / f"{ds}_{enc}_{method}_{seed}"
                    finish(run, 50)
                    score(run, ds, dual_reps, main_results)
                    main_rows.append({"dataset": ds, "encoder": enc, "method": method, "seed": seed,
                                      "run_dir": str(run), "eval_split": "val" if ds == "imagenet" else "test"})
    manifest = root / "main_runs.json"
    manifest.write_text(json.dumps(main_rows))

    # decorrelation-loss ablation runs
    dcor = root / "dcor"
    for ds in C.DATASETS:
        for enc in C.ENCODERS:
            for act in ("topk", "batch_topk", "jump_relu"):
                for seed in C.SEEDS:
                    names = ([f"dcor_imagenet_{enc}_{act}_span0_seed{seed}"] if ds == "imagenet" else
                             [f"dcor_open_images_{enc}_{act}_warm2500_span{g}_seed{seed}" for g in ("0", "0.01")])
                    for name in names:
                        finish(dcor / "checkpoints" / name, 40)
                        score(dcor / "checkpoints" / name, ds, dual_reps, dcor / "branch_metrics" / "results")
    return {"root": root, "manifest": manifest, "main_results": main_results, "dcor": dcor}


def run_py(script, *args):
    res = subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True,
                         env={**__import__("os").environ, "MPLBACKEND": "Agg"})
    assert res.returncode == 0, res.stderr[-2000:]
    return res.stdout


def test_grid_collector_tables_and_joint_vs_seq_plot(world):
    grid = world["root"] / "grid"
    run_py(ANA / "collect_grid.py", "--root", grid, "--main-manifest", world["manifest"],
           "--main-results", world["main_results"])
    rows = list(csv.DictReader(open(grid / "long.csv")))
    arms = {r["arm"] for r in rows}
    assert {"S", "F", "R0", "Z", "B256", "D4", "published_dual"} <= arms
    assert "PASS" in (grid / "REPORT.md").read_text()          # F-vs-S invariants + decomposition
    run_py(ANA / "make_grid_tables.py", "--long-csv", grid / "long.csv")
    tex = (grid / "frozen_vs_dualsae.tex").read_text()
    assert tex.count("Sequential Training") == 12 and "–" not in tex
    assert (grid / "projector_ablation.md").read_text().count("| Default |") == 12
    run_py(ANA / "plot_joint_vs_seq.py", grid, grid / "long.csv")
    assert (grid / "joint_vs_seq_tradeoff.pdf").exists() and (grid / "joint_vs_seq_tradeoff_all.pdf").exists()


def test_gamma_collector_and_tables(world):
    out = world["root"] / "gamma"
    run_py(ANA / "collect_span_ablation.py", "--grid-root", world["root"] / "grid", "--out", out,
           "--checkpoint", "final")
    run_py(ANA / "make_gamma_tables.py", "--long-csv", out / "long_final.csv")
    for name in ("gamma_ablation_table.tex", "gamma_r2.tex", "gamma_ms.tex"):
        assert (out / name).exists()
    assert (out / "gamma_r2.tex").read_text().count(r"\\") >= 13


def test_expansion_data_and_figures(world):
    out = world["root"] / "expansion"
    run_py(ANA / "collect_scale_ablation.py", "--grid-root", world["root"] / "grid", "--out", out, "--seeds", "42")
    run_py(ANA / "build_expansion_data.py", "--long-csv", out / "long.csv", "--out", out / "data.json",
           "--grid-root", world["root"] / "grid", "--seeds", "42")
    data = json.loads((out / "data.json").read_text())
    assert len(data) == 12
    assert data["imagenet|vit|batch_topk"]["sae"]["16"]["selection_rerun"].endswith("_sel10")
    assert set(data["imagenet|clip|topk"]["dual"]) == {"4", "8", "16", "32", "pub"}  # x32 trained, not plotted
    run_py(ANA / "plot_expansion.py", out / "data.json", out)
    run_py(ANA / "plot_expansion_main.py", out / "data.json", out)
    for name in ("expansion_ratio_imagenet.pdf", "expansion_ratio_openimages.pdf", "expansion_ratio_main.pdf"):
        assert (out / name).exists()


def test_branch_bars(world, tmp_path):
    csv_path = tmp_path / "branch_reps.csv"
    run_py(ANA / "collect_branch_reps.py", "--manifest", world["manifest"], "--results", world["main_results"],
           "--out", csv_path)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    png = tmp_path / "tsne.png"
    plt.figure(figsize=(2, 2)); plt.scatter([0, 1], [0, 1]); plt.savefig(png); plt.close()
    run_py(ANA / "plot_branch_bars.py", "--csv", csv_path, "--out-dir", tmp_path / "figs", "--tsne-png", png)
    for name in ("xm_vs_rhat_grid.pdf", "branch_decomposition.pdf", "xm_vs_rhat_openimages.pdf"):
        assert (tmp_path / "figs" / name).exists()


def test_dcor_table(world):
    run_py(ANA / "make_dcor_table.py", "--dcor-root", world["dcor"], "--main-manifest", world["manifest"],
           "--main-results", world["main_results"])
    md = (world["dcor"] / "dcor_ablation.md").read_text().splitlines()
    assert len(md) == 2 + 9 and all("–" not in line for line in md[2:])


def test_warmup_curves_from_tensorboard_and_logs(tmp_path):
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        from torch.utils.tensorboard import SummaryWriter
    tb = tmp_path / "tb"
    logs = tmp_path / "logs"
    logs.mkdir()
    for w in (0, 2500):
        for seed in (42, 43):
            writer = SummaryWriter(str(tb / f"warmup_open_images_clip_dual_batch_topk_w{w}_seed{seed}"))
            lines = []
            for epoch in range(5):
                r2, ms = 0.5 + 0.1 * epoch, 0.1 + 0.02 * epoch
                writer.add_scalar("val_r2", r2, epoch)
                writer.add_scalar("val_mono_mean", ms, epoch)
                lines += [f"Epoch {epoch + 1}/5 - Val   R2: {r2:.6f}", f"Epoch {epoch + 1}/5 - Val Mono (sparse): mean={ms:.4f}, median=0.0"]
            writer.close()
            (logs / f"run_warmup_{w}_seed_{seed}.txt").write_text("\n".join(lines) + "\n")
    out = run_py(ANA / "plot_warmup_curves.py", "--tb-root", tb, "--warmups", "0", "2500", "--seeds", "42", "43",
                 "--out", tmp_path / "w.png")
    assert "4 runs found" in out and (tmp_path / "w.png").exists()
    out = run_py(ANA / "plot_warmup_curves.py", "--logs", str(logs / "*.txt"), "--warmups", "0", "2500",
                 "--seed-per-warmup", "0:43", "2500:42", "--out", tmp_path / "w2.png")
    assert "4 runs found" in out and "warning" not in out
