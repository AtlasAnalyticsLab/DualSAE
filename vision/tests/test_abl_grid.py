#!/usr/bin/env python3
"""Tests of the ablation-grid definition and the plain drivers (no training, no data).

Each test pins a property the reported tables/figures rely on: every arm differs from the replayed
main-table command only where its ablation says, gamma / expansion aliases skip exactly the
duplicated runs, the sequential arm F reads the update count of the finished joint run, and the
drivers emit runnable main.py commands with explicit feature paths.

Run with:  python -m pytest vision/tests/test_abl_grid.py -q
"""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

VISION_DIR = Path(__file__).resolve().parents[1]
ABL = VISION_DIR / "scripts" / "ablations"
for p in (str(VISION_DIR), str(ABL), str(ABL / "grid")):
    if p not in sys.path:
        sys.path.insert(0, p)

import ablation_grid as C  # noqa: E402
import eval_rows as E  # noqa: E402


@pytest.fixture(autouse=True)
def grid_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FEATURES_ROOT", str(tmp_path / "features"))
    C.set_root(tmp_path / "grid")
    yield tmp_path / "grid"


def flags(argv):
    """argv -> {flag: value or True}; every flag in a main.py command takes at most one value."""
    out, i = {}, 0
    while i < len(argv):
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[argv[i]] = argv[i + 1]
            i += 2
        else:
            out[argv[i]] = True
            i += 1
    return out


def test_static_cells_cover_the_grid():
    rows = C.load_published_rows()
    assert len(rows) == 36
    keys = {(r["dataset"], r["encoder"], r["method"], r["seed"]) for r in rows}
    assert len(keys) == 36
    for r in rows:
        cfg = C.published_config(r)
        assert cfg["n_latents"] == 8192 and cfg["top_k"] == 64 and cfg["normalize"] is False
        # every replayed warm-up fits inside its run (otherwise the arm would be a plain SAE)
        assert cfg["warmup_steps"] < C.TOTAL_STEPS[r["dataset"]]


def test_grid_shape_and_unique_names():
    grid = C.build_grid()
    names = [c["exp_name"] for c in grid]
    assert len(names) == len(set(names))
    per_arm = {}
    for c in grid:
        per_arm[c["arm"]] = per_arm.get(c["arm"], 0) + 1
    for arm in C.CORE_ARMS:
        assert per_arm[arm] == 36, arm
    # ViT x8 is the 8192-latent anchor: never trained as a scale arm
    assert per_arm["E8"] == per_arm["SE8"] == 24


def test_each_arm_differs_from_r0_only_where_intended():
    by = {(c["dataset"], c["encoder"], c["activation"], c["seed"], c["arm"]): c for c in C.build_grid()}
    ignore = {"--exp_name", "--tb_dir"}
    for (ds, enc, act, seed, arm), cell in by.items():
        if arm == "R0":
            continue
        r0 = flags(by[(ds, enc, act, seed, "R0")]["argv"])
        mine = flags(cell["argv"])
        diff = {k for k in set(r0) | set(mine) if k not in ignore and r0.get(k) != mine.get(k)}
        if arm in C.ARCH_ARMS:
            assert diff == {f for f in C.ARCH_ARMS[arm] if f.startswith("--")}, (arm, diff)
        elif arm in C.SPAN_ARMS:
            assert diff == {"--span_coef"} and float(mine["--span_coef"]) == C.SPAN_ARMS[arm]
        elif arm == "Z":
            assert diff == {"--poly_warmup_zero"}
        elif arm == "S":
            assert diff == {"--warmup_steps", "--span_coef", "--poly_arch"}
        elif arm == "F":
            assert diff == {"--warmup_steps", "--freeze_sparse_from", "--skip_extra_eval"}
            assert mine["--freeze_sparse_from"].endswith(C.exp_name(ds, enc, act, "S", seed))
        elif arm in C.SCALE_DUAL_ARMS:
            assert diff <= {"--n_latents", "--topk_k"} and diff
        elif arm in C.SCALE_SAE_ARMS:
            assert diff - {"--n_latents", "--topk_k"} == {"--warmup_steps", "--span_coef", "--poly_arch"}


def test_r0_replays_the_cell_settings_and_explicit_feature_paths(grid_root):
    for row in C.load_published_rows():
        cell = [c for c in C.build_grid(["R0"]) if (c["dataset"], c["encoder"], c["seed"]) ==
                (row["dataset"], row["encoder"], row["seed"]) and c["activation"] == C.activation_of(row["method"])][0]
        f = flags(cell["argv"])
        cfg = C.published_config(row)
        assert float(f["--span_coef"]) == cfg["span_coef"]
        assert int(f["--warmup_steps"]) == cfg["warmup_steps"]
        assert int(f["--dead_check_interval"]) == cfg["dead_check_interval"]
        assert f["--train_path"].startswith(str(grid_root.parent / "features" / row["dataset"]))
        assert ("--test_path" in f) == (row["dataset"] == "open_images")


def test_span_alias_skips_exactly_the_main_table_gamma():
    grid = C.build_grid(list(C.SPAN_ARMS))
    for c in grid:
        assert C.SPAN_ARMS[c["arm"]] != c["published_span"]
    assert C.span_alias("G1e-2", 0.01) == "R0" and C.span_alias("G1e-2", 0.005) is None


def test_scale_settings():
    assert C.scale_setting("clip", "E4") == {"n_latents": 3072, "top_k": 64}
    assert C.scale_setting("sigclip", "SE16") == {"n_latents": 18432, "top_k": 64}
    assert C.scale_setting("vit", "K128") == {"n_latents": 8192, "top_k": 128}
    assert C.scale_alias("vit", "E8") == "R0" and C.scale_alias("vit", "SE8") == "S"
    assert C.scale_alias("clip", "E8") is None


def test_selection_reruns():
    reruns = C.build_selection_reruns()
    assert [c["exp_name"] for c in reruns] == [
        "abl_imagenet_sigclip_batch_topk_E16_seed42_sel29", "abl_imagenet_vit_batch_topk_SE16_seed42_sel10"]
    for c in reruns:
        f = flags(c["argv"])
        assert int(f["--select_from_epoch"]) == c["select_from_epoch"]
        assert f["--exp_name"] == c["exp_name"]


def _finish(run: Path, best_epoch: int):
    run.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": best_epoch}, run / "best_val_monoscore_model.pt")
    torch.save({"epoch": 50}, run / "autoencoder_final.pt")


def test_f_snapshot_plan_follows_the_finished_joint_run():
    row = [r for r in C.load_published_rows() if (r["dataset"], r["encoder"], r["method"], r["seed"]) ==
           ("imagenet", "clip", "dual_topk", 42)][0]
    tcfg = C.published_config(row)
    assert C.snapshot_plan(row, tcfg) is None            # nothing finished yet
    _finish(C.run_dir("imagenet", "clip", "topk", "Z", 42), 33)
    _finish(C.run_dir("imagenet", "clip", "topk", "R0", 42), 32)
    plan = C.snapshot_plan(row, tcfg)
    warm = tcfg["warmup_steps"]
    assert plan == {"Z_best": 33 * 625 - warm - 1, "D_best": 32 * 625 - warm - 1,
                    "dual_final": 50 * 625 - warm - 1}
    _finish(C.run_dir("imagenet", "clip", "topk", "S", 42), 50)
    f_cell = [c for c in C.build_grid(["F"]) if c["exp_name"] == C.exp_name("imagenet", "clip", "topk", "F", 42)][0]
    assert C.f_prerequisites_met(f_cell)
    assert flags(f_cell["argv"])["--snapshot_steps"] == ",".join(str(s) for s in sorted(set(plan.values())))


def test_eval_rows_for_finished_cells():
    cell = [c for c in C.build_grid(["S", "R0"]) if c["dataset"] == "open_images" and c["seed"] == 42][0]
    assert E.rows_for_cell(cell) == []
    _finish(C.OUTPUT_DIR / cell["exp_name"], 30)
    rows = E.rows_for_cell(cell)
    # OpenImagesV7: test and val, best and final (best epoch != 50)
    assert {(r["eval_split"], r["checkpoint"]) for r in rows} == {("test", "best"), ("val", "best"),
                                                                   ("test", "final"), ("val", "final")}
    want = E.REPS_NO_RESIDUAL if C.is_sae_arm(cell["arm"]) else E.REPS_DUAL
    assert all(r["reps"] == want for r in rows)


def test_run_grid_print_stage(grid_root):
    out = subprocess.run([sys.executable, str(ABL / "grid" / "run_grid.py"), "--stage", "print",
                          "--arms", "R0", "--only-dataset", "imagenet", "--only-encoder", "vit",
                          "--root", str(grid_root)], capture_output=True, text=True, check=True).stdout
    assert out.count("main.py --model dual") == 6     # 2 activations x 3 seeds


def test_dcor_and_warmup_drivers_emit_commands(tmp_path):
    out = subprocess.run([sys.executable, str(ABL / "dcor" / "run_dcor.py"), "--stage", "print",
                          "--root", str(tmp_path / "dcor")], capture_output=True, text=True, check=True).stdout
    lines = [l for l in out.splitlines() if l.startswith("$ ")]
    assert len(lines) == 27 + 54                      # ImageNet span-0 replays + OpenImagesV7 pairs
    jr_in = [l for l in lines if "dcor_imagenet_vit_jump_relu_span0_seed42" in l][0]
    assert "--jump_threshold_init 1.0" in jr_in and "--span_coef 0.0" in jr_in and "--normalize" in jr_in
    jr_oi = [l for l in lines if "dcor_open_images_vit_jump_relu_warm2500_span0_seed42" in l][0]
    assert "--jump_threshold_init 0.001" in jr_oi and "--warmup_steps 2500" in jr_oi
    assert subprocess.run(["bash", "-n", str(ABL / "warmup" / "run_warmup.sh")]).returncode == 0
    for script in sorted((ABL / "figures").glob("*.sh")):
        assert subprocess.run(["bash", "-n", str(script)]).returncode == 0, script
