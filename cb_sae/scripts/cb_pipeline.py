#!/usr/bin/env python3
"""Sequential, resumable driver for the CB-SAE / CB-DualSAE pipeline.

One CELL = (dataset, encoder, seed). Each cell has two ARMS that share every
CB setting and differ only in the base:

    plain  CB-SAE      concept head on a pruned Matryoshka-SAE           (--no-poly)
    dual   CB-DualSAE  concept head on a pruned Dual-Matryoshka-SAE, whose
                       Residual Branch ("poly") is warm-started from the base

Stages per arm (every stage is a `python -m featbsf.<module>` CLI with its
own skip-guard, so re-running a command continues where it stopped):

    convert   featbsf.convert_{plain_sae,dualsae}  parity-gated b=1 port
    dissect   featbsf.cb_dissect                   CLIP-Dissect I_m + dead mask
    retain    (driver)                             n_live >= R check, pair-matched
    prune     featbsf.cb_prune                     keep the top-R latents by I_m
    train     featbsf.cb_train                     concept-bottleneck head
    finalize  (driver)                             training_config.json for the evaluator
    eval      vision/branch_metrics.py             the five metrics

Subcommands:

    bases     discover the Matryoshka / Dual-Matryoshka base runs under a
              checkpoint root (from each run's training_config.json) -> bases.json
    prep      offline stage-0 steps per dataset (probe image embeds + gate)
    cell      run every stage for one (dataset, encoder, seed)
    grid      run every stage for every cell in bases.json (the retain check is
              then pair-matched over BOTH arms and ALL seeds of a
              (dataset, encoder), exactly as for the reported rows)
    collect   mean / sd over seeds of every evaluated CB run -> long CSV

Operating point of the CB rows in the main comparison (passed explicitly
because the stage CLIs default to other values): --retain 1024 --k-cb 5
--b-cb 1, interpretability ranking, VLG concept set. The other arms the
original runs covered are available with --retain 1024 2048 --k-cb 5 64.

Paths: FEATURES_ROOT (default ./features), CB_CKPT_DIR (./checkpoints/cb),
CB_WORK_DIR (./outputs/cb_sae), CB_VOCAB_DIR (./assets/cb_sae/concepts), all
relative to the launch directory (the repository root) and passed to every
stage as absolute paths. The bases are read from the main grid's run
directories (default ./checkpoints/main) and its Dual-Matryoshka selection
(default ./outputs/main/selection/dual_matryoshka_winners.json).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

CB_SAE = Path(__file__).resolve().parents[1]           # cb_sae/
REPO = CB_SAE.parent                                   # repository root
VISION = REPO / "vision"


def _abs_env() -> dict:
    """Child environment with every path variable made absolute.

    The stage modules resolve their (relative) defaults at import time
    against THEIR working directory; branch_metrics runs from vision/. Making
    the variables absolute here means every child sees the same tree no
    matter where it runs.
    """
    env = dict(os.environ)
    defaults = {"FEATURES_ROOT": "./features", "CB_WORK_DIR": "./outputs/cb_sae",
                "CB_CKPT_DIR": "./checkpoints/cb",
                "CB_VOCAB_DIR": "./assets/cb_sae/concepts"}
    for k, v in defaults.items():
        env[k] = str(Path(env.get(k, v)).resolve())
    if env.get("CB_LMDB_PATHS"):
        env["CB_LMDB_PATHS"] = str(Path(env["CB_LMDB_PATHS"]).resolve())
    env["PYTHONPATH"] = os.pathsep.join(
        [str(CB_SAE)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    # Runtime is fully offline: every download lives in prepare_assets.sh.
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return env


ENV = _abs_env()
# Import the shared config with the SAME absolute paths the children get.
os.environ.update({k: ENV[k] for k in ("FEATURES_ROOT", "CB_WORK_DIR", "CB_CKPT_DIR",
                                       "CB_VOCAB_DIR")})
if ENV.get("CB_LMDB_PATHS"):
    os.environ["CB_LMDB_PATHS"] = ENV["CB_LMDB_PATHS"]
sys.path.insert(0, str(CB_SAE))
from featbsf import config as FC  # noqa: E402
from featbsf import cb_config as FCB  # noqa: E402

CSET = "vlg"
RANK = "interp"
SEEDS_DEFAULT = (42, 43, 44)
# Main radius / concept budget of the reported rows; collect() labels every
# other arm with a suffix so arms never average into one another.
MAIN_RETAIN, MAIN_KCB = 1024, 5
# Split policy of the evaluation (one result file per split): ImageNet-1K is
# scored on its validation split; OpenImagesV7 on test, except class purity,
# which is read from the validation split.
EVAL_SPLITS = {"imagenet": ("val",), "open_images": ("test", "val")}
SPLIT_BY_METRIC = {
    "imagenet": {"R2": "val", "MS": "val", "PUR": "val", "KNN": "val", "LP": "val"},
    "open_images": {"R2": "test", "MS": "test", "PUR": "val", "KNN": "test", "LP": "test"},
}
METRICS = ("R2", "MS", "PUR", "KNN", "LP")


def _g(v: float) -> str:
    """Compact, round-trippable float tag: 0.005 -> '0.005', 5.0 -> '5'."""
    return f"{float(v):g}"


def run_stage(argv: list[str], log: Path, cwd: Path | None = None) -> None:
    """Run one stage, teeing its output to a log file; abort the driver on failure."""
    log.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(argv), flush=True)
    with open(log, "a") as fh:
        fh.write("+ " + " ".join(argv) + "\n")
        fh.flush()
        proc = subprocess.Popen(argv, cwd=cwd, env=ENV, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            # Flush per line: when the driver's own stdout is a file, progress
            # would otherwise only appear when the buffer fills.
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
            fh.flush()
        rc = proc.wait()
    if rc != 0:
        sys.exit(f"stage failed (exit {rc}): {' '.join(argv)} -- see {log}")


# ---------------------------------------------------------------------------
# Base discovery.
# ---------------------------------------------------------------------------
def _arm_of(tc: dict) -> str | None:
    """'plain' for a Matryoshka-SAE run, 'dual' for a Dual-Matryoshka run."""
    if tc.get("model_type") == "matryoshka":
        return "plain"
    if tc.get("model_type") == "dual" and tc.get("sparsity_mode") == "matryoshka":
        return "dual"
    return None


def converted_run_name(b: dict) -> str:
    """Name of the converted base inside CB_WORK_DIR/checkpoints.

    The name carries the base run's IDENTITY (for the dual arm its
    decorrelation coefficient and warm-up length), and every downstream
    artifact (dissect, prune, CB checkpoint) is named after it. A re-picked
    Dual-Matryoshka setting therefore gets a fresh conversion and a fresh
    chain instead of silently reusing artifacts of the previous base.
    """
    stem = f"cvt_{b['dataset']}_{b['encoder']}"
    if b["arm"] == "plain":
        return f"{stem}_matryoshka_seed{b['seed']}"
    return (f"{stem}_dual_matryoshka_span{_g(b['span_coef'])}"
            f"_warm{int(b['warmup_steps'])}_seed{b['seed']}")


def cmd_bases(a) -> int:
    root = Path(a.ckpt_root).resolve()
    choice = {}
    if a.dual_choice:
        raw = json.loads(Path(a.dual_choice).read_text())
        raw = raw.get("chosen", raw)
        choice = {k: (float(v["span_coef"]), int(v["warmup_steps"])) for k, v in raw.items()}
    found = defaultdict(list)
    for tc_path in sorted(root.rglob("training_config.json")):
        try:
            tc = json.loads(tc_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        arm = _arm_of(tc)
        if arm is None:
            continue
        if int(tc.get("n_latents") or 0) != a.n_latents:
            continue
        if int(tc.get("topk_k", tc.get("top_k")) or 0) != a.topk_k:
            continue
        src = tc_path.parent / "autoencoder_final.pt"
        if not src.exists():
            continue
        ds, enc, seed = tc.get("dataset_name"), tc.get("extractor_name"), tc.get("seed")
        if ds not in FC.DATASETS or enc not in FC.ENCODERS or seed is None:
            continue
        entry = {"dataset": ds, "encoder": enc, "seed": int(seed), "arm": arm,
                 "src_exp": tc_path.parent.name, "src_ckpt": str(src)}
        if arm == "dual":
            entry["span_coef"] = float(tc.get("span_coef", 0.01))
            entry["warmup_steps"] = int(tc.get("warmup_steps", 0))
        found[(ds, enc, int(seed), arm)].append(entry)

    out, problems = [], []
    for key, entries in sorted(found.items()):
        ds, enc, seed, arm = key
        if arm == "dual":
            cell = f"{ds}/{enc}"
            if cell in choice:
                span, warm = choice[cell]
                entries = [e for e in entries
                           if math.isclose(e["span_coef"], span) and e["warmup_steps"] == warm]
            elif a.dual_choice:
                problems.append(f"{cell}: not in {a.dual_choice}")
                continue
        if len(entries) != 1:
            problems.append(f"{key}: {len(entries)} candidate runs "
                            f"({[e['src_exp'] for e in entries]}); pass --dual-choice "
                            f"or move the extra runs out of {root}")
            continue
        e = entries[0]
        e["run"] = converted_run_name(e)
        out.append(e)
    for p in problems:
        print("  !", p)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} base(s) -> {a.out}")
    return 1 if problems else 0


def load_bases(path: str) -> list[dict]:
    bases = json.loads(Path(path).read_text())
    for b in bases:
        b.setdefault("run", converted_run_name(b))
    return bases


# ---------------------------------------------------------------------------
# Stages.
# ---------------------------------------------------------------------------
def _log(name: str) -> Path:
    return FC.LOG_DIR / f"{name}.log"


def stage_prep(dataset: str) -> None:
    """Probe-split image embeds + the alignment gate (offline, idempotent)."""
    run_stage([sys.executable, "-m", "featbsf.cb_text_embeds", "--img-embeds",
               "--dataset", dataset], _log(f"prep_{dataset}"))
    if FCB.alignment_ok_path(dataset).exists():
        print(f"[skip] alignment gate already passed for {dataset}")
        return
    run_stage([sys.executable, "-m", "featbsf.cb_text_embeds", "--check-alignment",
               "--dataset", dataset], _log(f"prep_{dataset}"))


def stage_convert(b: dict, device: str | None) -> None:
    run = b["run"]
    guard = FC.CKPT_DIR / run / "metrics.json"
    if guard.exists():
        # Identity check on top of the name: the recorded source must be the
        # base this entry points at, or the conversion is stale.
        m = json.loads(guard.read_text())
        src = Path(b["src_ckpt"])
        recorded = m.get("converted_from") or (m.get("extra") or {}).get("parent_run")
        if recorded not in (str(src), src.parent.name):
            sys.exit(f"{run}: existing conversion was built from {recorded!r}, "
                     f"not {src} -- remove {guard.parent} to re-convert")
        print(f"[skip] {run} already converted")
        return
    spec = FC.WORK_DIR / "specs" / f"{run}.json"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(json.dumps([{
        "run": run, "dataset": b["dataset"], "encoder": b["encoder"],
        "seed": b["seed"],
        "model_type": "matryoshka" if b["arm"] == "plain" else "dual",
        "variant": None if b["arm"] == "plain" else "dual_matryoshka_sae",
        "src_ckpt": b["src_ckpt"]}], indent=1))
    module = ("featbsf.convert_plain_sae" if b["arm"] == "plain"
              else "featbsf.convert_dualsae")
    argv = [sys.executable, "-m", module, "--spec", str(spec), "--only-run", run]
    if device:
        argv += ["--device", device]
    run_stage(argv, _log(f"convert_{run}"))


def stage_dissect(b: dict, device: str | None) -> None:
    argv = [sys.executable, "-m", "featbsf.cb_dissect", "--run", b["run"]]
    if device:
        argv += ["--device", device]
    run_stage(argv, _log(f"dissect_{b['run']}"))


def n_live(run: str) -> int:
    import numpy as np
    p = FCB.dissect_dir(run) / "dead_mask.npy"
    if not p.exists():
        sys.exit(f"{run}: no dead mask ({p}); run the dissect stage first")
    return int((~np.load(p)).sum())


def retain_plan(bases: list[dict], radii: list[int]) -> dict:
    """{R: set of admitted runs}, pair-matched per (dataset, encoder).

    A (dataset, encoder) is admitted at radius R only if EVERY one of its base
    runs (both arms, all seeds present) has at least R live latents: admitting
    a plain run whose dual twin cannot be pruned to R would produce a CB-SAE
    row with no CB-DualSAE counterpart, and admitting 2 of 3 seeds would make
    a seed mean that is not one.
    """
    by_cell = defaultdict(list)
    for b in bases:
        by_cell[(b["dataset"], b["encoder"])].append(b["run"])
    plan, report = {}, {}
    for r in radii:
        plan[r] = set()
        report[r] = {"admitted": [], "excluded": []}
        for (ds, enc), runs in sorted(by_cell.items()):
            live = {run: n_live(run) for run in runs}
            tight = min(live, key=live.get)
            if live[tight] >= r:
                plan[r].update(runs)
                report[r]["admitted"].append(f"{ds}/{enc}")
            else:
                report[r]["excluded"].append(
                    {"cell": f"{ds}/{enc}", "tightest_run": tight, "n_live": live[tight]})
    out = FC.WORK_DIR / "cb_retain_plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({str(r): v for r, v in report.items()}, indent=1))
    for r, v in report.items():
        excl = ", ".join(f"{e['cell']} (n_live {e['n_live']})" for e in v["excluded"])
        print(f"retain R={r}: {len(v['admitted'])} cell(s) admitted"
              + (f"; excluded: {excl}" if excl else ""))
    return plan


def stage_prune(b: dict, retain: int, device: str | None) -> None:
    argv = [sys.executable, "-m", "featbsf.cb_prune", "--run", b["run"],
            "--cset", CSET, "--retain", str(retain), "--rank", RANK]
    if device:
        argv += ["--device", device]
    run_stage(argv, _log(f"prune_{b['run']}_R{retain}"))


def cb_name(b: dict, retain: int, k_cb: int, b_cb: int) -> str:
    return FCB.cb_run_name(b["run"], CSET, b_cb, b["arm"] == "dual",
                           retain=retain, k_cb=k_cb, rank=RANK)


def stage_train(b: dict, retain: int, k_cb: int, b_cb: int, a) -> None:
    argv = [sys.executable, "-m", "featbsf.cb_train", "--run", b["run"],
            "--cset", CSET, "--retain", str(retain), "--rank", RANK,
            # b_cb=1 / k_cb=5 / R=1024: the operating point of the reported
            # rows; the CLI defaults (4 / concept-set default / 256) differ.
            "--b-cb", str(b_cb), "--k-cb", str(k_cb),
            # --seed is REQUIRED to equal the base run's seed (asserted).
            "--seed", str(b["seed"])]
    if b["arm"] == "plain":
        # A plain SAE has no Residual Branch to warm-start.
        argv.append("--no-poly")
    if a.cb_epochs is not None:
        argv += ["--epochs", str(a.cb_epochs)]
    if a.cb_batch is not None:
        argv += ["--batch", str(a.cb_batch)]
    if a.cb_warmup is not None:
        argv += ["--warmup", str(a.cb_warmup)]
    if a.num_workers is not None:
        argv += ["--num-workers", str(a.num_workers)]
    if a.device:
        argv += ["--device", a.device]
    run_stage(argv, _log(f"train_{cb_name(b, retain, k_cb, b_cb)}"))


def stage_finalize(b: dict, retain: int, k_cb: int, b_cb: int) -> Path:
    """Write training_config.json beside a trained CB checkpoint.

    A CB run is post-hoc: featbsf/cb_train.py emits a cb_dual_bsf `model.pt`,
    not the trainer's (training_config.json + best_val_monoscore_model.pt)
    pair. This writes the missing half so vision/cb_adapter.py can load the
    run and vision/branch_metrics.py scores a CB row through the same path as
    every other row. Idempotent.
    """
    use_poly = b["arm"] == "dual"
    d = FC.CKPT_DIR / cb_name(b, retain, k_cb, b_cb)
    if not (d / "model.pt").exists():
        sys.exit(f"{d}: not trained yet")
    out = d / "training_config.json"
    if out.exists():
        return d
    ds = b["dataset"]
    paths = FC.LMDB_PATHS[ds][b["encoder"]]
    cfg = {
        "model_type": "cb_dual_sae" if use_poly else "cb_sae",
        "exp_name": d.name, "seed": b["seed"],
        "dataset_name": ds, "extractor_name": b["encoder"],
        "retain": retain, "k_cb": k_cb, "b_cb": b_cb, "cset": CSET,
        "use_poly": use_poly, "base_exp_name": b["src_exp"],
        "converted_run": b["run"],
        # Ranking is interpretability-only: feature space has no generative
        # model to steer, so the CB-SAE paper's steerability term is undefined.
        "prune_rank": RANK,
        "n_latents": None,      # filled by the adapter from the checkpoint
        "batch_size": FC.BATCH,
        "normalize": False,
        "multilabel": ds == "open_images",
        # The evaluator asks for val_path / test_path; the split keys differ
        # per dataset ('validation' vs 'valid'), so map explicitly.
        "train_path": str(paths["train"]),
        "val_path": str(paths.get("validation") or paths.get("valid")),
        "test_path": str(paths["test"]) if paths.get("test") else None,
    }
    out.write_text(json.dumps(cfg, indent=1))
    return d


def results_dir(a) -> Path:
    return Path(a.eval_out).resolve() if getattr(a, "eval_out", None) \
        else FC.WORK_DIR / "branch_metrics"


def stage_eval(run_dir: Path, dataset: str, a) -> None:
    results = results_dir(a)
    for split in EVAL_SPLITS[dataset]:
        argv = [sys.executable, "branch_metrics.py", "--run-dir", str(run_dir),
                "--eval-split", split, "--checkpoint", "best",
                "--reps", "x", "xM", "r_hat", "xhat",
                "--retrieval-modes", "self", "bank", "--retrieval-k", "50",
                # 2048 = the training batch; a batch-global top-k gate makes the
                # codes depend on the encode batch size.
                "--encode-batch-size", "2048", "--knn-k", "20",
                "--out-dir", str(results)]
        if a.num_workers is not None:
            argv += ["--num-workers", str(a.num_workers)]
        if a.device:
            argv += ["--device", a.device]
        argv += a.eval_extra
        # branch_metrics imports the vision/ modules by bare name, so it runs
        # from vision/; PYTHONPATH (set in ENV) exposes featbsf/bsf to the
        # CB dispatch in vision/cb_adapter.py.
        run_stage(argv, _log(f"eval_{run_dir.name}_{split}"), cwd=VISION)


# Output of vision/scripts/main/pick_dual_winners.py --variant dual_matryoshka.
_DEFAULT_CHOICE = Path("./outputs/main/selection/dual_matryoshka_winners.json").resolve()

STAGES = ("convert", "dissect", "retain", "prune", "train", "finalize", "eval")


def run_cells(bases: list[dict], a) -> int:
    stages = set(a.stages)
    for ds in sorted({b["dataset"] for b in bases}):
        if "dissect" in stages or "prune" in stages or "train" in stages:
            stage_prep(ds)
    if "convert" in stages:
        for b in bases:
            stage_convert(b, a.device)
    if "dissect" in stages:
        for b in bases:
            stage_dissect(b, a.device)
    admitted = {r: {b["run"] for b in bases} for r in a.retain}
    if "retain" in stages:
        admitted = retain_plan(bases, a.retain)
    for r in a.retain:
        for b in bases:
            if b["run"] not in admitted[r]:
                continue
            if "prune" in stages:
                stage_prune(b, r, a.device)
            for k in a.k_cb:
                if "train" in stages:
                    stage_train(b, r, k, a.b_cb, a)
                if "finalize" in stages or "eval" in stages:
                    d = stage_finalize(b, r, k, a.b_cb)
                    if "eval" in stages:
                        stage_eval(d, b["dataset"], a)
    return 0


def cmd_cell(a) -> int:
    bases = [b for b in load_bases(a.bases)
             if b["dataset"] == a.dataset and b["encoder"] == a.encoder
             and b["seed"] == a.seed and b["arm"] in a.arms]
    if len(bases) != len(a.arms):
        sys.exit(f"bases.json has {len(bases)} of the {len(a.arms)} requested "
                 f"arm(s) for {a.dataset}/{a.encoder}/seed{a.seed}")
    return run_cells(bases, a)


def cmd_grid(a) -> int:
    bases = [b for b in load_bases(a.bases) if b["arm"] in a.arms
             and (not a.datasets or b["dataset"] in a.datasets)
             and (not a.encoders or b["encoder"] in a.encoders)
             and (not a.seeds or b["seed"] in a.seeds)]
    if not bases:
        sys.exit("no bases selected")
    return run_cells(bases, a)


def cmd_prep(a) -> int:
    for ds in a.datasets:
        stage_prep(ds)
    return 0


# ---------------------------------------------------------------------------
# Collection: one long CSV row per (dataset, encoder, method, metric).
# ---------------------------------------------------------------------------
def _read_result(p: Path) -> dict:
    d = json.loads(p.read_text())
    if d.get("status") not in (None, "ok", "complete"):
        raise ValueError(f"{p}: status={d.get('status')!r}")
    return {
        "R2": float(d["reps"]["xhat"]["r2_perfeat"]),
        # MonoScore scores the latents against the RAW input x.
        "MS": float(d["reps"]["x"]["monoscore"]["mean_all"]),
        "PUR": float(d["model_level"]["binary_purity"]["mean_active"]),
        "KNN": float(d["reps"]["xhat"]["knn"]["metric"]),
        "LP": float(d["reps"]["xhat"]["lp"]["metric"]),
        "_l0": float(d["model_level"]["eval_l0_mean"]),
        "_dict": float(d["model_level"]["dict_size"]),
        "_n_active": float(d["model_level"]["n_active_latents_eval"]),
    }


def _method(arm: str, retain: int, k_cb: int) -> str:
    m = "cb_dual_sae" if arm == "dual" else "cb_sae"
    if retain != MAIN_RETAIN:
        m += f"_R{retain}"
    if k_cb != MAIN_KCB:
        m += f"_k{k_cb}"
    return m


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def cmd_collect(a) -> int:
    results = results_dir(a)
    vals = defaultdict(lambda: defaultdict(dict))
    aux = defaultdict(lambda: defaultdict(dict))
    missing = 0
    for b in load_bases(a.bases):
        for r in a.retain:
            for k in a.k_cb:
                d = FC.CKPT_DIR / cb_name(b, r, k, a.b_cb)
                key = (b["dataset"], b["encoder"], _method(b["arm"], r, k))
                for metric in METRICS:
                    split = SPLIT_BY_METRIC[b["dataset"]][metric]
                    p = results / f"{d.name}__{split}.json"
                    if not p.exists():
                        missing += 1
                        continue
                    got = _read_result(p)
                    vals[key][metric][b["seed"]] = got[metric]
                    aux[key]["l0"][b["seed"]] = got["_l0"]
                    aux[key]["dead"][b["seed"]] = 1.0 - got["_n_active"] / got["_dict"]
                    aux[key]["units"][b["seed"]] = got["_dict"]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "encoder", "method", "metric", "split", "mean", "sd",
                    "n_seeds", "l0_mean", "dead_frac", "units"])
        for (ds, enc, meth), per_metric in sorted(vals.items()):
            l0 = _mean_sd(list(aux[(ds, enc, meth)]["l0"].values()))[0]
            dead = _mean_sd(list(aux[(ds, enc, meth)]["dead"].values()))[0]
            units = _mean_sd(list(aux[(ds, enc, meth)]["units"].values()))[0]
            for metric in METRICS:
                seeds = per_metric.get(metric, {})
                if not seeds:
                    continue
                m, sd = _mean_sd(list(seeds.values()))
                w.writerow([ds, enc, meth, metric, SPLIT_BY_METRIC[ds][metric],
                            f"{m:.6f}", f"{sd:.6f}", len(seeds), f"{l0:.2f}",
                            f"{dead:.4f}", f"{units:.0f}"])
    print(f"wrote {out} ({len(vals)} method rows; {missing} result file(s) missing)")
    return 0


# ---------------------------------------------------------------------------
def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bases", required=True, help="bases.json written by the 'bases' subcommand")
    p.add_argument("--arms", nargs="+", default=["plain", "dual"], choices=["plain", "dual"])
    p.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    # Operating point of the reported CB rows; pass several values for the
    # robustness arms (--retain 1024 2048 --k-cb 5 64).
    p.add_argument("--retain", nargs="+", type=int, default=[MAIN_RETAIN])
    p.add_argument("--k-cb", nargs="+", type=int, default=[MAIN_KCB])
    p.add_argument("--b-cb", type=int, default=1)
    p.add_argument("--device", default=None, help="passed to every stage (default: each stage's own)")
    p.add_argument("--eval-out", default=None,
                   help="branch_metrics result dir (default $CB_WORK_DIR/branch_metrics)")
    p.add_argument("--num-workers", type=int, default=None)
    # Testing knobs (leave unset for the reported protocol: 25 epochs, batch
    # 2048, 2500 CB-first warm-up steps -- the stage defaults).
    p.add_argument("--cb-epochs", type=int, default=None)
    p.add_argument("--cb-batch", type=int, default=None)
    p.add_argument("--cb-warmup", type=int, default=None)
    p.add_argument("--eval-extra", nargs=argparse.REMAINDER, default=[],
                   help="extra flags appended to branch_metrics.py (must come last)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("bases", help="discover base runs -> bases.json")
    p.add_argument("--ckpt-root", default="./checkpoints/main",
                   help="root holding the vision/main.py run directories")
    p.add_argument("--dual-choice", default=_DEFAULT_CHOICE if _DEFAULT_CHOICE.exists() else None,
                   help="JSON {'chosen': {'<dataset>/<encoder>': {'span_coef':..,"
                        "'warmup_steps':..}}} selecting the reported Dual-Matryoshka "
                        "setting per cell (needed when several settings exist)")
    p.add_argument("--n-latents", type=int, default=8192)
    p.add_argument("--topk-k", type=int, default=64)
    p.add_argument("--out", default=str(FC.WORK_DIR / "bases.json"))
    p.set_defaults(fn=cmd_bases)

    p = sub.add_parser("prep", help="offline stage-0 steps (probe embeds + alignment gate)")
    p.add_argument("--datasets", nargs="+", default=list(FC.DATASETS), choices=list(FC.DATASETS))
    p.set_defaults(fn=cmd_prep)

    p = sub.add_parser("cell", help="all stages for one (dataset, encoder, seed)")
    p.add_argument("--dataset", required=True, choices=list(FC.DATASETS))
    p.add_argument("--encoder", required=True, choices=list(FC.ENCODERS))
    p.add_argument("--seed", required=True, type=int)
    _common(p)
    p.set_defaults(fn=cmd_cell)

    p = sub.add_parser("grid", help="all stages for every cell in bases.json")
    p.add_argument("--datasets", nargs="+", default=None, choices=list(FC.DATASETS))
    p.add_argument("--encoders", nargs="+", default=None, choices=list(FC.ENCODERS))
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    _common(p)
    p.set_defaults(fn=cmd_grid)

    p = sub.add_parser("collect", help="seed mean/sd of every evaluated CB run -> CSV")
    p.add_argument("--bases", required=True)
    p.add_argument("--retain", nargs="+", type=int, default=[MAIN_RETAIN])
    p.add_argument("--k-cb", nargs="+", type=int, default=[MAIN_KCB])
    p.add_argument("--b-cb", type=int, default=1)
    p.add_argument("--eval-out", default=None,
                   help="branch_metrics result dir (default $CB_WORK_DIR/branch_metrics)")
    p.add_argument("--out", default=str(FC.WORK_DIR / "cb_long.csv"))
    p.set_defaults(fn=cmd_collect)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
