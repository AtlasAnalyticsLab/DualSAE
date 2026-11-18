"""Shared constants and readers for the SAE-vs-DualSAE analysis scripts.

Everything the collector, the table builders and the radar plotter agree on
lives here, so the method keys, the split policy and the metric->file-field
mapping are defined exactly once.

Two result sources exist for every trained run:

* **in-training** -- ``<run_dir>/best_metrics.json``, written by the trainers
  after re-evaluating the best-validation-MonoScore checkpoint (the same keys
  the trainers put into the optional W&B summary: ``BEST_R2``,
  ``BEST_MONOSCORE``, ``BEST_PURITY``, ``BEST_ACC_KNN``, ``BEST_ACC_LP`` and,
  when a test split exists, the ``BEST_TEST_*`` twins);
* **offline** -- ``<results_root>/<run_name>__<split>[__<checkpoint>].json``,
  written by ``vision/branch_metrics.py``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Metrics and split policy
# --------------------------------------------------------------------------- #
METRICS = ("R2", "MS", "PUR", "KNN", "LP")
METRIC_LABEL = {"R2": "R$^2$", "MS": "MS", "PUR": "Pur.", "KNN": "KNN", "LP": "LP"}

# Which split each column is read from. ImageNet-1K has no labelled test split,
# so every column is its validation split. On OpenImagesV7 the reconstruction,
# MonoScore and the two probes are read on the test split while purity is read
# on the validation split: the baseline trainers never compute a test-split
# purity, so val is the only split on which purity exists for every model type,
# and using it for both arms keeps the column internally consistent.
SPLIT_BY_METRIC: Dict[str, Dict[str, str]] = {
    "imagenet": {"R2": "val", "MS": "val", "PUR": "val", "KNN": "val", "LP": "val"},
    "open_images": {"R2": "test", "MS": "test", "PUR": "val", "KNN": "test", "LP": "test"},
}
DATASETS = ("imagenet", "open_images")
DATASET_LABEL = {"imagenet": "ImageNet-1K", "open_images": "OpenImagesV7"}
DATASET_SHORT = {"imagenet": "IN", "open_images": "OI"}

ENCODER_ORDER = ("vit", "clip", "sigclip")
ENCODER_LABEL = {"vit": "ViT", "clip": "CLIP", "sigclip": "SigLIP2"}

# --------------------------------------------------------------------------- #
# Method keys
# --------------------------------------------------------------------------- #
# (baseline key, DualSAE key, pair title). The pairing is the ONLY axis along
# which a comparison is ever formed: a baseline is compared with the DualSAE
# that uses the same sparsification rule in its Sparse Branch, never across
# rules. CB-SAE's base is the Matryoshka pair; its DualSAE twin is the CB head
# trained on the Dual-Matryoshka base.
PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("topk", "dual_topk", "TopK"),
    ("batch_topk", "dual_batch_topk", "BatchTopK"),
    ("jumprelu", "dual_jump_relu", "JumpReLU"),
    ("matryoshka", "dual_matryoshka", "MatryoshkaSAE"),
    ("gated", "dual_gated", "GatedSAE"),
    ("cb_sae", "cb_dual_sae", "CB-SAE"),
)
LABEL = {
    "topk": "TK", "dual_topk": "D-TK",
    "batch_topk": "BTK", "dual_batch_topk": "D-BTK",
    "jumprelu": "JR", "dual_jump_relu": "D-JR",
    "matryoshka": "MAT", "dual_matryoshka": "D-MAT",
    "gated": "GATE", "dual_gated": "D-GATE",
    "cb_sae": "CB", "cb_dual_sae": "CB-DUAL",
    "topk_pmatch": "TK (matched)", "batch_topk_pmatch": "BTK (matched)",
    "jumprelu_pmatch": "JR (matched)",
}
# Parameter-matched arm: baseline method -> (matched-arm key, DualSAE key).
PMATCH_PAIRS: Tuple[Tuple[str, str, str, str], ...] = (
    ("topk", "topk_pmatch", "dual_topk", "TopK"),
    ("batch_topk", "batch_topk_pmatch", "dual_batch_topk", "BatchTopK"),
    ("jumprelu", "jumprelu_pmatch", "dual_jump_relu", "JumpReLU"),
)
KNOWN_METHODS = frozenset(LABEL)

# Source policy that matches how each block of the comparison was produced:
# the TopK/BatchTopK/JumpReLU pairs and the parameter-matched arms were read
# from the trainers' own best-checkpoint evaluation, the Matryoshka/Gated pairs
# and the CB rows from the offline evaluator.
IN_TRAINING_METHODS = frozenset({
    "topk", "dual_topk", "batch_topk", "dual_batch_topk", "jumprelu", "dual_jump_relu",
    "topk_pmatch", "batch_topk_pmatch", "jumprelu_pmatch",
})
# CB models are built post hoc from a trained base and never pass through a
# trainer that writes best_metrics.json, so they can only be read offline.
OFFLINE_ONLY_METHODS = frozenset({"cb_sae", "cb_dual_sae"})
SOURCES = ("as_reported", "offline", "in_training")


def source_for(method: str, policy: str) -> str:
    """Resolve the result source ('offline' | 'in_training') of one method."""
    if policy not in SOURCES:
        raise ValueError(f"unknown source policy {policy!r}; use one of {SOURCES}")
    if method in OFFLINE_ONLY_METHODS:
        return "offline"
    if policy == "as_reported":
        return "in_training" if method in IN_TRAINING_METHODS else "offline"
    return policy


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def result_path(results_root: Path, run_dir: str, split: str,
                checkpoint: str = "best") -> Path:
    """Output path of ``branch_metrics.py`` for one (run, split, checkpoint).

    Only a non-default checkpoint is suffixed, mirroring the evaluator's own
    naming, so a run scored at both checkpoints has two distinct files.
    """
    tag = "" if checkpoint == "best" else f"__{checkpoint}"
    return Path(results_root) / f"{Path(run_dir).name}__{split}{tag}.json"


def read_offline(path: Path) -> Dict[str, float]:
    """Five reported columns + L0/dead diagnostics from one evaluator JSON.

    MonoScore is read on the INPUT representation ``x`` (the latents are
    scored against the embeddings they encode), all reconstruction metrics on
    ``xhat``; ``mean_all`` is the unweighted mean over every latent, with
    latents that never fire counted as 0.
    """
    d = json.loads(Path(path).read_text())
    if d.get("status") not in (None, "ok", "complete"):
        raise ValueError(f"{path}: status={d.get('status')!r}")
    ml = d["model_level"]
    dict_size = float(ml["dict_size"])
    return {
        "R2": float(d["reps"]["xhat"]["r2_perfeat"]),
        "MS": float(d["reps"]["x"]["monoscore"]["mean_all"]),
        "PUR": float(ml["binary_purity"]["mean_active"]),
        "KNN": float(d["reps"]["xhat"]["knn"]["metric"]),
        "LP": float(d["reps"]["xhat"]["lp"]["metric"]),
        "_l0": float(ml["eval_l0_mean"]),
        "_dead": (1.0 - float(ml["n_active_latents_eval"]) / dict_size) if dict_size else math.nan,
    }


# best_metrics.json key for each metric, per split.
_IN_TRAINING_SUFFIX = {"R2": "R2", "MS": "MONOSCORE", "PUR": "PURITY",
                       "KNN": "ACC_KNN", "LP": "ACC_LP"}


def in_training_key(metric: str, split: str) -> str:
    prefix = "BEST_" if split == "val" else "BEST_TEST_"
    return prefix + _IN_TRAINING_SUFFIX[metric]


def read_in_training(run_dir: Path) -> Dict[str, Any]:
    return json.loads((Path(run_dir) / "best_metrics.json").read_text())


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def mean_sd(values: Iterable[float]) -> Tuple[Optional[float], Optional[float], int]:
    """Mean and SAMPLE standard deviation (n-1); sd is None for n < 2."""
    xs = [float(v) for v in values]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
    return m, sd, len(xs)


# --------------------------------------------------------------------------- #
# Long-CSV loader shared by the builders
# --------------------------------------------------------------------------- #
def load_long(path: Path, checkpoint: str = "best") -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    """{(dataset, encoder, method, metric): row} from the collector's long CSV."""
    import csv
    out: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("checkpoint", "best") != checkpoint:
                continue
            key = (r["dataset"], r["encoder"], r["method"], r["metric"])
            if key in out:
                raise ValueError(f"duplicate long-CSV row for {key} in {path}")
            row: Dict[str, Any] = dict(r)
            for f in ("mean", "sd", "l0_mean", "dead_frac"):
                row[f] = float(r[f]) if r.get(f) not in (None, "") else None
            row["n_seeds"] = int(r["n_seeds"]) if r.get("n_seeds") else 0
            out[key] = row
    return out


def fmt_num(x: Optional[float], nd: int = 4) -> str:
    return "---" if x is None else f"{x:.{nd}f}"


def ordered_present(values: List[str], order: Iterable[str]) -> List[str]:
    order = list(order)
    return [v for v in order if v in values] + sorted(v for v in values if v not in order)
