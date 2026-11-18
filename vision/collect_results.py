#!/usr/bin/env python3
"""Collect the per-run best-epoch metrics of a training tree into one table.

Every finished training run drops two small JSON files into its own directory
``<output_dir>/<exp_name>/``:

* ``best_metrics.json``    -- the best-epoch metrics (written by the trainers
  through :func:`write_best_metrics`, which lives in this module so the writer
  and the reader of the schema stay in one place).
* ``training_config.json`` -- ``vars(TrainingConfig)`` of that run (written by
  ``main.py``; standalone ``python train_*.py`` runs do not write it).

This script walks such a tree and emits ONE row per run, joining the metrics
with the run metadata (one row per encoder x activation x seed x setting).

Usage
-----
    python collect_results.py --root ./checkpoints --output outputs/runs.csv
    python collect_results.py --root ./checkpoints --output outputs/runs.json --format json

Only the standard library is used (no torch, no pandas), so the table can be
built on a laptop from a copied checkpoint tree.

Conventions worth knowing before reading the output
---------------------------------------------------
* ``warmup_steps`` is always a column. It is only *filled* for DualSAE runs:
  the baseline trainers have no warm-up phase at all, so their cell is left
  EMPTY rather than 0 (a warm-up plot then drops the row instead of merging
  baselines into the warm-up=0 bucket of a Dual curve).
* ``decorrelation_coef`` is the paper name of the DualSAE ``span_coef``
  (coefficient gamma of L_dcor); it is read from ``span_coef``.
* ``dataset`` / ``extractor`` are not stored in ``TrainingConfig``; they are
  inferred from the LMDB paths recorded in it (and from the run name as a
  fallback). See :func:`_infer_dataset` / :func:`_infer_extractor`.
* Rows are sorted deterministically so re-running the collector on the same
  tree produces a byte-identical file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BEST_METRICS_FILENAME = 'best_metrics.json'
TRAINING_CONFIG_FILENAME = 'training_config.json'

# Metadata columns, in output order. Everything a reader needs to identify the
# cell a row belongs to; `run_dir`/`meta_source` are provenance, not settings.
METADATA_COLUMNS: Tuple[str, ...] = (
    'exp_name',
    'model',
    'sparsity_mode',
    'extractor',
    'dataset',
    'seed',
    'warmup_steps',
    'decorrelation_coef',
    'n_latents',
    'topk_k',
    'lr',
    'run_dir',
    'meta_source',
)

# Preferred order of the metric columns; any further key found in a
# best_metrics.json is appended afterwards in alphabetical order.
PREFERRED_METRIC_ORDER: Tuple[str, ...] = (
    'val_mono_mean',
    'BEST_MONOSCORE',
    'BEST_EPOCH',
    'BEST_R2',
    'BEST_PURITY',
    'BEST_ACC_KNN',
    'BEST_ACC_LP',
    'BEST_TEST_MONOSCORE',
    'BEST_TEST_R2',
    'BEST_TEST_PURITY',
    'BEST_TEST_ACC_KNN',
    'BEST_TEST_ACC_LP',
)

# Baseline model -> the MonoBranch activation it must be compared against.
# A baseline is only ever compared with the Dual model that uses the SAME
# activation, and per-activation filters match on the `sparsity_mode`
# column, so baselines get the matching value here too.
_BASELINE_SPARSITY_MODE = {
    'batch_topk': 'batch_topk',
    'jumprelu': 'jump_relu',
    'vanilla': 'vanilla',
}

# Encoder tags as they appear in feature LMDB names / run names. Order matters:
# the CLIP feature file is named ``clip_image_features_CLIP-ViT-L-14-*.lmdb``,
# i.e. it also contains the substring "vit", so the more specific tags have to
# be tested first.
_EXTRACTOR_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ('dino', 'dino'),
    ('siglip', 'sigclip'),
    ('sigclip', 'sigclip'),
    ('clip', 'clip'),
    ('vit', 'vit'),
)


# --------------------------------------------------------------------------- #
# Write side: used by train_dual.py / train_topk.py / train_other.py
# --------------------------------------------------------------------------- #

def _jsonable(value: Any) -> Optional[Any]:
    """Coerce a metric value to a JSON-safe scalar, or None if it cannot be used.

    Handles the value types the trainers actually produce: Python scalars and
    0-d numpy/torch values (both expose ``.item()``). Non-finite floats (NaN
    from a metric that was never computed, +-inf) are rejected because they are
    not valid JSON and would poison a downstream mean; the caller drops the key
    instead, which is the same signal as "this trainer does not report it".
    """
    if value is None:
        return None
    # numpy scalars / 0-d tensors -> Python scalar.
    if hasattr(value, 'item'):
        try:
            value = value.item()
        except Exception:
            return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(as_float):
        return None
    return as_float


def write_best_metrics(output_dir: str, metrics: Dict[str, Any],
                       filename: str = BEST_METRICS_FILENAME) -> Optional[Path]:
    """Serialize best-epoch ``metrics`` to ``<output_dir>/<filename>``.

    This is the on-disk twin of the W&B ``run.summary`` / TensorBoard ``best/*``
    entries the trainers already emit: same keys, same values, but readable
    without any optional dependency, so tables and figures can be built from a
    plain checkpoint tree.

    Contract:
    * Keys whose value is None or non-finite are DROPPED, so a run that skipped
      part of the evaluation simply lacks those keys (readers must treat a
      missing key as "not computed", never as zero).
    * The write is atomic (temp file + ``os.replace``) so an interrupted job can
      never leave a half-written file that later crashes the collector.
    * It NEVER raises: a metrics file is a convenience artefact and must not be
      able to kill a training run that has already finished and saved its
      checkpoints. Failures are reported as a warning and return None.
    """
    try:
        payload = {}
        for key, value in metrics.items():
            coerced = _jsonable(value)
            if coerced is None:
                continue
            payload[str(key)] = coerced

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path = out_dir / filename
        tmp_path = out_dir / f'.{filename}.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write('\n')
        os.replace(tmp_path, final_path)
        print(f"Best-epoch metrics written to {final_path}")
        return final_path
    except Exception as exc:  # noqa: BLE001 - never propagate into the training run
        print(f"Warning: could not write {filename} to {output_dir}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Read side: run-directory discovery and metadata extraction
# --------------------------------------------------------------------------- #

def _load_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load a JSON object, returning ``(data, error)``.

    A corrupt or non-object file yields ``(None, reason)`` so the caller can
    warn and keep going instead of aborting a sweep-wide collection.
    """
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001 - unreadable/corrupt file
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(data, dict):
        return None, 'top-level JSON value is not an object'
    return data, None


def _first_extractor_tag(text: str) -> str:
    """Return the encoder tag whose pattern appears first in ``text``."""
    lowered = text.lower()
    for pattern, tag in _EXTRACTOR_PATTERNS:
        if pattern in lowered:
            return tag
    return ''


def _infer_dataset(*candidates: Optional[str]) -> str:
    """Infer the dataset name from LMDB paths / the run name.

    ``TrainingConfig`` stores no dataset field (``main.py`` only uses
    ``--dataset`` to pick the config file), so the paths are the ground truth.
    OpenImages is checked first because its trainers key multi-label mode off
    exactly this substring.
    """
    for candidate in candidates:
        if not candidate:
            continue
        lowered = str(candidate).lower()
        if 'open_image' in lowered or 'openimages' in lowered:
            return 'open_images'
        if 'imagenet' in lowered or 'ilsvrc' in lowered:
            return 'imagenet'
    return ''


def _infer_extractor(*candidates: Optional[str]) -> str:
    """Infer the feature extractor from LMDB paths / the run name."""
    for candidate in candidates:
        if not candidate:
            continue
        tag = _first_extractor_tag(str(candidate))
        if tag:
            return tag
    return ''


def _parse_run_name(name: str) -> Dict[str, Any]:
    """Best-effort metadata recovery from a run directory name.

    Only used when ``training_config.json`` is absent (a standalone
    ``python train_dual.py ...`` run writes checkpoints but no config dump).
    The patterns follow exp_names such as
    ``imagenet_clip_dual_batch_topk_warmup2500_seed42``. Anything the name
    does not encode stays absent from the returned dict, and the caller leaves
    that column empty rather than guessing.
    """
    parsed: Dict[str, Any] = {}
    lowered = name.lower()

    dataset = _infer_dataset(lowered)
    if dataset:
        parsed['dataset'] = dataset
    extractor = _infer_extractor(lowered)
    if extractor:
        parsed['extractor'] = extractor

    # Model + activation. 'dual' wins over the activation tokens, which appear
    # in Dual run names too (``..._dual_batch_topk_...``).
    if 'dual' in lowered:
        parsed['model'] = 'dual'
        mode_match = re.search(r'dual[_-](batch_topk|jump_relu|jumprelu|topk|vanilla)', lowered)
        if mode_match:
            mode = mode_match.group(1)
            parsed['sparsity_mode'] = 'jump_relu' if mode == 'jumprelu' else mode
    elif 'batch_topk' in lowered:
        parsed['model'] = 'batch_topk'
        parsed['sparsity_mode'] = 'batch_topk'
    elif 'jumprelu' in lowered or 'jump_relu' in lowered:
        parsed['model'] = 'jumprelu'
        parsed['sparsity_mode'] = 'jump_relu'
    elif 'vanilla' in lowered:
        parsed['model'] = 'vanilla'
        parsed['sparsity_mode'] = 'vanilla'
    elif 'topk' in lowered:
        parsed['model'] = 'topk'
        parsed['sparsity_mode'] = 'topk'

    seed_match = re.search(r'seed[_-]?(\d+)', lowered)
    if seed_match:
        parsed['seed'] = int(seed_match.group(1))

    # Warm-up is Dual-only; a baseline name never carries it, which keeps the
    # column empty for baselines exactly as in the config-driven path.
    warmup_match = re.search(r'warmup[_-]?(?:steps[_-]?)?(\d+)', lowered)
    if warmup_match:
        parsed['warmup_steps'] = int(warmup_match.group(1))

    return parsed


def _metadata_from_config(cfg: Dict[str, Any], run_name: str) -> Dict[str, Any]:
    """Map one ``vars(TrainingConfig)`` dump onto the metadata columns.

    The three trainers use three different dataclasses, so the field names have
    to be normalized here:
    * ``train_dual``  has ``sparsity_mode``/``span_coef``/``warmup_steps``
      and calls the sparsity level ``top_k``.
    * ``train_topk``  has ``activation``/``tied_weights`` and ``topk_k``.
    * ``train_other`` has ``model_type`` and ``topk_k``.
    """
    meta: Dict[str, Any] = {}

    meta['exp_name'] = cfg.get('exp_name') or run_name

    # Model: only train_dual writes sparsity_mode, only train_other writes
    # model_type; a config with neither came from train_topk.
    if 'sparsity_mode' in cfg:
        meta['model'] = 'dual'
        meta['sparsity_mode'] = cfg.get('sparsity_mode', '')
    elif cfg.get('model_type'):
        model_type = str(cfg['model_type'])
        meta['model'] = model_type
        meta['sparsity_mode'] = _BASELINE_SPARSITY_MODE.get(model_type, model_type)
    elif 'activation' in cfg:
        meta['model'] = 'topk'
        # For the TopK trainer the activation IS the sparsity mechanism
        # (--activation relu turns the same module into a plain ReLU SAE).
        meta['sparsity_mode'] = 'topk' if cfg.get('activation') == 'topk' else str(cfg.get('activation', ''))

    train_path = cfg.get('train_path')
    val_path = cfg.get('val_path')
    meta['dataset'] = _infer_dataset(train_path, val_path, meta.get('exp_name'), run_name)
    meta['extractor'] = _infer_extractor(train_path, val_path, meta.get('exp_name'), run_name)

    for key in ('seed', 'n_latents', 'lr'):
        if cfg.get(key) is not None:
            meta[key] = cfg[key]
    # Paper name of span_coef (older dumps may already carry decorrelation_coef).
    gamma = cfg.get('span_coef', cfg.get('decorrelation_coef'))
    if gamma is not None:
        meta['decorrelation_coef'] = gamma

    # Sparsity level: train_dual names it top_k, the other two topk_k.
    if cfg.get('topk_k') is not None:
        meta['topk_k'] = cfg['topk_k']
    elif cfg.get('top_k') is not None:
        meta['topk_k'] = cfg['top_k']

    # Warm-up only exists in the Dual trainer; see the module docstring for why
    # the column stays empty (not 0) for the baselines.
    if cfg.get('warmup_steps') is not None:
        meta['warmup_steps'] = cfg['warmup_steps']

    return meta


def _collect_run(run_dir: Path, root: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build one output row for ``run_dir``; returns ``(row, skip_reason)``."""
    metrics, error = _load_json(run_dir / BEST_METRICS_FILENAME)
    if metrics is None:
        return None, f'unreadable {BEST_METRICS_FILENAME} ({error})'
    if not metrics:
        return None, f'empty {BEST_METRICS_FILENAME}'

    run_name = run_dir.name
    config_path = run_dir / TRAINING_CONFIG_FILENAME
    meta_source = 'none'
    meta: Dict[str, Any] = {}

    if config_path.exists():
        cfg, cfg_error = _load_json(config_path)
        if cfg is None:
            # A corrupt config must not lose the metrics: fall back to the name.
            print(f"Warning: ignoring unreadable {config_path} ({cfg_error}); "
                  f"falling back to the run name for metadata")
            meta_source = 'run_name'
            meta = _parse_run_name(run_name)
        else:
            meta_source = 'training_config.json'
            meta = _metadata_from_config(cfg, run_name)
    else:
        meta_source = 'run_name'
        meta = _parse_run_name(run_name)

    row: Dict[str, Any] = {column: '' for column in METADATA_COLUMNS}
    row.update({k: v for k, v in meta.items() if k in row})
    if not row['exp_name']:
        row['exp_name'] = run_name
    try:
        row['run_dir'] = run_dir.relative_to(root).as_posix()
    except ValueError:
        # --root given as a relative path while run_dir resolved elsewhere.
        row['run_dir'] = run_dir.as_posix()
    row['meta_source'] = meta_source

    # Metrics never overwrite an identification column; on a name clash the
    # metric is kept under a prefixed column so nothing is silently lost.
    for key, value in metrics.items():
        column = key if key not in METADATA_COLUMNS else f'metric_{key}'
        row[column] = value

    return row, None


def _find_run_dirs(root: Path) -> List[Path]:
    """Return every directory under ``root`` holding a best_metrics.json."""
    run_dirs: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip caches / hidden trees; also keeps the walk deterministic.
        dirnames[:] = sorted(d for d in dirnames if not d.startswith('.') and d != '__pycache__')
        if BEST_METRICS_FILENAME in filenames:
            run_dirs.append(Path(dirpath))
    return run_dirs


def _sort_key(row: Dict[str, Any]) -> Tuple:
    """Deterministic ordering: dataset/encoder/model/activation, then warm-up, seed."""
    def as_number(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            # Unparsable/empty values sort last but keep a stable position via
            # the string tail of the key.
            return math.inf

    return (
        str(row.get('dataset', '')),
        str(row.get('extractor', '')),
        str(row.get('model', '')),
        str(row.get('sparsity_mode', '')),
        as_number(row.get('warmup_steps')),
        as_number(row.get('seed')),
        str(row.get('exp_name', '')),
        str(row.get('run_dir', '')),
    )


def _column_order(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Metadata columns first, then metrics in a stable, readable order."""
    metric_keys = set()
    for row in rows:
        metric_keys.update(k for k in row if k not in METADATA_COLUMNS)

    columns = list(METADATA_COLUMNS)
    for key in PREFERRED_METRIC_ORDER:
        if key in metric_keys:
            columns.append(key)
            metric_keys.discard(key)
    columns.extend(sorted(metric_keys))
    return columns


def _write_table(rows: Sequence[Dict[str, Any]], output: Path, fmt: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == 'json':
        with output.open('w', encoding='utf-8') as f:
            json.dump(list(rows), f, indent=2)
            f.write('\n')
        return

    columns = _column_order(rows)
    with output.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval='', extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Collect best_metrics.json files from a checkpoint tree into one table',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--root', type=str, default='./checkpoints',
                        help='Directory tree to scan recursively for run directories')
    parser.add_argument('--output', type=str, default='results/results.csv',
                        help='Output table path (parent directories are created)')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'json'],
                        help='Output format; JSON emits one object per run')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"--root is not a directory: {root}")

    rows: List[Dict[str, Any]] = []
    skipped: List[Tuple[Path, str]] = []
    for run_dir in _find_run_dirs(root):
        row, reason = _collect_run(run_dir, root)
        if row is None:
            print(f"Warning: skipping {run_dir}: {reason}")
            skipped.append((run_dir, reason or 'unknown reason'))
            continue
        rows.append(row)

    rows.sort(key=_sort_key)

    output = Path(args.output)
    _write_table(rows, output, args.format)

    # One-line summary: how many runs made it into the table, how many did not
    # and why (reasons are de-duplicated so the line stays one line).
    if skipped:
        reason_counts: Dict[str, int] = {}
        for _, reason in skipped:
            key = reason.split(' (')[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1
        reason_text = ', '.join(f"{count} x {reason}" for reason, count in sorted(reason_counts.items()))
    else:
        reason_text = 'none'
    print(f"collect_results: {len(rows)} run(s) collected from {root}, "
          f"{len(skipped)} skipped ({reason_text}) -> {output}")


if __name__ == '__main__':
    main()
