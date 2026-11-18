#!/bin/bash
# ============================================================================
# prepare_assets.sh -- the ONLY network step of the language component.
#
# Downloads, at the pinned commits recorded in langdual/assets.py:
#   * EleutherAI/pythia-410m-deduped   (weights + tokenizer)
#   * Skylion007/openwebtext           plain_text/train-0000{0,1}-of-00080.parquet
#   * codeparrot/github-code-clean     data/train-0000{0,1}-of-00880.parquet
#   * BAAI/bge-large-en-v1.5, fancyzhx/ag_news, google-research-datasets/go_emotions
#                                      (text-embedding stage, required by llm_steering/)
# into $LANGDUAL_ROOT/raw/{models,datasets}/<org>__<name>, then re-loads the model and tokenizer
# with the network disabled.  Everything after this script runs fully offline.
#
#     LANGDUAL_ROOT=/path/to/outputs bash language/prepare_assets.sh
#
# Re-running is cheap: a snapshot whose sentinel and files are complete is skipped.
# ============================================================================
set -euo pipefail

LANG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$LANG_DIR/scripts/env.sh"
# This script, and only this script, talks to the hub.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

echo "=== [1/3] roots under $LANGDUAL_ROOT"
mkdir -p "$LANGDUAL_ROOT/raw/models" "$LANGDUAL_ROOT/raw/datasets" "$LANGDUAL_ROOT/tokens" \
         "$LANGDUAL_ROOT/resid" "$LANGDUAL_ROOT/runs" "$LANGDUAL_ROOT/results" "$LANGDUAL_ROOT/logs"

echo "=== [2/3] snapshot_download (skips anything already complete)"
"$LD_PY" - <<'PYDL'
import json, os, sys, time
from huggingface_hub import snapshot_download
from langdual import assets

failures = []
for asset in assets.manifest():
    ok, reason = assets.snapshot_status(asset)
    if ok:
        print(f"  [skip] {asset.manifest_key:16s} {asset.repo}")
        continue
    print(f"  [get ] {asset.manifest_key:16s} {asset.repo}@{asset.revision}  ({reason})", flush=True)
    t0 = time.time()
    try:
        path = snapshot_download(
            repo_id=asset.repo, repo_type=asset.repo_type, revision=asset.revision,
            local_dir=asset.local_dir, allow_patterns=list(asset.allow_patterns),
            ignore_patterns=list(asset.ignore_patterns) or None)
    except Exception as exc:                      # noqa: BLE001
        print(f"         FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        failures.append(asset.manifest_key)
        continue
    # Sentinel written LAST and atomically: it is both the skip-guard and the provenance record.
    sentinel = {
        "repo": asset.repo, "repo_type": asset.repo_type, "revision": asset.revision,
        "commit_sha": asset.revision, "allow_patterns": list(asset.allow_patterns),
        "ignore_patterns": list(asset.ignore_patterns),
        "required_globs": list(asset.required_globs),
        "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    dst = os.path.join(asset.local_dir, assets.SENTINEL_NAME)
    with open(dst + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(sentinel, fh, indent=2)
    os.replace(dst + ".tmp", dst)
    ok, reason = assets.snapshot_status(asset)
    print(f"         {'ok' if ok else 'INCOMPLETE'} in {time.time() - t0:.0f}s ({reason})")
    if not ok:
        failures.append(asset.manifest_key)
if failures:
    sys.exit(f"download failures: {failures}")
PYDL

echo "=== [3/3] offline re-load check (network disabled)"
(
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  "$LD_PY" - <<'PYCHK'
from transformers import AutoConfig, AutoTokenizer
from langdual import assets, config
path = assets.local_snapshot(config.LM_REPO)
cfg = AutoConfig.from_pretrained(path, local_files_only=True)
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
assert cfg.num_hidden_layers == 24 and cfg.hidden_size == config.LM_D_MODEL, cfg
assert max(config.LM_LAYERS) < cfg.num_hidden_layers
for name in config.LM_CORPUS_ORDER:
    for f in assets.lm_corpus_files(name):
        assert __import__("os").path.getsize(f) > 0, f
bge = AutoConfig.from_pretrained(assets.local_snapshot(config.TEXT_ENCODERS["bge"].repo),
                                 local_files_only=True)
assert bge.hidden_size == config.TEXT_ENCODERS["bge"].dim, bge
for name in config.CORPUS_ORDER:
    for f in assets.TEXT_CORPUS_ASSETS[name].local_files():
        assert __import__("os").path.getsize(f) > 0, f
print(f"  ok: {config.LM_REPO} ({cfg.num_hidden_layers} blocks, d={cfg.hidden_size}), "
      f"tokenizer vocab {len(tok)}, corpus shards present; bge-large and the labelled "
      f"corpora present")
PYCHK
)
echo "=== assets ready; next: bash language/run_pipeline.sh --stage pack"
