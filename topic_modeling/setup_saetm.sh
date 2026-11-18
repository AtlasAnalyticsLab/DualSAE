#!/usr/bin/env bash
# Fetch SAE-TM (Girrbach et al.) at the pinned commit and apply this repository's patch.
#
# SAE-TM is NOT redistributed here (its repository carries no licence). This script clones the
# upstream repository, checks out the exact commit the experiments used, and applies
# patches/saetm_5bbd60f.patch, which
#   * interpret_sae.py : adds --old-impl (load checkpoints written by this repo's vision/ trainers via
#                        topic_modeling/src/old_sae_adapter.py), JumpReLU/Dual sae-types, and replaces
#                        an import of a module that is missing upstream;
#   * sae_to_topic.py  : loads word2vec from a local file instead of downloading it, and computes the
#                        per-topic document coverage (cluster_ratio) in one vectorised step;
#   * evaluation.py    : loads word2vec from a local directory, imports vLLM lazily with the engine
#                        in-process, skips topic files that already have outputs (resumable), and adds
#                        --path-filter;
#   * baselines/utils/make_bow_cache_sharded.py : never downloads NLTK data (checks it is present);
#   * libraries/dictionary_learning/trainers/{__init__,dual_sae}.py : a dictionary_learning DualSAE class
#                        that interpret_sae.py imports at module level (not used by this pipeline).
# The metric definitions, prompts, sampling parameters and topic construction are unchanged.
#
# Needs internet (git clone). Usage:  bash topic_modeling/setup_saetm.sh [TARGET_DIR]
# TARGET_DIR defaults to $SAE_TM_DIR or topic_modeling/third_party/SAE-TM.
set -euo pipefail

TM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-${SAE_TM_DIR:-$TM_DIR/third_party/SAE-TM}}"
SAE_TM_URL="${SAE_TM_URL:-https://github.com/ExplainableML/SAE-TM.git}"
SAE_TM_COMMIT="5bbd60f62723e9a59c135fe7b2553db246fdf55e"
PATCH="$TM_DIR/patches/saetm_5bbd60f.patch"

if [ -f "$TARGET/.dualsae_patch_applied" ]; then
    echo "SAE-TM already set up at $TARGET"; exit 0
fi
if [ ! -d "$TARGET/.git" ]; then
    mkdir -p "$(dirname "$TARGET")"
    git clone "$SAE_TM_URL" "$TARGET"
fi
git -C "$TARGET" checkout --quiet "$SAE_TM_COMMIT"
git -C "$TARGET" apply --check "$PATCH"
git -C "$TARGET" apply "$PATCH"
echo "$SAE_TM_COMMIT" > "$TARGET/.dualsae_patch_applied"
echo "SAE-TM $SAE_TM_COMMIT + patch ready at $TARGET"
