#!/bin/bash
# ============================================================================
# make_figures.sh -- the top-activating-context pages behind the language qualitative figures.
#
# Each page shows, for six automatically selected Dual-TopK latents, five strongly and three
# moderately activating snippets from 98,304 held-out test tokens, next to the TopK-baseline
# latent whose activations correlate best with it over the same tokens.  All pages are built
# from TopK-family cells (Dual-TopK vs TopK, block 12, best-validation checkpoints):
#
#   page                                          Dual cell                                    figure
#   contexts_L12_owt_topk_seed42_tuned.html       abl_warmupB1250_B_L12_owt_topk_dual_seed42   main-text "Qualitative analysis on language
#                                                                                              representations", panel (a): Concept 3
#                                                                                              supplementary "person name" latents: Concept 5
#   contexts_L12_owt_topk_seed43_tuned.html       abl_warmupB1250_B_L12_owt_topk_dual_seed43   supplementary "professional title" latents: Concept 2
#   contexts_L12_ghcode_topk_seed43.html          B_L12_ghcode_topk_dual_seed43                supplementary "end-of-text" latents: Concept 1
#   contexts_L12_ghcode_topk_seed42.html          B_L12_ghcode_topk_dual_seed42                supplementary "class attribute" latents: Concept 1
#
# The baseline of every page is the main-grid TopK cell of the same corpus and seed
# (B_L12_<corpus>_topk_baseline_seed<S>).  Prerequisites: those cells trained, and the block-12
# test shards of both corpora extracted.  One short GPU pass per page.
#
#     bash language/scripts/make_figures.sh      # pages -> $LANGDUAL_ROOT/reports/qualitative/
# ============================================================================
set -euo pipefail
LANG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$LANG_DIR/scripts/env.sh"
cd "$LANG_DIR"

"$LD_PY" -m langdual.qualitative.top_contexts --layer 12 --corpus owt --activation topk --seed 42 \
    --dual-tag abl_warmupB1250_B_L12_owt_topk_dual_seed42 --suffix _tuned
"$LD_PY" -m langdual.qualitative.top_contexts --layer 12 --corpus owt --activation topk --seed 43 \
    --dual-tag abl_warmupB1250_B_L12_owt_topk_dual_seed43 --suffix _tuned
"$LD_PY" -m langdual.qualitative.top_contexts --layer 12 --corpus ghcode --activation topk --seed 43
"$LD_PY" -m langdual.qualitative.top_contexts --layer 12 --corpus ghcode --activation topk --seed 42
