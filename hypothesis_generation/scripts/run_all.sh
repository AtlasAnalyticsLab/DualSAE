#!/usr/bin/env bash
# Whole hypothesis-generation study, sequentially (resumable). Requires the SAE checkpoints from
# topic_modeling (runs imagenet_paired, imagenet_paired_span001, openimages_paired) and prepare_assets.sh.
set -euo pipefail
D="$(dirname "${BASH_SOURCE[0]}")"
for DS in imagenet openimages; do
    bash "$D/01_stage1.sh" "$DS"
    for RULE in lasso correlation separation_score; do bash "$D/02_cells.sh" "$DS" "$RULE"; done
done
bash "$D/03_results.sh"
bash "$D/04_figures.sh"
