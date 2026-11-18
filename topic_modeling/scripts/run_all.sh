#!/usr/bin/env bash
# Whole topic-modeling study, sequentially (resumable; re-run to continue after an interruption).
# Assumes prepare_assets.sh has been run and the SAE training feature LMDBs exist.
set -euo pipefail
D="$(dirname "${BASH_SOURCE[0]}")"
for DS in imagenet openimages; do bash "$D/01_captions.sh" "$DS"; bash "$D/02_caption_side.sh" "$DS"; done
for RUN in imagenet_paired imagenet_paired_span001 imagenet_warmup_sweep \
           openimages_paired openimages_warmup_sweep openimages_jr_warmup500; do
    bash "$D/03_train_topics.sh" "$RUN"
    bash "$D/04_evaluate.sh" "$RUN"
done
bash "$D/05_tables.sh"
bash "$D/06_figures.sh"
