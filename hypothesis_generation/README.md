# SAE for hypothesis generation (HypotheSAEs for vision, with DualSAE)

A vision adaptation of HypotheSAEs (Movva et al., arXiv 2502.04382). For every class:

1. select the SAE latents most predictive of the class (one-vs-rest);
2. have a VLM (InternVL2.5-8B) describe each selected latent as a natural-language hypothesis, from the
   latent's top-activating and non-activating training images;
3. score the hypotheses on held-out images. The VLM annotates whether each image satisfies each
   hypothesis, and HypotheSAEs' `score_hypotheses` fits a logistic model on these binary annotations.
   It reports AUROC and #Significant, the number of hypotheses significant after Bonferroni correction
   at family-wise α = 0.1.

Each DualSAE is compared only with its activation-matched baseline. There are 3 encoders × {TopK,
BatchTopK, JumpReLU} × {baseline, Dual} × seeds {42, 43, 44}, on ImageNet-1K and OpenImagesV7.

## Paper items → commands

| Paper item | Command | Output |
|---|---|---|
| Supplementary, "SAE for hypothesis generation": results table | `scripts/01_stage1.sh <ds>`; `scripts/02_cells.sh <ds> <rule>` for each dataset and each rule in {lasso, correlation, separation_score}; then `scripts/03_results.sh` | `$HG_OUT/tables/hypogen_long.csv`, `hypogen_<dataset>_<rule>.md` (one table per selection rule) |
| Supplementary figure "Class-level hypotheses for the prayer rug and bull mastiff classes" and main-text figure "Hypothesis generation for bull mastiff" | `scripts/04_figures.sh` | `$HG_OUT/figures/encoder_vit.pdf`. The paper figure shows its TopK and JumpReLU sections; the main-text figure shows the first JumpReLU and Dual-JumpReLU rows. |
| Main text, Downstream applications: hypothesis-generation paragraph | the tables above | |

`scripts/run_all.sh` runs everything sequentially. Every stage is resumable: outputs that already exist
are skipped, and the VLM caches are flushed incrementally.

## Install

```bash
pip install -r vision/requirements.txt -r topic_modeling/requirements.txt -r hypothesis_generation/requirements.txt
bash hypothesis_generation/prepare_assets.sh   # needs internet
```

`prepare_assets.sh` installs HypotheSAEs (Apache-2.0) at commit
`4573abd87482ad3f02f850e0ca38897cac3e699b` with `--no-deps`; only `hypothesaes.evaluation.score_hypotheses`
is used, unmodified. It also downloads:

- `OpenGVLab/InternVL2_5-8B`;
- the three vision encoders;
- the class-name files.

All pipeline code then runs offline. vLLM runs with the engine in-process
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`, set before vLLM is imported).

## Inputs

- **SAE checkpoints.** These are the SAEs of `topic_modeling/` (`configs/runs.json`), found through `$TM_OUT/runs/`:
  - ImageNet-1K Dual: run `imagenet_paired_span001` (γ = 0.01, warm-up 17500);
  - ImageNet-1K baselines: run `imagenet_paired`;
  - OpenImagesV7 baselines and Dual: run `openimages_paired` (γ = 0.01, warm-up 17500).

  Train them with `topic_modeling/scripts/03_train_topics.sh <run> --stages train`.
  `HG_DUAL_ROOT` / `HG_BASE_ROOT` override the roots.
- **ImageNet-1K images** (ImageFolder): `IMAGENET_TRAIN_DIR`, `IMAGENET_VAL_DIR`. Embeddings are
  re-extracted from these images with the SAEs' encoders and L2-normalised, so every activation row
  maps to an image file.
- **OpenImagesV7.**
  - Feature/label LMDBs: `FEATURES_OPENIMAGES/{test,valid}/<encoder>.lmdb` and `openimages_labels.lmdb`.
  - Images: `OPENIMAGES_TEST_DIR`, `OPENIMAGES_VALID_DIR`.
  - Localized-narratives JSONL files of both splits: `OPENIMAGES_NARRATIVES_DIR`.

  `hypogen/build_oi_data.py` maps LMDB rows to image ids. `hypogen/oi_verify_order.py --split valid`
  checks that mapping by re-embedding a sample of images; it should print PASS.
  - The **train role** (selection and the VLM's proposal images) is served by the OpenImagesV7 **test** split.
  - The **val role** (faithful scoring) is served by the **valid** split.

## Pipeline (hypogen/)

| Stage | Module | What |
|---|---|---|
| 1a | `extract_embeddings` / `build_oi_data` | Embeddings, manifests and labels. ImageNet train: 300 images per class (seed 12345); val: all. |
| 1b | `dump_activations` | Dense fp16 SAE activations; checkpoints loaded via `topic_modeling/src/old_sae_adapter.py`. |
| 2 | `select_features` | Per class, one-vs-rest selection. The VLM interprets the union of the latents selected with K = 16. In `selmethod` mode and on OpenImagesV7, the union is restricted to the faithful classes. |
| 3 | `interpret_latents` | VLM proposal from 8 top-activating and 8 zero-activating training images per latent (temperature 0.7, 64 tokens, engine seed = SAE seed). |
| 4 | `evaluate` | **proxy**: held-out logistic AUROC on the selected latents. **faithful**: VLM annotation (yes/no, temperature 0) of each hypothesis on held-out images, then `score_hypotheses(classification=True, corrected_pval_threshold=0.1)`. |
| 5 | `build_results` | Tables, one per selection rule. |

Selection rules:

- **lasso**: one L1-logistic fit per class, C = 1, on the positives plus 2000 sampled negatives; latents
  ranked by |coefficient|.
- **correlation**: Pearson correlation of each latent with the class indicator.
- **separation_score**: as in HypotheSAEs, over the top 100 activating rows. Computed in a vectorised form.

How the rules are run:

- `HG_MODE=main` runs the selection for all three rules (K ∈ {8, 16}), and interpretation plus faithful
  evaluation for lasso.
- `HG_MODE=selmethod HG_SEL_METHOD=<rule>` runs the faithful pipeline with correlation or separation_score
  (K = 16). Outputs go under `selmethod_<rule>/`. Hypotheses already produced in the lasso run are reused
  for the same latents (`02_cells.sh` copies them before interpreting).

Faithful evaluation uses 100 classes sampled with seed 7. Each class gets its validation positives (at
most 200, sampled with a seeded RNG) and 200 shared negatives. The (hypothesis, image) annotations are
cached and reused.

`build_results` reports every encoder × activation × dataset under each of the three rules separately:

- baseline and DualSAE mean ± sd over seeds of the per-cell macro average over classes;
- Δ = DualSAE − baseline.

Nothing is selected per cell.

Single cell, one rule:
`HG_DATASET=imagenet HG_MODE=selmethod HG_SEL_METHOD=correlation python -m hypogen.evaluate --stage faithful --encoder vit --activation jump_relu --variant dual --seed 42`.
`scripts/02_cells.sh <ds> <rule> <cell index>` runs one cell by index (`python -m hypogen.jobs --kind cells --index i`).
`scripts/slurm_template.sh` is a generic array template.

## Class-level hypothesis figures: selection rule (`hypogen/encoder_figs.py`)

There is one PDF per encoder, with the three activation pairs stacked (ImageNet-1K, seed 42).

1. **Rule per cell.** By default (`--selectors auto`) each encoder/activation cell uses the selection rule
   with the largest seed-averaged Dual − baseline #Significant on the faithful evaluation (ties → larger
   Δ AUROC). Explicit `enc/act=rule` entries override this.
2. **Class per section.** The faithful class with the largest per-class Dual − baseline #Significant that
   has ≥ 2 qualifying latents in both models.
3. **Manual overrides.** The reported figures used two manual class choices: ViT/JumpReLU → 243
   (bull mastiff) and CLIP/JumpReLU → 4. In those two cells the automatically chosen class showed visually
   off-topic tiles. They are passed as `--override` in `04_figures.sh`; remove the override to get the
   automatic choice.
4. **Rows.** The top-2 positive-coefficient selected latents per model (|coef| > 0.5 for lasso, > 0 for
   the other rules). Each row shows the latent's hypothesis and its five top-activating training images,
   i.e. the images the VLM saw.

`hypogen/visualize_qualitative.py` renders the same rows for a single cell, in concept or contrast layout.

## Implementation notes

- **Training of the SAEs.** The SAEs are those of the topic-modeling study, trained with the
  hyper-parameters and the checkpoint rule documented in `topic_modeling/README.md`. This includes:
  - the warm-up behaviour;
  - JumpReLU with `jump_threshold_init 0.001`, which gives dense codes;
  - for OpenImagesV7, a Dual warm-up (17500) longer than the 12,350 training steps, so the OpenImagesV7
    Dual checkpoints never trained their Residual Branch;
  - checkpoints that may come from early epochs. Check the selected epochs (stored in each checkpoint's
    `epoch` field).
- **VLM prompt.** The proposer prompt and its examples mention "ImageNet" for both datasets.
- **Image sizing.** Each image is downsized to at most 448 px and shown as a single tile
  (`max_dynamic_patch=1`); 16 images fit in `max_model_len` 12288.
- **AUROC.** AUROC and #Significant come from HypotheSAEs `score_hypotheses`, i.e. a fit and evaluation on
  the same held-out annotation set. Identical hypothesis strings within a class are counted once.
- **Fidelity.** Per-latent fidelity (`--with-fidelity`) is optional and was not used.

## Tests

```bash
python hypothesis_generation/tests/test_smoke.py              # offline end-to-end, VLM stubbed, synthetic data
python hypothesis_generation/tests/check_rescore.py --reference-root <existing HG_OUT> \
    --rule correlation --encoder vit --activation jump_relu --variant dual --seed 42
```

`check_rescore.py` re-scores one cell from an existing output tree's cached VLM annotations with the
current code and checks that `faithful.csv` is identical row by row. `hypogen/smoke.py` is a GPU sanity
check on real data. It needs the downloaded models, the ImageNet images, the validation feature LMDB and
trained checkpoints.
