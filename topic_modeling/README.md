# SAE as a topic model (SAE-TM with DualSAE)

This component runs the SAE-TM topic-modeling pipeline (Girrbach et al.) on the sparse codes of
standard SAEs and of DualSAE's Sparse Branch. It covers ImageNet-1K and OpenImagesV7, three vision
encoders (CLIP ViT-L/14 DataComp, SigLIP2 so400m-14-384, ViT-L/16-224), and three activations
(TopK, BatchTopK, JumpReLU). Each DualSAE is compared only with its activation-matched baseline.

## Paper items → commands

| Paper item | Command | Output |
|---|---|---|
| Supplementary, "SAE as a topic model": topic modeling results table (ImageNet-1K / OpenImagesV7, K ∈ {50, 100, 200, 300, 500}) | `scripts/01_captions.sh`, `02_caption_side.sh`, `03_train_topics.sh <run>`, `04_evaluate.sh <run>` for every run, then `scripts/05_tables.sh` | `$TM_OUT/tables/topic_table_long.csv`, `topic_table_<dataset>_<metric>.md` |
| Supplementary figure "Image–topic assignments between DualSAE and traditional SAEs" | `scripts/06_figures.sh` | `$TM_OUT/figures/img2topic_all_K100.{pdf,png}` |
| Main text, Downstream applications: "Image–topic assignment comparison (TopK on SigLIP2)" figure | `scripts/06_figures.sh` (the SigLIP/TopK row of the same figure; the paper shows a subset of its images) | as above |
| Main text, Downstream applications: topic-modeling paragraph | the table above | |

`scripts/run_all.sh` runs everything in order. Each step is resumable: re-run the same command after an
interruption and it continues.

## Install

```bash
pip install -r vision/requirements.txt -r topic_modeling/requirements.txt   # Python 3.11
bash topic_modeling/prepare_assets.sh     # downloads (needs internet) + SAE-TM checkout, see below
```

`prepare_assets.sh` is the only step that needs internet access. It downloads:

- the three vision encoders;
- `microsoft/phi-4`, the LLM judge for coherence and intruder, the topic labels and the image–topic judge;
- `OpenGVLab/InternVL3_5-8B`, the captioning model;
- `word2vec-google-news-300`;
- NLTK data;
- `imagenet_class_index.json`.

It then calls `setup_saetm.sh`. Everything else runs offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
vLLM engine in-process via `VLLM_ENABLE_V1_MULTIPROCESSING=0`).

### SAE-TM is not vendored

The SAE-TM repository has no licence, so none of its code is redistributed here.
`setup_saetm.sh` does the following:

- clones `https://github.com/ExplainableML/SAE-TM` at commit `5bbd60f62723e9a59c135fe7b2553db246fdf55e`
  into `topic_modeling/third_party/SAE-TM` (or `$SAE_TM_DIR`);
- applies `patches/saetm_5bbd60f.patch`. The header of `setup_saetm.sh` lists what the patch changes.

The checkout contains code under its own licences (`libraries/dictionary_learning`: MIT; `baselines/`:
adapted from TopMost, Apache-2.0).

## Data

Set the following environment variables. Defaults are relative to the working directory; see `scripts/env.sh`.

| Variable | Content |
|---|---|
| `FEATURES_IMAGENET` | SAE training features: `{train,validation}/<encoder>.lmdb` (ImageNet-1K). These are written by the vision feature extraction; see `vision/README.md`. |
| `FEATURES_OPENIMAGES` | `{train,valid}/<encoder>.lmdb` (OpenImagesV7). |
| `IMAGENET_VAL_DIR` | ImageNet-1K validation images in ImageFolder layout. These images are captioned and form the ImageNet topic corpus. |
| `OPENIMAGES_TEST_DIR` | OpenImagesV7 test images. These images are captioned and form the OpenImagesV7 topic corpus. |
| `TM_OUT` | All outputs (default `./outputs/topic_modeling`). |

LMDB names: `clip_image_features_CLIP-ViT-L-14-DataComp.lmdb`, `siglip2_image_features_siglip2-so400m-p14-384.lmdb`,
`vit_features_vit-large-p16-224.lmdb`.

## Configurations (`configs/runs.json`)

Every SAE grid that was trained for this study is listed below. Each run has 3 encoders × its trainers × seeds {42, 43, 44}.
Paper terms map to code names as follows:

- Sparse Branch = `mono`;
- Residual Branch = `poly`;
- decorrelation loss L_dcor = `span`, with coefficient γ = `span_coef`;
- the warm-up trains the Sparse Branch alone.

| Run | Dataset | SAEs | γ (`span_coef`) | warm-up steps |
|---|---|---|---|---|
| `imagenet_paired` | ImageNet-1K | TopK, BatchTopK, JumpReLU baselines + their Duals | 1.0 | 17500 |
| `imagenet_paired_span001` | ImageNet-1K | Dual-TopK/BatchTopK/JumpReLU | 0.01 | 17500 |
| `imagenet_warmup_sweep` | ImageNet-1K | Dual-TopK/BatchTopK/JumpReLU | 0.01 | 0, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 20000 |
| `openimages_paired` | OpenImagesV7 | baselines + Duals | 0.01 | 17500 (≥ total steps, see notes) |
| `openimages_warmup_sweep` | OpenImagesV7 | Dual-TopK/BatchTopK/JumpReLU | 0.01 | same 10 values (12500, 15000, 20000 ≥ total steps) |
| `openimages_jr_warmup500` | OpenImagesV7 | Dual-JumpReLU | 0.01 | 500 |

The table builder reports **every** DualSAE configuration (run × warm-up) as its own row against the
baseline of `imagenet_paired` / `openimages_paired`. The configurations are compared with each other
side by side; the builder never picks one per cell. Columns:

- baseline and DualSAE mean/sd over seeds (seeds are the only averaged dimension);
- Δ = mean(DualSAE) − mean(baseline);
- number of seeds;
- best-checkpoint epochs;
- the number of seeds whose selected checkpoint precedes the end of the warm-up (`pre_residual_seeds`).

There is one table per dataset and metric, with all three SAE-TM default metrics and every K reported separately.

## Running

```bash
source topic_modeling/scripts/env.sh                      # or export the variables yourself
bash topic_modeling/scripts/01_captions.sh imagenet       # GPU; optional START END for sharding
bash topic_modeling/scripts/02_caption_side.sh imagenet   # GPU (embeddings) + CPU (BoW)
bash topic_modeling/scripts/03_train_topics.sh imagenet_paired            # train + interpret + topics
bash topic_modeling/scripts/04_evaluate.sh imagenet_paired                # 3 metrics (vLLM)
bash topic_modeling/scripts/05_tables.sh
bash topic_modeling/scripts/06_figures.sh
```

Single cell:
`python topic_modeling/src/tm_driver.py cell --run imagenet_warmup_sweep --encoder clip --trainer DualTopKTrainer --seed 42 --warmup 2500`.

Other commands:

- `tm_driver.py list --run <run>` lists cells.
- `tm_driver.py train-argv ...` prints the exact training command.
- `--dry-run` prints commands without running them.

`scripts/slurm_template.sh` is a generic array-job template with placeholders.

Output layout:
`$TM_OUT/runs/<run>/<enc>/{trained_models,interpreted,topics,evaluation}/<Trainer>[/warmup<W>]/seed<S>[/K<K>]`.
The caption side is written to `$TM_OUT/{imagenet_val,openimages_test}/`.

## Pipeline per cell

1. **Train**: `vision/main.py` with
   - 8192 latents, k = 64;
   - batch 2048, 50 epochs, Adam lr 1e-4, eps 6.25e-10;
   - MonoLoss coefficient 0;
   - `best_val_monoscore_model.pt` is the epoch with the best validation MonoScore.

   Per trainer:
   - TopK: tied weights, `--force_normalize off`, dead-latent check every 100 steps.
   - BatchTopK: no extra flags.
   - JumpReLU: L1 1e-4, bandwidth 1e-3, `--jump_threshold_init 0.001`.
   - Dual: Residual Branch hidden width flag 512; warm-up check every 500 steps; dead check every 100 steps; γ and warm-up per run.

   These flags give the same training configuration as the standalone trainer invocations the runs used
   (`tests/test_train_config.py`).
2. **Interpret** (SAE-TM `interpret_sae.py --old-impl`): sparse codes of the caption-aligned validation
   embeddings; word-emission EM over the caption bag-of-words (50 epochs, batch 1024, seed 0).
3. **Topics** (SAE-TM `sae_to_topic.py`): K-means (random_state 42) over word2vec topic embeddings of
   the live features, for K ∈ {50, 100, 200, 300, 500}.
4. **Evaluate** (SAE-TM `evaluation.py`, defaults `--k 5 --n 4 --r 3`, judge `microsoft/phi-4`, max model length 4096):
   - **diversity**: WMD over the top-20 words;
   - **coherence**: LLM rating 0–100; temperature 0.7, top-p 0.9;
   - **intruder**: LLM picks the intruder among 4 topic words + 1 word from another topic; temperature 0.

## Image–topic figures: selection rule

`06_figures.sh` reproduces the procedure used for the figures (ImageNet-1K, run `imagenet_paired`, seed 42, K = 100):

1. **Labels.** `annotate_topics.py` asks phi-4 for a 1–3 word label for each topic of the Dual and baseline
   cells (temperature 0.2, top-p 0.9; abstract/concrete classification at temperature 0). Labels are only
   used as captions.
2. **Candidate images.** Per encoder/activation, each image's dominant topic is the topic with the largest
   activation relative to that topic's mean over all images (lift). The 1500 candidate images are taken by
   round-robin over the Dual model's dominant topics, highest lift first.
3. **Judging.** `judge_img2topic.py` shows phi-4 the image caption and the top-10 words of the Dual and the
   baseline dominant topic. It asks which matches better (A / B / TIE, temperature 0, 4 tokens), once in each
   order. The verdict is `dual` only if the Dual topic wins in both orders.
4. **Rendering.** `visualize_topics.py --figures img2topic-all` shows, per row, the first 10 `dual`-verdict
   images in the candidate order, captioned with the Dual (red) and baseline (blue) labels. The judged
   counts per cell (dual/base/tie) are written to `img2topic_judgments.csv` in each Dual topics directory.

## Implementation notes

- **Residual Branch.** With `poly_layers=1` the Residual Branch is `Linear(d, d) + GELU`; `--poly_hidden`
  is not used at one layer. During warm-up it is frozen, but its output at initialisation is still added
  to the reconstruction.
- **Warm-ups longer than training.** OpenImagesV7 has 507,444 training rows, so 50 epochs at batch 2048
  are 12,350 steps. Warm-ups of 12500, 15000, 17500 and 20000 steps therefore never leave the warm-up: the
  Residual Branch is never trained and the decorrelation loss is never applied. These cells are trained
  as they were (with `--allow_degenerate_warmup`) and flagged `warmup_ge_budget` in the tables.
- **Checkpoint selection.** The checkpoint is picked by validation MonoScore over all epochs, including
  warm-up epochs. The tables report the selected epochs. For JumpReLU/Dual-JumpReLU and for some other
  cells, the selected checkpoint is from the first epochs, and for Duals possibly from before the end of
  the warm-up.
- **JumpReLU threshold.** JumpReLU runs here use `jump_threshold_init 0.001`. With this initialisation
  the learned thresholds stay near 0.001, so most latents are active (the code is dense rather than
  k-sparse).
- **TopK baseline.** It uses tied encoder/decoder weights and no per-sample input standardisation.
- **BatchTopK codes.** BatchTopK (baseline and Dual) codes are computed with a batch-level top-k over the
  batch being encoded, so they depend on the batch composition: EM batches of 1024 and theta batches of
  8192, as in SAE-TM.
- **LLM sampling.** Word sampling for the coherence and intruder prompts uses Python's unseeded `random`
  (SAE-TM behaviour), so re-runs vary slightly.
- **Captions.** Captions are generated with `do_sample=True` and the model's default generation settings;
  the `--temperature` flag of `create_caption.py` is not forwarded. Captions of the OpenImagesV7 test split
  were merged from shards and repaired with `repair_captions.py`.
- **Ordering and sharing.** Caption-aligned embeddings follow the caption-file order. The documents and the
  bag-of-words (vocabulary ≤ 20,000, min document frequency 5) are shared by all encoders.
- **ImageNet-1K bag-of-words settings.** The exact caption-sharding and bag-of-words commands used for the
  ImageNet-1K runs were not recorded. `02_caption_side.sh` applies the OpenImagesV7 settings (vocabulary
  ≤ 20,000, min document frequency 5) to ImageNet-1K as well; a vocabulary built with different settings
  changes the topic-word metrics.
- **Topic labels in figures.** Labels are LLM summaries of each topic's top words.

## Tests

```bash
python topic_modeling/tests/test_train_config.py                      # driver flags == reported trainer configs
SAE_TM_DIR=<patched checkout> python topic_modeling/tests/test_smoke.py   # offline end-to-end, LLM stubbed
```
