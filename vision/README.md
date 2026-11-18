# DualSAE — vision core and main experiments

This directory holds the single implementation of every SAE and of DualSAE used in the repository
(the other components import it), plus everything for the vision experiments: feature extraction
(ImageNet-1K and OpenImagesV7; CLIP, SigLIP2, ViT), training, the five metrics (R², MonoScore,
binary purity, KNN, linear probe), the main SAE-vs-DualSAE comparison, the parameter-matched
baselines and the vision ablations (`scripts/ablations/`, documented in
[`scripts/ablations/README.md`](scripts/ablations/README.md)).

## Paper item → command

| Paper item | Commands |
|---|---|
| Sec. 5.1 "Performance between DualSAE and SAE": radar figure on ImageNet-1K; supplementary radar figure on OpenImagesV7; supplementary side-by-side table of all SAE variants and their DualSAE counterparts (TopK, BatchTopK, JumpReLU, MatryoshkaSAE, GatedSAE, CB-SAE) | train: `scripts/main/train_cells.py` (+ CB rows: [`cb_sae/`](../cb_sae/)); evaluate: `scripts/main/eval_runs.py`; collect: `analysis/main/collect_main.py`; table: `analysis/main/build_main_table.py`; figures: `analysis/main/plot_radar.py --dataset {imagenet,open_images}` — or all at once: `scripts/main/run_main.sh` |
| Sec. 5.2 "Capacity analysis": parameter-matched SAE table | train: `scripts/param_matched/train_param_matched.py`; table: `analysis/main/build_param_matched_table.py` (also run by `run_main.sh`) |
| Supplementary "Experimental details" (hyper-parameters, datasets, encoders, metrics) | `configs/main_cells.json` (the exact `main.py` arguments of every run of the comparison), `extract_imagenet.py`, `extract_open_images.py`, `loss.py` / `metrics.py` / `branch_metrics.py` |
| Selection of the GatedSAE sparsity coefficient and of the DualSAE decorrelation coefficient / warm-up per cell | `scripts/main/stage_cells.py`, `scripts/main/pick_gated_lambda.py`, `scripts/main/pick_dual_winners.py`, `configs/original_dual_sweep.json` (see "Selection" below) |
| Vision ablations (decorrelation coefficient, decorrelation loss, Residual Branch design, joint vs. sequential training, expansion ratio, warm-up, branch analyses, qualitative figures) | [`scripts/ablations/README.md`](scripts/ablations/README.md) |

## Terminology: paper vs. code

| Paper | Code |
|---|---|
| Sparse Branch, its reconstruction `x^M` | "mono" branch, `xM`, `W_enc` / `W_dec` / `b_enc` / `b_dec` |
| Residual Branch `r_hat = Q(W_P r)` | "poly" branch, `poly` module, `r_hat` |
| Decorrelation loss `L_dcor`, coefficient `gamma` | span-overlap penalty, `span_coef` (CLI alias `--decorrelation_coef`) |
| Stage 1 of two-stage training (length) | `--warmup_steps` |
| Sparsification rule `S(.)` of DualSAE | `--sparsity_mode {topk, batch_topk, jump_relu, matryoshka, gated, msae, vanilla}` |
| Sparsity coefficient `lambda` | `--l1_coef` (JumpReLU: L0 coefficient; GatedSAE: gate-path L1) |
| SigLIP2 encoder | `sigclip` (`--extractor sigclip`) |
| TK / BTK / JR / MAT / GATE / CB and D-* | method keys `topk`, `batch_topk`, `jumprelu`, `matryoshka`, `gated`, `cb_sae` and `dual_topk`, `dual_batch_topk`, `dual_jump_relu`, `dual_matryoshka`, `dual_gated`, `cb_dual_sae` |

## Install

```bash
pip install -r vision/requirements.txt
```

Python 3.11 and a CUDA GPU are assumed for the full experiments; the tests run on CPU.

## Assets and features

`prepare_assets.sh` performs every download (run it once on a machine with internet access):
the encoder weights (into `$HF_HOME`), ImageNet-1K from the HuggingFace Hub (gated: accept its
terms and log in first), and the OpenImagesV7 annotations, Localized Narratives and images
(`prepare_open_images.py` builds the local layout). Everything afterwards runs offline.

```bash
HF_HOME=./assets/hf DATA_ROOT=./data bash vision/prepare_assets.sh
```

Feature extraction writes one LMDB per (dataset, split, encoder) under `./features`:

```bash
cd vision
python extract_imagenet.py   --model clip --split train        # clip | siglip2 | vit ; train | validation
python extract_open_images.py --model clip --split train --dataset_dir ../data/open_images_v7   # train | valid | test
```

Layout: `features/imagenet/{train,validation}/<file>.lmdb` and
`features/open_images/{train,valid,test}/<file>.lmdb` (+ `openimages_labels.lmdb` per OpenImagesV7
split). `main.py --features_root` (env `FEATURES_ROOT`) resolves `--dataset/--extractor` to these
paths; `--train_path/--val_path/--test_path` override them.

## Training

A single run:

```bash
cd vision
python main.py --model batch_topk --dataset imagenet --extractor clip --seed 42 --lr 1e-4 \
    --exp_name my_btk --output_dir ./checkpoints
python main.py --model dual --sparsity_mode batch_topk --dataset imagenet --extractor clip --seed 42 \
    --decorrelation_coef 0.005 --warmup_steps 17500 --dead_check_interval 100 --lr 1e-4 \
    --exp_name my_dual_btk --output_dir ./checkpoints
```

`main.py --model` selects the trainer: `topk` → `train_topk.py`, `dual` → `train_dual.py`,
`batch_topk | vanilla | jumprelu | gated | matryoshka | msae` → `train_other.py`. Each run writes
`<output_dir>/<exp_name>/` with `training_config.json`, `best_val_monoscore_model.pt` (the checkpoint
with the best validation MonoScore, the one that is evaluated), `autoencoder_final.pt` and
`best_metrics.json` (the trainer's own evaluation of the best checkpoint: `BEST_R2`,
`BEST_MONOSCORE`, `BEST_PURITY`, `BEST_ACC_KNN`, `BEST_ACC_LP`, and the `BEST_TEST_*` twins when a
test split exists). TensorBoard (`--tb_dir`) and W&B (`--wandb_project`) are optional.

**The main grid** (3 encoders × 2 datasets × 10 methods × seeds 42/43/44 = 180 runs).
`configs/main_cells.json` stores the exact `main.py` arguments of every run; per-cell values that
differ from the defaults (learning rate, `--normalize`, `--l1_coef`, `--dead_check_interval`,
decorrelation coefficient, warm-up, GatedSAE lambda, ...) are passed explicitly there.

```bash
python scripts/main/train_cells.py --list                         # all cells and their status
python scripts/main/train_cells.py                                 # train everything (sequential, resumable)
python scripts/main/train_cells.py --dataset imagenet --encoder clip --method dual_topk --seed 42
python scripts/main/train_cells.py --group matryoshka_gated --print   # one command per cell (for a scheduler)
```

**Parameter-matched baselines** (TopK / BatchTopK / JumpReLU retrained with a wider dictionary;
everything else as the baseline cell):

```bash
python scripts/param_matched/train_param_matched.py --verify_params   # parameter counts
python scripts/param_matched/train_param_matched.py                   # train all 54 runs
```

### Selection

* **TopK / BatchTopK / JumpReLU pairs.** The decorrelation coefficient and warm-up of each Dual
  cell were chosen among the settings listed in `configs/original_dual_sweep.json` (every setting
  trained with all three seeds), by inspecting the validation metrics of those runs; the choice is
  not encoded in a script. `train_cells.py --sweep` trains that sweep; the chosen settings are the
  ones in `configs/main_cells.json`.
* **MatryoshkaSAE / GatedSAE pairs.** Scripted (`scripts/main/grid_config.py` defines the grids):

  ```bash
  S=scripts/main
  python $S/stage_cells.py --stage matryoshka --run
  python $S/stage_cells.py --stage gated-lambda --run          # seed-42 lambda grid of GatedSAE
  python $S/eval_runs.py --runs './checkpoints/main/t3x_*_gated_*'   # or l0_probe.py for L0 only
  python $S/l0_trajectory.py                                    # per-epoch L0 from TensorBoard events
  python $S/pick_gated_lambda.py                                # -> outputs/main/selection/gated_lambda_choice.json
  python $S/stage_cells.py --stage gated-lambda-bisect --run    # refine cells that bracket the target; re-pick
  python $S/stage_cells.py --stage gated --run                  # seeds 43/44 at the chosen lambda
  python $S/stage_cells.py --stage dual-gated-lambda --run      # lambda check on the Dual-Gated arm
  python $S/pick_gated_lambda.py --arm dual                     # (+ dual-gated-lambda-bisect / -floor if asked)
  python $S/stage_cells.py --stage dual-sweep --variant dual_matryoshka --run
  python $S/stage_cells.py --stage dual-sweep --variant dual_gated --run
  python $S/l0_trajectory.py
  python $S/pick_dual_winners.py --variant dual_matryoshka      # -> dual_matryoshka_winners.json
  python $S/pick_dual_winners.py --variant dual_gated
  python $S/stage_cells.py --stage dual-confirm --variant dual_matryoshka --run
  python $S/stage_cells.py --stage dual-confirm --variant dual_gated --run
  ```

  The rules are documented in the pickers: GatedSAE lambda = the value whose realized L0 is closest
  (in log space) to 64 within [48, 85] among converged runs; Dual winner = the setting with the best
  validation MonoScore on seed 42 among converged settings (best epoch ≥ 20 % of training), with the
  realized L0 of emergent-sparsity variants constrained to the same band. Without re-running the
  selection, `scripts/main/write_reported_selection.py` writes the four selection files for the
  reported settings (hyper-parameters only), which `stage_cells.py --stage dual-confirm` and the
  CB-SAE pipeline (`cb_sae/`) read.

## Evaluation and tables

```bash
python scripts/main/eval_runs.py --runs './checkpoints/main/*' './checkpoints/param_matched/*'
python analysis/main/collect_main.py --out ./outputs/main/main_long.csv
python analysis/main/build_main_table.py --long ./outputs/main/main_long.csv
python analysis/main/build_param_matched_table.py --long ./outputs/main/main_long.csv
python analysis/main/plot_radar.py --long ./outputs/main/main_long.csv --dataset imagenet
python analysis/main/plot_radar.py --long ./outputs/main/main_long.csv --dataset open_images
```

or `bash scripts/main/run_main.sh` (all steps; `STEPS=collect` to rebuild only the outputs).

* `branch_metrics.py` is the offline evaluator: R², MonoScore, binary purity, KNN and LP of one
  checkpoint, on four representations (`x`, `xM`, `r_hat`, `xhat`); `knn_lp_sparse.load_run`
  rebuilds any model type (CB-SAE runs through `cb_adapter.py`, which imports `cb_sae/`).
  `--features-root` remaps the feature paths recorded in a run when the features have moved.
* `collect_main.py --source as_reported` (default) reads the TopK / BatchTopK / JumpReLU pairs and
  the parameter-matched arms from the trainers' `best_metrics.json` and the MatryoshkaSAE / GatedSAE
  / CB rows from the offline evaluator; `--source offline` reads every row from the offline
  evaluator. Splits: ImageNet-1K validation for every column; OpenImagesV7 test for R², MonoScore,
  KNN and LP, and validation for binary purity. CB rows come from `cb_sae/`'s collector
  (`./outputs/cb_sae/cb_long.csv`, `--cb_long`) when it exists. Values are mean ± sample standard
  deviation over seeds.

## Implementation notes

* **Residual Branch.** At `--poly_layers 1` (all runs) the Residual Branch is `Linear(d, d)` followed
  by GELU; `--poly_hidden` is only used for `--poly_layers ≥ 2`.
* **Warm-up.** During warm-up the Residual Branch is frozen at its random initialization, but its
  output is still added to the reconstruction (`x_hat = x^M + r_hat` with the frozen projector), and
  the decorrelation loss is off. `--poly_warmup_zero` (an ablation option) holds `r_hat` at 0 instead.
  After warm-up the optimizer is rebuilt over the Sparse-Branch and Residual-Branch parameters.
* **Decorrelation loss.** `L_dcor` is computed per sample over the decoder rows of the latents active
  for that sample, divided by the number of active latents, and averaged over the batch.
* **Decoder normalization.** The TopK / BatchTopK / JumpReLU / Matryoshka / Gated baselines keep unit-norm
  decoder atoms with gradient projection; DualSAE normalizes `W_dec` along the feature dimension
  (`DualSAE.normalize_decoder_columns_`), so its atoms are not unit-norm.
* **TopK baseline.** Tied encoder/decoder weights and per-sample input standardization (`main.py`
  forces `normalize=True` for `--model topk`); Dual-TopK uses separate encoder/decoder weights and no
  input standardization. BatchTopK, Matryoshka and Gated use no standardization; JumpReLU and
  Dual-JumpReLU do (`--normalize`).
* **Auxiliary loss and dead latents.** TopK and BatchTopK/Matryoshka baselines use the AuxK loss
  (coefficient 1/32; auxiliary k = k for TopK, 512 for BatchTopK). DualSAE has no AuxK term; instead it
  re-initializes dead latents every `--dead_check_interval` steps. `train_other.py` (BatchTopK,
  JumpReLU, Gated, Matryoshka) has no dead-latent re-initialization.
* **Gradient clipping.** Only `train_other.py` clips (`--max_grad_norm`, effectively disabled for
  GatedSAE); `train_topk.py` and `train_dual.py` do not clip.
* **JumpReLU.** The threshold is initialized at θ = 1 (`log θ = 0`; `--jump_threshold_init` changes it).
  The parameter-matched JumpReLU runs used θ = 1e-3 (passed explicitly by
  `train_param_matched.py`). In Dual-JumpReLU the threshold trains during warm-up only (it is not part
  of the post-warm-up optimizer; `--dual_train_threshold_post_warmup` changes that).
* **GatedSAE.** Adam betas (0.0, 0.999) and a linear 2000-step ramp of the gate L1, as in the reference
  trainer; the Dual-Gated arm uses the default Adam betas, divides its gate L1 by the input dimension
  (its reconstruction and auxiliary terms are per-element means) and has its own calibrated lambda.
* **Checkpoint selection.** Every trainer keeps the epoch with the best validation MonoScore (any epoch);
  all metrics are reported for that checkpoint.
* **MonoScore** is averaged over all latents of the dictionary, latents that never fire (or fire on a
  single sample) contributing 0; the offline evaluator also stores the mean over active latents
  (`mean_active_paper`). **R²** is the mean over feature dimensions of the per-dimension R² (the
  offline evaluator also stores the global ratio, `r2_global`). **Binary purity** is the mean over
  latents that fire at least once on the split. **KNN** uses k = 20 cosine neighbours on L2-normalized
  reconstructions; **LP** is a linear layer trained 10 epochs (AdamW, lr 1e-3, weight decay 1e-4,
  batch 1024). On OpenImagesV7 KNN/LP are micro-F1 at threshold 0.5; the trainers average a per-batch
  micro-F1 weighted by batch size, the offline evaluator stores both that and the pooled micro-F1.
* **Step budget.** At batch 2048 an epoch is 625 steps on ImageNet-1K and 247 on OpenImagesV7 (12,350
  steps in 50 epochs); a warm-up at least as long as the run never unfreezes the Residual Branch
  (`main.py` refuses it unless `--allow_degenerate_warmup` is given).
* **OpenImagesV7 samples** are (image, Localized-Narratives caption) pairs, so images with several
  captions appear several times; only images with a narrative are used.

## Tests

```bash
cd vision && python -m pytest tests -q
```

`tests/test_smoke_train.py` trains every model type for 2 epochs on synthetic LMDBs (CPU) and runs the
offline evaluator; the other tests pin the models, the pickers and the evaluator.
`tests/test_equivalence_original.py` is a code-equivalence check against an original copy of the
training code (bit-identical training on synthetic data, bit-identical forward passes of trained
checkpoints); it is skipped unless `DUALSAE_ORIGINAL_CORE` (and `DUALSAE_EQUIV_RUNS`) are set. The
parity tests against the reference GatedSAE / Matryoshka trainers run when `DICTIONARY_LEARNING_PARENT`
points at a checkout of `saprmarks/dictionary_learning`.

## Code provenance

`models/topk_sae.py` follows the TopK autoencoder of `openai/sparse_autoencoder` (tied transpose,
latent statistics, AuxK); `models/{base,batch_topk_sae,jumprelu_sae,vanilla_sae}.py` follow the
BatchTopK SAE code of Bussmann et al.; `models/gated_sae.py` and `models/matryoshka_sae.py` are ports
of the GatedSAE and Matryoshka BatchTopK trainers of `saprmarks/dictionary_learning` (MIT); every
deliberate deviation is flagged in the module docstrings.
