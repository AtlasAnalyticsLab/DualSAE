# DualSAE vision ablations and analyses

Drivers (`vision/scripts/ablations/`) and table / figure builders (`vision/analysis/ablations/`)
for every vision ablation and analysis item of the paper outside the main comparison table.
Each item follows the same path: **train** (`vision/main.py`, with the exact arguments of the
reported runs) → **select** (best-validation-MonoScore checkpoint unless stated) → **evaluate**
(`vision/branch_metrics.py`) → **collect** → **table / figure**. Every table and figure is built
from the run and evaluation outputs; no result values are stored in this directory.

Terminology: the paper's *Sparse Branch* is called `mono` in the code (output `xM`, latent code
`z`); the *Residual Branch* is `poly` (output `r_hat`); the decorrelation loss L_dcor is the
"span" penalty and its coefficient gamma is `--span_coef`; the Stage-1 length is `--warmup_steps`.

## Coverage

| Paper item | Driver | Builder |
|---|---|---|
| Method: *Effect of warm-up length* (validation R² / MonoScore per epoch) | `warmup/run_warmup.sh` | `analysis/ablations/plot_warmup_curves.py` |
| Motivation: class purity of x̂ vs. residual retrieval + example retrievals | `figures/run_motivation.sh` | `motivation/top50_class_purity.py`, `motivation/retrieval_grid.py` |
| *Improved interpretability of sparse latents* (activating images, per-latent MonoScore / purity) | `figures/run_matched_latents.sh` | `latent_activation_grid.py` |
| *t-SNE visualization of sparse representations* | `figures/run_tsne_sparse.sh` | `tsne_sparse_codes.py` |
| *DualSAE vs. SAE with Residual Branch* (joint vs. sequential figure) and supplementary table *Sequential vs. joint training* | `grid/run_grid.py` (arms S, F, Z, R0) | `collect_grid.py` → `plot_joint_vs_seq.py`, `make_grid_tables.py` |
| *Branch decomposition in DualSAE* (a) joint t-SNE of x^M, r̂, x̂ | `figures/run_branch_tsne.sh` | `tsne_branches.py` |
| *Branch decomposition* (b) and supplementary *Class information in each branch on OpenImagesV7* | `branch_metrics/run_branch_metrics.py` (main-table runs) | `collect_branch_reps.py` → `plot_branch_bars.py` |
| *Effect of decorrelation loss on branch separation* (retrieval) | `figures/run_span_retrieval.sh` | `span_retrieval.py` |
| *Effect of the expansion ratio* (main) + supplementary expansion-ratio figures | `grid/run_grid.py` (arms E*, SE*, reruns) | `collect_scale_ablation.py` → `build_expansion_data.py` → `plot_expansion_main.py`, `plot_expansion.py` |
| Supplementary *Input–reconstruction alignment* | `figures/run_recon_alignment.sh` | `tsne_x_xhat.py` |
| Supplementary *Coefficient of the decorrelation loss* (R² and MonoScore vs. gamma) | `grid/run_grid.py` (arms G*) | `collect_span_ablation.py --checkpoint final` → `make_gamma_tables.py` |
| Supplementary *Decorrelation loss* (metric change without L_dcor) | `dcor/run_dcor.py` | `make_dcor_table.py` |
| Supplementary *Projector of the Residual Branch* | `grid/run_grid.py` (arms R0, B256, B512, D3, D4) | `collect_grid.py` → `make_grid_tables.py` |

## Install and assets

```bash
pip install -r vision/requirements.txt -r vision/scripts/ablations/requirements-ablations.txt
bash vision/scripts/ablations/prepare_assets.sh   # once, with internet: ImageNet-1K val images (HF), WordNet
```

Everything else runs offline (`common.sh` sets `HF_*_OFFLINE=1` and `WANDB_MODE=disabled`).
Image-grid figures read tiles from `IMAGE_ROOT/<path stored in the LMDB>` when `IMAGE_ROOT` is
set, otherwise from the cached Hugging Face `ILSVRC/imagenet-1k` validation split by LMDB index.

Environment variables used by all drivers:

| Variable | Meaning | Default |
|---|---|---|
| `FEATURES_ROOT` | feature LMDBs, `<root>/<dataset>/<split>/<file>.lmdb` (`feature_paths.py`) | `./features` |
| `ABL_ROOT` | all runs, evaluations, tables and figures of this suite | `./outputs/ablations` |
| `PY` | python interpreter | `python` |

## Main-table runs

Several items read the main-table runs (TopK / BatchTopK / JumpReLU SAEs and their DualSAE
pairs, `vision/scripts/main/`). List them and score them on every representation once:

```bash
python vision/scripts/ablations/make_run_manifest.py --checkpoints <main checkpoint dir> --out main_runs.json
python vision/scripts/ablations/branch_metrics/run_branch_metrics.py --manifest main_runs.json
```

`configs/main_dual_cells.json` holds the training settings of the main-table DualSAE runs (per
dataset, encoder, Sparse-Branch activation and seed: gamma, warm-up, learning rate, input
normalisation, dead-latent check interval, plus the shared settings), copied from those runs'
`training_config.json`. The ablation grid and the decorrelation-loss ablation replay them.

## Running

Full ablation grid (joint vs. sequential, projector, gamma, expansion), all stages in order:

```bash
MAIN_MANIFEST=main_runs.json MAIN_RESULTS=$ABL_ROOT/main_branch_metrics/results \
  bash vision/scripts/ablations/grid/run_all.sh
```

Single stage / cell (every stage skips finished work and can be re-run):

```bash
python vision/scripts/ablations/grid/run_grid.py --stage print --arms R0 --only-dataset imagenet  # commands only
python vision/scripts/ablations/grid/run_grid.py --stage train --arms B256 --index 0            # one cell
python vision/scripts/ablations/grid/run_grid.py --stage eval --shard 0/4                       # a quarter of the evaluations
```

On a cluster, submit any driver as an array with `slurm_template.sh` (fill in the placeholders);
`--index` addresses the full, stable list, so finished items are skipped.

Other items:

```bash
bash vision/scripts/ablations/warmup/run_warmup.sh
python vision/scripts/ablations/dcor/run_dcor.py --stage train && python vision/scripts/ablations/dcor/run_dcor.py --stage eval
python vision/analysis/ablations/make_dcor_table.py --main-manifest main_runs.json
bash vision/scripts/ablations/figures/run_motivation.sh        # and the other figures/*.sh
```

Figure drivers train the runs their figure used unless existing run directories are passed
(e.g. `BTK_RUN=<dir> DUAL_RUN=<dir> bash figures/run_matched_latents.sh`).

## Outputs (under `$ABL_ROOT`)

| Path | Content |
|---|---|
| `grid/checkpoints/abl_<dataset>_<encoder>_<activation>_<arm>_seed<s>/` | grid runs (F snapshots: `..._F_seed<s>__step<T>/`) |
| `grid/branch_metrics/results/` | evaluation JSONs |
| `grid/long.csv`, `grid/REPORT.md`, `grid/curves.csv` | collected values, checks, per-epoch curves |
| `grid/frozen_vs_dualsae.(tex\|md)`, `grid/projector_ablation.(tex\|md)`, `grid/joint_vs_seq_tradeoff*.pdf` | tables and figures |
| `gamma/long_final.csv`, `gamma/gamma_r2.tex`, `gamma/gamma_ms.tex`, `gamma/gamma_ablation_table.(tex\|md)` | gamma ablation |
| `expansion/long.csv`, `expansion/expansion_data.json`, `expansion/expansion_ratio_*.pdf` | expansion ratio |
| `dcor/dcor_ablation.(tex\|md\|csv)` | decorrelation-loss ablation |
| `main_branch_metrics/branch_reps.csv`, `.../figures/` | branch bars, branch-decomposition composite |
| `warmup/`, `figures/<item>/` | warm-up curves and the qualitative figures |

## Implementation notes

* **Residual Branch.** With `poly_layers = 1` (all reported DualSAE runs) the projector is
  `Linear(d, d)` followed by GELU; `--poly_hidden` is recorded in the configs but not read. The
  projector-ablation arms B<h> / D3 / D4 build `Linear → GELU → … → Linear` without a terminal GELU;
  B512G adds one. The grid also trains L1 (`Linear(d, d)`), B64, B2048 and B512G;
  `make_grid_tables.py --projector-arms` adds them to the table.
* **Warm-up (Stage 1).** During warm-up the projector is frozen but its output for the random
  initial weights is still added to the reconstruction, and gamma is forced to 0. Arm Z
  (`--poly_warmup_zero`) holds r̂ at 0 during warm-up instead.
* **Replayed settings.** Grid arms and the ImageNet-1K arm of the decorrelation-loss ablation use the
  per-cell gamma, warm-up and dead-latent check interval of the main-table runs
  (`configs/main_dual_cells.json`), which differ between cells and from `main.py`'s defaults. In
  some cells they also differ between seeds (see the JSON).
* **Sequential training (arm F)** freezes the Sparse Branch of arm S (a standard SAE trained with
  DualSAE's Sparse Branch code, no Residual Branch) and trains a fresh default projector with the
  cell's gamma and no warm-up. The table and figure read F after the same number of Residual Branch
  updates as the jointly trained R0 at its selected checkpoint (role `at_D_best`); the DualSAE
  points are the main-table runs (`--joint published_dual`, default) or R0 (`--joint R0`). The
  "SAE" point of the figure is arm S, not the main-table SAE.
* **Gamma tables** read the epoch-50 checkpoint (`collect_span_ablation.py --checkpoint final`);
  all other tables use the best-validation-MonoScore checkpoint. The gamma = 1e-2 column is R0
  wherever a cell's main-table gamma is exactly 0.01.
* **Expansion ratio** uses seed 42 and dictionaries of 4, 8, 16 (and 32, trained but not plotted)
  times the encoder width at k = 64; the standard SAE is arm SE<r>. Two x16 points are taken from
  reruns with restricted checkpoint selection (`ablation_grid.SELECTION_RERUNS`,
  `--select_from_epoch`). Dead-latent resampling is effectively off (`dead_steps_threshold` 1e7).
* **Decorrelation-loss ablation.** ImageNet-1K: main-table configuration with gamma = 0 against the
  main-table run. OpenImagesV7: a matched pair (gamma 0 vs. 0.01) at warm-up 2500 and lr 1e-4,
  trained with JumpReLU threshold init 1e-3. For OpenImagesV7 JumpReLU the best-validation
  checkpoint of both arms falls inside warm-up, where gamma is not active.
* **JumpReLU threshold init.** Runs that replay the main-table JumpReLU settings pass
  `--jump_threshold_init 1.0`; the OpenImagesV7 decorrelation pair passes `0.001`.
* **Warm-up figure.** The drawn runs are OpenImagesV7 / CLIP / Dual-BatchTopK (gamma 0.01), one run
  per warm-up length: `--seed-per-warmup 0:43 2500:42 5000:42 7500:43` selects the same runs;
  without it the curves are seed means. The full warm-up grid that was trained is listed in
  `warmup/run_warmup.sh`.
* **Branch bars.** KNN (k = 20, cosine) and LP (10 epochs) on x^M and r̂ of the main-table
  DualSAE runs; ImageNet-1K top-1 accuracy on val, OpenImagesV7 micro-F1 on test. Panel (a) of the
  composite comes from `figures/run_branch_tsne.sh`: a Dual-TopK CLIP run with warm-up 2500,
  seed 44, read at its epoch-40 snapshot, classes drawn with random state 0.
* **Qualitative figures.** Motivation: BatchTopK, CLIP, seed 42, epoch-50 checkpoint, query =
  validation index 1. Matched latents: SigLIP2, seed 43, main-table configurations; the rank of the
  drawn latent pair is not recorded. Sparse-code t-SNE: the 20 classes are the fixed list in
  `tsne_sparse_codes.py` (SigLIP2, seed 43; the Dual-JumpReLU run uses gamma 0.01, lr 5e-5, L0
  coefficient 1e-3, warm-up 17500). Input–reconstruction alignment: CLIP, BatchTopK seed 42 vs.
  Dual-BatchTopK seed 44 (gamma 0.01, warm-up 17500); the plot window is fixed to [-5, 5].
  Decorrelation retrieval: ViT, seed 42, warm-up 17500, gamma 0 vs. 2.5, query = validation index 4.
* **MonoScore** of a model is the mean of the per-latent MonoScores over all latents (latents that
  never fire count as 0); purity is binary purity over active latents.

## Tests

```bash
python -m pytest vision/tests/test_abl_grid.py vision/tests/test_abl_arms.py \
                 vision/tests/test_abl_builders.py vision/tests/test_abl_figures.py -q
```

CPU only, synthetic data, no downloads: grid definition and drivers; the grid architectures (S, F,
Z, projector options) through the real trainer and CLI; every collector / table / plot builder on
synthetic evaluation outputs; every qualitative-figure script on tiny trained models.
