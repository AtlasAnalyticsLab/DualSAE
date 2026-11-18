# DualSAE: Separating Interpretability and Reconstruction in Sparse Autoencoders

Code for the anonymous submission *"Separating Interpretability and Reconstruction in Sparse Autoencoders"*.

DualSAE augments a standard sparse autoencoder with a plug-in **Residual Branch**:

- the **Sparse Branch** is an ordinary SAE (TopK, BatchTopK, JumpReLU, Matryoshka, Gated, or CB-SAE) that maps an embedding `x` to a sparse code `z` and reconstructs `x^M = W_D z + b_D`;
- the **Residual Branch** reads the residual `r = x - x^M` and predicts `r_hat = Q(W_P r)`;
- the final reconstruction is `x_hat = x^M + r_hat`.

Training minimizes `L = L_rec(x, x_hat) + lambda * L_sparse + gamma * L_dcor`, where the **decorrelation loss** `L_dcor = ||D_x^T r_hat||^2` penalizes the Residual Branch output along the decoder directions of the latents active for `x`. Training is two-stage: the Sparse Branch is first trained alone (warm-up), then both branches are trained jointly.

This repository contains the code for **every experiment in the paper** (main text and supplementary material), organized by domain. Each directory has its own README with installation, asset preparation, and the exact commands for each table and figure.

## Repository layout

| Directory | Contents |
|---|---|
| [`vision/`](vision/) | Core library: all SAE baselines and DualSAE, training, feature extraction (ImageNet-1K, OpenImagesV7; CLIP, SigLIP2, ViT), the five metrics (R², MonoScore, class purity, KNN, linear probe), the main SAE-vs-DualSAE grid, parameter-matched baselines, and all vision ablations and analyses ([`vision/scripts/ablations/`](vision/scripts/ablations/)). |
| [`cb_sae/`](cb_sae/) | CB-SAE and CB-DualSAE: concept dissection, pruning and concept-bottleneck training on top of a Matryoshka / Dual-Matryoshka base. |
| [`language/`](language/) | SAEs and DualSAEs on language-model (Pythia) residual-stream activations of OpenWebText and GitHub code. |
| [`topic_modeling/`](topic_modeling/) | SAE-based topic modeling (built on SAE-TM, applied as a patch to the upstream repository). |
| [`hypothesis_generation/`](hypothesis_generation/) | SAE-based hypothesis generation with VLM-written latent descriptions (uses HypotheSAEs for scoring). |
| [`llm_steering/`](llm_steering/) | Steering Pythia with SAE / DualSAE decoder directions and evaluating the steered generations. |

All components import the models from `vision/` (`vision/models/`), so there is a single implementation of every SAE.

## Terminology: paper vs. code

The code predates the paper's final terminology. Internal names are kept so that checkpoints and run configurations stay loadable; the command-line interface accepts the paper's names as aliases.

| Paper | Code |
|---|---|
| Sparse Branch, `x^M` | "mono" branch, `xM`, `W_enc` / `W_dec` / `b_dec` |
| Residual Branch, `r_hat = Q(W_P r)` | "poly" branch, `poly` module, `r_hat` |
| Decorrelation loss `L_dcor`, coefficient `gamma` | span-overlap penalty, `span_coef` (`--decorrelation_coef` alias) |
| Two-stage training, Stage 1 length | `--warmup_steps` |
| Sparsification rule `S(.)` | `--sparsity_mode {topk,batch_topk,jump_relu,matryoshka,gated,...}` |

## Installation

Python 3.11+ and a CUDA GPU are assumed. Each component pins its own dependencies (they differ: the downstream applications need vLLM, the language experiments need `transformers` model hooks):

```bash
pip install -r vision/requirements.txt          # core + all vision experiments
pip install -r <component>/requirements.txt     # any other component you want to run
```

## Assets and offline execution

Every download (datasets, encoder weights, language models, judges) lives in a separate `prepare_assets.sh` per component, so that training and evaluation can run on machines without internet access. Run the preparation script once on a machine with internet access, then run everything else offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).

## Running experiments

All drivers are plain, sequential bash/python scripts with skip-if-done guards, so an interrupted grid can be re-launched and continues where it stopped. To use a cluster scheduler, wrap the single-cell commands (each driver documents its per-cell command) in your scheduler's job script; each component ships a `scripts/slurm_template.sh` with placeholders (e.g. [`vision/scripts/ablations/slurm_template.sh`](vision/scripts/ablations/slurm_template.sh)).

Some components consume the outputs of others; run them in this order:

1. `vision/` main grid (features, baselines, DualSAEs, selection) — everything else in vision builds on it.
2. `cb_sae/` — reads the Matryoshka / Dual-Matryoshka runs and the selection file written in step 1; its `cb_long.csv` is merged into the main table and radar figures by `vision/analysis/main/collect_main.py`.
3. `language/` — trains the language SAEs (including the extra runs used for steering: `--grid steering`) and, with `--stage textemb`, writes the corpora and document embeddings that `llm_steering/` reads.
4. `llm_steering/` — after step 3.

`topic_modeling/` and `hypothesis_generation/` train their own SAEs through `vision/main.py` and depend only on `vision/`.

Results are reported as mean ± standard deviation over seeds 42, 43, 44. Seeds are the only dimension that is averaged; every other dimension (encoder, dataset, activation, warm-up, gamma, K, ...) is reported separately. A baseline SAE is compared only with the DualSAE that uses the same sparsification rule (TopK vs. Dual-TopK, BatchTopK vs. Dual-BatchTopK, ...).

## Paper-to-code map

The entry points below run each experiment end to end; the linked README lists the individual steps, options and outputs. Vision ablation paths are relative to [`vision/scripts/ablations/`](vision/scripts/ablations/README.md) (drivers) and `vision/analysis/ablations/` (table and figure builders).

**Main paper**

| Paper item | Entry point | Details |
|---|---|---|
| Sec. 3, motivation figure (class purity of retrievals with the reconstruction vs. the residual) | `figures/run_motivation.sh` | [vision ablations](vision/scripts/ablations/README.md) |
| Sec. 4, *Effect of warm-up length* | `warmup/run_warmup.sh` → `plot_warmup_curves.py` | [vision ablations](vision/scripts/ablations/README.md) |
| Sec. 5, *Comparison of SAE and DualSAE on ImageNet-1K* (radar) | `vision/scripts/main/run_main.sh` (+ `cb_sae/scripts/run_grid.sh` for the CB-SAE rows) → `vision/analysis/main/plot_radar.py` | [vision](vision/README.md), [cb_sae](cb_sae/README.md) |
| Sec. 5, *Improved interpretability of sparse latents* | `figures/run_matched_latents.sh` | [vision ablations](vision/scripts/ablations/README.md) |
| Sec. 5, *t-SNE visualization of sparse representations* | `figures/run_tsne_sparse.sh` | [vision ablations](vision/scripts/ablations/README.md) |
| Sec. 5, capacity analysis: *Parameter-matched SAE comparison* | `vision/scripts/param_matched/train_param_matched.py` → `vision/analysis/main/build_param_matched_table.py` | [vision](vision/README.md) |
| Sec. 5, residual analysis: *DualSAE vs. SAE with Residual Branch* | `grid/run_grid.py` → `collect_grid.py` → `plot_joint_vs_seq.py` | [vision ablations](vision/scripts/ablations/README.md) |
| Sec. 5, *Branch decomposition in DualSAE* (t-SNE and per-branch metrics) | `figures/run_branch_tsne.sh`; `branch_metrics/run_branch_metrics.py` → `plot_branch_bars.py` | [vision ablations](vision/scripts/ablations/README.md) |
| Sec. 5, *Effect of decorrelation loss on branch separation* | `figures/run_span_retrieval.sh` | [vision ablations](vision/scripts/ablations/README.md) |
| Sec. 5, *Effect of the expansion ratio* | `grid/run_grid.py` → `collect_scale_ablation.py` → `build_expansion_data.py` → `plot_expansion_main.py` | [vision ablations](vision/scripts/ablations/README.md) |
| Sec. 5, language representation: results table and qualitative latent figure | `language/run_pipeline.sh` → `language/build_language_table.py`; `language/scripts/make_figures.sh` | [language](language/README.md) |
| Sec. 5, downstream: *Image–topic assignment comparison* | `topic_modeling/scripts/run_all.sh` (figures: `06_figures.sh`) | [topic_modeling](topic_modeling/README.md) |
| Sec. 5, downstream: *Hypothesis generation* figure | `hypothesis_generation/scripts/run_all.sh` (figures: `04_figures.sh`) | [hypothesis_generation](hypothesis_generation/README.md) |

**Supplementary material**

| Paper item | Entry point | Details |
|---|---|---|
| Experimental details: hyper-parameters, datasets, vision encoders, metrics | `vision/configs/main_cells.json` (exact arguments of every run), `vision/extract_imagenet.py`, `vision/extract_open_images.py`, `vision/loss.py`, `vision/metrics.py`, `vision/branch_metrics.py` | [vision](vision/README.md) |
| Detailed results on ImageNet-1K and OpenImagesV7 (side-by-side table, OpenImagesV7 radar) | `vision/scripts/main/run_main.sh` + `cb_sae/scripts/run_grid.sh` → `vision/analysis/main/build_main_table.py`, `plot_radar.py` | [vision](vision/README.md), [cb_sae](cb_sae/README.md) |
| Input–reconstruction alignment | `figures/run_recon_alignment.sh` | [vision ablations](vision/scripts/ablations/README.md) |
| Class information in each branch on OpenImagesV7 | `branch_metrics/run_branch_metrics.py` → `plot_branch_bars.py` | [vision ablations](vision/scripts/ablations/README.md) |
| Ablation: coefficient of the decorrelation loss (R² and MonoScore tables) | `grid/run_grid.py` → `collect_span_ablation.py` → `make_gamma_tables.py` | [vision ablations](vision/scripts/ablations/README.md) |
| Ablation: decorrelation loss | `dcor/run_dcor.py` → `make_dcor_table.py` | [vision ablations](vision/scripts/ablations/README.md) |
| Ablation: projector of the Residual Branch | `grid/run_grid.py` → `collect_grid.py` → `make_grid_tables.py` | [vision ablations](vision/scripts/ablations/README.md) |
| Ablation: joint vs. sequential training (all settings, figure and table) | `grid/run_grid.py` → `collect_grid.py` → `plot_joint_vs_seq.py`, `make_grid_tables.py` | [vision ablations](vision/scripts/ablations/README.md) |
| Ablation: expansion ratio (ImageNet-1K and OpenImagesV7 figures) | `grid/run_grid.py` → `collect_scale_ablation.py` → `build_expansion_data.py` → `plot_expansion.py` | [vision ablations](vision/scripts/ablations/README.md) |
| Evaluation on language-model representations (setup, metrics, qualitative latent comparisons) | `language/run_pipeline.sh`; `language/scripts/make_figures.sh` | [language](language/README.md) |
| SAE as a topic model (table and image–topic figures) | `topic_modeling/scripts/run_all.sh` (table: `05_tables.sh`, figures: `06_figures.sh`) | [topic_modeling](topic_modeling/README.md) |
| SAE for hypothesis generation (table and class-level hypothesis figure) | `hypothesis_generation/scripts/run_all.sh` (table: `03_results.sh`, figures: `04_figures.sh`) | [hypothesis_generation](hypothesis_generation/README.md) |
| LLM steering (quantitative table and steering examples) | `llm_steering/run_all.sh` → `python -m steering.make_table`, `python -m steering.select_examples` | [llm_steering](llm_steering/README.md) |

Every table and figure is produced by a script from the run and evaluation outputs; no result values are stored in this repository.

## License

MIT (see [`LICENSE`](LICENSE)), except for third-party code, which keeps its own license (noted in the component READMEs).
