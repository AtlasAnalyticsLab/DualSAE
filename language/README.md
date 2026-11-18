# Language: SAEs and DualSAE on the Pythia residual stream

This component trains standard SAEs and DualSAE on residual-stream activations of
`EleutherAI/pythia-410m-deduped` and evaluates them by splicing the reconstruction back into
the model. It uses the SAE/DualSAE core in `../vision` (the same `main.py`, trainers, models and
scorers) and adds the language-specific steps: corpus packing, activation extraction,
evaluation (ΔCE, KL, R², MonoScore), the table builder and the qualitative context pages.

Internally the experiments are called "Track B", which is why every cell tag starts with `B_`.

## What it covers

| Paper item | How to produce it |
|---|---|
| §5 "Language representation": table "Language-model evaluation" (OWT / GHC, block 12) | train and evaluate the main grid and the warm-up grid (below), then `python build_language_table.py` |
| §5 figure "Qualitative analysis on language representations", panel (a) | `bash scripts/make_figures.sh` (page `contexts_L12_owt_topk_seed42_tuned.html`) |
| Supplementary "Evaluation on language-model representations": setup, data preparation, activation normalisation, SAE configuration, metric definitions | the code paths below (see *Implementation notes*) |
| Supplementary figures "person name", "professional title", "end-of-text" and "class attribute" latents | `bash scripts/make_figures.sh` (the page and concept used for each figure are listed in that script) |
| §5 figure panel (b) (steering) and the LLM-steering supplement | `../llm_steering/`; it loads SAEs trained here and the outputs of the `textemb` stage (see *On-disk layout*) |
| Inputs of `../llm_steering/` (required by llm_steering/): materialised AG News / GoEmotions corpora and their bge-large-en-v1.5 document embeddings | `bash language/run_pipeline.sh --stage textemb` |

The table's rows come from both grids at block 12: main-grid cells and warm-up-ablation cells.
`build_language_table.py` writes one row per training configuration it finds. Each row lists
its run tags, warm-up, learning rate, γ, JumpReLU threshold init, the checkpoint scored and
its epochs, so every row can be traced to the runs behind it.

Every qualitative figure page is built from **TopK-family cells** (Dual-TopK vs TopK, block
12). Each page labels its two columns with the family of the cells it was drawn from.

## Install

```bash
pip install -r language/requirements.txt     # the environment the runs used (Python 3.11)
```

The code imports `../vision` (via `scripts/env.sh`, which sets `PYTHONPATH`), so keep the
repository layout intact.

## Assets (the only step that needs internet)

```bash
export LANGDUAL_ROOT=/path/to/outputs/language      # default language/outputs/language
bash language/prepare_assets.sh
```

This downloads the Pythia-410m-deduped snapshot and two parquet shards each of
`Skylion007/openwebtext` and `codeparrot/github-code-clean`. For the `textemb` stage it also
downloads `BAAI/bge-large-en-v1.5`, `fancyzhx/ag_news` and the `simplified` config of
`google-research-datasets/go_emotions`. All downloads are at the commits pinned in
`langdual/assets.py`. Every later stage runs fully offline (`HF_HUB_OFFLINE=1`).

## Running

All drivers are sequential and resumable: every unit has a skip-guard, so re-running a
command continues where a killed run stopped.

```bash
bash language/run_pipeline.sh --stage pack     --grid main      # token streams, both corpora (CPU)
bash language/run_pipeline.sh --stage extract  --grid main      # activations, blocks 6/12/18 (GPU)
bash language/run_pipeline.sh --stage train    --grid main      # 108 cells
bash language/run_pipeline.sh --stage eval     --grid main
bash language/run_pipeline.sh --stage train    --grid warmup    # 108 DualSAE warm-up cells (OWT)
bash language/run_pipeline.sh --stage eval     --grid warmup
bash language/run_pipeline.sh --stage train    --grid steering  # 27 SAEs used by llm_steering/
bash language/run_pipeline.sh --stage textemb                   # corpora/ + lmdb/bge/ for llm_steering/
python language/build_language_table.py                         # block 12; --layers 6 12 18 for all
bash language/scripts/make_figures.sh
```

Restrict a grid with `--layers "12"`, `--corpora "owt"`, `--activations "topk"`,
`--seeds "42"` or (ablation grids) `--values "1250"`. `--dry-run` prints the commands without
running them. For the block-12 language table alone:

```bash
bash language/run_pipeline.sh --stage all --grid main   --layers 12
bash language/run_pipeline.sh --stage all --grid warmup --layers 12
```

Single cell:

```bash
LD_DRY_RUN=1 bash language/train_cell.sh B_L12_owt_topk_dual_seed42   # print the exact argv
bash language/train_cell.sh B_L12_owt_topk_dual_seed42
(cd language && python -m langdual.lmsae.eval_trackb --tag B_L12_owt_topk_dual_seed42)
```

List cells with `python -m langdual.jobs --list main|ablation [filters]`. For a cluster,
`scripts/slurm_template.sh` wraps one cell (train and evaluate) with placeholder account and
partition.

### Grids

| grid | cells | contents |
|---|---:|---|
| `main` | 108 | corpora {OpenWebText `owt`, GitHub code `ghcode`} × blocks {6, 12, 18} × {TopK, BatchTopK, JumpReLU} × {standard SAE, DualSAE} × seeds {42, 43, 44}; DualSAE warm-up 2,500 steps |
| `warmup` (axis `warmupB`) | 108 | DualSAE only, OpenWebText, blocks {6, 12, 18} × the three activations × warm-up {0, 1250, 5000, 10000} × 3 seeds (the 2,500 default is the main grid) |
| `steering` (axes `tkuntied`, `dg0B1250`) | 27 | TopK baseline with an untied decoder (blocks 6/12/18), and DualSAE with γ = 0 at warm-up 1,250 (TopK/BatchTopK, blocks 6/12/18); trained here, evaluated by `llm_steering/` |

## On-disk layout (contract with `llm_steering/`)

Everything lives under `$LANGDUAL_ROOT`:

```
raw/models/EleutherAI__pythia-410m-deduped/        model + tokenizer snapshot (.langdual_snapshot.json)
raw/datasets/{Skylion007__openwebtext,codeparrot__github-code-clean}/
tokens/{train,val,test}.u16.npy (+ .meta.json)      OpenWebText token streams (uint16, packed)
tokens/ghcode_{train,val,test}.u16.npy              GitHub-code token streams
resid/L{6,12,18}/{train,val,test}/                  OpenWebText activations: shard_%05d.npy (fp16) + manifest.json
resid/ghcode/L{n}/{split}/                          GitHub-code activations
runs/B_L{n}_{corpus}_{act}_{variant}_seed{S}/       main-grid cells
ablation/runs/abl_warmupB{W}_B_L{n}_owt_{act}_dual_seed{S}/
ablation/runs/abl_tkuntied_B_L{n}_owt_topk_baseline_seed{S}/
ablation/runs/abl_dg0B1250_B_L{n}_owt_{topk,batch_topk}_dual_seed{S}/
results/trackb/<tag>.json, ablation/results/trackb/<tag>.json     evaluation outputs
raw/models/BAAI__bge-large-en-v1.5/                 text-embedding stage (required by llm_steering/):
raw/datasets/{fancyzhx__ag_news,google-research-datasets__go_emotions}/
corpora/{agnews,goemotions}/{train,val,test}.jsonl  canonical documents ({"id", "text"} per line)
corpora/{agnews,goemotions}/{split}_labels.npy      int64 (AG News) / uint8 multi-hot (GoEmotions, 28)
corpora/{agnews,goemotions}/split_indices.json, meta.json
lmdb/bge/{agnews,goemotions}/{split}.lmdb           bge-large document embeddings (keys below)
lmdb/_labels/goemotions/{split}_labels.lmdb         multi-hot label store referenced by __labels_lmdb__
reports/language_table.{md,csv,tex}, reports/qualitative/*.html
tensorboard/
```

The fields of a cell tag are as follows:
- `{n}` is the block, 6, 12 or 18.
- `{corpus}` is `owt` or `ghcode`.
- `{act}` is `topk`, `batch_topk` or `jump_relu`.
- `{variant}` is `baseline` (the standard SAE) or `dual`.

Each run directory holds these files:
- `best_val_monoscore_model.pt`: the epoch with the highest validation MonoScore; it carries `model_cfg`.
- `autoencoder_final.pt`: the end of training.
- `training_config.json`.
- `train_stdout.log`.
- `CELL_DONE.json`.

Each embedding LMDB holds these keys:
- `__len__` and `__shape__` (`"(1024,)"`).
- `"<i>"`: raw float32 bytes of row `i`, stored at the encoder's native scale; it is
  L2-normalised on read.
- `__path__<i>`: the document id `<corpus>/<split>/<i>`.
- AG News only: `__label__<i>`.
- GoEmotions only: `__multilabel__="1"` and an absolute `__labels_lmdb__` path.
- Provenance keys: `__encoder__`, `__encode_dtype__`, `__max_seq_length__`, `__truncated_fraction__`, ...

Row `i` of an LMDB is line `i` of the matching `corpora/<corpus>/<split>.jsonl`.

Load a model with `knn_lp_sparse.load_run(Path(run_dir), device, checkpoint="best"|"final")`
from `../vision`. Its input is the shard space: the raw residual stream divided by the
`norm_factor` stored in the shard `manifest.json` (`norm_convention: "divide"`).

## Outputs

`results/trackb/<tag>.json` holds the following blocks:
- `recon`: pooled R² `r2_global_a4`, mean-of-per-dimension R² `r2_perfeat_table3`, cosine, norm ratio, normalised MSE.
- `monoscore`: exact min-max MonoScore in three reductions, its null floor, and a quantile-robust variant.
- `sparsity`: L0 and the dead-latent fraction.
- `ce`: clean, spliced and zero-ablated CE, `frac_recovered`, and `kl_clean_vs_reconstructed`.
- `meta`: run configuration, the checkpoint scored and its epoch.

The table builder reads these four keys:

| column | key |
|---|---|
| ΔCE | `ce.loss_reconstructed − ce.loss_original` |
| KL | `ce.kl_clean_vs_reconstructed` |
| R² | `recon.r2_global_a4` |
| MS | `monoscore.minmax.mean_active_paper_a4` (`--ms-reduction all` gives the mean over all latents) |

## Implementation notes

**Names.** The paper's Sparse Branch is `mono` in code, its Residual Branch is `poly`, and the
coefficient γ of the decorrelation loss L_dcor is `--span_coef` (0.01).

**Activations and data.**
- The hooked site is the output of `model.gpt_neox.layers[n]` (0-indexed), i.e. HF
  `hidden_states[n+1]`, for n ∈ {6, 12, 18}.
- Extraction runs in fp32 with TF32 matmuls; activations are stored as fp16.
- Each (corpus, block) is divided by one global scale, sqrt(E‖x‖²) over the first 2M training
  tokens, folded into the stored shards. No per-sample normalisation is applied
  (`--force_normalize off` for every cell).
- Documents are packed in raw shard order with one end-of-text token each and no padding.
  Empty documents are skipped; there is no deduplication step.
- The training stream is exactly 64M tokens (31,250 steps × 2,048). Validation and test are
  200K tokens each, taken from documents after the training stream.

**Training.**
- All cells use 8,192 latents, batch 2,048, 31,250 steps (`--num_epochs 50 --steps_per_epoch 625`),
  Adam with eps 6.25e-10, and seeds 42/43/44.
- Learning rate is 1e-4 for TopK/BatchTopK and 5e-5 for JumpReLU (both arms of a pair).
- k = 64 applies to TopK/BatchTopK. JumpReLU sparsity is set by `--l1_coef 1e-4` and
  `--bandwidth 1e-3`.
- The JumpReLU threshold is initialised at θ = 1e-3 (`--jump_threshold_init 0.001`, passed
  explicitly because the core's default differs). With this initialisation and the
  residual-stream scale, the learned thresholds stay close to their initial value, so JumpReLU
  cells are far denser than k = 64. Their L0 is reported per row.
- In Dual-JumpReLU the threshold is trained during warm-up only: it is left out of the
  optimizer rebuilt when the Residual Branch is enabled.
- The Residual Branch at `--poly_layers 1` is a dense `Linear(1024, 1024)` + GELU reading the
  residual; `--poly_hidden` is inert on this path.
- During warm-up the Residual Branch is frozen, but its (random) output is still added to the
  reconstruction.
- Training paths differ by family:
  - The TopK baseline uses a tied decoder (`train_topk.py` default), AuxK (1/32) and dead-latent
    reinitialisation.
  - The BatchTopK and JumpReLU baselines use AuxK where implemented and gradient clipping (1.0),
    with no reinitialisation.
  - DualSAE uses dead-latent reinitialisation only (no AuxK, no clipping).
- The scored checkpoint is `best_val_monoscore_model.pt`, the epoch with the highest
  validation MonoScore. Its epoch is recorded in each result and shown by the table builder.
  `--checkpoint final` scores the end of training instead.

**Evaluation.**
- Metrics are computed on the first 198,656 test tokens (97 batches of 2,048; BatchTopK's
  top-k is batch-level, so the encode batch equals the training batch).
- MonoScore uses the L2-normalised raw residual activation as the probe space.
- CE and KL splice the reconstruction (× the norm factor) back into the hooked block on the
  test stream: 97 batches of 16 contexts × 128 tokens, fp32, no padding.
- The qualitative pages encode in training-sized batches, skip latents that fire at the first
  position of most contexts or on more than 5% of tokens, and pair each Dual latent with the
  baseline latent whose activations correlate best over the same tokens.

**Text-embedding stage.**
- AG News ships only train/test. A stratified validation split of 7,600 rows is carved from
  train (seed 42) and recorded in `split_indices.json`; GoEmotions uses its own validation split.
- Documents are embedded with `bge-large-en-v1.5` through sentence-transformers: no prefix,
  `max_seq_length` 512, fp32 weights with TF32 matmuls, batch 256, `normalize_embeddings=False`.
  The module chain ends in a Normalize layer, so the stored vectors are unit-norm anyway.

## Tests

```bash
python language/tests/smoke_test.py --work /tmp/lang_smoke    # synthetic, no downloads, ~1 min on a GPU
python -m langdual.jobs --selftest                             # registry (run inside language/)
```

The smoke test builds a tiny random GPT-NeoX model and tokenizer plus a synthetic parquet
corpus, then runs pack, extract, train (TopK and Dual-TopK), evaluate, the table builder and
one context page through the real code paths. `pyarrow` is needed for the pack step; without
it the step is skipped.

`tests/check_argv_equivalence.py` compares the argv that `train_cell.sh` builds with argv
recorded in worker logs from an earlier execution.
