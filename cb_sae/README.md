# CB-SAE and CB-DualSAE

This component builds the **CB-SAE** and **CB-DualSAE** rows of the SAE-vs-DualSAE comparison (the "CB" and "CB-DUAL" pairs of the supplementary comparison table on ImageNet-1K and OpenImagesV7, and the CB-SAE column of the radar figures of the main comparison).

CB-SAE (Kulkarni et al., *Interpretable and Steerable Concept Bottleneck Sparse Autoencoders*, arXiv 2512.10805) is applied **post hoc** to a trained SAE. It ranks the SAE latents by interpretability, keeps the top R, and trains a small concept-bottleneck head next to the frozen retained dictionary. Here it is applied to the paper-native base of CB-SAE, a Matryoshka (BatchTopK) SAE, in both arms of the pair:

| Row | Base (trained by `vision/`) | CB head | Residual Branch |
|---|---|---|---|
| CB (CB-SAE) | MatryoshkaSAE | trained | none |
| CB-DUAL (CB-DualSAE) | Dual-MatryoshkaSAE | trained | warm-started from the base and trained jointly with the head |

Both arms share every CB setting and differ only in the base. The bases are the MatryoshkaSAE and Dual-MatryoshkaSAE runs of the main grid (`vision/scripts/main/`). For the dual arm this is the Dual-Matryoshka setting (decorrelation coefficient, warm-up) that the main grid selected for each (dataset, encoder).

## Pipeline

```
stage 0  prepare_assets.sh / cb_pipeline.py prep   concept vocabularies, supervisor text embeddings,
                                                   probe-image embeddings, row-alignment gate
stage 1  convert   featbsf.convert_{plain_sae,dualsae}   base checkpoint -> b=1 featurizer (parity-gated)
stage 2  dissect   featbsf.cb_dissect                    CLIP-Dissect soft-WPMI: per-latent I_m, dead mask
stage 3  retain    (driver)                              admit R only if every base of the pair has >= R live latents
         prune     featbsf.cb_prune                      keep the top-R latents by I_m, build the concept set C
stage 4  train     featbsf.cb_train                      concept-bottleneck head (A/B/C alternating Adam)
         finalize  (driver)                              training_config.json for the evaluator
stage 5  eval      vision/branch_metrics.py              R^2, MonoScore, class purity, KNN, LP
         collect   cb_pipeline.py collect                seed mean +- sd -> long CSV
```

`scripts/cb_pipeline.py` runs every stage. It is sequential and resumable: each stage has its own skip-if-output-exists guard, so re-running a command continues where it stopped.

## Install

```bash
pip install -r vision/requirements.txt      # the evaluator and the base models
pip install -r cb_sae/requirements.txt      # CB stages (+ open_clip for asset preparation)
```

## Assets (the only step that needs internet)

```bash
bash cb_sae/prepare_assets.sh
```

The script does three things:

1. **Downloads the two concept vocabularies.** Their upstream repositories carry no license, so they are not redistributed. The script fetches them at pinned commits and checks their sha256:
   - the VLG-CBM ImageNet-filtered concept set (`Trustworthy-ML-Lab/VLG-CBM`, `concept_files/imagenet_filtered.txt`), used as the concept set C_user;
   - the CLIP-Dissect Broden label set (`Trustworthy-ML-Lab/CLIP-dissect`, `data/broden_labels_clean.txt`), used to rank latents by interpretability.
2. **Downloads the supervisor CLIP model** `laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K` (open_clip, ~1.6 GB) into the Hugging Face cache. This is the same checkpoint that extracted the `clip` feature LMDBs.
3. **Computes the text embeddings offline** (`python -m featbsf.cb_text_embeds --embeds`): bare concept strings, L2-normalized, plus the ImageNet class names for the alignment gate.

Everything after this step runs offline (`HF_HUB_OFFLINE=1`).

**The CB pipeline needs the `clip` feature LMDBs even when the substrate is `vit` or `sigclip`.** The concept pseudo-labels y = x_clip · T^T are computed from the CLIP rows of the same images, joined by row index. The probe-image embeddings are those CLIP rows of the evaluation split; no image model is run. The offline `prep` subcommand (run automatically by the drivers) dumps them and runs the row-alignment gate:
- ImageNet-1K: per-index label agreement across the three encoder LMDBs, plus a zero-shot top-1 check that text and image share one space.
- OpenImagesV7: row counts plus a representational-similarity row-correspondence test.

## Paths

All paths are relative to the directory you launch from (the repository root). Each can be overridden with an environment variable:

| Variable | Default | Contents |
|---|---|---|
| `FEATURES_ROOT` | `./features` | feature LMDBs, `<dataset>/<split dir>/<file>` (same layout as `vision/`); `CB_LMDB_PATHS=<json>` overrides individual stores |
| `CB_CKPT_DIR` | `./checkpoints/cb` | converted bases `cvt_*`, pruned bases `pruned_*`, CB runs `cbdual_*` |
| `CB_WORK_DIR` | `./outputs/cb_sae` | embeddings, dissect/prune artifacts, logs, TensorBoard, evaluation results, `bases.json`, `cb_long.csv` |
| `CB_VOCAB_DIR` | `./assets/cb_sae/concepts` | the two downloaded vocabularies |

The main grid's base runs are read from `./checkpoints/main` (`--ckpt-root` / `CKPT_ROOT`). Its Dual-Matryoshka selection is read from `./outputs/main/selection/dual_matryoshka_winners.json` (`--dual-choice` / `DUAL_CHOICE`).

## Running

Full grid: 2 datasets × 3 encoders × seeds 42/43/44 × both arms. This first needs the Matryoshka and Dual-Matryoshka runs of `vision/scripts/main/` and their Dual-Matryoshka selection.

```bash
bash cb_sae/scripts/run_grid.sh                 # the reported operating point: R=1024, k_cb=5, b_cb=1
ALL_ARMS=1 bash cb_sae/scripts/run_grid.sh      # also R in {1024, 2048} x k_cb in {5, 64}
```

`run_grid.sh` runs `cb_pipeline.py bases` (base discovery), then `grid` (all stages), then `collect`, which writes `$CB_WORK_DIR/cb_long.csv`. The CSV has one row per (dataset, encoder, method, metric), with the mean and sd over seeds, the evaluation split, L0, dead fraction and unit count. Method keys:
- `cb_sae` / `cb_dual_sae` at R=1024, k_cb=5;
- other arms carry suffixes, e.g. `cb_dual_sae_R2048_k64`.

Arms are never averaged into one another.

A single cell (both arms of one dataset / encoder / seed):

```bash
bash cb_sae/scripts/run_cell.sh imagenet vit 42
# equivalent:
python cb_sae/scripts/cb_pipeline.py cell --bases ./outputs/cb_sae/bases.json \
    --dataset imagenet --encoder vit --seed 42 --retain 1024 --k-cb 5 --b-cb 1
```

The stage modules can also be called directly (`python -m featbsf.cb_train --help`, with `cb_sae/` on `PYTHONPATH`). Their CLI defaults (`--retain 256`, `--b-cb 4`, `--k-cb` = 64 from the concept set) are **not** the operating point of the reported rows. The drivers always pass `--retain 1024 --b-cb 1 --k-cb 5` explicitly.

Per-stage commands, as issued by the driver:

```bash
python -m featbsf.convert_plain_sae --spec <spec.json> --only-run <run>      # CB-SAE base
python -m featbsf.convert_dualsae   --spec <spec.json> --only-run <run>      # CB-DualSAE base
python -m featbsf.cb_dissect --run <run>
python -m featbsf.cb_prune   --run <run> --cset vlg --retain 1024 --rank interp
python -m featbsf.cb_train   --run <run> --cset vlg --retain 1024 --rank interp \
                             --b-cb 1 --k-cb 5 --seed <seed> [--no-poly]
(cd vision && python branch_metrics.py --run-dir <cb run dir> --eval-split {val|test} --checkpoint best \
    --reps x xM r_hat xhat --retrieval-modes self bank --retrieval-k 50 \
    --encode-batch-size 2048 --knn-k 20 --out-dir <results>)
```

Evaluation splits:
- ImageNet-1K: `val`.
- OpenImagesV7: `test` for R², MonoScore, KNN and LP; `val` for class purity.

`vision/cb_adapter.py` loads a CB run through `bsf.cb_dual_bsf.load_cb_dual`, so the same evaluator that scores every other row of the comparison also scores the CB rows.

### Outputs

- `$CB_CKPT_DIR/cbdual_<base>_R<R>_Cvlg_bcb<b>_kcb<k>[_nopoly]/`: `model.pt` (CB checkpoint), `metrics.json` (holdout statistics in the model's normalized space), `cb_block_stats.npz` (per-unit fire rates), `training_config.json` (written by finalize).
- `$CB_WORK_DIR/cb/prune/<base>__vlg_R<R>/prune.json`: ranking, n_live, prune fidelity, y statistics. `concepts.json` records the concept set. It names the vocabulary by file name plus sha256, so the tree can be moved.
- `$CB_WORK_DIR/cb_retain_plan.json`: which (dataset, encoder) each R admits.
- `$CB_WORK_DIR/branch_metrics/<cb run>__<split>.json`: evaluation results.

### Resumability and the base identity

Every downstream artifact is named after the converted base:
- `cvt_<ds>_<enc>_matryoshka_seed<S>`;
- `cvt_<ds>_<enc>_dual_matryoshka_span<gamma>_warm<W>_seed<S>`.

The dual name includes the decorrelation coefficient and the warm-up length. If the main grid re-selects a different Dual-Matryoshka setting for a cell, its base is converted afresh and the whole CB chain re-runs. Existing artifacts of the previous base are never reused. The driver also checks that an existing conversion was built from the checkpoint it now points at, and refuses otherwise.

This is an orchestration change relative to the original runs. The original runs named the converted dual base without span/warm-up, so a conversion done before the Dual-Matryoshka selection changed was kept. No model code is affected.

## Tests

```bash
python cb_sae/tests/smoke_synthetic.py --out-dir /tmp/cb_smoke --device cpu
```

The smoke test builds tiny synthetic feature LMDBs, stand-in vocabularies and embedding caches, and random-init MatryoshkaSAE / Dual-MatryoshkaSAE bases. It then runs the real driver end to end (bases → prep → convert → dissect → retain → prune → train → finalize → `vision/branch_metrics.py` → collect) for both arms. It also re-runs the grid to check that every stage skips. No downloads; a few minutes on CPU.

## Implementation notes

- **Terminology.** Sparse Branch = "mono", Residual Branch = "poly", decorrelation loss L_dcor with coefficient gamma = "span penalty" / `lam_span` / `span_coef`. CB-SAE's concept encoder/decoder are `E_cb` / `D_cb`. The base runs' conversion folds the SAE's pre-decoder bias into the featurizer's mean (mu ← b_dec, scale = 1), so `E_cb` reads exactly CB-SAE's `v − b`.
- **b=1 featurizer ports.** Each base is converted into a b=1 featurizer (`bsf/sae_models.py`) that reproduces the native model exactly. Every conversion is parity-checked against the native `vision/models` model on 1024 evaluation rows, for both the deployment code and the full reconstruction, before it is written.
- **The Matryoshka base gates with a batch-global top-k**, at evaluation as well as at training. Consequences:
  1. Pruning is not an exact restriction of the parent's codes. Removing competitors promotes retained runners-up. `prune.json` records a per-sample top-k simulation of this effect (`prune_fidelity`); it is a proxy, not the batch-global gate itself.
  2. Codes depend on the encode batch size. The dissect probe pass uses batch 8192, CB training 2048, and evaluation 2048.
- **No steerability score.** CB-SAE ranks latents by interpretability plus steerability. Feature space has no generative model to steer, so ranking here uses interpretability only (min-max-normalized Broden CLIP-Dissect score I_m); this is CB-SAE's own interpretability-only arm. Latents that never fire on the probe split are ranked last.
- **Concept set.** C = C_user \ C_rsae: the VLG-CBM concepts already assigned (by CLIP-Dissect argmax) to some retained latent are removed. The ImageNet-filtered VLG-CBM set is also used for OpenImagesV7.
- **Supervisor.** The supervisor is the open_clip CLIP ViT-L/14 DataComp.XL model. For the `clip` substrate the supervisor image embeddings are the model's own input rows, so the concept targets are an affine function of the input on that encoder.
- **CB head.**
  - The head uses scalar concepts (b_cb = 1) with a per-sample top-k_cb of ReLU(E_cb x), active only in the decoder path.
  - The three CB-SAE objectives (reconstruction; interpretability on the dense concept activations; steerability through the deployment reconstruction) each have their own Adam optimizer and step in A → B → C order on every batch.
  - Optimizer settings: lr 1e-4, eps 1e-8, no gradient clipping.
  - Training: 25 epochs, batch 2048.
  - The cosine-cubed similarity mean-centres each concept column, as in the Label-free-CBM implementation. Eq. 11 of the CB-SAE paper does not show this centring.
- **CB-DualSAE.**
  - The Residual Branch keeps its trained weights from the Dual-Matryoshka base.
  - For the first 2500 steps it is frozen and the decorrelation term is off ("CB-first" warm-up). After that it is trained inside the reconstruction objective, with the decorrelation coefficient 0.01 now applied over both the retained latents' and the concept decoder's directions.
  - The steerability objective always uses the reconstruction without the Residual Branch.
- **Base checkpoint.** The bases are converted from each run's `autoencoder_final.pt` (end of training), as in the original runs.
- **Retain check.** A (dataset, encoder) is admitted at radius R only if every base run of the pair (both arms, all seeds) has at least R live latents. Otherwise that radius has a CB-SAE row without a CB-DualSAE counterpart, or a seed mean over fewer seeds.
- **Defaults vs. reported arm.** The stage CLIs keep their original defaults (`--retain 256 --b-cb 4`, k_cb from the concept set = 64). The drivers pass `--retain 1024 --b-cb 1 --k-cb 5` explicitly, the operating point of the CB rows in the main comparison. `--retain 1024 2048 --k-cb 5 64` selects the further arms that were run.
- **Code layout.** `featbsf/` and `bsf/` keep the module names of the code that produced the runs, so imports and checkpoint formats are unchanged. Only the parts the feature-space CB pipeline uses are included. `bsf/models.py` (the block-sparse featurizer base, Fel et al., arXiv 2606.25234) is kept whole because the checkpoint I/O and pruning dispatch reference its classes.
