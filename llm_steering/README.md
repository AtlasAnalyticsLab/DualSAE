# LLM steering with SAEs and DualSAEs (Pythia-410M)

This component steers `EleutherAI/pythia-410m-deduped` with single latents of residual-stream
SAEs and DualSAEs and grades the continuations. It is the code of the paper's **LLM steering**
experiments. Protocol and analysis: [PROTOCOL.md](PROTOCOL.md).

Terms: DualSAE's **Sparse Branch** is called `mono` in the code, its **Residual Branch** `poly`,
and the decorrelation loss L_dcor coefficient gamma is `span_coef`. Track B = SAEs on the Pythia
residual stream (OpenWebText); Track A = SAEs on document embeddings (only their bge-large
document embeddings of AG News / GoEmotions are reused here).

## What it covers

| Paper item | Command (after the cascade has run) |
|---|---|
| Supplementary, "LLM steering": table "Quantitative LLM steering on Pythia" | `python -m steering.make_table` (headline, primary health gate); `python -m steering.make_table --gate strict` (sensitivity) |
| Supplementary, "LLM steering": the statistics stated with that table (Welch per contrast, BH, verdict words) | `python -m steering.s9_report` -> `s9/tests.csv`, `s9/report.md` |
| Main text, "Qualitative analysis on language representations", panel (b) "Positive-sentiment steering", and the supplementary figures "Steering toward Sadness / Fear / Sports / Anger" | `python -m steering.select_examples --direction dual` (and `--direction baseline` for the mirrored pool) |

`run_all.sh` runs everything, including these three commands, from the trained SAEs.

### Example-pool selection rule

`select_examples.py` matches the two arms of a pair (DualSAE vs the traditional SAE of the same
activation, block and seed) on the same concept, latent rank, prompt and continuation index
(hence the same sampling random numbers), at mode (i), the `full` selection and the primary
budget P1, for pairs that pass the primary health gate. An item is a clear win for the arm named
by `--direction` when

- the winner's phi-4 concept rating is 2 and the loser's is 0, and
- P_winner(concept) - P_loser(concept) >= 0.5 (bge classifier), and
- the winner's fluency rating is >= 1.

Both directions use the same rule, and each run prints the size of both pools. Every record
keeps the block, activation, seed, run directories, latent ids, all ratings, and the full
continuations (printed verbatim in the `.md`).

## Install

```bash
pip install -r requirements.txt        # the versions the reported runs used (Python 3.11)
```

The SAE/DualSAE model classes are imported from `../vision` (no install needed).

## Assets and inputs

1. **Downloads** (once, on a machine with internet):

   ```bash
   export LANGDUAL_ROOT=/path/to/language/data      # the language pipeline's data root
   bash prepare_assets.sh                           # add --with-second-judge for InternVL3.5-8B
   ```

   Pythia-410M-deduped, bge-large-en-v1.5, phi-4 (judge), the official AxBench judge files,
   and the Amazon / AG News / GoEmotions dataset snapshots, all at pinned revisions (see
   `prepare_assets.py`), under `$LANGDUAL_ROOT/raw/`.

2. **Language-pipeline outputs** (produced by `../language`; same `$LANGDUAL_ROOT` layout,
   documented in `common/lang_config.py`):
   - trained SAE run directories (`autoencoder_final.pt`, `best_val_monoscore_model.pt`,
     `training_config.json`, `train_stdout.log`, `CELL_DONE.json`) for the arms listed in
     `common/arms.py`: blocks 6/12/18, seeds 42-44, TopK (untied baseline, DualSAE with
     warm-up 1250, DualSAE with gamma = 0), BatchTopK (baseline, DualSAE warm-up 1250,
     gamma = 0), and the JumpReLU arms (held by default);
   - OpenWebText residual shards of blocks 6, 12 and 18 (`resid/L<n>/{train,test}`, with their
     `manifest.json` and norm factor) and the packed token streams (`tokens/{train,test}.u16.npy`);
   - the materialised AG News / GoEmotions corpora (`corpora/`) and their bge-large document
     embeddings (`lmdb/bge/`).

   `python -m common.arms --summary` lists every arm and whether its files are present.

Runtime is fully offline (`env.sh` sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` and
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, the in-process vLLM engine).

## Running

```bash
export LANGDUAL_ROOT=/path/to/language/data
export A16_ROOT=/path/to/steering/outputs           # default ./outputs/llm_steering
bash run_all.sh                                     # full cascade, sequential, resumable
```

`run_all.sh` asks the planner which units are runnable, runs them one by one (one GPU), and
repeats until nothing is left; a killed run resumes at the next unfinished unit. Stages:

| Stage | Module | Unit |
|---|---|---|
| s0 | `steering.s0_rescore` | SAE: final-checkpoint rescore, D-43 health measurements, D-44 BatchTopK threshold |
| s1 | `steering.s1_select` | layer: concept data, per-latent AUROC selection, DiffMean and random directions |
| s2 | `steering.s2_cancel` | Dual SAE: analytic Residual-Branch cancellation (diagnostic) |
| s4a / s4j / s4p | `steering.s4_pilot --generate`, `steering.s7_judge --pilot`, `steering.s4_pilot --rule` | arm-free block-12 pilot that fixes the primary KL budget P1 |
| s5 | `steering.calibrate` | SAE / arm-free unit: KL-budget calibration |
| s6 | `steering.s6_generate` | generation unit: steered generation + judge-free metrics |
| s7c | `steering.s7_metrics --train-clf` | bge concept classifiers |
| s7j | `steering.s7_judge` | generation unit: phi-4 AxBench judging |
| s8 (optional) | `steering.s8_handlabel` | hand-label sheet and InternVL second judge (`A16_SECOND_JUDGE=1`) |
| s9 | `steering.s9_report` | report: per-seed table, tests, verdicts |

Single unit: `bash run_stage.sh s6 12` (unit list: `python -m steering.jobs --kind s6 --list`).
Progress: `python -m steering.plan --summary`. A batch-scheduler wrapper with placeholders is in
`scripts/slurm_template.sh` (array index = unit index).

Defaults reproduce the configuration of the reported runs; no argument needs to be set.

## Outputs (`$A16_ROOT`)

`s0/` rescore records, gate JSONs (`gates/d43*.json`); `s1/` selections; `s4/` pilot and
`p1_primary_budget.json`; `s5/` calibration; `s6/<unit>/` generations; `s7/judge`,
`s7/metrics`, `s7/clf`; `s9/per_seed.csv`, `s9/tests.csv`, `s9/tests_strict.csv`,
`s9/report.md`, `s9/steering_table{,_strict}.{tex,md,json}`,
`s9/examples_{dual,baseline}_wins.{json,md}`.

## Tests

```bash
python tests/smoke_test.py          # whole cascade on synthetic data, CPU, no downloads
python -m common.stats --selftest; python -m common.seeds --selftest
python -m common.gates --selftest; python -m steering.gates43 --selftest
python -m steering.jobs --selftest; python -m steering.s9_report --selftest
```

The smoke test builds a tiny random GPT-NeoX, synthetic corpora and randomly initialised SAE
checkpoints, stubs the judge and the bge encoder, relaxes the health-gate thresholds, and runs
every stage at block 12 for TopK and BatchTopK, then the table builder and the example selection.

## Implementation notes

- Every arm is scored at its final checkpoint (`autoencoder_final.pt`); the TopK baseline is the
  untied-weight TopK SAE; the Dual arms use a 1,250-step warm-up; `dual_g0` is DualSAE with
  gamma = 0. The JumpReLU pairs are held by default (they were not steered in the reported runs).
- Steering mode (i) adds `alpha * nf * u` to the output of block L at every position except the
  first, where `u` is the unit-norm decoder row of the latent (for DualSAE, a Sparse-Branch row),
  `nf` the shard norm factor, and `alpha` the value that gives the target next-token KL on 32
  held-out OpenWebText windows. Position 0 (the attention sink of packed windows) is never
  edited, and is excluded from selection statistics; SAEs were trained on all positions.
- BatchTopK SAEs (baseline and Dual) are encoded at inference with a frozen threshold estimated
  post hoc over 5,000 batches of 2,048 non-sink training rows, not with batch top-k.
- Sampling reproduces HF `generate()` for a model without a generation config: temperature 0.7,
  then top-k 50, then top-p 0.9; every continuation draws from its own seeded generator keyed by
  (mode, budget, concept, rank, prompt), so all arms share random numbers. The unsteered
  continuation has its own key.
- The judge is phi-4 with the official AxBench templates and rating parser; the instruction
  slot holds the neutral prompt (Pythia is a base model).
- The health gate exists in two versions: the primary gate (rules (a), (c) and the BatchTopK L0
  rule remove; rule (b) is a diagnostic) drives the cascade and the headline; the strict gate
  (all four rules) is reported as a sensitivity analysis.
- `make_table.py` labels each column from the `arm` field of the rows it aggregates and computes
  every marker from `tests.csv`; pairs removed by a gate are listed in the caption with their
  label.
- The calibration KL is accumulated in chunks of 8 windows, so the number of calibration windows
  must be a multiple of 8.
- The original pipeline also refused to run a stage until its protocol entries were recorded in
  a pre-registration file; that guard is not part of this release (PROTOCOL.md holds the
  protocol).
- The hard-coded residual width 1024 of the original code is read from
  `lang_config.LM_D_MODEL` (1024 unless overridden for the smoke test).
