# Steering protocol (pre-registered analysis and decision identifiers)

This file fixes how the steering experiment is run and analysed. The identifiers used in the
code comments (`D-xx`, `Cxx`, `Rx`, `Px`, section numbers such as `PROTOCOL 1.5`) are defined
here. Sections 2 and 3 of the original protocol concern other downstream applications and are
not part of this component; their numbers are kept free so that the section numbers cited in
the code stay valid.

Terms: **Track B** = SAEs trained on the residual stream of `EleutherAI/pythia-410m-deduped`
(output of block 6, 12 or 18) over OpenWebText (`owt`); **Track A** = SAEs on sentence-encoder
document embeddings (the steering pipeline only reuses Track A's bge-large document embeddings
of AG News and GoEmotions to train its concept classifier). In code, DualSAE's Sparse Branch is
the *mono* branch, its Residual Branch the *poly* branch, and the decorrelation coefficient
gamma is `span_coef`.

## 1. Shared rules

### 1.1 Pairs, arms and contrasts

- Pairs: TopK, BatchTopK, JumpReLU. Never compared across activations.
- Arms per pair: `baseline` (traditional SAE), `dual` (DualSAE), `dual_g0` (DualSAE with gamma =
  0). Run directories come from `common/arms.py`.

| Contrast | delta | Question |
|---|---|---|
| Dual-baseline | mean(dual) - mean(baseline) | Does DualSAE steer better than the traditional SAE of the same activation? |
| Dual-g0 | mean(dual) - mean(dual_g0) | Does the decorrelation loss itself matter? |

dual_g0 vs baseline is always exploratory.

### 1.2 Unit, test and multiplicity

- Unit of replication: one trained SAE. Seeds 42, 43 and 44 for every arm, so every contrast is
  3 + 3.
- Per-seed value: the metric computed on that SAE. Items that define the metric are averaged
  first (D-01): prompts, continuations, and the 3 latents of a concept.
- Test: unpaired Welch t (sample sd, ddof=1), two-sided, alpha 0.05.
- Printed for every test: delta, its 95% CI (Welch-Satterthwaite df), raw p, BH q, and the MDE80
  (exact noncentral t; about 3.07 x sd at 3+3).
- Multiplicity: Benjamini-Hochberg over all tests of the family (cells x metrics x contrasts)
  that remain after the health gates (1.5). A test is significant iff q < 0.10.
- Direction: every primary metric is higher-is-better, so delta > 0 favours the Dual.

### 1.3 Smallest effects of interest (SESOI)

| Metric | Scale | SESOI | Role |
|---|---|---|---|
| AxBench harmonic mean | 0-2 | 0.10 | co-primary |
| bge classifier P(concept) | 0-1 | 0.05 | co-primary |
| detection AUROC | 0-1 | 0.01 | exploratory (no verdict word) |

### 1.4 Verdict words

Each confirmatory test gets exactly one verdict: the first step whose condition holds.

| Step | Condition | Verdict |
|---|---|---|
| 1 | Removed by a health gate (1.5) | No verdict. Per-seed rows with the gate's label. |
| 2 | q < 0.10 and delta > 0 | "Dual better than <comparator>" |
| 2 | q < 0.10 and delta < 0 | "Dual worse than <comparator>" |
| 3 | q >= 0.10 and the 95% CI lies inside (-SESOI, +SESOI) | "equivalent" |
| 4 | Otherwise | "inconclusive (MDE80 = x)" |

- If step 2 applies and the 95% CI also lies inside +-SESOI, the verdict adds "(below SESOI)".
- Forbidden: "no effect", "parity", or any claim of sameness drawn from p > 0.05.
- Every win count prints n_tests and 0.05 x n_tests.
- Exploratory results print delta, CI and raw p only. No verdict words, no BH.

### 1.5 Health gates that remove tests from the family

Gates are evaluated from training logs and rescore outputs before any family metric is read.
All removals happen before BH.

| Gate | Applies to | Fails when | What leaves the family | Label |
|---|---|---|---|---|
| D-15 run gate | every JumpReLU run (all 3 arms) | final log_threshold mean <= ln(init) - 1, or eval L0 outside [16, 512] | baseline or dual fails: the whole pair; only dual_g0 fails: the Dual-g0 contrast | "not sparsity-matched" |
| D-15 L0 match | JumpReLU pairs | baseline and dual L0 differ by more than 2x | the whole pair | "not sparsity-matched" |
| D-15 L0 match, g0 | JumpReLU pairs | dual and dual_g0 L0 differ by more than 2x | the Dual-g0 contrast | "not sparsity-matched" |
| D-43 health | every arm and seed, at positions >= 1 | (a) a top-k arm's final val Active < 57.6 (0.9k); (b) final val reconstruction loss > 1.5x the run's own minimum over epochs; (c) CE recovered < 0.80, or a Dual/g0 arm below (pair baseline - 0.10); (D-44) a BatchTopK arm's inference L0 on non-sink test rows < 57.6 | the whole (layer, pair); the failing SAE is not calibrated, generated or judged | "collapsed" ("baseline collapsed" for a failing TopK baseline, D-18) |
| outlier flag | every confirmatory test | a seed has abs(x_i - mean(other seeds)) > 4 x sd_ref(metric, layer), sd_ref = pooled within-arm seed sd over all cells of that metric (one mode, selection and budget) | the contrasts that use the flagged arm, for that cell and metric | "not assessable" |

Two versions of the D-43 gate are computed (`steering/gates43.py`):
- **primary** (the headline, and the gate the cascade runs under): removal by (a), (c) and D-44;
  rule (b) is computed and reported as a diagnostic on every record but removes nothing;
- **strict**: (a), (b), (c) and D-44 all remove. Reported beside every confirmatory table as a
  sensitivity analysis, with BH computed separately within that version's family.

The JumpReLU pairs are held (not steered) unless `A16_ALLOW_JUMPRELU=1`; held family cells are
listed with the label "not sparsity-matched (D-15 pilot failed; cells held)".

### 1.6 Attribution rule

| Dual-baseline verdict | Dual-g0 verdict (same cell) | Statement |
|---|---|---|
| Dual better | Dual better | The Dual advantage is attributed to the decorrelation loss. |
| Dual better | q >= 0.10 | Joint dense-residual training effect, not attributed to the decorrelation loss. |
| Dual better | removed by a gate, or labelled | Dual advantage; attribution not assessable. |
| any other combination | any | Both verdicts are printed as they are, with no attribution sentence. |

### 1.7 Everything else is exploratory

Every result outside the family of 4.1 is exploratory: dual_g0 vs baseline, modes (ii') and
(iii), every non-primary budget, the live-matched selection, and every extra column.

## 4. Steering family

### 4.1 Confirmatory family

| Axis | Levels |
|---|---|
| Layer | 6, 12, 18 |
| Concept | the 10 concepts of 4.2 |
| Mode | (i) direction-add of the unit decoder atom (D-47) |
| KL budget | the primary budget P1 (one value, used at all 3 layers, R7) |
| Metric | AxBench harmonic mean (0-2); bge classifier P(concept) (0-1) |
| Pair | TopK (baseline = untied TopK), BatchTopK, JumpReLU |
| Contrast | Dual-baseline, Dual-g0 |
| Items | 60 neutral prompts x 2 continuations per selected latent (D-49) |

Maximum size: 3 layers x 10 concepts x 2 metrics x 3 pairs x 2 contrasts = 360 tests.

### 4.2 Concepts (D-45)

| Corpus | Concepts |
|---|---|
| amazon (canrager/amazon_reviews_mcauley_1and5) | positive (5-star), negative (1-star) |
| agnews | World, Sports, Business, Sci/Tech |
| goemotions | joy, anger, sadness, fear |

Per concept class: 2,000 train documents for selection and 1,000 test documents, fixed seed,
sampled after the amazon shuffle (capped by availability for GoEmotions single-label rows).

### 4.3 Latent selection and aggregation (D-46)

- Per SAE and concept: the AUROC of each latent's per-token code, mean-pooled over positions
  >= 1, for concept vs rest (the other classes of the concept's corpus).
- Excluded before ranking: sink latents (fire at position 0 in > 50% of windows), always-on
  latents (fire rate > 0.3) and dead latents.
- Selected: the top 3 latents by AUROC (DualSAE: latents of the Sparse Branch).
- Per-latent score: the metric averaged over all its items at the primary budget.
- Per-seed value for a concept: the MEAN of its reachable latents' scores. Never best-of-3.

### 4.4 Primary metrics

- AxBench harmonic mean: the official harmonic mean of the concept, instruction (= prompt) and
  fluency ratings, 0 if any rating is 0. phi-4 judge, official AxBench templates and parser,
  unparsable ratings = 0.0 (D-50).
- bge classifier P(concept): the concept classifier's probability of the target concept for
  each continuation (D-50).

### 4.5 Claim limits (D-47)

- Mode (i) supports one claim only: "at equal next-token KL, Dual's Sparse-Branch decoder rows
  are / are not better concept directions than the baseline's".
- No claim that the Sparse Branch carries the concept-causal part of the residual stream.
- Dual vs dual_g0 in mode (i) is the only contrast that isolates the decorrelation term.

### 4.6 If no budget qualifies

If P1 is "none", the family is empty; the report states that the headline is infeasible at
budgets {0.05, 0.1, 0.2, 0.4, 0.8, 1.6} nats and every steering result is exploratory.

### 4.7 Exploratory

Modes (ii') and (iii), the analytic cancellation and bypass-regime rows, every non-primary
budget, the live-matched selection row, detection AUROC, the concept / instruction / fluency
means as separate columns, the sensitivity row with parse failures dropped, perplexity,
distinct-2 and next-token KL, the DiffMean / random / unsteered reference rows, the second judge
and hand-label kappa, and dual_g0 vs baseline.

## 5. Rules

- R1 Verdict precedence: step 2 is checked before step 3; a significant result whose CI lies
  inside +-SESOI adds "(below SESOI)" (1.4).
- R2 (other application; not used here).
- R3 Removal scope of each gate: D-43 removes the (layer, pair). D-15 removes the pair when
  baseline or dual fails, and only the Dual-g0 contrast when dual_g0 fails or the dual/dual_g0 L0
  ratio exceeds 2x. The outlier flag removes the contrasts that use the flagged arm (1.5).
- R4 Application-specific labels keep the test in BH's m.
- R5 Unreachable latents: a selected latent that cannot reach the primary budget in mode (i)
  within the D-48 cap contributes no score; the concept's value is the mean over its reachable
  latents and the unreachable count is printed. If none is reachable the seed is missing for
  that concept. A test with fewer than 2 seeds in either arm is "not assessable".
- R6 The 20-prompt set used by the D-48 pilot and by non-headline cells is the first 20 lines of
  the P2 prompt file.
- R7 The primary budget found at block 12 applies to all three layers.
- R8 (other application; not used here).

## 6. Fixed inputs

- **P1** primary KL budget: decided by the arm-free block-12 pilot (D-48) and written by
  `python -m steering.s4_pilot --rule` to `$A16_ROOT/s4/p1_primary_budget.json` before any
  SAE-arm generation. A decided P1 is never re-written. Rule: the smallest budget in
  {0.05, 0.1, 0.2, 0.4, 0.8, 1.6} at which DiffMean's mean concept rating is >= 0.5 while the
  random directions' mean concept rating is <= 0.2 (mean over every (concept, direction, prompt)
  item of that family; unreachable directions contribute no items).
- **P2** neutral prompt file: `data/neutral_prompts.txt` (60 prompts, each <= 16 pythia tokens,
  free of concept words), sha256 `1851bf6910d4967b8c7cc0468a45d2d4222c38eee145f4b30b941cdd73e67d80`
  (pinned in `steering/base.py`; every generation stage refuses another file).
- P3 (other application; not used here).

## 7. Decision identifiers cited in the code

- **D-01** Average over seeds only (mean +- sample sd, ddof=1); every other axis (layer, mode,
  budget, concept, metric) stays a separate row/column. Items that define a metric (prompts,
  continuations, the 3 latents of a concept) are averaged inside the metric.
- **D-03** Per-cell tests and pre-registration of the analysis (this file).
- **D-04** Every result row carries: checkpoint name, best_epoch (true best-val epoch), eval L0,
  dead % and live-latent count, computed from the final-checkpoint codes at positions >= 1
  (BatchTopK: D-44 encoder).
- **D-10** Every arm is scored at `autoencoder_final.pt` (step 31,250), strict load;
  `best_val_monoscore_model.pt` only lends its `model_cfg`.
- **D-11** Re-seed every RNG explicitly AFTER all model loads (`BaseAutoencoder.__init__` calls
  `torch.manual_seed`); loads are made RNG-neutral.
- **D-15** JumpReLU sparsity-match gate (1.5; `common/gates.py`).
- **D-17** Dual vs dual_g0 contrasts require both arms trained by the same code version.
- **D-18** Track B arms: the TopK baseline is the UNTIED TopK SAE (`--no_tied_weights`, with the
  tied initialisation encoder = decoder^T at step 0); Dual-TopK / Dual-BatchTopK and their
  gamma = 0 twins use a 1,250-step warm-up; the BatchTopK baseline is the main language grid's.
  A layer whose TopK baseline fails D-43 is reported as "baseline collapsed" and its TopK pair
  is not steered.
- **D-19** SAEs are trained on all positions; position 0 (the attention sink of packed windows)
  is masked at INFERENCE only (selection, steering, KL calibration). Rows report non-sink and
  sink FVU and the excluded-latent fractions.
- **D-20** The judge is a pinned phi-4 snapshot (revision `932b33c0...`), loaded by local path.
- **D-21** Batch-safe encoding: TopK/JumpReLU per row (fixed GEMM shape); BatchTopK in batches of
  exactly 2,048 rows over a seeded permutation, then un-permuted (both arms see the same
  batches).
- **D-22** Every derived seed = `int.from_bytes(sha256(f'{base}|{key}').digest()[:4], 'little')`;
  Python `hash()` is never used.
- **D-43** Health gate of the steered SAEs (1.5), applied per arm and seed at the final
  checkpoint before any selection, calibration or generation.
- **D-44** BatchTopK inference: a frozen threshold, the EMA (beta 0.999) of the per-batch minimum
  positive activation of the model's own batch-top-k code, run post hoc over >= 5,000 train
  batches of 2,048 NON-SINK rows (both arms of a pair see the same batches). Tokens are then
  encoded independently. Each row reports the inference L0 and the training-time non-sink L0.
- **D-45** Concepts and concept data (4.2): documents tokenised to <= 128 tokens; tokens at
  positions >= 1 are encoded per token and the codes mean-pooled (a pooled residual is never
  encoded).
- **D-46** Latent selection (4.3), plus a live-matched control: n_match = the minimum number of
  eligible latents over the 9 SAEs of a (layer, pair); each SAE draws one seeded subset of
  n_match eligible latents and runs the same top-3 selection in it (judge-free metrics only).
- **D-47** Directions and modes, edits only at positions with position_ids >= 1 and mask 1:
  (i) direction-add x' = x + alpha * nf * u (u = unit decoder row; also the DiffMean direction
  and random unit directions); (ii') error-preserving clamp z'_j = c, x' = x + nf * (c - z_j) *
  W_dec[j]; (iii) Dual full re-run x'_s = xM(z') + poly(x_s - xM(z')) + (x_s - xhat_s)
  (diagnostic). The analytic poly cancellation is computed for every live Dual latent; a Dual
  with xM FVU >= 1 is labelled "bypass regime" and its mode (iii) rows are a bypass diagnostic.
- **D-48** Strength calibration: per (SAE, direction, mode), alpha (or c) is set so that the
  teacher-forced next-token KL(clean || edited) on held-out OpenWebText windows equals each
  budget in {0.05, 0.1, 0.2, 0.4, 0.8, 1.6} nats; the arm-free pilot fixes P1 (6).
- **D-49** Generation: headline cells (mode (i), P1, all arms, incl. the live-matched row) use
  60 prompts x 2 continuations, every other cell 20 x 1; an unsteered null row; 48 new tokens in
  fp32 with KV cache and left padding; sampling T = 0.7, top-k 50 (the default of HF
  `generate()` for a model without a generation config), top-p 0.9; per-item seeds from
  (mode, budget, concept, rank, prompt) only, so all arms share random numbers; an identity-hook
  test precedes generation.
- **D-50** Grading (4.4): phi-4 with the official AxBench judge templates and parser
  (stanfordnlp/axbench @ `41c8332`), three single-item prompts at T = 0 (max 768 tokens, one
  re-ask at 1536 if truncated), arm-blinded; the bge-large concept classifier (logistic
  regression, class_weight='balanced', on L2-normalised bge vectors); perplexity under clean
  pythia, distinct-2 and next-token KL. Every teacher-forced pass uses position_ids =
  cumsum(mask) - 1.
- **D-51** Stage organisation: per-cell 1:1 stages are merged (s7m inside s6); many-to-one stages
  start only when every input is on disk.

Implementation clarifications cited as `Cxx`:
- **C5** The judge sees only the concept description, the prompt and the continuation.
- **C7** Latent eligibility = live, non-sink, not always-on (D-46).
- **C17** The primary budget is fixed by the arm-free pilot before any SAE-arm generation (P1).
- **C18** Report sink and non-sink reconstruction separately (D-19).
- **C19** Identity-splice precondition before generation (D-49).
- **C20** Bypass-regime labelling of mode (iii) (D-47).
- **C29** Calibration search: log-grid scan in units of the layer's median non-sink token norm,
  best bracket, 8 bisection steps, cap 1e3 x that norm, "unreachable" recorded (never looped).
- **C30** Explicit position_ids and the 2-D padding mask for every edited or teacher-forced pass.
- **C31** Encode tokens, never pooled residuals; save only AUROC tables and selections.
