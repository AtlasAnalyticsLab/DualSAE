"""Shared helpers of the steering pipeline (vendored from the language application layer).

  lang_config  paths of the language pipeline's data tree (LANGDUAL_ROOT) and its constants.
  arms         registry mapping (layer, activation, arm, seed) to a trained run directory, and
               the contrast-level API (contrast_ref) the report builds Dual-vs-comparator rows
               from.
  gates        the D-15 JumpReLU sparsity-match gate (per-run records, per-contrast decision).
  sae_load     rebuild a trained SAE / DualSAE from its run directory (models from <repo>/vision).
  sae_io       ``load_arm`` (final checkpoint, D-10) and the batch-safe encoders:
               ``encode_rows`` (D-21) and the frozen BatchTopK inference threshold (D-44).
  lm_model     local loading of pythia-410m-deduped.
  text_encoder local loading of bge-large-en-v1.5 (concept classifier).
  loss_recovered_hooks  hook/splice primitives, the shard manifest and its norm factor.
  llm_backend  ``TextLLM``: the in-process phi-4 judge engine (vLLM), with the length guard, the
               per-process JSONL cache and the official AxBench rating parser.
  seeds        D-22 stable seed derivation (sha256, never Python ``hash()``) and RNG seeding.
  stats        Welch per cell, BH q, verdict words, the global-scale outlier flag.

Nothing is imported eagerly here: ``llm_backend`` must control when ``vllm`` is imported
(VLLM_ENABLE_V1_MULTIPROCESSING has to be set first), and ``sae_load`` edits ``sys.path``.  So
each consumer imports the module it needs explicitly.
"""
