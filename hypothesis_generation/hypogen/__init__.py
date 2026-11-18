"""hypogen — Hypothesis Generation with SAEs (vision adaptation of HypotheSAEs, arXiv 2502.04382).

Downstream task comparing each DualSAE with its activation-matched baseline SAE (TopK<->Dual-TopK, etc.):
  Stage 1a  extract_embeddings  re-extract per-(encoder,split) image embeddings + image manifest + labels
            (OpenImagesV7: build_oi_data builds the same files from the pre-extracted LMDBs)
  Stage 1b  dump_activations    load embeddings + an SAE checkpoint -> dense fp16 latent activations
  Stage 2   select_features     per-class one-vs-rest selection -> predictive latents (+ their union)
  Stage 3   interpret_latents   top/zero-activating images -> VLM (InternVL2.5-8B) -> NL hypothesis
  Stage 4   evaluate            proxy (held-out logistic AUROC) + faithful (VLM annotation -> score_hypotheses)
  Stage 5   build_results       per-cell eval CSVs -> tables, one per selection rule (seeds averaged only)

All artifacts are written under config.OUT_ROOT ($HG_OUT, default ./outputs/hypothesis_generation).
"""
