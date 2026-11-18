"""langdual -- SAEs and DualSAE on the residual stream of Pythia-410m-deduped.

Modules:
  config               registry of the main grid (cells, tags, paths, protocol constants)
  ablate               registry of the ablation cells (warm-up grid, steering SAEs)
  jobs                 cell -> worker-environment decoder used by train_cell.sh
  assets               pinned download descriptors (prepare_assets.sh)
  materialize_corpora  corpus -> packed token streams; labelled corpora -> jsonl + labels
  extract_text_embeddings  bge-large document embeddings of the labelled corpora (LMDB)
  lmsae                activation extraction, the activation-shard dataset, CE/KL splicing,
                       and the evaluator
  qualitative          top-activating context pages and per-token activation maps

Training and scoring reuse the SAE/DualSAE core in <repo>/vision (main.py, trainers, models,
knn_lp_sparse.load_run, branch_metrics), which reads the activation shards through
langdual.lmsae.resid_shards.
"""
