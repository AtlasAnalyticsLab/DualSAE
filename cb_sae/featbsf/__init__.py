"""featbsf — the feature-space CB-SAE / CB-DualSAE pipeline.

Runs on pre-extracted LMDB features (vision/dataset) and on Matryoshka-SAE /
Dual-Matryoshka-SAE checkpoints trained by vision/main.py.

Modules:
    config             paths (env-configurable) and protocol constants
    data               LMDB feature loaders (wraps vision/dataset/lmdb_features.py)
    adapters           rebuilds a vision/ SAE checkpoint from its own saved config
    convert_plain_sae  stage 1: plain SAE checkpoint -> b=1 featurizer (parity-gated)
    convert_dualsae    stage 1: DualSAE checkpoint  -> b=1 featurizer (parity-gated)
    cb_text_embeds     stage 0: concept text embeds, probe image embeds, alignment gate
    cb_config          CB constants, concept sets, artifact naming
    cb_data            paired substrate/supervisor loader, on-the-fly pseudo-labels
    cb_dissect         stage 2: CLIP-Dissect of every latent -> I_m, dead mask
    cb_prune           stage 3: keep the top-R latents, build the concept set
    cb_train           stage 4: concept-bottleneck head (+ Residual Branch) training
"""
