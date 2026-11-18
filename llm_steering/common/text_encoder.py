"""Loading the bge-large document encoder used by the concept classifier (offline).

Same protocol as the document-embedding LMDBs the classifier is trained on: a frozen
SentenceTransformer loaded by LOCAL path, fp32 weights with TF32 matmuls, and the encoder's
max_seq_length pinned explicitly (a checkpoint's own sentence_bert_config.json is not always
the value the embeddings were made with).
"""

from __future__ import annotations

import os

from common import lang_config as config


def resolve_snapshot(repo: str) -> str:
    """Local snapshot directory of ``repo``: ``$A16_BGE_PATH`` if set, else
    <LANGDUAL_ROOT>/raw/models/<org>__<name> (prepare_assets.sh)."""
    env = os.environ.get("A16_BGE_PATH")
    cands = ([env] if env else []) + [config.local_snapshot(repo, "model")]
    for c in cands:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"no local snapshot of {repo} (looked in {cands}); run "
                            f"prepare_assets.sh on a machine with internet first")


def load_encoder(enc: "config.TextEncoder", device: str):
    """A frozen SentenceTransformer, loaded from disk with the protocol knobs pinned."""
    from sentence_transformers import SentenceTransformer  # heavy; imported on demand
    import torch

    # TF32 is the recorded encode dtype (config.ENCODE_DTYPE == "fp32-tf32"); enabled
    # explicitly so the tag is truthful on every GPU.  Callers that also run full-fp32 passes
    # scope this themselves (s7_metrics._bge_precision).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = SentenceTransformer(resolve_snapshot(enc.repo), device=device, local_files_only=True)
    model.max_seq_length = int(enc.max_seq_length)
    model.eval()
    got = int(model.get_sentence_embedding_dimension())
    if got != enc.dim:
        raise ValueError(f"encoder {enc.name} ({enc.repo}) has dim {got}, expected {enc.dim}")
    return model
