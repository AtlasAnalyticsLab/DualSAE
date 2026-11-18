"""Real-data sanity check on a tiny slice (GPU; needs the downloaded models, the ImageNet images, the
validation feature LMDB and trained checkpoints). For the offline synthetic smoke test see
tests/test_smoke.py. Validates, in order:
  1. re-extraction matches training distribution: max cosine of re-embedded images to the SAE
     training feature LMDB (FEATURES_IMAGENET) ~ 1.0 (proves build_encoder + L2-norm reproduce the
     training embeddings);
  2. SAE loads via old_sae_adapter and encode() yields the expected dict_size + sparsity;
  3. vectorized correlation selection produces per-class latents;
  4. the VLM (InternVL2.5-8B) interprets a few latents from top/zero images and annotates an image.

Writes nothing to the real output tree. Run: python -m hypogen.smoke
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import ImageFolder

from hypogen import config
from hypogen.select_features import correlation_matrix

N_CLASSES = 4
PER_CLASS = 30
ENCODER = "clip"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"== smoke test on {device}; encoder={ENCODER} ==")

    # --- pick a tiny, balanced image slice from the on-disk val ImageFolder ---
    folder = ImageFolder(config.IMAGE_ROOTS["val"])
    by_class = {}
    for path, c in folder.samples:
        if c < N_CLASSES and len(by_class.get(c, [])) < PER_CLASS:
            by_class.setdefault(c, []).append(path)
    paths, labels = [], []
    for c in range(N_CLASSES):
        for p in by_class[c]:
            paths.append(p); labels.append(c)
    labels = np.array(labels)
    print(f"slice: {len(paths)} images over {N_CLASSES} classes")

    # --- re-extract embeddings (the exact training-time encoder) ---
    from prepare_topic_data import build_encoder
    preprocess, embed_fn = build_encoder(ENCODER, config.ENCODER_MODELS[ENCODER], device)
    batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
        emb = embed_fn(batch)
    emb = F.normalize(emb.float(), p=2, dim=-1)

    # (1) validity vs the training LMDB: each re-embedded image should have a near-duplicate row.
    import lmdb
    lp = config._LMDB_VERIFY[(ENCODER, "val")]
    env = lmdb.open(lp, readonly=True, lock=False)
    D = emb.shape[1]
    with env.begin() as txn:
        N = int(txn.get(b"__len__").decode())
        Mlmdb = np.zeros((N, D), dtype=np.float32)
        for i in range(N):
            Mlmdb[i] = np.frombuffer(txn.get(str(i).encode()), dtype=np.float32)
    Mlmdb /= (np.linalg.norm(Mlmdb, axis=1, keepdims=True) + 1e-8)
    sims = emb.cpu().numpy() @ Mlmdb.T
    maxcos = sims.max(axis=1)
    print(f"(1) re-extraction max-cosine to LMDB: mean={maxcos.mean():.4f} min={maxcos.min():.4f} "
          f"-> {'OK' if maxcos.mean() > 0.99 else 'MISMATCH!'}")

    # (2) SAE load + encode
    from old_sae_adapter import load_old_sae
    cell = config.Cell(ENCODER, "topk", "dual", 42)
    sae = load_old_sae(cell.ckpt_path, cell.sae_type, device=device).eval()
    with torch.no_grad():
        z = sae.encode(emb.to(device)).float().cpu().numpy()
    print(f"(2) encode: acts {z.shape}; mean nnz/row={float((z != 0).sum(1).mean()):.1f}")

    # (3) correlation selection
    corr = correlation_matrix(z.astype(np.float32), labels, N_CLASSES)
    sel = {c: np.argsort(-np.abs(corr[c]))[:8].tolist() for c in range(N_CLASSES)}
    print(f"(3) correlation top-8 per class (class0): {sel[0]}")

    # (4) VLM interpret + annotate
    from hypogen.vlm_backend import VLM
    vlm = VLM(seed=0)
    items = []
    probe_latents = sel[0][:3]
    for lat in probe_latents:
        col = z[:, lat]
        hi = np.argsort(-col)[:config.N_HIGH_INTERP]
        lo = np.where(col == 0)[0][:config.N_LOW_INTERP]
        if len(lo) == 0:
            lo = np.argsort(col)[:config.N_LOW_INTERP]
        pos = [Image.open(paths[i]).convert("RGB") for i in hi]
        neg = [Image.open(paths[i]).convert("RGB") for i in lo]
        items.append((pos, neg))
    hyps = vlm.interpret(items)
    for lat, h in zip(probe_latents, hyps):
        print(f"(4a) latent {lat} -> hypothesis: {h!r}")
    if hyps and hyps[0]:
        a = vlm.annotate([(Image.open(paths[0]).convert("RGB"), hyps[0]),
                          (Image.open(paths[-1]).convert("RGB"), hyps[0])])
        print(f"(4b) annotate 2 images with first hypothesis -> {a}")
    print("== smoke test complete ==")


if __name__ == "__main__":
    main()
