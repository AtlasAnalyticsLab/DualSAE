#!/usr/bin/env python3
"""End-to-end smoke test of the CB-SAE pipeline on synthetic features.

No downloads, no real data. Builds, under --out-dir:

  * tiny row-aligned feature LMDBs for the three encoders (real widths
    768 / 1024 / 1152, a few thousand rows) in the FEATURES_ROOT layout, with
    class-structured CLIP rows so the zero-shot alignment gate can pass;
  * stand-in concept vocabularies + their "text embeddings" in exactly the
    cache format featbsf.cb_text_embeds --embeds writes (that step needs the
    real CLIP weights, so it is the one stage this test replaces);
  * a Matryoshka-SAE and a Dual-Matryoshka-SAE base (vision/models classes,
    random init) saved in the trainer's checkpoint format;

then runs the real driver end to end: bases -> prep -> convert -> dissect
-> retain -> prune -> CB train -> finalize -> eval (vision/branch_metrics.py)
-> collect, for both arms, and checks that every stage produced its outputs.

Usage (CPU, ~a few minutes):
    python cb_sae/tests/smoke_synthetic.py --out-dir <dir> [--device cpu]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import lmdb
import numpy as np

CB_SAE = Path(__file__).resolve().parents[1]
REPO = CB_SAE.parent
DIMS = {"clip": 768, "vit": 1024, "sigclip": 1152}
FILES = {"clip": "clip_image_features_CLIP-ViT-L-14-DataComp.lmdb",
         "vit": "vit_features_vit-large-p16-224.lmdb",
         "sigclip": "siglip2_image_features_siglip2-so400m-p14-384.lmdb"}
SUPERVISOR_TAG = "open_clip:hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"


def write_lmdb(path: Path, x: np.ndarray, labels: np.ndarray) -> None:
    """The raw-bytes layout vision/dataset/lmdb_features.py reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=1 << 31)
    with env.begin(write=True) as txn:
        for i, (row, lab) in enumerate(zip(x.astype(np.float32), labels)):
            txn.put(str(i).encode(), row.tobytes())
            txn.put(f"__label__{i}".encode(), str(int(lab)).encode())
        txn.put(b"__len__", str(len(x)).encode())
        txn.put(b"__shape__", str((x.shape[1],)).encode())
    env.close()


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def make_features(root: Path, rng: np.random.Generator, n_train: int, n_val: int):
    # 1000 class prototypes in the supervisor space; rows use 20 of them so the
    # synthetic clip rows are strongly class-structured (zero-shot gate >= 0.55).
    protos = unit(rng.standard_normal((1000, 768)))
    maps = {enc: rng.standard_normal((768, d)) / np.sqrt(768)
            for enc, d in DIMS.items() if enc != "clip"}
    for split, n in (("train", n_train), ("validation", n_val)):
        lab = rng.integers(0, 20, size=n)
        clip = unit(protos[lab] + 0.8 * unit(rng.standard_normal((n, 768))))
        write_lmdb(root / "imagenet" / split / FILES["clip"], clip, lab)
        for enc, m in maps.items():
            x = clip @ m + 0.05 * rng.standard_normal((n, DIMS[enc]))
            write_lmdb(root / "imagenet" / split / FILES[enc], x, lab)
    return protos


def write_text_cache(embed_dir: Path, name: str, emb: np.ndarray, vocab: Path | None,
                     extra: dict | None = None) -> None:
    """Exactly the cache files cb_text_embeds --embeds writes (npy + sidecar)."""
    embed_dir.mkdir(parents=True, exist_ok=True)
    np.save(embed_dir / f"text_clipL14dc_{name}.npy", emb.astype(np.float16))
    side = {"encoder": SUPERVISOR_TAG, "n": int(emb.shape[0])}
    if vocab is not None:
        side.update(vocab=vocab.name, bare_strings=True,
                    sha16=hashlib.sha256(vocab.read_bytes()).hexdigest()[:16])
    side.update(extra or {})
    (embed_dir / f"text_clipL14dc_{name}.json").write_text(json.dumps(side))


def make_vocab(vocab_dir: Path, embed_dir: Path, protos: np.ndarray,
               rng: np.random.Generator) -> None:
    vocab_dir.mkdir(parents=True, exist_ok=True)
    for stem, n in (("vlgcbm_imagenet_filtered", 300), ("broden_labels_clean", 120)):
        v = vocab_dir / f"{stem}.txt"
        v.write_text("\n".join(f"{stem[:5]} concept {i}" for i in range(n)) + "\n")
        # Half of the concepts are near class prototypes so the CB head has
        # something to align to; the rest are random directions.
        emb = unit(np.concatenate([
            protos[rng.integers(0, 20, n // 2)] + 0.2 * unit(rng.standard_normal((n // 2, 768))),
            rng.standard_normal((n - n // 2, 768))]))
        write_text_cache(embed_dir, stem, emb, v)
    # Class-name embeddings for the zero-shot gate (= the prototypes).
    write_text_cache(embed_dir, "imagenet_classnames", protos, None,
                     {"vocab": "IMAGENET_CLASSNAMES", "template": "a photo of a {c}."})


def make_bases(ckpt_root: Path, d: int, n_lat: int, k: int, seed: int,
               spans_warms=((0.01, 2500),)) -> None:
    """A Matryoshka-SAE and Dual-Matryoshka-SAE base in the trainer format."""
    sys.path.insert(0, str(REPO / "vision"))
    import torch
    from models.matryoshka_sae import MatryoshkaSAE
    from models.dual_sae import DualSAE
    groups = [n_lat // 16, n_lat // 16, n_lat // 8, n_lat // 4]
    groups.append(n_lat - sum(groups))
    common = {"act_size": d, "dict_size": n_lat, "n_batches_to_dead": 5000,
              "seed": seed, "device": "cpu", "dtype": torch.float32,
              "input_unit_norm": False}
    runs = [("matryoshka", None, None)] + [("dual", s, w) for s, w in spans_warms]
    for kind, span, warm in runs:
        if kind == "matryoshka":
            model = MatryoshkaSAE({**common, "top_k": k, "group_sizes": groups,
                                   "l1_coeff": 0.0, "aux_penalty": 1 / 32})
            exp = f"smoke_imagenet_vit_matryoshka_seed{seed}"
            cfg = {"model_type": "matryoshka", "topk_k": k, "normalize": False,
                   "group_fractions": [g / n_lat for g in groups]}
        else:
            model = DualSAE({**common, "top_k": k, "l1_coeff": 1e-4, "bandwidth": 1e-3,
                             "poly_hidden": 512, "poly_layers": 1, "poly_hidden_dims": "",
                             "sparsity_mode": "matryoshka", "group_sizes": groups})
            exp = f"smoke_imagenet_vit_dual_matryoshka_span{span:g}_warm{warm}_seed{seed}"
            cfg = {"model_type": "dual", "sparsity_mode": "matryoshka", "top_k": k,
                   "poly_hidden": 512, "poly_layers": 1, "poly_hidden_dims": None,
                   "span_coef": span, "warmup_steps": warm, "normalize": False,
                   "l1_coef": 1e-4, "bandwidth": 1e-3}
        cfg.update({"n_latents": n_lat, "seed": seed, "batch_size": 2048,
                    "dataset_name": "imagenet", "extractor_name": "vit",
                    "exp_name": exp, "auxk_coef": 1 / 32})
        out = ckpt_root / exp
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": 1, "step": 1, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": {}, "config": cfg},
                   out / "autoencoder_final.pt")
        (out / "training_config.json").write_text(json.dumps(cfg, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=None,
                    help="scratch directory (default: a fresh temporary directory)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--n-val", type=int, default=2048)
    a = ap.parse_args()
    out = Path(a.out_dir or tempfile.mkdtemp(prefix="cb_sae_smoke_")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    env = dict(os.environ, FEATURES_ROOT=str(out / "features"),
               CB_WORK_DIR=str(out / "work"), CB_CKPT_DIR=str(out / "work" / "checkpoints"),
               CB_VOCAB_DIR=str(out / "concepts"),
               OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"),
               # Keep the source tree free of bytecode caches.
               PYTHONDONTWRITEBYTECODE="1")
    protos = make_features(out / "features", rng, a.n_train, a.n_val)
    make_vocab(out / "concepts", out / "work" / "cb" / "embeds", protos, rng)
    make_bases(out / "checkpoints", d=DIMS["vit"], n_lat=512, k=16, seed=42)

    drv = [sys.executable, str(CB_SAE / "scripts" / "cb_pipeline.py")]
    bases = out / "work" / "bases.json"

    def run(argv):
        print("$", " ".join(argv), flush=True)
        subprocess.run(argv, env=env, check=True)

    run(drv + ["bases", "--ckpt-root", str(out / "checkpoints"),
               "--n-latents", "512", "--topk-k", "16", "--out", str(bases)])
    run(drv + ["grid", "--bases", str(bases),
               "--retain", "128", "--k-cb", "5", "--b-cb", "1",
               "--device", a.device, "--num-workers", "0",
               # Testing knobs only: 1 epoch, small batch, short CB-first
               # warm-up so the Residual-Branch phase is exercised too.
               "--cb-epochs", "1", "--cb-batch", "256", "--cb-warmup", "4"])
    # Re-running must be a no-op (every stage skip-guards).
    run(drv + ["grid", "--bases", str(bases), "--retain", "128", "--k-cb", "5",
               "--b-cb", "1", "--device", a.device, "--num-workers", "0",
               "--cb-epochs", "1", "--cb-batch", "256", "--cb-warmup", "4",
               "--stages", "convert", "dissect", "prune", "train", "finalize"])
    csv_path = out / "work" / "cb_long.csv"
    run(drv + ["collect", "--bases", str(bases), "--retain", "128", "--k-cb", "5",
               "--out", str(csv_path)])

    # ---- checks --------------------------------------------------------------
    ck = out / "work" / "checkpoints"
    want = [
        ck / "cvt_imagenet_vit_matryoshka_seed42" / "metrics.json",
        ck / "cvt_imagenet_vit_dual_matryoshka_span0.01_warm2500_seed42" / "metrics.json",
    ]
    for arm, tag in (("cvt_imagenet_vit_matryoshka_seed42", "_nopoly"),
                     ("cvt_imagenet_vit_dual_matryoshka_span0.01_warm2500_seed42", "")):
        run_dir = ck / f"cbdual_{arm}_R128_Cvlg_bcb1_kcb5{tag}"
        want += [run_dir / "model.pt", run_dir / "metrics.json",
                 run_dir / "training_config.json",
                 out / "work" / "branch_metrics" / f"{run_dir.name}__val.json"]
        concepts = json.loads((out / "work" / "cb" / "prune" / f"{arm}__vlg_R128"
                               / "concepts.json").read_text())
        assert not Path(concepts["source"]).is_absolute(), concepts["source"]
    missing = [str(p) for p in want if not p.exists()]
    assert not missing, f"missing outputs: {missing}"
    rows = list(csv.DictReader(open(csv_path)))
    methods = {r["method"] for r in rows}
    # R=128 is a non-main radius, so the collector suffixes it.
    assert methods == {"cb_sae_R128", "cb_dual_sae_R128"}, methods
    assert len(rows) == 10, len(rows)
    for p in want:
        if p.name.endswith("__val.json"):
            d = json.loads(p.read_text())
            assert d["self_checks"]["all_ok"], (p, d["self_checks"])
    print(f"\nSMOKE TEST PASSED ({len(want)} outputs checked) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
