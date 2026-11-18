#!/usr/bin/env python3
"""Parameter-matched baselines: each TopK / BatchTopK / JumpReLU baseline of the main grid
retrained with a wider dictionary, so that its number of trainable parameters matches the
DualSAE it is compared with. Every other argument is the baseline cell's own (seed by seed,
from configs/main_cells.json); only ``--n_latents`` changes.

Dictionary sizes (``matched_n_latents``):

* BatchTopK / JumpReLU (separate encoder and decoder): a latent costs 2A+1 parameters (+1 for
  the JumpReLU threshold), the Residual Branch at poly_layers=1 is Linear(A, A) + GELU, i.e.
  A^2 + A parameters, so the match is 8192 + A/2: CLIP (A=768) 8576, ViT (A=1024) 8704,
  SigLIP2 (A=1152) 8768.
* TopK: the baseline is built with TIED weights (the decoder is a view of the encoder), so a
  latent costs only A+1 parameters and parity would need roughly twice the dictionary. The
  matched arm is run at 8704 latents for every encoder (dimension-matched, not
  parameter-matched); ``--verify_params`` prints the resulting counts.

``--verify_params`` instantiates the real model classes and prints the parameter counts of the
baseline at 8192, the baseline at the matched size and the DualSAE at 8192.

Examples
--------
    python scripts/param_matched/train_param_matched.py --verify_params
    python scripts/param_matched/train_param_matched.py --list
    python scripts/param_matched/train_param_matched.py --dataset imagenet --encoder vit
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

VISION = Path(__file__).resolve().parents[2]
CELLS = VISION / "configs" / "main_cells.json"
ENCODER_ACT_SIZE = {"clip": 768, "vit": 1024, "sigclip": 1152}
BASELINE_METHODS = ("topk", "batch_topk", "jumprelu")
DUAL_OF = {"topk": "dual_topk", "batch_topk": "dual_batch_topk", "jumprelu": "dual_jump_relu"}


def matched_n_latents(encoder: str, method: str) -> int:
    if method == "topk":
        return 8704  # dimension-matched for every encoder (see the module docstring)
    return 8192 + ENCODER_ACT_SIZE[encoder] // 2


def exp_name(cell: dict, n: int) -> str:
    return f"pmatch_{cell['dataset']}_{cell['encoder']}_{cell['method']}_n{n}_seed{cell['seed']}"


def matched_cells() -> list:
    out = []
    for c in json.loads(CELLS.read_text())["cells"]:
        if c["method"] not in BASELINE_METHODS:
            continue
        n = matched_n_latents(c["encoder"], c["method"])
        args = list(c["args"]) + ["--n_latents", str(n)]
        if c["method"] == "jumprelu":
            # Value used for the reported parameter-matched JumpReLU runs (theta = 1e-3), which
            # differs from the theta = 1.0 of the JumpReLU baselines at 8192 latents.
            args += ["--jump_threshold_init", "0.001"]
        out.append(dict(c, exp_name=exp_name(c, n), args=args, n_latents=n))
    return out


def verify_params() -> None:
    sys.path.insert(0, str(VISION))
    import torch
    from models.topk_sae import TopKSAE, TopK
    from models.batch_topk_sae import BatchTopKSAE
    from models.jumprelu_sae import JumpReLUSAE
    from models.dual_sae import DualSAE
    torch.set_num_threads(1)

    def cfg(act, dict_size, **kw):
        base = dict(act_size=act, dict_size=dict_size, seed=42, device="cpu", dtype=torch.float32,
                    top_k=64, l1_coeff=1e-4, bandwidth=1e-3, n_batches_to_dead=4882,
                    input_unit_norm=False, aux_penalty=1 / 32, top_k_aux=64)
        base.update(kw)
        return base

    def build(method, act, n):
        if method == "topk":
            return TopKSAE(n_latents=n, n_inputs=act, activation=TopK(64), tied=True, normalize=True)
        if method == "batch_topk":
            return BatchTopKSAE(cfg(act, n))
        if method == "jumprelu":
            return JumpReLUSAE(cfg(act, n))
        return DualSAE(cfg(act, n, sparsity_mode=method[len("dual_"):], poly_hidden=512, poly_layers=1))

    def n_params(m):  # .parameters() de-duplicates the tied encoder/decoder alias
        return sum(p.numel() for p in m.parameters())

    print(f"{'encoder':8} {'method':10} {'n':>5} {'base@8192':>12} {'base@n':>12} {'dual@8192':>12} {'rel.diff':>9}")
    for enc, act in ENCODER_ACT_SIZE.items():
        for m in BASELINE_METHODS:
            n = matched_n_latents(enc, m)
            b0, b1, d = n_params(build(m, act, 8192)), n_params(build(m, act, n)), n_params(build(DUAL_OF[m], act, 8192))
            print(f"{enc:8} {m:10} {n:5d} {b0:12,d} {b1:12,d} {d:12,d} {100 * (b1 - d) / d:+8.3f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["imagenet", "open_images"])
    ap.add_argument("--encoder", choices=sorted(ENCODER_ACT_SIZE))
    ap.add_argument("--method", choices=BASELINE_METHODS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--output_dir", default=os.environ.get("PMATCH_CKPT_DIR", "./checkpoints/param_matched"))
    ap.add_argument("--features_root", default=os.environ.get("FEATURES_ROOT", "./features"))
    ap.add_argument("--python", default=sys.executable)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--print", action="store_true")
    mode.add_argument("--verify_params", action="store_true")
    args = ap.parse_args()
    if args.verify_params:
        verify_params()
        return 0
    out_dir = Path(args.output_dir).resolve()
    for c in matched_cells():
        if any(v is not None and c[k] != v for k, v in
               (("dataset", args.dataset), ("encoder", args.encoder), ("method", args.method), ("seed", args.seed))):
            continue
        d = out_dir / c["exp_name"]
        done = (d / "autoencoder_final.pt").exists() and (d / "best_val_monoscore_model.pt").exists()
        if args.list:
            print(f"{'done' if done else 'todo'}  {c['exp_name']}")
            continue
        if done:
            continue
        cmd = [args.python, str(VISION / "main.py"), *c["args"], "--exp_name", c["exp_name"],
               "--output_dir", str(out_dir), "--features_root", str(Path(args.features_root).resolve()),
               "--skip_existing"]
        if args.print:
            print(" ".join(shlex.quote(x) for x in cmd))
            continue
        print(f"[train] {c['exp_name']}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(VISION))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
