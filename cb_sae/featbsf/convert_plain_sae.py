#!/usr/bin/env python3
"""Convert a plain (non-dual) vision/main.py SAE checkpoint into a b=1 BSF
featurizer, so the CB cascade can run on it.

Companion to convert_dualsae.py, which handles the DualSAE side. The CB-SAE
paper builds its concept bottleneck on a Matryoshka Batch-Top-k SAE, so the
CB pair needs BOTH arms converted: the Dual base (convert_dualsae.py) and the
PLAIN base (here).

THE CONVERSION IS EVAL-ONLY AND PARITY-GATED. The port is never trained, and
every conversion is verified on a real validation batch -- deployment code AND
full reconstruction -- against the native model, refusing to write on mismatch.
The point is that cb_dissect/cb_prune/cb_train see the EXACT function the
Matryoshka base was scored with, not a near-miss.

The gate that makes this non-trivial
------------------------------------
vision/models/matryoshka_sae.py (FLAG A1) gates with a **batch-global top-k at
eval as well as at train** (its prefix structure lives only in the loss). The BSF
BatchTopK featurizer's deployment gate is, by default, a per-neuron EMA
threshold. Porting onto the default would produce a model that loads cleanly,
scores plausibly, and is simply a DIFFERENT function from the row above it --
so the port is built with deploy_batch_global=True, and the parity batch proves
it rather than assuming it.

Consequence recorded downstream: a batch-global gate ranks neurons against each
other, so pruning is not an exact restriction (PRUNE_EXACT is False on such an
instance) and cb_prune records the measured retention.

Space contract (the b_dec fold), identical to convert_dualsae's non-LN path:
  mu <- b_dec, scale <- 1, port b_dec <- 0
so normalize(x) = x - b_dec == the native x_cent, E_cb(x_norm) is CB-SAE Eq. 6
exactly, and the standard inversion re-adds b_dec.

Usage (GPU or CPU, seconds per run):
    python -m featbsf.convert_plain_sae --spec <cells.json> [--device cuda]
where <cells.json> is a list of {run, dataset, encoder, seed, variant,
src_ckpt}. Skip-guard: CKPT_DIR/<run>/metrics.json (the driver puts the base
run's identity into <run>, so a different base never reuses a stale port).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

from featbsf import config as C
from featbsf import data as D
from featbsf.adapters import load_any
from featbsf.convert_dualsae import PARITY_ROWS, _parity
from bsf.models import save_bsf, load_bsf
from bsf.sae_models import build_sae

# vision/main.py --model  ->  the b=1 featurizer variant whose DEPLOYMENT gate
# reproduces it. 'matryoshka' maps to batchtopk_sae because Bussmann nesting is
# a LOSS, not a gate: a frozen base never computes the prefix terms, so the
# only thing that must match is the batch-global top-k.
PLAIN_SOURCES = {
    "matryoshka": "batchtopk_sae",
    "batch_topk": "batchtopk_sae",
    "topk": "topk_sae",
    "jumprelu": "jumprelu_sae",
}
# Models whose vision/ eval gate is batch-global rather than per-neuron.
_BATCH_GLOBAL = {"matryoshka", "batch_topk"}


def _build_port(model_type: str, cfg: dict, native) -> torch.nn.Module:
    variant = PLAIN_SOURCES[model_type]
    ln = bool(cfg.get("normalize", False))
    kw = dict(l1_coeff=float(cfg.get("l1_coef", 1e-4) or 1e-4),
              bandwidth=float(cfg.get("bandwidth", 1e-3) or 1e-3))
    if variant == "batchtopk_sae":
        kw = {"deploy_batch_global": model_type in _BATCH_GLOBAL}
    elif variant != "jumprelu_sae":
        kw = {}
    port = build_sae(variant, d=int(native.cfg["act_size"]),
                     n_latent=int(native.cfg["dict_size"]),
                     k=int(native.cfg["top_k"]), **kw)
    if ln:
        raise SystemExit(
            f"model_type={model_type!r} was trained with normalize=True "
            f"(per-sample LN). The plain port has no LN carrier: the b_dec "
            f"fold below assumes an affine global normalization. Convert it "
            f"through the LN path in convert_dualsae.py instead.")
    return port


@torch.no_grad()
def _map_weights(port, native, model_type: str) -> None:
    # MatryoshkaSAE/BatchTopKSAE encode as relu((x - b_dec) @ W_enc + b_enc);
    # the port's _pre_code is relu(enc(x_norm - b_dec)) with its own b_dec
    # zeroed, so enc.bias IS the native b_enc. (convert_dualsae asserts b_enc
    # is zero because DualSAE.forward never uses it -- the opposite is true
    # here: Matryoshka DOES use it, faithfully to the vendored trainer.)
    port.enc.weight.copy_(native.W_enc.t())
    port.enc.bias.copy_(native.b_enc)
    port.D.copy_(native.W_dec.unsqueeze(1))
    port.mu.copy_(native.b_dec)
    port.scale.fill_(1.0)
    port.b_dec.zero_()
    if model_type == "jumprelu":
        port.log_threshold.data.copy_(native.jumprelu.log_threshold.data)
        port.bandwidth = float(native.cfg.get("bandwidth", 1e-3))


def convert_one(cell: dict, device: str) -> dict:
    out_dir = C.CKPT_DIR / cell["run"]
    if (out_dir / "metrics.json").exists():
        return {"run": cell["run"], "skipped": True}

    src = Path(cell["src_ckpt"])
    if not src.exists():
        raise SystemExit(f"{cell['run']}: source checkpoint missing: {src}")
    adapted = load_any(src, device=device, which="final")
    native = adapted.model
    payload = torch.load(src, map_location=device, weights_only=False)
    cfg = payload.get("config", {})
    model_type = cell.get("model_type") or cfg.get("model_type")
    if model_type not in PLAIN_SOURCES:
        raise SystemExit(f"{cell['run']}: model_type {model_type!r} has no "
                         f"plain port (known: {sorted(PLAIN_SOURCES)})")

    port = _build_port(model_type, cfg, native).to(device)
    _map_weights(port, native, model_type)

    # Parity on the eval split, unshuffled: real data, deterministic, and at
    # PARITY_ROWS so a batch-global gate sees a realistic budget.
    loader = D.build_loader(cell["dataset"], cell["encoder"],
                            C.EVAL_SPLIT[cell["dataset"]],
                            batch_size=PARITY_ROWS, labels="none",
                            shuffle=False, num_workers=2)
    x = next(iter(loader)).to(device)
    stats = _parity(port, native, x)
    if not stats["parity_ok"]:
        raise SystemExit(f"{cell['run']}: PARITY FAILED {stats} — refusing to "
                         f"write a checkpoint that does not reproduce the "
                         f"source model")

    out_dir.mkdir(parents=True, exist_ok=True)
    extra = {"dataset": cell["dataset"], "extractor": cell["encoder"],
             "seed": cell["seed"], "converted": True,
             "parent_run": src.parent.name, "model_type": model_type}
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    os.close(fd)
    save_bsf(port.cpu(), tmp, extra=extra)
    os.replace(tmp, out_dir / "model.pt")
    # Round-trip gate: the payload must rebuild into the same function, which
    # is the extra_config/build_from_payload path every CB stage depends on.
    rebuilt = load_bsf(out_dir / "model.pt", device)
    rt = _parity(rebuilt, native, x)
    if not rt["parity_ok"]:
        raise SystemExit(f"{cell['run']}: ROUND-TRIP PARITY FAILED {rt}")
    (out_dir / "metrics.json").write_text(json.dumps(
        {"run": cell["run"], "parity": stats, "roundtrip": rt,
         "model_type": model_type, "extra": extra}, indent=1))
    return {"run": cell["run"], "skipped": False, **stats}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True, type=Path,
                    help="JSON list of cells: run/dataset/encoder/seed/src_ckpt")
    ap.add_argument("--only-run", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cells = json.loads(args.spec.read_text())
    if args.only_run:
        cells = [c for c in cells if c["run"] == args.only_run]
    n_done = 0
    for cell in cells:
        res = convert_one(cell, args.device)
        tag = "skip" if res.get("skipped") else "ok"
        print(f"[{tag}] {cell['run']}"
              + ("" if res.get("skipped")
                 else f"  max|dz|={res['parity_max_abs_z']:.2e}"
                      f"  swaps={res['parity_n_swapped']}"
                      f"  max|dx|={res['parity_max_abs_recon']:.2e}"))
        n_done += 0 if res.get("skipped") else 1
    print(f"converted {n_done}/{len(cells)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
