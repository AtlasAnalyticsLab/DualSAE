#!/usr/bin/env python3
"""Convert vision/main.py DualSAE checkpoints into b=1 DualSAEFeaturizer bases
for the CB cascade (CB-DualSAE).

Each listed DualSAE checkpoint (autoencoder_final.pt) becomes a save_bsf
payload under CKPT_DIR/<port_run>/model.pt (+ metrics.json, the CB stages'
base guard); <port_run> and the source checkpoint come from a --spec JSON
written by the driver (cb_sae/scripts/cb_pipeline.py).

THE CONVERSION IS EVAL-ONLY AND PARITY-GATED. The port is never trained
(DualSAE atoms are deliberately non-unit, so a training-time constraint
projection would destroy the loaded dictionary); it exists so
cb_dissect/cb_prune/cb_train see the EXACT function the DualSAE row was
scored with. Every conversion is verified on a real validation batch
(deployment code AND full reconstruction vs the native model) and refuses to
write on mismatch.

Space contract (the b_dec fold): DualSAE centers on a learned b_dec that
BSFBase does not have. For every mode except the LN one we FOLD it into the
port's mu (mu <- b_dec, scale <- 1, port b_dec <- 0), which makes
  normalize(x) = x - b_dec == DualSAE's x_cent,
  E_cb(x_norm)             == CB-SAE Eq. 6's E_cb(v - b) EXACTLY,
and the standard /scale + mu inversion re-adds b_dec. dual_jumprelu_sae
(input_unit_norm, per-sample LN) cannot fold — its normalize() is the LN and
mu is unused — so its b_dec stays a real port parameter (CBDualBSF.base_recon
adds it explicitly) and inversion goes through the denormalize() hook.

Usage (any node with the requirements installed; ~10-20 s per conversion):
    python -m featbsf.convert_dualsae --spec <cells.json> [--only-run <port_run>] [--device cuda]
where <cells.json> is a list of {run, dataset, encoder, seed, variant,
src_ckpt}. Skip-guard: CKPT_DIR/<port_run>/metrics.json (the driver puts the
base run's identity -- span, warm-up -- into <port_run>, so a different base
can never reuse a stale conversion).
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
from bsf.models import save_bsf, load_bsf
from bsf.sae_models import build_sae

# Parity tolerances (float32; the two implementations differ only in weight
# layout — Linear vs transposed matmul — and op order, so real divergence
# means a mapping bug, not noise). One legitimate divergence class exists:
# GATE BOUNDARY TIES — two pre-activations straddling a top-k threshold
# within float noise, where native and port keep DIFFERENT tied elements
# (observed live: 1 swapped latent of 8.4M on the very first batch_topk OIv7
# cell). Swapped positions are therefore forgiven ONLY when (a) they are
# vanishingly few, (b) each one provably sits AT the gate threshold, and
# (c) everything away from them still agrees to float precision.
TOL_Z = 1e-3          # max |z_port - z_native| on agreeing positions
TOL_TIE = 1e-3        # a swapped entry must be within this of the threshold
MAX_SWAP_FRAC = 1e-5  # cap on swapped positions (observed: ~1e-7)
TOL_RECON = 2e-4      # max recon diff on rows WITHOUT a swapped entry
TOL_RECON_SWAP = 0.05 # sanity bound on rows containing a boundary swap
PARITY_ROWS = 1024


def _source_ckpt(base: dict) -> Path:
    """Where the native vision/main.py checkpoint for this base lives.

    Every spec entry carries an explicit 'src_ckpt' (absolute path to the
    run's autoencoder_final.pt), written by the driver from the run's own
    training_config.json. (The original tool also had a fallback that looked
    the run up in a fixed checkpoint tree by name; it is not part of this
    release because a name lookup can silently pick up a same-named run
    trained under different hyperparameters.)
    """
    if not base.get("src_ckpt"):
        raise SystemExit(f"{base.get('run')}: spec entry has no 'src_ckpt'")
    return Path(base["src_ckpt"])


def _build_port(base: dict, cfg: dict, native) -> torch.nn.Module:
    """Fresh DualSAEFeaturizer with the source's own hyperparameters."""
    mode = native.sparsity_mode
    assert mode == {"dual_topk_sae": "topk", "dual_batchtopk_sae": "batch_topk",
                    "dual_jumprelu_sae": "jump_relu",
                    "dual_vanilla_sae": "vanilla",
                    "dual_matryoshka_sae": "matryoshka",
                    "dual_msae_sae": "msae"}[base["variant"]], \
        f"{base['run']}: checkpoint mode {mode!r} != variant {base['variant']}"
    ln = bool(cfg.get("normalize", False))
    # The LN preprocess exists ONLY in the jump_relu protocol; any other
    # combination means the checkpoint was trained off-protocol — refuse.
    assert ln == (mode == "jump_relu"), \
        f"{base['run']}: normalize={ln} unexpected for mode {mode}"
    assert int(cfg.get("poly_layers", 1)) == 1, \
        f"{base['run']}: poly_layers != 1 has no port equivalent"
    return build_sae(
        base["variant"], d=native.cfg["act_size"],
        n_latent=native.cfg["dict_size"], k=int(native.k_sparsity),
        poly_hidden=0,                              # 1-layer legacy poly
        lam_span=float(cfg.get("span_coef", 0.01)),
        l1_coeff=float(cfg.get("l1_coef", 1e-4)),
        bandwidth=float(cfg.get("bandwidth", 1e-3)),
        latent_soft_cap=float(native.soft_cap_val
                              if getattr(native, "soft_cap_val", None)
                              is not None else 30.0),
        # sae DualSAE gates batch-globally at EVAL for these two modes; the
        # port must reproduce that exact function (not its native EMA gate).
        deploy_batch_global=(mode in ("batch_topk", "matryoshka")),
        input_unit_norm=ln)


@torch.no_grad()
def _map_weights(port, native) -> None:
    b_enc_max = float(native.b_enc.abs().max())
    assert b_enc_max == 0.0, \
        (f"b_enc is nonzero (max {b_enc_max:g}) — DualSAE.forward never uses "
         f"b_enc so it must be exactly its zero init; a nonzero value means "
         f"an unexpected training path and the port's enc.bias would change "
         f"the encoder")
    port.enc.weight.copy_(native.W_enc.t())
    port.enc.bias.copy_(native.b_enc)               # zeros, asserted above
    port.D.copy_(native.W_dec.unsqueeze(1))
    port.poly.load_state_dict(native.poly.state_dict())
    if native.sparsity_mode == "jump_relu":
        port.log_threshold.data.copy_(native.jumprelu.log_threshold.data)
        port.bandwidth = float(native.bandwidth)
        # NO fold under LN: normalize() is the per-sample LN (mu unused), so
        # b_dec must survive as the port's own centering parameter.
        port.b_dec.copy_(native.b_dec)
    else:
        # The b_dec fold (module docstring): exactness of Eq. 6 + the
        # standard affine inversion, with the port's own b_dec zeroed.
        port.mu.copy_(native.b_dec)
        port.scale.fill_(1.0)
        port.b_dec.zero_()


@torch.no_grad()
def _parity(port, native, x: torch.Tensor) -> dict:
    """Deployment-code + full-recon parity on one real batch, forgiving ONLY
    provable gate-boundary ties (see the tolerance block above).

    Works for POLY-LESS ports too (featbsf/convert_plain_sae.py): a plain SAE
    has no poly branch, so its full reconstruction IS the mono one. Shared
    rather than duplicated because the tie-forgiveness logic below is the
    subtle part, and two copies of it would drift.
    """
    port.eval()
    native.eval()
    # Native: (pre, latents, recons) with recons already in input space.
    _, z_nat, recon_nat = native(x)
    # Port: deployment gate (what every CB stage consumes via encode()).
    x_norm = port.normalize(x)
    pre = port._pre_code(x_norm).squeeze(-1)        # (N, G) pre-gate code
    _, z_port, _ = port.encode(x_norm)
    z_port = z_port.squeeze(-1)
    xM = port.decode(z_port.unsqueeze(-1)) + port.b_dec
    # A plain SAE's reconstruction is the mono one; only a Dual carries poly.
    full = xM if not hasattr(port, "poly") else xM + port.poly(x_norm - xM)
    x_hat = port.denormalize(full)

    agree = (z_port != 0) == (z_nat != 0)
    swapped = ~agree
    n_swap = int(swapped.sum())
    dz = float(((z_port - z_nat).abs() * agree).max())

    # Boundary proof for every swapped position: its pre-activation must sit
    # within TOL_TIE of the gate threshold. Threshold = the smallest KEPT
    # value — global for the batch-global gates, per-row otherwise (for
    # jump_relu per-feature thresholds a swap can only be a >0-vs-0 knife
    # edge, where the kept-min bound still applies row-wise).
    tie_ok = True
    if n_swap:
        rows, cols = swapped.nonzero(as_tuple=True)
        kept = z_port != 0
        if getattr(port, "deploy_batch_global", False) \
                or getattr(port, "mono_mode", None) == "matryoshka":
            tau = z_port[kept].min() if bool(kept.any()) else pre.max()
            taus = tau.expand(rows.shape[0])
        else:
            big = torch.full_like(z_port, float("inf"))
            row_min = torch.where(kept, z_port, big).min(dim=1).values
            taus = row_min[rows]
        gap = (pre[rows, cols] - taus).abs()
        tie_ok = bool((gap <= TOL_TIE * (1 + taus.abs())).all())

    # Recon: strict away from swaps; sanity-bounded on the few affected rows
    # (one boundary atom in/out perturbs only those rows' reconstructions).
    row_has_swap = swapped.any(dim=1)
    dr_clean = float((x_hat - recon_nat)[~row_has_swap].abs().max()
                     if bool((~row_has_swap).any()) else 0.0)
    dr_swap = float((x_hat - recon_nat)[row_has_swap].abs().max()
                    if bool(row_has_swap.any()) else 0.0)

    ok = (dz < TOL_Z and tie_ok
          and n_swap <= max(4, int(MAX_SWAP_FRAC * swapped.numel()))
          and dr_clean < TOL_RECON and dr_swap < TOL_RECON_SWAP)
    return {"parity_ok": ok, "parity_max_abs_z": dz,
            "parity_n_swapped": n_swap, "parity_ties_proven": tie_ok,
            "parity_max_abs_recon": dr_clean,
            "parity_max_abs_recon_swaprows": dr_swap,
            "parity_rows": int(x.shape[0])}


def convert_one(base: dict, device: str) -> dict:
    out_dir = C.CKPT_DIR / base["run"]
    guard = out_dir / "metrics.json"
    if guard.exists():
        return {"run": base["run"], "skipped": True}

    src = _source_ckpt(base)
    adapted = load_any(src, device=device, which="final")
    native = adapted.model
    payload = torch.load(src, map_location=device, weights_only=False)
    cfg = payload.get("config", {})

    port = _build_port(base, cfg, native).to(device)
    _map_weights(port, native)

    # Parity batch: the eval split, unshuffled — real data, deterministic.
    loader = D.build_loader(base["dataset"], base["encoder"],
                            C.EVAL_SPLIT[base["dataset"]],
                            batch_size=PARITY_ROWS, labels="none",
                            shuffle=False, num_workers=2)
    x = next(iter(loader)).to(device)
    stats = _parity(port, native, x)
    if not stats["parity_ok"]:
        raise SystemExit(f"{base['run']}: PARITY FAILED {stats} — refusing "
                         f"to write a checkpoint that does not reproduce the "
                         f"source model")

    out_dir.mkdir(parents=True, exist_ok=True)
    extra = {"dataset": base["dataset"], "extractor": base["encoder"],
             "seed": base["seed"], "converted": True,
             "parent_run": src.parent.name}
    # Atomic writes (the CB stages may be listing concurrently).
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    os.close(fd)
    save_bsf(port.cpu(), tmp, extra=extra)
    os.replace(tmp, out_dir / "model.pt")
    # Round-trip gate: the payload must rebuild through load_bsf into the
    # same function (guards the extra_config/build_from_payload path the CB
    # stages depend on).
    rt = load_bsf(out_dir / "model.pt", device=device)
    rt_stats = _parity(rt, native, x)
    if not rt_stats["parity_ok"]:
        (out_dir / "model.pt").unlink()
        raise SystemExit(f"{base['run']}: ROUND-TRIP PARITY FAILED "
                         f"{rt_stats} — load_bsf rebuilt a different model")

    meta = {"variant": base["variant"], "dataset": base["dataset"],
            "extractor": base["encoder"], "seed": base["seed"],
            "converted_from": str(src), **stats,
            "roundtrip_max_abs_recon": rt_stats["parity_max_abs_recon"]}
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, guard)
    return {"run": base["run"], **stats}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only-run", default=None,
                    help="convert just this port run name")
    ap.add_argument("--spec", type=Path, required=True,
                    help="JSON list of base dicts to convert. Each entry needs "
                         "run/dataset/encoder/seed/variant/src_ckpt.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()
    bases = json.loads(args.spec.read_text())
    if args.only_run:
        bases = [b for b in bases if b["run"] == args.only_run]
        if not bases:
            sys.exit(f"unknown port run {args.only_run!r}")
    done = skipped = 0
    for base in bases:
        res = convert_one(base, args.device)
        if res.get("skipped"):
            skipped += 1
            continue
        done += 1
        print(f"[ok] {res['run']}  max|dz|={res['parity_max_abs_z']:.2e}  "
              f"swaps={res['parity_n_swapped']}  "
              f"max|dr|={res['parity_max_abs_recon']:.2e}", flush=True)
    print(f"converted {done}, skipped (already done) {skipped}, "
          f"total {len(bases)}")


if __name__ == "__main__":
    main()
