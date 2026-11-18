"""s2: analytic poly cancellation for every live Dual latent, before any generation (D-47, C20).

One unit = one Dual SAE (arms dual and dual_g0; jobs.py kind 's2').  No LM is involved.

For latent j with unit atom u_j = W_dec[j] / ||W_dec[j]|| and a mono edit that moves the mono
reconstruction by exactly alpha along u_j (latent change dz_j = alpha / ||W_dec[j]||; Dual atoms
are NOT unit-norm, so the edit is stated in residual units to make latents comparable):
    r        = x_s - xM(z)                          (shard space, inference encoder)
    d_mono   = alpha * u_j
    d_poly   = poly(r - d_mono) - poly(r)           (the poly branch's reaction under mode (iii))
    d        = d_mono + d_poly                      (the mode-(iii) displacement of x'_s)
reported per latent (mean over rows):
    net_along      <d, u_j> / alpha      fraction of the intended move that survives ("||proj
                                         delta||/alpha", signed: < 0 means over-cancelled)
    cancel_ratio   -<d_poly, d_mono> / ||d_mono||^2   (D-47 cancellation ratio)
    cos_poly_atom  cos(d_poly, W_j)
    total_ratio    ||d|| / alpha
at alpha in {0.25, 1, 4} x the layer's median non-sink token norm (poly is nonlinear, so the
answer depends on the edit size).  Rows: 256 seeded non-sink rows of owt test windows [0, 400)
(the s0 rows; the same rows for every SAE of a layer).  Latents: every LIVE latent from s0.
The cell label comes from s0: "bypass regime" iff xM FVU >= 1 (C20); in that regime mode (iii)
is a bypass diagnostic, never a property of DualSAE.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict

import numpy as np
import torch

from steering import base, s0_rescore, steer
from common import sae_io, seeds as _seeds

ALPHA_MULTS = (0.25, 1.0, 4.0)
N_ROWS = 256


@torch.no_grad()
def run(ref, device: str, n_rows: int = N_ROWS, max_latents: int = 0) -> Dict[str, Any]:
    t0 = time.time()
    layer = int(ref.key[1:])
    s0 = base.read_json(base.s0_record(ref.tag))
    lat = s0_rescore.load_latents(ref.tag)
    model, meta = sae_io.load_arm(ref.run_dir, device)
    if not meta["is_dual"]:
        raise ValueError(f"{ref.tag} is not a Dual")
    thr = base.read_json(base.s0_threshold(ref.tag))["threshold"] if meta["family"] == "batch_topk" else None
    A = steer.SAEAdapter(model, meta, thr)
    X, _ = sae_io.read_shard_rows(layer, "test", 0, base.GATE_WINDOWS * base.CTX_LEN,
                                  corpus=base.CORPUS, drop_sinks=True)
    pick = _seeds.rng(_seeds.derive_seed("a16.s2.rows", f"L{layer}")).choice(X.shape[0], n_rows, replace=False)
    x = torch.from_numpy(X[np.sort(pick)]).to(device)
    parts = A.dual_parts(x)
    r = x - parts["xM"]
    poly_r = A.poly(r)
    live = np.flatnonzero(lat["live"])
    if max_latents:
        live = live[:max_latents]
    m = float(s0["rows"]["median_nonsink_norm"])
    res: Dict[str, Any] = {}
    arrays: Dict[str, np.ndarray] = {"latents": live.astype(np.int32)}
    for am in ALPHA_MULTS:
        alpha = am * m
        cols = {k: np.zeros(live.size, np.float32) for k in
                ("net_along", "cancel_ratio", "cos_poly_atom", "total_ratio")}
        C = max(1, 32768 // n_rows)
        for s in range(0, live.size, C):
            j = torch.as_tensor(live[s:s + C], device=device)
            U = A.U[j]                                             # (c, D)
            d_mono = alpha * U                                     # (c, D)
            r_new = r[None] - d_mono[:, None]                      # (c, R, D)
            d_poly = A.poly(r_new.reshape(-1, r.shape[1])).reshape(r_new.shape) - poly_r[None]
            d = d_mono[:, None] + d_poly
            cols["net_along"][s:s + C] = ((d * U[:, None]).sum(-1) / alpha).mean(1).cpu().numpy()
            cols["cancel_ratio"][s:s + C] = (-(d_poly * U[:, None]).sum(-1) / alpha).mean(1).cpu().numpy()
            cols["cos_poly_atom"][s:s + C] = torch.nn.functional.cosine_similarity(
                d_poly, U[:, None].expand_as(d_poly), dim=-1).mean(1).cpu().numpy()
            cols["total_ratio"][s:s + C] = (d.norm(dim=-1) / alpha).mean(1).cpu().numpy()
        key = f"alpha{am:g}m"
        for k, v in cols.items():
            arrays[f"{key}__{k}"] = v
        q = lambda v: {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                       "p90": float(np.percentile(v, 90)), "mean": float(v.mean())}
        res[key] = {"alpha_shard": alpha, **{k: q(v) for k, v in cols.items()},
                    "frac_cancel_gt_0.5": float((cols["cancel_ratio"] > 0.5).mean()),
                    "frac_net_along_lt_0.5": float((cols["net_along"] < 0.5).mean())}
    out = {"tag": ref.tag, "layer": layer, "activation": ref.activation, "arm": ref.arm,
           "seed": ref.seed, "n_live_latents": int(live.size), "n_rows": n_rows,
           "median_nonsink_norm": m, "alpha_mults": list(ALPHA_MULTS),
           "xM_fvu_nonsink": s0["rows"]["xM_fvu_nonsink"],
           "poly_identity_ratio": s0["rows"]["poly_identity_ratio"],
           "bypass_regime": bool(s0["rows"]["bypass_regime"]),
           "label": "bypass regime" if s0["rows"]["bypass_regime"] else None,
           "summary": res, "smoke": base.IS_SMOKE,
           "seconds": round(time.time() - t0, 1), "complete": True}
    base.atomic_save_npz(base.s2_record(ref.tag).replace(".json", ".npz"), **arrays)
    base.atomic_write_json(base.s2_record(ref.tag), out)
    return out


def is_done(tag: str) -> bool:
    r = base.read_json(base.s2_record(tag))
    return bool(r and r.get("complete"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, help="jobs.py s2 index (base.dual_refs() order)")
    ap.add_argument("--tag")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-latents", type=int, default=0, help="smoke only: first N live latents")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    refs = base.dual_refs()
    ref = refs[a.index] if a.index is not None else next(r for r in refs if r.tag == a.tag)
    why = base.held_reason(ref)
    if why:
        print(f"[s2] {ref.tag}: not runnable -- {why}")
        return 3
    if is_done(ref.tag) and not a.force:
        print(f"[s2] {ref.tag}: already done")
        return 0
    if not s0_rescore.is_done(ref.tag):
        print(f"[s2] {ref.tag}: s0 record missing")
        return 4
    if a.max_latents and not base.IS_SMOKE:
        raise SystemExit("--max-latents is a smoke option")
    out = run(ref, a.device or base.device_default(), max_latents=a.max_latents)
    s = out["summary"]["alpha1m"]
    print(f"[s2] {ref.tag}: {out['n_live_latents']} live latents; at alpha = 1 x median norm: "
          f"cancel ratio median {s['cancel_ratio']['median']:.3f}, net along atom median "
          f"{s['net_along']['median']:.3f}, cos(d_poly, W_j) median {s['cos_poly_atom']['median']:.3f}; "
          f"label {out['label']} ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
