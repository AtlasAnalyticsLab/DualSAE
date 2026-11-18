"""s0: re-score every A16 SAE at its FINAL checkpoint and measure the D-43 health gate.

One unit = one Track B SAE (arms.iter_arms('B') order; jobs.py kind 's0').  Output:
base.s0_record(tag) (JSON, written last and atomically -- the skip-guard) plus, for BatchTopK
arms, base.s0_threshold(tag) (the D-44 threshold, written first so a killed job never re-runs
the 5,000-batch EMA).

What is measured (every quantity at positions >= 1 unless named 'sink'; D-19, D-43, D-44, C18,
C20):
  * training log (train_stdout.log): final val Active (gate a, top-k arms), final val
    reconstruction loss and its minimum over epochs (gate b).  "Reconstruction loss" = the
    logged 'Val L2' (Dual, BatchTopK, JumpReLU) or 'Val NMSE' (TopK baseline, whose objective is
    the per-sample NMSE; D-19).  Declared reading: 'Val Loss' also contains aux/span terms.
  * D-44 (BatchTopK arms): frozen inference threshold = sae_io.estimate_batchtopk_threshold over
    >= 5,000 NON-SINK train-shard batches of 2048 (batch order seeded from the LAYER, so every
    arm of a pair sees the same batches), and sae_io.d44_l0_columns (inference vs training-time
    non-sink L0 on test windows [0, 400)).
  * rows = owt TEST windows [0, 400) (51,200 rows: 50,800 non-sink + 400 sinks; the D-15 rows),
    encoded with the INFERENCE encoder (per-row TopK / JumpReLU forward; BatchTopK: D-44
    threshold).  L0, live count, dead % (D-04: relative to these rows), sink latents (fire at
    position 0 in > 50% of windows), always-on latents (fire rate > 0.3 on non-sink rows), the
    eligible count (live, non-sink, not always-on; D-46) and the excluded fractions.
  * reconstruction: non-sink FVU, sink FVU and sink relative error (C18); BatchTopK arms also
    under the training-style all-row 2048 batch top-k ("shown beside", D-43).  Duals: xM FVU,
    ||poly(r) - r|| / ||r|| (aggregate and per-row median), the bypass-regime label (xM FVU >= 1,
    C20) and the span penalty on test rows (the D-03 manipulation check; the trainer's
    span_overlap_penalty with its active_only setting).
  * CE recovered via common/loss_recovered_hooks primitives, reported BOTH ways
    (D-43 / task spec): splice at positions >= 1 with the position-0 residual left clean
    (PRIMARY, used by gate c) and splice at all positions (the published eval_trackb protocol).
    Each variant has its own zero-ablation reference (zero at positions >= 1 / everywhere).
    BatchTopK arms: inference-threshold encoder (primary) and all-row batch top-k (beside).
    Same token set as eval_trackb: the owt test stream in batches of 16 x 128 = 2048 rows.

The GATE VERDICT is not decided here: gate (c) for Dual/g0 arms compares with the pair
BASELINE's CE, so gates43.py combines the records of a (layer, activation) pair.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from common import lang_config as config
from steering import base, steer
from common import arms, sae_io, seeds as _seeds
from common import loss_recovered_hooks as H

_EPOCH_RE = re.compile(r"^Epoch (\d+)/(\d+) - Val Loss: ([0-9.eE+-]+)(.*)$")


# ----------------------------------------------------------------------------- training log
def parse_train_log(run_dir: str) -> Dict[str, Any]:
    """Per-epoch validation lines of train_stdout.log -> gate (a)/(b) inputs."""
    path = os.path.join(run_dir, "train_stdout.log")
    rows: Dict[int, Dict[str, float]] = {}
    n_total = None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = _EPOCH_RE.match(line.strip())
            if not m:
                continue
            ep, n_total = int(m.group(1)), int(m.group(2))
            d = {"val_loss": float(m.group(3))}
            for k, v in re.findall(r"Val ([A-Za-z0-9(),_ ]+?): ([0-9.eE+-]+)", m.group(4)):
                d[k.strip()] = float(v)
            rows[ep] = d                      # a re-logged epoch keeps its LAST line
    if not rows:
        raise RuntimeError(f"{path}: no 'Epoch N/M - Val Loss' lines")
    eps = sorted(rows)
    key = "L2" if "L2" in rows[eps[-1]] else ("NMSE" if "NMSE" in rows[eps[-1]] else None)
    if key is None:
        raise RuntimeError(f"{path}: neither 'Val L2' nor 'Val NMSE' on the final epoch line")
    recon = [rows[e][key] for e in eps if key in rows[e]]
    final = rows[eps[-1]]
    return {"n_epochs_logged": len(eps), "last_epoch": eps[-1], "n_epochs_total": n_total,
            "recon_key": f"Val {key}", "final_recon": final[key], "min_recon": min(recon),
            "min_recon_epoch": eps[int(np.argmin(recon))],
            "recon_ratio": final[key] / min(recon) if min(recon) > 0 else float("inf"),
            "final_active": final.get("Active"), "final_val_loss": final["val_loss"],
            "final_span": final.get("Span")}


# ----------------------------------------------------------------------------- D-44 threshold
def ensure_threshold(model, meta, ref, device: str, n_batches: int, allow_short: bool
                     ) -> Dict[str, Any]:
    path = base.s0_threshold(ref.tag)
    rec = base.read_json(path)
    if rec is not None and rec.get("n_batches") == n_batches and rec.get("tag") == ref.tag:
        return rec
    layer = int(ref.key[1:])
    it = sae_io.iter_nonsink_batches(config.resid_dir(layer, "train", base.CORPUS), n_batches,
                                     _seeds.derive_seed("a16.d44", f"L{layer}|train"))
    res = sae_io.estimate_batchtopk_threshold(model, meta, it, n_batches, device,
                                              allow_short=allow_short)
    res.update(tag=ref.tag, layer=layer, batch_seed_key=f"a16.d44|L{layer}|train")
    base.atomic_write_json(path, res)
    return res


# ----------------------------------------------------------------------------- encoders
def make_encoders(model, meta, threshold: Optional[float]):
    """name -> fn(x (N, D) torch on device) -> dict(z, xhat[, xM, r_hat]).  'inference' is the
    per-token encoder used for selection/steering; 'btk' (BatchTopK only) is the training-style
    batch top-k over whatever rows are passed (callers pass whole 2048-row batches)."""
    is_dual = bool(meta["is_dual"])

    def inference(x):
        with torch.no_grad(), sae_io._fp32_exact():
            if meta["family"] == "batch_topk":
                return sae_io.forward_thresholded(model, meta, x, threshold, fixed_rows=None)
            _, z, xhat = model(x)
            out = {"z": z, "xhat": xhat}
            if is_dual:
                out.update(xM=model._xM, r_hat=model._r_hat)
            return out

    def btk(x):
        with torch.no_grad(), sae_io._fp32_exact():
            _, z, xhat = model(x)
            out = {"z": z, "xhat": xhat}
            if is_dual:
                out.update(xM=model._xM, r_hat=model._r_hat)
            return out

    enc = {"inference": inference}
    if meta["family"] == "batch_topk":
        enc["btk"] = btk
    return enc


# ----------------------------------------------------------------------------- row statistics
@torch.no_grad()
def row_stats(model, meta, threshold, layer: int, device: str, n_windows: int) -> Tuple[Dict, Dict]:
    X, ids = sae_io.read_shard_rows(layer, "test", 0, n_windows * base.CTX_LEN, corpus=base.CORPUS)
    ns = (ids % base.CTX_LEN) != 0
    Xt = torch.from_numpy(X).to(device)
    Fdim = int(meta["dict_size"])
    fire_ns = torch.zeros(Fdim, device=device, dtype=torch.float64)
    fire_sink = torch.zeros(Fdim, device=device, dtype=torch.float64)
    l0_ns: List[torch.Tensor] = []
    acc = {k: 0.0 for k in ("sse_ns", "sse_sink", "sse_xM_ns", "poly_num", "poly_den", "span_sum")}
    per_row_poly: List[torch.Tensor] = []
    ns_t = torch.from_numpy(ns).to(device)
    enc = make_encoders(model, meta, threshold)["inference"]
    chunk = 4096
    is_dual = bool(meta["is_dual"])
    span_rows = 0
    # The trainer's span penalty setting (training_config.json 'active_only_span', default True
    # in dual_sae.compute_loss); read, never assumed, so the manipulation check uses the exact
    # penalty the run was trained with.
    active_only = True
    if is_dual:
        with open(os.path.join(meta["run_dir"], "training_config.json")) as fh:
            active_only = bool(json.load(fh).get("active_only_span", True))
    for s in range(0, Xt.shape[0], chunk):
        x = Xt[s:s + chunk]
        m = ns_t[s:s + chunk]
        out = enc(x)
        z = out["z"]
        nz = z != 0
        fire_ns += nz[m].sum(0).double()
        fire_sink += nz[~m].sum(0).double()
        l0_ns.append(nz[m].sum(1).float())
        err = (x - out["xhat"]).pow(2).sum(1).double()
        acc["sse_ns"] += float(err[m].sum())
        acc["sse_sink"] += float(err[~m].sum())
        if is_dual:
            acc["sse_xM_ns"] += float((x - out["xM"]).pow(2).sum(1).double()[m].sum())
            r = x - out["xM"]
            dpoly = (out["r_hat"] - r).pow(2).sum(1).double()
            rr = r.pow(2).sum(1).double()
            acc["poly_num"] += float(dpoly[m].sum())
            acc["poly_den"] += float(rr[m].sum())
            per_row_poly.append((dpoly[m] / rr[m].clamp_min(1e-30)).sqrt().float())
            from models.dual_sae import span_overlap_penalty   # sae/ on sys.path via sae_io
            if int(m.sum()):
                acc["span_sum"] += float(span_overlap_penalty(model.W_dec, out["r_hat"][m], z[m],
                                                              active_only=active_only)) * int(m.sum())
                span_rows += int(m.sum())
    Xns, Xsk = Xt[ns_t], Xt[~ns_t]
    tot_ns = float((Xns - Xns.mean(0)).pow(2).sum().double())
    tot_sk = float((Xsk - Xsk.mean(0)).pow(2).sum().double())
    n_ns, n_sk = int(ns.sum()), int((~ns).sum())
    fire_rate = (fire_ns / n_ns).float().cpu().numpy()
    sink_frac = (fire_sink / max(n_sk, 1)).float().cpu().numpy()
    live = fire_rate > 0
    sink_lat = sink_frac > base.SINK_FIRE_FRAC
    always_on = fire_rate > base.ALWAYS_ON_RATE
    eligible = live & ~sink_lat & ~always_on
    l0 = torch.cat(l0_ns)
    stats = {
        "rows": f"owt test windows [0, {n_windows}): {n_ns} non-sink + {n_sk} sink rows",
        "n_nonsink": n_ns, "n_sink": n_sk,
        "eval_l0": float(l0.mean()), "eval_l0_median": float(l0.median()),
        "live_latents": int(live.sum()), "dead_pct": 100.0 * (1 - live.mean()),
        "dict_size": int(len(live)),
        "n_sink_latents": int(sink_lat.sum()), "n_always_on": int(always_on.sum()),
        "n_eligible": int(eligible.sum()),
        "excluded_frac_of_dict": float((sink_lat | always_on).mean()),
        "excluded_frac_of_live": float(((sink_lat | always_on) & live).sum() / max(live.sum(), 1)),
        "fvu_nonsink": acc["sse_ns"] / tot_ns,
        "fvu_sink": acc["sse_sink"] / tot_sk if tot_sk > 0 else float("nan"),
        "sink_rel_err": acc["sse_sink"] / float(Xsk.pow(2).sum().double()),
        "median_nonsink_norm": float(Xns.norm(dim=1).median()),
        "sink_norm_mean": float(Xsk.norm(dim=1).mean()),
    }
    if is_dual:
        stats.update({
            "xM_fvu_nonsink": acc["sse_xM_ns"] / tot_ns,
            "poly_identity_ratio": math.sqrt(acc["poly_num"] / acc["poly_den"]),
            "poly_identity_ratio_row_median": float(torch.cat(per_row_poly).median()),
            "span_penalty_test_nonsink": acc["span_sum"] / max(span_rows, 1),
            "span_penalty_active_only": active_only,
        })
        stats["bypass_regime"] = bool(stats["xM_fvu_nonsink"] >= 1.0)
    lat = {"fire_rate_nonsink": fire_rate, "sink_fire_frac": sink_frac,
           "eligible": eligible, "live": live}
    return stats, lat


def btk_recon_stats(model, meta, layer: int, device: str, n_windows: int) -> Dict[str, float]:
    """BatchTopK 'shown beside': recon under the training-style all-row 2048 batch top-k
    (sae_io.forward_reps_rows: seeded permutation, 2048-row batches, same for every arm)."""
    X, ids = sae_io.read_shard_rows(layer, "test", 0, n_windows * base.CTX_LEN, corpus=base.CORPUS)
    ns = (ids % base.CTX_LEN) != 0
    perm_seed = _seeds.derive_seed("a16.s0.btk", f"L{layer}|test|0-{n_windows}")
    reps = sae_io.forward_reps_rows(model, meta, X, device, perm_seed, keys=("xhat",) + (("xM",) if meta["is_dual"] else ()))
    err = ((X - reps["xhat"]) ** 2).sum(1)
    tot = ((X[ns] - X[ns].mean(0)) ** 2).sum()
    out = {"btk_fvu_nonsink": float(err[ns].sum() / tot),
           "btk_eval_l0_nonsink": float(reps["l0"][ns].mean()),
           "btk_live_latents_nonsink": int(np.unique(reps["z"][np.flatnonzero(ns)].indices).size)}
    if meta["is_dual"]:
        out["btk_xM_fvu_nonsink"] = float((((X - reps["xM"]) ** 2).sum(1))[ns].sum() / tot)
    return out


# ----------------------------------------------------------------------------- CE recovered
@torch.no_grad()
def ce_recovered(lm, model, meta, threshold, layer: int, device: str, n_batches: int) -> Dict[str, Any]:
    nf, conv = steer.norm_factor(layer)
    module = lm.gpt_neox.layers[layer]
    encs = make_encoders(model, meta, threshold)
    variants = [(e, where) for e in encs for where in ("ge1", "all")]
    sums = {"orig": 0.0, "zero_ge1": 0.0, "zero_all": 0.0}
    sums.update({f"{e}_{w}": 0.0 for e, w in variants})
    kls = {f"{e}_{w}": 0.0 for e, w in variants}
    n_pred = 0
    batches = H.iter_token_batches(config.lm_token_stream(base.CORPUS, "test"),
                                   batch_sequences=base.CE_BATCH_SEQ, ctx_len=base.CTX_LEN,
                                   max_sequences=n_batches * base.CE_BATCH_SEQ, device=device)
    nb = 0
    for tokens in batches:
        B, L = tokens.shape
        pos = torch.arange(L, device=device)[None].expand(B, L)

        def run(fn):
            with H.splice_activation(module, fn):
                return lm(input_ids=tokens, position_ids=pos, use_cache=False).logits

        clean = run(None)
        sums["orig"] += float(H.next_token_ce(clean, tokens)) * B * (L - 1)

        def zero_ge1(h):
            out = torch.zeros_like(h)
            out[:, 0] = h[:, 0]
            return out
        sums["zero_all"] += float(H.next_token_ce(run(torch.zeros_like), tokens)) * B * (L - 1)
        sums["zero_ge1"] += float(H.next_token_ce(run(zero_ge1), tokens)) * B * (L - 1)
        for e, w in variants:
            def splice(h, e=e, w=w):
                flat = h.reshape(B * L, -1).float()
                xs = H.to_shard_space(flat, nf, conv)
                xh = H.from_shard_space(encs[e](xs)["xhat"], nf, conv).reshape(B, L, -1).to(h.dtype)
                if w == "ge1":
                    xh[:, 0] = h[:, 0]          # the sink residual stays clean (D-19)
                return xh
            lg = run(splice)
            sums[f"{e}_{w}"] += float(H.next_token_ce(lg, tokens)) * B * (L - 1)
            kls[f"{e}_{w}"] += H.next_token_kl(clean, lg) * B * (L - 1)
            del lg
        n_pred += B * (L - 1)
        nb += 1
        del clean
    if nb == 0:
        raise RuntimeError("no CE batches")
    m = {k: v / n_pred for k, v in sums.items()}
    out: Dict[str, Any] = {"n_batches": nb, "n_predicted": n_pred, "loss_original": m["orig"],
                           "loss_zero_ge1": m["zero_ge1"], "loss_zero_all": m["zero_all"]}
    for e, w in variants:
        k = f"{e}_{w}"
        out[f"loss_{k}"] = m[k]
        out[f"frac_recovered_{k}"] = H.frac_recovered(m["orig"], m[k], m[f"zero_{w}"])
        out[f"kl_{k}"] = kls[k] / n_pred
    out["frac_recovered_primary"] = out["frac_recovered_inference_ge1"]
    return out


# ----------------------------------------------------------------------------- driver
def rescore(ref, device: str, *, d44_batches: int = base.D44_BATCHES, allow_short: bool = False,
            n_windows: int = base.GATE_WINDOWS, ce_batches: Optional[int] = None,
            lm=None) -> Dict[str, Any]:
    t0 = time.time()
    layer = int(ref.key[1:])
    rec: Dict[str, Any] = {"tag": ref.tag, "run_dir": str(ref.run_dir), "layer": layer,
                           "activation": ref.activation, "arm": ref.arm, "seed": ref.seed,
                           "smoke": base.IS_SMOKE, "timings": {}}
    rec["train_log"] = parse_train_log(str(ref.run_dir))
    model, meta = sae_io.load_arm(ref.run_dir, device)
    rec.update(checkpoint=meta["checkpoint_name"], best_epoch=meta["best_val_epoch"],
               final_step=meta["final_step"], family=meta["family"], is_dual=bool(meta["is_dual"]))
    thr = None
    if meta["family"] == "batch_topk":
        t = time.time()
        th = ensure_threshold(model, meta, ref, device, d44_batches, allow_short)
        thr = th["threshold"]
        rec["d44"] = dict(th)
        rec["d44"].update(sae_io.d44_l0_columns(
            model, meta, thr, layer, device,
            _seeds.derive_seed("a16.d44cols", f"L{layer}|test|0-{n_windows}"), n_windows=n_windows))
        rec["timings"]["d44_s"] = round(time.time() - t, 1)
    t = time.time()
    stats, lat = row_stats(model, meta, thr, layer, device, n_windows)
    rec["rows"] = stats
    if meta["family"] == "batch_topk":
        rec["rows"].update(btk_recon_stats(model, meta, layer, device, n_windows))
    rec["timings"]["rows_s"] = round(time.time() - t, 1)
    t = time.time()
    own_lm = lm is None
    if own_lm:
        lm, _ = steer.load_lm(device)
    n_ce = ce_batches if ce_batches is not None else base.CE_TOKENS // (base.CE_BATCH_SEQ * base.CTX_LEN)
    rec["ce"] = ce_recovered(lm, model, meta, thr, layer, device, n_ce)
    rec["timings"]["ce_s"] = round(time.time() - t, 1)
    # D-04 row annotation (Track B: the D-44 / inference encoder at positions >= 1).
    rec["d04"] = {"checkpoint": rec["checkpoint"], "best_epoch": rec["best_epoch"],
                  "eval_l0": stats["eval_l0"], "dead_pct": stats["dead_pct"],
                  "live_latents": stats["live_latents"]}
    base.atomic_save_npz(base.s0_record(ref.tag).replace(".json", ".latents.npz"),
                         **{k: np.asarray(v) for k, v in lat.items()})
    rec["timings"]["total_s"] = round(time.time() - t0, 1)
    rec["complete"] = True
    base.atomic_write_json(base.s0_record(ref.tag), rec)
    return rec


def load_latents(tag: str) -> Dict[str, np.ndarray]:
    return dict(np.load(base.s0_record(tag).replace(".json", ".latents.npz")))


def is_done(tag: str) -> bool:
    r = base.read_json(base.s0_record(tag))
    return bool(r and r.get("complete")) and os.path.isfile(
        base.s0_record(tag).replace(".json", ".latents.npz"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, help="jobs.py s0 index (arms.iter_arms('B') order)")
    ap.add_argument("--tag")
    ap.add_argument("--device", default=None)
    ap.add_argument("--d44-batches", type=int, default=base.D44_BATCHES)
    ap.add_argument("--allow-short", action="store_true", help="smoke only: < 5000 D-44 batches")
    ap.add_argument("--windows", type=int, default=base.GATE_WINDOWS)
    ap.add_argument("--ce-batches", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    refs = base.sae_refs()
    ref = refs[a.index] if a.index is not None else next(r for r in refs if r.tag == a.tag)
    why = base.held_reason(ref)
    if why:
        print(f"[s0] {ref.tag}: not runnable -- {why}")
        return 3
    if is_done(ref.tag) and not a.force:
        print(f"[s0] {ref.tag}: already done")
        return 0
    rec = rescore(ref, a.device or base.device_default(), d44_batches=a.d44_batches,
                  allow_short=a.allow_short, n_windows=a.windows, ce_batches=a.ce_batches)
    print(f"[s0] {ref.tag}: L0 {rec['rows']['eval_l0']:.1f} live {rec['rows']['live_latents']} "
          f"fvu_ns {rec['rows']['fvu_nonsink']:.4f} CE(ge1) {rec['ce']['frac_recovered_primary']:.4f} "
          f"({rec['timings']['total_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
