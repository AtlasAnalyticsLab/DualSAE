"""CB stage 4: train the concept-bottleneck head (+ the Residual Branch for
CB-DualSAE) on top of the frozen pruned base.

Arms: CB-SAE = concept head on a pruned PLAIN base (--no-poly); CB-DualSAE =
concept head on a pruned DualSAE base, whose Residual Branch ("poly") is
warm-started from the base and trained jointly (objective A below).

The objective structure combines CB-SAE's alternating scheme with the Dual
poly branch (see bsf/cb_dual_bsf.py for the model math). Per minibatch, three
objectives on the SAME batch, each with its own Adam and an immediate step
(the CB-SAE paper's stated alternative to loss weighting; schedule A->B->C):

  A  reconstruction   L_r = MSE(xB' + x_cb [+ r_hat], x_norm)
                            [+ lam_span * span(r_hat vs retained ++ D_cb)]
                      updates E_cb, D_cb, AND poly       (CB Eq. 8 + Dual)
  B  interpretability L_int = cos^3(dense concept acts, y)
                      updates E_cb ONLY                  (CB Eq. 9)
  C  steerability     L_st = cos^3(acts of the DEPLOYMENT recon, y), with the
                      concept code DETACHED and E_cb acting as a frozen
                      function (its incidental grad is discarded)
                      updates D_cb ONLY                  (CB Eq. 10)

Dual mechanics grafted in:
  * CB-FIRST WARMUP (--warmup, default cb_config.CB_WARMUP): for the first W
    steps the poly stays frozen and objective A reconstructs with xB' + x_cb
    only (span is exactly 0) — the sparse concept head gets to capture the
    concept-aligned residual before the dense poly MLP can swallow it (the
    DualSAE two-stage warm-up idiom, pointed at the CB head).
  * The span penalty covers BOTH steerable dictionaries (retained blocks and
    concept blocks), so the poly output is pushed out of everything a user
    might steer.
  * L_st's cyclic reconstruction is the DEPLOYMENT recon (no r_hat): the
    steerability objective must describe the model as steered/deployed
    (the Dual training-only-poly contract).
  * No unit-norm projection on D_cb during training; the base decoder is
    frozen, so no constraint projection runs at all.

y is computed ON THE FLY from the paired clip rows (featbsf/cb_data.py); only
the validation y ever exists on disk (stage 3's y_val, used here only through
the same on-the-fly path for consistency).

Resumable: ckpt_last.pt every --ckpt-every steps + an epoch-seeded loader
factory (bit-faithful mid-epoch resume); metrics.json is the skip-guard.

Usage (--seed MUST match the base run's seed, enforced). The CB rows of the
main comparison use b_cb=1, R=1024, k_cb=5 (passed explicitly; the CLI
defaults are b_cb=4, R=256, k_cb from the concept set):
  python -m featbsf.cb_train --run <base_run> --cset vlg --retain 1024 \
      --rank interp --b-cb 1 --k-cb 5 --seed 42            # CB-DualSAE
  python -m featbsf.cb_train --run <base_run> --cset vlg --retain 1024 \
      --rank interp --b-cb 1 --k-cb 5 --seed 42 --no-poly  # CB-SAE
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bsf.cb_dual_bsf import CBDualBSF, save_cb_dual
from bsf.cb_sae import cosine_cubed_loss
from bsf.models import load_bsf
from featbsf import cb_config as CB
from featbsf import config as C
from featbsf.cb_data import build_paired_loader, load_concept_embeds, make_y


def _atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


@torch.no_grad()
def holdout_metrics(model: CBDualBSF, loader, T_cb: torch.Tensor,
                    device: str, center: bool = True,
                    max_rows: int = 50_000) -> dict:
    """Holdout pass over the (paired) validation loader, streamed:
      * R^2 decomposition in the model's NORMALIZED space — base-only /
        deployment (base+CB, the steerable recon) / full (+poly). The
        input-space metrics are computed by vision/branch_metrics.py.
      * heldout L_int / L_st (batch-computed like training, chunk-averaged).
      * dead_concept_frac + cb_l0 + the per-unit fire-rate vector, under the
        DEPLOYMENT gate (describes the model as it will be used).
    """
    was_training = model.training
    model.eval()
    sse = {"base": 0.0, "deploy": 0.0, "full": 0.0}
    sst = 0.0
    l_int = l_st = 0.0
    chunks = 0
    n_units = model.R + model.n_concepts
    fire = torch.zeros(n_units, device=device)
    rows = 0
    # Streaming mean for SS_tot, estimated over the SAME capped prefix
    # (single pass via accumulators).
    sum_x = None
    sum_x2 = 0.0
    for x_sub, x_clip in loader:
        x = x_sub.to(device, torch.float32)
        y = make_y(x_clip.to(device), T_cb)
        x_norm = model.normalize(x)
        parts = model.full_recon(x_norm)
        deploy = parts["xB"] + parts["x_cb"]
        sse["base"] += float(((x_norm - parts["xB"]) ** 2).sum())
        sse["deploy"] += float(((x_norm - deploy) ** 2).sum())
        sse["full"] += float(((x_norm - parts["x_hat"]) ** 2).sum())
        sum_x = (x_norm.sum(0) if sum_x is None else sum_x + x_norm.sum(0))
        sum_x2 += float((x_norm ** 2).sum())
        acts = model.concept_dense_acts(x_norm)
        l_int += float(cosine_cubed_loss(acts, y, center=center))
        l_st += float(cosine_cubed_loss(model.concept_dense_acts(deploy), y,
                                        center=center))
        fire += (model.unit_acts(x_norm) > 0).float().sum(0)
        chunks += 1
        rows += x.shape[0]
        if rows >= max_rows:
            break
    model.train(was_training)
    if rows == 0:
        return {}
    # SS_tot about the streamed mean: sum ||x||^2 - N ||mean||^2.
    mean = sum_x / rows
    ss_tot = max(sum_x2 - rows * float((mean ** 2).sum()), 1e-9)
    cb_fire = fire[model.R:]
    return {
        "r2_base_only": 1.0 - sse["base"] / ss_tot,
        "r2_deploy": 1.0 - sse["deploy"] / ss_tot,
        "r2_full": 1.0 - sse["full"] / ss_tot,
        "l_int": l_int / chunks, "l_st": l_st / chunks,
        "dead_concept_frac": float((cb_fire == 0).float().mean()),
        "cb_l0": float(cb_fire.sum().item() / rows),
        "dead_block_frac": float((fire[:model.R] == 0).float().mean()),
        "fire_rate": (fire / rows).cpu().numpy(),
        "n_rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="base (Dual)BSF run name")
    ap.add_argument("--cset", required=True, choices=list(CB.CONCEPT_SETS))
    ap.add_argument("--b-cb", type=int, default=CB.CB_B_MAIN)
    ap.add_argument("--no-poly", action="store_true",
                    help="the ablation arm isolating the Dual contribution: "
                         "CB head on a pruned PLAIN base, no poly branch")
    ap.add_argument("--retain", type=int, default=CB.CB_RETAIN)
    ap.add_argument("--rank", choices=("interp", "random"),
                    default="interp",
                    help="prune-ranking control; must match the prune stage")
    ap.add_argument("--k-cb", type=int, default=None,
                    help="ablation override of the concept-set default k_cb "
                         "(CONCEPT_SETS table); tags the run name '_kcb{v}'")
    ap.add_argument("--epochs", type=int, default=CB.CB_EPOCHS)
    ap.add_argument("--batch", type=int, default=CB.CB_BATCH)
    ap.add_argument("--lr", type=float, default=CB.CB_LR)
    ap.add_argument("--lam-span", type=float, default=CB.CB_LAM_SPAN)
    ap.add_argument("--warmup", type=int, default=CB.CB_WARMUP,
                    help="CB-first steps: poly frozen, span off")
    ap.add_argument("--seed", type=int, default=0,
                    help="CB-head init + shuffle seed (the driver passes the "
                         "base run's seed)")
    ap.add_argument("--no-center-loss", action="store_true",
                    help="literal Eq. 11 (no LF-CBM mean-centering)")
    ap.add_argument("--ckpt-every", type=int, default=CB.CB_CKPT_EVERY)
    ap.add_argument("--eval-every-epochs", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--tb-dir", type=Path, default=C.TB_DIR)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    use_poly = not args.no_poly
    center = not args.no_center_loss
    device = args.device

    # The CB run dir is owned by the grid cell keyed to the BASE run's seed;
    # cb_run_name does not encode --seed, so a hand-launched command with the
    # wrong (default) seed would occupy the cell with a wrong-seed model that
    # a collector then averages into the wrong slot. The base run name embeds
    # its seed — enforce agreement.
    m_seed = re.search(r"_seed(\d+)", args.run)
    if m_seed and int(m_seed.group(1)) != args.seed:
        raise SystemExit(
            f"--seed {args.seed} != the base run's seed {m_seed.group(1)} "
            f"({args.run}); pass --seed {m_seed.group(1)} (the driver does).")
    run = CB.cb_run_name(args.run, args.cset, args.b_cb, use_poly,
                         args.retain, args.k_cb, args.rank)
    out = C.CKPT_DIR / run
    if (out / "metrics.json").exists():
        print(f"[skip] {run} already trained")
        return
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- prune artifacts + pruned base --------------------------------------
    pdir = CB.prune_dir(args.run, args.cset, args.retain, args.rank)
    assert (pdir / "DONE").exists(), \
        f"prune incomplete for {pdir.name}"
    prune_meta = json.loads((pdir / "prune.json").read_text())
    dataset, encoder = prune_meta["dataset"], prune_meta["encoder"]
    T_cb, concepts = load_concept_embeds(pdir)
    T_cb = T_cb.to(device)
    n_cb, k_cb = concepts["n_cb"], concepts["k_cb"]
    if args.k_cb is not None:
        # Ablation override; the sizing-rule default stays in concepts.json.
        k_cb = int(args.k_cb)

    base = load_bsf(C.CKPT_DIR / CB.pruned_run_name(args.run, args.retain,
                                                   args.rank)
                    / "model.pt", device)
    assert base.G == args.retain, (base.G, args.retain)
    model = CBDualBSF(base, n_cb, k_cb, b_cb=args.b_cb,
                      use_poly=use_poly).to(device)
    model.lam_span = args.lam_span

    # ---- data ---------------------------------------------------------------
    def train_loader_fn(epoch: int):
        # Epoch-seeded generator -> bit-faithful mid-epoch resume.
        g = torch.Generator()
        g.manual_seed(args.seed * 10_000 + epoch)
        return build_paired_loader(dataset, encoder, "train",
                                   batch_size=args.batch, shuffle=True,
                                   num_workers=args.num_workers,
                                   drop_last=True, generator=g)

    steps_per_epoch = len(train_loader_fn(0))
    total_steps = steps_per_epoch * args.epochs

    def val_loader():
        return build_paired_loader(dataset, encoder,
                                   CB.probe_split(dataset),
                                   batch_size=8192, shuffle=False,
                                   num_workers=args.num_workers)

    # ---- optimizers: CB-SAE's 3-way partition + poly inside A ---------------
    # Optimizer DYNAMICS: three Adams at the default eps (1e-8), NO gradient
    # clipping. The base SAE protocol's clip=1.0 / eps=6.25e-10 governs base
    # training only — carrying it here
    # would be an undocumented hybrid: the cos^3 objectives produce small
    # gradients, so a 16x smaller Adam eps materially enlarges the effective
    # B/C step sizes relative to the reference, and per-objective clipping
    # rescales the three losses against each other (the reference relies on
    # the separate-Adams mechanism alone to balance them).
    p_enc = [model.E_cb.weight]
    p_dec = [model.D_cb]
    p_poly = list(model.base.poly.parameters()) if use_poly else []
    optA = torch.optim.Adam(p_enc + p_dec + p_poly, lr=args.lr)
    optB = torch.optim.Adam(p_enc, lr=args.lr)
    optC = torch.optim.Adam(p_dec, lr=args.lr)

    def zero_all():
        model.E_cb.weight.grad = None
        model.D_cb.grad = None
        for p in p_poly:
            p.grad = None

    # ---- resume -------------------------------------------------------------
    step, start_epoch, resume_skip = 0, 0, 0
    last = out / "ckpt_last.pt"
    if last.exists():
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        model._refresh_base_frames()          # stale QR cache after a reload
        optA.load_state_dict(st["optA"])
        optB.load_state_dict(st["optB"])
        optC.load_state_dict(st["optC"])
        step, start_epoch = st["step"], st["epoch"]
        resume_skip = st.get("batch_in_epoch", 0)
        print(f"[resume] step {step} epoch {start_epoch} skip {resume_skip}")

    from tensorboardX import SummaryWriter
    tb = SummaryWriter(str(args.tb_dir / run))

    print(f"run={run} |C|={n_cb} k_cb={k_cb} b_cb={args.b_cb} "
          f"poly={use_poly} steps={total_steps} "
          f"({steps_per_epoch}/epoch x {args.epochs})")

    usage = torch.zeros(n_cb, dtype=torch.long, device=device)
    window = 500
    t0 = time.time()
    model.train()                       # base stays eval (CBDualBSF.train)
    for epoch in range(start_epoch, args.epochs):
        for bi, (x_sub, x_clip) in enumerate(train_loader_fn(epoch)):
            if epoch == start_epoch and bi < resume_skip:
                continue
            x = x_sub.to(device, torch.float32, non_blocking=True)
            y = make_y(x_clip.to(device, non_blocking=True), T_cb)
            x_norm = model.normalize(x)
            in_warmup = step < args.warmup
            if use_poly:
                # Freeze/unfreeze as a pure function of `step` (resume-safe).
                model.base.poly.requires_grad_(not in_warmup)
            # Frozen branch once per batch — identical across A/B/C.
            with torch.no_grad():
                xB, sel_base = model.base_recon(x_norm)

            # -- A: reconstruction (+span): E_cb, D_cb, poly -----------------
            zero_all()
            c = model.concept_pre(x_norm)
            z_cb, sel_cb = model.sigma_cb(c)
            x_cb = model.cb_recon(z_cb)
            if use_poly and not in_warmup:
                r_hat = model.base.poly(x_norm - xB - x_cb)
                span = model.span_penalty(r_hat, sel_base, sel_cb)
                l_r = F.mse_loss(xB + x_cb + r_hat, x_norm) \
                    + args.lam_span * span
                poly_frac = (r_hat.pow(2).sum()
                             / (xB + x_cb + r_hat).pow(2).sum()
                             .clamp_min(1e-12)).detach()
            else:
                # CB-first phase (or the no-poly arm): steerable-only recon,
                # span exactly 0 — the dual-warmup idiom.
                l_r = F.mse_loss(xB + x_cb, x_norm)
                span = torch.zeros((), device=device)
                poly_frac = torch.zeros((), device=device)
            l_r.backward()
            optA.step()

            # -- B: interpretability on DENSE acts: E_cb only ----------------
            zero_all()
            l_int = cosine_cubed_loss(model.concept_dense_acts(x_norm), y,
                                      center=center)
            l_int.backward()
            optB.step()

            # -- C: steerability, cyclic through the DEPLOYMENT recon (no
            # poly), concept code detached, E_cb a frozen function: D_cb only
            zero_all()
            c_det = model.concept_pre(x_norm).detach()
            z_det, _ = model.sigma_cb(c_det)
            v_dep = xB + model.cb_recon(z_det)
            l_st = cosine_cubed_loss(model.concept_dense_acts(v_dep), y,
                                     center=center)
            l_st.backward()
            optC.step()

            usage += (z_det.norm(dim=-1) > 0).sum(0)
            step += 1
            if step % 100 == 0:
                tb.add_scalar("loss/recon", l_r.item(), step)
                tb.add_scalar("loss/int", l_int.item(), step)
                tb.add_scalar("loss/st", l_st.item(), step)
                tb.add_scalar("loss/span", float(span), step)
                tb.add_scalar("model/poly_frac", float(poly_frac), step)
                tb.add_scalar("model/cb_warmup", float(in_warmup), step)
            if step % window == 0:
                dead = float((usage == 0).float().mean())
                tb.add_scalar("model/dead_concept_frac", dead, step)
                usage.zero_()
                print(f"step {step}/{total_steps} recon={l_r.item():.4f} "
                      f"int={l_int.item():.4f} st={l_st.item():.4f} "
                      f"dead={dead:.3f} "
                      f"({(time.time() - t0) / max(step, 1):.3f}s/step)",
                      flush=True)
            if step % args.ckpt_every == 0:
                tmp = last.with_suffix(".tmp")
                torch.save({"model": model.state_dict(),
                            "optA": optA.state_dict(),
                            "optB": optB.state_dict(),
                            "optC": optC.state_dict(),
                            "step": step, "epoch": epoch,
                            "batch_in_epoch": bi + 1}, tmp)
                tmp.replace(last)

        if (epoch + 1) % args.eval_every_epochs == 0 \
                and epoch + 1 < args.epochs:
            hm = holdout_metrics(model, val_loader(), T_cb, device, center)
            for k in ("r2_base_only", "r2_deploy", "r2_full", "l_int",
                      "l_st", "dead_concept_frac", "cb_l0"):
                tb.add_scalar(f"eval/{k}", hm[k], step)
            print(f"epoch {epoch}: R2 base={hm['r2_base_only']:.4f} "
                  f"deploy={hm['r2_deploy']:.4f} full={hm['r2_full']:.4f} "
                  f"dead_cb={hm['dead_concept_frac']:.3f}", flush=True)

    # ---- final metrics + artifacts ------------------------------------------
    hm = holdout_metrics(model, val_loader(), T_cb, device, center)
    np.savez(out / "cb_block_stats.npz",
             fire_rate=hm.pop("fire_rate"),
             n_rows=hm["n_rows"], R=model.R, n_concepts=n_cb)
    save_cb_dual(model.cpu().eval(), out / "model.pt",
                 extra={"base_run": args.run,
                        "pruned_run": CB.pruned_run_name(args.run,
                                                         args.retain),
                        # Provenance: the prune dir is resolvable from these
                        # (a name parse would break on the optional tags).
                        "retain": args.retain, "k_cb_override": args.k_cb,
                        "rank": args.rank,
                        "dataset": dataset, "extractor": encoder,
                        "cset": args.cset, "seed": args.seed,
                        "concepts_sha": concepts["source_sha"],
                        "center_loss": center, "warmup": args.warmup,
                        "epochs": args.epochs})
    _atomic_json(out / "metrics.json", {
        "run": run, "variant": "cb_dual_bsf",
        "base_run": args.run, "dataset": dataset, "encoder": encoder,
        "seed": args.seed, "cset": args.cset,
        "retain": args.retain, "n_concepts": n_cb, "k_cb": k_cb,
        "b_cb": args.b_cb, "use_poly": use_poly,
        "lam_span": args.lam_span, "warmup": args.warmup,
        "epochs": args.epochs, "steps": step, "center_loss": center,
        # Holdout, normalized space, deployment gate.
        "heldout": {k: v for k, v in hm.items()},
        # Prune provenance carried here so the report never reopens prune.json
        # — and so r2_deploy is always read next to the fidelity caveat.
        "prune_exact": prune_meta["prune_fidelity"].get("prune_exact"),
        "prune_retention": prune_meta["prune_fidelity"].get("retention"),
        "effective_rank": prune_meta.get("effective_rank"),
        "wall_seconds": time.time() - t0,
    })
    tb.close()
    print(f"DONE {run}: R2 base={hm['r2_base_only']:.4f} "
          f"deploy={hm['r2_deploy']:.4f} full={hm['r2_full']:.4f} "
          f"dead_cb={hm['dead_concept_frac']:.3f} cb_l0={hm['cb_l0']:.2f}")


if __name__ == "__main__":
    main()
