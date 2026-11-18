"""s1: concept residuals, per-latent AUROC selection, DiffMean and random directions (D-45, D-46,
C7, C31).  One unit = one layer (jobs.py kind 's1'): a single pythia pass over the concept
documents feeds every eligible SAE of that layer.

Per document (concepts.py: 2,000 train + 1,000 test docs per concept class, <= 128 tokens):
  * the layer-L residual of every real token at positions >= 1 (right padding, explicit position
    ids; the sink at position 0 is dropped, D-19) is ENCODED PER TOKEN with the SAE's inference
    encoder (BatchTopK: D-44 frozen threshold) and the per-token codes are MEAN-POOLED over those
    positions.  A pooled residual is never encoded (off-distribution for token SAEs; C31).
  * the raw residual is mean-pooled over the same positions for DiffMean.
Selection per SAE and concept (D-46): per-latent AUROC of the pooled code, concept vs the rest of
the concept's corpus, on the TRAIN docs; computed on the GPU (Mann-Whitney with mid-ranks, so the
many exact zeros tie correctly).  Candidates = ELIGIBLE latents from s0 (live on the non-sink
test rows, not a sink latent, not always-on); top 3 by AUROC (ties -> lower index).  The best and
the mean AUROC of the 3 are reported on train and on the held-out TEST docs.
Live-matched control (D-46): n_match = min eligible count over the 9 SAEs {baseline, dual,
dual_g0} x seeds of the (layer, activation); each SAE draws ONE seeded subset of n_match eligible
latents (seed from its tag) and runs the same top-3 selection inside it.
DiffMean (D-47): unit(mean pooled residual of the concept's train docs - mean of the rest).
Random: 5 unit Gaussian directions per (layer, concept) from a D-22 seed (data-independent).
Only AUROC tables, selections and directions are saved (C31).

Units are resumable per SAE: the layer worker skips SAEs whose selection file exists and the
arm-free file when present, so a later JumpReLU release only adds the missing SAEs.
Live-matched refresh: n_match needs the s0 records of all 9 SAEs of the (layer, activation).
When some of them cannot exist yet (untrained / held JumpReLU cells released with
A16_ALLOW_JUMPRELU=1) the selection is written with livematched 'pending'.  Such a selection is
STALE as soon as n_match becomes computable: the layer unit is then 'todo' again and fills the
live-matched rows from the SAVED per-latent AUROC table (no LM pass; the full-dictionary
selection is left byte-identical, so nothing calibrated or generated from it is invalidated);
calibrate.py then adds only the new directions and s6 only the new live-matched cell.
D-18: SAEs excluded by gates43.excluded_reason (a collapsed TopK baseline at the layer) are not
selected at all ("its TopK pair is not steered").
Gate version: selection follows the PRIMARY D-43 gate (gates43 default; rule (b) is a diagnostic).  A selection is a function of the SAE and the concept data only;
the gate decides WHETHER an SAE is selected, never what is selected.  So a gate-rule change can
never make an existing selection wrong: is_done() re-reads the live verdict, an SAE that newly
passes has no selection yet and makes its layer 'todo', and one that newly fails is refused
downstream by s5/s6/s7 (gates43.sae_excluded).  Each record stores the rule it was selected under
('gate_rule', provenance only).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from steering import base, concepts, gates43, s0_rescore, steer
from common import arms, sae_io, seeds as _seeds
from common import lang_config as config
from common import loss_recovered_hooks as H


# ----------------------------------------------------------------------------- AUROC (GPU)
@torch.no_grad()
def auroc_cols(S: torch.Tensor, pos: torch.Tensor, col_chunk: int = 2048) -> torch.Tensor:
    """AUROC of every column of S (N, F) for the boolean labels ``pos`` (N,): Mann-Whitney U with
    mid-ranks (ties share the average rank), i.e. P(s_pos > s_neg) + 0.5 P(tie)."""
    n1 = int(pos.sum())
    n0 = int(pos.numel() - n1)
    if n1 == 0 or n0 == 0:
        raise ValueError("AUROC needs both classes")
    out = []
    for c in range(0, S.shape[1], col_chunk):
        St = S[:, c:c + col_chunk].T.contiguous().double()          # (f, N)
        srt, _ = St.sort(dim=1)
        lo = torch.searchsorted(srt, St, right=False)
        hi = torch.searchsorted(srt, St, right=True)
        rank = (lo + hi + 1).double() / 2.0                         # 1-based mid-rank
        r_pos = rank[:, pos].sum(1)
        out.append(((r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)).float())
    return torch.cat(out)


# ----------------------------------------------------------------------------- eligibility
def layer_sae_plan(layer: int) -> Tuple[List[Any], Dict[str, str]]:
    """(SAEs to select for, {tag: why skipped}) at one layer."""
    todo, skipped = [], {}
    verdicts: Dict[str, Dict[str, Any]] = {}          # one pair verdict per activation
    for ref in base.sae_refs():
        if int(ref.key[1:]) != layer:
            continue
        why = base.held_reason(ref)
        if why:
            skipped[ref.tag] = why
            continue
        if not s0_rescore.is_done(ref.tag):
            skipped[ref.tag] = "no s0 record"
            continue
        v = verdicts.setdefault(ref.activation, gates43.pair_verdict(layer, ref.activation))
        st = v["sae"][ref.tag]["status"]
        if st != "pass":
            skipped[ref.tag] = f"D-43 gate {st}"
            continue
        ex = gates43.excluded_reason(v, ref.tag)
        if ex:
            skipped[ref.tag] = f"excluded: {ex}"
            continue
        todo.append(ref)
    return todo, skipped


def n_match(layer: int, activation: str) -> Optional[int]:
    vals = []
    for arm in arms.ARMS:
        for s in arms.SEEDS:
            ref = arms.resolve_arm("B", layer, None, activation, arm, s)
            rec = base.read_json(base.s0_record(ref.tag)) if ref.tag else None
            if not rec or not rec.get("complete"):
                return None
            vals.append(int(rec["rows"]["n_eligible"]))
    return min(vals)


def livematched_entry(tag: str, nm_: Optional[int], a_tr: torch.Tensor, a_te: torch.Tensor,
                      elig: np.ndarray) -> Dict[str, Any]:
    """The D-46 live-matched selection of one SAE and concept: one seeded subset of n_match
    eligible latents (seed from the SAE tag, the same draw for every concept) and the top-3 by
    TRAIN AUROC inside it.  Shared by the layer pass and the stale refresh so both give the same
    answer from the same AUROC values."""
    if nm_ is None:
        return {"status": "pending", "why": "n_match needs all 9 s0 records"}
    cand = np.flatnonzero(elig)
    rng = _seeds.rng(_seeds.derive_seed("a16.livematch", tag))
    subset = np.sort(rng.choice(cand, size=min(nm_, cand.size), replace=False))
    top_lm = select_top(a_tr, subset)
    return {"status": "ok", "n_match": int(nm_), "latents": top_lm,
            "train_auroc": [float(a_tr[j]) for j in top_lm],
            "test_auroc": [float(a_te[j]) for j in top_lm]}


def livematched_stale(ref) -> bool:
    """A written selection whose live-matched rows are 'pending' although n_match is now
    computable (module docstring)."""
    sel = load_selection(ref.tag)
    if not sel:
        return False
    pend = any(e.get("livematched", {}).get("status") != "ok" for e in sel["concepts"].values())
    return pend and n_match(int(ref.key[1:]), ref.activation) is not None


def refresh_livematched(ref) -> bool:
    """Fill the pending live-matched rows of an existing selection from its saved AUROC table
    (CPU, seconds).  The full-dictionary entries are not touched.  Returns True if rewritten."""
    if not livematched_stale(ref):
        return False
    sel = load_selection(ref.tag)
    nm_ = n_match(int(ref.key[1:]), ref.activation)
    auc = np.load(base.s1_auroc(ref.tag))
    elig = s0_rescore.load_latents(ref.tag)["eligible"].astype(bool)
    for cid, e in sel["concepts"].items():
        if e.get("livematched", {}).get("status") == "ok":
            continue
        a_tr = torch.from_numpy(np.asarray(auc[f"train__{cid}"]))
        a_te = torch.from_numpy(np.asarray(auc[f"test__{cid}"]))
        e["livematched"] = dict(livematched_entry(ref.tag, nm_, a_tr, a_te, elig),
                                refreshed_from_saved_auroc=True)
    sel["livematched_refreshed"] = True
    base.atomic_write_json(base.s1_select(ref.tag), sel)
    base.log(f"s1 {ref.tag}: live-matched rows filled (n_match {nm_}) from the saved AUROC table")
    return True


def select_top(auc: torch.Tensor, candidates: np.ndarray, k: int = base.TOP_LATENTS) -> List[int]:
    """Top-k candidate latents by AUROC, ties broken by the lower latent index (stable)."""
    cand = np.sort(np.asarray(candidates, dtype=np.int64))
    if cand.size == 0:
        return []
    vals = auc[torch.as_tensor(cand, device=auc.device)].cpu().numpy()
    order = np.argsort(-vals, kind="stable")
    return cand[order[:k]].tolist()


def random_dirs(layer: int, cid: str, n: int = base.N_RANDOM, d_model: Optional[int] = None) -> np.ndarray:
    # d_model defaults to the LM residual width (1024 for pythia-410m-deduped).
    d_model = config.LM_D_MODEL if d_model is None else d_model
    out = np.zeros((n, d_model), np.float32)
    for r in range(n):
        g = torch.Generator(device="cpu")
        g.manual_seed(_seeds.derive_seed("a16.random", f"L{layer}|{cid}|{r}"))
        v = torch.randn(d_model, generator=g, dtype=torch.float64)
        out[r] = (v / v.norm()).numpy()
    return out


# ----------------------------------------------------------------------------- the layer pass
@torch.no_grad()
def run_layer(layer: int, device: str, n_train: int = concepts.N_TRAIN_PER_CLASS,
              n_test: int = concepts.N_TEST_PER_CLASS, doc_batch: int = 64,
              only_concepts: Optional[List[str]] = None, force: bool = False,
              smoke_tags: Optional[List[str]] = None) -> Dict[str, Any]:
    t0 = time.time()
    gates43.write_all()              # per-SAE D-43 gate JSONs on disk before any selection
    todo, skipped = layer_sae_plan(layer)
    if smoke_tags:
        # Smoke only: select for these SAEs whatever their D-43 state (a smoke has s0 records for
        # one seed, so every Dual's relative-CE check is 'pending' by construction).
        if not base.IS_SMOKE:
            raise RuntimeError("--smoke-tags is only allowed under an A16_ROOT smoke root")
        todo = [r for r in base.sae_refs() if r.tag in smoke_tags]
    if not force:
        for r in todo:
            refresh_livematched(r)           # stale live-matched rows: CPU refill, no LM pass
        todo = [r for r in todo if base.read_json(base.s1_select(r.tag)) is None]
    need_armfree = force or not base.read_json(base.s1_armfree(layer).replace(".npz", ".json"))
    base.log(f"s1 L{layer}: {len(todo)} SAEs to select, arm-free {'todo' if need_armfree else 'done'}; "
             f"skipped {len(skipped)}")
    if not todo and not need_armfree:
        return {"layer": layer, "status": "done"}
    lm, _ = steer.load_lm(device)
    nf, _ = steer.norm_factor(layer)
    module = lm.gpt_neox.layers[layer]
    sae = {}
    for ref in todo:
        model, meta = sae_io.load_arm(ref.run_dir, device)
        thr = None
        if meta["family"] == "batch_topk":
            thr = base.read_json(base.s0_threshold(ref.tag))["threshold"]
        sae[ref.tag] = (ref, steer.SAEAdapter(model, meta, thr),
                        s0_rescore.load_latents(ref.tag)["eligible"].astype(bool))
    concept_list = [c for c in concepts.CONCEPTS if only_concepts is None or c.cid in only_concepts]
    per_sae: Dict[str, Dict[str, Any]] = {t: {"concepts": {}} for t in sae}
    auroc_tab = {t: {"train": {}, "test": {}} for t in sae}
    armfree: Dict[str, Any] = {"diffmean": {}, "diffmean_norm": {}, "diffmean_auroc_test": {},
                               "diffmean_auroc_train": {}, "random": {}, "counts": {}}
    for corpus in concepts.CORPORA:
        cs = [c for c in concept_list if c.corpus == corpus]
        if not cs:
            continue
        docs = concepts.ensure_concept_docs(corpus, n_train, n_test)
        N = docs["ids"].shape[0]
        pooled_raw = torch.zeros((N, config.LM_D_MODEL), device=device, dtype=torch.float64)
        pooled = {t: torch.zeros((N, a.W.shape[0]), device=device, dtype=torch.float32)
                  for t, (_, a, _) in sae.items()}
        for s in range(0, N, doc_batch):
            ids = torch.from_numpy(docs["ids"][s:s + doc_batch].astype(np.int64)).to(device)
            lens = torch.from_numpy(docs["lens"][s:s + doc_batch].astype(np.int64)).to(device)
            B, T = ids.shape
            ar = torch.arange(T, device=device)[None].expand(B, T)
            att = (ar < lens[:, None]).long()
            store: List[torch.Tensor] = []
            with H.capture_activation(module, store):
                lm.gpt_neox(input_ids=ids, attention_mask=att, position_ids=ar, use_cache=False)
            hid = store[0].float()
            m = (ar >= 1) & att.bool()
            bi, ti = torch.nonzero(m, as_tuple=True)
            x_raw = hid[bi, ti]
            cnt = m.sum(1).double()
            doc = bi + s
            pooled_raw.index_add_(0, doc, x_raw.double())
            x_s = H.to_shard_space(x_raw, nf, "divide")
            for t, (_, A, _) in sae.items():
                z = A.encode(x_s)
                pooled[t].index_add_(0, doc, z)
            for t in pooled:
                pooled[t][s:s + B] /= cnt[:, None].float()
            pooled_raw[s:s + B] /= cnt[:, None]
        armfree["counts"][corpus] = {
            "n_docs": int(N), "n_dropped_short": int(docs["n_dropped_short"]),
            "per_class": {str(docs["classes"][ci]): {
                "train": int(((docs["cls"] == ci) & (docs["split"] == 0)).sum()),
                "test": int(((docs["cls"] == ci) & (docs["split"] == 1)).sum())}
                for ci in range(len(docs["classes"]))}}
        split_t = torch.from_numpy(docs["split"]).to(device)
        for c in cs:
            pm_tr, nm_tr = (torch.from_numpy(m_).to(device) for m_ in concepts.concept_masks(docs, c, 0))
            pm_te, nm_te = (torch.from_numpy(m_).to(device) for m_ in concepts.concept_masks(docs, c, 1))
            # DiffMean on TRAIN docs, in raw space (= shard space up to the scalar nf).
            dm = pooled_raw[pm_tr].mean(0) - pooled_raw[nm_tr].mean(0)
            armfree["diffmean_norm"][c.cid] = float(dm.norm())
            u = (dm / dm.norm()).float()
            armfree["diffmean"][c.cid] = u.cpu().numpy()
            for split_name, pm, nm in (("train", pm_tr, nm_tr), ("test", pm_te, nm_te)):
                sel = pm | nm
                proj = (pooled_raw[sel].float() @ u)[:, None]
                armfree[f"diffmean_auroc_{split_name}"][c.cid] = float(auroc_cols(proj, pm[sel])[0])
            armfree["random"][c.cid] = random_dirs(layer, c.cid)
            for t, (ref, A, elig) in sae.items():
                sel_tr, sel_te = pm_tr | nm_tr, pm_te | nm_te
                a_tr = auroc_cols(pooled[t][sel_tr], pm_tr[sel_tr])
                a_te = auroc_cols(pooled[t][sel_te], pm_te[sel_te])
                auroc_tab[t]["train"][c.cid] = a_tr.cpu().numpy()
                auroc_tab[t]["test"][c.cid] = a_te.cpu().numpy()
                top = select_top(a_tr, np.flatnonzero(elig))
                entry = {"latents": top,
                         "train_auroc": [float(a_tr[j]) for j in top],
                         "test_auroc": [float(a_te[j]) for j in top],
                         "atom_norm": [float(A.atom_norm[j]) for j in top]}
                for sp in ("train", "test"):
                    v = entry[f"{sp}_auroc"]
                    entry[f"{sp}_auroc_best"] = max(v) if v else float("nan")
                    entry[f"{sp}_auroc_mean"] = float(np.mean(v)) if v else float("nan")
                # live-matched control (pending when n_match is not computable yet; refreshed
                # later from the saved AUROC table by refresh_livematched)
                entry["livematched"] = livematched_entry(t, n_match(layer, ref.activation),
                                                         a_tr, a_te, elig)
                per_sae[t]["concepts"][c.cid] = entry
        del pooled, pooled_raw
        torch.cuda.empty_cache()
    for t, (ref, A, elig) in sae.items():
        base.atomic_save_npz(base.s1_auroc(t), **{
            f"{sp}__{cid}": v for sp in ("train", "test") for cid, v in auroc_tab[t][sp].items()})
        rec = {"tag": t, "layer": layer, "activation": ref.activation, "arm": ref.arm,
               "seed": ref.seed, "n_eligible": int(elig.sum()), "concepts": per_sae[t]["concepts"],
               "n_train_per_class": n_train, "n_test_per_class": n_test, "smoke": base.IS_SMOKE,
               "concept_registry_sha256": concepts.concept_registry_sha256(),
               # provenance only (module docstring): the D-43 rule this SAE was selected under
               "gate_rule": gates43.RULE_ID[gates43.PRIMARY], "complete": True}
        base.atomic_write_json(base.s1_select(t), rec)
    if need_armfree:
        cids = sorted(armfree["diffmean"])
        base.atomic_save_npz(base.s1_armfree(layer),
                             concept_ids=np.asarray(cids),
                             diffmean=np.stack([armfree["diffmean"][c] for c in cids]),
                             random=np.stack([armfree["random"][c] for c in cids]))
        base.atomic_write_json(base.s1_armfree(layer).replace(".npz", ".json"), {
            "layer": layer, "concept_ids": cids,
            "diffmean_norm_raw": armfree["diffmean_norm"],
            "diffmean_auroc_train": armfree["diffmean_auroc_train"],
            "diffmean_auroc_test": armfree["diffmean_auroc_test"],
            "counts": armfree["counts"], "n_random": base.N_RANDOM,
            "n_train_per_class": n_train, "n_test_per_class": n_test, "smoke": base.IS_SMOKE,
            "complete": True})
    base.log(f"s1 L{layer}: done in {time.time() - t0:.1f}s")
    return {"layer": layer, "status": "ran", "seconds": round(time.time() - t0, 1),
            "selected": list(sae), "skipped": skipped}


def is_done(layer: int) -> bool:
    af = base.read_json(base.s1_armfree(layer).replace(".npz", ".json"))
    if not (af and af.get("complete")):
        return False
    todo, _ = layer_sae_plan(layer)
    return all(base.read_json(base.s1_select(r.tag)) is not None and not livematched_stale(r)
               for r in todo)


def load_selection(tag: str) -> Optional[Dict[str, Any]]:
    return base.read_json(base.s1_select(tag))


def load_armfree(layer: int) -> Dict[str, Any]:
    d = dict(np.load(base.s1_armfree(layer)))
    d["meta"] = base.read_json(base.s1_armfree(layer).replace(".npz", ".json"))
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, help="jobs.py s1 index (0,1,2 = L6,L12,L18)")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-train", type=int, default=concepts.N_TRAIN_PER_CLASS)
    ap.add_argument("--n-test", type=int, default=concepts.N_TEST_PER_CLASS)
    ap.add_argument("--concepts", nargs="*", default=None, help="smoke: subset of concept ids")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke-tags", nargs="*", default=None)
    a = ap.parse_args()
    layer = a.layer if a.layer is not None else base.LAYERS[a.index]
    res = run_layer(layer, a.device or base.device_default(), a.n_train, a.n_test,
                    only_concepts=a.concepts, force=a.force, smoke_tags=a.smoke_tags)
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
