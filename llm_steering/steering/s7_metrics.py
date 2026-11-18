"""s7 (judge-free metrics; D-50): the bge concept classifier and per-continuation metrics.

s7c  classifier (one unit, arm-free): per concept, a binary logistic regression (concept vs the
     rest of its corpus; class_weight='balanced' so P = 0.5 is the decision boundary whatever the
     base rate; sklearn defaults otherwise: L2, C = 1, lbfgs) on L2-normalised bge-large-en-v1.5
     vectors of the concept corpus's TRAIN split:
       agnews      the Track A bge LMDB (all 112,400 train rows; C31)
       goemotions  the Track A bge LMDB, single-label rows (as in s1)
       amazon      10,000 train reviews per class (shuffled as in s1), encoded here with the bge
                   snapshot BY LOCAL PATH through common.text_encoder.load_encoder
                   -- exactly the Track A encoding (sentence-transformers, native output, then the
                   LMDB loader's per-row L2 normalisation).
     Validated on the TEST split (AUROC, balanced accuracy), recorded next to the weights.
     P(concept) (the pre-registered co-primary for mode (i), PROTOCOL 4.4) = the classifier's
     predicted probability for the continuation's bge vector.
s7m  per generation unit and shard: P(concept) (null rows: every concept), perplexity of the
     continuation under CLEAN pythia, distinct-2 (unique / total whitespace-word bigrams of the
     continuation), and next-token KL(clean || steered) averaged over the positions that predict
     the continuation tokens, with the unit's steering edit applied exactly as in generation
     (same hook, same parameters).  Every teacher-forced pass uses explicit
     position_ids = cumsum(mask) - 1 (C30); pads are excluded.  A continuation with no token
     before EOS gets NaN perplexity / KL (counted).
     Since D-51's merge, s7m runs INSIDE the s6 task (s6_generate.run_unit passes its pythia);
     this module's --index/--unit CLI stays for manual re-scoring.
Precision (two protocols in one process, kept apart on purpose):
  * bge encodes run under TF32 -- the Track A encode protocol (config.ENCODE_DTYPE
    'fp32-tf32') of the LMDB vectors the classifier is trained on, so a continuation is embedded
    exactly like the training rows;
  * every pythia pass (perplexity, next-token KL) runs in FULL fp32 like generation (steer
    module docstring).  extract_text_embeddings.load_encoder switches TF32 on GLOBALLY when bge is
    first loaded, which would silently leave every later pythia pass in TF32; _bge_precision()
    therefore scopes TF32 to the encode call and restores full fp32 afterwards (checked by
    _assert_full_fp32 before each teacher-forced pass).
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from common import lang_config as config
from steering import base, concepts, s1_select, s6_generate, steer
from common import sae_io


# ----------------------------------------------------------------------------- bge
_BGE = None


@contextlib.contextmanager
def _bge_precision():
    """TF32 on for the bge encode (Track A protocol), then back to FULL fp32 (TF32 off), which is
    the only precision A16 runs pythia in (module docstring)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        steer.set_full_fp32()


def _assert_full_fp32() -> None:
    if torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("TF32 is on before a pythia pass; A16 scores pythia in full fp32")


def bge_encode(texts: List[str], device: str, batch_size: int = 256) -> np.ndarray:
    """Track A's bge protocol: sentence-transformers from the local snapshot, native output,
    then per-row L2 normalisation identical to the LMDB loader (sae_io.normalize_like_lmdb)."""
    global _BGE
    with _bge_precision():
        if _BGE is None:
            from common.text_encoder import load_encoder
            _BGE = load_encoder(config.TEXT_ENCODERS["bge"], device)
        emb = _BGE.encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=False)
    return sae_io.normalize_like_lmdb(np.asarray(emb, dtype=np.float32))


# ----------------------------------------------------------------------------- s7c
def train_classifiers(device: str, n_amazon: int = 10000, cap_per_class: int = 0) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    t0 = time.time()
    summary: Dict[str, Any] = {}
    for corpus in concepts.CORPORA:
        data = {}
        for split, n_am in (("train", n_amazon), ("test", max(1000, n_amazon // 10))):
            X, texts, lab = concepts.classifier_rows(corpus, split, n_am)
            if X is None:
                X = bge_encode(texts, device)
            if cap_per_class:
                if not base.IS_SMOKE:
                    raise RuntimeError("cap_per_class is a smoke option")
                keep = np.concatenate([np.flatnonzero(lab == c)[:cap_per_class] for c in np.unique(lab)])
                X, lab = X[keep], lab[keep]
            data[split] = (X, lab)
        W, b, rows = [], [], {}
        cids = [c.cid for c in concepts.CONCEPTS if c.corpus == corpus]
        for c in (concepts.CONCEPT_BY_ID[x] for x in cids):
            Xtr, ltr = data["train"]
            Xte, lte = data["test"]
            ytr, yte = (ltr == c.cls).astype(int), (lte == c.cls).astype(int)
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(Xte)[:, 1]
            W.append(clf.coef_[0].astype(np.float32))
            b.append(float(clf.intercept_[0]))
            rows[c.cid] = {"test_auroc": float(roc_auc_score(yte, p)),
                           "test_balanced_acc": float(balanced_accuracy_score(yte, (p > 0.5).astype(int))),
                           "n_train": int(ytr.size), "n_train_pos": int(ytr.sum()),
                           "n_test": int(yte.size), "n_test_pos": int(yte.sum()),
                           "converged_iter": int(np.max(clf.n_iter_))}
        base.atomic_save_npz(base.s7_clf(corpus), cids=np.asarray(cids), W=np.stack(W),
                             b=np.asarray(b, np.float32))
        summary[corpus] = rows
        base.log(f"s7c {corpus}: {rows}")
    summary["seconds"] = round(time.time() - t0, 1)
    summary["smoke"] = base.IS_SMOKE
    summary["complete"] = True
    base.atomic_write_json(base.p("s7", "clf", "_summary.json"), summary)
    return summary


def clf_done() -> bool:
    r = base.read_json(base.p("s7", "clf", "_summary.json"))
    return bool(r and r.get("complete")) and all(os.path.isfile(base.s7_clf(c)) for c in concepts.CORPORA)


def load_classifiers() -> Dict[str, Any]:
    out = {}
    for corpus in concepts.CORPORA:
        d = np.load(base.s7_clf(corpus))
        for i, cid in enumerate(d["cids"]):
            out[str(cid)] = (d["W"][i], float(d["b"][i]))
    return out


# ----------------------------------------------------------------------------- s7m
def distinct2(text: Optional[str]) -> float:
    w = re.findall(r"\S+", text or "")
    bi = list(zip(w, w[1:]))
    return len(set(bi)) / len(bi) if bi else float("nan")


@torch.no_grad()
def unit_metrics(unit: str, device: str, max_rows: int = 0, batch: int = 128, *, lm=None,
                 tok=None) -> Dict[str, Any]:
    t0 = time.time()
    clfs = load_classifiers()
    if lm is None:
        lm, tok = steer.load_lm(device)
    steer.set_full_fp32()
    refs = {r.tag: r for r in base.sae_refs()}
    adapter, layer = None, base.PILOT_LAYER
    if unit.startswith("armfree_L"):
        layer = int(unit.split("L")[1])
    elif unit != "null":
        ref = refs[unit]
        layer = int(ref.key[1:])
        model, meta = sae_io.load_arm(ref.run_dir, device)
        thr = base.read_json(base.s0_threshold(ref.tag))["threshold"] if meta["family"] == "batch_topk" else None
        adapter = steer.SAEAdapter(model, meta, thr)
    nf, _ = steer.norm_factor(layer)
    hook = steer.EditHook(lm, layer, nf, adapter)
    out_dir = base.s7m_dir(unit)
    stats = {"unit": unit, "shards": {}}
    for spath in sorted(glob.glob(os.path.join(base.s6_dir(unit), "*.jsonl"))):
        opath = os.path.join(out_dir, os.path.basename(spath))
        if os.path.isfile(opath):
            continue
        rows = base.read_jsonl(spath)
        if max_rows:
            if not base.IS_SMOKE:
                raise RuntimeError("max_rows is a smoke option")
            rows = rows[:max_rows]
        ok = [r for r in rows if r["status"] == "ok"]
        emb = bge_encode([r["text"] for r in ok], device) if ok else np.zeros((0, 1024), np.float32)
        vecmap = None
        if ok and ok[0]["mode"] == "i":
            cell = {"mode": "i", "dirset": ok[0]["dirset"]}
            vecmap = s6_generate._vec_lookup(layer, cell, adapter)
        res: Dict[int, Dict[str, Any]] = {}
        for s in range(0, len(ok), batch):
            chunk = ok[s:s + batch]
            prompts = [tok(r["prompt"], add_special_tokens=False)["input_ids"] for r in chunk]
            conts = [r["gen_ids"] for r in chunk]
            _assert_full_fp32()
            lp_clean = steer.teacher_forced_logp(lm, prompts, conts, None)
            kls: List[float] = [float("nan")] * len(chunk)
            if chunk[0]["mode"] != "null":
                slots = [({"latent": r["latent"], "dir_key": r["dir_key"], "cal": {"param": r["param"]}},
                          r["prompt_idx"], r["cont"], r["item_seed"]) for r in chunk]
                s6_generate._configure_rows(hook, chunk[0]["mode"], slots, nf, device, vecmap)
                with hook.installed():
                    lp_st = steer.teacher_forced_logp(lm, prompts, conts, hook)
                for i, (a_, b_) in enumerate(zip(lp_clean, lp_st)):
                    if a_.shape[0]:
                        kls[i] = float((a_.exp() * (a_ - b_)).sum(-1).mean())
            else:
                kls = [0.0] * len(chunk)
            for i, (r, lpc) in enumerate(zip(chunk, lp_clean)):
                n = lpc.shape[0]
                if n:
                    tgt = torch.as_tensor(r["gen_ids"], device=lpc.device)
                    nll = float(-lpc.gather(1, tgt[:, None]).mean())
                else:
                    nll = float("nan")
                res[id(r)] = {"nll": nll, "ppl": float(np.exp(nll)) if np.isfinite(nll) else float("nan"),
                              "kl_cont": kls[i] if n else float("nan")}
        out_rows = []
        k = 0
        for r in rows:
            o = {kk: r.get(kk) for kk in ("unit", "layer", "tag", "activation", "arm", "seed", "mode",
                                          "dirset", "budget", "headline", "concept", "rank", "latent",
                                          "dir_key", "param", "calib_kl", "prompt_idx", "cont",
                                          "status", "n_tokens", "hit_eos")}
            if r["status"] == "ok":
                e = emb[k]
                k += 1
                cids = [c.cid for c in concepts.CONCEPTS] if r["mode"] == "null" else [r["concept"]]
                o["p_concept"] = {cid: float(1.0 / (1.0 + np.exp(-(e @ clfs[cid][0] + clfs[cid][1]))))
                                  for cid in cids}
                o["distinct2"] = distinct2(r["text"])
                o.update(res[id(r)])
            out_rows.append(o)
        base.atomic_write_jsonl(opath, out_rows)
        stats["shards"][os.path.basename(spath)] = {"rows": len(rows), "ok": len(ok)}
    stats["seconds"] = round(time.time() - t0, 1)
    base.atomic_write_json(os.path.join(out_dir, "_summary.json"), stats)
    return stats


def unit_done(unit: str) -> bool:
    shards = glob.glob(os.path.join(base.s6_dir(unit), "*.jsonl"))
    return bool(shards) and all(os.path.isfile(os.path.join(base.s7m_dir(unit), os.path.basename(s)))
                                for s in shards)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-clf", action="store_true", help="s7c")
    ap.add_argument("--index", type=int, help="jobs.py s7m index (base.gen_units() order)")
    ap.add_argument("--unit")
    ap.add_argument("--device", default=None)
    ap.add_argument("--cap-per-class", type=int, default=0, help="smoke only (s7c)")
    ap.add_argument("--n-amazon", type=int, default=10000)
    ap.add_argument("--max-rows", type=int, default=0, help="smoke only (s7m)")
    a = ap.parse_args()
    dev = a.device or base.device_default()
    if a.train_clf:
        if clf_done() and not a.cap_per_class:
            print("[s7c] classifiers already trained")
            return 0
        s = train_classifiers(dev, a.n_amazon, a.cap_per_class)
        print(f"[s7c] done in {s['seconds']}s")
        return 0
    unit = a.unit or base.gen_units()[a.index]
    if unit_done(unit) and not a.max_rows:
        print(f"[s7m] {unit}: already done")
        return 0
    s = unit_metrics(unit, dev, a.max_rows)
    print(f"[s7m] {unit}: {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
