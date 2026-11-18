"""Per-token SAE activations over input sentences, for the residual-stream SAE cells.

WHAT THIS PRODUCES, and why it is worth producing: every quantitative number is an average
over ~200k tokens. An average cannot show WHICH words a latent responds to, whether it fires on
a single token or spans a phrase, or where DualSAE's poly branch is doing the work the mono
branch did not. This module returns, per sentence:

    tokens      the decoded sub-word strings, in order
    z           (T, n_latents) sparse code -- the MONO branch for a DualSAE
    mono_norm   (T,) ||x^M||, how much of the activation the sparse dictionary explains
    poly_norm   (T,) ||r_hat||, DUAL ONLY -- where the dense residual branch is working
    resid_norm  (T,) ||x||, the raw activation magnitude, for normalising the two above

EXACTNESS. The activations must be the ones the quantitative tables were computed on, or the
figure illustrates a different model. Three things are therefore reused rather than
reimplemented: `ResidCapture` (the same hook, taking the block OUTPUT, not its input),
the cell's own norm factor read from its shard manifest (an expectation over that corpus'
training tokens -- using any other scale silently puts the input in a space the SAE never saw),
and `forward_reps` from the vision core's knn_lp_sparse.py, which handles the preprocess/postprocess inversion
that makes x^M and r_hat comparable to x.

ONE DELIBERATE DIFFERENCE from the quantitative path. BatchTopK selects its top-k over a WHOLE
batch, so a token's code depends on which other tokens share its batch. For a figure that must
be read per sentence, this module encodes ONE SENTENCE AT A TIME, so the code shown is the one
that sentence alone produces. That is the honest choice for a qualitative display, and it is
recorded in the output so a reader cannot mistake it for the L0=64-per-batch protocol.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from pathlib import Path as _Path
# language/ (for the langdual package) and <repo>/vision (the SAE core, imported by bare
# module name) on sys.path, derived from this file's location.
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "vision"))

from langdual import assets, config                      # noqa: E402
from langdual.lmsae import extract_resid as ER           # noqa: E402
from langdual.lmsae import loss_recovered_hooks as H     # noqa: E402


@dataclass
class SentenceMap:
    """Everything needed to render one sentence under one model."""
    text: str
    tokens: List[str]
    z: np.ndarray                      # (T, n_latents) float32
    resid_norm: np.ndarray             # (T,)
    mono_norm: np.ndarray              # (T,)
    poly_norm: Optional[np.ndarray]    # (T,) or None for a single-branch model
    meta: Dict[str, Any] = field(default_factory=dict)


def load_lm(device: torch.device, dtype: str = "float32"):
    """Pythia + tokenizer, strictly offline (local snapshot from prepare_assets.sh)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = assets.local_snapshot(config.LM_REPO)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype]
    lm = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                                              torch_dtype=torch_dtype)
    lm.to(device).eval()
    return lm, tok


def load_cell(tag: str, checkpoint: str = "best", device: Optional[torch.device] = None):
    """Rebuild one trained Track B SAE from its run directory.

    Returns ``(model, meta, layer, corpus, norm_factor, convention)``.  The norm factor comes
    from the SHARD MANIFEST of the cell's own corpus and layer, never recomputed: it is the
    scale the stored activations were folded by, and the SAE only understands that space.
    """
    from knn_lp_sparse import load_run          # the vision core is on sys.path
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cell = next((c for c in config.iter_cells_b() if c.tag == tag), None)
    if cell is None:
        # Ablation cells live in their own registry and their own run root; supporting them
        # here lets a page be drawn for an ablation Dual (e.g. a different warm-up length).
        from langdual import ablate
        cell = next((c for c in ablate.iter_cells() if c.tag == tag), None)
    if cell is None:
        raise KeyError(f"no Track B cell with tag {tag!r} in either registry")
    model, meta = load_run(__import__("pathlib").Path(cell.run_dir), device, checkpoint=checkpoint)
    layer = int(cell.key.lstrip("L"))
    manifest = H.read_shard_manifest(layer, "test", cell.corpus)
    nf, conv = H.norm_factor_from_manifest(manifest, verbose=False)
    meta = dict(meta, tag=tag, layer=layer, corpus=cell.corpus, checkpoint=checkpoint,
                norm_factor=nf, norm_convention=conv, is_dual=(meta.get("model_type") == "dual"))
    return model, meta


@torch.no_grad()
def map_sentences(sentences: Sequence[str], lm, tok, sae, meta: Dict[str, Any],
                  device: torch.device, max_tokens: int = 128,
                  joint_batch: bool = True, filler_rows: int = 0,
                  filler_split: str = "test") -> List[SentenceMap]:
    """Per-token activations for each sentence.

    ``joint_batch`` controls the ENCODE step, not the forward pass: activations are always
    captured one sentence at a time, but with joint_batch the whole probe set is encoded as a
    single batch and the code is split back per sentence afterwards.

    THIS IS NOT COSMETIC FOR BATCHTOPK. BatchTopK selects its top-k over an entire batch, so
    with one short sentence per batch the budget is dominated by the highest-magnitude token --
    which is the attention sink, at ~50x the norm of everything else. Measured on this probe
    set: 619 of the 772 latents a Dual-BatchTopK fires are sink latents under per-sentence
    encoding, leaving 2-3 usable features. Encoding jointly restores something close to the
    training-time selection (where the sink is ~1/128 of the batch rather than ~1/12) and is
    therefore the fair setting for any BatchTopK comparison. TopK is per-token and unaffected
    either way; the flag is kept so the difference can be demonstrated rather than assumed.
    """
    from knn_lp_sparse import forward_reps
    layer, nf, conv = meta["layer"], meta["norm_factor"], meta["norm_convention"]
    is_dual = bool(meta["is_dual"])
    out: List[SentenceMap] = []
    captured: List[tuple] = []
    cap = ER.ResidCapture(lm, [layer])
    try:
        for text in sentences:
            enc = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens)
            ids = enc["input_ids"].to(device)
            cap._buf.clear()
            try:
                lm(input_ids=ids, use_cache=False)
            except ER._StopForward:
                pass                      # expected: the hook aborts the rest of the forward
            hidden = cap._buf[layer][0]   # (T, D)
            # Fold the SAME scale the shards carry.  "divide" is the convention the extractor
            # writes; anything else would be a different space.
            x = hidden.float() / nf if conv == "divide" else hidden.float() * nf
            toks = [tok.decode([int(i)]) for i in ids[0].tolist()]
            captured.append((text, toks, x))
    finally:
        for h in cap._handles:
            h.remove()

    def _emit(text, toks, reps, sl):
        z = reps["z"][sl].detach().float().cpu().numpy()
        return SentenceMap(
            text=text, tokens=toks, z=z, meta=dict(meta, joint_batch=joint_batch),
            resid_norm=reps["x"][sl].norm(dim=-1).cpu().numpy(),
            mono_norm=reps["xM"][sl].detach().float().norm(dim=-1).cpu().numpy(),
            poly_norm=(reps["r_hat"][sl].detach().float().norm(dim=-1).cpu().numpy()
                       if is_dual and "r_hat" in reps else None))

    if joint_batch:
        parts = [x for _t, _k, x in captured]
        n_probe = sum(x.shape[0] for x in parts)
        if filler_rows > 0:
            # REAL CORPUS ROWS, appended purely to fix the batch COMPOSITION. BatchTopK's top-k
            # is taken over the whole batch, so the fraction of attention-sink tokens in that
            # batch changes which latents win the budget. A bare probe set is ~8% sink; a 2048-row
            # training batch is ~0.8%. Padding to the training batch size reproduces the
            # selection the model was trained under. The filler rows are DISCARDED before
            # rendering -- they exist to set the competition, not to be shown.
            fill = _filler_rows(meta, filler_rows, parts[0].device)
            if fill is not None:
                parts = parts + [fill]
        big = torch.cat(parts, dim=0)
        reps = forward_reps(sae, big, is_dual)
        off = 0
        for text, toks, x in captured:
            out.append(_emit(text, toks, reps, slice(off, off + x.shape[0])))
            off += x.shape[0]
    else:
        for text, toks, x in captured:
            reps = forward_reps(sae, x, is_dual)
            out.append(_emit(text, toks, reps, slice(0, x.shape[0])))
    return out


# THE ATTENTION SINK, and why every selector below has to know about it.
#
# Measured on pythia-410m block 12, this probe set: the FIRST position carries 81.8% of the
# total activation energy, with mean ||x|| = 8.01 against ~0.16 at every later position, a 50x
# gap. Worse for our purposes, it fires the SAME 64 latents in all 10 sentences -- literally
# zero content sensitivity -- while later positions use 227 distinct latents. Ranking latents by
# peak activation therefore returns the sink set every time, and a colour scale normalised to a
# sentence maximum paints position 0 solid and everything else white.
#
# The sink is a real property of the model, not a bug in this code, and the quantitative tables
# are unaffected (they average over ~200k tokens and the sink is a fixed ~1/128 of them). It is
# only fatal for a QUALITATIVE figure, whose entire job is to show content sensitivity.
SINK_POSITIONS = 1


def sink_latents(maps: Sequence[SentenceMap], positions: int = SINK_POSITIONS,
                 min_share: float = 0.9) -> List[int]:
    """Latents that fire at a sink position in at least `min_share` of sentences.

    Identified from the data rather than hard-coded, because the set is a property of the model
    and the layer, not a constant we can assume.
    """
    if not maps:
        return []
    hits = np.zeros(maps[0].z.shape[1], dtype=np.float32)
    for m in maps:
        hits += (m.z[:positions] > 0).any(axis=0).astype(np.float32)
    return [int(i) for i in np.nonzero(hits >= min_share * len(maps))[0]]


def _filler_rows(meta: Dict[str, Any], n: int, device) -> Optional[torch.Tensor]:
    """`n` real activation rows from the cell's own shard cache, already in shard space.

    Its own corpus and layer, never another: the rows only have to make the batch statistics
    realistic, and rows from a different distribution would set a different competition.
    """
    import glob
    base = config.resid_dir(meta["layer"], "test", meta["corpus"])
    files = sorted(glob.glob(os.path.join(base, "*.npy")))
    if not files:
        return None
    arr = np.load(files[0], mmap_mode="r")
    take = min(int(n), arr.shape[0])
    return torch.from_numpy(np.array(arr[:take])).float().to(device)


def top_latents(maps: Sequence[SentenceMap], n: int = 12, min_tokens: int = 1,
                skip_positions: int = SINK_POSITIONS,
                exclude_sink: bool = True,
                max_fire_rate: Optional[float] = 0.30,
                dedupe: Optional[float] = 0.9) -> List[int]:
    """The n latents with the largest peak activation, IGNORING the sink positions.

    Peak rather than mean, because a feature that fires hard on two tokens of one sentence is
    exactly what a qualitative figure is for, and a mean would rank it below a latent that fires
    weakly everywhere. But peak over ALL positions returns the attention sink and nothing else
    (see SINK_POSITIONS above), so the first `skip_positions` tokens are excluded from the
    ranking and any latent that is part of the constant sink set is dropped outright.
    """
    width = maps[0].z.shape[1]
    peak = np.zeros(width, dtype=np.float32)
    count = np.zeros_like(peak)
    for m in maps:
        body = m.z[skip_positions:]
        if body.size == 0:
            continue
        peak = np.maximum(peak, body.max(axis=0))
        count += (body > 0).sum(axis=0)
    peak[count < min_tokens] = -np.inf
    if exclude_sink:
        peak[np.asarray(sink_latents(maps, skip_positions), dtype=int)] = -np.inf

    # SELECTIVITY GATE. Peak alone is not enough: after the sink is removed the next-ranked
    # latents are "always-on" ones that fire weakly on EVERY token of every sentence. Those are
    # not features either -- they carry no information about which word they are on, and a
    # shaded row of uniform colour tells a reader nothing. A latent is kept only if it fires on
    # at most `max_fire_rate` of non-sink tokens, which is the operational content of "this
    # feature responds to something in particular".
    n_body = sum(max(0, len(m.z) - skip_positions) for m in maps)
    if n_body and max_fire_rate is not None:
        rate = count / float(n_body)
        peak[rate > max_fire_rate] = -np.inf
    order = [int(i) for i in np.argsort(-peak) if np.isfinite(peak[i])]
    if not dedupe:
        return order[:n]

    # DIVERSITY. Ranking alone returns near-duplicates: at layer 12 the top three selective
    # latents are all newline detectors, which makes three identical-looking panels. A latent is
    # skipped when its firing pattern across the whole token set correlates above `dedupe` with
    # one already chosen, so the figure spends its rows on distinct behaviours.
    pattern = np.concatenate([m.z[skip_positions:] for m in maps if len(m.z) > skip_positions])
    chosen: List[int] = []
    for i in order:
        vi = pattern[:, i]
        if vi.std() == 0:
            continue
        if any(abs(float(np.corrcoef(vi, pattern[:, j])[0, 1])) > dedupe for j in chosen):
            continue
        chosen.append(i)
        if len(chosen) >= n:
            break
    return chosen


def decoder_matrix(sae) -> torch.Tensor:
    """The decoder as ``(n_latents, d_model)``, whatever the family calls it.

    The three families disagree, and one of them is a trap. DualSAE and BatchTopKSAE expose
    ``W_dec`` already in (n_latents, d_model). TopKSAE has no ``W_dec`` at all: its decoder is
    TIED to the encoder and surfaces as ``decoder.weight`` in the TRANSPOSED (d_model, n_latents)
    layout. Reading that without transposing would silently compare a 1024-long slice of the
    wrong axis and produce cosine values that look plausible and mean nothing.
    """
    if hasattr(sae, "W_dec"):
        W = sae.W_dec
        return W.detach().float()                      # (n_latents, d_model)
    dec = getattr(sae, "decoder", None)
    if dec is not None and hasattr(dec, "weight"):
        return dec.weight.detach().float().t()         # (d_model, n_latents) -> transpose
    raise AttributeError(
        f"{type(sae).__name__} exposes neither W_dec nor decoder.weight; add it here rather "
        f"than guessing an axis")


def match_latent_by_decoder(src_sae, dst_sae, latent: int, topn: int = 1) -> List[int]:
    """Latents of dst whose decoder direction best matches src's `latent`.

    Used to put the SAME direction side by side across a baseline/dual pair. Ranking
    independently in each model would show each model's own best feature, which answers a
    different question than "what did the two models do with the same direction".
    """
    a = decoder_matrix(src_sae)[latent]
    B = decoder_matrix(dst_sae)
    if a.shape[-1] != B.shape[-1]:
        raise ValueError(f"decoder widths differ: {a.shape[-1]} vs {B.shape[-1]}")
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), B, dim=-1)
    return [int(i) for i in torch.argsort(cos, descending=True)[:topn].cpu()]


def decoder_cosine(src_sae, dst_sae, la: int, lb: int) -> float:
    a = decoder_matrix(src_sae)[la].unsqueeze(0)
    b = decoder_matrix(dst_sae)[lb].unsqueeze(0)
    return float(torch.nn.functional.cosine_similarity(a, b)[0])
