"""Top-activating text contexts per feature: the language analogue of top-activating images.

For each matched concept the page shows, side by side for the baseline and the dual:
  * a short AUTOMATIC interpretation, derived from token statistics and labelled as such;
  * five STRONGLY activating snippets, activated tokens shaded;
  * three MODERATELY activating snippets, drawn from a mid band of the activation range.

The moderate band is not filler. Top-5 panels systematically flatter a feature: the strongest
examples of a polysemantic latent are usually its dominant sense, and the second sense only
appears further down. Showing a mid band is the cheapest way to expose a mixed feature, so it is
a required part of the figure rather than an optional extra.

RETRIEVAL IS FROM THE REAL CORPUS, not from hand-written probes, and it costs no forward pass:
the extraction stage already wrote fp16 activations for the packed test stream, and row i of
those shards is token i of that stream. Verified: the stream decodes to running text and the
shard row count (199,936) equals the manifest's token count. So the contexts shown are genuine
corpus text the SAE was evaluated on.
"""
from __future__ import annotations

import argparse
import collections
import glob
import html
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from pathlib import Path as _Path
# language/ (for the langdual package) and <repo>/vision (the SAE core, imported by bare
# module name) on sys.path, derived from this file's location.
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "vision"))
from langdual import assets, config                          # noqa: E402
from langdual.qualitative import render as R                 # noqa: E402
from langdual.qualitative import token_maps as TM            # noqa: E402

CTX = config.LM_CTX_LEN          # contexts are independent; a snippet must not cross a boundary


@dataclass
class Snippet:
    pos: int
    peak: float
    tokens: List[str]
    values: np.ndarray
    peak_idx: int                # index of the peak token WITHIN the snippet


def load_slice(corpus: str, layer: int, n_rows: int, split: str = "test"
               ) -> Tuple[torch.Tensor, np.ndarray]:
    """(activations, token ids) for the first `n_rows` rows, already in shard space."""
    base = config.resid_dir(layer, split, corpus)
    files = sorted(glob.glob(os.path.join(base, "*.npy")))
    if not files:
        raise FileNotFoundError(f"no shards under {base}")
    acts, got = [], 0
    for f in files:
        a = np.load(f, mmap_mode="r")
        take = min(n_rows - got, a.shape[0])
        acts.append(np.array(a[:take]))
        got += take
        if got >= n_rows:
            break
    X = torch.from_numpy(np.concatenate(acts)).float()
    toks = np.load(config.lm_token_stream(corpus, split), mmap_mode="r")[:X.shape[0]]
    return X, np.array(toks)


@torch.no_grad()
def encode_columns(sae, X: torch.Tensor, latents: Optional[Sequence[int]], device,
                   is_dual: bool, batch: int = config.BATCH_SIZE
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Encode in TRAINING-SIZED batches; return (selected columns, per-latent max).

    The batch size is a protocol constant, not a throughput knob: BatchTopK takes its top-k over
    the whole batch, so a different batch changes which latents fire. Using config.BATCH_SIZE
    reproduces the selection the model was trained and scored under.
    """
    from knn_lp_sparse import forward_reps
    n = X.shape[0] - (X.shape[0] % batch)          # whole batches only, as the scorer does
    cols = [] if latents is not None else None
    peak = None
    for i in range(0, n, batch):
        z = forward_reps(sae, X[i:i + batch].to(device), is_dual)["z"].detach().float()
        m = z.max(dim=0).values.cpu().numpy()
        peak = m if peak is None else np.maximum(peak, m)
        if latents is not None:
            cols.append(z[:, list(latents)].cpu().numpy())
    return (np.concatenate(cols) if cols else np.zeros((0, 0), dtype=np.float32)), peak


@torch.no_grad()
def match_by_coactivation(sae_b, X: torch.Tensor, Vd: np.ndarray, device,
                          is_dual_b: bool, batch: int = config.BATCH_SIZE
                          ) -> Tuple[List[int], List[float]]:
    """For each dual column in `Vd`, the baseline latent whose corpus activation correlates best.

    WHY NOT DECODER COSINE. Matching by decoder direction is the obvious move and it does not
    work here: measured on this pair, the nearest baseline decoder to a dual latent sits at
    cosine 0.10-0.14, i.e. essentially orthogonal. The two models do not share a basis, so a
    geometric "nearest" is arbitrary and the side-by-side comparison it produces is meaningless.
    Correlating the two latents' activation patterns over the same corpus tokens asks the
    question the figure is actually about -- do these two features respond to the same things --
    and needs no shared basis.

    Accumulated in one streaming pass: sum(zb * vd), sum(zb), sum(zb^2) per baseline latent,
    so nothing of shape (n_tokens, 8192) is ever held.
    """
    from knn_lp_sparse import forward_reps
    n = X.shape[0] - (X.shape[0] % batch)
    k = Vd.shape[1]
    sxy = np.zeros((8192, k), dtype=np.float64)
    sy = np.zeros(8192, dtype=np.float64)
    syy = np.zeros(8192, dtype=np.float64)
    for i in range(0, n, batch):
        zb = forward_reps(sae_b, X[i:i + batch].to(device), is_dual_b)["z"].detach().float()
        zb_np = zb.cpu().numpy().astype(np.float64)
        sxy += zb_np.T @ Vd[i:i + batch].astype(np.float64)
        sy += zb_np.sum(0)
        syy += (zb_np ** 2).sum(0)
    sx = Vd[:n].sum(0).astype(np.float64)
    sxx = (Vd[:n].astype(np.float64) ** 2).sum(0)
    cov = sxy - np.outer(sy, sx) / n
    vy = np.maximum(syy - sy ** 2 / n, 1e-12)
    vx = np.maximum(sxx - sx ** 2 / n, 1e-12)
    corr = cov / np.sqrt(np.outer(vy, vx))
    best = [int(np.argmax(corr[:, j])) for j in range(k)]
    return best, [float(corr[b, j]) for j, b in enumerate(best)]


def _is_trivial(v: np.ndarray, toks: np.ndarray, tok, n_top: int = 120,
                thresh: float = 0.8) -> bool:
    """True when a latent's activating tokens are overwhelmingly whitespace or punctuation.

    These are REAL features -- newline and sentence-boundary detectors are among the strongest
    latents in this corpus -- but they are not concepts, and without this filter half the page
    comes back as newline detectors because newline carries the largest activations. Excluded
    from the DISPLAY only; nothing here changes what the model learned, and the count of what
    was skipped is printed so the omission is visible.
    """
    idx = np.argsort(-v)[:n_top]
    idx = idx[v[idx] > 0]
    if idx.size == 0:
        return True
    triv = sum(1 for i in idx
               if not tok.decode([int(toks[i])]).strip(" \t\n\r.,;:!?-–—\"'()[]{}/*+=#"))
    return triv / float(idx.size) >= thresh


def select_concepts(Z: np.ndarray, peak: np.ndarray, n: int, max_fire: float = 0.05,
                    dedupe: float = 0.6, toks: Optional[np.ndarray] = None,
                    tok=None) -> List[int]:
    """Latents that are live, SELECTIVE, non-sink and mutually distinct.

    Three filters, each answering a failure seen on this data:
      * sink -- a latent firing at the first token of every 128-token context is a constant;
      * always-on -- after the sink, the next-ranked latents fire on most tokens and mean nothing;
      * duplicates -- without this, all six selected concepts came back as newline detectors.
    """
    n_tok = Z.shape[0]
    fire = (Z > 0).mean(0)
    first = np.arange(n_tok) % CTX == 0
    sink_rate = (Z[first] > 0).mean(0)
    ok = (peak[:Z.shape[1]] > 0) & (fire > 0) & (fire <= max_fire) & (sink_rate < 0.5)
    order = [int(i) for i in np.argsort(-peak[:Z.shape[1]]) if ok[i]]
    chosen: List[int] = []
    n_trivial = 0
    for i in order:
        col = Z[:, i]
        if col.std() == 0:
            continue
        if toks is not None and tok is not None and _is_trivial(col, toks, tok):
            n_trivial += 1
            continue
        if any(abs(float(np.corrcoef(col, Z[:, j])[0, 1])) > dedupe for j in chosen):
            continue
        chosen.append(i)
        if len(chosen) >= n:
            break
    if n_trivial:
        print(f"  [select] skipped {n_trivial} whitespace/punctuation latents "
              f"(real features, not concepts)")
    return chosen


def snippets(v: np.ndarray, toks: np.ndarray, tok, k: int, window: int,
             lo: float, hi: float, suppress: int) -> List[Snippet]:
    """`k` snippets whose peak lies in [lo, hi], non-overlapping, inside one context.

    Non-overlapping matters: without suppression the five "strongest examples" are usually five
    views of one sentence, which shows a single occurrence five times rather than five pieces of
    evidence.
    """
    order = np.argsort(-v)
    used = np.zeros(len(v), dtype=bool)
    out: List[Snippet] = []
    for p in order:
        val = float(v[p])
        if val <= 0 or not (lo <= val <= hi) or used[p]:
            continue
        # Clamp to the context this token belongs to: contexts are independent documents
        # concatenated end to end, so a window spilling over the edge shows unrelated text.
        c0 = (p // CTX) * CTX
        s, e = max(c0, p - window), min(c0 + CTX, p + window + 1)
        out.append(Snippet(pos=int(p), peak=val,
                           tokens=[tok.decode([int(t)]) for t in toks[s:e]],
                           values=v[s:e].copy(), peak_idx=int(p - s)))
        used[max(0, p - suppress):p + suppress + 1] = True
        if len(out) >= k:
            break
    return out


def interpret(v: np.ndarray, toks: np.ndarray, tok, n_top: int = 200) -> str:
    """A one-line AUTOMATIC description from token statistics. Not an LLM reading."""
    idx = np.argsort(-v)[:n_top]
    idx = idx[v[idx] > 0]
    if idx.size == 0:
        return "silent on this slice"
    strings = [tok.decode([int(toks[i])]) for i in idx]
    c = collections.Counter(strings)
    top = c.most_common(5)
    share = 100.0 * top[0][1] / len(strings)
    shown = ", ".join(f"{s!r}" for s, _ in top)
    kind = ("single token type" if share > 70 else
            "few token types" if share > 35 else "many token types")
    return f"fires on {shown} &nbsp;·&nbsp; {kind} ({share:.0f}% is {top[0][0]!r})"


def _snip_html(s: Snippet, ramp: str, vmax: float) -> str:
    """One snippet row: shaded tokens, with the peak token marked."""
    toks = list(s.tokens)
    toks[s.peak_idx] = toks[s.peak_idx]          # marked via the peak value, not extra markup
    return (f'<div class="snip"><span class="pk">{s.peak:.2f}</span>'
            f'<span class="toks">{R.token_row(toks, s.values, ramp, vmax=vmax)}</span></div>')


def build(layer: int, corpus: str, activation: str, seed: int, checkpoint: str,
          n_rows: int, n_concepts: int, out_dir: str, device: torch.device,
          dual_tag: str = "", window: int = 12, suffix: str = "") -> str:
    base_tag = f"B_L{layer}_{corpus}_{activation}_baseline_seed{seed}"
    dual_tag = dual_tag or f"B_L{layer}_{corpus}_{activation}_dual_seed{seed}"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(assets.local_snapshot(config.LM_REPO),
                                        local_files_only=True)
    sae_b, meta_b = TM.load_cell(base_tag, checkpoint, device)
    sae_d, meta_d = TM.load_cell(dual_tag, checkpoint, device)
    X, toks = load_slice(corpus, layer, n_rows)

    # PASS 1 -- per-latent maxima only, so nothing of shape (n_tokens, 8192) is materialised.
    _, peak_d = encode_columns(sae_d, X, None, device, True)

    # PASS 2 -- a generous shortlist of the dual's strongest latents, then filter it down.
    short = [int(i) for i in np.argsort(-peak_d)[:400] if peak_d[i] > 0]
    Zs, _ = encode_columns(sae_d, X, short, device, True)
    picks = select_concepts(Zs, peak_d[short], n_concepts, toks=toks, tok=tok)
    lat_d = [short[i] for i in picks]
    Zd = Zs[:, picks]

    # PASS 3 -- functional matching. Decoder cosine is useless across these two models
    # (nearest neighbour sits at ~0.12), so the counterpart is the baseline latent whose corpus
    # activation correlates best with the dual one.
    lat_b, corr_b = match_by_coactivation(sae_b, X, Zd, device, meta_b["is_dual"])
    Zb, _ = encode_columns(sae_b, X, lat_b, device, meta_b["is_dual"])
    chosen = list(zip(lat_d, lat_b))
    n = Zd.shape[0]
    toks = toks[:n]

    base_lbl, dual_lbl = next((b, d) for a, b, d in config.ACTIVATION_PAIRS if a == activation)
    body: List[str] = []
    for ci, (ld, lb) in enumerate(chosen):
        cols = []
        for nm, Z, col, sae_x, lat, ramp in (
                (dual_lbl, Zd, ci, sae_d, ld, "mono"),
                (base_lbl, Zb, ci, sae_b, lb, "poly")):
            v = Z[:, col]
            mx = float(v.max())
            strong = snippets(v, toks, tok, 5, window, 0.60 * mx, mx, window)
            # The mid band is where a second sense hides; 25-55% of peak is low enough to leave
            # the dominant sense behind and high enough that the latent is genuinely responding.
            mod = snippets(v, toks, tok, 3, window, 0.25 * mx, 0.55 * mx, window)
            rows = "".join(_snip_html(s, ramp, mx) for s in strong)
            rows_m = "".join(_snip_html(s, ramp, mx) for s in mod)
            cols.append(
                f'<div class="col"><div class="colh">{html.escape(nm)} '
                f'<span class="lat">latent {lat}</span></div>'
                f'<p class="interp">{interpret(v, toks, tok)}</p>'
                f'<div class="lbl">strongly activating (5)</div>{rows or "<p class=none>none</p>"}'
                f'<div class="lbl">moderately activating (3) '
                f'<span class="mut">— where a second sense shows up</span></div>'
                f'{rows_m or "<p class=none>none in the 25–55% band</p>"}</div>')
        cs = TM.decoder_cosine(sae_d, sae_b, ld, lb)
        body.append(f'<h2>Concept {ci + 1} <span class="mut">dual {ld} &harr; baseline {lb}, '
                    f'co-activation r {corr_b[ci]:.3f} (decoder cos {cs:.3f})</span></h2>'
                    f'<div class="two">{"".join(cols)}</div>')

    extra_css = """
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:900px){.two{grid-template-columns:1fr}}
.col{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.colh{font-weight:650;margin-bottom:4px}.colh .lat{font-weight:400;color:var(--mut);font-size:.85rem}
.interp{font-size:.85rem;color:var(--mut);margin:0 0 10px}
.lbl{font-size:.78rem;font-weight:650;text-transform:uppercase;letter-spacing:.04em;
  color:var(--mut);margin:12px 0 4px}
.snip{display:flex;gap:8px;align-items:flex-start;margin:5px 0;font-size:12.5px}
.snip .pk{flex:0 0 42px;text-align:right;color:var(--mut);font:11.5px/2 ui-monospace,monospace}
.snip .toks{font:12.5px/1.95 ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-word}
.mut{font-weight:400;color:var(--mut)}
.none{color:var(--mut);font-size:.82rem;margin:4px 0}
h2 .mut{font-size:.82rem}
"""
    lede = (f"{config.LM_REPO} block {layer}, trained on "
            f"<code>{config.LM_CORPORA[corpus].repo}</code>, seed {seed}, {checkpoint} "
            f"checkpoint. Retrieved from {n:,} real tokens of the held-out "
            f"{corpus} test stream. Blue is {dual_lbl}; orange is {base_lbl}.")
    footer = (
        '<h2>How to read this</h2><table>'
        '<tr><th>Point</th><th>Detail</th></tr>'
        '<tr><td>Omitted</td><td>Latents whose activating tokens are >80% whitespace or punctuation are skipped for DISPLAY. They are real and among the strongest features in this corpus, but they are not concepts; the count skipped is printed at build time.</td></tr>'
        '<tr><td>Interpretation</td><td>AUTOMATIC, from token statistics over the 200 strongest '
        'positions. It is a description of what fires, not a semantic reading; an LLM judge '
        'would be needed for that.</td></tr>'
        '<tr><td>Strong vs moderate</td><td>Strong is 60-100% of the feature\'s peak, moderate is '
        '25-55%. The moderate band is included because a top-5 panel shows a polysemantic '
        'feature\'s dominant sense only.</td></tr>'
        '<tr><td>Snippets</td><td>Non-overlapping and clamped to one 128-token context, so five '
        'rows are five occurrences rather than five views of one.</td></tr>'
        '<tr><td>Pairing</td><td>Matched by CO-ACTIVATION over the same corpus tokens, not by '
        'decoder cosine: across these two models the nearest decoder sits at ~0.12, so a '
        'geometric match is arbitrary. A low r means the baseline has no functional '
        'counterpart, and the comparison is weak whatever the snippets show.</td></tr>'
        '<tr><td>Shade</td><td>Linear in activation, normalised per column by that feature\'s own '
        'peak, which is printed at the left of each row.</td></tr>'
        '<tr><td>Batching</td><td>Encoded in training-sized batches, so BatchTopK\'s batch-level '
        'top-k matches the protocol the tables use.</td></tr>'
        '</table>')
    doc = R.document(f"Top-activating contexts — block {layer}, {corpus}, {activation}",
                     lede, "\n".join(body), footer).replace("</style>", extra_css + "</style>")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"contexts_L{layer}_{corpus}_{activation}_seed{seed}{suffix}.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=12, choices=list(config.LM_LAYERS))
    p.add_argument("--corpus", default="owt", choices=list(config.LM_CORPUS_ORDER))
    p.add_argument("--activation", default="topk", choices=list(config.ACTIVATIONS))
    p.add_argument("--seed", type=int, default=config.SEEDS[0], choices=list(config.SEEDS))
    p.add_argument("--checkpoint", default="best", choices=("best", "final"))
    p.add_argument("--dual-tag", default="")
    p.add_argument("--rows", type=int, default=100_000, help="corpus tokens to retrieve from")
    p.add_argument("--concepts", type=int, default=6)
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--suffix", default="")
    p.add_argument("--out", default=os.path.join(config.REPORT_DIR, "qualitative"))
    a = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("wrote", build(a.layer, a.corpus, a.activation, a.seed, a.checkpoint, a.rows,
                         a.concepts, a.out, dev, a.dual_tag, a.window, a.suffix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
