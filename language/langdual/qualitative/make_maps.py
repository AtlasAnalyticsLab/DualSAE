"""Build per-token activation maps for one activation PAIR (standard SAE vs DualSAE).

The figure answers three questions the averaged tables cannot:

  1. WHICH WORDS a latent responds to, and whether it fires on one sub-word token or spans a
     phrase -- the "textual combination" question.
  2. WHETHER THE PAIR AGREES. The baseline and the dual are shown on the SAME sentence and, where
     possible, on the SAME decoder direction (matched by cosine), so the panels answer "what did
     these two models do with this direction", not "what is each model's own best feature".
  3. WHERE THE POLY BRANCH WORKS. DualSAE claims a division of labour: a sparse mono branch that
     carries interpretable features and a dense poly branch that absorbs the rest. Plotting
     ||r_hat|| per token next to the sparse code makes that claim visible or falsifies it. There
     is no per-latent poly panel because the poly branch is a dense MLP on the residual -- it has
     no latent basis, and inventing one would be a fiction.

Usage (one short GPU pass):
  python -m langdual.qualitative.make_maps --layer 12 --corpus owt --activation topk \
      --out <LANGDUAL_ROOT>/reports/qualitative
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import torch

from pathlib import Path as _Path
# language/ (for the langdual package) and <repo>/vision (the SAE core, imported by bare
# module name) on sys.path, derived from this file's location.
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "vision"))
from langdual import config                                   # noqa: E402
from langdual.qualitative import render as R                  # noqa: E402
from langdual.qualitative import token_maps as TM             # noqa: E402

# Default probe set. Chosen to make COMPOSITION visible rather than to be representative:
# each group holds sentences that share words but differ in how those words combine, so a
# feature that keys on a bag of tokens and a feature that keys on the construction look
# different. Kept short so a whole group fits on one screen.
PROBES: List[str] = [
    # sense disambiguation: same token, two meanings
    "The bank raised interest rates again this quarter.",
    "We sat on the river bank and watched the boats go by.",
    # syntactic role reversal: identical content words
    "The senator that the reporter attacked admitted the error.",
    "The reporter that the senator attacked admitted the error.",
    # negation and scope
    "The treatment was effective for most patients in the trial.",
    "The treatment was not effective for most patients in the trial.",
    # multi-word entity vs its parts
    "New York Times reporters covered the election in New York.",
    "Times are hard and the news from York is worse.",
    # code, to probe a code-trained dictionary on its own distribution
    "def parse(path):\n    with open(path) as fh:\n        return json.load(fh)",
    "for i in range(10):\n    total += values[i] * weights[i]",
]


def build(layer: int, corpus: str, activation: str, seed: int, n_latents: int,
          checkpoint: str, sentences: Sequence[str], out_dir: str,
          device: torch.device, dual_tag: str = "", suffix: str = "",
          filler_rows: int = 2048) -> str:
    base_tag = f"B_L{layer}_{corpus}_{activation}_baseline_seed{seed}"
    # The dual arm may be an ABLATION cell (e.g. another warm-up length), while the baseline
    # always comes from the main grid: warm-up is a DualSAE-only knob, so the main-grid baseline
    # is the matching standard SAE.
    dual_tag = dual_tag or f"B_L{layer}_{corpus}_{activation}_dual_seed{seed}"
    lm, tok = TM.load_lm(device)
    sae_b, meta_b = TM.load_cell(base_tag, checkpoint, device)
    sae_d, meta_d = TM.load_cell(dual_tag, checkpoint, device)
    maps_b = TM.map_sentences(sentences, lm, tok, sae_b, meta_b, device,
                              filler_rows=filler_rows)
    maps_d = TM.map_sentences(sentences, lm, tok, sae_d, meta_d, device,
                              filler_rows=filler_rows)

    # Rank on the DUAL, then carry each latent across to the baseline by decoder cosine. The dual
    # is the model under test, so its features are the ones worth interrogating; the baseline
    # panel is there to show what the same direction does in a single-branch model.
    lat_d = TM.top_latents(maps_d, n=n_latents)
    pairs = [(ld, TM.match_latent_by_decoder(sae_d, sae_b, ld, 1)[0]) for ld in lat_d]
    cos = {ld: TM.decoder_cosine(sae_d, sae_b, ld, lb) for ld, lb in pairs}

    body: List[str] = []
    sink = TM.sink_latents(maps_d)
    body.append("<h2>Where the poly branch works</h2>")
    body.append('<p class="note">Each branch is shown as a FRACTION of the token\'s own raw '
                'activation norm, not as a raw magnitude. Raw magnitudes are unreadable here: '
                'the first position of every sentence is an attention sink carrying 81.8% of '
                'the total energy, so an absolute colour scale paints it solid and leaves the '
                'rest of the sentence blank. The fraction is scale-free and shows the division '
                'of labour the architecture claims.</p>')
    for m in maps_d[:6]:
        denom = np.maximum(m.resid_norm, 1e-6)
        pan = [R.panel("mono branch", "||x^M|| / ||x||", m.tokens, m.mono_norm / denom, "mono")]
        if m.poly_norm is not None:
            pan.append(R.panel("poly branch", "||r_hat|| / ||x||", m.tokens,
                               m.poly_norm / denom, "poly"))
        pan.append(R.panel("raw activation", "||x|| (absolute; note the sink at position 0)",
                           m.tokens, m.resid_norm, "resid"))
        body.append(f'<div class="card"><p class="sent">{m.text!r}</p>' + "".join(pan) + "</div>")

    body.append(f"<h2>Matched latents, dual vs baseline</h2>")
    body.append('<p class="note">Each pair is the dual latent and the baseline latent whose '
                'decoder direction is closest to it, so both panels describe the same direction '
                'in activation space. Cosine is printed; a low cosine means the baseline has no '
                'close counterpart and the comparison is weak. Latents are ranked ignoring the '
                f'first {TM.SINK_POSITIONS} position(s), and the {len(sink)} latents that form '
                'the constant attention-sink set are excluded entirely: they fire on every '
                'sentence identically and carry no content.</p>')
    for ld, lb in pairs:
        rows = []
        for md, mb in zip(maps_d, maps_b):
            vd, vb = md.z[:, ld], mb.z[:, lb]
            if vd.max() <= 0 and vb.max() <= 0:
                continue          # neither model reacts: an all-white row carries no information
            rows.append(
                f'<div class="card"><p class="sent">{md.text!r}</p>'
                + R.panel(f"Dual latent {ld}", "sparse code", md.tokens, vd, "mono")
                + R.panel(f"Baseline latent {lb}", "sparse code", mb.tokens, vb, "mono")
                + "</div>")
        if rows:
            body.append(f'<h3>dual {ld} &harr; baseline {lb} '
                        f'<span style="font-weight:400;color:var(--mut)">'
                        f'(decoder cos {cos[ld]:.3f})</span></h3>')
            body.extend(rows)

    base_lbl, dual_lbl = next((b, d) for a, b, d in config.ACTIVATION_PAIRS if a == activation)
    lede = (f"{config.LM_REPO} block {layer}, trained on "
            f"<code>{config.LM_CORPORA[corpus].repo}</code>, seed {seed}, "
            f"{checkpoint} checkpoint. {base_lbl} (<code>{base_tag}</code>) against "
            f"{dual_lbl} (<code>{dual_tag}</code>).")
    footer = (
        '<h2>How to read this</h2><table>'
        '<tr><th>Point</th><th>Detail</th></tr>'
        '<tr><td>Shade</td><td>Linear in activation, normalised per panel by that panel\'s own '
        'maximum, which is printed. Compare colours WITHIN a panel; compare maxima ACROSS them.</td></tr>'
        '<tr><td>Tokens</td><td>Pythia sub-words with real boundaries. A feature firing on part '
        'of a word is a different claim from one firing on the whole word.</td></tr>'
        '<tr><td>Batching</td><td>One sentence per forward pass. BatchTopK normally selects '
        'top-k over a whole batch, so these codes are what each sentence alone produces, not '
        'the L0=64-per-batch protocol the tables use.</td></tr>'
        '<tr><td>Poly branch</td><td>Shown as a fraction of the token norm, not per latent. It '
        'is a dense MLP on the residual with no latent basis, so a per-feature panel would be a '
        'fiction.</td></tr>'
        '<tr><td>Attention sink</td><td>Position 0 carries 81.8% of activation energy on this '
        'probe set and fires the same 64 latents in every sentence. It is excluded from latent '
        'ranking and its latents are dropped, or the figure would show only the sink.</td></tr>'
        '</table>')
    doc = R.document(f"Token activation maps — block {layer}, {corpus}, {activation}",
                     lede, "\n".join(body), footer)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir,
                        f"tokenmap_L{layer}_{corpus}_{activation}_seed{seed}{suffix}.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=12, choices=list(config.LM_LAYERS))
    p.add_argument("--corpus", default="owt", choices=list(config.LM_CORPUS_ORDER))
    p.add_argument("--activation", default="topk", choices=list(config.ACTIVATIONS))
    p.add_argument("--seed", type=int, default=config.SEEDS[0], choices=list(config.SEEDS))
    p.add_argument("--n-latents", type=int, default=8)
    p.add_argument("--checkpoint", default="best", choices=("best", "final"))
    p.add_argument("--sentences-file", default=None,
                   help="one sentence per line; default is the built-in composition probe set")
    p.add_argument("--dual-tag", default="",
                   help="override the dual cell, e.g. an ablation tag at a tuned warm-up")
    p.add_argument("--suffix", default="", help="appended to the output filename")
    p.add_argument("--filler-rows", type=int, default=2048,
                   help="real corpus rows appended to the encode batch so BatchTopK's "
                        "batch-level selection matches training; discarded before rendering")
    p.add_argument("--out", default=os.path.join(config.REPORT_DIR, "qualitative"))
    a = p.parse_args(argv)
    sents = PROBES
    if a.sentences_file:
        with open(a.sentences_file, encoding="utf-8") as fh:
            sents = [ln.rstrip("\n") for ln in fh if ln.strip()]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = build(a.layer, a.corpus, a.activation, a.seed, a.n_latents, a.checkpoint,
                 sents, a.out, dev, dual_tag=a.dual_tag, suffix=a.suffix,
                 filler_rows=a.filler_rows)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
