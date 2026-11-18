"""Render per-token activations as shaded text.

COLOUR IS A MEASUREMENT, so the mapping is stated on every figure rather than left implicit:
shade is LINEAR in activation, normalised per panel by that panel's own maximum, and the maximum
is printed. A per-panel normalisation is the right choice for reading WHERE a feature fires; it
is the wrong choice for comparing HOW HARD two features fire, so any cross-panel comparison must
read the printed maxima, not the colours. Sub-word tokens are shown with their real boundaries
(a leading space is rendered as part of the preceding gap, never stripped) because a feature that
fires on "▁Man" but not "hattan" is a different claim than one that fires on the whole word.
"""
from __future__ import annotations

import html
from typing import Dict, List, Optional, Sequence

import numpy as np

# Two ramps so a reader can tell the panels apart at a glance without a legend lookup: the sparse
# code (Sparse Branch) is the object of interest; the poly (Residual Branch) norm is shown alongside.
RAMPS = {
    "mono": (34, 94, 168),     # blue
    "poly": (200, 90, 20),     # orange
    "resid": (90, 90, 90),     # grey
}


def _shade(tok: str, frac: float, rgb) -> str:
    """One token, background-shaded by `frac` in [0, 1]."""
    frac = 0.0 if not np.isfinite(frac) else float(max(0.0, min(1.0, frac)))
    r, g, b = rgb
    # Alpha rather than a blend so the page background shows through and text stays readable at
    # every intensity; the 0.88 cap keeps the darkest cell from swallowing the glyph.
    alpha = 0.88 * frac
    fg = "#fff" if frac > 0.55 else "#111"
    txt = html.escape(tok.replace(" ", " "))
    return (f'<span class="t" style="background:rgba({r},{g},{b},{alpha:.3f});color:{fg}" '
            f'title="{frac:.3f}">{txt}</span>')


def token_row(tokens: Sequence[str], values: np.ndarray, ramp: str = "mono",
              vmax: Optional[float] = None) -> str:
    v = np.asarray(values, dtype=np.float64)
    hi = float(vmax if vmax is not None else (v.max() if v.size else 0.0))
    frac = v / hi if hi > 0 else np.zeros_like(v)
    return "".join(_shade(t, f, RAMPS[ramp]) for t, f in zip(tokens, frac))


def panel(title: str, subtitle: str, tokens: Sequence[str], values: np.ndarray,
          ramp: str = "mono") -> str:
    hi = float(np.asarray(values).max()) if len(values) else 0.0
    return (f'<div class="panel"><div class="h">{html.escape(title)}'
            f'<span class="s">{html.escape(subtitle)} &nbsp;·&nbsp; max {hi:.3f}</span></div>'
            f'<div class="toks">{token_row(tokens, values, ramp, vmax=hi)}</div></div>')


CSS = """
:root{--bg:#ffffff;--fg:#16181d;--mut:#5b6270;--line:#e3e6ec;--card:#fbfcfe}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#14161a;--fg:#e9ecf1;--mut:#98a1b0;--line:#2a2e36;--card:#1b1e24}}
:root[data-theme="dark"]{--bg:#14161a;--fg:#e9ecf1;--mut:#98a1b0;--line:#2a2e36;--card:#1b1e24}
*{box-sizing:border-box}
body{margin:0;padding:24px 16px 64px;background:var(--bg);color:var(--fg);
  font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto}
h1{font-size:1.35rem;margin:0 0 6px}
h2{font-size:1.05rem;margin:32px 0 4px;padding-top:16px;border-top:1px solid var(--line)}
.lede{color:var(--mut);margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:14px 16px;margin:14px 0}
.sent{color:var(--mut);font-size:.86rem;margin:0 0 10px;word-break:break-word}
.panel{margin:10px 0}
.h{font-size:.82rem;font-weight:650;margin-bottom:5px}
.h .s{font-weight:400;color:var(--mut);margin-left:8px}
.toks{font:13.5px/2.05 ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-word}
.t{padding:2px 0;border-radius:3px}
.note{color:var(--mut);font-size:.84rem;margin:6px 0 0}
table{border-collapse:collapse;width:100%;font-size:.86rem;margin:10px 0}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left}
th{background:var(--card);font-weight:650}
code{background:var(--card);padding:1px 5px;border-radius:4px;font-size:.9em}
@media(max-width:640px){body{padding:16px}}
"""


def document(title: str, lede: str, body: str, footer: str = "") -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="lede">{lede}</p>
{body}
{footer}
</div></body></html>"""
