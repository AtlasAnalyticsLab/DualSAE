"""
Repair an LLM-generated caption JSONL whose caption *values* contain unescaped newlines or
quotes — which breaks line-delimited JSON (a single record gets split across physical lines, or a
stray `"` ends the string early). The OpenImages InternVL caption files have ~4% such records.

Approach (does NOT rely on per-line JSON validity):
  Records share a fixed outer schema  {"rel_path": "<id>", "caption": "<text>", "model": "<m>"} .
  We re-segment the whole file on that schema with a DOTALL regex whose caption group is bounded by
  the full `", "model":` delimiter, so embedded newlines and stray quotes inside the caption no
  longer matter. For each record we keep byte-identical text when it parses cleanly (json.loads on
  the matched object) and fall back to the regex-captured caption + whitespace normalization only
  when it doesn't. Output is a well-formed JSONL (proper json.dumps escaping) in original order.

Why normalization is safe: the caption is consumed only by the BoW tokenizer downstream, which keeps
  ^[a-z]+$ word tokens — collapsed whitespace and the odd leftover backslash are dropped there.

Usage:
  python repair_captions.py --in  <raw .jsonl> --out <cleaned .jsonl>
"""
import argparse
import glob as _glob
import json
import os
import re

# Record-boundary regex. DOTALL lets the (non-greedy) caption span embedded newlines; the caption
# end is anchored on the full `"<ws>,<ws>"model":` delimiter, so an unescaped quote inside the
# caption doesn't terminate the match early (it would have to be immediately followed by the whole
# delimiter, which never occurs in prose). rel_path/model values are clean (no quotes/braces).
REC = re.compile(
    r'\{\s*"rel_path"\s*:\s*"(?P<rel>[^"]+)"\s*,\s*'
    r'"caption"\s*:\s*"(?P<cap>.*?)"\s*,\s*'
    r'"model"\s*:\s*"[^"}]*"\s*\}',
    re.DOTALL,
)
_WS = re.compile(r"\s+")


def clean_caption(raw: str) -> str:
    """Best-effort de-escape + whitespace-normalize a caption captured from a broken record."""
    # Undo the JSON escapes we may have captured literally, then fold all whitespace (including the
    # unescaped newlines that split the record) into single spaces. A lone trailing backslash is
    # harmless (filtered by the BoW tokenizer).
    raw = (raw.replace('\\"', '"').replace("\\n", " ").replace("\\t", " ")
              .replace("\\r", " ").replace("\\\\", "\\"))
    return _WS.sub(" ", raw).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True,
                    help="Raw caption JSONL, or a GLOB matching several (e.g. '.../test_OpenGVLab/*.jsonl' "
                         "to merge the original file + all caption shards).")
    ap.add_argument("--out", dest="out", required=True, help="Output well-formed, de-duplicated JSONL.")
    args = ap.parse_args()

    files = sorted(_glob.glob(args.inp))
    if not files:
        raise SystemExit(f"No input files matched: {args.inp}")

    # Dedup by rel_path across all files. Each entry is [tier, caption] with tier 0=clean(json),
    # 1=regex-recovered. A lower tier always wins (a cleanly-recaptioned shard line beats a malformed
    # original line for the same image); within a tier the first occurrence wins. dict preserves
    # first-seen insertion order → deterministic output order (emb/BoW will follow this same order).
    best = {}
    n_markers = clean = recovered = 0
    for path in files:
        text = open(path, encoding="utf-8", errors="replace").read()
        n_markers += text.count('"rel_path"')      # ~one per logical record, regardless of line splits
        for m in REC.finditer(text):
            rel = m.group("rel")
            try:
                cap = json.loads(m.group(0))["caption"]   # clean record → exact original text
                tier = 0
                clean += 1
            except Exception:
                cap = clean_caption(m.group("cap"))        # broken record → recovered text
                tier = 1
                recovered += 1
            cap = cap.strip()
            if not cap:
                continue
            prev = best.get(rel)
            if prev is None or tier < prev[0]:
                best[rel] = [tier, cap]

    tmp = args.out + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as w:
        for rel, (tier, cap) in best.items():
            w.write(json.dumps({"rel_path": rel, "caption": cap}) + "\n")
    os.replace(tmp, args.out)

    kept_clean = sum(1 for v in best.values() if v[0] == 0)
    print(f"input files: {len(files)}")
    print(f"record markers (\"rel_path\") across files: {n_markers}  (parsed clean={clean}, recovered={recovered})")
    print(f"unique records written: {len(best)}  (kept clean={kept_clean}, kept recovered={len(best) - kept_clean})")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
