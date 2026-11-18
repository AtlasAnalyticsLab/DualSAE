"""s3: the 60 hand-written neutral prompts (D-49, PROTOCOL P2 and R6).

  * 60 prompts, one per line; the FIRST 20 lines are the 20-prompt set used by the D-48 pilot and
    by every non-headline cell (PROTOCOL R6).  Headline cells use all 60 x 2 continuations.
  * each <= 16 pythia tokens (no special tokens; position 0 is the sink whatever the token), so
    prompt + 48 new tokens stays inside the SAEs' 128-position training window;
  * free of concept words: no word of concepts.BANNED_PROMPT_WORDS (class names, every content
    word of the judge descriptions, and obvious synonyms), checked case-insensitively on word
    boundaries.  They are sentence openers about everyday, non-topical situations (houses,
    roads, meals, letters), deliberately avoiding sentiment, news, sport, business, technology
    and emotion vocabulary.

``--write`` writes the file ONCE (refuses to change an existing file) and prints its sha256 for
the P2 record; ``--check`` re-validates the file on disk.  After P2 is recorded, the sha256 is
pinned in base.PROMPTS_SHA256 and every generation stage refuses a different file.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from steering import base, concepts

PROMPTS = (
    # --- the 20-prompt set (PROTOCOL R6: the first 20 lines) ---
    "The next thing I remember is",
    "She opened the door and saw",
    "In the morning, we usually",
    "Here is what happened after lunch:",
    "My neighbor told me that the",
    "The old house at the end of the street",
    "When the train finally arrived,",
    "He picked up the pen and wrote",
    "The recipe calls for two cups of",
    "On the second floor of the building there is",
    "It was a quiet afternoon, and",
    "The letter began with the words",
    "After the meeting, they walked to",
    "The river runs through the middle of",
    "I have been thinking about the",
    "According to the instructions, you should",
    "The small village was known for its",
    "Last Tuesday, my cousin and I",
    "The book on the table was",
    "At the corner of the road, a man",
    # --- prompts 21-60 (headline cells only) ---
    "The garden behind the school has",
    "Before leaving, she checked the",
    "The story takes place in a",
    "Every summer, the family drives to",
    "The kitchen smelled of fresh",
    "He looked out of the window at the",
    "There are three things you need to know about",
    "The bus stopped in front of the",
    "My grandmother used to say that",
    "The map showed a path through the",
    "They spent the evening talking about",
    "The first chapter describes how the",
    "In the middle of the night, the",
    "The man at the desk asked me",
    "We packed our bags and headed to",
    "The painting on the wall shows a",
    "When I was younger, I often",
    "The road to the lake was",
    "She wrote a short note that said",
    "The children were sitting on the",
    "It took us two hours to find the",
    "The sign on the door read",
    "Yesterday I found an old photograph of",
    "The lights in the hallway were",
    "Our teacher asked us to describe",
    "The ship sailed slowly past the",
    "He explained how to fold the",
    "The weather this week has been",
    "At the bottom of the drawer, I found",
    "The bakery is next to the",
    "She turned the page and read",
    "The path through the forest leads to",
    "In the evening, the streets are",
    "The instructions on the card say",
    "My uncle and I decided to",
    "The clock on the wall said it was",
    "Behind the curtain there was a",
    "The last time we visited the",
    "He carried the chair up the",
    "The conversation started when",
)


def validate(lines, tok) -> list:
    problems = []
    if len(lines) != base.N_PROMPTS_HEADLINE:
        problems.append(f"{len(lines)} prompts, expected {base.N_PROMPTS_HEADLINE}")
    if len(set(lines)) != len(lines):
        problems.append("duplicate prompts")
    banned = concepts.BANNED_PROMPT_WORDS
    for i, ln in enumerate(lines):
        n = len(tok(ln, add_special_tokens=False)["input_ids"])
        if n > base.MAX_PROMPT_TOKENS or n < 2:
            problems.append(f"line {i + 1}: {n} tokens (need 2..{base.MAX_PROMPT_TOKENS}): {ln!r}")
        words = set(re.findall(r"[a-z]+", ln.lower()))
        hit = sorted(words & set(banned))
        if hit:
            problems.append(f"line {i + 1}: concept words {hit}: {ln!r}")
        if ln != ln.strip() or "\n" in ln:
            problems.append(f"line {i + 1}: leading/trailing whitespace")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    tok = concepts.load_tokenizer()
    body = "\n".join(PROMPTS) + "\n"
    if a.write:
        probs = validate(list(PROMPTS), tok)
        if probs:
            print("REFUSING to write:\n  " + "\n  ".join(probs))
            return 1
        if os.path.isfile(base.PROMPTS_PATH):
            if open(base.PROMPTS_PATH, encoding="utf-8").read() != body:
                print(f"REFUSING: {base.PROMPTS_PATH} exists with different content (P2 is fixed)")
                return 1
            print(f"{base.PROMPTS_PATH} already written (identical)")
        else:
            base.atomic_write_bytes(base.PROMPTS_PATH, body.encode("utf-8"))
            print(f"wrote {base.PROMPTS_PATH}")
    lines = base.check_prompts(require_pin=False)
    probs = validate(lines, tok)
    ntok = [len(tok(ln, add_special_tokens=False)["input_ids"]) for ln in lines]
    print(f"{len(lines)} prompts; pythia tokens min {min(ntok)} max {max(ntok)}; "
          f"banned-word list {len(concepts.BANNED_PROMPT_WORDS)} words")
    print(f"sha256 {base.sha256_file(base.PROMPTS_PATH)}  (P2)")
    if base.PROMPTS_SHA256 is not None:
        print(f"pinned  {base.PROMPTS_SHA256}  match={base.sha256_file(base.PROMPTS_PATH) == base.PROMPTS_SHA256}")
    if probs:
        print("PROBLEMS:\n  " + "\n  ".join(probs))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
