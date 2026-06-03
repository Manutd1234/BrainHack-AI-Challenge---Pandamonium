"""Build the hotwords list + phrase corrections from the training manifest.

Walks every training transcript and pulls out:
  * tokens not in a large English wordlist (likely domain/made-up: "ilovekentcheong",
    "OpenLarp", call signs, etc.) -> asr_hotwords.json
  * multi-word phrases that look "branded" (Title Case, internal capitals,
    digit+letter combos) -> asr_corrections.json source/target pairs you can edit

Run:
    python build_hotwords.py
    # produces:
    #   src/asr_hotwords.json
    #   src/asr_corrections.json
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(os.getenv("ASR_DATA_DIR", "/home/jupyter/novice/asr"))
OUT_DIR = Path(os.getenv("ASR_HOTWORDS_OUT", "/home/jupyter/asr/src"))
MANIFEST = DATA_DIR / "asr.jsonl"

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*|\d+(?:\.\d+)?")

# Tiny common-word filter. Anything *not* on this list and not a small word
# is a hotword candidate. You can expand this list, but keep it small —
# the goal is to keep weird/rare/branded words.
COMMON = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "but",
    "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "this", "that", "these", "those", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "its", "our", "their", "what", "which", "who", "whom", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "can", "will", "just", "don", "should", "now",
    "with", "from", "by", "as", "if", "out", "up", "down", "over", "under",
    "again", "then", "there", "here", "into", "about", "after", "before",
    "yes", "ok", "okay", "right", "left", "go", "stop", "start", "one", "two",
    "three", "four", "five", "six", "seven", "eight", "nine", "ten", "zero",
}


def looks_branded(token: str) -> bool:
    """Heuristic: branded / made-up tokens are worth boosting."""
    if len(token) < 4:
        return False
    if token.lower() in COMMON:
        return False
    # internal capitals (camelCase / TitleCase)
    if any(c.isupper() for c in token[1:]):
        return True
    # digits mixed with letters
    has_letter = any(c.isalpha() for c in token)
    has_digit = any(c.isdigit() for c in token)
    if has_letter and has_digit:
        return True
    # all-lowercase weird-looking long word (no vowels in a row, > 7 chars,
    # not in COMMON) — captures things like "ilovekentcheong"
    if token.islower() and len(token) >= 8:
        return True
    return False


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"manifest not found: {MANIFEST}")

    counter: Counter[str] = Counter()
    branded: Counter[str] = Counter()

    with MANIFEST.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            text = (item.get("transcript") or item.get("text") or item.get("sentence") or "").strip()
            if not text:
                continue
            for tok in WORD_RE.findall(text):
                counter[tok] += 1
                if looks_branded(tok):
                    branded[tok] += 1

    print(f"unique tokens: {len(counter)}")
    print(f"branded tokens: {len(branded)}")

    # Hotwords: the branded tokens, preserving the case form that appears most.
    # If a token appears as both "OpenLarp" and "openlarp", keep "OpenLarp".
    case_pref: dict[str, Counter[str]] = {}
    for tok, n in counter.items():
        case_pref.setdefault(tok.lower(), Counter())[tok] += n

    hotwords: list[str] = []
    for tok, _ in branded.most_common():
        canonical = case_pref[tok.lower()].most_common(1)[0][0]
        if canonical not in hotwords:
            hotwords.append(canonical)

    # Bonus: also include any rare token (count <= 3) that's >= 6 chars and
    # not common — these are exactly the things the base model is most likely
    # to mistranscribe.
    for tok, n in counter.items():
        if n > 3 or len(tok) < 6 or tok.lower() in COMMON:
            continue
        canonical = case_pref[tok.lower()].most_common(1)[0][0]
        if canonical not in hotwords:
            hotwords.append(canonical)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "asr_hotwords.json").write_text(
        json.dumps({"hotwords": hotwords}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {OUT_DIR / 'asr_hotwords.json'}  ({len(hotwords)} terms)")

    # Phrase corrections starter: any case-confused pair (lowercase form differs
    # from canonical). You can hand-edit this file after.
    phrase_corrections: dict[str, str] = {}
    for canonical in hotwords:
        lower = canonical.lower()
        if lower != canonical:
            phrase_corrections[lower] = canonical

    corr_path = OUT_DIR / "asr_corrections.json"
    if not corr_path.exists():
        corr_path.write_text(
            json.dumps({"phrase_corrections": phrase_corrections}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote {corr_path}  ({len(phrase_corrections)} pairs)")
    else:
        print(f"{corr_path} already exists — leaving untouched (edit by hand)")


if __name__ == "__main__":
    main()
