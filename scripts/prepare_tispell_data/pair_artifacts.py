"""Bracket / ornament cleaning and Latin / non-Tibetan-punct pair filters.

Pre-segmentation (full page transcript):
  - Strip bracket chars `()（）༼༽` (keep enclosed text)
  - Strip page ornaments U+0FD0–U+0FDA plus section/head marks
    ༁–༊, ༒, ༓ (exclude ༔ U+0F14)

Pair-level (after alignment):
  - Drop if Latin on either side — ASCII `[A-Za-z]` and fullwidth
    U+FF21–FF5A (ＩＳＯ …)
  - Drop if non-Tibetan punctuation/symbols on either side (— „ . " …);
    these sit where a Tibetan character should be, so stripping would gap.
"""

from __future__ import annotations

import re
import unicodedata

# ASCII + fullwidth Latin letter runs (U+FF21–FF5A = Ａ–Ｚａ–ｚ).
LATIN_RUN_RE = re.compile(r"[A-Za-z\uFF21-\uFF5A]+")
# ASCII, fullwidth, and Tibetan gugu brackets — strip chars, keep contents.
BRACKET_CHARS_RE = re.compile(r"[()（）༼༽]")
# Page ornaments / pecha flourishes / section-head marks — strip everywhere,
# keep the pair. U+0FD0–U+0FDA plus ༁–༊, ༒, ༓. Exclude ༔ (U+0F14).
ORNAMENT_CHARS_RE = re.compile(
    r"["
    r"\u0FD0-\u0FDA"  # ࿐–࿚
    r"\u0F01-\u0F0A"  # ༁–༊
    r"\u0F12\u0F13"  # ༒ ༓
    r"]"
)

# Extra punct/symbol chars beyond Unicode P*/S* that show up as OCR junk.
_EXTRA_NON_TIB_PUNCT = set(
    "—–―‐‑‒„“”‘’«»‹›…·•°′″´`~^|\\/<>{}[]@#$%&*_+=§†‡※◌×✳"
)


def paren_unmatched(text: str) -> bool:
    """True if any prefix depth goes negative or final depth ≠ 0 (ASCII only)."""
    bal = 0
    for ch in text:
        if ch == "(":
            bal += 1
        elif ch == ")":
            bal -= 1
            if bal < 0:
                return True
    return bal != 0


def has_bracket_chars(text: str) -> bool:
    return bool(BRACKET_CHARS_RE.search(text))


def has_balanced_parens(text: str) -> bool:
    """True if text contains (…) and parentheses are fully matched."""
    if "(" not in text:
        return False
    return not paren_unmatched(text)


def has_ornament_chars(text: str) -> bool:
    return bool(ORNAMENT_CHARS_RE.search(text))


def latin_runs(text: str) -> list[str]:
    return LATIN_RUN_RE.findall(text)


def has_latin(text: str) -> bool:
    return bool(LATIN_RUN_RE.search(text))


def one_side_latin_placeholder(src: str, tgt: str) -> bool:
    """Latin on exactly one side."""
    return has_latin(src) != has_latin(tgt)


def both_side_latin_placeholder(src: str, tgt: str) -> bool:
    """Latin on both sides."""
    return has_latin(src) and has_latin(tgt)


def any_side_latin_placeholder(src: str, tgt: str) -> bool:
    """Latin on either side (one or both)."""
    return has_latin(src) or has_latin(tgt)


def non_tibetan_punct_chars(text: str) -> list[str]:
    """Punctuation/symbols outside the Tibetan block (not digits/letters/space)."""
    found = []
    for ch in text:
        o = ord(ch)
        if 0x0F00 <= o <= 0x0FFF:
            continue
        if ch.isspace() or ch.isdigit():
            continue
        # Latin (ASCII or fullwidth) handled by the Latin filter.
        if "A" <= ch <= "Z" or "a" <= ch <= "z":
            continue
        if 0xFF21 <= o <= 0xFF5A:
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S") or ch in _EXTRA_NON_TIB_PUNCT:
            found.append(ch)
    return found


def has_non_tibetan_punct(text: str) -> bool:
    return bool(non_tibetan_punct_chars(text))


def any_side_non_tibetan_punct(src: str, tgt: str) -> bool:
    return has_non_tibetan_punct(src) or has_non_tibetan_punct(tgt)


def strip_bracket_chars(text: str) -> str:
    """Remove bracket characters; keep the enclosed text."""
    return BRACKET_CHARS_RE.sub("", text)


def strip_ornament_chars(text: str) -> str:
    """Remove page ornaments / section-head marks at all positions."""
    return ORNAMENT_CHARS_RE.sub("", text)


def strip_latin_placeholders(text: str) -> str:
    """Remove Latin placeholder runs (ASCII + fullwidth)."""
    return LATIN_RUN_RE.sub("", text)


def clean_page_artifacts(text: str) -> tuple[str, dict]:
    """Strip brackets (keep text) from a page transcript before segmentation.

    Ornaments are stripped at pair level after alignment (so transfer
    boundaries stay stable). Latin / non-Tibetan punct dropped at pair level.
    """
    stats = {
        "had_brackets": has_bracket_chars(text),
        "had_unmatched_parens": paren_unmatched(text) if ("(" in text or ")" in text) else False,
        "had_ornaments": has_ornament_chars(text),
        "had_latin": has_latin(text),
        "brackets_stripped": 0,
        "ornaments_stripped": 0,
        "latin_runs_stripped": 0,
    }
    if stats["had_brackets"]:
        stats["brackets_stripped"] = len(BRACKET_CHARS_RE.findall(text))
        text = strip_bracket_chars(text)
        # `(S)` / `(III)` become bare Latin after bracket strip.
        stats["had_latin"] = has_latin(text) or stats["had_latin"]
    return text, stats
