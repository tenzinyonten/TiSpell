"""Tibetan transcript normalization and sentence segmentation helpers."""

from __future__ import annotations

import re
import unicodedata

SHAD_RE = re.compile(r"[།༎༏༐༑]")
SHAD_END_RE = re.compile(r"[།༎༏༐༑]$")
WHITESPACE_RE = re.compile(r"\s+")
DOUBLE_TSHEG_RE = re.compile(r"[་༌]{2,}")
TSHEG_BEFORE_SHAD_RE = re.compile(r"[་༌]+([།༎༏༐༑])")
# Space adjacent to punctuation / syllable boundaries
SPACE_BEFORE_SHAD_RE = re.compile(r"\s+([།༎༏༐༑])")
SPACE_AFTER_TSHEG_RE = re.compile(r"([་༌])\s+")
# Space between Tibetan letters ≡ tsheg (canonicalize to tsheg).
# Exclude tsheg (0F0B–0F0C) and shad marks (0F0D–0F11) from either side.
SPACE_BETWEEN_TIBETAN_RE = re.compile(
    r"([\u0F00-\u0F0A\u0F12-\u0FFF])\s+([\u0F00-\u0F0A\u0F12-\u0FFF])"
)


def normalize_transcript(text: str | None) -> str | None:
    """NFC + whitespace + tsheg cleanup. Returns None for empty input."""
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFC", text)
    # Same leading/section mark, different codepoints: annotators often write
    # CHE MGO ༸ (U+0F38) where reviewers write digit-seven ༧ (U+0F27).
    # Map to the reviewer form (do not map ༧→༸ — ༧ is also a real digit).
    text = text.replace("\u0F38", "\u0F27")
    # Paluta ྅ (U+0F85) vs digit-three ༣ (U+0F23): one-way convention across
    # ume batches (do not map ༣→྅ — ༣ is a real digit).
    text = text.replace("\u0F85", "\u0F23")
    # Newlines are layout; collapse with other whitespace (do not use as splits).
    text = WHITESPACE_RE.sub(" ", text).strip()
    # ཐུག ། → ཐུག།
    text = SPACE_BEFORE_SHAD_RE.sub(r"\1", text)
    # ནི་ གཉེན → ནི་གཉེན
    text = SPACE_AFTER_TSHEG_RE.sub(r"\1", text)
    # འདིའི ལུང → འདིའི་ལུང  (tsheg and space in same position)
    prev = None
    while prev != text:
        prev = text
        text = SPACE_BETWEEN_TIBETAN_RE.sub(r"\1་\2", text)
    text = DOUBLE_TSHEG_RE.sub("་", text)
    text = TSHEG_BEFORE_SHAD_RE.sub(r"\1", text)
    text = text.strip()
    return text or None


def segment_sentences(text: str) -> list[str]:
    """Split on shad, keeping the shad on the preceding segment. No newline splits."""
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "")
    parts = SHAD_RE.split(text)
    delims = SHAD_RE.findall(text)
    sentences = []
    for i, part in enumerate(parts):
        s = WHITESPACE_RE.sub(" ", part).strip()
        if not s:
            continue
        if i < len(delims):
            s = s + delims[i]
        # Re-apply space/tsheg rules in case split reintroduced edge cases.
        s = normalize_transcript(s) or ""
        if s:
            sentences.append(s)
    return sentences


def filter_page_segments(
    segments: list[str],
    min_chars: int = 15,
    drop_first: bool = True,
) -> tuple[list[str], dict]:
    """Apply page-edge filters. Returns (kept, drop_counts)."""
    counts = {
        "before": len(segments),
        "dropped_first": 0,
        "dropped_no_shad": 0,
        "dropped_short": 0,
        "after": 0,
    }
    if not segments:
        return [], counts

    work = list(segments)
    if drop_first and work:
        work = work[1:]
        counts["dropped_first"] = 1

    kept = []
    for s in work:
        if not SHAD_END_RE.search(s):
            counts["dropped_no_shad"] += 1
            continue
        if len(s) < min_chars:
            counts["dropped_short"] += 1
            continue
        kept.append(s)
    counts["after"] = len(kept)
    return kept, counts
