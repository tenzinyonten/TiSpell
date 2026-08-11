"""Align annotator text to reviewer sentence boundaries via fast_antx.transfer.

Same pattern as the STT etext pipeline:

    source  = reviewer segments joined with newlines
    target  = annotator page text (continuous)
    result  = transfer(source, [["segment", "(\\n)"]], target).split("\\n")

The packaged dmp binary is x86_64-only, so we patch fast_antx to use the
Python diff_match_patch backend before calling transfer().
"""

from __future__ import annotations

import logging
import re

from diff_match_patch import diff_match_patch
from fast_antx.core import transfer

from tibetan_normalize import normalize_transcript, segment_sentences

logging.getLogger("fast_antx").setLevel(logging.ERROR)

ANNOTATION = [["segment", "(\n)"]]
# Plain leading shad / tsheg / whitespace from a cut landing on punctuation.
LEADING_BOUNDARY_RE = re.compile(r"^[\s།༎༏༐༑་༌]+")
# Boundary cut mid-syllable: stranded combining vowel / subjoined consonant /
# mark at segment start, with or without following shad/tsheg/whitespace.
# e.g. "ོ། སྲིད་…" / "ྲབཀྲ་…" / "࿁་བྱི་…" → strip the leading debris.
LEADING_DEBRIS_RE = re.compile(
    r"^["
    r"\u0F71-\u0F84"  # vowel signs, anusvara, visarga, …
    r"\u0F86-\u0F87"  # marks above
    r"\u0F8D-\u0FBC"  # subjoined consonants
    r"\u0F35\u0F37\u0F39"  # honorific / emphasis marks
    r"\u0FBE-\u0FCF"  # kurukha / cantillation / related marks (྾ ྿ ࿁ …)
    r"]+[།༎༏༐༑་༌\s]*"
)

_DMP = diff_match_patch()
_PATCHED = False


def _patch_fast_antx_python_dmp() -> None:
    """Replace native dmp subprocess with Python diff_match_patch."""
    global _PATCHED
    if _PATCHED:
        return
    import fast_antx.core as antx_core

    def get_diffs(text1, text2):
        diffs = _DMP.diff_main(text1, text2)
        _DMP.diff_cleanupSemantic(diffs)
        return [[t, txt] for t, txt in diffs]

    antx_core.get_diffs = get_diffs
    _PATCHED = True


def strip_leading_boundary_artifacts(text: str) -> str:
    """Remove leading boundary-transfer debris (combining marks, shad, ws)."""
    prev = None
    while prev != text:
        prev = text
        text = LEADING_DEBRIS_RE.sub("", text)
        text = LEADING_BOUNDARY_RE.sub("", text)
    return text


# Short complete syllable + shad stranded from the previous sentence.
# Matched via cross-side compare (not a blind pattern): only strip when
# removing it makes that side's opening match the other side.
_SHAD_CHARS = "\u0F0D\u0F0E\u0F0F\u0F10\u0F11"
_BODY_CHAR = r"[\u0F00-\u0F0C\u0F12-\u0FFF]"
LEADING_SYLLABLE_RE = re.compile(
    rf"^({_BODY_CHAR}{{1,6}}[{_SHAD_CHARS}])([\u0F0B\u0F0C\s]*)"
)


def _leading_syllable_prefix(text: str) -> tuple[str, str] | None:
    m = LEADING_SYLLABLE_RE.match(text)
    if not m:
        return None
    prefix = m.group(0)
    rest = text[len(prefix) :]
    if not rest:
        return None
    return prefix, rest


def _openings_match(a: str, b: str, min_len: int = 4) -> bool:
    if not a or not b:
        return False
    if a == b or a.startswith(b) or b.startswith(a):
        return True
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i >= min_len


def strip_leading_syllable_by_compare(src: str, tgt: str) -> tuple[str, str, bool]:
    """Strip a short leading syll+shad from one side if that matches the other.

    Returns (src, tgt, stripped).
    """
    for side in ("src", "tgt"):
        text, other = (src, tgt) if side == "src" else (tgt, src)
        sp = _leading_syllable_prefix(text)
        if not sp:
            continue
        prefix, rest = sp
        op = _leading_syllable_prefix(other)
        if op and op[0].rstrip() == prefix.rstrip():
            continue  # same prefix on both sides — content, not debris
        if _openings_match(text, other, min_len=4):
            continue  # already aligned at opening
        if not _openings_match(rest, other, min_len=4):
            continue
        if side == "src":
            return rest, tgt, True
        return src, rest, True
    return src, tgt, False


# Content punctuation mid-segment (༴ nyis shad, ༔ gter shad). Only strip when
# leading and removing the mark(+following tsheg/ws) makes openings match.
_LEADING_CONTENT_PUNCT = {"\u0F34", "\u0F14"}  # ༴ ༔
_LEADING_CONTENT_PUNCT_FOLLOW = {"\u0F0B", "\u0F0C", " ", "\t"}  # ་ ༌ ws
LEADING_ASCII_DIGITS_RE = re.compile(r"^[0-9]+")


def _leading_content_punct_prefix(text: str) -> tuple[str, str] | None:
    if not text or text[0] not in _LEADING_CONTENT_PUNCT:
        return None
    i = 0
    while i < len(text) and text[i] in _LEADING_CONTENT_PUNCT:
        i += 1
    j = i
    while j < len(text) and text[j] in _LEADING_CONTENT_PUNCT_FOLLOW:
        j += 1
    rest = text[j:]
    if not rest:
        return None
    return text[:j], rest


def strip_leading_content_punct_by_compare(src: str, tgt: str) -> tuple[str, str, bool]:
    """Strip leading ༴/༔ (+tsheg/ws) when that aligns openings; leave mid alone.

    Same cross-side rule as ``strip_leading_syllable_by_compare``. Identical
    leading marks on both sides are treated as content and kept. May strip
    more than once (e.g. ``༔་༔…``).
    """
    changed = False
    for _ in range(4):
        did = False
        for side in ("src", "tgt"):
            text, other = (src, tgt) if side == "src" else (tgt, src)
            sp = _leading_content_punct_prefix(text)
            if not sp:
                continue
            prefix, rest = sp
            op = _leading_content_punct_prefix(other)
            if op and op[0] == prefix:
                continue  # same leading mark run on both — content
            if _openings_match(text, other, min_len=4):
                continue  # already aligned at opening
            if not _openings_match(rest, other, min_len=4):
                continue
            if side == "src":
                src = rest
            else:
                tgt = rest
            changed = True
            did = True
            break
        if not did:
            break
    return src, tgt, changed


def has_leading_ascii_digits(text: str) -> bool:
    """True if text begins with one or more ASCII digits (folio/line debris)."""
    return bool(text) and LEADING_ASCII_DIGITS_RE.match(text) is not None


def flatten_layout(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "")


def align_annotator_to_reviewer(
    annotator: str,
    reviewer: str,
) -> tuple[list[tuple[str, str]], dict]:
    """Segment reviewer on shad; transfer those cuts onto annotator via fast_antx."""
    _patch_fast_antx_python_dmp()

    rev_flat = flatten_layout(reviewer)
    ann_flat = flatten_layout(annotator)
    rev_segs = segment_sentences(rev_flat)
    stats = {
        "reviewer_segs": len(rev_segs),
        "annotator_segs": 0,
        "count_agree": False,
        "n_pairs": 0,
    }
    if not rev_segs:
        return [], stats

    # Same as STT transfer_text():
    #   source_text = "\n".join(segments)
    #   transferred = transfer(source, annotation, target).split("\n")
    source_text = "\n".join(rev_segs)
    transferred = transfer(source_text, ANNOTATION, ann_flat, output="txt")
    ann_segs = []
    for s in str(transferred).split("\n"):
        if s == "":
            continue
        s = strip_leading_boundary_artifacts(s)
        s = normalize_transcript(s) or ""
        if s:
            ann_segs.append(s)
    # Same leading-debris + space/tsheg cleanup on reviewer segs.
    rev_segs_clean = []
    for s in rev_segs:
        s = strip_leading_boundary_artifacts(s)
        s = normalize_transcript(s) or ""
        if s:
            rev_segs_clean.append(s)
    rev_segs = rev_segs_clean

    stats["annotator_segs"] = len(ann_segs)
    stats["count_agree"] = len(ann_segs) == len(rev_segs)

    # Pad / truncate like the STT script so lengths match
    if len(ann_segs) > len(rev_segs):
        ann_segs = ann_segs[: len(rev_segs)]
    elif len(ann_segs) < len(rev_segs):
        ann_segs = ann_segs + [""] * (len(rev_segs) - len(ann_segs))

    pairs = [(a, r) for a, r in zip(ann_segs, rev_segs) if a]
    stats["n_pairs"] = len(pairs)
    return pairs, stats
