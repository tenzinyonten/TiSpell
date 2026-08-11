#!/usr/bin/env python3
"""Classify (source, target) pairs in the differing clean export by edit type.

Categories are assigned in order (first match wins). Does not filter or
modify training decisions — only labels and reports.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

PUNCT_RE = re.compile(r"[།༎༏༐༑་༌\s]+")
PUNCT_CHARS = set("།༎༏༐༑་༌")
WHITESPACE_RE = re.compile(r"\s+")
TSHEG_SPLIT_RE = re.compile(r"[་༌]")
SHAD_RE = re.compile(r"[།༎༏༐༑]")

CATEGORIES = [
    "punctuation_only",
    "whitespace_only",
    "single_char_substitution",
    "single_char_insert_delete",
    "syllable_level",
    "multi_edit",
    "large_rewrite",
]


def levenshtein(a: str, b: str) -> int:
    """Classic Wagner–Fischer Levenshtein distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Ensure b is shorter for memory
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def strip_tibetan_punct(text: str) -> str:
    return PUNCT_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub("", text)


def split_syllables(text: str) -> list[str]:
    """Tsheg-delimited units; trailing shad kept on the last unit if present."""
    text = text.strip()
    if not text:
        return []
    # Keep shad attached by temporarily protecting it, split on tsheg only.
    parts = TSHEG_SPLIT_RE.split(text)
    return [p for p in parts if p != ""]


def is_syllable_level(src: str, tgt: str, dist: int) -> bool:
    """True if differences are whole tsheg-delimited units only.

    Does not claim large rewrites: if char edit distance > 5 or >30% of the
    longer side, return False so those fall through to large_rewrite.
    """
    max_len = max(len(src), len(tgt), 1)
    if dist > 5 or (dist / max_len) > 0.30:
        return False
    s_syl = split_syllables(src)
    t_syl = split_syllables(tgt)
    if s_syl == t_syl:
        return False
    return _syllable_edits_explain(s_syl, t_syl, src, tgt)


def _syllable_edits_explain(
    s_syl: list[str], t_syl: list[str], src: str, tgt: str
) -> bool:
    """Check syllable-token alignment fully accounts for the string change."""
    # DP align syllables
    n, m = len(s_syl), len(t_syl)
    if n == 0 and m == 0:
        return False
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s_syl[i - 1] == t_syl[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    if dp[n][m] == 0:
        return False

    # Backtrace: rebuild strings from ops; must equal src/tgt when rejoined
    # with tsheg between units (best-effort). More reliable test:
    # every character-level unequal span should map to complete syllables.
    i, j = n, m
    ops = []  # list of ('equal'|'replace'|'delete'|'insert', s_tok, t_tok)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s_syl[i - 1] == t_syl[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("equal", s_syl[i - 1], t_syl[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("replace", s_syl[i - 1], t_syl[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("delete", s_syl[i - 1], ""))
            i -= 1
        else:
            ops.append(("insert", "", t_syl[j - 1]))
            j -= 1
    ops.reverse()

    # At least one non-equal op, and no 'replace' where tokens are equal
    # (already enforced). Require that unequal ops involve complete tokens
    # (always true by construction). Reject if either side has no tsheg and
    # both are single tokens that are character-near — those are char edits
    # already classified earlier when distance is 1; for longer within-token
    # edits, treating as syllable_level is acceptable when the whole segment
    # is one syllable, OR we can call it multi_edit/large_rewrite instead.
    # If both are single syllable (no internal tsheg), the "unit" is the
    # whole segment — that's not really syllable-level structure.
    has_tsheg = ("་" in src or "༌" in src) or ("་" in tgt or "༌" in tgt)
    if not has_tsheg:
        return False
    return any(op[0] != "equal" for op in ops)


def classify_punctuation_only_subtype(src: str, tgt: str) -> str:
    """Split punctuation_only pairs: terminal shad vs in-text tsheg vs other.

    Returns one of: ``terminal_shad_only``, ``tsheg_within``, ``other_punct``.
    Assumes the pair is already ``punctuation_only`` (letter content matches).
    """
    src, tgt = str(src), str(tgt)
    tsheg = set("་༌")

    def strip_trailing_shad_ws(s: str) -> str:
        return re.sub(r"[།༎༏༐༑\s]+$", "", s)

    def strip_tsheg(s: str) -> str:
        return "".join(c for c in s if c not in tsheg)

    s_core, t_core = strip_trailing_shad_ws(src), strip_trailing_shad_ws(tgt)
    if s_core == t_core and src != tgt:
        return "terminal_shad_only"

    if strip_tsheg(src) == strip_tsheg(tgt) and src != tgt:
        return "tsheg_within"

    def tsheg_signature(s: str) -> tuple[tuple[bool, ...], int]:
        sig: list[bool] = []
        prev_tsheg = False
        for c in s:
            if c in tsheg:
                prev_tsheg = True
            elif c in PUNCT_CHARS or c.isspace():
                pass
            else:
                sig.append(prev_tsheg)
                prev_tsheg = False
        return (tuple(sig), sum(1 for c in s if c in tsheg))

    if tsheg_signature(src) != tsheg_signature(tgt):
        return "tsheg_within"
    return "other_punct"


def classify_pair(src: str, tgt: str) -> tuple[str, int]:
    src = str(src)
    tgt = str(tgt)
    dist = levenshtein(src, tgt)

    if strip_tibetan_punct(src) == strip_tibetan_punct(tgt):
        return "punctuation_only", dist

    if collapse_whitespace(src) == collapse_whitespace(tgt):
        return "whitespace_only", dist

    if len(src) == len(tgt) and dist == 1:
        return "single_char_substitution", dist

    if abs(len(src) - len(tgt)) == 1 and dist == 1:
        return "single_char_insert_delete", dist

    if is_syllable_level(src, tgt, dist):
        return "syllable_level", dist

    max_len = max(len(src), len(tgt), 1)
    frac = dist / max_len
    if dist > 5 or frac > 0.30:
        return "large_rewrite", dist

    if 2 <= dist <= 5:
        return "multi_edit", dist

    # Fallback (e.g. dist==0 shouldn't appear in differing file)
    if dist == 0:
        return "whitespace_only", dist
    return "multi_edit", dist


def punct_substitution_breakdown(df: pd.DataFrame) -> Counter:
    """Count punctuation char substitutions for punctuation_only pairs."""
    counts: Counter = Counter()
    for src, tgt in zip(df["source"].astype(str), df["target"].astype(str)):
        # Align full strings; record punct↔punct or punct↔empty changes
        # Use a simple LCS-style scan via DP backtrace on short strings
        for a, b in _align_chars(src, tgt):
            if a == b:
                continue
            a_is_p = (a in PUNCT_CHARS) or (a != "" and a.isspace())
            b_is_p = (b in PUNCT_CHARS) or (b != "" and b.isspace())
            if a_is_p or b_is_p:
                left = a if a else "∅"
                right = b if b else "∅"
                if left.isspace():
                    left = "␠"
                if right.isspace():
                    right = "␠"
                counts[f"{left} → {right}"] += 1
    return counts


def _align_chars(a: str, b: str) -> list[tuple[str, str]]:
    """Return aligned char pairs ('' for gap) via Levenshtein backtrace."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            pairs.append((a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            pairs.append((a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((a[i - 1], ""))
            i -= 1
        else:
            pairs.append(("", b[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "dataset/ocr_annotation/exports/all_sentence_pairs_differing_clean_fixed.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "dataset/ocr_annotation/exports/all_sentence_pairs_differing_categorized.csv"
        ),
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} differing pairs from {args.input}")

    categories = []
    distances = []
    for src, tgt in zip(df["source"].astype(str), df["target"].astype(str)):
        cat, dist = classify_pair(src, tgt)
        categories.append(cat)
        distances.append(dist)

    df = df.copy()
    df["diff_category"] = categories
    df["edit_distance"] = distances

    # --- counts table ---
    counts = df["diff_category"].value_counts()
    n = len(df)
    print("\n" + "=" * 64)
    print("CATEGORY COUNTS")
    print("=" * 64)
    print(f"{'category':<28} {'count':>8} {'pct':>8}")
    print("-" * 46)
    for cat in CATEGORIES:
        c = int(counts.get(cat, 0))
        print(f"{cat:<28} {c:>8} {100*c/n:>7.2f}%")
    # any unexpected
    for cat in counts.index:
        if cat not in CATEGORIES:
            c = int(counts[cat])
            print(f"{cat:<28} {c:>8} {100*c/n:>7.2f}%")
    print("-" * 46)
    print(f"{'TOTAL':<28} {n:>8} {100:>7.2f}%")

    # punctuation breakdown
    punct_df = df[df["diff_category"] == "punctuation_only"]
    if len(punct_df):
        print("\n" + "=" * 64)
        print("PUNCTUATION_ONLY SUBSTITUTION BREAKDOWN")
        print("=" * 64)
        breakdown = punct_substitution_breakdown(punct_df)
        for k, v in breakdown.most_common(40):
            print(f"  {k}: {v}")
        if not breakdown:
            print("  (no punct-char alignments recorded)")

    ws = int(counts.get("whitespace_only", 0))
    if ws:
        print(
            f"\nNOTE: {ws} whitespace_only pairs remain — "
            "these should have been removed by normalization."
        )

    # samples
    rng = random.Random(args.seed)
    print("\n" + "=" * 64)
    print(f"SAMPLES ({args.samples} per category, seed={args.seed})")
    print("=" * 64)
    for cat in CATEGORIES:
        subset = df[df["diff_category"] == cat]
        print(f"\n----- {cat} (n={len(subset)}) -----")
        if subset.empty:
            print("  (none)")
            continue
        idxs = list(subset.index)
        chosen = idxs if len(idxs) <= args.samples else rng.sample(idxs, args.samples)
        for rank, idx in enumerate(chosen, 1):
            row = df.loc[idx]
            print(f"\n  [{rank}]")
            print(f"  SOURCE: {row['source']}")
            print(f"  TARGET: {row['target']}")
            print(f"  (edit distance: {row['edit_distance']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nWrote {len(df)} rows → {args.output}")
    print("(original file left unchanged)")


if __name__ == "__main__":
    main()
