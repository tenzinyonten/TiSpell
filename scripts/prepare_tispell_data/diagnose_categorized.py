#!/usr/bin/env python3
"""Diagnostics on differing_categorized.csv — report only, no writes."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

PUNCT_CHARS = set("།༎༏༐༑་༌")


def fmt_char(c: str) -> str:
    if c == "":
        return "∅"
    if c.isspace():
        return "␠"
    return c


def align_chars(a: str, b: str):
    pairs = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(a, b):
        if tag == "equal":
            for k in range(i2 - i1):
                pairs.append((a[i1 + k], b[j1 + k]))
        elif tag == "replace":
            la, lb = i2 - i1, j2 - j1
            for k in range(max(la, lb)):
                ca = a[i1 + k] if k < la else ""
                cb = b[j1 + k] if k < lb else ""
                pairs.append((ca, cb))
        elif tag == "delete":
            for k in range(i1, i2):
                pairs.append((a[k], ""))
        elif tag == "insert":
            for k in range(j1, j2):
                pairs.append(("", b[k]))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "dataset/ocr_annotation/exports/all_sentence_pairs_differing_categorized.csv"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}", flush=True)

    print("\n" + "=" * 70)
    print("PART A: DIVERSITY AND LEAKAGE")
    print("=" * 70)

    n_rows = len(df)
    n_distinct_targets = df["target"].astype(str).nunique()
    print("\nA1) Distinct targets vs total rows")
    print(f"  total rows:        {n_rows}")
    print(f"  distinct targets:  {n_distinct_targets}")
    print(f"  ratio (distinct/rows): {n_distinct_targets / n_rows:.4f}")
    print(f"  avg rows per distinct target: {n_rows / n_distinct_targets:.2f}")

    print(
        "\nA2) Dual-annotator sources for the same target "
        "(same page_id + segment_idx + correct)"
    )
    gcols = ["page_id", "segment_idx", "target"]
    identical_src = different_src = both_n = 0
    for _, g in df.groupby(gcols):
        roles = set(g["source_role"].astype(str))
        if "annotator_a" not in roles or "annotator_b" not in roles:
            continue
        sa = str(g.loc[g["source_role"] == "annotator_a", "source"].iloc[0])
        sb = str(g.loc[g["source_role"] == "annotator_b", "source"].iloc[0])
        both_n += 1
        if sa == sb:
            identical_src += 1
        else:
            different_src += 1
    print(f"  target segments with both annotators: {both_n}")
    if both_n:
        print(
            f"  sources identical:  {identical_src} "
            f"({100 * identical_src / both_n:.1f}%)"
        )
        print(
            f"  sources different:  {different_src} "
            f"({100 * different_src / both_n:.1f}%)"
        )

    print("\nA3) Near-duplicate leakage (test source ↔ closest train source)")
    print("  method: length band |ΔL|<=5; RapidFuzz Levenshtein.cdist; score_cutoff=5")
    test_df = df[df["split"] == "test"]
    train_df = df[df["split"] == "train"]
    test_sources = test_df["source"].astype(str).tolist()
    train_sources = train_df["source"].astype(str).tolist()
    test_uniq = list(dict.fromkeys(test_sources))
    train_uniq = list(dict.fromkeys(train_sources))
    print(f"  test rows: {len(test_sources)}  unique test sources: {len(test_uniq)}")
    print(f"  train rows: {len(train_sources)}  unique train sources: {len(train_uniq)}")

    train_by_len: dict[int, list[str]] = defaultdict(list)
    for s in train_uniq:
        train_by_len[len(s)].append(s)
    test_by_len: dict[int, list[str]] = defaultdict(list)
    for s in test_uniq:
        test_by_len[len(s)].append(s)

    BAND = 5
    CHUNK = 200
    uniq_min: dict[str, int] = {}
    done = 0
    for L, tlist in sorted(test_by_len.items()):
        cands: list[str] = []
        for Ll in range(max(0, L - BAND), L + BAND + 1):
            cands.extend(train_by_len.get(Ll, []))
        if not cands:
            for s in tlist:
                uniq_min[s] = 999
            done += len(tlist)
            continue
        for start in range(0, len(tlist), CHUNK):
            chunk = tlist[start : start + CHUNK]
            mat = cdist(
                chunk,
                cands,
                scorer=Levenshtein.distance,
                workers=-1,
                score_cutoff=5,
            )
            for i, s in enumerate(chunk):
                row = mat[i]
                uniq_min[s] = int(row.min()) if len(row) else 999
            done += len(chunk)
        if done % 1000 < CHUNK or done == len(test_uniq):
            print(f"  ... {done}/{len(test_uniq)} unique test sources", flush=True)

    row_mins = [uniq_min[s] for s in test_sources]
    n_test = len(row_mins)
    print(f"\n  Near-neighbour counts (over {n_test} test rows):")
    for t in (1, 2, 3, 5):
        c = sum(1 for d in row_mins if d <= t)
        print(f"    within edit distance {t}: {c}  ({100 * c / n_test:.2f}%)")

    print(f"\n  Near-neighbour counts (over {len(test_uniq)} unique test sources):")
    for t in (1, 2, 3, 5):
        c = sum(1 for s in test_uniq if uniq_min[s] <= t)
        print(f"    within edit distance {t}: {c}  ({100 * c / len(test_uniq):.2f}%)")

    exact0 = sum(1 for d in row_mins if d == 0)
    gt5 = sum(1 for d in row_mins if d > 5)
    print(
        f"\n  exact match dist=0 (test source also in train): "
        f"{exact0} ({100 * exact0 / n_test:.2f}%)"
    )
    print(f"  no train neighbour within 5: {gt5} ({100 * gt5 / n_test:.2f}%)")

    print("\n" + "=" * 70)
    print("PART B: SAMPLES TO READ")
    print("=" * 70)
    rng = random.Random(args.seed)

    print("\n----- B1) 10 random large_rewrite pairs -----")
    lr = df[df["diff_category"] == "large_rewrite"]
    idxs = list(lr.index)
    chosen = idxs if len(idxs) <= 10 else rng.sample(idxs, 10)
    for rank, idx in enumerate(chosen, 1):
        row = df.loc[idx]
        src, tgt = str(row["source"]), str(row["target"])
        dist = (
            int(row["edit_distance"])
            if "edit_distance" in row and pd.notna(row["edit_distance"])
            else Levenshtein.distance(src, tgt)
        )
        pct = 100 * dist / max(len(src), len(tgt), 1)
        print(f"\n[{rank}] page={row['page_id']} seg={row['segment_idx']} split={row['split']}")
        print(f"SOURCE: {src}")
        print(f"TARGET: {tgt}")
        print(f"(edit distance: {dist}, percent changed: {pct:.1f}%)")

    print("\n----- B2) Full punctuation_only substitution table -----")
    punct = df[df["diff_category"] == "punctuation_only"]
    counts: Counter = Counter()
    for src, tgt in zip(punct["source"].astype(str), punct["target"].astype(str)):
        for a, b in align_chars(src, tgt):
            if a == b:
                continue
            a_is = (a in PUNCT_CHARS) or (a != "" and a.isspace())
            b_is = (b in PUNCT_CHARS) or (b != "" and b.isspace())
            if a_is or b_is:
                counts[f"{fmt_char(a)} → {fmt_char(b)}"] += 1
    print(
        f"(from {len(punct)} punctuation_only pairs; "
        f"{sum(counts.values())} punct-involving ops)"
    )
    print(f"{'substitution':<20} {'count':>8}")
    print("-" * 30)
    for k, v in counts.most_common():
        print(f"{k:<20} {v:>8}")

    print("\n----- B3) 10 random syllable_level pairs -----")
    print(
        "Note: multi_edit=7/53k ⇒ syllable_level matches first "
        "for most 2–5-edit tsheg cases."
    )
    syl = df[df["diff_category"] == "syllable_level"]
    idxs = list(syl.index)
    chosen = idxs if len(idxs) <= 10 else rng.sample(idxs, 10)
    for rank, idx in enumerate(chosen, 1):
        row = df.loc[idx]
        src, tgt = str(row["source"]), str(row["target"])
        dist = (
            int(row["edit_distance"])
            if "edit_distance" in row and pd.notna(row["edit_distance"])
            else Levenshtein.distance(src, tgt)
        )
        print(f"\n[{rank}] page={row['page_id']} seg={row['segment_idx']} split={row['split']}")
        print(f"SOURCE: {src}")
        print(f"TARGET: {tgt}")
        print(f"(edit distance: {dist})")

    print("\n" + "=" * 70)
    print("END OF REPORT (no files written)")
    print("=" * 70)


if __name__ == "__main__":
    main()
