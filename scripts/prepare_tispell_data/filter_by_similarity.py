#!/usr/bin/env python3
"""Post-transfer SequenceMatcher similarity gate (same as old align_sentences).

Drops pairs with ratio < min_ratio. Writes filtered CSVs alongside inputs and
prints drop stats + near-cutoff samples.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_sentences import similarity
from categorize_diffs import CATEGORIES, classify_pair


def bucket(d: int) -> str:
    if d == 0:
        return "0"
    if d == 1:
        return "1"
    if d <= 5:
        return "2-5"
    if d <= 10:
        return "6-10"
    if d <= 30:
        return "11-30"
    return "31+"


def out_name(name: str, out_suffix: str) -> str:
    p = Path(name)
    stem = p.stem
    if stem.endswith("_antx"):
        stem = stem[: -len("_antx")] + f"_{out_suffix}"
    elif "_antx_" in stem:
        stem = stem.replace("_antx_", f"_{out_suffix}_")
    else:
        stem = f"{stem}_{out_suffix}"
    return f"{stem}{p.suffix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exports_dir",
        type=Path,
        default=Path("dataset/ocr_annotation/exports"),
    )
    parser.add_argument(
        "--differing",
        type=str,
        default="all_sentence_pairs_differing_clean_antx.csv",
    )
    parser.add_argument(
        "--identical",
        type=str,
        default="all_sentence_pairs_identical_clean_antx.csv",
    )
    parser.add_argument("--out_suffix", type=str, default="antx_sim")
    parser.add_argument("--min_ratio", type=float, default=0.55)
    parser.add_argument("--drop_samples", type=int, default=15)
    args = parser.parse_args()

    diff_in = args.exports_dir / args.differing
    ident_in = args.exports_dir / args.identical
    diff = pd.read_csv(diff_in)
    ident = pd.read_csv(ident_in)

    diff_out = args.exports_dir / out_name(args.differing, args.out_suffix)
    ident_out = args.exports_dir / out_name(args.identical, args.out_suffix)
    cat_path = (
        args.exports_dir
        / f"all_sentence_pairs_differing_categorized_{args.out_suffix}.csv"
    )

    records = []
    for kind, df in (("differing", diff), ("identical", ident)):
        for _, r in df.iterrows():
            src = str(r["source"])
            tgt = str(r["target"])
            records.append({**r.to_dict(), "_kind": kind, "_ratio": similarity(src, tgt)})

    all_df = pd.DataFrame(records)
    keep_mask = all_df["_ratio"] >= args.min_ratio
    kept = all_df[keep_mask].copy()
    dropped = all_df[~keep_mask].copy()

    kept_diff = kept[kept["_kind"] == "differing"].drop(columns=["_kind", "_ratio"])
    kept_ident = kept[kept["_kind"] == "identical"].drop(columns=["_kind", "_ratio"])

    eq = kept_diff["source"].astype(str) == kept_diff["target"].astype(str)
    if eq.any():
        kept_ident = pd.concat([kept_ident, kept_diff[eq]], ignore_index=True)
        kept_diff = kept_diff[~eq]

    kept_diff.to_csv(diff_out, index=False, encoding="utf-8")
    kept_ident.to_csv(ident_out, index=False, encoding="utf-8")

    n_before = len(all_df)
    n_dropped = len(dropped)
    n_kept = len(kept_diff) + len(kept_ident)
    n0 = len(kept_ident)

    diff_dists = [
        Levenshtein.distance(str(a), str(b))
        for a, b in zip(kept_diff["source"], kept_diff["target"])
    ]
    all_dists = [0] * len(kept_ident) + diff_dists
    b_all = Counter(bucket(d) for d in all_dists)
    b_diff = Counter(bucket(d) for d in diff_dists)

    cats = []
    ratios_kept_diff = []
    for src, tgt in zip(
        kept_diff["source"].astype(str), kept_diff["target"].astype(str)
    ):
        cat, _ = classify_pair(src, tgt)
        cats.append(cat)
        ratios_kept_diff.append(similarity(src, tgt))
    cat_counts = Counter(cats)

    print("=" * 64)
    print(f"SIMILARITY FILTER (min_ratio={args.min_ratio})")
    print("=" * 64)
    print(f"Input:  {diff_in.name} + {ident_in.name}")
    print(f"Before: {n_before}  (diff={len(diff)}, ident={len(ident)})")
    print(f"Dropped by similarity: {n_dropped}  ({100 * n_dropped / n_before:.2f}%)")
    print(f"Remaining: {n_kept}  (diff={len(kept_diff)}, ident={len(kept_ident)})")
    print(f"dist=0 rate: {n0}/{n_kept} = {100 * n0 / n_kept:.2f}%")

    order = ["0", "1", "2-5", "6-10", "11-30", "31+"]
    print("\nEdit-distance buckets (ALL remaining):")
    for k in order:
        print(f"  {k:>5}: {b_all[k]:7d}  ({100 * b_all[k] / n_kept:5.2f}%)")
    print("\nEdit-distance buckets (DIFFERING remaining):")
    denom = max(len(kept_diff), 1)
    for k in order:
        print(f"  {k:>5}: {b_diff[k]:7d}  ({100 * b_diff[k] / denom:5.2f}%)")

    print("\nCategory counts (remaining differing):")
    n_diff = len(kept_diff)
    for cat in CATEGORIES:
        c = cat_counts.get(cat, 0)
        pct = 100 * c / n_diff if n_diff else 0.0
        print(f"  {cat:<28} {c:7d}  ({pct:5.2f}%)")
    lr = cat_counts.get("large_rewrite", 0)
    print(f"\nlarge_rewrite: {lr}/{n_diff} = {100 * lr / n_diff:.2f}%")

    dropped_sorted = dropped.sort_values("_ratio", ascending=False)
    sample = dropped_sorted.head(args.drop_samples)
    print("\n" + "=" * 64)
    print(
        f"{args.drop_samples} DROPPED pairs "
        "(similarity descending — closest to cutoff first)"
    )
    print("=" * 64)
    for i, (_, r) in enumerate(sample.iterrows(), 1):
        print(
            f"\n[{i}] ratio={r['_ratio']:.4f}  kind={r['_kind']}  "
            f"page={r.get('page_id', '')}  seg={r.get('segment_idx', '')}"
        )
        print(f"  SOURCE: {r['source']}")
        print(f"  TARGET: {r['target']}")

    print("\nDropped ratio distribution:")
    edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55]
    ratios = dropped["_ratio"].tolist()
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = sum(1 for x in ratios if lo <= x < hi)
        print(f"  [{lo:.2f}, {hi:.2f}): {n}")

    cat_df = kept_diff.copy()
    cat_df["diff_category"] = cats
    cat_df["edit_distance"] = diff_dists
    cat_df["similarity_ratio"] = ratios_kept_diff
    cat_df.to_csv(cat_path, index=False, encoding="utf-8")

    print()
    print(f"Wrote {len(kept_diff)} → {diff_out}")
    print(f"Wrote {len(kept_ident)} → {ident_out}")
    print(f"Wrote {len(cat_df)} → {cat_path}")


if __name__ == "__main__":
    main()
