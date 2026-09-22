#!/usr/bin/env python3
"""Verify and fix clean sentence-pair exports before training.

Checks:
  1) Length cap vs max_char_length (default 380)
  2) Exact (source, target) duplicates and cross-split source leakage
  3) Page split integrity (annotator_a/b from same page share split)

Prints a full report first, then writes corrected exports alongside originals
(does not overwrite).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Prefer train when resolving cross-split collisions so eval stays cleaner.
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}


def length_stats(series: pd.Series) -> dict:
    s = series.astype(str).map(len)
    return {
        "n": int(len(s)),
        "mean": round(float(s.mean()), 1) if len(s) else 0,
        "p50": int(s.median()) if len(s) else 0,
        "p90": int(s.quantile(0.90)) if len(s) else 0,
        "p95": int(s.quantile(0.95)) if len(s) else 0,
        "p99": int(s.quantile(0.99)) if len(s) else 0,
        "max": int(s.max()) if len(s) else 0,
    }


def check_lengths(df: pd.DataFrame, max_chars: int, label: str) -> dict:
    src = df["source"].astype(str).map(len)
    tgt = df["target"].astype(str).map(len)
    over_src = src > max_chars
    over_tgt = tgt > max_chars
    over_either = over_src | over_tgt
    return {
        "label": label,
        "n": len(df),
        "src": length_stats(df["source"]),
        "tgt": length_stats(df["target"]),
        "over_src": int(over_src.sum()),
        "over_tgt": int(over_tgt.sum()),
        "over_either": int(over_either.sum()),
        "over_either_pct": round(100 * float(over_either.mean()), 2) if len(df) else 0.0,
        "mask_over": over_either,
    }


def check_duplicates_and_leakage(df: pd.DataFrame) -> dict:
    pair_key = df["source"].astype(str) + "\t" + df["target"].astype(str)
    vc = pair_key.value_counts()
    n_unique = int((vc == 1).sum() + (vc > 1).sum())  # unique keys
    n_dup_keys = int((vc > 1).sum())
    n_extra = int(len(df) - len(vc))

    # Cross-split source leakage
    src_to_splits: dict[str, set[str]] = defaultdict(set)
    src_to_rows: dict[str, list[int]] = defaultdict(list)
    for idx, (src, split) in enumerate(
        zip(df["source"].astype(str), df["split"].astype(str))
    ):
        src_to_splits[src].add(split)
        src_to_rows[src].append(idx)

    multi = {s: splits for s, splits in src_to_splits.items() if len(splits) > 1}
    overlap_types = Counter(tuple(sorted(sp)) for sp in multi.values())

    # Also target leakage (gold text in multiple splits) — useful signal
    tgt_to_splits: dict[str, set[str]] = defaultdict(set)
    for tgt, split in zip(df["target"].astype(str), df["split"].astype(str)):
        tgt_to_splits[tgt].add(split)
    multi_tgt = {t: sp for t, sp in tgt_to_splits.items() if len(sp) > 1}

    return {
        "n_rows": len(df),
        "n_unique_pairs": len(vc),
        "n_duplicate_pair_keys": n_dup_keys,
        "n_extra_duplicate_rows": n_extra,
        "n_sources_multi_split": len(multi),
        "source_overlap_types": {
            "/".join(k): int(v) for k, v in sorted(overlap_types.items())
        },
        "n_targets_multi_split": len(multi_tgt),
        "multi_source_map": multi,
        "pair_value_counts": vc,
        "pair_keys": pair_key,
    }


def check_page_split_integrity(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_pages": 0, "n_violations": 0, "violating_pages": []}
    g = df.groupby("page_id")["split"].nunique()
    bad_pages = g[g > 1].index.tolist()
    return {
        "n_pages": int(g.shape[0]),
        "n_violations": len(bad_pages),
        "violating_pages": bad_pages[:20],
    }


def print_check_report(
    diff: pd.DataFrame,
    ident: pd.DataFrame,
    max_chars: int,
) -> dict:
    print("=" * 64)
    print("DATASET CHECK SUMMARY (before fixes)")
    print("=" * 64)

    len_diff = check_lengths(diff, max_chars, "differing")
    len_ident = check_lengths(ident, max_chars, "identical")
    combined = pd.concat([diff, ident], ignore_index=True)
    dup = check_duplicates_and_leakage(combined)
    page = check_page_split_integrity(combined)

    print("\n1) LENGTH CAP (max_chars={})".format(max_chars))
    print("   Recommendation: DROP pairs exceeding the cap on either side.")
    print("   Why: further splitting requires aligned cut-points in both source")
    print("   and target; naive windowing breaks correspondence and can invent")
    print("   false error spans. Only ~2% of differing pairs are affected.")
    for L in (len_diff, len_ident):
        print(f"\n   [{L['label']}] n={L['n']}")
        print(
            f"     source chars: mean={L['src']['mean']} p50={L['src']['p50']} "
            f"p90={L['src']['p90']} p95={L['src']['p95']} p99={L['src']['p99']} "
            f"max={L['src']['max']}"
        )
        print(
            f"     target chars: mean={L['tgt']['mean']} p50={L['tgt']['p50']} "
            f"p90={L['tgt']['p90']} p95={L['tgt']['p95']} p99={L['tgt']['p99']} "
            f"max={L['tgt']['max']}"
        )
        print(
            f"     exceed {max_chars}: src={L['over_src']} tgt={L['over_tgt']} "
            f"either={L['over_either']} ({L['over_either_pct']}%)"
        )

    print("\n2) DUPLICATE / LEAKAGE")
    print(f"   total rows (diff+ident):     {dup['n_rows']}")
    print(f"   unique (source,target):      {dup['n_unique_pairs']}")
    print(f"   (source,target) keys with >1 row: {dup['n_duplicate_pair_keys']}")
    print(f"   extra duplicate rows:        {dup['n_extra_duplicate_rows']}")
    print(f"   sources in >1 split:         {dup['n_sources_multi_split']}")
    print(f"   source overlap by split pair:{dup['source_overlap_types']}")
    print(f"   targets in >1 split:         {dup['n_targets_multi_split']}")

    print("\n3) SPLIT INTEGRITY (same page_id → one split)")
    print(f"   pages: {page['n_pages']}")
    print(f"   violations: {page['n_violations']}")
    if page["violating_pages"]:
        print(f"   examples: {page['violating_pages']}")

    return {
        "len_diff": len_diff,
        "len_ident": len_ident,
        "dup": dup,
        "page": page,
    }


def drop_overlength(df: pd.DataFrame, max_chars: int) -> tuple[pd.DataFrame, int]:
    src = df["source"].astype(str).map(len)
    tgt = df["target"].astype(str).map(len)
    keep = (src <= max_chars) & (tgt <= max_chars)
    return df.loc[keep].copy(), int((~keep).sum())


def dedupe_exact_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep one row per exact (source, target). Prefer high-conf, then split priority."""
    if df.empty:
        return df.copy(), 0

    work = df.copy()
    work["_pair"] = work["source"].astype(str) + "\t" + work["target"].astype(str)
    work["_split_rank"] = work["split"].map(lambda s: SPLIT_PRIORITY.get(s, 99))
    if "high_confidence_error" in work.columns:
        work["_hc"] = work["high_confidence_error"].astype(bool).astype(int)
    else:
        work["_hc"] = 0
    if "both_annotators_agree" in work.columns:
        work["_agree"] = work["both_annotators_agree"].astype(bool).astype(int)
    else:
        work["_agree"] = 0

    # Sort so best row is first within each pair group:
    # high_conf desc, agree desc, prefer train (low rank), stable index
    work = work.sort_values(
        by=["_pair", "_hc", "_agree", "_split_rank"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    before = len(work)
    work = work.drop_duplicates(subset=["_pair"], keep="first")
    dropped = before - len(work)
    return work.drop(columns=["_pair", "_split_rank", "_hc", "_agree"]), dropped


def remove_cross_split_source_leakage(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """If a source appears in multiple splits, keep only the priority split's rows."""
    if df.empty:
        return df.copy(), 0

    src_to_splits = defaultdict(set)
    for src, split in zip(df["source"].astype(str), df["split"].astype(str)):
        src_to_splits[src].add(split)

    keep_split_for_src = {}
    for src, splits in src_to_splits.items():
        if len(splits) == 1:
            keep_split_for_src[src] = next(iter(splits))
        else:
            keep_split_for_src[src] = sorted(
                splits, key=lambda s: SPLIT_PRIORITY.get(s, 99)
            )[0]

    mask = [
        keep_split_for_src[src] == split
        for src, split in zip(df["source"].astype(str), df["split"].astype(str))
    ]
    kept = df.loc[mask].copy()
    return kept, int(len(df) - len(kept))


def enforce_page_split_integrity(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """If a page somehow has multiple splits, assign all rows to the priority split."""
    if df.empty:
        return df.copy(), 0

    page_splits = df.groupby("page_id")["split"].agg(lambda s: sorted(set(s)))
    multi = {pid: splits for pid, splits in page_splits.items() if len(splits) > 1}
    if not multi:
        return df.copy(), 0

    work = df.copy()
    n_changed = 0
    for pid, splits in multi.items():
        chosen = sorted(splits, key=lambda s: SPLIT_PRIORITY.get(s, 99))[0]
        idx = work["page_id"] == pid
        n_changed += int((work.loc[idx, "split"] != chosen).sum())
        work.loc[idx, "split"] = chosen
    return work, n_changed


def apply_fixes(
    diff: pd.DataFrame,
    ident: pd.DataFrame,
    max_chars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    stats = {}

    d1, n_d_len = drop_overlength(diff, max_chars)
    i1, n_i_len = drop_overlength(ident, max_chars)
    stats["dropped_overlength_differing"] = n_d_len
    stats["dropped_overlength_identical"] = n_i_len

    # Deduplicate within each file first, then across combined for leakage
    d2, n_d_dup = dedupe_exact_pairs(d1)
    i2, n_i_dup = dedupe_exact_pairs(i1)
    stats["dropped_exact_dup_differing"] = n_d_dup
    stats["dropped_exact_dup_identical"] = n_i_dup

    # Mark kind, combine, fix leakage on sources across the whole training set
    d2 = d2.copy()
    i2 = i2.copy()
    d2["_kind"] = "differing"
    i2["_kind"] = "identical"
    combined = pd.concat([d2, i2], ignore_index=True)

    # Exact (src,tgt) may still collide across differing/identical? unlikely.
    # Cross-split source leakage across both.
    combined2, n_leak = remove_cross_split_source_leakage(combined)
    stats["dropped_cross_split_source_leakage"] = n_leak

    combined3, n_page_fix = enforce_page_split_integrity(combined2)
    stats["page_split_rows_reassigned"] = n_page_fix

    # If the same (src,tgt) still exists in multiple splits after leakage pass
    # (different sources... no, same pair), dedupe again globally.
    before = len(combined3)
    combined3 = dedupe_exact_pairs(combined3)[0]
    stats["dropped_exact_dup_global"] = before - len(combined3)

    out_diff = combined3[combined3["_kind"] == "differing"].drop(columns=["_kind"])
    out_ident = combined3[combined3["_kind"] == "identical"].drop(columns=["_kind"])

    # Final verification stats
    final_combined = pd.concat([out_diff, out_ident], ignore_index=True)
    stats["final_differing"] = len(out_diff)
    stats["final_identical"] = len(out_ident)
    stats["final_length_violations"] = int(
        (
            (final_combined["source"].astype(str).map(len) > max_chars)
            | (final_combined["target"].astype(str).map(len) > max_chars)
        ).sum()
    )
    leak_after = check_duplicates_and_leakage(final_combined)
    stats["final_sources_multi_split"] = leak_after["n_sources_multi_split"]
    stats["final_extra_duplicate_rows"] = leak_after["n_extra_duplicate_rows"]
    stats["final_page_violations"] = check_page_split_integrity(final_combined)[
        "n_violations"
    ]
    stats["final_split_differing"] = out_diff["split"].value_counts().to_dict()
    stats["final_split_identical"] = out_ident["split"].value_counts().to_dict()
    return out_diff, out_ident, stats


def print_fix_report(stats: dict):
    print("\n" + "=" * 64)
    print("FIXES APPLIED")
    print("=" * 64)
    print(f"  dropped overlength differing:     {stats['dropped_overlength_differing']}")
    print(f"  dropped overlength identical:     {stats['dropped_overlength_identical']}")
    print(f"  dropped exact-dup differing:      {stats['dropped_exact_dup_differing']}")
    print(f"  dropped exact-dup identical:      {stats['dropped_exact_dup_identical']}")
    print(f"  dropped cross-split src leakage:  {stats['dropped_cross_split_source_leakage']}")
    print(f"  dropped exact-dup global:         {stats['dropped_exact_dup_global']}")
    print(f"  page split rows reassigned:       {stats['page_split_rows_reassigned']}")
    print()
    print(f"  final differing: {stats['final_differing']}  splits={stats['final_split_differing']}")
    print(f"  final identical: {stats['final_identical']}  splits={stats['final_split_identical']}")
    print()
    print("Post-fix verification:")
    print(f"  length violations remaining:      {stats['final_length_violations']}")
    print(f"  sources in >1 split remaining:    {stats['final_sources_multi_split']}")
    print(f"  extra exact-dup rows remaining:   {stats['final_extra_duplicate_rows']}")
    print(f"  page split violations remaining:  {stats['final_page_violations']}")


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
        default="all_sentence_pairs_differing_clean.csv",
    )
    parser.add_argument(
        "--identical",
        type=str,
        default="all_sentence_pairs_identical_clean.csv",
    )
    parser.add_argument("--max_chars", type=int, default=380)
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument(
        "--out_differing",
        type=str,
        default=None,
        help="Output differing filename (default: <stem>_fixed.csv)",
    )
    parser.add_argument(
        "--out_identical",
        type=str,
        default=None,
        help="Output identical filename (default: <stem>_fixed.csv)",
    )
    args = parser.parse_args()

    diff_path = args.exports_dir / args.differing
    ident_path = args.exports_dir / args.identical
    diff = pd.read_csv(diff_path)
    ident = pd.read_csv(ident_path)

    print_check_report(diff, ident, args.max_chars)

    if args.check_only:
        return

    out_diff, out_ident, stats = apply_fixes(diff, ident, args.max_chars)
    print_fix_report(stats)

    def default_fixed_name(name: str) -> str:
        p = Path(name)
        if p.stem.endswith("_fixed"):
            return name
        return f"{p.stem}_fixed{p.suffix}"

    out_diff_path = args.exports_dir / (
        args.out_differing or default_fixed_name(args.differing)
    )
    out_ident_path = args.exports_dir / (
        args.out_identical or default_fixed_name(args.identical)
    )
    out_diff.to_csv(out_diff_path, index=False, encoding="utf-8")
    out_ident.to_csv(out_ident_path, index=False, encoding="utf-8")
    print()
    print(f"Wrote {len(out_diff)} → {out_diff_path}")
    print(f"Wrote {len(out_ident)} → {out_ident_path}")
    print("(originals left unchanged)")


if __name__ == "__main__":
    main()
