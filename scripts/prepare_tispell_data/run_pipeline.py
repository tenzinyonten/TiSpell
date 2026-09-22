#!/usr/bin/env python3
"""Single entry point: OCR annotation CSVs → TiSpell sentence-pair exports.

End-to-end:
  normalize → shad-segment reviewer → antx boundary transfer → page-edge
  filter → similarity gate → page-level split → length cap / dedupe /
  leakage fix → categorize differing pairs.

Writes only:
  <out_dir>/differing_pairs.csv
  <out_dir>/identical_pairs.csv
  <out_dir>/README.md  (refreshed with final counts)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from categorize_diffs import (
    CATEGORIES,
    classify_pair,
    classify_punctuation_only_subtype,
)
from check_and_fix_exports import apply_fixes, print_fix_report
from pair_artifacts import any_side_latin_placeholder, paren_unmatched
from preprocess_clean_pairs import load_all_batches, print_report, process

DIFF_COLUMNS = [
    "page_id",
    "segment_idx",
    "source",
    "target",
    "both_annotators_agree",
    "split",
    "high_confidence_error",
    "batch_name",
    "source_role",
    "has_brackets",
    "diff_category",
    "edit_distance",
]

IDENT_COLUMNS = [
    "page_id",
    "segment_idx",
    "source",
    "target",
    "both_annotators_agree",
    "split",
    "high_confidence_error",
    "batch_name",
    "source_role",
    "has_brackets",
]


def drop_residual_latin(
    diff_df: pd.DataFrame, ident_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Drop any pair that still has Latin on either side (safety net after strip)."""
    stats = {
        "before_diff": len(diff_df),
        "before_ident": len(ident_df),
        "residual_unmatched_parens": 0,
        "dropped_residual_latin_diff": 0,
        "dropped_residual_latin_ident": 0,
        "has_brackets_diff": 0,
        "has_brackets_ident": 0,
    }

    def split_latin(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
        if df.empty:
            return df.copy(), 0, 0
        src = df["source"].astype(str)
        tgt = df["target"].astype(str)
        um = sum(paren_unmatched(a) or paren_unmatched(b) for a, b in zip(src, tgt))
        mask = [any_side_latin_placeholder(a, b) for a, b in zip(src, tgt)]
        kept = df.loc[[not m for m in mask]].copy()
        return kept, um, int(sum(mask))

    d2, um_d, drop_d = split_latin(diff_df)
    i2, um_i, drop_i = split_latin(ident_df)
    stats["residual_unmatched_parens"] = um_d + um_i
    stats["dropped_residual_latin_diff"] = drop_d
    stats["dropped_residual_latin_ident"] = drop_i
    if "has_brackets" in d2.columns and len(d2):
        stats["has_brackets_diff"] = int(d2["has_brackets"].sum())
    if "has_brackets" in i2.columns and len(i2):
        stats["has_brackets_ident"] = int(i2["has_brackets"].sum())
    stats["after_diff"] = len(d2)
    stats["after_ident"] = len(i2)
    return d2, i2, stats


_TSHEG_CHARS = set("་༌")


def _strip_tsheg(text: str) -> str:
    return "".join(c for c in text if c not in _TSHEG_CHARS)


def _agree_true(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return bool(value)


def is_agreed_tsheg_removal(src: str, tgt: str) -> bool:
    """True if only diffs are tsheg(s) present in src and absent in tgt."""
    if src == tgt or _strip_tsheg(src) != _strip_tsheg(tgt):
        return False
    ns = sum(1 for c in src if c in _TSHEG_CHARS)
    nt = sum(1 for c in tgt if c in _TSHEG_CHARS)
    return ns > nt


def drop_agreed_tsheg_removals(diff_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop reviewer tsheg-only deletions where both annotators kept the tsheg.

    Manual review found these are reviewer errors, not corrections. Drop the
    pair entirely; do not rewrite the target.
    """
    if diff_df.empty:
        return diff_df.copy(), 0
    mask = [
        not (
            _agree_true(r.get("both_annotators_agree"))
            and is_agreed_tsheg_removal(str(r["source"]), str(r["target"]))
        )
        for _, r in diff_df.iterrows()
    ]
    kept = diff_df.loc[mask].copy()
    return kept, int(len(diff_df) - len(kept))


def drop_punctuation_only(diff_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop terminal-shad-only and other non-tsheg punctuation_only pairs.

    Keep ``tsheg_within`` punctuation_only pairs (in-text tsheg insert/delete
    is a real spelling/segmentation signal).
    """
    if diff_df.empty or "diff_category" not in diff_df.columns:
        return diff_df.copy(), 0
    keep_rows = []
    n_drop = 0
    for _, r in diff_df.iterrows():
        if str(r["diff_category"]) != "punctuation_only":
            keep_rows.append(True)
            continue
        sub = classify_punctuation_only_subtype(str(r["source"]), str(r["target"]))
        if sub == "tsheg_within":
            keep_rows.append(True)
        else:
            keep_rows.append(False)
            n_drop += 1
    kept = diff_df.loc[keep_rows].copy()
    return kept, n_drop


def categorize(df: pd.DataFrame) -> pd.DataFrame:
    cats, dists = [], []
    for src, tgt in zip(df["source"].astype(str), df["target"].astype(str)):
        cat, dist = classify_pair(src, tgt)
        cats.append(cat)
        dists.append(dist)
    out = df.copy()
    out["diff_category"] = cats
    out["edit_distance"] = dists
    return out


def write_readme(
    path: Path,
    *,
    n_diff: int,
    n_ident: int,
    split_diff: dict,
    split_ident: dict,
    cat_counts: dict,
    defaults: dict,
) -> None:
    total = n_diff + n_ident
    cat_lines = "\n".join(
        f"| {c} | {cat_counts.get(c, 0)} | "
        f"{100 * cat_counts.get(c, 0) / n_diff:.2f}% |"
        for c in CATEGORIES
    )
    text = f"""# OCR annotation → TiSpell sentence pairs

Final exports from the OCR double-annotation batches
(`annotator_a` / `annotator_b` → `final_reviewer` as gold).

## Output files

| File | Rows | Contents |
|------|------|----------|
| `differing_pairs.csv` | {n_diff} | Source ≠ target after normalization. Includes edit category. |
| `identical_pairs.csv` | {n_ident} | Source == target after normalization (positive / no-op examples). |
| `README.md` | — | This file. |

**Total pairs: {total}**
(differing split: `{split_diff}`; identical split: `{split_ident}`)

## Columns

Shared columns:

| Column | Meaning |
|--------|---------|
| `page_id` | `{{batch_name}}/{{image_id}}` — page-level unit for splits |
| `segment_idx` | Index of the reviewer shad-segment on that page (before page-edge drop of first) |
| `source` | Annotator text |
| `target` | Final-reviewer gold text |
| `both_annotators_agree` | True if annotator_a and annotator_b produced the same source for this segment |
| `high_confidence_error` | True if both agree **and** source ≠ target |
| `split` | `train` / `val` / `test` (assigned by `page_id`, not by shuffling sentences) |
| `batch_name` | Source batch |
| `source_role` | `annotator_a` or `annotator_b` |
| `has_brackets` | True if either page transcript had bracket chars before they were stripped (text kept) |

Differing-only:

| Column | Meaning |
|--------|---------|
| `diff_category` | Edit type (first-match order; see below) |
| `edit_distance` | Character Levenshtein distance |

### `diff_category` values (first match wins)

`punctuation_only` → `whitespace_only` → `single_char_substitution` →
`single_char_insert_delete` → `syllable_level` → `multi_edit` → `large_rewrite`

| category | count | pct of differing |
|----------|------:|-----------------:|
{cat_lines}

## Pipeline filters and thresholds

Applied in order:

1. **Pre-segmentation artifact cleaning** (full page transcript)
   - Strip bracket characters `()（）༼༽` but **keep** the enclosed text
   - Done before shad splits so a cut cannot orphan a bracket half
   - `has_brackets=True` marks pages that had brackets before stripping

1b. **Pair-level artifact filters** (after alignment)
   - Strip page ornaments U+0FD0–U+0FDA plus section/head marks
     ༁–༊, ༒, ༓ (exclude ༔; keep the pair)
   - Strip leading ༴/༔ (+tsheg/ws) when openings then match; leave mid alone
   - Drop pairs with leading ASCII digits; leave mid-string digits alone
   - Drop pairs with Latin on **either** side — ASCII `[A-Za-z]` and
     fullwidth U+FF21–FF5A (`ＩＳＯ` …)
   - Drop pairs with non-Tibetan punctuation/symbols (`—` `„` `.` `"` …);
     these sit where a Tibetan character should be

2. **Unicode / tsheg / space normalization** (`tibetan_normalize.normalize_transcript`)
   - NFC
   - Convention maps: ༸→༧ (U+0F38→U+0F27), ྅→༣ (U+0F85→U+0F23)
   - Collapse whitespace to single spaces
   - Remove spaces before shad; remove spaces after tsheg
   - Treat tsheg ↔ space between Tibetan letters as equivalent (canonicalize to tsheg)
   - Collapse repeated tsheg; drop tsheg immediately before shad

3. **Segmentation** — split gold (`final_reviewer`) on shad only (`།༎༏༐༑`);
   do **not** split on newlines.

4. **Boundary transfer** — join gold segments with `\\n`, transfer those cuts onto
   the annotator page via `fast_antx.core.transfer` (Python `diff_match_patch`
   backend; native dmp binary is x86_64-only). Strip leading boundary debris
   from transferred segments (any leading combining vowel/subjoined/mark,
   with or without following shad/tsheg; also bare shad/tsheg/whitespace).
   Also strip a short leading syllable+shad from one side when removing it
   makes that side's opening match the other (stranded previous sentence).
   Strip leading ༴/༔ (+ following tsheg/whitespace) under the same openings-
   match rule; mid-string occurrences are left alone. Drop pairs with leading
   ASCII digits (folio/line numbers); mid-string digits are left alone.

5. **Page-edge filter** (per aligned pair)
   - Completeness from **target (reviewer)** only: require gold ends with shad
   - Missing shad on the annotator side is kept (learnable error)
   - Keep first/last when target is shad-complete
   - Require both sides length ≥ **{defaults['min_chars']}** chars

6. **Similarity gate** — drop pairs with SequenceMatcher ratio
   `< {defaults['min_ratio']}` (same normalization as the old aligner:
   NFC, strip whitespace for the ratio only).

7. **Page-level split** — shuffle `page_id`s with seed **{defaults['seed']}**;
   assign **{defaults['train_pct']:.0%} / {defaults['val_pct']:.0%} / {defaults['test_pct']:.0%}**
   train/val/test. All segments from a page share one split.

8. **Length cap** — drop any pair where source or target exceeds
   **{defaults['max_chars']}** chars (~300 tokens at ~1.27 chars/token;
   backbone context 512).

9. **Exact dedupe** — keep one row per exact `(source, target)`;
   prefer high-confidence / annotator-agree / train.

10. **Cross-split source leakage** — if the same `source` string appears in
   more than one split, keep only the priority split (train > val > test).

11. **Agreed tsheg-removal drop** — drop differing pairs where the only edit is
    one or more tshegs present in the annotator source and absent in the
    reviewer target, **and** `both_annotators_agree=True`. Manual review
    found these are reviewer errors; pairs are dropped (target not rewritten).

12. **Punctuation-only drop** — after categorization, drop `punctuation_only`
    pairs that are terminal-shad-only or other non-tsheg punct. Keep
    in-text tsheg insert/delete pairs (`tsheg_within`) as spelling signal.

## How to re-run

From the `TiSpell` directory, with the project venv active:

```bash
python scripts/prepare_tispell_data/run_pipeline.py \\
  --input_dir dataset/ocr_annotation/raw \\
  --out_dir dataset/ocr_annotation/exports
```

Optional flags:

```bash
python scripts/prepare_tispell_data/run_pipeline.py --help
```

Defaults: `--min_chars {defaults['min_chars']}`, `--min_ratio {defaults['min_ratio']}`,
`--max_chars {defaults['max_chars']}`, `--seed {defaults['seed']}`.

Input: all `*.csv` under `--input_dir` (uchen / ume batches). Pair definition is
always annotator → `final_reviewer`.

This command **overwrites** `differing_pairs.csv` and `identical_pairs.csv` in
`--out_dir` and refreshes this README. It does not write intermediate CSVs.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("dataset/ocr_annotation/raw"),
    )
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("dataset/ocr_annotation/exports"),
    )
    parser.add_argument("--min_chars", type=int, default=8)
    parser.add_argument("--min_ratio", type=float, default=0.55)
    parser.add_argument("--max_chars", type=int, default=380)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split_ratios",
        type=float,
        nargs=3,
        default=[0.9, 0.05, 0.05],
        metavar=("TRAIN", "VAL", "TEST"),
    )
    args = parser.parse_args()

    defaults = {
        "min_chars": args.min_chars,
        "min_ratio": args.min_ratio,
        "max_chars": args.max_chars,
        "seed": args.seed,
        "train_pct": args.split_ratios[0],
        "val_pct": args.split_ratios[1],
        "test_pct": args.split_ratios[2],
    }

    df = load_all_batches(args.input_dir, args.files)
    print(f"Loaded {len(df)} page rows from {args.input_dir}")

    differing, identical, report, _ = process(
        df,
        min_chars=args.min_chars,
        seed=args.seed,
        min_ratio=args.min_ratio,
        ratios=tuple(args.split_ratios),
    )
    print_report(report, differing, identical, seed=args.seed)

    diff_df = pd.DataFrame(differing)
    ident_df = pd.DataFrame(identical)
    diff_df, ident_df, art_stats = drop_residual_latin(diff_df, ident_df)
    print("\n" + "=" * 64)
    print("ARTIFACT CLEANING (pre-segmentation; residual Latin drop)")
    print("=" * 64)
    atot = report.get("artifact_totals") or {}
    um_before = atot.get("gold_segs_unmatched_before_strip", 0)
    um_after = atot.get("gold_segs_unmatched_after_strip", 0)
    print(
        f"  pairs leading-debris stripped: {atot.get('pairs_leading_debris_stripped', 0)}"
    )
    print(
        f"  leading-debris → identical:    "
        f"{atot.get('leading_debris_became_identical', 0)}"
    )
    print(
        f"  pairs leading-syllable stripped: "
        f"{atot.get('pairs_leading_syllable_stripped', 0)}"
    )
    print(
        f"  leading-syllable → identical:  "
        f"{atot.get('leading_syllable_became_identical', 0)}"
    )
    print(
        f"  pairs leading ༴/༔ stripped:    "
        f"{atot.get('pairs_leading_content_punct_stripped', 0)}"
    )
    print(
        f"  leading ༴/༔ → identical:       "
        f"{atot.get('leading_content_punct_became_identical', 0)}"
    )
    print(
        f"  pairs ornament-stripped:       {atot.get('pairs_ornament_stripped', 0)}"
    )
    print(
        f"  ornament-strip → identical:    "
        f"{atot.get('ornament_strip_became_identical', 0)}"
    )
    print(
        f"  dropped leading ASCII digits:  "
        f"{atot.get('dropped_leading_ascii_digit_pairs', 0)}"
    )
    print(
        f"  dropped one-side Latin pairs:  {atot.get('dropped_one_side_latin_pairs', 0)}"
    )
    print(
        f"  dropped both-side Latin pairs: {atot.get('dropped_both_side_latin_pairs', 0)}"
    )
    print(
        f"  dropped any-Latin pairs total: {atot.get('dropped_any_latin_pairs', 0)}"
    )
    print(
        f"  dropped non-Tibetan punct:     "
        f"{atot.get('dropped_non_tibetan_punct_pairs', 0)}"
    )
    print(f"  gold segs unmatched before strip: {um_before}")
    print(f"  gold segs unmatched after strip:  {um_after}")
    print(f"  gold segs no longer unmatched:    {um_before - um_after}")
    print(
        f"  residual unmatched parens in pairs: {art_stats['residual_unmatched_parens']}"
    )
    print(
        f"  dropped residual Latin differing:   {art_stats['dropped_residual_latin_diff']}"
    )
    print(
        f"  dropped residual Latin identical:   {art_stats['dropped_residual_latin_ident']}"
    )
    print(
        f"  has_brackets=True: differing={art_stats['has_brackets_diff']}  "
        f"identical={art_stats['has_brackets_ident']}"
    )

    out_diff, out_ident, fix_stats = apply_fixes(diff_df, ident_df, args.max_chars)
    print_fix_report(fix_stats)

    out_diff, n_tsheg_drop = drop_agreed_tsheg_removals(out_diff)
    print(
        f"\n  dropped agreed tsheg-only removals: {n_tsheg_drop}"
        f"  (differing now {len(out_diff)})"
    )
    # Refresh final counts after the post-fix drop.
    fix_stats["final_differing"] = len(out_diff)
    fix_stats["final_split_differing"] = out_diff["split"].value_counts().to_dict()

    out_diff = categorize(out_diff)
    out_diff, n_punct_drop = drop_punctuation_only(out_diff)
    print(
        f"\n  dropped terminal-shad/other punct_only: {n_punct_drop}"
        f"  (kept tsheg-within punct_only; differing now {len(out_diff)})"
    )
    fix_stats["final_differing"] = len(out_diff)
    fix_stats["final_split_differing"] = out_diff["split"].value_counts().to_dict()

    cat_counts = Counter(out_diff["diff_category"].tolist())
    print("\nCategory counts (final differing):")
    for cat in CATEGORIES:
        c = cat_counts.get(cat, 0)
        print(f"  {cat:<28} {c:7d}  ({100 * c / max(len(out_diff), 1):5.2f}%)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    diff_path = args.out_dir / "differing_pairs.csv"
    ident_path = args.out_dir / "identical_pairs.csv"
    out_diff.reindex(columns=DIFF_COLUMNS).to_csv(
        diff_path, index=False, encoding="utf-8"
    )
    out_ident.reindex(columns=IDENT_COLUMNS).to_csv(
        ident_path, index=False, encoding="utf-8"
    )

    write_readme(
        args.out_dir / "README.md",
        n_diff=len(out_diff),
        n_ident=len(out_ident),
        split_diff=out_diff["split"].value_counts().to_dict(),
        split_ident=out_ident["split"].value_counts().to_dict(),
        cat_counts=dict(cat_counts),
        defaults=defaults,
    )

    print()
    print(f"Wrote {len(out_diff)} → {diff_path}")
    print(f"Wrote {len(out_ident)} → {ident_path}")
    print(f"Wrote README → {args.out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
