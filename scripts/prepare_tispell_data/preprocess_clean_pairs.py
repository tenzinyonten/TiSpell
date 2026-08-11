#!/usr/bin/env python3
"""Preprocess page-level OCR transcripts into clean sentence pairs for TiSpell.

Pipeline:
  1) Unicode/whitespace/tsheg normalization
  2) Shad sentence segmentation on final_reviewer (gold)
  3) antx-style boundary transfer onto annotator (Python DMP)
  4) Strip leading boundary debris from transferred segments
     (vowel/subjoined + shad, bare shad/tsheg/whitespace)
  5) Page-edge filtering on aligned pairs
  6) Annotator-agreement flags
  7) Page-level 80/10/10 train/val/test split

Writes differing and identical clean CSVs separately.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_sentences import similarity
from alignment_transfer import (
    align_annotator_to_reviewer,
    has_leading_ascii_digits,
    strip_leading_boundary_artifacts,
    strip_leading_content_punct_by_compare,
    strip_leading_syllable_by_compare,
)
from pair_artifacts import (
    any_side_non_tibetan_punct,
    both_side_latin_placeholder,
    clean_page_artifacts,
    one_side_latin_placeholder,
    paren_unmatched,
    strip_ornament_chars,
)
from process_batches import normalize_batch_df, norm_text as raw_norm_text
from tibetan_normalize import SHAD_END_RE, normalize_transcript, segment_sentences

OUTPUT_COLUMNS = [
    "page_id",
    "segment_idx",
    "source",
    "target",
    "both_annotators_agree",
    "high_confidence_error",
    "split",
    "batch_name",
    "source_role",
    "has_brackets",
]


def load_all_batches(input_dir: Path, files: list[str] | None) -> pd.DataFrame:
    paths = (
        [input_dir / f for f in files]
        if files
        else sorted(input_dir.glob("*.csv"))
    )
    frames = []
    for path in paths:
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        raw = pd.read_csv(path)
        frames.append(normalize_batch_df(raw, fallback_batch_name=path.stem))
    if not frames:
        raise SystemExit(f"No CSV files found under {input_dir}")
    return pd.concat(frames, ignore_index=True)


def assign_splits(page_ids: list[str], seed: int, ratios=(0.8, 0.1, 0.1)) -> dict[str, str]:
    ids = sorted(set(page_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    split_map = {}
    for i, pid in enumerate(ids):
        if i < n_train:
            split_map[pid] = "train"
        elif i < n_train + n_val:
            split_map[pid] = "val"
        else:
            split_map[pid] = "test"
    if n >= 3:
        counts = defaultdict(int)
        for s in split_map.values():
            counts[s] += 1
        train_ids = [p for p, s in split_map.items() if s == "train"]
        for need in ("val", "test"):
            if counts[need] == 0 and train_ids:
                pid = train_ids.pop()
                split_map[pid] = need
                counts[need] += 1
                counts["train"] -= 1
    return split_map


def filter_aligned_pairs(
    pairs: list[tuple[str, str]],
    min_chars: int,
    min_ratio: float = 0.55,
) -> tuple[list[dict], dict]:
    """Page-edge + post-transfer similarity filters on aligned pairs.

    Completeness is judged from the **target (reviewer)** only: if gold ends
    in shad, the sentence is complete. A missing shad on the annotator side
    is a real error to learn, not a drop reason. First/last segments are kept
    when the target is shad-complete.
    """
    counts = {
        "before": len(pairs),
        "dropped_no_shad": 0,
        "dropped_short": 0,
        "dropped_low_sim": 0,
        "recovered_src_missing_shad": 0,  # tgt has shad, src does not
        "recovered_first": 0,
        "kept_last": 0,
        "after": 0,
    }
    kept: list[dict] = []
    n = len(pairs)
    for i, (src, tgt) in enumerate(pairs):
        if not SHAD_END_RE.search(tgt):
            counts["dropped_no_shad"] += 1
            continue
        if len(tgt) < min_chars or len(src) < min_chars:
            counts["dropped_short"] += 1
            continue
        if similarity(src, tgt) < min_ratio:
            counts["dropped_low_sim"] += 1
            continue
        if not SHAD_END_RE.search(src):
            counts["recovered_src_missing_shad"] += 1
        if i == 0:
            counts["recovered_first"] += 1
        if i == n - 1:
            counts["kept_last"] += 1
        kept.append({"source": src, "target": tgt, "target_idx": i})
    counts["after"] = len(kept)
    return kept, counts


def process(
    df: pd.DataFrame,
    min_chars: int,
    seed: int,
    min_ratio: float = 0.55,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
):
    filter_totals = defaultdict(int)
    align_totals = defaultdict(int)
    artifact_totals = defaultdict(int)
    recovered_examples: list[dict] = []
    encoding_noise_pairs = 0
    raw_differing_page_slots = 0
    pages_used = 0
    page_records = []

    for _, row in df.iterrows():
        batch = str(row.get("batch_name", ""))
        image_id = str(row.get("image_id", ""))
        page_id = f"{batch}/{image_id}"

        raw_gold = raw_norm_text(row["final_reviewer_transcript"])
        if raw_gold is None:
            continue

        # Counterfactual: segment without artifact strip to measure how many
        # gold segments inherit unmatched brackets from a cut-through span.
        gold_unclean = normalize_transcript(raw_gold)
        if gold_unclean is not None:
            for seg in segment_sentences(gold_unclean):
                artifact_totals["gold_segs_before_strip"] += 1
                if paren_unmatched(seg):
                    artifact_totals["gold_segs_unmatched_before_strip"] += 1

        # Strip brackets (keep text) + Latin placeholders BEFORE segmentation
        # so a shad inside (…) cannot split the span into unmatched halves.
        gold_clean, gold_art = clean_page_artifacts(raw_gold)
        for k, v in gold_art.items():
            if isinstance(v, bool):
                artifact_totals[f"gold_{k}"] += int(v)
            else:
                artifact_totals[f"gold_{k}"] += v

        gold = normalize_transcript(gold_clean)
        if gold is None:
            continue

        pages_used += 1
        gold_segs = segment_sentences(gold)
        filter_totals["gold_before"] += len(gold_segs)
        for seg in gold_segs:
            artifact_totals["gold_segs_after_strip"] += 1
            if paren_unmatched(seg):
                artifact_totals["gold_segs_unmatched_after_strip"] += 1

        role_alignments = {}
        role_had_brackets = {}
        for role, tcol in (
            ("annotator_a", "annotator_a_transcript"),
            ("annotator_b", "annotator_b_transcript"),
        ):
            raw_src = raw_norm_text(row[tcol])
            if raw_src is None:
                role_alignments[role] = []
                role_had_brackets[role] = False
                continue

            src_clean, src_art = clean_page_artifacts(raw_src)
            for k, v in src_art.items():
                if isinstance(v, bool):
                    artifact_totals[f"src_{k}"] += int(v)
                else:
                    artifact_totals[f"src_{k}"] += v
            role_had_brackets[role] = bool(src_art["had_brackets"] or gold_art["had_brackets"])

            src_norm = normalize_transcript(src_clean)
            if src_norm is None:
                role_alignments[role] = []
                continue
            if raw_src != raw_gold:
                raw_differing_page_slots += 1
                if src_norm == gold:
                    encoding_noise_pairs += 1

            pairs, stats = align_annotator_to_reviewer(src_norm, gold)
            align_totals["transfer_slots"] += 1
            align_totals["reviewer_segs"] += stats["reviewer_segs"]
            align_totals["annotator_segs"] += stats["annotator_segs"]
            if stats["count_agree"]:
                align_totals["count_agree"] += 1
            else:
                align_totals["count_disagree"] += 1
            align_totals["paired_before_filter"] += stats["n_pairs"]

            # Residual unmatched after pre-seg bracket strip (should be ~0).
            for src, tgt in pairs:
                if paren_unmatched(src) or paren_unmatched(tgt):
                    artifact_totals["pairs_still_unmatched_parens"] += 1

            kept, counts = filter_aligned_pairs(
                pairs, min_chars=min_chars, min_ratio=min_ratio
            )
            for k, v in counts.items():
                filter_totals[f"pair_{k}"] += v

            # Pair-level cleanup: ornaments → combining-mark debris →
            # cross-side leading syllable+shad → leading ༴/༔ when openings match.
            orn_clean = []
            for a in kept:
                src, tgt = a["source"], a["target"]
                src_o, tgt_o = strip_ornament_chars(src), strip_ornament_chars(tgt)
                if src_o != src or tgt_o != tgt:
                    artifact_totals["pairs_ornament_stripped"] += 1
                    if src != tgt and src_o == tgt_o:
                        artifact_totals["ornament_strip_became_identical"] += 1
                src_d = strip_leading_boundary_artifacts(src_o)
                tgt_d = strip_leading_boundary_artifacts(tgt_o)
                if src_d != src_o or tgt_d != tgt_o:
                    artifact_totals["pairs_leading_debris_stripped"] += 1
                    if src_o != tgt_o and src_d == tgt_d:
                        artifact_totals["leading_debris_became_identical"] += 1
                src_s, tgt_s, did_syll = strip_leading_syllable_by_compare(src_d, tgt_d)
                if did_syll:
                    artifact_totals["pairs_leading_syllable_stripped"] += 1
                    if src_d != tgt_d and src_s == tgt_s:
                        artifact_totals["leading_syllable_became_identical"] += 1
                src_p, tgt_p, did_punct = strip_leading_content_punct_by_compare(
                    src_s, tgt_s
                )
                if did_punct:
                    artifact_totals["pairs_leading_content_punct_stripped"] += 1
                    if src_s != tgt_s and src_p == tgt_p:
                        artifact_totals["leading_content_punct_became_identical"] += 1
                src2 = normalize_transcript(src_p) or ""
                tgt2 = normalize_transcript(tgt_p) or ""
                if not src2 or not tgt2:
                    artifact_totals["dropped_empty_after_ornament_strip"] += 1
                    continue
                orn_clean.append({**a, "source": src2, "target": tgt2})
            kept = orn_clean

            # Drop pairs with Latin on either side (ASCII + fullwidth), pair-level.
            latin_clean = []
            for a in kept:
                src, tgt = a["source"], a["target"]
                if both_side_latin_placeholder(src, tgt):
                    artifact_totals["dropped_both_side_latin_pairs"] += 1
                    artifact_totals["dropped_any_latin_pairs"] += 1
                    continue
                if one_side_latin_placeholder(src, tgt):
                    artifact_totals["dropped_one_side_latin_pairs"] += 1
                    artifact_totals["dropped_any_latin_pairs"] += 1
                    continue
                latin_clean.append(a)
            kept = latin_clean

            # Drop pairs with non-Tibetan punctuation (— „ . " …) on either side.
            punct_clean = []
            for a in kept:
                src, tgt = a["source"], a["target"]
                if any_side_non_tibetan_punct(src, tgt):
                    artifact_totals["dropped_non_tibetan_punct_pairs"] += 1
                    continue
                punct_clean.append(a)
            kept = punct_clean

            # Drop pairs with leading ASCII digits (folio/line numbers); leave mid alone.
            digit_clean = []
            for a in kept:
                src, tgt = a["source"], a["target"]
                if has_leading_ascii_digits(src) or has_leading_ascii_digits(tgt):
                    artifact_totals["dropped_leading_ascii_digit_pairs"] += 1
                    continue
                digit_clean.append(a)
            kept = digit_clean

            if len(recovered_examples) < 40:
                for a in kept:
                    if a["target_idx"] == 0:
                        recovered_examples.append(
                            {
                                "page_id": page_id,
                                "source": a["source"],
                                "target": a["target"],
                                "role": role,
                            }
                        )
                        if len(recovered_examples) >= 40:
                            break
            role_alignments[role] = kept

        by_target = {"annotator_a": {}, "annotator_b": {}}
        for role, aligned in role_alignments.items():
            for a in aligned:
                by_target[role][a["target_idx"]] = a["source"]

        page_pairs = []
        for role, aligned in role_alignments.items():
            for a in aligned:
                t_idx = a["target_idx"]
                src_a = by_target["annotator_a"].get(t_idx)
                src_b = by_target["annotator_b"].get(t_idx)
                both_agree = (
                    src_a is not None and src_b is not None and src_a == src_b
                )
                identical = a["source"] == a["target"]
                high_conf = both_agree and not identical
                page_pairs.append(
                    {
                        "page_id": page_id,
                        "segment_idx": t_idx,
                        "source": a["source"],
                        "target": a["target"],
                        "both_annotators_agree": both_agree,
                        "high_confidence_error": high_conf,
                        "batch_name": batch,
                        "source_role": role,
                        "identical": identical,
                        "has_brackets": role_had_brackets.get(role, False),
                    }
                )

        if page_pairs:
            page_records.append({"page_id": page_id, "pairs": page_pairs})

    split_map = assign_splits(
        [p["page_id"] for p in page_records], seed=seed, ratios=ratios
    )

    differing, identical = [], []
    for page in page_records:
        split = split_map[page["page_id"]]
        for pair in page["pairs"]:
            row_out = {
                "page_id": pair["page_id"],
                "segment_idx": pair["segment_idx"],
                "source": pair["source"],
                "target": pair["target"],
                "both_annotators_agree": pair["both_annotators_agree"],
                "high_confidence_error": pair["high_confidence_error"],
                "split": split,
                "batch_name": pair["batch_name"],
                "source_role": pair["source_role"],
                "has_brackets": pair["has_brackets"],
            }
            if pair["identical"]:
                identical.append(row_out)
            else:
                differing.append(row_out)

    report = {
        "pages_used": pages_used,
        "pages_with_pairs": len(page_records),
        "filter_totals": dict(filter_totals),
        "align_totals": dict(align_totals),
        "artifact_totals": dict(artifact_totals),
        "recovered_examples": recovered_examples,
        "encoding_noise_page_pairs": encoding_noise_pairs,
        "raw_differing_page_slots_checked": raw_differing_page_slots,
        "n_differing": len(differing),
        "n_identical": len(identical),
        "high_confidence_errors": sum(
            1 for r in differing if r["high_confidence_error"]
        ),
        "both_agree_differing": sum(
            1 for r in differing if r["both_annotators_agree"]
        ),
        "has_brackets_differing": sum(1 for r in differing if r["has_brackets"]),
        "has_brackets_identical": sum(1 for r in identical if r["has_brackets"]),
    }
    return differing, identical, report, split_map


def print_report(report: dict, differing: list[dict], identical: list[dict], seed: int):
    ft = report["filter_totals"]
    at = report["align_totals"]
    print("=" * 60)
    print("PREPROCESS REPORT (antx boundary transfer)")
    print("=" * 60)
    print(f"Total pages used:              {report['pages_used']}")
    print(f"Pages with ≥1 kept pair:       {report['pages_with_pairs']}")
    print()
    slots = at.get("transfer_slots", 0)
    agree = at.get("count_agree", 0)
    disagree = at.get("count_disagree", 0)
    print("Boundary-transfer segment count agreement:")
    print(f"  annotator↔reviewer slots:    {slots}")
    print(f"  count agree:                 {agree}"
          f" ({100 * agree / slots:.1f}%)" if slots else "")
    print(f"  count disagree:              {disagree}"
          f" ({100 * disagree / slots:.1f}%)" if slots else "")
    print(f"  reviewer segments (sum):     {at.get('reviewer_segs', 0)}")
    print(f"  annotator segments (sum):    {at.get('annotator_segs', 0)}")
    print(f"  paired before page filter:   {at.get('paired_before_filter', 0)}")
    print()
    print("Page-edge filtering (aligned pairs):")
    print(f"  gold segments (raw count):   {ft.get('gold_before', 0)}")
    print(f"  pairs before filter:         {ft.get('pair_before', 0)}")
    print(f"  dropped no-shad (target):    {ft.get('pair_dropped_no_shad', 0)}")
    print(f"  dropped short (<min):        {ft.get('pair_dropped_short', 0)}")
    print(f"  dropped low similarity:      {ft.get('pair_dropped_low_sim', 0)}")
    print(
        f"  recovered src-missing-shad:  {ft.get('pair_recovered_src_missing_shad', 0)}"
        "  (tgt ends shad, src does not)"
    )
    print(f"  recovered first (idx=0):     {ft.get('pair_recovered_first', 0)}")
    print(f"  kept last (tgt shad-complete): {ft.get('pair_kept_last', 0)}")
    print(f"  pairs after filter:          {ft.get('pair_after', 0)}")
    exs = report.get("recovered_examples") or []
    if exs:
        print()
        print("10 recovered first-segment examples (old pipeline dropped these):")
        for i, ex in enumerate(exs[:10], 1):
            print(f"\n  [{i}] page={ex['page_id']} role={ex['role']}")
            print(f"      SRC: {ex['source']}")
            print(f"      TGT: {ex['target']}")
    print()
    atot = report.get("artifact_totals") or {}
    um_before = atot.get("gold_segs_unmatched_before_strip", 0)
    um_after = atot.get("gold_segs_unmatched_after_strip", 0)
    print("Pre-segmentation artifact cleaning (brackets + Latin):")
    print(f"  gold pages with brackets:       {atot.get('gold_had_brackets', 0)}")
    print(f"  gold pages unmatched parens:    {atot.get('gold_had_unmatched_parens', 0)}")
    print(f"  gold pages with Latin runs:     {atot.get('gold_had_latin', 0)}")
    print(f"  src pages with brackets:        {atot.get('src_had_brackets', 0)}")
    print(f"  src pages unmatched parens:     {atot.get('src_had_unmatched_parens', 0)}")
    print(f"  src pages with Latin runs:      {atot.get('src_had_latin', 0)}")
    print(f"  bracket chars stripped (gold):  {atot.get('gold_brackets_stripped', 0)}")
    print(f"  bracket chars stripped (src):   {atot.get('src_brackets_stripped', 0)}")
    print(f"  gold ornaments stripped:          {atot.get('gold_ornaments_stripped', 0)}")
    print(f"  src ornaments stripped:           {atot.get('src_ornaments_stripped', 0)}")
    print(
        f"  pairs leading-debris stripped:    "
        f"{atot.get('pairs_leading_debris_stripped', 0)}"
    )
    print(
        f"  leading-debris → identical:       "
        f"{atot.get('leading_debris_became_identical', 0)}"
    )
    print(
        f"  pairs leading-syllable stripped:  "
        f"{atot.get('pairs_leading_syllable_stripped', 0)}"
    )
    print(
        f"  leading-syllable → identical:     "
        f"{atot.get('leading_syllable_became_identical', 0)}"
    )
    print(
        f"  pairs leading ༴/༔ stripped:       "
        f"{atot.get('pairs_leading_content_punct_stripped', 0)}"
    )
    print(
        f"  leading ༴/༔ → identical:          "
        f"{atot.get('leading_content_punct_became_identical', 0)}"
    )
    print(f"  pairs ornament-stripped:          {atot.get('pairs_ornament_stripped', 0)}")
    print(
        f"  ornament-strip → identical:       "
        f"{atot.get('ornament_strip_became_identical', 0)}"
    )
    print(
        f"  dropped one-side Latin pairs:   "
        f"{atot.get('dropped_one_side_latin_pairs', 0)}"
    )
    print(
        f"  dropped both-side Latin pairs:  "
        f"{atot.get('dropped_both_side_latin_pairs', 0)}"
    )
    print(
        f"  dropped any-Latin pairs total:  "
        f"{atot.get('dropped_any_latin_pairs', 0)}"
    )
    print(
        f"  dropped non-Tibetan punct pairs: "
        f"{atot.get('dropped_non_tibetan_punct_pairs', 0)}"
    )
    print(
        f"  dropped leading ASCII digit pairs: "
        f"{atot.get('dropped_leading_ascii_digit_pairs', 0)}"
    )
    print(f"  gold segs unmatched BEFORE strip: {um_before}")
    print(f"  gold segs unmatched AFTER strip:  {um_after}")
    print(f"  gold segs no longer unmatched:    {um_before - um_after}")
    print(
        f"  aligned pairs still unmatched:    "
        f"{atot.get('pairs_still_unmatched_parens', 0)}"
    )
    print(
        f"  has_brackets flag: differing={report.get('has_brackets_differing', 0)}  "
        f"identical={report.get('has_brackets_identical', 0)}"
    )
    print()
    print(
        "Encoding-noise page pairs "
        "(raw annotator≠final, but identical after normalization): "
        f"{report['encoding_noise_page_pairs']} / "
        f"{report['raw_differing_page_slots_checked']} checked slots"
    )
    print()
    print(f"Final differing pairs:  {report['n_differing']}")
    print(f"Final identical pairs:  {report['n_identical']}")
    print(
        f"  of differing, both annotators agree: {report['both_agree_differing']} "
        f"(high-confidence errors: {report['high_confidence_errors']})"
    )

    def split_counts(rows):
        c = defaultdict(int)
        for r in rows:
            c[r["split"]] += 1
        return dict(c)

    print()
    print("Per-split counts:")
    print(f"  differing: {split_counts(differing)}")
    print(f"  identical: {split_counts(identical)}")

    print()
    print("-" * 60)
    print("20 randomly sampled DIFFERING pairs (eye-check):")
    print("-" * 60)
    rng = random.Random(seed)
    sample = differing if len(differing) <= 20 else rng.sample(differing, 20)
    for i, r in enumerate(sample, 1):
        flag = []
        if r["both_annotators_agree"]:
            flag.append("both_agree")
        if r["high_confidence_error"]:
            flag.append("high_conf_error")
        flags = ",".join(flag) if flag else "-"
        print(f"\n[{i}] page={r['page_id']} seg={r['segment_idx']} "
              f"split={r['split']} flags={flags}")
        print(f"  source: {r['source']}")
        print(f"  target: {r['target']}")


def main():
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
    parser.add_argument(
        "--suffix",
        type=str,
        default="antx",
        help="Filename suffix before .csv (writes alongside existing exports)",
    )
    parser.add_argument("--min_chars", type=int, default=15)
    parser.add_argument(
        "--min_ratio",
        type=float,
        default=0.55,
        help="Post-transfer SequenceMatcher ratio gate (old align_sentences default)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_all_batches(args.input_dir, args.files)
    print(f"Loaded {len(df)} page rows from {args.input_dir}")

    differing, identical, report, _ = process(
        df,
        min_chars=args.min_chars,
        seed=args.seed,
        min_ratio=args.min_ratio,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.suffix}" if args.suffix else ""
    diff_path = args.out_dir / f"all_sentence_pairs_differing_clean{tag}.csv"
    ident_path = args.out_dir / f"all_sentence_pairs_identical_clean{tag}.csv"

    cols = [
        "page_id",
        "segment_idx",
        "source",
        "target",
        "both_annotators_agree",
        "split",
        "high_confidence_error",
        "batch_name",
        "source_role",
    ]
    pd.DataFrame(differing).reindex(columns=cols).to_csv(
        diff_path, index=False, encoding="utf-8"
    )
    pd.DataFrame(identical).reindex(columns=cols).to_csv(
        ident_path, index=False, encoding="utf-8"
    )

    print_report(report, differing, identical, seed=args.seed)
    print()
    print(f"Wrote {len(differing)} → {diff_path}")
    print(f"Wrote {len(identical)} → {ident_path}")


if __name__ == "__main__":
    main()
