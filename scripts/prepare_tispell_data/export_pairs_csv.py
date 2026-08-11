#!/usr/bin/env python3
"""Export source/target transcript pairs to CSV.

Keeps identical and differing sentence pairs in separate files so the
training mix can be decided later.

Writes:
  - sentence pairs (differing)
  - sentence pairs (identical)
  - page pairs (differing full transcripts)
  - page pairs (identical full transcripts)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_sentences import align_sentences, segment_sentences


def norm(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def export_sentence_csvs(raw_csv: Path, out_dir: Path) -> tuple[int, int]:
    raw = pd.read_csv(raw_csv)
    ident_rows = []
    diff_rows = []

    for _, row in raw.iterrows():
        gold = norm(row["final_reviewer_transcript"])
        for role, tcol, ucol in (
            ("annotator_a", "annotator_a_transcript", "annotator_a_username"),
            ("annotator_b", "annotator_b_transcript", "annotator_b_username"),
        ):
            src = norm(row[tcol])
            if src is None or gold is None:
                continue
            for a in align_sentences(segment_sentences(src), segment_sentences(gold)):
                rec = {
                    "task_id": row["task_id"],
                    "image_id": row["image_id"],
                    "source_role": role,
                    "annotator_username": row[ucol],
                    "final_reviewer_username": row["final_reviewer_username"],
                    "source": a["source"],
                    "target": a["target"],
                    "similarity": a["similarity"],
                    "source_idx": a["source_idx"],
                    "target_idx": a["target_idx"],
                    "identical": a["identical"],
                }
                if a["identical"]:
                    ident_rows.append(rec)
                else:
                    diff_rows.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    diff_path = out_dir / "uchen_batch_3_sentence_pairs_differing.csv"
    ident_path = out_dir / "uchen_batch_3_sentence_pairs_identical.csv"
    # Legacy alias for differing pairs
    legacy_path = out_dir / "uchen_batch_3_sentence_pairs.csv"

    pd.DataFrame(diff_rows).to_csv(diff_path, index=False, encoding="utf-8")
    pd.DataFrame(ident_rows).to_csv(ident_path, index=False, encoding="utf-8")
    pd.DataFrame(diff_rows).to_csv(legacy_path, index=False, encoding="utf-8")
    return len(diff_rows), len(ident_rows)


def export_page_csvs(raw_csv: Path, out_dir: Path) -> tuple[int, int]:
    raw = pd.read_csv(raw_csv)
    ident_rows = []
    diff_rows = []

    for _, row in raw.iterrows():
        gold = norm(row["final_reviewer_transcript"])
        for role, tcol, ucol in (
            ("annotator_a", "annotator_a_transcript", "annotator_a_username"),
            ("annotator_b", "annotator_b_transcript", "annotator_b_username"),
        ):
            src = norm(row[tcol])
            if src is None or gold is None:
                continue
            rec = {
                "batch_id": row["batch_id"],
                "batch_name": row["batch_name"],
                "task_id": row["task_id"],
                "image_id": row["image_id"],
                "image_url": row["image_url"],
                "orientation": row["orientation"],
                "state": row["state"],
                "source_role": role,
                "annotator_username": row[ucol],
                "reviewer_a_username": row["reviewer_a_username"],
                "reviewer_b_username": row["reviewer_b_username"],
                "final_reviewer_username": row["final_reviewer_username"],
                "source": src,
                "target": gold,
                "identical": src == gold,
            }
            if src == gold:
                ident_rows.append(rec)
            else:
                diff_rows.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diff_rows).to_csv(
        out_dir / "uchen_batch_3_page_pairs_differing.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(ident_rows).to_csv(
        out_dir / "uchen_batch_3_page_pairs_identical.csv", index=False, encoding="utf-8"
    )
    # Legacy alias
    pd.DataFrame(diff_rows).to_csv(
        out_dir / "uchen_batch_3_page_pairs.csv", index=False, encoding="utf-8"
    )
    return len(diff_rows), len(ident_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("dataset/ocr_annotation/uchen_batch_3.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("dataset/ocr_annotation/exports"),
    )
    args = parser.parse_args()

    n_diff_s, n_ident_s = export_sentence_csvs(args.csv, args.out_dir)
    n_diff_p, n_ident_p = export_page_csvs(args.csv, args.out_dir)

    print("Sentence pairs:")
    print(f"  differing: {n_diff_s} → {args.out_dir / 'uchen_batch_3_sentence_pairs_differing.csv'}")
    print(f"  identical: {n_ident_s} → {args.out_dir / 'uchen_batch_3_sentence_pairs_identical.csv'}")
    print("Page pairs:")
    print(f"  differing: {n_diff_p} → {args.out_dir / 'uchen_batch_3_page_pairs_differing.csv'}")
    print(f"  identical: {n_ident_p} → {args.out_dir / 'uchen_batch_3_page_pairs_identical.csv'}")


if __name__ == "__main__":
    main()
