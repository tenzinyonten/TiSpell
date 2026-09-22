#!/usr/bin/env python3
"""Sub-task 1: Parse uchen_batch_3.csv and count usable annotator/reviewer pairs.

A usable spell-correction pair is:
  source = annotator transcript (noisy OCR transcription)
  target = final_reviewer transcript (gold)

Each page yields up to two pairs (annotator_a and annotator_b vs final_reviewer).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


EMPTY_MARKERS = {"", "nan", "none", "null"}


def normalize_transcript(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in EMPTY_MARKERS:
        return None
    return text


def load_batch(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "task_id",
        "image_id",
        "state",
        "annotator_a_transcript",
        "annotator_b_transcript",
        "final_reviewer_transcript",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    return df


def extract_page_pairs(df: pd.DataFrame, require_diff: bool = True) -> list[dict]:
    """Extract annotator → final_reviewer pairs at page level."""
    pairs = []
    for _, row in df.iterrows():
        gold = normalize_transcript(row["final_reviewer_transcript"])
        if gold is None:
            continue
        for role, col in (
            ("annotator_a", "annotator_a_transcript"),
            ("annotator_b", "annotator_b_transcript"),
        ):
            src = normalize_transcript(row[col])
            if src is None:
                continue
            if require_diff and src == gold:
                continue
            pairs.append(
                {
                    "task_id": row["task_id"],
                    "image_id": row["image_id"],
                    "batch_name": row.get("batch_name", ""),
                    "state": row.get("state", ""),
                    "orientation": row.get("orientation", ""),
                    "source_role": role,
                    "source_username": row.get(f"{role}_username", ""),
                    "target_role": "final_reviewer",
                    "target_username": row.get("final_reviewer_username", ""),
                    "source": src,
                    "target": gold,
                    "identical": src == gold,
                    "source_chars": len(src),
                    "target_chars": len(gold),
                }
            )
    return pairs


def summarize(df: pd.DataFrame, pairs: list[dict], pairs_incl_identical: list[dict]) -> dict:
    empty = {}
    for col in (
        "annotator_a_transcript",
        "annotator_b_transcript",
        "reviewer_a_transcript",
        "reviewer_b_transcript",
        "final_reviewer_transcript",
    ):
        if col in df.columns:
            empty[col] = int(sum(1 for x in df[col] if normalize_transcript(x) is None))

    pages_with_diff = len({p["task_id"] for p in pairs})
    summary = {
        "n_pages": int(len(df)),
        "states": df["state"].value_counts().to_dict() if "state" in df.columns else {},
        "empty_transcripts": empty,
        "page_pairs_differing": len(pairs),
        "page_pairs_including_identical": len(pairs_incl_identical),
        "identical_page_pairs": len(pairs_incl_identical) - len(pairs),
        "pages_with_at_least_one_differing_pair": pages_with_diff,
        "by_source_role": {
            role: sum(1 for p in pairs if p["source_role"] == role)
            for role in ("annotator_a", "annotator_b")
        },
        "mean_source_chars": round(
            sum(p["source_chars"] for p in pairs) / len(pairs), 1
        )
        if pairs
        else 0,
        "mean_target_chars": round(
            sum(p["target_chars"] for p in pairs) / len(pairs), 1
        )
        if pairs
        else 0,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("dataset/ocr_annotation/uchen_batch_3.csv"),
        help="Path to uchen_batch_3.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("dataset/ocr_annotation/parsed"),
        help="Directory for parsed pairs and summary",
    )
    parser.add_argument(
        "--include_identical",
        action="store_true",
        help="Also keep pairs where source == target",
    )
    args = parser.parse_args()

    df = load_batch(args.csv)
    pairs_diff = extract_page_pairs(df, require_diff=True)
    pairs_all = extract_page_pairs(df, require_diff=False)
    pairs = pairs_all if args.include_identical else pairs_diff
    summary = summarize(df, pairs_diff, pairs_all)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.out_dir / "page_pairs.jsonl"
    summary_path = args.out_dir / "parse_summary.json"

    with open(pairs_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== uchen_batch_3 parse summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(pairs)} page pairs → {pairs_path}")
    print(f"Wrote summary → {summary_path}")


if __name__ == "__main__":
    main()
