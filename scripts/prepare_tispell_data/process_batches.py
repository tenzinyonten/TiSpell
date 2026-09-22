#!/usr/bin/env python3
"""Process one or more OCR annotation batch CSVs into sentence/page pair exports.

Supports both schemas:
  A) uchen_batch_3 / ume_batch_3 style:
     annotator_*_transcript, final_reviewer_transcript, image_id, ...
  B) ITv2 export style:
     annotator_*_response, final_reviewer_response, file_number, batch, ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_sentences import align_sentences, segment_sentences

EMPTY_MARKERS = {"", "nan", "none", "null"}

# Canonical column names used after normalization
CANON = {
    "batch_name",
    "task_id",
    "image_id",
    "image_url",
    "state",
    "annotator_a_username",
    "annotator_a_transcript",
    "annotator_b_username",
    "annotator_b_transcript",
    "reviewer_a_username",
    "reviewer_a_transcript",
    "reviewer_b_username",
    "reviewer_b_transcript",
    "final_reviewer_username",
    "final_reviewer_transcript",
}


def norm_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in EMPTY_MARKERS:
        return None
    return text


def normalize_batch_df(df: pd.DataFrame, fallback_batch_name: str) -> pd.DataFrame:
    """Map either schema onto a common column set."""
    cols = set(df.columns)
    out = pd.DataFrame()

    if "final_reviewer_transcript" in cols:
        # Schema A
        out["batch_name"] = df.get("batch_name", fallback_batch_name)
        out["task_id"] = df["task_id"]
        out["image_id"] = df.get("image_id", df.get("file_number", ""))
        out["image_url"] = df.get("image_url", "")
        out["state"] = df.get("state", "")
        out["annotator_a_username"] = df.get("annotator_a_username", "")
        out["annotator_a_transcript"] = df["annotator_a_transcript"]
        out["annotator_b_username"] = df.get("annotator_b_username", "")
        out["annotator_b_transcript"] = df["annotator_b_transcript"]
        out["reviewer_a_username"] = df.get("reviewer_a_username", "")
        out["reviewer_a_transcript"] = df.get("reviewer_a_transcript", "")
        out["reviewer_b_username"] = df.get("reviewer_b_username", "")
        out["reviewer_b_transcript"] = df.get("reviewer_b_transcript", "")
        out["final_reviewer_username"] = df.get("final_reviewer_username", "")
        out["final_reviewer_transcript"] = df["final_reviewer_transcript"]
    elif "final_reviewer_response" in cols:
        # Schema B (ITv2)
        out["batch_name"] = df.get("batch", fallback_batch_name)
        out["task_id"] = df["task_id"]
        out["image_id"] = df.get("file_number", "")
        out["image_url"] = df.get("image_url", "")
        out["state"] = df.get("state", "")
        out["annotator_a_username"] = df.get("annotator_a", "")
        out["annotator_a_transcript"] = df["annotator_a_response"]
        out["annotator_b_username"] = df.get("annotator_b", "")
        out["annotator_b_transcript"] = df["annotator_b_response"]
        out["reviewer_a_username"] = df.get("reviewer_a", "")
        out["reviewer_a_transcript"] = df.get("reviewer_a_response", "")
        out["reviewer_b_username"] = df.get("reviewer_b", "")
        out["reviewer_b_transcript"] = df.get("reviewer_b_response", "")
        out["final_reviewer_username"] = df.get("final_reviewer", "")
        out["final_reviewer_transcript"] = df["final_reviewer_response"]
    else:
        raise ValueError(
            f"Unrecognized schema. Columns: {sorted(cols)}"
        )

    # Ensure batch_name is a series of strings
    if not isinstance(out["batch_name"], pd.Series):
        out["batch_name"] = fallback_batch_name
    out["batch_name"] = out["batch_name"].fillna(fallback_batch_name).astype(str)
    return out


def process_batch(df: pd.DataFrame, batch_name: str) -> dict:
    """Return page/sentence identical+differing rows and a summary."""
    page_diff, page_ident = [], []
    sent_diff, sent_ident = [], []

    for _, row in df.iterrows():
        gold = norm_text(row["final_reviewer_transcript"])
        bname = row.get("batch_name", batch_name)
        for role, tcol, ucol in (
            ("annotator_a", "annotator_a_transcript", "annotator_a_username"),
            ("annotator_b", "annotator_b_transcript", "annotator_b_username"),
        ):
            src = norm_text(row[tcol])
            if src is None or gold is None:
                continue

            page_rec = {
                "batch_name": bname,
                "task_id": row["task_id"],
                "image_id": row["image_id"],
                "image_url": row.get("image_url", ""),
                "state": row.get("state", ""),
                "source_role": role,
                "annotator_username": row.get(ucol, ""),
                "final_reviewer_username": row.get("final_reviewer_username", ""),
                "source": src,
                "target": gold,
                "identical": src == gold,
            }
            if src == gold:
                page_ident.append(page_rec)
            else:
                page_diff.append(page_rec)

            for a in align_sentences(segment_sentences(src), segment_sentences(gold)):
                sent_rec = {
                    "batch_name": bname,
                    "task_id": row["task_id"],
                    "image_id": row["image_id"],
                    "source_role": role,
                    "annotator_username": row.get(ucol, ""),
                    "final_reviewer_username": row.get("final_reviewer_username", ""),
                    "source": a["source"],
                    "target": a["target"],
                    "similarity": a["similarity"],
                    "source_idx": a["source_idx"],
                    "target_idx": a["target_idx"],
                    "identical": a["identical"],
                }
                if a["identical"]:
                    sent_ident.append(sent_rec)
                else:
                    sent_diff.append(sent_rec)

    summary = {
        "batch_name": batch_name,
        "n_pages": int(len(df)),
        "page_pairs_differing": len(page_diff),
        "page_pairs_identical": len(page_ident),
        "sentence_pairs_differing": len(sent_diff),
        "sentence_pairs_identical": len(sent_ident),
        "sentence_pairs_total": len(sent_diff) + len(sent_ident),
    }
    return {
        "summary": summary,
        "page_diff": page_diff,
        "page_ident": page_ident,
        "sent_diff": sent_diff,
        "sent_ident": sent_ident,
    }


def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("dataset/ocr_annotation/raw"),
        help="Directory containing batch CSVs",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional specific CSV filenames (default: all *.csv in input_dir)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("dataset/ocr_annotation/exports"),
    )
    parser.add_argument(
        "--summary_path",
        type=Path,
        default=Path("dataset/ocr_annotation/parsed/batches_summary.json"),
    )
    args = parser.parse_args()

    if args.files:
        paths = [args.input_dir / f for f in args.files]
    else:
        paths = sorted(args.input_dir.glob("*.csv"))

    if not paths:
        raise SystemExit(f"No CSV files found under {args.input_dir}")

    all_page_diff, all_page_ident = [], []
    all_sent_diff, all_sent_ident = [], []
    summaries = []

    for path in paths:
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        fallback = path.stem
        raw = pd.read_csv(path)
        df = normalize_batch_df(raw, fallback_batch_name=fallback)
        print(f"Processing {path.name} ({len(df)} pages)...")
        result = process_batch(df, batch_name=fallback)
        summaries.append(result["summary"])

        # per-batch exports
        batch_dir = args.out_dir / fallback
        write_csv(result["page_diff"], batch_dir / "page_pairs_differing.csv")
        write_csv(result["page_ident"], batch_dir / "page_pairs_identical.csv")
        write_csv(result["sent_diff"], batch_dir / "sentence_pairs_differing.csv")
        write_csv(result["sent_ident"], batch_dir / "sentence_pairs_identical.csv")

        all_page_diff.extend(result["page_diff"])
        all_page_ident.extend(result["page_ident"])
        all_sent_diff.extend(result["sent_diff"])
        all_sent_ident.extend(result["sent_ident"])

        s = result["summary"]
        print(
            f"  pages={s['n_pages']} | "
            f"page diff/ident={s['page_pairs_differing']}/{s['page_pairs_identical']} | "
            f"sent diff/ident={s['sentence_pairs_differing']}/{s['sentence_pairs_identical']}"
        )

    # combined exports
    write_csv(all_page_diff, args.out_dir / "all_page_pairs_differing.csv")
    write_csv(all_page_ident, args.out_dir / "all_page_pairs_identical.csv")
    write_csv(all_sent_diff, args.out_dir / "all_sentence_pairs_differing.csv")
    write_csv(all_sent_ident, args.out_dir / "all_sentence_pairs_identical.csv")

    total = {
        "batches": summaries,
        "totals": {
            "n_pages": sum(s["n_pages"] for s in summaries),
            "page_pairs_differing": len(all_page_diff),
            "page_pairs_identical": len(all_page_ident),
            "sentence_pairs_differing": len(all_sent_diff),
            "sentence_pairs_identical": len(all_sent_ident),
        },
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(total, f, ensure_ascii=False, indent=2)

    print("\n=== TOTALS ===")
    print(json.dumps(total["totals"], indent=2))
    print(f"\nCombined exports → {args.out_dir}/all_*.csv")
    print(f"Summary → {args.summary_path}")


if __name__ == "__main__":
    main()
