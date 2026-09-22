#!/usr/bin/env python3
"""Inspect uchen_batch_3 CSV and parsed annotator→reviewer pairs.

Examples:
  python scripts/prepare_tispell_data/inspect_uchen_batch.py
  python scripts/prepare_tispell_data/inspect_uchen_batch.py --samples 5
  python scripts/prepare_tispell_data/inspect_uchen_batch.py --task_id 1rvmPhZllIpimGeCUVUmX
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from difflib import SequenceMatcher, ndiff
from pathlib import Path

import pandas as pd


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def char_edits(source: str, target: str) -> dict:
    sm = SequenceMatcher(None, source, target)
    inserts = deletes = replaces = equal = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            equal += i2 - i1
        elif tag == "insert":
            inserts += j2 - j1
        elif tag == "delete":
            deletes += i2 - i1
        elif tag == "replace":
            replaces += max(i2 - i1, j2 - j1)
    return {
        "inserts": inserts,
        "deletes": deletes,
        "replaces": replaces,
        "equal": equal,
        "char_error_rate": round(
            (inserts + deletes + replaces) / max(1, len(target)), 4
        ),
        "similarity": round(sm.ratio(), 4),
    }


def inline_diff(source: str, target: str, max_chars: int = 200) -> str:
    """Compact +/- view of differing characters."""
    parts = []
    for token in ndiff(source, target):
        if token.startswith("  "):
            parts.append(token[2:])
        elif token.startswith("- "):
            parts.append(f"[-{token[2:]}-]")
        elif token.startswith("+ "):
            parts.append(f"[+{token[2:]}+]")
    text = "".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def inspect_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    print_section(f"CSV: {csv_path}")
    print(f"rows: {len(df)}")
    print(f"columns ({len(df.columns)}): {list(df.columns)}")
    if "state" in df.columns:
        print(f"state: {df['state'].value_counts().to_dict()}")
    if "orientation" in df.columns:
        print(f"orientation: {df['orientation'].value_counts().to_dict()}")

    transcript_cols = [c for c in df.columns if c.endswith("_transcript")]
    print("\ntranscript emptiness / lengths:")
    for col in transcript_cols:
        non_null = df[col].fillna("").astype(str).str.strip()
        empty = int((non_null == "").sum())
        lengths = non_null.map(len)
        print(
            f"  {col}: empty={empty}, "
            f"mean_len={lengths.mean():.0f}, "
            f"min={lengths.min()}, max={lengths.max()}"
        )

    user_cols = [c for c in df.columns if c.endswith("_username")]
    if user_cols:
        print("\nusernames:")
        for col in user_cols:
            counts = df[col].fillna("").astype(str).str.strip().value_counts()
            top = ", ".join(f"{k}={v}" for k, v in counts.head(5).items())
            print(f"  {col}: {df[col].nunique()} unique | top: {top}")


def inspect_page_pairs(path: Path, samples: int, seed: int):
    pairs = load_jsonl(path)
    print_section(f"Page pairs: {path}")
    print(f"n_pairs: {len(pairs)}")
    by_role = Counter(p["source_role"] for p in pairs)
    print(f"by_source_role: {dict(by_role)}")
    if not pairs:
        return
    src_lens = [p["source_chars"] for p in pairs]
    tgt_lens = [p["target_chars"] for p in pairs]
    print(
        f"source chars: mean={sum(src_lens)/len(src_lens):.0f}, "
        f"min={min(src_lens)}, max={max(src_lens)}"
    )
    print(
        f"target chars: mean={sum(tgt_lens)/len(tgt_lens):.0f}, "
        f"min={min(tgt_lens)}, max={max(tgt_lens)}"
    )

    rng = random.Random(seed)
    for i, p in enumerate(rng.sample(pairs, min(samples, len(pairs))), 1):
        edits = char_edits(p["source"], p["target"])
        print(f"\n--- page sample {i} | {p['image_id']} | {p['source_role']} ---")
        print(f"task_id: {p['task_id']}")
        print(f"similarity={edits['similarity']} CER={edits['char_error_rate']}")
        print(f"source[:160]: {p['source'][:160]}")
        print(f"target[:160]: {p['target'][:160]}")


def inspect_sentence_pairs(path: Path, samples: int, seed: int, task_id: str | None):
    pairs = load_jsonl(path)
    if task_id:
        pairs = [p for p in pairs if p.get("task_id") == task_id]
    print_section(f"Sentence pairs: {path}")
    print(f"n_pairs: {len(pairs)}" + (f" (filtered task_id={task_id})" if task_id else ""))
    if not pairs:
        return

    sims = [p.get("similarity", SequenceMatcher(None, p["source"], p["target"]).ratio()) for p in pairs]
    cers = [char_edits(p["source"], p["target"])["char_error_rate"] for p in pairs]
    src_lens = [len(p["source"]) for p in pairs]
    print(
        f"similarity: mean={sum(sims)/len(sims):.4f}, "
        f"min={min(sims):.4f}, max={max(sims):.4f}"
    )
    print(
        f"CER: mean={sum(cers)/len(cers):.4f}, "
        f"min={min(cers):.4f}, max={max(cers):.4f}"
    )
    print(
        f"source chars: mean={sum(src_lens)/len(src_lens):.1f}, "
        f"min={min(src_lens)}, max={max(src_lens)}"
    )
    print(f"by_source_role: {dict(Counter(p['source_role'] for p in pairs))}")

    # length buckets
    buckets = Counter()
    for n in src_lens:
        if n < 20:
            buckets["<20"] += 1
        elif n < 50:
            buckets["20-49"] += 1
        elif n < 100:
            buckets["50-99"] += 1
        else:
            buckets["100+"] += 1
    print(f"source length buckets: {dict(sorted(buckets.items()))}")

    rng = random.Random(seed)
    chosen = rng.sample(pairs, min(samples, len(pairs)))
    for i, p in enumerate(chosen, 1):
        edits = char_edits(p["source"], p["target"])
        print(f"\n--- sentence sample {i} | {p.get('image_id', '')} | {p['source_role']} ---")
        print(f"task_id: {p['task_id']}  sim={p.get('similarity', edits['similarity'])}")
        print(f"CER={edits['char_error_rate']}  edits={edits['inserts']}+{edits['deletes']}+{edits['replaces']} (ins/del/rep)")
        print(f"source: {p['source']}")
        print(f"target: {p['target']}")
        print(f"diff:   {inline_diff(p['source'], p['target'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("dataset/ocr_annotation/uchen_batch_3.csv"),
    )
    parser.add_argument(
        "--parsed_dir",
        type=Path,
        default=Path("dataset/ocr_annotation/parsed"),
    )
    parser.add_argument("--samples", type=int, default=3, help="Random samples to print")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_id", type=str, default=None, help="Filter sentence pairs by task_id")
    parser.add_argument(
        "--only",
        choices=("all", "csv", "page", "sentence"),
        default="all",
        help="Which sections to show",
    )
    args = parser.parse_args()

    if args.only in ("all", "csv"):
        if args.csv.exists():
            inspect_csv(args.csv)
        else:
            print(f"[skip] CSV not found: {args.csv}")

    page_path = args.parsed_dir / "page_pairs.jsonl"
    if args.only in ("all", "page"):
        if page_path.exists():
            inspect_page_pairs(page_path, args.samples, args.seed)
        else:
            print(f"[skip] page pairs not found: {page_path}")
            print("  run: python scripts/prepare_tispell_data/parse_uchen_batch.py")

    sent_path = args.parsed_dir / "sentence_pairs.jsonl"
    if args.only in ("all", "sentence"):
        if sent_path.exists():
            inspect_sentence_pairs(sent_path, args.samples, args.seed, args.task_id)
        else:
            print(f"[skip] sentence pairs not found: {sent_path}")
            print("  run: python scripts/prepare_tispell_data/align_sentences.py")

    print()


if __name__ == "__main__":
    main()
