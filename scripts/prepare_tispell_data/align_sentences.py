#!/usr/bin/env python3
"""Sub-task 2: Segment page transcripts into sentences and align annotator → reviewer.

Segmentation uses Tibetan shad punctuation (། ༎ ༏ ༐ ༑), matching TiSpell's
dataloader utilities. Alignment is greedy sequence matching on normalized
syllable strings with a length/similarity gate so we only keep confident pairs.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

SHAD_RE = re.compile(r"[།༎༏༐༑]")
TSHEG_RE = re.compile(r"[་༌]")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    """Light normalization for alignment only (not for training text)."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\n", "").replace("\r", "")
    text = WHITESPACE_RE.sub("", text)
    return text


def segment_sentences(text: str, min_chars: int = 8) -> list[str]:
    """Split on Tibetan sentence-final punctuation; keep delimiter on the left."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Soft-join line breaks inside a page (OCR wraps lines mid-sentence).
    text = text.replace("\n", "")
    parts = SHAD_RE.split(text)
    delims = SHAD_RE.findall(text)
    sentences = []
    for i, part in enumerate(parts):
        s = part.strip()
        if not s:
            continue
        if i < len(delims):
            s = s + delims[i]
        s = WHITESPACE_RE.sub("", s)
        if len(s) >= min_chars:
            sentences.append(s)
    return sentences


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_for_match(a), normalize_for_match(b)).ratio()


def align_sentences(
    source_sents: list[str],
    target_sents: list[str],
    min_ratio: float = 0.55,
    max_len_ratio: float = 2.5,
) -> list[dict]:
    """Greedy monotonic alignment of source sentences to target sentences."""
    aligned = []
    j = 0
    for i, src in enumerate(source_sents):
        if j >= len(target_sents):
            break
        best_j = None
        best_score = -1.0
        # Look ahead a small window to absorb split/merge differences.
        for k in range(j, min(j + 3, len(target_sents))):
            tgt = target_sents[k]
            len_ratio = max(len(src), len(tgt)) / max(1, min(len(src), len(tgt)))
            if len_ratio > max_len_ratio:
                continue
            score = similarity(src, tgt)
            if score > best_score:
                best_score = score
                best_j = k
        if best_j is None or best_score < min_ratio:
            continue
        tgt = target_sents[best_j]
        aligned.append(
            {
                "source": src,
                "target": tgt,
                "similarity": round(best_score, 4),
                "source_idx": i,
                "target_idx": best_j,
                "identical": src == tgt,
            }
        )
        j = best_j + 1
    return aligned


def load_page_pairs(path: Path) -> list[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--page_pairs",
        type=Path,
        default=Path("dataset/ocr_annotation/parsed/page_pairs.jsonl"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("dataset/ocr_annotation/parsed"),
    )
    parser.add_argument("--min_chars", type=int, default=8)
    parser.add_argument("--min_ratio", type=float, default=0.55)
    args = parser.parse_args()

    page_pairs = load_page_pairs(args.page_pairs)
    sentence_pairs = []
    page_stats = []

    for page in page_pairs:
        src_sents = segment_sentences(page["source"], min_chars=args.min_chars)
        tgt_sents = segment_sentences(page["target"], min_chars=args.min_chars)
        aligned = align_sentences(src_sents, tgt_sents, min_ratio=args.min_ratio)
        page_stats.append(
            {
                "task_id": page["task_id"],
                "source_role": page["source_role"],
                "n_source_sents": len(src_sents),
                "n_target_sents": len(tgt_sents),
                "n_aligned": len(aligned),
                "n_aligned_diff": sum(1 for a in aligned if not a["identical"]),
            }
        )
        for a in aligned:
            if a["identical"]:
                continue
            sentence_pairs.append(
                {
                    "task_id": page["task_id"],
                    "image_id": page["image_id"],
                    "source_role": page["source_role"],
                    "source_username": page.get("source_username", ""),
                    "target_username": page.get("target_username", ""),
                    "source": a["source"],
                    "target": a["target"],
                    "similarity": a["similarity"],
                    "source_idx": a["source_idx"],
                    "target_idx": a["target_idx"],
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pairs = args.out_dir / "sentence_pairs.jsonl"
    out_stats = args.out_dir / "align_summary.json"

    with open(out_pairs, "w", encoding="utf-8") as f:
        for p in sentence_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    n_src = sum(s["n_source_sents"] for s in page_stats)
    n_tgt = sum(s["n_target_sents"] for s in page_stats)
    n_aligned_all = sum(s["n_aligned"] for s in page_stats)
    n_aligned_diff = sum(s["n_aligned_diff"] for s in page_stats)
    summary = {
        "n_page_pairs": len(page_pairs),
        "n_source_sentences": n_src,
        "n_target_sentences": n_tgt,
        "n_aligned_including_identical": n_aligned_all,
        "n_aligned_differing": n_aligned_diff,
        "alignment_rate_vs_source": round(n_aligned_all / n_src, 4) if n_src else 0,
        "mean_similarity_diff_pairs": round(
            sum(p["similarity"] for p in sentence_pairs) / len(sentence_pairs), 4
        )
        if sentence_pairs
        else 0,
        "mean_source_chars": round(
            sum(len(p["source"]) for p in sentence_pairs) / len(sentence_pairs), 1
        )
        if sentence_pairs
        else 0,
        "min_ratio": args.min_ratio,
        "min_chars": args.min_chars,
    }

    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== sentence alignment summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(sentence_pairs)} differing sentence pairs → {out_pairs}")


if __name__ == "__main__":
    main()
