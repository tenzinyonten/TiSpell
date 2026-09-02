#!/usr/bin/env python3
"""Check whether reviewer (target) Tibetan text looks correct via Gemini.

Reads differing_pairs.csv, samples/limits rows, and asks Gemini whether each
``target`` has correct Tibetan spelling/grammar. Appends verdicts to CSV after
each batch and resumes from any existing output (skips already-written pairs).

Requires ``GEMINI_API_KEY`` in the environment (never hardcoded).

Example:
  export GEMINI_API_KEY=...
  python scripts/check_tibetan_targets_gemini.py --sample --limit 100
  # later, after a crash / quota hit:
  python scripts/check_tibetan_targets_gemini.py --limit 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    from google import genai
    from google.genai import types
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "google-genai is required. Install with: pip install google-genai"
    ) from e


DEFAULT_INPUT = Path("dataset/ocr_annotation/exports/differing_pairs.csv")
DEFAULT_OUTPUT = Path("dataset/ocr_annotation/exports/gemini_target_check.csv")
DEFAULT_MODEL = "gemini-2.5-flash"

VERDICTS = ("correct", "incorrect", "unsure")
OUTPUT_COLS = ["page_id", "target", "verdict", "error_count", "error_description"]

SYSTEM_PROMPT = """You are a strict Tibetan (Uchen) orthographic proofreader for OCR error detection.

You will receive Tibetan text snippets claimed to be correct reviewer transcriptions.
Your SOLE task: judge whether each snippet contains a genuine physical spelling or
particle-orthography error. Judge ORTHOGRAPHY ONLY — never style, meaning, or usage.

FLAG AS "incorrect":
1. Malformed or impossible character stacks, broken glyph topologies, corrupted vowels (e.g. སྡུཾལ་).
2. Incorrect particle spelling or suffix-harmony violations (e.g. བདགིས་ for བདག་གིས་, བཀག་བ་ for བཀག་པ་).
3. Missing or extra tsheg causing invalid merged/split syllables, or corrupted root letters.

DO NOT flag (verdict "correct"):
1. Stylistic redundancy, clumsy phrasing, awkward prose (e.g. སེམས་ཐག་non-standard punctuation spacing.
3. Old Tibetan orthography (e.g. མྱེད་ for མེད་), archaic particles, ritual/medical jargon.
4. SCRIBAL ABBREVIATIONS (standard manuscript conventions, NOT errors):
   - Anusvara U+0F7E (ཾ) for a final མ (e.g. སེཾས་ for སེམས་, རྣཾས་ for རྣམས་, ཟླ་གཾ་ for ཟླ་གམ་).
   - U+0F4C (ཌ) for a final གས (e.g. བཞུཌ་ for བཞུགས་, ཚོཌ་ for ཚོགས་, གླེཌ་ for གླེགས་).
   Treat these as CORRECT unless there is a separate, real error.

If every syllable and particle is spelled correctly (allowing the conventions above),
the verdict is "correct", even if the phrasing is repetitive or clumsy.

Rules:
- "correct" when confident the orthography is fine.
- "incorrect" only for real spelling/particle errors OTHER than the accepted conventions above.
- "unsure" when not confident — do NOT guess.
- For "incorrect": set error_count to the number of distinct spelling problems and
  describe them briefly in English in error_description.
- For "correct"/"unsure": error_count = 0, error_description = "".
- Preserve the input index of each item.
- Respond with JSON only (no markdown fences).
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "verdict": {
                        "type": "string",
                        "enum": list(VERDICTS),
                    },
                    "error_count": {"type": "integer"},
                    "error_description": {"type": "string"},
                },
                "required": [
                    "index",
                    "verdict",
                    "error_count",
                    "error_description",
                ],
            },
        }
    },
    "required": ["results"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max number of rows to check (default 100; use 0 for all)",
    )
    p.add_argument(
        "--sample",
        action="store_true",
        help="Randomly sample --limit rows instead of taking the first rows",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for --sample")
    p.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of targets per Gemini request (default 5)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model id (default {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--max_retries",
        type=int,
        default=6,
        help="Max retries per batch on rate-limit / transient errors",
    )
    p.add_argument(
        "--request_pause",
        type=float,
        default=1.0,
        help="Seconds to sleep between successful batches",
    )
    p.add_argument(
        "--dedupe_targets",
        action="store_true",
        help="Check each unique target once (resume key becomes target only)",
    )
    return p.parse_args()


def _is_rate_limit(exc: BaseException) -> bool:
    text = str(exc).lower()
    if "429" in text or "rate" in text or "resource_exhausted" in text:
        return True
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code == 429


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _normalize_verdict(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s in VERDICTS:
        return s
    if s in ("ok", "yes", "good", "valid"):
        return "correct"
    if s in ("bad", "no", "error", "wrong", "invalid"):
        return "incorrect"
    return "unsure"


class BatchAPIError(RuntimeError):
    """Raised when a batch fails after retries (nothing written for that batch)."""


def call_gemini_batch(
    client: genai.Client,
    model: str,
    batch_rows: list[dict],
    *,
    max_retries: int,
) -> list[dict]:
    """Ask Gemini about a small batch of targets. Returns one dict per input index.

    Raises BatchAPIError if the request fails after retries (caller should not
    append those rows, so a resume can retry them).
    """
    payload = [
        {"index": i, "text": row["target"]} for i, row in enumerate(batch_rows)
    ]
    user_prompt = (
        "Evaluate each Tibetan text below.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    last_err: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                ),
            )
            raw = (response.text or "").strip()
            if not raw:
                raise ValueError("Empty response from Gemini")
            parsed = _extract_json(raw)
            results = parsed.get("results", parsed if isinstance(parsed, list) else [])
            if not isinstance(results, list):
                raise ValueError(f"Unexpected JSON shape: {parsed!r}")

            by_index: dict[int, dict] = {}
            for item in results:
                try:
                    idx = int(item["index"])
                except (KeyError, TypeError, ValueError):
                    continue
                verdict = _normalize_verdict(item.get("verdict"))
                try:
                    err_n = int(item.get("error_count", 0) or 0)
                except (TypeError, ValueError):
                    err_n = 0
                desc = str(item.get("error_description") or "")
                if verdict != "incorrect":
                    err_n = 0
                    if verdict == "correct":
                        desc = ""
                by_index[idx] = {
                    "index": idx,
                    "verdict": verdict,
                    "error_count": err_n,
                    "error_description": desc,
                }

            out = []
            for i in range(len(batch_rows)):
                if i in by_index:
                    out.append(by_index[i])
                else:
                    out.append(
                        {
                            "index": i,
                            "verdict": "unsure",
                            "error_count": 0,
                            "error_description": (
                                "model omitted this item; marked unsure"
                            ),
                        }
                    )
            return out
        except Exception as exc:  # noqa: BLE001 — retry transient API failures
            last_err = exc
            if attempt >= max_retries:
                break
            base = 2.0 ** attempt
            sleep_s = base * (8.0 if _is_rate_limit(exc) else 2.0)
            sleep_s = min(sleep_s, 120.0)
            print(
                f"  retry {attempt + 1}/{max_retries} after error: {exc!s} "
                f"(sleep {sleep_s:.1f}s)",
                file=sys.stderr,
            )
            time.sleep(sleep_s)

    raise BatchAPIError(f"API failure after retries: {last_err}")


def select_rows(df: pd.DataFrame, *, limit: int, sample: bool, seed: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(df):
        out = df
    elif sample:
        out = df.sample(n=limit, random_state=seed)
    else:
        out = df.head(limit)
    return out.reset_index(drop=True)


def load_done_keys(output: Path, *, by_target: bool) -> set:
    """Return keys already present in the output CSV for resume."""
    if not output.is_file() or output.stat().st_size == 0:
        return set()
    try:
        prev = pd.read_csv(output, dtype=str, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not read existing output ({exc}); starting fresh")
        return set()
    if prev.empty:
        return set()
    if by_target:
        if "target" not in prev.columns:
            return set()
        return set(prev["target"].astype(str))
    # page_id alone is not unique (many segments per page); use (page_id, target).
    if not {"page_id", "target"}.issubset(prev.columns):
        return set()
    return set(zip(prev["page_id"].astype(str), prev["target"].astype(str)))


def append_batch_rows(output: Path, rows: list[dict]) -> None:
    """Append judged rows to CSV, writing a header only if the file is new/empty."""
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not output.is_file()) or output.stat().st_size == 0
    with output.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=OUTPUT_COLS, extrasaction="ignore", lineterminator="\n"
        )
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in OUTPUT_COLS})
        f.flush()
        os.fsync(f.fileno())


def print_summary(output: Path) -> None:
    if not output.is_file() or output.stat().st_size == 0:
        print("\nNo results to summarize.")
        return
    out = pd.read_csv(output, dtype=str, keep_default_na=False)
    counts = Counter(out["verdict"].astype(str))
    print(f"\nWrote/updated → {output}  (total rows on disk: {len(out)})")
    print("\nVerdict summary (full output file):")
    for key in VERDICTS:
        n = counts.get(key, 0)
        print(f"  {key:<10} {n:5d}  ({100 * n / max(len(out), 1):5.1f}%)")
    other = sum(v for k, v in counts.items() if k not in VERDICTS)
    if other:
        print(f"  {'other':<10} {other:5d}")
    print(f"  {'total':<10} {len(out):5d}")


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY is not set. Export it before running:\n"
            "  export GEMINI_API_KEY=your_key_here"
        )

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)
    required = {"page_id", "target"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input missing columns: {sorted(missing)}")

    df = df.copy()
    df["target"] = df["target"].astype(str)
    df["page_id"] = df["page_id"].astype(str)
    work = select_rows(df, limit=args.limit, sample=args.sample, seed=args.seed)
    print(
        f"Loaded {len(df)} pairs; selected {len(work)} "
        f"({'sample' if args.sample else 'head' if args.limit > 0 else 'all'}, "
        f"seed={args.seed})"
    )

    done = load_done_keys(args.output, by_target=args.dedupe_targets)
    if done:
        before = len(work)
        if args.dedupe_targets:
            work = work[~work["target"].isin(done)].reset_index(drop=True)
            print(
                f"Resume: {len(done)} unique targets already in {args.output}; "
                f"skipping {before - len(work)} selected rows → {len(work)} pending"
            )
        else:
            keys = list(zip(work["page_id"], work["target"]))
            mask = [k not in done for k in keys]
            work = work.loc[mask].reset_index(drop=True)
            print(
                f"Resume: {len(done)} (page_id, target) pairs already in "
                f"{args.output}; skipping {before - len(work)} → {len(work)} pending"
            )

    if work.empty:
        print("Nothing left to check.")
        print_summary(args.output)
        return

    if args.dedupe_targets:
        check_df = (
            work[["page_id", "target"]]
            .drop_duplicates(subset=["target"], keep="first")
            .reset_index(drop=True)
        )
        print(f"Deduped pending to {len(check_df)} unique targets")
    else:
        check_df = work[["page_id", "target"]].reset_index(drop=True)

    client = genai.Client(api_key=api_key)
    n_batches = (len(check_df) + args.batch_size - 1) // args.batch_size
    n_written = 0
    failed_batches = 0

    for b_idx in range(n_batches):
        start = b_idx * args.batch_size
        end = min(start + args.batch_size, len(check_df))
        batch_rows = check_df.iloc[start:end].to_dict("records")
        print(f"Batch {b_idx + 1}/{n_batches}  rows {start}-{end - 1}")
        try:
            judged = call_gemini_batch(
                client,
                args.model,
                batch_rows,
                max_retries=args.max_retries,
            )
        except BatchAPIError as exc:
            failed_batches += 1
            print(
                f"  batch not written (will retry on resume): {exc}",
                file=sys.stderr,
            )
            # Stop on rate-limit exhaustion so we don't burn remaining quota
            # spinning; already-written batches are safe on disk.
            if _is_rate_limit(exc) or "429" in str(exc) or "resource_exhausted" in str(exc).lower():
                print(
                    "Stopping early due to rate limit / quota. "
                    "Re-run the same command to resume.",
                    file=sys.stderr,
                )
                break
            continue

        out_rows = []
        for row, judgment in zip(batch_rows, judged):
            out_rows.append(
                {
                    "page_id": row["page_id"],
                    "target": row["target"],
                    "verdict": judgment["verdict"],
                    "error_count": judgment["error_count"],
                    "error_description": judgment["error_description"],
                }
            )
        append_batch_rows(args.output, out_rows)
        n_written += len(out_rows)
        print(f"  appended {len(out_rows)} rows (session total {n_written})")

        if b_idx + 1 < n_batches and args.request_pause > 0:
            time.sleep(args.request_pause)

    print(f"\nSession appended {n_written} rows; failed batches={failed_batches}")
    print_summary(args.output)
    if failed_batches:
        sys.exit(1)


if __name__ == "__main__":
    main()
