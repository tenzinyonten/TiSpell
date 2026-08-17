#!/usr/bin/env python3
"""Re-judge a sample of already-checked rows on a second model and compare.

Answers one question: did the model tier / batch size actually change the
verdicts, or would a stronger model have said the same thing?

Reuses the prompt and retry logic from check_tibetan_targets_gemini.py so the
only variables are --model and --batch_size.

Example:
  export GEMINI_API_KEY=...
  python scripts/compare_model_agreement.py \
      --n 200 --model gemini-3.6-flash --batch_size 10
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_tibetan_targets_gemini import (  # noqa: E402
    BatchAPIError,
    VERDICTS,
    call_gemini_batch,
)

from google import genai  # noqa: E402

DEFAULT_BASELINE = Path("dataset/ocr_annotation/exports/gemini_target_check.csv")
DEFAULT_OUTPUT = Path("dataset/ocr_annotation/exports/model_agreement_check.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE,
                   help="CSV of verdicts from the original run")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--n", type=int, default=200,
                   help="How many rows to re-judge (default 200)")
    p.add_argument("--model", type=str, default="gemini-3.6-flash",
                   help="Model to re-judge with")
    p.add_argument("--batch_size", type=int, default=10,
                   help="Rows per request. Keep small - large batches are the "
                        "suspected cause of cross-row attribution errors.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max_retries", type=int, default=6)
    p.add_argument("--request_pause", type=float, default=2.0)
    return p.parse_args()


def main() -> None:
    import os
    import time

    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set.")

    if not args.baseline.is_file():
        raise SystemExit(f"Baseline not found: {args.baseline}")

    base = pd.read_csv(args.baseline, dtype=str, keep_default_na=False)
    for col in ("page_id", "target", "verdict"):
        if col not in base.columns:
            raise SystemExit(f"Baseline missing column: {col}")

    # dedupe on target - the original run had ~6k duplicate extras, and
    # re-judging the same string twice tells us nothing.
    base = base.drop_duplicates(subset=["target"], keep="first")
    base = base[base["verdict"].isin(VERDICTS)].reset_index(drop=True)
    print(f"{len(base)} unique judged rows in baseline")

    sample = base.sample(n=min(args.n, len(base)), random_state=args.seed)
    sample = sample.reset_index(drop=True)
    print(f"re-judging {len(sample)} rows on {args.model} "
          f"at batch_size {args.batch_size}\n")

    client = genai.Client(api_key=api_key)
    rows = sample[["page_id", "target"]].to_dict("records")

    new_verdicts: list[dict] = []
    n_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    for b in range(n_batches):
        chunk = rows[b * args.batch_size:(b + 1) * args.batch_size]
        print(f"batch {b + 1}/{n_batches}")
        try:
            judged = call_gemini_batch(
                client, args.model, chunk, max_retries=args.max_retries
            )
        except BatchAPIError as exc:
            print(f"  batch failed, skipping: {exc}", file=sys.stderr)
            judged = [{"verdict": "", "error_count": 0,
                       "error_description": "batch failed"} for _ in chunk]
        for row, j in zip(chunk, judged):
            new_verdicts.append({
                "page_id": row["page_id"],
                "target": row["target"],
                "verdict_new": j["verdict"],
                "error_count_new": j["error_count"],
                "error_description_new": j["error_description"],
            })
        if b + 1 < n_batches and args.request_pause > 0:
            time.sleep(args.request_pause)

    new_df = pd.DataFrame(new_verdicts)
    merged = sample.merge(new_df[["target", "verdict_new", "error_count_new",
                                  "error_description_new"]],
                          on="target", how="left")
    merged = merged.rename(columns={"verdict": "verdict_old",
                                    "error_count": "error_count_old"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    # ---- comparison ----
    comp = merged[merged["verdict_new"].isin(VERDICTS)]
    n = len(comp)
    if n == 0:
        raise SystemExit("no comparable rows - all batches failed")

    agree = (comp["verdict_old"] == comp["verdict_new"]).sum()
    print(f"\n{'=' * 52}")
    print(f"compared {n} rows")
    print(f"agreement: {agree}/{n} = {agree / n:.1%}")

    old_inc = (comp["verdict_old"] == "incorrect").mean()
    new_inc = (comp["verdict_new"] == "incorrect").mean()
    print(f"\nincorrect rate, old model: {old_inc:.1%}")
    print(f"incorrect rate, new model: {new_inc:.1%}")
    print(f"difference: {abs(new_inc - old_inc) * 100:.1f} points")

    print("\nconfusion (old -> new):")
    conf = Counter(zip(comp["verdict_old"], comp["verdict_new"]))
    for (o, nv), c in sorted(conf.items(), key=lambda x: -x[1]):
        mark = "" if o == nv else "   <- disagreement"
        print(f"  {o:<10} -> {nv:<10} {c:4d}{mark}")

    print(f"\nfull side-by-side written to {args.output}")
    print("\nREAD THE DISAGREEMENTS YOURSELF - I can't tell you which model was "
          "right, only that they differed. Open the CSV and check a dozen rows "
          "where verdict_old != verdict_new.")


if __name__ == "__main__":
    main()