"""
No-op baseline.

Scores a "model" that outputs its input unchanged, using token-level
precision/recall/F1 over the whole sequence -- the same shape of metric the
TiSpell paper reports.

Why this matters: in the TiSpell paper's Table 1, a dummy model that makes no
corrections at all scores 89.26 F1 on mixed corruption, and their model scores
93.43. So the model contributes about 4 points over doing nothing. Without the
equivalent number on this dataset, our 0.7897 cannot be compared to theirs at
all.

Two numbers come out of this:

  1. no-op F1   -- what you get for free
  2. the gap    -- no-op vs. the trained model, i.e. the work the model does

Run this on the validation set. No GPU needed.

Usage:
    pip install pandas
    python noop_baseline.py
"""

import re
import sys
from collections import Counter

import pandas as pd

# Syllable delimiters: tsheg, shad and relatives, whitespace
DELIM = re.compile(
    r'[\u0f0b\u0f0c\u0f0d\u0f0e\u0f0f\u0f10\u0f11\u0f12\u0f14\s]+'
)


def tokenize(text: str):
    """Split into syllables. Empty pieces dropped."""
    if not isinstance(text, str):
        return []
    return [t for t in DELIM.split(text) if t]


def prf_sequence(pred_toks, gold_toks):
    """Token-level P/R/F1 for one sequence, multiset-based.

    Matching is by token identity with multiplicity, which is how sequence-level
    correction is normally scored when positions can shift (insertions and
    deletions move everything after them).
    """
    if not pred_toks and not gold_toks:
        return 1.0, 1.0, 1.0
    if not pred_toks or not gold_toks:
        return 0.0, 0.0, 0.0

    pred_c = Counter(pred_toks)
    gold_c = Counter(gold_toks)
    overlap = sum((pred_c & gold_c).values())

    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate(df, pred_col, gold_col='target'):
    """Mean P/R/F1 overall and per diff_category."""
    rows = []
    for _, r in df.iterrows():
        p, rec, f1 = prf_sequence(tokenize(r[pred_col]), tokenize(r[gold_col]))
        rows.append({
            'diff_category': r['diff_category'],
            'precision': p,
            'recall': rec,
            'f1': f1,
        })
    scored = pd.DataFrame(rows)

    overall = scored[['precision', 'recall', 'f1']].mean()
    per_cat = scored.groupby('diff_category')[['precision', 'recall', 'f1']].agg(
        ['mean', 'count'])
    return overall, per_cat, scored


def main(path='validation.csv'):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        sys.exit(f"Could not read {path}: {e}")

    for col in ('source', 'target'):
        if col not in df.columns:
            sys.exit(f"Missing '{col}' column. Found: {list(df.columns)}")

    if 'diff_category' not in df.columns:
        df['diff_category'] = 'unspecified'
    df['diff_category'] = df['diff_category'].fillna('unspecified')
    df['source'] = df['source'].fillna('').astype(str)
    df['target'] = df['target'].fillna('').astype(str)

    print(f"Loaded {len(df):,} pairs from {path}\n")

    # The no-op model: prediction == input, unchanged.
    overall, per_cat, _ = evaluate(df, pred_col='source')

    print("=" * 70)
    print("  NO-OP BASELINE  (model outputs its input unchanged)")
    print("=" * 70)
    print(f"\nOverall   precision={overall['precision']:.4f}  "
          f"recall={overall['recall']:.4f}  f1={overall['f1']:.4f}\n")

    print("Per category:")
    print("-" * 70)
    flat = per_cat.copy()
    flat.columns = ['_'.join(c) for c in flat.columns]
    flat = flat[['f1_mean', 'precision_mean', 'recall_mean', 'f1_count']]
    flat.columns = ['f1', 'precision', 'recall', 'n']
    print(flat.sort_values('n', ascending=False).to_string(
        float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)
    noop = overall['f1']
    trained = 0.7897  # best run: lr 2e-5 + dropout 0.2
    print(f"""
  No-op baseline on this data : {noop:.4f}
  Best trained model          : {trained:.4f}
  Gap (work the model does)   : {trained - noop:+.4f}

  For comparison, from the TiSpell paper's Table 1 (mixed corruption):

    dummy (no correction)     : 0.8926
    TiSpell-RoBERTa           : 0.9343
    Gap                       : +0.0417

  If the gap here is larger than +0.0417, this model is doing more actual
  correction than TiSpell does on its own benchmark -- even though the
  headline number is lower. That is the honest comparison.

  Caveat worth stating when reporting this: the metric here is token-level
  over syllables, matched as a multiset. The paper does not fully specify
  its scoring, so this reproduces the shape of the metric, not necessarily
  the exact implementation. The no-op number is directly comparable to the
  paper's dummy row in spirit -- both answer "what do you get for free?"
""")


if __name__ == "__main__":
    main('validation.csv')