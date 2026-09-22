#!/usr/bin/env python3
"""
Fix the split leakage between dataset versions.

THE BUG
-------
v11 added more real pairs and then re-split everything at random. A pair that
was in v10's TEST could land in v11's TRAIN, so a model trained on v11 has
already seen v10-era test pairs. Any v10-vs-v11 comparison on those pairs is
inflated.

THE FIX
-------
Split assignments must be STABLE across versions:
  1. Any pair that already existed in v10 keeps v10's split, forever.
  2. Genuinely new pairs get a fresh assignment.
  3. Assignment is done per PAGE (page_id), not per pair, so all pairs from one
     manuscript page live in the same split. Pages share scribe/scan artifacts,
     so splitting mid-page is soft leakage even within one version.

Result: no pair ever changes split, in this version or any future one.

USAGE
    python fix_splits.py \
        --v10 v10_hf/train.csv v10_hf/val.csv v10_hf/test.csv \
        --v11 data/final/v11_pairs/differing_pairs.csv \
        --out data/final/v11_fixed_split/differing_pairs.csv

If your v10 files already carry a `split` column in one file, pass it once:
    --v10 path/to/v10_all.csv
"""
import argparse
import hashlib
import os

import pandas as pd

TRAIN, VAL, TEST = 0.90, 0.05, 0.05
SEED = 42


def pair_key(src, tgt):
    """Stable identity for a pair, independent of row order or file."""
    return hashlib.md5(f'{str(src)}\x00{str(tgt)}'.encode('utf-8')).hexdigest()


def load_v10_assignments(paths):
    """pair_key -> split, from the v10 files (the authority)."""
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        if 'split' not in d.columns:
            # infer split from the filename: train.csv / val.csv / test.csv
            base = os.path.basename(p).lower()
            if 'train' in base:
                d['split'] = 'train'
            elif 'val' in base:
                d['split'] = 'val'
            elif 'test' in base:
                d['split'] = 'test'
            else:
                raise SystemExit(f'cannot infer split for {p}; add a split column')
        frames.append(d[['source', 'target', 'split']])
    v10 = pd.concat(frames, ignore_index=True)
    v10['key'] = [pair_key(s, t) for s, t in zip(v10.source, v10.target)]
    # if a pair somehow appears twice with different splits in v10, keep the
    # most conservative (test wins, then val) so it never leaks into train
    rank = {'test': 0, 'val': 1, 'train': 2}
    v10['r'] = v10.split.map(rank)
    v10 = v10.sort_values('r').drop_duplicates('key', keep='first')
    print(f'v10 assignments: {len(v10):,} pairs '
          f'({v10.split.value_counts().to_dict()})')
    return dict(zip(v10.key, v10.split))


def assign_new_pages(df, seed=SEED):
    """Assign a split to each page_id not already fixed by v10."""
    rng = pd.Series(sorted(df.page_id.astype(str).unique())).sample(
        frac=1.0, random_state=seed).tolist()
    n = len(rng)
    n_tr = int(n * TRAIN)
    n_va = int(n * VAL)
    out = {}
    for i, page in enumerate(rng):
        out[page] = 'train' if i < n_tr else ('val' if i < n_tr + n_va else 'test')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v10', nargs='+', required=True,
                    help='v10 file(s): either one file with a split column, '
                         'or train.csv val.csv test.csv')
    ap.add_argument('--v11', required=True, help='v11 pairs to re-split')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    v10_split = load_v10_assignments(args.v10)

    d = pd.read_csv(args.v11)
    print(f'v11 pairs: {len(d):,}')
    if 'page_id' not in d.columns:
        raise SystemExit('v11 file has no page_id column; page-level split '
                         'needs it. Add page_id (or doc_id) first.')

    d['key'] = [pair_key(s, t) for s, t in zip(d.source, d.target)]
    d['inherited'] = d.key.map(v10_split)

    n_inherit = d.inherited.notna().sum()
    print(f'  inherited from v10: {n_inherit:,} '
          f'({n_inherit / len(d) * 100:.1f}%)')
    print(f'  new pairs:          {len(d) - n_inherit:,}')

    # A page must land in ONE split. If any pair on a page was fixed by v10,
    # the whole page follows the most conservative inherited value.
    rank = {'test': 0, 'val': 1, 'train': 2}
    known = d[d.inherited.notna()].copy()
    known['r'] = known.inherited.map(rank)
    page_forced = (known.sort_values('r')
                        .drop_duplicates('page_id', keep='first')
                        .set_index('page_id')['inherited'].to_dict())
    print(f'  pages pinned by v10: {len(page_forced):,}')

    free = d[~d.page_id.astype(str).isin(page_forced)]
    page_new = assign_new_pages(free) if len(free) else {}
    print(f'  pages newly assigned: {len(page_new):,}')

    page_split = {**page_new, **page_forced}
    d['split'] = d.page_id.astype(str).map(page_split)
    if d.split.isna().any():
        raise SystemExit('BUG: some pairs got no split')

    d = d.drop(columns=['key', 'inherited'])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    d.to_csv(args.out, index=False)

    print(f'\nwrote {args.out}')
    print('final splits:', d.split.value_counts().to_dict())

    # ---- verification -----------------------------------------------------
    print('\nVERIFY (all should be 0)')
    d2 = pd.read_csv(args.out)
    d2['key'] = [pair_key(s, t) for s, t in zip(d2.source, d2.target)]

    moved = sum(1 for k, s in zip(d2.key, d2.split)
                if k in v10_split and v10_split[k] != s)
    print(f'  pairs whose split changed vs v10: {moved}')

    spanning = (d2.groupby('page_id').split.nunique() > 1).sum()
    print(f'  pages spanning multiple splits:   {spanning}')

    tr = set(d2[d2.split == 'train'].key)
    te = set(d2[d2.split == 'test'].key)
    va = set(d2[d2.split == 'val'].key)
    print(f'  train∩test pairs:                 {len(tr & te)}')
    print(f'  train∩val pairs:                  {len(tr & va)}')


if __name__ == '__main__':
    main()