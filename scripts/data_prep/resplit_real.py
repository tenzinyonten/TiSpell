#!/usr/bin/env python3
"""
Re-split the REAL pairs in v11 so they inherit v10's assignments.

Rule:
  - real pair that exists in v10  -> keep v10's split (train/val/test)
  - real pair that is new in v11  -> 90% train / 5% val / 5% test
  - synthetic pairs               -> untouched

New pairs are assigned by page_id so all pairs from one manuscript page land
in the same split (prevents same-page leakage).

Usage:
  python resplit_real.py \
      --v10 v10_hf/train.csv v10_hf/val.csv v10_hf/test.csv \
      --v11 data/hf/v11_fresh/differing_pairs.csv \
      --out v11_resplit/differing_pairs.csv
"""
import argparse, hashlib, os
import pandas as pd

VAL_FRAC, TEST_FRAC = 0.05, 0.05
SEED = 42
# a pair is "real" if its category names a real OCR error type
REAL_MARKERS = ('real', 'syllable_level', 'single_char_substitution',
                'single_char_insert_delete')


def key(s, t):
    return hashlib.md5(f'{str(s)}\x00{str(t)}'.encode('utf-8')).hexdigest()


def is_real(cat):
    c = str(cat)
    return c.startswith('real') or c in REAL_MARKERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v10', nargs='+', required=True)
    ap.add_argument('--v11', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # ---- v10 assignments -------------------------------------------------
    frames = []
    for p in args.v10:
        d = pd.read_csv(p)
        if 'split' not in d.columns:
            b = os.path.basename(p).lower()
            d['split'] = ('train' if 'train' in b else
                          'val' if 'val' in b else
                          'test' if 'test' in b else None)
            if d['split'].isna().any():
                raise SystemExit(f'cannot infer split for {p}')
        frames.append(d[['source', 'target', 'split']])
    v10 = pd.concat(frames, ignore_index=True)
    v10['k'] = [key(s, t) for s, t in zip(v10.source, v10.target)]
    rank = {'test': 0, 'val': 1, 'train': 2}          # test wins ties
    v10['r'] = v10.split.map(rank)
    v10 = v10.sort_values('r').drop_duplicates('k', keep='first')
    v10_map = dict(zip(v10.k, v10.split))
    print(f'v10 pairs: {len(v10):,}  {v10.split.value_counts().to_dict()}')

    # ---- v11 -------------------------------------------------------------
    d = pd.read_csv(args.v11)
    d['k'] = [key(s, t) for s, t in zip(d.source, d.target)]
    d['_real'] = d.diff_category.map(is_real)
    print(f'\nv11 pairs: {len(d):,}   real: {d._real.sum():,}   '
          f'synthetic: {(~d._real).sum():,}')

    real = d[d._real].copy()
    real['inherited'] = real.k.map(v10_map)
    n_inh = real.inherited.notna().sum()
    print(f'  real inheriting v10 split: {n_inh:,}')
    print(f'  real that are new:         {len(real) - n_inh:,}')

    # new real pairs -> split by page, 90/5/5
    new = real[real.inherited.isna()]
    page_split = {}
    if len(new):
        pages = pd.Series(sorted(new.page_id.astype(str).unique())).sample(
            frac=1.0, random_state=SEED).tolist()
        n = len(pages)
        n_val = int(round(n * VAL_FRAC))
        n_test = int(round(n * TEST_FRAC))
        for i, pg in enumerate(pages):
            page_split[pg] = ('val' if i < n_val else
                              'test' if i < n_val + n_test else 'train')
        print(f'  new real pages: {n:,} -> '
              f'val {n_val}, test {n_test}, train {n - n_val - n_test}')

    new_split = real.page_id.astype(str).map(page_split)
    real['split'] = real.inherited.fillna(new_split)
    if real.split.isna().any():
        raise SystemExit('BUG: some real pairs unassigned')

    d.loc[real.index, 'split'] = real.split.values
    d = d.drop(columns=['k', '_real'])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    d.to_csv(args.out, index=False)

    print(f'\nwrote {args.out}')
    print('all splits: ', d.split.value_counts().to_dict())
    rr = d[d.diff_category.map(is_real)]
    print('real splits:', rr.split.value_counts().to_dict())

    # ---- verify ----------------------------------------------------------
    print('\nVERIFY (want 0)')
    chk = pd.read_csv(args.out)
    chk['k'] = [key(s, t) for s, t in zip(chk.source, chk.target)]
    chk_real = chk[chk.diff_category.map(is_real)]
    moved = sum(1 for k_, s in zip(chk_real.k, chk_real.split)
                if k_ in v10_map and v10_map[k_] != s)
    print(f'  real pairs disagreeing with v10: {moved}')
    span = (chk_real.groupby('page_id').split.nunique() > 1).sum()
    print(f'  real pages spanning splits:      {span}')
    tr = set(chk_real[chk_real.split == 'train'].k)
    te = set(chk_real[chk_real.split == 'test'].k)
    print(f'  real train∩test:                 {len(tr & te)}')


if __name__ == '__main__':
    main()