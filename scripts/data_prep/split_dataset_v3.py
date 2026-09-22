#!/usr/bin/env python3
"""
Split the combined dataset into train/val/test.

Adds two columns:
  density      - corruptions per character of the target. Normalises for
                 length, which is what makes the three sources comparable:
                 real pairs are shorter and denser than news, so a raw
                 per-sentence count hides the difference.
  density_bin  - low/mid/high terciles, since stratification needs a
                 discrete key.

Two rules for the split itself:
  - group by document: all pairs from one source sentence (synthetic) or one
    manuscript page (real) land in the same split, so the model never sees a
    test answer during training and cannot learn page-specific artifacts.
  - stratify on source x density_bin x first error type, so each split
    carries the same mix. Cells with too few groups fall back to source
    alone, since stratification fails on classes smaller than the split count.

Also reports the no-op baseline per source, using TiSpell's own metric so
the numbers are directly comparable to what train.py prints.

Usage:
    python3 split_dataset_v3.py \
        --input data/final/combined_all_v3.csv \
        --outdir data/final/v3 \
        --train 0.90 --val 0.05 --test 0.05
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metrics import compute_precision_recall_f1

MIN_PER_CELL = 20   # below this, a stratification cell collapses to source


def syllables(text):
    return [s for s in re.split(r'[་༌]', str(text).replace(' ', '')) if s]


def noop_f1(sources, targets):
    """Score you get by changing nothing, in train.py's metric."""
    tot = 0.0
    for s, t in zip(sources, targets):
        _, _, f1 = compute_precision_recall_f1(syllables(s), syllables(t))
        tot += f1
    return tot / len(sources) if len(sources) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--train', type=float, default=0.90)
    ap.add_argument('--val', type=float, default=0.05)
    ap.add_argument('--test', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6

    d = pd.read_csv(args.input)
    print(f'{len(d)} pairs')

    # ---------------------------------------------------------- new columns
    tgt_len = d.target.astype(str).str.len().clip(lower=1)
    d['density'] = (d.n_corruptions.fillna(0) / tgt_len).round(5)
    d['error_type'] = (d.corruption_types.fillna('clean').astype(str)
                       .str.split('|').str[0])

    # terciles over the corrupted rows only; clean rows are their own bin,
    # otherwise a third of the "low" bucket is just zeros
    nonzero = d.density > 0
    d['density_bin'] = 'clean'
    d.loc[nonzero, 'density_bin'] = pd.qcut(
        d.loc[nonzero, 'density'], 3, labels=['low', 'mid', 'high'])

    print('\ndensity by source (corruptions per character):')
    for name, sub in d.groupby('source_corpus'):
        nz = sub[sub.density > 0].density
        print(f'  {name:10s} median {nz.median():.4f}   '
              f'mean {nz.mean():.4f}   n={len(sub)}')
    print('\ndensity_bin counts:', d.density_bin.value_counts().to_dict())

    # ---------------------------------------------------------- grouping
    d['group'] = d.apply(
        lambda r: f'real:{r.doc_id}' if r.source_corpus == 'real'
        else r.source_sentence_id, axis=1)
    print(f'\n{d.group.nunique()} groups')

    # One row per group. A group can span several error types and bins, so
    # take the first of each - the split is at group granularity regardless.
    groups = (d.groupby('group')
              .agg(source_corpus=('source_corpus', 'first'),
                   density_bin=('density_bin', 'first'),
                   error_type=('error_type', 'first'))
              .reset_index())

    key = (groups.source_corpus.astype(str) + '|' +
           groups.density_bin.astype(str) + '|' +
           groups.error_type.astype(str))
    counts = key.value_counts()
    thin = set(counts[counts < MIN_PER_CELL].index)
    if thin:
        print(f'{len(thin)} stratification cells under {MIN_PER_CELL} groups; '
              f'those fall back to source alone')
    from collections import Counter
    strat = [k if k not in thin else k.split('|')[0] for k in key]
    frac_tmp = args.val + args.test
    sc = Counter(strat)
    min_safe = max(4, int(2 / max(frac_tmp * 0.5, 1e-6)))
    groups['strat'] = [x if sc[x] >= min_safe else 'ALL' for x in strat]
    print(f'{groups.strat.nunique()} stratification cells')

    tr_g, tmp_g = train_test_split(
        groups, test_size=(args.val + args.test),
        stratify=groups.strat, random_state=args.seed)
    val_g, test_g = train_test_split(
        tmp_g, test_size=args.test / (args.val + args.test),
        random_state=args.seed)

    splits = {
        'train': d[d.group.isin(tr_g.group)],
        'val': d[d.group.isin(val_g.group)],
        'test': d[d.group.isin(test_g.group)],
    }

    # ---------------------------------------------------------- invariants
    seen = {}
    for name, sub in splits.items():
        for g in sub.group.unique():
            assert g not in seen, f'group {g} in {seen[g]} and {name}'
            seen[g] = name
    assert sum(len(s) for s in splits.values()) == len(d), 'lost rows'
    print('no group appears in more than one split\n')

    # ---------------------------------------------------------- write
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, sub in splits.items():
        sub.drop(columns=['group']).to_csv(outdir / f'{name}.csv', index=False)
        by_src = '  '.join(f'{k} {v}' for k, v in
                           sub.source_corpus.value_counts().items())
        print(f'{name:6s} {len(sub):6d}  {by_src}')

    # ---------------------------------------------------------- baseline
    test = splits['test']
    print('\nno-op baseline on test (TiSpell metric, weighted syllable F1):')
    for name, sub in test.groupby('source_corpus'):
        print(f'  {name:10s} f1={noop_f1(list(sub.source), list(sub.target)):.4f}'
              f'   ({len(sub)} pairs)')
    print(f'  {"overall":10s} '
          f'f1={noop_f1(list(test.source), list(test.target)):.4f}')

    # ---------------------------------------------------------- test mix
    print('\ntest set by source and density bin:')
    print(pd.crosstab(test.source_corpus, test.density_bin).to_string())
    print('\ntest set by source and error type (thin cells flagged):')
    ct = pd.crosstab(test.source_corpus, test.error_type)
    print(ct.to_string())
    thin_cells = [(r, c) for r in ct.index for c in ct.columns
                  if 0 < ct.loc[r, c] < 30]
    for r, c in thin_cells:
        print(f'  thin: {r} / {c} = {ct.loc[r, c]}')


if __name__ == '__main__':
    main()
    