#!/usr/bin/env python3
"""
Merge the real OCR annotator pairs into the synthetic dataset.

Filters on the real side, in order:
  1. drop large_rewrite and multi_edit - edit distance runs to 161 there,
     which is rewriting rather than spelling correction
  2. drop targets containing non-Tibetan characters
  3. drop targets containing mantra marks (visarga, anusvara) - Sanskrit
     ritual text, excluded from the synthetic side too
  4. drop targets failing StructureCheck - the annotators transcribe the
     manuscript faithfully, so some "correct" readings are shapes no writer
     could produce (e.g. stacked forms that cannot occur). This replaces the
     has_brackets flag, which on inspection contains no bracket characters
     and whose meaning is undocumented.
  5. drop targets with a syllable hunspell does not know

Steps 2-5 are the same guarantees the synthetic targets get, so all three
sources are clean by the same rule.

page_id becomes doc_id so the split groups by manuscript page: pairs from
one page share scribe and scanning artifacts.

Usage:
    python3 merge_real_v2.py \
        --synthetic data/synthetic/combined_v3.csv \
        --real TiSpell/dataset/ocr_annotation/exports/control/differing_pairs.csv \
        --tispell_corrupt tispell_corrupt \
        --hunspell_dic hunspell-bo/bo \
        --n-real 10000 \
        --out data/final/combined_all_v4.csv
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_combined import (TSHEG, StructureCheck, hunspell_invalid,
                               load_tispell_corrupt)

DROP_CATEGORIES = ('large_rewrite', 'multi_edit', 'punctuation_only')
MANTRA_MARKS = 'ཿཾ'
# U+0FBE illegible-text placeholder, U+0F1D repetition mark,
# U+0F3C/U+0F3D Tibetan brackets - annotator markup, not text
PLACEHOLDER_MARKS = '\u0fbe\u0f1d\u0f3c\u0f3d'


def is_tibetan(text):
    return all('\u0f00' <= c <= '\u0fff' or c.isspace() for c in str(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synthetic', required=True)
    ap.add_argument('--real', required=True)
    ap.add_argument('--tispell_corrupt',
                    default='scripts/data_prep/tispell_corrupt')
    ap.add_argument('--hunspell_dic', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-real', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    _, _, _, letters, _ = load_tispell_corrupt(args.tispell_corrupt)
    check = StructureCheck(letters)

    syn = pd.read_csv(args.synthetic)
    print(f'synthetic: {len(syn)} pairs')

    real = pd.read_csv(args.real)
    n0 = len(real)
    print(f'real: {n0} pairs')

    real = real.dropna(subset=['source', 'target'])
    real = real[real.source.astype(str) != real.target.astype(str)]

    real = real[~real.diff_category.isin(DROP_CATEGORIES)]
    print(f'  after dropping {"/".join(DROP_CATEGORIES)}: {len(real)}')

    real = real[real.target.apply(is_tibetan) & real.source.apply(is_tibetan)]
    print(f'  after non-Tibetan characters: {len(real)}')

    real = real[~real.target.astype(str).str.contains(f'[{MANTRA_MARKS}]')]
    print(f'  after mantra marks: {len(real)}')
    has_num = (real.target.astype(str).str.contains(r'[\u0f20-\u0f29]') |
               real.source.astype(str).str.contains(r'[\u0f20-\u0f29]'))
    real = real[~has_num]
    print(f'  after Tibetan digits: {len(real)}')

    has_ph = (real.target.astype(str).str.contains(f'[{PLACEHOLDER_MARKS}]') |
              real.source.astype(str).str.contains(f'[{PLACEHOLDER_MARKS}]'))
    real = real[~has_ph]
    print(f'  after placeholder marks: {len(real)}')

    ok = real.target.astype(str).apply(
        lambda t: all(check.ok(s) for s in t.split(TSHEG) if s))
    real = real[ok]
    print(f'  after StructureCheck on targets: {len(real)}')

    syls = sorted({s for t in real.target.astype(str)
                   for s in t.split(TSHEG) if s})
    bad = set(hunspell_invalid(syls, args.hunspell_dic)) & set(syls)
    real = real[real.target.astype(str).apply(
        lambda t: not any(s in bad for s in t.split(TSHEG) if s))]
    print(f'  after hunspell on targets: {len(real)}  '
          f'({100 * len(real) / n0:.0f}% of original)')

    if args.n_real and len(real) > args.n_real:
        real = real.sample(n=args.n_real, random_state=args.seed)
        print(f'  sampled down to {len(real)}')
    elif args.n_real:
        print(f'  WARNING: only {len(real)} available, wanted {args.n_real}')

    out = pd.DataFrame({
        'source': real.source.astype(str),
        'target': real.target.astype(str),
        'corruption_types': real.diff_category.astype(str),
        'n_corruptions': real.edit_distance.fillna(0).astype(int),
        'source_corpus': 'real',
        'doc_id': real.page_id.astype(str),
        'source_sentence_id': [
            'real:' + hashlib.md5(f'{p}:{i}'.encode()).hexdigest()[:12]
            for p, i in zip(real.page_id, real.segment_idx)],
    })

    combined = pd.concat([syn, out], ignore_index=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    print(f'\nwrote {len(combined)} pairs to {out_path}')
    print('\nby source:')
    for name, sub in combined.groupby('source_corpus'):
        lens = sorted(sub.target.astype(str).str.len())
        print(f'  {name:10s} {len(sub):6d}  '
              f'median target {lens[len(lens) // 2]} chars')
    print('\nreal pairs by category:')
    print(combined[combined.source_corpus == 'real']
          .corruption_types.value_counts().to_string())
    print(f'\ndistinct pages in real: '
          f'{combined[combined.source_corpus == "real"].doc_id.nunique()}')


if __name__ == '__main__':
    main()