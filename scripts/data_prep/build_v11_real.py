#!/usr/bin/env python3
"""
Build the v11 real-pair dataset:
  - 26,931 Gemini-clean differing pairs (from exports_v2, artifact-aware)
  - 9,000 identical pairs, filtered to target length 10-200 chars, randomly picked
Combine into one file for the final Gemini verification pass.

Run from repo root:  python build_v11_real.py
"""
import pandas as pd

EXPORTS = 'dataset/ocr_annotation/exports_v2'
VERDICTS = 'dataset/ocr_annotation/exports/gemini_target_check_v1_noconventions.csv'
SEED = 42
N_IDENTICAL = 9000
MIN_CHARS, MAX_CHARS = 10, 200

TSHEG = '\u0f0b'; DDA = '\u0f4c'; GA_SA = '\u0f42\u0f66'
ANU = '\u0f7e'; MA = '\u0f58'


def is_tibetan(t):
    return all('\u0f00' <= c <= '\u0fff' or c.isspace() for c in str(t))


def main():
    # ---- DIFFERING: 26,931 Gemini-clean (artifact-aware) --------------------
    d = pd.read_csv(f'{EXPORTS}/differing_pairs.csv')
    d = d[~d.diff_category.isin(['large_rewrite', 'multi_edit', 'punctuation_only'])].copy()

    g = pd.read_csv(VERDICTS)
    incorrect = set(g.loc[g.verdict == 'incorrect', 'target'])

    # anusvara whitelist from corpus (only ཾ->མ that also occur spelled with མ)
    blob = ' '.join(list(d.source.astype(str)) + list(d.target.astype(str)))
    syls = set(blob.replace(TSHEG, ' ').split())
    anu_map = {s: s.replace(ANU, MA) for s in syls
               if ANU in s and s.replace(ANU, MA) in syls}

    def norm(t):
        t = str(t)
        return TSHEG.join(anu_map.get(s, s).replace(DDA, GA_SA)
                          for s in t.split(TSHEG))

    # real error = flagged AND normalizing does NOT change it (artifact-only kept)
    d['real_error'] = d.target.isin(incorrect) & d.target.map(lambda t: norm(t) == str(t))
    differing = d[~d.real_error].drop(columns=['real_error']).copy()
    print(f'clean differing pairs: {len(differing)}')

    # ---- IDENTICAL: length 10-200 on target, then random 9,000 --------------
    i = pd.read_csv(f'{EXPORTS}/identical_pairs.csv')
    print(f'identical total: {len(i)}')

    # basic Tibetan cleaning
    i = i[i.target.apply(is_tibetan)]
    i = i[~i.target.astype(str).str.contains(r'[\u0f20-\u0f29]')]  # no digits

    # LENGTH FILTER on identical (10-200 chars on target)
    tlen = i.target.astype(str).str.len()
    i = i[(tlen >= MIN_CHARS) & (tlen <= MAX_CHARS)]
    print(f'identical after length {MIN_CHARS}-{MAX_CHARS}: {len(i)}')

    # random pick 9,000
    if len(i) >= N_IDENTICAL:
        identical = i.sample(n=N_IDENTICAL, random_state=SEED)
    else:
        identical = i
        print(f'WARNING: only {len(i)} identical available, wanted {N_IDENTICAL}')
    print(f'identical picked: {len(identical)}')

    # ---- COMBINE ------------------------------------------------------------
    # tag which is which so downstream can tell them apart
    differing['pair_type'] = 'differing'
    identical['pair_type'] = 'identical'
    combined = pd.concat([differing, identical], ignore_index=True)

    out = f'{EXPORTS}/v11_real_combined.csv'
    combined.to_csv(out, index=False)
    print(f'\nwrote {len(combined)} pairs to {out}')
    print(f'  differing: {(combined.pair_type=="differing").sum()}')
    print(f'  identical: {(combined.pair_type=="identical").sum()}')
    print('\nNext: run Gemini verification on this file, then it is the final v11 real set.')


if __name__ == '__main__':
    main()