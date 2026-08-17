#!/usr/bin/env python3
"""
Generate synthetic Tibetan spelling-correction pairs from clean text.

Two corruption types:
  type 2  syllable -> valid homophone syllable   (Levenshtein bound 2)
  type 3  syllable -> unattested near-homophone  (IPA bound 1)

NOTE ON TYPE 3: bophono converts written -> spoken and only accepts valid
Tibetan, so it cannot score invented syllables. In practice only ~982
syllables survive the type 3 filter and almost none of them occur in real
corpus text (overlap of 3 out of 1,666 in a test run), so type 3 fires at
~0%. Run with --p_type3 0 until a written-perturbation rule set exists.

Output is a CSV of (source, target) pairs where the target is the
original clean sentence and the source is the corrupted version.

Usage:
    python generate_synthetic.py \
        --corpus ~/dharmaduta-ml/BoCorpus/bo_corpus.parquet \
        --syllables /tmp/syllables.txt \
        --out synthetic_pairs.csv \
        --p_type3 0 \
        --n 100000
"""

import argparse
import csv
import random
import re
import subprocess
import sys
from collections import defaultdict

import pandas as pd
import Levenshtein
from bophono import UnicodeToApi

SHAD = r'[།༎༏༐༑]'
TSHEG = '་'


# ---------------------------------------------------------------- helpers

def log(msg):
    print(msg, file=sys.stderr, flush=True)


def hunspell_invalid(items, dic='bo'):
    """Return the subset of `items` that hunspell rejects."""
    items = list(items)
    if not items:
        return set()
    proc = subprocess.run(
        ['hunspell', '-d', dic, '-l'],
        input='\n'.join(items) + '\n',
        capture_output=True, text=True,
    )
    return set(proc.stdout.split())


def syllables_of(sentence):
    return [s for s in sentence.replace('\n', TSHEG).split(TSHEG) if s]


# ---------------------------------------------------------------- indexes

def build_type2_index(syllable_file, conv, bound):
    """syllable -> list of valid homophones within `bound` edits."""
    groups = defaultdict(list)
    for line in open(syllable_file):
        s = line.strip()
        if not s:
            continue
        try:
            ipa = conv.get_api(s)
        except Exception:
            continue
        if ipa:
            groups[ipa].append(s)

    index = defaultdict(list)
    for members in groups.values():
        if len(members) < 2:
            continue
        for a in members:
            for b in members:
                if a != b and Levenshtein.distance(a, b) <= bound:
                    index[a].append(b)
    return index


def build_type3_index(syllable_file, conv, ipa_bound, require_same_onset):
    """syllable -> list of INVALID syllables that sound close.

    Largely non-functional: see the module docstring. Kept so the approach
    is documented and so a future rule-based version has somewhere to live.
    """
    syls = [l.strip() for l in open(syllable_file) if l.strip()]
    valid = set(syls)
    chars = sorted({c for s in syls for c in s})

    # generate single-character perturbations
    cands = {}
    for s in syls:
        try:
            ipa = conv.get_api(s)
        except Exception:
            continue
        if not ipa:
            continue
        for i in range(len(s)):
            for c in chars:
                if c == s[i]:
                    continue
                cand = s[:i] + c + s[i + 1:]
                if cand not in valid:
                    cands.setdefault(cand, (s, ipa))

    log(f'  {len(cands)} perturbation candidates, checking validity...')
    invalid = hunspell_invalid(cands.keys())

    index = defaultdict(list)
    for cand in cands:
        if cand not in invalid:
            continue
        src, src_ipa = cands[cand]
        try:
            cand_ipa = conv.get_api(cand)
        except Exception:
            continue
        if not cand_ipa:
            continue
        if Levenshtein.distance(cand_ipa, src_ipa) > ipa_bound:
            continue
        if require_same_onset and cand_ipa[:1] != src_ipa[:1]:
            continue
        index[src].append(cand)
    return index


# ---------------------------------------------------------------- corpus

def clean_sentences(parquet_path, min_syllables, need, rng):
    """Reservoir-sample `need` sentences uniformly across the WHOLE corpus.

    The previous version returned as soon as it had enough, which meant every
    sentence came from the first few documents. Reservoir sampling gives a
    uniform draw over all documents without holding 12M sentences in memory.
    """
    df = pd.read_parquet(parquet_path)
    reservoir = []
    seen = set()
    n_seen = 0

    for doc in df['text'].astype(str):
        for raw in re.split(SHAD, doc):
            s = raw.strip()
            if not s or s in seen:
                continue
            syls = syllables_of(s)
            if len(syls) < min_syllables:
                continue
            seen.add(s)
            n_seen += 1
            if len(reservoir) < need:
                reservoir.append((s, syls))
            else:
                j = rng.randrange(n_seen)
                if j < need:
                    reservoir[j] = (s, syls)

    log(f'  saw {n_seen} eligible sentences across the corpus')
    return reservoir


def filter_valid(sentences):
    """Drop sentences containing any syllable hunspell rejects."""
    all_syls = set()
    for _, syls in sentences:
        all_syls.update(syls)
    log(f'  checking {len(all_syls)} distinct syllables...')
    bad = hunspell_invalid(all_syls)
    out = [(s, syls) for s, syls in sentences
           if not any(x in bad for x in syls)]
    return out, bad


# ---------------------------------------------------------------- corrupt

def pick_error_count(rng, weights):
    r = rng.random()
    cum = 0.0
    for n, w in weights:
        cum += w
        if r < cum:
            return n
    return weights[-1][0]


def corrupt(syls, idx2, idx3, n_errors, p_type3, rng):
    """Return (corrupted_sentence, list_of_(from, to, type))."""
    positions = [
        i for i, s in enumerate(syls)
        if idx2.get(s) or idx3.get(s)
    ]
    if not positions:
        return None, []

    n = min(n_errors, len(positions))
    picks = rng.sample(positions, n)
    out = list(syls)
    changes = []

    for i in picks:
        s = out[i]
        use3 = rng.random() < p_type3 and bool(idx3.get(s))
        if use3:
            new, kind = rng.choice(idx3[s]), 3
        elif idx2.get(s):
            new, kind = rng.choice(idx2[s]), 2
        elif idx3.get(s):
            new, kind = rng.choice(idx3[s]), 3
        else:
            continue
        out[i] = new
        changes.append((s, new, kind))

    if not changes:
        return None, []
    return TSHEG.join(out), changes


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True)
    p.add_argument('--syllables', default='/tmp/syllables.txt')
    p.add_argument('--out', default='synthetic_pairs.csv')
    p.add_argument('--n', type=int, default=100000,
                   help='number of pairs to generate')
    p.add_argument('--min_syllables', type=int, default=8)
    p.add_argument('--type2_bound', type=int, default=2)
    p.add_argument('--type3_ipa_bound', type=int, default=1)
    p.add_argument('--p_type3', type=float, default=0.0,
                   help='share of substitutions using type 3 '
                        '(default 0 - type 3 is not currently reachable)')
    p.add_argument('--skip_type3_index', action='store_true',
                   help='do not build the type 3 index at all (faster)')
    p.add_argument('--require_same_onset', action='store_true',
                   help='type 3: keep the first IPA symbol unchanged')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sample_out', default='synthetic_sample.txt',
                   help='readable sample for manual review')
    args = p.parse_args()

    rng = random.Random(args.seed)
    conv = UnicodeToApi(schema='MST')

    log('building type 2 index (valid homophones)...')
    idx2 = build_type2_index(args.syllables, conv, args.type2_bound)
    log(f'  {len(idx2)} syllables have a valid homophone')

    if args.skip_type3_index or args.p_type3 <= 0:
        log('skipping type 3 index (p_type3 = 0)')
        idx3 = {}
    else:
        log('building type 3 index (unattested near-homophones)...')
        idx3 = build_type3_index(args.syllables, conv,
                                 args.type3_ipa_bound,
                                 args.require_same_onset)
        with open('/tmp/idx3_keys.txt', 'w') as f:
            f.write('\n'.join(sorted(idx3)) + '\n')
        log(f'  {len(idx3)} syllables have an unattested near-homophone')

    # read generously; validity filtering and corruption both lose some
    log('reading corpus (reservoir sampling)...')
    pool = clean_sentences(args.corpus, args.min_syllables,
                           args.n * 4, rng)
    log(f'  {len(pool)} candidate sentences')

    log('filtering to valid Tibetan...')
    pool, _ = filter_valid(pool)
    log(f'  {len(pool)} clean sentences')

    rng.shuffle(pool)

    weights = [(1, 0.6), (2, 0.3), (3, 0.1)]
    rows = []
    type_counts = defaultdict(int)
    count_hist = defaultdict(int)
    skipped = 0

    log('corrupting...')
    for sent, syls in pool:
        if len(rows) >= args.n:
            break
        n_err = pick_error_count(rng, weights)
        corrupted, changes = corrupt(syls, idx2, idx3, n_err,
                                     args.p_type3, rng)
        if not corrupted or corrupted == sent:
            skipped += 1
            continue
        rows.append({
            'source': corrupted,
            'target': sent,
            'n_errors': len(changes),
            'types': ','.join(str(k) for _, _, k in changes),
            'changes': ';'.join(f'{a}>{b}' for a, b, _ in changes),
        })
        count_hist[len(changes)] += 1
        for _, _, k in changes:
            type_counts[k] += 1

    if not rows:
        log('no pairs generated - check the syllable file and indexes')
        sys.exit(1)

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(args.sample_out, 'w') as f:
        for r in rng.sample(rows, min(60, len(rows))):
            f.write(f"[{r['changes']}]  types={r['types']}\n")
            f.write(f"  corr: {r['source']}\n")
            f.write(f"  orig: {r['target']}\n\n")

    total_sub = sum(type_counts.values()) or 1
    log('')
    log(f'wrote {len(rows)} pairs to {args.out}')
    log(f'skipped {skipped} sentences with no usable substitution')
    log(f'errors per sentence: {dict(sorted(count_hist.items()))}')
    log(f'substitutions: type2={type_counts[2]} '
        f'({type_counts[2] / total_sub:.1%}), '
        f'type3={type_counts[3]} ({type_counts[3] / total_sub:.1%})')
    log(f'sample for review: {args.sample_out}')


if __name__ == '__main__':
    main()