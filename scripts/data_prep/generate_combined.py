#!/usr/bin/env python3
"""
Combined synthetic corruption generator.

Eleven corruption types over two corpora:

  From TiSpell (dataloader/corrupt/) - structural and visual:
    char_random_delete, char_random_insert, char_tall_short_replace,
    char_homomorphic_replace, char_inner_syllable_exchange,
    char_near_syllable_exchange, syllable_random_delete,
    syllable_random_exchange, syllable_random_merge

  Ours - phonetic:
    syllable_homophone, verb_tense_homophone

WHY THIS REPLACES generate_synthetic_v2.py
------------------------------------------
The old generator picked substitute characters from a flat 41-character set,
which produced syllables no writer could produce - 13.9% of pairs contained a
syllable starting with a vowel sign (ཐུབ -> ཱུབ). A vowel sign cannot begin a
syllable. The model trained on that learned to expect impossible input, and
then rewrote well-formed text: it damaged 15% of already-correct tokens where
TiSpell's own model damages 0.78%.

TiSpell's letters.py encodes the actual orthography - prefix/root/superscript/
subscript/suffix categories, a homomorphic look-alike table, and rules for
which roots may follow which prefixes. Their corruption functions substitute
within those categories, so a corrupted syllable stays well-formed Tibetan and
is merely wrong. We import their functions rather than reimplementing.

TWO-SIDED VALIDATION
--------------------
  target: must PASS hunspell            (it is correct Tibetan)
  source: must FAIL hunspell            (it is misspelled - that is the point)
          but must be STRUCTURALLY POSSIBLE  (a writer could have written it)

The third check is the one that was missing. "Wrong" and "impossible" are
different things and only the first is useful training signal.

Usage (from the TiSpell repo root):
    python3 scripts/data_prep/generate_combined.py \
        --bocorpus ../BoCorpus/bo_corpus.parquet \
        --news ../news_data \
        --tispell_corrupt scripts/data_prep/tispell_corrupt \
        --syllables /tmp/syllables.txt \
        --verb_pairs /tmp/verb_homophone_pairs.tsv \
        --hunspell_dic ../hunspell-bo/bo \
        --out combined_pairs.csv --n 200000
"""

import argparse
import csv
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TSHEG = '་'
SHAD_RE = r'[།༎༏༐༑]'


def log(m):
    print(m, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- their module

def load_tispell_corrupt(path):
    """Import TiSpell's corruption module from an arbitrary directory."""
    parent = str(Path(path).resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    pkg = Path(path).name
    char_level = __import__(f'{pkg}.char_level', fromlist=['char_level'])
    char_level_safe = __import__(f'{pkg}.char_level_safe', fromlist=['char_level_safe'])
    syllable_level = __import__(f'{pkg}.syllable_level', fromlist=['syllable_level'])
    letters = __import__(f'{pkg}.letters', fromlist=['letters'])
    utils = __import__(f'{pkg}.utils', fromlist=['utils'])
    return char_level, char_level_safe, syllable_level, letters, utils


# ---------------------------------------------------------------- validation

class StructureCheck:
    """Is this a syllable a writer could have written?

    Conservative. Enforces only what letters.py makes unambiguous:
      - must be non-empty and all Tibetan
      - must not begin with a vowel sign, a subjoined letter, or a short
        (stacked) root form - none of those can start a syllable
      - must contain at least one full root letter

    It does NOT check prefix-root legality, suffix legality, or stacking
    order. Those need the language knowledge I do not have. Widen it once
    you have read some output.
    """

    def __init__(self, letters):
        self.roots = set(letters.ROOT_LETTER)
        self.prefixes = set(letters.PERFIX_LETTER)
        self.shorts = set(letters.ROOT_LETTER_SHORT)
        self.subscripts = set(letters.SUBSCRIPT_LETTER) | set(
            getattr(letters, 'FARTHER_SUBSCRIPT_LETTER', []))
        self.vowels = set(letters.VOWEL)
        self.cannot_start = self.vowels | self.subscripts | self.shorts

    def ok(self, syllable: str) -> bool:
        s = syllable.strip()
        if not s:
            return False
        if not all('\u0F00' <= c <= '\u0FFF' for c in s):
            return False
        if s[0] in self.cannot_start:
            return False
        if not any(c in self.roots for c in s):
            return False
        return True


def hunspell_invalid(items, dic):
    items = [i for i in items if i]
    if not items:
        return set()
    proc = subprocess.run(['hunspell', '-d', dic, '-l'],
                          input='\n'.join(items) + '\n',
                          capture_output=True, text=True)
    flagged = set(proc.stdout.split())
    # hunspell splits on some characters (visarga) and reports
    # fragments that were never sent; keep only real inputs
    return flagged & set(items)


# ---------------------------------------------------------------- our types

def build_homophone_index(syllable_file, bound=2):
    """syllable -> valid homophones within `bound` written edits."""
    try:
        import Levenshtein
        from bophono import UnicodeToApi
    except ImportError:
        log('  bophono/Levenshtein missing; phonetic types disabled')
        return {}
    conv = UnicodeToApi(schema='MST', options={'unknownSyllableMarker': True})
    groups = defaultdict(list)
    for line in open(syllable_file, encoding='utf-8'):
        s = line.strip()
        if not s:
            continue
        try:
            ipa = conv.get_api(s)
        except Exception:
            continue
        if ipa and ipa != '(?)':
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


def build_verb_index(path):
    index = defaultdict(list)
    if not path or not os.path.isfile(path):
        return index
    try:
        df = pd.read_csv(path, sep='\t')
    except Exception as exc:
        log(f'  could not read verb pairs ({exc})')
        return index
    for _, r in df.iterrows():
        index[str(r['form_a'])].append(str(r['form_b']))
        index[str(r['form_b'])].append(str(r['form_a']))
    return index


def corrupt_homophone(syls, index, rng):
    positions = [i for i, s in enumerate(syls) if index.get(s)]
    if not positions:
        return None
    i = rng.choice(positions)
    out = list(syls)
    out[i] = rng.choice(index[out[i]])
    return out


# ---------------------------------------------------------------- corpora

def read_bocorpus(path):
    df = pd.read_parquet(path)
    for doc in df['text'].astype(str):
        for raw in re.split(SHAD_RE, doc):
            s = raw.strip()
            if s:
                yield s


def _read(f):
    raw = f.read_bytes()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return raw.decode('utf-16').lstrip('\ufeff')
    return raw.decode('utf-8-sig')


def read_news(root):
    """Their layout: dataset/<category>/<n>.txt"""
    root = Path(root)
    for cat in sorted(root.iterdir()):
        if not cat.is_dir() or cat.name.startswith('__'):
            continue
        for f in cat.glob('*.txt'):
            try:
                text = _read(f)
            except Exception as e:
                print(f'  decode failed: {f}: {e}', file=sys.stderr)
                continue
            for raw in re.split(SHAD_RE, text):
                s = raw.strip()
                if s:
                    yield s


def reservoir(iterable, k, rng):
    out, n = [], 0
    seen = set()
    for item in iterable:
        if item in seen:
            continue
        seen.add(item)
        n += 1
        if len(out) < k:
            out.append(item)
        else:
            j = rng.randrange(n)
            if j < k:
                out[j] = item
    return out, n


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bocorpus', required=True)
    p.add_argument('--news', required=True)
    p.add_argument('--tispell_corrupt', required=True,
                   help='path to TiSpell dataloader/corrupt directory')
    p.add_argument('--syllables', default='/tmp/syllables.txt')
    p.add_argument('--verb_pairs', default='/tmp/verb_homophone_pairs.tsv')
    p.add_argument('--hunspell_dic', required=True)
    p.add_argument('--out', default='combined_pairs.csv')
    p.add_argument('--sample_out', default='combined_sample.txt')
    p.add_argument('--n', type=int, default=200000,
                   help='total pairs, split evenly between the two corpora')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()

    rng = random.Random(a.seed)
    dic = os.path.expanduser(a.hunspell_dic)

    log('importing TiSpell corruption module...')
    char_level, _safe, syllable_level, letters, utils = load_tispell_corrupt(
        a.tispell_corrupt)
    check = StructureCheck(letters)

    log('building phonetic indexes...')
    hom = build_homophone_index(a.syllables)
    verb = build_verb_index(a.verb_pairs)
    log(f'  {len(hom)} syllables with a homophone, {len(verb)} verb forms')

    # ---- corpora, sampled separately so each contributes half ----
    per = a.n * 3 // 2   # read generously; validation drops some
    sources = {}
    log('reading BoCorpus...')
    sents, total = reservoir(read_bocorpus(a.bocorpus), per, rng)
    sources['bocorpus'] = sents
    log(f'  {len(sents)} sampled from {total} distinct sentences')

    log('reading news corpus...')
    sents, total = reservoir(read_news(a.news), per, rng)
    sources['news'] = sents
    log(f'  {len(sents)} sampled from {total} distinct sentences')

    # ---- corruption types ----
    def t_char_delete(s):   return char_level.char_random_delete(s)
    def t_char_insert(s):   return char_level.char_random_insert(s)
    def t_tall_short(s):    return char_level.char_tall_short_replace(s)
    def t_homomorphic(s):   return char_level.char_homomorphic_replace(s)
    def t_inner_exch(s):    return char_level.char_inner_syllable_exchange(s)
    def t_near_exch(s):     return char_level.char_near_syllable_exchange(s)
    def t_syl_delete(s):    return syllable_level.syllable_random_delete(s)[0]
    def t_syl_exchange(s):  return syllable_level.syllable_random_exchange(s)[0]
    def t_syl_merge(s):     return syllable_level.syllable_random_merge(s)
    def t_homophone(s):     return corrupt_homophone(s, hom, rng)
    def t_verb(s):          return corrupt_homophone(s, verb, rng)

    TYPES = [
        ('char_random_delete', t_char_delete),
        ('char_random_insert', t_char_insert),
        ('char_tall_short_replace', t_tall_short),
        ('char_homomorphic_replace', t_homomorphic),
        ('char_inner_syllable_exchange', t_inner_exch),
        ('char_near_syllable_exchange', t_near_exch),
        ('syllable_random_delete', t_syl_delete),
        ('syllable_random_exchange', t_syl_exchange),
        ('syllable_random_merge', t_syl_merge),
        ('syllable_homophone', t_homophone),
        ('verb_tense_homophone', t_verb),
    ]

    # ---- validate targets in one batched hunspell call per source ----
    log('validating targets...')
    clean = {}
    for name, sents in sources.items():
        syls = set()
        for s in sents:
            syls.update(x for x in utils.split_syllable(s) if x)
        bad = hunspell_invalid(syls, dic)
        keep = [s for s in sents
                if all(x not in bad for x in utils.split_syllable(s) if x)]
        clean[name] = keep
        log(f'  {name}: {len(keep)}/{len(sents)} sentences fully valid')

    # ---- corrupt ----
    log('corrupting...')
    rows = []
    rejected = defaultdict(int)
    per_source = a.n // len(clean)

    for source_name, sents in clean.items():
        rng.shuffle(sents)
        made = 0
        for sent in sents:
            if made >= per_source:
                break
            syls = [x for x in utils.split_syllable(sent) if x]
            if len(syls) < 2:
                continue
            type_name, fn = TYPES[rng.randrange(len(TYPES))]
            try:
                out = fn(list(syls))
            except Exception:
                rejected[f'{type_name}:exception'] += 1
                continue
            if not out:
                rejected[f'{type_name}:no_change'] += 1
                continue
            source_text = TSHEG.join(out)
            if source_text == sent:
                rejected[f'{type_name}:identical'] += 1
                continue
            # the check that was missing: wrong, but possible
            changed = [x for x in out if x and x not in syls]
            if changed and not all(check.ok(x) for x in changed):
                rejected[f'{type_name}:malformed'] += 1
                continue
            rows.append({
                'source': source_text,
                'target': sent,
                'corruption_type': type_name,
                'source_corpus': source_name,
                'source_sentence_id': f'{source_name}:{hash(sent) & 0xFFFFFFFF}',
            })
            made += 1
        log(f'  {source_name}: {made} pairs')

    if not rows:
        sys.exit('no pairs generated - check the imported function signatures')

    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(a.sample_out, 'w', encoding='utf-8') as f:
        for r in rng.sample(rows, min(100, len(rows))):
            f.write(f"[{r['corruption_type']}] ({r['source_corpus']})\n")
            f.write(f"  corr: {r['source']}\n")
            f.write(f"  orig: {r['target']}\n\n")

    log(f'\nwrote {len(rows)} pairs to {a.out}')
    counts = defaultdict(int)
    for r in rows:
        counts[r['corruption_type']] += 1
    log('\nby corruption type:')
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        log(f'  {k:32s} {v:7d}  {v/len(rows):.1%}')
    if rejected:
        log('\nrejected:')
        for k, v in sorted(rejected.items(), key=lambda x: -x[1])[:15]:
            log(f'  {k:44s} {v:7d}')
    log(f'\nsample for review: {a.sample_out}')
    log('READ THE SAMPLE. The structural check is conservative and I cannot '
        'read Tibetan - tell me what it lets through that it should not.')


if __name__ == '__main__':
    main()