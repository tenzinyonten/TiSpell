"""
Mine the real error distribution from the #407 annotator/reviewer pairs.

Why: the synthetic corruption script needs to know what errors actually occur.
Guessing produces a model that fixes errors nobody makes. This measures the real
distribution instead, so corruption frequencies can be matched to it.

It also answers a design question directly: are the real errors PHONOLOGICAL
(sound-alike confusions, what someone writing by ear produces) or VISUAL
(misread stacks, dropped vowel signs, what OCR produces)? The corruption types
currently proposed are phonological; if the real errors are mostly visual, the
generated data will not transfer.

Output:
  - the most common character-level substitutions, with counts
  - the most common syllable-level substitutions
  - a breakdown by diff_category
  - a rough phonological-vs-visual split based on whether the confused
    characters share a phonetic class or a visual/structural one
  - real_error_distribution.csv, for feeding into the corruption script

Usage:
    pip install pandas
    python mine_error_distribution.py
"""

import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import pandas as pd

BASE = 'dataset/ocr_annotation/exports/hf_differing/'

TSHEG = '\u0f0b'
DELIM = re.compile(r'[\u0f0b\u0f0c\u0f0d\u0f0e\u0f0f\u0f10\u0f11\u0f12\u0f14\s]+')

BASE_CONSONANTS = set(chr(c) for c in range(0x0F40, 0x0F6C))
VOWEL_SIGNS = {'\u0f72', '\u0f74', '\u0f7a', '\u0f7b', '\u0f7c', '\u0f7d'}
SUBJOINED = set(chr(c) for c in range(0x0F90, 0x0FBD))

# Rough phonetic groupings. Tibetan consonants come in series that share a place
# of articulation and differ in aspiration/voicing -- these are the pairs a
# writer confuses BY EAR.
PHONETIC_SERIES = [
    set('\u0f40\u0f41\u0f42\u0f43'),          # ka kha ga gha
    set('\u0f44'),                             # nga
    set('\u0f45\u0f46\u0f47'),                # ca cha ja
    set('\u0f49'),                             # nya
    set('\u0f4f\u0f50\u0f51\u0f52'),          # ta tha da dha
    set('\u0f53'),                             # na
    set('\u0f54\u0f55\u0f56\u0f57'),          # pa pha ba bha
    set('\u0f58'),                             # ma
    set('\u0f59\u0f5a\u0f5b'),                # tsa tsha dza
    set('\u0f5d'),                             # wa
    set('\u0f5e\u0f5f'),                      # zha za
    set('\u0f60'),                             # 'a
    set('\u0f61'),                             # ya
    set('\u0f62'),                             # ra
    set('\u0f63'),                             # la
    set('\u0f64\u0f65\u0f66'),                # sha ssa sa
    set('\u0f67'),                             # ha
    set('\u0f68'),                             # a
]

# Characters that look alike or sit in similar structural positions -- the pairs
# OCR confuses BY SIGHT.
VISUAL_PAIRS = [
    ('\u0f42', '\u0f44'),   # ga / nga
    ('\u0f51', '\u0f53'),   # da / na
    ('\u0f54', '\u0f55'),   # pa / pha
    ('\u0f56', '\u0f58'),   # ba / ma
    ('\u0f5a', '\u0f5b'),   # tsha / dza
    ('\u0f63', '\u0f64'),   # la / sha
    ('\u0f66', '\u0f67'),   # sa / ha
    ('\u0f72', '\u0f74'),   # i / u  (above vs below the base)
    ('\u0f7a', '\u0f7c'),   # e / o
    ('\u0f7e', '\u0f7f'),   # anusvara / visarga
]
VISUAL_SET = set()
for a, b in VISUAL_PAIRS:
    VISUAL_SET.add(frozenset((a, b)))


def same_phonetic_series(a: str, b: str) -> bool:
    for series in PHONETIC_SERIES:
        if a in series and b in series:
            return True
    return False


def visually_confusable(a: str, b: str) -> bool:
    return frozenset((a, b)) in VISUAL_SET


def char_class(ch: str) -> str:
    if ch in BASE_CONSONANTS:
        return 'consonant'
    if ch in VOWEL_SIGNS:
        return 'vowel'
    if ch in SUBJOINED:
        return 'subjoined'
    if ch == TSHEG:
        return 'tsheg'
    return 'other'


def align_ops(src: str, tgt: str):
    """Yield (op, src_fragment, tgt_fragment) for the differences."""
    sm = SequenceMatcher(None, src, tgt, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != 'equal':
            yield op, src[i1:i2], tgt[j1:j2]


def main():
    frames = []
    for split in ('train', 'validation', 'test'):
        try:
            df = pd.read_csv(BASE + split + '.csv')
            df['split'] = split
            frames.append(df)
        except Exception as e:
            print(f"Skipping {split}: {e}")

    if not frames:
        sys.exit("No CSVs found. Run from the TiSpell repo root.")

    df = pd.concat(frames, ignore_index=True)
    df['source'] = df['source'].fillna('').astype(str)
    df['target'] = df['target'].fillna('').astype(str)
    if 'diff_category' not in df.columns:
        df['diff_category'] = 'unspecified'

    print(f"Analysing {len(df):,} pairs\n")

    char_subs = Counter()
    char_subs_by_cat = defaultdict(Counter)
    syl_subs = Counter()
    op_counts = Counter()
    class_transitions = Counter()

    for src, tgt, cat in zip(df['source'], df['target'], df['diff_category']):
        # Character level
        for op, s_frag, t_frag in align_ops(src, tgt):
            op_counts[op] += 1
            if op == 'replace' and len(s_frag) == 1 and len(t_frag) == 1:
                char_subs[(s_frag, t_frag)] += 1
                char_subs_by_cat[cat][(s_frag, t_frag)] += 1
                class_transitions[(char_class(s_frag), char_class(t_frag))] += 1
            elif op == 'delete' and len(s_frag) == 1:
                class_transitions[(char_class(s_frag), 'DELETED')] += 1
            elif op == 'insert' and len(t_frag) == 1:
                class_transitions[('INSERTED', char_class(t_frag))] += 1

        # Syllable level
        s_syls = [s for s in DELIM.split(src) if s]
        t_syls = [s for s in DELIM.split(tgt) if s]
        sm = SequenceMatcher(None, s_syls, t_syls, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == 'replace' and (i2 - i1) == 1 and (j2 - j1) == 1:
                syl_subs[(s_syls[i1], t_syls[j1])] += 1

    # ---- Report -------------------------------------------------------------
    print("=" * 72)
    print("  EDIT OPERATIONS")
    print("=" * 72)
    total_ops = sum(op_counts.values())
    for op, n in op_counts.most_common():
        print(f"  {op:<10}: {n:>8,}  ({n / total_ops:.1%})")

    print("\n" + "=" * 72)
    print("  TOP 30 CHARACTER SUBSTITUTIONS")
    print("=" * 72)
    print(f"  {'from':<6} {'to':<6} {'count':>8}   phonetic?  visual?")
    print("  " + "-" * 46)
    for (a, b), n in char_subs.most_common(30):
        ph = 'yes' if same_phonetic_series(a, b) else '-'
        vi = 'yes' if visually_confusable(a, b) else '-'
        print(f"  {a:<6} {b:<6} {n:>8,}   {ph:<10} {vi}")

    # ---- The design question ------------------------------------------------
    total_sub = sum(char_subs.values())
    phon = sum(n for (a, b), n in char_subs.items() if same_phonetic_series(a, b))
    vis = sum(n for (a, b), n in char_subs.items() if visually_confusable(a, b))
    both = sum(n for (a, b), n in char_subs.items()
               if same_phonetic_series(a, b) and visually_confusable(a, b))

    print("\n" + "=" * 72)
    print("  PHONOLOGICAL vs VISUAL")
    print("=" * 72)
    print(f"""
  single-character substitutions : {total_sub:,}
    same phonetic series          : {phon:,}  ({phon / max(1, total_sub):.1%})
    visually confusable           : {vis:,}  ({vis / max(1, total_sub):.1%})
    both                          : {both:,}  ({both / max(1, total_sub):.1%})
    neither                       : {total_sub - phon - vis + both:,}

  If the phonetic share dominates, the proposed homophone-based corruptions
  match the real distribution. If the visual share dominates, the generated
  data will not resemble the errors this corpus actually contains, and visual
  confusion pairs should be added.

  Caveat: these groupings are approximate, and many pairs are both phonetically
  and visually close. Read the top-30 table above and judge it yourself rather
  than relying on the percentages alone.
""")

    print("=" * 72)
    print("  WHAT GETS CONFUSED WITH WHAT (character classes)")
    print("=" * 72)
    for (a, b), n in class_transitions.most_common(15):
        print(f"  {a:<12} -> {b:<12} {n:>8,}")

    print("\n" + "=" * 72)
    print("  TOP 25 SYLLABLE SUBSTITUTIONS")
    print("=" * 72)
    for (a, b), n in syl_subs.most_common(25):
        print(f"  {a:<16} -> {b:<16} {n:>7,}")

    # ---- Export -------------------------------------------------------------
    rows = [{
        'from': a, 'to': b, 'count': n,
        'from_codepoint': f'U+{ord(a):04X}' if len(a) == 1 else '',
        'to_codepoint': f'U+{ord(b):04X}' if len(b) == 1 else '',
        'same_phonetic_series': same_phonetic_series(a, b),
        'visually_confusable': visually_confusable(a, b),
        'frequency': n / max(1, total_sub),
    } for (a, b), n in char_subs.most_common()]

    pd.DataFrame(rows).to_csv('real_error_distribution.csv', index=False)

    syl_rows = [{'from': a, 'to': b, 'count': n}
                for (a, b), n in syl_subs.most_common()]
    pd.DataFrame(syl_rows).to_csv('real_syllable_substitutions.csv', index=False)

    print(f"""

Wrote:
  real_error_distribution.csv        {len(rows):,} character substitutions
  real_syllable_substitutions.csv    {len(syl_rows):,} syllable substitutions

Use the `frequency` column to weight the corruption script, so the synthetic
errors occur at the rates actually observed rather than uniformly.

Note the ཌ/གས and ཾ/མ encoding artifact will show up prominently here. Decide
whether to normalise it out first -- corrupting text to reproduce an encoding
artifact is probably not what you want.
""")


if __name__ == "__main__":
    main()