"""
Target-side spelling and orthography check.

Question this answers: are the reviewer's corrections (the `target` column)
actually valid Tibetan? If reviewers made mistakes, the model is being trained
toward wrong answers.

Runs on the TARGET column only. Source errors are expected — that's the point of
the dataset. What matters is whether the "right answer" is right.

Sanskrit transliteration (mantras, dharanis) legitimately breaks native Tibetan
syllable rules, so those syllables are detected and skipped rather than counted
as errors.

Usage:
    pip install pandas tqdm botok
    python check_target_spelling.py
"""

import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

# ==============================================================================
# TIBETAN ORTHOGRAPHY
# ==============================================================================

# The 30 consonants, U+0F40 – U+0F6C
BASE_CONSONANTS = set(chr(c) for c in range(0x0F40, 0x0F6C))

# Vowel signs
VOWEL_SIGNS = {'\u0f72', '\u0f74', '\u0f7a', '\u0f7b', '\u0f7c', '\u0f7d'}

# Subjoined consonants, U+0F90 – U+0FBC
SUBJOINED = set(chr(c) for c in range(0x0F90, 0x0FBD))

# Structural rules for NATIVE Tibetan syllables
VALID_PREFIXES = {'\u0f42', '\u0f51', '\u0f56', '\u0f58', '\u0f60'}  # ག ད བ མ འ
VALID_SUFFIXES = {                                                    # ག ང ད ན བ མ འ ར ལ ས
    '\u0f42', '\u0f44', '\u0f51', '\u0f53', '\u0f56',
    '\u0f58', '\u0f60', '\u0f62', '\u0f63', '\u0f66',
}
VALID_SECOND_SUFFIXES = {'\u0f66', '\u0f51'}                          # ས ད

# Superscripts (can sit above a root): ར ལ ས
VALID_SUPERSCRIPTS = {'\u0f62', '\u0f63', '\u0f66'}

# Subscripts: ྱ ྲ ླ ྭ
VALID_SUBSCRIPTS = {'\u0fb1', '\u0fb2', '\u0fb3', '\u0fad'}

# ---- Sanskrit / transliteration markers --------------------------------------
# These appear in mantras and Sanskrit loanwords, which do NOT follow native
# Tibetan syllable rules. Any syllable containing one of these is skipped.
SANSKRIT_MARKERS = {
    '\u0f71',  # ཱ  long vowel a
    '\u0f73',  # ཱི long i
    '\u0f75',  # ཱུ long u
    '\u0f76', '\u0f77', '\u0f78', '\u0f79',  # vocalic r/l
    '\u0f80', '\u0f81',  # short/long reversed i
    '\u0f7e',  # ཾ  anusvara
    '\u0f7f',  # ཿ  visarga
    '\u0f82', '\u0f83',  # ྂ ྃ nada / candrabindu
    '\u0f84',  # ྄  halanta
    '\u0f85',  # ྅
    '\u0f39',  # ྐ  tsa-phru
    '\u0f35', '\u0f37',  # ordinary/lower dots
}

# Retroflex consonants used almost exclusively in Sanskrit transliteration
SANSKRIT_CONSONANTS = {
    '\u0f4a', '\u0f4b', '\u0f4c', '\u0f4d', '\u0f4e', '\u0f4f',  # ཊ ཋ ཌ ཌྷ ཎ ཏ-retroflex
    '\u0f69', '\u0f6a',
}
SANSKRIT_SUBJOINED = {
    '\u0f9a', '\u0f9b', '\u0f9c', '\u0f9d', '\u0f9e',
    '\u0fb9', '\u0fba', '\u0fbb', '\u0fbc',
}

# ---- Punctuation and delimiters ----------------------------------------------
# NOTE: this is the FULL set. The earlier audit script only had U+0F0B/0D/0E,
# which is why legitimate punctuation-only pairs were being mis-flagged.
TSHEG_LIKE = {'\u0f0b', '\u0f0c'}                      # ་ ༌
SHAD_LIKE = {'\u0f0d', '\u0f0e', '\u0f0f', '\u0f10',   # ། ༎ ༏ ༐
             '\u0f11', '\u0f12', '\u0f14'}             # ༑ ༒ ༔
PUNCTUATION = TSHEG_LIKE | SHAD_LIKE | {
    '\u0f04', '\u0f05', '\u0f06', '\u0f07', '\u0f08',
    '\u0f3a', '\u0f3b', '\u0f3c', '\u0f3d',
    '\u0f13', '\u0f85',
}

# Tibetan digits ༠–༩ and ༪–༳
TIBETAN_DIGITS = set(chr(c) for c in range(0x0F20, 0x0F34))

SYLLABLE_SPLIT = re.compile(
    r'[\u0f0b\u0f0c\u0f0d\u0f0e\u0f0f\u0f10\u0f11\u0f12\u0f14\s]+'
)

LATIN_OR_DIGIT = re.compile(r'[A-Za-z0-9]')
TIBETAN_RANGE = re.compile(r'[\u0f00-\u0fff]')


# ==============================================================================
# botok (optional but preferred)
# ==============================================================================

def load_botok():
    """botok is OpenPecha's Tibetan tokenizer and knows real syllable structure.
    If it isn't installed we fall back to the hand-written rules below."""
    try:
        from botok import WordTokenizer
        return WordTokenizer()
    except Exception as e:
        print(f"botok unavailable ({e}) — using built-in rules only.\n")
        return None


# ==============================================================================
# CLASSIFICATION
# ==============================================================================

def is_sanskrit_syllable(syl: str) -> bool:
    """True if the syllable looks like Sanskrit transliteration.

    These are common in mantras and dharanis in this corpus and legitimately
    violate native Tibetan orthography, so they must not be counted as errors.
    """
    for ch in syl:
        if ch in SANSKRIT_MARKERS or ch in SANSKRIT_CONSONANTS or ch in SANSKRIT_SUBJOINED:
            return True
    # Two or more subjoined consonants stacked is a strong Sanskrit signal
    if sum(1 for ch in syl if ch in SUBJOINED) >= 2:
        return True
    return False


def strip_punctuation(syl: str) -> str:
    return ''.join(ch for ch in syl if ch not in PUNCTUATION)


def validate_native_syllable(syl: str) -> List[str]:
    """Check a single native Tibetan syllable against orthographic rules.

    Returns a list of problems; empty means the syllable looks valid.
    """
    problems = []
    s = strip_punctuation(syl)

    if not s:
        return problems
    if not TIBETAN_RANGE.search(s):
        return problems              # Latin/digits/other — not our business
    if all(ch in TIBETAN_DIGITS for ch in s):
        return problems              # pure number

    consonants = [ch for ch in s if ch in BASE_CONSONANTS]
    vowels = [ch for ch in s if ch in VOWEL_SIGNS]
    subjoined = [ch for ch in s if ch in SUBJOINED]

    # A syllable needs at least one consonant
    if not consonants:
        problems.append("no base consonant")
        return problems

    # More than one vowel sign in a native syllable is invalid
    if len(vowels) > 1:
        problems.append(f"{len(vowels)} vowel signs")

    # A combining mark before any base consonant has nothing to attach to
    seen_base = False
    for ch in s:
        if ch in BASE_CONSONANTS:
            seen_base = True
        elif (ch in SUBJOINED or ch in VOWEL_SIGNS) and not seen_base:
            problems.append(f"combining mark U+{ord(ch):04X} before any base")
            break

    # Native syllables have at most 7 slots; more than 6 base consonants is
    # structurally impossible
    if len(consonants) > 6:
        problems.append(f"{len(consonants)} base consonants")

    # Subjoined consonant that isn't a legal subscript
    for ch in subjoined:
        if ch not in VALID_SUBSCRIPTS:
            problems.append(f"unusual subjoined U+{ord(ch):04X}")
            break

    # Prefix/suffix checks — only meaningful for plain CVC-shaped syllables
    if len(consonants) >= 2 and not subjoined:
        first, last = consonants[0], consonants[-1]
        if len(consonants) >= 3 and first not in VALID_PREFIXES \
                and first not in VALID_SUPERSCRIPTS:
            problems.append(f"invalid prefix/superscript '{first}'")
        if last not in VALID_SUFFIXES and last not in VALID_SECOND_SUFFIXES:
            problems.append(f"invalid suffix '{last}'")

    return problems


def check_text(text: str) -> Tuple[List[Tuple[str, List[str]]], int, int]:
    """Validate every syllable in a piece of text.

    Returns (problem_syllables, n_checked, n_sanskrit_skipped).
    """
    problems = []
    checked = 0
    skipped = 0

    for syl in SYLLABLE_SPLIT.split(text):
        if not syl.strip():
            continue
        if is_sanskrit_syllable(syl):
            skipped += 1
            continue
        checked += 1
        issues = validate_native_syllable(syl)
        if issues:
            problems.append((syl, issues))

    return problems, checked, skipped


# ==============================================================================
# MAIN
# ==============================================================================

def run(file_paths: Dict[str, str], sample_size: int = 20):
    tokenizer = load_botok()

    frames = []
    for split, path in file_paths.items():
        try:
            df = pd.read_csv(path)
            df['split'] = split
            frames.append(df)
            print(f"Loaded {split}: {len(df):,} rows")
        except Exception as e:
            print(f"Skipping {split} ({path}): {e}")

    if not frames:
        sys.exit("No CSVs loaded.")

    df = pd.concat(frames, ignore_index=True)

    if 'target' not in df.columns:
        sys.exit(f"No 'target' column. Found: {list(df.columns)}")

    if 'diff_category' not in df.columns:
        df['diff_category'] = 'unspecified'
    df['diff_category'] = df['diff_category'].fillna('unspecified')
    df['target'] = df['target'].fillna('').astype(str)
    has_source = 'source' in df.columns
    if has_source:
        df['source'] = df['source'].fillna('').astype(str)

    print(f"\nChecking {len(df):,} target strings...\n")

    flagged = []
    total_syllables = 0
    total_sanskrit = 0
    problem_counter = Counter()

    rows = zip(df.index, df['target'], df['diff_category'], df['split'],
               df['source'] if has_source else df['target'])

    for idx, tgt, cat, split, src in tqdm(rows, total=len(df),
                                          desc="Validating targets"):
        problems, checked, skipped = check_text(tgt)
        total_syllables += checked
        total_sanskrit += skipped

        if problems:
            for _, issues in problems:
                for issue in issues:
                    # Normalise the issue text so counts group sensibly
                    key = re.sub(r"'.'", "'X'", issue)
                    key = re.sub(r'U\+[0-9A-F]{4}', 'U+XXXX', key)
                    key = re.sub(r'^\d+ ', 'N ', key)
                    problem_counter[key] += 1

            flagged.append({
                'row_id': idx,
                'split': split,
                'diff_category': cat,
                'n_problem_syllables': len(problems),
                'problem_syllables': ' | '.join(s for s, _ in problems),
                'problems': ' | '.join('; '.join(i) for _, i in problems),
                'source': src if has_source else '',
                'target': tgt,
            })

    flagged_df = pd.DataFrame(flagged)

    # ---- Report --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("           TARGET-SIDE SPELLING / ORTHOGRAPHY CHECK")
    print("=" * 78)
    print(f"Rows checked                 : {len(df):,}")
    print(f"Native syllables validated   : {total_syllables:,}")
    print(f"Sanskrit syllables skipped   : {total_sanskrit:,} "
          f"({total_sanskrit / max(1, total_syllables + total_sanskrit):.1%})")
    print(f"Rows with a suspect target   : {len(flagged_df):,} "
          f"({len(flagged_df) / len(df):.2%})")

    if flagged_df.empty:
        print("\nNo orthography problems found in the reviewer targets.")
        return

    print("\n" + "-" * 78)
    print("PROBLEM TYPES")
    print("-" * 78)
    for problem, count in problem_counter.most_common():
        print(f"  {problem:<45}: {count:>7,}")

    print("\n" + "-" * 78)
    print("SUSPECT TARGETS BY CATEGORY")
    print("-" * 78)
    by_cat = flagged_df.groupby('diff_category').size()
    totals = df.groupby('diff_category').size()
    summary = pd.DataFrame({
        'flagged': by_cat,
        'total': totals,
    }).fillna(0).astype(int)
    summary['rate'] = (summary['flagged'] / summary['total']).map('{:.1%}'.format)
    print(summary.sort_values('flagged', ascending=False).to_string())

    print("\n" + "-" * 78)
    print(f"EXAMPLES (worst {sample_size} by number of problem syllables)")
    print("-" * 78)
    worst = flagged_df.sort_values('n_problem_syllables', ascending=False).head(sample_size)
    for i, (_, row) in enumerate(worst.iterrows(), 1):
        print(f"\n[{i}] split={row['split']} category={row['diff_category']} "
              f"problems={row['n_problem_syllables']}")
        if has_source:
            print(f"    SOURCE: {row['source']}")
        print(f"    TARGET: {row['target']}")
        print(f"    SUSPECT: {row['problem_syllables']}")
        print(f"    WHY: {row['problems']}")

    out = "target_spelling_flagged.csv"
    flagged_df.to_csv(out, index=False, encoding='utf-8')
    print(f"\n\nWrote {len(flagged_df):,} flagged rows to '{out}'.")
    print("\nRead a sample by hand before acting on any of this. A flag means")
    print("'worth a look', not 'definitely wrong' — archaic spellings, proper")
    print("nouns, and unusual transliterations will all show up here.")


if __name__ == "__main__":
    run({
        'train': 'train.csv',
        'validation': 'validation.csv',
        'test': 'test.csv',
    })