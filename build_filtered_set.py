"""
Build a filtered training set from the Gemini target verdicts.

Drops train rows whose target Gemini judged "incorrect" -- EXCEPT where the only
problem is the known encoding artifact (ཌ standing in for གས, ཾ for མ). Those
are not real errors: the same words appear both ways throughout the corpus, and
both characters occur at similar rates in source and target, so they originate
upstream of the reviewers.

Rows are tested by normalising the target and checking whether that changes it.
If normalising changes the text, the flag was (at least partly) the artifact and
the row is kept. The ORIGINAL text is written out, not the normalised form --
this keeps train in the same encoding as val and test.

Rows with no Gemini verdict are kept. Coverage is only ~65%, so dropping
unchecked rows would discard a third of the data for no evidential reason.

Validation and test are copied verbatim and checksummed, so results stay
comparable to the four completed runs (0.7725 / 0.7878 / 0.7897 / 0.7804).

Nothing is overwritten.

Usage:
    python build_filtered_set.py
"""

import hashlib
import os
import shutil
import sys

import pandas as pd

EXPORTS = 'dataset/ocr_annotation/exports/'
SRC = EXPORTS + 'hf_differing/'
OUT = EXPORTS + 'hf_differing_filtered/'
GEMINI = EXPORTS + 'gemini_target_check_v1_noconventions.csv'

TSHEG = '\u0f0b'
SHAD = '\u0f0d'
DDA = '\u0f4c'          # ཌ
GA_SA = '\u0f42\u0f66'  # གས
ANUSVARA = '\u0f7e'     # ཾ
MA = '\u0f58'           # མ


def file_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_anusvara_map(texts) -> dict:
    """Whitelist of anusvara syllables that also appear spelled with མ.

    Only these are treated as the artifact. The rest are Sanskrit, where the
    anusvara is correct -- ཧཱུཾ, ཨོཾ and so on must not be touched.
    """
    blob = ' '.join(t for t in texts if isinstance(t, str))
    syls = set(blob.replace(TSHEG, ' ').replace(SHAD, ' ').split())
    return {s: s.replace(ANUSVARA, MA) for s in syls
            if ANUSVARA in s and s.replace(ANUSVARA, MA) in syls}


def make_normaliser(anu_map: dict):
    def norm(text):
        if not isinstance(text, str):
            return ''
        return TSHEG.join(
            anu_map.get(s, s).replace(DDA, GA_SA) for s in text.split(TSHEG)
        )
    return norm


def main():
    if not os.path.isdir(SRC):
        sys.exit(f"Not found: {SRC}\nRun this from the TiSpell repo root.")

    os.makedirs(OUT, exist_ok=True)

    try:
        gemini = pd.read_csv(GEMINI)
    except Exception as e:
        sys.exit(f"Could not read {GEMINI}: {e}")

    print(f"Gemini verdicts: {len(gemini):,} rows, "
          f"{gemini['target'].nunique():,} unique targets")
    print(gemini['verdict'].value_counts().to_string())
    print()

    incorrect = set(gemini.loc[gemini['verdict'] == 'incorrect', 'target'])
    judged = set(gemini['target'])

    train = pd.read_csv(SRC + 'train.csv')
    train['source'] = train['source'].fillna('').astype(str)
    train['target'] = train['target'].fillna('').astype(str)

    # ---- Normaliser, built from the corpus itself ---------------------------
    anu_map = build_anusvara_map(list(train['source']) + list(train['target']))
    norm = make_normaliser(anu_map)
    print(f"anusvara whitelist: {len(anu_map):,} syllable forms "
          f"(rest treated as Sanskrit and left alone)\n")

    train['_norm'] = train['target'].map(norm)
    train['_judged'] = train['target'].isin(judged)
    train['_flagged'] = train['target'].isin(incorrect)

    # Artifact-only: flagged, but normalising changes the text
    train['_artifact_only'] = train['_flagged'] & (train['target'] != train['_norm'])

    # Real error: flagged, and normalising leaves it unchanged
    train['_drop'] = train['_flagged'] & (train['target'] == train['_norm'])

    before = len(train)
    flagged = int(train['_flagged'].sum())
    artifact = int(train['_artifact_only'].sum())
    dropped = int(train['_drop'].sum())
    unchecked = int((~train['_judged']).sum())

    filtered = train.loc[~train['_drop']].drop(
        columns=['_norm', '_judged', '_flagged', '_artifact_only', '_drop'])
    filtered.to_csv(OUT + 'train.csv', index=False, encoding='utf-8')

    # ---- Val / test: verbatim ----------------------------------------------
    print("=" * 72)
    print("  VALIDATION AND TEST -- COPIED VERBATIM")
    print("=" * 72)

    integrity_ok = True
    for split in ('validation', 'test'):
        src_path, out_path = SRC + split + '.csv', OUT + split + '.csv'

        before_sum = file_checksum(src_path)
        before_rows = len(pd.read_csv(src_path))

        shutil.copy2(src_path, out_path)

        after_sum = file_checksum(out_path)
        after_rows = len(pd.read_csv(out_path))

        same = (before_sum == after_sum) and (before_rows == after_rows)
        integrity_ok = integrity_ok and same

        print(f"\n  {split}")
        print(f"    rows      : {before_rows:,}  ->  {after_rows:,}")
        print(f"    sha256[16]: {before_sum}  ->  {after_sum}")
        print(f"    identical : {'YES' if same else 'NO  <-- PROBLEM'}")

    # ---- Report -------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  TRAIN SPLIT")
    print("=" * 72)
    print(f"  before             : {before:,}")
    print(f"  flagged by Gemini  : {flagged:,}  ({flagged / before:.1%})")
    print(f"    of which artifact: {artifact:,}  "
          f"({artifact / max(1, flagged):.1%} of flagged)  -- KEPT")
    print(f"    real errors      : {dropped:,}  "
          f"({dropped / max(1, flagged):.1%} of flagged)  -- DROPPED")
    print(f"  no verdict         : {unchecked:,}  "
          f"({unchecked / before:.1%})  -- KEPT")
    print()
    print(f"  remaining          : {len(filtered):,}  "
          f"({len(filtered) / before:.1%})")
    print(f"  kept vs. dropping all flagged: +{artifact:,} rows")

    print("\n" + "-" * 72)
    print("  PER CATEGORY")
    print("-" * 72)
    comp = pd.DataFrame({
        'before': train['diff_category'].value_counts(),
        'artifact_kept': train.loc[train['_artifact_only'],
                                   'diff_category'].value_counts(),
        'dropped': train.loc[train['_drop'], 'diff_category'].value_counts(),
        'after': filtered['diff_category'].value_counts(),
    }).fillna(0).astype(int)
    comp['drop_rate'] = (comp['dropped'] / comp['before']).map('{:.1%}'.format)
    comp['share_after'] = (comp['after'] / len(filtered)).map('{:.1%}'.format)
    print(comp.sort_values('before', ascending=False).to_string())

    print(f"""
{'=' * 72}
  WROTE
{'=' * 72}

  {OUT}train.csv       {len(filtered):,} rows
  {OUT}validation.csv  unchanged
  {OUT}test.csv        unchanged

  Originals in {SRC} untouched.
  Integrity check: {'PASS' if integrity_ok else 'FAIL -- do not proceed'}

  Text is written in its ORIGINAL encoding, not normalised, so train stays
  consistent with val and test.

  Run with:
    python train.py --device cuda:0 --epochs 15 \\
        --learning_rate 2e-5 --dropout 0.2 --use_wandb \\
        --data_path {OUT}

{'=' * 72}
  INTERPRETING THE RESULT
{'=' * 72}

  Two things change at once: target quality and training set size. If F1
  moves, that alone does not say which caused it. A size-matched random
  control run is what separates them.

  Gemini coverage is ~65%, so the kept set mixes verified-correct and
  unverified rows. Cleaner than the original, not clean.

  Validation still contains flagged targets. Scoring against imperfect
  references caps the achievable number regardless of how good the model is.
""")


if __name__ == "__main__":
    main()