"""
Build a size-matched random control for the Gemini-clean experiment.

The confound this removes:

    Training on train_gemini_clean.csv (12,016 pairs) and comparing against
    0.7897 (trained on 43,015 pairs) conflates two changes -- target quality
    AND training set size. If F1 drops you cannot tell which caused it.

The fix is a control set of the same size, sampled at random from the full
training data, so the ONLY difference between the two runs is target quality.

Two variants are written:

    train_random_control.csv        12,016 pairs sampled uniformly at random
    train_random_stratified.csv     12,016 pairs sampled to match the clean
                                    set's diff_category distribution

The stratified one is the stricter control: it holds both size AND category mix
constant, so the only remaining difference is whether Gemini judged the target
correct. Prefer it if you only have budget for one control run.

Usage:
    python build_control_set.py
"""

import sys

import pandas as pd

BASE = 'dataset/ocr_annotation/exports/hf_differing/'
SEED = 42


def main():
    try:
        train = pd.read_csv(BASE + 'train.csv')
        clean = pd.read_csv(BASE + 'train_gemini_clean.csv')
    except Exception as e:
        sys.exit(f"Could not read inputs: {e}\n"
                 f"Run this from the TiSpell repo root.")

    n = len(clean)
    print(f"Full train        : {len(train):,} pairs")
    print(f"Gemini-clean      : {n:,} pairs")

    # ---- Control 1: uniform random ------------------------------------------
    random_control = train.sample(n=n, random_state=SEED)
    random_control.to_csv(BASE + 'train_random_control.csv', index=False)

    # ---- Control 2: stratified to match the clean category mix --------------
    clean_dist = clean['diff_category'].value_counts()

    parts = []
    shortfall = 0
    for cat, want in clean_dist.items():
        pool = train[train['diff_category'] == cat]
        take = min(want, len(pool))
        if take < want:
            shortfall += want - take
            print(f"  note: only {len(pool):,} '{cat}' pairs available, "
                  f"wanted {want:,}")
        parts.append(pool.sample(n=take, random_state=SEED))

    stratified = pd.concat(parts, ignore_index=True)

    # Top up from the remaining pool if any category ran short
    if shortfall:
        used = set(stratified.index)
        remaining = train.drop(index=[i for i in used if i in train.index],
                               errors='ignore')
        if len(remaining) >= shortfall:
            stratified = pd.concat(
                [stratified, remaining.sample(n=shortfall, random_state=SEED)],
                ignore_index=True)

    stratified = stratified.sample(frac=1, random_state=SEED).reset_index(drop=True)
    stratified.to_csv(BASE + 'train_random_stratified.csv', index=False)

    # ---- Report -------------------------------------------------------------
    print(f"\nWrote:")
    print(f"  train_random_control.csv     {len(random_control):,} pairs")
    print(f"  train_random_stratified.csv  {len(stratified):,} pairs")

    print("\nCategory distributions:\n")
    comparison = pd.DataFrame({
        'clean': clean['diff_category'].value_counts(),
        'random': random_control['diff_category'].value_counts(),
        'stratified': stratified['diff_category'].value_counts(),
        'full_train': train['diff_category'].value_counts(),
    }).fillna(0).astype(int)
    print(comparison.to_string())

    # How much of the clean set the control accidentally overlaps
    clean_targets = set(clean['target'])
    overlap_r = random_control['target'].isin(clean_targets).mean()
    overlap_s = stratified['target'].isin(clean_targets).mean()
    print(f"\nOverlap with the clean set (lower is a cleaner contrast):")
    print(f"  random     : {overlap_r:.1%}")
    print(f"  stratified : {overlap_s:.1%}")
    print("  ~28% is expected, since the clean pairs are 28% of the full set.")

    print(f"""
{'=' * 72}
  RUNS TO DO
{'=' * 72}

  A) clean targets
     python train.py --device cuda:0 --epochs 15 \\
         --learning_rate 2e-5 --dropout 0.2 --use_wandb \\
         --train_file .../train_gemini_clean.csv

  B) size-matched control
     python train.py --device cuda:0 --epochs 15 \\
         --learning_rate 2e-5 --dropout 0.2 --use_wandb \\
         --train_file .../train_random_stratified.csv

  Same hyperparameters, same size, same category mix. The only difference is
  whether Gemini judged the target correct.

  ~2.5h each, roughly $1 each.

{'=' * 72}
  HOW TO READ IT
{'=' * 72}

  A clearly above B   ->  target quality is the ceiling. Filtering the data
                          is the highest-value next step, and there is a case
                          for re-annotating or sourcing better targets.

  A and B about equal ->  target quality is not the ceiling, at least not at
                          this scale. The limit is the model, the task, or
                          the amount of data.

  A below B           ->  worth a hard look at whether Gemini's verdicts are
                          reliable. Read 30 'incorrect' judgements by hand
                          before believing this one.

  Note both runs will likely score BELOW 0.7897 simply because 12k pairs is
  28% of the original training data. That drop is not the result -- the
  A-versus-B gap is.

  Both are scored on the UNCHANGED validation set, so they remain comparable
  to the four earlier runs.
""")


if __name__ == "__main__":
    main()