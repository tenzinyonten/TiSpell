#!/usr/bin/env python3
"""
Reshape synthetic pairs into the format OCRAnnotationDataset expects.

The dataloader wants a directory containing:
    differing_pairs.csv   - corrupted/clean pairs, with a `split` column
    identical_pairs.csv   - clean/clean pairs, with a `split` column

Required columns (from dataloader/ocr_pairs.py):
    page_id, segment_idx, source, target, split, diff_category

Two things this handles that matter:

1. IDENTICAL PAIRS. The synthetic output is 100% corrupted. Training on that
   alone teaches the model that every sentence needs changing. The dataloader
   expects a pool of clean/clean pairs so it learns to leave correct text
   alone, so we hold out uncorrupted sentences for that.

2. VAL/TEST ARE REAL. Val drives early stopping and checkpoint selection.
   If val is synthetic, you select the checkpoint that best reproduces your
   own corruption rules. Both val and test come from real annotator pairs.

Usage:
    python make_tispell_dataset.py \
        --synthetic synthetic_pairs_100k.csv \
        --real_pairs TiSpell/dataset/ocr_annotation/exports/differing_pairs.csv \
        --real_identical TiSpell/dataset/ocr_annotation/exports/identical_pairs.csv \
        --verdicts TiSpell/dataset/ocr_annotation/exports/gemini_target_check.csv \
        --outdir TiSpell/dataset/synthetic_v1
"""

import argparse
import os
import sys

import pandas as pd

COLS = ["page_id", "segment_idx", "source", "target", "split", "diff_category"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", required=True)
    p.add_argument("--real_pairs", required=True,
                   help="real differing_pairs.csv - source of val/test")
    p.add_argument("--real_identical", default=None,
                   help="real identical_pairs.csv - clean pairs for val/test")
    p.add_argument("--verdicts", required=True,
                   help="gemini_target_check.csv - to keep only clean targets")
    p.add_argument("--outdir", required=True)
    p.add_argument("--identical_frac", type=float, default=0.20,
                   help="clean pairs as a fraction of synthetic training rows")
    p.add_argument("--max_char_length", type=int, default=380)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---------------------------------------------------------- synthetic
    syn = pd.read_csv(args.synthetic)
    syn = syn.drop_duplicates(subset=["source", "target"])
    syn = syn[
        (syn["source"].astype(str).str.len() <= args.max_char_length)
        & (syn["target"].astype(str).str.len() <= args.max_char_length)
    ]
    syn = syn.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print(f"synthetic pairs usable: {len(syn)}")

    syn_diff = pd.DataFrame({
        "page_id": ["synthetic"] * len(syn),
        "segment_idx": range(len(syn)),
        "source": syn["source"],
        "target": syn["target"],
        "split": "train",
        "diff_category": "synthetic_type2",
    })

    # Clean pairs: reuse synthetic TARGETS as source==target. These sentences
    # are already validated Tibetan, so they are the right clean examples.
    n_ident = int(len(syn) * args.identical_frac)
    ident_src = syn["target"].drop_duplicates().head(n_ident)
    syn_ident = pd.DataFrame({
        "page_id": ["synthetic"] * len(ident_src),
        "segment_idx": range(len(ident_src)),
        "source": ident_src.values,
        "target": ident_src.values,
        "split": "train",
        "diff_category": "identical",
    })
    print(f"synthetic identical pairs: {len(syn_ident)}")

    # ---------------------------------------------------------- real
    verdicts = pd.read_csv(args.verdicts, dtype=str, keep_default_na=False)
    clean = set(verdicts.loc[verdicts["verdict"] == "correct", "target"])
    print(f"targets judged clean: {len(clean)}")

    real = pd.read_csv(args.real_pairs)
    real = real[real["target"].astype(str).isin(clean)].copy()
    real = real[
        (real["source"].astype(str).str.len() <= args.max_char_length)
        & (real["target"].astype(str).str.len() <= args.max_char_length)
    ]
    if "diff_category" not in real.columns:
        real["diff_category"] = "unknown"

    # Reuse the real data's own split column for val/test; ignore its train rows
    # (that data is what we pivoted away from).
    real_val = real[real["split"].astype(str) == "val"].copy()
    real_test = real[real["split"].astype(str) == "test"].copy()
    if real_val.empty or real_test.empty:
        print("WARNING: real val or test is empty after filtering to clean "
              "targets. Check the split column values:",
              real["split"].astype(str).unique()[:10], file=sys.stderr)

    for frame in (real_val, real_test):
        if "segment_idx" not in frame.columns:
            frame["segment_idx"] = range(len(frame))

    real_val = real_val.reindex(columns=COLS)
    real_test = real_test.reindex(columns=COLS)
    print(f"real val: {len(real_val)}   real test: {len(real_test)}")

    # ---------------------------------------------------------- identical
    ident_parts = [syn_ident]
    if args.real_identical and os.path.isfile(args.real_identical):
        ri = pd.read_csv(args.real_identical)
        ri = ri[ri["target"].astype(str).isin(clean)].copy()
        ri = ri[ri["split"].astype(str).isin(["val", "test"])]
        ri["diff_category"] = "identical"
        if "segment_idx" not in ri.columns:
            ri["segment_idx"] = range(len(ri))
        ri = ri.reindex(columns=COLS)
        ident_parts.append(ri)
        print(f"real identical (val+test): {len(ri)}")

    # ---------------------------------------------------------- write
    differing = pd.concat([syn_diff, real_val, real_test], ignore_index=True)
    identical = pd.concat(ident_parts, ignore_index=True)

    differing.to_csv(f"{args.outdir}/differing_pairs.csv", index=False)
    identical.to_csv(f"{args.outdir}/identical_pairs.csv", index=False)

    print(f"\nwritten to {args.outdir}/")
    print(differing.groupby(["split", "diff_category"]).size().to_string())
    print("\nNOTE: train is synthetic, val/test are real. Val drives early "
          "stopping, so the checkpoint is selected on real-world performance.")


if __name__ == "__main__":
    main()