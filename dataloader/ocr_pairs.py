"""OCR annotation sentence-pair dataset for TiSpell.

Loads differing_pairs.csv + identical_pairs.csv and respects the existing
`split` column (train / val / test). No synthetic corruption.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


def _subsample_identical_for_ratio(
    diff: pd.DataFrame,
    ident: pd.DataFrame,
    ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Keep all differing; sample identical so they are `ratio` of the result.

    Solve n_i / (n_d + n_i) = ratio  ⇒  n_i = n_d * ratio / (1 - ratio).
    """
    if ratio <= 0:
        return diff.reset_index(drop=True)
    if ratio >= 1.0:
        return pd.concat([diff, ident], ignore_index=True)

    n_diff = len(diff)
    n_ident_target = int(round(n_diff * ratio / (1.0 - ratio)))
    n_ident_target = min(n_ident_target, len(ident))
    if n_ident_target <= 0:
        return diff.reset_index(drop=True)
    ident_keep = ident.sample(n=n_ident_target, random_state=seed)
    return pd.concat([diff, ident_keep], ignore_index=True)


class OCRAnnotationDataset(Dataset):
    """Parallel (source → target) pairs from the OCR annotation exports."""

    def __init__(
        self,
        args,
        tokenizer,
        split: str,
        *,
        differing_name: str = "differing_pairs.csv",
        identical_name: str = "identical_pairs.csv",
    ):
        self.tokenizer = tokenizer
        self.max_subword_length = args.max_subword_length
        self.max_char_length = args.max_char_length
        data_dir = Path(args.data_path)
        seed = getattr(args, "seed", 42)

        diff_path = data_dir / differing_name
        ident_path = data_dir / identical_name
        if not diff_path.is_file():
            raise FileNotFoundError(f"Missing differing pairs: {diff_path}")
        if not ident_path.is_file():
            raise FileNotFoundError(f"Missing identical pairs: {ident_path}")

        cols = [
            "source",
            "target",
            "is_identical",
            "diff_category",
            "page_id",
            "segment_idx",
        ]

        diff = pd.read_csv(diff_path)
        diff = diff[diff["split"].astype(str) == split].copy()
        diff["is_identical"] = False
        if "diff_category" not in diff.columns:
            diff["diff_category"] = "unknown"
        diff = diff[cols]

        ident = pd.read_csv(ident_path)
        ident = ident[ident["split"].astype(str) == split].copy()
        ident["is_identical"] = True
        ident["diff_category"] = "identical"
        ident = ident[cols]

        # Safety: enforce char cap (exports should already be ≤ max_char_length).
        def _cap(frame: pd.DataFrame) -> pd.DataFrame:
            src_ok = frame["source"].astype(str).map(len) <= self.max_char_length
            tgt_ok = frame["target"].astype(str).map(len) <= self.max_char_length
            return frame.loc[src_ok & tgt_ok].reset_index(drop=True)

        diff = _cap(diff)
        ident = _cap(ident)

        identical_ratio = float(getattr(args, "identical_ratio", 1.0))
        if split == "train":
            n_ident_before = len(ident)
            df = _subsample_identical_for_ratio(diff, ident, identical_ratio, seed)
            n_ident_after = int(df["is_identical"].sum())
            n_diff = int((~df["is_identical"]).sum())
            ratio_actual = n_ident_after / max(len(df), 1)
            print(
                f"OCRAnnotationDataset split=train: identical subsample "
                f"{n_ident_before} → {n_ident_after} "
                f"(target ratio={identical_ratio:.2f}, actual={ratio_actual:.3f}); "
                f"differing={n_diff}, total={len(df)}"
            )
        else:
            # Val / test: keep all identical pairs.
            df = pd.concat([diff, ident], ignore_index=True)

        subset = getattr(args, "subset", 1.0)
        if subset is not None and subset < 1.0:
            if subset <= 0:
                raise ValueError(f"--subset must be in (0, 1], got {subset}")
            n = max(1, int(len(df) * subset))
            df = df.sample(n=n, random_state=seed).reset_index(drop=True)

        self.df = df.reset_index(drop=True)
        n_diff = int((~self.df["is_identical"]).sum())
        n_ident = int(self.df["is_identical"].sum())
        extra = ""
        if subset is not None and subset < 1.0:
            extra = f", subset={subset}"
        if split != "train":
            print(
                f"OCRAnnotationDataset split={split}: "
                f"{len(self.df)} pairs (differing={n_diff}, identical={n_ident})"
                f"{extra}"
            )
        elif subset is not None and subset < 1.0:
            print(
                f"OCRAnnotationDataset split=train after subset={subset}: "
                f"{len(self.df)} pairs (differing={n_diff}, identical={n_ident})"
            )

    def __len__(self) -> int:
        return len(self.df)

    def _tokenize(self, text: str) -> dict:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_subword_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        source = str(row["source"])
        target = str(row["target"])
        return {
            # Keep key names close to the original trainer: input vs gold.
            "random_corrupt": self._tokenize(source),
            "target": self._tokenize(target),
            # Character-head target: gold tokens (no synthetic mask on real pairs).
            "mask": self._tokenize(target),
            "diff_category": str(row["diff_category"]),
        }
