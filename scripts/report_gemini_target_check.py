#!/usr/bin/env python3
"""Analyze gemini_target_check.csv and write a short markdown report.

Covers verdict mix, script/batch breakdown, length correlation, and a
cross-row contamination check on Tibetan tokens cited in error_description.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("dataset/ocr_annotation/exports/gemini_target_check.csv")
DEFAULT_REPORT = Path(
    "dataset/ocr_annotation/exports/gemini_target_check_report.md"
)
DEFAULT_CORRECT_OUT = Path(
    "dataset/ocr_annotation/exports/gemini_target_check_correct.csv"
)

TIBETAN_TOKEN_RE = re.compile(r"[\u0F00-\u0FFF]+(?:\s*[\u0F00-\u0FFF]+)*")

MANTRA_HINT_RE = re.compile(
    r"(ཨོཾ|ཨཱོྃ|ཧཱུཾ|ཧཱུྃ|ཕཊ|བཛྲ|སརྦ|ཏཐཱ|གཏ|སྭཱ|ཨཱཿ|ཧྲཱི|མ་ཎི)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument(
        "--correct_out",
        type=Path,
        default=DEFAULT_CORRECT_OUT,
        help="CSV of all rows with verdict=correct",
    )
    p.add_argument(
        "--correct_sample",
        type=int,
        default=40,
        help="How many correct rows to embed in the markdown report",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def script_of(page_id: str) -> str:
    prefix = str(page_id).split("/", 1)[0].lower()
    if prefix.startswith("uchen"):
        return "uchen"
    if prefix.startswith("ume"):
        return "ume"
    return "other"


def batch_of(page_id: str) -> str:
    return str(page_id).split("/", 1)[0]


def pct(n: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{100.0 * n / total:5.1f}%"


def verdict_table(df: pd.DataFrame) -> str:
    total = len(df)
    lines = [
        "| verdict | count | % |",
        "|---|---:|---:|",
    ]
    for v in ("correct", "incorrect", "unsure"):
        n = int((df["verdict"] == v).sum())
        lines.append(f"| {v} | {n} | {pct(n, total)} |")
    other = int((~df["verdict"].isin(["correct", "incorrect", "unsure"])).sum())
    if other:
        lines.append(f"| other | {other} | {pct(other, total)} |")
    lines.append(f"| **total** | **{total}** | 100% |")
    return "\n".join(lines)


def error_count_stats(incorrect: pd.DataFrame) -> tuple[str, str]:
    if incorrect.empty:
        return "_No incorrect rows._", ""
    ec = pd.to_numeric(incorrect["error_count"], errors="coerce").fillna(0)
    summary = (
        f"- n = {len(ec)}\n"
        f"- mean = {ec.mean():.2f}\n"
        f"- median = {ec.median():.1f}\n"
        f"- max = {int(ec.max())}\n"
        f"- zeros (error_count=0) = {int((ec == 0).sum())}"
    )
    bins = [0, 1, 2, 3, 4, 5, 10, 20, 50, 10**9]
    labels = ["0", "1", "2", "3", "4", "5-9", "10-19", "20-49", "50+"]
    cats = pd.cut(ec, bins=bins, labels=labels, right=False, include_lowest=True)
    hist = cats.value_counts().reindex(labels, fill_value=0)
    lines = [
        "| error_count | count | % of incorrect |",
        "|---|---:|---:|",
    ]
    for lab, n in hist.items():
        lines.append(f"| {lab} | {int(n)} | {pct(int(n), len(ec))} |")
    return summary, "\n".join(lines)


def breakdown_table(df: pd.DataFrame, key: str) -> str:
    rows = []
    for name, g in df.groupby(key, sort=False):
        n = len(g)
        c = int((g["verdict"] == "correct").sum())
        i = int((g["verdict"] == "incorrect").sum())
        u = int((g["verdict"] == "unsure").sum())
        rows.append((name, n, c, i, u))
    rows.sort(key=lambda r: (-r[1], str(r[0])))
    lines = [
        f"| {key} | n | correct | incorrect | unsure | % correct | % incorrect |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, n, c, i, u in rows:
        lines.append(
            f"| {name} | {n} | {c} | {i} | {u} | {pct(c, n)} | {pct(i, n)} |"
        )
    return "\n".join(lines)


def length_analysis(df: pd.DataFrame) -> str:
    df = df.copy()
    df["len"] = df["target"].astype(str).map(len)
    overall = df["len"]
    lines = [
        "### Overall target length (characters)",
        f"- mean = {overall.mean():.1f}",
        f"- median = {overall.median():.1f}",
        f"- p10 / p90 = {overall.quantile(0.10):.0f} / {overall.quantile(0.90):.0f}",
        f"- min / max = {int(overall.min())} / {int(overall.max())}",
        "",
        "### Length by verdict",
        "| verdict | n | mean len | median len |",
        "|---|---:|---:|---:|",
    ]
    for v in ("correct", "incorrect", "unsure"):
        g = df.loc[df["verdict"] == v, "len"]
        if g.empty:
            continue
        lines.append(
            f"| {v} | {len(g)} | {g.mean():.1f} | {g.median():.1f} |"
        )

    try:
        df["len_q"] = pd.qcut(df["len"], 5, duplicates="drop")
    except ValueError:
        df["len_q"] = pd.cut(df["len"], 5)
    lines += [
        "",
        "### Incorrect rate by target-length quintile",
        "| length bin | n | % incorrect | mean len |",
        "|---|---:|---:|---:|",
    ]
    for bin_label, g in df.groupby("len_q", observed=True):
        n = len(g)
        inc = int((g["verdict"] == "incorrect").sum())
        lines.append(
            f"| `{bin_label}` | {n} | {pct(inc, n)} | {g['len'].mean():.1f} |"
        )

    corr = df.loc[df["verdict"].isin(["correct", "incorrect"])].copy()
    if len(corr) >= 2:
        y = (corr["verdict"] == "incorrect").astype(float)
        x = corr["len"].astype(float)
        if x.std() > 0 and y.std() > 0:
            r = float(np.corrcoef(x, y)[0, 1])
            lines += [
                "",
                f"Pearson correlation(length, is_incorrect) = **{r:.3f}** "
                f"(positive ⇒ longer targets more often marked incorrect).",
            ]
    return "\n".join(lines)


def extract_tibetan_tokens(text: str) -> list[str]:
    if not text:
        return []
    toks = []
    for m in TIBETAN_TOKEN_RE.finditer(str(text)):
        tok = m.group(0).strip()
        letters = re.sub(r"[\u0F0B-\u0F14\s]+", "", tok)
        if len(letters) < 2:
            continue
        toks.append(tok)
    return toks


def contamination_check(
    df: pd.DataFrame, *, neighbor_window: int = 5, rare_max_targets: int = 3
) -> tuple[str, pd.DataFrame]:
    """Flag incorrect rows whose error_description cites tokens from other targets.

    Tokens absent from the own target may be suggested corrections (common
    vocabulary) or leaked forms from another row in the same Gemini batch.
    We report:

    1. Broad: token ∉ own target but ∈ some other target (high false-positive
       rate from ordinary suggested corrections).
    2. Rare: same, but the token occurs in ≤ ``rare_max_targets`` other
       targets (more likely an idiosyncratic OCR form leaked across rows).
    3. Neighbor: same, but the other target is within ±``neighbor_window``
       rows in file order (matches batch_size-style prompting leakage).
    """
    incorrect = df[df["verdict"] == "incorrect"].reset_index(drop=True)
    if incorrect.empty:
        return "_No incorrect rows._", pd.DataFrame()

    targets = incorrect["target"].astype(str).tolist()
    desc_tokens_per_row = [
        extract_tibetan_tokens(d) for d in incorrect["error_description"]
    ]
    # Only tokens that are missing from at least one citing row need indexing.
    candidate_tokens: set[str] = set()
    for i, toks in enumerate(desc_tokens_per_row):
        own = targets[i]
        for tok in toks:
            if tok not in own:
                candidate_tokens.add(tok)

    token_to_rows: dict[str, set[int]] = defaultdict(set)
    for i, tgt in enumerate(targets):
        for tok in candidate_tokens:
            if tok in tgt:
                token_to_rows[tok].add(i)

    affected_rows = []
    n_broad = n_rare = n_neighbor = 0
    n_broad_hits = n_rare_hits = n_neighbor_hits = 0

    for i, toks in enumerate(desc_tokens_per_row):
        own = targets[i]
        broad, rare, neighbor = [], [], []
        for tok in toks:
            if tok in own:
                continue
            others = token_to_rows.get(tok, set()) - {i}
            if not others:
                continue
            broad.append(tok)
            n_broad_hits += 1
            if len(others) <= rare_max_targets:
                rare.append(tok)
                n_rare_hits += 1
            if any(abs(j - i) <= neighbor_window for j in others):
                neighbor.append(tok)
                n_neighbor_hits += 1

        if broad:
            n_broad += 1
        if rare:
            n_rare += 1
        if neighbor:
            n_neighbor += 1
        if rare or neighbor:
            row = incorrect.iloc[i]
            affected_rows.append(
                {
                    "page_id": row["page_id"],
                    "target": row["target"],
                    "error_description": row["error_description"],
                    "rare_cross_tokens": " | ".join(dict.fromkeys(rare)),
                    "neighbor_cross_tokens": " | ".join(dict.fromkeys(neighbor)),
                    "n_rare": len(set(rare)),
                    "n_neighbor": len(set(neighbor)),
                    "n_broad": len(set(broad)),
                }
            )

    aff = pd.DataFrame(affected_rows)
    n_aff = len(aff)
    lines = [
        f"- Incorrect rows checked: **{len(incorrect)}**",
        f"- Candidate Tibetan tokens in descriptions (absent from own target): "
        f"**{len(candidate_tokens)}**",
        "",
        f"- **Broad** (token ∉ own target, ∈ some other target): "
        f"**{n_broad}** rows ({pct(n_broad, len(incorrect))} of incorrect); "
        f"{n_broad_hits} token-hits. *Mostly suggested corrections that are "
        f"ordinary Tibetan vocabulary — treat as an upper bound.*",
        f"- **Rare** (same, but token occurs in ≤{rare_max_targets} other "
        f"targets): **{n_rare}** rows ({pct(n_rare, len(incorrect))}); "
        f"{n_rare_hits} token-hits. *Stronger contamination signal.*",
        f"- **Neighbor** (same, other target within ±{neighbor_window} rows): "
        f"**{n_neighbor}** rows ({pct(n_neighbor, len(incorrect))}); "
        f"{n_neighbor_hits} token-hits. *Matches batched-prompt leakage.*",
        f"- Rows in companion CSV (rare ∪ neighbor): **{n_aff}**",
    ]
    show = aff.head(15) if not aff.empty else aff
    if not show.empty:
        lines += [
            "",
            "Sample flagged rows (up to 15; rare ∪ neighbor):",
            "",
            "| page_id | rare tokens | neighbor tokens | error_description (trim) |",
            "|---|---|---|---|",
        ]
        for _, r in show.iterrows():
            desc = str(r["error_description"]).replace("|", "\\|")[:100]
            rare_s = str(r["rare_cross_tokens"]).replace("|", "\\|")[:60]
            neigh_s = str(r["neighbor_cross_tokens"]).replace("|", "\\|")[:60]
            pid = str(r["page_id"]).replace("|", "\\|")
            lines.append(f"| `{pid}` | {rare_s} | {neigh_s} | {desc} |")
    return "\n".join(lines), aff


def looks_like_mantra(text: str) -> bool:
    return bool(MANTRA_HINT_RE.search(text))


def correct_section(
    df: pd.DataFrame, sample_n: int, seed: int
) -> tuple[str, pd.DataFrame]:
    correct = df[df["verdict"] == "correct"].copy()
    correct["len"] = correct["target"].astype(str).map(len)
    correct["mantra_hint"] = correct["target"].astype(str).map(looks_like_mantra)
    n = len(correct)
    n_mantra = int(correct["mantra_hint"].sum())
    lines = [
        f"- Correct rows: **{n}**",
        f"- Heuristic mantra/Sanskrit-marker hits "
        f"(ཨོཾ / ཧཱུཾ / བཛྲ / …): **{n_mantra}** ({pct(n_mantra, n)})",
        f"- Mean / median length: "
        f"{correct['len'].mean():.1f} / {correct['len'].median():.1f}"
        if n
        else "- Mean / median length: n/a",
        "",
        "Full correct-row listing is written to the companion CSV "
        "(too large to embed). Sample below "
        f"({min(sample_n, n)} rows; mantra-hint rows preferentially included).",
        "",
    ]
    if n == 0:
        return "\n".join(lines), correct

    mantra = correct[correct["mantra_hint"]]
    other = correct[~correct["mantra_hint"]]
    n_m = min(len(mantra), max(sample_n // 2, 1))
    n_o = min(len(other), sample_n - n_m)
    parts = []
    if n_m:
        parts.append(mantra.sample(n=n_m, random_state=seed))
    if n_o:
        parts.append(other.sample(n=n_o, random_state=seed))
    sample = pd.concat(parts, ignore_index=True) if parts else correct.head(0)

    lines += [
        "| mantra_hint | len | page_id | target |",
        "|---|---:|---|---|",
    ]
    for _, r in sample.iterrows():
        tgt = str(r["target"]).replace("|", "\\|").replace("\n", " ")
        if len(tgt) > 160:
            tgt = tgt[:160] + "…"
        pid = str(r["page_id"]).replace("|", "\\|")
        lines.append(
            f"| {bool(r['mantra_hint'])} | {int(r['len'])} | `{pid}` | {tgt} |"
        )
    return "\n".join(lines), correct


def analyze(df_all: pd.DataFrame, args: argparse.Namespace):
    df_all = df_all.copy()
    df_all["verdict"] = df_all["verdict"].astype(str).str.strip().str.lower()
    df_all["page_id"] = df_all["page_id"].astype(str)
    df_all["target"] = df_all["target"].astype(str)
    df_all["error_description"] = df_all["error_description"].astype(str)
    df_all["error_count"] = pd.to_numeric(
        df_all["error_count"], errors="coerce"
    ).fillna(0)
    df_all["script"] = df_all["page_id"].map(script_of)
    df_all["batch"] = df_all["page_id"].map(batch_of)

    n_all = len(df_all)
    dup_mask = df_all.duplicated(subset=["page_id", "target"], keep="first")
    n_dup_extra = int(dup_mask.sum())
    df_uniq = df_all.loc[~dup_mask].copy()
    n_uniq = len(df_uniq)

    def block(title: str, df: pd.DataFrame) -> list[str]:
        incorrect = df[df["verdict"] == "incorrect"]
        ec_summary, ec_hist = error_count_stats(incorrect)
        return [
            f"## {title}",
            "",
            f"Rows: **{len(df)}**",
            "",
            "### Verdict distribution",
            verdict_table(df),
            "",
            "### error_count (incorrect rows only)",
            ec_summary,
            "",
            ec_hist,
            "",
            "### By script (page_id prefix)",
            breakdown_table(df, "script"),
            "",
            "### By batch",
            breakdown_table(df, "batch"),
            "",
            "### Target length vs verdict",
            length_analysis(df),
            "",
        ]

    contam_text_all, _ = contamination_check(df_all)
    contam_text_uniq, contam_df_uniq = contamination_check(df_uniq)
    correct_md, correct_df = correct_section(
        df_uniq, args.correct_sample, args.seed
    )

    parts = [
        "# Gemini target-check report",
        "",
        f"Source: `{args.input}`",
        "",
        "Reviewer targets are corrected toward the **manuscript**, not toward "
        "normative Tibetan, so preserved scribal errors are expected.",
        "",
        "## 1. Row counts and duplicates",
        "",
        f"- Total rows: **{n_all}**",
        f"- Exact duplicate rows on `(page_id, target)` (extras beyond first): "
        f"**{n_dup_extra}**",
        f"- Unique `(page_id, target)` rows: **{n_uniq}**",
        "",
        "All statistics below are reported **with duplicates** (full file) and "
        "**without duplicates** (first occurrence kept).",
        "",
    ]
    parts += block("2–5. With duplicates (full file)", df_all)
    parts += block("2–5. Without duplicates", df_uniq)
    parts += [
        "## 6. Cross-row contamination check",
        "",
        "For each incorrect row, Tibetan-script tokens are extracted from "
        "`error_description`. Tokens present in the row's own `target` are "
        "ignored. Tokens absent from the own target may be suggested "
        "corrections **or** forms leaked from another row in a batched prompt. "
        "Because ordinary Tibetan suggestions also appear elsewhere in the "
        "corpus, we report a broad upper bound plus stricter rare-token and "
        "neighbor-window signals.",
        "",
        "### With duplicates",
        contam_text_all,
        "",
        "### Without duplicates",
        contam_text_uniq,
        "",
        "## 7. Rows marked `correct`",
        "",
        "Listed on unique `(page_id, target)` rows. Many may be Sanskrit mantras "
        "or ritual formulas the model cannot reliably judge.",
        "",
        correct_md,
        "",
    ]
    return "\n".join(parts), correct_df, contam_df_uniq


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    needed = {"page_id", "target", "verdict", "error_count", "error_description"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    report, correct_df, contam_df = analyze(df, args)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"Wrote report → {args.report}")

    args.correct_out.parent.mkdir(parents=True, exist_ok=True)
    correct_df[
        [
            "page_id",
            "target",
            "verdict",
            "error_count",
            "error_description",
            "mantra_hint",
        ]
    ].to_csv(args.correct_out, index=False, encoding="utf-8")
    print(f"Wrote correct rows → {args.correct_out}  ({len(correct_df)} rows)")

    if not contam_df.empty:
        contam_path = args.report.with_name(
            args.report.stem + "_contamination_rows.csv"
        )
        contam_df.to_csv(contam_path, index=False, encoding="utf-8")
        print(
            f"Wrote contamination flags → {contam_path}  ({len(contam_df)} rows)"
        )

    n = len(df)
    n_dup = int(df.duplicated(["page_id", "target"]).sum())
    vc = df["verdict"].str.lower().value_counts()
    print(
        f"\nTeaser: rows={n} dup_extras={n_dup} "
        f"correct={vc.get('correct', 0)} incorrect={vc.get('incorrect', 0)} "
        f"unsure={vc.get('unsure', 0)}"
    )


if __name__ == "__main__":
    main()
