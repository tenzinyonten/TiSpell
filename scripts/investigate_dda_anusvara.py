#!/usr/bin/env python3
"""Characterize systematic DDA/anusvara substitutions in hf_differing.

Steps 1-3 only: report evidence. Does NOT modify any CSV.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Optional, Set

import pandas as pd

DDA = "\u0F4C"  # Tibetan letter DDA
ANUSVARA = "\u0F7E"  # anusvara
GAS = "\u0F42\u0F66"  # GA + SA
MA = "\u0F58"  # MA

TSHEG_SPLIT = re.compile(r"[\u0F0B\u0F0C\u0F0D\u0F0E\u0F0F\u0F10\u0F11\u0F14\s]+")

SANSKRIT_MARKERS_EXCL_ANUSVARA = {
    "\u0F71", "\u0F73", "\u0F75", "\u0F76", "\u0F77", "\u0F78", "\u0F79",
    "\u0F80", "\u0F81", "\u0F7F", "\u0F82", "\u0F83", "\u0F84",
}
SANSKRIT_RETROFLEX_EXCL_DDA = {
    "\u0F4A", "\u0F4B", "\u0F4D", "\u0F4E", "\u0F4F", "\u0F69", "\u0F6A",
}
SANSKRIT_SUBJOINED = {
    "\u0F9A", "\u0F9B", "\u0F9C", "\u0F9D", "\u0F9E", "\u0F9F",
    "\u0FB9", "\u0FBA", "\u0FBB", "\u0FBC",
}
KNOWN_MANTRA_SYLS = {
    "\u0F67\u0F71\u0F74\u0F7E",  # hung with anusvara
    "\u0F67\u0F71\u0F74\u0F83",  # hung with nada
    "\u0F68\u0F7C\u0F7E",        # om
    "\u0F68\u0F71\u0F7C\u0F83",  # om long
    "\u0F68\u0F71\u0F7F",        # ah visarga
    "\u0F66\u0FAD\u0F71\u0F67\u0F71",  # swaha (approx)
    "\u0F55\u0F4A\u0F84",        # phat + halanta
    "\u0F55\u0F4A",              # phat
}

DEFAULT_DIR = Path("dataset/ocr_annotation/exports/hf_differing")
DEFAULT_REPORT = Path(
    "dataset/ocr_annotation/exports/dda_anusvara_investigation.md"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--top_n", type=int, default=40)
    return p.parse_args()


def load_botok():
    from botok import WordTokenizer
    return WordTokenizer()


def botok_is_word(wt, syl: str) -> bool:
    if not syl:
        return False
    text = syl if syl.endswith("\u0F0B") else syl + "\u0F0B"
    try:
        toks = wt.tokenize(text)
    except Exception:
        return False
    text_toks = [t for t in toks if getattr(t, "chunk_type", None) == "TEXT"]
    if len(text_toks) != 1:
        return False
    t = text_toks[0]
    pos = (t.pos or "").upper()
    if pos in {"NON_WORD", "UNK", ""}:
        return False
    core = t.text.rstrip("\u0F0B\u0F0C")
    return core == syl.rstrip("\u0F0B\u0F0C")


def syllables(text: str):
    return [s for s in TSHEG_SPLIT.split(str(text)) if s]


def is_sanskrit_context(syl: str, ignore_chars: Optional[Set[str]] = None) -> bool:
    ignore = ignore_chars or set()
    core = syl.rstrip("\u0F0B\u0F0C")
    if core in KNOWN_MANTRA_SYLS:
        return True
    for m in KNOWN_MANTRA_SYLS:
        if m in syl:
            return True
    for ch in syl:
        if ch in ignore:
            continue
        if ch in SANSKRIT_MARKERS_EXCL_ANUSVARA:
            return True
        if ch in SANSKRIT_RETROFLEX_EXCL_DDA:
            return True
        if ch in SANSKRIT_SUBJOINED:
            return True
    if sum(1 for ch in syl if "\u0F90" <= ch <= "\u0FBC") >= 2:
        return True
    return False


def propose_dda_sub(syl: str) -> Optional[str]:
    if DDA not in syl:
        return None
    return syl.replace(DDA, GAS)


def propose_anusvara_sub(syl: str) -> Optional[str]:
    if ANUSVARA not in syl:
        return None
    return syl.replace(ANUSVARA, MA)


def pct(n, d) -> str:
    if not d:
        return "n/a"
    return f"{100.0 * n / d:.1f}%"


def analyse_char(
    frames,
    char: str,
    name: str,
    propose: Callable,
    wt,
    ignore_for_sanskrit: Set[str],
    top_n: int,
) -> dict:
    form_counts: Counter = Counter()
    col_occ = {"source": 0, "target": 0}
    row_with = {"source": Counter(), "target": Counter()}
    row_total = Counter()
    sanskrit_occ = 0
    native_occ = 0
    form_meta = {}

    for split, df in frames.items():
        row_total[split] = len(df)
        for col in ("source", "target"):
            series = df[col].astype(str)
            has = series.str.contains(char, regex=False)
            row_with[col][split] = int(has.sum())
            for text in series[has]:
                col_occ[col] += text.count(char)
                for syl in syllables(text):
                    if char not in syl:
                        continue
                    form_counts[syl] += 1
                    skt = is_sanskrit_context(syl, ignore_chars=ignore_for_sanskrit)
                    proposed = propose(syl)
                    if syl not in form_meta:
                        form_meta[syl] = {
                            "proposed": proposed,
                            "valid_before": botok_is_word(wt, syl),
                            "valid_after": botok_is_word(wt, proposed) if proposed else False,
                            "sanskrit": skt,
                            "n": 0,
                        }
                    form_meta[syl]["n"] += 1
                    # Keep Sanskrit flag if any occurrence looks Sanskrit
                    form_meta[syl]["sanskrit"] = form_meta[syl]["sanskrit"] or skt

    # Recount occ with final form_meta sanskrit flags
    sanskrit_occ = 0
    native_occ = 0
    for syl, n in form_counts.items():
        if form_meta[syl]["sanskrit"]:
            sanskrit_occ += n
        else:
            native_occ += n

    sub_valid = sub_invalid = already_word_before = sub_sanskrit_skipped = 0
    examples_valid, examples_invalid, examples_sanskrit = [], [], []

    for syl, n in form_counts.most_common():
        meta = form_meta[syl]
        if meta["sanskrit"]:
            sub_sanskrit_skipped += n
            if len(examples_sanskrit) < 25:
                examples_sanskrit.append((syl, meta["proposed"], n))
            continue
        if meta["valid_before"]:
            already_word_before += n
        if meta["valid_after"]:
            sub_valid += n
            if len(examples_valid) < 30:
                examples_valid.append((syl, meta["proposed"], n))
        else:
            sub_invalid += n
            if len(examples_invalid) < 30:
                examples_invalid.append((syl, meta["proposed"], n))

    return {
        "name": name,
        "char": char,
        "codepoint": f"U+{ord(char):04X}",
        "col_occ": col_occ,
        "row_with": row_with,
        "row_total": row_total,
        "n_forms": len(form_counts),
        "form_counts": form_counts,
        "form_meta": form_meta,
        "sanskrit_occ": sanskrit_occ,
        "native_occ": native_occ,
        "sub_valid": sub_valid,
        "sub_invalid": sub_invalid,
        "sub_sanskrit_skipped": sub_sanskrit_skipped,
        "already_word_before": already_word_before,
        "examples_valid": examples_valid,
        "examples_invalid": examples_invalid,
        "examples_sanskrit": examples_sanskrit,
        "native_total": native_occ,
        "top_n": top_n,
    }


def render_char_section(stats: dict) -> str:
    lines = []
    lines.append(f"## Character: {stats['char']} ({stats['codepoint']}) — {stats['name']}")
    lines.append("")
    tot_rows = sum(stats["row_total"].values())
    lines.append("### Frequency (rows containing the character)")
    lines.append("")
    lines.append("| split | n rows | source has char | % | target has char | % |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split in ("train", "validation", "test"):
        n = stats["row_total"].get(split, 0)
        ns = stats["row_with"]["source"].get(split, 0)
        nt = stats["row_with"]["target"].get(split, 0)
        lines.append(f"| {split} | {n} | {ns} | {pct(ns, n)} | {nt} | {pct(nt, n)} |")
    ns_all = sum(stats["row_with"]["source"].values())
    nt_all = sum(stats["row_with"]["target"].values())
    lines.append(
        f"| **all** | **{tot_rows}** | **{ns_all}** | **{pct(ns_all, tot_rows)}** | "
        f"**{nt_all}** | **{pct(nt_all, tot_rows)}** |"
    )
    lines.append("")
    lines.append(
        f"Raw character occurrences: source={stats['col_occ']['source']}, "
        f"target={stats['col_occ']['target']}"
    )
    lines.append(f"Distinct syllable forms containing the char: **{stats['n_forms']}**")
    lines.append("")
    lines.append("### Sanskrit vs native context (per syllable occurrence)")
    lines.append("")
    tot = stats["sanskrit_occ"] + stats["native_occ"]
    lines.append(
        f"- Sanskrit-context occurrences: **{stats['sanskrit_occ']}** "
        f"({pct(stats['sanskrit_occ'], tot)})"
    )
    lines.append(
        f"- Native-context occurrences: **{stats['native_occ']}** "
        f"({pct(stats['native_occ'], tot)})"
    )
    lines.append(
        "  (Detector ignores the character under study, so anusvara is not "
        "circularly counted as Sanskrit.)"
    )
    lines.append("")
    lines.append("### Substitution -> botok lexicon (native context only)")
    lines.append("")
    nt = stats["native_total"]
    lines.append(
        f"- Native occurrences where substitution yields a botok word: "
        f"**{stats['sub_valid']}** ({pct(stats['sub_valid'], nt)})"
    )
    lines.append(
        f"- Native occurrences where substitution yields NON_WORD / nonsense: "
        f"**{stats['sub_invalid']}** ({pct(stats['sub_invalid'], nt)})"
    )
    lines.append(
        f"- Native occurrences already a botok word *before* substitution: "
        f"**{stats['already_word_before']}** ({pct(stats['already_word_before'], nt)})"
    )
    lines.append(
        f"- Sanskrit-context occurrences (must preserve): "
        f"**{stats['sub_sanskrit_skipped']}**"
    )
    lines.append("")
    lines.append(f"### Top {stats['top_n']} syllable forms")
    lines.append("")
    lines.append("| n | form | proposed | skt? | word_before | word_after |")
    lines.append("|---:|---|---|---|---|---|")
    for syl, n in stats["form_counts"].most_common(stats["top_n"]):
        m = stats["form_meta"][syl]
        lines.append(
            f"| {n} | {syl} | {m['proposed'] or ''} | {m['sanskrit']} | "
            f"{m['valid_before']} | {m['valid_after']} |"
        )
    lines.append("")
    lines.append("### Examples: native, substitution VALID")
    lines.append("")
    for syl, prop, n in stats["examples_valid"][:20]:
        lines.append(f"- `{syl}` -> `{prop}`  (n={n})")
    lines.append("")
    lines.append("### Examples: native, substitution INVALID")
    lines.append("")
    if stats["examples_invalid"]:
        for syl, prop, n in stats["examples_invalid"][:20]:
            lines.append(f"- `{syl}` -> `{prop}`  (n={n})")
    else:
        lines.append("_none in sample_")
    lines.append("")
    lines.append("### Examples: Sanskrit context (must keep)")
    lines.append("")
    if stats["examples_sanskrit"]:
        for syl, prop, n in stats["examples_sanskrit"][:20]:
            lines.append(f"- `{syl}` (would become `{prop}`, n={n})")
    else:
        lines.append("_none found_")
    lines.append("")
    return "\n".join(lines)


def decision_block(stats: dict) -> str:
    nt = stats["native_total"] or 1
    valid_rate = stats["sub_valid"] / nt
    support = valid_rate >= 0.90 and stats["sub_valid"] > 100
    tot = stats["sanskrit_occ"] + stats["native_occ"]
    return (
        f"- Native substitution->valid rate: **{pct(stats['sub_valid'], stats['native_total'])}** "
        f"({stats['sub_valid']}/{stats['native_total']})\n"
        f"- Sanskrit-context share of occurrences: "
        f"**{pct(stats['sanskrit_occ'], tot)}**\n"
        f"- Hypothesis support (>=90% native->valid and n>100): "
        f"**{'YES' if support else 'NO / WEAK'}**\n"
    )


def main() -> None:
    args = parse_args()
    splits = {}
    for name in ("train", "validation", "test"):
        path = args.data_dir / f"{name}.csv"
        if not path.is_file():
            raise SystemExit(f"Missing {path}")
        splits[name] = pd.read_csv(path, dtype=str, keep_default_na=False)
        print(f"Loaded {name}: {len(splits[name])} rows")

    print("Loading botok...")
    wt = load_botok()

    print("Analysing DDA...")
    dda = analyse_char(
        splits, DDA, "DDA / proposed DDA->GAS",
        propose_dda_sub, wt, {DDA}, args.top_n,
    )
    print("Analysing anusvara...")
    anus = analyse_char(
        splits, ANUSVARA, "anusvara / proposed anusvara->MA",
        propose_anusvara_sub, wt, {ANUSVARA}, args.top_n,
    )

    repo_notes = """## 3. Existing codebase conventions

Searched TiSpell for normalisation / codepoint maps / transliteration tables.

| Location | Finding |
|---|---|
| `scripts/prepare_tispell_data/tibetan_normalize.py` | Maps CHE MGO->digit7 and paluta->digit3. **No** DDA/anusvara mapping. |
| `datachecker.py` | Treats **both** anusvara (U+0F7E) and DDA (U+0F4C) as *Sanskrit markers* and **skips** those syllables in native orthography checks. That assumes they are always legitimate transliteration — which conflicts with the OCR-substitution hypothesis for native words. |
| `alignment_transfer.py` | Mentions anusvara only inside a leading-debris combining-mark class. No substitution. |
| Docs / other maps | No deliberate DDA->GAS or anusvara->MA convention found. |

**Conclusion from repo search:** this is **not** an intentional documented normalisation. The closest related code *protects* these characters from being flagged as errors by classifying them as Sanskrit.
"""

    parts = [
        "# Investigation: DDA (U+0F4C) and anusvara (U+0F7E) systematic substitutions",
        "",
        f"Data: `{args.data_dir}/{{train,validation,test}}.csv` (hf_differing).",
        "",
        "Proposed substitutions (NOT applied in this report):",
        "",
        "- DDA (U+0F4C) -> GA+SA (U+0F42 U+0F66)",
        "- anusvara (U+0F7E) -> MA (U+0F58)",
        "",
        "botok WordTokenizer used as lexicon oracle (NON_WORD => invalid).",
        "",
        "## Summary decision",
        "",
        "### DDA -> GAS",
        decision_block(dda),
        "### anusvara -> MA",
        decision_block(anus),
        "",
        repo_notes,
        "",
        render_char_section(dda),
        render_char_section(anus),
        "## Hard constraints reminder",
        "",
        "- No CSV was modified by this script.",
        "- Validation/test must not be silently rewritten: four prior runs "
        "(0.7725 / 0.7878 / 0.7897 / 0.7804) used the current validation targets.",
        "",
    ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(parts), encoding="utf-8")
    print(f"\nWrote {args.report}")
    for label, st in (("DDA", dda), ("ANUSVARA", anus)):
        print(
            f"{label}: native_valid={st['sub_valid']}/{st['native_total']} "
            f"({pct(st['sub_valid'], st['native_total'])}), "
            f"sanskrit={st['sanskrit_occ']}, forms={st['n_forms']}"
        )


if __name__ == "__main__":
    main()
