"""High-confidence subpatterns: before NON_WORD + after WORD; ANU+SA cluster."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from botok import WordTokenizer

DDA, ANU, GAS, MA, SA = "\u0F4C", "\u0F7E", "\u0F42\u0F66", "\u0F58", "\u0F66"
SPLIT = re.compile(r"[\u0F0B\u0F0C\u0F0D\u0F0E\u0F0F\u0F10\u0F11\u0F14\s]+")
wt = WordTokenizer()

SKT_MARK = set(
    "\u0F71\u0F73\u0F75\u0F76\u0F77\u0F78\u0F79\u0F80\u0F81\u0F7F\u0F82\u0F83\u0F84"
)
SKT_RET = set("\u0F4A\u0F4B\u0F4D\u0F4E\u0F4F\u0F69\u0F6A")
MANTRA = {
    "\u0F68\u0F7C\u0F7E",
    "\u0F67\u0F71\u0F74\u0F7E",
    "\u0F67\u0F71\u0F74\u0F83",
    "\u0F68\u0F71\u0F7F",
    "\u0F68\u0F71\u0F7C\u0F83",
    "\u0F55\u0F4A",
    "\u0F55\u0F4A\u0F84",
}


def classify(syl: str) -> str:
    text = syl if syl.endswith("\u0F0B") else syl + "\u0F0B"
    toks = [t for t in wt.tokenize(text) if getattr(t, "chunk_type", None) == "TEXT"]
    if len(toks) != 1:
        return "MULTI_OR_EMPTY"
    pos = (toks[0].pos or "").upper()
    core = syl.rstrip("\u0F0B\u0F0C")
    if toks[0].text.rstrip("\u0F0B\u0F0C") != core:
        return "MISMATCH"
    if pos in {"NON_WORD", "UNK", ""}:
        return "NON_WORD"
    return "WORD"


def skt(syl: str, ignore: set[str]) -> bool:
    core = syl.rstrip("\u0F0B\u0F0C")
    if core in MANTRA or any(m in syl for m in MANTRA):
        return True
    for ch in syl:
        if ch in ignore:
            continue
        if ch in SKT_MARK or ch in SKT_RET:
            return True
    if sum(1 for ch in syl if "\u0F90" <= ch <= "\u0FBC") >= 2:
        return True
    return False


def gather(df: pd.DataFrame, char: str) -> Counter:
    c: Counter = Counter()
    for col in ("source", "target"):
        for text in df[col].astype(str):
            if char not in text:
                continue
            for syl in SPLIT.split(text):
                if char in syl:
                    c[syl] += 1
    return c


def bucket(forms: Counter, pred, ignore: set[str], label: str) -> None:
    buckets: dict[tuple[str, str], int] = defaultdict(int)
    examples: dict[tuple[str, str], list] = defaultdict(list)
    skt_n = 0
    for syl, n in forms.items():
        if skt(syl, ignore):
            skt_n += n
            continue
        prop = pred(syl)
        b, a = classify(syl), classify(prop)
        buckets[(b, a)] += n
        if len(examples[(b, a)]) < 8:
            examples[(b, a)].append((n, syl, prop))
    tot = sum(buckets.values())
    print(f"\n=== {label}  native_n={tot} skt={skt_n} ===")
    for k in sorted(buckets, key=lambda x: -buckets[x]):
        pct = 100 * buckets[k] / tot if tot else 0
        print(f"  {k[0]:14s} -> {k[1]:14s}  n={buckets[k]:6d} ({pct:5.1f}%)")
        for n, s, p in sorted(examples[k], key=lambda x: -x[0])[:5]:
            print(f"      ex n={n}  {s} -> {p}")
    hc = buckets[("NON_WORD", "WORD")]
    print(f"  HIGH_CONF (NON_WORD->WORD): {hc}/{tot} = {100 * hc / tot if tot else 0:.1f}%")
    aw = sum(v for (b, a), v in buckets.items() if a == "WORD")
    print(f"  AFTER_WORD (any before): {aw}/{tot} = {100 * aw / tot if tot else 0:.1f}%")


def apply_hc(text: str, kind: str) -> tuple[str, int]:
    if kind == "dda":
        char, pred, ignore = DDA, lambda s: s.replace(DDA, GAS), {DDA}
    elif kind == "anu_sa":
        char, pred, ignore = ANU, lambda s: s.replace(ANU, MA), {ANU}
    else:
        raise ValueError(kind)
    if char not in text:
        return text, 0
    parts = SPLIT.split(text)
    seps = SPLIT.findall(text)
    out: list[str] = []
    nchg = 0
    for i, syl in enumerate(parts):
        ns = syl
        if kind == "anu_sa":
            ok = char in syl and (ANU + SA) in syl and syl.count(ANU) == 1
        else:
            ok = (
                char in syl
                and syl.count(DDA) == 1
                and syl.rstrip("\u0F0B\u0F0C").endswith(DDA)
            )
        if ok and not skt(syl, ignore):
            prop = pred(syl)
            if classify(syl) == "NON_WORD" and classify(prop) == "WORD":
                ns = prop
                nchg += 1
        out.append(ns)
        if i < len(seps):
            out.append(seps[i])
    return "".join(out), nchg


def main() -> None:
    base = Path(
        "/Users/tenzinyonten/dharmaduta-ml/TiSpell/dataset/ocr_annotation/exports/hf_differing"
    )
    df = pd.concat(
        [
            pd.read_csv(base / f"{n}.csv", dtype=str, keep_default_na=False)
            for n in ("train", "validation", "test")
        ],
        ignore_index=True,
    )
    dda, anu = gather(df, DDA), gather(df, ANU)

    bucket(dda, lambda s: s.replace(DDA, GAS), {DDA}, "DDA all -> GAS")
    final_dda = Counter(
        {
            s: n
            for s, n in dda.items()
            if s.count(DDA) == 1 and s.rstrip("\u0F0B\u0F0C").endswith(DDA)
        }
    )
    bucket(final_dda, lambda s: s.replace(DDA, GAS), {DDA}, "DDA single-final -> GAS")

    bucket(anu, lambda s: s.replace(ANU, MA), {ANU}, "ANU all -> MA")
    anu_sa = Counter(
        {s: n for s, n in anu.items() if (ANU + SA) in s and s.count(ANU) == 1}
    )
    bucket(anu_sa, lambda s: s.replace(ANU, MA), {ANU}, "ANU: pattern *ANU+SA* (single)")
    final_a = Counter(
        {
            s: n
            for s, n in anu.items()
            if s.count(ANU) == 1 and s.rstrip("\u0F0B\u0F0C").endswith(ANU)
        }
    )
    bucket(final_a, lambda s: s.replace(ANU, MA), {ANU}, "ANU: single-final")

    print("\n=== Row-level impact estimate (train only, high-conf syl replace) ===")
    train = pd.read_csv(base / "train.csv", dtype=str, keep_default_na=False)
    for kind in ("dda", "anu_sa"):
        src_ch = tgt_ch = both = newly_id = 0
        for _, row in train.iterrows():
            s0, t0 = row["source"], row["target"]
            s1, ns = apply_hc(s0, kind)
            t1, nt = apply_hc(t0, kind)
            if ns:
                src_ch += 1
            if nt:
                tgt_ch += 1
            if ns or nt:
                both += 1
            if s0 != t0 and s1 == t1:
                newly_id += 1
        print(
            f"{kind}: rows_src={src_ch} rows_tgt={tgt_ch} rows_any={both} "
            f"newly_identical={newly_id} / {len(train)}"
        )

    anu_tot = sum(anu.values())
    anu_skt = sum(n for s, n in anu.items() if skt(s, {ANU}))
    print(
        f"\nANU syllable occurrences: total={anu_tot} skt={anu_skt} "
        f"({100 * anu_skt / anu_tot:.1f}%) native={anu_tot - anu_skt} "
        f"({100 * (anu_tot - anu_skt) / anu_tot:.1f}%)"
    )
    dda_tot = sum(dda.values())
    dda_skt = sum(n for s, n in dda.items() if skt(s, {DDA}))
    print(
        f"DDA syllable occurrences: total={dda_tot} skt={dda_skt} "
        f"({100 * dda_skt / dda_tot:.1f}%) native={dda_tot - dda_skt} "
        f"({100 * (dda_tot - dda_skt) / dda_tot:.1f}%)"
    )


if __name__ == "__main__":
    main()
