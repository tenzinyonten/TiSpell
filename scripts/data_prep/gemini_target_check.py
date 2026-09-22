#!/usr/bin/env python3
"""Ask Gemini YES/NO whether each training target is valid Tibetan. Flags the YESes.

Targets are packed --batch at a time into one numbered request, which is what
makes a full 150k-row pass affordable. Every reply is checked for one verdict
per input line; a batch that comes back misaligned is bisected and retried, so
a garbled reply costs a few extra calls instead of silently shifting verdicts
onto the wrong rows. Verdicts are cached per target, so runs resume for free.
"""
import argparse, os, sys, json, re, time, logging, threading, unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit("pip install google-genai")

# google-genai warns once per call about thought_signature parts we don't use.
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

nfc = lambda s: unicodedata.normalize("NFC", str(s))

RULES = """You are a strict Tibetan (Uchen) orthographic proofreader specializing in classical, Dunhuang, legal, and Vajrayana liturgical texts.
YOUR ONLY TASK: Determine if the input text contains GENUINE physical spelling/orthographic errors.
STRICT CONSTRAINTS (DO NOT FLAG AS ERRORS):
1. Old Tibetan Orthography (Dunhuang style): ya-btags after labials (e.g. myed for med) are VALID.
2. Classical / Ritual Jargon: Abhidharma, Dzogchen, Tantra terms are VALID.
3. Archaic particle usage and incomplete text fragments are VALID. Do not check grammar or syntax.
4. Non-standard tsheg placements or sentence fragments are NOT spelling errors.
5. Rare Lineage & Administrative Vocabulary: Historical legal terms (e.g. mag-gyos) and ritual offering items (e.g. gyad) are VALID.
6. A truncated or clause-final fragment is NOT an error. Missing associative or case particles are GRAMMAR, not spelling. Do not flag either.
7. Do NOT flag the following valid classical Tibetan text patterns as errors:
   - Anusvara/bindu (ཾ) used as nasal shorthand (e.g., སེཾས, གསུཾ, རྒྷཾས).
   - Scribal abbreviations replacing final གས with reversed ཌ (e.g., ཐུཌ, ཕྱོཌ, ཚོཌ).
   - Classical repetition or abbreviation marks (༴).
   - Valid Sanskrit mantras and transliterated character stacks (e.g., ཨོཾ, ཧྲཱིཿ, བཛྲ, ཕཊ).
GENUINE ERRORS TO FLAG:
- Impossible root-letter stacks (broken Tibetan glyph topology).
- Wrong, missing, or swapped subjoined letters/vowels that violate script construction.
- Blatant typos forming invalid Tibetan syllables, including a syllable duplicated in place."""

SINGLE_PROMPT = RULES + """
Reply ONLY "YES" if there is a real physical spelling error.
Reply ONLY "NO" if the spelling and character stacks are valid.
No explanations."""

BATCH_PROMPT = RULES + """
You will receive N numbered Tibetan texts, one per line.
Reply with exactly N lines, nothing else, in the same order:
<number>: YES   (that text has a real physical spelling error)
<number>: NO    (that text is orthographically valid)
Judge every line independently. Do not merge, skip, reorder, or explain."""

VERDICT_RE = re.compile(r"(\d+)\s*[:.\)-]*\s*(YES|NO)", re.I)


def make_config(prompt, no_thinking, max_tokens):
    """A YES/NO verdict needs no reasoning tokens, and thinking is the main
    cost driver here. Not every model accepts a zero budget, hence probe()."""
    kw = {}
    if no_thinking:
        kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(
        system_instruction=prompt, temperature=0,
        max_output_tokens=max_tokens, **kw)


def probe(client, model, prompt, max_tokens):
    """Return whether the model accepts a zero thinking budget."""
    for no_thinking in (True, False):
        try:
            client.models.generate_content(
                model=model, contents="1. \u0f56\u0f0b",
                config=make_config(prompt, no_thinking, max_tokens))
            print(f"  thinking {'disabled' if no_thinking else 'on (not disableable)'}")
            return no_thinking
        except Exception as e:
            if not no_thinking:
                sys.exit(f"model {model} rejected a basic request: {e}")
    return False


def call(client, model, config, body):
    for a in range(5):
        try:
            r = client.models.generate_content(model=model, contents=body,
                                               config=config)
            return (r.text or "").strip()
        except Exception:
            time.sleep(min(2 ** a, 8))
    return None


def parse_one(text):
    """Last standalone YES/NO in the reply — thinking models bury it."""
    hits = re.findall(r"\b(YES|NO)\b", (text or "").upper())
    return hits[-1] if hits else "?"


def ask_batch(client, model, config, single_config, texts):
    """Return one verdict per text. Bisects on a misaligned reply."""
    if len(texts) == 1:
        out = call(client, model, single_config, texts[0])
        if out is None:
            return ["ERR"]
        return [parse_one(out)]

    # Newlines would break the one-text-per-line contract.
    body = "\n".join(f"{i + 1}. {t.replace(chr(10), ' ')}"
                     for i, t in enumerate(texts))
    out = call(client, model, config, body)
    if out is not None:
        got = {}
        for num, v in VERDICT_RE.findall(out):
            idx = int(num) - 1
            if 0 <= idx < len(texts) and idx not in got:
                got[idx] = v.upper()
        if len(got) == len(texts):
            return [got[i] for i in range(len(texts))]

    mid = len(texts) // 2
    return (ask_batch(client, model, config, single_config, texts[:mid])
            + ask_batch(client, model, config, single_config, texts[mid:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--sample", type=int, default=0, help="0 = all rows")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--batch", type=int, default=50, help="targets per request")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="gemini_target_flagged.csv")
    ap.add_argument("--cache", default="data/cache/gemini_target_cache.jsonl")
    args = ap.parse_args()
    if "GEMINI_API_KEY" not in os.environ: sys.exit("export GEMINI_API_KEY=...")

    df = pd.concat([pd.read_csv(p) for p in args.data], ignore_index=True)
    if args.skip: df = df.iloc[args.skip:].copy()
    if args.sample: df = df.head(args.sample).copy()
    df[args.target_col] = df[args.target_col].fillna("").map(nfc)

    cache = {}
    if os.path.exists(args.cache):
        for line in open(args.cache):
            try:
                r = json.loads(line); cache[r["t"]] = r["v"]
            except Exception:
                pass

    uniq = list(dict.fromkeys(df[args.target_col]))
    todo = [t for t in uniq if t and t not in cache]
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"{len(df):,} rows | {len(uniq):,} unique | {len(uniq) - len(todo):,} cached "
          f"| {len(todo):,} to fetch in {len(batches):,} calls of {args.batch} "
          f"| {args.workers} workers | {args.model}")

    if batches:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        batch_tokens = 12 * args.batch + 64
        no_think = probe(client, args.model, BATCH_PROMPT, batch_tokens)
        cfg = make_config(BATCH_PROMPT, no_think, batch_tokens)
        single = make_config(SINGLE_PROMPT, no_think, 4)
        lock = threading.Lock()
        t0 = time.time()
        n_done = 0
        with open(args.cache, "a") as cf, \
                ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(ask_batch, client, args.model, cfg, single, b): b
                    for b in batches}
            for k, fut in enumerate(as_completed(futs), 1):
                b = futs[fut]
                for t, v in zip(b, fut.result()):
                    with lock:
                        cache[t] = v
                        if v in ("YES", "NO"):
                            cf.write(json.dumps({"t": t, "v": v},
                                                ensure_ascii=False) + "\n")
                    n_done += 1
                with lock:
                    cf.flush()
                if k % 10 == 0 or k == len(batches):
                    rate = n_done / max(time.time() - t0, 1e-9)
                    eta = (len(todo) - n_done) / max(rate, 1e-9)
                    print(f"  {k}/{len(batches)} calls, {n_done:,}/{len(todo):,} "
                          f"targets, {rate:.0f}/s, eta {eta/60:.1f}m")

    df["gemini_valid"] = [cache.get(t, "ERR") for t in df[args.target_col]]
    n = len(df)
    bad = int(df.gemini_valid.isin(("ERR", "?")).sum())
    yes = int((df.gemini_valid == "YES").sum())
    print(f"\nGemini says HAS_ERROR: {yes:,}/{n:,} = {yes/n:.2%}"
          + (f"   ({bad:,} unanswered)" if bad else ""))
    for c in ("source_corpus", "diff_category"):
        if c in df.columns:
            r = df.groupby(c).gemini_valid.apply(lambda s: (s == "YES").mean())
            print(f"  by {c}: " + ", ".join(f"{k} {v:.2%}" for k, v in r.items()))

    flagged = df[df.gemini_valid == "YES"]
    keep = [c for c in (args.target_col, "source", "source_corpus",
                        "diff_category", "source_sentence_id", "gemini_valid")
            if c in flagged.columns]
    flagged[keep].to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(flagged):,} rows)")
    print("\n--- flagged examples (eyeball: really bad, or Gemini over-flagging?) ---")
    for t in flagged[args.target_col].head(10):
        print(t)


if __name__ == "__main__":
    main()
