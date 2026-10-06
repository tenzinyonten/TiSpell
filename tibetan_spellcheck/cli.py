"""tibetan-spellcheck: correct a text file or a string with the v12b model."""
import argparse
import sys
import time
from pathlib import Path

from tqdm import tqdm

from .correct import BATCH_SIZE, CHUNKS, correct, correct_lines
from .model import DEVICE, get_model


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="tibetan-spellcheck",
                                description="Tibetan spell checker (BDRC/tibetan-byt5-v12b).")
    p.add_argument("input", nargs="?", type=Path, help="input .txt file")
    p.add_argument("-o", "--output", type=Path,
                   help="output .txt file (default: <input>.corrected.txt)")
    p.add_argument("--text", help="correct this string and print the result")
    p.add_argument("--add-shad-space", action="store_true",
                   help="add a space after a shad when text follows it")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help="chunks per generate() call (1 gives the unbatched result)")
    args = p.parse_args(argv)
    if (args.input is None) == (args.text is None):
        p.error("give either an input file or --text")
    return args


def main(argv=None):
    args = parse_args(argv)
    get_model()

    if args.text is not None:
        print(correct([args.text], add_shad_space=args.add_shad_space)[0])
        return 0

    raw = args.input.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        print("[warn] input is not valid UTF-8, bad bytes replaced", file=sys.stderr)
        text = raw.decode("utf-8", errors="replace")

    lines = text.split("\n")
    start = time.time()
    with tqdm(desc="chunks", unit="chunk") as bar:
        bar.total = sum(1 for l in lines for part in CHUNKS.findall(l) if part.strip())
        out, bad = correct_lines(lines, add_shad_space=args.add_shad_space,
                                 batch_size=args.batch_size, progress=bar.update)
    failed = [n + 1 for n in bad]
    for n in failed[:10]:
        print(f"[warn] line {n} failed, kept as is", file=sys.stderr)
    elapsed = time.time() - start

    dest = args.output or args.input.with_suffix(".corrected.txt")
    dest.write_text("\n".join(out), encoding="utf-8")

    changed = sum(1 for a, b in zip(lines, out) if a != b)
    print(f"device: {DEVICE}, batch size: {args.batch_size}")
    print(f"lines: {len(lines)} in, {len(out)} out")
    print(f"changed: {changed}, failed: {len(failed)}"
          + (f" (lines {failed[:10]}{'...' if len(failed) > 10 else ''})" if failed else ""))
    print(f"time: {elapsed:.1f}s")
    print(f"wrote {dest}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
