import re

import torch

from .model import DEVICE, get_model

TSHEG = "\u0f0b"

CHUNKS = re.compile(r"[^།]*།+|[^།]+")
VOWEL_FIX = {"འཾ": "འི", "འྀ": "འི"}

def normalize(text):
    syls, out, i = text.split(TSHEG), [], 0
    while i < len(syls):
        j = i
        while j + 1 < len(syls) and syls[j + 1] == syls[i]:
            j += 1
        out.extend(syls[i:min(j, i + 1) + 1])
        i = j + 1
    text = re.sub(r'(.)\1{3,}', r'\1', TSHEG.join(out))
    for bad, good in VOWEL_FIX.items():
        text = text.replace(bad, good)
    return text

BATCH_SIZE = 32

def max_new_tokens_for(chunk):
    return int(len(chunk.encode("utf-8")) * 1.5)

def generate_batch(chunks, device):
    """One generate() call over chunks. Every chunk keeps its own length cap
    on the result, so a chunk never gets more tokens than it would alone."""
    tok, m = get_model()
    caps = [max_new_tokens_for(c) for c in chunks]
    enc = tok(chunks, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=max(caps), num_beams=3,
                         repetition_penalty=1.1, early_stopping=True)
    return [tok.decode(row[:cap + 1], skip_special_tokens=True).replace(" ", "")
            for row, cap in zip(out, caps)]

def correct_chunks(chunks, device=DEVICE, batch_size=BATCH_SIZE, progress=None):
    """Correct many chunks in batches, shortest first so padding stays small.
    Returns (outputs, failed): outputs[i] is None where chunk i failed. If a
    batch raises, its chunks are retried one by one."""
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i].encode("utf-8")))
    outs = [None] * len(chunks)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        try:
            for i, o in zip(idx, generate_batch([chunks[i] for i in idx], device)):
                outs[i] = o
        except Exception:
            for i in idx:
                try:
                    outs[i] = generate_batch([chunks[i]], device)[0]
                except Exception:
                    pass
        if progress is not None:
            progress(len(idx))
    return outs, [i for i, o in enumerate(outs) if o is None]

def correct_chunk(chunk, device=DEVICE):
    return generate_batch([chunk], device)[0]

SHAD_AFTER = re.compile("།(?![།\\s\"'”’»)\\]」』])(?=\\S)")

def space_after_shad(text):
    return SHAD_AFTER.sub("། ", text)

def correct_lines(lines, device=DEVICE, normalize_output=True, add_shad_space=False,
                  batch_size=BATCH_SIZE, progress=None):
    """Correct a list of lines. Returns (results, failed): results[i] is the
    corrected line, or the original line if any of its chunks failed, and
    failed lists those line indexes."""
    pieces = [CHUNKS.findall(line) for line in lines]
    todo, owner = [], []
    for n, parts in enumerate(pieces):
        for part in parts:
            if part.strip():
                todo.append(part)
                owner.append(n)
    outs, bad_chunks = correct_chunks(todo, device, batch_size, progress)
    failed = sorted({owner[i] for i in bad_chunks})
    done = iter(outs)
    results = []
    for n, parts in enumerate(pieces):
        got = [next(done) if part.strip() else part for part in parts]
        if n in failed:
            results.append(lines[n])
            continue
        out = "".join(got)
        if normalize_output:
            out = normalize(out)
        if add_shad_space:
            out = space_after_shad(out)
        results.append(out)
    return results, failed

def correct(texts, device=DEVICE, normalize_output=True, add_shad_space=False,
            batch_size=BATCH_SIZE):
    split = [t.split("\n") for t in texts]
    flat = [line for lines in split for line in lines]
    results, failed = correct_lines(flat, device, normalize_output,
                                    add_shad_space, batch_size)
    if failed:
        raise RuntimeError(f"{len(failed)} line(s) failed to correct")
    it = iter(results)
    return ["\n".join(next(it) for _ in lines) for lines in split]
