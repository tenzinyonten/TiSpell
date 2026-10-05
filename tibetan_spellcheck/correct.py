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

def correct_chunk(chunk, device):
    tok, m = get_model()
    enc = tok(chunk, return_tensors="pt").to(device)
    cap = int(len(chunk.encode("utf-8")) * 1.5)
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=cap, num_beams=3,
                         repetition_penalty=1.1, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True).replace(" ", "")

SHAD_AFTER = re.compile("།(?![།\\s\"'”’»)\\]」』])(?=\\S)")

def space_after_shad(text):
    return SHAD_AFTER.sub("། ", text)

def correct_line(line, device):
    parts = [c if not c.strip() else correct_chunk(c, device)
             for c in CHUNKS.findall(line)]
    return "".join(parts)

def correct(texts, device=DEVICE, normalize_output=True, add_shad_space=False):
    results = []
    for text in texts:
        lines = []
        for line in text.split("\n"):
            out = correct_line(line, device)
            if normalize_output:
                out = normalize(out)
            if add_shad_space:
                out = space_after_shad(out)
            lines.append(out)
        results.append("\n".join(lines))
    return results
