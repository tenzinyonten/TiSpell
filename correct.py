import os, re, torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

TSHEG = "\u0f0b"
MODEL_ID = "BDRC/tibetan-byt5-v12b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("google/byt5-small")
m = T5ForConditionalGeneration.from_pretrained(
    MODEL_ID, token=os.environ.get("HF_TOKEN")).to(DEVICE).eval()

CHUNKS = re.compile(r"[^\n།]*།+|[^\n།]+|\n")
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
    enc = tok(chunk, return_tensors="pt").to(device)
    cap = int(len(chunk.encode("utf-8")) * 1.5)
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=cap, num_beams=3,
                         repetition_penalty=1.1, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True).replace(" ", "")

def correct(texts, device=DEVICE, normalize_output=True):
    results = []
    for text in texts:
        parts = [c if c == "\n" or not c.strip() else correct_chunk(c, device)
                 for c in CHUNKS.findall(text)]
        out = "".join(parts)
        results.append(normalize(out) if normalize_output else out)
    return results
