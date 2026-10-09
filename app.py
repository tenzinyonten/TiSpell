"""Gradio demo for the BDRC/tibetan-byt5-v12b spell checker."""
import os
import re

import gradio as gr
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

try:
    from botok.utils.corpus_normalization import normalize_corpus
except ImportError:
    normalize_corpus = None

MODEL_ID = "BDRC/tibetan-byt5-v12b"
TOKEN = os.environ.get("HF_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
model = T5ForConditionalGeneration.from_pretrained(MODEL_ID, token=TOKEN)
model.to(device).eval()

SYLLABLE_RUN = re.compile(r"([^་།\s]+་)\1{2,}")
CHAR_RUN = re.compile(r"(.)\1{2,}")

EXAMPLES = [
    ["བཀྲ་ཤིས་ཀྱིས་དཔེ་དེབ་གསར་པ་ཞིགཉོས་བྱུང་།", False],
    ["གཞོན་སྐྱེས་རྣམས་སློབ་གྲྭ་ཁག་དུ་འགྲོ་བཞིན་ཡོད།", False],
    ["བོད་གྱི་རིག་གཞུང་ནི་ལོ་ངོ་སྟོང་ཕྲག་མང་པོའི་རིང་ལ་དར་ཞིང་རྒྱས།", False],
    ["རི་མོ་འདི་ནི་ཤིན་ཏུམཛེས་པོ་ཞིག་འདུག", False],
    ["བད་ཀྱི་ཐེག་པ་ཆེན་པོའི་ཆོས་སྐཏ་རིགས།", False],
]


def collapse_repeats(text):
    text = SYLLABLE_RUN.sub(r"\1", text)
    return CHAR_RUN.sub(r"\1", text)


def correct_chunk(chunk):
    input_ids = tokenizer(chunk, return_tensors="pt").input_ids.to(device)
    # Keep decode length near the input so the model cannot pad the
    # chunk with extra syllables (len*1.5 was enough room to hallucinate).
    # +2 still produced དང་པ; *1.1 stopped that on short chunks.
    n = input_ids.shape[1]
    max_new_tokens = max(1, int(n * 1.1))
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def correct_line(line):
    parts = re.split(r"(།+)", line)
    result = []
    for i, part in enumerate(parts):
        if not part.strip() or part.startswith("།"):
            result.append(part)
            continue
        corrected = collapse_repeats(correct_chunk(part))
        nxt = parts[i + 1] if i + 1 < len(parts) else ""
        if nxt.startswith("།"):
            corrected = corrected.rstrip("།")
        result.append(corrected)
    return "".join(result)


def correct(text):
    # Split on lines first so paragraph breaks survive; shad-split only
    # inside each line, then rejoin with the original newlines.
    return "\n".join(correct_line(line) for line in text.split("\n"))


def run(text, use_botok):
    corrected = correct(text)
    if use_botok and normalize_corpus is not None:
        corrected = normalize_corpus(corrected)
    return corrected


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(label="Input Tibetan text", lines=6),
        gr.Checkbox(
            label="Botok normalization",
            value=False,
            interactive=normalize_corpus is not None,
            info=None if normalize_corpus is not None else "botok is not installed",
        ),
    ],
    outputs=gr.Textbox(label="Corrected text", lines=6),
    examples=EXAMPLES,
    title="Tibetan spell checker",
    description=f"Model: {MODEL_ID}",
)

if __name__ == "__main__":
    demo.launch(share=True)
