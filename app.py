"""Gradio demo for the BDRC/tibetan-byt5-v12b spell checker."""
import os
import re

import gradio as gr
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

MODEL_ID = "BDRC/tibetan-byt5-v12b"
TOKEN = os.environ.get("HF_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
model = T5ForConditionalGeneration.from_pretrained(MODEL_ID, token=TOKEN)
model.to(device).eval()

SYLLABLE_RUN = re.compile(r"([^་།\s]+་)\1{2,}")
CHAR_RUN = re.compile(r"(.)\1{2,}")
SHAD_GAP = re.compile("།(?![།\\s\"'”’»)\\]」』])(?=\\S)")

EXAMPLES = [
    ["བོད་ཀྱི་སྐད་ཡིག་ནི་ཧ་ཅང་ཡག་པོ་ཡིན།"],
    ["ང་ཚོས་དགེ་བའི་ལས་ལ་འབད་དགོས་ཀྱི་ཡོད།"],
    ["དེ་རིང་གནམ་གཤིས་ཧ་ཅང་བཟང་པོ་འདུག།"],
]


def normalize_tsegs(text: str) -> str:
    text = re.sub(r"\s+་", "་ ", text)
    text = re.sub(r" +", " ", text)
    text = SHAD_GAP.sub("། ", text)
    return text.strip()


def collapse_repeats(text):
    text = SYLLABLE_RUN.sub(r"\1", text)
    return CHAR_RUN.sub(r"\1", text)


def correct_chunk(chunk):
    input_ids = tokenizer(chunk, return_tensors="pt").input_ids.to(device)
    max_new_tokens = int(input_ids.shape[1] * 1.5)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def correct(text):
    parts = re.split(r"(།+)", normalize_tsegs(text))
    result = []
    for part in parts:
        if not part.strip() or part.startswith("།"):
            result.append(part)
            continue
        result.append(collapse_repeats(correct_chunk(part)))
    return normalize_tsegs("".join(result))


demo = gr.Interface(
    fn=correct,
    inputs=gr.Textbox(label="Input Tibetan text", lines=6),
    outputs=gr.Textbox(label="Corrected text", lines=6),
    examples=EXAMPLES,
    title="Tibetan spell checker",
    description=f"Model: {MODEL_ID}",
)

if __name__ == "__main__":
    demo.launch(share=True)
