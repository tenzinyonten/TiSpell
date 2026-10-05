"""Loads the v12b model and the byt5-small tokenizer once and shares them."""
import os
import threading

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

MODEL_ID = "BDRC/tibetan-byt5-v12b"
TOKENIZER_ID = "google/byt5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_lock = threading.Lock()
_loaded = None


def get_model():
    global _loaded
    with _lock:
        if _loaded is None:
            tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
            model = T5ForConditionalGeneration.from_pretrained(
                MODEL_ID, token=os.environ.get("HF_TOKEN")).to(DEVICE).eval()
            _loaded = (tok, model)
    return _loaded
