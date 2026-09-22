"""Correct Tibetan sentences with the semi-mask model (default)."""
import argparse

import torch
from transformers import AutoTokenizer

from model.tispell_roberta import TiSpell_RoBERTa_SemiMask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="weights/tispell_roberta_best")
    p.add_argument("--device", default="cpu")
    p.add_argument("text", nargs="+", help="Tibetan sentence(s) to correct")
    args = p.parse_args()

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = TiSpell_RoBERTa_SemiMask(args.model, tok)
    weights = f"{args.model}/tispell_roberta.pth"
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    enc = tok(
        args.text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=384,
    ).to(device)
    with torch.no_grad():
        logits, _ = model(**enc)
    pred = tok.batch_decode(logits.argmax(-1), skip_special_tokens=True)
    for src, out in zip(args.text, pred):
        print(src)
        print(out.replace(" ", ""))
        print()


if __name__ == "__main__":
    main()
