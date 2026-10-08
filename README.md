# tibetan-spellcheck

Tibetan spell checker built on [BDRC/tibetan-byt5-v12b](https://huggingface.co/BDRC/tibetan-byt5-v12b).
One model load, three ways in: Python, command line, REST API.

The model is private. Set `HF_TOKEN` before the first run. The tokenizer is
loaded from `google/byt5-small`. CUDA is used when available.

## Install

    pip install -e .

This installs only the runtime packages and takes torch unpinned, so you get
a build that matches your CUDA. Do not install from `requirements.txt`, it is
the training file and pins `torch==2.4.1+cu118`.

## Python

    from tibetan_spellcheck import correct
    correct(["ཀ་ཁ།"], add_shad_space=False)

## Command line

    tibetan-spellcheck input.txt -o output.txt
    tibetan-spellcheck --text "ཇི་བཞིན་..."
    tibetan-spellcheck input.txt --add-shad-space

The input can be any length. Line breaks are kept, so the output has the same
number of lines. A line that fails is copied through unchanged and listed in
the summary. A progress bar shows while it runs.

## REST API

    uvicorn tibetan_spellcheck.api:app --host 0.0.0.0 --port 8000

    curl -X POST localhost:8000/correct -H 'Content-Type: application/json' \
         -d '{"text": "ཀ་ཁ།", "add_shad_space": false}'
    # {"corrected": "..."}

`GET /health` returns the model id and device. The model loads once at startup.

## What correct() does

Each line is split on shad, each piece is corrected on its own (3 beams,
repetition penalty 1.1, `max_new_tokens` = 1.5 times the piece's UTF-8 bytes),
then a small normalizer runs: runs of 3 or more repeated syllables become 2,
and `འཾ` and `འྀ` become `འི`. `add_shad_space` adds one space after a shad
that is directly followed by text. It is off by default.
