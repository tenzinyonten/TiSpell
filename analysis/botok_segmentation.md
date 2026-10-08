# Botok as a segment tokenizer for the spell checker

Assessment only; nothing integrated. Tested against a clone of `Webuddhist-tech/Botok` (HEAD `449e454`, July 2026) on 150 multi-line passages built from real OCR lines in `v12b_resplit` (8 lines per passage, avg 337 chars).

## Short answer

- **`corpus_normalization.py` is not a segmenter.** It cleans characters. The one function that does segment-like work (`normalize_for_perplexity`) deletes tshegs and rewrites shads, so it cannot feed the model.
- **Botok does have a sentence tokenizer** (`botok/tokenizers/sentencetokenizer.py`). It gives larger, more sentence-like chunks than the app's shad split, but it still depends on a shad being present, and some of its chunks (7 of 574 here, up to 949 characters) came out longer than the 380 characters the model was trained on.
- **Recommendation: don't integrate it for this.** A cheap merge of adjacent shad pieces up to a length cap would capture most of the benefit with no dependency. Reasons and numbers below.

## 1. What `corpus_normalization.py` does

All functions are string-to-string cleanup; none returns boundaries.

| Function | What it does |
|---|---|
| `normalize_spaces` | Collapses repeated newlines and spaces, strips spaces beside newlines, maps Unicode spaces to ASCII. Optionally removes a space after a tsheg or before one. |
| `normalize_corpus` | NFC, converts line-break variants to `\n`, removes zero-width characters and control characters, normalizes spaces, applies `normalize_unicode`, maps ༌ (U+0F0C) to ་ and ༎ (U+0F0E) to two shads. |
| `merge_lines` | Joins a word-wrapped passage into one line (removes `\n`, fixes tsheg runs and letter-at-line-end). Meant for page OCR where newlines are not sentence boundaries. |
| `normalize_for_perplexity` | Builds a space-separated token string for LM perplexity: removes **all tshegs**, turns every punctuation/space run into ` ། `, replaces digits with `D`, strips non-Tibetan characters, splits case affixes (རྒྱལའི → རྒྱལ འི), optionally spaces out Sanskrit syllables. |

Example from the test run: `མཆོད་པ་དང་གཏོར་མ་བྱིན་རླབས་བྱས་ཏེ་གོང་ལྟར་འབུལ།` becomes `མཆོད པ དང གཏོར མ བྱིན རླབས བྱས ཏེ གོང ལྟར འབུལ །`.

Consequences for this app:

- `normalize_for_perplexity` is unusable as model input. The model was trained on tsheg-delimited text, and the output could not be mapped back to the user's text.
- `normalize_corpus` is mild and could be applied as input cleanup (zero-width characters, odd spaces, CRLF). It would silently change user text, including control characters and U+0F0C/U+0F0E, so the output would differ from the input in places the model never touched. Worth considering only as a deliberate, documented pre-clean, separate from segmentation.
- `merge_lines` does the opposite of what the app does today. The app splits on `\n` to keep paragraph breaks; `merge_lines` removes them. It is only relevant if the input is hard-wrapped OCR where line ends are not sentence ends.

## 2. Botok's actual segmentation

`get_sentence_indices(tokens)` runs after word tokenization and POS tagging, and applies these rules in order:

1. Ending particle (གོ ངོ དོ ནོ བོ མོ འོ རོ ལོ སོ ཏོ) followed by punctuation.
2. Clause-boundary words (སྟེ ཏེ དེ ནས ན ལ ཞིང ཅིག ཤོག) followed by punctuation.
3. Verb followed by punctuation (excluding པ/བ-type nominalizers).
4. Sentences without a verb are joined to a neighbour.

Every rule requires a punctuation token. Text with no shads is returned as one chunk, the same as the current app (confirmed on a 3-sentence run-on string). So it does not segment un-punctuated text.

## 3. Measured comparison with the current shad/newline split

150 passages, 1,220 shad-delimited pieces under the current app logic:

| | App (newline, then shad) | Botok sentence tokenizer |
|---|---:|---:|
| Chunks | 1,220 | 574 |
| Median chunk length (chars) | 31 | 63 |
| Mean / p90 / max | 40 / 71 / 320 | 88 / 182 / 949 |
| Chunks over 380 chars | 0 | 7 |
| Time for all 150 passages | ~0 s | 2.3 s |
| Exact text round trip | n/a | 150 of 150 (chunks joined equal the input) |

For reference, training targets in `v12b_resplit` have a median of 56 characters (p10 31, p90 134); real-pair targets have a median of 34 (p90 83).

What this says:

- **Chunk size.** The app's median chunk (31 chars) is shorter than the typical training sentence (56). Botok's median (63) is closer to the training distribution, so it would give the model slightly more context per call, which is relevant to agreement errors that depend on neighbouring words. The cost is the long tail: 7 chunks beyond the training length, up to 949 chars.
- **Boundary quality.** In the sample, Botok mostly grouped consecutive shad pieces into larger units; it split inside an existing shad piece in only 2 of 1,220 cases. So it rarely finds a boundary the shad split misses; it mainly merges. I did not check by hand whether the merges are linguistically right. In the example I looked at (`ཀ་ར་དྱུག་འགྱུར་རྟ་ལ་བསྐྱོན། གཞན་ཡང་རྒྱ་ཊོ་གཡན་པ་དང།`) it joined two lines that the shad split keeps separate, and in this verse text that is a coin flip.
- **Dependence on input quality.** The boundaries come from POS tags, and the POS tagger sees the misspelled input. A mis-tagged word could move a boundary. I did not measure how often this happens; the 2 extra splits suggest it is rare on this sample.

## 4. Integration cost

| | |
|---|---|
| Python dependencies | Declared: `pyyaml`, `requests`. Light. |
| Data download | On first use Botok downloads a dictionary pack from GitHub (`Esukhia/botok-data` releases). A deployed Space needs network at start-up or the pack baked into the image. |
| Start-up | About 3.5 s to load the dictionary trie (measured with the Botok 0.9.0 installed in the project venv; the pack is built for 1.1.6, which prints a version warning). |
| Speed | 2.3 s for about 50k characters of tokenization plus sentence logic (about 20k chars/s). Small next to ByT5 generation, but not zero for long inputs. |
| Version risk | The fork reports itself as 0.9.0 in `botok/vars.py`. The Botok installed in the project venv is 0.9.0 and does **not** contain `corpus_normalization.py` at all; PyPI 1.1.6 does. Pin an explicit version or a git commit if you ever use it. |
| Maintenance | Boundary changes would change what the model sees for every request; every future Botok release could shift the chunking. The existing shad split is stable and testable. |
| Failure modes to handle | Chunks over the trained length (need a fallback split), text with no shads (no help), and the round-trip guarantee holds in my test but should be asserted. |

## 5. Recommendation

Botok's sentence tokenizer is not worth integrating for chunking. The gain is bigger chunks, not better boundaries, and that gain is available without it:

1. **Merge adjacent shad pieces** until a character cap near the training median-to-p90 range (for example 100 to 130 characters), never crossing a newline. This gives Botok-like chunk sizes with no dependency, no download and no tagger in the loop.
2. If boundaries on un-shaded text matter, that is a different problem (Botok does not solve it either); evaluate it on real un-punctuated OCR pages.
3. If you want an input clean-up step, consider `normalize_corpus` on its own, with the change-tracking implications noted above, but evaluate it separately from segmentation.

To decide between the options with evidence, the missing measurement is the model's accuracy per chunking strategy. Run the model over the same passages with (a) the current split, (b) the merged-pieces split, and (c) Botok, and compare against the corrected targets. I have not run the model, so I make no claim about accuracy.
