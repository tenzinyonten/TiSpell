# Manager follow-up: why a particle error survived and a correct word got replaced

Data: `v12b_resplit/differing_pairs.csv` (206,671 rows; train 198,839 / val 3,993 / test 3,839).
Generator: `scripts/pipeline/generate_v2b.py`, with helpers in `scripts/generate_combined.py` and `tispell_corrupt/`.
No code was changed. All numbers come from scripts run against the CSV (one-off scripts, not saved in the repo).

## Summary

1. **Particle agreement is barely represented in the real data.** Only 166 swap events between two particles of the same class (ཀྱི/གྱི/གི, ཏུ/དུ, ཏེ/དེ/སྟེ) occur in the 18,540 real pairs (0.9%). The other ~1,670 such swaps are synthetic. About 76% of those come from `char_homomorphic_replace`, which swaps one random letter cluster with no regard to agreement.
2. **Agreement is rule-governed for ཀྱི/གྱི/གི and partly for ཏུ/དུ, but not for ཏེ/དེ/སྟེ.** In the clean targets the previous syllable's final letter predicts the genitive 98.7% of the time and ཏུ/དུ 92.9%. It predicts ཏེ/དེ/སྟེ only 67.9%. The real targets also contain lexical exceptions (ཀུན་ཏུ, ཤིན་ཏུ) and some labels that contradict each other.
3. **The "correct word replaced by a different word" problem has a likely training-data cause, and it is broader than whole-word replacement in `real:syllable_level`.** That category does contain whole-syllable swaps, but they are only 765 pairs. Much larger sources teach valid-word-to-valid-word edits:
   - Botok finds that in about half of `real:syllable_level` pairs (50.1% of a 1,500-pair sample) every changed source word is already a dictionary word.
   - The three synthetic types that swap one valid syllable for another (`real_confusion_replace`, `syllable_homophone`, `verb_tense_homophone`) are about 30k first-labelled rows. They are built from symmetric tables, so the model sees both A→B and B→A.
4. **Botok is worth a limited integration, not a full one.** Its syllable parser is useful for a particle rule. Its word-validity signal is too weak to gate edits. Details in section 4.

Caveat on the "real" label: I treat `real:syllable_level`, `real:single_char_substitution` and `real:single_char_insert_delete` as real (18,540 pairs). `real_confusion_replace` is a synthetic type, built from a table of real OCR confusions.

---

## 1. How each error type is created

`diff_category` is not the full list of corruptions in a row. For synthetic rows it holds only the **first** corruption applied (`merge_v12.py:to_trainer`). Sentences receive up to `len(syllables)//9` corruptions (minimum 1), in up to 6 passes over the corruptor list. The per-type probabilities below are applied per pass.

**Synthetic types** (probability per pass in brackets; all operate on the syllable list of a clean, hunspell-validated corpus sentence, and the clean sentence is the target):

| Type | How it is made |
|---|---|
| `char_homomorphic_replace` [0.55] | Picks one letter cluster (a base letter plus its stacked/vowel marks) from a table of real OCR look-alikes (e.g. བ→པ) and swaps it, weighted by how often that confusion occurred in real OCR. Exempt from the structure check. |
| `real_confusion_replace` [0.30] | Replaces a whole syllable with a syllable it was really confused with in OCR output (from `real_syllable_confusions.tsv`, 860 rows; e.g. ཀྱིས→ཀྱི, ལས→ལ, པར→པ), weighted by count. Exempt from the structure check. |
| `syllable_random_merge` [0.08] | Glues a syllable to the next one, deleting the tsheg between them (ཞེས་ཡོངས→ཞེསཡོངས). |
| `syllable_homophone` [0.06] | Replaces a syllable with another valid syllable that sounds the same (bophono IPA match) and is within 2 written edits. The lookup table is symmetric. |
| `verb_tense_homophone` [0.05] | Replaces a verb form with another tense form of the same verb that sounds the same (`verb_homophone_pairs.tsv`, 3,920 rows). Both directions are loaded, so past→present and present→past both occur. |
| `char_random_insert` [0.04] | Inserts one random root letter at a cluster boundary in a random syllable. |
| `char_inner_syllable_exchange` [0.03] | Swaps two adjacent letter clusters inside one syllable. |
| `char_near_syllable_exchange` [0.03] | Swaps the last cluster of one syllable with the first cluster of the next. |
| `char_random_delete` [0.02] | Half the time removes a trailing shad (དང།→དང, the missing-shad error). Otherwise deletes one random cluster from a multi-cluster syllable. |

Rules that apply across types: changes must be at least 2 syllables apart; the non-exempt types are reverted if the result is structurally impossible; 18% of sentences are left uncorrupted on purpose.

**Real types** (from the annotation pipeline, `TiSpell/scripts/prepare_tispell_data/categorize_diffs.py`; first match wins):
- `real:single_char_substitution`: the real OCR text and the corrected text differ by one character substituted.
- `real:single_char_insert_delete`: they differ by one character inserted or deleted.
- `real:syllable_level`: the edit distance is at most 5 characters and at most 30% of the string, and the differences fall entirely on whole tsheg-delimited units. A 2-character fix inside one syllable also qualifies; "syllable level" does not mean "whole word replaced".

**Other labels:**
- `ungrouped` (5,742) and `clean` (1,845): these are sentences that were not corrupted, so source = target except that the source lacks the final tsheg (7,520 of 7,587 rows have `target == source + "་"`). They are effectively identity pairs mislabelled as differing (`TSHEG.join(out)` drops the trailing tsheg, `generate_v2b.py:391`). Side effect: they may teach the model to append a tsheg at the end of a sentence.

Counts by `diff_category` (differing file): char_homomorphic_replace 83,677; syllable_random_merge 25,347; real_confusion_replace 18,785; char_random_insert 14,304; char_random_delete 13,493; real:syllable_level 9,720; char_inner_syllable_exchange 7,360; syllable_homophone 6,696; real:single_char_substitution 6,344; char_near_syllable_exchange 6,141; ungrouped 5,742; verb_tense_homophone 4,741; real:single_char_insert_delete 2,476; clean 1,845.

---

## 2. Edit-size distribution of the real pairs (18,540)

Method: whole-string Levenshtein distance, plus a syllable-level alignment on tsheg boundaries. A pair is "1 character" if the distance is 1. Otherwise it is classified by how many syllables differ. For a single replaced syllable I computed character similarity (difflib ratio) between the old and new syllable: ≥0.5 counts as "partial", <0.5 as "whole syllable replaced". The 0.5 cutoff is my choice and is arbitrary.

| Edit size | `syllable_level` | `single_char_substitution` | `single_char_insert_delete` | All real | Share |
|---|---:|---:|---:|---:|---:|
| 1 character | 0 | 6,344 | 2,476 | **8,820** | 47.6% |
| 1 syllable, partial change (similarity ≥ 0.5) | 2,519 | 0 | 0 | **2,519** | 13.6% |
| 1 syllable inserted/deleted | 1,448 | 0 | 0 | **1,448** | 7.8% |
| 1 syllable, whole replaced (similarity < 0.5) | 765 | 0 | 0 | **765** | 4.1% |
| 2+ syllables | 4,988 | 0 | 0 | **4,988** | 26.9% |
| Total | 9,720 | 6,344 | 2,476 | 18,540 | |

Notes:
- Every `syllable_level` pair has an edit distance of 2 to 5 (2: 5,148; 3: 2,389; 4: 1,421; 5: 762). The category cannot contain large rewrites by construction.
- The 4,988 "2+ syllables" pairs are mostly several small fixes in different syllables, not large replacements.

**Does the suspicion hold?** Partly. Whole-syllable replacement is 765 pairs: 7.9% of `syllable_level`, 4.1% of real pairs and 0.4% of all differing rows. That is not a large share. The 10 examples below (random sample, seed 1) show that only about half are the "valid word swapped for another valid word" pattern:

| # | Changed syllable (source → target) | Context | What it looks like |
|---|---|---|---|
| 1 | སྟོང → རྡོར | དགྱེས་**སྟོང**་འབུམ → དགྱེས་**རྡོར**་འབུམ | valid word → valid word |
| 2 | སྒྲ → ༣ | རིང་པོ་**སྒྲ**་འཁྱོག → རིང་པོ་**༣**་འཁྱོག | OCR digit artifact |
| 3 | དུ → བར | ...འཁོད་**དུ**་གྱུར → ...འཁོད་**བར**་གྱུར | function word → different function word |
| 4 | ཡང → སྤང | རྩོད་པ་**ཡང**་བའོ → རྩོད་པ་**སྤང**་བའོ | valid word → valid word (ཡང is a common particle) |
| 5 | དྷཾ → ཧཾ | **དྷཾ**་ནི་དབུས་སུ → **ཧཾ**་ནི་དབུས་སུ | Sanskrit seed syllable |
| 6 | ཚན → ཆེན | ཡུམ་**ཚན**་མོ → ཡུམ་**ཆེན**་མོ | valid word → valid word |
| 7 | ཆོང → ཚོད | ན་**ཆོང**་ཕྲ་བའི → ན་**ཚོད**་ཕྲ་བའི | OCR-style confusion |
| 8 | རྡི → དྷི | བོ་**རྡི**་སཏྭ → བོ་**དྷི**་སཏྭ | Sanskrit transliteration |
| 9 | རབ༤ → དབང | དང་པོ་**རབ༤**་བཞིའི → དང་པོ་**དབང**་བཞིའི | OCR garbage |
| 10 | ོཔོ → སོ | མཆིས་**ོཔོ** → མཆིས་**སོ** | OCR garbage |

So 4 of 10 (1, 3, 4, 6) teach "replace a valid, contextually plausible word with another". The others are garbage-to-word fixes, which are the intended behaviour.

**A better measure than shape (Botok, section 4):** in a random sample of 1,500 `real:syllable_level` pairs, 50.1% had every changed source word recognised by Botok's dictionary (all real categories: 57.6%). So real over-correction pressure comes mostly from real errors that happen to be dictionary words, whatever the edit size. The model has to use context to decide, and it sees these cases in both directions.

**Larger source of valid-for-valid swaps (synthetic):** the three types that replace one valid syllable with another make up about 30,200 first-labelled rows (14.6% of the file): `real_confusion_replace` 18,785, `syllable_homophone` 6,696, `verb_tense_homophone` 4,741. They are symmetric. I counted, for each type, how many (source syllable → target syllable) edits also appear in the reverse direction elsewhere in the data:

| Type | Edits (same-length rows) | Also seen reversed |
|---|---:|---:|
| `verb_tense_homophone` | 5,944 | 40.4% (e.g. བྱས→བྱེད 225×, བྱེད→བྱས 131×) |
| `real_confusion_replace` | 31,818 | 39.7% (e.g. ཀྱིས→ཀྱི 522×, ལས→ལ 423×, པར→པ 621×, པ→པའི 363×) |
| `syllable_homophone` | 8,681 | 15.6% |

Whenever the same two valid words appear as "wrong → right" in one row and "right → wrong" in another, the model can only choose by context. In ambiguous contexts it will sometimes flip a correct word. This is the most direct data-side explanation I found for problem (2). It is a hypothesis from the data; I have not run the model to confirm it. Also note that `real_confusion_replace` swaps function words and particles (ཀྱིས/ཀྱི, ལས/ལ, པ/པར/པའི), which the particle count in section 3 does not include.

---

## 3. Particle coverage

Particles tracked: genitive ཀྱི གྱི གི ཡི འི; terminative/connective ཏུ དུ ཏེ དེ སྟེ. Standalone syllables only (an attached འི such as རྒྱལ་པོའི is not a separate syllable here and is not counted). A "swap" is a one-syllable-for-one-syllable replacement where both sides are in the list and differ.

| Measure | Count |
|---|---:|
| Pairs where a listed particle changes (any way) | 19,944 (9.6% of all differing) |
| ...of which real pairs | 1,256 (6.8% of real) |
| Particle ↔ non-particle events (garbled into or restored from a non-particle) | 18,647 |
| Particle inserted/deleted events | 418 |
| **Particle ↔ particle swaps (agreement-type errors)** | **1,832** |
| ...of which in real pairs | 166 (0.9% of real pairs) |
| ...of which from `char_homomorphic_replace` | 1,386 (75.6%) |

Other contributors to the 1,832: `real:single_char_substitution` 95, `real_confusion_replace` 91, `real:syllable_level` 66, `char_random_insert` 52, `char_random_delete` 44, `syllable_random_merge` 32, `syllable_homophone` 29. All swaps stay within a class (no genitive↔terminative swaps).

Most frequent swaps (source → target): གྱི→ཀྱི 657; ཏུ→དུ 360; དུ→ཏུ 238; ཀྱི→གྱི 211; ཏེ→དེ 101; གི→གྱི 89; ཏེ→སྟེ 60; དེ→ཏེ 58; སྟེ→ཏེ 51.

Five examples (source → target; the changed particle is marked). The first three are real pairs; the last two are synthetic.

1. real:single_char_substitution, ཏུ→དུ: `དམ་པ་རྣམས་ལ་ཤིན་ཏུ་མངའ་བ་ཡིན།` → `དམ་པ་རྣམས་ལ་ཤིན་དུ་མངའ་བ་ཡིན།` (**the target here arguably looks wrong**; see the next section)
2. real:single_char_substitution, གྱི→ཀྱི: `སྒྱུ་ལུས་གྱི་སྒོས་བཀའ་བྱེད།` → `སྒྱུ་ལུས་ཀྱི་སྒོས་བཀའ་བྱེད།` (a correct agreement fix, after ས)
3. real:single_char_substitution, དུ→ཏུ: `ངག་དུ་བདེན་པའི་ཚིག་འདི་བརྗོད།` → `ངག་ཏུ་བདེན་པའི་ཚིག་འདི་བརྗོད།` (a correct agreement fix, after ག)
4. synthetic real_confusion_replace, གི→གྱི: `...སྐྱེ་འཇིག་ཅན་གི་རྟགས་ལ...` → `...སྐྱེ་འཇིག་ཅན་གྱི་རྟགས་ལ...` (after ན)
5. synthetic char_homomorphic_replace, གི→གྱི: `...རོགས་སྐྱོར་གི་ལས་དོན་སྤེལ་བའི་རེད` → `...རོགས་སྐྱོར་གྱི་ལས་དོན་སྤེལ་བ་རེད` (context window from a long multi-error row; after ར)

### How deterministic is agreement in this data?

I tabulated, for every standalone particle in the 236,717 unique targets, the final letter of the preceding syllable (ག ང ད ན བ མ འ ར ལ ས, or "vowel" if none) against the particle that followed. This is derived from the corpus targets, not from a grammar book.

| Particle set | Majority-rule accuracy on all targets | Notes |
|---|---:|---|
| ཀྱི/གྱི/གི/ཡི | **98.7%** (n=136,325) | Clean split: ཀྱི after ས ད བ; གི after ང ག; གྱི after ན ར ལ མ; ཡི after a vowel (85%) or འ. |
| ཏུ/དུ | **92.9%** (n=60,591) | Clean for ག བ (ཏུ ≈98%) and ང ད ར མ (དུ ≈99.9%); ambiguous after ན (དུ 82%) and ལ (དུ 72%). |
| ཏེ/དེ/སྟེ | **67.9%** (n=67,972) | Not predictable from the final letter alone; ད is the only strong case (དེ 99.5%). |

Checked on the particle swaps in the data (target side of the swap, previous-syllable rule): genitive 99.5% (n=955), ཏུ/དུ 88.4% (n=596), ཏེ/དེ/སྟེ 64.0% (n=239). That check is circular for synthetic rows, because their targets also fed the table. The real-only subset is less circular but small: genitive 87.5% (n=40), ཏུ/དུ 77.2% (n=92), ཏེ/དེ/སྟེ 34.4% (n=32).

Two problems for a pure rule:
- **Lexical exceptions.** ཀུན་ཏུ (ན would normally give དུ) appears in 670 targets and ཀུན་དུ in 434. Real targets have ཤིན་ཏུ 353 times and ཤིན་དུ 19. A letter rule would "fix" correct ཀུན་ཏུ and ཤིན་ཏུ.
- **Inconsistent real labels.** Five real pairs change ཤིན་ཏུ→ཤིན་དུ (example 1 above), while another real pair changes ཤིན་དུ→ཤིན་ཏུ. The first direction is the minority spelling in the corpus. So part of the real data teaches the model to apply the mechanical rule where the lexical form is correct. This is label noise worth cleaning, and it may contribute to the over-editing in problem (2).

For problem (1) the numbers say the model has very little real agreement signal to learn from (166 events, 0.9%). The synthetic signal (1,666 events) exists, but it comes from random cluster swaps, so the particle is corrupted in the same way as any other letter, and the model learns that "particle looks odd" without a strong cue for which form is right. I did not run the model, so I cannot say which of these explains the specific miss your manager saw.

---

## 4. Botok check

I tested the Botok in the project venv (`venv`, version 0.9.0; the bundled trie was built for 1.1.6, which prints a version warning; PyPI's latest is 1.1.6). Loading takes about 3.5 seconds.

### What it offers that is relevant

- **Tokenization and POS tagging** (`WordTokenizer`), with a dictionary built from ACTib, Tsikchen and other sources. Tokens carry `pos`, `lemma`, `affix_host` and `affix`. With `split_affixes=True` it splits attached particles (རྒྱལ་པོའི → རྒྱལ་པོ + འི).
- **Particle table** (`resources/particles.tsv`). All genitive forms (གི ཀྱི གྱི འི ཡི) share one lemma, as do ལ-forms (སུ ར རུ ཏུ ན ལ དུ) and སྟེ-forms (སྟེ ཏེ དེ). It lists forms but has no agreement rules.
- **Syllable parser** (`botok.textunits.sylcomponents.SylComponents`): `get_parts("ཚིག")` → `('ཚ', 'ིག')`, root letter, `is_thame` and a check against Classical Tibetan syllable norms (`SylComponents.json`). This can give the final letter of the previous syllable.

### (a) Deterministic particle-agreement rule

Botok does not enforce agreement. In my test it tagged both ཐམས་ཅད་**ཀྱི** and ཐམས་ཅད་**གྱི** as `PART` with no warning. It supplies the ingredients (tokenization, particle identification, syllable parts) but you would write the rule yourself.

The rule is feasible for the genitive (98.7% rule accuracy), reasonable for ཏུ/དུ with an exception list, and not feasible for ཏེ/དེ/སྟེ. A rule that covered genitive only, with a small exception list (ཀུན་ཏུ, ཤིན་ཏུ, and others to be found), would be a post-processing step with high precision. Even for the genitive, the table above is the best evidence available; I would validate it against a held-out set of real pairs before relying on it. A simpler way to get the final letter is the last character of the syllable, so Botok's parser is a convenience, not a necessity, for this rule.

### (b) Checking whether a changed word was already valid

Botok marks unknown words with `pos == "NON_WORD"` (e.g. ཞེསཡོངས་ in the missing-tsheg example). I measured how well that separates wrong from correct (random samples, 1,500 per row):

| Sample | Result |
|---|---|
| Correct sentences (identical pairs): tokens tagged NON_WORD | 1.0% of tokens; 8.5% of sentences have at least one |
| Wrong words in real pairs (changed source tokens): NON_WORD | 28.5% |
| Wrong words in `real:syllable_level`: NON_WORD | 32.2% |
| Real pairs where **every** changed source word is a valid dictionary word | 57.6% (`syllable_level`: 50.1%) |
| Wrong words in synthetic pairs: NON_WORD | 32.9% |
| Correct target words in real pairs tagged NON_WORD | 15.8% |

What this means:
- If used as a hard gate ("only allow the model to change a word Botok calls NON_WORD"), it would block roughly 70% of genuine real corrections, because most real errors produce a word that exists in the dictionary (for example ཚན vs ཆེན, ཉིས vs གཉིས, which Botok tags as a normal NUM).
- It would also let through nonsense Botok accepts (མི་**ད་པ**་ was tagged as an adverb).
- 16% of correct real target words are NON_WORD (Sanskrit transliteration, rare words, OCR-style spellings), so a veto-on-NON_WORD target check would also reject good corrections.

So Botok can say "this token is not in my dictionary", which is useful as one soft feature. It cannot say "this word is correct in this context", which is what over-editing prevention needs. A softer use that might help: treat a model edit as suspicious when both the source and the new word are dictionary words and the POS or lemma of the source fits its neighbours, and require higher model confidence in that case. That needs the model's per-token scores, which I did not look at.

### Is it worth integrating?

- **Particle rule: yes, as a narrow post-processor** for the genitive (and possibly ཏུ/དུ with exceptions), written against the final-letter table, with Botok used for tokenization and for the affixed འི case. The main cost is maintaining an exception list. The dependency is heavy for a rule that needs only a syllable splitter; a ~20-line own implementation would do the same job.
- **Validity gate: not as a hard filter.** The dictionary agrees with about 30% of genuine errors and misses real-word errors, so it would suppress true corrections. A soft signal is possible but unproven.
- Practical cost: a new runtime dependency with a multi-second start-up and a dictionary download on first use. The installed version and trie version differ (0.9.0 vs 1.1.6).

---

## 5. Suggested next steps (not done; your call)

1. Get the model's actual output on your manager's two failing sentences, and run a few hundred identical (clean) sentences through it to measure the over-edit rate directly. This would confirm or reject the hypotheses above.
2. Clean the conflicting real labels (ཤིན་ཏུ/ཤིན་དུ and similar). Check how many real pairs apply the mechanical particle rule against the corpus-majority spelling.
3. Consider making the symmetric synthetic types one-directional (only produce the less common form from the more common one), so the same valid pair is not both "wrong" and "right".
4. Fix the 7,520 identity rows labelled `ungrouped`/`clean` that differ only by the trailing tsheg (move them to the identical file, or restore the tsheg in the source).
5. If particles matter, add targeted agreement-aware synthetic errors for ཀྱི/གྱི/གི and ཏུ/དུ, since the existing particle swaps come from random cluster swaps.
