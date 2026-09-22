import bophono, csv
from collections import defaultdict
from itertools import combinations

conv = bophono.UnicodeToApi(schema="MST", options={"unknownSyllableMarker": True})

lemma_forms = defaultdict(list)
for line in open("/tmp/verb_forms.tsv", encoding="utf-8"):
    form, tag, lemma = line.rstrip("\n").split("\t")[:3]
    lemma_forms[lemma].append((form, tag))

pairs = []
for lemma, forms in lemma_forms.items():
    ipa = {}
    for f, _ in forms:
        try:
            ipa[f] = conv.get_api(f)
        except Exception:
            ipa[f] = None
    for (fa, ta), (fb, tb) in combinations(forms, 2):
        if fa != fb and ta != tb and ipa[fa] and ipa[fa] == ipa[fb]:
            pairs.append((lemma, fa, ta, fb, tb, ipa[fa]))

print(f"{len(pairs)} homophone tense pairs across {len(set(p[0] for p in pairs))} lemmas")
with open("/tmp/verb_homophone_pairs.tsv", "w", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["lemma","form_a","tag_a","form_b","tag_b","ipa"])
    w.writerows(pairs)