"""
Grapheme-safe character-level corruptions for Tibetan.

Drop-in replacements for the buggy functions in tispell_corrupt/char_level.py.
Every function operates on GRAPHEME CLUSTERS (base consonant + its stacked
subjoins + vowel signs), never raw Unicode code points, so a vowel sign can
never be orphaned from its base. Requires the `regex` module (\X = grapheme).

Keeps the same signature: takes List[str] (syllables), returns List[str].
"""
import random
import regex

from .letters import (TIBETAN_LETTER, ROOT_LETTER, ROOT_LETTER_SHORT,
                      HOMOMORPHIC_LETTER, short_to_tall, tall_to_short)


def _clusters(syllable):
    """Split a syllable into extended grapheme clusters."""
    return regex.findall(r"\X", syllable)


def char_random_delete(syllable_list):
    """Delete one whole grapheme cluster from a random multi-cluster syllable."""
    out = list(syllable_list)
    cands = [i for i, s in enumerate(out) if len(_clusters(s)) >= 2]
    if not cands:
        return out
    idx = random.choice(cands)
    cl = _clusters(out[idx])
    del cl[random.randrange(len(cl))]
    out[idx] = "".join(cl)
    return out


def char_random_insert(syllable_list):
    """Insert a random root letter at a cluster BOUNDARY (never mid-cluster)."""
    out = list(syllable_list)
    if not out:
        return out
    idx = random.randrange(len(out))
    cl = _clusters(out[idx])
    # insert only at boundaries between clusters (or ends) so we never split a stack
    pos = random.randint(0, len(cl))
    new_char = random.choice(ROOT_LETTER)      # a bare root letter is a plausible insert
    cl.insert(pos, new_char)
    out[idx] = "".join(cl)
    return out


def char_inner_syllable_exchange(syllable_list):
    """Swap two ADJACENT grapheme clusters within a syllable."""
    out = list(syllable_list)
    cands = [i for i, s in enumerate(out) if len(_clusters(s)) >= 2]
    if not cands:
        return out
    idx = random.choice(cands)
    cl = _clusters(out[idx])
    j = random.randint(0, len(cl) - 2)
    cl[j], cl[j + 1] = cl[j + 1], cl[j]
    out[idx] = "".join(cl)
    return out


def char_near_syllable_exchange(syllable_list):
    """Swap one grapheme cluster between two ADJACENT syllables."""
    out = list(syllable_list)
    if len(out) < 2:
        return out
    i = random.randint(0, len(out) - 2)
    ci, cj = _clusters(out[i]), _clusters(out[i + 1])
    if not ci or not cj:
        return out
    # swap the last cluster of syll i with the first cluster of syll i+1
    ci[-1], cj[0] = cj[0], ci[-1]
    out[i] = "".join(ci)
    out[i + 1] = "".join(cj)
    return out


# --- these two were already fine (single-char replace at valid positions) ---

def char_tall_short_replace(syllable_list):
    out = list(syllable_list)
    if not out:
        return out
    idx = random.randrange(len(out))
    cl = _clusters(out[idx])
    if not cl:
        return out
    j = random.randrange(len(cl))
    # a cluster's base is its first char; replace only if it's a tall/short root
    base = cl[j][0]
    if base in ROOT_LETTER:
        cl[j] = tall_to_short(base) + cl[j][1:]
    elif base in ROOT_LETTER_SHORT:
        cl[j] = short_to_tall(base) + cl[j][1:]
    out[idx] = "".join(cl)
    return out


def char_homomorphic_replace(syllable_list):
    out = list(syllable_list)
    cands = [i for i, s in enumerate(out)
             if any(c in HOMOMORPHIC_LETTER for c in s)]
    if not cands:
        return out
    idx = random.choice(cands)
    s = out[idx]
    positions = [k for k, c in enumerate(s) if c in HOMOMORPHIC_LETTER]
    k = random.choice(positions)
    # replace that homoglyph char with one of its variants
    variants = HOMOMORPHIC_LETTER[s[k]] if isinstance(HOMOMORPHIC_LETTER, dict) else None
    if variants:
        out[idx] = s[:k] + random.choice(variants) + s[k+1:]
    return out