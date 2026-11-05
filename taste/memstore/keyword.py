"""Keyword matching for full-text search: one tokeniser, one stemmer, one scorer, all pinned here.

Nothing here knows about git. A baseline that keeps the same text somewhere else can import this
module and rank by exactly the same rules, so that a difference between the two is a difference in
what they stored and never in how they matched.

* **Tokens.** Lowercase, split at anything that is not a letter or a digit, then stem. An underscore
  splits; camelCase does not. There is no stop list: rare words outweigh common ones through the
  score.
* **Stemmer.** Porter's 1980 algorithm with the three departures of his own reference code: a
  ``logi`` rule, ``bli`` in place of ``abli``, and words of one or two letters left alone. Written
  here rather than imported, so that its behaviour cannot move with a library's version.
* **Score.** Okapi BM25 with k1 1.5 and b 0.75, the idf ``ln(1 + (N - n + 0.5) / (n + 0.5))``. Each
  distinct query word counts once. Ties go to the document added last: for a memory, the newest of
  two equal matches is the natural one to read first.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable
from functools import lru_cache

K1 = 1.5
B = 0.75

WORD = re.compile(r"[^\W_]+")


def tokens(text: str) -> list[str]:
    """The stemmed words of ``text``, in order."""
    return [stem(word.lower()) for word in WORD.findall(text)]


# ------------------------------------------------------------------ Porter's stemmer


def _consonant(word: str, i: int) -> bool:
    letter = word[i]
    if letter in "aeiou":
        return False
    if letter == "y":
        return i == 0 or not _consonant(word, i - 1)
    return True


def _measure(word: str) -> int:
    """The m of [C](VC)^m[V]: how many times a vowel run is followed by a consonant run."""
    shape = "".join("c" if _consonant(word, i) else "v" for i in range(len(word)))
    return shape.count("vc")


def _has_vowel(word: str) -> bool:
    return any(not _consonant(word, i) for i in range(len(word)))


def _double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _consonant(word, len(word) - 1)


def _cvc(word: str) -> bool:
    """Ends consonant, vowel, consonant, where the last is not w, x or y."""
    n = len(word)
    return (
        n >= 3
        and _consonant(word, n - 3)
        and not _consonant(word, n - 2)
        and _consonant(word, n - 1)
        and word[-1] not in "wxy"
    )


def _positive(word: str) -> bool:
    return _measure(word) > 0


def _above_one(word: str) -> bool:
    return _measure(word) > 1


_Rule = tuple[str, str, Callable[[str], bool]]


def _apply(word: str, rules: Iterable[_Rule]) -> str:
    """The first rule whose suffix matches decides, whether or not its condition holds."""
    for suffix, replacement, condition in rules:
        if word.endswith(suffix):
            base = word[: len(word) - len(suffix)]
            return base + replacement if condition(base) else word
    return word


_STEP_2: tuple[_Rule, ...] = tuple(
    (suffix, replacement, _positive)
    for suffix, replacement in (
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"),
        ("bli", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
        ("ization", "ize"), ("ation", "ate"), ("ator", "ate"), ("alism", "al"),
        ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
        ("iviti", "ive"), ("biliti", "ble"), ("logi", "log"),
    )
)
_STEP_3: tuple[_Rule, ...] = tuple(
    (suffix, replacement, _positive)
    for suffix, replacement in (
        ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"),
        ("ful", ""), ("ness", ""),
    )
)
_STEP_4: tuple[_Rule, ...] = (
    ("al", "", _above_one), ("ance", "", _above_one), ("ence", "", _above_one),
    ("er", "", _above_one), ("ic", "", _above_one), ("able", "", _above_one),
    ("ible", "", _above_one), ("ant", "", _above_one), ("ement", "", _above_one),
    ("ment", "", _above_one), ("ent", "", _above_one),
    ("ion", "", lambda base: _measure(base) > 1 and base[-1:] in ("s", "t")),
    ("ou", "", _above_one), ("ism", "", _above_one), ("ate", "", _above_one),
    ("iti", "", _above_one), ("ous", "", _above_one), ("ive", "", _above_one),
    ("ize", "", _above_one),
)


def _step_1a(word: str) -> str:
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _step_1b(word: str) -> str:
    if word.endswith("eed"):
        base = word[:-3]
        return base + "ee" if _positive(base) else word
    for suffix in ("ed", "ing"):
        if word.endswith(suffix):
            base = word[: -len(suffix)]
            if _has_vowel(base):
                return _after_1b(base)
            return word
    return word


def _after_1b(base: str) -> str:
    if base.endswith(("at", "bl", "iz")):
        return base + "e"
    if _double_consonant(base):
        return base[:-1] if base[-1] not in "lsz" else base
    if _measure(base) == 1 and _cvc(base):
        return base + "e"
    return base


def _step_1c(word: str) -> str:
    return word[:-1] + "i" if word.endswith("y") and _has_vowel(word[:-1]) else word


def _step_5(word: str) -> str:
    if word.endswith("e"):
        base = word[:-1]
        if _measure(base) > 1 or (_measure(base) == 1 and not _cvc(base)):
            word = base
    if word.endswith("ll") and _measure(word[:-1]) > 1:
        word = word[:-1]
    return word


@lru_cache(maxsize=1 << 18)
def stem(word: str) -> str:
    """The Porter stem of one lowercase word."""
    if len(word) <= 2:
        return word
    word = _step_1c(_step_1b(_step_1a(word)))
    word = _apply(_apply(_apply(word, _STEP_2), _STEP_3), _STEP_4)
    return _step_5(word)


# ------------------------------------------------------------------ the score


class Bm25:
    """An inverted index that grows by whole documents and ranks them on demand."""

    def __init__(self) -> None:
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._lengths: list[int] = []
        self._total_length = 0

    def __len__(self) -> int:
        return len(self._lengths)

    def add(self, terms: Iterable[str]) -> int:
        """Index one document's terms and return its number."""
        doc = len(self._lengths)
        counts = Counter(terms)
        for term, frequency in counts.items():
            self._postings.setdefault(term, []).append((doc, frequency))
        length = sum(counts.values())
        self._lengths.append(length)
        self._total_length += length
        return doc

    def rank(self, query: Iterable[str], limit: int | None = None) -> list[tuple[int, float]]:
        """(document, score) for the documents holding a query word, best first."""
        documents = len(self._lengths)
        if not documents:
            return []
        average = self._total_length / documents
        scores: dict[int, float] = {}
        # dict.fromkeys, not set: the sum of floats depends on its order, and a set's order
        # changes with the process's hash seed.
        for term in dict.fromkeys(query):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (documents - len(postings) + 0.5) / (len(postings) + 0.5))
            for doc, frequency in postings:
                norm = K1 * (1 - B + B * self._lengths[doc] / average) if average else K1
                scores[doc] = scores.get(doc, 0.0) + idf * frequency * (K1 + 1) / (frequency + norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], -item[0]))
        return ranked if limit is None else ranked[:limit]
