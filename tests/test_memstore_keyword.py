"""The tokeniser, stemmer and scorer behind full-text search.

Two self-run keyword results on one benchmark differed by 11 points from tokenising and stemming
alone, so these are pinned by known answers and not left to a library's defaults.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest

from taste.memstore.keyword import K1, B, Bm25, stem, tokens

# Every pair is a word and its final stem. The words are the examples in Porter's 1980 paper,
# "An algorithm for suffix stripping", carried through all of the algorithm's steps.
PORTER_EXAMPLES = {
    "caresses": "caress", "ponies": "poni", "ties": "ti", "caress": "caress", "cats": "cat",
    "feed": "feed", "agreed": "agre", "plastered": "plaster", "bled": "bled",
    "motoring": "motor", "sing": "sing", "conflated": "conflat", "troubled": "troubl",
    "sized": "size", "hopping": "hop", "tanned": "tan", "falling": "fall", "hissing": "hiss",
    "fizzed": "fizz", "failing": "fail", "filing": "file", "happy": "happi", "sky": "sky",
    "relational": "relat", "conditional": "condit", "rational": "ration", "valenci": "valenc",
    "hesitanci": "hesit", "digitizer": "digit", "radicalli": "radic", "differentli": "differ",
    "vileli": "vile", "analogousli": "analog", "vietnamization": "vietnam",
    "predication": "predic", "operator": "oper", "feudalism": "feudal", "decisiveness": "decis",
    "hopefulness": "hope", "callousness": "callous", "formaliti": "formal",
    "sensitiviti": "sensit", "triplicate": "triplic", "formative": "form", "formalize": "formal",
    "electriciti": "electr", "electrical": "electr", "hopeful": "hope", "goodness": "good",
    "revival": "reviv", "allowance": "allow", "inference": "infer", "airliner": "airlin",
    "gyroscopic": "gyroscop", "adjustable": "adjust", "defensible": "defens", "irritant": "irrit",
    "replacement": "replac", "adjustment": "adjust", "dependent": "depend", "adoption": "adopt",
    "homologous": "homolog", "communism": "commun", "activate": "activ", "angulariti": "angular",
    "effective": "effect", "bowdlerize": "bowdler", "probate": "probat", "rate": "rate",
    "cease": "ceas", "controll": "control", "roll": "roll", "generalizations": "gener",
    "oscillators": "oscil",
}


@pytest.mark.parametrize(("word", "expected"), PORTER_EXAMPLES.items())
def test_the_stemmer_gives_the_papers_answers(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_the_stemmer_keeps_the_reference_codes_three_departures_from_the_paper() -> None:
    # From Porter's own notes on the reference implementation: a "logi" rule, "bli" for "abli",
    # and no stemming of words of one or two letters.
    assert stem("analogi") == "analog"
    assert stem("conformabli") == "conform"
    assert stem("is") == "is" and stem("as") == "as"


def test_the_stemmer_agrees_with_the_reference_mode_of_nltk() -> None:
    nltk_stem = pytest.importorskip("nltk.stem.porter")
    reference = nltk_stem.PorterStemmer(mode=nltk_stem.PorterStemmer.MARTIN_EXTENSIONS)
    bases = ["connect", "relate", "general", "sense", "bake", "hop", "fall", "agree", "happy", "organ", "nation", "form", "adjust", "electric", "sign", "deploy", "retain", "memory", "search", "history", "commit", "branch", "merge"]
    endings = ["s", "es", "ed", "ing", "ly", "er", "ers", "ion", "ions", "ation", "ational", "ment", "ness", "ful", "ity", "ize", "izer", "able", "ism", "ive", "al"]
    for word in {base + ending for base in bases for ending in endings} | set(PORTER_EXAMPLES):
        assert stem(word) == reference.stem(word, to_lowercase=False), word


def test_tokens_are_lowercased_stemmed_words_split_at_anything_but_letters_and_digits() -> None:
    assert tokens("Deploying_the v2.1 BUILDS, don't!") == ["deploi", "the", "v2", "1", "build", "don", "t"]
    assert tokens("café 日本語") == ["café", "日本語"]
    assert tokens("  \n\t ") == []


def test_tokens_split_at_bytes_that_are_not_text() -> None:
    assert tokens("one\udc80two") == ["on", "two"]


def test_the_score_is_the_stated_formula() -> None:
    index = Bm25()
    index.add(["a", "b"])
    index.add(["a", "c", "c", "c"])
    index.add(["d"])

    ((doc, score),) = index.rank(["c"])

    average = (2 + 4 + 1) / 3
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 3 * (K1 + 1) / (3 + K1 * (1 - B + B * 4 / average))
    assert doc == 1
    assert score == pytest.approx(expected)
    assert (K1, B) == (1.5, 0.75)


def test_every_query_word_adds_its_part_and_a_repeated_word_counts_once() -> None:
    index = Bm25()
    index.add(["x", "y"])
    index.add(["x"])
    index.add(["z"])
    ((_, both), (_, single)) = index.rank(["x", "y"])
    assert both > single
    assert index.rank(["x", "y", "x"]) == index.rank(["x", "y"])


def test_ties_go_to_the_document_added_last() -> None:
    """For a memory, the newest of two equal matches is the natural one to read first."""
    index = Bm25()
    for _ in range(4):
        index.add(["same", "words"])
    assert [doc for doc, _ in index.rank(["same"])] == [3, 2, 1, 0]


def test_the_limit_and_no_match() -> None:
    index = Bm25()
    index.add(["a"])
    index.add(["a"])
    assert len(index.rank(["a"], limit=1)) == 1
    assert index.rank(["missing"]) == []
    assert index.rank([]) == []
    assert Bm25().rank(["a"]) == []


def test_a_document_added_later_counts_in_the_next_ranking() -> None:
    index = Bm25()
    index.add(["a"])
    before = index.rank(["a"])[0][1]
    index.add(["b"])
    assert index.rank(["a"])[0][1] != before
    assert index.rank(["b"])[0][0] == 1


def test_scores_do_not_depend_on_the_process_hash_seed() -> None:
    """Python orders strings differently per process, and float sums depend on order."""
    code = ("from taste.memstore.keyword import Bm25\n"
            "index = Bm25()\n"
            "for doc in range(50):\n"
            "    index.add([f'w{(doc * k) % 17}' for k in range(1, 9)])\n"
            "print(repr(index.rank([f'w{k}' for k in range(17)], limit=50)))\n")
    outputs = {subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed},
                              capture_output=True, text=True, check=True).stdout
               for seed in ("1", "2", "3")}
    assert len(outputs) == 1
