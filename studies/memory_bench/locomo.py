"""LoCoMo as the comparison uses it: the dialogue only, one store per conversation, its own score.

``observation`` and ``session_summary`` are generated text and never reach a memory. The reader's prompt
strings and the judge-free scoring are the benchmark's own (task_eval/gpt_utils.py and evaluation.py). The
token F1 needs no judge, so it survives every judge dispute, and it leads. Its official form stems with NLTK's
default Porter mode, so the stemmer is passed in; a judge's prompt, where one is used, is LongMemEval's plain
one, chosen in advance, and adversarial questions are scored apart by their own rule.

The official scorer stems with NLTK's default ``PorterStemmer()``, whose extensions differ from Porter's paper on
words an answer can hold ("skies" is "sky" there and "ski" in Taste's search), so scores use exactly that
stemmer, at a pinned NLTK version (``official_stem``). Taste's own stemmer stays in search; the two never mix.
"""

from __future__ import annotations

import hashlib
import re
import string
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from statistics import fmean

from studies.context_rot.lme_clean import neutral
from studies.memory_bench.core import Item

_START = (
    "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and "
    "the date of each conversation is wriiten at the beginning of the conversation.\n\n"
)
_QA = (
    "\nBased on the above context, write an answer in the form of a short phrase for the following question. "
    "Answer with exact words from the context whenever possible.\n\nQuestion: {} Short answer:\n"
)
_QA_ADVERSARIAL = "\nBased on the above context, answer the following question.\n\nQuestion: {} Short answer:\n"
_NOT_THERE = "Not mentioned in the conversation"
CUTS = (1, 3, 5, 10, 30, 50)
NLTK_VERSION = "3.10.4"


def official_stem() -> Callable[[str], str]:
    """The stemmer of the benchmark's evaluation.py (``ps = PorterStemmer()``), at the pinned NLTK version."""
    import nltk
    from nltk.stem.porter import PorterStemmer

    if nltk.__version__ != NLTK_VERSION:
        raise RuntimeError(f"scores are pinned to NLTK {NLTK_VERSION}; this is {nltk.__version__}")
    return PorterStemmer().stem


@dataclass(frozen=True)
class Gold:
    """Never shown to a memory."""

    question_id: str
    question_type: str
    category: int
    answer: str
    adversarial: bool
    key: dict
    evidence_sessions: tuple
    evidence_rounds: tuple


def _sessions(conversation: dict, sample_id: str, captions: bool = True) -> tuple[list[dict], dict[str, str], dict[str, tuple[str, int]]]:
    numbers = sorted(int(k.split("_")[1]) for k in conversation if re.fullmatch(r"session_\d+", k))
    sessions, session_of, turn_of = [], {}, {}
    for number in numbers:
        sid = neutral("s", f"{sample_id}/{number}")
        session_of[str(number)] = sid
        turns = []
        for index, dialog in enumerate(conversation[f"session_{number}"]):
            caption = dialog.get("blip_caption")
            text = dialog["text"] + (f" and shared {caption}." if caption and captions else "")
            turns.append({"role": dialog["speaker"], "content": text})
            turn_of[dialog["dia_id"]] = (sid, index)
        sessions.append({"sid": sid, "date": conversation[f"session_{number}_date_time"], "turns": turns,
                         "each_turn_is_a_round": True})
    return sessions, session_of, turn_of


def _asked(item: dict, question_id: str) -> tuple[str, dict]:
    """The question as the benchmark asks it. Its adversarial options were ordered by chance; here by a hash."""
    if item["category"] == 2:
        return item["question"] + " Use DATE of CONVERSATION to answer with an approximate date.", {}
    if item["category"] == 5:
        wrong = str(item.get("adversarial_answer", item.get("answer")))
        first = hashlib.sha256(question_id.encode()).digest()[0] % 2 == 0
        key = {"a": _NOT_THERE, "b": wrong} if first else {"a": wrong, "b": _NOT_THERE}
        return f"{item['question']} Select the correct answer: (a) {key['a']} (b) {key['b']}. ", key
    return item["question"], {}


def load(samples: list[dict], *, captions: bool = True) -> tuple[list[dict], dict[str, Gold]]:
    """(one view per conversation, the gold for each question by neutral id). The benchmark's reader prompt shows an
    image's caption, so every system ingests it; MemPalace's runner leaves it out, so ``captions=False`` is only for
    checking an adapter against its committed lists."""
    views, golds = [], {}
    for sample in samples:
        conversation = sample["conversation"]
        sessions, session_of, turn_of = _sessions(conversation, sample["sample_id"], captions)
        questions = []
        for index, item in enumerate(sample["qa"]):
            question_id = f"{sample['sample_id']}:{index}"
            qid = neutral("q", question_id)
            asked, key = _asked(item, question_id)
            questions.append({"qid": qid, "query": item["question"], "asked": asked, "category": item["category"]})
            golds[qid] = Gold(
                question_id=question_id,
                question_type=f"category-{item['category']}",
                category=item["category"],
                answer=str(item.get("answer", item.get("adversarial_answer"))),
                adversarial=item["category"] == 5,
                key=key,
                evidence_sessions=tuple(session_of.get(e.split(":")[0][1:]) for e in item["evidence"]),
                evidence_rounds=tuple(turn_of.get(e) for e in item["evidence"]),
            )
        views.append({"conversation": sample["sample_id"], "speakers": (conversation["speaker_a"], conversation["speaker_b"]),
                      "sessions": sessions, "questions": questions})
    return views, golds


def reference(samples: list[dict]) -> list[dict]:
    """What LongMemEval's judge code needs, for the questions a judge scores. "multi-session" routes to its plain
    prompt. Adversarial questions are left out: they have their own rule."""
    return [
        {"question_id": f"{sample['sample_id']}:{index}", "question_type": "multi-session",
         "question": item["question"], "answer": str(item["answer"])}
        for sample in samples
        for index, item in enumerate(sample["qa"])
        if item["category"] != 5
    ]


def dialog_text(item: Item) -> str:
    return "".join(f'{turn["role"]} said, "{turn["content"]}"\n' for turn in item.turns)


def reader_prompt(items: Iterable[Item], view: dict, question: dict) -> str:
    """The benchmark's prompt. Turns retrieved from one session sit together, in order, under its date, and
    sessions go in the order they happened (its dates are written in words, so they cannot be sorted as text)."""
    position = {session["sid"]: number for number, session in enumerate(view["sessions"])}
    groups: dict[str, list[Item]] = {}
    for item in items:
        groups.setdefault(item.session, []).append(item)
    blocks = []
    for sid in sorted(groups, key=lambda s: position.get(s, len(position))):
        parts = sorted(groups[sid], key=lambda item: item.rounds[0] if item.rounds else 1 << 30)
        lines = "".join(dialog_text(part) for part in parts)
        blocks.append(f"DATE: {parts[0].date}\nCONVERSATION:\n{lines}\n\n")
    template = _QA_ADVERSARIAL if question["category"] == 5 else _QA
    return _START.format(*view["speakers"]) + "".join(blocks) + "\n\n" + template.format(question["asked"])


# ------------------------------------------------------------------ the judge-free score


def normalize_answer(text: str) -> str:
    text = text.replace(",", "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the|and)\b", " ", text).split())


def f1_score(prediction: str, truth: str, stem: Callable[[str], str]) -> float:
    predicted = [stem(word) for word in normalize_answer(prediction).split()]
    gold = [stem(word) for word in normalize_answer(truth).split()]
    same = sum((Counter(predicted) & Counter(gold)).values())
    if same == 0:
        return 0
    precision, recall = same / len(predicted), same / len(gold)
    return 2 * precision * recall / (precision + recall)


def f1_multi(prediction: str, truth: str, stem: Callable[[str], str]) -> float:
    predictions = [part.strip() for part in prediction.split(",")]
    return fmean(max(f1_score(p, part.strip(), stem) for p in predictions) for part in truth.split(","))


def qa_score(category: int, prediction: str, answer: str, stem: Callable[[str], str]) -> float:
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category in (2, 3, 4):
        return f1_score(prediction, answer, stem)
    if category == 1:
        return f1_multi(prediction, answer, stem)
    raise ValueError(f"category {category} has no F1 rule")


def score_answers(answers: Sequence[dict], view_golds: dict[str, Gold],
                  stem: Callable[[str], str]) -> tuple[list[dict], dict]:
    """Every question's judge-free score, and the summary: categories 1-4 together, adversarial apart.

    An answer the reader returned empty scores 0. A question with no answer at all stops the scoring: the run
    is not finished, and leaving it out would change the denominator without saying so."""
    by_qid = {answer["qid"]: answer for answer in answers}
    missing = sorted(set(view_golds) - set(by_qid))
    if missing:
        raise ValueError(f"{len(missing)} questions have no answer; finish the run before scoring")
    rows = []
    for qid, gold in sorted(view_golds.items(), key=lambda item: item[1].question_id):
        said = by_qid[qid]["hypothesis"]
        value = (adversarial_score(said, gold.key) if gold.adversarial
                 else qa_score(gold.category, said, gold.answer, stem))
        rows.append({"qid": qid, "question_id": gold.question_id, "category": gold.category, "score": float(value)})
    scored = [row for row in rows if row["category"] != 5]
    adversarial = [row["score"] for row in rows if row["category"] == 5]
    summary = {
        "questions": len(rows),
        "f1_categories_1_to_4": fmean(row["score"] for row in scored) if scored else None,
        "scored_1_to_4": len(scored),
        "per_category": {c: fmean(r["score"] for r in rows if r["category"] == c)
                         for c in sorted({r["category"] for r in rows})},
        "adversarial": fmean(adversarial) if adversarial else None,
        "adversarial_questions": len(adversarial),
    }
    return rows, summary


def adversarial_score(prediction: str, key: dict) -> int:
    said = prediction.strip().lower()
    if len(said) == 1:
        said = key["a"] if "a" in said else key["b"]
    elif len(said) == 3:
        said = key["a"] if "(a)" in said else key["b"]
    said = said.lower()
    return int("no information available" in said or "not mentioned" in said)


# ------------------------------------------------------------------ retrieval


def _share(found: set, wanted: Sequence) -> float:
    return sum(value in found for value in wanted) / len(wanted) if wanted else 1.0


def locomo_scores(ranked: Sequence[Item], delivered: Sequence[Item], gold: Gold) -> dict[str, float]:
    """The share of the evidence found, as the benchmark and MemPalace's runner count it: a question with no
    evidence scores 1. An evidence id that names no turn counts as not found."""
    sessions = list(dict.fromkeys(item.session for item in ranked))
    scores: dict[str, float] = {}
    for cut in CUTS:
        scores[f"session_recall@{cut}"] = _share(set(sessions[:cut]), gold.evidence_sessions)
        scores[f"turn_recall@{cut}"] = _share(
            {(item.session, number) for item in ranked[:cut] for number in item.rounds}, gold.evidence_rounds
        )
    scores["session_recall@budget"] = _share({item.session for item in delivered}, gold.evidence_sessions)
    scores["turn_recall@budget"] = _share(
        {(item.session, number) for item in delivered for number in item.rounds}, gold.evidence_rounds
    )
    return scores


def conversation_split(ids: Sequence[str], dev_count: int, seed: int) -> tuple[list[str], list[str]]:
    """Whole conversations, since the questions of one share its store."""
    ordered = sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    return sorted(ordered[:dev_count]), sorted(ordered[dev_count:])
