"""LongMemEval as the comparison uses it: what a memory may see, and what only the scorer may.

``lme_clean`` strips every marker of the answer and gives neutral ids. Sessions that appear twice in one
question's history are kept once, for every system alike (the file has 13 such repeats, and a store that
dedupes bytes would otherwise differ from a flat list for no reason worth measuring).
"""

from __future__ import annotations

from dataclasses import dataclass

from studies.context_rot.lme_clean import clean, neutral


@dataclass(frozen=True)
class Gold:
    """Never shown to a memory."""

    question_id: str
    question_type: str
    answer: object
    abstention: bool
    sessions: frozenset[str]
    rounds: frozenset[tuple[str, int]]
    evidence: tuple[str, ...]


def _split_rounds(turns: list[dict]) -> list[list[dict]]:
    """A round is a user turn and the replies after it; anything before the first user turn is its own."""
    out: list[list[dict]] = []
    for turn in turns:
        if turn["role"] == "user" or not out:
            out.append([])
        out[-1].append(turn)
    return out


def rounds(session: dict) -> list[list[dict]]:
    return _split_rounds(session["turns"])


def unit_text(date: str, turns: list[dict]) -> str:
    """What every keyword or dense system indexes for a unit: the same date line, both roles."""
    return f"Session date: {date}\n" + "".join(f"{turn['role']}: {turn['content']}\n" for turn in turns)


def _gold(item: dict, sessions: list[dict]) -> Gold:
    by_sid = {
        neutral("s", f"{item['question_id']}/{original}"): (original, turns)
        for original, turns in zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True)
    }
    answer_sessions, answer_rounds, evidence = set(), set(), []
    for session in sessions:
        original, turns = by_sid[session["sid"]]
        if original in item["answer_session_ids"]:
            answer_sessions.add(session["sid"])
        for number, part in enumerate(_split_rounds(turns)):
            marked = [turn for turn in part if turn.get("has_answer")]
            if marked:
                answer_rounds.add((session["sid"], number))
                evidence.extend(turn["content"] for turn in marked)
    return Gold(
        question_id=item["question_id"],
        question_type=item["question_type"],
        answer=item["answer"],
        abstention=item["question_id"].endswith("_abs"),
        sessions=frozenset(answer_sessions),
        rounds=frozenset(answer_rounds),
        evidence=tuple(evidence),
    )


def load(instances: list[dict]) -> tuple[list[dict], dict[str, Gold]]:
    """(what a memory may see, oldest session first; the gold for each by neutral question id)."""
    views, _ = clean(instances)
    golds = {}
    for view, item in zip(views, instances, strict=True):
        seen: dict[str, dict] = {}
        for session in view["sessions"]:
            if session["sid"] in seen and seen[session["sid"]]["turns"] != session["turns"]:
                raise ValueError(f"{item['question_id']}: one session id with two different contents")
            seen.setdefault(session["sid"], session)
        view["sessions"] = list(seen.values())
        golds[view["qid"]] = _gold(item, view["sessions"])
    return views, golds
