"""Strip from LongMemEval everything that marks the answer, before any memory sees it.

The released files mark where the answer is: evidence turns carry ``has_answer``,
evidence sessions are named ``answer_<id>`` (and some others ``noans_<id>``),
``answer_session_ids`` lists them, ``question_type`` names the type, and an ``_abs``
suffix on the question id marks the 30 unanswerable questions (compare-longmemeval.md
1.7, 7). A memory under test gets none of that. It gets, per question, a neutral id, the
question, its date, and the sessions in date order, each with a neutral id, its date and
its turns (role and content only). The answer and the reference stay out.

Neutral ids are a keyed hash of the original, so they do not reveal file order (the files
are grouped by type) or the original name. A private map, never given to the memory,
turns them back into original ids for retrieval scoring.

    python -m studies.context_rot.lme_clean <longmemeval_s.json> <cleaned.json> <private-map.json>
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

KEY = b"taste-context-rot-2026-11"
_DATE = re.compile(r"(\d{4})/(\d{2})/(\d{2}) \(\w+\) (\d{2}):(\d{2})")


def neutral(kind: str, original: str) -> str:
    return kind + hashlib.blake2b(original.encode(), key=KEY, digest_size=6).hexdigest()


def _when(date: str) -> tuple:
    match = _DATE.fullmatch(date.strip())
    if match is None:
        raise ValueError(f"unexpected date {date!r}")
    return tuple(int(part) for part in match.groups())


def clean(instances: list[dict]) -> tuple[list[dict], dict]:
    """Return (what the memory may see, the private map back to original ids)."""
    cleaned, private = [], {}
    for item in instances:
        qid = neutral("q", item["question_id"])
        sessions, back = [], {}
        order = sorted(range(len(item["haystack_sessions"])), key=lambda i: _when(item["haystack_dates"][i]))
        for i in order:
            original = item["haystack_session_ids"][i]
            sid = neutral("s", f"{item['question_id']}/{original}")
            back[sid] = original
            sessions.append({"sid": sid, "date": item["haystack_dates"][i],
                             "turns": [{"role": turn["role"], "content": turn["content"]}
                                       for turn in item["haystack_sessions"][i]]})
        cleaned.append({"qid": qid, "question": item["question"], "question_date": item["question_date"],
                        "sessions": sessions})
        private[qid] = {"question_id": item["question_id"], "sessions": back}
    return cleaned, private


def main(argv=None) -> int:
    source, target, private_path = (argv or sys.argv[1:])[:3]
    cleaned, private = clean(json.loads(Path(source).read_text()))
    Path(target).write_text(json.dumps(cleaned))
    Path(private_path).write_text(json.dumps(private))
    print(f"{len(cleaned)} questions cleaned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
