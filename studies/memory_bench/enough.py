"""Did what a system delivered hold enough to answer the question? Asked of every system alike, with Mem0's own published judge.

Whether the right session or round arrived says little when a memory is written in other words, so the same question goes to
every system's delivered text, in the words Mem0's makers use for their own retrieval mode. The prompts are read from that
repository's file at a pinned commit (``COMMIT``), never copied here, and never by running the file.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

COMMIT = "4b61c5d31b9c668a12b4f5e78064248a02c82d2b"  # mem0ai/memory-benchmarks, benchmarks/longmemeval/run.py
PROMPTS_SHA256 = "99bd3d6d9a69072f9125550a0409f0de0aeaec66309416060db3daf12e137861"  # that file at that commit
_NAMES = ("RETRIEVAL_JUDGE_SYSTEM", "RETRIEVAL_JUDGE_PROMPT")


def load_prompts(path: Path, sha256: str | None = PROMPTS_SHA256) -> tuple[str, str]:
    """(system prompt, template), read as constants from the file's syntax tree. Any other file than the one named is refused."""
    if sha256 and hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha256:
        raise SystemExit(f"{path} is not the file at commit {COMMIT}")
    found: dict[str, str] = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in _NAMES:
            found[node.targets[0].id] = ast.literal_eval(node.value)
    return found[_NAMES[0]], found[_NAMES[1]]


def delivered_text(prompt: str) -> str:
    """What the reader's prompt holds of a system's memory, between the benchmark's own markers."""
    start, end = prompt.find("History Chats:"), prompt.rfind("Current Date:")
    if start < 0 or end < 0:
        return ""  # no memory, by the benchmark's own form of the prompt
    return prompt[start + len("History Chats:") : end].strip()


def parse_judgment(text: str) -> bool | None:
    """PASS or FAIL from the judge's JSON; if the JSON is broken, from its judgment field alone; otherwise unknown."""
    start, stop = text.find("{"), text.rfind("}")
    try:
        value = json.loads(text[start : stop + 1]) if start >= 0 else None
    except json.JSONDecodeError:
        value = None
    said = value.get("judgment") if isinstance(value, dict) else None
    if not isinstance(said, str):
        match = re.search(r'"judgment"\s*:\s*"(PASS|FAIL)', text or "")
        said = match.group(1) if match else None
    return {"PASS": True, "FAIL": False}.get((said or "").strip().upper()[:4])


def judge_all(records: list[dict], references: dict[str, dict], prompts: tuple[str, str], ask: Callable[[str, str], Any], out: Path, *,
              workers: int) -> dict[str, int]:
    """Judge every record not yet judged. A failed call leaves no row and is asked again next time."""
    system, template = prompts
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["question_id"] for line in out.read_text().splitlines()} if out.exists() else set()
    lock = threading.Lock()
    errors = [0]

    def one(record: dict) -> None:
        reference = references[record["question_id"]]
        text = delivered_text(record["prompt"])
        user = template.format(question=reference["question"], question_date=reference.get("question_date") or "(not specified)",
                               answer=str(reference["answer"]), profile_section="", num_memories=len(record["delivered"]),
                               memories_text=text or "(None)")
        try:
            reply = ask(user, system)
        except Exception:
            with lock:
                errors[0] += 1
            return
        row = {"question_id": record["question_id"], "qid": record["qid"], "prompt_sha256": record["prompt_sha256"],
               "enough": parse_judgment(reply.text), "raw": reply.text, "stop_reason": reply.stop_reason}
        with lock, out.open("a") as written:
            written.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, [record for record in records if record["question_id"] not in done]))
    rows = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    return {"questions": len(records), "judged": len(rows), "errors": errors[0], "unread": sum(row["enough"] is None for row in rows),
            "enough": sum(row["enough"] is True for row in rows)}
