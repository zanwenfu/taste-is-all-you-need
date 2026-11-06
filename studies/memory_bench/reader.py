"""The reader stage and the free judge: the served model, asked as the benchmark asks it.

Answers are written as they arrive, so a stopped job loses nothing. A failed call goes to a file of its
own and is asked again next time; the benchmark's own script leaves such a question out of its results
without saying so.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def served_ask(client: Any, model: str, *, max_tokens: int, effort: str | None = None) -> Callable[[str], dict]:
    """One user message, temperature 0, as run_generation.py sends it. A model that reasons gets its effort."""
    extra = {"reasoning_effort": effort} if effort else {}

    def ask(prompt: str) -> dict:
        reply = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], n=1, temperature=0, max_tokens=max_tokens, **extra
        )
        choice = reply.choices[0]
        return {"text": choice.message.content or "", "finish_reason": choice.finish_reason,
                "prompt_tokens": reply.usage.prompt_tokens, "completion_tokens": reply.usage.completion_tokens}

    return ask


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


TRIES = 3  # a model that reasons first can use up its whole cap and show nothing; on full context gpt-oss-120b did so on 2 of 50 at 12,288 tokens, and each answered when asked again


def read_all(records_path: Path, ask: Callable[[str], dict], out: Path, *, workers: int,
             settings: dict[str, Any] | None = None) -> dict[str, int]:
    """Ask every record's prompt that has no answer yet, and count what came back. A reply with no text is asked again, up to ``TRIES`` in all, the same for every row and on
    the record, since an empty answer says nothing about the memory it was read from. An answer to a prompt that has since changed stops the stage: mixing the two would be silent.
    ``settings`` (model, cap, effort) goes on every answer."""
    records_path, out = Path(records_path), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    failed_path = out.with_name(out.name + ".failed")
    records = _rows(records_path)
    hashes = {record["question_id"]: record["prompt_sha256"] for record in records}
    done = {row["question_id"]: row.get("prompt_sha256") for row in _rows(out)}
    stale = [question for question, seen in done.items() if question in hashes and hashes[question] != seen]
    if stale:
        raise ValueError(f"{len(stale)} stale answers in {out}: their prompts have changed, for example {stale[0]}")
    todo = [record for record in records if record["question_id"] not in done]
    lock = threading.Lock()
    failed = [0]

    def one(record: dict) -> None:
        base = {"question_id": record["question_id"], "qid": record["qid"], "system": record["system"],
                "prompt_sha256": record["prompt_sha256"]}
        try:
            tries = 0
            while tries < TRIES:
                tries += 1
                reply = ask(record["prompt"])
                if reply["text"].strip():
                    break
        except Exception as error:
            with lock, failed_path.open("a") as written:
                failed[0] += 1
                written.write(json.dumps({**base, "error": f"{type(error).__name__}: {error}"}) + "\n")
            return
        row = {**base, **(settings or {}), "hypothesis": reply["text"], "finish_reason": reply["finish_reason"], "tries": tries,
               "prompt_tokens": reply["prompt_tokens"], "completion_tokens": reply["completion_tokens"]}
        with lock, out.open("a") as written:
            written.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, todo))
    answers = _rows(out)
    return {"questions": len(records), "asked": len(todo), "failed": failed[0],
            "empty": sum(not row["hypothesis"].strip() for row in answers),
            "cut_off": sum(row["finish_reason"] == "length" for row in answers), "asked_again": sum(row.get("tries", 1) > 1 for row in answers)}


class _Reply:
    def __init__(self, text: str, input_tokens: int, output_tokens: int, stop_reason: str) -> None:
        self.text, self.input_tokens, self.output_tokens, self.stop_reason = text, input_tokens, output_tokens, stop_reason

    def cost(self, _model: str) -> float:
        return 0.0  # the served model is free; only its GPU-hours count


class ServedJudge:
    """The served model behind the interface ``lme_judge.judge_file`` expects, at temperature 0 as the official judge is. It judges every
    part; the paid judge reads the same answers beside it, on the user's yes."""

    def __init__(self, client: Any, model: str, *, effort: str = "low", max_tokens: int = 512) -> None:
        self.client, self.model, self.effort, self.max_tokens = client, model, effort, max_tokens

    def ask(self, user: str, system: str | None = None) -> _Reply:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        reply = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=self.max_tokens, temperature=0,
                                                    reasoning_effort=self.effort)
        choice = reply.choices[0]
        return _Reply(choice.message.content or "", reply.usage.prompt_tokens, reply.usage.completion_tokens,
                      choice.finish_reason)
