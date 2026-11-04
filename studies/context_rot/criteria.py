"""Criterion records: the user's standing instructions, written once as each message arrives.

Taste keeps a request's standing criteria as records sent with every decision, never
summarized. Here they are filled from the conversation instead of a form: each user message
is read once, when it arrives, with nothing after it, and every instruction meant to hold
beyond the current request is kept word for word, pointing at its message. Only quotes found
verbatim in the message are kept; the rest are counted and dropped. At compaction the context
is the summary plus these records, verbatim.

The reader's prompt was written without the benchmark's rules in view and is not tuned on them.
Close to the benchmark paper's own fix (a small model keeps a list of standing rules); what is
measured here is what it keeps across repeated compaction, what it adds on rule-free
histories, and what it misses when the rule carries no "for the rest of this session" cue.
"""

from __future__ import annotations

import json
import re

READER_SYSTEM = (
    "You read one message that a user sent to an AI assistant during a long working session. "
    "Find every instruction in it that the user means to keep applying after the current request is done: "
    "a rule, constraint, preference or format for the rest of the session, for all later answers, or for a "
    "kind of task whenever it comes up. Leave out requests that are finished once they are answered, "
    "questions, and facts about the task. Copy each instruction word for word from the message, without "
    "changing, shortening or joining anything. Reply with JSON only, in this shape: "
    '{"standing": ["<exact words from the message>", ...]}. Use an empty list when there is none.'
)


def reader_prompt(message: str) -> list[dict]:
    return [{"role": "system", "content": READER_SYSTEM},
            {"role": "user", "content": f"<message>\n{message}\n</message>"}]


def parse_reply(reply: str, message: str) -> tuple[list[str], list[str]]:
    """(kept, dropped): quotes found verbatim in the message, and those that were not."""
    text = (reply or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        quotes = json.loads(text).get("standing", [])
    except (json.JSONDecodeError, AttributeError):
        return [], []
    kept, dropped = [], []
    for quote in quotes:
        if not isinstance(quote, str) or not quote.strip():
            continue
        quote = quote.strip()
        (kept if quote in message else dropped).append(quote)
    return kept, dropped


def build_records(messages: list[dict], read, cache: dict) -> list[dict]:
    """Records for one history. ``read(text)`` returns the kept quotes of one user message;
    ``cache`` maps message text to them, so a message seen before is not read again."""
    records, seen = [], set()
    for index, message in enumerate(messages):
        if message["role"] != "user":
            continue
        if message["content"] not in cache:
            cache[message["content"]] = read(message["content"])
        for quote in cache[message["content"]]:
            if quote not in seen:  # an instruction repeated word for word is one record, at its first message
                seen.add(quote)
                records.append({"message": index, "text": quote})
    return records


def records_block(records: list[dict]) -> str:
    return "\n".join(f"- {record['text']}" for record in records) or "(none)"


def served_reader(client, model: str, log: list | None = None, max_tokens: int = 2048):
    """``read(text)`` for build_records, asking the served model; dropped quotes go to ``log``."""

    def read(text: str) -> list[str]:
        reply = client.chat.completions.create(model=model, messages=reader_prompt(text), max_tokens=max_tokens,
                                               reasoning_effort="low")
        kept, dropped = parse_reply(reply.choices[0].message.content or "", text)
        if log is not None:
            log.append({"kept": len(kept), "dropped": dropped, "finish": reply.choices[0].finish_reason})
        return kept

    return read


def records_for(instances, read, workers: int = 16) -> dict[str, list[dict]]:
    """Records for every case; each distinct user message is read once, in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    texts = sorted({m["content"] for inst in instances for m in inst.messages if m["role"] == "user"})
    with ThreadPoolExecutor(workers) as pool:
        cache = dict(zip(texts, pool.map(read, texts)))
    return {inst.id: build_records(inst.messages, read, cache) for inst in instances}
