"""Repeated compaction on the rule-loss benchmark: k rounds of summary, then the probes.

Each case of the paper's ~100K-token set keeps its history and its rule; only the number of
compactions changes. The history is cut into k consecutive parts of about equal tokens, at
message boundaries. Round 1 summarizes part 1 with the package's prompt as published. Each
later round summarizes the previous summary together with the next part:

  pi_mono    Pi's own update step (pi-mono f129ac93, compaction.ts, generateSummary): the new
             messages, then the previous summary in <previous-summary> tags followed by Pi's
             UPDATE_SUMMARIZATION_PROMPT, under the package's Pi system prompt.
  anthropic  the previous summary stays in the history as an assistant message, and the
             package's prompt summarizes it with the new part, as compaction does when it runs
             again over a history that already holds a compaction.

After round k the package's probes run on four contexts: the last summary alone; the summary
plus every user turn verbatim from the record; the summary plus the same number of tokens of
the newest text (the blocks of record_arms); and the summary plus the criterion records
(criteria.py), the user's standing instructions written down verbatim as each message arrived. Every round's summary is judged for retention
with the package's judge prompt, asked of the served model (free; its agreement with the
paper's GPT-5.4 judge is measured on the single-compaction run). k = 1 is the published single
compaction and comes from the baseline run.

    python -m studies.context_rot.repeated --prompt pi_mono --ks 2,4,8,16 --histories 0:10 \\
        --probe-url http://127.0.0.1:8000/v1 --tokenizer <gpt-oss folder> --out runs/conpact/repeated/pi_mono
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from studies.context_rot.record_arms import HEADERS, arm_context, recent_block, user_record

# pi-mono f129ac93, packages/coding-agent/src/core/compaction/compaction.ts, verbatim.
PI_UPDATE_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


def split_parts(messages: list[dict], sizes: list[int], k: int) -> list[list[dict]]:
    """k consecutive, non-empty parts; each cut at the message boundary nearest j/k of the tokens."""
    if len(messages) < k:
        raise ValueError(f"{len(messages)} messages cannot make {k} parts")
    cumulative = [0]
    for size in sizes:
        cumulative.append(cumulative[-1] + size)
    total, cuts = cumulative[-1], [0]
    for j in range(1, k):
        low, high = cuts[-1] + 1, len(messages) - (k - j)
        cuts.append(min(range(low, high + 1), key=lambda b: (abs(cumulative[b] - total * j / k), b)))
    cuts.append(len(messages))
    return [messages[a:b] for a, b in zip(cuts, cuts[1:])]


def round_prompt(option: str, previous: str | None, part: list[dict], package_prompt) -> list[dict]:
    """The summarizer's messages for one round. ``package_prompt`` is conpact's summarization_prompt."""
    if previous is None:
        return package_prompt(option, part)
    if option == "pi_mono":
        system = package_prompt("pi_mono", [])[0]
        return [system, *part, {"role": "user", "content": f"<previous-summary>\n{previous}\n</previous-summary>\n\n"
                                                           f"{PI_UPDATE_PROMPT}"}]
    if option == "anthropic":
        return package_prompt("anthropic", [{"role": "assistant", "content": previous}, *part])
    raise ValueError(option)


def run_one(inst, k: int, option: str, sizes: list[int], summarize, judge, probe, count, package_prompt,
            blocks: dict | None = None) -> dict:
    """k rounds on one case, every round judged, then the probes on each context.

    ``blocks`` adds contexts: name -> (block text, header), each placed after the last summary.
    """
    parts = split_parts(inst.messages, sizes, k)
    summaries, retention, previous = [], [], None
    for part in parts:
        previous = summarize(round_prompt(option, previous, part, package_prompt))
        summaries.append(previous)
        retention.append(judge(inst.sc_text, previous))
    summary = [{"role": "assistant", "content": previous}]
    record = user_record(inst.messages)
    contexts = {"compaction": summary,
                "user_record": arm_context(summary, record, HEADERS["user_record"]),
                "recent_control": arm_context(summary, recent_block(inst.messages, count(record), count),
                                              HEADERS["recent_control"])}
    for name, (block, header) in (blocks or {}).items():
        contexts[name] = arm_context(summary, block, header)
    part_tokens, offset = [], 0
    for part in parts:
        part_tokens.append(sum(sizes[offset:offset + len(part)]))
        offset += len(part)
    row = {"id": inst.id, "haystack_id": inst.haystack_id, "sc_id": inst.sc_id, "sc_type": inst.sc_type,
           "compliant_letter": inst.compliant_letter, "prompt": option, "k": k, "part_tokens": part_tokens,
           "summaries": summaries, "retention": retention, "record_tokens": count(record)}
    for arm, context in contexts.items():
        row[f"{arm}_output"], row[f"{arm}_compliant"] = probe(inst, context, inst.probe_prompt)
        row[f"{arm}_upper_output"], row[f"{arm}_upper_compliant"] = probe(inst, context, inst.post_sc_probe_prompt)
    return row


def done_keys(out: Path) -> set[tuple[str, int]]:
    path = Path(out) / "results.jsonl"
    if not path.exists():
        return set()
    return {(row["id"], row["k"]) for row in map(json.loads, path.read_text().splitlines())}


class ServedChat:
    """The served model behind the RetentionJudge interface (``ask(user, system)``)."""

    def __init__(self, client, model: str, max_tokens: int = 2048) -> None:
        self.client, self.model, self.max_tokens = client, model, max_tokens

    def ask(self, user: str, system: str | None = None):
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        reply = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=self.max_tokens,
                                                    reasoning_effort="low")
        return type("Reply", (), {"text": reply.choices[0].message.content or "",
                                  "input_tokens": reply.usage.prompt_tokens,
                                  "output_tokens": reply.usage.completion_tokens,
                                  "stop_reason": reply.choices[0].finish_reason})()

    def settings(self) -> dict:
        return {"model": self.model, "effort": "low", "max_tokens": self.max_tokens}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True, choices=("pi_mono", "anthropic"))
    parser.add_argument("--ks", default="2,4,8,16", help="numbers of compaction rounds, comma-separated")
    parser.add_argument("--histories", default="0:10", help="START:END, histories in file order, end excluded")
    parser.add_argument("--source", default="hermes")
    parser.add_argument("--probe-url", required=True)
    parser.add_argument("--probe-model", default="openai/gpt-oss-120b")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--no-criteria", dest="criteria", action="store_false",
                        help="skip the summary plus criterion records (read once per distinct user message)")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)

    from conpact import Endpoint, load_static
    from conpact.evaluate import _probe
    from conpact.prompts import judge_prompts, summarization_prompt
    from transformers import AutoTokenizer

    from studies.context_rot.conpact_bridge import DATA_REVISION, OUTPUT_CAP, RetentionJudge, history_range

    tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    start, end = (int(part) for part in arguments.histories.split(":"))
    instances = history_range(load_static(arguments.source, revision=DATA_REVISION), start, end)
    ks = [int(k) for k in arguments.ks.split(",")]
    endpoint = Endpoint(arguments.probe_model, arguments.probe_url, "EMPTY",
                        {"reasoning_effort": "low", "max_tokens": OUTPUT_CAP})
    client = endpoint.client()
    judge = RetentionJudge(ServedChat(client, arguments.probe_model))

    def summarize(messages: list[dict]) -> str:
        reply = client.chat.completions.create(model=arguments.probe_model, messages=messages, **endpoint.params)
        return reply.choices[0].message.content or ""

    def judge_one(sc_text: str, summary: str):
        return judge.verdict(*judge_prompts(sc_text, [{"role": "assistant", "content": summary}]))

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    blocks_by_case: dict = {}
    if arguments.criteria:
        from studies.context_rot.criteria import records_block, records_for, served_reader

        log: list = []
        records = records_for(instances, served_reader(client, arguments.probe_model, log), arguments.workers)
        (out / "records.json").write_text(json.dumps({"records": records, "reads": log}, indent=1))
        blocks_by_case = {case: {"criteria": (records_block(found), HEADERS["criteria"])}
                          for case, found in records.items()}
    done, lock = done_keys(out), threading.Lock()
    sizes_by_text: dict[str, int] = {}

    def sizes_of(messages: list[dict]) -> list[int]:
        with lock:
            missing = [m["content"] for m in messages if m["content"] not in sizes_by_text]
        counted = {text: count(text) for text in missing}
        with lock:
            sizes_by_text.update(counted)
            return [sizes_by_text[m["content"]] for m in messages]

    def one(inst) -> None:
        sizes = sizes_of(inst.messages)
        for k in ks:  # one case's rounds in a row: the shorter first parts share the longer ones' prefix
            if (inst.id, k) in done:
                continue
            row = run_one(inst, k, arguments.prompt, sizes, summarize, judge_one,
                          lambda i, context, turn: _probe(client, endpoint, i, context, turn), count,
                          summarization_prompt, blocks_by_case.get(inst.id))
            with lock, (out / "results.jsonl").open("a") as written:
                written.write(json.dumps(row) + "\n")

    # Case order is history-major, so the 15 rules of one history run together and share its later parts.
    with ThreadPoolExecutor(arguments.workers) as pool:
        list(pool.map(one, instances))
    print(json.dumps({"prompt": arguments.prompt, "ks": ks, "cases": len(instances),
                      "rows": len(done_keys(out)), "judge_calls": judge.calls, "judge_unclear": judge.unparseable}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
