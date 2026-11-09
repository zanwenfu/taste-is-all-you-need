"""The compaction baseline with one sentence added to its prompt, so the store and the instruction can be read apart.

A notes file whose writer is told to keep what the user asked it to keep doing holds a rule far more often than a summary
does. That could be the store or that one sentence. The same sentence on the benchmark's own summary prompt answers it.

    python -m studies.memory_probe.told_summary --inputs runs/probe/inputs --prompt anthropic --limit 50 \\
        --url http://127.0.0.1:8000/v1 --model openai/gpt-oss-120b --out runs/probe/told-anthropic

The folder it writes holds its own cases and summaries, in the shape ``studies.memory_probe.run`` reads.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from studies.memory_probe.replies import read_summary

CLAUSE = "Keep anything the user asked the assistant to keep doing."
# A second model may carry on the work in place of summarizing (Laguna S 2.1, Nov 6 and 9): with read_summaries, a reply is read by the long run's rule
# (what the prompt asked for, nothing around it) and asked again when it holds none; after the last try the package's own reply is kept, and the row says so.
SUMMARY_TRIES = 3


def told_prompt(prompt: list[dict]) -> list[dict]:
    """The package's prompt with the sentence after its own last instruction, and nothing else changed."""
    *before, last = prompt
    return [*before, {**last, "content": f"{last['content']}\n\n{CLAUSE}"}]


def write_summaries(cases: list[dict], option: str, ask: Callable[[list[dict]], tuple[str, str]], out: Path, *, workers: int,
                    package_prompt: Callable[[str, list[dict]], list[dict]], read_summaries: bool = False) -> dict[str, int]:
    """One round per case. A summary the output cap cut stops the run: its end would be lost silently, and this arm would
    look worse for no real reason. A stopped run picks up where it was."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.jsonl").write_text("".join(json.dumps(case) + "\n" for case in cases))
    path = out / "summaries.jsonl"
    done = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()
    todo = [case for case in cases if case["id"] not in done]
    lock = threading.Lock()

    def one(case: dict) -> None:
        messages = told_prompt(package_prompt(option, case["messages"]))
        row = {"id": case["id"], "prompt": option, "k": 1}
        if not read_summaries:
            text, finish = ask(messages)
            if finish == "length" or not text.strip():
                raise ValueError(f"{case['id']}: the summary was cut off or empty")
            row["summary"] = text
        else:
            for tries in range(1, SUMMARY_TRIES + 1):
                text, finish = ask(messages)
                summary = None if finish == "length" else read_summary(option, text)
                if summary:
                    break
            if summary:
                row |= {"summary": summary, "read": True, "tries": tries}
            elif finish == "length" or not text.strip():
                raise ValueError(f"{case['id']}: the summary was cut off or empty")
            else:
                row |= {"summary": text, "read": False, "tries": tries}
        with lock, path.open("a") as written:
            written.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, todo))
    return {"cases": len(cases), "written": len(todo)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("inputs", "url", "model", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--prompt", required=True, choices=("pi_mono", "anthropic"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--effort", default="low", help='the model\'s reasoning effort; "none" sends none')
    parser.add_argument("--output-cap", type=int, help="the summary's output cap; the package's 4,096 when not given")
    parser.add_argument("--read-summaries", action="store_true",
                        help="read each reply by the long run's rule and ask again when it holds no summary (a second model's summaries)")
    arguments = parser.parse_args(argv)

    from conpact import Endpoint
    from conpact.prompts import summarization_prompt

    from studies.context_rot.conpact_bridge import OUTPUT_CAP, effort_arg, probe_params

    endpoint = Endpoint(arguments.model, arguments.url, "EMPTY", probe_params(effort_arg(arguments.effort), arguments.output_cap or OUTPUT_CAP))
    client = endpoint.client()

    def ask(messages: list[dict]) -> tuple[str, str]:
        reply = client.chat.completions.create(model=arguments.model, messages=messages, **endpoint.params)
        return reply.choices[0].message.content or "", reply.choices[0].finish_reason

    cases = [json.loads(line) for line in (Path(arguments.inputs) / "cases.jsonl").read_text().splitlines()]
    cases = cases[: arguments.limit] if arguments.limit else cases
    print(json.dumps(write_summaries(cases, arguments.prompt, ask, Path(arguments.out), workers=arguments.workers,
                                     package_prompt=summarization_prompt, read_summaries=arguments.read_summaries)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
