"""Whether what a store delivered held the rule, when it may say it in its own words.

A whole-text match answers this for the arms that hold the user's own words. A notes file is rewritten by a model and may
paraphrase, so its arms are read by the benchmark package's own retention judge, asked of the served model, on the text
each case delivered. That is the judge the study already calibrated against GPT-5.4 at one compaction.

    python -m studies.memory_probe.judge --results run/results.jsonl --cases inputs/cases.jsonl --url URL --model NAME
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def judge_found(results: Path, rules: dict[str, str], verdict: Callable[[str, list[dict]], bool | None], *,
                workers: int) -> dict[str, int]:
    """One row per case in found.served.jsonl beside the results. Nothing delivered is not found, with no call."""
    results = Path(results)
    out = results.with_name("found.served.jsonl")
    done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    todo = [row for row in rows if row["id"] not in done]
    lock = threading.Lock()
    asked = [0]

    def one(row: dict) -> None:
        text = row.get("delivered_text") or ""
        answer = False
        if text.strip():
            answer = verdict(rules[row["id"]], [{"role": "assistant", "content": text}])
        with lock, out.open("a") as written:
            written.write(json.dumps({"id": row["id"], "found_by_judge": answer}) + "\n")
            asked[0] += bool(text.strip())

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, todo))
    verdicts = [json.loads(line)["found_by_judge"] for line in out.read_text().splitlines()]
    return {"cases": len(verdicts), "asked": asked[0], "found": sum(value is True for value in verdicts),
            "unclear": sum(value is None for value in verdicts)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("results", "cases", "url", "model"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args(argv)

    from conpact import Endpoint
    from conpact.prompts import judge_prompts

    from studies.context_rot.conpact_bridge import RetentionJudge
    from studies.context_rot.repeated import ServedChat

    client = Endpoint(arguments.model, arguments.url, "EMPTY", {}).client()
    judge = RetentionJudge(ServedChat(client, arguments.model))
    rules = {case["id"]: case["sc_text"] for case in map(json.loads, Path(arguments.cases).read_text().splitlines())}
    print(json.dumps(judge_found(Path(arguments.results), rules,
                                 lambda rule, delivered: judge.verdict(*judge_prompts(rule, delivered)), workers=arguments.workers)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
