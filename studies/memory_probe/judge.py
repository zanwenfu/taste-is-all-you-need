"""Whether what a store delivered held the rule, when it may say it in its own words.

A whole-text match answers this for the arms that hold the user's own words. A notes file is rewritten by a model and may
paraphrase, so its arms are read by the benchmark package's own retention judge, asked of the served model, on the text
each case delivered. That is the judge the study already calibrated against GPT-5.4 at one compaction.

    python -m studies.memory_probe.judge --results run/results.jsonl --cases inputs/cases.jsonl --url URL --model NAME
    python -m studies.memory_probe.judge --notes notes-told.jsonl --out stored-told.served.jsonl --cases inputs/cases.jsonl --url URL --model NAME
    python -m studies.memory_probe.judge --results run/results.jsonl --summaries inputs/summaries.jsonl --prompt anthropic --k 1 --cases inputs/cases.jsonl --url URL --model NAME

The second form asks whether a notes file holds the rule at all, in its newest version and in any version: the "stored" step, which an append-only record has by
construction and a file a model rewrites may not. The third reads a summary row's own summaries, whose block is the summary itself: what compaction kept.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def summaries_of(path: Path, prompt: str, k: int) -> dict[str, str]:
    """This prompt's and this depth's summaries, by case. A case found twice is refused."""
    found: dict[str, str] = {}
    for row in map(json.loads, Path(path).read_text().splitlines()):
        if row["prompt"] == prompt and row["k"] == k:
            if row["id"] in found:
                raise ValueError(f"{row['id']} has two summaries at {prompt}, {k} rounds: found twice")
            found[row["id"]] = row["summary"]
    return found


def judge_found(results: Path, rules: dict[str, str], verdict: Callable[[str, list[dict]], bool | None], *,
                workers: int, texts: dict[str, str] | None = None) -> dict[str, int]:
    """One row per case in found.served.jsonl beside the results. Nothing delivered is not found, with no call. A row whose memory is its summary delivers no
    block of its own, so ``texts`` gives the judge the summary to read, and a case with none is refused."""
    results = Path(results)
    out = results.with_name("found.served.jsonl")
    done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    todo = [row for row in rows if row["id"] not in done]
    lock = threading.Lock()
    asked = [0]

    def one(row: dict) -> None:
        text = texts[row["id"]] if texts is not None else row.get("delivered_text") or ""
        answer = False
        if text.strip():
            answer = verdict(rules[row["id"]], [{"role": "assistant", "content": text}])
        with lock, out.open("a") as written:
            written.write(json.dumps({"id": row["id"], "found_by_judge": answer}) + "\n")
            asked[0] += bool(text.strip())

    if texts is not None:
        for row in todo:
            texts[row["id"]]  # before any call
    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, todo))
    verdicts = [json.loads(line)["found_by_judge"] for line in out.read_text().splitlines()]
    return {"cases": len(verdicts), "asked": asked[0], "found": sum(value is True for value in verdicts),
            "unclear": sum(value is None for value in verdicts)}


def judge_stored(notes: Path, rules: dict[str, str], verdict: Callable[[str, list[dict]], bool | None], out: Path, *, workers: int) -> dict[str, int]:
    """One row per case. A rule in the newest version is stored, with nothing more to ask; otherwise the older versions are read from the first, until one holds it."""
    out = Path(out)
    done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
    rows = [json.loads(line) for line in Path(notes).read_text().splitlines()]
    lock = threading.Lock()

    def one(row: dict) -> None:
        rule = rules[row["id"]]

        def holds_it(version: str) -> bool | None:
            return verdict(rule, [{"role": "assistant", "content": version}])

        newest = holds_it(row["versions"][-1])
        anywhere, asked, unclear = newest, 1, newest is None
        if newest is not True:
            anywhere = False
            for version in row["versions"][:-1]:
                asked += 1
                answer = holds_it(version)
                unclear = unclear or answer is None
                if answer is True:
                    anywhere = True
                    break
            else:
                anywhere = None if unclear else False
        with lock, out.open("a") as written:
            written.write(json.dumps({"id": row["id"], "newest": newest, "any": anywhere, "asked": asked}) + "\n")

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, [row for row in rows if row["id"] not in done]))
    verdicts = [json.loads(line) for line in out.read_text().splitlines()]
    return {"cases": len(verdicts), "stored_newest": sum(row["newest"] is True for row in verdicts), "stored_any": sum(row["any"] is True for row in verdicts),
            "unclear": sum(row["newest"] is None or row["any"] is None for row in verdicts), "asked": sum(row["asked"] for row in verdicts)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("cases", "url", "model"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("results", "notes", "out", "summaries", "prompt"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--k", type=int)
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args(argv)
    if bool(arguments.results) == bool(arguments.notes) or bool(arguments.notes) != bool(arguments.out):
        parser.error("give --results, or --notes and --out")
    if arguments.summaries and (arguments.notes or not arguments.prompt or arguments.k is None):
        parser.error("--summaries goes with --results, --prompt and --k")

    from conpact import Endpoint
    from conpact.prompts import judge_prompts

    from studies.context_rot.conpact_bridge import RetentionJudge
    from studies.context_rot.repeated import ServedChat

    client = Endpoint(arguments.model, arguments.url, "EMPTY", {}).client()
    judge = RetentionJudge(ServedChat(client, arguments.model))
    rules = {case["id"]: case["sc_text"] for case in map(json.loads, Path(arguments.cases).read_text().splitlines())}

    def verdict(rule: str, delivered: list[dict]) -> bool | None:
        return judge.verdict(*judge_prompts(rule, delivered))

    if arguments.notes:
        print(json.dumps(judge_stored(Path(arguments.notes), rules, verdict, Path(arguments.out), workers=arguments.workers)))
    else:
        texts = summaries_of(Path(arguments.summaries), arguments.prompt, arguments.k) if arguments.summaries else None
        print(json.dumps(judge_found(Path(arguments.results), rules, verdict, workers=arguments.workers, texts=texts)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
