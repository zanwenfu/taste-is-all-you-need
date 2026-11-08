"""Paper 1's probe runs: every case of a set, one arm, one condition, one summary prompt and depth.

    python -m studies.memory_probe.run --inputs runs/probe/inputs --prompt anthropic --k 1 --arm a5 --condition ii \\
        --url http://127.0.0.1:8000/v1 --model openai/gpt-oss-120b --tokenizer <gpt-oss folder>/tokenizer.json \\
        --workdir /tmp/probe --out runs/probe/a5-ii-anthropic-k1/results.jsonl --workers 12 [--limit 50]

Each case gets a fresh store. Results are written as they finish, and a stopped run picks up where it was. The
summary counts the steps (looked, found, obeyed), the tool calls that could not be parsed, and the ledger. Building a store is
Python work on one core however many threads wait on it, so a row runs fastest as many processes, each with ``--shard I/N``, joined
by ``studies.memory_probe.merge``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from studies.memory_ledger import check, code_digest, code_version
from studies.memory_probe.cases import _NEEDS_ITEMS, _NEEDS_NOTES, _NEEDS_RECORDS, ARMS, run_case
from studies.memory_probe.tool_loop import effort_arg, http_chat


def history_id(case_id: str) -> str:
    """A cued case has the published case's history word for word, so it shares that case's summaries, records and notes."""
    return case_id.removesuffix(":cued")


def shard_of(cases: list[dict], shard: str | None) -> list[dict]:
    """Every Nth case from the Ith, so that N processes hold every case once."""
    if shard is None:
        return cases
    index, total = (int(part) for part in shard.split("/"))
    if not 0 <= index < total:
        raise ValueError(f"no shard {shard}")
    return cases[index::total]


def delivered_blocks(path: Path, prompt: str) -> dict[str, str]:
    """A summary prompt's own rule records, by case (see ``export.delivered``)."""
    return {row["id"]: row["block"] for row in map(json.loads, Path(path).read_text().splitlines())
            if row["prompt"] == prompt}


def delivered_items(path: Path, prompt: str) -> dict[str, list[dict]]:
    """The same records with the message each came from, by case."""
    return {row["id"]: row["items"] for row in map(json.loads, Path(path).read_text().splitlines()) if row["prompt"] == prompt}


def _rate(rows: list[dict], key: str) -> dict[str, Any]:
    values = [row[key] for row in rows if row[key] is not None]
    return {"n": len(values), "true": sum(map(bool, values)), "pct": 100 * sum(map(bool, values)) / len(values)
            if values else None, "unreadable": len(rows) - len(values)}


def summarize(rows: list[dict]) -> dict[str, Any]:
    calls = [call for row in rows for call in row["calls"]]
    return {
        "cases": len(rows),
        "looked": _rate(rows, "looked"), "found": _rate(rows, "found"),
        "rule_turn_reached": _rate(rows, "rule_turn_reached"), "compliant": _rate(rows, "compliant"),
        "tool_calls": len(calls), "unparsed_arguments": sum(row["unparsed_arguments"] for row in rows),
        "tool_errors": sum(row["tool_errors"] for row in rows),
        "budget_refusals": sum(row["budget_refusals"] for row in rows),
        "round_limit_hit": sum(row["round_limit_hit"] for row in rows),
        "final_names_a_tool": sum(row["final_names_a_tool"] for row in rows),
        "acted": sum(bool(row.get("action")) for row in rows),
        "searches_mean": sum(row["searches"] for row in rows) / len(rows) if rows else None,
        "ledger": check([row["ledger"] for row in rows]),
    }


def _same_settings(out: Path, identity: dict[str, Any]) -> None:
    """A second run carries on only with what the first was told, or two budgets or two wordings would sit in one file."""
    path = out.with_name(out.stem + ".settings.json")
    if path.exists():
        before = json.loads(path.read_text())
        changed = sorted(key for key in identity if before.get(key) != identity[key])
        if changed:
            raise SystemExit(f"{out} was started with other settings: {changed}")
    path.write_text(json.dumps(identity, indent=1))


def run_all(cases: list[dict], summaries: dict[str, str], *, chat: Callable, out: Path, workers: int,
            delivered: dict[str, str] | None = None, notes: dict[str, list[str]] | None = None,
            items: dict[str, list[dict]] | None = None, identity: dict[str, Any] | None = None, **settings: Any) -> dict[str, Any]:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if identity is not None:
        _same_settings(out, identity)
    out.touch()  # a shard with no cases is still a shard
    done = {json.loads(line)["id"] for line in out.read_text().splitlines()}
    lock = threading.Lock()

    def one(case: dict) -> None:
        source = history_id(case["id"])
        extra = {"delivered": delivered[source]} if delivered is not None else {}
        if notes is not None:
            extra["notes"] = notes[source]
        if items is not None:
            extra["items"] = items[source]
        row = run_case(case, summaries[source], chat=chat, **extra, **settings)
        row["final_names_a_tool"] = any(name in row["final"] for name in ("memory_search", "memory_fetch"))
        with lock, out.open("a") as written:
            written.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, [case for case in cases if case["id"] not in done]))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    if sorted(row["id"] for row in rows) != sorted(case["id"] for case in cases):
        raise ValueError("the results do not hold every case exactly once")
    summary = summarize(rows)
    out.with_name(out.stem + ".summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("inputs", "url", "model", "tokenizer", "workdir", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--prompt", required=True, choices=("pi_mono", "anthropic"))
    parser.add_argument("--k", required=True, type=int, choices=(1, 16))
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--notes", help="the notes arms: the notes versions (studies.memory_probe.notes)")
    parser.add_argument("--keep", default="newest", choices=("newest", "oldest"), help="a2a: which end is kept")
    parser.add_argument("--condition", required=True, choices=("i", "ii", "iii", "iv", "v", "vi", "vii", "viii"))
    parser.add_argument("--limit", type=int, help="the first N cases in file order, for a smoke run")
    parser.add_argument("--shard", help="I/N: this process holds every Nth case from the Ith")
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--budget", type=int, default=4096, help="memory tokens a probe may receive in all")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--effort", default="low", help='the reasoning effort sent with each call; "none" sends none, for a model that takes no such setting')
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--upper", action="store_true", help="ask the restated probe instead (the ceiling)")
    parser.add_argument("--headers", help="the header arm's stored headers (studies.memory_probe.headers)")
    arguments = parser.parse_args(argv)

    from tokenizers import Tokenizer

    from studies.memory_probe.headers import StoredTurnHeaders

    inputs = Path(arguments.inputs)
    cases = [json.loads(line) for line in (inputs / "cases.jsonl").read_text().splitlines()]
    cases = shard_of(cases[: arguments.limit] if arguments.limit else cases, arguments.shard)
    summaries = {row["id"]: row["summary"] for row in map(json.loads, (inputs / "summaries.jsonl").read_text().splitlines())
                 if row["prompt"] == arguments.prompt and row["k"] == arguments.k}
    tokenizer = Tokenizer.from_file(arguments.tokenizer)
    blocks = None
    if arguments.arm in _NEEDS_RECORDS:
        blocks = delivered_blocks(inputs / "delivered.jsonl", arguments.prompt)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    chat = http_chat(arguments.url, arguments.model, max_tokens=arguments.max_tokens, effort=effort_arg(arguments.effort))
    Path(arguments.workdir).mkdir(parents=True, exist_ok=True)
    versions = None
    if arguments.arm in _NEEDS_NOTES:
        versions = {row["id"]: row["versions"] for row in map(json.loads, Path(arguments.notes).read_text().splitlines())}
    items = delivered_items(inputs / "delivered.jsonl", arguments.prompt) if arguments.arm in _NEEDS_ITEMS else None

    def digest(path: str | Path | None) -> str | None:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).exists() else None

    identity = {"arm": arguments.arm, "condition": arguments.condition, "prompt": arguments.prompt, "k": arguments.k,
                "upper": arguments.upper, "budget": arguments.budget, "keep": arguments.keep, "max_rounds": arguments.max_rounds,
                "model": arguments.model, "limit": arguments.limit, "shard": arguments.shard, "cases_sha256": digest(inputs / "cases.jsonl"),
                "summaries_sha256": digest(inputs / "summaries.jsonl"), "notes_sha256": digest(arguments.notes),
                "headers_sha256": digest(arguments.headers), "delivered_sha256": digest(inputs / "delivered.jsonl"),
                "tokenizer_sha256": digest(arguments.tokenizer), "commit": code_version()[0], "code_digest": code_digest()}
    # A row's settings name its cap and effort only where they are not the ones every row before had, so that those rows carry on as they were.
    if arguments.effort != "low":
        identity["effort"] = arguments.effort
    if arguments.max_tokens != 4096:
        identity["max_tokens"] = arguments.max_tokens
    headers = StoredTurnHeaders(Path(arguments.headers)) if arguments.headers else None
    summary = run_all(cases, summaries, chat=chat, out=Path(arguments.out), workers=arguments.workers, delivered=blocks,
                      notes=versions, items=items, identity=identity,
                      keep=arguments.keep,
                      arm=arguments.arm, condition=arguments.condition, workdir=Path(arguments.workdir),
                      max_rounds=arguments.max_rounds, budget=arguments.budget, count=count,
                      prompt=arguments.prompt, k=arguments.k, upper=arguments.upper,
                      header=headers)
    print(json.dumps({key: summary[key] for key in ("cases", "looked", "found", "compliant", "acted", "unparsed_arguments")}))
    if headers is not None and headers.missing:
        print(f"{headers.missing} turns had no header, so this row is not what it says", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
