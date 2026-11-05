"""ConPact's SWE-Natural set in the comparison: 126 rules from real GitHub issues, each with the coding-agent log that
resolved the issue, and whether an agent still obeys the rule after its history is replaced.

The main table asks no question. Each system hands over what it would put in front of an agent (``memories``), cut to
one token cap in its own order, and that takes the place of the history. A second table lets each system search with
the probe's request, the same cap, and is never pooled with the first. There is no rule-free baseline in these logs, so
this reports compliance and retention, not effective retention.

    python -m studies.memory_bench.swe export --out cases/                      # inside the container: it needs conpact
    python -m studies.memory_bench.swe rank --views cases/views.jsonl --system taste --query-free --workdir /tmp/x --out r.jsonl
    python -m studies.memory_bench.swe blocks --rankings r.jsonl --cases cases/cases.jsonl --tokenizer <folder> --out b.jsonl
    python -m studies.memory_bench.swe answer --blocks b.jsonl --cases cases/cases.jsonl --url URL --model NAME --out a.jsonl
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

from studies.context_rot.lme_clean import neutral
from studies.memory_bench.lme import _split_rounds

FIXED_DATE = "2026/01/01 (Thu) 00:00"  # the logs carry no dates, and every system's ingest wants one
CAP = 1000
CASE_FIELDS = ("id", "haystack_id", "sc_id", "sc_type", "sc_text", "compliant_letter", "system_prompt", "developer_prompt",
               "probe", "probe_prompt", "post_sc_probe_prompt", "messages")


def views_of(cases: list[dict]) -> list[dict]:
    """The shared ingest: per case its log, each round (a user message and what follows) as a session of its own, so
    that "one commit, one header, one add per session" means one per round for every system alike."""
    units = []
    for case in cases:
        sessions = [{"sid": neutral("s", f"{case['id']}/{number}"), "date": FIXED_DATE,
                     "turns": [{"role": turn["role"], "content": str(turn["content"])} for turn in part]}
                    for number, part in enumerate(_split_rounds(case["messages"]))]
        qid = neutral("q", case["id"])
        units.append({"unit": qid, "sessions": sessions, "queries": [{"qid": qid, "query": case["probe"]}]})
    return units


def fill(texts: list[str], cap: int, count: Callable[[str], int]) -> str:
    """In the system's own order until the cap. The one that does not fit is cut between words, and filling stops."""
    kept, used = [], 0
    for text in texts:
        size = count(text)
        if used + size <= cap:
            kept.append(text)
            used += size
            continue
        words = text.split(" ")
        low, high = 0, len(words)
        while low < high:
            middle = (low + high + 1) // 2
            if count(" ".join(words[:middle])) <= cap - used:
                low = middle
            else:
                high = middle - 1
        if low:
            kept.append(" ".join(words[:low]))
        break
    return "\n\n".join(kept)


def read_blocks(rankings: Path, cases: list[dict], *, cap: int, count: Callable[[str], int]) -> dict[str, dict]:
    """Per case, a rankings file's items cut to the cap. An item is one session here, so one round."""
    rows = {row["qid"]: row for row in map(json.loads, Path(rankings).read_text().splitlines())}
    blocks = {}
    for case in cases:
        row = rows[neutral("q", case["id"])]
        items = _pieces(row["items"])
        block = fill(items, cap, count)
        blocks[case["id"]] = {"block": block, "tokens": count(block), "offered": len(items), "system": row["system"],
                              "ledger": row["ledger"]}
    return blocks


def _pieces(items: list[dict]) -> list[str]:
    """Turn by turn, so that a round that does not fit is cut between its turns first."""
    pieces = []
    for item in items:
        turns = item["turns"]
        if len(turns) == 1 and turns[0]["role"] == "memory":
            pieces.append(turns[0]["content"])
        else:
            pieces += [f"{turn['role']}: {turn['content']}" for turn in turns]
    return pieces


def answer_all(cases: list[dict], blocks: dict[str, dict], chat: Callable, out: Path, *, workers: int, system: str) -> dict[str, Any]:
    """Ask every case not yet answered, with the block in place of its history, and count what came back."""
    from studies.memory_probe.cases import case_messages
    from studies.memory_probe.tool_loop import grade

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
    lock = threading.Lock()

    def one(case: dict) -> None:
        block = blocks[case["id"]]
        reply = chat(case_messages(case, block["block"], condition="i", upper=False), None)
        choice = reply.choices[0]
        final = choice.message.content or ""
        row = {"id": case["id"], "system": system, "sc_type": case["sc_type"], "final": final,
               "finish_reason": choice.finish_reason, "compliant": grade(final, case["compliant_letter"]),
               "block_tokens": block["tokens"], "usage": [reply.usage.prompt_tokens, reply.usage.completion_tokens],
               "prompt_sha256": hashlib.sha256(block["block"].encode("utf-8", "surrogateescape")).hexdigest()}
        with lock, out.open("a") as written:
            written.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, [case for case in cases if case["id"] not in done]))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    if sorted(row["id"] for row in rows) != sorted(case["id"] for case in cases):
        raise ValueError("the answers do not hold every case exactly once")
    summary = {"system": system, "cases": len(rows), "compliant": sum(row["compliant"] is True for row in rows),
               "unreadable": sum(row["compliant"] is None for row in rows),
               "block_tokens_mean": sum(row["block_tokens"] for row in rows) / len(rows) if rows else None}
    out.with_name(out.stem + ".summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _read_cases(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def export(arguments: argparse.Namespace) -> int:
    from conpact import load_static

    from studies.context_rot.conpact_bridge import DATA_REVISION

    instances = load_static("swe_natural", revision=DATA_REVISION)
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [{key: getattr(instance, key) for key in CASE_FIELDS} for instance in instances]
    (out / "cases.jsonl").write_text("".join(json.dumps(case) + "\n" for case in cases))
    (out / "views.jsonl").write_text("".join(json.dumps(unit) + "\n" for unit in views_of(cases)))
    print(json.dumps({"cases": len(cases), "revision": DATA_REVISION,
                      "views_sha256": hashlib.sha256((out / "views.jsonl").read_bytes()).hexdigest()}))
    return 0


def rank(arguments: argparse.Namespace) -> int:
    """Taste's store or the keyword list, run on the shared ingest as the other systems are."""
    from studies.memory_bench.adapters.common import QueryFree, run_adapter
    from studies.memory_bench.headers import stored_writer
    from studies.memory_bench.systems import FlatBm25, TasteGit

    Path(arguments.workdir).mkdir(parents=True, exist_ok=True)
    shard = tuple(int(part) for part in arguments.shard.split("/"))

    def make() -> Any:
        if arguments.system == "flat":
            system: Any = FlatBm25()
        else:
            system = TasteGit(Path(arguments.workdir), mode="delivered" if arguments.query_free else "single_query",
                              header_writer=stored_writer(Path(arguments.headers)) if arguments.headers else None)
        return QueryFree(system) if arguments.query_free else _Searching(system)

    name = f"{arguments.system}{'_headers' if arguments.headers else ''}{'_free' if arguments.query_free else '_search'}"
    print(json.dumps(run_adapter(Path(arguments.views), Path(arguments.out), make, system=name, code="studies/memory_bench/swe.py",
                                 versions={}, limit=arguments.limit, shard=shard)))
    return 0


class _Searching:
    """Taste's store and the keyword list, in the shape every adapter has."""

    def __init__(self, system: Any) -> None:
        self.system = system

    def ingest(self, sessions: list[dict]) -> None:
        self.system.ingest(sessions)

    def retrieve(self, query: str, *, date: str | None = None) -> list:
        return self.system.retrieve(query)

    def ledger(self) -> dict[str, Any]:
        return self.system.ledger()

    def close(self) -> None:
        self.system.close()


def blocks_stage(arguments: argparse.Namespace) -> int:
    from tokenizers import Tokenizer

    path = Path(arguments.tokenizer)
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json" if path.is_dir() else path))
    blocks = read_blocks(Path(arguments.rankings), _read_cases(arguments.cases), cap=arguments.cap,
                         count=lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids))
    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps({"id": key, **value}) + "\n" for key, value in blocks.items()))
    print(json.dumps({"cases": len(blocks), "tokens_mean": sum(b["tokens"] for b in blocks.values()) / len(blocks),
                      "empty": sum(not b["block"].strip() for b in blocks.values())}))
    return 0


def answer(arguments: argparse.Namespace) -> int:
    from studies.memory_probe.tool_loop import http_chat

    blocks = {row["id"]: row for row in map(json.loads, Path(arguments.blocks).read_text().splitlines())}
    cases = [case for case in _read_cases(arguments.cases) if case["id"] in blocks]
    system = next(iter(blocks.values()))["system"]
    chat = http_chat(arguments.url, arguments.model, max_tokens=arguments.max_tokens)
    print(json.dumps(answer_all(cases, blocks, chat, Path(arguments.out), workers=arguments.workers, system=system)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    stage = stages.add_parser("export")
    stage.add_argument("--out", required=True)
    stage.set_defaults(run=export)
    stage = stages.add_parser("rank")
    for name in ("views", "workdir", "out"):
        stage.add_argument(f"--{name}", required=True)
    stage.add_argument("--system", required=True, choices=("taste", "flat"))
    stage.add_argument("--headers", help="Taste with its model-written headers")
    stage.add_argument("--query-free", action="store_true")
    stage.add_argument("--limit", type=int)
    stage.add_argument("--shard", default="0/1")
    stage.set_defaults(run=rank)
    stage = stages.add_parser("blocks")
    for name in ("rankings", "cases", "tokenizer", "out"):
        stage.add_argument(f"--{name}", required=True)
    stage.add_argument("--cap", type=int, default=CAP)
    stage.set_defaults(run=blocks_stage)
    stage = stages.add_parser("answer")
    for name in ("blocks", "cases", "url", "model", "out"):
        stage.add_argument(f"--{name}", required=True)
    stage.add_argument("--max-tokens", type=int, default=4096)
    stage.add_argument("--workers", type=int, default=12)
    stage.set_defaults(run=answer)
    arguments = parser.parse_args(argv)
    return arguments.run(arguments)


if __name__ == "__main__":
    sys.exit(main())
