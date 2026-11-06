"""ConPact's SWE-Natural set in the comparison: 126 rules from real GitHub issues, each with the coding-agent log that
resolved the issue, and whether an agent still obeys the rule after its history is replaced.

The main table asks no question. Each system hands over what it would put in front of an agent (``memories``), cut to
one token cap in its own order, and that takes the place of the history. A second table lets each system search with
the probe's request, the same cap, and is never pooled with the first. There is no rule-free baseline in these logs, so
this reports compliance, and whether the block holds the rule by the retention judge in ``studies.memory_probe.judge``, run on the
blocks as a stage of its own; it reports no effective retention.

    python -m studies.memory_bench.swe export --out cases/                      # inside the container: it needs conpact
    python -m studies.memory_bench.swe rank --views cases/views.jsonl --system taste --query-free --workdir /tmp/x --out r.jsonl
    python -m studies.memory_bench.swe blocks --rankings r.jsonl --cases cases/cases.jsonl --tokenizer <folder> --out b.jsonl
    python -m studies.memory_bench.swe answer --blocks b.jsonl --cases cases/cases.jsonl --url URL --model NAME --out a.jsonl
    python -m studies.memory_bench.swe report --root runs/swe-X --commit X --digest D --out report.md

The stages run as ``jobs/swe.sbatch`` and ``jobs/swe_read.sbatch``. An answer on record stands only if this code makes its block again byte for byte, so that a later commit
reads nothing twice, and ``report`` writes no table from a run that is not whole.
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
from studies.memory_bench.core import fill
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


def none_rankings(cases: list[dict], out: Path) -> None:
    """The row with no memory: an empty block in place of the history, read by the same stage as every other row."""
    from studies.memory_ledger import BASELINE, stand_in_record

    ledger = stand_in_record(BASELINE, code="no memory")
    Path(out).write_text("".join(json.dumps({"qid": neutral("q", case["id"]), "items": [], "system": "none", "ledger": ledger}) + "\n" for case in cases))


def _rows_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _block_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()


def carry(blocks: Path, answers: Path, out: Path) -> dict[str, int]:
    """Keep each answer on record whose block this code makes again byte for byte. A case whose block changed, or that was never read, is left for
    the answer stage, which asks only what has no answer, so nothing that stands is read twice and the set's cases are read once."""
    wanted = {row["id"]: _block_sha(row["block"]) for row in _rows_of(blocks)}
    old = {row["id"]: row for row in _rows_of(answers)} if Path(answers).exists() else {}
    kept = [old[case] for case, sha in wanted.items() if case in old and old[case]["prompt_sha256"] == sha]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in kept))
    return {"kept": len(kept), "changed": sum(case in old for case in wanted) - len(kept), "never_read": sum(case not in old for case in wanted)}


def carry_verdicts(blocks: Path, judged: Path, out: Path) -> dict[str, int]:
    """The same for the retention judge: a verdict stands only for the block it read. ``out`` then holds every block to read and the verdicts that stand."""
    rows = _rows_of(blocks)
    folder = Path(judged)
    read = {row["id"]: row.get("delivered_text") or "" for row in _rows_of(folder / "results.jsonl")} if (folder / "results.jsonl").exists() else {}
    verdicts = {row["id"]: row for row in _rows_of(folder / "found.served.jsonl")} if (folder / "found.served.jsonl").exists() else {}
    kept = [verdicts[row["id"]] for row in rows if row["id"] in verdicts and read.get(row["id"]) == row["block"]]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text("".join(json.dumps({"id": row["id"], "delivered_text": row["block"]}) + "\n" for row in rows))
    (out / "found.served.jsonl").write_text("".join(json.dumps(row) + "\n" for row in kept))
    return {"kept": len(kept), "to_judge": len(rows) - len(kept)}


# Every row of the protocol's two tables, at both caps, and no other. No memory has one cap: an empty block is the same at any.
FREE_ROWS = ("flat_free", "taste_free", "taste_headers_free", "mem0f_free", "mempalace_free")
SEARCH_ROWS = ("flat_search", "taste_search", "taste_headers_search", "dense_search", "mem0f_search", "mempalace_search")
CAPS = (CAP, 8000)
NONE_ROW = f"none-{CAP}"
BASELINE_PROMPTS = ("anthropic", "pi_mono")
# Mem0's one ingest of this set took about 15 GPU-hours and ran on this commit, on the protocol's settings. Its adapter's code has not changed since but for a docstring, which a test holds.
MEM0_INGEST_COMMIT = "a43bb54bb31725046c4585a1a163ece60207be91"
MEM0_INGEST = {"mem0ai": "2.2.1", "embedder": "sentence-transformers/all-MiniLM-L6-v2", "model": "openai/gpt-oss-120b", "max_tokens": 4096, "effort": "low", "top_k": 1000, "date_prompt": True}


def required_rows() -> list[str]:
    return [NONE_ROW, *(f"{row}-{cap}" for row in (*FREE_ROWS, *SEARCH_ROWS) for cap in CAPS)]


def _check_rankings(path: Path, name: str, *, commit: str, digest: str) -> list[str]:
    if not path.exists():
        return [f"{name}: no rankings"]
    problems: list[str] = []
    for versions in {json.dumps(row.get("versions", {}), sort_keys=True) for row in _rows_of(path)}:
        made = json.loads(versions)
        if name.startswith("mem0"):
            if made.get("study_commit") != MEM0_INGEST_COMMIT:
                problems.append(f"{name}: rankings made by {made.get('study_commit')}, not the ingest's commit")
            if any(made.get(key) != value for key, value in MEM0_INGEST.items()):
                problems.append(f"{name}: rankings made with other settings than the protocol's")
        elif made.get("study_commit") != commit or made.get("code_digest") != digest:
            problems.append(f"{name}: rankings made by {made.get('study_commit')}, not this run's commit")
    return problems


def check_coding(root: Path, *, commit: str, digest: str) -> list[str]:
    """Why a table of this set may not be written, or nothing: every row and no other, every case once, each answer read from the block on record, each verdict
    on it, and every ranking from this commit, or Mem0's from its one ingest on the protocol's settings."""
    root = Path(root)
    problems: list[str] = []
    ids = sorted(row["id"] for row in _rows_of(root / "cases" / "cases.jsonl"))
    wanted = required_rows()
    for path in sorted((root / "blocks").glob("*.jsonl")):
        if path.stem not in wanted:
            problems.append(f"{path.stem}: not a row of the protocol")
    for row in wanted:
        name = row.rsplit("-", 1)[0]
        blocks_path, answers_path = root / "blocks" / f"{row}.jsonl", root / "answers" / f"{row}.jsonl"
        if not blocks_path.exists():
            problems.append(f"{row}: no blocks")
            continue
        blocks = {item["id"]: item["block"] for item in _rows_of(blocks_path)}
        if sorted(blocks) != ids:
            problems.append(f"{row}: blocks hold {len(blocks)} of {len(ids)}")
        if not answers_path.exists():
            problems.append(f"{row}: no answers")
            continue
        answers = _rows_of(answers_path)
        if sorted(item["id"] for item in answers) != ids:
            problems.append(f"{row}: answers hold {len({item['id'] for item in answers})} of {len(ids)}")
        other = sum(item["id"] in blocks and item["prompt_sha256"] != _block_sha(blocks[item["id"]]) for item in answers)
        if other:
            problems.append(f"{row}: {other} answers were read from another block")
        if name == "none":
            continue
        folder = root / "judged" / row
        if not (folder / "found.served.jsonl").exists() or not (folder / "results.jsonl").exists():
            problems.append(f"{row}: no verdicts")
        else:
            read = {item["id"]: item.get("delivered_text") or "" for item in _rows_of(folder / "results.jsonl")}
            if read != blocks:
                problems.append(f"{row}: the judge read another block")
            if sorted(item["id"] for item in _rows_of(folder / "found.served.jsonl")) != ids:
                problems.append(f"{row}: verdicts do not hold every case once")
    for name in (*FREE_ROWS, *SEARCH_ROWS):
        problems += _check_rankings(root / "rankings" / f"{name}.jsonl", name, commit=commit, digest=digest)
    for prompt in BASELINE_PROMPTS:
        if not (root / "baselines" / f"baseline-{prompt}.report.json").exists():
            problems.append(f"no baseline for {prompt}")
    return problems


def _paired(mine: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Cases both read, by the sign test. An unread answer is left out of a difference."""
    from studies.memory_bench.analyze import sign_test

    both = [(mine[case], other[case]) for case in mine if case in other and mine[case] is not None and other[case] is not None]
    better, worse = sum(a and not b for a, b in both), sum(b and not a for a, b in both)
    return {"better": better, "worse": worse, "p": sign_test(better, worse)}


def coding_table(root: Path) -> list[dict[str, Any]]:
    root = Path(root)
    marks = {row: {item["id"]: item["compliant"] for item in _rows_of(root / "answers" / f"{row}.jsonl")} for row in required_rows()}
    table = []
    for row in required_rows():
        name, cap = row.rsplit("-", 1)
        answers = list(marks[row].values())
        held = None
        unclear = 0
        if name != "none":
            verdicts = [item["found_by_judge"] for item in _rows_of(root / "judged" / row / "found.served.jsonl")]
            held, unclear = sum(value is True for value in verdicts), sum(value is None for value in verdicts)
        blocks = _rows_of(root / "blocks" / f"{row}.jsonl")
        taste = f"taste_{'free' if name.endswith('_free') else 'search'}-{cap}"
        table.append({"row": row, "cases": len(answers), "obeyed": sum(value is True for value in answers), "read": sum(value is not None for value in answers),
                      "unread": sum(value is None for value in answers), "held": held, "held_unclear": unclear,
                      "tokens_mean": sum(item["tokens"] for item in blocks) / len(blocks) if blocks else 0,
                      "against_none": None if name == "none" else _paired(marks[row], marks[NONE_ROW]),
                      "against_taste": None if name == "none" or row == taste else _paired(marks[row], marks[taste])})
    return table


def coding_baselines(root: Path) -> dict[str, dict[str, float]]:
    """The benchmark's own rows, under each summary prompt, as obeyed of the set."""
    baselines = {}
    for prompt in BASELINE_PROMPTS:
        report = json.loads((Path(root) / "baselines" / f"baseline-{prompt}.report.json").read_text())
        row = next(item for item in report["package_summary"] if item["source"] == "swe_natural" and item["sc_type"] == "all")
        share = row["n"] / 100
        baselines[prompt] = {"compaction": row["compaction_compliance_pct"] * share, "restated": row["upper_bound_compliance_pct"] * share,
                             "whole log": row["full_with_sc_compliance_pct"] * share, "whole log, rule taken out": row["full_without_sc_compliance_pct"] * share}
    return baselines


def render_coding(table: list[dict[str, Any]], baselines: dict[str, dict[str, float]]) -> str:
    def pair(cell: dict[str, Any] | None) -> str:
        return "" if cell is None else f"+{cell['better']} -{cell['worse']}, p {cell['p']:.3f}"

    lines = ["| row | obeyed | read | held | held unclear | tokens | against no memory | against Taste |", "|---|---|---|---|---|---|---|---|"]
    for row in table:
        lines.append(f"| {row['row']} | {row['obeyed']} | {row['read']} | {'' if row['held'] is None else row['held']} | {row['held_unclear'] or ''} | {row['tokens_mean']:.0f} | {pair(row['against_none'])} | {pair(row['against_taste'])} |")
    lines += ["", "The benchmark's own rows, obeyed of the set:", "", "| prompt | compaction | restated | whole log | whole log, rule taken out |", "|---|---|---|---|---|"]
    for prompt, row in baselines.items():
        lines.append(f"| {prompt} | {row['compaction']:.0f} | {row['restated']:.0f} | {row['whole log']:.0f} | {row['whole log, rule taken out']:.0f} |")
    return "\n".join(lines) + "\n"


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


def headers_stage(arguments: argparse.Namespace) -> int:
    from studies.memory_bench.headers import write_all
    from studies.memory_probe.headers import served_ask

    views = [json.loads(line) for line in Path(arguments.views).read_text().splitlines()]
    counts = write_all(views, served_ask(arguments.url, arguments.model, max_tokens=arguments.max_tokens), Path(arguments.out),
                       workers=arguments.workers)
    print(json.dumps(counts))
    return 0 if counts["failed"] == 0 else 1


def answer(arguments: argparse.Namespace) -> int:
    from studies.memory_probe.tool_loop import http_chat

    blocks = {row["id"]: row for row in map(json.loads, Path(arguments.blocks).read_text().splitlines())}
    cases = [case for case in _read_cases(arguments.cases) if case["id"] in blocks]
    system = next(iter(blocks.values()))["system"]
    chat = http_chat(arguments.url, arguments.model, max_tokens=arguments.max_tokens)
    print(json.dumps(answer_all(cases, blocks, chat, Path(arguments.out), workers=arguments.workers, system=system)))
    return 0


def none_stage(arguments: argparse.Namespace) -> int:
    none_rankings(_read_cases(arguments.cases), Path(arguments.out))
    return 0


def carry_stage(arguments: argparse.Namespace) -> int:
    print(json.dumps({"answers": carry(Path(arguments.blocks), Path(arguments.answers), Path(arguments.out)),
                      **({"verdicts": carry_verdicts(Path(arguments.blocks), Path(arguments.judged), Path(arguments.judged_out))} if arguments.judged else {})}))
    return 0


def report_stage(arguments: argparse.Namespace) -> int:
    root = Path(arguments.root)
    problems = check_coding(root, commit=arguments.commit, digest=arguments.digest)
    if problems:
        print(f"{len(problems)} problems, and no table:", *problems[:40], sep="\n  ")
        return 1
    table = coding_table(root)
    out = Path(arguments.out)
    out.write_text(f"The coding set, code {arguments.commit}, Mem0's ingest {MEM0_INGEST_COMMIT}.\n\n" + render_coding(table, coding_baselines(root)))
    out.with_suffix(".json").write_text(json.dumps(table, indent=1))
    print(f"{len(table)} rows at {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    stage = stages.add_parser("export")
    stage.add_argument("--out", required=True)
    stage.set_defaults(run=export)
    stage = stages.add_parser("none")
    for name in ("cases", "out"):
        stage.add_argument(f"--{name}", required=True)
    stage.set_defaults(run=none_stage)
    stage = stages.add_parser("carry")
    for name in ("blocks", "answers", "out"):
        stage.add_argument(f"--{name}", required=True)
    stage.add_argument("--judged")
    stage.add_argument("--judged-out")
    stage.set_defaults(run=carry_stage)
    stage = stages.add_parser("report")
    for name in ("root", "commit", "digest", "out"):
        stage.add_argument(f"--{name}", required=True)
    stage.set_defaults(run=report_stage)
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
    stage = stages.add_parser("headers")
    for name in ("views", "url", "model", "out"):
        stage.add_argument(f"--{name}", required=True)
    stage.add_argument("--max-tokens", type=int, default=1024)
    stage.add_argument("--workers", type=int, default=16)
    stage.set_defaults(run=headers_stage)

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
