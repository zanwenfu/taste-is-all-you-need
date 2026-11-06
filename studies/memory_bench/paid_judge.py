"""The second judge on the real test: GPT-5.4 on Amazon Bedrock, paid from AWS credit, on the user's yes of Nov 6 2026.

It reads the LongMemEval answers of the rows the main table reads, for both readers, with LongMemEval's own prompts, under one spend limit over the whole job. About 9,900
verdicts at about $0.00133 each, by the 50 of the sample: about $13, stopping by itself at $20, and each report says the moment the spend passes the estimate. Bedrock's batch
table lists no GPT-5.4, so these are ordinary calls. The key stays on the laptop, so this runs there, over small copies of the answers and a reference cut to what a judge reads,
and its verdicts go back beside the answers. Batches stay small and each is read before the next.

    python -m studies.memory_bench.paid_judge reference --data lme.json --split split.json --part held_out --out reference.json   # where the data is
    python -m studies.memory_bench.paid_judge run --root copy/ --reference reference.json --row oss/mem0-8000 --first 5 --part-split split.json
    python -m studies.memory_bench.paid_judge run --root copy/ --reference reference.json --row oss/mem0-8000 --stand-in     # no key, no call
    python -m studies.memory_bench.paid_judge status --root copy/
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

from studies.context_rot.lme_judge import JUDGES, judge_file
from studies.memory_bench.analyze import (
    ALONE,
    ARMS,
    MAIN_BUDGET,
    OURS,
    _sha256,
    readers_of,
    verdicts,
)

SHORT_NAME = "gpt-5.4"
ESTIMATE_USD = 13.0
STOP_AT_USD = 20.0
MAX_TOKENS = 512  # the sample's largest output was 107 tokens
EFFORT = "low"
FIELDS = ("question_id", "question_type", "question", "answer", "question_date")
_ANSWERS = "answers.jsonl"
_VERDICTS = f"{_ANSWERS}.eval-results-{SHORT_NAME}"
_FREE = f"{_ANSWERS}.eval-results-served"


def rows_judged() -> list[tuple[str, str]]:
    """(reader, row) for every row the main table reads, but no memory, which holds nothing for a memory's judge to tell apart."""
    rows = [f"{arm}-{MAIN_BUDGET}" for arm in (*OURS, *ARMS["lme"])] + [row for row in ALONE["lme"] if row != f"none-{MAIN_BUDGET}"]
    return sorted((reader, row) for row in rows for reader in readers_of(row))


VERDICTS = 450 * 22  # the real test's questions, by these rows


def _folder(root: Path, reader: str, row: str) -> Path:
    return Path(root) / f"answers-{reader}" / "lme" / row


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _cost(path: Path) -> float:
    return sum(row["autoeval_label"].get("cost_usd_estimate") or 0.0 for row in _lines(path))


def spent(root: Path) -> float:
    """What this job has spent so far, by the prices in bedrock.py, over every row judged."""
    return sum(_cost(path) for path in Path(root).glob(f"answers-*/lme/*/{_VERDICTS}"))


def judged(root: Path) -> int:
    return sum(len(_lines(path)) for path in Path(root).glob(f"answers-*/lme/*/{_VERDICTS}"))


def run_row(root: Path, reference: Path, reader: str, row: str, judge: Any, *, allowed: set[str] | None, first: int | None = None, workers: int = 8,
            stop_at_usd: float = STOP_AT_USD, estimate_usd: float = ESTIMATE_USD, commit: str | None = None) -> dict[str, Any]:
    """Judge this row's answers with no verdict yet, or its next ``first``, and report what came back. ``allowed`` holds the real test's questions, and an answer to anything
    else is refused before any call, since the user's yes covers the real test; None only for a stand-in."""
    if (reader, row) not in rows_judged():
        raise ValueError(f"{reader}/{row} is not a row of this job")
    folder = _folder(root, reader, row)
    answers = folder / _ANSWERS
    if allowed is not None:
        other = {entry["question_id"] for entry in _lines(answers)} - allowed
        if other:
            raise ValueError(f"{len(other)} answers of {reader}/{row} are not the real test's")
    out = folder / _VERDICTS
    (folder / f"{_VERDICTS}.settings.json").write_text(json.dumps({
        "model": judge.model, "effort": judge.effort, "max_tokens": judge.max_tokens, "commit": commit,
        "answers_sha256": _sha256(answers), "reference_sha256": _sha256(reference)}, indent=1))
    judge_file(answers, reference, judge, short_name=SHORT_NAME, workers=workers, stop_at_usd=stop_at_usd, spent_elsewhere=spent(root) - _cost(out), first=first)
    return report_row(root, reader, row, estimate_usd=estimate_usd)


def report_row(root: Path, reader: str, row: str, *, estimate_usd: float = ESTIMATE_USD) -> dict[str, Any]:
    folder = _folder(root, reader, row)
    rows = _lines(folder / _VERDICTS)
    labels = {entry["question_id"]: entry["autoeval_label"]["label"] for entry in rows}
    free = verdicts(folder / _FREE) if (folder / _FREE).exists() else {}
    both = [question for question, label in labels.items() if label is not None and question in free]
    total, count = spent(root), judged(root)
    outputs = sorted(entry["autoeval_label"].get("output_tokens") or 0 for entry in rows)
    return {"row": f"{reader}/{row}", "judged": len(rows), "of": len(_lines(folder / _ANSWERS)), "yes": sum(label is True for label in labels.values()),
            "no": sum(label is False for label in labels.values()), "unparsed": sum(label is None for label in labels.values()),
            "cut_off": sum(entry["autoeval_label"].get("stop_reason") == "max_tokens" for entry in rows), "most_output_tokens": outputs[-1] if outputs else 0,
            "free_read": len(both), "free_agrees": sum(labels[question] == free[question] for question in both),
            "spent_total": total, "judged_total": count, "projected_total": total / count * VERDICTS if count else None, "past_estimate": total > estimate_usd}


def cut_reference(data: Path, split: Path, part: str, out: Path) -> int:
    """What a judge reads of each question of this part, from the file the reader's prompt came from. No history, so it is small enough to leave the cluster."""
    wanted = set(json.loads(Path(split).read_text())[part])
    rows = [{key: entry[key] for key in FIELDS} for entry in json.loads(Path(data).read_text()) if entry["question_id"] in wanted]
    Path(out).write_text(json.dumps(rows))
    return len(rows)


def _stand_in() -> Any:
    """No key and no call: the whole path, for the free dry run. Each verdict is a yes at the sample's mean price."""
    class StandIn:
        model, effort, max_tokens = "stand-in", EFFORT, MAX_TOKENS

        def ask(self, prompt: str) -> types.SimpleNamespace:
            return types.SimpleNamespace(text="yes", input_tokens=1, output_tokens=1, stop_reason="end_turn", cost=lambda model: 0.00133)

    return StandIn()


def _run(arguments: argparse.Namespace) -> int:
    reader, row = arguments.row.split("/", 1)
    if arguments.stand_in:
        judge, allowed = _stand_in(), None
    else:
        from studies.context_rot.bedrock import Converse

        if not arguments.part_split:
            raise ValueError("the paid judge needs --part-split, to hold it to the real test")
        allowed = set(json.loads(Path(arguments.part_split).read_text())["held_out"])
        judge = Converse(JUDGES[SHORT_NAME], effort=EFFORT, max_tokens=MAX_TOKENS)
    print(json.dumps(run_row(Path(arguments.root), Path(arguments.reference), reader, row, judge, allowed=allowed, first=arguments.first, workers=arguments.workers,
                             stop_at_usd=arguments.stop_at_usd, commit=arguments.commit)))
    return 0


def _status(arguments: argparse.Namespace) -> int:
    for reader, row in rows_judged():
        if (_folder(Path(arguments.root), reader, row) / _VERDICTS).exists():
            print(json.dumps(report_row(Path(arguments.root), reader, row)))
    print(json.dumps({"spent_total": spent(Path(arguments.root)), "judged_total": judged(Path(arguments.root)), "of": VERDICTS}))
    return 0


def _reference(arguments: argparse.Namespace) -> int:
    print(cut_reference(Path(arguments.data), Path(arguments.split), arguments.part, Path(arguments.out)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    stage = stages.add_parser("run")
    for name in ("root", "reference", "row"):
        stage.add_argument(f"--{name}", required=True)
    stage.add_argument("--first", type=int, help="only the next this many, so that a batch can be read before the next")
    stage.add_argument("--workers", type=int, default=8)
    stage.add_argument("--stop-at-usd", type=float, default=STOP_AT_USD)
    stage.add_argument("--commit")
    stage.add_argument("--part-split", help="the split file: an answer outside its held_out part is refused before any call")
    stage.add_argument("--stand-in", action="store_true", help="no key, no call, for the dry run")
    stage.set_defaults(action=_run)
    stage = stages.add_parser("status")
    stage.add_argument("--root", required=True)
    stage.set_defaults(action=_status)
    stage = stages.add_parser("reference")
    for name in ("data", "split", "part", "out"):
        stage.add_argument(f"--{name}", required=True)
    stage.set_defaults(action=_reference)
    arguments = parser.parse_args(argv)
    return arguments.action(arguments)


if __name__ == "__main__":
    sys.exit(main())
