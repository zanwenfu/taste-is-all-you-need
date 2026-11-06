"""The second judge on the real test: GPT-5.4 on Amazon Bedrock, paid from AWS credit, on the user's yes of Nov 6 2026.

It reads the LongMemEval answers of the rows the main table reads, for both readers, with LongMemEval's own prompts, under one spend limit over the whole job. About 9,900
verdicts at about $0.00133 each, by the 50 of the sample: about $13, stopping by itself at $20, and each report says the moment the spend passes the estimate. Bedrock's batch
table lists no GPT-5.4, so these are ordinary calls. The key stays on the laptop, so this runs there, over small copies of the answers and a reference cut to what a judge reads,
and its verdicts go back beside the answers. Batches stay small and each is read before the next.

    python -m studies.memory_bench.paid_judge reference --data lme.json --split split.json --part held_out --out reference.json   # where the data is, after the part opens
    python -m studies.memory_bench.paid_judge run --root copy/ --reference reference.json --row oss/mem0-8000 --first 5 --part-split split.json
    python -m studies.memory_bench.paid_judge run --root copy/ --reference reference.json --row oss/mem0-8000 --stand-in     # no key, no call, in a file of its own
    python -m studies.memory_bench.paid_judge status --root copy/
    bash studies/memory_bench/paid_judge_files.sh fetch <run on the cluster> copy/    # the answers and free verdicts of these rows, small files, to where the key is
    bash studies/memory_bench/paid_judge_files.sh push copy/ <run on the cluster>     # the verdicts back beside them, each checked by md5
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from studies.context_rot.lme_judge import JUDGES, judge_file
from studies.memory_bench.analyze import SECOND_JUDGE, _sha256, second_judge_rows, verdicts
from studies.memory_bench.core import DATA_SHA256
from studies.memory_bench.run import OPENS_AT, _now

STAND_IN = "stand-in"
ESTIMATE_USD = 13.0
STOP_AT_USD = 20.0
MAX_TOKENS = 512  # the sample's largest output was 107 tokens
EFFORT = "low"
FIELDS = ("question_id", "question_type", "question", "answer", "question_date")
_ANSWERS = "answers.jsonl"
_FREE = f"{_ANSWERS}.eval-results-served"
VERDICTS = 450 * 22  # the real test's questions, by these rows


def _name(short_name: str) -> str:
    return f"{_ANSWERS}.eval-results-{short_name}"


@contextmanager
def only_one(root: Path) -> Iterator[None]:
    """One copy of this job at a time: two would each count the other's spend as a snapshot, and could pay twice for one verdict. The system frees the lock if a copy dies."""
    with (Path(root) / ".paid-judge.lock").open("a") as held:
        try:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("another copy of this job is already running") from error
        yield


def _folder(root: Path, reader: str, row: str) -> Path:
    return Path(root) / f"answers-{reader}" / "lme" / row


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _cost(path: Path) -> float:
    return sum(row["autoeval_label"].get("cost_usd_estimate") or 0.0 for row in _lines(path))


def spent(root: Path, short_name: str = SECOND_JUDGE) -> float:
    """What this job has spent so far, by the prices in bedrock.py, over every row judged."""
    return sum(_cost(path) for path in Path(root).glob(f"answers-*/lme/*/{_name(short_name)}"))


def judged(root: Path, short_name: str = SECOND_JUDGE) -> int:
    return sum(len(_lines(path)) for path in Path(root).glob(f"answers-*/lme/*/{_name(short_name)}"))


def run_row(root: Path, reference: Path, reader: str, row: str, judge: Any, *, allowed: set[str] | None, first: int | None = None, workers: int = 8,
            stop_at_usd: float = STOP_AT_USD, estimate_usd: float = ESTIMATE_USD, commit: str | None = None, short_name: str = SECOND_JUDGE) -> dict[str, Any]:
    """Judge this row's answers with no verdict yet, or its next ``first``, and report what came back. ``allowed`` holds the real test's questions, and an answer to anything
    else is refused before any call, since the user's yes covers the real test; None only for a stand-in. A stand-in writes to a file of its own, so that its verdicts can
    never count as the paid judge's."""
    if (reader, row) not in second_judge_rows():
        raise ValueError(f"{reader}/{row} is not a row of this job")
    folder = _folder(root, reader, row)
    answers = folder / _ANSWERS
    if allowed is not None:
        other = {entry["question_id"] for entry in _lines(answers)} - allowed
        if other:
            raise ValueError(f"{len(other)} answers of {reader}/{row} are not the real test's")
    out = folder / _name(short_name)
    theirs = {entry["autoeval_label"].get("model") for entry in _lines(out)} - {judge.model}
    if theirs:
        raise ValueError(f"{out} holds verdicts by another model: {sorted(map(str, theirs))}")
    settings = {"model": judge.model, "effort": judge.effort, "max_tokens": judge.max_tokens, "answers_sha256": _sha256(answers), "reference_sha256": _sha256(reference)}
    path = folder / f"{_name(short_name)}.settings.json"
    if path.exists() and out.exists() and _lines(out):
        before = json.loads(path.read_text())
        changed = [key for key, value in settings.items() if before.get(key) != value]
        if changed:
            raise ValueError(f"{out}: its verdicts were made on other settings: {changed}")
    path.write_text(json.dumps({**settings, "commit": commit}, indent=1))
    with only_one(root):
        judge_file(answers, reference, judge, short_name=short_name, workers=workers, stop_at_usd=stop_at_usd, spent_elsewhere=spent(root, short_name) - _cost(out), first=first)
    return report_row(root, reader, row, estimate_usd=estimate_usd, short_name=short_name)


def report_row(root: Path, reader: str, row: str, *, estimate_usd: float = ESTIMATE_USD, short_name: str = SECOND_JUDGE) -> dict[str, Any]:
    folder = _folder(root, reader, row)
    rows = _lines(folder / _name(short_name))
    labels = {entry["question_id"]: entry["autoeval_label"]["label"] for entry in rows}
    free = verdicts(folder / _FREE) if (folder / _FREE).exists() else {}
    both = [question for question, label in labels.items() if label is not None and question in free]
    total, count = spent(root, short_name), judged(root, short_name)
    outputs = sorted(entry["autoeval_label"].get("output_tokens") or 0 for entry in rows)
    return {"row": f"{reader}/{row}", "judged": len(rows), "of": len(_lines(folder / _ANSWERS)), "yes": sum(label is True for label in labels.values()),
            "no": sum(label is False for label in labels.values()), "unparsed": sum(label is None for label in labels.values()),
            "cut_off": sum(entry["autoeval_label"].get("stop_reason") == "max_tokens" for entry in rows), "most_output_tokens": outputs[-1] if outputs else 0,
            "free_read": len(both), "free_agrees": sum(labels[question] == free[question] for question in both),
            "spent_total": total, "judged_total": count, "projected_total": total / count * VERDICTS if count else None, "past_estimate": total > estimate_usd}


def cut_reference(data: Path, split: Path, part: str, out: Path, *, sha256: str = DATA_SHA256["lme"], now: datetime | None = None) -> int:
    """What a judge reads of each question of this part, from the file the reader's prompt came from. No history, so it is small enough to leave the cluster. Refuses any other
    file, and the real test's gold before its date, as every stage that reads gold does."""
    if _sha256(Path(data)) != sha256:
        raise SystemExit(f"{data} is not the benchmark's file")
    if part == "held_out" and (now or _now()) < OPENS_AT:
        raise SystemExit(f"the held-out part opens at {OPENS_AT.isoformat()}: nothing reads its gold before then")
    wanted = set(json.loads(Path(split).read_text())[part])
    rows = [{key: entry[key] for key in FIELDS} for entry in json.loads(Path(data).read_text()) if entry["question_id"] in wanted]
    Path(out).write_text(json.dumps(rows))
    Path(f"{out}.source.json").write_text(json.dumps({"part": part, "data_sha256": sha256}))
    return len(rows)


def _stand_in() -> Any:
    """No key and no call: the whole path, for the free dry run. Each verdict is a yes at the sample's mean price."""
    class StandIn:
        model, effort, max_tokens = STAND_IN, EFFORT, MAX_TOKENS

        def ask(self, prompt: str) -> types.SimpleNamespace:
            return types.SimpleNamespace(text="yes", input_tokens=1, output_tokens=1, stop_reason="end_turn", cost=lambda model: 0.00133)

    return StandIn()


def _run(arguments: argparse.Namespace) -> int:
    reader, row = arguments.row.split("/", 1)
    if arguments.stand_in:
        judge, allowed, name = _stand_in(), None, STAND_IN
    else:
        from studies.context_rot.bedrock import Converse

        if not arguments.part_split:
            raise ValueError("the paid judge needs --part-split, to hold it to the real test")
        allowed = set(json.loads(Path(arguments.part_split).read_text())["held_out"])
        judge, name = Converse(JUDGES[SECOND_JUDGE], effort=EFFORT, max_tokens=MAX_TOKENS), SECOND_JUDGE
    print(json.dumps(run_row(Path(arguments.root), Path(arguments.reference), reader, row, judge, allowed=allowed, first=arguments.first, workers=arguments.workers,
                             stop_at_usd=arguments.stop_at_usd, commit=arguments.commit, short_name=name)))
    return 0


def _status(arguments: argparse.Namespace) -> int:
    name = STAND_IN if arguments.stand_in else SECOND_JUDGE
    root = Path(arguments.root)
    for reader, row in second_judge_rows():
        if (_folder(root, reader, row) / _name(name)).exists():
            print(json.dumps(report_row(root, reader, row, short_name=name)))
    print(json.dumps({"spent_total": spent(root, name), "judged_total": judged(root, name), "of": VERDICTS}))
    return 0


def _list_rows(arguments: argparse.Namespace) -> int:
    print("\n".join(f"answers-{reader}/lme/{row}" for reader, row in second_judge_rows()))
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
    stage.add_argument("--stand-in", action="store_true")
    stage.set_defaults(action=_status)
    stage = stages.add_parser("rows")
    stage.set_defaults(action=_list_rows)
    stage = stages.add_parser("reference")
    for name in ("data", "split", "part", "out"):
        stage.add_argument(f"--{name}", required=True)
    stage.set_defaults(action=_reference)
    arguments = parser.parse_args(argv)
    return arguments.action(arguments)


if __name__ == "__main__":
    sys.exit(main())
