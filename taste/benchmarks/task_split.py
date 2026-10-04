"""Split a benchmark's tasks into tuning and test sets by a fixed, written rule.

Settings may be tuned only on tasks whose results are never reported, so the
split is decided before any run and committed with the rule that made it:

- Difficulty is the stratum. Each difficulty gets seats in proportion to its
  share of the dataset, rounded by the largest-remainder method (ties by name).
- Inside a stratum, tasks are taken in the order of SHA-256 of the salt and the
  task name, so the choice is fixed by the salt and owes nothing to judgement.
- A category takes at most its proportional share of the seats, rounded up,
  while other tasks of the stratum remain; a stratum that runs out of those
  fills its quota in the same order regardless of category.
- Strata are filled in name order.

The record pins the dataset by a digest of every file in it. Test tasks are
selected only for a registered study whose registration names this exact split
record, by the SHA-256 of its bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "taste.benchmarks/TaskSplit/1"
PARTS = ("tuning", "test")


class SplitRefused(RuntimeError):
    """A selection the split's rule does not allow."""


@dataclass(frozen=True)
class TaskInfo:
    name: str
    category: str
    difficulty: str


@dataclass(frozen=True)
class Split:
    tuning: tuple[str, ...]
    test: tuple[str, ...]


def _order(salt, name):
    return hashlib.sha256(f"{salt}\0{name}".encode()).hexdigest()


def _seats(counts, size, total):
    shares = {key: size * count / total for key, count in counts.items()}
    seats = {key: math.floor(share) for key, share in shares.items()}
    left = size - sum(seats.values())
    for key in sorted(shares, key=lambda name: (-(shares[name] - seats[name]), name))[:left]:
        seats[key] += 1
    return seats


def split_tasks(tasks, *, tuning_size, salt):
    """The tuning and test task names, each sorted by name."""
    tasks = list(tasks)
    names = [item.name for item in tasks]
    if len(set(names)) != len(names):
        raise ValueError("task names must be unique")
    if type(tuning_size) is not int or not 0 < tuning_size < len(tasks):
        raise ValueError("the tuning set must be nonempty and smaller than the dataset")
    if not isinstance(salt, str) or not salt:
        raise ValueError("a salt is required")
    total = len(tasks)
    quotas = _seats(Counter(item.difficulty for item in tasks), tuning_size, total)
    caps = {category: math.ceil(tuning_size * count / total)
            for category, count in Counter(item.category for item in tasks).items()}
    taken, used = [], Counter()
    for difficulty in sorted(quotas):
        stratum = sorted((item for item in tasks if item.difficulty == difficulty),
                         key=lambda item: _order(salt, item.name))
        chosen = []
        for item in stratum:
            if len(chosen) == quotas[difficulty]:
                break
            if used[item.category] < caps[item.category]:
                chosen.append(item)
                used[item.category] += 1
        for item in stratum:
            if len(chosen) == quotas[difficulty]:
                break
            if item not in chosen:
                chosen.append(item)
                used[item.category] += 1
        taken.extend(chosen)
    tuning = {item.name for item in taken}
    return Split(tuple(sorted(tuning)), tuple(sorted(set(names) - tuning)))


def load_tasks(dataset_dir):
    """Every task directory's name, category and difficulty, sorted by name."""
    tasks = []
    for path in sorted(Path(dataset_dir).iterdir()):
        if not (path / "task.toml").is_file():
            continue
        with open(path / "task.toml", "rb") as handle:
            metadata = tomllib.load(handle).get("metadata", {})
        for key in ("category", "difficulty"):
            if not isinstance(metadata.get(key), str) or not metadata[key]:
                raise ValueError(f"task {path.name} has no {key}")
        tasks.append(TaskInfo(path.name, metadata["category"], metadata["difficulty"]))
    if not tasks:
        raise ValueError("no task directories in the dataset")
    return tuple(tasks)


def dataset_digest(dataset_dir):
    """SHA-256 over every file's path relative to the dataset and its content."""
    root = Path(dataset_dir)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big") + relative)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + digest.hexdigest()


def make_split_record(dataset_dir, *, dataset, tuning_size, salt):
    split = split_tasks(load_tasks(dataset_dir), tuning_size=tuning_size, salt=salt)
    return {"schema": SCHEMA, "dataset": dataset, "dataset_digest": dataset_digest(dataset_dir),
            "rule": {"tuning_size": tuning_size, "salt": salt, "stratum": "difficulty",
                     "order": "sha256(salt NUL name)",
                     "category_cap": "ceil(tuning_size * category_tasks / tasks)"},
            "tuning": list(split.tuning), "test": list(split.test)}


def select(record_path, dataset_dir, part, into, *, study=None, registrations=None):
    """Copy one part's task directories into ``into``, for a benchmark runner.

    Test tasks need a study registered before the run whose registration names
    the SHA-256 of this split record's bytes.
    """
    if part not in PARTS:
        raise SplitRefused(f"part must be one of {', '.join(PARTS)}")
    raw = Path(record_path).read_bytes()
    record = json.loads(raw)
    if record.get("schema") != SCHEMA:
        raise SplitRefused("not a task split record")
    if dataset_digest(dataset_dir) != record["dataset_digest"]:
        raise SplitRefused("the dataset differs from the one the split was made from")
    if part == "test":
        if study is None:
            raise SplitRefused("test tasks run only for a registered study")
        registration = Path(registrations or "data/studies") / f"{study}.json"
        if not registration.is_file():
            raise SplitRefused(f"study {study!r} is not registered")
        if json.loads(registration.read_text()).get("split_sha256") != hashlib.sha256(raw).hexdigest():
            raise SplitRefused(f"study {study!r} is registered for another split")
    target = Path(into)
    if target.exists():
        raise SplitRefused(f"{target} exists; selections are made into a new directory")
    target.mkdir(parents=True)
    names = list(record[part])
    for name in names:
        shutil.copytree(Path(dataset_dir) / name, target / name, symlinks=True)
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make", help="write a split record for a dataset directory")
    make.add_argument("dataset_dir", type=Path)
    make.add_argument("--dataset", required=True, help="the dataset's registry name")
    make.add_argument("--size", type=int, required=True, help="number of tuning tasks")
    make.add_argument("--salt", required=True)
    make.add_argument("--out", type=Path, required=True)
    pick = commands.add_parser("select", help="copy one part's tasks into a new directory")
    pick.add_argument("record", type=Path)
    pick.add_argument("dataset_dir", type=Path)
    pick.add_argument("part", choices=PARTS)
    pick.add_argument("into", type=Path)
    pick.add_argument("--study")
    pick.add_argument("--registrations", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "make":
        if arguments.out.exists():
            print(f"{arguments.out} exists; a split is never rewritten", file=sys.stderr)
            return 1
        record = make_split_record(arguments.dataset_dir, dataset=arguments.dataset,
                                   tuning_size=arguments.size, salt=arguments.salt)
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        print(f"{len(record['tuning'])} tuning and {len(record['test'])} test tasks -> {arguments.out}")
        return 0
    try:
        names = select(arguments.record, arguments.dataset_dir, arguments.part, arguments.into,
                       study=arguments.study, registrations=arguments.registrations)
    except SplitRefused as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 2
    print(f"{len(names)} {arguments.part} tasks -> {arguments.into}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
