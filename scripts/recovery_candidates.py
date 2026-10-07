"""The recovery study's calibration candidates, chosen by a written rule.

    python scripts/recovery_candidates.py --pro /root/recovery-bench/swebench-pro-os/v2 \\
        --tb3 /root/recovery-bench/terminal-bench --pro-seats 242 \\
        --salt taste-recovery-calibration-2026-10-07 --out candidates.json \\
        [--drop task="its reference solution fails here"]

The study's states are files: a branch brings back the agent's container's
files, never a running process or another container. So a task qualifies only
if its hidden tests grade files the agent left in its own container. A task
the rule admits is still dropped, with its reason recorded, when a check run
here shows it cannot be graded fairly (its reference solution fails).

SWE-Bench Pro V2: every verifier applies the task's tests to the repository in
the agent's container and runs them itself, so all 642 tasks qualify. The
HARD-51 subset is left out (it was chosen to separate frontier models). The
rest are stratified by language (the most common extension among the files
the reference patch changes) and by the size of the reference patch (changed
lines, terciles within the language). Each stratum gets seats in proportion
to its size, rounded by the largest remainder (ties by name); inside a stratum
tasks are taken in the order of SHA-256 of the salt and the task name.

Terminal-Bench 3.0: a task qualifies when its verifier runs in a separate
container (Harbor stops the agent's container first and hands the verifier
only the declared artifact files), its environment has no service but the
agent's own container (a sidecar's state is neither in the agent's files nor
restorable with them, and several sidecar tasks collect their artifacts from
the running services), and it needs no GPU (the measurement VM has none).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.benchmarks.task_split import dataset_digest

SCHEMA = "taste.recovery/CalibrationCandidates/1"
LANGUAGES = {".py": "python", ".go": "go", ".js": "js-ts", ".jsx": "js-ts", ".ts": "js-ts", ".tsx": "js-ts",
             ".mjs": "js-ts", ".cjs": "js-ts"}
SIZES = ("small", "medium", "large")


def _order(salt, name):
    return hashlib.sha256(f"{salt}\0{name}".encode()).hexdigest()


def patch_facts(diff):
    """Language and changed lines of a unified diff."""
    files = re.findall(r"^diff --git a/(\S+)", diff, re.M)
    kinds = Counter(LANGUAGES[suffix] for path in files if (suffix := Path(path).suffix.lower()) in LANGUAGES)
    # Files in no listed language (changelogs, templates, styles) never decide it.
    language = min(kinds, key=lambda kind: (-kinds[kind], kind)) if kinds else "other"
    lines = sum(1 for line in diff.splitlines()
                if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
    return language, lines


def pro_tasks(root):
    """SWE-Bench Pro V2 tasks with their stratum facts and HARD-51 membership."""
    root = Path(root)
    hard = set((root / "hard51_ids.txt").read_text().split()) if (root / "hard51_ids.txt").is_file() else set()
    tasks = []
    for task in sorted(path for path in (root / "tasks").iterdir() if (path / "task.toml").is_file()):
        language, lines = patch_facts((task / "solution" / "gold_patch.diff").read_text(errors="replace"))
        tasks.append({"task": task.name, "language": language, "lines": lines,
                      "hard51": task.name in hard or task.name.removeprefix("instance_") in hard})
    return tasks


def size_terciles(tasks):
    """Each task's patch-size tercile within its language, smallest first (ties by name)."""
    by_language = defaultdict(list)
    for task in tasks:
        by_language[task["language"]].append(task)
    sizes = {}
    for group in by_language.values():
        group.sort(key=lambda task: (task["lines"], task["task"]))
        for index, task in enumerate(group):
            sizes[task["task"]] = SIZES[index * len(SIZES) // len(group)]
    return sizes


def _seats(counts, size):
    total = sum(counts.values())
    shares = {key: size * count / total for key, count in counts.items()}
    seats = {key: math.floor(share) for key, share in shares.items()}
    left = size - sum(seats.values())
    for key in sorted(shares, key=lambda name: (-(shares[name] - seats[name]), name))[:left]:
        seats[key] += 1
    return seats


def choose_pro(tasks, seats, salt):
    """The stratified sample of non-HARD-51 tasks, and its strata."""
    eligible = [task for task in tasks if not task["hard51"]]
    if not 0 < seats <= len(eligible):
        raise ValueError(f"seats must be between 1 and {len(eligible)}")
    sizes = size_terciles(eligible)
    strata = defaultdict(list)
    for task in eligible:
        strata[f"{task['language']}/{sizes[task['task']]}"].append(task["task"])
    quotas = _seats({key: len(names) for key, names in strata.items()}, seats)
    chosen = {key: sorted(sorted(names, key=lambda name: _order(salt, name))[:quotas[key]])
              for key, names in sorted(strata.items())}
    return chosen, {key: {"tasks": len(strata[key]), "seats": quotas[key]} for key in sorted(strata)}


def compose_services(path):
    """Service names of a Compose file (top-level `services:` keys)."""
    services, inside = [], False
    for line in path.read_text().splitlines():
        if re.match(r"^services:\s*$", line):
            inside = True
        elif inside and re.match(r"^\S", line):
            inside = False
        elif inside and (match := re.match(r"^  ([A-Za-z0-9_.-]+):\s*$", line)):
            services.append(match.group(1))
    return services


def tb3_verdict(task):
    """Whether a Terminal-Bench task's hidden tests grade only the agent's files; if not, why."""
    config = tomllib.loads((task / "task.toml").read_text())
    verifier = config.get("verifier", {})
    if verifier.get("environment_mode") != "separate":
        return False, "verifier shares the agent's container (read its tests)"
    compose = task / "environment" / "docker-compose.yaml"
    sidecars = [name for name in compose_services(compose) if name != "main"] if compose.is_file() else []
    if sidecars:
        return False, "sidecar services: " + ", ".join(sidecars)
    gpus = max(int(config.get("environment", {}).get("gpus") or 0),
               int((verifier.get("environment") or {}).get("gpus") or 0))
    if gpus:
        return False, "needs a GPU"
    return True, ""


def candidates(pro_root, tb3_root, pro_seats, salt, drop=None):
    """The record: the rule, and per benchmark the tasks and a digest of the dataset they came from.

    `drop` maps a task the rule would admit to the reason it is left out.
    """
    drop = dict(drop or {})
    record = {"schema": SCHEMA, "salt": salt, "rule": __doc__.split("\n\n", 2)[2].strip()}
    if pro_root:
        chosen, strata = choose_pro(pro_tasks(pro_root), pro_seats, salt)
        names = sorted(name for group in chosen.values() for name in group)
        record["swebench_pro_v2"] = {
            "dataset": "github.com/scaleapi/SWE-bench_Pro-os v2/tasks",
            "dataset_digest": dataset_digest(Path(pro_root) / "tasks"), "strata": strata,
            "tasks": [name for name in names if name not in drop],
            "excluded": {name: drop[name] for name in names if name in drop}}
    if tb3_root:
        verdicts = {task.name: tb3_verdict(task) for task in sorted(Path(tb3_root).iterdir())
                    if (task / "task.toml").is_file()}
        verdicts.update({name: (False, why) for name, why in drop.items() if verdicts.get(name, (False,))[0]})
        record["terminal_bench_3"] = {
            "dataset": "terminal-bench/terminal-bench@3.0.0", "dataset_digest": dataset_digest(tb3_root),
            "tasks": [name for name, (ok, _) in verdicts.items() if ok],
            "excluded": {name: why for name, (ok, why) in verdicts.items() if not ok}}
    unknown = set(drop) - {name for part in ("swebench_pro_v2", "terminal_bench_3") if part in record
                           for name in record[part]["excluded"]}
    if unknown:
        raise ValueError("--drop names tasks that are not candidates: " + ", ".join(sorted(unknown)))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pro", type=Path, help="SWE-Bench Pro V2's v2/ directory")
    parser.add_argument("--tb3", type=Path, help="Terminal-Bench 3.0's downloaded tasks")
    parser.add_argument("--pro-seats", type=int, default=242)
    parser.add_argument("--salt", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--drop", action="append", default=[], metavar="TASK=REASON",
                        help="leave out a task the rule admits, for the reason given")
    arguments = parser.parse_args(argv)
    drop = dict(item.split("=", 1) for item in arguments.drop)
    record = candidates(arguments.pro, arguments.tb3, arguments.pro_seats, arguments.salt, drop)
    arguments.out.write_text(json.dumps(record, indent=1) + "\n")
    for key in ("swebench_pro_v2", "terminal_bench_3"):
        if key in record:
            print(f"{key}: {len(record[key]['tasks'])} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
