"""Pooled obedience of two arms on the tasks both reached, and an interval for their difference resampled over sessions.

Each session folder holds one summary per arm (FOLDER/ARM/summary.json), as studies.long_run.analyze reads them. For a pair of arms and a rule, a
session counts on the tasks both arms reached where the rule applied in both; the pooled rate is obeyed over applies, summed over those tasks and
sessions. The interval resamples whole sessions with replacement (tasks of one conversation are not independent), a fixed number of times from a
fixed seed, and gives the 2.5th and 97.5th percentiles of the pooled difference. An invalid summary is left out and counted. Standard library only.

    python -m studies.long_run.intervals FOLDER [FOLDER ...] --pairs taste_rules:mem0_search,rules_file:compaction --rules python_B,ls_one --out intervals.md
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

Row = tuple[int, int, int, int]  # obeyed and applies for arm a, then for arm b, on one session's shared tasks


def load(folders: list[Path]) -> dict[str, dict[str, dict]]:
    """Every arm's summary in each session folder, by session name."""
    runs: dict[str, dict[str, dict]] = {}
    for folder in map(Path, folders):
        runs[folder.name] = {path.parent.name: json.loads(path.read_text()) for path in sorted(folder.glob("*/summary.json"))}
    return runs


def shared_counts(a: dict, b: dict, rule: str) -> Row:
    """Obeyed and applies of each arm over the tasks both reached where the rule applied in both."""
    mine = a["scores"].get(rule, {}).get("by_task", {})
    theirs = b["scores"].get(rule, {}).get("by_task", {})
    shared = [task for task in sorted(set(mine) & set(theirs)) if mine[task]["applies"] and theirs[task]["applies"]]
    return (sum(mine[t]["obeyed"] for t in shared), sum(mine[t]["applies"] for t in shared),
            sum(theirs[t]["obeyed"] for t in shared), sum(theirs[t]["applies"] for t in shared))


def pooled_difference(rows: list[Row]) -> float:
    oa, aa, ob, ab = (sum(row[i] for row in rows) for i in range(4))
    return oa / aa - ob / ab


def resampled_interval(rows: list[Row], draws: int = 10000, seed: int = 0) -> tuple[float, float]:
    """The 2.5th and 97.5th percentiles of the pooled difference over sessions drawn with replacement."""
    pick = random.Random(seed)
    values = []
    for _ in range(draws):
        sample = [rows[pick.randrange(len(rows))] for _ in rows]
        if sum(row[1] for row in sample) and sum(row[3] for row in sample):
            values.append(pooled_difference(sample))
    values.sort()
    return values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]


def compare(runs: dict[str, dict[str, dict]], arm_a: str, arm_b: str, rule: str, draws: int = 10000, seed: int = 0) -> dict[str, Any]:
    rows: list[Row] = []
    invalid = missing = cut_short = 0
    for arms in runs.values():
        if arm_a not in arms or arm_b not in arms:
            missing += 1
            continue
        if not arms[arm_a].get("valid", True) or not arms[arm_b].get("valid", True):
            invalid += 1
            continue
        cut_short += arms[arm_a].get("ended_by") is not None or arms[arm_b].get("ended_by") is not None
        row = shared_counts(arms[arm_a], arms[arm_b], rule)
        if row[1] and row[3]:
            rows.append(row)
    result: dict[str, Any] = {"a": arm_a, "b": arm_b, "rule": rule, "sessions": len(rows), "left_out_invalid": invalid,
                              "left_out_missing": missing, "cut_short_included": cut_short}
    if rows:
        oa, aa, ob, ab = (sum(row[i] for row in rows) for i in range(4))
        low, high = resampled_interval(rows, draws, seed)
        result |= {"obeyed_a": oa, "applies_a": aa, "obeyed_b": ob, "applies_b": ab, "difference": pooled_difference(rows),
                   "low": low, "high": high, "sessions_a_higher": sum(r[0] / r[1] > r[2] / r[3] for r in rows),
                   "sessions_b_higher": sum(r[0] / r[1] < r[2] / r[3] for r in rows)}
    return result


def table(results: list[dict[str, Any]], draws: int, seed: int) -> str:
    lines = [f"Pooled obedience on shared tasks; 95% interval of the difference from {draws} draws of whole sessions (seed {seed}).", "",
             "| rule | arm a | arm b | sessions | a obeyed | b obeyed | difference (a - b) | 95% interval | sessions a higher / b higher | left out |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if not r["sessions"]:
            lines.append(f"| {r['rule']} | {r['a']} | {r['b']} | 0 | | | | | | {r['left_out_invalid']} invalid, {r['left_out_missing']} missing |")
            continue
        lines.append(f"| {r['rule']} | {r['a']} | {r['b']} | {r['sessions']} | {r['obeyed_a']}/{r['applies_a']} ({r['obeyed_a'] / r['applies_a']:.0%}) "
                     f"| {r['obeyed_b']}/{r['applies_b']} ({r['obeyed_b'] / r['applies_b']:.0%}) | {r['difference']:+.1%} | {r['low']:+.1%} to {r['high']:+.1%} "
                     f"| {r['sessions_a_higher']} / {r['sessions_b_higher']} | {r['left_out_invalid']} invalid, {r['left_out_missing']} missing; "
                     f"{r['cut_short_included']} cut short, on tasks reached |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--pairs", required=True, help="arm_a:arm_b,... compared as a minus b")
    parser.add_argument("--rules", default="python_B,ls_one,session_tag")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    runs = load(args.folders)
    results = [compare(runs, a, b, rule, args.draws, args.seed)
               for rule in args.rules.split(",") for a, b in (pair.split(":") for pair in args.pairs.split(","))]
    Path(args.out).write_text(table(results, args.draws, args.seed))
    Path(args.out).with_suffix(".json").write_text(json.dumps(results, indent=1))
    print(Path(args.out).read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
