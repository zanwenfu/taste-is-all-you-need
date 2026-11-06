"""The long run's tables: each arm against compaction alone, across seeds, task by task.

Arms can end early, so a rule's obedience compares only tasks both arms reached. Tasks of one conversation are not independent, so the test is over sessions, each
compared on the tasks both arms reached in it; the count over tasks is shown as a description. The tag is reported as ran and as known, since a reply with no command can
hold it. A session that is invalid (a planted text sent again, a ledger refused, nothing planted) is left out of every total and counted, and so is how many ended early.

    python -m studies.long_run.analyze runs/seed2-1 runs/seed3-1 --out report.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

BASE = "compaction"
SPANS = ((0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 10**9))


def load(folders: list[Path]) -> dict[str, dict[str, dict]]:
    """{seed folder: {arm: summary}}"""
    return {folder.name: {path.parent.name: json.loads(path.read_text()) for path in sorted(folder.glob("*/summary.json"))} for folder in folders}


def sign_test(better: int, worse: int) -> float:
    n = better + worse
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(better, worse) + 1)) / 2**n) if n else 1.0


def _rate(cell: dict[str, int]) -> float:
    return cell["obeyed"] / cell["applies"]


def compare_rule(runs: dict[str, dict[str, dict]], arm: str, rule: str) -> dict[str, Any]:
    """Pooled obedience, and task by task against compaction alone on tasks both reached where the rule applied in both."""
    pooled = {"obeyed": 0, "applies": 0, "base_obeyed": 0, "base_applies": 0, "tasks": 0, "better": 0, "worse": 0, "same": 0,
              "sessions": 0, "sessions_better": 0, "sessions_worse": 0}
    for arms in runs.values():
        if arm not in arms or BASE not in arms or not arms[arm].get("valid", True) or not arms[BASE].get("valid", True):
            continue
        mine, theirs = arms[arm]["scores"].get(rule, {}).get("by_task", {}), arms[BASE]["scores"].get(rule, {}).get("by_task", {})
        shared = sorted(set(mine) & set(theirs))
        if shared:
            rates = [sum(side[task]["obeyed"] for task in shared) / sum(side[task]["applies"] for task in shared) for side in (mine, theirs)]
            pooled["sessions"] += 1
            pooled["sessions_better"] += rates[0] > rates[1]
            pooled["sessions_worse"] += rates[0] < rates[1]
        for task in shared:
            a, b = mine[task], theirs[task]
            pooled["obeyed"] += a["obeyed"]
            pooled["applies"] += a["applies"]
            pooled["base_obeyed"] += b["obeyed"]
            pooled["base_applies"] += b["applies"]
            pooled["tasks"] += 1
            pooled["better" if _rate(a) > _rate(b) else "worse" if _rate(a) < _rate(b) else "same"] += 1
    pooled["p"] = sign_test(pooled["better"], pooled["worse"])
    pooled["sessions_p"] = sign_test(pooled["sessions_better"], pooled["sessions_worse"])
    return pooled


def arm_table(runs: dict[str, dict[str, dict]]) -> list[dict[str, Any]]:
    arms = sorted({arm for seed in runs.values() for arm in seed})
    rules = sorted({rule for seed in runs.values() for summary in seed.values() for rule in summary["scores"]})
    table = []
    for arm in arms:
        every = [seed[arm] for seed in runs.values() if arm in seed]
        summaries = [s for s in every if s.get("valid", True)]
        entry: dict[str, Any] = {
            "arm": arm, "sessions": len(every), "invalid": len(every) - len(summaries), "ended_early": sum(bool(s.get("ended_by")) for s in every),
            "calls": sum(s["calls"] for s in summaries), "compactions": sum(s["compactions"] for s in summaries),
            "no_command": sum(s.get("no_command", 0) for s in summaries),
            "submitted": sum(o["exit_status"] == "Submitted" for s in summaries for o in s["outcomes"]),
            "tasks": sum(len(s["outcomes"]) for s in summaries),
            "machines_failed": sum(o["exit_status"] == "MachineFailed" for s in summaries for o in s["outcomes"]),
            "looks": max((sum(s["scores"][rule]["looks"] for s in summaries if rule in s["scores"]) for rule in rules), default=0),
            "rules": {}}
        for rule in rules:
            scores = [s["scores"][rule] for s in summaries if rule in s["scores"]]
            cell: dict[str, Any] = {"obeyed": sum(x["obeyed"] for x in scores), "applies": sum(x["applies"] for x in scores),
                                    "finds": sum(x["finds"] for x in scores)}
            if any("knows" in x for x in scores):
                cell["knows"], cell["knows_of"] = sum(x.get("knows", 0) for x in scores), sum(x.get("knows_of", 0) for x in scores)
            if arm != BASE:
                cell["against_compaction"] = compare_rule(runs, arm, rule)
            entry["rules"][rule] = cell
        table.append(entry)
    return table


def _span(compactions: int) -> str:
    low, high = next(span for span in SPANS if span[0] <= compactions <= span[1])
    return str(low) if low == high else f"{low}-{high}" if high < 10**9 else f"{low}+"


def compaction_table(runs: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict[str, list[int]]]]:
    """{rule: {arm: {compactions so far: [obeyed, applies]}}}, over the valid runs: the question is what is lost as compactions add up."""
    table: dict[str, dict[str, dict[str, list[int]]]] = {}
    for arms in runs.values():
        for arm, summary in arms.items():
            if not summary.get("valid", True):
                continue
            for rule, score in summary["scores"].items():
                for compactions, (obeyed, applies) in score.get("by_compactions", {}).items():
                    cell = table.setdefault(rule, {}).setdefault(arm, {}).setdefault(_span(int(compactions)), [0, 0])
                    cell[0] += obeyed
                    cell[1] += applies
    return table


def render_compactions(table: dict[str, dict[str, dict[str, list[int]]]]) -> str:
    spans = [_span(low) for low, _ in SPANS]
    lines = []
    for rule in sorted(table):
        lines += ["", f"{rule}, obeyed by compactions so far", "", "| arm | " + " | ".join(spans) + " |", "|" + "---|" * (len(spans) + 1)]
        for arm in sorted(table[rule]):
            cells = [("/".join(str(n) for n in table[rule][arm][span]) if span in table[rule][arm] else "-") for span in spans]
            lines.append("| " + " | ".join([arm, *cells]) + " |")
    return "\n".join(lines) + "\n"


def render(table: list[dict[str, Any]]) -> str:
    rules = sorted({rule for entry in table for rule in entry["rules"]})
    head = ["arm", "sessions", "invalid", "ended early", "tasks", "submitted", "machines failed", "calls", "compactions", "no command", "looks"]
    lines = ["| " + " | ".join(head + rules) + " |", "|" + "---|" * (len(head) + len(rules))]
    for entry in table:
        cells = [entry["arm"], entry["sessions"], entry["invalid"], entry["ended_early"], entry["tasks"], entry["submitted"], entry["machines_failed"], entry["calls"], entry["compactions"],
                 entry["no_command"], entry["looks"]]
        for rule in rules:
            cell = entry["rules"].get(rule, {})
            text = f"{cell.get('obeyed', 0)}/{cell.get('applies', 0)}"
            if "knows" in cell:
                text += f" knows {cell['knows']}/{cell['knows_of']}"
            pair = cell.get("against_compaction")
            if pair:
                text += f" (paired over {pair['sessions']} sessions, +{pair['sessions_better']} -{pair['sessions_worse']}, p {pair['sessions_p']:.3f}; over {pair['tasks']} tasks, +{pair['better']} -{pair['worse']})"
            cells.append(text)
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)
    runs = load([Path(folder) for folder in arguments.folders])
    table = arm_table(runs)
    out = Path(arguments.out)
    out.write_text(render(table) + render_compactions(compaction_table(runs)))
    out.with_suffix(".json").write_text(json.dumps(table, indent=1))
    print(f"{len(table)} arms at {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
