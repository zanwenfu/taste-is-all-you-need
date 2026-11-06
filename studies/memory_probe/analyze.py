"""The rule-loss tables: every row of a run read against the summary alone it was given, the rule restated, and no rule at all.

A row is ``<label>/<arm>-<condition>-<prompt>-k<k>-b<budget>/results.jsonl``. Compliance is never read against zero, since a model the rule
never reached obeys it about half the time: every row carries the no-rule rate on the same cases, and effective retention as the benchmark
defines it. Paired tests are per case against the summary alone, by an exact sign test, with an interval from a bootstrap over histories, since
the 15 cases of a history share it. A rule the model does not obey even when told at that moment cannot be lost, so each row is also given
per rule and on the cases the restated row obeyed. Where a later statement reversed a rule, ``old_text`` says whether the old one came back.

    python -m studies.memory_probe.analyze --run runs/probe/matrix2 --inputs runs/probe/inputs-items --prompt anthropic --k 1 --out report.md
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

from studies.context_rot.compare import cluster_interval, effective_retention, paired, wilson
from studies.memory_probe.tool_loop import holds

BASE, CEILING = "summary", "restated"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_run(root: Path, prompt: str, k: int, budget: int = 4096) -> dict[str, list[dict]]:
    """Every row of this prompt, depth and budget, by label."""
    found: dict[str, list[dict]] = {}
    for path in sorted(Path(root).glob(f"*/*-{prompt}-k{k}-b{budget}/results.jsonl")):
        rows = _rows(path)
        verdicts = path.with_name("found.served.jsonl")
        if verdicts.exists():
            said = {row["id"]: row["found_by_judge"] for row in _rows(verdicts)}
            rows = [{**row, "held": said.get(row["id"])} for row in rows]
        found[path.parent.parent.name] = rows
    return found


def floors_from(paths: list[Path]) -> dict[str, bool | None]:
    """The no-rule answers from a baseline's own run. A cued case keeps the published case's summaries, but its question, and so its no-rule rate, is its own."""
    return {row["id"]: row["full_without_sc_compliant"] for path in paths for row in _rows(Path(path))}


def difference_interval(base: dict[str, dict], other: dict[str, dict], *, draws: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(difference in points, and its 95% interval), drawing whole histories with replacement. Unreadable answers count as not obeyed."""
    histories: dict[str, list[tuple[bool, bool]]] = {}
    for case, row in base.items():
        histories.setdefault(row["haystack_id"], []).append((row["compliant"] is True, other[case]["compliant"] is True))
    names = sorted(histories)
    def mean(pairs):
        return 100 * statistics.fmean(b - a for a, b in pairs)
    point = mean([pair for name in names for pair in histories[name]])
    generator = random.Random(seed)
    means = sorted(mean([pair for _ in names for pair in histories[generator.choice(names)]]) for _ in range(draws))
    return point, means[int(0.025 * draws)], means[min(draws - 1, int(0.975 * draws))]


def _obeyed(rows: list[dict]) -> int:
    return sum(row["compliant"] is True for row in rows)


def row_entry(label: str, rows: list[dict], *, floor: dict[str, bool | None], base: dict[str, dict] | None, ceiling: dict[str, dict] | None,
              old_text: dict[str, str] | None = None) -> dict[str, Any]:
    n = len(rows)
    obeyed = _obeyed(rows)
    entry: dict[str, Any] = {"row": label, "n": n, "obeyed": obeyed, "unread": sum(row["compliant"] is None for row in rows),
                             "looked": sum(bool(row["looked"]) for row in rows), "found": sum(bool(row["found"]) for row in rows),
                             "acted": sum(bool(row.get("action")) for row in rows), "in_summary": sum(bool(row.get("in_summary")) for row in rows),
                             "wilson": wilson(obeyed, n)}
    if any(row["compliant"] is not None for row in rows) and {row["id"] for row in rows} <= set(floor):
        no_rule = sum(floor[row["id"]] is True for row in rows)
        entry["no_rule"] = no_rule
        if ceiling is not None and {row["id"] for row in rows} <= set(ceiling):
            top = sum(ceiling[row["id"]]["compliant"] is True for row in rows)
            entry["restated"] = top
            entry["effective_retention"] = effective_retention(100 * obeyed / n, 100 * no_rule / n, 100 * top / n)
            could = [row for row in rows if ceiling[row["id"]]["compliant"] is True]
            entry["obeyed_where_restated_is"] = [_obeyed(could), len(could)]
    if "held" in rows[0]:
        held = [row for row in rows if row["held"] is True]
        entry["held"] = len(held)
        entry["obeyed_if_held"] = [_obeyed(held), len(held)]
        entry["obeyed_if_not"] = [obeyed - _obeyed(held), n - len(held)]
    if base is not None and label != BASE and {row["id"] for row in rows} <= set(base) and any(row["compliant"] is not None for row in rows):
        mine = {row["id"]: row for row in rows}
        shared = {case: base[case] for case in mine}
        entry["against_summary"] = paired(shared, mine, "compliant", "compliant")
        entry["difference"] = difference_interval(shared, mine)
    if any(row["compliant"] is not None for row in rows):
        entry["history_interval"] = cluster_interval(rows, "compliant")
    per_rule: dict[str, list[int]] = {}
    for row in rows:
        cell = per_rule.setdefault(str(row["sc_id"]), [0, 0])
        cell[0] += row["compliant"] is True
        cell[1] += 1
    entry["per_rule"] = per_rule
    if old_text and any("delivered_text" in row for row in rows):
        entry["old_rule_came_back"] = sum(holds(str(row.get("delivered_text", "")), old_text.get(row["id"], "\0")) for row in rows)
    return entry


def tables(run: dict[str, list[dict]], floor: dict[str, bool | None], old_text: dict[str, str] | None = None) -> list[dict[str, Any]]:
    def by_id(label):
        return {row["id"]: row for row in run[label]} if label in run else None
    base, ceiling = by_id(BASE), by_id(CEILING)
    return [row_entry(label, rows, floor=floor, base=base, ceiling=ceiling, old_text=old_text) for label, rows in sorted(run.items())]


def effective_rounds(rows: list[dict], *, smallest: int = 200) -> dict[str, Any]:
    """How many of a run's rounds folded enough to be a round: a history with one huge turn leaves many that fold almost nothing."""
    counts = [sum(tokens >= smallest for tokens in row["part_tokens"]) for row in rows if row.get("part_tokens")]
    return {"cases": len(counts), "median": statistics.median(counts) if counts else None, "smallest": min(counts, default=None),
            "largest": max(counts, default=None)}


def rounds_from(folder: Path, k: int) -> dict[str, Any]:
    """Effective rounds from the compaction records, ``<history>/results.jsonl``, which hold each case's part sizes for every depth."""
    return effective_rounds([row for path in sorted(Path(folder).glob("*/results.jsonl")) for row in _rows(path) if row.get("k") == k])


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    if isinstance(value, (list, tuple)):
        return "/".join(_cell(part) for part in value)
    return str(value)


def render(entries: list[dict[str, Any]], title: str) -> str:
    columns = ("row", "n", "obeyed", "unread", "no_rule", "restated", "effective_retention", "obeyed_where_restated_is", "held", "obeyed_if_held", "obeyed_if_not",
               "looked", "found", "in_summary", "acted", "old_rule_came_back")
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " | better | worse | p | difference [95%] |", "|" + "---|" * (len(columns) + 4)]
    for entry in entries:
        against = entry.get("against_summary", {})
        difference = entry.get("difference")
        shown = f"{difference[0]:+.1f} [{difference[1]:+.1f}, {difference[2]:+.1f}]" if difference else "-"
        lines.append("| " + " | ".join(_cell(entry.get(column)) for column in columns)
                     + f" | {_cell(against.get('better'))} | {_cell(against.get('worse'))} | {_cell(against.get('sign_test_p'))} | {shown} |")
    return "\n".join(lines) + "\n"


def render_rules(entries: list[dict[str, Any]]) -> str:
    """Each rule of every row that has answers, so that a rule the model leaves even when restated is seen as such."""
    shown = [entry for entry in entries if entry["n"] > entry["unread"]]
    rules = sorted({rule for entry in shown for rule in entry["per_rule"]}, key=lambda rule: (len(rule), rule))
    lines = ["| row | " + " | ".join(rules) + " |", "|" + "---|" * (len(rules) + 1)]
    lines += ["| " + " | ".join([entry["row"], *("/".join(str(n) for n in entry["per_rule"].get(rule, [0, 0])) for rule in rules)]) + " |" for entry in shown]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("run", "inputs", "prompt", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--base-cases", help="the unchanged cases, to tell whether a reversed rule's old text came back")
    parser.add_argument("--floors", nargs="+", help="a baseline's results, for cases whose no-rule rate is not in the inputs")
    parser.add_argument("--rounds-from", help="the compaction records of this prompt, to report how many rounds folded enough to be a round")
    arguments = parser.parse_args(argv)
    inputs = Path(arguments.inputs)
    floor = {row["id"]: row["floor_compliant"] for row in _rows(inputs / "summaries.jsonl")
             if row["prompt"] == arguments.prompt and row["k"] == arguments.k}
    if arguments.floors:
        floor = floors_from([Path(path) for path in arguments.floors])
    old_text = None
    if arguments.base_cases:
        old = {row["id"]: row["sc_text"] for row in _rows(Path(arguments.base_cases))}
        old_text = {f"{case}:{suffix}": text for case, text in old.items() for suffix in ("revised", "withdrawn")}
    entries = tables(load_run(Path(arguments.run), arguments.prompt, arguments.k, arguments.budget), floor, old_text)
    out = Path(arguments.out)
    text = render(entries, f"{Path(arguments.run).name}, {arguments.prompt}, {arguments.k} round(s), budget {arguments.budget}") + "\n" + render_rules(entries)
    if arguments.rounds_from:
        text += f"\nRounds of at least 200 tokens: {json.dumps(rounds_from(Path(arguments.rounds_from), arguments.k))}\n"
    out.write_text(text)
    out.with_suffix(".json").write_text(json.dumps(entries, indent=1))
    print(f"{len(entries)} rows at {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
