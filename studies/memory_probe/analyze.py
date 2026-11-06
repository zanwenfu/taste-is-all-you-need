"""The rule-loss tables: every row of a run read against the summary alone it was given, the rule restated, and no rule at all.

A row is ``<label>/<arm>-<condition>-<prompt>-k<k>-b<budget>/results.jsonl``. Compliance is never read against zero, since a model the rule
never reached obeys it about half the time: every row carries the no-rule rate on the same cases, and effective retention as the benchmark
defines it. Paired tests are per case against the summary alone, by an exact sign test, with an interval from a bootstrap over histories, since
the 15 cases of a history share it. A rule the model does not obey even when told at that moment cannot be lost, so each row is also given
per rule and on the cases the restated row obeyed. Where a later statement reversed a rule, ``old_text`` says whether the old one came back. Rates leave out answers that
could not be read, as the benchmark's summary does, and say how many. Where a rule was taken back, obeying it is the failure, so such a row has no effective retention, and it is
read against the no-rule rate. The rows of a table must have run on one version of the code.

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


def one_code(root: Path, prompt: str, k: int, budget: int = 4096) -> dict[str, Any]:
    """The code the rows of a table ran on. Rows from two versions of it are not one table."""
    seen: dict[tuple, list[str]] = {}
    for path in sorted(Path(root).glob(f"*/*-{prompt}-k{k}-b{budget}/results.settings.json")):
        said = json.loads(path.read_text())
        seen.setdefault((said.get("commit"), said.get("code_digest")), []).append(path.parent.parent.name)
    if len(seen) > 1:
        raise ValueError(f"these rows ran on two versions of the code: {seen}")
    return next(({"code_digest": digest, "commit": commit} for commit, digest in seen), {})


def load_stored(root: Path) -> dict[str, dict[str, dict]]:
    """What the judge said each notes file holds, by the wording of its note-taker."""
    found = {}
    for style in ("told", "neutral"):
        path = Path(root) / f"stored-{style}.served.jsonl"
        if path.exists():
            found[style] = {row["id"]: row for row in _rows(path)}
    return found


def floors_from(paths: list[Path]) -> dict[str, bool | None]:
    """The no-rule answers from a baseline's own run. A cued case keeps the published case's summaries, but its question, and so its no-rule rate, is its own."""
    return {row["id"]: row["full_without_sc_compliant"] for path in paths for row in _rows(Path(path))}


def difference_interval(base: dict[str, dict], other: dict[str, dict], *, draws: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(difference in points, and its 95% interval), drawing whole histories with replacement, over the pairs where both answers could be read."""
    histories: dict[str, list[tuple[bool, bool]]] = {}
    for case, row in base.items():
        if row["compliant"] is not None and other[case]["compliant"] is not None:
            histories.setdefault(row["haystack_id"], []).append((row["compliant"], other[case]["compliant"]))
    names = sorted(histories)
    def mean(pairs):
        return 100 * statistics.fmean(b - a for a, b in pairs)
    point = mean([pair for name in names for pair in histories[name]])
    generator = random.Random(seed)
    means = sorted(mean([pair for _ in names for pair in histories[generator.choice(names)]]) for _ in range(draws))
    return point, means[int(0.025 * draws)], means[min(draws - 1, int(0.975 * draws))]


def _obeyed(rows: list[dict]) -> int:
    return sum(row["compliant"] is True for row in rows)


_TURN_ARMS = ("a2a", "a3", "a4", "a4q", "a5", "a5q", "a7", "a7q")
_LAST_NOTES = ("a2c", "a2cq", "a2cw")
_EVERY_NOTES = ("a2d", "a2dq", "a2dw")


def _stored(label: str, rows: list[dict], stored: dict[str, dict[str, dict]] | None) -> int | None:
    """The stored step: every turn is in a turn store, so how many cases have the rule's whole text in one; a rule list or a notes file holds what the judge read in it."""
    arm = rows[0].get("arm")
    if arm in _TURN_ARMS:
        return sum(bool(row.get("rule_turns")) for row in rows)
    if arm in ("a6", "a6h") and "held" in rows[0]:
        return sum(row["held"] is True for row in rows)
    verdicts = (stored or {}).get(label.rsplit("-", 1)[-1])
    if verdicts and arm in (*_LAST_NOTES, *_EVERY_NOTES):
        return sum(verdicts.get(row["id"], {}).get("newest" if arm in _LAST_NOTES else "any") is True for row in rows)
    return None


def _stored_unclear(label: str, rows: list[dict], stored: dict[str, dict[str, dict]] | None) -> int:
    """The notes files the judge could not read either way: not counted as holding the rule, and not lost among those that do not."""
    arm = rows[0].get("arm")
    verdicts = (stored or {}).get(label.rsplit("-", 1)[-1])
    if not verdicts or arm not in (*_LAST_NOTES, *_EVERY_NOTES):
        return 0
    return sum(row["id"] in verdicts and verdicts[row["id"]].get("newest" if arm in _LAST_NOTES else "any") is None for row in rows)


def row_entry(label: str, rows: list[dict], *, floor: dict[str, bool | None], base: dict[str, dict] | None, ceiling: dict[str, dict] | None,
              old_text: dict[str, str] | None = None, stored: dict[str, dict[str, dict]] | None = None) -> dict[str, Any]:
    n = len(rows)
    obeyed = _obeyed(rows)
    readable = n - sum(row["compliant"] is None for row in rows)
    withdrawn = all(str(row["id"]).endswith(":withdrawn") for row in rows)
    entry: dict[str, Any] = {"row": label, "n": n, "obeyed": obeyed, "unread": n - readable,
                             "looked": sum(bool(row["looked"]) for row in rows), "found": sum(bool(row["found"]) for row in rows),
                             "acted": sum(bool(row.get("action")) for row in rows), "in_summary": sum(bool(row.get("in_summary")) for row in rows),
                             "wilson": wilson(obeyed, readable)}
    in_store = _stored(label, rows, stored)
    if in_store is not None:
        entry["stored"] = in_store
    if _stored_unclear(label, rows, stored):
        entry["stored_unclear"] = _stored_unclear(label, rows, stored)
    if withdrawn:
        entry["still_obeying"] = obeyed
    if readable and {row["id"] for row in rows} <= set(floor):
        floors = [floor[row["id"]] for row in rows if floor[row["id"]] is not None]
        no_rule = sum(value is True for value in floors)
        entry["no_rule"] = no_rule
        if ceiling is not None and {row["id"] for row in rows} <= set(ceiling):
            tops = [ceiling[row["id"]]["compliant"] for row in rows if ceiling[row["id"]]["compliant"] is not None]
            top = sum(value is True for value in tops)
            entry["restated"] = top
            if floors and tops and not withdrawn:
                entry["effective_retention"] = effective_retention(100 * obeyed / readable, 100 * no_rule / len(floors), 100 * top / len(tops))
            could = [row for row in rows if ceiling[row["id"]]["compliant"] is True and row["compliant"] is not None]
            entry["obeyed_where_restated_is"] = [_obeyed(could), len(could)]
    if "held" in rows[0]:
        held = [row for row in rows if row["held"] is True]
        without = [row for row in rows if row["held"] is False]
        entry["held"] = len(held)
        entry["held_unclear"] = sum(row["held"] is None for row in rows)
        entry["obeyed_if_held"] = [_obeyed(held), sum(row["compliant"] is not None for row in held)]
        entry["obeyed_if_not"] = [_obeyed(without), sum(row["compliant"] is not None for row in without)]
    if base is not None and label != BASE and {row["id"] for row in rows} <= set(base) and readable:
        mine = {row["id"]: row for row in rows}
        shared = {case: base[case] for case in mine}
        pair = paired(shared, mine, "compliant", "compliant")
        if withdrawn:
            pair["more_often"], pair["less_often"] = pair.pop("better"), pair.pop("worse")
        entry["against_summary"] = pair
        entry["difference"] = difference_interval(shared, mine)
    if readable:
        entry["history_interval"] = cluster_interval(rows, "compliant")
    per_rule: dict[str, list[int]] = {}
    for row in rows:
        cell = per_rule.setdefault(str(row["sc_id"]), [0, 0])
        cell[0] += row["compliant"] is True
        cell[1] += row["compliant"] is not None
    entry["per_rule"] = per_rule
    if old_text and any("delivered_text" in row for row in rows):
        entry["old_rule_came_back"] = sum(holds(str(row.get("delivered_text", "")), old_text.get(row["id"], "\0")) for row in rows)
    return entry


def tables(run: dict[str, list[dict]], floor: dict[str, bool | None], old_text: dict[str, str] | None = None,
           stored: dict[str, dict[str, dict]] | None = None) -> list[dict[str, Any]]:
    def by_id(label):
        return {row["id"]: row for row in run[label]} if label in run else None
    base, ceiling = by_id(BASE), by_id(CEILING)
    return [row_entry(label, rows, floor=floor, base=base, ceiling=ceiling, old_text=old_text, stored=stored) for label, rows in sorted(run.items())]


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
    columns = ("row", "n", "stored", "stored_unclear", "looked", "found", "held", "held_unclear", "obeyed", "unread", "no_rule", "restated", "effective_retention", "obeyed_where_restated_is",
               "obeyed_if_held", "obeyed_if_not", "in_summary", "acted", "old_rule_came_back")
    withdrawn = any("still_obeying" in entry for entry in entries)
    more, less = ("more often", "less often") if withdrawn else ("better", "worse")
    lines = [f"# {title}", "", *(("A rule was taken back here: obeyed means still obeying it, read against the no-rule rate, and lower is closer.", "") if withdrawn else ()),
             "| " + " | ".join(columns) + f" | {more} | {less} | p | difference [95%] |", "|" + "---|" * (len(columns) + 4)]
    for entry in entries:
        against = dict(entry.get("against_summary", {}))
        against["better"], against["worse"] = against.get("more_often", against.get("better")), against.get("less_often", against.get("worse"))
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
    parser.add_argument("--mixed", action="store_true", help="allow rows from two versions of the code, and say so; never for a table in the paper")
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
    run = Path(arguments.run)
    try:
        code = one_code(run, arguments.prompt, arguments.k, arguments.budget)
    except ValueError as error:
        if not arguments.mixed:
            raise
        code = {"mixed": str(error)}
    entries = tables(load_run(run, arguments.prompt, arguments.k, arguments.budget), floor, old_text, load_stored(run))
    out = Path(arguments.out)
    text = render(entries, f"{run.name}, {arguments.prompt}, {arguments.k} round(s), budget {arguments.budget}") + f"\nCode: {json.dumps(code)}\n\n" + render_rules(entries)
    if arguments.rounds_from:
        text += f"\nRounds of at least 200 tokens: {json.dumps(rounds_from(Path(arguments.rounds_from), arguments.k))}\n"
    out.write_text(text)
    out.with_suffix(".json").write_text(json.dumps(entries, indent=1))
    print(f"{len(entries)} rows at {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
