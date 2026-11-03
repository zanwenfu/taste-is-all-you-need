"""LongMemEval's three aggregates, plus what the official script leaves out.

The official ``print_qa_metrics.py`` (commit 9e0b455) prints, over six question types:
per-type accuracy, the task-averaged accuracy (mean of the six type means), the overall
accuracy (mean over every question) and the abstention accuracy (over ``_abs`` ids, which
also count inside their own type). It asserts that the judge was ``gpt-4o-2024-08-06``,
so it refuses any other judge's file by design; this module reproduces its arithmetic
without that assertion and never relabels a judge.

Added: how many questions have no verdict file entry (``missing``), how many verdicts
were unparseable, every aggregate both with unparseable verdicts counted as wrong and with
them left out, and agreement between two judges on the same hypotheses.

    python -m studies.context_rot.lme_metrics <eval-results> <reference.json> [--other <eval-results>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TYPES = ("single-session-user", "single-session-preference", "single-session-assistant", "multi-session",
         "temporal-reasoning", "knowledge-update")


def _mean(values):
    return sum(values) / len(values) if values else None


def _aggregates(rows, refs, unparseable_as):
    by_type = {t: [] for t in TYPES}
    abstention = []
    for row in rows:
        label = row["autoeval_label"]["label"]
        if label is None:
            if unparseable_as is None:
                continue
            label = unparseable_as
        value = 1 if label else 0
        by_type[refs[row["question_id"]]["question_type"]].append(value)
        if "_abs" in row["question_id"]:
            abstention.append(value)
    present = [values for values in by_type.values() if values]
    return {"per_type": {t: _mean(v) for t, v in by_type.items()},
            "task_averaged": _mean([_mean(v) for v in present]),
            "overall": _mean([x for values in by_type.values() for x in values]),
            "abstention": _mean(abstention)}


def summarize(rows: list[dict], refs: dict[str, dict]) -> dict:
    """Aggregates over verdict rows; ``refs`` maps question id to the reference entry."""
    rows = [row for row in rows if row["question_id"] in refs]
    wrong = _aggregates(rows, refs, unparseable_as=False)
    excluded = _aggregates(rows, refs, unparseable_as=None)
    return {
        "questions": len(refs), "judged": len(rows), "missing": len(set(refs) - {r["question_id"] for r in rows}),
        "unparseable": sum(row["autoeval_label"]["label"] is None for row in rows),
        "judges": sorted({row["autoeval_label"]["model"] for row in rows}),
        "overall_as_wrong": wrong["overall"], "overall_excluding": excluded["overall"],
        "task_averaged_as_wrong": wrong["task_averaged"], "task_averaged_excluding": excluded["task_averaged"],
        "abstention": wrong["abstention"], "per_type_as_wrong": wrong["per_type"],
        "per_type_excluding": excluded["per_type"],
    }


def agreement(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Question-by-question agreement of two judges, and Cohen's kappa, where both gave a verdict."""
    a = {row["question_id"]: row["autoeval_label"]["label"] for row in rows_a}
    b = {row["question_id"]: row["autoeval_label"]["label"] for row in rows_b}
    shared = sorted(set(a) & set(b))
    both = [(a[q], b[q]) for q in shared if a[q] is not None and b[q] is not None]
    agree = sum(x == y for x, y in both)
    kappa = None
    if both:
        observed = agree / len(both)
        p_a = sum(x for x, _ in both) / len(both)
        p_b = sum(y for _, y in both) / len(both)
        chance = p_a * p_b + (1 - p_a) * (1 - p_b)
        kappa = (observed - chance) / (1 - chance) if chance < 1 else None
    return {"shared": len(shared), "compared": len(both), "agree": agree,
            "either_unparseable": len(shared) - len(both), "kappa": kappa,
            "disagreements": [q for q in shared if a[q] is not None and b[q] is not None and a[q] != b[q]]}


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results")
    parser.add_argument("reference")
    parser.add_argument("--other", help="a second judge's results on the same hypotheses")
    arguments = parser.parse_args(argv)
    refs = {entry["question_id"]: entry for entry in json.loads(Path(arguments.reference).read_text())}
    out = {"summary": summarize(_rows(arguments.results), refs)}
    if arguments.other:
        out["other"] = summarize(_rows(arguments.other), refs)
        out["agreement"] = agreement(_rows(arguments.results), _rows(arguments.other))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
