"""Registered comparisons between arms of Harbor trials, paired by task.

    python scripts/study_report.py --arm alone=/root/tb/jobs/s-alone --arm taste=/root/tb/jobs/s-taste \\
        --arm continue=/root/tb/jobs/s-continue --compare taste:alone --compare taste:continue \\
        [--trials /var/lib/taste-trials] [--resamples 10000] [--seed 0] [--json out.json]

For each comparison OTHER:BASE, over the tasks both arms ran: each task's
share of runs solved in either arm, and their difference (OTHER minus BASE).
Reported per comparison: the mean difference over tasks; a percentile
bootstrap 95% interval over tasks; a two-sided sign-flip permutation test of
the mean difference over tasks (exact up to 20 tasks that differ, otherwise
from the given number of random sign flips), which is the primary test; the
exact two-sided sign test; and the primary p-values adjusted by Holm's method
over the comparisons named. Costs per trial and per solved task, by arm, come
from Taste's settled records, as in the go/no-go report.

Every random draw comes from one generator seeded with --seed, so the same
records and seed give the same report.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("go_no_go_report", Path(__file__).resolve().parent / "go_no_go_report.py")
_records = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_records)

EXACT_TASKS = 20


def task_rates(rows):
    """Each task's share of its runs that the verifier passed."""
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(1.0 if row["solved"] else 0.0)
    return {task: math.fsum(values) / len(values) for task, values in by_task.items()}


def bootstrap_interval(differences, resamples, rng, level=0.95):
    """Percentile interval of the mean difference, resampling tasks with replacement."""
    n = len(differences)
    means = sorted(math.fsum(rng.choice(differences) for _ in range(n)) / n for _ in range(resamples))
    low = means[math.floor((1 - level) / 2 * resamples)]
    high = means[min(resamples - 1, math.ceil((1 + level) / 2 * resamples) - 1)]
    return low, high


def permutation_p(differences, resamples, rng):
    """Two-sided sign-flip test of the mean difference: exact for few differing tasks."""
    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return 1.0
    observed = abs(math.fsum(nonzero))
    tolerance = 1e-12
    if len(nonzero) <= EXACT_TASKS:
        flips = itertools.product((1, -1), repeat=len(nonzero))
        extreme = sum(abs(math.fsum(s * v for s, v in zip(signs, nonzero, strict=True))) >= observed - tolerance
                      for signs in flips)
        return extreme / 2 ** len(nonzero)
    extreme = sum(abs(math.fsum(rng.choice((1, -1)) * v for v in nonzero)) >= observed - tolerance
                  for _ in range(resamples))
    # Counting the observed assignment keeps a Monte Carlo p-value valid.
    return (extreme + 1) / (resamples + 1)


def holm(p_values):
    """Holm's step-down adjustment, in the order given."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def compare(base_rows, other_rows, resamples, rng):
    base, other = task_rates(base_rows), task_rates(other_rows)
    tasks = sorted(set(base) & set(other))
    differences = [other[task] - base[task] for task in tasks]
    wins = sum(value > 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    low, high = bootstrap_interval(differences, resamples, rng) if tasks else (None, None)
    return {"tasks": len(tasks), "mean_difference": math.fsum(differences) / len(tasks) if tasks else None,
            "bootstrap_95": [low, high], "other_better": wins, "base_better": losses,
            "same": len(tasks) - wins - losses, "permutation_p": permutation_p(differences, resamples, rng),
            "sign_test_p": _records.sign_test(wins, losses),
            "per_task": {task: {"base": base[task], "other": other[task]} for task in tasks}}


def report(arms, comparisons, trials_root, resamples=10000, seed=0):
    loaded = {name: _records.load_arm(path, trials_root) for name, path in arms}
    rng = random.Random(seed)
    result = {"arms": {name: _records.summarize(rows) for name, rows in loaded.items()},
              "comparisons": [], "resamples": resamples, "seed": seed}
    for other, base in comparisons:
        if other not in loaded or base not in loaded:
            raise ValueError(f"comparison {other}:{base} names an arm that was not given")
        result["comparisons"].append({"other": other, "base": base,
                                      **compare(loaded[base], loaded[other], resamples, rng)})
    adjusted_p = holm([item["permutation_p"] for item in result["comparisons"]])
    for item, adjusted in zip(result["comparisons"], adjusted_p, strict=True):
        item["holm_p"] = adjusted
    return result


def markdown(result):
    lines = ["| Arm | Solved | Trials | $/trial | $/solved |", "| --- | --- | --- | --- | --- |"]
    for name, arm in result["arms"].items():
        lines.append(f"| {name} | {arm['solved']} | {arm['trials']} | {arm['usd_per_trial']} | "
                     f"{arm['usd_per_solved_task']} |")
    lines += ["", "| Comparison | Tasks | Mean difference | Bootstrap 95% | Better / worse / same | "
                  "Permutation p | Holm p | Sign test p |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for item in result["comparisons"]:
        low, high = item["bootstrap_95"]
        lines.append(f"| {item['other']} vs {item['base']} | {item['tasks']} | {item['mean_difference']:.3f} | "
                     f"[{low:.3f}, {high:.3f}] | {item['other_better']} / {item['base_better']} / {item['same']} | "
                     f"{item['permutation_p']:.4g} | {item['holm_p']:.4g} | {item['sign_test_p']:.4g} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", action="append", required=True, help="name=harbor job directory")
    parser.add_argument("--compare", action="append", required=True, help="other:base, as registered")
    parser.add_argument("--trials", default="/var/lib/taste-trials", help="Taste's trial records")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="also write the full result here")
    arguments = parser.parse_args(argv)
    arms = [tuple(item.split("=", 1)) for item in arguments.arm]
    comparisons = [tuple(item.split(":", 1)) for item in arguments.compare]
    result = report(arms, comparisons, arguments.trials, arguments.resamples, arguments.seed)
    if arguments.json:
        arguments.json.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(markdown(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
