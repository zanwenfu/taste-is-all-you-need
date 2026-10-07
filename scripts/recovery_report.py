"""What the recovery study found: the map, rewind-point choices, recoveries, gaming flags, a selector.

    python scripts/recovery_report.py --map /root/study/map.json --runs /root/study/recoveries.json \\
        [--reader /root/study/reader.json] [--compare continue:retry ...] [--features a,b,c] \\
        [--resamples 10000] [--seed 0] [--folds 5] [--depth 2] [--min-leaf 5] \\
        [--json report.json] [--markdown report.md]

Reads only the drivers' state files (which hold every finished trial's
reward, cost and time) and the trajectory reader's choices.

- The map: per run, the curve v(k) with Wilson 95% intervals; the decisive
  steps, absolute and as a share of the run's length; how many failed runs
  were recoverable at some step after the start (a branch from some step k >= 1
  succeeded), with a Wilson interval over runs; curves that recover after a drop.
- Rewind-point choices against the measured truth, for the reader, each rule
  and the oracle: the distance of the first wrong step each implies (the step
  after its rewind point) from the decisive step d, and v at the chosen step
  (the probe at or before it: a step function, exact where it was probed).
- The recoveries, paired by failed run: per run, each recovery's share of
  repeats solved and its mean dollars, tokens, wall-clock and rounds. For each
  comparison OTHER:BASE (every pair, unless --compare names them) the mean
  difference over runs, a percentile bootstrap 95% interval over runs and a
  two-sided sign-flip permutation p (from scripts/study_report.py), with
  Holm's adjustment over the comparisons of each measure.
- Fix or gaming: final changes touching test files or test configuration, or
  adding skip markers, per recovery, against the hidden tests' verdict.
- A selector: a policy tree on what a harness observes (run length, the
  checker's confidence, whether visible tests ever passed), scored by
  cross-validation on the counterfactual outcomes, against every fixed
  recovery and against the best fixed recovery chosen the same way.

Every random draw comes from one generator seeded with --seed.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.agents.trajectory_reader import rewind_point
from taste.recovery_study import selector
from taste.recovery_study.map_driver import MapDriver
from taste.recovery_study.recovery_driver import RecoveryDriver, readings
from taste.recovery_study.search import FINAL, wilson

_SPEC = importlib.util.spec_from_file_location("study_report", Path(__file__).resolve().parent / "study_report.py")
_study = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_study)

VIEWS = ("retry", "retry_feedback", "rewind_oracle", "rewind_reader", "rewind_rules", "continue", "continue_bare")
# A deployed harness cannot rerun to find r*: the selector chooses among the others.
SELECTABLE = ("retry", "retry_feedback", "rewind_reader", "rewind_rules", "continue", "continue_bare")
# usd and seconds are everything an episode spent, the checker's included;
# agent_usd and agent_seconds are what its budget was charged (the agent's).
# solved: the episode's last trial passed the hidden tests (what a harness would keep);
# ever_solved: some trial of it did (a recovery that found a fix the checker then sent back).
MEASURES = ("solved", "ever_solved", "usd", "agent_usd", "tokens", "seconds", "agent_seconds", "rounds")
HIGHER_IS_BETTER = ("solved", "ever_solved")
METHODS = ("reader", "visible_tests", "before_large_edit", "start", "rules", "oracle_highest_v", "oracle_latest")
FEATURES = ("run_length", "checker_confidence", "visible_tests_passed")


def rounded(value):
    """Six decimals, and never a negative zero from summing differences that cancel."""
    return round(value, 6) + 0.0


def describe(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"n": 0}
    quartiles = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]] * 3
    return {"n": len(values), "mean": round(statistics.fmean(values), 6), "min": values[0],
            "q1": round(quartiles[0], 6), "median": round(quartiles[1], 6), "q3": round(quartiles[2], 6),
            "max": values[-1]}


def rate(flags):
    flags = list(flags)
    count = sum(1 for flag in flags if flag)
    low, high = wilson(count, len(flags))
    return {"runs": len(flags), "count": count, "rate": round(count / len(flags), 6) if flags else None,
            "wilson_95": [round(low, 6), round(high, 6)]}


# The map ------------------------------------------------------------------

def map_section(summary):
    runs = {run_id: run for run_id, run in summary.items() if run["status"] != "skipped"}
    final = {run_id: run for run_id, run in runs.items() if run["status"] in FINAL}
    done = {run_id: run for run_id, run in runs.items() if run["status"] == "done"}
    fractions = [run["decisive_fraction"] for run in done.values()]
    histogram = Counter(min(9, int(fraction * 10)) for fraction in fractions)
    return {
        "runs": len(runs), "skipped": len(summary) - len(runs),
        "statuses": dict(sorted(Counter(run["status"] for run in runs.values()).items())),
        "decisive_step": describe(run["decisive_step"] for run in done.values()),
        "decisive_fraction": describe(fractions),
        "decisive_fraction_deciles": {f"{tenth / 10:.1f}-{(tenth + 1) / 10:.1f}": histogram.get(tenth, 0)
                                      for tenth in range(10)},
        "recoverable_after_start": rate(run["recoverable"] for run in final.values()),
        "winnable_until_submission": rate(run["decisive_step"] == run["steps"] for run in done.values()),
        "non_monotone": sum(1 for run in final.values() if not run["monotone"]),
        "with_earlier_drops": sum(1 for run in final.values() if run["earlier_drops"]),
        "passed_then_failed": sum(1 for run in runs.values() if 1.0 in run.get("state_rewards", {}).values()),
        "spend_usd": round(math.fsum(run["spend"]["usd"] for run in runs.values()), 6),
        "trials": sum(run["spend"]["trials"] for run in runs.values()),
        "curves": {run_id: run["curve"] for run_id, run in runs.items()},
    }


def v_at(curve, step):
    """v at a step: the probe at it, else the nearest probed step before it (a step function)."""
    known = [point for point in curve if point["step"] <= step and point["trials"]]
    return known[-1]["v"] if known else None


def choices(run, reading):
    """Each method's rewind point for a mapped run (None where it abstains)."""
    last = run["steps"] - 1
    picked = {name: run["rules"].get(name) for name in ("visible_tests", "before_large_edit", "start", "rules")}
    picked["reader"] = None if reading is None else max(0, min(rewind_point(reading), last))
    picked["oracle_highest_v"] = run["rewind"]["highest_v"]
    picked["oracle_latest"] = run["rewind"]["latest"]
    return picked


def choice_section(summary, reader):
    rows = {method: [] for method in METHODS}
    for run_id, run in summary.items():
        if run["status"] != "done":
            continue
        for method, step in choices(run, reader.get(run_id)).items():
            if step is None:
                continue
            distance = step + 1 - run["decisive_step"]
            rows[method].append({"run": run_id, "step": step, "distance": distance, "v": v_at(run["curve"], step)})
    out = {}
    for method, items in rows.items():
        if not items:
            out[method] = {"runs": 0}
            continue
        distances = [item["distance"] for item in items]
        values = [item["v"] for item in items if item["v"] is not None]
        out[method] = {"runs": len(items), "exact": round(sum(d == 0 for d in distances) / len(items), 6),
                       "within_2": round(sum(abs(d) <= 2 for d in distances) / len(items), 6),
                       "before_decisive": round(sum(d <= 0 for d in distances) / len(items), 6),
                       "mean_abs_distance": round(statistics.fmean(abs(d) for d in distances), 6),
                       "median_distance": statistics.median(distances),
                       "mean_v": round(statistics.fmean(values), 6) if values else None, "choices": items}
    return out


# The recoveries -----------------------------------------------------------

def views(run):
    """Each recovery's finished episodes for a run; rewind arms seen once per source that chose them."""
    found = {}
    for name, arm in run.get("arms", {}).items():
        episodes = [episode for episode in arm["episodes"] if episode]
        if arm["recovery"] == "rewind":
            for source, step in run["sources"].items():
                if step == arm["step"]:
                    found[f"rewind_{source}"] = episodes
        else:
            found[name] = episodes
    return {name: episodes for name, episodes in found.items() if episodes}


def per_run(episodes):
    def mean(key):
        return statistics.fmean(float(episode.get(key) or 0) for episode in episodes)
    return {"solved": statistics.fmean(1.0 if episode["solved"] else 0.0 for episode in episodes),
            "ever_solved": statistics.fmean(1.0 if ever_solved(episode) else 0.0 for episode in episodes),
            **{key: mean(key) for key in MEASURES if key not in HIGHER_IS_BETTER}, "episodes": len(episodes)}


def ever_solved(episode):
    """Whether some trial of the episode passed the hidden tests: its last, or one a check looked at."""
    return bool(episode["solved"]) or any(item.get("solved") for item in episode.get("verdicts") or ())


def recovery_table(summary):
    """run -> view -> the means of its finished episodes."""
    table = {}
    for run_id, run in summary.items():
        if run.get("status") == "rejected":
            found = {name: per_run(episodes) for name, episodes in views(run).items()}
            if found:
                table[run_id] = found
    return table


def compare(table, other, base, measure, resamples, rng):
    runs = sorted(run for run, found in table.items() if other in found and base in found)
    differences = [table[run][other][measure] - table[run][base][measure] for run in runs]
    if not runs:
        return {"runs": 0, "mean_difference": None, "bootstrap_95": [None, None], "permutation_p": 1.0}
    low, high = _study.bootstrap_interval(differences, resamples, rng)
    return {"runs": len(runs), "mean_difference": rounded(math.fsum(differences) / len(runs)),
            "bootstrap_95": [rounded(low), rounded(high)],
            "other_better": sum(1 for d in differences if (d > 0) == (measure in HIGHER_IS_BETTER) and d != 0),
            "base_better": sum(1 for d in differences if (d < 0) == (measure in HIGHER_IS_BETTER) and d != 0),
            "permutation_p": _study.permutation_p(differences, resamples, rng)}


def recovery_section(summary, comparisons, resamples, rng):
    table = recovery_table(summary)
    present = [view for view in VIEWS if any(view in found for found in table.values())]
    arms = {}
    for view in present:
        episodes = [episode for run in summary.values() if run.get("status") == "rejected"
                    for episode in views(run).get(view, ())]
        shares = [found[view]["solved"] for found in table.values() if view in found]
        low, high = _study.bootstrap_interval(shares, resamples, rng) if shares else (None, None)
        solved = sum(1 for episode in episodes if episode["solved"])
        ever = [found[view]["ever_solved"] for found in table.values() if view in found]
        arms[view] = {"runs": len(shares), "episodes": len(episodes),
                      "solved_share": round(statistics.fmean(shares), 6) if shares else None,
                      "ever_solved_share": round(statistics.fmean(ever), 6) if ever else None,
                      "solved_share_bootstrap_95": [low, high],
                      "usd_per_episode": round(statistics.fmean(e["usd"] for e in episodes), 6) if episodes else None,
                      "usd_per_solved": round(math.fsum(e["usd"] for e in episodes) / solved, 6) if solved else None,
                      "seconds_per_episode": round(statistics.fmean(e["seconds"] for e in episodes), 1) if episodes else None,
                      **{f"{key}_per_episode": round(statistics.fmean(float(e.get(key) or 0) for e in episodes), 6)
                         if episodes else None
                         for key in ("agent_usd", "check_usd", "agent_seconds", "check_seconds")},
                      "tokens_per_episode": round(statistics.fmean(e["tokens"] for e in episodes), 1) if episodes else None,
                      "rounds_per_episode": round(statistics.fmean(e["rounds"] for e in episodes), 3) if episodes else None,
                      "ends": dict(sorted(Counter(e["end"] for e in episodes).items()))}
    pairs = comparisons or [(other, base) for base, other in itertools.combinations(present, 2)]
    measures = {}
    for measure in MEASURES:
        items = [{"other": other, "base": base, **compare(table, other, base, measure, resamples, rng)}
                 for other, base in pairs]
        for item, adjusted in zip(items, _study.holm([item["permutation_p"] for item in items]), strict=True):
            item["holm_p"] = adjusted
        measures[measure] = items
    return {"arms": arms, "comparisons": measures, "per_run": table}


def gaming_section(summary):
    out = {}
    for run_id, run in summary.items():
        if run.get("status") != "rejected":
            continue
        for view, episodes in views(run).items():
            entry = out.setdefault(view, {"episodes": 0, "flagged": 0, "flagged_solved": 0, "flagged_unsolved": 0,
                                          "test_files": 0, "check_files": 0, "skip_markers": 0, "items": []})
            for episode in episodes:
                entry["episodes"] += 1
                flags = episode["flags"]
                if not flags:
                    continue
                entry["flagged"] += 1
                entry["flagged_solved" if episode["solved"] else "flagged_unsolved"] += 1
                entry["test_files"] += any(flag.startswith("test_file:") for flag in flags)
                entry["check_files"] += any(flag.startswith("check_file:") for flag in flags)
                entry["skip_markers"] += "skip_marker" in flags
                entry["items"].append({"run": run_id, "solved": episode["solved"], "flags": flags,
                                       "paths": episode["changed_paths"]})
    return out


def checker_section(summary):
    initial = [run["verdict"] for run in summary.values() if run.get("verdict")]
    confusion = Counter()
    for run in summary.values():
        for episodes in views(run).values() if run.get("status") == "rejected" else ():
            for episode in episodes:
                for item in episode["verdicts"]:
                    confusion[f"{item['verdict']}_{'solved' if item['solved'] else 'unsolved'}"] += 1
    confidences = [item.get("confidence") for item in initial if isinstance(item.get("confidence"), (int, float))]
    return {"initial": dict(Counter(item["verdict"] for item in initial)),
            "initial_confidence": describe(confidences),
            "rounds": {key: confusion.get(key, 0) for key in ("done_solved", "done_unsolved", "not_done_solved",
                                                              "not_done_unsolved")}}


def selector_section(summary, features, folds, depth, min_leaf, seed, resamples, rng):
    table = recovery_table(summary)
    arms = [view for view in SELECTABLE if table and all(view in found for found in table.values())]
    rows, names = [], []
    for run_id, found in sorted(table.items()):
        run = summary[run_id]
        observed = {**run["features"], "checker_confidence": (run["verdict"] or {}).get("confidence")}
        rows.append({"features": observed, "outcomes": {arm: found[arm]["solved"] for arm in arms},
                     "costs": {arm: found[arm]["usd"] for arm in arms}})
        names.append(run_id)
    usable = [name for name in features if rows and all(isinstance(row["features"].get(name), (int, float, bool))
                                                         for row in rows)]
    if len(rows) < 2 or len(arms) < 2:
        return {"runs": len(rows), "arms": arms, "features": usable, "note": "too few runs or recoveries"}
    chosen = selector.cross_validate(rows, arms, usable, folds=folds, depth=depth, min_leaf=min_leaf, seed=seed)
    fixed = selector.cross_validate(rows, arms, usable, folds=folds, depth=0, min_leaf=min_leaf, seed=seed)
    scored = {"selector": [row["outcomes"][arm] for row, arm in zip(rows, chosen, strict=True)],
              "best_fixed_by_cv": [row["outcomes"][arm] for row, arm in zip(rows, fixed, strict=True)],
              **{arm: [row["outcomes"][arm] for row in rows] for arm in arms},
              "per_run_best": [max(row["outcomes"].values()) for row in rows]}
    costs = [row["costs"][arm] for row, arm in zip(rows, chosen, strict=True)]
    against = {}
    for name in [*arms, "best_fixed_by_cv"]:
        differences = [a - b for a, b in zip(scored["selector"], scored[name], strict=True)]
        low, high = _study.bootstrap_interval(differences, resamples, rng)
        against[name] = {"mean_difference": rounded(statistics.fmean(differences)),
                         "bootstrap_95": [rounded(low), rounded(high)],
                         "permutation_p": _study.permutation_p(differences, resamples, rng)}
    tree = selector.fit(rows, arms, usable, depth=depth, min_leaf=min_leaf)
    return {"runs": len(rows), "arms": arms, "features": usable, "folds": folds, "depth": depth,
            "min_leaf": min_leaf, "solved_share": {name: round(statistics.fmean(values), 6)
                                                   for name, values in scored.items()},
            "selector_usd_per_run": round(statistics.fmean(costs), 6),
            "selector_choices": dict(sorted(Counter(chosen).items())),
            "selector_minus": against, "tree": selector.describe(tree),
            "choices": dict(zip(names, chosen, strict=True))}


def report(map_summary, recovery_summary, reader, *, comparisons=None, features=FEATURES, resamples=10000,
           seed=0, folds=5, depth=2, min_leaf=5):
    rng = random.Random(seed)
    result = {"seed": seed, "resamples": resamples}
    if map_summary is not None:
        result["map"] = map_section(map_summary)
        result["choices"] = choice_section(map_summary, reader)
    if recovery_summary is not None:
        result["checker"] = checker_section(recovery_summary)
        result["recoveries"] = recovery_section(recovery_summary, comparisons, resamples, rng)
        result["gaming"] = gaming_section(recovery_summary)
        result["selector"] = selector_section(recovery_summary, features, folds, depth, min_leaf, seed,
                                              resamples, rng)
    return result


# Markdown -----------------------------------------------------------------

def _f(value, digits=3):
    return "" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def markdown(result):
    lines = ["# Recovery study report", ""]
    if "map" in result:
        found = result["map"]
        recoverable, to_end = found["recoverable_after_start"], found["winnable_until_submission"]
        step, share = found["decisive_step"], found["decisive_fraction"]
        lines += ["## Where failed runs become unwinnable", "",
                  f"{found['runs']} failed runs mapped ({found['skipped']} left out); statuses: "
                  f"{json.dumps(found['statuses'])}. {found['trials']} trials, ${found['spend_usd']:.2f}.", "",
                  f"- Recoverable after the start (some branch from a step k >= 1 succeeded): "
                  f"{recoverable['count']}/{recoverable['runs']} = {_f(recoverable['rate'])} "
                  f"(Wilson 95% [{_f(recoverable['wilson_95'][0])}, {_f(recoverable['wilson_95'][1])}]).",
                  f"- Winnable until the submission itself (d = T): {to_end['count']}/{to_end['runs']}.",
                  f"- Decisive step d: median {_f(step.get('median'), 1)} (quartiles {_f(step.get('q1'), 1)} to "
                  f"{_f(step.get('q3'), 1)}, n = {step['n']}); as a share of the run: median "
                  f"{_f(share.get('median'))} (quartiles {_f(share.get('q1'))} to {_f(share.get('q3'))}).",
                  f"- Curves that recover after a drop: {found['non_monotone']}; runs with an earlier drop: "
                  f"{found['with_earlier_drops']}; runs whose state passed the hidden tests at some probed step: "
                  f"{found['passed_then_failed']}.", "",
                  "| d / T | " + " | ".join(found["decisive_fraction_deciles"]) + " |",
                  "| --- | " + " | ".join("---" for _ in found["decisive_fraction_deciles"]) + " |",
                  "| runs | " + " | ".join(str(v) for v in found["decisive_fraction_deciles"].values()) + " |", ""]
    if "choices" in result:
        lines += ["## Choosing the rewind point without reruns", "",
                  "Distance: the first wrong step a method implies (the step after its rewind point) minus d.", "",
                  "| Method | Runs | Exact | Within 2 | At or before d | Mean abs distance | Median distance | Mean v at choice |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for method, item in result["choices"].items():
            if item["runs"]:
                lines.append(f"| {method} | {item['runs']} | {_f(item['exact'])} | {_f(item['within_2'])} | "
                             f"{_f(item['before_decisive'])} | {_f(item['mean_abs_distance'], 2)} | "
                             f"{item['median_distance']} | {_f(item['mean_v'])} |")
        lines.append("")
    if "recoveries" in result:
        checker = result["checker"]
        lines += ["## The checker", "", f"Its verdicts on the failed base runs: {json.dumps(checker['initial'])}. "
                  f"On the recovery trials it checked, its verdict against the hidden tests' grade: "
                  f"{json.dumps(checker['rounds'])}.", "",
                  "## Recoveries, paired by failed run", "",
                  "Dollars and seconds are everything an episode spent; its budget was charged only the "
                  "agent's (agent $, agent s), and the checker's are beside them.", "",
                  "| Recovery | Runs | Episodes | Solved share | Bootstrap 95% | Solved at some round | $/episode | "
                  "agent $ | checker $ | $/solved | s/episode | agent s | Tokens/episode | Rounds | Ends |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for view, arm in result["recoveries"]["arms"].items():
            low, high = arm["solved_share_bootstrap_95"]
            lines.append(f"| {view} | {arm['runs']} | {arm['episodes']} | {_f(arm['solved_share'])} | "
                         f"[{_f(low)}, {_f(high)}] | {_f(arm['ever_solved_share'])} | {_f(arm['usd_per_episode'], 4)} | "
                         f"{_f(arm['agent_usd_per_episode'], 4)} | {_f(arm['check_usd_per_episode'], 4)} | "
                         f"{_f(arm['usd_per_solved'], 4)} | {_f(arm['seconds_per_episode'], 0)} | "
                         f"{_f(arm['agent_seconds_per_episode'], 0)} | {_f(arm['tokens_per_episode'], 0)} | "
                         f"{_f(arm['rounds_per_episode'], 2)} | {json.dumps(arm['ends'])} |")
        for measure, items in result["recoveries"]["comparisons"].items():
            lines += ["", f"{measure}: other minus base, over the runs both have; Holm over these comparisons.", "",
                      "| Other vs base | Runs | Mean difference | Bootstrap 95% | Permutation p | Holm p |",
                      "| --- | --- | --- | --- | --- | --- |"]
            for item in items:
                low, high = item["bootstrap_95"]
                lines.append(f"| {item['other']} vs {item['base']} | {item['runs']} | {_f(item['mean_difference'], 4)} | "
                             f"[{_f(low, 4)}, {_f(high, 4)}] | {item['permutation_p']:.4g} | {item['holm_p']:.4g} |")
        lines += ["", "## Fix or gaming", "",
                  "| Recovery | Episodes | Flagged | Flagged, solved | Flagged, unsolved | Test files | Checks | Skip markers |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for view, item in result["gaming"].items():
            lines.append(f"| {view} | {item['episodes']} | {item['flagged']} | {item['flagged_solved']} | "
                         f"{item['flagged_unsolved']} | {item['test_files']} | {item['check_files']} | "
                         f"{item['skip_markers']} |")
        chosen = result["selector"]
        lines += ["", "## A selector", ""]
        if "note" in chosen:
            lines.append(f"Not fitted: {chosen['note']} ({chosen['runs']} runs).")
        else:
            lines += [f"{chosen['runs']} runs, recoveries {', '.join(chosen['arms'])}; features "
                      f"{', '.join(chosen['features']) or '(none)'}; {chosen['folds']}-fold cross-validation, "
                      f"depth {chosen['depth']}, at least {chosen['min_leaf']} runs a leaf.", "",
                      "| Policy | Solved share |", "| --- | --- |"]
            lines += [f"| {name} | {_f(value)} |" for name, value in chosen["solved_share"].items()]
            lines += ["", "| Selector minus | Mean difference | Bootstrap 95% | Permutation p |", "| --- | --- | --- | --- |"]
            for name, item in chosen["selector_minus"].items():
                lines.append(f"| {name} | {_f(item['mean_difference'], 4)} | [{_f(item['bootstrap_95'][0], 4)}, "
                             f"{_f(item['bootstrap_95'][1], 4)}] | {item['permutation_p']:.4g} |")
            lines += ["", "Tree fitted on every run:", "", "```", *chosen["tree"], "```"]
    if "map" in result:
        lines += ["", "## Curves", "", "v(k) per run: successes / branches from step k, with the Wilson 95% "
                  "interval; step 0 is the task's runs from scratch.", "", "| Run | Curve |", "| --- | --- |"]
        for run_id, curve in result["map"]["curves"].items():
            lines.append(f"| {run_id} | " + "; ".join(f"{p['step']}: {p['successes']}/{p['trials']} "
                                                       f"[{p['low']:.2f}, {p['high']:.2f}]" for p in curve) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, help="the map's state file")
    parser.add_argument("--runs", type=Path, help="the recoveries' state file")
    parser.add_argument("--reader", type=Path, help="the trajectory reader's readings (JSON)")
    parser.add_argument("--compare", action="append", help="OTHER:BASE, as registered (default: every pair)")
    parser.add_argument("--features", default=",".join(FEATURES), help="the selector's features, comma-separated")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--min-leaf", type=int, default=5)
    parser.add_argument("--json", type=Path, help="also write the full result here")
    parser.add_argument("--markdown", type=Path, help="also write the summary here")
    arguments = parser.parse_args(argv)
    if not arguments.map and not arguments.runs:
        parser.error("give --map, --runs or both")
    result = report(MapDriver.open(arguments.map).summary() if arguments.map else None,
                    RecoveryDriver.open(arguments.runs).summary() if arguments.runs else None,
                    readings(arguments.reader),
                    comparisons=[tuple(item.split(":", 1)) for item in arguments.compare or ()],
                    features=[name for name in arguments.features.split(",") if name],
                    resamples=arguments.resamples, seed=arguments.seed, folds=arguments.folds,
                    depth=arguments.depth, min_leaf=arguments.min_leaf)
    text = markdown(result)
    if arguments.json:
        arguments.json.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    if arguments.markdown:
        arguments.markdown.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
