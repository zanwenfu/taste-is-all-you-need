"""The recovery study's figures, drawn from scripts/recovery_report.py's --json result.

    python scripts/recovery_figures.py --report report.json --out figures/ [--format png|pdf]

- curves: v(k) per mapped run (its search ended), the Wilson 95% interval at
  each probed step, the decisive step d dashed;
- decisive: where failed runs become unwinnable, as a share of the run (d / T);
- recoveries: each recovery's solved share (bootstrap 95% over runs), its
  paired difference from plain retry (bootstrap 95%), and its dollars and
  seconds per episode, the agent's (what the budget was charged) and the
  checker's;
- checker: the checker's verdicts on the recovery trials it checked, against
  the hidden tests;
- choices: how far the first wrong step each rewind-point method implies lies
  from d (the oracle's latest winnable step is d - 1 by definition, so it is
  left out).

What each figure shows is computed by a plain function; only ``render`` draws,
and needs matplotlib (pip install '.[figures]').
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.recovery_study.search import FINAL

FIGURES = ("curves", "decisive", "recoveries", "checker", "choices")
# The report's order of the recoveries (its --json file sorts keys, so the order is not kept there).
VIEWS = ("retry", "retry_feedback", "rewind_oracle", "rewind_reader", "rewind_rules", "continue", "continue_bare",
         "retry_bare", "rewind_bare_oracle", "rewind_bare_reader", "rewind_bare_rules",
         "retry_tests", "rewind_tests_oracle", "rewind_tests_reader", "rewind_tests_rules", "continue_tests",
         "continue_tests_code", "continue_tests_full", "continue_both", "continue_tests_hidden",
         "agentrewind", "agentrewind_norestore")
METHODS = ("reader", "rules", "before_large_edit", "visible_tests", "start", "oracle_highest_v")
LABELS = {"retry": "Plain retry", "retry_feedback": "Retry + feedback", "rewind_oracle": "Rewind to r* (oracle)",
          "rewind_reader": "Rewind (reader)", "rewind_rules": "Rewind (rules)", "continue": "Continue + feedback",
          "continue_bare": "Continue, bare", "retry_bare": "Retry, bare", "rewind_bare_oracle": "Rewind to r*, bare",
          "rewind_bare_reader": "Rewind (reader), bare", "rewind_bare_rules": "Rewind (rules), bare",
          "retry_tests": "Retry + failed tests", "rewind_tests_oracle": "Rewind to r* + failed tests",
          "rewind_tests_reader": "Rewind (reader) + failed tests", "rewind_tests_rules": "Rewind (rules) + failed tests",
          "continue_tests": "Continue + failed tests", "continue_tests_code": "Continue + failed tests and their code",
          "continue_tests_full": "Continue + failed tests, wide output",
          "continue_both": "Continue + feedback and failed tests",
          "continue_tests_hidden": "Continue + failed tests, said to be hidden",
          "agentrewind": "AgentRewind", "agentrewind_norestore": "AgentRewind's note, no rewind",
          "reader": "Trajectory reader", "rules": "Rules",
          "before_large_edit": "Before the last large edit", "visible_tests": "Visible tests last passed",
          "start": "Start (retry)", "oracle_highest_v": "Oracle: highest v"}

# The reference palette's first two categorical slots (validated, light), its ink and a sequential blue.
SURFACE, INK, SECONDARY, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281")


# What each figure shows ---------------------------------------------------

def curves(result):
    """Each mapped run whose search ended: its probed points and decisive step, in run order."""
    per_run = result["map"]["per_run"]
    panels = []
    for run_id, curve in sorted(result["map"]["curves"].items()):
        info = per_run[run_id]
        if info["status"] not in FINAL:
            continue
        panels.append({"run": run_id, "status": info["status"], "steps": info["steps"],
                       "decisive": info["decisive_step"],
                       "points": [{key: point[key] for key in ("step", "v", "low", "high", "trials")}
                                  for point in curve if point["trials"]]})
    return panels


def decisive_fractions(result):
    return sorted(info["decisive_fraction"] for info in result["map"]["per_run"].values()
                  if info["status"] == "done")


def paired(items, other, base):
    """The registered OTHER - BASE difference in solved share, turned round if it was BASE - OTHER."""
    for item in items:
        if (item["other"], item["base"]) == (other, base):
            return {"mean": item["mean_difference"], "interval": list(item["bootstrap_95"]), "runs": item["runs"]}
        if (item["other"], item["base"]) == (base, other):
            low, high = item["bootstrap_95"]
            return {"mean": -item["mean_difference"], "interval": [-high, -low], "runs": item["runs"]}
    return None


def recoveries(result, base="retry"):
    found = result["recoveries"]
    rows = []
    for view in sorted(found["arms"], key=lambda view: (VIEWS.index(view) if view in VIEWS else len(VIEWS), view)):
        arm = found["arms"][view]
        rows.append({"view": view, "runs": arm["runs"], "solved": arm["solved_share"],
                     "interval": arm["solved_share_bootstrap_95"],
                     "difference": None if view == base else paired(found["comparisons"]["solved"], view, base),
                     **{key: arm[f"{key}_per_episode"] or 0.0
                        for key in ("agent_usd", "check_usd", "agent_seconds", "check_seconds")}})
    return rows


def checker_cells(result):
    """[[done & passed, done & failed], [not done & passed, not done & failed]]."""
    rounds = result["checker"]["rounds"]
    return [[rounds["done_solved"], rounds["done_unsolved"]], [rounds["not_done_solved"], rounds["not_done_unsolved"]]]


def distances(result):
    found = result["choices"]
    return {method: [item["distance"] for item in found[method]["choices"]]
            for method in METHODS if found.get(method, {}).get("runs")}


# Drawing ------------------------------------------------------------------

def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "text.color": INK, "axes.labelcolor": SECONDARY, "xtick.color": SECONDARY, "ytick.color": SECONDARY,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 7})


def _short(run_id):
    """A run's task without the benchmark's prefix, and the trial's suffix that tells its runs apart."""
    parts = run_id.split("__")
    return f"{parts[1]} {parts[-1]}" if len(parts) >= 3 else run_id


def _draw_curves(plt, result):
    panels = curves(result)
    columns = 8
    rows = max(1, -(-len(panels) // columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.55, rows * 1.25 + 0.6), sharey=True,
                                squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, panel in zip(axes.flat, panels, strict=False):
        steps = [p["step"] for p in panel["points"]]
        values = [p["v"] for p in panel["points"]]
        below = [p["v"] - p["low"] for p in panel["points"]]
        above = [p["high"] - p["v"] for p in panel["points"]]
        if panel["decisive"] is not None:
            ax.axvline(panel["decisive"], color=MUTED, linewidth=1, linestyle=(0, (3, 2)))
        ax.plot(steps, values, color=BLUE, linewidth=1)
        ax.errorbar(steps, values, yerr=[below, above], fmt="o", color=BLUE, markersize=2.5, elinewidth=0.8,
                    capsize=0)
        ax.set_xlim(-0.5, panel["steps"] + 0.5)
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks([0, 0.5, 1])
        title = _short(panel["run"]) + ("" if panel["status"] == "done" else f" ({panel['status']})")
        ax.set_title(title, loc="left", fontsize=5.5, color=SECONDARY)
        ax.tick_params(labelsize=6, length=2)
    figure.supxlabel("step k (dashed: decisive step d)", fontsize=8, color=SECONDARY)
    figure.supylabel("v(k): branches from step k that pass", fontsize=8, color=SECONDARY)
    figure.suptitle(f"Success chance of a branch from each probed step, {len(panels)} failed runs "
                    "(Wilson 95% intervals)", fontsize=9, color=INK, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.35 / figure.get_figheight()))
    return figure


def _draw_decisive(plt, result):
    fractions = decisive_fractions(result)
    figure, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.hist(fractions, bins=[tenth / 10 for tenth in range(11)], color=BLUE, edgecolor=SURFACE, linewidth=2)
    ax.set_xlim(0, 1)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("decisive step as a share of the run (d / T)")
    ax.set_ylabel("failed runs")
    ax.grid(axis="x", visible=False)
    median = statistics.median(fractions) if fractions else None
    ax.set_title(f"Where failed runs become unwinnable (n = {len(fractions)}"
                 + (f", median {median:.2f})" if median is not None else ")"), loc="left")
    figure.tight_layout()
    return figure


def _draw_recoveries(plt, result):
    rows = recoveries(result)
    figure, axes = plt.subplots(1, 4, figsize=(10.5, 0.42 * len(rows) + 1.3), sharey=True)
    ticks = list(range(len(rows)))[::-1]
    names = [f"{LABELS.get(row['view'], row['view'])} ({row['runs']})" for row in rows]
    solved, difference, dollars, seconds = axes
    for y, row in zip(ticks, rows, strict=True):
        low, high = row["interval"]
        if low is not None:
            solved.plot([low, high], [y, y], color=BLUE, linewidth=1.5)
        solved.plot([row["solved"]], [y], "o", color=BLUE, markersize=5)
        found = row["difference"]
        if found:
            difference.plot(found["interval"], [y, y], color=BLUE, linewidth=1.5)
            difference.plot([found["mean"]], [y], "o", color=BLUE, markersize=5)
        elif row["view"] == "retry":
            difference.text(0, y, "base", ha="center", va="center", color=SECONDARY, fontsize=7)
    for ax, (agent, check, unit) in ((dollars, ("agent_usd", "check_usd", "$ per episode")),
                                     (seconds, ("agent_seconds", "check_seconds", "seconds per episode"))):
        ax.barh(ticks, [row[agent] for row in rows], color=BLUE, height=0.6, edgecolor=SURFACE, linewidth=2,
                label="agent (charged to the budget)")
        ax.barh(ticks, [row[check] for row in rows], left=[row[agent] for row in rows], color=ORANGE,
                height=0.6, edgecolor=SURFACE, linewidth=2, label="checker")
        ax.set_xlabel(unit)
        ax.grid(axis="y", visible=False)
    seconds.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.2)
    difference.axvline(0, color=MUTED, linewidth=1)
    solved.set_xlim(0, 1)
    solved.set_xlabel("solved share (bootstrap 95% over runs)")
    difference.set_xlabel("minus plain retry, paired (bootstrap 95%)")
    solved.set_yticks(ticks, names)
    for ax in axes:
        ax.grid(axis="y", visible=False)
    figure.suptitle("Recoveries from the same failed runs at equal budget (runs in brackets)", fontsize=9,
                    color=INK, x=0.01, ha="left")
    figure.tight_layout()
    return figure


def _draw_checker(plt, result):
    from matplotlib.colors import LinearSegmentedColormap

    cells = checker_cells(result)
    total = sum(map(sum, cells)) or 1
    figure, ax = plt.subplots(figsize=(3.8, 2.6))
    ax.imshow(cells, cmap=LinearSegmentedColormap.from_list("blue", SEQUENTIAL), vmin=0,
              vmax=max(max(map(max, cells)), 1))
    top = max(max(map(max, cells)), 1)
    for i, row in enumerate(cells):
        share = sum(row) or 1
        for j, count in enumerate(row):
            ax.text(j, i, f"{count}\n{count / share:.0%} of row", ha="center", va="center", fontsize=8,
                    color=SURFACE if count > top * 0.55 else INK)
    ax.set_xticks([0, 1], ["hidden tests pass", "hidden tests fail"])
    ax.set_yticks([0, 1], ["checker: done", "checker: not done"])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"The checker's verdicts on {total} recovery trials", loc="left")
    figure.tight_layout()
    return figure


def _draw_choices(plt, result):
    found = distances(result)
    rng = random.Random(0)
    figure, ax = plt.subplots(figsize=(6.0, 0.45 * max(len(found), 1) + 1.0))
    ticks = list(range(len(found)))[::-1]
    for y, values in zip(ticks, found.values(), strict=True):
        ax.scatter([value + rng.uniform(-0.15, 0.15) for value in values],
                   [y + rng.uniform(-0.18, 0.18) for _ in values], s=9, color=BLUE, alpha=0.55, linewidths=0)
        ax.plot([statistics.median(values)] * 2, [y - 0.3, y + 0.3], color=INK, linewidth=1.5)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(ticks, [f"{LABELS.get(method, method)} ({len(values)})" for method, values in found.items()])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("first wrong step the method implies minus the decisive step (bar: median)")
    ax.set_title("Choosing the rewind point without reruns, against the map (runs in brackets)", loc="left")
    figure.tight_layout()
    return figure


DRAW = {"curves": _draw_curves, "decisive": _draw_decisive, "recoveries": _draw_recoveries,
        "checker": _draw_checker, "choices": _draw_choices}
NEEDS = {"curves": "map", "decisive": "map", "choices": "choices", "recoveries": "recoveries", "checker": "checker"}


def render(result, out, fmt="png"):
    """Draw every figure the report has the data for; the paths written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name in FIGURES:
        if NEEDS[name] not in result:
            continue
        figure = DRAW[name](plt, result)
        path = out / f"{name}.{fmt}"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        written.append(path)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True, help="recovery_report.py's --json output")
    parser.add_argument("--out", type=Path, required=True, help="the directory to write the figures to")
    parser.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    arguments = parser.parse_args(argv)
    for path in render(json.loads(arguments.report.read_text()), arguments.out, arguments.format):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
