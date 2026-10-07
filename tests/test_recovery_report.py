"""The recovery study's analysis, checked on synthetic summaries with known answers."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

from taste.recovery_study import selector
from taste.recovery_study.search import point, wilson
from tests.recovery_fakes import FakeHarbor, agent_steps, base_jobs, branches, recoveries
from tests.test_recovery_map import run_to_end
from tests.test_recovery_runs import drive

_SPEC = importlib.util.spec_from_file_location(
    "recovery_report", Path(__file__).resolve().parents[1] / "scripts" / "recovery_report.py")
report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report)


def mapped(status, steps, decisive, probes, *, rules=None, rewind=None, monotone=True, drops=()):
    curve = [point(step, s, n) for step, (s, n) in sorted(probes.items())]
    winnable = [p for p in curve if p["step"] >= 1 and p["successes"]]
    return {"task": "t", "base": {}, "status": status, "steps": steps, "decisive_step": decisive,
            "decisive_fraction": None if decisive is None else decisive / steps, "curve": curve,
            "recoverable": bool(winnable), "monotone": monotone, "earlier_drops": list(drops),
            "rewind": rewind or {"highest_v": 0, "latest": 0}, "state_rewards": {},
            "rules": rules or {"visible_tests": None, "before_large_edit": None, "start": 0, "rules": 0},
            "spend": {"usd": 1.0, "trials": 10}}


def test_the_map_section_counts_decisive_steps_recoverable_runs_and_drops():
    summary = {
        "r1": mapped("done", 20, 12, {0: (1, 2), 10: (4, 8), 11: (8, 16), 12: (0, 16)}),
        "r2": mapped("done", 10, 10, {0: (1, 3), 5: (2, 8), 9: (1, 16)}),
        "r3": mapped("done", 16, 1, {0: (1, 2), 1: (0, 16)}),
        "r4": mapped("censored", 30, None, {0: (1, 2), 7: (3, 8)}, monotone=False, drops=[{"after": 3, "by": 5}]),
        "r5": {"task": "t", "base": {}, "status": "skipped", "skipped": "1 steps"},
    }
    found = report.map_section(summary)
    assert (found["runs"], found["skipped"], found["statuses"]) == (4, 1, {"censored": 1, "done": 3})
    assert found["decisive_step"] == {"n": 3, "mean": 7.666667, "min": 1, "q1": 5.5, "median": 10.0, "q3": 11.0,
                                      "max": 12}
    assert found["decisive_fraction"]["median"] == 0.6
    assert found["decisive_fraction_deciles"]["0.6-0.7"] == 1 and found["decisive_fraction_deciles"]["0.9-1.0"] == 1
    assert found["decisive_fraction_deciles"]["0.0-0.1"] == 1
    low, high = wilson(3, 4)
    assert found["recoverable_after_start"] == {"runs": 4, "count": 3, "rate": 0.75,
                                                "wilson_95": [round(low, 6), round(high, 6)]}
    assert found["winnable_until_submission"]["count"] == 1
    assert (found["non_monotone"], found["with_earlier_drops"], found["spend_usd"]) == (1, 1, 4.0)


def test_rewind_choices_are_scored_by_their_distance_from_the_decisive_step_and_v_there():
    summary = {
        "r1": mapped("done", 20, 12, {0: (1, 2), 10: (4, 8), 11: (8, 16), 12: (0, 16), 15: (0, 8)},
                     rules={"visible_tests": 11, "before_large_edit": 5, "start": 0, "rules": 11},
                     rewind={"highest_v": 11, "latest": 11}),
        "r2": mapped("done", 10, 10, {0: (1, 3), 5: (2, 8), 9: (1, 16)},
                     rules={"visible_tests": None, "before_large_edit": 3, "start": 0, "rules": 3},
                     rewind={"highest_v": 5, "latest": 9}),
        "r3": mapped("searching", 10, None, {0: (1, 2)}),
    }
    found = report.choice_section(summary, {"r1": {"step": 12}, "r2": {"step": 10, "reason": "r", "confidence": 0.4}})
    assert found["reader"]["runs"] == 2 and found["reader"]["exact"] == 1.0
    assert found["visible_tests"]["runs"] == 1 and found["visible_tests"]["mean_v"] == 0.5
    edit = found["before_large_edit"]
    assert (edit["exact"], edit["within_2"], edit["before_decisive"], edit["mean_abs_distance"]) == (0.0, 0.0, 1.0, 6.0)
    assert edit["median_distance"] == -6 and edit["mean_v"] == pytest.approx((0.5 + 1 / 3) / 2, abs=1e-6)
    assert found["start"]["median_distance"] == -10.0
    rules = found["rules"]
    assert (rules["exact"], rules["within_2"], rules["mean_abs_distance"]) == (0.5, 0.5, 3.0)
    assert [item["distance"] for item in found["oracle_highest_v"]["choices"]] == [0, -4]
    assert found["oracle_latest"]["exact"] == 1.0


def episodes(solved, usd, count=2, flags=()):
    return [{"end": "done" if solved else "budget", "rounds": 1 if solved else 3, "reward": float(solved),
             "solved": solved, "usd": usd, "seconds": 100.0, "tokens": 1000, "agent_usd": usd, "check_usd": 0.0,
             "verdicts": [{"verdict": "done", "confidence": 0.9, "solved": solved}], "changed_paths": [],
             "flags": list(flags)} for _ in range(count)]


def rejected(arms, features=None, confidence=0.8, sources=None):
    return {"task": "t", "base": {}, "status": "rejected", "steps": 10, "budget": {"usd": 1.0, "seconds": 600},
            "features": features or {"run_length": 10, "visible_tests_passed": False},
            "rules": {}, "verdict": {"verdict": "not_done", "confidence": confidence}, "sources": sources or {},
            "arms": {name: {"recovery": name.split("@")[0], "step": int(name.split("@")[1]) if "@" in name else 0,
                            "episodes": items} for name, items in arms.items()}}


def test_recoveries_are_compared_paired_by_run_with_bootstrap_permutation_and_holm():
    summary = {f"r{i}": rejected({"retry": episodes(False, 0.5), "continue": episodes(True, 0.3)}) for i in range(6)}
    summary["r9"] = {"task": "t", "base": {}, "status": "accepted"}
    found = report.recovery_section(summary, None, 2000, random.Random(3))
    assert found["arms"]["continue"]["solved_share"] == 1.0 and found["arms"]["retry"]["solved_share"] == 0.0
    assert found["arms"]["continue"]["usd_per_solved"] == 0.3 and found["arms"]["retry"]["usd_per_solved"] is None
    [solved] = found["comparisons"]["solved"]
    assert (solved["other"], solved["base"], solved["runs"], solved["mean_difference"]) == ("continue", "retry", 6, 1.0)
    # Six runs differ, all one way: 2 of the 2**6 sign assignments are as extreme.
    assert solved["permutation_p"] == pytest.approx(2 / 64) and solved["holm_p"] == pytest.approx(2 / 64)
    assert solved["bootstrap_95"] == [1.0, 1.0] and solved["other_better"] == 6
    [cost] = found["comparisons"]["usd"]
    assert cost["mean_difference"] == pytest.approx(-0.2) and cost["other_better"] == 6
    # What the budget was charged (the agent's) is compared too, and the checker's is shown beside it.
    [charged] = found["comparisons"]["agent_usd"]
    assert charged["mean_difference"] == pytest.approx(-0.2) and found["arms"]["continue"]["check_usd_per_episode"] == 0.0
    registered = report.recovery_section(summary, [("retry", "continue"), ("continue", "retry")], 500,
                                         random.Random(0))
    assert [item["holm_p"] for item in registered["comparisons"]["solved"]] == pytest.approx([2 * 2 / 64] * 2)


def test_rewind_sources_that_chose_one_step_are_each_seen_with_its_episodes():
    run = rejected({"rewind@3": episodes(True, 0.4)}, sources={"oracle": 3, "rules": 3, "reader": 5})
    assert sorted(report.views(run)) == ["rewind_oracle", "rewind_rules"]


def test_gaming_flags_are_counted_against_the_hidden_tests():
    summary = {"r1": rejected({"continue": episodes(True, 0.3, flags=["test_file:tests/test_a.py"])
                               + episodes(False, 0.3, count=1, flags=["skip_marker"])})}
    found = report.gaming_section(summary)["continue"]
    assert (found["episodes"], found["flagged"], found["flagged_solved"], found["flagged_unsolved"]) == (3, 3, 2, 1)
    assert (found["test_files"], found["skip_markers"]) == (2, 1)


def test_the_selector_learns_which_recovery_suits_which_runs_under_cross_validation():
    # Short runs (1-10 steps) are solved by a retry, long ones (21-30) by continuing.
    summary = {}
    for length in [*range(1, 11), *range(21, 31)]:
        short = length <= 10
        summary[f"r{length}"] = rejected({"retry": episodes(short, 0.5), "continue": episodes(not short, 0.3)},
                                         features={"run_length": length, "visible_tests_passed": False})
    found = report.selector_section(summary, report.FEATURES, 5, 2, 2, 0, 500, random.Random(0))
    assert found["arms"] == ["retry", "continue"] and found["features"] == list(report.FEATURES)
    assert found["solved_share"]["selector"] == 1.0 and found["solved_share"]["per_run_best"] == 1.0
    assert found["solved_share"]["retry"] == 0.5 and found["solved_share"]["continue"] == 0.5
    assert found["selector_minus"]["retry"]["mean_difference"] == 0.5
    assert found["tree"][0] == "if run_length <= 15.5:"
    assert report.selector_section({"r1": summary["r1"]}, report.FEATURES, 5, 2, 2, 0, 100,
                                   random.Random(0))["note"] == "too few runs or recoveries"


def test_a_tree_keeps_a_split_only_when_it_helps():
    rows = [{"features": {"x": x}, "outcomes": {"a": 1.0, "b": 0.0}, "costs": {"a": 1.0, "b": 1.0}} for x in range(10)]
    assert selector.fit(rows, ["a", "b"], ["x"], depth=2, min_leaf=2) == selector.Leaf("a", 10)
    assert selector.choose(selector.Leaf("b", 1), {"x": 3}) == "b"
    tied = [{"features": {}, "outcomes": {"a": 0.5, "b": 0.5}, "costs": {"a": 2.0, "b": 1.0}}]
    assert selector.best_arm(tied, ["a", "b"]) == "b"


def test_the_report_runs_on_the_drivers_state_files(tmp_path, capsys):
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    [run_id] = base_jobs(jobs_dir, trials, [agent_steps(12, tests_pass_at=(4,))],
                         calibration=[(1.0, 0.5, 300.0), (0.0, 0.7, 600.0)])
    common = {"jobs_dir": str(jobs_dir), "trials_root": str(trials), "run_harbor": "/r.sh"}
    harbor = FakeHarbor(trials, branches(lambda k, i: k <= 6 and i % 2 == 0))
    run_to_end(tmp_path / "map.json", {**common, "mode": "rebuild"}, jobs_dir, harbor)
    from taste.recovery_study.map_driver import MapDriver
    oracle = MapDriver.open(tmp_path / "map.json").summary()
    drive(tmp_path / "rec.json", {**common, "repeats": 2, "recoveries": ["retry", "rewind", "continue"],
                                  "rewind_sources": ["oracle", "rules"]},
          jobs_dir, FakeHarbor(trials, recoveries()), oracle=oracle)
    reader = tmp_path / "reader.json"
    reader.write_text(json.dumps({run_id: {"step": 8, "reason": "a wrong diagnosis", "confidence": 0.6}}))
    out = tmp_path / "report.json"
    assert report.main(["--map", str(tmp_path / "map.json"), "--runs", str(tmp_path / "rec.json"), "--reader",
                        str(reader), "--resamples", "200", "--json", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["map"]["statuses"] == {"done": 1} and result["choices"]["reader"]["runs"] == 1
    assert result["choices"]["reader"]["choices"][0]["distance"] == 1
    assert set(result["recoveries"]["arms"]) == {"retry", "rewind_oracle", "rewind_rules", "continue"}
    assert result["selector"]["note"] == "too few runs or recoveries"
    text = capsys.readouterr().out
    assert "## Where failed runs become unwinnable" in text and "## Recoveries, paired by failed run" in text
    assert "| rewind_oracle |" in text
    assert f"| {run_id} | 0: 1/3 [" in text and "## Curves" in text


def test_an_episode_that_solved_then_broke_its_fix_counts_as_solved_at_some_round():
    # Round one passed the hidden tests, the checker rejected it, round two broke it.
    broke = [{"end": "budget", "rounds": 2, "reward": 0.0, "solved": False, "usd": 0.2, "seconds": 100.0,
              "tokens": 1000, "agent_usd": 0.1, "check_usd": 0.1, "changed_paths": [], "flags": [],
              "verdicts": [{"verdict": "not_done", "confidence": 0.99, "solved": True}]}] * 2
    summary = {f"r{i}": rejected({"continue": broke, "retry": episodes(False, 0.5)}) for i in range(4)}
    found = report.recovery_section(summary, None, 500, random.Random(1))
    assert found["arms"]["continue"]["solved_share"] == 0.0
    assert found["arms"]["continue"]["ever_solved_share"] == 1.0 and found["arms"]["retry"]["ever_solved_share"] == 0.0
    [ever] = found["comparisons"]["ever_solved"]
    assert (ever["other"], ever["base"], ever["mean_difference"]) == ("continue", "retry", 1.0)
