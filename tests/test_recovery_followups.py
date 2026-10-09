"""Follow-ups on a finished study: recoveries at another budget from its checks, and a scan of its map."""

from __future__ import annotations

import json

from taste.recovery_study.driver import RUNNING, STARTED
from taste.recovery_study.map_driver import MapDriver
from tests.recovery_fakes import FakeHarbor, branches, recoveries
from tests.test_recovery_map import cli as map_cli
from tests.test_recovery_map import run_to_end
from tests.test_recovery_map import study as map_study
from tests.test_recovery_runs import cli as runs_cli
from tests.test_recovery_runs import step
from tests.test_recovery_runs import study as runs_study

# Recoveries at another budget ------------------------------------------------------------


def checked(tmp_path, given, jobs_dir, harbor):
    """A first study that has checked the base run and started its recoveries; its saved final state."""
    path = tmp_path / "rec.json"
    _, [saved] = step(path, given, jobs_dir)
    harbor.run([saved])
    _, [check] = step(path, given, jobs_dir)
    harbor.run([check])
    first, _ = step(path, given, jobs_dir)
    return path, first, saved


def test_a_study_at_another_budget_takes_each_runs_initial_check_from_the_first(tmp_path):
    # The same rejected runs, recovered at twice the budget: the base runs are not checked again,
    # so both budgets recover the same runs from the same verdicts and saved final states.
    jobs_dir, trials, given, run_id = runs_study(tmp_path)
    harbor = FakeHarbor(trials, recoveries())
    first_path, first, saved = checked(tmp_path, given, jobs_dir, harbor)
    # The first study's budget is taken and scaled, not worked out again from runs that came later.
    stored = json.loads(first_path.read_text())
    stored["runs"][run_id]["budget"].update(usd=0.5, seconds=500.0)
    first_path.write_text(json.dumps(stored))
    again = {**given, "prefix": "rec2", "budget_runs": 2.0, "checks_from": str(first_path),
             "recoveries": ["retry", "continue", "continue_bare"]}
    second, specs = step(tmp_path / "rec2.json", again, jobs_dir)
    run = second.runs[run_id]
    assert run["status"] == "rejected" and run["check"] == first.runs[run_id]["check"]
    assert {second.jobs[spec.name]["purpose"] for spec in specs} == {"first"}
    assert all(spec.name.startswith("rec2-") for spec in specs)
    assert (run["budget"]["usd"], run["budget"]["seconds"], run["budget"]["base_runs"]) == (1.0, 1000.0, 2.0)
    assert float(specs[0].settings()["spend_cap_usd"]) == 1.0
    # Continue restores the final state the first study saved, whose job is on record here too.
    [ct] = [spec for spec in specs if spec.name.split("-")[-2] == "ct"]
    assert ct.settings()["branch_checkpoint"] == saved.name
    assert second.jobs[saved.name]["taken_from"] == str(first_path)
    assert second.summary()[run_id]["verdict"]["verdict"] == "not_done"


def test_a_run_the_first_study_has_not_checked_waits_for_its_check(tmp_path):
    jobs_dir, trials, given, run_id = runs_study(tmp_path)
    harbor, first_path = FakeHarbor(trials, recoveries()), tmp_path / "rec.json"
    _, [saved] = step(first_path, given, jobs_dir)
    again = {**given, "prefix": "rec2", "checks_from": str(first_path)}
    second, specs = step(tmp_path / "rec2.json", again, jobs_dir)
    assert specs == [] and second.summary()[run_id]["status"] == "checking"
    harbor.run([saved])
    _, [check] = step(first_path, given, jobs_dir)
    harbor.run([check])
    step(first_path, given, jobs_dir)
    second, specs = step(tmp_path / "rec2.json", again, jobs_dir)
    assert specs and second.runs[run_id]["status"] == "rejected"


def test_the_command_line_takes_checks_from_another_state(tmp_path):
    jobs_dir, trials, given, run_id = runs_study(tmp_path)
    first_path, _, _ = checked(tmp_path, given, jobs_dir, FakeHarbor(trials, recoveries()))
    state, spool = tmp_path / "rec2.json", tmp_path / "spool"
    assert runs_cli.main(["--state", str(state), "--base-job", str(jobs_dir / "base"), "--calibration",
                          str(jobs_dir / "calibration"), "--jobs-dir", str(jobs_dir), "--trials", str(trials),
                          "--prefix", "rec2", "--rewind-sources", "rules", "--checks-from", str(first_path),
                          "--spool", str(spool)]) == STARTED
    stored = json.loads(state.read_text())
    assert stored["settings"]["checks_from"] == str(first_path) and stored["runs"][run_id]["status"] == "rejected"
    queued = [path.name.split("-", 1)[1] for path in spool.glob("*.job")]
    assert queued and all(name.startswith("rec2-") for name in queued)


# A scan of the map -------------------------------------------------------------------------


def mapped(tmp_path):
    """A finished map of one monotone run in restore mode, the boundary between steps 11 and 12."""
    jobs_dir, trials, given = map_study(tmp_path, mode="restore")
    harbor = FakeHarbor(trials, branches(lambda k, i: k <= 11 and i % 2 == 0))
    run_to_end(tmp_path / "map.json", given, jobs_dir, harbor)
    return trials, given


def test_a_scan_is_seeded_from_finished_runs_and_probes_only_the_steps_not_yet_probed(tmp_path):
    _, given = mapped(tmp_path)
    source = MapDriver.open(tmp_path / "map.json")
    scan = MapDriver.open(tmp_path / "scan.json", {**given, "prefix": "scan", "scan_every": 3, "seed": "scan"})
    [run_id] = scan.seed(source, 20)
    run = scan.runs[run_id]
    assert run["scan"] == [3, 6, 9, 12, 15, 18] and run["seeded_from"] == str(tmp_path / "map.json")
    # Every probe and checkpoint the map made is the scan's too: steps 12 and 15 were probed.
    assert set(source.jobs) <= set(scan.jobs)
    assert all(scan.jobs[name]["taken_from"] == str(tmp_path / "map.json") for name in source.jobs)
    scan.advance()
    made = sorted(int(spec.settings()["branch_step"]) for spec in scan.new if spec.settings()["branch_live"] == "off")
    assert made == [3, 6, 9, 18] and all(spec.name.startswith("scan-") for spec in scan.new)
    # A seeded state is seeded once: its runs stand.
    assert scan.seed(source, 20) == []


def test_a_seeded_scan_reports_whether_each_curve_is_monotone(tmp_path):
    trials, given = mapped(tmp_path)
    # Branches from step 15 and later win again: the curve is not monotone, and the scan finds it.
    harbor = FakeHarbor(trials, branches(lambda k, i: (k <= 11 or k >= 15) and i % 2 == 0))
    path = tmp_path / "scan.json"
    scan = MapDriver.open(path, {**given, "prefix": "scan", "scan_every": 3, "seed": "scan"})
    scan.seed(MapDriver.open(tmp_path / "map.json"), 20)
    scan.save()
    for _ in range(30):
        driver = MapDriver.open(path)
        driver.refresh()
        specs = driver.advance()
        driver.save()
        if not specs:
            break
        harbor.run(specs)
    [summary] = driver.summary().values()
    assert summary["scanned"] and not summary["monotone"]
    assert summary["earlier_drops"] and summary["decisive_step"] > 15


def test_the_command_line_seeds_a_scan_with_the_maps_settings(tmp_path):
    mapped(tmp_path)
    state, spool = tmp_path / "scan.json", tmp_path / "spool"
    options = ["--state", str(state), "--seed-from", str(tmp_path / "map.json"), "--seed-runs", "20",
               "--scan-every", "3", "--prefix", "scan1", "--seed", "recovery-scan", "--spool", str(spool)]
    assert map_cli.main(options) == STARTED
    stored = json.loads(state.read_text())
    source = json.loads((tmp_path / "map.json").read_text())
    differ = ("prefix", "scan_every", "seed")
    assert {k: v for k, v in stored["settings"].items() if k not in differ} == {
        k: v for k, v in source["settings"].items() if k not in differ}
    assert len(stored["runs"]) == 1 and stored["sources"] == {"base": [], "calibration": []}
    # Run again as a loop does, with the same command: nothing is seeded twice.
    assert map_cli.main(options) in (STARTED, RUNNING) and len(json.loads(state.read_text())["runs"]) == 1
