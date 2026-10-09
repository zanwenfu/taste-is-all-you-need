"""The map driver, run to its end against synthetic Harbor jobs: probes, checkpoints, resuming."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from taste.recovery_study.driver import FINISHED, FORGIVEN_CRASHES, RUNNING, STARTED
from taste.recovery_study.jobs import JobSpec, run_key
from taste.recovery_study.map_driver import MapDriver
from tests.recovery_fakes import BASE_SETTINGS, FakeHarbor, agent_steps, base_jobs, branches, trial

_SPEC = importlib.util.spec_from_file_location(
    "recovery_map", Path(__file__).resolve().parents[1] / "scripts" / "recovery_map.py")
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)


def study(tmp_path, steps=20, runs=None, **settings):
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    base_jobs(jobs_dir, trials, runs or [agent_steps(steps)],
              calibration=[(1.0, 0.5, 300.0), (0.0, 0.7, 600.0)])
    given = {"jobs_dir": str(jobs_dir), "trials_root": str(trials), "run_harbor": "/opt/taste/run-harbor.sh",
             "mode": "rebuild", **settings}
    return jobs_dir, trials, given


def step(path, given, jobs_dir):
    """One run of the driver, as its command line does it, from the state on disk."""
    driver = MapDriver.open(path, given)
    driver.add_sources([jobs_dir / "base"], [jobs_dir / "calibration"])
    driver.refresh()
    specs = driver.advance()
    driver.save()
    return driver, specs


def run_to_end(path, given, jobs_dir, harbor, limit=30):
    rounds = []
    for _ in range(limit):
        driver, specs = step(path, given, jobs_dir)
        if not specs:
            return driver, rounds
        rounds.append([(spec.settings()["branch_step"], spec.settings()["branch_live"], spec.attempts)
                       for spec in specs])
        harbor.run(specs)
    raise AssertionError("the map did not finish")


def probed(rounds):
    return [[int(step) for step, live, _ in batch if live == "on"] for batch in rounds]


def test_a_monotone_run_is_bisected_then_both_sides_of_the_boundary_get_sixteen(tmp_path):
    jobs_dir, trials, given = study(tmp_path)
    harbor = FakeHarbor(trials, branches(lambda k, i: k <= 11 and i % 2 == 0))
    driver, rounds = run_to_end(tmp_path / "map.json", given, jobs_dir, harbor)
    assert probed(rounds) == [[10], [15], [12], [11], [11, 12]]
    [(run_id, summary)] = driver.summary().items()
    assert summary["status"] == "done" and summary["decisive_step"] == 12 and summary["steps"] == 20
    assert summary["rewind"] == {"highest_v": 11, "latest": 11}
    curve = {point["step"]: (point["successes"], point["trials"]) for point in summary["curve"]}
    # v(0): the task's runs from scratch, calibration and base runs alike.
    assert curve == {0: (1, 3), 10: (4, 8), 11: (8, 16), 12: (0, 16), 15: (0, 8)}
    assert summary["spend"]["trials"] == 48 and summary["spend"]["usd"] == pytest.approx(0.96)
    # Every branch is the base run's settings and the branch's own, nothing else.
    first = JobSpec.from_dict(driver.jobs[harbor.started[0]]["spec"])
    assert first.settings() == {"agent": "mini-swe-agent", "services": "none", "reply_reserve_seconds": "10",
                                "branch": str(tmp_path / "map.d" / "scripts" / f"{summary['base']['token']}.json"),
                                "branch_step": "10", "branch_mode": "rebuild", "branch_live": "on"}
    assert first.argv[:9] == ("/opt/taste/run-harbor.sh", first.name, "/study/tasks", "-i", "task-a", "-k", "8",
                              "-n", "8")
    assert dict(first.env)["MODEL"] == "azure/gpt-6-luna" and first.name == f"map-{run_key(run_id)}-s10-b1"


def test_restore_mode_makes_a_checkpoint_first_and_the_next_ones_while_a_probe_runs(tmp_path):
    jobs_dir, trials, given = study(tmp_path, mode="restore")
    harbor = FakeHarbor(trials, branches(lambda k, i: k <= 11 and i % 2 == 0,
                                         state=lambda k: 1.0 if k == 5 else 0.0))
    driver, rounds = run_to_end(tmp_path / "map.json", given, jobs_dir, harbor)
    made = [[int(step) for step, live, _ in batch if live == "off"] for batch in rounds]
    assert probed(rounds) == [[], [10], [15], [12], [11], [11, 12]]
    assert made == [[10, 5, 15], [], [12, 17], [11, 13], [], []]
    assert all(attempts == 1 for batch in rounds for _, live, attempts in batch if live == "off")
    [summary] = driver.summary().values()
    assert summary["decisive_step"] == 12 and summary["state_rewards"]["5"] == 1.0
    restore = [JobSpec.from_dict(job["spec"]).settings() for job in driver.jobs.values() if job["purpose"] == "probe"]
    assert all(s["branch_mode"] == "restore" and s["branch_checkpoint"].startswith("map-") for s in restore)
    checkpoints = [JobSpec.from_dict(job["spec"]) for job in driver.jobs.values() if job["purpose"] == "checkpoint"]
    assert all(c.settings()["branch_checkpoint"] == c.name and c.settings()["branch_mode"] == "rebuild"
               for c in checkpoints)


def test_a_recovering_curve_is_found_by_the_scan_and_the_earlier_drop_reported(tmp_path):
    def wins(k, i):
        return (k <= 6 or 13 <= k <= 16) and i % 2 == 0

    jobs_dir, trials, given = study(tmp_path, steps=24, scan_every=4, scan_fraction=1.0)
    driver, rounds = run_to_end(tmp_path / "scan.json", given, jobs_dir, FakeHarbor(trials, branches(wins)))
    assert probed(rounds) == [[4, 8, 12, 16, 20], [18], [17], [16, 17]]
    [summary] = driver.summary().values()
    assert summary["decisive_step"] == 17 and summary["earlier_drops"] == [{"after": 4, "by": 8}]
    assert not summary["monotone"]
    # Without the scan, the search takes the curve as monotone and stops at the first drop.
    other = tmp_path / "other"
    jobs_dir, trials, given = study(other, steps=24)
    driver, rounds = run_to_end(other / "map.json", given, jobs_dir, FakeHarbor(trials, branches(wins)))
    assert probed(rounds) == [[12], [6], [9], [7], [6, 7]]
    assert next(iter(driver.summary().values()))["decisive_step"] == 7


def test_a_step_whose_branches_are_unfaithful_is_retried_once_then_censors_the_run(tmp_path):
    jobs_dir, trials, given = study(tmp_path)
    harbor = FakeHarbor(trials, branches(lambda k, i: i % 2 == 0, unfaithful_from=8))
    driver, rounds = run_to_end(tmp_path / "map.json", given, jobs_dir, harbor)
    assert probed(rounds) == [[10], [10], [5], [7], [8], [8]]
    [summary] = driver.summary().values()
    assert summary["status"] == "censored" and summary["rewind"]["latest"] == 7
    assert summary["decisive_step"] is None and summary["spend"]["not_usable"] == 32


def test_the_driver_resumes_from_its_state_and_never_starts_a_job_twice(tmp_path):
    jobs_dir, trials, given = study(tmp_path)
    path = tmp_path / "map.json"
    harbor = FakeHarbor(trials, branches(lambda k, i: k <= 11 and i % 2 == 0))
    _, specs = step(path, given, jobs_dir)
    assert [spec.attempts for spec in specs] == [8]
    # Not started yet, then half written: nothing new is decided.
    assert step(path, given, jobs_dir)[1] == []
    harbor.run(specs, upto=4)
    driver, again = step(path, given, jobs_dir)
    assert again == [] and not driver.jobs[specs[0].name]["complete"]
    harbor.run(specs, already=4)
    driver, following = step(path, given, jobs_dir)
    assert [spec.settings()["branch_step"] for spec in following] == ["15"]
    assert driver.jobs[specs[0].name]["complete"] and len(driver.jobs[specs[0].name]["results"]) == 8
    # A second decision on the same state adds nothing.
    assert driver.advance() == []
    # A job that never ran can be closed; its probe is then started again under a new name.
    driver = MapDriver.open(path, given)
    driver.close(following[0].name)
    replacement = driver.advance()
    assert [spec.name for spec in replacement] == [following[0].name[:-1] + "2"]


def test_the_studys_settings_are_fixed_when_it_starts(tmp_path):
    jobs_dir, _, given = study(tmp_path, k=4)
    path = tmp_path / "map.json"
    step(path, given, jobs_dir)
    assert MapDriver.open(path, {"k": 4}).settings["k"] == 4
    assert MapDriver.open(path, {}).settings["k"] == 4
    with pytest.raises(ValueError, match="other settings for: k"):
        MapDriver.open(path, {"k": 8})
    with pytest.raises(ValueError, match="unknown settings"):
        MapDriver.open(path, {"kk": 8})


def test_runs_the_study_cannot_branch_are_left_out_with_the_reason(tmp_path):
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    base_jobs(jobs_dir, trials, [agent_steps(1)], task="task-short")
    base_jobs(tmp_path / "other", trials, [agent_steps(9)], task="task-old", model="azure/gpt-5.6-luna")
    driver = MapDriver.open(tmp_path / "map.json", {"jobs_dir": str(jobs_dir), "trials_root": str(trials)})
    driver.add_sources([jobs_dir / "base", tmp_path / "other" / "base"])
    reasons = sorted(run["skipped"] for run in driver.runs.values())
    assert reasons == ["1 steps", "base model azure/gpt-5.6-luna is not this study's gpt-6-luna"]
    assert driver.advance() == []


def test_a_run_whose_agent_never_started_is_neither_a_base_run_nor_a_run_from_scratch(tmp_path):
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    base_jobs(jobs_dir, trials, [agent_steps(9), []], calibration=[(1.0, 0.5, 300.0)])
    # Graded 0 on a container the agent never touched: Taste failed to start it.
    trial(jobs_dir, "calibration", "task-a", reward=0.0, trials_root=trials, steps=[], settings=BASE_SETTINGS)
    driver = MapDriver.open(tmp_path / "map.json", {"jobs_dir": str(jobs_dir), "trials_root": str(trials)})
    driver.add_sources([jobs_dir / "base"], [jobs_dir / "calibration"])
    assert sorted(run.get("skipped", "") for run in driver.runs.values()) == ["", "the agent never took a step"]
    # One success from scratch and the one failed base run that started.
    assert driver.v0("task-a") == [1, 2]


def test_the_command_line_writes_a_launcher_and_says_how_the_study_stands(tmp_path, capsys):
    jobs_dir, trials, _ = study(tmp_path)
    state, launcher, summary = tmp_path / "map.json", tmp_path / "next.sh", tmp_path / "summary.json"
    options = ["--state", str(state), "--base-job", str(jobs_dir / "base"), "--calibration",
               str(jobs_dir / "calibration"), "--jobs-dir", str(jobs_dir), "--trials", str(trials),
               "--mode", "rebuild", "--prefix", "pilot", "--launcher", str(launcher), "--json", str(summary)]
    assert cli.main(options) == STARTED
    assert "start pilot-" in launcher.read_text() and "run-harbor.sh" in launcher.read_text()
    first = [line for line in launcher.read_text().splitlines() if line.startswith("start ")]
    # Nothing new while the job runs; the launcher still lists it, to start it if it never started.
    assert cli.main(["--state", str(state), "--launcher", str(launcher)]) == RUNNING
    assert [line for line in launcher.read_text().splitlines() if line.startswith("start ")] == first
    assert len(first) == 1
    harbor = FakeHarbor(trials, branches(lambda k, i: k <= 11 and i % 2 == 0))
    for _ in range(10):
        stored = json.loads(state.read_text())
        harbor.run([JobSpec.from_dict(job["spec"]) for job in stored["jobs"].values() if not job["complete"]
                    and job["spec"]["name"] not in harbor.started])
        status = cli.main(["--state", str(state), "--json", str(summary)])
        if status == FINISHED:
            break
    assert status == FINISHED
    assert next(iter(json.loads(summary.read_text()).values()))["decisive_step"] == 12
    assert "| done |" in capsys.readouterr().out


def test_with_a_spool_the_command_line_queues_its_jobs_and_ends_on_how_the_study_stands(tmp_path, capsys):
    jobs_dir, trials, _ = study(tmp_path)
    spool = tmp_path / "spool"
    assert cli.main(["--state", str(tmp_path / "map.json"), "--base-job", str(jobs_dir / "base"),
                     "--calibration", str(jobs_dir / "calibration"), "--jobs-dir", str(jobs_dir),
                     "--trials", str(trials), "--mode", "rebuild", "--prefix", "pilot", "--spool", str(spool)]) == STARTED
    assert len(list(spool.glob("*.job"))) == 1
    # No commands to start by hand: the last line says how the study stands, and where its jobs wait.
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last == f"1 new jobs, 1 not finished; queued in {spool} for the host's scheduler."


def test_a_job_whose_harbor_stopped_without_closing_it_is_taken_as_finished(tmp_path):
    # Harbor can stop with an error before writing a job's result (a check it could not set up):
    # once its unit has stopped, the job is finished with what it has, and is marked crashed.
    jobs_dir, _, given = study(tmp_path)
    driver, specs = step(tmp_path / "map.json", given, jobs_dir)
    [spec] = specs
    (jobs_dir / spec.name).mkdir(parents=True)
    assert driver.refresh(unit_active=lambda name: True) == 0 and not driver.jobs[spec.name]["complete"]
    assert driver.refresh(unit_active=lambda name: False) == 1
    assert driver.jobs[spec.name]["complete"] and driver.jobs[spec.name]["crashed"]
    assert driver.jobs[spec.name]["results"] == []


def crashed(path, given, jobs_dir, name):
    """Harbor ended job ``name`` before any trial (its network check failed); the driver's next run."""
    (jobs_dir / name).mkdir(parents=True)
    driver = MapDriver.open(path, given)
    driver.add_sources([jobs_dir / "base"], [jobs_dir / "calibration"])
    driver.refresh(unit_active=lambda job: False)
    specs = driver.advance()
    driver.save()
    return specs


def test_a_job_harbor_ended_before_any_trial_is_not_an_attempt_until_it_keeps_happening(tmp_path):
    # A busy Docker fails Harbor's network check and the job ends before its trials: the probe is
    # started again in full. Past a few such crashes they count, so a job that can never start ends.
    jobs_dir, _, given = study(tmp_path)
    path = tmp_path / "map.json"
    _, [spec] = step(path, given, jobs_dir)
    started = [spec]
    for _ in range(FORGIVEN_CRASHES + 1):
        [spec] = crashed(path, given, jobs_dir, started[-1].name)
        started.append(spec)
    assert [(s.settings()["branch_step"], s.attempts) for s in started] == [("10", 8)] * (FORGIVEN_CRASHES + 2)
    # The probe's attempts are twice its eight: one counted crash leaves room for one more job, then none.
    assert not any(s.settings()["branch_step"] == "10" for s in crashed(path, given, jobs_dir, started[-1].name))


def test_a_checkpoint_whose_save_failed_is_not_restored_from_and_is_made_again(tmp_path):
    # The rebuild was faithful but saving its files failed: no checkpoint of that name exists.
    jobs_dir, trials, given = study(tmp_path, mode="restore")
    made, failed = branches(lambda k, i: True), []

    def outcome(spec, task, settings, harbor):
        found = made(spec, task, settings, harbor)
        if settings.get("branch_live") == "off" and not failed:
            failed.append(spec.name)
            found["branch"] = {"faithful": True}
        return found

    harbor, path = FakeHarbor(trials, outcome), tmp_path / "map.json"
    _, specs = step(path, given, jobs_dir)
    harbor.run(specs)
    driver, specs = step(path, given, jobs_dir)
    assert failed[0].endswith("-s10-c1")
    assert [(s.name.rsplit("-", 2)[1:], s.settings()["branch_live"]) for s in specs] == [(["s10", "c2"], "off")]
    assert not any(job.get("checkpoint") == failed[0] for job in driver.jobs.values() if job["purpose"] == "probe")


def test_a_voided_checkpoint_and_the_probe_it_spoiled_are_made_again(tmp_path):
    # Found after the fact: a checkpoint that did not restore exactly, so its branches failed before
    # the agent ran. Both are voided with the reason; neither is an attempt, and both are made again.
    jobs_dir, trials, given = study(tmp_path, mode="restore")
    harbor, path = FakeHarbor(trials, branches(lambda k, i: k <= 11 and i % 2 == 0)), tmp_path / "map.json"
    _, specs = step(path, given, jobs_dir)
    harbor.run(specs)
    driver, [probe] = step(path, given, jobs_dir)
    harbor.run([probe])
    spoiled = driver.jobs[probe.name]["checkpoint"]
    driver = MapDriver.open(path, given)
    driver.refresh()
    for name in (spoiled, probe.name):
        driver.void(name, "the checkpoint did not restore exactly")
    driver.save()
    driver, specs = step(path, given, jobs_dir)
    assert [(s.name.rsplit("-", 2)[1:], s.settings()["branch_live"]) for s in specs] == [(["s10", "c2"], "off")]
    harbor.run(specs)
    driver, specs = step(path, given, jobs_dir)
    [again] = [s for s in specs if s.settings()["branch_step"] == "10"]
    assert again.attempts == 8 and again.settings()["branch_checkpoint"].endswith("-s10-c2")
    assert driver.jobs[spoiled]["void"] == "the checkpoint did not restore exactly"
    assert driver.results([probe.name]) == []


def test_a_step_whose_checkpoint_came_out_partial_is_branched_by_rebuilding(tmp_path):
    # Over the size cap, a checkpoint lacks paths a restore needs, and every restore from it would
    # fail before the agent ran: the step's branches rebuild their prefix themselves instead.
    jobs_dir, trials, given = study(tmp_path, mode="restore")
    made = branches(lambda k, i: True)

    def outcome(spec, task, settings, harbor):
        found = made(spec, task, settings, harbor)
        if settings.get("branch_live") == "off" and settings["branch_step"] == "10":
            found["branch"]["saved_checkpoint"] = {"saved": True, "partial": True}
        return found

    harbor, path = FakeHarbor(trials, outcome), tmp_path / "map.json"
    _, specs = step(path, given, jobs_dir)
    harbor.run(specs)
    driver, specs = step(path, given, jobs_dir)
    [probe] = [s for s in specs if s.settings()["branch_step"] == "10"]
    settings = probe.settings()
    assert (settings["branch_mode"], settings["branch_live"], probe.attempts) == ("rebuild", "on", 8)
    assert "branch_checkpoint" not in settings
    assert "partial" in driver.jobs[probe.name]["rebuilt_because"]
    # No other checkpoint of the step is made, and the map lends none for it to a rewind.
    assert not any(s.settings()["branch_live"] == "off" for s in specs if s.settings()["branch_step"] == "10")
    assert "10" not in next(iter(driver.summary().values()))["checkpoints"]
