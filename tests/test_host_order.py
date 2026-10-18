"""Taking back the trials a rebuild stopped at a cut-off directory walk, and reopening the episodes they ended."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from taste.recovery_study import host_order
from taste.recovery_study.recovery_driver import RecoveryDriver
from tests.recovery_fakes import FakeHarbor, recoveries
from tests.test_recovery_runs import step, study

_SPEC = importlib.util.spec_from_file_location(
    "reopen_host_ordered", Path(__file__).resolve().parents[1] / "scripts" / "reopen_host_ordered.py")
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)


def installed(root):
    """A copy of the code as a host installs it: its run-harbor.sh, and the export template that runs it."""
    harbor = root / "infra" / "azure" / "run-harbor.sh"
    harbor.parent.mkdir(parents=True)
    harbor.write_text("#!/bin/sh\n")
    return {"run_harbor": str(harbor),
            "export_template": f"PYTHONPATH={root} python3 -m taste.benchmarks.replay_export --record {{record}} "
                               "--out {out}"}


def walking(spoiling):
    """Recovery outcomes in which the trials ``spoiling(spec)`` names stop in their rebuild, before the agent."""
    base = recoveries()

    def outcome(spec, task, settings, harbor):
        fields = base(spec, task, settings, harbor)
        if spoiling(spec):
            fields["branch"] = {"faithful": False, "unfaithful_at": 3}
            if settings.get("branch_live") == "off":
                fields.update(reward=None)
        return fields
    return outcome


def drive(path, given, jobs_dir, harbor, limit=60):
    for _ in range(limit):
        driver, specs = step(path, given, jobs_dir)
        if not specs:
            return driver
        harbor.run(specs)
    raise AssertionError("the recoveries did not finish")


def test_round_one_jobs_the_walk_spoiled_are_voided_and_their_episodes_start_again_on_the_new_code(tmp_path):
    old, new = installed(tmp_path / "code-old"), installed(tmp_path / "code-new")
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["retry", "continue"], **old)
    fixed = []
    spoiling = lambda spec: "-ct-f" in spec.name and not fixed  # noqa: E731
    path = tmp_path / "rec.json"
    driver = drive(path, given, jobs_dir, FakeHarbor(trials, walking(spoiling)))
    arms = driver.runs[run_id]["arms"]
    assert [episode["end"] for episode in arms["continue"]["episodes"]] == ["agent_failed", "agent_failed"]
    assert "agent_failed" not in [episode["end"] for episode in arms["retry"]["episodes"]]
    spoiled_jobs = list(arms["continue"]["first"])
    assert len(spoiled_jobs) == 2

    driver = RecoveryDriver.open(path)
    done = host_order.reopen(driver, is_spoiled=lambda summary: summary["faithful"] is False)
    assert done["voided"] == spoiled_jobs and done["left"] == []
    assert done["reopened"] == [(run_id, "continue", 0, "first"), (run_id, "continue", 1, "first")]
    assert all(driver.jobs[name]["void"] == host_order.REASON for name in spoiled_jobs)
    changes = host_order.move_code(driver, tmp_path / "code-new")
    assert changes == {name: [old[name], new[name]] for name in ("run_harbor", "export_template")}
    host_order.amend(driver, {"by": "test", "voided": done["voided"], "settings": changes})
    driver.save()
    # Nothing more the second time.
    again = host_order.reopen(RecoveryDriver.open(path), is_spoiled=lambda summary: summary["faithful"] is False)
    assert again == {"voided": [], "reopened": [], "left": []}

    fixed.append(True)
    given = {**given, **new}
    driver, specs = step(path, given, jobs_dir)
    [first] = specs
    assert first.name.endswith("-ct-f3") and first.argv[0] == new["run_harbor"] and first.attempts == 2
    assert [entry["by"] for entry in driver.data["amendments"]] == ["test"]
    harbor = FakeHarbor(trials, walking(spoiling))
    harbor.run(specs)
    driver = drive(path, given, jobs_dir, harbor)
    ends = [episode["end"] for episode in driver.runs[run_id]["arms"]["continue"]["episodes"]]
    assert ends == ["budget", "budget"]


def test_a_check_the_walk_spoiled_is_made_again(tmp_path):
    # The final state of round one's trial could not be brought back, nor checked: the check failed. Voided, the
    # episode's check starts again where it was, with new names.
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["retry_feedback"], check_attempts=1,
                                           **installed(tmp_path / "code"))
    fixed = []
    spoiling = lambda spec: "-rf-e1r1-" in spec.name and not fixed  # noqa: E731
    path = tmp_path / "rec.json"
    driver = drive(path, given, jobs_dir, FakeHarbor(trials, walking(spoiling)))
    episodes = driver.runs[run_id]["arms"]["retry_feedback"]["episodes"]
    assert episodes[0]["end"] == "check_failed"
    check = episodes[0]["rounds"][-1]["check"]
    spoiled_jobs = [*check["checkpoints"], *check["jobs"]]
    assert [name.rsplit("-", 1)[1] for name in spoiled_jobs] == ["s1", "c1"]

    driver = RecoveryDriver.open(path)
    done = host_order.reopen(driver, is_spoiled=lambda summary: summary["faithful"] is False)
    assert done["voided"] == spoiled_jobs and done["reopened"] == [(run_id, "retry_feedback", 0, "check")]
    driver.save()
    fixed.append(True)
    driver, specs = step(path, given, jobs_dir)
    assert [spec.name.rsplit("-", 1)[1] for spec in specs] == ["s2"]
    harbor = FakeHarbor(trials, walking(spoiling))
    harbor.run(specs)
    driver = drive(path, given, jobs_dir, harbor)
    assert driver.runs[run_id]["arms"]["retry_feedback"]["episodes"][0]["end"] == "budget"


def test_jobs_with_a_trial_that_was_not_spoiled_are_kept(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["continue"], **installed(tmp_path / "code"))
    path = tmp_path / "rec.json"
    drive(path, given, jobs_dir, FakeHarbor(trials, walking(lambda spec: "-ct-f" in spec.name)))
    driver = RecoveryDriver.open(path)
    [first, second] = driver.runs[run_id]["arms"]["continue"]["first"]
    # Only the first job's first trial stopped at a walk; the others at a difference in the files.
    walk = driver.jobs[first]["results"][0]["trial"]
    done = host_order.reopen(driver, is_spoiled=lambda summary: summary["trial"] == walk)
    assert done == {"voided": [], "reopened": [], "left": []}
    assert not any(driver.jobs[name].get("void") for name in (first, second))


def test_the_code_is_moved_only_to_an_installed_copy_the_template_runs(tmp_path):
    driver = SimpleNamespace(settings={"run_harbor": "/opt/taste/aaa/infra/azure/run-harbor.sh",
                                       "export_template": "python3 -m taste.benchmarks.replay_export --record {record}"
                                                          " --out {out}"})
    with pytest.raises(ValueError, match="run-harbor"):
        host_order.move_code(driver, tmp_path / "nonexistent")
    with pytest.raises(ValueError, match="export template"):
        host_order.move_code(driver, Path(installed(tmp_path / "code")["run_harbor"]).parents[2])
    assert driver.settings["run_harbor"] == "/opt/taste/aaa/infra/azure/run-harbor.sh"


def test_the_command_line_reports_until_it_is_told_to_apply(tmp_path, capsys, monkeypatch):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["continue"], **installed(tmp_path / "code"))
    path = tmp_path / "rec.json"
    drive(path, given, jobs_dir, FakeHarbor(trials, walking(lambda spec: "-ct-f" in spec.name)))
    monkeypatch.setattr(host_order, "spoiled", lambda summary, root: summary["faithful"] is False)
    before = path.read_text()
    assert cli.main(["--state", str(path)]) == 0
    report = json.loads(capsys.readouterr().out.split("\nvoid ")[0])
    assert report["applied"] is False and report["voided"] == {"first": 2} and report["reopened"] == {"first": 2}
    assert path.read_text() == before
    assert cli.main(["--state", str(path), "--apply"]) == 0
    capsys.readouterr()
    driver = RecoveryDriver.open(path)
    [amendment] = driver.data["amendments"]
    assert amendment["by"] == "reopen_host_ordered" and len(amendment["voided"]) == 2
    assert [episode["end"] for episode in driver.runs[run_id]["arms"]["continue"]["episodes"]] == [None, None]
