"""A rewind arm's rounds back to the start that the harness stopped ungraded, taken back and graded."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from taste.recovery_study import rewind_to_start
from taste.recovery_study.recovery_driver import RecoveryDriver
from tests.recovery_fakes import FakeHarbor
from tests.test_recovery_agentrewind import committing, rewinding, settings_of
from tests.test_recovery_runs import drive, study

_SPEC = importlib.util.spec_from_file_location(
    "reopen_rewind_to_start", Path(__file__).resolve().parents[1] / "scripts" / "reopen_rewind_to_start.py")
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)


def back_to(target, fixed):
    """The rewind arm's first round commits back to ``target`` with no round left; until ``fixed``, the harness
    stops the round that grades that state as one whose agent never started."""
    base = rewinding()

    def outcome(spec, task, settings, harbor):
        if settings.get("branch_live") == "off" and "branch_checkpoint" not in settings and not fixed:
            return {"reward": None, "cost": 0.0, "seconds": 5.0, "exception": "HostedAgentNotStarted",
                    "branch": {"faithful": True}}
        fields = base(spec, task, settings, harbor)
        if fields.get("stopped_by") == "backtrack_commit":
            fields["steps"] = committing(target)
        return fields
    return outcome


def failed_at_the_start(tmp_path, target="llm-1"):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["agentrewind"], repeats=1, min_round_share=0.8)
    fixed = []
    harbor = FakeHarbor(trials, back_to(target, fixed))
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, harbor)
    return driver, (jobs_dir, given, run_id, harbor, fixed)


def test_a_round_back_to_the_start_the_harness_stopped_is_voided_and_graded_on_the_new_code(tmp_path, capsys):
    driver, (jobs_dir, given, run_id, harbor, fixed) = failed_at_the_start(tmp_path)
    stopped = settings_of(driver, "agentrewind")
    assert [s["branch_step"] for s in stopped] == ["0", "0"] and {s["branch_live"] for s in stopped} == {"off"}
    [episode] = driver.runs[run_id]["arms"]["agentrewind"]["episodes"]
    assert episode["end"] == "agent_failed" and len(episode["rounds"]) == 1
    path = tmp_path / "rec.json"
    before = path.read_text()

    # Without --apply it only reports.
    assert cli.main(["--state", str(path)]) == 0
    report = json.loads(capsys.readouterr().out.split("\nvoid ")[0])
    assert report["voided"] == 2 and report["reopened"] == 1 and report["applied"] is False
    assert path.read_text() == before

    fixed.append(True)
    assert cli.main(["--state", str(path), "--apply"]) == 0
    driver = RecoveryDriver.open(path)
    [amendment] = driver.data["amendments"]
    assert amendment["by"] == "reopen_rewind_to_start" and amendment["reopened"] == [[run_id, "agentrewind", 0, 2]]
    assert all(driver.jobs[name].get("void") == rewind_to_start.REASON for name in amendment["voided"])

    driver, _ = drive(path, given, jobs_dir, harbor)
    graded = settings_of(driver, "agentrewind")
    assert len(graded) == 3 and graded[-1]["branch_step"] == "0" and graded[-1]["branch_live"] == "off"
    [episode] = driver.runs[run_id]["arms"]["agentrewind"]["episodes"]
    # The voided jobs stay the round's own, so the new one took the next name.
    assert episode["rounds"][-1]["agent_jobs"][:2] == amendment["voided"]
    assert episode["rounds"][-1]["agent_jobs"][-1].endswith("-a3")
    [restored] = driver.summary()[run_id]["arms"]["agentrewind"]["episodes"]
    assert restored["end"] == "budget" and restored["solved"] is False and restored["reward"] == 0.0
    # Running it again does nothing more.
    assert rewind_to_start.reopen(RecoveryDriver.open(path)) == {"voided": [], "reopened": []}


def test_a_round_with_steps_to_replay_that_never_started_is_left_as_it_ended(tmp_path):
    # Back to the second call, the round has a step to answer from the record: no step taken is a fault of the
    # harness that its retries were for, not this one.
    driver, (_, _, run_id, _, _) = failed_at_the_start(tmp_path, target="llm-2")
    assert [s["branch_step"] for s in settings_of(driver, "agentrewind")] == ["1", "1"]
    assert driver.runs[run_id]["arms"]["agentrewind"]["episodes"][0]["end"] == "agent_failed"
    assert rewind_to_start.reopen(driver) == {"voided": [], "reopened": []}
