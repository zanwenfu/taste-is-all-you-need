"""Each episode's final state checked against one requirement of its task (ISSTA plan, item 13; draft 3, finding 3).

Where the task text says the reviewer was right, the hidden tests cannot say whether keeping going kept the gap it named
or a restart closed it. The requirement driver checks the state that ended each episode again with the same checker,
told the base run's task and the requirement the task-text readers quoted, and records its verdict and evidence. The
checker here is the tests' stand-in (``tests.recovery_fakes``): no model runs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from taste.agents.checker import checker_task
from taste.recovery_study import jobs
from taste.recovery_study.driver import FINISHED, STARTED
from taste.recovery_study.jobs import JobSpec
from taste.recovery_study.requirement_driver import (
    CLOSING,
    RequirementDriver,
    _quoted,
    requirement_task,
    requirements,
)
from tests.recovery_fakes import NOT_DONE, FakeHarbor, recoveries
from tests.test_recovery_blind_recheck import told
from tests.test_recovery_final_change import changing
from tests.test_recovery_final_saves import base_saved, saving
from tests.test_recovery_messages import graded_recoveries
from tests.test_recovery_messages import study as messages_study
from tests.test_recovery_runs import FEEDBACK, drive, step

ROUTE = {"SECRETS": "/root/taste-secrets/bedrock.env"}
REQUIREMENT = "Fix the bug."


@pytest.fixture
def finished(tmp_path):
    """A finished recovery state: plain retry, start over and keep going, every trial's final state saved and kept."""
    jobs_dir, trials, given, run_id = messages_study(tmp_path, recoveries=["retry", "retry_feedback", "continue"],
                                                     save_final=True, keep_final_saves=True, final_change_chars=60000)
    base = base_saved(jobs_dir, run_id)
    given = {**given, "prefix": "rec1", "checker_model": "gpt-6-luna", "checker_env": ROUTE, "check_attempts": 3}
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir,
                      FakeHarbor(trials, told(changing(saving(graded_recoveries())))))
    return tmp_path / "rec.json", driver, trials, run_id, base


def finals(driver, run_id):
    """(arm, episode index, its outcome's final state) of every episode of the run."""
    return [(name, index, outcome["final"]) for name, arm in driver.summary()[run_id]["arms"].items()
            for index, outcome in enumerate(arm["episodes"])]


def kept_file(tmp_path, driver, run_id, *, drop=(), base=None):
    """The saves this host keeps: each episode's last trial's own save (and the base run's), but those in ``drop``."""
    names = [final["save"] for _, _, final in finals(driver, run_id)] + ([base] if base else [])
    path = tmp_path / "kept.txt"
    path.write_text("".join(name + "\n" for name in names if name not in drop))
    return path


def requirements_file(tmp_path, run_id, text=REQUIREMENT):
    path = tmp_path / "requirements.json"
    path.write_text(json.dumps({run_id: text}))
    return path


def check(tmp_path, harbor, **given):
    path = tmp_path / "req.json"
    for _ in range(30):
        driver = RequirementDriver.open(path, given)
        driver.refresh()
        specs = driver.advance()
        driver.save()
        if not specs:
            return driver
        harbor.run(specs)
    raise AssertionError("the requirement checks did not finish")


def specs(driver, purpose):
    return [(name, JobSpec.from_dict(job["spec"])) for name, job in driver.jobs.items()
            if job.get("purpose") == purpose and not job.get("taken_from")]


def text(spec):
    return Path(spec.settings()[jobs.SETTING["task_text"]]).read_text()


def settings_for(tmp_path, source, theirs, run_id, **given):
    return {"source": str(source), "kept_saves": str(kept_file(tmp_path, theirs, run_id)),
            "requirements": str(requirements_file(tmp_path, run_id)), "prefix": "req1", **given}


def test_each_ended_episode_of_the_arms_is_checked_told_the_task_and_the_requirement_alone(tmp_path, finished):
    source, theirs, trials, run_id, _ = finished
    driver = check(tmp_path, FakeHarbor(trials, recoveries()), **settings_for(tmp_path, source, theirs, run_id))
    items = driver.data["items"]
    # Keep going and start over, two episodes each; plain retry is not asked for.
    assert sorted(items) == [f"{run_id}:{arm}:e{index}" for arm in ("continue", "retry_feedback") for index in (1, 2)]
    checks = specs(driver, "check")
    assert len(checks) == len(items)
    for _, spec in checks:
        told_now = text(spec)
        assert "<task>\nFix the bug.\n</task>" in told_now and f"<requirement>\n{REQUIREMENT}\n</requirement>" in told_now
        assert "<final_message>\n(none)\n</final_message>" in told_now
        assert "<submission>\n(empty)\n</submission>" in told_now
        # Not what any agent was told: not the reviewer's reasons, not the rejection.
        assert "rejected by a reviewer" not in told_now and FEEDBACK.splitlines()[0] not in told_now
        assert told_now.endswith("\n" + CLOSING)
        assert spec.settings()["agent"] == "checker" and dict(spec.env)["SECRETS"] == ROUTE["SECRETS"]


def test_each_state_is_the_save_of_the_trial_that_ended_the_episode_or_a_faithful_rebuild(tmp_path, finished):
    source, theirs, trials, run_id, _ = finished
    ended = {f"{run_id}:{arm}:e{index + 1}": final for arm, index, final in finals(theirs, run_id)}
    gone = ended[f"{run_id}:continue:e1"]["save"]
    given = settings_for(tmp_path, source, theirs, run_id)
    given["kept_saves"] = str(kept_file(tmp_path, theirs, run_id, drop={gone}))
    driver = check(tmp_path, FakeHarbor(trials, recoveries()), **given)
    restored = {driver.jobs[name]["requirement"]: spec.settings()[jobs.SETTING["checkpoint"]]
                for name, spec in specs(driver, "check")}
    for item_id, final in ended.items():
        if not item_id.endswith((":continue:e1", ":continue:e2", ":retry_feedback:e1", ":retry_feedback:e2")):
            continue
        if final["save"] == gone:
            # Pruned from this host: rebuilt from the record, checked against it, then restored for the check.
            [rebuilt] = specs(driver, "final_state")
            assert restored[item_id] == rebuilt[0]
        else:
            assert restored[item_id] == final["save"]
    rows = driver.summary()
    assert sorted(row["state"] for row in rows.values()) == ["rebuilt", "same save", "same save", "same save"]


def test_the_summary_holds_the_verdict_and_the_evidence_on_the_requirement(tmp_path, finished):
    source, theirs, trials, run_id, _ = finished
    driver = check(tmp_path, FakeHarbor(trials, recoveries()), **settings_for(tmp_path, source, theirs, run_id))
    rows = driver.summary()
    assert rows and all(row["verdict"] == "not_done" for row in rows.values())
    for item_id, row in rows.items():
        assert row["unmet_requirement"] == NOT_DONE["unmet_requirement"] and row["evidence"] == NOT_DONE["evidence"]
        assert row["requirement"] == REQUIREMENT and row["requirement_in_task"] is True
        assert row["ended"] == "budget" and row["solved"] is False
        # The source's own check of the same state, if it had one: a round the budget ended is never checked there.
        episode = theirs.runs[run_id]["arms"][row["arm"]]["episodes"][row["episode"]]
        first = episode["rounds"][row["round"] - 1]["check"]["verdict"]
        assert row["first_verdict"] == (theirs.trial(first)["checker"]["verdict"] if first else None), item_id
        assert row["final_change"] == {"diff_chars": None, "cut": False, "head_moved": None}
        assert row["check_usd"] == pytest.approx(0.05) and row["state"] == "same save"
    # Another state on the same jobs folder: its own prefix, as every state has.
    done = check(tmp_path / "done", FakeHarbor(trials, recoveries(verdict=lambda spec: "done")),
                 **settings_for(tmp_path, source, theirs, run_id, prefix="req2"))
    assert all(row["verdict"] == "done" and row["unmet_requirement"] is None for row in done.summary().values())


def test_with_check_base_the_rejected_base_state_is_checked_too(tmp_path, finished):
    source, theirs, trials, run_id, base = finished
    given = settings_for(tmp_path, source, theirs, run_id, check_base=True, arms=["continue"])
    given["kept_saves"] = str(kept_file(tmp_path, theirs, run_id, base=base))
    driver = check(tmp_path, FakeHarbor(trials, recoveries()), **given)
    assert sorted(driver.data["items"]) == [f"{run_id}:base", f"{run_id}:continue:e1", f"{run_id}:continue:e2"]
    [spec] = [spec for name, spec in specs(driver, "check") if driver.jobs[name]["requirement"].endswith(":base")]
    assert spec.settings()[jobs.SETTING["checkpoint"]] == base
    row = driver.summary()[f"{run_id}:base"]
    assert (row["base"], row["first_verdict"], row["verdict"], row["state"]) == (True, "not_done", "not_done",
                                                                                 "same save")


def test_the_recovery_state_is_only_read(tmp_path, finished):
    source, theirs, trials, run_id, _ = finished
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    driver = check(tmp_path, FakeHarbor(trials, recoveries()), **settings_for(tmp_path, source, theirs, run_id))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before == driver.data["source_sha256"]


def test_settings_no_check_could_be_built_from_are_refused(tmp_path, finished):
    source, theirs, _trials, run_id, _ = finished
    given = settings_for(tmp_path, source, theirs, run_id)
    (tmp_path / "x").mkdir()
    for changes, message in (({"requirements": str(tmp_path / "missing.json")}, "requirements"),
                             ({"requirements": str(requirements_file(tmp_path / "x", "no-such-run"))},
                              "no-such-run"),
                             ({"arms": [], "check_base": False}, "nothing to check"),
                             ({"arms": ["nonsense"]}, "unknown")):
        with pytest.raises(ValueError, match=message):
            RequirementDriver.open(tmp_path / "req.json", {**given, **changes})
        assert not (tmp_path / "req.json").exists()
    for bad in ({run_id: ""}, [{"run": run_id}], {run_id: "x" * 2001}):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="requirement"):
            requirements(path)


def test_a_recovery_state_still_running_is_refused(tmp_path):
    jobs_dir, trials, given, run_id = messages_study(tmp_path, recoveries=["continue"], save_final=True)
    base_saved(jobs_dir, run_id)
    harbor = FakeHarbor(trials, saving(graded_recoveries()))
    for _ in range(3):
        _, emitted = step(tmp_path / "rec.json", given, jobs_dir)
        harbor.run(emitted)
    (tmp_path / "kept.txt").write_text("")
    with pytest.raises(ValueError, match="still running"):
        RequirementDriver.open(tmp_path / "req.json", {"source": str(tmp_path / "rec.json"),
                                                       "kept_saves": str(tmp_path / "kept.txt"), "prefix": "req1",
                                                       "requirements": str(requirements_file(tmp_path, run_id))})


def test_only_the_checkers_task_text_names_the_requirement_just_before_its_closing_line():
    first = checker_task("Fix the parser.\n- Empty input parses to [].", "", "", changed_paths=["/app/p.py"])
    named = requirement_task(first, "Empty input parses to [].")
    assert named.startswith(first[:-len(CLOSING)]) and named.endswith("\n" + CLOSING)
    assert "<changed_paths>\n/app/p.py\n</changed_paths>" in named
    assert named.count("<requirement>\nEmpty input parses to [].\n</requirement>") == 1
    assert "Check this requirement alone." in named
    with pytest.raises(ValueError, match="closing line"):
        requirement_task("A task text of another form.", "x")
    assert _quoted("Empty input   parses\nto [].", "Fix the parser.\n- Empty input parses to [].")
    assert not _quoted("Empty input raises.", "Fix the parser.\n- Empty input parses to [].")


def test_the_command_line_runs_the_checks_to_their_end(tmp_path, finished, capsys):
    spec = importlib.util.spec_from_file_location(
        "recovery_requirements_cli", Path(__file__).resolve().parents[1] / "scripts" / "recovery_requirements.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    source, theirs, trials, run_id, _ = finished
    path, harbor = tmp_path / "req.json", FakeHarbor(trials, recoveries(verdict=lambda spec: "done"))
    argv = ["--state", str(path), "--source", str(source), "--kept-saves", str(kept_file(tmp_path, theirs, run_id)),
            "--requirements", str(requirements_file(tmp_path, run_id)), "--prefix", "req1",
            "--arms", "continue,retry_feedback", "--launcher", str(tmp_path / "next.sh"),
            "--json", str(tmp_path / "req-summary.json")]
    status, done = cli.main(argv), set()
    assert status == STARTED and (tmp_path / "next.sh").exists()
    for _ in range(10):
        driver = RequirementDriver.open(path)
        started = [JobSpec.from_dict(job["spec"]) for name, job in driver.jobs.items()
                   if not job["complete"] and name not in done]
        harbor.run(started)
        done |= {spec.name for spec in started}
        status = cli.main(argv)
        if status == FINISHED:
            break
    assert status == FINISHED
    rows = json.loads((tmp_path / "req-summary.json").read_text())
    assert len(rows) == 4 and all(row["verdict"] == "done" for row in rows.values())
    assert "| continue | 2 | 2 | 2 | 0 | 0 |" in capsys.readouterr().out
