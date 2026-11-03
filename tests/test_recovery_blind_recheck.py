"""The blind re-check: the judged final states checked again by the same checker, told the original task alone."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from taste.recovery_study import jobs
from taste.recovery_study.jobs import JobSpec
from taste.recovery_study.recheck_driver import RecheckDriver
from tests.recovery_fakes import FakeHarbor, recoveries
from tests.test_recovery_final_saves import base_saved, saving
from tests.test_recovery_messages import graded_recoveries
from tests.test_recovery_messages import study as messages_study
from tests.test_recovery_runs import FEEDBACK, drive

ROUTE = {"SECRETS": "/root/taste-secrets/bedrock.env"}


def told(outcome):
    """Outcomes whose agent records hold the task as the agent got it: with its suffix, as the harness gives it."""
    def wrapped(spec, task, settings, harbor):
        fields = outcome(spec, task, settings, harbor)
        suffix = settings.get(jobs.SETTING["suffix"])
        if suffix and "steps" in fields:
            fields["task_text"] = "Fix the bug.\n\n" + Path(suffix).read_text()
        return fields
    return wrapped


@pytest.fixture
def finished(tmp_path):
    """A finished recovery state whose every check rejected: retry, retry_feedback and continue, each trial saved."""
    jobs_dir, trials, given, run_id = messages_study(tmp_path, recoveries=["retry", "retry_feedback", "continue"],
                                                     save_final=True)
    base_saved(jobs_dir, run_id)
    given = {**given, "prefix": "rec1", "checker_model": "gpt-6-luna", "checker_env": ROUTE, "check_attempts": 3}
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, told(saving(graded_recoveries()))))
    return tmp_path / "rec.json", driver, trials


def judged(driver):
    """(recovery, round's check) for every judged round of the recovery state."""
    return [(arm["recovery"], step["check"]) for run in driver.runs.values() for arm in run["arms"].values()
            for episode in arm["episodes"] for step in episode["rounds"]
            if step.get("agent") and step["check"]["verdict"]]


def kept_file(tmp_path, driver, *, drop=()):
    """The saves this host keeps: every check's own save but those in ``drop``."""
    path = tmp_path / "kept.txt"
    path.write_text("".join(check["own"][0] + "\n" for _, check in judged(driver) if check["own"][0] not in drop))
    return path


def recheck(tmp_path, harbor, **given):
    path = tmp_path / "blind.json"
    for _ in range(30):
        driver = RecheckDriver.open(path, given)
        driver.refresh()
        specs = driver.advance()
        driver.save()
        if not specs:
            return driver
        harbor.run(specs)
    raise AssertionError("the re-checks did not finish")


def specs(driver, purpose):
    return [(name, JobSpec.from_dict(job["spec"])) for name, job in driver.jobs.items()
            if job.get("purpose") == purpose and not job.get("taken_from")]


def text(spec):
    return Path(spec.settings()[jobs.SETTING["task_text"]]).read_text()


def test_every_reasons_round_and_a_sample_of_plain_retry_rounds_is_checked_again_told_the_task_alone(tmp_path,
                                                                                                     finished):
    source, theirs, trials = finished
    rounds = judged(theirs)
    reasons = [check for recovery, check in rounds if recovery in ("retry_feedback", "continue")]
    assert reasons and any(recovery == "retry" for recovery, _ in rounds)
    # What the first checks were told: retry_feedback's task holds the reviewer's rejection, and every check the
    # agent's own account of its work.
    informed = [text(spec) for name, spec in specs(theirs, "check") if "-rf-" in name]
    assert informed and all("A previous attempt was rejected by a reviewer" in item for item in informed)
    driver = recheck(tmp_path, FakeHarbor(trials, recoveries(verdict=lambda spec: "done")), source=str(source),
                     kept_saves=str(kept_file(tmp_path, theirs)), prefix="bl1", control_sample=1)
    items = driver.data["items"]
    assert len(items) == len(reasons) + 1
    assert [item["recovery"] for item in items.values() if item["sample"]] == ["retry"]
    checks = specs(driver, "check")
    assert len(checks) == len(items)
    for _, spec in checks:
        told_now = text(spec)
        assert "<task>\nFix the bug.\n</task>" in told_now
        assert "rejected by a reviewer" not in told_now and FEEDBACK.splitlines()[0] not in told_now
        assert "<final_message>\n(none)\n</final_message>" in told_now
        assert "<submission>\n(empty)\n</submission>" in told_now


def test_the_same_checker_checks_the_same_save_or_a_faithful_rebuild_of_a_pruned_one(tmp_path, finished):
    source, theirs, trials = finished
    gone = next(check["own"][0] for recovery, check in judged(theirs) if recovery == "retry_feedback")
    driver = recheck(tmp_path, FakeHarbor(trials, recoveries(verdict=lambda spec: "done")), source=str(source),
                     kept_saves=str(kept_file(tmp_path, theirs, drop={gone})), prefix="bl1", control_sample=1)
    [model] = {dict(spec.env)["MODEL"] for _, spec in specs(theirs, "check")}
    for _, spec in specs(driver, "check"):
        settings = spec.settings()
        assert dict(spec.env)["MODEL"] == model and dict(spec.env)["SECRETS"] == ROUTE["SECRETS"]
        assert settings["worker_effort"] == theirs.settings["checker_effort"]
        assert settings[jobs.SETTING["mode"]] == "restore"
    restored = {spec.settings()[jobs.SETTING["checkpoint"]] for _, spec in specs(driver, "check")}
    first = {check["own"][0] for _, check in judged(theirs)}
    # Every kept save is restored as it was checked; the pruned one is rebuilt from the record first, in this state.
    [rebuilt] = specs(driver, "final_state")
    assert gone not in restored and rebuilt[0] in restored
    assert restored - {rebuilt[0]} <= first and len(restored) == len(driver.data["items"])
    rows = driver.summary()
    assert sorted(row["state"] for row in rows.values()).count("rebuilt") == 1
    assert all(row["state"] == "same save" for row in rows.values() if row["state"] != "rebuilt")


def test_with_rebuilds_off_a_pruned_save_is_left_out(tmp_path, finished):
    source, theirs, trials = finished
    gone = next(check["own"][0] for recovery, check in judged(theirs) if recovery == "continue")
    driver = recheck(tmp_path, FakeHarbor(trials, recoveries(verdict=lambda spec: "done")), source=str(source),
                     kept_saves=str(kept_file(tmp_path, theirs, drop={gone})), prefix="bl1", control_sample=0,
                     rebuild=False)
    ends = [item["end"] for item in driver.data["items"].values()]
    assert ends.count("save_gone") == 1 and ends.count(None) == len(ends) - 1
    assert not specs(driver, "final_state") and len(specs(driver, "check")) == len(ends) - 1


def test_the_summary_pairs_each_first_verdict_with_the_blind_one(tmp_path, finished):
    source, theirs, trials = finished
    driver = recheck(tmp_path, FakeHarbor(trials, recoveries(verdict=lambda spec: "done")), source=str(source),
                     kept_saves=str(kept_file(tmp_path, theirs)), prefix="bl1", control_sample=1)
    rows = driver.summary()
    assert rows and all((row["informed"], row["blind"], row["solved"]) == ("not_done", "done", False)
                        for row in rows.values())
    differs = {row["recovery"]: row["told_task_differs"] for row in rows.values()}
    assert differs == {"retry_feedback": True, "continue": False, "retry": False}
    assert all(row["check_usd"] == pytest.approx(0.05) for row in rows.values())


def test_the_recovery_state_is_only_read(tmp_path, finished):
    source, theirs, trials = finished
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    driver = recheck(tmp_path, FakeHarbor(trials, recoveries(verdict=lambda spec: "done")), source=str(source),
                     kept_saves=str(kept_file(tmp_path, theirs)), prefix="bl1", control_sample=1)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before == driver.data["source_sha256"]


def test_a_route_restated_at_creation_is_kept_and_reported(tmp_path, finished):
    source, theirs, trials = finished
    other = {"SECRETS": "/root/taste-secrets/other.env"}
    driver = recheck(tmp_path, FakeHarbor(trials, recoveries(verdict=lambda spec: "done")), source=str(source),
                     kept_saves=str(kept_file(tmp_path, theirs)), prefix="bl1", control_sample=0, checker_env=other)
    assert driver.data["restated"] == {"checker_env": {"first": ROUTE, "now": other}}
    assert all(dict(spec.env)["SECRETS"] == other["SECRETS"] for _, spec in specs(driver, "check"))


def test_a_route_the_recovery_state_never_set_is_reported_as_restated(tmp_path, finished):
    import json

    source, theirs, trials = finished
    # States from before the route setting existed hold no checker_env at all.
    data = json.loads(source.read_text())
    del data["settings"]["checker_env"]
    older = tmp_path / "older-rec.json"
    older.write_text(json.dumps(data))
    driver = recheck(tmp_path, FakeHarbor(trials, recoveries(verdict=lambda spec: "done")), source=str(older),
                     kept_saves=str(kept_file(tmp_path, theirs)), prefix="bl1", control_sample=0, checker_env=ROUTE)
    assert driver.data["restated"] == {"checker_env": {"first": None, "now": ROUTE}}
    assert all(dict(spec.env)["SECRETS"] == ROUTE["SECRETS"] for _, spec in specs(driver, "check"))


@pytest.mark.parametrize("given, message", [
    ({"source": None}, "source"),
    ({"kept_saves": "missing"}, "kept_saves"),
    ({"arms": ["continue"], "control": ["continue"]}, "both"),
    ({"arms": ["nonsense"]}, "unknown"),
    ({"control_sample": -1}, "control_sample"),
])
def test_settings_no_re_check_could_be_built_from_are_refused(tmp_path, finished, given, message):
    source, theirs, _ = finished
    settings = {"source": str(source), "kept_saves": str(kept_file(tmp_path, theirs)), "prefix": "bl1", **given}
    with pytest.raises(ValueError, match=message):
        RecheckDriver.open(tmp_path / "blind.json", settings)
    assert not (tmp_path / "blind.json").exists()


def test_the_command_line_runs_the_re_checks_to_their_end(tmp_path, finished, capsys):
    import importlib.util
    import json

    from taste.recovery_study.driver import FINISHED, STARTED

    spec = importlib.util.spec_from_file_location(
        "recovery_recheck_cli", Path(__file__).resolve().parents[1] / "scripts" / "recovery_recheck.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    source, theirs, trials = finished
    path, harbor = tmp_path / "blind.json", FakeHarbor(trials, recoveries(verdict=lambda spec: "done"))
    argv = ["--state", str(path), "--source", str(source), "--kept-saves", str(kept_file(tmp_path, theirs)),
            "--prefix", "bl1", "--control-sample", "1", "--launcher", str(tmp_path / "next.sh"),
            "--json", str(tmp_path / "blind-summary.json")]
    status, done = cli.main(argv), set()
    assert status == STARTED and (tmp_path / "next.sh").exists()
    for _ in range(10):
        driver = RecheckDriver.open(path)
        started = [JobSpec.from_dict(job["spec"]) for name, job in driver.jobs.items()
                   if not job["complete"] and name not in done]
        harbor.run(started)
        done |= {spec.name for spec in started}
        status = cli.main(argv)
        if status == FINISHED:
            break
    assert status == FINISHED
    rows = json.loads((tmp_path / "blind-summary.json").read_text())
    assert rows and all(row["blind"] == "done" for row in rows.values())
    assert "| retry_feedback |" in capsys.readouterr().out


def test_with_rebuilds_off_the_control_is_sampled_from_the_rounds_still_saved(tmp_path, finished):
    source, theirs, trials = finished
    retries = [check["own"][0] for recovery, check in judged(theirs) if recovery == "retry"]
    assert len(retries) >= 2
    driver = recheck(tmp_path, FakeHarbor(trials, recoveries(verdict=lambda spec: "done")), source=str(source),
                     kept_saves=str(kept_file(tmp_path, theirs, drop=set(retries[1:]))), prefix="bl1",
                     control_sample=5, rebuild=False)
    sampled = [item for item in driver.data["items"].values() if item["sample"]]
    assert [item["first_saves"][0] for item in sampled] == retries[:1] and sampled[0]["end"] is None
