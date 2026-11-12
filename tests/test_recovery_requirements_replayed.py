"""The requirement check on states rebuilt from the record: its own rebuilds, or the replay driver's.

Trials made before ``final_change_chars`` existed hold no final change, and their saves went with a deleted disk. The
requirement driver can (1) ask its own rebuilds of a state to read the final change (``final_change_chars``), and
(2) take the replay driver's kept rebuilds (``replayed``): it restores the very checkpoint whose diff and checks the
replay recorded, and reads the diff from there. Both are off by default, and then nothing changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taste.recovery_study import jobs, replay_driver
from taste.recovery_study.jobs import JobSpec
from taste.recovery_study.replay_driver import ReplayDriver
from taste.recovery_study.requirement_driver import RequirementDriver
from tests.recovery_fakes import FakeHarbor, recoveries
from tests.test_recovery_blind_recheck import told
from tests.test_recovery_messages import graded_recoveries
from tests.test_recovery_messages import study as messages_study
from tests.test_recovery_replay import (
    DRY,
    MODE_ONLY,
    TEXT,
    base_saved,
    fake_export,
    own_saves,
    rebuilds,
    replay,
)
from tests.test_recovery_runs import drive

REQUIREMENT = "Fix the bug."


@pytest.fixture
def old(tmp_path):
    """A finished recovery state as the Nov 7-8 run left it: every live trial saved its own final files, none read
    its final change."""
    jobs_dir, trials, given, run_id = messages_study(tmp_path, recoveries=["retry_feedback", "continue"],
                                                     save_final=True)
    base_saved(jobs_dir, run_id)
    drive(tmp_path / "rec.json", {**given, "prefix": "rec1", "check_attempts": 3}, jobs_dir,
          FakeHarbor(trials, told(own_saves(graded_recoveries()))))
    return tmp_path / "rec.json", trials, run_id


def reading(outcome, diff=MODE_ONLY + TEXT):
    """Outcomes whose rebuilds with live off read their final change when the settings ask (as harbor_agent does)."""
    def wrapped(spec, task, settings, harbor):
        fields = outcome(spec, task, settings, harbor)
        if settings.get("final_change_chars") and settings.get("agent") != "checker":
            fields["change"] = {"read": True, "diff": diff, "diff_chars": len(diff), "cut": False, "complete": True,
                                "paths": ["M\tbin/run", "M\tsrc/app.py"], "head_moved": False,
                                "cap_chars": int(settings["final_change_chars"])}
        return fields
    return wrapped


def files(tmp_path, run_id, kept=()):
    (tmp_path / "requirements.json").write_text(json.dumps({run_id: REQUIREMENT}))
    (tmp_path / "kept.txt").write_text("".join(name + "\n" for name in kept))
    return {"requirements": str(tmp_path / "requirements.json"), "kept_saves": str(tmp_path / "kept.txt")}


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


def settings_of(driver, purpose):
    return {name: JobSpec.from_dict(job["spec"]).settings() for name, job in driver.jobs.items()
            if job.get("purpose") == purpose and not job.get("taken_from")}


def test_by_default_a_rebuild_reads_no_change_and_the_summary_has_none(tmp_path, old):
    source, trials, run_id = old
    driver = check(tmp_path, FakeHarbor(trials, reading(recoveries())),
                   source=str(source), prefix="req1", **files(tmp_path, run_id))
    rebuilt = settings_of(driver, "final_state")
    assert len(rebuilt) == 4 and all("final_change_chars" not in settings for settings in rebuilt.values())
    rows = driver.summary()
    assert {row["state"] for row in rows.values()} == {"rebuilt"}
    assert all(row["final_change"] is None and row["final_change_from"] is None for row in rows.values())


def test_with_final_change_chars_its_own_rebuilds_read_the_change_and_the_summary_reads_it_there(tmp_path, old):
    source, trials, run_id = old
    driver = check(tmp_path, FakeHarbor(trials, reading(recoveries())),
                   source=str(source), prefix="req1", final_change_chars=60000, **files(tmp_path, run_id))
    rebuilt = settings_of(driver, "final_state")
    assert all(settings["final_change_chars"] == "60000" and settings["branch_live"] == "off"
               for settings in rebuilt.values())
    # The checker's own trials read nothing: only the rebuild of the state does.
    assert all("final_change_chars" not in settings for settings in settings_of(driver, "check").values())
    for row in driver.summary().values():
        assert row["final_change_from"] == "rebuild" and row["state"] == "rebuilt"
        assert row["final_change"] == {"diff_chars": len(TEXT), "cut": False, "head_moved": False}


def replayed_state(tmp_path, source, trials, monkeypatch, **harbor):
    monkeypatch.setattr(replay_driver, "export_record", fake_export())
    given = {"source": str(source), "env": {"SECRETS": DRY}, "prefix": "rpl", "records_root": str(trials)}
    driver = replay(tmp_path / "replay", FakeHarbor(trials, rebuilds(source, **harbor)), **given)
    return tmp_path / "replay" / "replay.json", driver


def test_with_replayed_each_state_is_the_replays_kept_save_and_its_diff(tmp_path, old, monkeypatch):
    source, trials, run_id = old
    (tmp_path / "replay").mkdir()
    lost_one = {}

    def faithful(spec, number):
        lost_one.setdefault("token", Path(spec.settings()["branch"]).stem)
        return Path(spec.settings()["branch"]).stem != lost_one["token"]
    path, replays = replayed_state(tmp_path, source, trials, monkeypatch, faithful=faithful)
    rows = replays.summary()
    kept = {item_id: row["checkpoint"] for item_id, row in rows.items() if row["kept"]}
    lost = [item_id for item_id, row in rows.items() if not row["kept"]]
    assert len(kept) == 3 and len(lost) == 1
    driver = check(tmp_path, FakeHarbor(trials, recoveries()), source=str(source), prefix="req1",
                   replayed=str(path), **files(tmp_path, run_id, kept=kept.values()))
    assert settings_of(driver, "final_state") == {}  # nothing rebuilt again
    checks = {driver.jobs[name]["requirement"]: settings for name, settings in settings_of(driver, "check").items()}
    assert sorted(checks) == sorted(kept)
    for item_id, settings in checks.items():
        assert settings[jobs.SETTING["mode"]] == "restore" and settings[jobs.SETTING["checkpoint"]] == kept[item_id]
        assert settings[jobs.SETTING["step"]] == str(replays.items[item_id]["step"])
    summary = driver.summary()
    for item_id in kept:
        assert summary[item_id]["state"] == "replayed" and summary[item_id]["final_change_from"] == "replay"
        assert summary[item_id]["final_change"] == {"diff_chars": len(TEXT), "cut": False, "head_moved": False}
    # A state the replay could not keep is not rebuilt again here: its checks would not be the replay's.
    [gone] = lost
    assert summary[gone]["end"] == "replay_lost" and summary[gone]["verdict"] is None
    # The checker is told the replay's list of changed files, as it is told a save's.
    told_now = Path(checks[sorted(kept)[0]][jobs.SETTING["task_text"]]).read_text()
    assert "<changed_paths>\n/app/src/app.py\n/root/.cache/pip/http/ab\n</changed_paths>" in told_now


def test_a_replay_save_this_host_no_longer_keeps_is_rebuilt_as_before(tmp_path, old, monkeypatch):
    source, trials, run_id = old
    (tmp_path / "replay").mkdir()
    path, _ = replayed_state(tmp_path, source, trials, monkeypatch)
    driver = check(tmp_path, FakeHarbor(trials, recoveries()), source=str(source), prefix="req1",
                   replayed=str(path), **files(tmp_path, run_id, kept=()))
    assert len(settings_of(driver, "final_state")) == 4
    assert {row["state"] for row in driver.summary().values()} == {"rebuilt"}


def test_a_replay_of_another_recovery_state_is_refused(tmp_path, old, monkeypatch):
    source, trials, run_id = old
    (tmp_path / "replay").mkdir()
    path, _ = replayed_state(tmp_path, source, trials, monkeypatch)
    other = tmp_path / "other.json"
    other.write_text(source.read_text().replace('"prefix": "rec1"', '"prefix": "rec1" '))
    with pytest.raises(ValueError, match="another recovery state"):
        RequirementDriver.open(tmp_path / "req.json", {"source": str(other), "prefix": "req1",
                                                       "replayed": str(path), **files(tmp_path, run_id)})
    with pytest.raises(ValueError, match="replayed"):
        RequirementDriver.open(tmp_path / "req.json", {"source": str(source), "prefix": "req1",
                                                       "replayed": str(tmp_path / "none.json"),
                                                       **files(tmp_path, run_id)})


def test_the_replay_keeps_the_state_the_requirement_check_restores(tmp_path, old, monkeypatch):
    """The replay's item ids are the requirement driver's: every check maps to one replayed item."""
    source, trials, run_id = old
    (tmp_path / "replay").mkdir()
    path, _ = replayed_state(tmp_path, source, trials, monkeypatch)
    driver = RequirementDriver.open(tmp_path / "req.json", {"source": str(source), "prefix": "req1",
                                                            **files(tmp_path, run_id)})
    assert sorted(driver.items) == sorted(ReplayDriver(path, json.loads(path.read_text())).items)
