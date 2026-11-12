"""Each episode's final code rebuilt from its record with no model call (taste.recovery_study.replay_driver).

The recovery state here is the study's own fake (``tests.recovery_fakes``), run to its end; the replay script a
record exports is made up from the trial's settled record (``fake_export``), and the rebuild trials are written by
the fake Harbor. No Docker, no model, no network.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from taste.recovery_study import jobs, records, replay_driver
from taste.recovery_study.driver import FINISHED, RUNNING, STARTED
from taste.recovery_study.jobs import JobSpec
from taste.recovery_study.replay_driver import ReplayDriver
from tests.recovery_fakes import FakeHarbor
from tests.test_recovery_blind_recheck import told
from tests.test_recovery_messages import graded_recoveries
from tests.test_recovery_messages import study as messages_study
from tests.test_recovery_runs import drive

DRY = "/root/taste-secrets-dry/dry-bedrock.env"
REAL = "/root/taste-secrets/bedrock.env"
WORKDIR = "/app"
SAVED = ["/app/src/app.py", "/root/.cache/pip/http/ab"]
MODE_ONLY = "diff --git a/bin/run b/bin/run\nold mode 100644\nnew mode 100755\n"
TEXT = "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n"


def own_saves(outcome):
    """Live trials save their own final files, the save's manifest naming paths as Docker does (absolute)."""
    def wrapped(spec, task, settings, harbor):
        fields = outcome(spec, task, settings, harbor)
        if settings.get("final_checkpoint") and settings.get("agent") != "checker":
            fields["final"] = {"saved": True, "partial": False, "changed_paths": list(SAVED)}
        return fields
    return wrapped


def base_saved(jobs_dir, run_id):
    path = jobs_dir / "base" / run_id / "result.json"
    result = json.loads(path.read_text())
    result["config"]["agent"]["kwargs"]["final_checkpoint"] = "base-final"
    result["agent_result"]["metadata"]["taste"]["final_checkpoint"] = {"saved": True, "partial": False,
                                                                       "changed_paths": list(SAVED)}
    path.write_text(json.dumps(result))


@pytest.fixture
def finished(tmp_path):
    """A finished recovery state: plain retry, start over and keep going, two episodes each; every live trial saved
    its own final files; the study's model route is a real one."""
    jobs_dir, trials, given, run_id = messages_study(tmp_path, recoveries=["retry", "retry_feedback", "continue"],
                                                     save_final=True)
    base_saved(jobs_dir, run_id)
    given = {**given, "prefix": "rec1", "env": {"SECRETS": REAL}}
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir,
                      FakeHarbor(trials, told(own_saves(graded_recoveries(solved_from_round=2)))))
    return tmp_path / "rec.json", driver, trials, run_id


def fake_export(dropped=0):
    """A replay script made up from a record: one step per step of its settled record (``dropped`` fewer, and said
    so), every command's exact text known, the task's folder /app."""
    def export(record):
        nested = json.loads((Path(record) / "controller" / "trajectory.json").read_text())
        steps = len(records.agent_view(nested)["steps"])
        script = {"schema": "taste.branch/ReplayScript/1", "task": "Fix the bug.",
                  "source": {"trial": Path(record).name, "workdir": WORKDIR, "dropped_steps": dropped},
                  "steps": [{"runs": [{"command": "true", "executed": "true"}]} for _ in range(steps - dropped)]}
        raw = (json.dumps(script, sort_keys=True) + "\n").encode()
        return raw, script
    return export


@pytest.fixture(autouse=True)
def exported(monkeypatch):
    monkeypatch.setattr(replay_driver, "export_record", fake_export())


def rebuilds(source, *, faithful=lambda spec, n: True, reward=None, changed=lambda spec: list(SAVED),
             diff=MODE_ONLY + TEXT, cost=0.0):
    """Rebuild outcomes: the grade the record holds (or ``reward(spec)``), the checkpoint the job names with its
    changed paths, the final change the settings ask for."""
    theirs = json.loads(Path(source).read_text())
    graded = {summary["token"]: summary["reward"] for job in theirs["jobs"].values()
              for summary in job["results"] or ()}
    graded.update({run["base"]["token"]: run["base"]["reward"] for run in theirs["runs"].values()})
    tried = {}

    def outcome(spec, task, settings, harbor):
        assert settings["branch_live"] == "off" and settings["branch_mode"] == "rebuild"
        token = Path(settings["branch"]).stem
        tried[token] = tried.get(token, 0) + 1
        good = faithful(spec, tried[token])
        fields = {"reward": graded[token] if reward is None else reward(spec), "cost": cost, "seconds": 40.0,
                  "branch": {"faithful": good, "prefix_seconds": 35.0,
                             **({"checkpoint": settings["branch_checkpoint"], "changed_paths": changed(spec)}
                                if good and settings.get("branch_checkpoint") else {})}}
        if settings.get("final_change_chars"):
            fields["change"] = {"read": True, "diff": diff, "diff_chars": len(diff), "cut": False, "complete": True,
                                "paths": ["M\tbin/run", "M\tsrc/app.py"], "base": "a" * 40, "head": "a" * 40,
                                "head_moved": False, "cap_chars": int(settings["final_change_chars"])}
        return fields
    return outcome


def given_for(source, trials, **given):
    return {"source": str(source), "env": {"SECRETS": DRY}, "prefix": "rpl", "records_root": str(trials), **given}


def replay(tmp_path, harbor, *, limit=20, **given):
    path = tmp_path / "replay.json"
    for _ in range(limit):
        driver = ReplayDriver.open(path, given)
        driver.refresh()
        specs = driver.advance()
        driver.save()
        if not specs:
            return driver
        harbor.run(specs)
    raise AssertionError("the replays did not finish")


def specs_of(driver):
    return [JobSpec.from_dict(job["spec"]) for job in driver.jobs.values()]


def last_trials(theirs, run_id, arms=("continue", "retry_feedback")):
    """{item id: the summary of the trial that ended the episode}."""
    found = {}
    for name in arms:
        for index, episode in enumerate(theirs.runs[run_id]["arms"][name]["episodes"]):
            found[f"{run_id}:{name}:e{index + 1}"] = theirs.trial(episode["rounds"][-1]["agent"])
    return found


def test_each_ended_episodes_last_trial_is_rebuilt_at_its_last_step_with_live_off(tmp_path, finished):
    source, theirs, trials, run_id = finished
    driver = ReplayDriver.open(tmp_path / "replay.json", given_for(source, trials))
    specs = driver.advance()
    ended = last_trials(theirs, run_id)
    assert sorted(driver.items) == sorted(ended) and len(specs) == 4
    for spec in specs:
        item_id = driver.jobs[spec.name]["item"]
        trial, settings = ended[item_id], spec.settings()
        script = Path(settings[jobs.SETTING["script"]])
        assert script == driver.work_dir() / "scripts" / f"{trial['token']}.json" and script.is_file()
        assert settings[jobs.SETTING["step"]] == str(trial["steps"])
        assert (settings["branch_mode"], settings["branch_live"]) == ("rebuild", "off")
        assert settings["branch_checkpoint"] == spec.name and settings["final_change_chars"] == "60000"
        # The recorded trial's own settings, without its caps, its suffix, its override and its own save.
        assert settings["agent"] == "mini-swe-agent" and settings["services"] == "none"
        for gone in ("spend_cap_usd", "agent_timeout_sec", "task_suffix", "branch_override", "branch_note",
                     "final_checkpoint"):
            assert gone not in settings, gone
        env = dict(spec.env)
        assert env["SECRETS"] == DRY and env["TASTE_TRIALS_ROOT"] == theirs.settings["trials_root"]
        assert env["MODEL"] == "azure/gpt-6-luna" and spec.attempts == 1 and spec.concurrent == 1
        [prepare] = spec.prepare
        assert str(Path(trials) / trial["token"]) in prepare[-1] and "replay_export" in prepare[-1]


def test_a_faithful_rebuild_that_passes_the_checks_keeps_its_episode_on_the_first_try(tmp_path, finished):
    source, theirs, trials, run_id = finished
    harbor = FakeHarbor(trials, rebuilds(source))
    driver = replay(tmp_path, harbor, **given_for(source, trials))
    rows = driver.summary()
    assert len(rows) == 4 and all(row["end"] == "kept" and row["tries"] == 1 for row in rows.values())
    for item_id, row in rows.items():
        recorded = last_trials(theirs, run_id)[item_id]
        assert row["grade"] == {"recorded": recorded["reward"], "rebuilt": recorded["reward"], "equal": True}
        assert row["files"]["judged"] and row["files"]["equal"] and row["recorded"]["changed_from"] == "own save"
        # The diff without the mode-only file; the save the check will restore.
        assert row["change"]["diff"] == TEXT and row["change"]["mode_only_paths"] == ["bin/run"]
        assert row["change"]["paths"] == ["M\tsrc/app.py"] and row["checkpoint"] == driver.items[item_id]["tries"][0]
        assert row["changed_paths"] == SAVED and row["cost_usd"] == 0.0
    # Keep going's later rounds passed the hidden tests: the rebuilt grade is the recorded one, whatever it was.
    assert {row["grade"]["recorded"] for row in rows.values()} == {1.0}
    assert driver.model_cost() == 0.0


def test_an_unfaithful_rebuild_is_tried_again_up_to_the_tries_then_its_episode_is_lost(tmp_path, finished):
    source, theirs, trials, run_id = finished
    ended = last_trials(theirs, run_id)
    never = ended[f"{run_id}:continue:e1"]["token"]
    late = ended[f"{run_id}:retry_feedback:e2"]["token"]

    def faithful(spec, number):
        token = Path(spec.settings()["branch"]).stem
        return token not in (never, late) or (token == late and number == 3)
    driver = replay(tmp_path, FakeHarbor(trials, rebuilds(source, faithful=faithful)), **given_for(source, trials))
    rows = driver.summary()
    assert rows[f"{run_id}:continue:e1"]["end"] == "never_faithful" and rows[f"{run_id}:continue:e1"]["tries"] == 3
    assert rows[f"{run_id}:continue:e1"]["try_whys"] == ["unfaithful"] * 3
    assert rows[f"{run_id}:retry_feedback:e2"]["end"] == "kept" and rows[f"{run_id}:retry_feedback:e2"]["tries"] == 3
    assert sum(1 for spec in specs_of(driver) if driver.jobs[spec.name]["item"] == f"{run_id}:continue:e1") == 3


def test_a_faithful_rebuild_graded_otherwise_or_changing_other_files_is_not_kept(tmp_path, finished):
    source, theirs, trials, run_id = finished
    ended = last_trials(theirs, run_id)
    graded_otherwise, other_files = ended[f"{run_id}:continue:e1"]["token"], ended[f"{run_id}:continue:e2"]["token"]

    def token(spec):
        return Path(spec.settings()["branch"]).stem
    harbor = FakeHarbor(trials, rebuilds(
        source, reward=lambda spec: 0.0 if token(spec) == graded_otherwise else 1.0,
        changed=lambda spec: ["/app/src/app.py", "/app/stray.py"] if token(spec) == other_files else list(SAVED)))
    rows = replay(tmp_path, harbor, **given_for(source, trials)).summary()
    assert rows[f"{run_id}:continue:e1"]["end"] == "grade_differs"
    assert rows[f"{run_id}:continue:e1"]["grade"] == {"recorded": 1.0, "rebuilt": 0.0, "equal": False}
    assert rows[f"{run_id}:continue:e2"]["end"] == "files_differ"
    assert rows[f"{run_id}:continue:e2"]["files"]["only_rebuilt"] == ["stray.py"]
    assert all(rows[item]["tries"] == 3 for item in (f"{run_id}:continue:e1", f"{run_id}:continue:e2"))
    assert {row["end"] for item, row in rows.items() if ":retry_feedback:" in item} == {"kept"}


def test_a_rebuild_that_cost_model_dollars_is_never_kept(tmp_path, finished):
    source, _theirs, trials, _ = finished
    rows = replay(tmp_path, FakeHarbor(trials, rebuilds(source, cost=0.01)), **given_for(source, trials)).summary()
    assert {row["end"] for row in rows.values()} == {"model_cost"}


def test_a_rebuilds_save_over_its_size_cap_is_no_state_to_restore_and_its_list_is_not_judged(tmp_path, finished):
    source, _theirs, trials, _ = finished
    plain = rebuilds(source, changed=lambda spec: ["/app/src/app.py"])

    def partial(spec, task, settings, harbor):
        fields = plain(spec, task, settings, harbor)
        fields["branch"]["saved_checkpoint"] = {"saved": True, "partial": True}
        return fields
    rows = replay(tmp_path, FakeHarbor(trials, partial), **given_for(source, trials)).summary()
    for row in rows.values():
        assert row["end"] == "kept" and row["checkpoint"] is None
        assert row["files"] == {"judged": False, "equal": None, "why": "the rebuild's save is partial (over its size cap)"}


def test_base_states_are_rebuilt_too_when_asked(tmp_path, finished):
    source, theirs, trials, run_id = finished
    driver = replay(tmp_path, FakeHarbor(trials, rebuilds(source)), **given_for(source, trials, base=True,
                                                                                    arms=[]))
    [(item_id, row)] = driver.summary().items()
    run = theirs.runs[run_id]
    assert item_id == f"{run_id}:base" and row["base"] and row["end"] == "kept"
    assert row["token"] == run["base"]["token"] and row["step"] == run["steps"]
    assert row["recorded"] == {"reward": 0.0, "changed_from": "own save", "changed_paths": 2}


def test_without_the_trials_own_saves_the_list_is_the_save_its_check_rebuilt(tmp_path):
    jobs_dir, trials, given, _run_id = messages_study(tmp_path, recoveries=["continue"])
    drive(tmp_path / "rec.json", {**given, "prefix": "rec1"}, jobs_dir,
                      FakeHarbor(trials, graded_recoveries()))
    driver = ReplayDriver.open(tmp_path / "replay.json", given_for(tmp_path / "rec.json", trials, arms=["continue"],
                                                                   base=True))
    for item_id, item in driver.items.items():
        if item["base"]:
            # The base run's check rebuilt its state and saved it: that save's manifest is the list.
            assert item["recorded"]["changed_from"] == "check save"
            assert item["recorded"]["changed_paths"] == ["src/app.py"]
        else:
            # A round the budget ended was never checked: nothing lists its files, which are not judged.
            assert item["recorded"]["changed_from"] is None, item_id


def test_a_missing_record_or_a_failed_export_ends_the_item_with_no_job(tmp_path, finished, monkeypatch):
    source, _theirs, trials, _ = finished
    driver = ReplayDriver.open(tmp_path / "a.json", given_for(source, trials, records_root=str(tmp_path / "none")))
    assert driver.advance() == [] and {item["end"] for item in driver.items.values()} == {"no_record"}

    def broken(record):
        raise ValueError("this run was a branch that did not match its record")
    monkeypatch.setattr(replay_driver, "export_record", broken)
    driver = ReplayDriver.open(tmp_path / "b.json", given_for(source, trials))
    assert driver.advance() == []
    assert {(item["end"], item["why"]) for item in driver.items.values()} == {
        ("export_failed", "ValueError: this run was a branch that did not match its record")}


def test_a_record_that_stops_mid_command_is_rebuilt_to_its_last_whole_step_and_flagged(tmp_path, finished,
                                                                                      monkeypatch):
    source, _theirs, trials, _run_id = finished
    monkeypatch.setattr(replay_driver, "export_record", fake_export(dropped=1))
    driver = replay(tmp_path, FakeHarbor(trials, rebuilds(source)), **given_for(source, trials))
    for item_id, row in driver.summary().items():
        assert row["short"] and row["step"] == row["steps"] - 1 and row["dropped_steps"] == 1
        assert row["end"] == "short_record" and row["tries"] == 1 and row["faithful"], item_id


def test_named_episodes_or_a_seeded_sample_spread_over_the_arms_are_the_items(tmp_path, finished):
    source, _theirs, trials, run_id = finished
    named = ReplayDriver.open(tmp_path / "a.json", given_for(source, trials, episodes=[f"{run_id}:continue:e2"]))
    assert list(named.items) == [f"{run_id}:continue:e2"]
    sample = ReplayDriver.open(tmp_path / "b.json", given_for(source, trials, sample=2, sample_seed=7))
    assert sorted(item["recovery"] for item in sample.items.values()) == ["continue", "retry_feedback"]
    again = ReplayDriver.open(tmp_path / "c.json", given_for(source, trials, sample=2, sample_seed=7))
    assert list(again.items) == list(sample.items)
    with pytest.raises(ValueError, match="not in the recovery state"):
        ReplayDriver.open(tmp_path / "d.json", given_for(source, trials, episodes=[f"{run_id}:continue:e9"]))


def test_the_export_command_is_this_codes_and_the_tokens_are_those_of_the_items_taken(tmp_path, finished):
    source, theirs, trials, run_id = finished
    data = json.loads(source.read_text())
    data["settings"]["export_template"] = "PYTHONPATH=/opt/taste/old python3 -m taste.benchmarks.replay_export " \
                                          "--record {record} --out {out}"
    (tmp_path / "old.json").write_text(json.dumps(data))
    driver = ReplayDriver.open(tmp_path / "replay.json", given_for(tmp_path / "old.json", trials, sample=1))
    [spec] = driver.advance()
    [prepare] = spec.prepare
    assert "/opt/taste/old" not in prepare[-1]
    assert f"PYTHONPATH={Path(replay_driver.__file__).resolve().parents[2]}" in prepare[-1]
    [item] = driver.items.values()
    assert driver.tokens() == [item["token"], theirs.runs[run_id]["base"]["token"]]


def test_settings_that_could_bill_a_model_or_build_no_rebuild_are_refused(tmp_path, finished):
    source, _theirs, trials, _ = finished
    for changes, message in (({"env": {}}, "SECRETS"), ({"env": {"SECRETS": REAL}}, "own model route"),
                             ({"tries": 0}, "tries"), ({"arms": ["nonsense"]}, "unknown arms"),
                             ({"arms": [], "base": False}, "nothing to rebuild"),
                             ({"final_change_chars": 5}, "final_change_chars"),
                             ({"only_runs": ["no-such-run"]}, "no-such-run"),
                             ({"episodes": ["not an id"]}, "not an item id"),
                             ({"source": str(tmp_path / "missing.json")}, "source")):
        with pytest.raises(ValueError, match=message):
            ReplayDriver.open(tmp_path / "replay.json", {**given_for(source, trials), **changes})
        assert not (tmp_path / "replay.json").exists()


def test_a_recovery_state_still_running_is_refused_and_a_finished_one_is_only_read(tmp_path, finished):
    source, _theirs, trials, _ = finished
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    driver = replay(tmp_path, FakeHarbor(trials, rebuilds(source)), **given_for(source, trials))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before == driver.data["source_sha256"]
    data = json.loads(source.read_text())
    next(iter(data["runs"].values()))["arms"]["continue"]["episodes"][0]["end"] = None
    (tmp_path / "running.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="still running"):
        ReplayDriver.open(tmp_path / "other.json", given_for(tmp_path / "running.json", trials))


def cli():
    spec = importlib.util.spec_from_file_location(
        "recovery_replay_cli", Path(__file__).resolve().parents[1] / "scripts" / "recovery_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def argv_for(tmp_path, source, trials, *more):
    return ["--state", str(tmp_path / "replay.json"), "--source", str(source), "--env", f"SECRETS={DRY}",
            "--records-root", str(trials), "--prefix", "rpl", *more]


def test_the_dry_run_walks_every_stage_and_starts_and_writes_nothing(tmp_path, finished, capsys):
    source, theirs, trials, run_id = finished
    requirements = tmp_path / "requirements.json"
    requirements.write_text(json.dumps({run_id: "Empty input must parse to an empty list."}))
    status = cli().main(argv_for(tmp_path, source, trials, "--base", "true", "--dry-run", "--requirements",
                                 str(requirements), "--tokens-out", str(tmp_path / "tokens.txt")))
    out = capsys.readouterr().out
    assert status == 0 and not (tmp_path / "replay.json").exists() and not (tmp_path / "replay.d").exists()
    for stage in ("1 restore:", "dropped 0, workdir /app", "3 rebuild, try 1 of 3 at step",
                  "4 checks: recorded grade", "5 requirement: 'Empty input must parse"):
        assert out.count(stage) == 5, stage
    # The base run took six steps, each restart's last trial five.
    assert out.count("2 export: 5 steps") == 4 and out.count("2 export: 6 steps") == 1
    assert "run-harbor.sh" in out and "branch_live=off" in out and "final_change_chars=60000" in out
    assert '"items continue": 2' in out and '"items retry_feedback": 2' in out and '"items base": 1' in out
    tokens = (tmp_path / "tokens.txt").read_text().split()
    items = last_trials(theirs, run_id)
    assert sorted(tokens) == sorted({trial["token"] for trial in items.values()} | {theirs.runs[run_id]["base"]["token"]})
    # Without the records the walk says so, and still prints the command it would run.
    cli().main(argv_for(tmp_path, source, trials, "--dry-run", "--records-root", str(tmp_path / "none")))
    out = capsys.readouterr().out
    assert out.count("missing journal, ledger, trajectory") == 4 and out.count("2 export: no_record") == 4
    assert out.count("run-harbor.sh") == 4
    with pytest.raises(SystemExit):
        cli().main(argv_for(tmp_path, source, trials, "--dry-run", "--spool", str(tmp_path / "spool")))


def test_the_command_line_runs_the_replays_to_their_end_and_prints_the_loss_table(tmp_path, finished, capsys):
    source, _theirs, trials, _ = finished
    harbor = FakeHarbor(trials, rebuilds(source))
    argv = argv_for(tmp_path, source, trials, "--launcher", str(tmp_path / "next.sh"), "--json",
                    str(tmp_path / "summary.json"), "--tokens-out", str(tmp_path / "tokens.txt"))
    module, done = cli(), set()
    status = module.main(argv)
    assert status == STARTED and (tmp_path / "next.sh").exists()
    for _ in range(5):
        driver = ReplayDriver.open(tmp_path / "replay.json")
        started = [spec for spec in specs_of(driver) if spec.name not in done]
        harbor.run(started)
        done |= {spec.name for spec in started}
        status = module.main(argv)
        if status == FINISHED:
            break
    assert status == FINISHED
    out = capsys.readouterr().out
    assert "| azure/gpt-6-luna | continue | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 (0.0%) |" in out
    assert "Model cost of the rebuilds: $0.000000" in out
    rows = json.loads((tmp_path / "summary.json").read_text())
    assert len(rows) == 4 and all(row["kept"] for row in rows.values())
    assert module.main(["--tables", str(tmp_path / "replay.json")]) == 0
    assert "| azure/gpt-6-luna | retry_feedback | 2 | 2 |" in capsys.readouterr().out


def test_the_loop_runs_again_until_every_item_has_ended(tmp_path, finished, capsys):
    source, _theirs, trials, _ = finished
    harbor, queued = FakeHarbor(trials, rebuilds(source)), []

    def sleep(seconds):
        driver = ReplayDriver.open(tmp_path / "replay.json")
        started = [spec for spec in specs_of(driver) if spec.name not in queued]
        harbor.run(started)
        queued.extend(spec.name for spec in started)
    status = cli().main(argv_for(tmp_path, source, trials, "--spool", str(tmp_path / "spool"), "--loop", "1"),
                        sleep=sleep)
    assert status == FINISHED and len(queued) == 4
    assert status != RUNNING and "0 not finished" in capsys.readouterr().out
