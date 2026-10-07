"""A branch's settings reach the worker exactly, and are absent from every wire when off."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace

import pytest

from taste.benchmarks import branch_trial
from taste.benchmarks.harbor_settings import TrialSettings, task_instruction
from taste.brains.azure_execution_policy import AzureExecutionPolicy
from taste.brains.azure_worker_policy import AzureWorkerPolicy
from taste.brains.branch_replay import SCRIPT_SCHEMA, SENTINEL, BranchPolicy
from taste.brains.docker_checkpoint import CheckpointManifest
from taste.brains.worker_admission import EntrypointInputError
from tests.test_azure_worker_policy import assignment
from tests.test_branch_replay import BASE, OUTPUTS, Agent, Live, Recorder

ENDPOINT = "https://test-resource.openai.azure.com/openai/v1/"
ALONE = {"model": "gpt-5.6-luna", "agent": "mini-swe-agent", "services": "none"}


def policy_for(settings, branch=None):
    return settings.policy(ENDPOINT, time.time() + 1500, owner_token="a" * 32, container_id="c" * 64,
                           workdir="/app", branch=branch)


def fork(**changes):
    return BranchPolicy(**{"script": "/var/lib/taste-trials/t/branch-script.json", "script_sha256": "b" * 64,
                           "step": 7, **changes})


def test_a_branch_reaches_the_policy_the_worker_and_the_disclosure_only_when_set():
    settings = TrialSettings.from_options({**ALONE, "branch": "/root/s.json", "branch_step": "7",
                                           "branch_override": "append", "branch_note": "/root/note.txt",
                                           "branch_tolerance": "2"})
    assert (settings.branch_step, settings.branch_tolerance, settings.branch_live) == (7, 2, "on")
    branch = settings.branch_policy("/var/lib/taste-trials/t/branch-script.json", "b" * 64, "a note")
    assert branch == fork(override="append", note="a note", tolerance=2)
    on, off = policy_for(settings, branch), policy_for(TrialSettings.from_options(ALONE))
    # Off, the policy keeps its exact earlier wire form; on, it round-trips.
    assert "branch" not in off.to_dict() and "branch" not in off.worker_resources()
    assert on.to_dict()["branch"] == branch.to_dict() == on.worker_resources()["branch"]
    assert AzureExecutionPolicy.from_dict(json.loads(json.dumps(on.to_dict()))) == on
    with pytest.raises(ValueError, match="fields or schema"):
        AzureExecutionPolicy.from_dict({**off.to_dict(), "branch": None})
    disclosed = settings.disclosure()["branch"]
    assert disclosed == {"script": "/root/s.json", "step": 7, "mode": "rebuild", "live": "on", "tolerance": 2,
                         "override": "append", "note": "/root/note.txt"}
    assert "branch" not in TrialSettings.from_options(ALONE).disclosure()


def test_the_worker_admits_its_branch_from_the_assignment():
    raw = {"worker_agent": "mini-swe-agent", "services": "none", "branch": fork(live=False).to_dict()}
    policy = AzureWorkerPolicy.from_assignment(assignment(**raw))
    assert policy.branch == fork(live=False) and not policy.supervised
    assert AzureWorkerPolicy.from_assignment(assignment(worker_agent="mini-swe-agent")).branch is None
    without_replay = {key: value for key, value in raw["branch"].items() if key != "replay"}
    for damage in ({**raw, "services": "all"}, {"worker_agent": "mini-swe-agent", "branch": raw["branch"]},
                   {**raw, "branch": {**raw["branch"], "step": -1}},
                   {**raw, "branch": {**raw["branch"], "override": ""}},
                   {**raw, "branch": {**raw["branch"], "extra": 1}}, {**raw, "branch": without_replay}):
        with pytest.raises(EntrypointInputError):
            AzureWorkerPolicy.from_assignment(assignment(**damage))


def test_a_branch_policy_is_refused_when_it_cannot_be_served():
    for changes, match in [({"script": "relative.json"}, "absolute"), ({"script_sha256": "x"}, "SHA-256"),
                           ({"mode": "copy"}, "rebuild or restore"), ({"override": "append"}, "needs a note"),
                           ({"note": "text"}, "needs a note"), ({"override": "edit", "note": "x"}, "append or reject"),
                           ({"override": "append", "note": "x", "live": False}, "branch_live=on"),
                           ({"override": "append", "note": "x", "replay": False}, "branch_replay=on"),
                           ({"tolerance": -1}, "nonnegative"), ({"live": "yes"}, "true or false"),
                           ({"replay": "no"}, "true or false")]:
        with pytest.raises(ValueError, match=match):
            fork(**changes)
    supervised = TrialSettings.from_options({"model": "gpt-5.6-luna", "agent": "mini-swe-agent"})
    with pytest.raises(ValueError, match="hosted agent run alone"):
        policy_for(supervised, fork())


@pytest.mark.parametrize("options,match", [
    ({"branch_step": "3"}, "needs branch="),
    ({"task_suffix": "/x", "branch": "/s"}, "fresh trials"),
    ({"branch": "/s", "services": "all"}, "services none"),
    ({"branch": "/s", "branch_mode": "copy"}, "rebuild or restore"),
    ({"branch": "/s", "branch_live": "maybe"}, "on or off"),
    ({"branch": "/s", "branch_override": "append"}, "go together"),
    ({"branch": "/s", "branch_note": "/n"}, "go together"),
    ({"branch": "/s", "branch_override": "reject", "branch_note": "/n", "branch_live": "off"}, "branch_live=on"),
    ({"branch": "/s", "branch_mode": "restore"}, "branch_checkpoint"),
    ({"branch": "/s", "branch_checkpoint": "/c"}, "branch_live=off"),
    ({"branch": "/s", "branch_step": "-1"}, "nonnegative"),
    ({"branch": "/s", "branch_replay": "maybe"}, "on or off"),
    ({"branch": "/s", "branch_replay": "off", "branch_override": "reject", "branch_note": "/n"}, "branch_replay=on"),
    ({"branch_replay": "off"}, "needs branch="),
])
def test_branch_settings_are_admitted_not_guessed(options, match):
    with pytest.raises(ValueError, match=match):
        TrialSettings.from_options({**ALONE, **options})


def test_a_checker_is_given_a_runs_final_files_and_its_own_task():
    """The checker trial: another agent, the run's last step's files, no replay, a task text of its own."""
    settings = TrialSettings.from_options({**ALONE, "agent": "checker", "branch": "/root/s.json",
                                           "branch_step": "9", "branch_replay": "off",
                                           "task_text": "/root/checker-task.txt"})
    branch = settings.branch_policy("/var/lib/taste-trials/t/branch-script.json", "b" * 64)
    assert branch == fork(step=9, replay=False) and branch.to_dict()["replay"] is False
    policy = policy_for(settings, branch)
    assert policy.worker_agent == "checker" and policy.worker_resources()["branch"]["replay"] is False
    disclosed = settings.disclosure()
    assert disclosed["branch"]["replay"] == "off" and disclosed["task_text"] == "/root/checker-task.txt"
    final = TrialSettings.from_options({**ALONE, "final_checkpoint": "/root/final/t"})
    assert final.disclosure()["final_checkpoint"] == "/root/final/t"


def test_a_fresh_trial_can_be_given_another_task_text_or_more_of_it(tmp_path):
    (tmp_path / "suffix.txt").write_text("A previous attempt was rejected by a reviewer: tabs are dropped.\n")
    (tmp_path / "task.txt").write_text("Make the lexer tests pass.\n")
    settings = TrialSettings.from_options({**ALONE, "task_suffix": str(tmp_path / "suffix.txt")})
    assert task_instruction(settings, "Fix the parser.\n") == (
        "Fix the parser.\n\nA previous attempt was rejected by a reviewer: tabs are dropped.\n")
    both = replace(settings, task_text=str(tmp_path / "task.txt"))
    assert task_instruction(both, "Fix the parser.").startswith("Make the lexer tests pass.\n\nA previous")
    assert task_instruction(TrialSettings.from_options(ALONE), "Fix the parser.") == "Fix the parser."
    assert settings.disclosure()["task_suffix"] == str(tmp_path / "suffix.txt")
    (tmp_path / "empty.txt").write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        task_instruction(replace(settings, task_suffix=str(tmp_path / "empty.txt")), "Fix the parser.")


def _script(tmp_path):
    recorder = Recorder(Live(BASE, OUTPUTS))
    Agent().run(recorder)
    path = tmp_path / "script.json"
    path.write_text(json.dumps(recorder.script().to_dict()))
    return path


def test_the_owner_reads_and_checks_the_branch_before_the_trial(tmp_path):
    path = _script(tmp_path)
    (tmp_path / "note.txt").write_text("Submission rejected by a reviewer: tabs are dropped.")
    settings = TrialSettings.from_options({**ALONE, "branch": str(path), "branch_step": "3",
                                           "branch_override": "reject", "branch_note": str(tmp_path / "note.txt")})
    inputs = branch_trial.load_inputs(settings)
    assert inputs.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert inputs.note.startswith("Submission rejected") and len(inputs.script.steps) == 3
    disclosed = branch_trial.disclosure(settings, inputs)
    assert disclosed["steps"] == 3 and disclosed["override"] == "reject" and "note_sha256" in disclosed
    assert branch_trial.load_inputs(TrialSettings.from_options(ALONE)) is None
    with pytest.raises(ValueError, match="not one"):
        branch_trial.load_inputs(replace(settings, branch_step=2))
    with pytest.raises(ValueError, match="between 0 and 3"):
        branch_trial.load_inputs(replace(settings, branch_step=4, branch_override="", branch_note=""))


def test_a_saved_checkpoint_is_refused_for_another_branch(tmp_path):
    path = _script(tmp_path)
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    settings = TrialSettings.from_options({**ALONE, "branch": str(path), "branch_step": "2",
                                           "branch_mode": "restore", "branch_checkpoint": str(directory)})
    inputs = branch_trial.load_inputs(settings)
    (directory / branch_trial.IDENTITY).write_text(json.dumps(
        {"schema": branch_trial.IDENTITY_SCHEMA, "script_sha256": inputs.sha256, "step": 1}))

    class Backend:
        def restore(self, *args, **kwargs):
            raise AssertionError("nothing is restored for another branch")

    with pytest.raises(ValueError, match="another replay script or step"):
        branch_trial.restore_checkpoint(Backend(), settings, inputs)


def test_a_rebuild_with_live_off_saves_its_checkpoint_once_and_only_when_faithful(tmp_path):
    path = _script(tmp_path)
    directory = tmp_path / "checkpoint"
    settings = TrialSettings.from_options({**ALONE, "branch": str(path), "branch_step": "2", "branch_live": "off",
                                           "branch_checkpoint": str(directory)})
    inputs = branch_trial.load_inputs(settings)
    manifest = CheckpointManifest("c" * 64, "sha256:" + "d" * 64, "e" * 64 + ".tar", "e" * 64, 10,
                                  (("/app/x", "added"),), ("/app/x",), (), (), (), 1.0)

    class Backend:
        def __init__(self):
            self.taken = 0

        def checkpoint(self, target):
            self.taken += 1
            (target / manifest.tar).write_bytes(b"tar")
            return manifest

    backend = Backend()
    assert branch_trial.save_checkpoint(backend, settings, inputs, {"faithful": False}, "t")["saved"] is False
    assert backend.taken == 0
    saved = branch_trial.save_checkpoint(backend, settings, inputs, {"faithful": True}, "t")
    assert saved["saved"] and saved["tar_sha256"] == "e" * 64 and backend.taken == 1
    assert CheckpointManifest.from_dict(json.loads((directory / branch_trial.MANIFEST).read_text())) == manifest
    identity = json.loads((directory / branch_trial.IDENTITY).read_text())
    assert (identity["script_sha256"], identity["step"]) == (inputs.sha256, 2)
    again = branch_trial.save_checkpoint(backend, settings, inputs, {"faithful": True}, "t")
    assert again["saved"] is False and backend.taken == 1
    with pytest.raises(ValueError, match="manifest"):
        CheckpointManifest.from_dict({**manifest.to_dict(), "tar": "../elsewhere.tar"})


def test_a_trials_final_files_are_restored_for_the_last_step_of_its_own_script(tmp_path):
    """A checker of a run: the files the run's trial left, put back for its replay script's last step."""
    from taste.brains.docker_checkpoint import RestoreReceipt

    recorder = Recorder(Live(BASE, OUTPUTS))
    Agent().run(recorder)
    value = recorder.script().to_dict()
    value["source"] = {**value["source"], "trial": "f00d", "dropped_steps": 0}
    path = tmp_path / "script.json"
    path.write_text(json.dumps(value))
    directory = tmp_path / "final"
    manifest = CheckpointManifest("c" * 64, "sha256:" + "d" * 64, "e" * 64 + ".tar", "e" * 64, 10,
                                  (("/app/x", "added"),), ("/app/x",), (), (), (), 1.0)

    class Backend:
        restored = None

        def checkpoint(self, target):
            (target / manifest.tar).write_bytes(b"tar")
            return manifest

        def restore(self, saved, target, *, any_container=False):
            self.restored = (saved, target, any_container)
            return RestoreReceipt(saved.tar_sha256, (), (), (), (), 0.1)

    backend = Backend()
    saved = branch_trial.save_final_checkpoint(backend, directory, "f00d")
    assert saved["saved"] and json.loads((directory / branch_trial.IDENTITY).read_text())["final"] is True
    settings = TrialSettings.from_options({**ALONE, "agent": "checker", "branch": str(path), "branch_step": "3",
                                           "branch_replay": "off", "branch_mode": "restore",
                                           "branch_checkpoint": str(directory)})
    inputs = branch_trial.load_inputs(settings)
    assert branch_trial.restore_checkpoint(backend, settings, inputs)["exact"] is True
    assert backend.restored == (manifest, directory, True)
    # Not for another step of that run, nor for another trial's script.
    identity = json.loads((directory / branch_trial.IDENTITY).read_text())
    assert not branch_trial.names_branch(identity, inputs, 2)
    other = branch_trial.BranchInputs(inputs.raw, replace(inputs.script, source={**inputs.script.source,
                                                                                 "trial": "beef"}), "")
    assert not branch_trial.names_branch(identity, other, 3)
    cut = branch_trial.BranchInputs(inputs.raw, replace(inputs.script, source={**inputs.script.source,
                                                                               "dropped_steps": 1}), "")
    assert not branch_trial.names_branch(identity, cut, 3)


def test_the_replay_outcome_is_read_from_the_settled_trajectory():
    nested = {"subagent_trajectories": [{"extra": {}}, {"extra": {"branch": {"faithful": True, "step": 2}}}]}
    assert branch_trial.replay_outcome(nested) == {"faithful": True, "step": 2}
    assert branch_trial.replay_outcome(None) is None


def test_a_script_records_its_schema_and_submission():
    recorder = Recorder(Live(BASE, OUTPUTS))
    Agent().run(recorder)
    value = recorder.script().to_dict()
    assert value["schema"] == SCRIPT_SCHEMA and value["submission_step"] == 3
    assert value["steps"][2]["runs"][0]["output"] == f"{SENTINEL}\n"
