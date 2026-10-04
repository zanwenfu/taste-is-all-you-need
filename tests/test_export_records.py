"""A study's export: its records without credentials, checksummed, and enough to recompute the report."""

import importlib.util
import json
from pathlib import Path

import pytest

from tests.test_go_no_go_report import trial

_ROOT = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter, study = _load("export_records"), _load("study_report")


@pytest.fixture
def records(tmp_path):
    jobs, trials = tmp_path / "jobs", tmp_path / "trials"
    for index in range(4):
        for arm, reward in (("alone", 0.0), ("taste", 1.0 if index < 3 else 0.0)):
            trial(jobs, trials, arm, f"task-{index}", 1, reward=reward,
                  subs=[("mini-swe-agent", 0.01, "delivered", "Submitted", "")])
            name = f"task-{index}__{arm}1"
            (jobs / arm / name / "verifier").mkdir()
            (jobs / arm / name / "verifier" / "reward.txt").write_text(str(reward))
            # Not part of a record: an agent log stays behind.
            (jobs / arm / name / "agent").mkdir()
            (jobs / arm / name / "agent" / "log.txt").write_text("raw agent log")
            (trials / name / "controller" / "outcome.json").write_text("{}")
    for arm in ("alone", "taste"):
        (jobs / arm / "config.json").write_text(json.dumps({"job_name": arm}))
    return jobs, trials


def test_the_export_recomputes_the_report_and_verifies(records, tmp_path):
    jobs, trials = records
    out = tmp_path / "export"
    manifest = exporter.export([jobs / "alone", jobs / "taste"], trials, out)
    assert exporter.verify(out) == []
    assert "jobs/taste/task-0__taste1/verifier/reward.txt" in manifest["files"]
    assert "trials/task-0__taste1/controller/trajectory.json" in manifest["files"]
    assert not any("agent/log.txt" in name for name in manifest["files"])
    arms = [("alone", "alone"), ("taste", "taste")]

    def numbers(root, trial_root):
        result = study.report([(name, root / job) for name, job in arms], [("taste", "alone")], trial_root,
                              resamples=200, seed=1)
        return json.dumps(result, sort_keys=True)

    assert numbers(out / "jobs", out / "trials") == numbers(jobs, trials)


def test_a_changed_file_fails_verification(records, tmp_path):
    jobs, trials = records
    out = tmp_path / "export"
    exporter.export([jobs / "taste"], trials, out)
    (out / "jobs" / "taste" / "task-0__taste1" / "result.json").write_text("{}")
    (out / "extra.txt").write_text("added later")
    assert exporter.verify(out) == ["changed: jobs/taste/task-0__taste1/result.json",
                                    "not in the manifest: extra.txt"]


def test_an_export_holding_a_credential_is_refused(records, tmp_path, monkeypatch):
    jobs, trials = records
    secret = "s3cret-value-" + "x" * 20
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", secret)
    (trials / "task-1__taste1" / "controller" / "outcome.json").write_text(json.dumps({"leak": secret}))
    out = tmp_path / "export"
    with pytest.raises(ValueError, match="contains the value of AZURE_OPENAI_API_KEY") as caught:
        exporter.export([jobs / "taste"], trials, out, forbid_env=["AZURE_OPENAI_API_KEY"])
    assert secret not in str(caught.value)
    assert not out.exists() and not (tmp_path / "export.partial").exists()
