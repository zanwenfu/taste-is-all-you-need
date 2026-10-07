"""Twin checks: what the study's checker checked, checked again by another model, and scored."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from tests.recovery_fakes import FakeHarbor, recoveries
from tests.test_recovery_runs import drive, study

_SPEC = importlib.util.spec_from_file_location(
    "twin_checks", Path(__file__).resolve().parents[1] / "scripts" / "twin_checks.py")
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)


def test_every_finished_check_gets_a_twin_on_the_same_files_and_the_twins_are_scored(tmp_path, capsys):
    jobs_dir, trials, given, _ = study(tmp_path, recoveries=["continue"], repeats=1)
    path = tmp_path / "rec.json"
    # The study's checker rejects round one and accepts round two, whose trial the hidden tests pass.
    driver, _ = drive(path, given, jobs_dir, FakeHarbor(trials, recoveries(
        verdict=lambda spec: "done" if "-e1r2-" in spec.name else "not_done",
        reward=lambda spec: 1.0 if "-e1r2-" in spec.name else 0.0)))
    originals = [name for name, job in driver.jobs.items() if job["purpose"] == "check"]
    specs = cli.twins(json.loads(path.read_text()), model="gpt-6-luna", effort="low", prefix="twin")
    assert len(specs) == len(originals) == 3
    for spec in specs:
        original = driver.jobs[spec.name.replace("twin-", "rec-", 1)]["spec"]
        assert spec.name.startswith("twin-") and dict(spec.env)["MODEL"] == "azure/gpt-6-luna"
        settings = spec.settings()
        assert settings["worker_effort"] == "low" and settings["agent"] == "checker"
        # The same checker task on the same final state as the original check.
        kept = {key: value for key, value in settings.items() if key != "worker_effort"}
        assert kept == {key: value for key, value in
                        dict(item.split("=", 1) for item in original["argv"][original["argv"].index("--ak") + 1::2]).items()
                        if key != "worker_effort"}
    # The twin model says done everywhere: right once (round two), wrong twice.
    FakeHarbor(trials, recoveries(verdict=lambda spec: "done")).run(specs)
    found = cli.score(json.loads(path.read_text()), prefix="twin")
    assert found["checks"] == 3 and found["agreement"] == 1 / 3
    assert found["original"]["accuracy"] == 1.0 and found["twin"]["accuracy"] == 1 / 3
    assert found["original"]["false_done"] == 0 and found["twin"]["false_done"] == 2
    assert found["twin"]["usd_per_check"] == 0.05
    assert cli.main(["report", "--state", str(path), "--prefix", "twin"]) == 0
    assert "agreement 0.33" in capsys.readouterr().out
