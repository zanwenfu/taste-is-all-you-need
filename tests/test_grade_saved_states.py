"""What grading saved states reports: the control over final saves, then the failed runs that held a pass."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from grade_saved_states import graded, job_name, plan_rows, summarize


def trial(job, name, task, reward, token, trials, events, *, final=None):
    folder = job / name
    folder.mkdir(parents=True)
    taste = {"trial": token, **({"final_checkpoint": final} if final else {})}
    (folder / "result.json").write_text(json.dumps({
        "task_name": "terminal-bench/" + task, "trial_name": name, "agent_result": {"metadata": {"taste": taste}},
        "verifier_result": {"rewards": {"reward": reward}}}))
    terminal = trials / token / "controller" / "terminal"
    (terminal / "checkpoints").mkdir(parents=True)
    connection = sqlite3.connect(terminal / "terminal.sqlite3")
    connection.execute("CREATE TABLE events (seq INTEGER, kind TEXT, payload TEXT)")
    for number, (kind, payload) in enumerate(events):
        connection.execute("INSERT INTO events VALUES (?,?,?)", (number, kind, json.dumps(payload)))
        if kind == "checkpoint":
            (terminal / "checkpoints" / f"{payload['checkpoint_id']}.json").write_text("{}")
            (terminal / "checkpoints" / f"{payload['tar_sha256']}.tar").write_bytes(b"x")
    connection.commit()
    connection.close()


def saved(name, **overrides):
    return "checkpoint", {"checkpoint_id": name, "tar_sha256": name * 8, "bytes": 10, "partial": False, **overrides}


def test_a_plan_lists_each_gradable_state_in_order_and_counts_the_rest(tmp_path):
    job, trials = tmp_path / "job", tmp_path / "trials"
    trial(job, "t1__a", "kv", 0.0, "tok", trials, [
        saved("cp1"), ("checkpoint_failed", {"checkpoint_id": "cp2"}), saved("cp3", partial=True), saved("cp4")],
        final={"saved": True, "directory": "/saves/f", "bytes": 5})
    rows, skipped = plan_rows(job, trials)
    assert [(row["checkpoint_id"], row["order"], row["kind"]) for row in rows] == [
        ("cp1", 1, "checkpoint"), ("cp4", 2, "checkpoint"), ("final", 3, "final")]
    assert rows[-1]["state"] == "/saves/f/manifest.json"
    assert skipped == {"failed": 1, "partial": 1}


def test_a_job_name_is_one_harbor_accepts():
    assert job_name({"trial": "kv.store__Ab1", "order": 3, "kind": "checkpoint"}) == "state-kv-store__Ab1-03-checkpoint"


def row(trial_name, kind, reward, state_reward, task="kv"):
    return {"job": "j", "trial": trial_name, "task": task, "reward": reward, "kind": kind,
            "state_reward": state_reward}


def test_the_summary_keeps_the_control_apart_from_the_result():
    summary = summarize([
        row("failed-held", "checkpoint", 0.0, 1.0), row("failed-held", "final", 0.0, 0.0),
        row("failed-never", "checkpoint", 0.0, 0.0), row("failed-never", "final", 0.0, 1.0),
        row("solved", "checkpoint", 1.0, 1.0), row("solved", "final", 1.0, 1.0),
        row("lost", "checkpoint", 0.0, None)])
    assert summary["runs_failed_at_end"] == 3 and summary["failed_runs_with_an_earlier_passing_state"] == 1
    assert summary["control_final_saves"] == 3 and summary["control_agree"] == 2
    assert summary["control_disagree"] == ["kv/failed-never"]
    assert summary["states_ungraded"] == 1 and summary["held_by_task"] == {"kv": 1}


def test_a_state_whose_restore_was_not_exact_gets_no_reward(tmp_path):
    entry = {"trial": "t", "order": 1, "kind": "checkpoint"}
    folder = tmp_path / job_name(entry) / "t0"
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps({
        "agent_result": {"metadata": {"saved_state": {"exact": False}}}, "verifier_result": {"rewards": {"reward": 1.0}}}))
    assert graded([entry], tmp_path)[0]["state_reward"] is None
