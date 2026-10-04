"""Outside-process result exchange, interrupted accounting and replay boundaries."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from taste.brains.azure_goal_credentials import GOAL_CREDENTIAL_NAME, encode_azure_goal_credentials
from taste.brains.azure_goal_handoff import (
    MAX_RESULT_BYTES,
    grading_ready,
    handoff_command,
    perform,
    preparation_bytes,
    read_handoff,
    settled_outcome,
)
from taste.brains.central_runtime import BudgetState, GoalOutcome
from taste.brains.goal_entrypoint import GoalInputError, GoalProcessInput
from tests.test_azure_central_host import goal as _goal
from tests.test_azure_central_host import policy as _policy
from tests.test_azure_goal_entrypoint import BOOTSTRAP

goal = _goal
policy = _policy


def write_input(path, raw):
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def process(path, digest, output, operation, env, bootstrap=""):
    argv = handoff_command(path, digest, output, operation=operation)
    argv[3] = argv[3].replace("from taste.brains.azure_goal_handoff", bootstrap + "\nfrom taste.brains.azure_goal_handoff", 1)
    return subprocess.run(argv, cwd=path.parent, env=env, capture_output=True, text=True, timeout=40)


def preparation(tmp_path, goal, policy):
    root = tmp_path / "workspace"
    root.mkdir()
    raw = preparation_bytes(root, "handoff", goal, policy, max_generations=2,
                            wall_clock_seconds=120, max_planner_failures=1)
    path = tmp_path / "prepare.json"
    return path, write_input(path, raw)


@pytest.mark.parametrize("lost_reply", [False, True])
def test_real_process_exchange_and_credential_free_settlement_preserve_cost(tmp_path, goal, policy, lost_reply):
    path, digest = preparation(tmp_path, goal, policy)
    counter = tmp_path / "wire-count"
    env = {key: value for key, value in os.environ.items()
           if key not in {"CREDENTIALS_DIRECTORY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL"}}
    env.update(TASTE_TEST_WIRE_COUNT=str(counter), OPENAI_API_KEY="personal-never-use",
               ANTHROPIC_API_KEY="claude-never-use", OPENAI_BASE_URL="https://wrong.invalid/v1")
    output = tmp_path / "prepared"
    prepared = process(path, digest, output, "prepare", env, BOOTSTRAP)
    assert prepared.returncode == 0, prepared.stderr
    assert not counter.exists()
    value, _ = read_handoff(output, digest, "prepare", service_uid=os.geteuid())
    config = GoalProcessInput.from_bytes(json.dumps(value).encode())
    assert config.repo_root == str(tmp_path / "workspace")
    input_path = tmp_path / "goal.json"
    input_digest = write_input(input_path, config.to_bytes())
    credentials = tmp_path / "credentials"
    credentials.mkdir(mode=0o700)
    secret = credentials / GOAL_CREDENTIAL_NAME
    secret.write_bytes(encode_azure_goal_credentials(config, "azure-test-only"))
    secret.chmod(0o400)
    env["CREDENTIALS_DIRECTORY"] = str(credentials)
    bootstrap = BOOTSTRAP
    if lost_reply:
        bootstrap = bootstrap.replace("return httpx.Response(200, json={",
            "raise httpx.ReadTimeout('reply lost after dispatch', request=wire)\n    return httpx.Response(200, json={")
    outcomes = []
    for operation in ("run", "settle"):
        if operation == "settle":
            secret.unlink()
        result_dir = tmp_path / operation
        child = process(input_path, input_digest, result_dir, operation, env, bootstrap)
        assert child.returncode == 0, child.stderr
        value, result_digest = read_handoff(result_dir, input_digest, operation, service_uid=os.geteuid())
        assert result_digest == hashlib.sha256((result_dir / "result.json").read_bytes()).hexdigest()
        outcome = settled_outcome(value, config)
        outcomes.append(outcome)
        assert grading_ready(outcome) == (not lost_reply)
        assert outcome.complete == (not lost_reply)
        # A lost reply has an unknown cost. It is carried as a reservation of the
        # most that call could have cost: still not "grading ready", but no
        # longer a budget nobody can reason about.
        assert not outcome.budget.unknown_planner_attempt_ids
        assert (outcome.budget.reserved_usd > 0) == lost_reply
        assert outcome.budget.known_spent_usd >= 0 if lost_reply else outcome.budget.known_spent_usd > 0
        assert counter.read_text().splitlines() == ["one Azure planner request"]
        assert "azure-test-only" not in (result_dir / "result.json").read_text() + child.stdout + child.stderr
    assert outcomes[0] == outcomes[1]
    # A second launcher cannot reclaim even a previously successful exchange.
    replay = process(input_path, input_digest, tmp_path / "run", "run", env, bootstrap)
    assert replay.returncode == 70
    assert counter.read_text().splitlines() == ["one Azure planner request"]
    # Preparation cannot replace already-admitted durable state.
    repeated = process(path, digest, tmp_path / "prepare-again", "prepare", env, bootstrap)
    assert repeated.returncode == 70 and not (tmp_path / "prepare-again").exists()


def test_failed_run_leaves_intent_for_recovery_and_does_not_fall_back(tmp_path, goal, policy, monkeypatch):
    path, digest = preparation(tmp_path, goal, policy)
    value = perform(path, digest, tmp_path / "prepared", operation="prepare")
    config = GoalProcessInput.from_bytes(json.dumps(value).encode())
    path = tmp_path / "goal.json"
    digest = write_input(path, config.to_bytes())
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "must-not-fall-back")
    output = tmp_path / "run"
    with pytest.raises(GoalInputError, match="credentials"):
        perform(path, digest, output, operation="run")
    assert (output / "intent.json").is_file() and not (output / "result.json").exists()
    with pytest.raises(FileNotFoundError):
        read_handoff(output, digest, "run", service_uid=os.geteuid())
    with pytest.raises(FileExistsError):
        perform(path, digest, output, operation="run")
    # Recovery uses a new operation, needs no key, and never starts planning.
    result = perform(path, digest, tmp_path / "settle", operation="settle")
    outcome = settled_outcome(result, config)
    assert outcome.stop_reason == "cancelled" and not grading_ready(outcome)
    assert outcome.budget.enforceable and outcome.budget.known_spent_usd == 0


@pytest.mark.parametrize("damage", ["digest", "source", "session", "limits", "deadline", "duplicate", "existing", "dirty"])
def test_preparation_rejects_invalid_or_replayed_input_before_mutation(tmp_path, goal, policy, damage):
    path, digest = preparation(tmp_path, goal, policy)
    output = tmp_path / "result"
    raw = path.read_bytes()
    value = json.loads(raw)
    if damage == "digest":
        digest = "0" * 64
    elif damage in {"source", "session", "limits", "deadline"}:
        key, replacement = {"source": ("python_source_sha256", "0" * 64), "session": ("session", "../other"),
                            "limits": ("wall_clock_seconds", True), "deadline": ("wall_clock_seconds", 1)}[damage]
        value[key] = replacement
        digest = write_input(path, json.dumps(value).encode())
    elif damage == "duplicate":
        digest = write_input(path, raw.rstrip()[:-1] + b',"session":"other"}')
    elif damage == "existing":
        output.mkdir(mode=0o700)
    else:
        (tmp_path / "workspace" / "existing").touch()
    with pytest.raises((ValueError, FileExistsError)):
        perform(path, digest, output, operation="prepare")
    assert not (tmp_path / "workspace" / ".git").exists()
    assert not (output / "result.json").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="write-permission rejection requires an unprivileged process")
def test_unwritable_worktree_parent_fails_before_partial_repository_creation(tmp_path, goal, policy):
    path, digest = preparation(tmp_path, goal, policy)
    mode = tmp_path.stat().st_mode & 0o777
    tmp_path.chmod(0o500)
    try:
        with pytest.raises(GoalInputError, match="enclosing workspace"):
            perform(path, digest, tmp_path / "result", operation="prepare")
        assert not (tmp_path / "workspace/.git").exists()
        assert not (tmp_path / "result").exists()
    finally:
        tmp_path.chmod(mode)


@pytest.mark.parametrize("damage", ["missing", "symlink", "hardlink", "fifo", "large", "mode", "uid",
                                     "directory_mode", "directory_symlink", "intent", "binding", "duplicate"])
def test_observer_rejects_incomplete_unbound_or_unsafe_exchange(tmp_path, damage):
    output = tmp_path / "result"
    output.mkdir(mode=0o700)
    binding = {"schema": "taste.brains/AzureGoalHandoff/1", "input_sha256": "a" * 64, "operation": "run"}
    intent = output / "intent.json"
    intent.write_text(json.dumps(binding))
    intent.chmod(0o600)
    path = output / "result.json"
    raw = json.dumps({**binding, "result": {}})
    path.write_text(raw)
    path.chmod(0o600)
    uid = os.geteuid()
    if damage == "missing":
        path.unlink()
    elif damage == "symlink":
        path.rename(output / "saved")
        path.symlink_to(output / "saved")
    elif damage == "hardlink":
        os.link(path, output / "alias")
    elif damage == "fifo":
        path.unlink()
        os.mkfifo(path)
    elif damage == "large":
        path.write_bytes(b"x" * (MAX_RESULT_BYTES + 1))
    elif damage == "mode":
        path.chmod(0o644)
    elif damage == "uid":
        uid += 1
    elif damage == "directory_mode":
        output.chmod(0o755)
    elif damage == "directory_symlink":
        output.rename(tmp_path / "saved")
        output.symlink_to(tmp_path / "saved", target_is_directory=True)
    elif damage == "intent":
        intent.write_text(json.dumps({**binding, "operation": "settle"}))
    elif damage == "binding":
        path.write_text(json.dumps({**binding, "input_sha256": "b" * 64, "result": {}}))
    else:
        path.write_text(raw[:-1] + ',"result":{}}')
    with pytest.raises((GoalInputError, OSError)):
        read_handoff(output, "a" * 64, "run", service_uid=uid)


def valid_outcome(goal, reason="complete"):
    return GoalOutcome(goal.goal_id, reason, reason == "complete", 1, 1,
                       budget=BudgetState(goal.budget_usd, 3, 0, worker_spent_usd=2, planner_spent_usd=1)).to_dict()


@pytest.mark.parametrize("field,replacement", [
    ("goal_id", "foreign"), ("complete", 1), ("complete", False), ("cycles", True), ("generations", -1),
    ("completion_reason", []), ("delivered_assignment_ids", ["a", "a"]),
    ("budget.limit_usd", 101), ("budget.known_spent_usd", float("nan")), ("budget.reserved_usd", -1),
    ("budget.planner_spent_usd", True), ("budget.worker_spent_usd", 0),
    ("budget.unknown_run_ids", ["a", "a"]), ("budget.unknown_planner_attempt_ids", [""]),
])
def test_result_rejects_identity_and_accounting_contradictions(goal, field, replacement):
    value = valid_outcome(goal)
    target = value
    if "." in field:
        parent, field = field.split(".")
        target = target[parent]
    target[field] = replacement
    with pytest.raises(GoalInputError):
        settled_outcome(value, SimpleNamespace(goal=goal))


@pytest.mark.parametrize("reason", ["complete", "generation_bound", "wall_clock", "budget_blocked", "spend_cap",
                                    "cancelled", "interrupted", "runtime_error", "planner_failed", "foreign"])
def test_only_known_settled_bounded_stops_can_proceed_to_grading(goal, reason):
    result = settled_outcome(valid_outcome(goal, reason), SimpleNamespace(goal=goal))
    assert grading_ready(result) == (reason in {
        "complete", "generation_bound", "wall_clock", "budget_blocked", "spend_cap"})


@pytest.mark.parametrize("reason,flags", [("complete", []), ("spend_cap", []), ("planner_failed", ["stopped:planner_failed"]),
                                          ("runtime_error", ["stopped:runtime_error"])])
def test_a_settled_run_that_stopped_otherwise_is_flagged_as_stopped_not_unsettled(goal, reason, flags):
    from taste.benchmarks.azure_terminal_trial import grading_flags

    assert grading_flags(settled_outcome(valid_outcome(goal, reason), SimpleNamespace(goal=goal))) == flags
    raw = valid_outcome(goal, reason)
    raw["budget"]["reserved_usd"] = 0.1
    assert grading_flags(settled_outcome(raw, SimpleNamespace(goal=goal))) == ["accounting_unsettled"]


@pytest.mark.parametrize("change", ["reserved", "unknown_worker", "live_worker", "unknown_planner", "over_budget"])
def test_complete_does_not_override_unsettled_or_over_budget_accounting(goal, change):
    raw = valid_outcome(goal)
    budget = raw["budget"]
    if change == "reserved":
        budget["reserved_usd"] = 0.1
    elif change == "over_budget":
        budget.update(known_spent_usd=101, worker_spent_usd=100)
    else:
        field = {"unknown_worker": "unknown_run_ids", "live_worker": "unbounded_live_run_ids",
                 "unknown_planner": "unknown_planner_attempt_ids"}[change]
        budget[field] = ["unsettled"]
    outcome = settled_outcome(raw, SimpleNamespace(goal=goal))
    assert outcome.complete and not grading_ready(outcome)
