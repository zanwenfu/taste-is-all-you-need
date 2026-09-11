"""The supervisor-to-worker subprocess boundary is exact and fail closed."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("claude_agent_sdk", reason="the worker process needs claude-agent-sdk")

from claude_agent_sdk import ResultMessage

from taste.brains.contract import CONTRACT_PATH, Contract
from taste.brains.monitor_judge import JUDGEMENT_SCHEMA, TERMINAL_JUDGEMENT_SCHEMA
from taste.brains.records import ArtifactSpec, Assignment, WorkerReport, contract_digest
from taste.brains.subbrain import SubBrain, SubBrainResult
from taste.brains.supervisor import LaunchSpec
from taste.brains.worker_entrypoint import (
    EntrypointConfig,
    WorkerExitCode,
    assignment_run_id,
    execute_worker,
    worker_command,
    worker_command_factory,
)
from taste.brains.worker_runtime import ASSIGNMENT_PATH, WORKER_REPORT_PATH
from taste.llm import MODEL_MONITOR
from taste.memstore import BranchBusy, Store

POLL = 0.001
QUIET = 0.005


def _assignment(
    store: Store,
    *,
    resources: dict[str, Any] | None = None,
    budget_usd: float | None = None,
) -> tuple[Assignment, str]:
    contract = Contract(
        identity="worker-1",
        task="write the parser",
        outputs=("parser.py",),
        success_criteria=("parser.py contains the checked implementation",),
        budget_usd=budget_usd,
    )
    branch = store.branch(contract.identity, producer="central-supervisor")
    assignment = Assignment(
        assignment_id="parser-assignment",
        generation=1,
        attempt=0,
        contract=contract,
        contract_digest=contract_digest(contract),
        base_state_id=branch.head.id,
        outputs=(
            ArtifactSpec(
                artifact_id="parser-output",
                path="parser.py",
                description="checked parser implementation",
            ),
        ),
        model="claude-sonnet-5",
        resources=resources or {},
    )
    branch.write(CONTRACT_PATH, contract.to_json())
    branch.write(ASSIGNMENT_PATH, assignment.to_json())
    prepared = branch.checkpoint("prepared exact worker assignment")
    branch.close()
    return assignment, prepared.id


def _ready_path(store: Store, run_id: str) -> Path:
    key = hashlib.sha256(run_id.encode()).hexdigest()
    return store.backend.common_dir / f"taste-supervisor.{store.session}.{key}.ready.json"


def _environment(store: Store, assignment: Assignment) -> dict[str, str]:
    run_id = assignment_run_id(assignment)
    return {
        "TASTE_WORKER_RUN_ID": run_id,
        "TASTE_WORKER_LAUNCH_TOKEN": "a" * 32,
        "TASTE_WORKER_READY_PATH": str(_ready_path(store, run_id)),
    }


def _config(
    root: Path,
    prepared: str,
    *,
    model: str = "claude-sonnet-5",
    monitor_budget_usd: float | None = None,
) -> EntrypointConfig:
    return EntrypointConfig(
        repo_root=root,
        session="session-1",
        worker="worker-1",
        expected_model=model,
        prepared_state_id=prepared,
        monitor_budget_usd=monitor_budget_usd,
        poll_interval=POLL,
        terminal_quiet_period=QUIET,
        shutdown_timeout=0.02,
    )


class FakeMonitorLLM:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.ready_models: list[str] = []
        self.calls: list[dict[str, Any]] = []

    def ensure_ready(self, *models: str) -> None:
        self.ready_models.extend(models)

    def call(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        terminal = "PINNED TERMINAL MONITOR OBSERVATION" in kwargs["messages"][0]["content"]
        if terminal:
            payload = {
                "schema": TERMINAL_JUDGEMENT_SCHEMA,
                "severity": "fine",
                "reason": "the exact checkpoint satisfies the contract",
                "evidence": ["parser.py is present in the pinned state"],
                "suggestion": "",
                "resolved_finding_ids": [],
                "unresolved_finding_ids": [],
            }
        else:
            payload = {
                "schema": JUDGEMENT_SCHEMA,
                "severity": "fine",
                "reason": "the worker remains aligned",
                "evidence": ["the observed events match the task"],
                "suggestion": "",
            }
        return SimpleNamespace(
            tool_calls=(),
            stop_reason="end_turn",
            summary_text=json.dumps(payload),
            model=MODEL_MONITOR,
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=10,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
            ),
        )


def _claim(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "summary": f"worker reported {status}",
        "evidence": ["parser.py inspected"],
        "accepted_inbox_ids": [],
        "accepted_verdicts": {},
    }


def _result(status: str) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=1,
        session_id="sdk-session-1",
        total_cost_usd=0.1,
        stop_reason=None,
        terminal_reason="completed",
        structured_output=_claim(status),
        errors=None,
        origin=None,
    )


class ScriptedClient:
    def __init__(self, status: str = "completed", *, cancel_disconnect: bool = False) -> None:
        self.options: Any = None
        self.status = status
        self.cancel_disconnect = cancel_disconnect
        self.queried = asyncio.Event()
        self.disconnected = asyncio.Event()
        self.calls: list[str] = []

    async def connect(self) -> None:
        self.calls.append("connect")

    async def query(self, _prompt: str) -> None:
        self.calls.append("query")
        Path(self.options.cwd, "parser.py").write_text("IMPLEMENTED = True\n", encoding="utf-8")
        self.queried.set()

    async def receive_messages(self):
        await self.queried.wait()
        yield _result(self.status)
        await self.disconnected.wait()

    async def disconnect(self) -> None:
        self.calls.append("disconnect")
        if self.cancel_disconnect:
            raise asyncio.CancelledError
        self.disconnected.set()


def _client_factory(client: ScriptedClient):
    def factory(options: Any) -> ScriptedClient:
        client.options = options
        return client

    return factory


@pytest.mark.parametrize(
    "failure",
    ["malformed", "contract_mismatch", "model", "run_id", "ready_path"],
)
def test_bad_durable_or_launch_binding_spawns_no_sdk_client(
    tmp_path: Path,
    failure: str,
) -> None:
    root = tmp_path / "repo"
    store = Store.open(root, "session-1")
    assignment, prepared = _assignment(store)
    if failure == "malformed":
        branch = store.branch("worker-1")
        branch.write(ASSIGNMENT_PATH, "{not-json}\n")
        prepared = branch.checkpoint("malformed assignment is durable").id
        branch.close()
    if failure == "contract_mismatch":
        branch = store.branch("worker-1")
        conflicting = Contract(
            identity="worker-1",
            task="a different task",
            success_criteria=("different criterion",),
        )
        branch.write(CONTRACT_PATH, conflicting.to_json())
        prepared = branch.checkpoint("mismatched contract is durable").id
        branch.close()
    environment = _environment(store, assignment)
    if failure == "run_id":
        environment["TASTE_WORKER_RUN_ID"] = "worker-run." + "f" * 64
    if failure == "ready_path":
        environment["TASTE_WORKER_READY_PATH"] = str(tmp_path / "wrong.ready.json")
    config = _config(
        root, prepared, model="wrong-model" if failure == "model" else assignment.model
    )
    sdk_clients = 0
    llms = 0

    def client_factory(_options: Any) -> None:
        nonlocal sdk_clients
        sdk_clients += 1

    def llm_factory(**_kwargs: Any) -> FakeMonitorLLM:
        nonlocal llms
        llms += 1
        return FakeMonitorLLM()

    exit_code = asyncio.run(
        execute_worker(
            config,
            store=store,
            environ=environment,
            llm_factory=llm_factory,
            client_factory=client_factory,
        )
    )

    assert exit_code is WorkerExitCode.INPUT_REJECTED
    assert sdk_clients == 0
    assert llms == 0
    assert store.view("worker-1").holder is None
    store.close()


def test_budgeted_injected_client_is_rejected_before_monitor_or_client(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    store = Store.open(root, "session-1")
    assignment, prepared = _assignment(store, budget_usd=100.0)
    environment = _environment(store, assignment)
    made_clients = 0
    made_llms = 0

    def client_factory(_options: Any) -> None:
        nonlocal made_clients
        made_clients += 1

    def llm_factory(**_kwargs: Any) -> FakeMonitorLLM:
        nonlocal made_llms
        made_llms += 1
        return FakeMonitorLLM()

    exit_code = asyncio.run(
        execute_worker(
            _config(root, prepared),
            store=store,
            environ=environment,
            llm_factory=llm_factory,
            client_factory=client_factory,
        )
    )

    assert exit_code is WorkerExitCode.INPUT_REJECTED
    assert made_clients == 0
    assert made_llms == 0
    assert store.view("worker-1").holder is None
    store.close()


def test_exact_env_binding_marks_ready_and_completes_scripted_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    store = Store.open(root, "session-1")
    assignment, prepared = _assignment(store)
    environment = _environment(store, assignment)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    llm = FakeMonitorLLM()
    client = ScriptedClient()

    exit_code = asyncio.run(
        execute_worker(
            _config(root, prepared),
            store=store,
            environ=environment,
            llm_factory=lambda **_kwargs: llm,
            client_factory=_client_factory(client),
        )
    )

    assert exit_code is WorkerExitCode.COMPLETED
    ready = json.loads(Path(environment["TASTE_WORKER_READY_PATH"]).read_text())
    assert ready == {
        "schema": "taste.brains/WorkerReady/1",
        "run_id": assignment_run_id(assignment),
        "launch_token": "a" * 32,
        "pid": __import__("os").getpid(),
    }
    report = WorkerReport.from_json(store.view("worker-1").head.read(WORKER_REPORT_PATH) or "")
    assert report.completed is True
    assert report.run_id == assignment_run_id(assignment)
    assert report.final_state_id == report.outputs[0].state_id
    assert client.calls[0] == "connect"
    assert client.calls[-1] == "disconnect"
    assert llm.ready_models == [MODEL_MONITOR]
    assert any("PINNED TERMINAL" in call["messages"][0]["content"] for call in llm.calls)
    assert store.view("worker-1").holder is None
    store.close()


def test_exact_assignment_monitor_budget_reaches_only_the_monitor_llm(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    store = Store.open(root, "session-1")
    assignment, prepared = _assignment(
        store,
        resources={"monitor_budget_usd": 0.75},
    )
    captured: dict[str, Any] = {}

    def llm_factory(**kwargs: Any) -> FakeMonitorLLM:
        captured.update(kwargs)
        return FakeMonitorLLM(**kwargs)

    exit_code = asyncio.run(
        execute_worker(
            _config(root, prepared, monitor_budget_usd=0.75),
            store=store,
            environ=_environment(store, assignment),
            llm_factory=llm_factory,
            client_factory=_client_factory(ScriptedClient()),
            ready_callback=lambda: None,
        )
    )

    assert exit_code is WorkerExitCode.COMPLETED
    assert captured["budget_usd"] == 0.75
    assert captured["cap_on"] == "billed"
    assert captured["max_attempts"] == 1
    assert set(captured) == {
        "env_dir",
        "budget_usd",
        "run_id",
        "max_attempts",
        "cap_on",
    }
    store.close()


def test_descendant_head_after_prepare_is_rejected_before_llm_or_sdk(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    store = Store.open(root, "session-1")
    assignment, prepared = _assignment(store)
    branch = store.branch("worker-1")
    branch.write("injected.py", "not part of the prepared assignment\n")
    branch.checkpoint("unexpected descendant before worker launch")
    branch.close()
    called = {"llm": 0, "sdk": 0}

    def llm_factory(**_kwargs: Any) -> FakeMonitorLLM:
        called["llm"] += 1
        return FakeMonitorLLM()

    def client_factory(_options: Any) -> None:
        called["sdk"] += 1

    exit_code = asyncio.run(
        execute_worker(
            _config(root, prepared),
            store=store,
            environ=_environment(store, assignment),
            llm_factory=llm_factory,
            client_factory=client_factory,
        )
    )

    assert exit_code is WorkerExitCode.INPUT_REJECTED
    assert called == {"llm": 0, "sdk": 0}
    assert store.view("worker-1").holder is None
    store.close()


@pytest.mark.parametrize(
    ("contract_field", "paths"),
    [
        ("inputs", ("undeclared-input.txt",)),
        ("outputs", ("different-output.py",)),
    ],
)
def test_contract_and_structured_io_mismatch_is_rejected_before_llm_or_sdk(
    tmp_path: Path,
    contract_field: str,
    paths: tuple[str, ...],
) -> None:
    root = tmp_path / "repo"
    store = Store.open(root, "session-1")
    assignment, _prepared = _assignment(store)
    contract = replace(assignment.contract, **{contract_field: paths})
    mismatched = replace(
        assignment,
        contract=contract,
        contract_digest=contract_digest(contract),
    )
    branch = store.branch(assignment.worker)
    branch.write(CONTRACT_PATH, contract.to_json())
    branch.write(ASSIGNMENT_PATH, mismatched.to_json())
    prepared = branch.checkpoint("durable mismatched contract and typed artifacts").id
    branch.close()
    called = {"llm": 0, "sdk": 0}

    def llm_factory(**_kwargs: Any) -> FakeMonitorLLM:
        called["llm"] += 1
        return FakeMonitorLLM()

    def client_factory(_options: Any) -> None:
        called["sdk"] += 1

    exit_code = asyncio.run(
        execute_worker(
            _config(root, prepared),
            store=store,
            environ=_environment(store, mismatched),
            llm_factory=llm_factory,
            client_factory=client_factory,
        )
    )

    assert exit_code is WorkerExitCode.INPUT_REJECTED
    assert called == {"llm": 0, "sdk": 0}
    assert store.view(assignment.worker).holder is None
    store.close()


@pytest.mark.parametrize("race", ["commit", "dirty"])
def test_post_preflight_head_or_dirty_race_never_constructs_sdk(
    tmp_path: Path,
    race: str,
) -> None:
    root = tmp_path / "repo"
    store = Store.open(root, "session-1")
    assignment, prepared = _assignment(store)
    made: list[SubBrain] = []
    sdk_clients = 0

    def racing_brain(opened: Store, contract: Contract, **kwargs: Any) -> SubBrain:
        brain = SubBrain(opened, contract, **kwargs)
        made.append(brain)
        brain.branch.write("parser.py", "injected between read and lease validation\n")
        if race == "commit":
            brain.checkpoint("racing commit after lease-free preflight")
        return brain

    def client_factory(_options: Any) -> None:
        nonlocal sdk_clients
        sdk_clients += 1

    exit_code = asyncio.run(
        execute_worker(
            _config(root, prepared),
            store=store,
            environ=_environment(store, assignment),
            llm_factory=lambda **_kwargs: FakeMonitorLLM(),
            brain_factory=racing_brain,
            client_factory=client_factory,
        )
    )

    assert exit_code is WorkerExitCode.INPUT_REJECTED
    assert sdk_clients == 0
    assert made
    assert store.view("worker-1").holder is None
    store.close()


def test_valid_blocked_report_has_distinct_noncompletion_exit(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    store = Store.open(root, "session-1")
    assignment, prepared = _assignment(store)
    environment = _environment(store, assignment)
    client = ScriptedClient(status="blocked")

    exit_code = asyncio.run(
        execute_worker(
            _config(root, prepared),
            store=store,
            environ=environment,
            llm_factory=lambda **_kwargs: FakeMonitorLLM(),
            client_factory=_client_factory(client),
            ready_callback=lambda: None,
        )
    )

    report = WorkerReport.from_json(store.view("worker-1").head.read(WORKER_REPORT_PATH) or "")
    assert exit_code is WorkerExitCode.INCOMPLETE
    assert report.completed is False
    assert report.terminal_reason == "blocked"
    assert store.view("worker-1").holder is None
    store.close()


def test_shutdown_unconfirmed_keeps_lease_and_publishes_no_report(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    store = Store.open(root, "session-1")
    assignment, prepared = _assignment(store)
    environment = _environment(store, assignment)
    client = ScriptedClient(cancel_disconnect=True)

    exit_code = asyncio.run(
        execute_worker(
            _config(root, prepared),
            store=store,
            environ=environment,
            llm_factory=lambda **_kwargs: FakeMonitorLLM(),
            client_factory=_client_factory(client),
            ready_callback=lambda: None,
        )
    )

    assert exit_code is WorkerExitCode.SHUTDOWN_UNCONFIRMED
    assert store.view("worker-1").holder is not None
    assert store.view("worker-1").head.read(WORKER_REPORT_PATH) is None
    observer = Store.open(root, "session-1")
    with pytest.raises(BranchBusy):
        observer.branch("worker-1")
    observer.close()
    store.close()


def test_cleanup_failure_is_categorical_and_does_not_claim_completion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    store = Store.open(root, "session-1")
    assignment, prepared = _assignment(store)
    made: list[SubBrain] = []

    def broken_close_brain(opened: Store, contract: Contract, **kwargs: Any) -> SubBrain:
        brain = SubBrain(opened, contract, **kwargs)
        made.append(brain)

        def fail_close() -> None:
            raise OSError("simulated lease cleanup failure")

        brain.close = fail_close  # type: ignore[method-assign]
        return brain

    class ImmediateRuntime:
        def __init__(self, brain: SubBrain, _monitor: Any, **_kwargs: Any) -> None:
            self.brain = brain

        async def run(self) -> SubBrainResult:
            return SubBrainResult(identity=self.brain.contract.identity, completed=True)

    exit_code = asyncio.run(
        execute_worker(
            _config(root, prepared),
            store=store,
            environ=_environment(store, assignment),
            llm_factory=lambda **_kwargs: FakeMonitorLLM(),
            brain_factory=broken_close_brain,
            runtime_factory=ImmediateRuntime,  # type: ignore[arg-type]
            ready_callback=lambda: None,
        )
    )

    assert exit_code is WorkerExitCode.RUNTIME_FAILURE
    observer = Store.open(root, "session-1")
    with pytest.raises(BranchBusy):
        observer.branch("worker-1")
    observer.close()
    # Explicit test cleanup stands in for production process exit.
    made[0].branch.close()
    store.close()


def test_async_cancellation_reaps_client_and_releases_lease(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    store = Store.open(root, "session-1")
    assignment, prepared = _assignment(store)
    environment = _environment(store, assignment)
    client = ScriptedClient()

    # With no result, the runtime remains live until this process boundary is
    # cancelled. Disconnect still has to complete before the exit is classified.
    async def no_results():
        await client.queried.wait()
        await client.disconnected.wait()
        if False:  # make this an async generator without emitting false evidence
            yield _result("completed")

    client.receive_messages = no_results  # type: ignore[method-assign]

    async def cancel_run() -> WorkerExitCode:
        task = asyncio.create_task(
            execute_worker(
                _config(root, prepared),
                store=store,
                environ=environment,
                llm_factory=lambda **_kwargs: FakeMonitorLLM(),
                client_factory=_client_factory(client),
                ready_callback=lambda: None,
            )
        )
        await client.queried.wait()
        task.cancel()
        return await task

    exit_code = asyncio.run(cancel_run())

    assert exit_code is WorkerExitCode.INTERRUPTED
    assert client.calls[-1] == "disconnect"
    assert store.view("worker-1").holder is None
    assert store.view("worker-1").head.read(WORKER_REPORT_PATH) is None
    store.close()


def test_launcher_command_is_exact_argv_without_credentials(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    store = Store.open(root, "session-1")
    assignment, prepared = _assignment(store)
    spec = LaunchSpec(
        run_id=assignment_run_id(assignment),
        assignment=assignment,
        prepared_state_id=prepared,
        worktree=store.worktree_path_for(assignment.worker),
        deadline_at="2030-01-01T00:00:00Z",
        readiness_path=_ready_path(store, assignment_run_id(assignment)),
        launch_path=tmp_path / "launch.json",
    )

    argv = worker_command(
        spec,
        repo_root=root,
        session="session-1",
        python_executable=sys.executable,
    )

    assert argv[:3] == (sys.executable, "-m", "taste.brains.worker_entrypoint")
    assert argv[argv.index("--worker") + 1] == assignment.worker
    assert argv[argv.index("--model") + 1] == assignment.model
    assert argv[argv.index("--prepared-state") + 1] == prepared
    assert "--monitor-budget-usd" not in argv
    assert not any("key" in item.lower() or "token" in item.lower() for item in argv)
    store.close()


def test_launcher_command_passes_only_validated_assignment_monitor_budget(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    store = Store.open(root, "session-1")
    assignment, prepared = _assignment(
        store,
        resources={"monitor_budget_usd": 0.75, "api_key": "never-on-argv"},
    )
    spec = LaunchSpec(
        run_id=assignment_run_id(assignment),
        assignment=assignment,
        prepared_state_id=prepared,
        worktree=store.worktree_path_for(assignment.worker),
        deadline_at="2030-01-01T00:00:00Z",
        readiness_path=_ready_path(store, assignment_run_id(assignment)),
        launch_path=tmp_path / "launch.json",
    )

    argv = worker_command_factory(
        root,
        "session-1",
        python_executable=sys.executable,
    )(spec)

    assert argv[argv.index("--monitor-budget-usd") + 1] == "0.75"
    assert "never-on-argv" not in argv
    store.close()


@pytest.mark.parametrize("invalid", [True, 0, -0.1, "0.5"])
def test_launcher_rejects_invalid_assignment_monitor_budget(
    tmp_path: Path,
    invalid: object,
) -> None:
    root = tmp_path / "repo"
    store = Store.open(root, "session-1")
    assignment, prepared = _assignment(
        store,
        resources={"monitor_budget_usd": invalid},
    )
    spec = LaunchSpec(
        run_id=assignment_run_id(assignment),
        assignment=assignment,
        prepared_state_id=prepared,
        worktree=store.worktree_path_for(assignment.worker),
        deadline_at="2030-01-01T00:00:00Z",
        readiness_path=_ready_path(store, assignment_run_id(assignment)),
        launch_path=tmp_path / "launch.json",
    )

    with pytest.raises(ValueError, match="monitor_budget_usd"):
        worker_command(spec, repo_root=root, session="session-1")
    store.close()
