"""Fault-oriented tests for the live SubBrain/MonitorBrain host boundary."""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from taste.memstore import BranchBusy, Store, Verdict

pytest.importorskip("claude_agent_sdk", reason="the brain layer needs claude-agent-sdk")

from claude_agent_sdk import (
    ConversationResetMessage,
    MirrorErrorMessage,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    UserMessage,
)

from taste.brains.communication import Communicator, Message
from taste.brains.contract import CONTRACT_PATH, Contract
from taste.brains.monitor import Judgement, MonitorBrain, Severity, TerminalDecision
from taste.brains.records import ArtifactSpec, Assignment, WorkerReport, contract_digest
from taste.brains.subbrain import SubBrain
from taste.brains.worker_runtime import (
    ASSIGNMENT_PATH,
    WORKER_REPORT_PATH,
    WORKER_RESULT_SCHEMA,
    ContractMismatch,
    ShutdownUnconfirmed,
    WorkerRuntime,
)
from taste.pricing import ensure_priced, max_call_cost_usd

POLL = 0.001
QUIET = 0.005


def reset_safe_budget(*, threshold: float = 1.0) -> float:
    model = "claude-sonnet-5"
    price = ensure_priced(model)
    exposure = max_call_cost_usd(
        model,
        max_output_tokens=price.context_window,
        max_attempts=1,
        cap_on="billed",
    )
    return 2 * (exposure + threshold)


def a_contract(**overrides: Any) -> Contract:
    values = {
        "identity": "worker-1",
        "task": "build the parser",
        "outputs": ("parser.py",),
        "success_criteria": ("the parser tests pass",),
    }
    return Contract(**{**values, **overrides})


@pytest.fixture
def store(tmp_path: Path) -> Store:
    opened = Store.open(tmp_path / "repo", "s1")
    yield opened
    opened.close()


def scaffold(store: Store, contract: Contract | None = None) -> Contract:
    contract = contract or a_contract()
    brain = SubBrain(store, contract)
    brain.install_contract()
    brain.checkpoint("central accepted the worker contract")
    brain.close()
    return contract


def scaffold_assignment(
    store: Store,
    *,
    output: str = "parser.py",
    assignment_id: str = "build-parser",
    generation: int = 1,
    budget_usd: float | None = None,
) -> Assignment:
    contract = a_contract(outputs=(output,), budget_usd=budget_usd)
    brain = SubBrain(store, contract)
    base_state = brain.branch.head.meta.parents[0]
    assignment = Assignment(
        assignment_id=assignment_id,
        generation=generation,
        attempt=0,
        contract=contract,
        contract_digest=contract_digest(contract),
        base_state_id=base_state,
        outputs=(ArtifactSpec("parser", output, description="the parser"),),
    )
    brain.install_contract()
    brain.branch.write(ASSIGNMENT_PATH, assignment.to_json())
    brain.checkpoint("central accepted the typed assignment")
    brain.close()
    return assignment


def completed_claim(
    *,
    accepted_inbox_ids: tuple[str, ...] = (),
    accepted_verdicts: dict[str, int] | None = None,
    status: str = "completed",
    summary: str = "implemented and checked the parser",
    evidence: tuple[str, ...] = ("parser.py exists", "parser tests passed"),
) -> dict[str, Any]:
    return {
        "status": status,
        "summary": summary,
        "evidence": list(evidence),
        "accepted_inbox_ids": list(accepted_inbox_ids),
        "accepted_verdicts": dict(accepted_verdicts or {}),
    }


def result_message(
    *,
    session_id: str = "session-1",
    terminal_reason: str | None = "completed",
    stop_reason: str | None = None,
    structured_output: Any = None,
    is_error: bool = False,
    errors: list[str] | None = None,
    origin: dict[str, Any] | None = None,
    turns: int = 1,
    cost: float | None = 0.25,
    uuid: str | None = None,
    model_usage: dict[str, Any] | None = None,
) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=is_error,
        num_turns=turns,
        session_id=session_id,
        total_cost_usd=cost,
        stop_reason=stop_reason,
        terminal_reason=terminal_reason,
        structured_output=structured_output,
        errors=errors,
        origin=origin,
        uuid=uuid,
        model_usage=model_usage,
    )


def auditable_result_message(
    *,
    cost: float = 0.25,
    turns: int = 1,
    session_id: str = "session-1",
    uuid: str = "result-1",
    **kwargs: Any,
) -> ResultMessage:
    model = "claude-sonnet-5"
    window = ensure_priced(model).context_window
    # This model bills output at $10/M tokens. Keep the fake provider's
    # cumulative token ledger internally consistent with its billed total so
    # budgeted tests exercise the same identity check as production results.
    output_tokens = round(cost * 100_000)
    return result_message(
        cost=cost,
        turns=turns,
        session_id=session_id,
        uuid=uuid,
        model_usage={
            model: {
                "inputTokens": 0,
                "outputTokens": output_tokens,
                "cacheReadInputTokens": 0,
                "cacheCreationInputTokens": 0,
                "webSearchRequests": 0,
                "costUSD": cost,
                "contextWindow": window,
                "maxOutputTokens": window,
                "canonicalModel": model,
                "provider": "firstParty",
            }
        },
        **kwargs,
    )


class ScriptedClient:
    """A persistent stream: it ends only when the host disconnects it."""

    def __init__(self, options: Any, messages: list[Any] | None = None) -> None:
        self.options = options
        self.messages = list(messages or [result_message()])
        self.calls: list[tuple[str, Any]] = []
        self.queried = asyncio.Event()
        self.disconnected = asyncio.Event()

    async def connect(self) -> None:
        self.calls.append(("connect", None))

    async def query(self, prompt: str) -> None:
        self.calls.append(("query", prompt))
        self.queried.set()

    async def receive_messages(self):
        await self.queried.wait()
        for message in self.messages:
            await asyncio.sleep(0)
            yield message
        await self.disconnected.wait()

    async def disconnect(self) -> None:
        self.calls.append(("disconnect", None))
        self.disconnected.set()


class QuietMonitor:
    def __init__(
        self,
        *,
        model_calls: int = 0,
        cost_known: bool = True,
        cost_usd: float | None = 0.0,
    ) -> None:
        self.clients: list[Any] = []
        self.drained = False
        self.model_calls = model_calls
        self.cost_known = cost_known
        self.cost_usd = cost_usd

    @property
    def pending_actions(self) -> tuple[Any, ...]:
        return ()

    async def cycle(self, client: Any):
        self.clients.append(client)
        return None, None

    async def drain(self, client: Any, *, final: bool = False):
        self.clients.append(client)
        self.drained = final
        return []

    async def certify_terminal(self, state: Any, *, context: Any):
        durable = Contract.from_json(state.read("contract.json") or "")
        return SimpleNamespace(
            state_id=state.id,
            contract_digest=contract_digest(durable),
            acceptable=True,
            failure="",
            judgement=SimpleNamespace(
                severity=SimpleNamespace(value="fine"),
            ),
        )

    def report(self) -> dict[str, Any]:
        return {
            "worker": "worker-1",
            "worst": "fine",
            "current": "fine",
            "current_state": None,
            "terminal_assessment": None,
            "judgements": 0,
            "model_calls": self.model_calls,
            "cost_known": self.cost_known,
            "cost_usd": self.cost_usd,
            "pending_actions": [],
            "interventions": [],
            "alive": True,
        }


def runtime_for(
    store: Store,
    contract: Contract,
    client: Any,
    monitor: Any | None = None,
    *,
    assignment: Assignment | None = None,
    brain: SubBrain | None = None,
    run_id: str | None = None,
    communicator: Communicator | None = None,
) -> WorkerRuntime:
    def factory(options: Any) -> Any:
        client.options = options
        return client

    # Budgeted production code must select WorkerRuntime's built-in SDK client.
    # These protocol tests replace the already-selected client internally so
    # they can exercise reset/accounting frames without exposing a production
    # constructor bypass.
    budgeted = contract.budget_usd is not None
    runtime = WorkerRuntime(
        brain
        or SubBrain(store, contract, model=assignment.model if assignment else "claude-sonnet-5"),
        monitor or QuietMonitor(),
        assignment=assignment,
        run_id=run_id,
        communicator=communicator,
        client_factory=None if budgeted else factory,
        poll_interval=POLL,
        terminal_quiet_period=QUIET,
        shutdown_timeout=1,
    )
    if budgeted:
        test_cli_evidence = {
            "claude_agent_sdk_version": "0.2.152",
            "claude_code_version": "2.1.259",
            "cli_sha256": "884baa38fe1a624be25c4a91568bf5a08b5cf4e7d7acf29b7760e3525d964898",
        }
        runtime._preflight_budgeted_cli = lambda: (  # type: ignore[method-assign]
            Path(__file__).resolve(),
            dict(test_cli_evidence),
        )
        runtime._make_client = factory  # type: ignore[method-assign]
        abort_now = getattr(client, "abort_now", None)
        wait_reaped = getattr(client, "wait_reaped", None)
        if callable(abort_now) and callable(wait_reaped):

            class TestBudgetProcess:
                returncode: int | None = None

                def kill(self) -> None:
                    result = abort_now()
                    if inspect.isawaitable(result):
                        close = getattr(result, "close", None)
                        if callable(close):
                            close()
                        raise RuntimeError("test budget process kill must be synchronous")

                async def wait(self) -> int | None:
                    result = wait_reaped()
                    if inspect.isawaitable(result):
                        result = await result
                    if result is True:
                        self.returncode = -9
                    return self.returncode

            process = TestBudgetProcess()

            # This is deliberately an internal test splice: public budgeted
            # factories and duck-typed clients remain rejected by production
            # code.  The adapter exercises the same synchronous process
            # kill/reap fields used by the audited subprocess transport.
            runtime._assert_budgeted_client_identity = lambda: None  # type: ignore[method-assign]

            def arm_test_process() -> dict[str, Any]:
                runtime._budget_abort_transport = SimpleNamespace(_ready=True)
                runtime._budget_abort_process = process
                return dict(test_cli_evidence)

            runtime._arm_budgeted_process_boundary = arm_test_process  # type: ignore[method-assign]
    return runtime


def test_stream_is_durable_but_an_untyped_sdk_turn_is_never_task_completion(
    store: Store,
) -> None:
    contract = scaffold(store)
    stream = StreamEvent(
        uuid="message-1",
        session_id="session-1",
        event={"type": "content_block_delta", "delta": {"text": "inspecting"}},
    )
    client = ScriptedClient(
        None,
        [SystemMessage("init", {"session_id": "session-1"}), stream, result_message()],
    )
    monitor = QuietMonitor()

    outcome = asyncio.run(runtime_for(store, contract, client, monitor).run())

    assert not outcome.completed
    assert outcome.terminal_reason == "turn_complete_without_assignment"
    assert outcome.cost_usd == 0.25
    assert monitor.drained
    assert monitor.clients and all(seen is client for seen in monitor.clients)
    assert [call[0] for call in client.calls][0:2] == ["connect", "query"]
    assert client.calls[-1][0] == "disconnect"
    head = store.view(contract.identity).head
    stream_events = [
        event
        for event in head.transcript.turns
        if event.get("kind") == "sdk_message" and event.get("message_type") == "StreamEvent"
    ]
    assert stream_events[0]["message"]["event"]["delta"]["text"] == "inspecting"
    report = json.loads(head.read(WORKER_REPORT_PATH) or "null")
    assert report["completed_claim"] is False
    assert "no typed assignment" in report["uncertainty"][0]
    assert outcome.states == [head.meta.parents[0], head.id]


def test_exact_mirror_error_is_a_durability_failure_even_if_sdk_continues(
    store: Store,
) -> None:
    contract = scaffold(store)
    mirror_error = MirrorErrorMessage(
        subtype="mirror_error",
        data={"session_id": "session-1"},
        error="three append attempts failed",
    )
    client = ScriptedClient(None, [mirror_error, result_message()])
    monitor = QuietMonitor()

    outcome = asyncio.run(runtime_for(store, contract, client, monitor).run())

    assert not outcome.completed
    assert outcome.terminal_reason == "mirror_error"
    assert "three append attempts failed" in outcome.error
    assert outcome.cost_usd is None
    report = json.loads(store.view(contract.identity).head.read(WORKER_REPORT_PATH) or "null")
    assert report["durability_ok"] is False
    assert not monitor.drained, "an error path must not send new monitor queries without a reader"


def test_slow_monitor_does_not_stall_stream_consumption(store: Store) -> None:
    contract = scaffold(store)
    stream_consumed = asyncio.Event()

    class StreamingClient(ScriptedClient):
        async def receive_messages(self):
            await self.queried.wait()
            stream_consumed.set()
            yield StreamEvent(
                uuid="m1",
                session_id="session-1",
                event={"type": "content_block_delta", "delta": {"text": "moving"}},
            )
            yield result_message()
            await self.disconnected.wait()

    class SlowMonitor(QuietMonitor):
        async def cycle(self, client: Any):
            self.clients.append(client)
            await asyncio.wait_for(stream_consumed.wait(), timeout=1)
            return None, None

    client = StreamingClient(None)
    outcome = asyncio.run(runtime_for(store, contract, client, SlowMonitor()).run())

    assert stream_consumed.is_set()
    assert outcome.terminal_reason == "turn_complete_without_assignment"
    assert not outcome.error


def test_two_submitted_queries_may_be_covered_by_one_result(store: Store) -> None:
    contract = scaffold(store)
    message_id = ""

    class CoalescingClient(ScriptedClient):
        def __init__(self) -> None:
            super().__init__(None, [])
            self.query_count = 0
            self.two_queries = asyncio.Event()

        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            self.query_count += 1
            if self.query_count == 2:
                self.two_queries.set()

        async def receive_messages(self):
            await self.two_queries.wait()
            yield result_message(
                structured_output=completed_claim(
                    status="continue",
                    summary="handled coalesced prompts",
                    evidence=(),
                    accepted_inbox_ids=(message_id,),
                )
            )
            await self.disconnected.wait()

    async def scenario() -> tuple[Any, CoalescingClient]:
        nonlocal message_id
        client = CoalescingClient()
        task = asyncio.create_task(runtime_for(store, contract, client).run())
        await asyncio.wait_for(client.queried.wait(), timeout=1)
        message_id = store.send(contract.identity, {"kind": "note", "text": "one more thing"})
        await asyncio.wait_for(client.two_queries.wait(), timeout=1)
        return await asyncio.wait_for(task, timeout=2), client

    outcome, client = asyncio.run(scenario())

    assert client.query_count == 2
    assert outcome.terminal_reason == "turn_complete_without_assignment"
    assert not outcome.error


def test_delegated_task_interim_result_does_not_end_run(store: Store) -> None:
    assignment = scaffold_assignment(store)
    brain = SubBrain(store, assignment.contract, model=assignment.model)

    started = TaskStartedMessage(
        subtype="task_started",
        data={},
        task_id="task-1",
        description="delegated parser work",
        uuid="start-1",
        session_id="session-1",
        task_type="local_agent",
    )
    finished = TaskNotificationMessage(
        subtype="task_notification",
        data={},
        task_id="task-1",
        status="completed",
        output_file="",
        summary="done",
        uuid="done-1",
        session_id="session-1",
    )

    class TaskClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("parser.py", "def parse(text):\n    return text\n")

        async def receive_messages(self):
            await self.queried.wait()
            yield started
            yield result_message(
                structured_output=completed_claim(
                    status="continue",
                    summary="delegated",
                    evidence=(),
                )
            )
            await asyncio.sleep(QUIET * 2)
            assert not self.disconnected.is_set()
            yield finished
            yield result_message(structured_output=completed_claim(), turns=2, cost=0.4)
            await self.disconnected.wait()

    client = TaskClient(None, [])
    outcome = asyncio.run(
        runtime_for(store, assignment.contract, client, assignment=assignment, brain=brain).run()
    )

    assert not outcome.completed
    report = WorkerReport.from_json(
        store.view(assignment.worker).head.read(WORKER_REPORT_PATH) or ""
    )
    assert any("no proof-safe" in reason for reason in report.uncertainty_reasons)
    assert outcome.turns == 2
    assert outcome.cost_usd == 0.4


def test_task_updated_terminal_status_also_clears_task(store: Store) -> None:
    contract = scaffold(store)
    started = TaskStartedMessage(
        subtype="task_started",
        data={},
        task_id="task-1",
        description="workflow",
        uuid="start-1",
        session_id="session-1",
        task_type="local_workflow",
    )
    updated = TaskUpdatedMessage(
        subtype="task_updated",
        data={},
        task_id="task-1",
        patch={"status": "completed"},
        status="completed",
        session_id="session-1",
        uuid="updated-1",
    )
    client = ScriptedClient(None, [started, result_message(), updated, result_message(turns=2)])

    outcome = asyncio.run(runtime_for(store, contract, client).run())

    assert outcome.turns == 2
    assert not outcome.error


def test_activity_after_old_result_requires_a_new_result_before_disconnect(
    store: Store,
) -> None:
    contract = scaffold(store)

    class InjectedTurnClient(ScriptedClient):
        async def receive_messages(self):
            await self.queried.wait()
            yield result_message()
            yield UserMessage("peer asks a question", origin={"kind": "peer"})
            await asyncio.sleep(QUIET * 2)
            assert not self.disconnected.is_set()
            yield result_message(origin={"kind": "human"}, turns=2, cost=0.4)
            await self.disconnected.wait()

    outcome = asyncio.run(runtime_for(store, contract, InjectedTurnClient(None, [])).run())

    assert outcome.turns == 2
    assert not outcome.error


def test_conversation_reset_invalidates_old_completion_and_sums_new_totals(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store)
    brain = SubBrain(store, assignment.contract, model=assignment.model)

    class ResetClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("parser.py", "def parse(text):\n    return text\n")

        async def receive_messages(self):
            await self.queried.wait()
            yield result_message(structured_output=completed_claim(), turns=2, cost=0.5)
            yield ConversationResetMessage(
                new_conversation_id="conversation-2",
                uuid="reset-1",
                session_id="session-1",
            )
            await asyncio.sleep(QUIET * 2)
            assert not self.disconnected.is_set()
            yield result_message(
                session_id="session-2",
                origin={"kind": "human"},
                structured_output=completed_claim(),
                turns=1,
                cost=0.2,
            )
            await self.disconnected.wait()

    outcome = asyncio.run(
        runtime_for(
            store,
            assignment.contract,
            ResetClient(None, []),
            assignment=assignment,
            brain=brain,
        ).run()
    )

    assert outcome.completed
    assert outcome.turns == 3
    assert outcome.cost_usd == pytest.approx(0.7)


def test_budgeted_reset_synchronously_aborts_cli_before_any_await(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store, budget_usd=reset_safe_budget())

    class BudgetResetClient(ScriptedClient):
        def __init__(self) -> None:
            super().__init__(None, [])
            self.aborted = False
            self.interrupted = False

        def abort_now(self) -> None:
            self.calls.append(("abort_now", None))
            self.aborted = True
            self.disconnected.set()

        async def wait_reaped(self) -> bool:
            return self.aborted

        async def interrupt(self) -> None:
            self.interrupted = True
            await asyncio.Event().wait()

        async def receive_messages(self):
            await self.queried.wait()
            yield SystemMessage("init", {"session_id": "session-1"})
            yield auditable_result_message(turns=1, cost=0.5)
            yield ConversationResetMessage(
                new_conversation_id="conversation-2",
                uuid="budget-reset",
                session_id="session-1",
            )
            assert self.aborted
            await self.disconnected.wait()

    client = BudgetResetClient()
    outcome = asyncio.run(
        runtime_for(
            store,
            assignment.contract,
            client,
            assignment=assignment,
        ).run()
    )

    assert not outcome.completed
    assert "provider-side cost guard" in outcome.error
    assert client.aborted
    assert not client.interrupted
    assert [kind for kind, _ in client.calls].index("abort_now") < [
        kind for kind, _ in client.calls
    ].index("disconnect")


def test_budgeted_public_client_factory_is_rejected_before_client_creation(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store, budget_usd=reset_safe_budget())
    made_clients = 0

    def factory(_options: Any) -> ScriptedClient:
        nonlocal made_clients
        made_clients += 1
        return ScriptedClient(None, [auditable_result_message()])

    runtime = WorkerRuntime(
        SubBrain(store, assignment.contract, model=assignment.model),
        QuietMonitor(),
        assignment=assignment,
        client_factory=factory,
        poll_interval=POLL,
        terminal_quiet_period=QUIET,
        shutdown_timeout=1,
    )
    with pytest.raises(ContractMismatch, match="built-in audited SDK client"):
        asyncio.run(runtime.run())

    assert made_clients == 0
    assert not runtime._budget_journal_path().exists()


def test_budgeted_cli_preflight_failure_precedes_client_and_spend_intent(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store, budget_usd=reset_safe_budget())
    runtime = WorkerRuntime(
        SubBrain(store, assignment.contract, model=assignment.model),
        QuietMonitor(),
        assignment=assignment,
    )
    made_clients = 0

    def make_client(_options: Any) -> None:
        nonlocal made_clients
        made_clients += 1

    def reject_cli() -> tuple[Path, dict[str, Any]]:
        raise ContractMismatch("bundled CLI bytes are not in the audited artifact allowlist")

    runtime._make_client = make_client  # type: ignore[method-assign]
    runtime._preflight_budgeted_cli = reject_cli  # type: ignore[method-assign]
    with pytest.raises(ContractMismatch, match="artifact allowlist"):
        asyncio.run(runtime.run())

    assert made_clients == 0
    assert not runtime._budget_journal_path().exists()


def test_budgeted_cli_preflight_requires_checked_in_executable_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import taste.brains.worker_runtime as worker_runtime_module

    monkeypatch.setattr(
        worker_runtime_module,
        "_AUDITED_CLAUDE_CODE_SHA256",
        frozenset({"0" * 64}),
    )
    with pytest.raises(ContractMismatch, match="artifact allowlist"):
        WorkerRuntime._preflight_budgeted_cli()


def test_budgeted_run_is_spend_once_even_if_session_sidecar_is_deleted(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store, budget_usd=reset_safe_budget())

    class BudgetClient(ScriptedClient):
        def abort_now(self) -> None:
            self.calls.append(("abort_now", None))

        async def wait_reaped(self) -> bool:
            return True

    first_client = BudgetClient(None, [auditable_result_message()])
    first = runtime_for(
        store,
        assignment.contract,
        first_client,
        assignment=assignment,
    )
    asyncio.run(first.run())
    journal = first._budget_journal_path()
    binding = first._binding_path()
    assert journal.exists() and binding.exists()
    binding.unlink()
    report_before = store.view(assignment.worker).head.read(WORKER_REPORT_PATH)

    replay_client = BudgetClient(None, [auditable_result_message()])
    replay = runtime_for(
        store,
        assignment.contract,
        replay_client,
        assignment=assignment,
    )
    with pytest.raises(ContractMismatch, match="automatic provider replay"):
        asyncio.run(replay.run())

    assert replay_client.calls == []
    assert store.view(assignment.worker).head.read(WORKER_REPORT_PATH) == report_before


def test_budgeted_reset_without_reap_proof_never_publishes_report(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store, budget_usd=reset_safe_budget())

    class UnreapedClient(ScriptedClient):
        def abort_now(self) -> None:
            self.disconnected.set()

        async def wait_reaped(self) -> bool:
            return False

        async def receive_messages(self):
            await self.queried.wait()
            yield SystemMessage("init", {"session_id": "session-1"})
            yield auditable_result_message(cost=0.5)
            yield ConversationResetMessage(
                new_conversation_id="conversation-2",
                uuid="unreaped-reset",
                session_id="session-1",
            )
            await self.disconnected.wait()

    runtime = runtime_for(
        store,
        assignment.contract,
        UnreapedClient(None, []),
        assignment=assignment,
    )
    with pytest.raises(ShutdownUnconfirmed, match="proven reaped"):
        asyncio.run(runtime.run())

    assert store.view(assignment.worker).head.read(WORKER_REPORT_PATH) is None


def test_budgeted_client_without_sync_kill_boundary_never_receives_query(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store, budget_usd=reset_safe_budget())
    client = ScriptedClient(None, [auditable_result_message()])

    runtime = runtime_for(
        store,
        assignment.contract,
        client,
        assignment=assignment,
    )
    with pytest.raises(ContractMismatch, match="exact built-in SDK client"):
        asyncio.run(runtime.run())

    assert client.calls == []
    assert not runtime._budget_journal_path().exists()


def test_budgeted_missing_process_binding_never_queries_or_reports_known_cost(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store, budget_usd=reset_safe_budget())

    class BoundaryClient(ScriptedClient):
        def abort_now(self) -> None:
            pass

        async def wait_reaped(self) -> bool:
            return True

    client = BoundaryClient(None, [auditable_result_message()])
    runtime = runtime_for(store, assignment.contract, client, assignment=assignment)
    runtime._arm_budgeted_process_boundary = lambda: {}  # type: ignore[method-assign]

    outcome = asyncio.run(runtime.run())

    assert not outcome.completed
    assert outcome.cost_usd is None
    assert [kind for kind, _payload in client.calls] == ["connect", "disconnect"]
    records = runtime._budget_journal_records()
    assert [record.event for record in records] == [
        "connection_intent",
        "connection_outcome",
    ]
    assert records[-1].payload["accounting_unknown"] is True


def test_budget_journal_rejects_out_of_order_duplicate_and_post_outcome_events(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store, budget_usd=reset_safe_budget())
    runtime = runtime_for(
        store,
        assignment.contract,
        ScriptedClient(None, []),
        assignment=assignment,
    )
    runtime.assignment = runtime._validate_durable_input()
    evidence = {
        "claude_agent_sdk_version": "0.2.152",
        "claude_code_version": "2.1.259",
        "cli_sha256": "884baa38fe1a624be25c4a91568bf5a08b5cf4e7d7acf29b7760e3525d964898",
    }
    try:
        runtime._append_budget_journal("connection_intent", {"cli_identity": evidence})
        with pytest.raises(ContractMismatch, match="preceded exact process binding"):
            runtime._append_budget_journal("result_accounting", {})
        runtime._append_budget_journal("process_bound", evidence)
        with pytest.raises(ContractMismatch, match="out of order"):
            runtime._append_budget_journal("process_bound", evidence)
        runtime._append_budget_journal("connection_outcome", {"accounting_unknown": False})
        with pytest.raises(ContractMismatch, match="outcome is not terminal"):
            runtime._append_budget_journal("identity_violation", {})
    finally:
        runtime.brain.close()


def test_budgeted_production_boundary_binds_the_audited_bundled_cli(
    store: Store,
) -> None:
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    assignment = scaffold_assignment(store, budget_usd=reset_safe_budget())
    brain = SubBrain(store, assignment.contract, model=assignment.model)
    runtime = WorkerRuntime(
        brain,
        QuietMonitor(),
        assignment=assignment,
        poll_interval=POLL,
        terminal_quiet_period=QUIET,
        shutdown_timeout=1,
    )
    try:
        cli_path, cli_evidence = runtime._preflight_budgeted_cli()
    except ContractMismatch as exc:
        if "artifact allowlist" in str(exc):
            brain.close()
            pytest.skip("this platform's bundled CLI has not been audited")
        raise
    runtime._budget_cli_path = cli_path
    runtime._budget_cli_evidence = cli_evidence
    transport = SubprocessCLITransport("", brain.options())
    transport._cli_path = str(cli_path)

    class LiveProcess:
        returncode = None

        def kill(self) -> None:
            pass

    transport._process = LiveProcess()  # type: ignore[assignment]

    from claude_agent_sdk import ClaudeSDKClient

    runtime._client = ClaudeSDKClient(options=brain.options())
    runtime._client._transport = transport
    try:
        runtime._arm_budgeted_process_boundary()
        assert runtime._budget_cli_evidence is not None
        assert runtime._budget_cli_evidence["claude_agent_sdk_version"] == "0.2.152"
        assert runtime._budget_cli_evidence["claude_code_version"] == "2.1.259"
        assert runtime._budget_cli_evidence["cli_sha256"] == cli_evidence["cli_sha256"]
    finally:
        brain.close()


def test_budgeted_duck_typed_abort_client_cannot_arm_production_boundary(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store, budget_usd=reset_safe_budget())
    brain = SubBrain(store, assignment.contract, model=assignment.model)
    runtime = WorkerRuntime(brain, QuietMonitor(), assignment=assignment)

    class ClaimedBoundary:
        def abort_now(self) -> None:
            pass

        async def wait_reaped(self) -> bool:
            return True

    runtime._client = ClaimedBoundary()
    try:
        with pytest.raises(ContractMismatch, match="exact built-in SDK client"):
            runtime._arm_budgeted_process_boundary()
    finally:
        brain.close()


def test_deleted_budget_authority_does_not_turn_prior_provider_work_fresh(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store, budget_usd=reset_safe_budget())

    class BudgetClient(ScriptedClient):
        def abort_now(self) -> None:
            pass

        async def wait_reaped(self) -> bool:
            return True

    first = runtime_for(
        store,
        assignment.contract,
        BudgetClient(None, [auditable_result_message()]),
        assignment=assignment,
    )
    asyncio.run(first.run())
    first._budget_journal_path().unlink()
    first._binding_path().unlink()

    replay_client = BudgetClient(None, [auditable_result_message()])
    replay = runtime_for(
        store,
        assignment.contract,
        replay_client,
        assignment=assignment,
    )
    with pytest.raises(ContractMismatch, match="prior provider evidence"):
        asyncio.run(replay.run())

    assert replay_client.calls == []


def test_budgeted_run_id_cannot_select_a_fresh_spend_journal(store: Store) -> None:
    assignment = scaffold_assignment(store, budget_usd=reset_safe_budget())
    client = ScriptedClient(None, [auditable_result_message()])

    with pytest.raises(ContractMismatch, match="derived from the exact durable assignment"):
        asyncio.run(
            runtime_for(
                store,
                assignment.contract,
                client,
                assignment=assignment,
                run_id="forged-fresh-budget-namespace",
            ).run()
        )

    assert client.calls == []


@pytest.mark.parametrize("violation", ["decrease", "uuid-rewrite"])
def test_budgeted_result_accounting_never_moves_backward_or_rewrites_uuid(
    store: Store, violation: str
) -> None:
    assignment = scaffold_assignment(store, budget_usd=reset_safe_budget())
    first = auditable_result_message(cost=0.8, turns=2, uuid="stable-result")
    second = (
        auditable_result_message(cost=0.2, turns=1, uuid="later-result")
        if violation == "decrease"
        else replace(first, result="different bytes under the same UUID")
    )

    class AccountingClient(ScriptedClient):
        def abort_now(self) -> None:
            self.disconnected.set()

        async def wait_reaped(self) -> bool:
            return True

        async def receive_messages(self):
            await self.queried.wait()
            yield SystemMessage("init", {"session_id": "session-1"})
            yield first
            yield second
            await self.disconnected.wait()

    runtime = runtime_for(
        store,
        assignment.contract,
        AccountingClient(None, []),
        assignment=assignment,
    )
    outcome = asyncio.run(runtime.run())

    expected = "decreased" if violation == "decrease" else "different bytes"
    assert expected in outcome.error
    assert outcome.cost_usd is None
    assert any(record.event == "identity_violation" for record in runtime._budget_journal_records())


def test_budgeted_result_rejects_cost_that_disagrees_with_priced_tokens(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store, budget_usd=reset_safe_budget())
    valid = auditable_result_message(cost=0.25)
    usage = {key: dict(value) for key, value in (valid.model_usage or {}).items()}
    usage["claude-sonnet-5"]["costUSD"] = 0.20
    forged = replace(valid, total_cost_usd=0.20, model_usage=usage)

    class MispricedClient(ScriptedClient):
        def abort_now(self) -> None:
            self.disconnected.set()

        async def wait_reaped(self) -> bool:
            return True

    runtime = runtime_for(
        store,
        assignment.contract,
        MispricedClient(None, [forged]),
        assignment=assignment,
    )
    outcome = asyncio.run(runtime.run())

    assert "disagrees with priced token usage" in outcome.error
    assert outcome.cost_usd is None
    assert any(record.event == "identity_violation" for record in runtime._budget_journal_records())


def test_conversation_reset_and_accounting_survive_crashes_and_bind_new_session(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    assignment = scaffold_assignment(store)
    run_id = "reset-crash-recovery"
    interrupted_brain = SubBrain(store, assignment.contract, model=assignment.model)
    interrupted = runtime_for(
        store,
        assignment.contract,
        ScriptedClient(None, []),
        assignment=assignment,
        brain=interrupted_brain,
        run_id=run_id,
    )
    interrupted.assignment = interrupted._validate_durable_input()
    interrupted._bind_session("session-1")
    interrupted._accounting_latest = result_message(turns=2, cost=0.5)
    interrupted._persist_current_accounting()
    marker_path = interrupted._binding_path()
    interrupted_brain.close()

    # The first process stopped after the Result. A reset that is the first
    # message after resume must still fold that exact persisted total forward.
    reset_brain = SubBrain(store, assignment.contract, model=assignment.model)
    reset_runtime = runtime_for(
        store,
        assignment.contract,
        ScriptedClient(None, []),
        assignment=assignment,
        brain=reset_brain,
        run_id=run_id,
    )
    reset_runtime.assignment = reset_runtime._validate_durable_input()
    assert reset_runtime._resume_session_id() == "session-1"
    assert reset_runtime._accounting_latest is not None
    assert reset_runtime._accounting_latest.num_turns == 2
    assert reset_runtime._accounting_latest.total_cost_usd == pytest.approx(0.5)

    original_turn = reset_brain.branch.turn

    def crash_on_reset_journal(**event: Any) -> None:
        if event.get("kind") == "sdk_message":
            raise RuntimeError("simulated crash immediately after reset marker")
        original_turn(**event)

    monkeypatch.setattr(reset_brain.branch, "turn", crash_on_reset_journal)
    with pytest.raises(RuntimeError, match="simulated crash"):
        reset_runtime._record_message(
            ConversationResetMessage(
                new_conversation_id="conversation-2",
                uuid="reset-crash",
                session_id="session-1",
            )
        )
    pending = json.loads(marker_path.read_text(encoding="utf-8"))
    assert pending["reset_pending"] is True
    assert "session_id" not in pending
    assert pending["previous_session_id"] == "session-1"
    assert pending["turns_before_reset"] == 2
    assert pending["cost_usd_before_reset"] == pytest.approx(0.5)
    reset_brain.close()

    replacement_brain = SubBrain(store, assignment.contract, model=assignment.model)

    class ReplacementClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            replacement_brain.branch.write("parser.py", "def parse(text):\n    return text\n")

    replacement = ReplacementClient(
        None,
        [
            result_message(
                session_id="session-2",
                structured_output=completed_claim(),
                turns=1,
                cost=0.2,
            )
        ],
    )
    outcome = asyncio.run(
        runtime_for(
            store,
            assignment.contract,
            replacement,
            assignment=assignment,
            brain=replacement_brain,
            run_id=run_id,
        ).run()
    )

    assert replacement.options.resume is None
    assert outcome.completed
    assert outcome.turns == 3
    assert outcome.cost_usd == pytest.approx(0.7)
    rebound = json.loads(marker_path.read_text(encoding="utf-8"))
    assert "reset_pending" not in rebound
    assert rebound["session_id"] == "session-2"
    assert rebound["previous_session_id"] == "session-1"
    assert rebound["turns_before_reset"] == 2
    assert rebound["cost_usd_before_reset"] == pytest.approx(0.5)


def test_opening_error_remains_sticky_after_later_success(store: Store) -> None:
    assignment = scaffold_assignment(store)
    brain = SubBrain(store, assignment.contract, model=assignment.model)

    class ProducingClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("parser.py", "def parse(text):\n    return text\n")

    client = ProducingClient(
        None,
        [
            result_message(is_error=True, errors=["opening API failed"]),
            result_message(structured_output=completed_claim(), turns=2, cost=0.5),
        ],
    )
    outcome = asyncio.run(
        runtime_for(store, assignment.contract, client, assignment=assignment, brain=brain).run()
    )

    assert not outcome.completed
    report = WorkerReport.from_json(
        store.view(assignment.worker).head.read(WORKER_REPORT_PATH) or ""
    )
    assert any("opening API failed" in reason for reason in report.uncertainty_reasons)
    assert outcome.cost_usd == 0.5, "the latest result is a running accounting total"


def test_adverse_stop_reason_with_no_terminal_reason_is_not_success(store: Store) -> None:
    assignment = scaffold_assignment(store)
    brain = SubBrain(store, assignment.contract, model=assignment.model)

    class ProducingClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("parser.py", "def parse(text):\n    return text\n")

    client = ProducingClient(
        None,
        [
            result_message(
                terminal_reason=None,
                stop_reason="max_turns",
                structured_output=completed_claim(),
            )
        ],
    )
    outcome = asyncio.run(
        runtime_for(store, assignment.contract, client, assignment=assignment, brain=brain).run()
    )

    assert not outcome.completed
    assert outcome.terminal_reason == "sdk_error"
    report = WorkerReport.from_json(
        store.view(assignment.worker).head.read(WORKER_REPORT_PATH) or ""
    )
    assert any("max_turns" in reason for reason in report.uncertainty_reasons)


def test_unexpected_abort_is_a_terminal_sdk_failure(store: Store) -> None:
    contract = scaffold(store)
    client = ScriptedClient(None, [result_message(terminal_reason="aborted_streaming")])

    outcome = asyncio.run(runtime_for(store, contract, client).run())

    assert not outcome.completed
    assert outcome.terminal_reason == "sdk_error"
    assert "unexpected SDK abort" in (
        store.view(contract.identity).head.read(WORKER_REPORT_PATH) or ""
    )


def test_one_expected_abort_cannot_excuse_a_second_abort(store: Store) -> None:
    contract = scaffold(store)
    client = ScriptedClient(
        None,
        [
            result_message(terminal_reason="aborted_streaming"),
            result_message(terminal_reason="aborted_streaming", turns=2),
        ],
    )
    runtime = runtime_for(store, contract, client)
    runtime._expected_abort_tokens = 1

    outcome = asyncio.run(runtime.run())

    assert outcome.terminal_reason == "sdk_error"
    assert "unexpected SDK abort" in outcome.error or outcome.error == ""
    report = json.loads(store.view(contract.identity).head.read(WORKER_REPORT_PATH) or "null")
    assert any("unexpected SDK abort" in reason for reason in report["uncertainty"])


def test_bypass_result_without_any_terminal_reason_cannot_end_host_run(
    store: Store,
) -> None:
    contract = scaffold(store)
    client = ScriptedClient(
        None,
        [
            result_message(terminal_reason=None, stop_reason=None),
            result_message(origin={"kind": "human"}, turns=2, cost=0.4),
        ],
    )

    outcome = asyncio.run(runtime_for(store, contract, client).run())

    assert outcome.turns == 2
    assert outcome.cost_usd == 0.4
    assert not outcome.error


def test_incidental_peer_result_cannot_mask_human_prompt_error(store: Store) -> None:
    contract = scaffold(store)
    client = ScriptedClient(
        None,
        [
            result_message(is_error=True, errors=["human prompt failed"], cost=0.2),
            result_message(origin={"kind": "peer"}, turns=9, cost=9.0),
            result_message(origin={"kind": "human"}, turns=10, cost=9.2),
        ],
    )

    outcome = asyncio.run(runtime_for(store, contract, client).run())

    report = json.loads(store.view(contract.identity).head.read(WORKER_REPORT_PATH) or "null")
    assert any("human prompt failed" in item for item in report["uncertainty"])
    assert outcome.cost_usd == 9.2
    assert outcome.turns == 10


def test_live_inbox_is_acked_only_after_a_relevant_success_result(store: Store) -> None:
    contract = scaffold(store)
    message_id = ""

    class InboxClient(ScriptedClient):
        def __init__(self) -> None:
            super().__init__(None, [])
            self.inbox_queried = asyncio.Event()

        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            if prompt.startswith("[inbox "):
                self.inbox_queried.set()

        async def receive_messages(self):
            await self.inbox_queried.wait()
            yield result_message(
                structured_output=completed_claim(
                    status="continue",
                    summary="handled central message",
                    evidence=(),
                    accepted_inbox_ids=(message_id,),
                )
            )
            await self.disconnected.wait()

    async def scenario() -> tuple[Any, str, InboxClient]:
        nonlocal message_id
        client = InboxClient()
        task = asyncio.create_task(runtime_for(store, contract, client).run())
        await asyncio.wait_for(client.queried.wait(), timeout=1)
        message_id = store.send(
            contract.identity,
            {"kind": "artifact_request", "need": "parser-fixture"},
            sender="central",
        )
        await asyncio.wait_for(client.inbox_queried.wait(), timeout=1)
        return await asyncio.wait_for(task, timeout=2), message_id, client

    outcome, message_id, client = asyncio.run(scenario())

    assert not outcome.completed
    assert store.inbox(contract.identity) == []
    prompts = [
        body for kind, body in client.calls if kind == "query" and body.startswith("[inbox ")
    ]
    assert len(prompts) == 1 and message_id[:10] in prompts[0]
    accepted = [
        event
        for event in store.view(contract.identity).head.transcript.turns
        if event.get("kind") == "inbox_accepted"
    ]
    assert [event["message_id"] for event in accepted] == [message_id]


def test_production_typed_inbox_accepts_only_exact_current_generation(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store)
    communicator = Communicator(store)
    item = communicator.send(
        Message.create(
            idempotency_key="current-instruction",
            kind="instruction",
            sender="central",
            recipient=assignment.worker,
            generation=assignment.generation,
            payload={"task": "include the parser fixture"},
        )
    )
    client = ScriptedClient(
        None,
        [
            result_message(
                structured_output=completed_claim(
                    status="continue",
                    evidence=(),
                    accepted_inbox_ids=(item.inbox_id,),
                )
            )
        ],
    )
    runtime = runtime_for(
        store,
        assignment.contract,
        client,
        assignment=assignment,
        run_id="typed-current-run",
        communicator=communicator,
    )

    asyncio.run(runtime.run())

    assert communicator.pending(assignment.worker) == ()
    accepted = [
        event for event in runtime.brain._recorded_turns() if event.get("kind") == "inbox_accepted"
    ]
    assert accepted[-1]["semantic_message_id"] == item.message.message_id
    assert accepted[-1]["message_generation"] == assignment.generation


def test_typed_inbox_recovers_crash_after_acceptance_before_cursor(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    assignment = scaffold_assignment(store)
    communicator = Communicator(store)
    item = communicator.send(
        Message.create(
            idempotency_key="crash-boundary",
            kind="instruction",
            sender="central",
            recipient=assignment.worker,
            generation=assignment.generation,
        )
    )
    runtime = runtime_for(
        store,
        assignment.contract,
        ScriptedClient(None),
        assignment=assignment,
        run_id="typed-crash-run",
        communicator=communicator,
    )
    messages = runtime._typed_pending_envelopes()

    def crash_after_acceptance(_branch: str, _message_id: str) -> None:
        raise RuntimeError("crash after durable acceptance")

    async def interrupted_acceptance() -> None:
        await runtime._register_inbox(messages, epoch=1)
        await runtime._accept_deliveries(
            {
                "accepted_inbox_ids": [item.inbox_id],
                "accepted_verdicts": {},
            },
            submitted_epoch=1,
        )

    with monkeypatch.context() as patcher:
        patcher.setattr(
            store,
            "mark_inbox_seen",
            crash_after_acceptance,
        )
        with pytest.raises(RuntimeError, match="crash after durable acceptance"):
            asyncio.run(interrupted_acceptance())

    assert [pending.inbox_id for pending in communicator.pending(assignment.worker)] == [
        item.inbox_id
    ]
    runtime.brain.close()

    recovered = runtime_for(
        store,
        assignment.contract,
        ScriptedClient(None),
        assignment=assignment,
        run_id="typed-crash-run",
        communicator=Communicator(store),
    )
    try:
        recovered._reconcile_delivery_markers()
        assert communicator.pending(assignment.worker) == ()
        accepted = [
            event
            for event in recovered.brain._recorded_turns()
            if event.get("kind") == "inbox_accepted" and event.get("message_id") == item.inbox_id
        ]
        assert len(accepted) == 1
    finally:
        recovered.brain.close()


def test_production_typed_inbox_durably_retires_stale_oldest_message(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store, generation=2)
    communicator = Communicator(store)
    item = communicator.send(
        Message.create(
            idempotency_key="stale-instruction",
            kind="instruction",
            sender="central",
            recipient=assignment.worker,
            generation=1,
            payload={"task": "obsolete"},
        )
    )
    client = ScriptedClient(None)
    runtime = runtime_for(
        store,
        assignment.contract,
        client,
        assignment=assignment,
        run_id="typed-stale-run",
        communicator=communicator,
    )

    asyncio.run(runtime.run())

    assert communicator.pending(assignment.worker) == ()
    retired = [
        event
        for event in runtime.brain._recorded_turns()
        if event.get("kind") == "inbox_retired_stale"
    ]
    assert retired[-1]["message_id"] == item.inbox_id
    assert retired[-1]["current_generation"] == assignment.generation
    assert not any(item.inbox_id in body for kind, body in client.calls if kind == "query")


def test_production_typed_inbox_leaves_future_message_for_future_worker(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store)
    communicator = Communicator(store)
    item = communicator.send(
        Message.create(
            idempotency_key="future-instruction",
            kind="instruction",
            sender="central",
            recipient=assignment.worker,
            generation=2,
            payload={"task": "not for this run"},
        )
    )
    client = ScriptedClient(None)
    runtime = runtime_for(
        store,
        assignment.contract,
        client,
        assignment=assignment,
        run_id="typed-future-run",
        communicator=communicator,
    )

    asyncio.run(runtime.run())

    assert [pending.inbox_id for pending in communicator.pending(assignment.worker)] == [
        item.inbox_id
    ]
    assert not any(item.inbox_id in body for kind, body in client.calls if kind == "query")


def test_later_inbox_echo_cannot_acknowledge_an_earlier_gap(store: Store) -> None:
    contract = scaffold(store)
    runtime = runtime_for(store, contract, ScriptedClient(None))
    first_id = store.send(contract.identity, {"kind": "first"}, sender="central")
    second_id = store.send(contract.identity, {"kind": "second"}, sender="central")
    messages = store.inbox(contract.identity)

    async def scenario() -> None:
        await runtime._register_inbox(messages, epoch=1)
        await runtime._accept_deliveries(
            {
                "accepted_inbox_ids": [second_id],
                "accepted_verdicts": {},
            },
            submitted_epoch=1,
        )
        assert [message["id"] for message in store.inbox(contract.identity)] == [
            first_id,
            second_id,
        ]

        # The later acceptance evidence is retained.  Once the missing oldest
        # message is accepted, both form one contiguous prefix and settle in
        # their durable inbox order.
        await runtime._accept_deliveries(
            {
                "accepted_inbox_ids": [first_id],
                "accepted_verdicts": {},
            },
            submitted_epoch=1,
        )

    try:
        asyncio.run(scenario())
        assert store.inbox(contract.identity) == []
        accepted = [
            event["message_id"]
            for event in runtime.brain._recorded_turns()
            if event.get("kind") == "inbox_accepted"
        ]
        assert accepted == [first_id, second_id]
    finally:
        runtime.brain.close()


def test_recovery_does_not_advance_cursor_across_an_unaccepted_gap(store: Store) -> None:
    contract = scaffold(store)
    runtime = runtime_for(store, contract, ScriptedClient(None))
    first_id = store.send(contract.identity, {"kind": "first"}, sender="central")
    second_id = store.send(contract.identity, {"kind": "second"}, sender="central")
    runtime.brain.branch.turn(kind="inbox_accepted", message_id=second_id)

    try:
        runtime._reconcile_delivery_markers()
        assert [message["id"] for message in store.inbox(contract.identity)] == [
            first_id,
            second_id,
        ]
    finally:
        runtime.brain.close()


def test_injected_task_result_cannot_ack_a_host_inbox_prompt(store: Store) -> None:
    contract = scaffold(store)
    message_id = ""

    class InjectedFirstClient(ScriptedClient):
        def __init__(self) -> None:
            super().__init__(None, [])
            self.inbox_queried = asyncio.Event()

        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            if prompt.startswith("[inbox "):
                self.inbox_queried.set()

        async def receive_messages(self):
            await self.inbox_queried.wait()
            echoed = completed_claim(
                status="continue",
                summary="task claims it handled the message",
                evidence=(),
                accepted_inbox_ids=(message_id,),
            )
            yield result_message(
                origin={"kind": "task-notification"},
                structured_output=echoed,
            )
            await asyncio.sleep(QUIET * 2)
            assert [item["id"] for item in store.inbox(contract.identity)] == [message_id]
            yield result_message(
                origin={"kind": "human"},
                turns=2,
                structured_output=completed_claim(
                    status="continue",
                    summary="host turn handled the message",
                    evidence=(),
                    accepted_inbox_ids=(message_id,),
                ),
            )
            await self.disconnected.wait()

    async def scenario() -> Any:
        nonlocal message_id
        client = InjectedFirstClient()
        task = asyncio.create_task(runtime_for(store, contract, client).run())
        await asyncio.wait_for(client.queried.wait(), timeout=1)
        message_id = store.send(contract.identity, {"kind": "must-reach-human-turn"})
        return await asyncio.wait_for(task, timeout=2)

    asyncio.run(scenario())

    assert store.inbox(contract.identity) == []


def test_success_result_without_exact_echo_leaves_inbox_for_replay(
    store: Store,
) -> None:
    contract = scaffold(store)

    class NoEchoClient(ScriptedClient):
        def __init__(self) -> None:
            super().__init__(None, [])
            self.inbox_queried = asyncio.Event()

        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            if prompt.startswith("[inbox "):
                self.inbox_queried.set()

        async def receive_messages(self):
            await self.inbox_queried.wait()
            yield result_message(
                structured_output=completed_claim(
                    status="continue",
                    summary="did not acknowledge the message",
                    evidence=(),
                )
            )
            await self.disconnected.wait()

    async def scenario() -> str:
        client = NoEchoClient()
        task = asyncio.create_task(runtime_for(store, contract, client).run())
        await asyncio.wait_for(client.queried.wait(), timeout=1)
        message_id = store.send(contract.identity, {"kind": "must-be-echoed"})
        await asyncio.wait_for(client.inbox_queried.wait(), timeout=1)
        await asyncio.sleep(QUIET * 3)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return message_id

    message_id = asyncio.run(scenario())

    assert [item["id"] for item in store.inbox(contract.identity)] == [message_id]


def test_query_pipe_write_without_result_leaves_inbox_for_replay(store: Store) -> None:
    contract = scaffold(store)

    class NoResultClient(ScriptedClient):
        def __init__(self) -> None:
            super().__init__(None, [])
            self.inbox_queried = asyncio.Event()

        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            if prompt.startswith("[inbox "):
                self.inbox_queried.set()

        async def receive_messages(self):
            await self.queried.wait()
            await self.disconnected.wait()
            if False:  # pragma: no cover - keep this an async generator
                yield None

    async def scenario() -> tuple[str, NoResultClient]:
        client = NoResultClient()
        task = asyncio.create_task(runtime_for(store, contract, client).run())
        await asyncio.wait_for(client.queried.wait(), timeout=1)
        message_id = store.send(contract.identity, {"kind": "do-not-lose"})
        await asyncio.wait_for(client.inbox_queried.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return message_id, client

    message_id, _ = asyncio.run(scenario())

    assert [message["id"] for message in store.inbox(contract.identity)] == [message_id]
    turns = store.view(contract.identity).pending_turns()
    assert any(event.get("kind") == "inbox_submitted" for event in turns)
    assert not any(event.get("kind") == "inbox_accepted" for event in turns)


def test_queued_old_result_cannot_ack_a_later_blocked_inbox_query(store: Store) -> None:
    contract = scaffold(store)

    class RacingClient(ScriptedClient):
        def __init__(self) -> None:
            super().__init__(None, [])
            self.inbox_write_started = asyncio.Event()
            self.release_inbox_write = asyncio.Event()
            self.old_result_yielded = asyncio.Event()

        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            if prompt.startswith("[inbox "):
                self.inbox_write_started.set()
                await self.release_inbox_write.wait()

        async def receive_messages(self):
            await self.inbox_write_started.wait()
            yield result_message()  # belongs to opening; inbox write has not returned
            self.old_result_yielded.set()
            await self.disconnected.wait()

    async def scenario() -> str:
        client = RacingClient()
        task = asyncio.create_task(runtime_for(store, contract, client).run())
        await asyncio.wait_for(client.queried.wait(), timeout=1)
        message_id = store.send(contract.identity, {"kind": "causality-race"})
        await asyncio.wait_for(client.inbox_write_started.wait(), timeout=1)
        await asyncio.wait_for(client.old_result_yielded.wait(), timeout=1)
        await asyncio.sleep(0)
        client.release_inbox_write.set()
        await asyncio.sleep(QUIET * 3)
        assert not task.done(), "the old opening result was attributed to the inbox query"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return message_id

    message_id = asyncio.run(scenario())

    assert [item["id"] for item in store.inbox(contract.identity)] == [message_id]


def test_hallucinated_verdict_count_cannot_ack_a_later_verdict(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store)
    brain = SubBrain(store, assignment.contract, model=assignment.model)
    target = brain.branch.head
    store.judge(
        target,
        Verdict(
            "unknown",
            by="monitor/worker-1",
            detail="first warning",
            failure_class="drifting",
        ),
    )

    class HallucinatingClient(ScriptedClient):
        def __init__(self) -> None:
            super().__init__(None, [])
            self.result_consumed = asyncio.Event()

        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("parser.py", "def parse(text):\n    return text\n")

        async def receive_messages(self):
            await self.queried.wait()
            yield result_message(
                structured_output=completed_claim(accepted_verdicts={target.id: 2})
            )
            self.result_consumed.set()
            await self.disconnected.wait()

    async def scenario() -> None:
        client = HallucinatingClient()
        task = asyncio.create_task(
            runtime_for(
                store,
                assignment.contract,
                client,
                assignment=assignment,
                brain=brain,
            ).run()
        )
        await asyncio.wait_for(client.result_consumed.wait(), timeout=1)
        await asyncio.sleep(QUIET * 2)
        store.judge(
            target,
            Verdict(
                "unknown",
                by="monitor/worker-1",
                detail="second warning",
                failure_class="drifting",
            ),
        )
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    reopened = store.branch(assignment.worker)
    try:
        assert [verdict.detail for verdict in reopened.unacked_verdicts()] == [
            "first warning",
            "second warning",
        ]
    finally:
        reopened.close()


def test_session_resume_is_bound_to_exact_run_not_newest_mtime(store: Store) -> None:
    contract = scaffold(store)
    captured: list[Any] = []

    def factory(options: Any) -> ScriptedClient:
        captured.append(options)
        return ScriptedClient(options, [result_message(session_id="bound-session")])

    first = WorkerRuntime(
        SubBrain(store, contract),
        QuietMonitor(),
        run_id="run-exact",
        client_factory=factory,
        poll_interval=POLL,
        terminal_quiet_period=QUIET,
        shutdown_timeout=1,
    )
    asyncio.run(first.run())
    second = WorkerRuntime(
        SubBrain(store, contract),
        QuietMonitor(),
        run_id="run-exact",
        client_factory=factory,
        poll_interval=POLL,
        terminal_quiet_period=QUIET,
        shutdown_timeout=1,
    )
    asyncio.run(second.run())

    assert captured[0].resume is None
    assert captured[1].resume == "bound-session"


def test_revised_assignment_bytes_never_resume_superseded_sdk_context(
    store: Store,
) -> None:
    original = scaffold_assignment(store)
    first_brain = SubBrain(store, original.contract, model=original.model)

    class FirstClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            first_brain.branch.write("parser.py", "old assignment output\n")

    first_client = FirstClient(
        None,
        [
            result_message(
                session_id="old-assignment-session",
                structured_output=completed_claim(),
            )
        ],
    )
    asyncio.run(
        runtime_for(
            store,
            original.contract,
            first_client,
            assignment=original,
            brain=first_brain,
        ).run()
    )

    revised_contract = replace(original.contract, outputs=("parser-v2.py",))
    revised = replace(
        original,
        contract=revised_contract,
        contract_digest=contract_digest(revised_contract),
        outputs=(ArtifactSpec("parser-v2", "parser-v2.py"),),
    )
    installer = SubBrain(store, revised.contract, model=revised.model)
    installer.branch.write(CONTRACT_PATH, revised.contract.to_json())
    installer.branch.write(ASSIGNMENT_PATH, revised.to_json())
    installer.checkpoint("central revised the exact assignment")
    installer.close()
    second_brain = SubBrain(store, revised.contract, model=revised.model)

    class SecondClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            second_brain.branch.write("parser-v2.py", "new assignment output\n")

    second_client = SecondClient(
        None,
        [
            result_message(
                session_id="new-assignment-session",
                structured_output=completed_claim(),
            )
        ],
    )
    asyncio.run(
        runtime_for(
            store,
            revised.contract,
            second_client,
            assignment=revised,
            brain=second_brain,
        ).run()
    )

    assert first_client.options.resume is None
    assert second_client.options.resume is None


@pytest.mark.parametrize("run_id", ["", "x" * 257])
def test_invalid_caller_run_id_is_rejected_before_sdk_start(store: Store, run_id: str) -> None:
    contract = scaffold(store)
    client = ScriptedClient(None)

    with pytest.raises(ContractMismatch, match="run_id"):
        asyncio.run(runtime_for(store, contract, client, run_id=run_id).run())

    assert client.calls == []


def test_typed_assignment_pins_outputs_and_requires_explicit_claim(store: Store) -> None:
    assignment = scaffold_assignment(store)
    brain = SubBrain(store, assignment.contract, model=assignment.model)

    class ProducingClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("parser.py", "def parse(text):\n    return text\n")

    client = ProducingClient(None, [result_message(structured_output=completed_claim())])
    outcome = asyncio.run(
        runtime_for(
            store,
            assignment.contract,
            client,
            assignment=assignment,
            brain=brain,
            run_id="run-build-parser-0",
        ).run()
    )

    assert outcome.completed
    assert client.options.output_format == {"type": "json_schema", "schema": WORKER_RESULT_SCHEMA}
    assert set(client.options.disallowed_tools or ()) >= {"Agent", "Task"}
    opening = next(body for kind, body in client.calls if kind == "query")
    assert assignment.to_json().strip() in opening
    report_head = store.view(assignment.worker).head
    report = WorkerReport.from_json(report_head.read(WORKER_REPORT_PATH) or "")
    assert report.final_state_id == report_head.meta.parents[0]
    assert report.completed and not report.uncertain
    assert report.outputs[0].blob_id == store.state(report.final_state_id).blob("parser.py")


def test_worker_report_cost_sums_target_and_monitor_billed_cost(store: Store) -> None:
    assignment = scaffold_assignment(store)
    brain = SubBrain(store, assignment.contract, model=assignment.model)

    class ProducingClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("parser.py", "def parse(text):\n    return text\n")

    monitor = QuietMonitor(model_calls=2, cost_known=True, cost_usd=0.4)
    outcome = asyncio.run(
        runtime_for(
            store,
            assignment.contract,
            ProducingClient(
                None,
                [result_message(cost=0.25, structured_output=completed_claim())],
            ),
            monitor,
            assignment=assignment,
            brain=brain,
        ).run()
    )

    report = WorkerReport.from_json(
        store.view(assignment.worker).head.read(WORKER_REPORT_PATH) or ""
    )
    assert outcome.cost_usd == pytest.approx(0.65)
    assert report.cost_usd == pytest.approx(0.65)
    assert report.metadata["monitor"]["model_calls"] == 2
    assert report.metadata["monitor"]["cost_usd"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "failure",
    ["target-unknown", "monitor-unknown", "monitor-missing", "monitor-malformed", "assessment"],
)
def test_worker_report_cost_fails_closed_on_any_accounting_uncertainty(
    store: Store,
    failure: str,
) -> None:
    assignment = scaffold_assignment(store)
    brain = SubBrain(store, assignment.contract, model=assignment.model)

    class ProducingClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("parser.py", "def parse(text):\n    return text\n")

    monitor = QuietMonitor(model_calls=1, cost_known=True, cost_usd=0.4)
    if failure == "monitor-unknown":
        monitor.cost_known = False
        monitor.cost_usd = None
    elif failure in {"monitor-missing", "monitor-malformed"}:
        original_report = monitor.report

        def broken_report() -> dict[str, Any]:
            report = original_report()
            if failure == "monitor-missing":
                report.pop("cost_known")
            else:
                report["cost_usd"] = "not-a-number"
            return report

        monitor.report = broken_report  # type: ignore[method-assign]
    elif failure == "assessment":

        async def failed_assessment(state: Any, *, context: Any) -> Any:
            raise RuntimeError("monitor provider failed")

        monitor.certify_terminal = failed_assessment  # type: ignore[method-assign]

    target_cost = None if failure == "target-unknown" else 0.25
    outcome = asyncio.run(
        runtime_for(
            store,
            assignment.contract,
            ProducingClient(
                None,
                [
                    result_message(
                        cost=target_cost,
                        structured_output=completed_claim(),
                    )
                ],
            ),
            monitor,
            assignment=assignment,
            brain=brain,
        ).run()
    )

    report = WorkerReport.from_json(
        store.view(assignment.worker).head.read(WORKER_REPORT_PATH) or ""
    )
    assert outcome.cost_usd is None
    assert report.cost_usd is None


def test_terminal_monitor_rejection_defeats_otherwise_valid_completion(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store)
    brain = SubBrain(store, assignment.contract, model=assignment.model)

    class ProducingClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("parser.py", "def parse(text):\n    return text\n")

    class RejectingMonitor(QuietMonitor):
        async def certify_terminal(self, state: Any, *, context: Any):
            durable = Contract.from_json(state.read("contract.json") or "")
            return SimpleNamespace(
                state_id=state.id,
                contract_digest=contract_digest(durable),
                acceptable=False,
                failure="",
                judgement=SimpleNamespace(
                    severity=SimpleNamespace(value="wrong"),
                ),
            )

        def report(self) -> dict[str, Any]:
            return {**super().report(), "current": "wrong", "current_state": "pinned"}

    outcome = asyncio.run(
        runtime_for(
            store,
            assignment.contract,
            ProducingClient(None, [result_message(structured_output=completed_claim())]),
            RejectingMonitor(),
            assignment=assignment,
            brain=brain,
        ).run()
    )

    assert not outcome.completed
    assert outcome.terminal_reason == "monitor_rejected"


def test_real_monitor_certifies_the_exact_work_checkpoint(store: Store) -> None:
    assignment = scaffold_assignment(store)
    brain = SubBrain(store, assignment.contract, model=assignment.model)

    class FineJudge:
        def __call__(self, contract: Any, batch: Any, view: Any) -> Judgement:
            return Judgement(Severity.FINE, "progress is on contract", cost_usd=0.0)

        def judge_terminal(
            self,
            contract: Any,
            state: Any,
            context: Any,
            findings: list[dict[str, Any]],
        ) -> TerminalDecision:
            return TerminalDecision(
                Judgement(
                    Severity.FINE,
                    "the exact immutable output satisfies the assignment",
                    evidence=(state.id,),
                    cost_usd=0.0,
                ),
                resolved_finding_ids=tuple(item["id"] for item in findings),
            )

    class ProducingClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("parser.py", "def parse(text):\n    return text\n")

    monitor = MonitorBrain(store, assignment.contract, FineJudge(), batch_size=100)
    outcome = asyncio.run(
        runtime_for(
            store,
            assignment.contract,
            ProducingClient(None, [result_message(structured_output=completed_claim())]),
            monitor,
            assignment=assignment,
            brain=brain,
        ).run()
    )

    assert outcome.completed
    report = WorkerReport.from_json(
        store.view(assignment.worker).head.read(WORKER_REPORT_PATH) or ""
    )
    assessment = report.metadata["monitor"]["terminal_assessment"]
    assert assessment["acceptable"] is True
    assert assessment["state_id"] == report.final_state_id


def test_a_live_drifting_verdict_is_recorded_without_touching_the_worker(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store)
    brain = SubBrain(store, assignment.contract, model=assignment.model)

    class CorrectingJudge:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, contract: Any, batch: Any, view: Any) -> Judgement:
            self.calls += 1
            if self.calls == 1:
                return Judgement(
                    Severity.DRIFTING,
                    "check the parser output before finishing",
                    suggestion="verify the exact file",
                    cost_usd=0.0,
                )
            return Judgement(Severity.FINE, "correction is on track", cost_usd=0.0)

        def judge_terminal(
            self,
            contract: Any,
            state: Any,
            context: Any,
            findings: list[dict[str, Any]],
        ) -> TerminalDecision:
            return TerminalDecision(
                Judgement(Severity.FINE, "the drift was corrected", cost_usd=0.0),
                resolved_finding_ids=tuple(item["id"] for item in findings),
            )

    class ProducingClient(ScriptedClient):
        """Works and finishes. Nothing the monitor thinks reaches it."""

        def __init__(self) -> None:
            super().__init__(None, [result_message(structured_output=completed_claim())])

        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("parser.py", "def parse(text):\n    return text\n")

    client = ProducingClient()
    monitor = MonitorBrain(
        store,
        assignment.contract,
        CorrectingJudge(),
        batch_size=1,
    )
    outcome = asyncio.run(
        runtime_for(
            store,
            assignment.contract,
            client,
            monitor,
            assignment=assignment,
            brain=brain,
        ).run()
    )

    assert outcome.completed
    report = WorkerReport.from_json(
        store.view(assignment.worker).head.read(WORKER_REPORT_PATH) or ""
    )
    # The verdict is in the report the central brain reads, which is the whole
    # point of judging: it decides what to do about the drift, not the monitor.
    assert report.metadata["monitor"]["worst"] == "drifting"
    assert report.metadata["monitor"]["terminal_assessment"]["acceptable"] is True
    assert not report.uncertain

    # Nothing was said to the worker. Every prompt it saw came from the host.
    assert not any(
        isinstance(prompt, str) and prompt.startswith("[monitor]")
        for name, prompt in client.calls
        if name == "query"
    )
    assert not [
        event
        for event in store.view(assignment.worker).head.transcript.turns
        if event.get("kind") in {"monitor_feedback_submitted", "monitor_feedback_accepted"}
    ]

    # And because nothing was delivered, nothing was acknowledged in-run: the
    # drifting verdict is still waiting on the branch for whoever reads next.
    reopened = store.branch(assignment.worker)
    try:
        assert any(
            verdict.failure_class == "drifting" for verdict in reopened.unacked_verdicts()
        )
    finally:
        reopened.close()


def test_exact_fine_terminal_assessment_can_resolve_historical_drift(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store)
    brain = SubBrain(store, assignment.contract, model=assignment.model)

    class ProducingClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("parser.py", "def parse(text):\n    return text\n")

    class CorrectedMonitor(QuietMonitor):
        def report(self) -> dict[str, Any]:
            return {
                **super().report(),
                "worst": "drifting",
                "current": "fine",
                "current_state": "pinned",
            }

    outcome = asyncio.run(
        runtime_for(
            store,
            assignment.contract,
            ProducingClient(None, [result_message(structured_output=completed_claim())]),
            CorrectedMonitor(),
            assignment=assignment,
            brain=brain,
        ).run()
    )

    assert outcome.completed
    report = WorkerReport.from_json(
        store.view(assignment.worker).head.read(WORKER_REPORT_PATH) or ""
    )
    assert report.monitor_severity == "fine"
    assert report.metadata["monitor"]["worst"] == "drifting"
    assert not report.uncertain
    assert not report.uncertainty_reasons


def test_max_length_assignment_id_still_produces_bounded_report_ids(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store, assignment_id="a" * 256)
    brain = SubBrain(store, assignment.contract, model=assignment.model)

    class ProducingClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("parser.py", "def parse(text):\n    return text\n")

    outcome = asyncio.run(
        runtime_for(
            store,
            assignment.contract,
            ProducingClient(None, [result_message(structured_output=completed_claim())]),
            assignment=assignment,
            brain=brain,
        ).run()
    )

    assert outcome.completed
    report = WorkerReport.from_json(
        store.view(assignment.worker).head.read(WORKER_REPORT_PATH) or ""
    )
    assert report.assignment_id == "a" * 256
    assert report.report_id.startswith("worker-report.")
    assert report.run_id.startswith("worker-run.")
    assert len(report.report_id) < 256
    assert len(report.run_id) < 256


@pytest.mark.parametrize("output", ["contract.json", ".git/config"])
def test_assignment_with_private_output_is_rejected_before_sdk_start(
    store: Store, output: str
) -> None:
    assignment = scaffold_assignment(store, output=output)
    client = ScriptedClient(None)

    with pytest.raises(ContractMismatch, match="invalid product path"):
        asyncio.run(
            runtime_for(
                store,
                assignment.contract,
                client,
                assignment=assignment,
            ).run()
        )

    assert client.calls == []


@pytest.mark.parametrize(
    ("contract_field", "paths", "match"),
    [
        ("inputs", ("undeclared-input.txt",), "contract inputs"),
        ("outputs", ("different-output.py",), "contract outputs"),
    ],
)
def test_runtime_rejects_contract_and_structured_io_disagreement_before_sdk(
    store: Store,
    contract_field: str,
    paths: tuple[str, ...],
    match: str,
) -> None:
    assignment = scaffold_assignment(store)
    contract = replace(assignment.contract, **{contract_field: paths})
    mismatched = replace(
        assignment,
        contract=contract,
        contract_digest=contract_digest(contract),
    )
    branch = store.branch(assignment.worker)
    branch.write(CONTRACT_PATH, contract.to_json())
    branch.write(ASSIGNMENT_PATH, mismatched.to_json())
    branch.checkpoint("durable mismatched contract and typed artifacts")
    branch.close()
    client = ScriptedClient(None)

    with pytest.raises(ContractMismatch, match=match):
        asyncio.run(
            runtime_for(
                store,
                contract,
                client,
                assignment=mismatched,
            ).run()
        )

    assert client.calls == []


def test_symbolic_assignment_base_is_rejected_by_typed_record_schema(
    store: Store,
) -> None:
    original = scaffold_assignment(store)

    with pytest.raises(ValueError, match=r"full lowercase.*object ID"):
        replace(original, base_state_id="HEAD")


def test_missing_required_output_defeats_a_completed_claim(store: Store) -> None:
    assignment = scaffold_assignment(store, output="missing.py")
    client = ScriptedClient(None, [result_message(structured_output=completed_claim())])
    outcome = asyncio.run(
        runtime_for(store, assignment.contract, client, assignment=assignment).run()
    )

    assert not outcome.completed
    assert outcome.terminal_reason == "output_validation_error"
    report = WorkerReport.from_json(
        store.view(assignment.worker).head.read(WORKER_REPORT_PATH) or ""
    )
    assert any("missing.py" in reason for reason in report.uncertainty_reasons)


def test_directory_output_is_rejected_as_a_product_artifact(store: Store) -> None:
    assignment = scaffold_assignment(store, output="pkg")
    brain = SubBrain(store, assignment.contract, model=assignment.model)

    class ProducingClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("pkg/parser.py", "def parse(text):\n    return text\n")

    outcome = asyncio.run(
        runtime_for(
            store,
            assignment.contract,
            ProducingClient(None, [result_message(structured_output=completed_claim())]),
            assignment=assignment,
            brain=brain,
        ).run()
    )

    assert not outcome.completed
    assert outcome.terminal_reason == "output_validation_error"
    report = WorkerReport.from_json(
        store.view(assignment.worker).head.read(WORKER_REPORT_PATH) or ""
    )
    assert any("directory" in reason for reason in report.uncertainty_reasons)


def test_later_relevant_result_without_claim_invalidates_earlier_completion(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store)
    brain = SubBrain(store, assignment.contract, model=assignment.model)

    class ProducingClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("parser.py", "def parse(text):\n    return text\n")

    client = ProducingClient(
        None,
        [
            result_message(structured_output=completed_claim()),
            result_message(origin={"kind": "auto-continuation"}, turns=2, cost=0.5),
        ],
    )
    outcome = asyncio.run(
        runtime_for(
            store,
            assignment.contract,
            client,
            assignment=assignment,
            brain=brain,
        ).run()
    )

    assert not outcome.completed
    assert outcome.terminal_reason == "missing_completion_claim"
    report = WorkerReport.from_json(
        store.view(assignment.worker).head.read(WORKER_REPORT_PATH) or ""
    )
    assert any(
        "no valid structured completion claim" in reason for reason in report.uncertainty_reasons
    )


@pytest.mark.parametrize("control_path", ["contract.json", "assignment.json"])
def test_worker_cannot_complete_after_mutating_control_record(
    store: Store, control_path: str
) -> None:
    assignment = scaffold_assignment(store)
    brain = SubBrain(store, assignment.contract, model=assignment.model)

    class MutatingClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("parser.py", "def parse(text):\n    return text\n")
            brain.branch.write(control_path, "worker changed central truth\n")

    client = MutatingClient(None, [result_message(structured_output=completed_claim())])
    outcome = asyncio.run(
        runtime_for(store, assignment.contract, client, assignment=assignment, brain=brain).run()
    )

    assert not outcome.completed
    assert outcome.terminal_reason == "control_integrity_error"
    report = WorkerReport.from_json(
        store.view(assignment.worker).head.read(WORKER_REPORT_PATH) or ""
    )
    assert any(control_path in reason for reason in report.uncertainty_reasons)


@pytest.mark.parametrize("control_path", ["contract.json", "assignment.json"])
def test_worker_cannot_replace_control_record_with_symlink(store: Store, control_path: str) -> None:
    assignment = scaffold_assignment(store)
    brain = SubBrain(store, assignment.contract, model=assignment.model)

    class SymlinkingClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("parser.py", "def parse(text):\n    return text\n")
            control = brain.branch.path(control_path)
            copy = brain.branch.path(f"{control_path}.copy")
            copy.write_bytes(control.read_bytes())
            control.unlink()
            control.symlink_to(copy.name)

    client = SymlinkingClient(None, [result_message(structured_output=completed_claim())])
    outcome = asyncio.run(
        runtime_for(
            store,
            assignment.contract,
            client,
            assignment=assignment,
            brain=brain,
        ).run()
    )

    assert not outcome.completed
    assert outcome.terminal_reason == "control_integrity_error"
    report_head = store.view(assignment.worker).head
    report = WorkerReport.from_json(report_head.read(WORKER_REPORT_PATH) or "")
    assert any(control_path in reason for reason in report.uncertainty_reasons)
    work_state = store.state(report.final_state_id)
    assert work_state.read(control_path) == f"{control_path}.copy"


def test_broken_report_symlink_is_evidence_and_never_redirects_host_report(
    store: Store,
) -> None:
    assignment = scaffold_assignment(store)
    brain = SubBrain(store, assignment.contract, model=assignment.model)

    class SymlinkingClient(ScriptedClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            brain.branch.write("parser.py", "def parse(text):\n    return text\n")
            brain.branch.path(WORKER_REPORT_PATH).symlink_to("missing-report-target")

    client = SymlinkingClient(None, [result_message(structured_output=completed_claim())])
    outcome = asyncio.run(
        runtime_for(
            store,
            assignment.contract,
            client,
            assignment=assignment,
            brain=brain,
        ).run()
    )

    assert not outcome.completed
    assert outcome.terminal_reason == "control_integrity_error"
    report_head = store.view(assignment.worker).head
    report = WorkerReport.from_json(report_head.read(WORKER_REPORT_PATH) or "")
    assert report.final_state_id == report_head.meta.parents[0]
    assert store.state(report.final_state_id).read(WORKER_REPORT_PATH) == ("missing-report-target")
    assert not brain.branch.path("missing-report-target").exists()


def test_post_result_system_tail_is_recorded_before_report(store: Store) -> None:
    contract = scaffold(store)

    class TailClient(ScriptedClient):
        async def receive_messages(self):
            await self.queried.wait()
            yield result_message()
            await self.disconnected.wait()
            yield SystemMessage("session_state_changed", {"session_id": "session-1"})

    client = TailClient(None, [])
    asyncio.run(runtime_for(store, contract, client).run())

    messages = [
        event.get("message_type")
        for event in store.view(contract.identity).head.transcript.turns
        if event.get("kind") == "sdk_message"
    ]
    assert messages[-1] == "SystemMessage"


def test_tail_session_mismatch_becomes_a_fail_closed_report(store: Store) -> None:
    contract = scaffold(store)

    class TailClient(ScriptedClient):
        async def receive_messages(self):
            await self.queried.wait()
            yield result_message(session_id="session-1")
            await self.disconnected.wait()
            yield SystemMessage("session_state_changed", {"session_id": "session-2"})

    outcome = asyncio.run(runtime_for(store, contract, TailClient(None, [])).run())

    assert outcome.terminal_reason == "runtime_error"
    report = json.loads(store.view(contract.identity).head.read(WORKER_REPORT_PATH) or "null")
    assert any("without a ConversationResetMessage" in item for item in report["uncertainty"])


def test_final_mirror_error_emitted_during_disconnect_is_not_lost(store: Store) -> None:
    contract = scaffold(store)

    class TailMirrorClient(ScriptedClient):
        async def receive_messages(self):
            await self.queried.wait()
            yield result_message()
            await self.disconnected.wait()
            yield MirrorErrorMessage(
                subtype="mirror_error",
                data={"session_id": "session-1"},
                error="final flush failed",
            )

    outcome = asyncio.run(runtime_for(store, contract, TailMirrorClient(None, [])).run())

    assert outcome.terminal_reason == "mirror_error"
    assert outcome.cost_usd is None
    assert "final flush failed" in outcome.error


def test_disconnect_is_allowed_to_finish_past_reader_timeout(store: Store) -> None:
    contract = scaffold(store)

    class SlowDisconnectClient(ScriptedClient):
        def __init__(self) -> None:
            super().__init__(None)
            self.cancelled = False

        async def disconnect(self) -> None:
            try:
                await asyncio.sleep(0.02)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            await super().disconnect()

    client = SlowDisconnectClient()
    runtime = runtime_for(store, contract, client)
    runtime.shutdown_timeout = 0.001

    outcome = asyncio.run(runtime.run())

    assert outcome.terminal_reason == "turn_complete_without_assignment"
    assert not client.cancelled


def test_cancellation_waits_for_disconnect_and_records_final_tail(
    store: Store,
) -> None:
    contract = scaffold(store)

    class BlockingDisconnectClient(ScriptedClient):
        def __init__(self) -> None:
            super().__init__(None, [])
            self.disconnect_started = asyncio.Event()
            self.release_disconnect = asyncio.Event()

        async def receive_messages(self):
            await self.queried.wait()
            yield result_message()
            await self.disconnected.wait()
            yield MirrorErrorMessage(
                subtype="mirror_error",
                data={"session_id": "session-1"},
                error="final cancellation flush failed",
            )

        async def disconnect(self) -> None:
            self.disconnect_started.set()
            await self.release_disconnect.wait()
            await super().disconnect()

    async def scenario() -> None:
        client = BlockingDisconnectClient()
        task = asyncio.create_task(runtime_for(store, contract, client).run())
        await asyncio.wait_for(client.disconnect_started.wait(), timeout=1)
        task.cancel()
        client.release_disconnect.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    pending = store.view(contract.identity).pending_turns()
    assert any(
        event.get("kind") == "sdk_message" and event.get("message_type") == "MirrorErrorMessage"
        for event in pending
    )
    reopened = store.branch(contract.identity)
    reopened.close()


def test_unconfirmed_disconnect_keeps_branch_lease_and_writes_no_report(
    store: Store,
) -> None:
    contract = scaffold(store)
    brain = SubBrain(store, contract)

    class CancelledDisconnectClient(ScriptedClient):
        async def disconnect(self) -> None:
            raise asyncio.CancelledError

    runtime = runtime_for(
        store,
        contract,
        CancelledDisconnectClient(None),
        brain=brain,
    )

    with pytest.raises(ShutdownUnconfirmed):
        asyncio.run(runtime.run())

    assert store.view(contract.identity).head.read(WORKER_REPORT_PATH) is None
    reopened = Store.open(store.root, store.session)
    try:
        with pytest.raises(BranchBusy):
            reopened.branch(contract.identity)
    finally:
        reopened.close()
        brain.close()


def test_adapter_ledger_catches_a_mirror_error_missing_from_sdk_stream(
    store: Store,
) -> None:
    contract = scaffold(store)
    brain = SubBrain(store, contract)
    brain.sessions.unresolved_mirror_batches = lambda: (  # type: ignore[method-assign]
        {"fingerprint": "a" * 64, "status": "failed"},
    )
    client = ScriptedClient(None, [result_message()])

    outcome = asyncio.run(runtime_for(store, contract, client, brain=brain).run())

    assert outcome.terminal_reason == "mirror_error"
    assert outcome.cost_usd is None
    assert "unresolved session mirror batches" in outcome.error


def test_failed_connect_still_closes_partially_started_client(store: Store) -> None:
    contract = scaffold(store)

    class FailedConnectClient(ScriptedClient):
        async def connect(self) -> None:
            self.calls.append(("connect", None))
            raise RuntimeError("startup handshake failed")

    client = FailedConnectClient(None, [])
    outcome = asyncio.run(runtime_for(store, contract, client).run())

    assert [kind for kind, _ in client.calls] == ["connect", "disconnect"]
    assert "startup handshake failed" in outcome.error


def test_stream_failure_waits_for_inflight_monitor_cycle_and_never_drains_again(
    store: Store,
) -> None:
    contract = scaffold(store)
    judge_started = threading.Event()
    release_judge = threading.Event()

    class BrokenClient(ScriptedClient):
        async def receive_messages(self):
            await self.queried.wait()
            while not judge_started.is_set():
                await asyncio.sleep(0)
            raise RuntimeError("stream broke")
            yield  # pragma: no cover

    class BlockingMonitor(QuietMonitor):
        def __init__(self) -> None:
            super().__init__()
            self.cycle_active = False
            self.drain_calls = 0

        async def cycle(self, client: Any):
            self.cycle_active = True
            judge_started.set()
            await asyncio.to_thread(release_judge.wait)
            self.cycle_active = False
            return None, None

        async def drain(self, client: Any, *, final: bool = False):
            assert not self.cycle_active
            self.drain_calls += 1
            return []

    async def scenario() -> tuple[Any, BlockingMonitor, list[str]]:
        monitor = BlockingMonitor()
        ready: list[str] = []
        runtime = runtime_for(store, contract, BrokenClient(None, []), monitor)
        runtime.ready = ready.append
        task = asyncio.create_task(runtime.run())
        while not judge_started.is_set():
            await asyncio.sleep(0)
        asyncio.get_running_loop().call_later(0.02, release_judge.set)
        return await asyncio.wait_for(task, timeout=2), monitor, ready

    outcome, monitor, ready = asyncio.run(scenario())

    assert "stream broke" in outcome.error
    assert monitor.drain_calls == 0
    assert not monitor.cycle_active
    assert ready == []


def test_runtime_refuses_contract_other_than_checkpointed_one(store: Store) -> None:
    scaffold(store, a_contract(task="the accepted task"))
    wrong = a_contract(task="a silently revised task")
    made_clients = 0

    def make_client(options: Any) -> ScriptedClient:
        nonlocal made_clients
        made_clients += 1
        return ScriptedClient(options)

    with pytest.raises(ContractMismatch, match="checkpointed contract"):
        asyncio.run(
            WorkerRuntime(SubBrain(store, wrong), QuietMonitor(), client_factory=make_client).run()
        )

    assert made_clients == 0
    reopened = store.branch(wrong.identity)
    reopened.close()
