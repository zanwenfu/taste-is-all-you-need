from __future__ import annotations

import asyncio
import copy
import errno
import json
import multiprocessing as mp
import os
import shutil
import signal
import sys
import time
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import taste.brains.supervisor as supervisor_module
from taste.brains.contract import Contract
from taste.brains.monitor import Judgement, MonitorBrain, Severity, TerminalDecision
from taste.brains.records import (
    ArtifactRef,
    ArtifactSpec,
    Assignment,
    WorkerReport,
    contract_digest,
)
from taste.brains.supervisor import (
    AssignmentIdentityConflict,
    CentralSupervisor,
    DeliveryRejected,
    InvalidWorkerReport,
    ProcessExit,
    StaleGeneration,
    SubprocessLauncher,
    SupervisorError,
    SupervisorLedgerCorruption,
    SupervisorRun,
    SupervisorStateConflict,
)
from taste.brains.worker_runtime import ASSIGNMENT_PATH, WORKER_REPORT_PATH
from taste.memstore import Store
from taste.memstore.backend import GitBackend


class FakeHandle:
    def __init__(self, pid: int = 4101) -> None:
        self.pid = pid
        self.process_group_id = pid
        self.launch_token = f"token-{pid}"
        self.process_identity = f"identity-{pid}"
        self.exit: ProcessExit | None = None
        self.is_ready = False
        self.die_during_ready = False
        self.termination_calls = 0

    def poll(self) -> ProcessExit | None:
        return self.exit

    def ready(self) -> bool:
        if self.die_during_ready:
            self.exit = ProcessExit(exit_code=1, reaped=True)
        return self.is_ready

    def terminate_tree(self, grace_seconds: float) -> ProcessExit:
        self.termination_calls += 1
        if self.exit is None:
            self.exit = ProcessExit(signal=9, reaped=True)
        return self.exit


class FakeLauncher:
    def __init__(self, handle: FakeHandle | None = None) -> None:
        self.handle = handle or FakeHandle()
        self.launch_calls = 0
        self.recover_calls = 0
        self.fail_launches = 0
        self.before_launch: Any = None
        self.launched = False

    def launch(self, spec):
        self.launch_calls += 1
        if self.before_launch is not None:
            self.before_launch(spec)
        if self.fail_launches:
            self.fail_launches -= 1
            raise RuntimeError("injected launch gap")
        self.launched = True
        return self.handle

    def recover(self, spec):
        self.recover_calls += 1
        return self.handle if self.launched else None

    def cancel(self, spec, grace_seconds):
        return self.handle.terminate_tree(grace_seconds) if self.launched else ProcessExit(reaped=True)


class TrackingControlLock:
    def __init__(self) -> None:
        self.active = False
        self.entries = 0

    def __enter__(self):
        assert not self.active
        self.active = True
        self.entries += 1
        return self

    def __exit__(self, *_exc):
        self.active = False


@dataclass
class FakeClock:
    value: datetime = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


@pytest.fixture
def store(tmp_path: Path):
    opened = Store.open(tmp_path / "repo", "central-test")
    yield opened
    opened.close()


def assignment_for(
    supervisor: CentralSupervisor,
    *,
    assignment_id: str = "build-parser",
    generation: int = 1,
    attempt: int = 0,
    worker: str = "worker-1",
    inputs: tuple[ArtifactRef, ...] = (),
) -> Assignment:
    contract = Contract(
        identity=worker,
        task="build the parser",
        inputs=tuple(item.path for item in inputs),
        outputs=("parser.py",),
        success_criteria=("the parser is correct",),
    )
    return Assignment(
        assignment_id=assignment_id,
        generation=generation,
        attempt=attempt,
        contract=contract,
        contract_digest=contract_digest(contract),
        base_state_id=supervisor.integration.head.id,
        inputs=inputs,
        outputs=(ArtifactSpec(artifact_id="parser", path="parser.py"),),
        model="claude-sonnet-5",
    )


def test_supervisor_run_wire_is_strict_and_phase_consistent(store: Store) -> None:
    supervisor = CentralSupervisor(store, launcher=FakeLauncher())
    prepared = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)
    raw = prepared.to_dict()

    encoded = json.dumps(raw, sort_keys=True)
    assert SupervisorRun.from_json(encoded) == prepared

    invalid: list[tuple[dict[str, Any], str]] = []
    wrong_timeout = copy.deepcopy(raw)
    wrong_timeout["wall_timeout_seconds"] = True
    invalid.append((wrong_timeout, "wall_timeout_seconds"))
    wrong_sequence = copy.deepcopy(raw)
    wrong_sequence["sequence"] = 1.5
    invalid.append((wrong_sequence, "sequence"))
    wrong_tuple = copy.deepcopy(raw)
    wrong_tuple["conflict_paths"] = "parser.py"
    invalid.append((wrong_tuple, "JSON arrays"))
    inconsistent_terminal = copy.deepcopy(raw)
    inconsistent_terminal["terminal_reason"] = "process_failed"
    invalid.append((inconsistent_terminal, "terminal phase"))
    incomplete_process = copy.deepcopy(raw)
    incomplete_process["pid"] = 4101
    invalid.append((incomplete_process, "process identity fields"))
    inconsistent_report = copy.deepcopy(raw)
    inconsistent_report["report_id"] = "report-1"
    inconsistent_report["report_state_id"] = "0" * 40
    invalid.append((inconsistent_report, "report phase"))

    for candidate, match in invalid:
        with pytest.raises(ValueError, match=match):
            SupervisorRun.from_dict(candidate)

    with pytest.raises(ValueError, match="duplicate key"):
        SupervisorRun.from_json(encoded[:-1] + ',"schema":"duplicate"}')
    with pytest.raises(ValueError, match="non-JSON number"):
        SupervisorRun.from_json(
            encoded.replace('"wall_timeout_seconds": 30.0', '"wall_timeout_seconds": NaN')
        )


def test_transition_rejects_a_stale_durable_run_snapshot(store: Store) -> None:
    supervisor = CentralSupervisor(store, launcher=FakeLauncher())
    prepared = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)

    current = supervisor._transition(prepared, kind="first_observation")
    with pytest.raises(SupervisorStateConflict, match="stale"):
        supervisor._transition(prepared, kind="competing_observation")

    assert supervisor.get(prepared.run_id) == current


def test_supervisor_run_index_preserves_creation_order(store: Store) -> None:
    supervisor = CentralSupervisor(store, launcher=FakeLauncher())
    first = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)
    second = supervisor.prepare(
        assignment_for(
            supervisor,
            assignment_id="build-renderer",
            worker="worker-2",
        ),
        wall_timeout_seconds=30,
    )

    assert supervisor.runs() == (first, second)


def test_supervisor_run_index_detects_deleted_run_record(store: Store) -> None:
    supervisor = CentralSupervisor(store, launcher=FakeLauncher())
    prepared = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)

    supervisor.control.path(supervisor_module._run_path(prepared.run_id)).unlink()
    supervisor.control.checkpoint("delete durable run record")

    with pytest.raises(SupervisorLedgerCorruption, match="run files disagree"):
        supervisor.runs()


def test_supervisor_run_history_detects_delete_then_restore(store: Store) -> None:
    supervisor = CentralSupervisor(store, launcher=FakeLauncher())
    prepared = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)
    path = supervisor_module._run_path(prepared.run_id)
    original = supervisor.control.head.read(path)
    assert original is not None

    supervisor.control.path(path).unlink()
    supervisor.control.checkpoint("temporarily delete durable run record")
    supervisor.control.write(path, original)
    supervisor.control.checkpoint("restore durable run record")

    with pytest.raises(SupervisorLedgerCorruption, match="deleted from durable history"):
        supervisor.runs()
    with pytest.raises(SupervisorLedgerCorruption, match="deleted from durable history"):
        supervisor.get(prepared.run_id)
    with pytest.raises(SupervisorLedgerCorruption, match="deleted from durable history"):
        supervisor.start(prepared.run_id, active_generation=1)
    assert supervisor.launcher.launch_calls == 0


def test_supervisor_run_index_detects_historical_rewrite(store: Store) -> None:
    supervisor = CentralSupervisor(store, launcher=FakeLauncher())
    supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)
    supervisor.prepare(
        assignment_for(
            supervisor,
            assignment_id="build-renderer",
            worker="worker-2",
        ),
        wall_timeout_seconds=30,
    )
    raw = json.loads(
        supervisor.control.head.read(supervisor_module.RUN_INDEX_PATH) or ""
    )
    raw["runs"][0]["assignment_digest"] = f"sha256:{'0' * 64}"
    supervisor.control.write(
        supervisor_module.RUN_INDEX_PATH,
        json.dumps(raw, sort_keys=True),
    )
    supervisor.control.checkpoint("rewrite durable run index")

    with pytest.raises(SupervisorLedgerCorruption, match="not append-only"):
        supervisor.runs()


def input_ref(supervisor: CentralSupervisor) -> ArtifactRef:
    producer = supervisor.store.branch(
        "producer", from_state=supervisor.integration.head, producer="producer"
    )
    try:
        producer.write("schema.txt", b"exact input bytes\n")
        state = producer.checkpoint("publish exact input")
    finally:
        producer.close()
    return ArtifactRef(
        artifact_id="schema",
        branch="producer",
        state_id=state.id,
        path="schema.txt",
        blob_id=state.blob("schema.txt") or "",
    )


def install_report(
    supervisor: CentralSupervisor,
    assignment: Assignment,
    run_id: str,
    *,
    completed: bool = True,
    uncertain: bool = False,
    monitor_current: str = "fine",
    assessment_state: str | None = None,
) -> tuple[WorkerReport, str]:
    worker = supervisor.store.branch(assignment.worker)
    try:
        worker.write("parser.py", "def parse(text):\n    return text\n")
        work = worker.checkpoint("terminal worker product")
        exact_assessment_state = assessment_state or work.id
        output = ArtifactRef(
            artifact_id="parser",
            branch=assignment.worker,
            state_id=work.id,
            path="parser.py",
            blob_id=work.blob("parser.py") or "",
        )

        class FineTerminalJudge:
            def judge_terminal(self, _contract, _state, _context, findings):
                finding_ids = tuple(finding.id for finding in findings)
                return TerminalDecision(
                    judgement=Judgement(Severity.FINE, "exact state is complete"),
                    resolved_finding_ids=finding_ids,
                )

        monitor_brain = MonitorBrain(
            supervisor.store, assignment.contract, FineTerminalJudge(), batch_size=100
        )
        certified = asyncio.run(monitor_brain.certify_terminal(work, context={"test": True}))
        assessment = certified.to_dict()
        assessment["state_id"] = exact_assessment_state
        report = WorkerReport(
            report_id=f"report-{assignment.generation}-{assignment.attempt}",
            run_id=run_id,
            assignment_id=assignment.assignment_id,
            worker=assignment.worker,
            generation=assignment.generation,
            attempt=assignment.attempt,
            contract_digest=assignment.contract_digest,
            base_state_id=assignment.base_state_id,
            final_state_id=work.id,
            at=work.meta.created_at,
            completed=completed,
            terminal_reason="completed" if completed else "incomplete",
            outputs=(output,),
            monitor_severity=monitor_current,
            uncertain=uncertain,
            uncertainty_reasons=("unknown side effect",) if uncertain else (),
            metadata={
                "durability_ok": True,
                "monitor": {
                    "worst": "drifting",
                    "current": monitor_current,
                    "current_state": exact_assessment_state,
                    "pending_actions": [],
                    "terminal_assessment": assessment,
                },
            },
        )
        worker.write(WORKER_REPORT_PATH, report.to_json())
        report_state = worker.checkpoint("durable worker report")
    finally:
        worker.close()
    return report, report_state.id


def test_prepare_projects_exact_inputs_and_spawn_intent_precedes_launch(store: Store) -> None:
    launcher = FakeLauncher()
    clock = FakeClock()
    supervisor = CentralSupervisor(store, launcher=launcher, clock=clock)
    source = input_ref(supervisor)
    assignment = assignment_for(supervisor, inputs=(source,))
    prepared = supervisor.prepare(
        assignment, wall_timeout_seconds=30, active_generation=1
    )

    state = store.state(prepared.prepared_state_id)
    assert state.read("contract.json") == assignment.contract.to_json()
    assert state.read(ASSIGNMENT_PATH) == assignment.to_json()
    assert state.blob("schema.txt") == source.blob_id
    assert launcher.launch_calls == 0

    def observe_before_launch(spec) -> None:
        durable = supervisor.get(spec.run_id)
        assert durable.phase == "spawn_intent"
        assert durable.deadline_at is not None
        assert store.state(durable.prepared_state_id).read(ASSIGNMENT_PATH) == assignment.to_json()

    launcher.before_launch = observe_before_launch
    spawned = supervisor.start(prepared.run_id, active_generation=1)
    assert spawned.phase == "spawned"
    assert launcher.launch_calls == 1
    assert supervisor.start(prepared.run_id, active_generation=1).phase == "spawned"
    assert launcher.launch_calls == 1


@pytest.mark.parametrize("mode", ["executable", "symlink"])
def test_prepare_and_reopen_preserve_input_modes(store: Store, mode: str) -> None:
    launcher = FakeLauncher()
    supervisor = CentralSupervisor(store, launcher=launcher)
    producer = store.branch("producer")
    producer.write("input", "#!/bin/sh\nexit 0\n")
    if mode == "symlink":
        producer.path("input").unlink()
        producer.path("input").symlink_to("missing-target")
    else:
        producer.path("input").chmod(0o755)
    source = producer.checkpoint("publish mode-sensitive input")
    producer.close()
    artifact = ArtifactRef("input", "producer", source.id, "input", source.blob("input"))
    assignment = assignment_for(supervisor, inputs=(artifact,))
    prepared = supervisor.prepare(assignment, wall_timeout_seconds=30)
    expected = store.backend.entry_at(source.id, "input")
    assert store.backend.entry_at(prepared.prepared_state_id, "input") == expected
    supervisor.close()

    reopened = CentralSupervisor(store, launcher=launcher)
    assert reopened.prepare(assignment, wall_timeout_seconds=30) == prepared
    assert reopened.start(prepared.run_id, active_generation=1).phase == "spawned"
    assert launcher.launch_calls == 1
    reopened.close()


def test_prepare_rejects_same_bytes_with_conflicting_input_modes(store: Store) -> None:
    launcher = FakeLauncher()
    supervisor = CentralSupervisor(store, launcher=launcher)
    producer = store.branch("producer")
    producer.write("input", "exit 0\n")
    regular = producer.checkpoint("ordinary file")
    producer.path("input").chmod(0o755)
    executable = producer.checkpoint("same bytes, executable")
    producer.close()
    artifacts = tuple(
        ArtifactRef(identity, "producer", state.id, "input", state.blob("input"))
        for identity, state in (("regular", regular), ("executable", executable))
    )
    assert artifacts[0].blob_id == artifacts[1].blob_id
    with pytest.raises(AssignmentIdentityConflict, match="different bytes or modes"):
        supervisor.prepare(assignment_for(supervisor, inputs=artifacts), wall_timeout_seconds=30)
    assert not store.view("worker-1").exists()
    assert launcher.launch_calls == 0
    supervisor.close()


def test_coordinator_can_share_control_branch_and_mutation_lock(store: Store) -> None:
    control = store.branch("shared-control", producer="coordinator")
    integration = store.branch("shared-integration", producer="coordinator")
    lock = TrackingControlLock()
    launcher = FakeLauncher()
    launcher.before_launch = lambda _spec: assert_control_unlocked(lock)
    supervisor = CentralSupervisor(
        store,
        launcher=launcher,
        control_branch=control,
        integration_branch=integration,
        control_lock=lock,
    )
    assignment = assignment_for(supervisor)
    run = supervisor.prepare(assignment, wall_timeout_seconds=10)
    supervisor.start(run.run_id, active_generation=1)
    assert lock.entries > 0

    supervisor.close()
    # Injected branches remain coordinator-owned and writable.
    with lock:
        control.write("plans/current.json", "{}\n")
        control.checkpoint("planner can still use shared control")
    assert control.head.read("plans/current.json") == "{}\n"
    integration.close()
    control.close()


def assert_control_unlocked(lock: TrackingControlLock) -> None:
    assert not lock.active


def test_invalid_base_or_input_spawns_nothing(store: Store) -> None:
    launcher = FakeLauncher()
    supervisor = CentralSupervisor(store, launcher=launcher)
    source = input_ref(supervisor)
    wrong = ArtifactRef(
        artifact_id=source.artifact_id,
        branch=source.branch,
        state_id=source.state_id,
        path=source.path,
        blob_id="0" * 40,
    )
    with pytest.raises(AssignmentIdentityConflict, match="declared file bytes"):
        supervisor.prepare(
            assignment_for(supervisor, inputs=(wrong,)), wall_timeout_seconds=10
        )
    assert launcher.launch_calls == 0


@pytest.mark.parametrize(
    ("contract_field", "value", "match"),
    [
        ("inputs", ("another-input.txt",), "contract inputs"),
        ("outputs", ("another-output.txt",), "contract outputs"),
    ],
)
def test_supervisor_rejects_contract_and_structured_io_disagreement(
    store: Store,
    contract_field: str,
    value: tuple[str, ...],
    match: str,
) -> None:
    supervisor = CentralSupervisor(store, launcher=FakeLauncher())
    source = input_ref(supervisor)
    assignment = assignment_for(supervisor, inputs=(source,))
    changed_contract = replace(assignment.contract, **{contract_field: value})
    mismatched = replace(
        assignment,
        contract=changed_contract,
        contract_digest=contract_digest(changed_contract),
    )

    with pytest.raises(AssignmentIdentityConflict, match=match):
        supervisor.prepare(mismatched, wall_timeout_seconds=10)


def test_prepare_never_follows_a_final_control_symlink(store: Store, tmp_path: Path) -> None:
    supervisor = CentralSupervisor(store, launcher=FakeLauncher())
    outside = tmp_path / "outside-contract.txt"
    outside.write_text("must remain untouched\n")
    supervisor.integration.path("contract.json").symlink_to(outside)
    supervisor.integration.checkpoint("hostile final control symlink")
    assignment = assignment_for(supervisor)

    with pytest.raises(AssignmentIdentityConflict, match="traverses symlink"):
        supervisor.prepare(assignment, wall_timeout_seconds=10)

    assert outside.read_text() == "must remain untouched\n"


def test_prepare_never_traverses_a_parent_symlink_for_an_input(
    store: Store, tmp_path: Path
) -> None:
    supervisor = CentralSupervisor(store, launcher=FakeLauncher())
    producer = store.branch(
        "nested-producer", from_state=supervisor.integration.head, producer="producer"
    )
    try:
        producer.write("nested/schema.txt", "trusted source\n")
        source_state = producer.checkpoint("source with a real directory")
    finally:
        producer.close()
    source = ArtifactRef(
        artifact_id="nested-schema",
        branch="nested-producer",
        state_id=source_state.id,
        path="nested/schema.txt",
        blob_id=source_state.blob("nested/schema.txt") or "",
    )

    outside = tmp_path / "outside-directory"
    outside.mkdir()
    (outside / "schema.txt").write_text("must remain untouched\n")
    supervisor.integration.path("nested").symlink_to(outside, target_is_directory=True)
    supervisor.integration.checkpoint("hostile parent symlink")
    assignment = assignment_for(supervisor, inputs=(source,))

    with pytest.raises(AssignmentIdentityConflict, match="traverses symlink"):
        supervisor.prepare(assignment, wall_timeout_seconds=10)

    assert (outside / "schema.txt").read_text() == "must remain untouched\n"


def test_generation_is_fenced_before_prepare_start_and_collect(store: Store) -> None:
    launcher = FakeLauncher()
    supervisor = CentralSupervisor(store, launcher=launcher)
    assignment = assignment_for(supervisor, generation=2)
    with pytest.raises(StaleGeneration):
        supervisor.prepare(assignment, wall_timeout_seconds=10, active_generation=3)
    run = supervisor.prepare(assignment, wall_timeout_seconds=10, active_generation=2)
    with pytest.raises(StaleGeneration):
        supervisor.start(run.run_id, active_generation=3)
    assert launcher.launch_calls == 0


def test_spawn_intent_survives_launch_failure_and_restart_without_planner_call(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    first_store = Store.open(root, "central-test")
    launcher = FakeLauncher()
    launcher.fail_launches = 1
    first = CentralSupervisor(first_store, launcher=launcher)
    assignment = assignment_for(first)
    run = first.prepare(assignment, wall_timeout_seconds=30)
    with pytest.raises(RuntimeError, match="injected"):
        first.start(run.run_id, active_generation=1)
    assert first.get(run.run_id).phase == "spawn_intent"
    first.close()
    first_store.close()

    second_store = Store.open(root, "central-test")
    second = CentralSupervisor(second_store, launcher=launcher)
    (reconciled,) = second.reconcile(active_generation=1)
    assert reconciled.phase == "spawned"
    assert launcher.launch_calls == 2
    second.close()
    second_store.close()


def test_expired_spawn_intent_never_launches_after_the_wall_deadline(store: Store) -> None:
    clock = FakeClock()
    launcher = FakeLauncher()
    launcher.fail_launches = 1
    supervisor = CentralSupervisor(store, launcher=launcher, clock=clock)
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=5)
    with pytest.raises(RuntimeError, match="injected"):
        supervisor.start(run.run_id, active_generation=1)
    clock.advance(6)

    terminal = supervisor.start(run.run_id, active_generation=1)

    assert terminal.terminal_reason == "wall_timeout"
    assert terminal.reaped
    assert launcher.launch_calls == 1


def test_readiness_cannot_outlive_a_racing_process_failure(store: Store) -> None:
    handle = FakeHandle()
    handle.is_ready = True
    handle.die_during_ready = True
    supervisor = CentralSupervisor(store, launcher=FakeLauncher(handle))
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=20)
    terminal = supervisor.start(run.run_id, active_generation=1)
    assert terminal.phase == "terminal"
    assert not terminal.ready
    assert terminal.ready_at is None
    assert terminal.terminal_reason == "process_failed"


def test_hard_wall_timeout_kills_tree_then_captures_dirty_work(store: Store) -> None:
    clock = FakeClock()
    handle = FakeHandle()
    supervisor = CentralSupervisor(
        store,
        launcher=FakeLauncher(handle),
        clock=clock,
        termination_grace=0,
    )
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=5)
    supervisor.start(run.run_id, active_generation=1)
    worker_path = store.worktree_path_for("worker-1")
    (worker_path / "precious.txt").write_text("dirty work survives timeout\n")

    clock.advance(6)
    terminal = supervisor.poll(run.run_id, active_generation=1)

    assert terminal.terminal_reason == "wall_timeout"
    assert handle.termination_calls == 1
    assert store.view("worker-1").head.read("precious.txt") == "dirty work survives timeout\n"
    assert not worker_path.exists()


def test_restart_recovers_live_process_instead_of_launching_duplicate(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    store1 = Store.open(root, "central-test")
    launcher = FakeLauncher()
    first = CentralSupervisor(store1, launcher=launcher)
    assignment = assignment_for(first)
    run = first.prepare(assignment, wall_timeout_seconds=30)
    first.start(run.run_id, active_generation=1)
    first.close()
    store1.close()

    store2 = Store.open(root, "central-test")
    second = CentralSupervisor(store2, launcher=launcher)
    (recovered,) = second.reconcile(active_generation=1)
    assert recovered.phase == "spawned"
    assert recovered.pid == launcher.handle.pid
    assert launcher.launch_calls == 1
    second.close()
    store2.close()


def test_restart_refuses_process_evidence_for_a_different_launch(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    store1 = Store.open(root, "central-test")
    original = FakeLauncher(FakeHandle(4101))
    first = CentralSupervisor(store1, launcher=original)
    run = first.prepare(assignment_for(first), wall_timeout_seconds=30)
    first.start(run.run_id, active_generation=1)
    first.close()
    store1.close()

    store2 = Store.open(root, "central-test")
    wrong = FakeLauncher(FakeHandle(9999))
    wrong.launched = True
    second = CentralSupervisor(store2, launcher=wrong)
    with pytest.raises(SupervisorError, match="differs from the durable spawned event"):
        second.reconcile(active_generation=1)
    assert wrong.handle.termination_calls == 0
    second.close()
    store2.close()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fork handshake")
def test_child_persists_pid_when_parent_dies_in_fork_record_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    first_store = Store.open(root, "central-test")
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import time; "
            "p=Path('started.txt'); p.write_text((p.read_text() if p.exists() else '')+'once\\n'); "
            "time.sleep(30)"
        ),
    ]
    first_launcher = SubprocessLauncher(command)
    first = CentralSupervisor(first_store, launcher=first_launcher)
    assignment = assignment_for(first)
    run = first.prepare(assignment, wall_timeout_seconds=60)
    real_wait = supervisor_module._await_launch_evidence

    def parent_dies_after_popen(*_args, **_kwargs):
        raise RuntimeError("simulated parent SIGKILL after fork")

    monkeypatch.setattr(supervisor_module, "_await_launch_evidence", parent_dies_after_popen)
    with pytest.raises(RuntimeError, match="parent SIGKILL"):
        first.start(run.run_id, active_generation=1)
    assert first.get(run.run_id).phase == "spawn_intent"
    monkeypatch.setattr(supervisor_module, "_await_launch_evidence", real_wait)

    _ready, launch_path = first._sidecars(run.run_id)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            if "SubprocessLaunch/1" in launch_path.read_text():
                break
        except FileNotFoundError:
            pass
        time.sleep(0.02)
    assert "SubprocessLaunch/1" in launch_path.read_text()
    first.close()
    first_store.close()

    second_store = Store.open(root, "central-test")
    second = CentralSupervisor(second_store, launcher=SubprocessLauncher(command))
    (recovered,) = second.reconcile(active_generation=1)
    assert recovered.phase == "spawned"
    started = second_store.worktree_path_for("worker-1") / "started.txt"
    assert started.read_text() == "once\n"
    second.stop(run.run_id, "test_stop")
    assert second_store.view("worker-1").head.read("started.txt") == "once\n"
    second.close()
    second_store.close()


def test_recovered_launcher_never_signals_a_reused_root_pid(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = CentralSupervisor(store, launcher=FakeLauncher())
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)
    spec = supervisor._spec(replace(run, deadline_at="2026-09-10T12:01:00Z"))
    handle = supervisor_module._SubprocessHandle(
        spec=spec,
        pid=8123,
        process_group_id=8123,
        launch_token="old-token",
        process_identity="old-os-birth",
        popen=None,
    )
    signals: list[tuple[str, int, int]] = []
    monkeypatch.setattr(
        supervisor_module,
        "_process_table",
        lambda: {8123: (1, 8123, "different-new-os-birth")},
    )
    monkeypatch.setattr(supervisor_module, "_lock_is_held", lambda _path: False)
    monkeypatch.setattr(
        supervisor_module.os,
        "kill",
        lambda pid, sig: signals.append(("pid", pid, sig)),
    )
    monkeypatch.setattr(
        supervisor_module.os,
        "killpg",
        lambda pgid, sig: signals.append(("group", pgid, sig)),
    )

    exit_status = handle.terminate_tree(0)

    assert exit_status == ProcessExit(reaped=False)
    assert signals == []


def test_recovered_launcher_never_claims_an_orphaned_reused_group(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = CentralSupervisor(store, launcher=FakeLauncher())
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)
    spec = supervisor._spec(replace(run, deadline_at="2026-09-10T12:01:00Z"))
    handle = supervisor_module._SubprocessHandle(
        spec=spec,
        pid=8123,
        process_group_id=8123,
        launch_token="old-token",
        process_identity="old-os-birth",
        popen=None,
    )
    signals: list[tuple[int, int]] = []
    # The original root is gone.  A held inherited lock is insufficient to
    # prove that a new orphan carrying the same numeric PGID belongs to it.
    monkeypatch.setattr(
        supervisor_module,
        "_process_table",
        lambda: {8124: (1, 8123, "unrelated-new-os-birth")},
    )
    monkeypatch.setattr(supervisor_module, "_lock_is_held", lambda _path: True)
    monkeypatch.setattr(
        supervisor_module.os,
        "kill",
        lambda pid, sig: signals.append((pid, sig)),
    )

    assert handle.terminate_tree(0) == ProcessExit(reaped=False)
    assert signals == []


def test_birth_fenced_descendant_remains_owned_after_detach_and_reparent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = iter(
        (
            {
                8123: (1, 8123, "root-birth"),
                8124: (8123, 9000, "child-birth"),
            },
            {8124: (1, 9000, "child-birth")},
        )
    )
    monkeypatch.setattr(supervisor_module, "_process_table", lambda: next(snapshots))

    first, safe_group = supervisor_module._tree_members(
        8123,
        8123,
        "root-birth",
        allow_orphaned_group=True,
        known={8123: "root-birth"},
    )
    second, second_safe_group = supervisor_module._tree_members(
        8123,
        8123,
        "root-birth",
        allow_orphaned_group=True,
        known=first,
    )

    assert safe_group
    assert first == {8123: "root-birth", 8124: "child-birth"}
    assert second == {8124: "child-birth"}
    assert not second_safe_group


def test_reused_root_does_not_release_an_already_fenced_descendant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        supervisor_module,
        "_process_table",
        lambda: {
            8123: (1, 8123, "unrelated-root-birth"),
            8124: (1, 9000, "owned-child-birth"),
        },
    )

    members, safe_group = supervisor_module._tree_members(
        8123,
        8123,
        "original-root-birth",
        allow_orphaned_group=True,
        known={8123: "original-root-birth", 8124: "owned-child-birth"},
    )

    assert members == {8124: "owned-child-birth"}
    assert not safe_group


def test_observed_descendant_birth_survives_handle_reconstruction(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = CentralSupervisor(store, launcher=FakeLauncher())
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)
    spec = supervisor._spec(replace(run, deadline_at="2026-09-10T12:01:00Z"))
    root_pid = 8123
    child_pid = 8124
    table = {
        root_pid: (1, root_pid, "root-birth"),
        child_pid: (root_pid, 9000, "child-birth"),
    }
    monkeypatch.setattr(supervisor_module, "_process_table", lambda: dict(table))

    first = supervisor_module._SubprocessHandle(
        spec=spec,
        pid=root_pid,
        process_group_id=root_pid,
        launch_token="durable-token",
        process_identity="root-birth",
        popen=None,
    )
    assert first._observe_members(root_live=True)[child_pid] == "child-birth"

    # Reconstruct the handle after the child has detached and the root has
    # disappeared.  Its exact birth remains owned without relying on ancestry,
    # process-group reuse, or another marker scan.
    table.clear()
    table[child_pid] = (1, 9000, "child-birth")
    recovered = supervisor_module._SubprocessHandle(
        spec=spec,
        pid=root_pid,
        process_group_id=root_pid,
        launch_token="durable-token",
        process_identity="root-birth",
        popen=None,
    )

    assert recovered._observe_members(root_live=False) == {
        root_pid: "root-birth",
        child_pid: "child-birth",
    }
    record = json.loads(supervisor_module._members_path(spec.launch_path).read_text())
    assert record["members"] == [
        {"pid": root_pid, "process_identity": "root-birth"},
        {"pid": child_pid, "process_identity": "child-birth"},
    ]


def test_exact_terminal_report_delivers_despite_resolved_historical_drift(store: Store) -> None:
    handle = FakeHandle()
    supervisor = CentralSupervisor(store, launcher=FakeLauncher(handle))
    assignment = assignment_for(supervisor)
    run = supervisor.prepare(assignment, wall_timeout_seconds=30)
    supervisor.start(run.run_id, active_generation=1)
    expected, report_state = install_report(supervisor, assignment, run.run_id)
    handle.exit = ProcessExit(exit_code=0, reaped=True)
    terminal = supervisor.poll(run.run_id, active_generation=1)
    assert terminal.phase == "terminal"

    accepted = supervisor.collect(run.run_id, active_generation=1)
    assert accepted == expected
    assert supervisor.get(run.run_id).report_state_id == report_state
    delivered = supervisor.deliver(run.run_id, active_generation=1)
    assert delivered.ok
    assert supervisor.integration.head.read("parser.py") == "def parse(text):\n    return text\n"
    assert supervisor.get(run.run_id).phase == "delivered"


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"uncertain": True}, "unresolved uncertainty"),
        ({"completed": False}, "completion claim"),
        ({"assessment_state": "0" * 40}, "terminal monitor assessment"),
    ],
)
def test_delivery_rejects_uncertified_reports(store: Store, changes, match: str) -> None:
    handle = FakeHandle()
    supervisor = CentralSupervisor(store, launcher=FakeLauncher(handle))
    assignment = assignment_for(supervisor)
    run = supervisor.prepare(assignment, wall_timeout_seconds=30)
    supervisor.start(run.run_id, active_generation=1)
    install_report(supervisor, assignment, run.run_id, **changes)
    handle.exit = ProcessExit(exit_code=0, reaped=True)
    supervisor.poll(run.run_id, active_generation=1)
    with pytest.raises(DeliveryRejected, match=match):
        supervisor.deliver(run.run_id, active_generation=1)


def test_report_with_wrong_run_identity_is_never_accepted(store: Store) -> None:
    handle = FakeHandle()
    supervisor = CentralSupervisor(store, launcher=FakeLauncher(handle))
    assignment = assignment_for(supervisor)
    run = supervisor.prepare(assignment, wall_timeout_seconds=30)
    supervisor.start(run.run_id, active_generation=1)
    install_report(supervisor, assignment, "some-other-run")
    handle.exit = ProcessExit(exit_code=0, reaped=True)
    supervisor.poll(run.run_id, active_generation=1)
    with pytest.raises(InvalidWorkerReport, match="identity"):
        supervisor.collect(run.run_id, active_generation=1)


def test_nonzero_process_exit_blocks_an_otherwise_valid_report(store: Store) -> None:
    handle = FakeHandle()
    supervisor = CentralSupervisor(store, launcher=FakeLauncher(handle))
    assignment = assignment_for(supervisor)
    run = supervisor.prepare(assignment, wall_timeout_seconds=30)
    supervisor.start(run.run_id, active_generation=1)
    install_report(supervisor, assignment, run.run_id)
    handle.exit = ProcessExit(exit_code=2, reaped=True)
    supervisor.poll(run.run_id, active_generation=1)
    with pytest.raises(DeliveryRejected, match="process_failed"):
        supervisor.deliver(run.run_id, active_generation=1)


def test_unknown_unreaped_process_exit_cannot_deliver_valid_report(store: Store) -> None:
    handle = FakeHandle()
    supervisor = CentralSupervisor(store, launcher=FakeLauncher(handle))
    assignment = assignment_for(supervisor)
    run = supervisor.prepare(assignment, wall_timeout_seconds=30)
    supervisor.start(run.run_id, active_generation=1)
    install_report(supervisor, assignment, run.run_id)
    handle.exit = ProcessExit(reaped=False)

    terminal = supervisor.poll(run.run_id, active_generation=1)

    assert terminal.uncertain
    with pytest.raises(DeliveryRejected, match="process outcome remains uncertain"):
        supervisor.deliver(run.run_id, active_generation=1)


@pytest.mark.parametrize("trigger", ["exit", "stop", "timeout", "prepared", "unlaunched", "lost"])
def test_capture_failure_records_exit_and_retries_after_restart_without_relaunch(
    store: Store, monkeypatch, trigger: str,
) -> None:
    clock = FakeClock()
    launcher = FakeLauncher()
    supervisor = CentralSupervisor(store, launcher=launcher, clock=clock)
    assignment = assignment_for(supervisor)
    run = supervisor.prepare(assignment, wall_timeout_seconds=30)
    if trigger == "unlaunched":
        launcher.fail_launches = 1
        with pytest.raises(RuntimeError, match="injected launch gap"):
            supervisor.start(run.run_id, active_generation=1)
    elif trigger != "prepared":
        supervisor.start(run.run_id, active_generation=1)
    worker = store.branch(assignment.worker)
    worker.write("precious.txt", "only uncommitted copy")
    worktree = worker.worktree
    worker.close()
    if trigger == "exit":
        launcher.handle.exit = ProcessExit(exit_code=17, reaped=True)
    if trigger in {"timeout", "unlaunched"}:
        clock.advance(31)
    if trigger == "lost":
        supervisor._handles.clear()
        launcher.launched = False
    real_stage = GitBackend.stage_all

    def disk_full(backend):
        if backend.path == worktree:
            raise OSError(errno.ENOSPC, "simulated worker snapshot disk full")
        real_stage(backend)

    with monkeypatch.context() as fault:
        fault.setattr(GitBackend, "stage_all", disk_full)
        terminal = (
            supervisor.stop(run.run_id)
            if trigger in {"stop", "prepared"}
            else supervisor.poll(run.run_id, active_generation=1)
        )
        assert terminal.terminal
        assert terminal.recovery_state_id is None
        assert terminal.recovery_status == "capture_failed"
        assert "disk full" in terminal.recovery_error
        assert terminal.uncertain
        if trigger == "exit":
            assert terminal.exit_code == 17 and terminal.reaped
        assert (worktree / "precious.txt").read_text() == "only uncommitted copy"
        assert store.view(assignment.worker).head.read("precious.txt") is None
        with pytest.raises(AssignmentIdentityConflict, match="recovery"):
            supervisor.prepare(assignment_for(supervisor, attempt=1), wall_timeout_seconds=30)
        with pytest.raises(InvalidWorkerReport, match="recovery"):
            supervisor.collect(run.run_id, active_generation=1)
        supervisor.close()
        with (
            closing(Store.open(store.root, store.session)) as reopened_store,
            CentralSupervisor(reopened_store, launcher=launcher, clock=clock) as reopened,
        ):
            assert reopened.get(run.run_id) == terminal
            # Repeated failure keeps the same evidence without an endless
            # stream of identical checkpoints or another termination.
            assert reopened.reconcile(active_generation=1) == (terminal,)
    launches = launcher.launch_calls
    terminations = launcher.handle.termination_calls
    with (
        closing(Store.open(store.root, store.session)) as reopened_store,
        CentralSupervisor(reopened_store, launcher=launcher, clock=clock) as reopened,
    ):
        recovered, = reopened.reconcile(active_generation=1)
        assert recovered.terminal
        assert recovered.recovery_status == "complete"
        assert not recovered.recovery_error
        assert recovered.recovery_state_id is not None
        assert reopened_store.state(recovered.recovery_state_id).read("precious.txt") == "only uncommitted copy"
        assert not worktree.exists()
        assert reopened.reconcile(active_generation=1) == (recovered,)
    assert launcher.launch_calls == launches
    assert launcher.handle.termination_calls == terminations


def test_cleanup_failure_retains_exact_capture_and_recovers_without_recapturing(
    store: Store, monkeypatch,
) -> None:
    launcher = FakeLauncher()
    supervisor = CentralSupervisor(store, launcher=launcher)
    assignment = assignment_for(supervisor)
    run = supervisor.prepare(assignment, wall_timeout_seconds=30)
    supervisor.start(run.run_id, active_generation=1)
    worker = store.branch(assignment.worker)
    worker.write("precious.txt", "keep")
    worktree = worker.worktree
    worker.close()
    launcher.handle.exit = ProcessExit(exit_code=0, reaped=True)

    def cannot_remove(_path):
        raise PermissionError("simulated removal denied")

    with monkeypatch.context() as fault:
        fault.setattr(store.backend, "worktree_remove", cannot_remove)
        terminal = supervisor.poll(run.run_id, active_generation=1)
        assert terminal.terminal and terminal.reaped and terminal.exit_code == 0
        assert terminal.recovery_status == "cleanup_failed"
        assert "removal denied" in terminal.recovery_error
        assert store.state(terminal.recovery_state_id).read("precious.txt") == "keep"
        assert worktree.exists()
    supervisor.close()
    with (
        closing(Store.open(store.root, store.session)) as reopened_store,
        CentralSupervisor(reopened_store, launcher=launcher) as reopened,
    ):
        recovered, = reopened.reconcile(active_generation=1)
        assert recovered.recovery_status == "complete"
        assert recovered.recovery_state_id == terminal.recovery_state_id
        assert not recovered.uncertain
        assert not worktree.exists()
    assert launcher.launch_calls == launcher.handle.termination_calls == 1


def test_exit_is_durable_before_interruption_inside_snapshot(store: Store, monkeypatch) -> None:
    launcher = FakeLauncher()
    supervisor = CentralSupervisor(store, launcher=launcher)
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)
    supervisor.start(run.run_id, active_generation=1)
    worker = store.branch(run.assignment.worker)
    worker.write("precious.txt", "keep")
    worktree = worker.worktree
    worker.close()
    launcher.handle.exit = ProcessExit(exit_code=3, reaped=True)
    real_stage = GitBackend.stage_all

    def interrupted(backend):
        if backend.path == worktree:
            durable = supervisor.get(run.run_id)
            assert durable.terminal and durable.exit_code == 3
            raise SystemExit("interrupted snapshot")
        real_stage(backend)

    with monkeypatch.context() as fault:
        fault.setattr(GitBackend, "stage_all", interrupted)
        with pytest.raises(SystemExit, match="interrupted snapshot"):
            supervisor.poll(run.run_id, active_generation=1)
    supervisor.close()
    with (
        closing(Store.open(store.root, store.session)) as reopened_store,
        CentralSupervisor(reopened_store, launcher=launcher) as reopened,
    ):
        durable = reopened.get(run.run_id)
        assert durable.terminal and durable.exit_code == 3
        assert durable.recovery_status == "pending"
        recovered, = reopened.reconcile(active_generation=1)
        assert reopened_store.state(recovered.recovery_state_id).read("precious.txt") == "keep"
    assert launcher.launch_calls == launcher.handle.termination_calls == 1


def test_stale_worktree_preservation_is_not_misreported_as_complete_capture(store: Store) -> None:
    launcher = FakeLauncher()
    supervisor = CentralSupervisor(store, launcher=launcher)
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)
    supervisor.start(run.run_id, active_generation=1)
    worker = store.branch(run.assignment.worker)
    worker.write("precious.txt", "only copy")
    worktree = worker.worktree
    worker.close()
    (worktree / ".git").unlink()
    launcher.handle.exit = ProcessExit(exit_code=0, reaped=True)
    terminal = supervisor.poll(run.run_id, active_generation=1)
    assert terminal.terminal and terminal.recovery_status == "unavailable"
    assert terminal.recovery_state_id is None
    assert str(worktree) in terminal.recovery_error
    assert (worktree / "precious.txt").read_text() == "only copy"
    assert not store.worktree_recoveries(run.assignment.worker)
    assert supervisor.reconcile(active_generation=1) == (terminal,)
    with pytest.raises(AssignmentIdentityConflict, match="recovery"):
        supervisor.prepare(assignment_for(supervisor, attempt=1), wall_timeout_seconds=30)
    supervisor.close()


def test_missing_worktree_without_capture_is_not_recreated_as_success(store: Store) -> None:
    launcher = FakeLauncher()
    supervisor = CentralSupervisor(store, launcher=launcher)
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)
    supervisor.start(run.run_id, active_generation=1)
    worktree = store.worktree_path_for(run.assignment.worker)
    shutil.rmtree(worktree)
    launcher.handle.exit = ProcessExit(exit_code=1, reaped=True)
    terminal = supervisor.poll(run.run_id, active_generation=1)
    assert terminal.recovery_status == "unavailable"
    assert terminal.recovery_state_id is None
    assert "Missing worktree" in terminal.recovery_error
    assert not worktree.exists()
    assert supervisor.reconcile(active_generation=1) == (terminal,)
    supervisor.close()


def test_missing_worker_ref_records_exit_without_inventing_recovery_lineage(store: Store) -> None:
    launcher = FakeLauncher()
    supervisor = CentralSupervisor(store, launcher=launcher)
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)
    supervisor.start(run.run_id, active_generation=1)
    worker = store.branch(run.assignment.worker)
    worker.write("precious.txt", "only copy")
    worktree = worker.worktree
    worker.close()
    store.backend.delete_ref(store.ref_for(run.assignment.worker))
    launcher.handle.exit = ProcessExit(exit_code=1, reaped=True)
    terminal = supervisor.poll(run.run_id, active_generation=1)
    assert terminal.terminal and terminal.exit_code == 1
    assert terminal.recovery_status == "unavailable"
    assert terminal.recovery_state_id is None
    assert "Missing worker branch ref" in terminal.recovery_error
    assert (worktree / "precious.txt").read_text() == "only copy"
    assert store.backend.ref_sha(store.ref_for(run.assignment.worker)) is None
    supervisor.close()


@pytest.mark.parametrize("terminal", [False, True])
def test_legacy_supervisor_wire_roundtrip_preserves_immutable_observations(
    store: Store, terminal: bool,
) -> None:
    with CentralSupervisor(store, launcher=FakeLauncher()) as supervisor:
        run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)
        if terminal:
            run = supervisor.stop(run.run_id)
        raw = run.to_dict()
        raw["schema"] = "taste.brains/SupervisorRun/1"
        raw.pop("recovery_status")
        raw.pop("recovery_error")
        restored = SupervisorRun.from_dict(raw)
        assert restored.to_dict() == raw
        assert restored.recovery_status == ("complete" if terminal else "not_started")
        if terminal:
            raw["recovery_state_id"] = None
            with pytest.raises(ValueError, match="exact recovery state"):
                SupervisorRun.from_dict(raw)


def test_legacy_run_transitions_to_new_wire_without_rewriting_history(store: Store, monkeypatch) -> None:
    real_serialize = SupervisorRun.to_dict

    def legacy_serialize(run):
        raw = real_serialize(run)
        raw["schema"] = "taste.brains/SupervisorRun/1"
        raw.pop("recovery_status", None)
        raw.pop("recovery_error", None)
        return raw

    with CentralSupervisor(store, launcher=FakeLauncher()) as supervisor:
        with monkeypatch.context() as legacy_writer:
            legacy_writer.setattr(SupervisorRun, "to_dict", legacy_serialize)
            prepared = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)
        old_head = supervisor.control.head
        old_bytes = old_head.read(supervisor_module._run_path(prepared.run_id))
        assert supervisor.get(prepared.run_id).to_dict()["schema"] == "taste.brains/SupervisorRun/1"
        terminal = supervisor.stop(prepared.run_id)
        assert terminal.to_dict()["schema"] == supervisor_module.RUN_SCHEMA
        assert terminal.recovery_status == "complete"
        assert supervisor.runs() == (terminal,)
        assert old_head.read(supervisor_module._run_path(prepared.run_id)) == old_bytes


@pytest.mark.parametrize("boundary", ["process_terminal", "worker_captured", "worker_recovered"])
def test_control_persistence_failure_is_explicit_and_storage_recovery_remains_retryable(
    store: Store, monkeypatch, boundary: str,
) -> None:
    launcher = FakeLauncher()
    supervisor = CentralSupervisor(store, launcher=launcher)
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)
    supervisor.start(run.run_id, active_generation=1)
    worker = store.branch(run.assignment.worker)
    worker.write("precious.txt", "keep")
    worktree = worker.worktree
    worker.close()
    launcher.handle.exit = ProcessExit(exit_code=0, reaped=True)
    real_checkpoint = supervisor.control.checkpoint

    def cannot_persist(reason, *args, **kwargs):
        if reason.startswith(f"supervisor {boundary}:"):
            raise OSError(errno.ENOSPC, "simulated control disk full")
        return real_checkpoint(reason, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(supervisor.control, "checkpoint", cannot_persist)
        with pytest.raises(OSError, match="control disk full"):
            supervisor.poll(run.run_id, active_generation=1)
    durable = supervisor.get(run.run_id)
    if boundary == "process_terminal":
        assert not durable.terminal
        assert store.view(run.assignment.worker).head.read("precious.txt") is None
    else:
        assert durable.terminal and durable.exit_code == 0
        assert durable.recovery_status == ("pending" if boundary == "worker_captured" else "cleanup_pending")
    if boundary != "worker_recovered":
        assert (worktree / "precious.txt").read_text() == "keep"
    supervisor.close()
    with (
        closing(Store.open(store.root, store.session)) as reopened_store,
        CentralSupervisor(reopened_store, launcher=launcher) as reopened,
    ):
        recovered, = reopened.reconcile(active_generation=1)
        assert recovered.recovery_status == "complete"
        assert reopened_store.state(recovered.recovery_state_id).read("precious.txt") == "keep"
        assert not worktree.exists()
    assert launcher.launch_calls == 1


def _killed_during_terminal_snapshot(root: str, session: str, reached) -> None:
    store = Store.open(Path(root), session)
    launcher = FakeLauncher()
    supervisor = CentralSupervisor(store, launcher=launcher)
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)
    supervisor.start(run.run_id, active_generation=1)
    worker = store.branch(run.assignment.worker)
    worker.write("precious.txt", "survives SIGKILL")
    worktree = worker.worktree
    worker.close()
    launcher.handle.exit = ProcessExit(exit_code=17, reaped=True)
    real_stage = GitBackend.stage_all

    def kill_at_capture(backend):
        if backend.path == worktree:
            reached.set()
            os.kill(os.getpid(), signal.SIGKILL)
        real_stage(backend)

    GitBackend.stage_all = kill_at_capture
    supervisor.poll(run.run_id, active_generation=1)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX SIGKILL and fork")
def test_real_supervisor_kill_after_exit_record_resumes_capture(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    context = mp.get_context("fork")
    reached = context.Event()
    process = context.Process(target=_killed_during_terminal_snapshot, args=(str(root), "s", reached))
    process.start()
    try:
        assert reached.wait(30), "child did not reach the capture boundary"
        process.join(30)
        assert process.exitcode == -signal.SIGKILL
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
    launcher = FakeLauncher()
    with closing(Store.open(root, "s")) as reopened_store, CentralSupervisor(reopened_store, launcher=launcher) as reopened:
        durable, = reopened.runs()
        assert durable.terminal and durable.exit_code == 17 and durable.reaped
        assert durable.recovery_status == "pending"
        recovered, = reopened.reconcile(active_generation=1)
        assert recovered.recovery_status == "complete"
        assert reopened_store.state(recovered.recovery_state_id).read("precious.txt") == "survives SIGKILL"
    assert launcher.launch_calls == launcher.recover_calls == launcher.handle.termination_calls == 0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_subprocess_launcher_observes_ready_and_reaps_process_group(store: Store) -> None:
    command = [
        sys.executable,
        "-c",
        (
            "from taste.brains.supervisor import mark_worker_ready; "
            "import time; mark_worker_ready(); time.sleep(30)"
        ),
    ]
    launcher = SubprocessLauncher(command)
    supervisor = CentralSupervisor(store, launcher=launcher, termination_grace=0.1)
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=20)
    supervisor.start(run.run_id, active_generation=1)
    deadline = time.monotonic() + 5
    observed = supervisor.get(run.run_id)
    while time.monotonic() < deadline and not observed.ready:
        time.sleep(0.02)
        observed = supervisor.poll(run.run_id, active_generation=1)
    assert observed.ready

    terminal = supervisor.stop(run.run_id, "test_stop")
    assert terminal.phase == "terminal"
    assert terminal.reaped
    assert launcher.recover(supervisor._spec(observed)).poll() is not None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX descendant identity")
def test_subprocess_launcher_reaps_a_descendant_that_detaches_its_group(
    store: Store,
) -> None:
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import subprocess, sys, time; "
            "from taste.brains.supervisor import mark_worker_ready, _process_identity; "
            "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
            "start_new_session=True); "
            "Path('child.json').write_text(str(child.pid)+'\\n'+_process_identity(child.pid)); "
            "mark_worker_ready(); time.sleep(30)"
        ),
    ]
    launcher = SubprocessLauncher(command)
    supervisor = CentralSupervisor(store, launcher=launcher, termination_grace=0.1)
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=20)
    child_pid = 0
    child_identity = ""
    try:
        supervisor.start(run.run_id, active_generation=1)
        marker = store.worktree_path_for("worker-1") / "child.json"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.02)
            supervisor.poll(run.run_id, active_generation=1)
        child_pid_text, child_identity = marker.read_text().splitlines()
        child_pid = int(child_pid_text)
        assert supervisor_module._process_identity(child_pid) == child_identity

        terminal = supervisor.stop(run.run_id, "test_detached_stop")

        assert terminal.phase == "terminal"
        assert terminal.reaped
        deadline = time.monotonic() + 5
        while (
            time.monotonic() < deadline
            and supervisor_module._process_identity(child_pid) == child_identity
        ):
            time.sleep(0.02)
        assert supervisor_module._process_identity(child_pid) != child_identity
    finally:
        if (
            child_pid > 0
            and child_identity
            and supervisor_module._process_identity(child_pid) == child_identity
        ):
            os.kill(child_pid, 9)


@pytest.mark.skipif(
    sys.platform != "darwin" and not sys.platform.startswith("linux"),
    reason="process-environment marker discovery requires Darwin or Linux",
)
def test_subprocess_launcher_finds_detached_child_when_root_exits_before_first_poll(
    store: Store,
) -> None:
    class RootExitedBeforeReturnLauncher(SubprocessLauncher):
        def launch(self, spec):
            handle = super().launch(spec)
            assert isinstance(handle, supervisor_module._SubprocessHandle)
            assert handle._popen is not None
            handle._popen.wait(timeout=5)
            assert not handle._observed_root_live
            return handle

    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import subprocess, sys; "
            "from taste.brains.supervisor import _process_identity; "
            "child=subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(30)'], start_new_session=True); "
            "Path('child.json').write_text(str(child.pid)+'\\n'+_process_identity(child.pid))"
        ),
    ]
    launcher = RootExitedBeforeReturnLauncher(command)
    supervisor = CentralSupervisor(store, launcher=launcher, termination_grace=0.1)
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=20)
    child_pid = 0
    child_identity = ""
    try:
        terminal = supervisor.start(run.run_id, active_generation=1)
        child_record = store.view("worker-1").head.read("child.json")
        assert child_record is not None
        child_pid_text, child_identity = child_record.splitlines()
        child_pid = int(child_pid_text)

        assert terminal.phase == "terminal"
        assert terminal.uncertain
        assert not terminal.reaped
        deadline = time.monotonic() + 5
        while (
            time.monotonic() < deadline
            and supervisor_module._process_identity(child_pid) == child_identity
        ):
            time.sleep(0.02)
        assert supervisor_module._process_identity(child_pid) != child_identity

        member_record = json.loads(
            supervisor_module._members_path(supervisor._spec(terminal).launch_path).read_text()
        )
        assert {"pid": child_pid, "process_identity": child_identity} in member_record[
            "members"
        ]
    finally:
        if (
            child_pid > 0
            and child_identity
            and supervisor_module._process_identity(child_pid) == child_identity
        ):
            os.kill(child_pid, 9)


def test_recovered_root_exit_is_not_hidden_by_a_descendant_lease_lock(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inherited flock denotes the run, not continued root liveness."""
    supervisor = CentralSupervisor(store, launcher=FakeLauncher())
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=20)
    spawned = supervisor.start(run.run_id, active_generation=1)
    handle = supervisor_module._SubprocessHandle(
        spec=supervisor._spec(spawned),
        pid=spawned.pid or 4101,
        process_group_id=spawned.process_group_id or 4101,
        launch_token=spawned.launch_token or "token",
        process_identity="exact-birth",
        popen=None,
    )
    monkeypatch.setattr(supervisor_module.sys, "platform", "linux")
    monkeypatch.setattr(supervisor_module, "_linux_process_info", lambda _pid: None)
    monkeypatch.setattr(supervisor_module, "_lock_is_held", lambda _path: True)

    assert handle.poll() == ProcessExit(reaped=False)
