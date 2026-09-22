"""Regressions for report, transcript and real process import boundaries."""
import asyncio
import subprocess
import sys

import pytest

from taste.brains.records import WorkerReport
from taste.brains.session_store import MemstoreSessionStore
from taste.brains.supervisor import CentralSupervisor, ProcessExit, SubprocessLauncher
from taste.brains.worker_entrypoint import worker_command
from taste.brains.worker_runtime import WORKER_REPORT_PATH
from taste.memstore import Store
from tests.test_brains_central_runtime import FakeLauncher, ScriptedTransport, simple_goal, stack
from tests.test_brains_central_runtime import assignment_for as planned_assignment
from tests.test_brains_runtime_boundaries import one
from tests.test_brains_supervisor import assignment_for


@pytest.fixture
def store(tmp_path):
    opened = Store.open(tmp_path / "repo", "boundaries")
    yield opened
    opened.close()


def test_incomplete_report_without_outputs_reaches_next_planner_with_cost_and_reason(store):
    transport, launcher = ScriptedTransport(one), FakeLauncher()
    runtime, _ = stack(store, simple_goal(), transport, launcher)
    runtime.cycle()
    runtime.cycle()
    (run,) = runtime.supervisor.runs()
    worker = store.branch(run.assignment.worker)
    state = worker.checkpoint("failed before producing product")
    report = WorkerReport(
        report_id="failed-report", run_id=run.run_id,
        assignment_id=run.assignment.assignment_id, worker=run.assignment.worker,
        generation=1, attempt=run.assignment.attempt,
        contract_digest=run.assignment.contract_digest, base_state_id=run.assignment.base_state_id,
        final_state_id=state.id, at=state.meta.created_at, completed=False,
        terminal_reason="command_denied", outputs=(), cost_usd=1.75,
        metadata={"denials": ["missing compiler"]},
    )
    worker.write(WORKER_REPORT_PATH, report.to_json())
    worker.checkpoint("persist failure report")
    worker.close()
    launcher.handles[run.run_id].exit = ProcessExit(exit_code=1, reaped=True)
    # This replan is inspected before its new worker can run.
    def respond(request, prompt):
        assert "command_denied" in prompt
        assert "missing compiler" in prompt
        assert "1.75" in prompt
        return (planned_assignment(request, "next", "worker-next", "next.txt"),)

    transport.responder = respond
    outcome = runtime.cycle()
    assert any(t.kind == "worker_incomplete" for t in outcome.triggers)
    assert runtime.supervisor.get(run.run_id).report_id == report.report_id
    assert outcome.budget.worker_spent_usd == 1.75
    assert runtime.integration.head.read("product.txt") is None


@pytest.mark.parametrize("checkpoint_delete", [False, True])
def test_deleted_committed_transcript_stays_deleted_across_adapter_and_store_reopen(store, checkpoint_delete):
    key = {"project_key": "project", "session_id": "session"}
    child = {**key, "subpath": "child"}
    adapter = MemstoreSessionStore(store, "brain")
    for item in (key, child):
        asyncio.run(adapter.append(item, [{"type": "user", "uuid": "turn", "message": {"role": "user", "content": "hello"}}]))
    branch = store.branch("brain")
    original = branch.checkpoint("saved transcripts")
    branch.write("unrelated-dirty.txt", "must not be committed by delete")
    asyncio.run(adapter.delete(key))
    assert branch.head.id == original.id
    if checkpoint_delete:
        branch.checkpoint("intentional user checkpoint")
    root, session = store.root, store.session
    store.close()
    reopened = Store.open(root, session)
    try:
        adapter = MemstoreSessionStore(reopened, "brain")
        assert asyncio.run(adapter.load(key)) is None
        assert asyncio.run(adapter.load(child)) is None
        assert asyncio.run(adapter.list_sessions("project")) == []
        assert asyncio.run(adapter.list_subkeys(key)) == []
        # The old state is retained, and a deliberate rollback restores it.
        assert original.read(adapter._path(key)) is not None
        reopened.branch("brain").rollback(reopened.state(original.id), "restore deleted transcript")
        assert asyncio.run(adapter.load(key)) is not None
    finally:
        reopened.close()


def test_actual_launcher_and_worker_argv_ignore_task_python_modules(store, tmp_path):
    marker = tmp_path / "exec-ok"
    launcher = SubprocessLauncher([sys.executable, "-I", "-c",
                                  f"from pathlib import Path; Path({str(marker)!r}).touch()"])
    supervisor = CentralSupervisor(store, launcher=launcher)
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=20)
    path = store.worktree_path_for(run.assignment.worker)
    for name in ("types.py", "logging.py", "git.py", "json.py"):
        (path / name).write_text("raise RuntimeError('task module shadowed harness')\n")
    # Prepared-state integrity forbids arbitrary changes after prepare. Capture
    # the files in the product base instead, then prepare an exact fresh run.
    for name in ("types.py", "logging.py", "git.py", "json.py"):
        (path / name).unlink()
        supervisor.integration.write(name, "raise RuntimeError('task module shadowed harness')\n")
    supervisor.integration.checkpoint("task contains ordinary Python module names")
    supervisor.stop(run.run_id)
    run = supervisor.prepare(assignment_for(supervisor, assignment_id="shadow", worker="shadow"),
                             wall_timeout_seconds=20)
    try:
        supervisor.start(run.run_id, active_generation=1)
        supervisor.wait(run.run_id, active_generation=1)
        assert marker.exists()
        spec = supervisor._spec(supervisor.get(run.run_id))
        argv = worker_command(spec, repo_root=store.root, session=store.session)
        # Use the real generated command from a hostile task cwd, without
        # reaching any model-client constructor.
        cwd = tmp_path / "shadow-cwd"
        cwd.mkdir()
        (cwd / "types.py").write_text("raise RuntimeError('wrong types')")
        result = subprocess.run([*argv, "--help"], cwd=cwd, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        assert "--prepared-state" in result.stdout
    finally:
        supervisor.stop(run.run_id)
        supervisor.close()
