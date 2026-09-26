"""New assignments on reused workers retain prior history and paid receipts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from taste.brains.azure_worker_entrypoint import run_directory
from taste.brains.azure_worker_launch import worker_command
from taste.brains.communication import Communicator, Message
from taste.brains.supervisor import CentralSupervisor, SubprocessLauncher
from taste.brains.worker_protocol import WORKER_REPORT_PATH, assignment_run_id
from taste.memstore import Branch
from tests.test_azure_worker_policy import environment
from tests.test_azure_worker_process import BOOTSTRAP, assigned
from tests.test_azure_worker_runtime import worker as _worker
from tests.test_worker_admission import launch as _launch

launch = _launch
worker = _worker


def _journal_hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("next_generation,crash", [(1, None), (2, None), (1, "before"), (1, "after")])
def test_second_real_worker_attempt_has_fresh_context_and_preserves_first_receipts(
        worker, next_generation, crash, monkeypatch):
    def command(spec):
        bootstrap = BOOTSTRAP
        if spec.assignment.attempt == 0:
            bootstrap = bootstrap.replace("install(network)", """
import json
from tests.test_azure_worker_runtime import accepted, message
install(network, worker_reply=lambda number, payload: [
    message(json.dumps({**accepted(payload), "status": "blocked"}))])
""")
        argv = list(worker_command(spec, repo_root=worker.store.root, session=worker.store.session))
        # Test response helpers import the entrypoint for their own tests;
        # the actual child must execute a fresh module, as production does.
        bootstrap += "\nsys.modules.pop('taste.brains.azure_worker_entrypoint', None)\n"
        argv[3] = argv[3].replace("import runpy;", bootstrap + "\nimport runpy;", 1)
        return argv

    supervisor = CentralSupervisor(worker.store, launcher=SubprocessLauncher(command, env=environment()))
    first_assignment = assigned(supervisor, worker)
    runs = []
    try:
        first = supervisor.prepare(first_assignment, wall_timeout_seconds=30)
        runs.append(first.run_id)
        inbox = Communicator(worker.store).send(Message.create(
            idempotency_key="first-attempt-feedback", kind="feedback", sender="central",
            recipient=first_assignment.worker, generation=1,
            payload={"instruction": "produce the declared artifact"}))
        supervisor.start(first.run_id, active_generation=1)
        terminal = supervisor.wait(first.run_id, active_generation=1)
        assert terminal.reaped and terminal.exit_code == 10, terminal.to_dict()
        old_report = supervisor.collect(first.run_id, active_generation=1)
        old_report_state = worker.store.state(supervisor.get(first.run_id).report_state_id)
        assert not old_report.completed and old_report.cost_usd is not None
        assert old_report_state.read(WORKER_REPORT_PATH) == old_report.to_json()
        old_transcript = old_report_state.transcript.turns
        assert any(turn.get("kind") == "responses_binding" for turn in old_transcript)
        old_directory = run_directory(worker.store, first.run_id)
        old_receipts = _journal_hashes(old_directory)
        assert len(old_receipts) >= 2
        assert Communicator(worker.store).pending(first_assignment.worker) == ()

        second_assignment = replace(first_assignment, attempt=1, generation=next_generation)
        if crash:
            original = Branch.checkpoint
            failed = []

            def interrupted(branch, reason, **kwargs):
                if branch.name == first_assignment.worker and reason.startswith("prepare ") and not failed:
                    failed.append(True)
                    if crash == "after":
                        original(branch, reason, **kwargs)
                    raise OSError("interrupted assignment preparation")
                return original(branch, reason, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(Branch, "checkpoint", interrupted)
                with pytest.raises(OSError, match="interrupted assignment preparation"):
                    supervisor.prepare(second_assignment, wall_timeout_seconds=30)
            assert failed
            assert _journal_hashes(old_directory) == old_receipts
        second = supervisor.prepare(second_assignment, wall_timeout_seconds=30)
        runs.append(second.run_id)
        repeated = supervisor.prepare(second_assignment, wall_timeout_seconds=30)
        assert repeated.prepared_state_id == second.prepared_state_id
        supervisor.start(second.run_id, active_generation=next_generation)
        second_exit = supervisor.wait(second.run_id, active_generation=next_generation)
        assert second_exit.reaped and second_exit.exit_code == 0, second_exit.to_dict()
        new_report = supervisor.collect(second.run_id, active_generation=next_generation)
        assert new_report.completed and not new_report.uncertain
        assert new_report.cost_usd == pytest.approx(0.001464)
        assert new_report.run_id != old_report.run_id
        assert _journal_hashes(old_directory) == old_receipts

        prepared = worker.store.state(second.prepared_state_id)
        assert prepared.transcript.turns == ()
        assert prepared.read(WORKER_REPORT_PATH) is None
        assert old_report_state.read(WORKER_REPORT_PATH) == old_report.to_json()
        assert old_report_state.transcript.turns == old_transcript
        assert worker.store.backend.is_ancestor(old_report_state.id, prepared.id)
        assert assignment_run_id(second_assignment) == new_report.run_id
        turns = worker.store.state(new_report.final_state_id).transcript.turns
        bindings = [turn["binding"]["run_id"] for turn in turns if turn.get("kind") == "responses_binding"]
        assert bindings == [second.run_id]
        inputs = [json.loads(turn["content"]) for turn in turns if turn.get("kind") == "responses_input"
                  and turn.get("id", "").startswith("inbox.")]
        assert any(item["inbox_id"] == inbox.inbox_id for item in inputs) == (next_generation == 1)
        assert supervisor.deliver(second.run_id, active_generation=next_generation).ok
        assert supervisor.integration.head.read("output.txt") == "correct"
    finally:
        for run_id in runs:
            if not supervisor.get(run_id).terminal:
                supervisor.stop(run_id, "test_cleanup")


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo"])
def test_unsafe_old_report_is_rejected_before_prepare_changes_any_input(worker, tmp_path, kind):
    import os

    from taste.brains.supervisor import AssignmentIdentityConflict
    from tests.test_brains_supervisor import FakeLauncher

    supervisor = CentralSupervisor(worker.store, launcher=FakeLauncher())
    assignment = assigned(supervisor, worker)
    first = supervisor.prepare(assignment, wall_timeout_seconds=30)
    supervisor.stop(first.run_id)
    branch = worker.store.branch(assignment.worker)
    outside = tmp_path / "outside.txt"
    outside.write_text("retain outside bytes")
    path = branch.path(WORKER_REPORT_PATH)
    if kind == "symlink":
        path.symlink_to(outside)
    elif kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    before = branch.head.id
    contract_text = branch.path("contract.json").read_bytes()
    assignment_text = branch.path("assignment.json").read_bytes()
    branch.close()
    with pytest.raises(AssignmentIdentityConflict):
        supervisor.prepare(replace(assignment, attempt=1), wall_timeout_seconds=30)
    assert worker.store.view(assignment.worker).head.id == before
    assert path.parent.joinpath("contract.json").read_bytes() == contract_text
    assert path.parent.joinpath("assignment.json").read_bytes() == assignment_text
    assert outside.read_text() == "retain outside bytes"


def test_invalid_azure_policy_is_rejected_before_worker_creation(worker):
    from taste.brains.worker_admission import EntrypointInputError
    from tests.test_brains_supervisor import FakeLauncher

    supervisor = CentralSupervisor(worker.store, launcher=FakeLauncher())
    assignment = assigned(supervisor, worker)
    raw = dict(assignment.resources["azure_openai"], pricing_sha="changed")
    assignment = replace(assignment, resources={**assignment.resources, "azure_openai": raw})
    before = supervisor.control.head.id
    with pytest.raises(EntrypointInputError):
        supervisor.prepare(assignment, wall_timeout_seconds=30)
    assert not worker.store.view(assignment.worker).exists()
    assert supervisor.control.head.id == before


def test_repeating_same_preparation_cannot_clear_pending_context_or_private_journal(worker):
    from tests.test_brains_supervisor import FakeLauncher

    supervisor = CentralSupervisor(worker.store, launcher=FakeLauncher())
    assignment = assigned(supervisor, worker)
    first = supervisor.prepare(assignment, wall_timeout_seconds=30)
    branch = worker.store.branch(assignment.worker)
    branch.turn(kind="in_progress_marker", content="must survive prepare replay")
    branch.close()
    directory = run_directory(worker.store, first.run_id)
    directory.mkdir(mode=0o700)
    receipt = directory / "pending-dispatch-proof"
    receipt.write_bytes(b"retained unknown outcome")
    repeated = supervisor.prepare(assignment, wall_timeout_seconds=30)
    assert repeated.prepared_state_id == first.prepared_state_id
    assert worker.store.view(assignment.worker).pending_turns() == [
        {"kind": "in_progress_marker", "content": "must survive prepare replay"}]
    assert receipt.read_bytes() == b"retained unknown outcome"
