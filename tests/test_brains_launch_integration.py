"""The two halves that had never met: a real launcher and a real entrypoint.

Everything else in the brain-layer suite tests one half. ``test_brains_supervisor``
spawns genuine processes, but they are ``python -c 'time.sleep(30)'`` -- process
mechanics with no worker inside. ``test_brains_worker_entrypoint`` drives the
real entrypoint, but in-process through ``execute_worker`` with injected
factories -- a worker with no process around it. ``test_brains_central_runtime``
proves three-worker coordination on a ``FakeLauncher``.

So the seam between them was never exercised: nothing checked that the argv
``worker_command`` builds actually starts, that the environment
``SubprocessLauncher`` exports is the environment ``worker_entrypoint``
validates, or that a process launched this way binds to the exact prepared
assignment.

**What these tests deliberately do not claim.** A worker cannot certify
completion without a model, so these prove the launch seam and the durable
binding, not a finished run. Proving completion needs a live model and belongs
in a separate, paid test.

What a credential-free run *does* reach is worth stating, because it was
measured rather than predicted: the process validates its durable input, takes
the branch lease, builds brain and monitor, runs the runtime, publishes a
durable ``WorkerReport`` with ``completed=False``, and exits ``INCOMPLETE``
(10). It fails closed and says so in the store. An earlier draft of this file
asserted ``INFRA_FAILURE`` (71) from reading ``execute_worker``; the code
reaches its reporting boundary instead, which is the stronger behaviour.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from taste.memstore import Store

pytest.importorskip("claude_agent_sdk", reason="the brain layer needs claude-agent-sdk")

from taste.brains.contract import CONTRACT_PATH, Contract
from taste.brains.records import ArtifactSpec, Assignment, contract_digest
from taste.brains.supervisor import CentralSupervisor, SubprocessLauncher
from taste.brains.worker_entrypoint import (
    WorkerExitCode,
    assignment_run_id,
    worker_command,
    worker_command_factory,
)
from taste.brains.worker_runtime import ASSIGNMENT_PATH, WORKER_REPORT_PATH

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX process groups and fork handshake"
)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "repo", "session-1")
    yield s
    s.close()


def an_assignment(supervisor: CentralSupervisor, *, worker: str = "worker-1") -> Assignment:
    contract = Contract(
        identity=worker,
        task="write the parser",
        outputs=("parser.py",),
        success_criteria=("parser.py contains the checked implementation",),
    )
    return Assignment(
        assignment_id="parser-assignment",
        generation=1,
        attempt=0,
        contract=contract,
        contract_digest=contract_digest(contract),
        base_state_id=supervisor.integration.head.id,
        outputs=(ArtifactSpec(artifact_id="parser-output", path="parser.py"),),
        model="claude-sonnet-5",
    )


def _credential_free_env() -> dict[str, str]:
    """The inherited environment with every model credential removed.

    The worker must fail at the model boundary for a stated reason, not
    accidentally succeed because the developer running the suite happens to be
    authenticated. Removing the credentials is what makes the assertion mean
    something.
    """
    env = dict(os.environ)
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "OPENAI_API_KEY",
    ):
        env.pop(name, None)
    env["TASTE_NO_DOTENV"] = "1"
    return env


# --------------------------------------------------------------- the argv seam


def test_the_launcher_command_starts_a_real_python_module(store: Store) -> None:
    """The argv is not merely well-formed: it names a module that imports.

    A typo in the module path, or an entrypoint that cannot be imported, would
    surface in production as a process that dies instantly with no durable
    evidence -- which the supervisor would report as an uncertain outcome.
    """
    supervisor = CentralSupervisor(store, launcher=SubprocessLauncher(["/bin/true"]))
    assignment = an_assignment(supervisor)
    run = supervisor.prepare(assignment, wall_timeout_seconds=30)
    # start() is what records the durable deadline a LaunchSpec requires; a
    # prepared run has no process boundary yet. /bin/true exits immediately,
    # which is all this test needs from the process itself.
    supervisor.start(run.run_id, active_generation=1)
    spec = supervisor._spec(supervisor.get(run.run_id))

    argv = worker_command(spec, repo_root=store.root, session="session-1")

    assert argv[:3] == (sys.executable, "-m", "taste.brains.worker_entrypoint")
    # The module really is importable and really does parse arguments.
    probe = subprocess.run(
        [sys.executable, "-m", "taste.brains.worker_entrypoint", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stderr[-400:]
    assert "--prepared-state" in probe.stdout


def test_the_command_binds_to_the_exact_prepared_state(store: Store) -> None:
    """The launch argv carries the state the supervisor actually checkpointed.

    This is the binding the entrypoint re-validates before taking a lease, so a
    drift here is the difference between a worker reading its own assignment and
    a worker reading someone else's.
    """
    supervisor = CentralSupervisor(store, launcher=SubprocessLauncher(["/bin/true"]))
    assignment = an_assignment(supervisor)
    run = supervisor.prepare(assignment, wall_timeout_seconds=30)
    supervisor.start(run.run_id, active_generation=1)
    spec = supervisor._spec(supervisor.get(run.run_id))

    argv = list(worker_command(spec, repo_root=store.root, session="session-1"))
    prepared = argv[argv.index("--prepared-state") + 1]

    assert prepared == run.prepared_state_id
    # and that state really does hold this exact assignment
    assert store.state(prepared).read(ASSIGNMENT_PATH) == assignment.to_json()
    assert store.state(prepared).read(CONTRACT_PATH) == assignment.contract.to_json()
    assert argv[argv.index("--worker") + 1] == assignment.contract.identity
    assert "ANTHROPIC_API_KEY" not in " ".join(argv)


# ------------------------------------------------- launcher meets entrypoint


def test_a_real_launcher_runs_the_real_entrypoint_against_its_assignment(
    store: Store,
) -> None:
    """The seam itself: production argv, production launcher, real process.

    The assertion is that the process got far enough to publish a durable
    report and decline completion -- not that it crashed in some particular
    way. A bad argv, a missing module, or an environment the entrypoint rejects
    would all fail earlier and leave no report at all, which is exactly what
    this distinguishes.
    """
    launcher = SubprocessLauncher(
        worker_command_factory(store.root, "session-1"),
        env=_credential_free_env(),
        handshake_timeout=30.0,
    )
    supervisor = CentralSupervisor(store, launcher=launcher, termination_grace=0.5)
    assignment = an_assignment(supervisor)
    run = supervisor.prepare(assignment, wall_timeout_seconds=60)

    supervisor.start(run.run_id, active_generation=1)
    observed = supervisor.get(run.run_id)
    handle = launcher.recover(supervisor._spec(observed))
    assert handle is not None, "the launcher did not produce process evidence"

    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and handle.poll() is None:
        time.sleep(0.05)
    exit_code = handle.poll()

    assert exit_code is not None, "the worker process never exited"
    assert exit_code.exit_code == int(WorkerExitCode.INCOMPLETE), (
        f"expected a validated runtime to decline completion, got {exit_code}; "
        "INPUT_REJECTED (65) or INFRA_FAILURE (71) means the launch seam broke "
        "before the worker could reach its reporting boundary"
    )

    # The exit code alone could be a coincidence. The durable report is what
    # proves the worker validated its input, took its lease, and reported.
    report = store.view(assignment.contract.identity).head.read(WORKER_REPORT_PATH)
    assert report is not None, "a validated runtime published no durable report"
    assert json.loads(report)["completed"] is False


def test_the_launcher_exports_the_environment_the_entrypoint_requires(
    store: Store,
) -> None:
    """Three variables bind a process to one run, and both sides must agree.

    ``SubprocessLauncher`` writes them; ``worker_entrypoint`` refuses to start
    without them. They are defined in different modules, so nothing but a test
    that runs both keeps them in step.
    """
    probe = [
        sys.executable,
        "-c",
        (
            "import json, os, time; from pathlib import Path; "
            "from taste.brains.supervisor import mark_worker_ready; "
            "Path('worker-env.json').write_text(json.dumps("
            "{k: v for k, v in os.environ.items() if k.startswith('TASTE_WORKER_')})); "
            "mark_worker_ready(); time.sleep(30)"
        ),
    ]
    launcher = SubprocessLauncher(probe, handshake_timeout=30.0)
    supervisor = CentralSupervisor(store, launcher=launcher, termination_grace=0.5)
    run = supervisor.prepare(an_assignment(supervisor), wall_timeout_seconds=60)

    supervisor.start(run.run_id, active_generation=1)
    captured = store.worktree_path_for("worker-1") / "worker-env.json"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not captured.exists():
        time.sleep(0.02)
    assert captured.exists(), "the launched process never observed its environment"
    exported = json.loads(captured.read_text())

    assert exported["TASTE_WORKER_RUN_ID"] == assignment_run_id(
        supervisor.get(run.run_id).assignment
    )
    assert len(exported["TASTE_WORKER_LAUNCH_TOKEN"]) == 32
    assert Path(exported["TASTE_WORKER_READY_PATH"]).parent.exists()

    supervisor.stop(run.run_id, "test finished")


def test_a_launched_worker_is_reaped_with_its_descendants(store: Store) -> None:
    """Stopping a worker must stop what the worker started.

    Proven elsewhere against a synthetic command; proven here through the
    production launch path, because that is where the process group is actually
    established.
    """
    probe = [
        sys.executable,
        "-c",
        (
            "import subprocess, sys, time; "
            "from taste.brains.supervisor import mark_worker_ready; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "mark_worker_ready(); time.sleep(60)"
        ),
    ]
    launcher = SubprocessLauncher(probe, handshake_timeout=30.0)
    supervisor = CentralSupervisor(store, launcher=launcher, termination_grace=0.5)
    run = supervisor.prepare(an_assignment(supervisor), wall_timeout_seconds=60)

    supervisor.start(run.run_id, active_generation=1)
    deadline = time.monotonic() + 30
    observed = supervisor.get(run.run_id)
    while time.monotonic() < deadline and not observed.ready:
        time.sleep(0.05)
        observed = supervisor.poll(run.run_id, active_generation=1)
    assert observed.ready, "the launched worker never signalled readiness"

    terminal = supervisor.stop(run.run_id, "test_stop")

    assert terminal.phase == "terminal"
    assert terminal.reaped, "the supervisor could not confirm the tree was reaped"
