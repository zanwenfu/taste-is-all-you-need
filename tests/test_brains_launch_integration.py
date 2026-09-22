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

**What these tests claim, and what they do not.** The subject is the seam, not
the outcome: that the argv starts, that the environment the launcher exports is
the one the entrypoint validates, and that the process binds to its exact
prepared assignment. How far the worker then gets depends on the machine.

An earlier version of this file asserted that a worker *cannot* certify
completion here, on the reasoning that certification needs a live model. That
stopped being true: with a credential reachable, this test now routinely exits
0 with a completed report. The premise was not wrong when written -- the
monitor could not parse a verdict, so no run had ever finished -- and leaving
it in place would have been a stale rationale outliving its evidence.

So three exits are all consistent with a healthy seam, and which one occurs is
a property of the machine:

* ``COMPLETED`` (0) -- a credential was reachable and the run finished. This is
  the *strongest* evidence for the seam, because reaching completion means
  every earlier step worked;
* ``INCOMPLETE`` (10) -- the provider validated, the runtime ran, and published
  a durable ``WorkerReport`` declining completion;
* ``INFRA_FAILURE`` (71) -- ``ensure_ready`` could not validate the monitor
  provider, so the run stopped before the lease and published no report.

Only ``INPUT_REJECTED`` (65) indicts the seam, because it means the process
could not bind to its own durable assignment.

An earlier draft asserted 10 alone, having measured it here. That measurement
was contaminated: ``LLM`` calls ``load_dotenv(_find_env(...))``, which walks up
the directory tree and finds this repository's own ``.env``, so a credential is
reachable on a developer machine no matter what the test strips from the
environment. CI has no ``.env`` and got 71 on both 3.11 and 3.12. The lesson is
in the assertion now: pin the property, not the environment.
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

SEAM_INTACT = frozenset(
    {
        int(WorkerExitCode.COMPLETED),
        int(WorkerExitCode.INCOMPLETE),
        int(WorkerExitCode.INFRA_FAILURE),
    }
)
"""Exits that are all consistent with a working launch seam.

Which one occurs depends on whether a model credential is reachable, which is a
property of the machine rather than of the seam. ``INPUT_REJECTED`` is
deliberately absent: it is the one exit that indicts the seam itself.
"""

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
    """The inherited environment with the obvious model credentials removed.

    Note what this cannot do: ``LLM`` loads ``.env`` by walking up from the
    repository root, so on a developer machine a key stays reachable however
    the environment is scrubbed. That is why the assertions accept either
    fail-closed boundary instead of pinning the one this machine produces.
    """
    env = dict(os.environ)
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "OPENAI_API_KEY",
    ):
        env.pop(name, None)
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

    assert argv[:3] == (sys.executable, "-I", "-c")
    # The module really is importable and really does parse arguments.
    probe = subprocess.run(
        [*argv, "--help"],
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

    The assertion is that the process got far enough for its exit and its
    durable report to agree -- not that it stopped in one particular place. A
    bad argv, a missing module, or an environment the entrypoint rejects would
    all fail earlier and leave no report at all, which is what this
    distinguishes.
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

    # Wide on purpose. This deadline exists to tell "never started" from
    # "still working", not to assert a speed. When the worker could not reach a
    # model it died in seconds and 90s was generous; now that it completes,
    # measured single-worker runs land at 35-80s and the same 90s straddled the
    # distribution, making this test flaky by construction. 240s is roughly
    # three times the slowest completion observed and still fails fast when the
    # seam is genuinely broken.
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline and handle.poll() is None:
        time.sleep(0.05)
    exit_code = handle.poll()

    assert exit_code is not None, (
        "the worker process never exited; the seam started it but nothing "
        "terminated it within the deadline"
    )
    observed = exit_code.exit_code
    assert observed in SEAM_INTACT, (
        f"expected an exit consistent with a working seam, got {exit_code}. "
        f"INPUT_REJECTED ({int(WorkerExitCode.INPUT_REJECTED)}) means the process "
        "could not bind to its own durable assignment, which is the seam this "
        "test exists to check; anything else means it never ran at all."
    )

    # Whether a report exists, and what it says, follows from WHERE the run
    # stopped. Asserting that correspondence is stronger than asserting any
    # one exit code, and it is what survives the machine changing underneath.
    report = store.view(assignment.contract.identity).head.read(WORKER_REPORT_PATH)
    if observed == int(WorkerExitCode.COMPLETED):
        assert report is not None, "a completed run published no durable report"
        assert json.loads(report)["completed"] is True
    elif observed == int(WorkerExitCode.INCOMPLETE):
        assert report is not None, "a validated runtime published no durable report"
        assert json.loads(report)["completed"] is False
    else:
        assert report is None, (
            "the run stopped at provider validation, before the lease, so it "
            "must not have published a report"
        )


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
