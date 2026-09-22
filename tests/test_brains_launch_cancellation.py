"""Cancellation across the real fork / PID publication / exec boundaries."""
from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from pathlib import Path

import pytest

from taste.brains import supervisor as module
from taste.brains.supervisor import CentralSupervisor, SubprocessLauncher, SupervisorError
from taste.memstore import Store
from tests.test_brains_supervisor import FakeLauncher, assignment_for


def wait_for(path: Path) -> None:
    deadline = time.monotonic() + 10
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists(), f"child did not reach barrier: {path}"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux process discovery")
@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("boundary", ["before_import", "before_pid", "before_exec"])
def test_stop_fences_delayed_bootstrap(tmp_path, monkeypatch, restart, boundary):
    entered, release, executed = (tmp_path / name for name in ("entered", "release", "executed"))
    processes = []
    popen = subprocess.Popen

    def paused_popen(argv, *args, **kwargs):
        if "-c" in argv and "_subprocess_bootstrap" in argv[argv.index("-c") + 1]:
            argv = list(argv)
            index = argv.index("-c") + 1
            pause = (
                f"from pathlib import Path; import time; Path({str(entered)!r}).touch(); "
                f"\nwhile not Path({str(release)!r}).exists(): time.sleep(0.01)\n"
            )
            if boundary == "before_import":
                argv[index] = pause + argv[index]
            else:
                # The production import setup is retained, including -I and
                # its explicit code root. Install the hook immediately before
                # the child enters the actual bootstrap implementation.
                hook = (
                    "import taste.brains.supervisor as m\n"
                    "def pause(*a, **kw):\n"
                    + "\n".join("    " + line for line in pause.splitlines()) + "\n"
                )
                if boundary == "before_pid":
                    hook += "    return original(*a, **kw)\noriginal=m._atomic_json\nm._atomic_json=pause\n"
                else:
                    hook += "    return original(*a, **kw)\noriginal=m.os.execvpe\nm.os.execvpe=pause\n"
                code = argv[index]
                call = "_subprocess_bootstrap(sys.argv[1:])"
                assert call in code
                argv[index] = code.replace(call, "\n" + hook + "\n" + call)
            child = popen(argv, *args, **kwargs)
            processes.append(child)
            return child
        return popen(argv, *args, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", paused_popen)
    store = Store.open(tmp_path / "repo", "cancel-test")
    command = [sys.executable, "-c", f"from pathlib import Path; Path({str(executed)!r}).touch()"]
    launcher = SubprocessLauncher(command, handshake_timeout=0.1)
    supervisor = CentralSupervisor(store, launcher=launcher, termination_grace=0.02)
    try:
        run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)
        # A timeout can occur before import or while the child holds its exec
        # fence. In either case stop must forbid this exact run from executing.
        try:
            supervisor.start(run.run_id, active_generation=1)
        except SupervisorError as exc:
            assert "handshake" in str(exc)
        wait_for(entered)
        spec = supervisor._spec(supervisor.get(run.run_id))
        if restart:
            supervisor.close()
            store.close()
            store = Store.open(tmp_path / "repo", "cancel-test")
            launcher = SubprocessLauncher(command, handshake_timeout=0.1)
            supervisor = CentralSupervisor(store, launcher=launcher, termination_grace=0.02)
        stopped = supervisor.stop(run.run_id)
        assert stopped.terminal and stopped.recovery_status == "complete"
        release.touch()
        for process in processes:
            process.wait(timeout=5)
        assert not executed.exists()
        # A new launcher has no in-memory knowledge of the earlier stop.
        with pytest.raises(SupervisorError, match="cancel"):
            SubprocessLauncher(command).launch(spec)
        supervisor.reconcile(active_generation=1)
        assert not executed.exists()
    finally:
        release.touch()
        for process in processes:
            if process.poll() is None:
                process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
        supervisor.close()
        store.close()


def test_failed_stop_remains_durable_and_retries_without_launch(tmp_path):
    class FailingCancel(FakeLauncher):
        fail = True

        def cancel(self, spec, grace_seconds):
            if self.fail:
                raise SupervisorError("injected cancellation failure")
            return self.handle.terminate_tree(grace_seconds)

    store = Store.open(tmp_path / "repo", "cancel-test")
    launcher = FailingCancel()
    supervisor = CentralSupervisor(store, launcher=launcher)
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=30)
    launcher.fail_launches = 1
    with pytest.raises(RuntimeError, match="launch gap"):
        supervisor.start(run.run_id, active_generation=1)
    try:
        with pytest.raises(SupervisorError, match="cancellation failure"):
            supervisor.stop(run.run_id, "user_cancelled")
        pending = supervisor.get(run.run_id)
        assert not pending.terminal and not pending.ready
        assert store.worktree_path_for(run.assignment.worker).exists()
        supervisor.close()
        store.close()
        store = Store.open(tmp_path / "repo", "cancel-test")
        supervisor = CentralSupervisor(store, launcher=launcher)
        launcher.fail = False
        stopped = supervisor.poll(run.run_id, active_generation=1)
        assert stopped.terminal and stopped.terminal_reason == "user_cancelled"
        assert launcher.launch_calls == 1
    finally:
        supervisor.close()
        store.close()
