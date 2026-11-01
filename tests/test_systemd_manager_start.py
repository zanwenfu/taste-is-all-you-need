"""The service manager's launch waits out a busy host's start queue (study S pilot, Nov 1 2026).

With 20 trials starting at once, systemd took ~17 s to start a settlement service. systemd-run waited on it for 15 s,
timed out, and the scope fenced a launch that then ran: two finished runs lost their record. The launch now waits as
long as the host's start queue plausibly takes; the other control commands keep their short bound.
"""

import subprocess
import types

from taste.brains import process_scope
from taste.brains.process_scope import ScopeSpec, SystemdManager


def started(monkeypatch, tmp_path):
    calls = []

    def run(argv, **options):
        calls.append((argv, options))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(process_scope.subprocess, "run", run)
    monkeypatch.setattr(process_scope.os, "geteuid", lambda: 0)
    monkeypatch.setattr(process_scope.os, "getgrouplist", lambda name, gid: [gid])
    monkeypatch.setattr(process_scope.pwd, "getpwuid", lambda uid: types.SimpleNamespace(pw_name="worker", pw_gid=uid))
    is_file = process_scope.Path.is_file
    monkeypatch.setattr(process_scope.Path, "is_file",
                        lambda self: str(self) == "/sys/fs/cgroup/cgroup.controllers" or is_file(self))
    SystemdManager().start("taste-goal-x", "taste-goal:x", ScopeSpec(("/usr/bin/python3", "-V"), str(tmp_path), 1000, 6))
    return calls


def test_a_launch_waits_out_a_busy_start_queue(monkeypatch, tmp_path):
    (argv, options), = started(monkeypatch, tmp_path)
    assert argv[0] == "systemd-run" and "--service-type=exec" in argv
    assert options["timeout"] >= process_scope.START_ACK_SECONDS >= 120


def test_other_control_commands_keep_their_short_bound(monkeypatch):
    calls = []
    monkeypatch.setattr(process_scope.subprocess, "run",
                        lambda argv, **options: calls.append(options)
                        or subprocess.CompletedProcess(argv, 0, "LoadState=not-found\n", ""))
    assert SystemdManager().inspect("taste-goal-x") is None
    assert calls[0]["timeout"] == 15
