"""Real Unix HTTP transport, fake daemon; no Docker privileges or model calls."""

from __future__ import annotations

import asyncio
import copy
import json
import socketserver
import tempfile
import threading
import time
import tracemalloc
from dataclasses import replace
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from taste.brains import docker_terminal as transport
from taste.brains.docker_terminal import (
    DockerTerminalBackend,
    DockerTransportError,
)
from taste.brains.terminal_broker import (
    TerminalBinding,
    TerminalBroker,
    TerminalConflict,
    TerminalFenced,
    TerminalRequest,
    TerminalResult,
)
from tests.test_goal_cancellation import _leaves, wait_event

CONTAINER = "a" * 64
EXEC = "b" * 64
TOKEN = "c" * 32
STARTED = "2026-09-26T12:00:00.000000000Z"


def frame(stream, payload):
    return bytes([stream, 0, 0, 0]) + len(payload).to_bytes(4, "big") + payload


def request(**changes):
    return replace(TerminalRequest("effect_1", "worker_A", "echo task", "/tmp", 3), **changes)


class Daemon:
    def __init__(self, directory):
        self.info = {
            "Id": CONTAINER, "Config": {"Labels": {transport.OWNER_LABEL: TOKEN}},
            "State": {"Running": True, "Paused": False, "Restarting": False, "Dead": False,
                      "Pid": 42, "StartedAt": STARTED},
            "HostConfig": {"AutoRemove": False, "RestartPolicy": {"Name": "no"}},
        }
        self.exec_info = {"ID": EXEC, "ContainerID": CONTAINER, "Running": False, "ExitCode": 7}
        self.calls, self.errors = [], []
        self.started, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.parts = lambda: iter([frame(1, b"out\0\xff"), frame(2, b"error\x80")])
        self.upgrade, self.chunked = True, False
        self.stop_failure, self.live_cgroup = False, False
        self.override = None
        self.cgroups = directory / "cgroup"
        self.cgroups.mkdir()
        (self.cgroups / "cgroup.controllers").write_text("cpu memory pids")
        self.group = self.cgroups / "docker" / CONTAINER
        self.group.mkdir(parents=True)
        (self.group / "cgroup.events").write_text("populated 1\nfrozen 0\n")
        self.proc = directory / "proc"
        (self.proc / "42").mkdir(parents=True)
        (self.proc / "42" / "cgroup").write_text(f"0::/docker/{CONTAINER}\n")

    def handle(self, handler):
        length = int(handler.headers.get("Content-Length", 0))
        body = json.loads(handler.rfile.read(length)) if length else None
        path = handler.path.removeprefix("/v1.51")
        assert handler.path.startswith("/v1.51/")
        self.calls.append((handler.command, path, body))
        if self.override and self.override(handler, path):
            return
        if path == f"/containers/{CONTAINER}/json":
            self.reply(handler, self.info)
        elif path == f"/containers/{CONTAINER}/exec":
            self.reply(handler, {"Id": EXEC}, status=201)
        elif path == f"/exec/{EXEC}/start":
            assert body == {"Detach": False, "Tty": False}
            self.started.set()
            assert self.release.wait(3), "fake daemon stream was never released"
            status = "101 Switching Protocols" if self.upgrade else "200 OK"
            headers = "Connection: Upgrade\r\nUpgrade: tcp\r\n" if self.upgrade else "Connection: close\r\n"
            if self.chunked:
                headers += "Transfer-Encoding: chunked\r\n"
            handler.connection.sendall((f"HTTP/1.1 {status}\r\n{headers}"
                "Content-Type: application/vnd.docker.raw-stream\r\n\r\n").encode())
            for part in self.parts():
                if self.chunked:
                    part = f"{len(part):x}\r\n".encode() + part + b"\r\n"
                handler.connection.sendall(part)
            if self.chunked:
                handler.connection.sendall(b"0\r\n\r\n")
        elif path == f"/exec/{EXEC}/json":
            self.reply(handler, self.exec_info)
        elif path == f"/containers/{CONTAINER}/stop?t=1":
            self.info["State"]["Running"] = False
            if not self.live_cgroup:
                (self.group / "cgroup.events").write_text("populated 0\nfrozen 0\n")
            self.release.set()
            self.reply(handler, None, status=500 if self.stop_failure else 204)
        else:
            raise AssertionError(f"unexpected daemon operation: {path}")

    @staticmethod
    def reply(handler, value, *, status=200, extra_length=0):
        data = b"" if value is None else json.dumps(value).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data) + extra_length))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(data)


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    daemon = Daemon(tmp_path)
    monkeypatch.setattr(transport, "CGROUP_ROOT", daemon.cgroups)
    monkeypatch.setattr(transport, "PROC_ROOT", daemon.proc)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            try:
                daemon.handle(self)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected when cancellation/deadlines close the peer.
            except BaseException as exc:
                daemon.errors.append(exc)
            finally:
                self.close_connection = True

        do_POST = do_GET

        def log_message(self, *_args):
            pass

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        pass

    # Keep the socket below Linux's sockaddr_un path limit even with long test names.
    with tempfile.TemporaryDirectory(prefix="taste-wire-", dir="/tmp") as socket_dir:
        daemon.socket = str(Path(socket_dir) / "docker.sock")
        with Server(daemon.socket, Handler) as server:
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
            thread.start()
            try:
                yield daemon
            finally:
                daemon.release.set()
                server.shutdown()
                thread.join(3)
                assert not thread.is_alive()
        assert not daemon.errors


def backend(daemon, **kwargs):
    return DockerTerminalBackend.admit(daemon.socket, CONTAINER, TOKEN, time.time() + 20, **kwargs)


@pytest.mark.parametrize("upgrade,chunked", [(True, False), (False, False), (False, True)])
def test_real_http_binary_stream_and_exit_receipt_reopen(daemon, tmp_path, upgrade, chunked):
    daemon.upgrade, daemon.chunked = upgrade, chunked
    # Deliberately split frame headers and multi-byte output across HTTP writes.
    raw = frame(1, b"\xff\0abc") + frame(2, b"\x80error")
    daemon.parts = lambda: (bytes([b]) for b in raw)
    executor = backend(daemon, output_limit=3)
    owner = TerminalBroker.create(tmp_path / "ledger",
        TerminalBinding("trial", CONTAINER, executor.binding.deadline_unix, 3), executor)
    expected = TerminalResult(7, b"\xff\0a", b"\x80er", 2, 3)
    assert asyncio.run(owner.execute(request())) == expected
    owner.close()
    recovered = TerminalBroker.open(tmp_path / "ledger", owner.binding,
                                    DockerTerminalBackend(executor.binding))

    async def replay():
        assert await recovered.execute(request()) == expected
        await recovered.abort()

    try:
        asyncio.run(replay())
        creates = [body for _, path, body in daemon.calls if path.endswith("/exec")]
        assert creates == [{"AttachStdout": True, "AttachStderr": True, "AttachStdin": False,
                            "Tty": False, "Cmd": ["/bin/sh", "-c", "echo task"], "WorkingDir": "/tmp"}]
        assert recovered.phase == "stopped"
    finally:
        recovered.close()


def test_large_unterminated_output_is_drained_with_bounded_memory(daemon):
    size = 32 * 1024 * 1024

    def parts():
        yield bytes([1, 0, 0, 0]) + size.to_bytes(4, "big")
        for _ in range(size // 65536):
            yield b"x" * 65536
        yield frame(2, b"tail")

    daemon.parts = parts
    executor = backend(daemon, output_limit=1024)
    tracemalloc.start()
    try:
        result = executor.execute(request())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result == TerminalResult(7, b"x" * 1024, b"tail", size - 1024, 0)
    assert peak < 4 * 1024 * 1024
    assert executor.stop_and_confirm() == CONTAINER


@pytest.mark.parametrize("raw", [b"\1\0", frame(3, b"daemon failure"),
    b"\1\1\0\0\0\0\0\0", b"\1\0\0\0\xff\xff\xff\xffx"])
def test_malformed_or_huge_incomplete_frame_fences_and_stops(daemon, tmp_path, raw):
    daemon.parts = lambda: iter([raw])
    executor = backend(daemon)
    owner = TerminalBroker.create(tmp_path / "ledger",
        TerminalBinding("trial", CONTAINER, executor.binding.deadline_unix, 3), executor)
    try:
        with pytest.raises(DockerTransportError):
            asyncio.run(owner.execute(request()))
        assert owner.phase == "stopped" and not daemon.info["State"]["Running"]
        with pytest.raises(TerminalFenced):
            owner.lookup(request())
    finally:
        owner.close()


@pytest.mark.parametrize("change", [{"Running": True}, {"ExitCode": None}, {"ExitCode": True},
                                   {"ContainerID": "d" * 64}, {"ID": "d" * 64}])
def test_eof_is_not_a_success_receipt_without_matching_exec_exit(daemon, change):
    executor = backend(daemon)
    daemon.exec_info.update(change)
    with pytest.raises(DockerTransportError, match="confirmed exit"):
        executor.execute(request())
    with pytest.raises(TerminalFenced):
        executor.execute(request(request_id="retry"))
    assert executor.stop_and_confirm() == CONTAINER


@pytest.mark.parametrize("fault", ["owner", "restart", "policy", "paused", "stopped"])
def test_container_changes_are_refused_before_an_exec_is_created(daemon, fault):
    executor = backend(daemon)
    if fault == "owner":
        daemon.info["Config"]["Labels"][transport.OWNER_LABEL] = "d" * 32
    elif fault == "restart":
        daemon.info["State"]["StartedAt"] = "different_start"
    elif fault == "policy":
        daemon.info["HostConfig"]["RestartPolicy"]["Name"] = "always"
    elif fault == "paused":
        daemon.info["State"]["Paused"] = True
    else:
        daemon.info["State"]["Running"] = False
    with pytest.raises((TerminalConflict, TerminalFenced)):
        executor.execute(request())
    assert not any(path.endswith("/exec") for _, path, _ in daemon.calls)


@pytest.mark.parametrize("ending", ["cancel", "timeout", "abort"])
def test_active_command_settles_across_http_broker_and_cgroup_boundary(daemon, tmp_path, ending):
    executor = backend(daemon)
    daemon.release.clear()
    owner = TerminalBroker.create(tmp_path / "ledger",
        TerminalBinding("trial", CONTAINER, executor.binding.deadline_unix, 3), executor)

    async def scenario():
        active = asyncio.create_task(owner.execute(request(timeout_seconds=0.15 if ending == "timeout" else 3)))
        await wait_event(daemon.started)
        started = time.monotonic()
        if ending == "cancel":
            active.cancel()
            active.cancel()
        elif ending == "abort":
            with pytest.raises(BaseExceptionGroup):
                await owner.abort()
        with pytest.raises(BaseException) as caught:
            await active
        leaves = _leaves(caught.value)
        assert any(isinstance(e, TimeoutError if ending == "timeout" else asyncio.CancelledError) for e in leaves)
        assert owner.phase == "stopped"
        assert time.monotonic() - started < 1.5, "shutdown waited for the command timeout"
        assert not daemon.info["State"]["Running"]
        with pytest.raises(TerminalFenced):
            await owner.execute(request(request_id="late"))

    try:
        asyncio.run(scenario())
    finally:
        owner.close()


@pytest.mark.parametrize("stage", ["headers", "stream"])
def test_absolute_deadline_bounds_slow_trickle_not_just_idle_time(daemon, stage):
    executor = backend(daemon)

    def override(handler, path):
        if path != f"/exec/{EXEC}/start":
            return False
        if stage == "headers":
            parts = [b"HTTP/1.1 101 Switching Protocols\r\nX-Slow: ", *([b"a"] * 30)]
        else:
            parts = [b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: tcp\r\n"
                     b"Content-Type: application/vnd.docker.raw-stream\r\n\r\n",
                     *([frame(1, b"a")] * 30)]
        for part in parts:
            handler.connection.sendall(part)
            time.sleep(0.03)
        return True

    daemon.override = override
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        executor.execute(request(timeout_seconds=0.16))
    assert time.monotonic() - start < 0.7
    assert executor.stop_and_confirm() == CONTAINER


@pytest.mark.parametrize("fault", ["truncated", "large", "wrong_exec_id", "error_status"])
def test_control_reply_faults_never_start_a_command(daemon, fault):
    executor = backend(daemon)

    def override(handler, path):
        if path != f"/containers/{CONTAINER}/exec":
            return False
        if fault == "truncated":
            daemon.reply(handler, {"Id": EXEC}, status=201, extra_length=1)
        elif fault == "large":
            daemon.reply(handler, {"Id": EXEC, "padding": "x" * transport.CONTROL_BYTES}, status=201)
        elif fault == "wrong_exec_id":
            daemon.reply(handler, {"Id": "../../host"}, status=201)
        else:
            daemon.reply(handler, {"message": "daemon error"}, status=500)
        return True

    daemon.override = override
    with pytest.raises(DockerTransportError):
        executor.execute(request())
    assert not daemon.started.is_set()
    assert executor.stop_and_confirm() == CONTAINER


def test_lost_stop_reply_can_be_reconciled_without_readmission(daemon):
    executor = backend(daemon)
    daemon.stop_failure = True
    with pytest.raises(DockerTransportError):
        executor.stop_and_confirm()
    assert not daemon.info["State"]["Running"]
    with pytest.raises(TerminalFenced):
        executor.execute(request())
    recovered = DockerTerminalBackend(executor.binding)
    assert recovered.stop_and_confirm() == CONTAINER
    assert sum(path.endswith("/stop?t=1") for _, path, _ in daemon.calls) == 1


@pytest.mark.parametrize("fault", ["populated", "missing_mount", "changed_identity"])
def test_stopped_daemon_report_alone_does_not_prove_drainage(daemon, monkeypatch, fault):
    executor = backend(daemon)
    monkeypatch.setattr(transport, "CONTROL_SECONDS", 0.12)
    if fault == "populated":
        daemon.live_cgroup = True
    elif fault == "missing_mount":
        (daemon.cgroups / "cgroup.controllers").unlink()
    else:
        daemon.info["Id"] = "d" * 64
    with pytest.raises((TimeoutError, TerminalFenced, TerminalConflict)):
        executor.stop_and_confirm()


@pytest.mark.parametrize("changes", [
    {"container_id": "short"}, {"socket_path": "tcp://remote"}, {"owner_token": "wrong"},
    {"output_limit": True}, {"output_limit": 0}, {"output_limit": 1024 * 1024 + 1},
    {"deadline_unix": float("nan")}, {"cgroup_path": f"/../../{CONTAINER}"},
])
def test_invalid_binding_is_rejected_before_daemon_contact(daemon, changes):
    original = backend(daemon).binding
    before = copy.deepcopy(daemon.calls)
    with pytest.raises(ValueError):
        replace(original, **changes)
    assert daemon.calls == before


def test_original_expired_deadline_cannot_be_refreshed_on_recovery(daemon):
    original = backend(daemon).binding
    recovered = DockerTerminalBackend(replace(original, deadline_unix=time.time() - 1))
    before = copy.deepcopy(daemon.calls)
    with pytest.raises(TimeoutError):
        recovered.execute(request())
    assert daemon.calls == before
    assert recovered.stop_and_confirm() == CONTAINER


def test_stop_racing_exec_creation_prevents_delayed_start(daemon):
    executor = backend(daemon)
    created, release = threading.Event(), threading.Event()

    def override(handler, path):
        if not path.endswith("/exec"):
            return False
        created.set()
        assert release.wait(3)
        daemon.reply(handler, {"Id": EXEC}, status=201)
        return True

    daemon.override = override
    outcomes = []

    def execute():
        try:
            outcomes.append(executor.execute(request()))
        except Exception as exc:
            outcomes.append(exc)

    thread = threading.Thread(target=execute)
    thread.start()
    try:
        assert created.wait(2)
        with pytest.raises(TerminalFenced):
            executor.execute(request(request_id="concurrent"))
        assert executor.stop_and_confirm() == CONTAINER
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive() and len(outcomes) == 1
    assert isinstance(outcomes[0], Exception)
    assert not daemon.started.is_set()
    assert sum(path.endswith("/exec") for _, path, _ in daemon.calls) == 1
