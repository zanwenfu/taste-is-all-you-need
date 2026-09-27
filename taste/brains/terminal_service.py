"""Authenticated Linux Unix-socket access to an outside-container broker.

One framed request per connection. The peer's UID and a private bearer grant
both bind access; only public scope is ever supplied to model tools. Production
workers should use a different UID from the daemon-owning controller. Give
their group traverse/connect access, never directory write or Docker access.

Client write-EOF means cancellation. The service retains the broker operation
until it settles, then sends an acknowledgement on the still-readable socket.
A killed/disconnected client follows the same path. An outside lifecycle owner
must still stop the environment if this service itself is killed.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import re
import socket
import stat
import struct
import sys
import time
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path

from taste.brains.terminal_broker import (
    TerminalBinding,
    TerminalBroker,
    TerminalConflict,
    TerminalFenced,
    TerminalRequest,
    TerminalResult,
    _identifier,
    _settle,
)

REQUEST_BYTES = 512 * 1024
RESPONSE_BYTES = 3 * 1024 * 1024
HANDSHAKE_SECONDS = 5
SETTLEMENT_SECONDS = 20
WRITE_SECONDS = 3


class TerminalAccessDenied(ValueError):
    """The terminal service refused this request's authority or shape."""


class TerminalUnavailable(RuntimeError):
    """No confirmed terminal outcome was received. Never automatically retry."""


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode()


def _socket_path(value):
    if (not isinstance(value, str) or not value.startswith("/") or "\0" in value
            or len(value.encode()) > 107):
        raise ValueError("terminal service requires an absolute Linux Unix socket path")


def _uid(value):
    if type(value) is not int or not 0 <= value < 2**32 - 1:
        raise ValueError("terminal peer UID must be a nonnegative integer")


def _peer_uid(writer):
    raw = writer.get_extra_info("socket").getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    return struct.unpack("iII", raw)[1]


@dataclass(frozen=True)
class TerminalGrant:
    binding: TerminalBinding
    actor_id: str
    max_timeout_seconds: float = 300

    def __post_init__(self):
        if not isinstance(self.binding, TerminalBinding):
            raise ValueError("terminal grant requires its exact task binding")
        _identifier(self.actor_id)
        if (type(self.max_timeout_seconds) not in (int, float)
                or not math.isfinite(self.max_timeout_seconds)
                or not 0 < self.max_timeout_seconds <= 3600):
            raise ValueError("terminal command allowance must be positive and at most one hour")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TerminalCredential:
    socket_path: str
    server_uid: int
    worker_uid: int
    grant: TerminalGrant
    token: str = field(repr=False)

    def __post_init__(self):
        _socket_path(self.socket_path)
        _uid(self.server_uid)
        _uid(self.worker_uid)
        if not isinstance(self.grant, TerminalGrant):
            raise ValueError("terminal credential requires its admitted grant")
        if not isinstance(self.token, str) or re.fullmatch(r"[0-9a-f]{64}", self.token) is None:
            raise ValueError("terminal bearer token must be 32 random bytes encoded as hex")

    def to_dict(self):
        """PRIVATE launch material. Never place this in Git, prompts or reports."""
        return {"schema": "taste.brains/TerminalCredential/1", **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        try:
            if (not isinstance(value, dict) or set(value) != {
                    "schema", "socket_path", "server_uid", "worker_uid", "grant", "token"}
                    or value["schema"] != "taste.brains/TerminalCredential/1"):
                raise ValueError
            grant = dict(value["grant"])
            grant["binding"] = TerminalBinding(**grant["binding"])
            return cls(value["socket_path"], value["server_uid"], value["worker_uid"],
                       TerminalGrant(**grant), value["token"])
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("invalid private terminal credential") from exc


def _result_payload(result):
    return {"return_code": result.return_code,
            "stdout": base64.b64encode(result.stdout).decode("ascii"),
            "stderr": base64.b64encode(result.stderr).decode("ascii"),
            "stdout_dropped_bytes": result.stdout_dropped_bytes,
            "stderr_dropped_bytes": result.stderr_dropped_bytes}


def _decode_result(value):
    if not isinstance(value, dict) or set(value) != {
        "return_code", "stdout", "stderr", "stdout_dropped_bytes", "stderr_dropped_bytes",
    }:
        raise TerminalUnavailable("invalid terminal receipt fields")
    try:
        return TerminalResult(value["return_code"], base64.b64decode(value["stdout"], validate=True),
                              base64.b64decode(value["stderr"], validate=True),
                              value["stdout_dropped_bytes"], value["stderr_dropped_bytes"])
    except (ValueError, TypeError) as exc:
        raise TerminalUnavailable("invalid terminal receipt data") from exc


async def _read(reader, limit):
    size = int.from_bytes(await reader.readexactly(4), "big")
    if not 0 < size <= limit:
        raise TerminalAccessDenied("terminal message exceeds its byte limit")
    return _decode_message(await reader.readexactly(size))


def _decode_message(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate terminal message field")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, RecursionError) as exc:
        raise TerminalAccessDenied("invalid terminal message") from exc
    if not isinstance(value, dict):
        raise TerminalAccessDenied("terminal message must be an object")
    return value


async def _write(writer, value, limit):
    data = _json(value)
    if len(data) > limit:
        raise TerminalAccessDenied("terminal message exceeds its byte limit")
    writer.write(len(data).to_bytes(4, "big") + data)
    await writer.drain()


class TerminalService:
    """Own every accepted handler and its broker effect through shutdown."""

    def __init__(self, broker: TerminalBroker, credentials=(), *, max_connections=16, issuer=None):
        if not isinstance(broker, TerminalBroker):
            raise TypeError("a TerminalBroker is required")
        credentials = tuple(credentials)
        if not (0 if issuer is not None else 1) <= len(credentials) <= 256 or not all(isinstance(item, TerminalCredential) for item in credentials):
            raise ValueError("terminal service requires 1-256 worker grants")
        if issuer is not None:
            from taste.brains.terminal_issuer import TerminalIssuerCredential

            if (not isinstance(issuer, TerminalIssuerCredential) or issuer.policy.binding != broker.binding
                    or issuer.server_uid != os.geteuid()):
                raise ValueError("terminal issuer must bind this exact service")
        if type(max_connections) is not int or not 1 <= max_connections <= 256:
            raise ValueError("terminal service connection limit must be 1-256")
        self.broker, self.max_connections = broker, max_connections
        self._credentials = {}
        self._issuer, self._issued = issuer, {}
        self.path = Path(issuer.socket_path if issuer is not None else credentials[0].socket_path)
        actors = set()
        for credential in credentials:
            if (not isinstance(credential, TerminalCredential)
                    or credential.grant.binding != broker.binding
                    or credential.server_uid != os.geteuid()
                    or credential.socket_path != str(self.path)
                    or credential.grant.actor_id in actors):
                raise ValueError("terminal grants must name unique actors in this exact service")
            key = hashlib.sha256(credential.token.encode()).digest()
            if issuer is not None and credential.token == issuer.token:
                raise ValueError("terminal worker and issuer cannot share a bearer token")
            if key in self._credentials:
                raise ValueError("terminal grants cannot share bearer tokens")
            actors.add(credential.grant.actor_id)
            self._credentials[key] = credential
        self._pid, self._loop = os.getpid(), None
        self._server = self._shutdown_task = None
        self._handlers = set()
        self._writers = set()
        self._closing = False
        self._identity = None

    def authorize(self, credential: TerminalCredential):
        """Trusted lifecycle owner only, on the service loop; not an RPC method.

        Register an exact prepared assignment before launching its worker. A
        repeated identical registration is harmless; token rotation for an
        already registered actor is refused. No model can mint actor grants.
        """
        self._on_loop()
        if self._closing or self._server is None or self.broker.phase != "ready":
            raise TerminalFenced("terminal service is not accepting actor grants")
        if (not isinstance(credential, TerminalCredential) or credential.grant.binding != self.broker.binding
                or credential.server_uid != os.geteuid() or credential.socket_path != str(self.path)):
            raise TerminalAccessDenied("terminal credential does not identify this service")
        key = hashlib.sha256(credential.token.encode()).digest()
        if self._issuer is not None and credential.token == self._issuer.token:
            raise TerminalAccessDenied("terminal worker cannot use the issuer token")
        if self._credentials.get(key) == credential:
            return
        if key in self._credentials or any(
                existing.grant.actor_id == credential.grant.actor_id for existing in self._credentials.values()):
            raise TerminalConflict("terminal actor or token was already registered")
        if len(self._credentials) >= 256:
            raise TerminalFenced("terminal actor grant limit reached")
        self._credentials[key] = credential

    def seal_for_grading(self) -> TerminalBinding:
        """Trusted lifecycle handoff, never a worker RPC; keep receipts readable.

        Call only after draining agent processes. The broker refuses to seal
        an active command. Successful sealing survives broker restart and
        prevents new effects while the outside verifier uses the live container.
        close() still stops the environment after grading or on any failure.
        """
        self._on_loop()
        if self._closing or self._server is None:
            raise TerminalFenced("terminal service is not available for grading handoff")
        return self.broker.seal_for_grading()

    def _on_loop(self):
        if os.getpid() != self._pid:
            raise TerminalFenced("terminal service belongs to another process")
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise TerminalFenced("terminal service belongs to another event loop")
        self._loop = loop

    async def start(self, *, worker_gid=None):
        self._on_loop()
        if self._server is not None or self._closing:
            raise TerminalFenced("terminal service cannot be started again")
        if not hasattr(socket, "SO_PEERCRED"):
            raise TerminalFenced("terminal service requires Linux peer credentials")
        if worker_gid is not None:
            _uid(worker_gid)
        # A fresh private directory prevents socket replacement by workers.
        self.path.parent.mkdir(mode=0o700)
        if worker_gid is not None:
            os.chown(self.path.parent, -1, worker_gid)
            self.path.parent.chmod(0o710)
        # Bind synchronously under the private parent, set permissions, and
        # only then let the event loop accept connections.
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.path))
            self.path.chmod(0o600 if worker_gid is None else 0o660)
            if worker_gid is not None:
                os.chown(self.path, -1, worker_gid)
            info = self.path.lstat()
            self._identity = (info.st_dev, info.st_ino)
            listener.listen(self.max_connections)
            listener.setblocking(False)
            # Python 3.13+ otherwise unlinks the socket at server.close(),
            # before we have proved broker drainage or checked its identity.
            cleanup = {"cleanup_socket": False} if sys.version_info >= (3, 13) else {}
            self._server = await asyncio.start_unix_server(
                self._accept, sock=listener, limit=REQUEST_BYTES + 4, **cleanup)
        except BaseException:
            listener.close()
            self._closing = True
            raise
        return self

    def _accept(self, reader, writer):
        if self._closing or len(self._handlers) >= self.max_connections:
            writer.transport.abort()
            return
        task = asyncio.create_task(self._handle(reader, writer))
        self._handlers.add(task)
        self._writers.add(writer)
        task.add_done_callback(self._handlers.discard)

    def _authorize(self, message, writer):
        if set(message) != {"version", "token", "grant", "operation", "arguments"} or type(message["version"]) is not int or message["version"] != 1:
            raise TerminalAccessDenied("invalid terminal request fields")
        token = message["token"]
        if not isinstance(token, str) or re.fullmatch(r"[0-9a-f]{64}", token) is None:
            raise TerminalAccessDenied("terminal access denied")
        credential = self._credentials.get(hashlib.sha256(token.encode()).digest())
        if (credential is None or _json(credential.grant.to_dict()) != _json(message["grant"])
                or _peer_uid(writer) != credential.worker_uid):
            raise TerminalAccessDenied("terminal access denied")
        return credential.grant

    async def _execute(self, request, reader):
        operation = asyncio.create_task(self.broker.execute(request))
        disconnected = asyncio.create_task(reader.read(1))
        try:
            done, _ = await asyncio.wait((operation, disconnected), return_when=asyncio.FIRST_COMPLETED)
            if disconnected in done:
                # EOF or trailing request bytes both revoke this connection.
                operation.cancel()
                await _settle(operation)
                raise TerminalUnavailable("terminal caller disconnected")
            return operation.result()
        finally:
            disconnected.cancel()
            with suppress(asyncio.CancelledError):
                await _settle(disconnected)
            if not operation.done():
                operation.cancel()
                await _settle(operation)

    async def _handle(self, reader, writer):
        reply = {"version": 1, "status": "denied"}
        dispatched = False
        try:
            async with asyncio.timeout(HANDSHAKE_SECONDS):
                message = await _read(reader, REQUEST_BYTES)
                if message.get("operation") in ("issue", "issuer_ping"):
                    from taste.brains.terminal_issuer import handle_coordinator_request

                    result = handle_coordinator_request(self, message, writer)
                    reply.update(status="ok", scope=self._issuer.public_scope(), **result)
                    return
                grant = self._authorize(message, writer)
            reply["grant"] = grant.to_dict()
            operation, arguments = message["operation"], message["arguments"]
            if not isinstance(arguments, dict):
                raise TerminalAccessDenied("invalid terminal arguments")
            if operation == "ping" and not arguments:
                reply.update(status="ok", phase=self.broker.phase)
            elif operation == "lookup" and set(arguments) == {"request_id"}:
                reply["request_id"] = arguments["request_id"]
                result = self.broker.lookup_actor(arguments["request_id"], grant.actor_id)
                reply.update(status="missing" if result is None else "ok")
                if result is not None:
                    reply["result"] = _result_payload(result)
            elif operation == "execute":
                request = TerminalRequest(**arguments)
                if request.actor_id != grant.actor_id or request.timeout_seconds > grant.max_timeout_seconds:
                    raise TerminalAccessDenied("terminal request exceeds its actor grant")
                reply["request_id"] = request.request_id
                if self._closing:
                    raise TerminalFenced("terminal service is closing")
                dispatched = True
                result = await self._execute(request, reader)
                reply.update(status="ok", result=_result_payload(result))
            else:
                raise TerminalAccessDenied("unknown terminal operation")
        except TerminalConflict:
            reply["status"] = "conflict"
        except TerminalFenced:
            reply["status"] = "fenced"
        except (TerminalAccessDenied, ValueError, TypeError):
            reply["status"] = "uncertain" if dispatched else "denied"
        except (Exception, asyncio.CancelledError, BaseExceptionGroup):
            # Never serialize transport exception text or private credentials.
            reply["status"] = "uncertain"
        finally:
            try:
                async with asyncio.timeout(WRITE_SECONDS):
                    await _write(writer, reply, RESPONSE_BYTES)
            except (Exception, asyncio.CancelledError):
                pass
            writer.close()
            try:
                async with asyncio.timeout(WRITE_SECONDS):
                    await writer.wait_closed()
            except (Exception, asyncio.CancelledError):
                writer.transport.abort()
            self._writers.discard(writer)

    async def close(self):
        self._on_loop()
        self._closing = True
        if self._shutdown_task is None or self._shutdown_task.done():
            self._shutdown_task = asyncio.create_task(self._shutdown())
        try:
            await asyncio.wait((self._shutdown_task,))
            return self._shutdown_task.result()
        except asyncio.CancelledError as original:
            try:
                while True:
                    try:
                        await _settle(self._shutdown_task)
                        break
                    except asyncio.CancelledError:
                        # Whole-loop cancellation can reach the owner before
                        # its coroutine enters. Keep this caller responsible.
                        self._shutdown_task = asyncio.create_task(self._shutdown())
            except BaseException as failure:
                raise BaseExceptionGroup("terminal service cancellation and shutdown", [original, failure]) from None
            raise

    async def _shutdown(self):
        if self._server is not None:
            self._server.close()
        for writer in tuple(self._writers):
            # Wake incomplete handshake/idle connections too. Effect ownership
            # remains with their handlers and the broker below.
            writer.transport.abort()
        failure = None
        try:
            await self.broker.abort()
        except BaseException as exc:
            failure = exc
        for task in tuple(self._handlers):
            try:
                await _settle(task)
            except BaseException as exc:
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure
        if self._server is not None:
            await self._server.wait_closed()
        if self._identity is not None:
            try:
                info = self.path.lstat()
            except FileNotFoundError:
                return
            if not stat.S_ISSOCK(info.st_mode) or (info.st_dev, info.st_ino) != self._identity:
                raise TerminalConflict("terminal service socket was replaced")
            self.path.unlink()


class TerminalClient:
    def __init__(self, credential: TerminalCredential):
        if not isinstance(credential, TerminalCredential):
            raise TypeError("a private TerminalCredential is required")
        self.credential = credential
        self._pid = os.getpid()

    async def _exchange(self, operation, arguments, state):
        credential = self.credential
        writer = None
        # A request can queue behind other actors until the fixed trial end.
        # The post-deadline allowance is for receiving stop/uncertainty proof,
        # never for extending command admission or retrying a request.
        allowance = (HANDSHAKE_SECONDS if operation != "execute" else
                     max(0, credential.grant.binding.deadline_unix - time.time()) + SETTLEMENT_SECONDS)
        try:
            async with asyncio.timeout(allowance):
                if state["cancelled"]:
                    raise asyncio.CancelledError
                reader, writer = await asyncio.open_unix_connection(credential.socket_path, limit=RESPONSE_BYTES + 4)
                state["writer"] = writer
                if _peer_uid(writer) != credential.server_uid:
                    raise TerminalAccessDenied("terminal server UID differs from the admitted controller")
                if state["cancelled"]:
                    raise asyncio.CancelledError
                await _write(writer, {"version": 1, "token": credential.token,
                    "grant": credential.grant.to_dict(), "operation": operation, "arguments": arguments}, REQUEST_BYTES)
                try:
                    response = await _read(reader, RESPONSE_BYTES)
                except TerminalAccessDenied as exc:
                    raise TerminalUnavailable("invalid terminal reply framing") from exc
                if response == {"version": 1, "status": "denied"}:
                    raise TerminalAccessDenied("terminal access denied")
                if (type(response.get("version")) is not int or response["version"] != 1
                        or _json(response.get("grant")) != _json(credential.grant.to_dict())):
                    raise TerminalUnavailable("terminal reply does not identify the admitted grant")
                status = response.get("status")
                if status == "denied":
                    raise TerminalAccessDenied("terminal request denied")
                if status == "conflict":
                    raise TerminalConflict("terminal request conflicts with the durable effect")
                if status == "fenced":
                    raise TerminalFenced("terminal environment is fenced")
                if status not in {"ok", "missing"}:
                    raise TerminalUnavailable("terminal operation did not complete with a receipt")
                if operation == "ping":
                    if set(response) != {"version", "status", "grant", "phase"} or response["phase"] not in {"ready", "sealed", "fenced", "stopped"}:
                        raise TerminalUnavailable("invalid terminal readiness reply")
                    return response["phase"]
                if response.get("request_id") != arguments["request_id"]:
                    raise TerminalUnavailable("terminal reply identifies another effect")
                if status == "missing" and operation == "lookup" and set(response) == {"version", "status", "grant", "request_id"}:
                    return None
                if status != "ok" or set(response) != {"version", "status", "grant", "request_id", "result"}:
                    raise TerminalUnavailable("invalid terminal receipt reply")
                return _decode_result(response["result"])
        except (OSError, TimeoutError, asyncio.IncompleteReadError) as exc:
            raise TerminalUnavailable("terminal connection ended without a confirmed reply") from exc
        finally:
            if writer is not None:
                writer.close()
                try:
                    async with asyncio.timeout(WRITE_SECONDS):
                        await writer.wait_closed()
                except (Exception, asyncio.CancelledError):
                    writer.transport.abort()

    async def _call(self, operation, arguments):
        if os.getpid() != self._pid or os.geteuid() != self.credential.worker_uid:
            raise TerminalAccessDenied("terminal client belongs to another process or UID")
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        state = {"writer": None, "cancelled": False}
        task = asyncio.create_task(self._exchange(operation, arguments, state))
        try:
            await asyncio.wait((task,))
            return task.result()
        except asyncio.CancelledError as original:
            state["cancelled"] = True
            writer = state["writer"]
            if writer is not None:
                with suppress(OSError, RuntimeError):
                    writer.write_eof()
            try:
                await _settle(task)
            except asyncio.CancelledError:
                pass
            except BaseException as failure:
                raise BaseExceptionGroup("terminal cancellation and remote settlement", [original, failure]) from None
            raise

    async def ping(self):
        return await self._call("ping", {})

    async def execute(self, request: TerminalRequest):
        if (not isinstance(request, TerminalRequest) or request.actor_id != self.credential.grant.actor_id
                or request.timeout_seconds > self.credential.grant.max_timeout_seconds):
            raise TerminalAccessDenied("terminal request differs from its actor grant")
        return await self._call("execute", asdict(request))

    async def lookup(self, request_id):
        _identifier(request_id)
        return await self._call("lookup", {"request_id": request_id})
