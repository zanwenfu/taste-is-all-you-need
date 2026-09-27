"""Private coordinator capability for issuing assignment-scoped terminal grants.

This role cannot execute terminal commands. Worker credentials cannot issue
grants. The trusted coordinator binds each request to a prepared assignment;
the broker independently checks its terminal policy and derives the actor ID.
Issuance is idempotent only within this service lifetime. Service death requires
outside-owner cleanup, not a new issuer and speculative worker replay.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import socket
import struct
import time
from dataclasses import dataclass, field

from taste.brains.records import Assignment, _state_id
from taste.brains.terminal_broker import TerminalConflict, TerminalFenced
from taste.brains.terminal_service import (
    HANDSHAKE_SECONDS,
    REQUEST_BYTES,
    TerminalAccessDenied,
    TerminalCredential,
    TerminalUnavailable,
    _decode_message,
    _json,
    _peer_uid,
    _socket_path,
    _uid,
)
from taste.brains.terminal_worker_policy import TerminalWorkerPolicy
from taste.brains.worker_admission import EntrypointInputError
from taste.brains.worker_protocol import assignment_run_id


@dataclass(frozen=True)
class TerminalIssuerCredential:
    socket_path: str
    server_uid: int
    coordinator_uid: int
    input_sha256: str
    policy: TerminalWorkerPolicy
    token: str = field(repr=False)

    def __post_init__(self):
        _socket_path(self.socket_path)
        _uid(self.server_uid)
        _uid(self.coordinator_uid)
        if not isinstance(self.policy, TerminalWorkerPolicy):
            raise ValueError("terminal issuer requires its admitted worker policy")
        for value in (self.input_sha256, self.token):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("terminal issuer requires a goal input digest and private random token")

    def public_scope(self):
        return {"input_sha256": self.input_sha256, "policy": self.policy.to_dict()}

    def to_dict(self):
        """PRIVATE launch material. Never serialize into prompts or reports."""
        return {"schema": "taste.brains/TerminalIssuerCredential/1", "socket_path": self.socket_path,
                "server_uid": self.server_uid, "coordinator_uid": self.coordinator_uid,
                **self.public_scope(), "token": self.token}

    @classmethod
    def from_dict(cls, value):
        try:
            if (not isinstance(value, dict) or set(value) != {"schema", "socket_path", "server_uid",
                    "coordinator_uid", "input_sha256", "policy", "token"}
                    or value["schema"] != "taste.brains/TerminalIssuerCredential/1"):
                raise ValueError
            return cls(value["socket_path"], value["server_uid"], value["coordinator_uid"],
                       value["input_sha256"], TerminalWorkerPolicy.from_dict(value["policy"]), value["token"])
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("invalid private terminal issuer credential") from exc


def _grant(issuer, assignment):
    try:
        return issuer.policy.grant(assignment)
    except EntrypointInputError as exc:
        raise TerminalAccessDenied("terminal assignment differs from the issuer policy") from exc


def issue_for_coordinator(service, message, writer):
    """Service-loop transaction; never waits between admission and registration."""
    issuer = service._issuer
    if (issuer is None or set(message) != {"version", "token", "scope", "operation", "arguments"}
            or type(message["version"]) is not int or message["version"] != 1
            or message["operation"] != "issue" or not isinstance(message["token"], str)
            or re.fullmatch(r"[0-9a-f]{64}", message["token"]) is None
            or not hmac.compare_digest(hashlib.sha256(message["token"].encode()).digest(),
                                       hashlib.sha256(issuer.token.encode()).digest())
            or _peer_uid(writer) != issuer.coordinator_uid
            or _json(message["scope"]) != _json(issuer.public_scope())):
        raise TerminalAccessDenied("terminal issuer access denied")
    if (service._closing or service.broker.phase != "ready"
            or time.time() >= issuer.policy.binding.deadline_unix):
        raise TerminalFenced("terminal issuer is not accepting assignments")
    arguments = message["arguments"]
    if not isinstance(arguments, dict) or set(arguments) != {"assignment", "prepared_state_id"}:
        raise TerminalAccessDenied("invalid terminal issuance arguments")
    assignment = Assignment.from_dict(arguments["assignment"])
    prepared = _state_id(arguments["prepared_state_id"], "prepared_state_id")
    grant = _grant(issuer, assignment)
    run_id = assignment_run_id(assignment)
    prior = service._issued.get(run_id)
    if prior is not None:
        if prior[0] != prepared:
            raise TerminalConflict("terminal assignment already binds a different prepared checkpoint")
        return prior[1]
    credential = TerminalCredential(issuer.socket_path, issuer.server_uid, issuer.coordinator_uid,
                                    grant, secrets.token_hex(32))
    service.authorize(credential)
    service._issued[run_id] = (prepared, credential)
    return credential


class TerminalIssuerClient:
    """Bounded synchronous launch hook, used in the coordinator's driver thread."""

    def __init__(self, credential: TerminalIssuerCredential):
        if not isinstance(credential, TerminalIssuerCredential):
            raise TypeError("a private TerminalIssuerCredential is required")
        self.credential = credential
        self._pid = os.getpid()

    def __call__(self, spec):
        if spec.run_id != assignment_run_id(spec.assignment):
            raise TerminalAccessDenied("terminal launch identity differs from its assignment")
        return self.issue(spec.assignment, spec.prepared_state_id)

    def issue(self, assignment, prepared_state_id):
        issuer = self.credential
        if os.getpid() != self._pid or os.geteuid() != issuer.coordinator_uid:
            raise TerminalAccessDenied("terminal issuer belongs to another process or UID")
        grant = _grant(issuer, assignment)
        _state_id(prepared_state_id, "prepared_state_id")
        allowance = min(HANDSHAKE_SECONDS, issuer.policy.binding.deadline_unix - time.time())
        if allowance <= 0:
            raise TerminalFenced("terminal assignment deadline expired")
        deadline = time.monotonic() + allowance
        wire = _json({"version": 1, "token": issuer.token, "scope": issuer.public_scope(),
                      "operation": "issue", "arguments": {
                          "assignment": assignment.to_dict(), "prepared_state_id": prepared_state_id}})
        if len(wire) > REQUEST_BYTES:
            raise TerminalAccessDenied("terminal assignment exceeds its issuance byte limit")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                def remaining():
                    value = deadline - time.monotonic()
                    if value <= 0:
                        raise TimeoutError
                    connection.settimeout(value)

                def receive(size):
                    result = bytearray()
                    while len(result) < size:
                        remaining()
                        part = connection.recv(size - len(result))
                        if not part:
                            raise TerminalUnavailable("terminal issuer reply was incomplete")
                        result.extend(part)
                    return bytes(result)

                remaining()
                connection.connect(issuer.socket_path)
                peer = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
                if struct.unpack("iII", peer)[1] != issuer.server_uid:
                    raise TerminalAccessDenied("terminal issuer server UID differs")
                remaining()
                connection.sendall(len(wire).to_bytes(4, "big") + wire)
                size = int.from_bytes(receive(4), "big")
                if not 0 < size <= 16384:
                    raise TerminalUnavailable("terminal issuer reply exceeds its byte limit")
                try:
                    response = _decode_message(receive(size))
                except TerminalAccessDenied as exc:
                    raise TerminalUnavailable("invalid terminal issuer reply") from exc
        except (OSError, TimeoutError) as exc:
            raise TerminalUnavailable("terminal issuance ended without a confirmed reply") from exc
        if response == {"version": 1, "status": "denied"}:
            raise TerminalAccessDenied("terminal issuer access denied")
        if response == {"version": 1, "status": "conflict"}:
            raise TerminalConflict("terminal issuance conflicts with its prepared checkpoint")
        if response == {"version": 1, "status": "fenced"}:
            raise TerminalFenced("terminal issuer is fenced")
        if (set(response) != {"version", "status", "scope", "credential"}
                or type(response["version"]) is not int or response["version"] != 1
                or response["status"] != "ok" or _json(response["scope"]) != _json(issuer.public_scope())):
            raise TerminalUnavailable("terminal issuer reply does not bind the admitted scope")
        try:
            credential = TerminalCredential.from_dict(response["credential"])
        except ValueError as exc:
            raise TerminalUnavailable("terminal issuer returned an invalid credential") from exc
        if (credential.socket_path != issuer.socket_path or credential.server_uid != issuer.server_uid
                or credential.worker_uid != issuer.coordinator_uid or credential.grant != grant
                or credential.token == issuer.token):
            raise TerminalUnavailable("terminal issuer returned a different assignment grant")
        return credential
