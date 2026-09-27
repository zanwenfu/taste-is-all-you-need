"""Public terminal scope in assignments; private credentials outside memory.

The actor identity is derived from the complete immutable assignment, including
its attempt. Credentials are installed for its exact prepared checkpoint before
launch. No socket path, bearer token or credential file path enters the prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from taste.brains.terminal_broker import TerminalBinding
from taste.brains.terminal_service import TerminalClient, TerminalCredential, TerminalGrant
from taste.brains.worker_admission import EntrypointInputError
from taste.brains.worker_protocol import assignment_run_id

TERMINAL_POLICY_KEY = "task_terminal"
_SCHEMA = "taste.brains/TerminalWorkerPolicy/1"
_MAX_CREDENTIAL_BYTES = 8192


@dataclass(frozen=True)
class TerminalWorkerPolicy:
    binding: TerminalBinding
    max_timeout_seconds: float = 300

    def __post_init__(self):
        TerminalGrant(self.binding, "policy_validation", self.max_timeout_seconds)

    def to_dict(self):
        return {"schema": _SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value, Mapping) or set(value) != {"schema", "binding", "max_timeout_seconds"}
                or value["schema"] != _SCHEMA or not isinstance(value["binding"], Mapping)):
            raise ValueError("invalid terminal worker policy")
        try:
            return cls(TerminalBinding(**value["binding"]), value["max_timeout_seconds"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid terminal worker policy limits") from exc

    @classmethod
    def from_assignment(cls, assignment):
        if TERMINAL_POLICY_KEY not in assignment.resources:
            return None
        try:
            policy = cls.from_dict(assignment.resources[TERMINAL_POLICY_KEY])
            azure = assignment.resources.get("azure_openai")
            if not isinstance(azure, Mapping) or policy.binding.deadline_unix != azure.get("deadline_unix"):
                raise ValueError("terminal and Azure assignment deadlines differ")
            return policy
        except ValueError as exc:
            raise EntrypointInputError("assignment terminal policy is invalid") from exc

    def grant(self, assignment):
        if self != self.from_assignment(assignment):
            raise EntrypointInputError("terminal policy differs from the assignment")
        return TerminalGrant(self.binding, "worker_" + assignment_run_id(assignment).split(".")[1],
                             self.max_timeout_seconds)


def credential_path(store, assignment):
    key = hashlib.sha256(assignment_run_id(assignment).encode()).hexdigest()
    return store.backend.common_dir / f"taste-terminal.{store.session}.{key}.json"


def _read_private(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
            raise EntrypointInputError("terminal launch credentials must be private, owned and regular")
        raw = handle.read(_MAX_CREDENTIAL_BYTES + 1)
        if len(raw) > _MAX_CREDENTIAL_BYTES:
            raise EntrypointInputError("terminal launch credentials exceeded their byte limit")
        return raw


def install_terminal_credential(store, assignment, prepared_state_id, credential: TerminalCredential):
    """Trusted coordinator hook; call only after exact assignment preparation."""
    policy = TerminalWorkerPolicy.from_assignment(assignment)
    if (policy is None or not isinstance(credential, TerminalCredential)
            or credential.grant != policy.grant(assignment) or credential.worker_uid != os.geteuid()):
        raise EntrypointInputError("terminal credential differs from the prepared assignment")
    prepared = store.state(prepared_state_id)
    from taste.brains.worker_protocol import ASSIGNMENT_PATH

    if prepared.read(ASSIGNMENT_PATH) != assignment.to_json():
        raise EntrypointInputError("terminal credential requires the exact prepared assignment")
    payload = {"schema": "taste.brains/TerminalWorkerCredential/1",
               "run_id": assignment_run_id(assignment), "prepared_state_id": prepared_state_id,
               "credential": credential.to_dict()}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > _MAX_CREDENTIAL_BYTES:
        raise EntrypointInputError("terminal launch credentials exceeded their byte limit")
    path = credential_path(store, assignment)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        if _read_private(path) != raw:
            raise EntrypointInputError("terminal launch credentials already bind different input") from None
        return path
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return path


def load_terminal_client(store, assignment, prepared_state_id):
    policy = TerminalWorkerPolicy.from_assignment(assignment)
    if policy is None:
        return None
    try:
        value = json.loads(_read_private(credential_path(store, assignment)))
        if (not isinstance(value, dict) or set(value) != {"schema", "run_id", "prepared_state_id", "credential"}
                or value["schema"] != "taste.brains/TerminalWorkerCredential/1"
                or value["run_id"] != assignment_run_id(assignment)
                or value["prepared_state_id"] != prepared_state_id):
            raise ValueError("terminal credential launch binding changed")
        credential = TerminalCredential.from_dict(value["credential"])
        if credential.grant != policy.grant(assignment) or credential.worker_uid != os.geteuid():
            raise ValueError("terminal credential grant changed")
        return TerminalClient(credential)
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
        raise EntrypointInputError("private terminal launch credentials were rejected") from exc
