"""Azure goal secrets delivered separately from hash-pinned public admission."""

from __future__ import annotations

import hashlib
import os
import stat
import struct
from pathlib import Path

from taste.brains.azure_execution_policy import POLICY_KEY, AzureExecutionPolicy
from taste.brains.goal_entrypoint import GoalInputError, _canonical, _decode
from taste.brains.process_credentials import MAX_CREDENTIAL_BYTES
from taste.brains.terminal_issuer import TerminalIssuerClient, TerminalIssuerCredential
from taste.providers.base import ProtocolFailure

GOAL_CREDENTIAL_NAME = "taste-azure-goal.json"
_SCHEMA = "taste.brains/AzureGoalCredentials/1"


def _private_service_copy(fd, info):
    if info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o400:
        return True
    if info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o440:
        return False
    # Newer systemd keeps ownership with root and adds exactly one POSIX ACL
    # reader. The group mode bits represent the ACL mask, not group access.
    # Require root read, this UID read, and no group/other/extra named access.
    undefined = 2**32 - 1
    entries = ((1, 4, undefined), (2, 4, os.geteuid()), (4, 0, undefined),
               (16, 4, undefined), (32, 0, undefined))
    expected = struct.pack("<I", 2) + b"".join(struct.pack("<HHI", *item) for item in entries)
    return os.getxattr(fd, "system.posix_acl_access") == expected


def _validate(config, value):
    if (not isinstance(value, dict) or set(value) != {"schema", "input_sha256", "api_key", "terminal_issuer"}
            or value["schema"] != _SCHEMA
            or value["input_sha256"] != hashlib.sha256(config.to_bytes()).hexdigest()):
        raise GoalInputError("private credentials do not bind this exact prepared goal input")
    key = value["api_key"]
    if (not isinstance(key, str) or not 1 <= len(key) <= 4096
            or any(not 33 <= ord(char) <= 126 for char in key)):
        raise GoalInputError("Azure goal credential must be a bounded nonempty ASCII API key")
    policy = AzureExecutionPolicy.from_dict(config.goal.metadata.get(POLICY_KEY))
    environment = {"AZURE_OPENAI_BASE_URL": policy.endpoint, "AZURE_OPENAI_API_KEY": key}
    policy.azure_config(environment)
    issuer = (None if value["terminal_issuer"] is None
              else TerminalIssuerCredential.from_dict(value["terminal_issuer"]))
    if ((policy.terminal is None) != (issuer is None)
            or (issuer is not None and (issuer.policy != policy.terminal
                                       or issuer.input_sha256 != value["input_sha256"]))):
        raise GoalInputError("private terminal issuer differs from this prepared goal policy")
    return environment, issuer


def encode_azure_goal_credentials(config, api_key: str, *, terminal_issuer=None) -> bytes:
    """Outside controller only: return bytes for a private ScopeCredential.

    The returned secret must never enter Git, argv, a public manifest or a model
    prompt. The immutable goal hash prevents credentials being moved to another
    goal, policy, source revision or deadline. No environment fallback is used.
    """
    if terminal_issuer is not None and not isinstance(terminal_issuer, TerminalIssuerCredential):
        raise GoalInputError("private goal requires a TerminalIssuerCredential")
    value = {"schema": _SCHEMA, "input_sha256": hashlib.sha256(config.to_bytes()).hexdigest(),
             "api_key": api_key, "terminal_issuer": terminal_issuer.to_dict() if terminal_issuer else None}
    _validate(config, value)
    raw = _canonical(value).encode()
    if len(raw) > MAX_CREDENTIAL_BYTES:
        raise GoalInputError("private goal credential exceeded its byte limit")
    return raw


def load_azure_goal_credentials(config, directory):
    """Read the systemd-provided credential copy before any host or paid call."""
    try:
        path = Path(directory)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("invalid credential directory")
        parent = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(parent)
            if info.st_uid not in {0, os.geteuid()} or info.st_mode & 0o022:
                raise ValueError("credential directory is not trusted")
            fd = os.open(GOAL_CREDENTIAL_NAME, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
            with os.fdopen(fd, "rb") as handle:
                info = os.fstat(handle.fileno())
                if (not stat.S_ISREG(info.st_mode) or not _private_service_copy(handle.fileno(), info)
                        or info.st_nlink != 1
                        or not 0 < info.st_size <= MAX_CREDENTIAL_BYTES):
                    raise ValueError("credential file must be owned, private and read-only")
                raw = handle.read(MAX_CREDENTIAL_BYTES + 1)
                if len(raw) > MAX_CREDENTIAL_BYTES:
                    raise ValueError("credential file too large")
        finally:
            os.close(parent)
        environment, issuer = _validate(config, _decode(raw))
        if issuer is not None and issuer.coordinator_uid != os.geteuid():
            raise ValueError("issuer belongs to another coordinator UID")
        return environment, TerminalIssuerClient(issuer) if issuer is not None else None
    except (ValueError, TypeError, KeyError, OSError, RecursionError, ProtocolFailure):
        # Do not include JSON/parser/provider exception text: it can contain
        # credential bytes. The CLI also emits only a fixed failure message.
        raise GoalInputError("private Azure goal credentials were rejected") from None
