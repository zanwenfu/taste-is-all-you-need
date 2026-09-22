"""Single-owner, crash-durable admission records for potentially paid cells."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class SweepBusy(RuntimeError):
    """Another driver owns this ledger; no second driver may admit work."""


class UnsettledSweepAttempt(RuntimeError):
    """An admitted attempt lacks a durable ending and cannot be retried safely."""


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate receipt field")
        result[key] = value
    return result


def _sync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _sync_dir(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class SweepJournal:
    """Receipts are retained; the pending pointer is cleared only after commit."""

    def __init__(self, ledger_dir: Path):
        self.root = Path(ledger_dir) / ".sweep-journal"
        self.pending_path = self.root / "pending.json"
        self._owner = None
        self._directory_owner: int | None = None
        self._active: dict[str, Any] | None = None

    def __enter__(self) -> SweepJournal:
        self.root.mkdir(parents=True, exist_ok=True)
        _sync_dir(self.root.parent)
        try:
            # Directory flock also coordinates with read-only report readers;
            # they need no writable lock file, including on historical ledgers.
            self._directory_owner = os.open(self.root.parent, os.O_RDONLY)
            fcntl.flock(self._directory_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._owner = (self.root / "owner.lock").open("a+")
            fcntl.flock(self._owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._active = self.pending()
        except BlockingIOError as exc:
            self.__exit__()
            raise SweepBusy(f"another sweep owns {self.root.parent}") from exc
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_exc: Any) -> None:
        owner, self._owner = self._owner, None
        directory, self._directory_owner = self._directory_owner, None
        try:
            if owner is not None:
                owner.close()  # Release the flock without unlinking its inode.
        finally:
            if directory is not None:
                os.close(directory)

    def assert_owner(self, ledger_dir: Path) -> None:
        if (self._owner is None or self._directory_owner is None
                or self.root.parent.resolve() != Path(ledger_dir).resolve()):
            raise SweepBusy("ledger mutation requires its active sweep owner")

    def _read(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(), object_pairs_hook=_unique_fields)
            if not isinstance(value, dict):
                raise ValueError("receipt is not an object")
            return value
        except (OSError, ValueError) as exc:
            raise UnsettledSweepAttempt(f"unreadable sweep receipt: {path}") from exc

    def pending(self) -> dict[str, Any] | None:
        if not self.pending_path.exists():
            return None
        raw = self._read(self.pending_path)
        if (set(raw) != {"schema", "attempt_id", "cell", "attempts_made"}
                or raw["schema"] != "taste/SweepAdmission/1"
                or not isinstance(raw["attempt_id"], str)
                or len(raw["attempt_id"]) != 32
                or any(char not in "0123456789abcdef" for char in raw["attempt_id"])
                or not isinstance(raw["cell"], dict)
                or set(raw["cell"]) != {"task", "arm", "trial"}
                or type(raw["attempts_made"]) is not int or raw["attempts_made"] < 1):
            raise UnsettledSweepAttempt(f"malformed sweep admission: {self.pending_path}")
        for name in ("task", "arm"):
            if not isinstance(raw["cell"][name], str) or not raw["cell"][name]:
                raise UnsettledSweepAttempt("malformed admitted cell identity")
        if type(raw["cell"]["trial"]) is not int:
            raise UnsettledSweepAttempt("malformed admitted trial identity")
        original = self._read(self.root / raw["attempt_id"] / "admission.json")
        if original != raw:
            raise UnsettledSweepAttempt("pending pointer differs from its admission receipt")
        return raw

    def begin(self, cell: Mapping[str, Any], attempts_made: int) -> str:
        if self._active is not None or self.pending_path.exists():
            raise UnsettledSweepAttempt("previous admitted sweep attempt is not settled")
        identity = uuid.uuid4().hex
        receipt_dir = self.root / identity
        receipt_dir.mkdir()
        _sync_dir(self.root)
        admission = {"schema": "taste/SweepAdmission/1", "attempt_id": identity,
                     "cell": dict(cell), "attempts_made": attempts_made}
        _write_json(receipt_dir / "admission.json", admission)
        _write_json(self.pending_path, admission)
        self._active = admission
        return identity

    def _receipt_path(self, kind: str) -> Path:
        if kind not in {"execution", "ending", "failure", "resources"}:
            raise ValueError("unknown sweep receipt kind")
        if self._active is None:
            raise UnsettledSweepAttempt("there is no active sweep admission")
        return self.root / self._active["attempt_id"] / f"{kind}.json"

    def record(self, kind: str, record: Mapping[str, Any]) -> None:
        path = self._receipt_path(kind)
        value = {"schema": "taste/SweepReceipt/1", "admission": self._active,
                 "record": dict(record)}
        # Compare the persisted JSON representation: dataclass histories use
        # tuples in memory and arrays on disk, including empty histories.
        value = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
        self._validate_receipt(value)
        if kind == "ending":
            self.assert_resources_settled()
            self._assert_execution_preserved(value["record"])
        if path.exists():
            if self._read(path) != value:
                raise UnsettledSweepAttempt("an immutable sweep receipt changed")
            return
        _write_json(path, value)

    def _validate_receipt(self, value: Mapping[str, Any]) -> None:
        record = value.get("record")
        active = self._active
        if (active is None or set(value) != {"schema", "admission", "record"}
                or value.get("schema") != "taste/SweepReceipt/1"
                or value.get("admission") != active or not isinstance(record, dict)
                or record.get("attempt_id") != active["attempt_id"]
                or record.get("attempts_made") != active["attempts_made"]
                or any(record.get(key) != val for key, val in active["cell"].items())):
            raise UnsettledSweepAttempt("sweep receipt is not bound to its admission")

    def _assert_execution_preserved(self, record: Mapping[str, Any]) -> None:
        self.assert_resources_settled()
        path = self._receipt_path("execution")
        if not path.exists():
            return
        execution = self._read(path)
        self._validate_receipt(execution)
        for key in ("billed_usd", "work_usd", "cache_delta_usd", "session_id", "final_sha",
                    "config_hash", "prior_attempts"):
            if execution["record"].get(key) != record.get(key):
                raise UnsettledSweepAttempt(f"ending changed execution evidence: {key}")

    def assert_resources_settled(self) -> None:
        if self._active is None:
            return
        path = self._receipt_path("resources")
        if path.exists():
            self._validate_receipt(self._read(path))
            raise UnsettledSweepAttempt(
                f"admitted attempt {self._active['attempt_id']}: resource cleanup is unconfirmed; "
                f"execution and grading recovery are blocked; evidence: {path}"
            )

    def execution(self) -> dict[str, Any]:
        """Read the bound receipt needed for grading without running the agent."""
        self.assert_resources_settled()
        path = self._receipt_path("execution")
        if not path.exists():
            raise UnsettledSweepAttempt("execution cost is unknown; no execution receipt is available")
        value = self._read(path)
        self._validate_receipt(value)
        return value["record"]

    def record_grading_failure(self, record: Mapping[str, Any]) -> None:
        """Retain each grading failure without replacing execution evidence."""
        self.execution()  # Grading requires a completed, durably bound execution.
        path = self._receipt_path("execution").with_name(f"grading-failure-{uuid.uuid4().hex}.json")
        value = {"schema": "taste/SweepReceipt/1", "admission": self._active, "record": dict(record)}
        self._validate_receipt(value)
        _write_json(path, value)

    def ending(self, *, allow_incomplete: bool = False) -> dict[str, Any] | None:
        if self._active is None:
            return None
        self.assert_resources_settled()
        path = self._receipt_path("ending")
        if not path.exists():
            if allow_incomplete:
                return None
            execution = self._receipt_path("execution")
            detail = "execution receipt preserved; grading is incomplete" if execution.exists() else "execution cost is unknown"
            raise UnsettledSweepAttempt(
                f"admitted attempt {self._active['attempt_id']} has no ending ({detail}); "
                f"automatic retry is blocked; evidence: {path.parent}"
            )
        value = self._read(path)
        self._validate_receipt(value)
        self._assert_execution_preserved(value["record"])
        return value["record"]

    def finish(self) -> None:
        if self._active is None:
            raise UnsettledSweepAttempt("no active admission can be finalized")
        self.assert_resources_settled()
        self.pending_path.unlink()
        _sync_dir(self.root)
        self._active = None
