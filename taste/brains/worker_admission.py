"""Exact worker launch admission, independent of any provider SDK.

Read immutable control records before acquiring a write lease, then repeat the
identity/cleanliness check under that lease before any model or tool operation.
Both historical workers and the Azure worker use these same launch bindings.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from taste.brains.contract import CONTRACT_PATH, Contract
from taste.brains.delivery import validate_artifact_path
from taste.brains.monitor import BATCH_SIZE
from taste.brains.records import Assignment
from taste.brains.worker_protocol import (
    ASSIGNMENT_PATH,
    _assignment_monitor_budget_usd,
    assignment_run_id,
)
from taste.llm import MODEL_MONITOR
from taste.memstore import Branch, Store
from taste.memstore.backend import BLOB_MODES
from taste.memstore.store import _check_name

_EXACT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_RUN_ID = re.compile(r"worker-run\.[0-9a-f]{64}\Z")
_LAUNCH_TOKEN = re.compile(r"[0-9a-f]{32}\Z")
_REGULAR_GIT_MODES = frozenset({"100644"})


class WorkerExitCode(IntEnum):
    """Stable process outcomes consumed alongside the durable WorkerReport.

    ``INCOMPLETE`` means a validated runtime reached its reporting boundary
    but did not certify completion.  The remaining non-zero codes mean no
    validated terminal report may be inferred from process exit alone.
    """

    COMPLETED = 0
    INCOMPLETE = 10
    INPUT_REJECTED = 65
    RUNTIME_FAILURE = 70
    INFRA_FAILURE = 71
    SHUTDOWN_UNCONFIRMED = 72
    INTERRUPTED = 130



class EntrypointInputError(RuntimeError):
    """The process launch is not bound to one exact durable assignment."""



@dataclass(frozen=True, slots=True)
class EntrypointConfig:
    """Non-secret, immutable worker process configuration."""

    repo_root: Path
    session: str
    worker: str
    expected_model: str
    prepared_state_id: str
    monitor_model: str = MODEL_MONITOR
    monitor_max_tokens: int = 1024
    monitor_batch_size: int = BATCH_SIZE
    monitor_budget_usd: float | None = None
    poll_interval: float = 0.05
    terminal_quiet_period: float = 0.25
    shutdown_timeout: float = 10.0

    def __post_init__(self) -> None:
        root = Path(self.repo_root).expanduser()
        try:
            root = root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("repo_root must name an existing directory") from exc
        if not root.is_dir() or not (root / ".git").exists():
            raise ValueError("repo_root must name an existing Git worktree")
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(self, "session", _check_name(self.session, "session"))
        object.__setattr__(self, "worker", _check_name(self.worker, "worker"))
        for name in ("expected_model", "monitor_model"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
                or len(value) > 256
                or any(ord(character) < 32 for character in value)
            ):
                raise ValueError(f"{name} must be a stable non-empty model id")
        if _EXACT_OBJECT_ID.fullmatch(self.prepared_state_id) is None:
            raise ValueError("prepared_state_id must be a full lowercase object id")
        for name in ("monitor_max_tokens", "monitor_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.monitor_budget_usd is not None and (
            isinstance(self.monitor_budget_usd, bool)
            or not isinstance(self.monitor_budget_usd, (int, float))
            or not 0 < float(self.monitor_budget_usd) < float("inf")
        ):
            raise ValueError("monitor_budget_usd must be finite and positive")
        if self.monitor_budget_usd is not None:
            object.__setattr__(self, "monitor_budget_usd", float(self.monitor_budget_usd))
        for name in ("poll_interval", "terminal_quiet_period", "shutdown_timeout"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < float(value) < float("inf")
            ):
                raise ValueError(f"{name} must be finite and positive")



@dataclass(frozen=True, slots=True)
class DurableWorkerInput:
    """Lease-free, exact launch input accepted from the worker branch."""

    assignment: Assignment
    contract: Contract
    head_state_id: str
    run_id: str



def _expected_readiness_path(store: Store, run_id: str) -> Path:
    key = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return store.backend.common_dir / f"taste-supervisor.{store.session}.{key}.ready.json"



def _validated_environment(
    store: Store,
    assignment: Assignment,
    environ: Mapping[str, str],
) -> str:
    run_id = environ.get("TASTE_WORKER_RUN_ID", "")
    if _RUN_ID.fullmatch(run_id) is None or run_id != assignment_run_id(assignment):
        raise EntrypointInputError("supervisor run id does not match the exact assignment")
    token = environ.get("TASTE_WORKER_LAUNCH_TOKEN", "")
    if _LAUNCH_TOKEN.fullmatch(token) is None:
        raise EntrypointInputError("supervisor launch token is missing or malformed")
    raw_ready_path = environ.get("TASTE_WORKER_READY_PATH", "")
    ready_path = Path(raw_ready_path)
    if not ready_path.is_absolute() or ready_path != _expected_readiness_path(store, run_id):
        raise EntrypointInputError("supervisor readiness path is not bound to this run")
    return run_id



def _require_regular_record(store: Store, state_id: str, path: str) -> str:
    entry = store.backend.entry_at(state_id, path)
    if entry is None or entry.mode not in _REGULAR_GIT_MODES:
        raise EntrypointInputError(f"durable {path} is not a regular control record")
    try:
        raw = store.state(state_id).read(path)
    except (OSError, UnicodeError) as exc:
        raise EntrypointInputError(f"durable {path} is not readable UTF-8") from exc
    if raw is None:
        raise EntrypointInputError(f"durable {path} is missing")
    return raw



def _load_durable_input(
    store: Store,
    config: EntrypointConfig,
    environ: Mapping[str, str],
) -> DurableWorkerInput:
    """Read and validate launch truth without opening a writable Branch."""
    if store.root != config.repo_root or store.session != config.session:
        raise EntrypointInputError("store does not match the admitted repo and session")
    view = store.view(config.worker)
    if not view.exists():
        raise EntrypointInputError("worker branch does not exist")
    try:
        head = view.head
        head_meta = head.meta
    except Exception as exc:
        raise EntrypointInputError("worker branch head is unreadable") from exc
    if head_meta.session != config.session or head_meta.branch != config.worker:
        raise EntrypointInputError("worker branch identity does not match this process")
    if head.id != config.prepared_state_id:
        raise EntrypointInputError("worker head changed after exact assignment preparation")

    contract_text = _require_regular_record(store, head.id, CONTRACT_PATH)
    assignment_text = _require_regular_record(store, head.id, ASSIGNMENT_PATH)
    try:
        contract = Contract.from_json(contract_text)
        assignment = Assignment.from_json(assignment_text)
    except Exception as exc:
        raise EntrypointInputError("durable worker control records are malformed") from exc

    # Canonical bytes reject duplicate/unknown fields even though the legacy
    # Contract decoder itself remains permissive for old callers.
    if contract.to_json() != contract_text:
        raise EntrypointInputError("durable contract is not canonical")
    if assignment.to_json() != assignment_text:
        raise EntrypointInputError("durable assignment is not canonical")
    if assignment.contract != contract or assignment.contract.to_json() != contract_text:
        raise EntrypointInputError("durable assignment and contract disagree")
    if assignment.worker != config.worker:
        raise EntrypointInputError("assignment worker identity does not match this process")
    if assignment.model != config.expected_model:
        raise EntrypointInputError("assignment model does not match this process")
    if contract.inputs != tuple(item.path for item in assignment.inputs):
        raise EntrypointInputError("contract inputs do not match the exact structured input paths")
    if contract.outputs != tuple(item.path for item in assignment.outputs):
        raise EntrypointInputError(
            "contract outputs do not match the exact structured output paths"
        )
    try:
        assignment_monitor_budget = _assignment_monitor_budget_usd(assignment)
    except ValueError as exc:
        raise EntrypointInputError("assignment monitor budget is invalid") from exc
    if config.monitor_budget_usd != assignment_monitor_budget:
        raise EntrypointInputError("monitor budget does not match the exact durable assignment")

    try:
        prepared = store.state(config.prepared_state_id)
        prepared_meta = prepared.meta
    except Exception as exc:
        raise EntrypointInputError("prepared state does not exist") from exc
    if prepared_meta.session != config.session or prepared_meta.branch != config.worker:
        raise EntrypointInputError("prepared state belongs to another worker or session")
    if (
        _require_regular_record(store, prepared.id, CONTRACT_PATH) != contract_text
        or _require_regular_record(store, prepared.id, ASSIGNMENT_PATH) != assignment_text
    ):
        raise EntrypointInputError("prepared control records differ from the live assignment")

    # These checks formerly lived only inside the Claude runtime. They are
    # admission invariants for every provider, before taking a worker lease.
    if _EXACT_OBJECT_ID.fullmatch(assignment.base_state_id) is None:
        raise EntrypointInputError("assignment base state must be an exact immutable id")
    try:
        base = store.state(assignment.base_state_id)
        ancestor = store.backend.is_ancestor(base.id, prepared.id)
    except Exception as exc:
        raise EntrypointInputError("assignment base state is unavailable") from exc
    if base.id != assignment.base_state_id or not ancestor:
        raise EntrypointInputError("assignment base state is not an ancestor of prepared work")
    for artifact in (*assignment.inputs, *assignment.outputs):
        try:
            validate_artifact_path(artifact.path)
        except ValueError as exc:
            raise EntrypointInputError("assignment contains an invalid artifact path") from exc

    for artifact in assignment.inputs:
        if _EXACT_OBJECT_ID.fullmatch(artifact.state_id) is None:
            raise EntrypointInputError("input source must be an exact immutable state id")
        try:
            source_state = store.state(artifact.state_id)
            source_meta = source_state.meta
        except Exception as exc:
            raise EntrypointInputError("input source state is unavailable") from exc
        if (source_state.id != artifact.state_id or source_meta.session != config.session
                or source_meta.branch != artifact.branch):
            raise EntrypointInputError("input source belongs to another session or branch")
        source = store.backend.entry_at(artifact.state_id, artifact.path)
        projected = store.backend.entry_at(prepared.id, artifact.path)
        if (source is None or source.mode not in BLOB_MODES
                or source.sha != artifact.blob_id or projected != source):
            raise EntrypointInputError(
                f"prepared input {artifact.artifact_id!r} bytes or mode do not match its source"
            )

    run_id = _validated_environment(store, assignment, environ)
    return DurableWorkerInput(
        assignment=assignment,
        contract=contract,
        head_state_id=head.id,
        run_id=run_id,
    )


def validate_acquired_branch(
    branch: Branch, store: Store, config: EntrypointConfig, durable: DurableWorkerInput,
) -> None:
    """Close the read/acquire race before a provider or tool can run."""
    if branch.store is not store:
        raise EntrypointInputError("worker branch changed the exact store")
    if branch.name != config.worker or branch.head.id != durable.head_state_id:
        raise EntrypointInputError("worker head changed while acquiring its lease")
    if branch.head.id != config.prepared_state_id:
        raise EntrypointInputError("worker branch differs from its prepared state")
    if branch.dirty_paths():
        raise EntrypointInputError("worker worktree was dirty when its lease was acquired")
