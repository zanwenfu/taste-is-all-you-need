"""Production, telemetry-bearing LLM transport for the central planner.

The central planner owns durable request/outcome records.  This module owns
the provider-facing boundary: preflight, one tool-free planner call, strict
completion validation, and canonical per-call pricing.  A provider response
can be charged yet unusable; :class:`PlannerCompletionError` therefore carries
any exact usage and response bytes recovered before validation failed.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from taste.llm import LLM, MODEL_PLANNER
from taste.memstore import Branch, State, Store
from taste.pricing import (
    PricingError,
    call_cost,
    ensure_priced,
    max_call_cost_usd,
    table_sha,
)
from taste.providers.base import Usage

__all__ = [
    "PLANNER_RECEIPT_BRANCH",
    "PLANNER_RECEIPT_INTENT_SCHEMA",
    "PLANNER_RECEIPT_SCHEMA",
    "PLANNER_TELEMETRY_SCHEMA",
    "PLANNER_TRANSPORT_ROOT",
    "PLANNER_USAGE_SCHEMA",
    "LLMPlannerTransport",
    "PlannerCompletion",
    "PlannerCompletionError",
    "PlannerReceiptError",
    "PlannerTelemetry",
    "PlannerTransportEvidence",
    "PlannerUsage",
    "assert_planner_transport_history",
    "load_planner_transport_evidence",
    "planner_transport_evidences",
    "planner_transport_intent_path",
    "planner_transport_outcome_path",
]

PLANNER_USAGE_SCHEMA = "taste.brains/PlannerUsage/1"
PLANNER_TELEMETRY_SCHEMA = "taste.brains/PlannerTelemetry/1"
PLANNER_RECEIPT_INTENT_SCHEMA = "taste.brains/PlannerReceiptIntent/1"
PLANNER_RECEIPT_SCHEMA = "taste.brains/PlannerReceipt/2"
PLANNER_RECEIPT_BRANCH = "planner-receipts"
PLANNER_TRANSPORT_ROOT = ".taste/planner/transport"

_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _count(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative integer")
    return value


def _cost(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{where} must be a non-negative finite number")
    return result


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: bytes | str) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PlannerUsage:
    """One provider completion's normalized, disjoint token buckets."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
        ):
            object.__setattr__(self, name, _count(getattr(self, name), name))
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning_tokens cannot exceed output_tokens")

    @classmethod
    def from_provider(cls, usage: Any) -> PlannerUsage:
        if not isinstance(usage, Usage):
            raise ValueError("planner completion usage is not normalized Usage")
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            reasoning_tokens=usage.reasoning_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLANNER_USAGE_SCHEMA,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

    @classmethod
    def from_dict(cls, value: Any) -> PlannerUsage:
        if not isinstance(value, dict):
            raise ValueError("planner usage must be an object")
        required = {
            "schema",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
        }
        if set(value) != required or value.get("schema") != PLANNER_USAGE_SCHEMA:
            raise ValueError("planner usage fields or schema are invalid")
        return cls(
            input_tokens=value["input_tokens"],
            output_tokens=value["output_tokens"],
            cache_read_tokens=value["cache_read_tokens"],
            cache_write_tokens=value["cache_write_tokens"],
            reasoning_tokens=value["reasoning_tokens"],
        )


@dataclass(frozen=True)
class PlannerTelemetry:
    """Auditable model identity, tokens and both canonical cost currencies."""

    source: str
    cost_known: bool
    requested_model: str | None = None
    model: str | None = None
    provider: str | None = None
    usage: PlannerUsage | None = None
    billed_usd: float | None = None
    work_usd: float | None = None
    pricing_table_sha: str | None = None
    pricing_as_of: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _text(self.source, "telemetry source"))
        if not isinstance(self.cost_known, bool):
            raise ValueError("cost_known must be boolean")
        for name in ("requested_model", "model", "provider", "pricing_table_sha", "pricing_as_of"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        if self.usage is not None and not isinstance(self.usage, PlannerUsage):
            raise ValueError("usage must be PlannerUsage or None")

        if self.cost_known:
            if self.billed_usd is None or self.work_usd is None:
                raise ValueError("known planner cost requires billed and work costs")
            object.__setattr__(self, "billed_usd", _cost(self.billed_usd, "billed_usd"))
            object.__setattr__(self, "work_usd", _cost(self.work_usd, "work_usd"))
            if self.source in {"legacy_string_zero", "not_called_zero", "not_dispatched_zero"}:
                if self.usage != PlannerUsage(0, 0, 0, 0, 0):
                    raise ValueError("synthetic zero telemetry must contain zero usage")
                if self.source == "not_dispatched_zero" and self.requested_model is None:
                    raise ValueError("not-dispatched telemetry requires the requested model")
                if any(
                    value is not None
                    for value in (
                        self.requested_model if self.source != "not_dispatched_zero" else None,
                        self.model,
                        self.provider,
                        self.pricing_table_sha,
                        self.pricing_as_of,
                    )
                ):
                    raise ValueError("synthetic zero telemetry cannot claim a model or price")
                if self.billed_usd != 0.0 or self.work_usd != 0.0:
                    raise ValueError("synthetic zero telemetry must have zero cost")
            elif any(
                value is None
                for value in (
                    self.requested_model,
                    self.model,
                    self.provider,
                    self.usage,
                    self.pricing_table_sha,
                    self.pricing_as_of,
                )
            ):
                raise ValueError("known LLM telemetry is missing an audit field")
        elif self.billed_usd is not None or self.work_usd is not None:
            raise ValueError("unknown planner cost cannot contain cost values")

    @classmethod
    def synthetic_zero(cls, source: str = "legacy_string_zero") -> PlannerTelemetry:
        return cls(
            source=source,
            cost_known=True,
            usage=PlannerUsage(0, 0, 0, 0, 0),
            billed_usd=0.0,
            work_usd=0.0,
        )

    @classmethod
    def not_dispatched(cls, requested_model: str) -> PlannerTelemetry:
        """An admitted call stopped locally before invoking the LLM facade."""
        return cls(source="not_dispatched_zero", cost_known=True, requested_model=requested_model,
                   usage=PlannerUsage(0, 0, 0, 0, 0), billed_usd=0.0, work_usd=0.0)

    @classmethod
    def unknown(
        cls,
        *,
        source: str,
        requested_model: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        usage: PlannerUsage | None = None,
    ) -> PlannerTelemetry:
        return cls(
            source=source,
            cost_known=False,
            requested_model=requested_model,
            model=model,
            provider=provider,
            usage=usage,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLANNER_TELEMETRY_SCHEMA,
            "source": self.source,
            "cost_known": self.cost_known,
            "requested_model": self.requested_model,
            "model": self.model,
            "provider": self.provider,
            "usage": None if self.usage is None else self.usage.to_dict(),
            "billed_usd": self.billed_usd,
            "work_usd": self.work_usd,
            "pricing_table_sha": self.pricing_table_sha,
            "pricing_as_of": self.pricing_as_of,
        }

    @classmethod
    def from_dict(cls, value: Any) -> PlannerTelemetry:
        if not isinstance(value, dict):
            raise ValueError("planner telemetry must be an object")
        required = {
            "schema",
            "source",
            "cost_known",
            "requested_model",
            "model",
            "provider",
            "usage",
            "billed_usd",
            "work_usd",
            "pricing_table_sha",
            "pricing_as_of",
        }
        if set(value) != required or value.get("schema") != PLANNER_TELEMETRY_SCHEMA:
            raise ValueError("planner telemetry fields or schema are invalid")
        return cls(
            source=value["source"],
            cost_known=value["cost_known"],
            requested_model=value["requested_model"],
            model=value["model"],
            provider=value["provider"],
            usage=None if value["usage"] is None else PlannerUsage.from_dict(value["usage"]),
            billed_usd=value["billed_usd"],
            work_usd=value["work_usd"],
            pricing_table_sha=value["pricing_table_sha"],
            pricing_as_of=value["pricing_as_of"],
        )


@dataclass(frozen=True)
class PlannerCompletion:
    """A complete textual planner response and its exact call telemetry."""

    text: str
    telemetry: PlannerTelemetry

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip() or "\x00" in self.text:
            raise ValueError("planner completion text must be non-empty")
        if not isinstance(self.telemetry, PlannerTelemetry):
            raise ValueError("planner completion telemetry is invalid")
        if not self.telemetry.cost_known:
            raise ValueError("a completed planner call must have known cost")


class PlannerCompletionError(RuntimeError):
    """A charged or malformed provider completion cannot be used safely."""

    failure_kind = "infra"
    planner_terminal = True

    def __init__(
        self,
        message: str,
        *,
        telemetry: PlannerTelemetry,
        response: str | None = None,
    ) -> None:
        super().__init__(message)
        self.telemetry = telemetry
        self.response = response


class PlannerReceiptError(RuntimeError):
    """The durable transport receipt is malformed or identity-conflicting."""

    failure_kind = "infra"
    planner_terminal = True


def planner_transport_intent_path(request_id: str) -> str:
    request_id = _text(request_id, "planner request_id")
    return f"{PLANNER_TRANSPORT_ROOT}/{_sha256(request_id)}/intent.json"


def planner_transport_outcome_path(request_id: str) -> str:
    request_id = _text(request_id, "planner request_id")
    return f"{PLANNER_TRANSPORT_ROOT}/{_sha256(request_id)}/outcome.json"


def _strict_record(text: str, where: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PlannerReceiptError(f"{where} contains duplicate key {key!r}")
            result[key] = value
        return result

    def no_constant(value: str) -> Any:
        raise PlannerReceiptError(f"{where} contains non-JSON number {value}")

    try:
        raw = json.loads(text, object_pairs_hook=unique, parse_constant=no_constant)
    except json.JSONDecodeError as exc:
        raise PlannerReceiptError(f"{where} is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise PlannerReceiptError(f"{where} must be a JSON object")
    return raw


def _validate_binding(binding: Any) -> dict[str, Any]:
    required = {
        "system_sha256",
        "prompt_sha256",
        "model",
        "model_sha256",
        "config_sha256",
    }
    if (
        not isinstance(binding, dict)
        or set(binding) != required
        or not all(isinstance(value, str) and value for value in binding.values())
        or any(
            len(binding[name]) != 64
            or any(character not in "0123456789abcdef" for character in binding[name])
            for name in (
                "system_sha256",
                "prompt_sha256",
                "model_sha256",
                "config_sha256",
            )
        )
        or binding["model_sha256"] != _sha256(binding["model"])
    ):
        raise PlannerReceiptError("planner receipt request binding is invalid")
    return binding


def _validate_admission(admission: Any) -> dict[str, Any]:
    if not isinstance(admission, dict) or set(admission) != {
        "remaining_usd",
        "max_billed_call_usd",
    }:
        raise PlannerReceiptError("planner receipt admission fields are invalid")
    try:
        ceiling = _cost(admission["max_billed_call_usd"], "max_billed_call_usd")
        remaining_raw = admission["remaining_usd"]
        remaining = None if remaining_raw is None else _cost(remaining_raw, "remaining_usd")
    except ValueError as exc:
        raise PlannerReceiptError("planner receipt admission is invalid") from exc
    if ceiling <= 0 or (remaining is not None and remaining < ceiling):
        raise PlannerReceiptError("planner receipt admission did not cover call exposure")
    return {
        "remaining_usd": remaining,
        "max_billed_call_usd": ceiling,
    }


def _validate_scope(scope: Any, request_id: str) -> dict[str, Any] | None:
    if scope is None:
        return None
    required = {
        "goal_id",
        "operation_id",
        "planning_request_id",
        "attempt_id",
        "attempt",
        "at",
        "central_request_path",
        "central_request_sha256",
        "central_intent_path",
        "central_intent_sha256",
    }
    if not isinstance(scope, dict) or set(scope) != required:
        raise PlannerReceiptError("planner receipt scope fields are invalid")
    for name in (
        "goal_id",
        "operation_id",
        "planning_request_id",
        "attempt_id",
        "at",
        "central_request_path",
        "central_intent_path",
    ):
        try:
            _text(scope[name], f"planner receipt scope {name}")
        except ValueError as exc:
            raise PlannerReceiptError("planner receipt scope identity is invalid") from exc
    if scope["attempt_id"] != request_id:
        raise PlannerReceiptError("planner receipt scope attempt identity is invalid")
    attempt = scope["attempt"]
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise PlannerReceiptError("planner receipt scope attempt number is invalid")
    for name in ("central_request_sha256", "central_intent_sha256"):
        digest = scope[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise PlannerReceiptError("planner receipt scope digest is invalid")
    return scope


@dataclass(frozen=True)
class PlannerTransportEvidence:
    """Exact immutable provider-boundary evidence for one planner attempt."""

    request_id: str
    binding: dict[str, Any]
    admission: dict[str, Any]
    scope: dict[str, Any] | None
    status: str
    response: str | None
    telemetry: PlannerTelemetry
    error: str | None


def load_planner_transport_evidence(
    state: State, request_id: str
) -> PlannerTransportEvidence | None:
    """Load and cross-check one attempt's immutable intent/outcome pair."""
    request_id = _text(request_id, "planner request_id")
    intent_path = planner_transport_intent_path(request_id)
    outcome_path = planner_transport_outcome_path(request_id)
    intent_raw = state.read(intent_path)
    outcome_raw = state.read(outcome_path)
    if intent_raw is None:
        if outcome_raw is not None:
            raise PlannerReceiptError("planner transport outcome has no durable intent")
        return None

    intent = _strict_record(intent_raw, "planner receipt intent")
    if (
        set(intent) != {"schema", "request_id", "binding", "admission", "scope"}
        or intent.get("schema") != PLANNER_RECEIPT_INTENT_SCHEMA
        or intent.get("request_id") != request_id
    ):
        raise PlannerReceiptError("planner receipt intent fields or identity are invalid")
    binding = _validate_binding(intent["binding"])
    admission = _validate_admission(intent["admission"])
    scope = _validate_scope(intent["scope"], request_id)
    if outcome_raw is None:
        return PlannerTransportEvidence(
            request_id=request_id,
            binding=binding,
            admission=admission,
            scope=scope,
            status="pending",
            response=None,
            telemetry=PlannerTelemetry.unknown(
                source="pending_receipt",
                requested_model=binding["model"],
            ),
            error=None,
        )

    outcome = _strict_record(outcome_raw, "planner receipt outcome")
    required = {
        "schema",
        "request_id",
        "intent_sha256",
        "status",
        "response",
        "response_sha256",
        "telemetry",
        "error",
    }
    if (
        set(outcome) != required
        or outcome.get("schema") != PLANNER_RECEIPT_SCHEMA
        or outcome.get("request_id") != request_id
        or outcome.get("intent_sha256") != _sha256(intent_raw)
    ):
        raise PlannerReceiptError("planner receipt outcome fields or intent identity are invalid")
    status = outcome["status"]
    if status not in {"completed", "invalid", "ambiguous"}:
        raise PlannerReceiptError("planner receipt outcome status is invalid")
    response = outcome["response"]
    if response is not None and not isinstance(response, str):
        raise PlannerReceiptError("planner receipt response is invalid")
    if outcome["response_sha256"] != (None if response is None else _sha256(response)):
        raise PlannerReceiptError("planner receipt response digest does not match")
    try:
        telemetry = PlannerTelemetry.from_dict(outcome["telemetry"])
    except ValueError as exc:
        raise PlannerReceiptError("planner receipt telemetry is invalid") from exc
    if telemetry.requested_model != binding["model"]:
        raise PlannerReceiptError("planner receipt telemetry model binding is invalid")
    error = outcome["error"]
    if status == "completed":
        if (
            not telemetry.cost_known
            or telemetry.model != binding["model"]
            or response is None
            or not response.strip()
            or error is not None
        ):
            raise PlannerReceiptError("completed planner receipt invariants are invalid")
    elif not isinstance(error, str) or not error.strip():
        raise PlannerReceiptError("failed planner receipt has no error detail")
    return PlannerTransportEvidence(
        request_id=request_id,
        binding=binding,
        admission=admission,
        scope=scope,
        status=status,
        response=response,
        telemetry=telemetry,
        error=error,
    )


def planner_transport_evidences(state: State) -> tuple[PlannerTransportEvidence, ...]:
    """Enumerate every transport intent in one exact journal state."""
    prefix = f"{PLANNER_TRANSPORT_ROOT}/"
    intent_ids: set[str] = set()
    outcome_ids: set[str] = set()
    for path in state.files():
        if not path.startswith(prefix):
            continue
        if path.endswith("/intent.json"):
            kind = "intent"
        elif path.endswith("/outcome.json"):
            kind = "outcome"
        else:
            raise PlannerReceiptError(
                f"planner transport authority contains unexpected path {path!r}"
            )
        raw = state.read(path)
        if raw is None:
            raise PlannerReceiptError("planner transport record disappeared during audit")
        record = _strict_record(raw, f"planner receipt {kind}")
        request_id = record.get("request_id")
        if not isinstance(request_id, str):
            raise PlannerReceiptError(f"planner receipt {kind} has no request identity")
        expected_path = (
            planner_transport_intent_path(request_id)
            if kind == "intent"
            else planner_transport_outcome_path(request_id)
        )
        if expected_path != path:
            raise PlannerReceiptError(f"planner receipt {kind} path has the wrong identity")
        destination = intent_ids if kind == "intent" else outcome_ids
        if request_id in destination:
            raise PlannerReceiptError(f"planner receipt {kind} identity is duplicated")
        destination.add(request_id)
    outcome_without_intent = outcome_ids - intent_ids
    if outcome_without_intent:
        raise PlannerReceiptError(
            "planner transport outcome has no durable intent: "
            f"{sorted(outcome_without_intent)[0]!r}"
        )
    return tuple(
        evidence
        for request_id in sorted(intent_ids)
        if (evidence := load_planner_transport_evidence(state, request_id)) is not None
    )


def assert_planner_transport_history(control: Branch) -> None:
    """Prove immutable transport evidence was never deleted or rewritten."""
    if not isinstance(control, Branch):
        raise TypeError("planner transport control must be a Branch")

    def protected(path: str) -> bool:
        return path.startswith(f"{PLANNER_TRANSPORT_ROOT}/") and path.endswith(
            ("/intent.json", "/outcome.json")
        )

    established: dict[str, str] = {}
    try:
        for state in reversed(control.history()):
            present = {
                path: raw
                for path in state.files()
                if protected(path) and (raw := state.read(path)) is not None
            }
            missing = established.keys() - present.keys()
            if missing:
                raise PlannerReceiptError(
                    f"planner transport history deleted immutable evidence {sorted(missing)[0]!r}"
                )
            for path, raw in present.items():
                previous = established.setdefault(path, raw)
                if previous != raw:
                    raise PlannerReceiptError(
                        f"planner transport history rewrote immutable evidence {path!r}"
                    )
    except PlannerReceiptError:
        raise
    except Exception as exc:
        raise PlannerReceiptError("planner transport history is unreadable") from exc


class LLMPlannerTransport:
    """Strict planner calls with a crash-safe, exactly-once receipt boundary.

    The sidecar lock is held from intent publication through receipt
    publication. A process death releases the kernel lock but leaves its
    pending intent. The next process reports that attempt as cost-unknown and
    does not guess whether the provider charged it by calling again.

    ``max_prompt_bytes`` bounds combined serialized system/user input before
    dispatch. It is an admission guard; observation construction and paged
    evidence access have separate resource requirements.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        store: Store,
        control: Branch,
        journal: Branch,
        mutation_lock: threading.RLock,
        model: str = MODEL_PLANNER,
        max_tokens: int = 8192,
        max_prompt_bytes: int = 192 * 1024,
    ) -> None:
        if not isinstance(store, Store):
            raise TypeError("planner transport store must be a Store")
        if not isinstance(control, Branch) or control.store is not store:
            raise ValueError("planner transport control must be the exact writable Store branch")
        if not isinstance(journal, Branch) or journal.store is not store or journal is control:
            raise ValueError("planner transport journal must be a distinct writable Store branch")
        if not isinstance(mutation_lock, type(threading.RLock())):
            raise TypeError("planner transport mutation_lock must be a threading.RLock")
        if not callable(getattr(llm, "call", None)):
            raise TypeError("planner llm must provide call()")
        ensure_ready = getattr(llm, "ensure_ready", None)
        if not callable(ensure_ready):
            raise TypeError("planner llm must provide ensure_ready()")
        self.model = _text(model, "planner model")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("planner max_tokens must be a positive integer")
        if type(max_prompt_bytes) is not int or max_prompt_bytes < 1:
            raise ValueError("planner max_prompt_bytes must be a positive integer")
        self.llm = llm
        self.store = store
        self.control = control
        self.journal = journal
        self.mutation_lock = mutation_lock
        self.max_tokens = max_tokens
        self.max_prompt_bytes = max_prompt_bytes
        self.journal_branch = journal.name
        self._ensure_ready = ensure_ready
        self._bound_remaining_usd: float | None = None
        self._planner_bound = False
        self._deadline: float | None = None

    def bind_deadline(self, *, remaining_seconds: float) -> None:
        if not math.isfinite(remaining_seconds) or remaining_seconds <= 0:
            raise ValueError("planner deadline must be finite and positive")
        self._deadline = time.monotonic() + remaining_seconds

    def _lock_path(self):
        return self.store.sidecar("planner-transport-lock", self.journal.name)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        lock_path = self._lock_path()
        lock_key = str(lock_path)
        with _PROCESS_LOCKS_GUARD:
            process_lock = _PROCESS_LOCKS.setdefault(lock_key, threading.RLock())
        with process_lock:
            handle = open(lock_path, "a+b")  # noqa: SIM115 - lock owns descriptor lifetime
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()

    def _max_attempts(self) -> int:
        value = getattr(self.llm, "max_attempts", 1)
        if isinstance(value, bool) or not isinstance(value, int) or value != 1:
            raise PlannerReceiptError(
                "planner LLM max_attempts must be exactly 1: a retry is a "
                "separate possibly billed request and needs its own durable receipt"
            )
        return value

    def max_billed_call_usd(self) -> float:
        """Conservative billed exposure of one facade call, including retries."""
        try:
            return max_call_cost_usd(
                self.model,
                max_output_tokens=self.max_tokens,
                max_attempts=self._max_attempts(),
                cap_on="billed",
            )
        except (PricingError, TypeError, ValueError) as exc:
            raise PlannerReceiptError(
                "planner billed call exposure is not finite and positive"
            ) from exc

    def bind_budget(self, *, remaining_usd: float) -> None:
        """Bind the next genuinely new provider call to durable remaining budget."""
        try:
            remaining = _cost(remaining_usd, "remaining_usd")
        except ValueError as exc:
            raise PlannerReceiptError("planner remaining budget is invalid") from exc
        exposure = self.max_billed_call_usd()
        if remaining < exposure:
            raise PlannerReceiptError(
                "planner remaining budget does not cover worst-case billed call exposure"
            )
        with self.mutation_lock:
            self._bound_remaining_usd = remaining

    def _call_config(self) -> dict[str, Any]:
        price = ensure_priced(self.model)
        llm_budget = getattr(self.llm, "budget_usd", None)
        if llm_budget is not None:
            try:
                llm_budget = _cost(llm_budget, "planner LLM budget_usd")
            except ValueError as exc:
                raise PlannerReceiptError("planner LLM budget is invalid") from exc
        cap_on = getattr(self.llm, "cap_on", None)
        if cap_on is not None and cap_on not in {"billed", "work"}:
            raise PlannerReceiptError("planner LLM cap currency is invalid")
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "max_prompt_bytes": self.max_prompt_bytes,
            "temperature": 0.0,
            "tools": None,
            "role": "planner",
            "llm_max_attempts": self._max_attempts(),
            "llm_budget_usd": llm_budget,
            "llm_cap_on": cap_on,
            "pricing_table_sha": table_sha(),
            "pricing_as_of": price.as_of,
            "context_window": price.context_window,
            "max_billed_call_usd": self.max_billed_call_usd(),
        }

    def _binding(self, *, system: str, prompt: str) -> dict[str, Any]:
        config = self._call_config()
        return {
            "system_sha256": _sha256(system),
            "prompt_sha256": _sha256(prompt),
            "model": self.model,
            "model_sha256": _sha256(self.model),
            "config_sha256": _sha256(_canonical(config)),
        }

    @staticmethod
    def _intent_record(
        request_id: str,
        binding: dict[str, Any],
        admission: dict[str, Any],
        scope: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "schema": PLANNER_RECEIPT_INTENT_SCHEMA,
            "request_id": request_id,
            "binding": binding,
            "admission": admission,
            "scope": scope,
        }

    @staticmethod
    def _terminal_receipt(
        request_id: str,
        intent_raw: str,
        *,
        status: str,
        telemetry: PlannerTelemetry,
        response: str | None,
        error: str | None,
    ) -> dict[str, Any]:
        return {
            "schema": PLANNER_RECEIPT_SCHEMA,
            "request_id": request_id,
            "intent_sha256": _sha256(intent_raw),
            "status": status,
            "response": response,
            "response_sha256": None if response is None else _sha256(response),
            "telemetry": telemetry.to_dict(),
            "error": error,
        }

    def _replay(
        self,
        evidence: PlannerTransportEvidence,
        *,
        binding: dict[str, Any],
    ) -> PlannerCompletion:
        if evidence.binding != binding:
            raise PlannerReceiptError(
                "planner request_id was reused with different prompt, model, or call config"
            )
        status = evidence.status
        if status == "completed":
            return PlannerCompletion(
                text=evidence.response or "",
                telemetry=evidence.telemetry,
            )
        if status == "invalid":
            raise PlannerCompletionError(
                evidence.error or "invalid planner completion",
                telemetry=evidence.telemetry,
                response=evidence.response,
            )
        if status == "ambiguous":
            telemetry = evidence.telemetry
            detail = evidence.error or "ambiguous planner completion"
        else:
            telemetry = evidence.telemetry
            detail = (
                "planner request has a surviving pending receipt; provider completion and "
                "cost are ambiguous, so this attempt will not be called again"
            )
        raise PlannerCompletionError(detail, telemetry=telemetry)

    def bind_planner(self) -> None:
        """Require every subsequent call to bind an exact CentralPlanner attempt."""
        with self.mutation_lock:
            self._planner_bound = True

    def _attempt_scope(self, request_id: str, prompt: str) -> dict[str, Any] | None:
        candidates: list[tuple[str, str, dict[str, Any]]] = []
        prefix = ".taste/planner/operations/"
        for path in self.control.head.files():
            if not path.startswith(prefix) or not path.endswith("/intent.json"):
                continue
            raw = self.control.head.read(path)
            if raw is None:
                raise PlannerReceiptError("central planner attempt intent disappeared")
            intent = _strict_record(raw, "central planner attempt intent")
            if intent.get("attempt_id") == request_id:
                candidates.append((path, raw, intent))
        if not candidates:
            if self._planner_bound:
                raise PlannerReceiptError(
                    "planner transport call has no exact central attempt intent"
                )
            return None
        if len(candidates) != 1:
            raise PlannerReceiptError("planner transport attempt identity is ambiguous")

        intent_path, intent_raw, intent = candidates[0]
        required_intent = {
            "schema",
            "request_id",
            "attempt_id",
            "attempt",
            "at",
        }
        if (
            set(intent) != required_intent
            or intent.get("schema") != "taste.brains/PlanningAttempt/1"
            or intent.get("attempt_id") != request_id
        ):
            raise PlannerReceiptError("central planner attempt intent is malformed")
        attempt = intent.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise PlannerReceiptError("central planner attempt number is invalid")
        root = intent_path.rsplit("/attempts/", 1)[0]
        request_path = f"{root}/request.json"
        request_raw = self.control.head.read(request_path)
        if request_raw is None:
            raise PlannerReceiptError("central planner request is missing")
        request = _strict_record(request_raw, "central planner request")
        prompt_record = _strict_record(prompt, "central planner prompt")
        prompt_request = prompt_record.get("request")
        goal = request.get("goal")
        if (
            not isinstance(prompt_request, dict)
            or prompt_request != request
            or not isinstance(goal, dict)
            or intent.get("request_id") != request.get("request_id")
            or not isinstance(goal.get("goal_id"), str)
            or not isinstance(request.get("operation_id"), str)
            or not isinstance(request.get("request_id"), str)
            or not isinstance(intent.get("at"), str)
        ):
            raise PlannerReceiptError(
                "planner prompt, request, and attempt intent are not exactly bound"
            )
        return {
            "goal_id": goal["goal_id"],
            "operation_id": request["operation_id"],
            "planning_request_id": request["request_id"],
            "attempt_id": request_id,
            "attempt": attempt,
            "at": intent["at"],
            "central_request_path": request_path,
            "central_request_sha256": _sha256(request_raw),
            "central_intent_path": intent_path,
            "central_intent_sha256": _sha256(intent_raw),
        }

    def _scope_is_current(self, scope: dict[str, Any] | None) -> bool:
        if scope is None:
            return not self._planner_bound
        request_raw = self.control.head.read(scope["central_request_path"])
        intent_raw = self.control.head.read(scope["central_intent_path"])
        return (
            request_raw is not None
            and intent_raw is not None
            and _sha256(request_raw) == scope["central_request_sha256"]
            and _sha256(intent_raw) == scope["central_intent_sha256"]
        )

    def _admission(self) -> dict[str, Any]:
        exposure = self.max_billed_call_usd()
        remaining = self._bound_remaining_usd
        self._bound_remaining_usd = None
        if getattr(self.llm, "budget_usd", None) is not None and remaining is None:
            raise PlannerReceiptError(
                "budgeted planner LLM requires bind_budget() before a new provider call"
            )
        if remaining is not None and remaining < exposure:
            raise PlannerReceiptError(
                "planner remaining budget does not cover worst-case billed call exposure"
            )
        return {
            "remaining_usd": remaining,
            "max_billed_call_usd": exposure,
        }

    def _commit_record(
        self,
        branch: Branch,
        path: str,
        payload: dict[str, Any],
        reason: str,
    ) -> str:
        wanted = json.dumps(payload, indent=1, sort_keys=True) + "\n"
        existing = branch.head.read(path)
        if existing is not None:
            if existing != wanted:
                raise PlannerReceiptError(
                    f"immutable planner transport record {path!r} changed identity"
                )
            return existing
        historical = {raw for state in branch.history() if (raw := state.read(path)) is not None}
        if historical:
            raise PlannerReceiptError(
                f"immutable planner transport record {path!r} disappeared from current state"
            )
        if branch.dirty_paths():
            raise PlannerReceiptError(
                f"planner transport cannot checkpoint dirty shared branch {branch.name!r}"
            )
        state = branch.checkpoint(reason, records={path: payload})
        recorded = state.read(path)
        if recorded != wanted:
            raise PlannerReceiptError("planner transport record checkpoint changed exact bytes")
        # A dirty shared worktree could otherwise smuggle deletion of an older
        # receipt into this checkpoint immediately before another paid call.
        assert_planner_transport_history(branch)
        return recorded

    def _paired_evidences(self) -> tuple[PlannerTransportEvidence, ...]:
        """Audit both ledgers once and return the independent authority."""
        assert_planner_transport_history(self.journal)
        assert_planner_transport_history(self.control)
        journal_by_request = {
            evidence.request_id: evidence
            for evidence in planner_transport_evidences(self.journal.head)
        }
        control_by_request = {
            evidence.request_id: evidence
            for evidence in planner_transport_evidences(self.control.head)
        }
        missing_authority = control_by_request.keys() - journal_by_request.keys()
        if missing_authority:
            raise PlannerReceiptError(
                "control acknowledges planner transport evidence absent from authority: "
                f"{sorted(missing_authority)[0]!r}"
            )
        for mirrored_id in control_by_request.keys() & journal_by_request.keys():
            intent_path = planner_transport_intent_path(mirrored_id)
            if self.journal.head.read(intent_path) != self.control.head.read(intent_path):
                raise PlannerReceiptError("planner transport intent mirrors disagree")
            outcome_path = planner_transport_outcome_path(mirrored_id)
            journal_outcome = self.journal.head.read(outcome_path)
            control_outcome = self.control.head.read(outcome_path)
            if journal_outcome is None and control_outcome is not None:
                raise PlannerReceiptError(
                    "control outcome is absent from planner receipt authority"
                )
            if (
                journal_outcome is not None
                and control_outcome is not None
                and journal_outcome != control_outcome
            ):
                raise PlannerReceiptError("planner transport outcome mirrors disagree")
        return tuple(journal_by_request[key] for key in sorted(journal_by_request))

    def _paired_evidence(self, request_id: str) -> PlannerTransportEvidence | None:
        for evidence in self._paired_evidences():
            if evidence.request_id == request_id:
                return evidence
        return None

    def _mirror_authority(self, request_id: str) -> None:
        # The transport is reached directly as well as through CentralPlanner,
        # so it is its own first touch of a control branch this coordinator
        # owns. Same act, same declaration: the receipt journal lives on
        # another branch and survives the rewind, which is what makes this a
        # repair rather than a worker publishing over its own lost states.
        self.control.adopt_rewind_if_any(
            evidence=f"{self.journal.name}:{request_id}",
            reason=(
                "control branch was rewound; replaying planner transport "
                f"authority for {request_id} from the receipt journal"
            ),
        )
        for path, reason in (
            (
                planner_transport_intent_path(request_id),
                f"planner transport intent: {request_id}",
            ),
            (
                planner_transport_outcome_path(request_id),
                f"planner transport outcome: {request_id}",
            ),
        ):
            raw = self.journal.head.read(path)
            if raw is None or self.control.head.read(path) is not None:
                continue
            payload = _strict_record(raw, "planner receipt authority record")
            mirrored = self._commit_record(self.control, path, payload, reason)
            if mirrored != raw:
                raise PlannerReceiptError("planner transport mirror changed authority bytes")

    def _commit_authoritative_pair(
        self,
        path: str,
        payload: dict[str, Any],
        reason: str,
    ) -> str:
        authoritative = self._commit_record(self.journal, path, payload, reason)
        mirrored = self._commit_record(self.control, path, payload, reason)
        if mirrored != authoritative:
            raise PlannerReceiptError("planner transport authority and mirror bytes disagree")
        return authoritative

    def _telemetry(self, completion: Any) -> PlannerTelemetry:
        actual_model = getattr(completion, "model", None)
        provider = getattr(completion, "provider", None)
        actual_model = (
            actual_model if isinstance(actual_model, str) and actual_model.strip() else None
        )
        provider = provider if isinstance(provider, str) and provider.strip() else None
        try:
            usage = PlannerUsage.from_provider(getattr(completion, "usage", None))
        except (PricingError, TypeError, ValueError):
            return PlannerTelemetry.unknown(
                source="invalid_completion_usage",
                requested_model=self.model,
                model=actual_model,
                provider=provider,
            )
        if actual_model is None or provider is None:
            return PlannerTelemetry.unknown(
                source="invalid_completion_identity",
                requested_model=self.model,
                model=actual_model,
                provider=provider,
                usage=usage,
            )
        try:
            price = ensure_priced(actual_model)
            if provider != price.provider:
                return PlannerTelemetry.unknown(
                    source="provider_model_mismatch",
                    requested_model=self.model,
                    model=actual_model,
                    provider=provider,
                    usage=usage,
                )
            billed, work = call_cost(
                actual_model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                reasoning_tokens=usage.reasoning_tokens,
            )
        except (PricingError, TypeError, ValueError):
            return PlannerTelemetry.unknown(
                source="unpriceable_completion",
                requested_model=self.model,
                model=actual_model,
                provider=provider,
                usage=usage,
            )
        return PlannerTelemetry(
            source="llm_completion",
            cost_known=True,
            requested_model=self.model,
            model=actual_model,
            provider=provider,
            usage=usage,
            billed_usd=billed,
            work_usd=work,
            pricing_table_sha=table_sha(),
            pricing_as_of=price.as_of,
        )

    def _validate_completion(self, completion: Any) -> PlannerCompletion:
        telemetry = self._telemetry(completion)
        text_blocks = getattr(completion, "text_blocks", None)
        response = (
            "\n".join(text_blocks)
            if isinstance(text_blocks, tuple) and all(isinstance(item, str) for item in text_blocks)
            else None
        )
        if not telemetry.cost_known:
            raise PlannerCompletionError(
                "planner completion has unauditable usage or model identity",
                telemetry=telemetry,
                response=response,
            )
        if telemetry.model != self.model:
            raise PlannerCompletionError(
                f"planner provider returned model {telemetry.model!r}, expected {self.model!r}",
                telemetry=telemetry,
                response=response,
            )
        tool_calls = getattr(completion, "tool_calls", None)
        if not isinstance(tool_calls, tuple) or tool_calls:
            raise PlannerCompletionError(
                "planner completion must contain zero tool calls",
                telemetry=telemetry,
                response=response,
            )
        stop_reason = getattr(completion, "stop_reason", None)
        if stop_reason != "end_turn":
            raise PlannerCompletionError(
                f"planner completion did not finish (stop_reason={stop_reason!r})",
                telemetry=telemetry,
                response=response,
            )
        if response is None or not response.strip():
            raise PlannerCompletionError(
                "planner completion contains no non-empty text response",
                telemetry=telemetry,
                response=response,
            )
        return PlannerCompletion(text=response, telemetry=telemetry)

    def complete(self, *, request_id: str, system: str, prompt: str) -> PlannerCompletion:
        request_id = _text(request_id, "planner request_id")
        system = _text(system, "planner system prompt")
        prompt = _text(prompt, "planner prompt")
        with self.mutation_lock, self._locked():
            binding = self._binding(system=system, prompt=prompt)
            evidence = self._paired_evidence(request_id)
            if evidence is not None:
                # A budget reservation authorizes a new effect, not replay.
                self._bound_remaining_usd = None
                if not self._scope_is_current(evidence.scope):
                    raise PlannerReceiptError(
                        "planner receipt authority is orphaned from current central attempt"
                    )
                self._mirror_authority(request_id)
                evidence = self._paired_evidence(request_id)
                assert evidence is not None
                return self._replay(evidence, binding=binding)

            input_bytes = len(system.encode("utf-8")) + len(prompt.encode("utf-8"))
            if input_bytes > self.max_prompt_bytes:
                raise PlannerCompletionError(
                    f"planner input is {input_bytes} bytes; limit is {self.max_prompt_bytes} bytes",
                    telemetry=PlannerTelemetry.synthetic_zero("not_called_zero"),
                )
            scope = self._attempt_scope(request_id, prompt)
            admission = self._admission()
            # Preflight is non-billable and intentionally precedes the
            # intent. A failed credential check remains safe to retry.
            try:
                self._ensure_ready(self.model)
            except Exception as exc:
                raise PlannerCompletionError(
                    f"planner preflight failed: {type(exc).__name__}: {exc}",
                    telemetry=PlannerTelemetry.synthetic_zero("not_called_zero"),
                ) from exc
            intent_path = planner_transport_intent_path(request_id)
            intent_raw = self._commit_authoritative_pair(
                intent_path,
                self._intent_record(request_id, binding, admission, scope),
                f"planner transport intent: {request_id}",
            )

            dispatched = False
            try:
                call_options: dict[str, Any] = {}
                if self._deadline is not None:
                    remaining = self._deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("planner deadline elapsed before provider dispatch")
                    call_options["timeout_seconds"] = remaining
                dispatched = True
                completion = self.llm.call(
                    model=self.model,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                    tools=None,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    role="planner",
                    **call_options,
                )
            except Exception as exc:
                telemetry = (
                    PlannerTelemetry.unknown(source="provider_exception", requested_model=self.model)
                    if dispatched else PlannerTelemetry.not_dispatched(self.model)
                )
                detail = f"{type(exc).__name__}: {exc}"
                outcome = self._terminal_receipt(
                    request_id,
                    intent_raw,
                    status="ambiguous" if dispatched else "invalid",
                    telemetry=telemetry,
                    response=None,
                    error=detail,
                )
                try:
                    self._commit_authoritative_pair(
                        planner_transport_outcome_path(request_id),
                        outcome,
                        f"planner transport outcome: {request_id}",
                    )
                except Exception as persist_error:
                    raise PlannerCompletionError(
                        f"planner provider exception receipt failed: {persist_error}",
                        telemetry=telemetry,
                    ) from persist_error
                raise PlannerCompletionError(detail, telemetry=telemetry) from exc

            try:
                result = self._validate_completion(completion)
            except PlannerCompletionError as exc:
                outcome = self._terminal_receipt(
                    request_id,
                    intent_raw,
                    status="invalid",
                    telemetry=exc.telemetry,
                    response=exc.response,
                    error=str(exc),
                )
                try:
                    self._commit_authoritative_pair(
                        planner_transport_outcome_path(request_id),
                        outcome,
                        f"planner transport outcome: {request_id}",
                    )
                except Exception as persist_error:
                    raise PlannerCompletionError(
                        f"planner invalid-completion receipt failed: {persist_error}",
                        telemetry=exc.telemetry,
                        response=exc.response,
                    ) from persist_error
                raise

            outcome = self._terminal_receipt(
                request_id,
                intent_raw,
                status="completed",
                telemetry=result.telemetry,
                response=result.text,
                error=None,
            )
            try:
                self._commit_authoritative_pair(
                    planner_transport_outcome_path(request_id),
                    outcome,
                    f"planner transport outcome: {request_id}",
                )
            except Exception as exc:
                raise PlannerCompletionError(
                    f"planner completion receipt failed: {exc}",
                    telemetry=result.telemetry,
                    response=result.text,
                ) from exc
            return result
