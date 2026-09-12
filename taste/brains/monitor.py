"""The monitor brain: watches a sub-brain and says what it sees.

One monitor per sub-brain, created with it, never separately -- a worker
without its monitor is not a smaller version of the system, it is a different
thing. Both are built from the same :class:`~taste.brains.contract.Contract`,
so the thing doing the work and the thing grading it read identical criteria.

**It observes. It does not act.** No nudge, no interrupt, no demotion, at any
severity. This is the correction that matters most in this file, because the
opposite was tried and measured: with a lever, the monitor interrupted its
worker into failure and then judged it for having been interrupted. One live
run produced six ``ResultMessage``s, every one ``is_error`` with
``terminal_reason`` ``aborted_*``, no ``end_turn``, no completion -- and a
verdict reading "The worker run was interrupted by the user before any work
could be completed", about an interruption the monitor itself had caused.
Twenty-six verdicts, six interrupts, zero completed turns.

A colocated observer is the wrong place for that authority. It sees a partial
view, seconds old, of work still in flight, and it cannot see the plan the work
belongs to or the other workers sharing it. The central brain can see all of
that -- it created this worker and this monitor together, from one contract --
so the central brain decides and this reports.

**Reporting is not a weaker lever; it is a different one.** A verdict is
written to the exact observed state with ``store.judge``, which needs no lease.
It reaches the worker on its next wake through ``Resume.unacked``, and the
central brain through ``report()`` in the ``WorkerReport``. What was removed is
only the mid-flight shove.

**It lives in the sub-brain's process.** Still true, and still not a style
choice: a second process that resumed a live session id and called
``interrupt()`` got success back with the right session id while the worker
ignored it and finished all eight writes. Cross-process control fails
*silently*. That measurement is why control was colocated; it is also, read
again, an early sign that control belonged somewhere else entirely.

**It judges off the memstore tail, never from a hook.** A ``PreToolUse`` hook
runs in the sub-brain's own event loop and the agent awaits it, so cost is
added 1:1 to every matched call against a ~20 ms budget; one model verdict costs
a median 2.18 s, about 100x that. Worse, ``HookMatcher(timeout=)`` fails closed,
so a slow judge becomes a tool outage rather than a slow tool.

**So it cannot veto a single call, and does not pretend to.** Interception is a
race governed by ``slack = tool duration - decision latency``: against an
instant tool a monitor deliberating >=10 ms leaked the side effect 9 times out
of 9. Prevention belongs to the jail and the sandbox.

Severity says how confident the observation is, not what to do about it:

===========  ==========================================================
 fine        nothing in the evidence contradicts the contract
 drifting    something looks off, but the work is still plausibly on track
 wrong       there is positive evidence the work is off-contract
 lost        the evidence says continuing this plan will not get there
===========  ==========================================================
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from taste.brains.contract import CONTRACT_PATH, Contract
from taste.brains.records import Assignment, contract_digest
from taste.memstore import Store, Verdict

__all__ = [
    "BATCH_SIZE",
    "UNJUDGED_MESSAGE_TYPES",
    "Judgement",
    "MonitorAction",
    "MonitorBrain",
    "Severity",
    "TerminalAssessment",
    "TerminalDecision",
]

BATCH_SIZE = 10
"""How many events a monitor judges per model call.

Batched because a persistent per-event monitor re-reads its whole transcript
each turn, so its cost climbs with the event count. Judging N events in one
stateless call was measured 10.6x cheaper and 2.9x faster than one call per
event, and it stays flat instead of climbing.
"""

UNJUDGED_MESSAGE_TYPES = frozenset({"StreamEvent"})
"""SDK message types the monitor does not judge, and why there are any.

A ``StreamEvent`` is one token-level delta of a message still being written:
``{"delta": {"partial_json": "-------"}}`` is a representative one, in full.
It is the transport, not the work. The same turn journal carries the finished
``AssistantMessage`` right after it, with the entire tool call and its input.

Measured on one live run: 2,277 recorded events, of which 2,160 were
``StreamEvent``. At ``BATCH_SIZE`` that is 228 model calls -- about sixteen
minutes of judging for seventy-five seconds of work, and the worker cannot
exit until its terminal drain finishes. Excluding them leaves 117 events,
twelve batches, under a minute.

This narrows what the *monitor reads*, never what the branch records. The
durable transcript keeps every message, because "nothing is lost" is a claim
of the layer beneath this one and is not ours to trade for a faster judge.

It must stay a constant rather than a parameter. ``_cursor`` compares content
fingerprints, so a monitor that filtered on one run and not on its restart
would see its saved prefix rewind and re-judge work it had already judged.
"""

_STATE_SCHEMA = "taste.brains/MonitorState/3"
_LEGACY_STATE_SCHEMAS = {None, "taste.brains/MonitorState/2"}


class Severity(StrEnum):
    """How wrong things are, which is what chooses the response.

    Ordered, so a monitor can compare. The distinction that matters is between
    ``WRONG`` -- the brain can still fix this itself, given the feedback -- and
    ``LOST``, where more of the same only spends budget.
    """

    FINE = "fine"
    DRIFTING = "drifting"
    WRONG = "wrong"
    LOST = "lost"

    @property
    def rank(self) -> int:
        return ["fine", "drifting", "wrong", "lost"].index(self.value)


@dataclass(frozen=True)
class Judgement:
    """One monitor verdict on a stretch of a sub-brain's work."""

    severity: Severity
    reason: str
    evidence: tuple[str, ...] = ()
    suggestion: str = ""
    # Optional audit fields are excluded from equality deliberately.  A
    # restarted monitor must still match a caller's logical judgement to the
    # durable pending action even when the caller does not have the original
    # model telemetry in hand. ``cost_usd`` is the billed price of this exact
    # judgement call, not a running monitor total.
    model: str | None = field(default=None, compare=False)
    raw_response: str | None = field(default=None, compare=False, repr=False)
    cost_usd: float | None = field(default=None, compare=False)

    @property
    def acceptable(self) -> bool:
        return self.severity is Severity.FINE

    def to_verdict(self, by: str) -> Verdict:
        """As a memstore verdict, which is how it reaches the worker.

        ``fail`` covers WRONG and LOST; DRIFTING is ``unknown`` because it is a
        warning rather than a judgement that the work is bad, and a monitor
        that cries failure at every wobble is one a brain learns to ignore.
        """
        status = {
            Severity.FINE: "pass",
            Severity.DRIFTING: "unknown",
            Severity.WRONG: "fail",
            Severity.LOST: "fail",
        }[self.severity]
        return Verdict(
            status=status,  # type: ignore[arg-type]
            by=by,
            detail=self.reason,
            failure_class=self.severity.value,
        )

    def as_feedback(self) -> str:
        """What the worker is told, in the second person."""
        parts = [self.reason]
        if self.evidence:
            parts.append("Specifically: " + "; ".join(self.evidence))
        if self.suggestion:
            parts.append(self.suggestion)
        return " ".join(parts)


def _judgement_dict(judgement: Judgement) -> dict[str, Any]:
    return {
        "severity": judgement.severity.value,
        "reason": judgement.reason,
        "evidence": list(judgement.evidence),
        "suggestion": judgement.suggestion,
        "model": judgement.model,
        "raw_response": judgement.raw_response,
        "cost_usd": judgement.cost_usd,
    }


def _judgement_from_dict(raw: dict[str, Any]) -> Judgement:
    return Judgement(
        severity=Severity(raw["severity"]),
        reason=raw.get("reason", ""),
        evidence=tuple(raw.get("evidence", ())),
        suggestion=raw.get("suggestion", ""),
        model=raw.get("model"),
        raw_response=raw.get("raw_response"),
        cost_usd=raw.get("cost_usd"),
    )


@dataclass(frozen=True)
class TerminalDecision:
    """The judge's answer about one immutable terminal work state.

    Historical findings must be partitioned explicitly. A model cannot make a
    prior WRONG or LOST judgement disappear merely by returning ``fine`` for a
    later state; it has to say which exact finding ids the state resolves and
    which remain open.
    """

    judgement: Judgement
    resolved_finding_ids: tuple[str, ...] = ()
    unresolved_finding_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.judgement, Judgement):
            raise TypeError("terminal decision judgement must be a Judgement")
        for name, values in (
            ("resolved_finding_ids", self.resolved_finding_ids),
            ("unresolved_finding_ids", self.unresolved_finding_ids),
        ):
            if not isinstance(values, tuple) or not all(
                isinstance(value, str) and value for value in values
            ):
                raise TypeError(f"terminal decision {name} must be a tuple of ids")
            if len(set(values)) != len(values):
                raise ValueError(f"terminal decision {name} contains duplicate ids")
        overlap = set(self.resolved_finding_ids) & set(self.unresolved_finding_ids)
        if overlap:
            raise ValueError(f"terminal decision resolves and leaves open the same ids: {overlap}")


def _terminal_assessment_id(
    *,
    state_id: str,
    contract_digest_value: str,
    context_digest: str,
    finding_ids: tuple[str, ...],
    decision: TerminalDecision,
    failure: str,
) -> str:
    payload = json.dumps(
        {
            "state_id": state_id,
            "contract_digest": contract_digest_value,
            "context_digest": context_digest,
            "finding_ids": list(finding_ids),
            "judgement": _judgement_dict(decision.judgement),
            "resolved_finding_ids": list(decision.resolved_finding_ids),
            "unresolved_finding_ids": list(decision.unresolved_finding_ids),
            "failure": failure,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8", "surrogateescape")
    return hashlib.sha256(payload).hexdigest()


def _terminal_context_digest(context_json: str) -> str:
    return "sha256:" + hashlib.sha256(
        context_json.encode("utf-8", "surrogateescape")
    ).hexdigest()


@dataclass(frozen=True)
class TerminalAssessment:
    """A durable current assessment pinned to one immutable work State."""

    id: str
    state_id: str
    contract_digest: str
    context_json: str
    context_digest: str
    finding_ids: tuple[str, ...]
    decision: TerminalDecision
    assessed_at: str
    failure: str = ""

    @property
    def judgement(self) -> Judgement:
        return self.decision.judgement

    @property
    def resolved_finding_ids(self) -> tuple[str, ...]:
        return self.decision.resolved_finding_ids

    @property
    def unresolved_finding_ids(self) -> tuple[str, ...]:
        return self.decision.unresolved_finding_ids

    @property
    def acceptable(self) -> bool:
        return bool(
            not self.failure
            and self.judgement.severity is Severity.FINE
            and not self.unresolved_finding_ids
            and set(self.resolved_finding_ids) == set(self.finding_ids)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state_id": self.state_id,
            "contract_digest": self.contract_digest,
            "context_json": self.context_json,
            "context_digest": self.context_digest,
            "finding_ids": list(self.finding_ids),
            "judgement": _judgement_dict(self.judgement),
            "resolved_finding_ids": list(self.resolved_finding_ids),
            "unresolved_finding_ids": list(self.unresolved_finding_ids),
            "assessed_at": self.assessed_at,
            "failure": self.failure,
            "acceptable": self.acceptable,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TerminalAssessment:
        try:
            context_json = raw["context_json"]
            if not isinstance(context_json, str) or not isinstance(json.loads(context_json), dict):
                raise ValueError("terminal assessment context_json must encode an object")
            decision = TerminalDecision(
                judgement=_judgement_from_dict(raw["judgement"]),
                resolved_finding_ids=tuple(raw["resolved_finding_ids"]),
                unresolved_finding_ids=tuple(raw["unresolved_finding_ids"]),
            )
            assessment = cls(
                id=raw["id"],
                state_id=raw["state_id"],
                contract_digest=raw["contract_digest"],
                context_json=context_json,
                context_digest=raw["context_digest"],
                finding_ids=tuple(raw["finding_ids"]),
                decision=decision,
                assessed_at=raw["assessed_at"],
                failure=raw.get("failure", ""),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid terminal assessment in monitor state") from exc
        wanted = _terminal_assessment_id(
            state_id=assessment.state_id,
            contract_digest_value=assessment.contract_digest,
            context_digest=assessment.context_digest,
            finding_ids=assessment.finding_ids,
            decision=assessment.decision,
            failure=assessment.failure,
        )
        if assessment.id != wanted:
            raise ValueError("terminal assessment id does not match its contents")
        if assessment.context_digest != _terminal_context_digest(assessment.context_json):
            raise ValueError("terminal assessment context digest does not match its contents")
        if not all(
            isinstance(value, str) and value
            for value in (
                assessment.id,
                assessment.state_id,
                assessment.contract_digest,
                assessment.context_digest,
                assessment.assessed_at,
            )
        ) or not isinstance(assessment.failure, str):
            raise ValueError("terminal assessment contains invalid scalar fields")
        if not all(isinstance(value, str) and value for value in assessment.finding_ids):
            raise ValueError("terminal assessment finding ids are invalid")
        if len(set(assessment.finding_ids)) != len(assessment.finding_ids):
            raise ValueError("terminal assessment finding ids contain duplicates")
        if (
            set(assessment.resolved_finding_ids)
            | set(assessment.unresolved_finding_ids)
        ) != set(assessment.finding_ids):
            raise ValueError("terminal assessment does not partition its finding ids")
        recorded_acceptable = raw.get("acceptable")
        if recorded_acceptable is not None and recorded_acceptable is not assessment.acceptable:
            raise ValueError("terminal assessment acceptable flag does not match its contents")
        return assessment


def _fingerprint(event: dict[str, Any]) -> str:
    """A stable identity for one observed event, across process restarts."""
    payload = json.dumps(
        event,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8", "surrogateescape")
    return hashlib.sha256(payload).hexdigest()


def _common_prefix(left: list[str], right: tuple[str, ...]) -> int:
    through = 0
    for ours, current in zip(left, right, strict=False):
        if ours != current:
            break
        through += 1
    return through


@dataclass(frozen=True)
class MonitorAction:
    """A judgement durably waiting to be delivered to its worker.

    ``observed_head`` and ``batch`` pin the evidence.  A judge can take seconds;
    attaching its verdict to whatever head happens to be current afterwards
    would make a true judgement about one trajectory look like a judgement of
    another.  Completed actions remain in ``MonitorState.actions`` as audit
    history; ``pending_action_ids`` is the replay queue.
    """

    id: str
    observed_head: str
    batch: tuple[dict[str, Any], ...]
    fingerprints: tuple[str, ...]
    judgement: Judgement
    verdict_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "observed_head": self.observed_head,
            "batch": list(self.batch),
            "fingerprints": list(self.fingerprints),
            "judgement": _judgement_dict(self.judgement),
            "verdict_at": self.verdict_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MonitorAction:
        return cls(
            id=raw["id"],
            observed_head=raw["observed_head"],
            batch=tuple(raw.get("batch", ())),
            fingerprints=tuple(raw.get("fingerprints", ())),
            judgement=_judgement_from_dict(raw["judgement"]),
            verdict_at=raw["verdict_at"],
        )


@dataclass(frozen=True)
class _Observed:
    """One coherent read of a branch head plus the journal extending it."""

    view: Any
    head: Any
    events: tuple[dict[str, Any], ...]
    fingerprints: tuple[str, ...]


class _PinnedView:
    """The ordinary BranchView with the observed head held still for a judge."""

    def __init__(self, observed: _Observed) -> None:
        self._live = observed.view
        self.head = observed.head
        self.observed_events = observed.events

    def exists(self) -> bool:
        return True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._live, name)


@dataclass
class MonitorState:
    """What the monitor has seen, so it can be resumed like anything else."""

    judged_through: int = 0
    """Compatibility count for readers of the original sidecar schema.

    Correctness rests on ``fingerprints``, not this integer.  A cumulative
    cursor cannot distinguish an extending trajectory from a rollback to a
    shorter prefix followed by different work: it points past the replacement
    events and silently skips them.
    """
    fingerprints: list[str] = field(default_factory=list)
    """The content-identified prefix of the current trajectory already judged."""
    judgements: list[Judgement] = field(default_factory=list)
    actions: list[MonitorAction] = field(default_factory=list)
    pending_action_ids: list[str] = field(default_factory=list)
    interventions: list[tuple[str, str]] = field(default_factory=list)
    terminal_assessments: list[TerminalAssessment] = field(default_factory=list)

    @property
    def worst(self) -> Severity:
        if not self.judgements:
            return Severity.FINE
        return max((j.severity for j in self.judgements), key=lambda s: s.rank)


class MonitorBrain:
    """Watches one sub-brain: reads its events, judges, and escalates.

    ``judge`` is injected rather than hardcoded so the loop can be tested
    without a model. A real monitor passes an LLM-backed judge; the mechanics
    of batching, escalation and recording are the same either way.
    """

    def __init__(
        self,
        store: Store,
        contract: Contract,
        judge: Any,
        *,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.store = store
        self.contract = contract
        self.identity = f"monitor/{contract.identity}"
        self.contract_digest = contract_digest(contract)
        self._judge = judge
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("monitor batch_size must be a positive integer")
        self.batch_size = batch_size
        self._state_lock = threading.RLock()
        self._state_needs_upgrade = False
        self.state = self._load_state()
        if self._state_needs_upgrade:
            self._save_state()

    def _state_path(self):
        digest_hex = self.contract_digest.removeprefix("sha256:")
        return self.store.sidecar(
            "monitor",
            self.contract.identity,
            f".contract-{digest_hex}",
        )

    def _legacy_state_path(self):
        """The identity-only location used before contract-scoped state."""
        return self.store.sidecar("monitor", self.contract.identity)

    def _durable_contract_digest(self) -> str | None:
        """Identify an unscoped legacy sidecar without guessing its owner.

        Old sidecars did not contain their Contract.  The checkpointed worker
        brief is the only durable evidence that can safely assign one to a
        digest; if that evidence is absent or differs, leave the legacy file in
        place for an auditor instead of contaminating a new contract.
        """
        view = self.store.view(self.contract.identity)
        if not view.exists():
            return None
        raw = view.read(CONTRACT_PATH)
        if raw is None:
            return None
        try:
            return contract_digest(Contract.from_json(raw))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _state_from_raw(self, raw: dict[str, Any]) -> MonitorState:
        actions = [MonitorAction.from_dict(a) for a in raw.get("actions", ())]
        pending_action_ids = list(raw.get("pending_action_ids", ()))
        known = {action.id for action in actions}
        missing = [action_id for action_id in pending_action_ids if action_id not in known]
        if missing:
            raise ValueError(f"monitor state has pending actions with no record: {missing}")
        assessments = [
            TerminalAssessment.from_dict(item)
            for item in raw.get("terminal_assessments", ())
        ]
        for assessment in assessments:
            if assessment.contract_digest != self.contract_digest:
                raise ValueError("terminal assessment has the wrong Contract digest")
        state_ids = [assessment.state_id for assessment in assessments]
        if len(set(state_ids)) != len(state_ids):
            raise ValueError("monitor state contains multiple assessments for one State")
        return MonitorState(
            judged_through=int(raw.get("judged_through", 0)),
            fingerprints=list(raw.get("fingerprints", ())),
            judgements=[_judgement_from_dict(j) for j in raw.get("judgements", ())],
            actions=actions,
            pending_action_ids=pending_action_ids,
            interventions=[tuple(i) for i in raw.get("interventions", [])],
            terminal_assessments=assessments,
        )

    def _load_state(self) -> MonitorState:
        path = self._state_path()
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            schema = raw.get("schema")
            if schema not in {*_LEGACY_STATE_SCHEMAS, _STATE_SCHEMA}:
                raise ValueError(f"unsupported monitor state schema: {schema!r}")
            recorded_digest = raw.get("contract_digest")
            if recorded_digest is not None and recorded_digest != self.contract_digest:
                raise ValueError(
                    "monitor state contract digest does not match its scoped filename"
                )
            if schema != _STATE_SCHEMA or recorded_digest is None:
                # This can occur if a process died after the atomic legacy
                # rename but before the schema-stamping rewrite below.
                self._state_needs_upgrade = True
            return self._state_from_raw(raw)

        legacy = self._legacy_state_path()
        if not legacy.exists():
            return MonitorState()
        raw = json.loads(legacy.read_text(encoding="utf-8"))
        recorded_digest = raw.get("contract_digest")
        if recorded_digest is not None and not isinstance(recorded_digest, str):
            raise ValueError("legacy monitor state contract_digest must be a string")
        owner = recorded_digest or self._durable_contract_digest()
        if owner != self.contract_digest:
            # An identity-only file cannot safely follow a revised Contract.
            # It remains readable at its historical path and is never deleted.
            return MonitorState()

        state = self._state_from_raw(raw)
        # Rename first: after this boundary even a kill cannot let another
        # contract revision claim the unscoped state.  __init__ then rewrites
        # it atomically with the explicit schema and digest.
        os.replace(legacy, path)
        self._state_needs_upgrade = True
        return state

    def _save_state(self) -> None:
        """Atomically replace the monitor's replay state.

        A monitor dies with its worker.  ``Path.write_text`` truncates the old
        file before writing the new one, so a kill in between turned a useful
        report into invalid JSON and made restart impossible.  The temporary
        file is fsynced, renamed in the same directory, then the directory is
        fsynced; readers therefore see the complete old state or the complete
        new state.
        """
        path = self._state_path()
        with self._state_lock:
            payload = json.dumps(
                {
                    "schema": _STATE_SCHEMA,
                    "contract_digest": self.contract_digest,
                    "judged_through": self.state.judged_through,
                    "fingerprints": self.state.fingerprints,
                    # Persisted because report() is what the central brain reads to
                    # decide whether to re-plan: without these a restarted monitor
                    # said worst="fine" for a worker it had just judged LOST.
                    "judgements": [_judgement_dict(j) for j in self.state.judgements],
                    "actions": [a.to_dict() for a in self.state.actions],
                    "pending_action_ids": self.state.pending_action_ids,
                    "interventions": [list(i) for i in self.state.interventions],
                    "terminal_assessments": [
                        assessment.to_dict()
                        for assessment in self.state.terminal_assessments
                    ],
                },
                sort_keys=True,
            )
            fd, temporary = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            tmp_path = os.fspath(temporary)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, path)
                tmp_path = ""
                try:
                    directory = os.open(path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                except OSError:
                    # Some filesystems do not permit directory fsync.  The file
                    # itself is still atomic and durable at the rename boundary.
                    pass
            finally:
                if tmp_path:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(tmp_path)

    @property
    def pending_actions(self) -> tuple[MonitorAction, ...]:
        """Judgements durably awaiting their response, oldest first."""
        by_id = {action.id: action for action in self.state.actions}
        return tuple(
            by_id[action_id]
            for action_id in self.state.pending_action_ids
            if action_id in by_id
        )

    # ------------------------------------------------------------------ reading

    def view(self) -> Any:
        """A lease-free window on the branch being written.

        The worker holds the write lease for the whole run, so this has to take
        nothing: a monitor that needed the lease could only watch a brain that
        had stopped, which is the opposite of the point.
        """
        return self.store.view(self.contract.identity)

    def _observation(self) -> _Observed | None:
        """Read a head and the journal extending that exact head coherently.

        ``BranchView.head`` is intentionally live.  Reading it once for the
        committed transcript and again inside ``pending_turns`` can straddle a
        checkpoint and splice two different trajectories together.  Rechecking
        the head gives us an instant at which both halves belonged together;
        anything appended just afterwards is simply input to the next tick.

        The stream is narrowed here, once, so every judging path sees the same
        events and the content cursor stays stable.  See
        :data:`UNJUDGED_MESSAGE_TYPES`.
        """
        view = self.view()
        for _ in range(8):
            if not view.exists():
                return None
            head = view.head
            committed = list(head.transcript.turns)
            pending = list(view.pending_turns())
            if view.head.id == head.id:
                events = self._judgeable(*committed, *pending)
                return _Observed(
                    view=view,
                    head=head,
                    events=events,
                    fingerprints=tuple(_fingerprint(event) for event in events),
                )
        # A worker checkpointing continuously can keep invalidating the live
        # tail read.  The latest committed head is still a coherent, conservative
        # snapshot; its new journal will be picked up by a later tick.
        head = view.head
        events = self._judgeable(*head.transcript.turns)
        return _Observed(
            view=view,
            head=head,
            events=events,
            fingerprints=tuple(_fingerprint(event) for event in events),
        )

    @staticmethod
    def _judgeable(*turns: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        """The recorded turns worth spending a model call on."""
        return tuple(
            turn
            for turn in turns
            if turn.get("message_type") not in UNJUDGED_MESSAGE_TYPES
        )

    def observations(self) -> list[dict[str, Any]]:
        """What the worker has recorded since its last state, minus the noise.

        Comes from the same journal the WAL writes, so a monitor sees tool
        intents and results as they are fsynced rather than at checkpoint
        boundaries.  Token-level stream deltas are left out -- see
        :data:`UNJUDGED_MESSAGE_TYPES`; the branch still records them, this
        monitor simply does not read them.
        """
        observed = self._observation()
        return list(observed.events) if observed is not None else []

    def unjudged(self) -> list[dict[str, Any]]:
        observed = self._observation()
        if observed is None:
            return []
        through, _ = self._cursor(observed)
        return list(observed.events[through:])

    def _cursor(self, observed: _Observed) -> tuple[int, bool]:
        """Where the judged trajectory and the current one stop agreeing.

        Returns ``(prefix_length, state_changed)``.  The latter is true for a
        legacy integer cursor being upgraded or for a divergence that rewinds
        the saved prefix.  Content, not numeric position, decides what has
        already been seen.
        """
        if not self.state.fingerprints and self.state.judged_through:
            # One-time migration from the original schema.  An integer cannot
            # prove that any event in a possibly rolled-back trajectory is the
            # event it once counted.  Re-judging is noisy; skipping replacement
            # work is silent, so the safe upgrade boundary is zero.
            return 0, True
        through = _common_prefix(self.state.fingerprints, observed.fingerprints)
        changed = through < len(self.state.fingerprints)
        return through, changed

    def worker_is_alive(self) -> bool:
        """Whether anyone still holds the branch.

        A status claim from a dead brain is a confident lie, so liveness comes
        from the lease rather than from anything the worker said about itself.
        """
        return self.view().holder is not None

    # ------------------------------------------------------------------ judging

    def _make_action(
        self,
        judgement: Judgement,
        observed: _Observed,
        batch: tuple[dict[str, Any], ...],
        fingerprints: tuple[str, ...],
    ) -> MonitorAction:
        ordinal = len(self.state.actions)
        raw = json.dumps(
            {
                "worker": self.contract.identity,
                "contract_digest": self.contract_digest,
                "head": observed.head.id,
                "fingerprints": fingerprints,
                "judgement": _judgement_dict(judgement),
                "ordinal": ordinal,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return MonitorAction(
            id=hashlib.sha256(raw).hexdigest(),
            observed_head=observed.head.id,
            batch=batch,
            fingerprints=fingerprints,
            judgement=judgement,
            # This timestamp is minted once with the durable action.  Replaying
            # after a kill uses the same value, which lets record() recognize a
            # verdict that landed just before the old process died.
            verdict_at=datetime.now(UTC).isoformat(timespec="microseconds"),
        )

    def tick(self, *, force: bool = False) -> Judgement | None:
        """Judge the next batch, or return None if there is nothing new.

        Only whole batches are judged while work continues; a partial batch is
        left for the next tick so the monitor is not billed for re-reading the
        same half-finished stretch.  ``force=True`` is the terminal-drain path:
        the runtime has stopped model input but still holds the branch lease so
        it can checkpoint the final report before releasing it.
        """
        observed = self._observation()
        if observed is None:
            return None
        old_fingerprints = list(self.state.fingerprints)
        old_through = self.state.judged_through
        through, cursor_changed = self._cursor(observed)
        if cursor_changed:
            self.state.fingerprints = list(observed.fingerprints[:through])
            self.state.judged_through = through
        pending = list(observed.events[through:])
        if not pending:
            if cursor_changed:
                try:
                    self._save_state()
                except BaseException:
                    self.state.fingerprints = old_fingerprints
                    self.state.judged_through = old_through
                    raise
            return None
        alive = self.worker_is_alive()
        if len(pending) < self.batch_size and alive and not force:
            if cursor_changed:
                try:
                    self._save_state()
                except BaseException:
                    self.state.fingerprints = old_fingerprints
                    self.state.judged_through = old_through
                    raise
            return None
        batch = tuple(pending[: self.batch_size])
        batch_fingerprints = observed.fingerprints[through : through + len(batch)]
        try:
            judgement = self._judge(self.contract, list(batch), _PinnedView(observed))
        except BaseException:
            self.state.fingerprints = old_fingerprints
            self.state.judged_through = old_through
            raise
        if not isinstance(judgement, Judgement):
            self.state.fingerprints = old_fingerprints
            self.state.judged_through = old_through
            raise TypeError(f"monitor judge returned {type(judgement).__name__}, not Judgement")

        action = self._make_action(judgement, observed, batch, batch_fingerprints)
        self.state.fingerprints = list(observed.fingerprints[: through + len(batch)])
        self.state.judged_through = len(self.state.fingerprints)
        self.state.judgements.append(judgement)
        self.state.actions.append(action)
        self.state.pending_action_ids.append(action.id)
        try:
            # Cursor, judgement and pending response become visible together.
            # A kill sees all three or none, never "judged" with no action left
            # to replay.
            self._save_state()
        except BaseException:
            self.state.pending_action_ids.pop()
            self.state.actions.pop()
            self.state.judgements.pop()
            self.state.fingerprints = old_fingerprints
            self.state.judged_through = old_through
            raise
        return judgement

    def record(self, judgement: Judgement) -> None:
        """Put the verdict where the worker will find it on its next wake.

        ``store.judge`` needs no lease, which is what lets the monitor speak
        while the worker is still writing.
        """
        for action in self.pending_actions:
            if action.judgement == judgement:
                self._record_action(action)
                return
        view = self.view()
        if view.exists():
            self.store.judge(view.head, judgement.to_verdict(self.identity))

    def _record_action(self, action: MonitorAction) -> None:
        """Record once, on the state against which the judgement was made."""
        plain = action.judgement.to_verdict(self.identity)
        verdict = Verdict(
            status=plain.status,
            by=plain.by,
            detail=plain.detail,
            failure_class=plain.failure_class,
            at=action.verdict_at,
        )
        target = self.store.state(action.observed_head)
        # A kill can land after store.judge() and before the pending action is
        # cleared.  ``verdict_at`` was minted with the action and survives that
        # kill, so replay recognizes the verdict rather than appending it twice.
        if verdict not in target.verdicts:
            self.store.judge(target, verdict)

    def _queue_direct(self, judgement: Judgement) -> MonitorAction:
        """Give the legacy ``respond(judgement)`` API the same durable boundary."""
        observed = self._observation()
        if observed is None:
            raise RuntimeError(f"cannot respond: branch {self.contract.identity!r} has no state")
        action = self._make_action(judgement, observed, (), ())
        self.state.judgements.append(judgement)
        self.state.actions.append(action)
        self.state.pending_action_ids.append(action.id)
        try:
            self._save_state()
        except BaseException:
            self.state.pending_action_ids.pop()
            self.state.actions.pop()
            self.state.judgements.pop()
            raise
        return action

    def _pending_for(self, judgement: Judgement) -> MonitorAction:
        pending = self.pending_actions
        if not pending:
            return self._queue_direct(judgement)
        for action in pending:
            if action.judgement == judgement:
                return action
        raise ValueError(
            "respond() received a judgement other than the durable pending action; "
            "use cycle() to drain monitor responses in order"
        )

    def _finish_action(
        self,
        action: MonitorAction,
        intervention: tuple[str, str] | None,
    ) -> None:
        old_pending = list(self.state.pending_action_ids)
        old_interventions = len(self.state.interventions)
        if intervention is not None:
            self.state.interventions.append(intervention)
        self.state.pending_action_ids = [
            action_id for action_id in self.state.pending_action_ids if action_id != action.id
        ]
        try:
            self._save_state()
        except BaseException:
            self.state.pending_action_ids = old_pending
            del self.state.interventions[old_interventions:]
            raise

    # ------------------------------------------------------------------ acting

    async def respond(self, judgement: Judgement, client: Any) -> str:
        """Record the judgement. Returns the observation it was filed under.

        ``client`` is accepted and deliberately unused: this monitor observes
        and never acts. See :meth:`_respond_action`.
        """
        return await self._respond_action(self._pending_for(judgement), client)

    async def _respond_action(self, action: MonitorAction, client: Any) -> str:
        """File a verdict. Touch nothing.

        **The monitor does not act on the worker, at any severity.** It used
        to: drifting nudged, wrong interrupted, lost demoted to plan mode. That
        was measured end to end and it broke the thing it was watching.

        What happened, in one live run: the worker never completed a single
        turn. Six ``ResultMessage``s, every one ``is_error`` with
        ``terminal_reason`` ``aborted_streaming`` or ``aborted_tools`` -- the
        monitor's own ``interrupt()`` landing mid-stream. No ``end_turn`` ever
        arrived, so the runtime never held a terminal candidate, so the run
        could not finish. And because an aborted run *looks* like a broken run
        to the next batch of events, the monitor then judged the worker for the
        damage it had just done. Verdict 13 of that run, verbatim: "The worker
        run was interrupted by the user before any work could be completed."

        The loop closes on itself: interrupt, aborted turn, adverse verdict,
        interrupt. Twenty-six verdicts, six interrupts, zero completions.

        So the lever is gone. A colocated observer should never have had one:
        it judges a partial view of work in flight, on evidence that is seconds
        old, and it cannot see the plan the work belongs to. The brain that can
        see all of that is the central brain, which created worker and monitor
        together from one contract. It decides; this reports.

        Nothing is lost by reporting alone. ``_record_action`` above has
        already written the verdict to the observed state with ``store.judge``,
        which needs no lease -- so it reaches the worker on its next wake
        through ``Resume.unacked``, and reaches the central brain through
        ``report()`` in the ``WorkerReport``. What is withheld is only the
        mid-flight shove.

        There is deliberately no ``final`` parameter. ``cycle`` and ``drain``
        still take one -- it decides whether a partial last batch is judged --
        but it never reached here for any reason except suppressing escalation,
        and there is no longer any to suppress.
        """
        judgement = action.judgement
        self._record_action(action)
        rung = (
            "none"
            if judgement.severity is Severity.FINE
            else f"flagged-{judgement.severity.value}"
        )
        intervention = (rung, judgement.reason) if rung != "none" else None
        self._finish_action(action, intervention)
        return rung

    async def cycle(
        self,
        client: Any,
        *,
        final: bool = False,
    ) -> tuple[Judgement | None, str | None]:
        """Replay one response or judge and respond to one new batch.

        This is the runtime API.  Pending actions always go first, closing the
        kill boundary between ``tick`` and ``respond``.  The synchronous judge
        runs in a worker thread so a multi-second model call cannot stall the
        SDK stream in the sub-brain's event loop.
        """
        pending = self.pending_actions
        if pending:
            action = pending[0]
            return action.judgement, await self._respond_action(action, client)

        judgement = await asyncio.to_thread(self.tick, force=final)
        if judgement is None:
            return None, None
        pending = self.pending_actions
        if not pending:  # pragma: no cover - tick publishes both atomically
            raise RuntimeError("monitor persisted a judgement without its pending action")
        action = pending[0]
        return action.judgement, await self._respond_action(action, client)

    async def drain(
        self,
        client: Any,
        *,
        final: bool = False,
    ) -> list[tuple[Judgement, str]]:
        """Respond to everything currently available, in durable action order.

        With ``final=True``, a partial last batch is judged even while the worker
        lease remains held.  The runtime uses that after stopping model input,
        then checkpoints its terminal report, and only then releases the lease.
        """
        drained: list[tuple[Judgement, str]] = []
        # This terminates because the monitor no longer talks to the worker.
        # Judging creates no new events, so the unjudged tail only shrinks. It
        # was not always so: when every verdict queried the worker, the answer
        # became the next batch and this loop had no reason to stop.
        while True:
            judgement, rung = await self.cycle(client, final=final)
            if judgement is None:
                return drained
            if rung is None:  # pragma: no cover - a judgement always gets a rung
                raise RuntimeError("monitor response completed without a rung")
            drained.append((judgement, rung))

    # ------------------------------------------------------ terminal boundary

    def _terminal_findings(self) -> tuple[dict[str, Any], ...]:
        """Every adverse historical finding, with a stable audit identity."""
        findings: list[dict[str, Any]] = []
        actions_align = len(self.state.actions) == len(self.state.judgements)
        for ordinal, judgement in enumerate(self.state.judgements):
            if judgement.severity is Severity.FINE:
                continue
            action = self.state.actions[ordinal] if actions_align else None
            if action is not None and action.judgement != judgement:
                action = None
            if action is not None:
                finding_id = action.id
                observed_head = action.observed_head
                fingerprints = list(action.fingerprints)
            else:
                payload = json.dumps(
                    {
                        "contract_digest": self.contract_digest,
                        "ordinal": ordinal,
                        "judgement": _judgement_dict(judgement),
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8", "surrogateescape")
                finding_id = "legacy-" + hashlib.sha256(payload).hexdigest()
                observed_head = None
                fingerprints = []
            findings.append(
                {
                    "id": finding_id,
                    "ordinal": ordinal,
                    "observed_head": observed_head,
                    "fingerprints": fingerprints,
                    "severity": judgement.severity.value,
                    "reason": judgement.reason,
                    "evidence": list(judgement.evidence),
                    "suggestion": judgement.suggestion,
                    "verdict_delivery_pending": finding_id in self.state.pending_action_ids,
                }
            )
        return tuple(findings)

    def _exact_terminal_state(self, state: Any) -> Any:
        state_id = getattr(state, "id", None)
        if not isinstance(state_id, str) or not state_id:
            raise ValueError("terminal certification requires an immutable memstore State")
        exact = self.store.state(state_id)
        if exact.meta.branch != self.contract.identity:
            raise ValueError(
                f"terminal State belongs to {exact.meta.branch!r}, not {self.contract.identity!r}"
            )
        raw_contract = exact.read(CONTRACT_PATH)
        if raw_contract is None:
            raise ValueError(f"terminal State has no durable Contract at {CONTRACT_PATH}")
        try:
            durable_contract = Contract.from_json(raw_contract)
        except Exception as exc:
            raise ValueError("terminal State has a malformed durable Contract") from exc
        if (
            durable_contract != self.contract
            or contract_digest(durable_contract) != self.contract_digest
        ):
            raise ValueError("terminal State Contract differs from the monitor Contract")

        raw_assignment = exact.read("assignment.json")
        if raw_assignment is not None:
            try:
                assignment = Assignment.from_json(raw_assignment)
            except Exception as exc:
                raise ValueError("terminal State has a malformed Assignment") from exc
            if (
                assignment.contract != self.contract
                or assignment.contract_digest != self.contract_digest
                or assignment.worker != self.contract.identity
            ):
                raise ValueError("terminal State Assignment and Contract disagree")
        return exact

    def _terminal_context(
        self,
        context: Mapping[str, Any] | None,
        findings: tuple[dict[str, Any], ...],
    ) -> tuple[str, str]:
        if context is None:
            caller: Any = {}
        elif isinstance(context, Mapping):
            caller = dict(context)
        else:
            raise TypeError("terminal context must be a JSON object")
        wrapped = {
            "terminal": caller,
            "monitor": {
                "historical_worst": self.state.worst.value,
                "judgement_count": len(self.state.judgements),
                "judged_event_fingerprints": list(self.state.fingerprints),
                "pending_action_ids": list(self.state.pending_action_ids),
                "interventions": [list(item) for item in self.state.interventions],
                "historical_finding_ids": [item["id"] for item in findings],
            },
        }
        text = json.dumps(
            wrapped,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return text, _terminal_context_digest(text)

    @staticmethod
    def _fail_closed_terminal_decision(
        finding_ids: tuple[str, ...], failure: str
    ) -> TerminalDecision:
        return TerminalDecision(
            judgement=Judgement(
                severity=Severity.LOST,
                reason="Terminal monitor certification failed closed.",
                evidence=(failure,),
                suggestion="Do not publish this work until a valid terminal assessment exists.",
            ),
            unresolved_finding_ids=finding_ids,
        )

    @staticmethod
    def _validate_terminal_decision(
        decision: TerminalDecision,
        finding_ids: tuple[str, ...],
    ) -> None:
        if not isinstance(decision, TerminalDecision):
            raise TypeError(
                f"terminal judge returned {type(decision).__name__}, not TerminalDecision"
            )
        judgement = decision.judgement
        if not isinstance(judgement.severity, Severity):
            raise TypeError("terminal judgement has an invalid severity")
        if not isinstance(judgement.reason, str) or not judgement.reason.strip():
            raise ValueError("terminal judgement reason must not be empty")
        if not isinstance(judgement.evidence, tuple) or not all(
            isinstance(item, str) and item.strip() for item in judgement.evidence
        ):
            raise TypeError("terminal judgement evidence must be a tuple of non-empty strings")
        if not isinstance(judgement.suggestion, str):
            raise TypeError("terminal judgement suggestion must be text")
        # The serialization is part of the durable assessment identity. This
        # also rejects NaN/Infinity cost values from non-production judges.
        json.dumps(_judgement_dict(judgement), allow_nan=False)
        returned = set(decision.resolved_finding_ids) | set(
            decision.unresolved_finding_ids
        )
        if returned != set(finding_ids):
            raise ValueError(
                "terminal judge did not partition the exact historical finding ids"
            )
        if judgement.severity is Severity.FINE and decision.unresolved_finding_ids:
            raise ValueError("a fine terminal decision cannot leave findings unresolved")

    def _new_terminal_assessment(
        self,
        *,
        state_id: str,
        context_json: str,
        context_digest: str,
        finding_ids: tuple[str, ...],
        decision: TerminalDecision,
        failure: str = "",
    ) -> TerminalAssessment:
        return TerminalAssessment(
            id=_terminal_assessment_id(
                state_id=state_id,
                contract_digest_value=self.contract_digest,
                context_digest=context_digest,
                finding_ids=finding_ids,
                decision=decision,
                failure=failure,
            ),
            state_id=state_id,
            contract_digest=self.contract_digest,
            context_json=context_json,
            context_digest=context_digest,
            finding_ids=finding_ids,
            decision=decision,
            assessed_at=datetime.now(UTC).isoformat(timespec="microseconds"),
            failure=failure,
        )

    def _certify_terminal_sync(
        self,
        state: Any,
        context: Mapping[str, Any] | None,
    ) -> TerminalAssessment:
        """Synchronous implementation run wholly inside one worker thread."""
        with self._state_lock:
            exact = self._exact_terminal_state(state)
            findings = self._terminal_findings()
            finding_ids = tuple(item["id"] for item in findings)

            preflight_failure = ""
            try:
                context_json, context_digest = self._terminal_context(context, findings)
            except Exception as exc:
                preflight_failure = f"invalid terminal context: {type(exc).__name__}: {exc}"[:512]
                context_json, context_digest = self._terminal_context(
                    {"invalid_terminal_context": preflight_failure}, findings
                )

            existing = next(
                (
                    assessment
                    for assessment in self.state.terminal_assessments
                    if assessment.state_id == exact.id
                ),
                None,
            )
            if existing is not None:
                if (
                    existing.context_digest != context_digest
                    or existing.context_json != context_json
                    or existing.finding_ids != finding_ids
                ):
                    raise RuntimeError(
                        "terminal State was already assessed with different context or findings"
                    )
                return existing

            # Narrowed the same way the monitor's own reading is, or this
            # comparison is between two different vocabularies and can never
            # hold: the saved prefix counts judgeable events, the state counts
            # every recorded turn including the token deltas nobody judges.
            exact_fingerprints = tuple(
                _fingerprint(event) for event in self._judgeable(*exact.transcript.turns)
            )
            if self.state.pending_action_ids:
                preflight_failure = preflight_failure or "monitor actions remain pending"
            elif tuple(self.state.fingerprints) != exact_fingerprints:
                preflight_failure = preflight_failure or (
                    "terminal State contains events that were not completely judged"
                )

            decision: TerminalDecision
            failure = preflight_failure
            if failure:
                decision = self._fail_closed_terminal_decision(finding_ids, failure)
            else:
                terminal_judge = getattr(self._judge, "judge_terminal", None)
                if not callable(terminal_judge):
                    failure = "configured monitor judge has no terminal assessment capability"
                    decision = self._fail_closed_terminal_decision(finding_ids, failure)
                else:
                    try:
                        candidate = terminal_judge(
                            self.contract,
                            exact,
                            json.loads(context_json),
                            list(findings),
                        )
                        self._validate_terminal_decision(candidate, finding_ids)
                        decision = candidate
                    except Exception as exc:
                        failure = f"terminal judge failed: {type(exc).__name__}: {exc}"[:512]
                        decision = self._fail_closed_terminal_decision(finding_ids, failure)

            assessment = self._new_terminal_assessment(
                state_id=exact.id,
                context_json=context_json,
                context_digest=context_digest,
                finding_ids=finding_ids,
                decision=decision,
                failure=failure,
            )
            self.state.terminal_assessments.append(assessment)
            try:
                self._save_state()
            except BaseException:
                self.state.terminal_assessments.pop()
                raise
            return assessment

    async def certify_terminal(
        self,
        state: Any,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> TerminalAssessment:
        """Assess one work checkpoint without blocking the SDK event loop.

        The complete judge-and-persist operation runs in one thread. Shielding
        means caller cancellation cannot cancel the awaitable while that thread
        is still publishing its answer; after a cancellation, the method waits
        for the durable old-or-new sidecar boundary before propagating it.
        """
        task = asyncio.create_task(
            asyncio.to_thread(self._certify_terminal_sync, state, context)
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await task
            raise

    # ------------------------------------------------------------------ report

    def report(self) -> dict[str, Any]:
        """What the central brain reads to decide whether to re-plan."""
        current = (
            self.state.terminal_assessments[-1]
            if self.state.terminal_assessments
            else None
        )
        call_judgements = [
            *self.state.judgements,
            *(assessment.judgement for assessment in self.state.terminal_assessments),
        ]
        costs: list[float] = []
        cost_known = True
        for judgement in call_judgements:
            value = judgement.cost_usd
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                cost_known = False
                break
            costs.append(float(value))
        cost_usd: float | None = None
        if cost_known:
            try:
                cost_usd = math.fsum(costs)
            except OverflowError:
                cost_known = False
        return {
            "worker": self.contract.identity,
            "contract_digest": self.contract_digest,
            "worst": self.state.worst.value,
            "judgements": len(self.state.judgements),
            "model_calls": len(call_judgements),
            "cost_known": cost_known,
            "cost_usd": cost_usd if cost_known else None,
            "pending_actions": [action.id for action in self.pending_actions],
            "interventions": list(self.state.interventions),
            "current": current.judgement.severity.value if current is not None else None,
            "current_state": current.state_id if current is not None else None,
            "terminal_assessment": current.to_dict() if current is not None else None,
            "alive": self.worker_is_alive(),
        }


def batch_prompt(contract: Contract, batch: list[dict[str, Any]]) -> str:
    """What an LLM judge is asked, given a batch of events.

    The criteria are quoted from the contract rather than paraphrased, so the
    worker and its monitor are held to the same words.
    """
    lines = [contract.judging_brief(), "", "Here is what it has done since you last looked:"]
    for event in batch:
        kind = event.get("kind") or event.get("role") or "event"
        body = json.dumps(
            {k: v for k, v in event.items() if k not in ("at",)}, sort_keys=True
        )
        lines.append(f"- {kind}: {body[:600]}")
    lines += [
        "",
        "Judge only what you can see. Answer with a severity and one sentence:",
        "  fine      - on track",
        "  drifting  - still recoverable, but say what to change",
        "  wrong     - it is doing the wrong thing and must be stopped",
        "  lost      - more of this will not help; the plan needs changing",
    ]
    return "\n".join(lines)
