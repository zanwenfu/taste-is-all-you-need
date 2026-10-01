"""Bind worker feedback acceptance to exact, durable Responses submissions.

Observing an inbox entry is not acceptance. Only a complete typed model claim,
verified against the inputs of its saved provider receipt, advances cursors.
Claim receipts precede cursor changes so interrupted acceptance can be replayed.
Memory rollback re-exposes current-generation history whose context was lost.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from taste.brains.communication import AcceptanceReceipt, Communicator, GenerationStatus, Message
from taste.brains.records import Assignment
from taste.brains.responses_conversation import CompletedResponsesTurn, ResponsesConversation
from taste.brains.worker_protocol import ContractMismatch, assignment_run_id

_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _loads(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate field")
            result[key] = value
        return result

    def invalid(_value):
        raise ValueError("non-finite constant")

    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


class InvalidWorkerClaim(ContractMismatch):
    """A reply that is not an acceptable claim, for a reason the worker can correct.

    Its message is written for the worker: what was wrong and what is valid.
    Integrity failures (context that changed identity) stay ContractMismatch.
    """


class TruncatedWorkerReply(InvalidWorkerClaim):
    """The model's reply reached its output limit before it ended; nothing in it ran."""


@dataclass(frozen=True)
class WorkerClaim:
    status: str
    summary: str
    evidence: tuple[str, ...]
    accepted_inbox_ids: tuple[str, ...]
    accepted_verdicts: Mapping[str, int]


def parse_worker_claim(turn: CompletedResponsesTurn) -> WorkerClaim:
    reply = turn.completion
    if reply.stop_reason == "max_tokens" and not reply.tool_calls:
        raise TruncatedWorkerReply("it reached the output limit before it ended")
    if reply.stop_reason != "end_turn" or reply.tool_calls:
        raise ContractMismatch("worker claim requires a complete, tool-free model reply")
    text = reply.summary_text
    if len(text.encode()) > 65_536:
        raise InvalidWorkerClaim("the reply is longer than 64 KiB")
    try:
        try:
            raw = _loads(text)
        except ValueError as exc:
            raise ValueError(f"it is not one JSON object ({exc})") from exc
        required = {"status", "summary", "evidence", "accepted_inbox_ids", "accepted_verdicts"}
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("it must be one JSON object with exactly the fields " + ", ".join(sorted(required)))
        if raw["status"] not in {"completed", "blocked", "continue"} or not isinstance(raw["summary"], str):
            raise ValueError("status must be completed, blocked or continue, and summary a string")
        evidence, inbox, verdicts = raw["evidence"], raw["accepted_inbox_ids"], raw["accepted_verdicts"]
        if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
            raise ValueError("evidence must be an array of strings")
        if raw["status"] == "completed" and not any(item.strip() for item in evidence):
            raise ValueError("a completed claim needs at least one concrete evidence string")
        if (not isinstance(inbox, list) or not all(isinstance(item, str) and _OBJECT_ID.fullmatch(item) for item in inbox)
                or len(inbox) != len(set(inbox))):
            raise ValueError("accepted_inbox_ids must be distinct inbox ids copied from your inputs")
        if not isinstance(verdicts, dict):
            raise ValueError("accepted_verdicts must be an object mapping state ids to counts")
        # An entry that acknowledges zero verdicts acknowledges nothing. Measured
        # live, one such stray entry cost a finished worker its whole run.
        verdicts = {key: count for key, count in verdicts.items() if not (type(count) is int and count == 0)}
        if not all(isinstance(key, str) and _OBJECT_ID.fullmatch(key) and type(count) is int and count > 0
                   for key, count in verdicts.items()):
            raise ValueError("accepted_verdicts keys must be state ids copied from a verdicts input, "
                             "each with a positive whole count")
    except (ValueError, TypeError) as exc:
        raise InvalidWorkerClaim(str(exc)) from exc
    return WorkerClaim(raw["status"], raw["summary"], tuple(evidence), tuple(inbox), MappingProxyType(verdicts))


class ResponsesFeedback:
    def __init__(self, conversation: ResponsesConversation, assignment: Assignment):
        self.conversation, self.assignment = conversation, assignment
        self.branch = conversation.branch
        self.communicator = Communicator(self.branch.store)
        if (assignment.worker != self.branch.name or assignment.model != conversation.session.binding.model
                or conversation.session.binding.role != "worker"
                or assignment_run_id(assignment) != conversation.session.binding.run_id):
            raise ContractMismatch("feedback is not bound to the exact worker assignment")

    @staticmethod
    def _envelope(item):
        return {"kind": "inbox", "inbox_id": item.inbox_id,
                "received_at": item.received_at, "message": item.message.to_dict()}

    def _events(self):
        with self.branch._mutation_lock:
            return list(self.branch.head.transcript.turns) + self.branch.view.pending_turns()

    def _validate_claim(self, turn):
        claim = parse_worker_claim(turn)
        delivered = {}
        verdicts = {}
        for identifier, content in turn.submitted_inputs.items():
            if identifier.startswith("inbox."):
                value = _loads(content)
                item = Message.from_dict(value["message"])
                if (identifier != "inbox." + value["inbox_id"] or value["kind"] != "inbox"
                        or item.recipient != self.assignment.worker or item.generation != self.assignment.generation):
                    raise ContractMismatch("submitted inbox context changed identity or generation")
                delivered[value["inbox_id"]] = value
            elif identifier.startswith("verdicts."):
                value = _loads(content)
                if identifier != "verdicts." + _digest(value) or value["kind"] != "verdicts":
                    raise ContractMismatch("submitted verdict context changed identity")
                for row in value["states"]:
                    count = len(row["verdicts"])
                    if row["through"] != count:
                        raise ContractMismatch("verdict count differs from its submitted evidence")
                    verdicts[row["state_id"]] = max(verdicts.get(row["state_id"], 0), count)
        unknown = sorted(set(claim.accepted_inbox_ids) - set(delivered))
        if unknown:
            raise InvalidWorkerClaim(
                "accepted_inbox_ids names messages that were not in your inputs: " + ", ".join(unknown)
                + ". The inbox ids you were given: " + (", ".join(sorted(delivered)) or "none"))
        excess = {state: count for state, count in claim.accepted_verdicts.items()
                  if count > verdicts.get(state, 0)}
        if excess:
            raise InvalidWorkerClaim(
                "accepted_verdicts acknowledges more than you were shown: " + _json(excess)
                + ". The most you may acknowledge: " + _json(verdicts))
        return claim, delivered

    def _accepted(self):
        inbox, verdicts = {}, {}
        for event in self._events():
            if event.get("kind") != "worker_claim":
                continue
            if set(event) != {"kind", "request_id"}:
                raise ContractMismatch("worker claim receipt has unexpected fields")
            turn = self.conversation.completed_turn(event["request_id"])
            claim, delivered = self._validate_claim(turn)
            for identifier in claim.accepted_inbox_ids:
                evidence = {"request_id": turn.request_id, "envelope": delivered[identifier]}
                prior = inbox.get(identifier)
                if prior is not None and prior["envelope"] != evidence["envelope"]:
                    raise ContractMismatch("accepted inbox message changed its exact bytes")
                inbox.setdefault(identifier, evidence)
            for state, count in claim.accepted_verdicts.items():
                verdicts[state] = max(verdicts.get(state, 0), count)
        return inbox, verdicts

    def _boundary(self, item, disposition, generation, *, proof=None):
        record = {
            "kind": "inbox_accepted" if disposition == "accepted" else "inbox_retired_stale",
            "message_id": item.inbox_id, "semantic_message_id": item.message.message_id,
            "message_digest": _digest(item.message.to_dict()), "disposition": disposition,
            "current_generation": generation, "message_generation": item.message.generation,
            "request_id": proof,
        }
        existing = [event for event in self._events() if event.get("kind") == record["kind"]
                    and event.get("message_id") == item.inbox_id]
        if any(event != record for event in existing):
            raise ContractMismatch("durable inbox acceptance changed its binding")
        if not existing:
            self.branch.turn(**record)
        return AcceptanceReceipt.for_message(item, disposition=disposition, current_generation=generation,
                                             durable_ref="worker-turn." + _digest(record))

    def reconcile(self):
        """Complete a durable accepted prefix, never jump over an unaccepted item."""
        accepted, verdicts = self._accepted()
        for item in self.communicator.pending(self.assignment.worker):
            status = self.communicator.generation_status(item.message, self.assignment.generation)
            if status is GenerationStatus.STALE:
                self.communicator.retire_stale(item, current_generation=self.assignment.generation,
                                              boundary=self._boundary)
            elif status is GenerationStatus.CURRENT and item.inbox_id in accepted:
                evidence = accepted[item.inbox_id]
                if evidence["envelope"] != self._envelope(item):
                    raise ContractMismatch("acknowledged inbox differs from its delivered bytes")
                self.communicator.accept(
                    item, current_generation=self.assignment.generation,
                    boundary=lambda entry, disposition, generation, proof=evidence["request_id"]: self._boundary(
                        entry, disposition, generation, proof=proof),
                )
            else:
                break
        # Verdicts beyond the lookback window remain outside this operation;
        # only visible prefixes can be acknowledged through Branch's API.
        visible = self.branch.verdict_watermark()
        through = {state: count for state, count in verdicts.items() if state in visible}
        if through:
            self.branch.acknowledge(through=through)

    def observe_pending(self) -> tuple[str, ...]:
        self.reconcile()
        pending_ids = {item.inbox_id for item in self.communicator.pending(self.assignment.worker)}
        inputs = []
        pending_current = False
        for item in self.communicator.history(self.assignment.worker):
            status = self.communicator.generation_status(item.message, self.assignment.generation)
            if status is GenerationStatus.FUTURE:
                break
            if status is GenerationStatus.STALE:
                if item.inbox_id in pending_ids and pending_current:
                    break
                continue
            pending_current |= item.inbox_id in pending_ids
            identifier = "inbox." + item.inbox_id
            # History also re-exposes messages accepted before a memory
            # rollback. A persistent inbox cursor must not hide lost context.
            self.conversation.observe(identifier, _json(self._envelope(item)))
            inputs.append(identifier)
        with self.branch._mutation_lock:
            # Read content and counts from one snapshot, so a racing verdict
            # cannot be acknowledged merely because a later count included it.
            snapshot = self.branch._recent_verdicts()
            states = [{"state_id": state, "through": len(rows), "verdicts": [v.to_dict() for v in rows]}
                      for state, rows in snapshot]
        if states:
            payload = {"kind": "verdicts", "states": states}
            identifier = "verdicts." + _digest(payload)
            self.conversation.observe(identifier, _json(payload))
            inputs.append(identifier)
        return tuple(inputs)

    def accept_latest(self) -> WorkerClaim:
        turn = self.conversation.completed_turn()
        claim, _ = self._validate_claim(turn)
        record = {"kind": "worker_claim", "request_id": turn.request_id}
        if record not in self._events():
            self.branch.turn(**record)
        self.reconcile()
        return claim
