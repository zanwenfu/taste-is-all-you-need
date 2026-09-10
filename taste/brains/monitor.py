"""The monitor brain: watches a sub-brain, judges it, and can stop it.

One monitor per sub-brain, created with it, never separately -- a worker
without its monitor is not a smaller version of the system, it is a different
thing. Both are built from the same :class:`~taste.brains.contract.Contract`,
so the thing doing the work and the thing grading it read identical criteria.

**It lives in the sub-brain's process.** Not a style choice: measured, a second
process that resumed a live session id and called ``interrupt()`` got success
back with the right session id while the worker ignored it and finished all
eight writes. Cross-process control fails *silently*, which is the worst way for
a safety mechanism to fail. Rungs (a)-(d) below need the live client object.

**It judges off the memstore tail, never from a hook.** A ``PreToolUse`` hook
runs in the sub-brain's own event loop and the agent awaits it, so cost is
added 1:1 to every matched call against a ~20 ms budget; one model verdict costs
a median 2.18 s, about 100x that. Worse, ``HookMatcher(timeout=)`` fails closed,
so a slow judge becomes a tool outage rather than a slow tool.

**So it cannot veto a single call, and does not pretend to.** Interception is a
race governed by ``slack = tool duration - decision latency``: against an
instant tool a monitor deliberating >=10 ms leaked the side effect 9 times out
of 9. Prevention belongs to the jail and the sandbox. The monitor's lever is
``interrupt()``, which stops the *run*, plus the rungs below.

The ladder, by severity, each rung measured working on a live brain:

===========  ==========================================================
 nudge       ``PermissionResultDeny(message=...)`` -- the reason reaches
             the model verbatim and it adapts instead of retrying
 correct     ``PermissionResultAllow(updated_input=...)`` -- but NOT
             silent: a model saw a rewritten path echoed back and called
             it suspicious, so path rewrites need saying out loud
 demote      ``set_permission_mode('plan')`` -- reasoning survives,
             editing stops, and it reverses cleanly
 interrupt   ``interrupt()`` -- ``terminal_reason`` is ``aborted_*``;
             the client stays usable afterwards
 redo        fork from an earlier state -- the abandoned trajectory stays
             intact and readable, which is what makes it a rollback
             rather than a deletion
===========  ==========================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from taste.brains.contract import Contract
from taste.memstore import Store, Verdict

__all__ = ["BATCH_SIZE", "Judgement", "MonitorBrain", "Severity"]

BATCH_SIZE = 10
"""How many events a monitor judges per model call.

Batched because a persistent per-event monitor re-reads its whole transcript
each turn, so its cost climbs with the event count. Judging N events in one
stateless call was measured 10.6x cheaper and 2.9x faster than one call per
event, and it stays flat instead of climbing.
"""


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


@dataclass
class MonitorState:
    """What the monitor has seen, so it can be resumed like anything else."""

    judged_through: int = 0
    """How far this monitor has read, per state.

    Per state, not one running count: the journal is keyed on the branch head,
    so a checkpoint starts a fresh one. A single index carried across that
    boundary pointed past the new journal's start and silently discarded as
    many fresh events as it had already judged -- a monitor that looked
    healthy while never seeing the work.
    """
    judgements: list[Judgement] = field(default_factory=list)
    interventions: list[tuple[str, str]] = field(default_factory=list)

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
        self._judge = judge
        self.batch_size = batch_size
        self.state = self._load_state()

    def _state_path(self):
        return self.store.sidecar("monitor", self.contract.identity)

    def _load_state(self) -> MonitorState:
        path = self._state_path()
        if not path.exists():
            return MonitorState()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return MonitorState(
            judged_through=int(raw.get("judged_through", 0)),
            judgements=[
                Judgement(
                    severity=Severity(j["severity"]),
                    reason=j.get("reason", ""),
                    evidence=tuple(j.get("evidence", ())),
                    suggestion=j.get("suggestion", ""),
                )
                for j in raw.get("judgements", ())
            ],
            interventions=[tuple(i) for i in raw.get("interventions", [])],
        )

    def _save_state(self) -> None:
        self._state_path().write_text(
            json.dumps(
                {
                    "judged_through": self.state.judged_through,
                    # Persisted because report() is what the central brain
                    # reads to decide whether to re-plan: without these a
                    # restarted monitor said worst="fine" for a worker it had
                    # just judged LOST.
                    "judgements": [
                        {
                            "severity": j.severity.value,
                            "reason": j.reason,
                            "evidence": list(j.evidence),
                            "suggestion": j.suggestion,
                        }
                        for j in self.state.judgements
                    ],
                    "interventions": [list(i) for i in self.state.interventions],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ reading

    def view(self) -> Any:
        """A lease-free window on the branch being written.

        The worker holds the write lease for the whole run, so this has to take
        nothing: a monitor that needed the lease could only watch a brain that
        had stopped, which is the opposite of the point.
        """
        return self.store.view(self.contract.identity)

    def observations(self) -> list[dict[str, Any]]:
        """Everything the worker has recorded since its last state.

        Comes from the same journal the WAL writes, so a monitor sees tool
        intents and results as they are fsynced rather than at checkpoint
        boundaries.
        """
        view = self.view()
        if not view.exists():
            return []
        # Both sources. The live journal is how a monitor sees intents and
        # results as they are fsynced; the committed transcript is where those
        # same turns go when the worker checkpoints, because publish_state
        # folds them into the state and unlinks the journal. Reading only the
        # journal destroyed any partial batch held back before a checkpoint:
        # for a worker checkpointing every N events at batch size B, exactly
        # N mod B events were never judged, at every checkpoint.
        return list(view.head.transcript.turns) + list(view.pending_turns())

    def unjudged(self) -> list[dict[str, Any]]:
        return self.observations()[self.state.judged_through :]

    def worker_is_alive(self) -> bool:
        """Whether anyone still holds the branch.

        A status claim from a dead brain is a confident lie, so liveness comes
        from the lease rather than from anything the worker said about itself.
        """
        return self.view().holder is not None

    # ------------------------------------------------------------------ judging

    def tick(self) -> Judgement | None:
        """Judge the next batch, or return None if there is nothing new.

        Only whole batches are judged while work continues; a partial batch is
        left for the next tick so the monitor is not billed for re-reading the
        same half-finished stretch.
        """
        pending = self.unjudged()
        if not pending:
            return None
        alive = self.worker_is_alive()
        if len(pending) < self.batch_size and alive:
            return None
        batch = pending[: self.batch_size]
        judgement = self._judge(self.contract, batch, self.view())
        self.state.judged_through += len(batch)
        self.state.judgements.append(judgement)
        self._save_state()
        return judgement

    def record(self, judgement: Judgement) -> None:
        """Put the verdict where the worker will find it on its next wake.

        ``store.judge`` needs no lease, which is what lets the monitor speak
        while the worker is still writing.
        """
        view = self.view()
        if view.exists():
            self.store.judge(view.head, judgement.to_verdict(self.identity))

    # ------------------------------------------------------------------ acting

    async def respond(self, judgement: Judgement, client: Any) -> str:
        """Escalate according to severity. Returns the rung taken.

        ``client`` is the worker's live ``ClaudeSDKClient``, held in this same
        process. A resumed client in another process accepts these calls and
        affects nothing.
        """
        self.record(judgement)
        rung = "none"
        if judgement.severity is Severity.DRIFTING:
            # A nudge: say it and let the brain course-correct. Interrupting a
            # brain that is merely wobbling costs more than it saves.
            rung = "nudge"
            await client.query(
                f"[monitor] {judgement.as_feedback()}\n"
                "Continue if you disagree, but say why."
            )
        elif judgement.severity is Severity.WRONG:
            # Stop the current turn, then say what was wrong. Order matters:
            # feedback sent to a brain mid-tool-call is read after the tool it
            # was meant to prevent.
            rung = "interrupt"
            await client.interrupt()
            await client.query(
                f"[monitor] I stopped you. {judgement.as_feedback()}\n"
                "Before continuing, check the state of your worktree: a tool "
                "may have completed even though it was interrupted."
            )
        elif judgement.severity is Severity.LOST:
            # Demote rather than kill. The reasoning is the expensive part and
            # it survives; a brain in plan mode can still report what it knows,
            # which is what the central brain needs in order to re-plan.
            rung = "demote"
            await client.interrupt()
            await client.set_permission_mode("plan")
            await client.query(
                f"[monitor] I have stopped your editing. {judgement.as_feedback()}\n"
                "Do not attempt further changes. Report what you tried, what "
                "you learned, and what you think should happen instead."
            )
        if rung != "none":
            self.state.interventions.append((rung, judgement.reason))
            self._save_state()
        return rung

    # ------------------------------------------------------------------ report

    def report(self) -> dict[str, Any]:
        """What the central brain reads to decide whether to re-plan."""
        return {
            "worker": self.contract.identity,
            "worst": self.state.worst.value,
            "judgements": len(self.state.judgements),
            "interventions": list(self.state.interventions),
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
