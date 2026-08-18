"""The inode table: one scannable card per checkpoint.

A commit tells you *that* something changed. Reading it tells you *what*.
Between those two costs sits the thing this module provides: a small,
uniform card per checkpoint — intent, files touched, verdict, cost, outcome
— cheap enough to list a whole branch, informative enough to decide which
node is worth paging in. Scan the index, then ``git show`` exactly one
commit. That is the demand-paging read path, with the index it was always
missing.

**Where cards live.** A git *note* on the checkpoint, plus an append-only
JSONL line. Notes were chosen over a tracked file for three reasons, each
disqualifying on its own:

* a tracked file is part of the diff the Monitor judges, so the harness
  would be grading its own bookkeeping;
* ``reset --hard`` destroys it, which loses the record of exactly the
  failed attempts the record exists to explain;
* it conflicts on merge in parallel waves.

A note has none of those properties. The JSONL mirror is the cross-run
store: unlike ``events.jsonl`` it is never truncated, so it accumulates
across every run in a workspace, and it survives even when the commit it
describes has been rolled away.

**Anchors.** Before a rollback discards commits, the journal points a ref at
them under ``refs/taste/attempt/*``. The work stays reachable and readable
(``git show``) instead of becoming an unreferenced object awaiting gc. This
is what makes a failed attempt inspectable after the fact.

Every write here is non-fatal. Bookkeeping must never be able to fail a step.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from taste.memory import Memory

# Bump when a field changes meaning. Readers tolerate unknown keys and
# missing ones, so an old card stays readable by a newer harness.
CARD_SCHEMA = "taste.card/1"

NOTES_REF = "taste-cards"
ANCHOR_PREFIX = "refs/taste/attempt"

# Notes live on one ref, which git locks per write. Parallel workers in a wave
# contend for it; serializing in-process plus a short retry covers both this
# process's threads and any external writer holding the lock briefly.
_NOTE_LOCK = threading.Lock()
_NOTE_WRITE_ATTEMPTS = 4

# A card is an index entry, not an archive. These caps keep it scannable and
# stop a pathological step from writing a megabyte of bookkeeping per commit.
MAX_FILES = 40
MAX_SUMMARY = 200


@dataclass(frozen=True)
class FileChange:
    path: str
    added: int = 0
    deleted: int = 0

    def render(self) -> str:
        return f"{self.path} +{self.added}-{self.deleted}"


@dataclass(frozen=True)
class CheckpointCard:
    """What happened at one checkpoint, in the form you want before deciding
    whether to read the diff."""

    session: str
    step_id: str
    sha: str
    kind: Literal["plan", "step", "merge", "abort"] = "step"
    schema: str = CARD_SCHEMA
    attempt: int = 1
    parent_sha: str | None = None
    branch: str = ""
    intent: str = ""
    depends_on: tuple[str, ...] = ()
    verdict: Literal["pass", "fail", "aborted", "n/a"] = "n/a"
    verdict_reason: str = ""
    verification_kind: str = ""
    verification_command: str | None = None
    files: tuple[FileChange, ...] = ()
    files_truncated: bool = False
    diff_lines: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    stopped_reason: str = ""
    failure_class: str | None = None
    action: str | None = None
    cost_usd: float | None = None
    elapsed_s: float = 0.0
    # The worker's own account of what it did. UNTRUSTED — an agent
    # describing its own work is exactly the self-assessment the Monitor
    # exists to counterweight. Kept for context, never for judgement.
    summary: str = ""
    ts: float = field(default_factory=time.time)

    def one_line(self) -> str:
        """A single scannable row: the whole point of the index."""
        bits = [f"{self.sha[:7]}", f"{self.step_id:<10}", f"{self.verdict.upper():<7}"]
        if self.failure_class:
            bits.append(f"{self.failure_class}->{self.action or '?'}")
        if self.files:
            adds = sum(f.added for f in self.files)
            dels = sum(f.deleted for f in self.files)
            bits.append(f"{len(self.files)}f +{adds}-{dels}")
        if self.cost_usd:
            bits.append(f"${self.cost_usd:.4f}")
        return "  ".join(bits)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def to_yaml_block(self) -> str:
        """This one card as YAML — the detail view for a single node."""
        return BranchIndex(ref=self.branch, cards=(self,)).to_yaml()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CheckpointCard:
        """Tolerant read: unknown keys ignored, missing keys defaulted.

        Cards are written by one harness version and read by another, so a
        strict parse would turn a schema change into unreadable history.
        """
        known = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in raw.items() if k in known}
        data["files"] = tuple(
            FileChange(**{k: v for k, v in f.items() if k in FileChange.__dataclass_fields__})
            for f in raw.get("files", [])
            if isinstance(f, dict)
        )
        data["depends_on"] = tuple(raw.get("depends_on", ()))
        return cls(**data)


@dataclass(frozen=True)
class BranchIndex:
    """Every card on a branch, newest last."""

    ref: str
    cards: tuple[CheckpointCard, ...]
    degraded: int = 0
    """Commits with no card, reconstructed from the commit trailer alone."""
    source: Literal["notes", "jsonl", "commits"] = "notes"

    def render(self, *, limit: int | None = None) -> str:
        cards = self.cards[-limit:] if limit else self.cards
        lines = [c.one_line() for c in cards]
        if self.degraded:
            lines.append(f"({self.degraded} checkpoint(s) without cards)")
        return "\n".join(lines)

    def to_yaml(self, *, limit: int | None = None) -> str:
        """YAML is a rendering, not the storage format.

        Cards are JSON in git notes; this projects them for humans. Written
        by hand rather than pulling in PyYAML — the shape is fixed and
        shallow, and a dependency for a projection nobody parses is a poor
        trade.
        """
        cards = self.cards[-limit:] if limit else self.cards
        out = [f"ref: {self.ref}", f"checkpoints: {len(cards)}", "cards:"]
        for c in cards:
            out.append(f"  - sha: {c.sha[:12]}")
            out.append(f"    step: {c.step_id}")
            out.append(f"    verdict: {c.verdict}")
            if c.intent:
                out.append(f"    intent: {_yaml_scalar(c.intent)}")
            if c.failure_class:
                out.append(f"    failure_class: {c.failure_class}")
            if c.action:
                out.append(f"    action: {c.action}")
            if c.files:
                out.append("    files:")
                out.extend(f"      - {f.render()}" for f in c.files)
            if c.cost_usd is not None:
                out.append(f"    cost_usd: {c.cost_usd:.6f}")
            if c.attempt != 1:
                out.append(f"    attempt: {c.attempt}")
        return "\n".join(out) + "\n"


class Journal:
    """Writes cards and anchors. Owns ``refs/notes/taste-cards`` and
    ``refs/taste/attempt/*`` exclusively — no other module writes there.

    Disabled by default so today's behavior is bit-identical; the kernel
    switches it on via config.
    """

    def __init__(self, memory: Memory, *, gitdir: Path, enabled: bool = True) -> None:
        self.memory = memory
        self.enabled = enabled
        self.path = Path(gitdir) / "journal.jsonl"
        self.errors = 0

    # ------------------------------------------------------------ writes

    def card(self, card: CheckpointCard) -> bool:
        """Record a card. Returns whether it was fully persisted.

        Both writes are best-effort: bookkeeping must never be the reason a
        step fails. A failure increments ``errors`` and is otherwise silent
        to the caller.
        """
        if not self.enabled:
            return False
        card = _bounded(card)
        body = card.to_json()
        ok = True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(body + "\n")
        except OSError:
            self.errors += 1
            ok = False
        # Parallel steps write the one notes ref concurrently, and git takes a
        # lock on it. Without a retry the losers fail, the fail-open handler
        # swallows it, and cards vanish for exactly the runs that need them
        # most. The JSONL line above is already safely appended either way.
        for attempt in range(_NOTE_WRITE_ATTEMPTS):
            try:
                with _NOTE_LOCK:
                    self.memory.write_note(card.sha, body, ref=NOTES_REF)
                return ok
            except Exception:
                if attempt == _NOTE_WRITE_ATTEMPTS - 1:
                    self.errors += 1
                    return False
                time.sleep(0.05 * (attempt + 1))
        return ok

    def anchor(self, *, session: str, step_id: str, attempt: int, sha: str) -> str | None:
        """Keep ``sha`` reachable before a rollback discards it."""
        if not self.enabled:
            return None
        ref = f"{ANCHOR_PREFIX}/{session}/{step_id}/{attempt}"
        try:
            self.memory.anchor(ref, sha)
        except Exception:
            self.errors += 1
            return None
        return ref

    # ------------------------------------------------------------ reads

    def read(self, sha: str) -> CheckpointCard | None:
        note = self.memory.read_note(sha, ref=NOTES_REF)
        if note:
            try:
                return CheckpointCard.from_dict(json.loads(note))
            except (ValueError, TypeError):
                pass
        return self._from_jsonl(sha)

    def anchors(self) -> list[tuple[str, str]]:
        return self.memory.list_refs(f"{ANCHOR_PREFIX}/")

    def prune_anchors(self, *, session: str | None = None) -> int:
        """Drop attempt anchors, letting gc reclaim the discarded attempts."""
        removed = 0
        for ref, _sha in self.anchors():
            if session and f"/{session}/" not in ref:
                continue
            self.memory.delete_ref(ref)
            removed += 1
        return removed

    def _from_jsonl(self, sha: str) -> CheckpointCard | None:
        """Fallback for a card whose commit was rolled away with its note."""
        if not self.path.exists():
            return None
        found = None
        for line in self.path.read_text().splitlines():
            if sha[:7] not in line:
                continue
            try:
                raw = json.loads(line)
            except ValueError:
                continue
            if raw.get("sha", "").startswith(sha[:7]):
                found = CheckpointCard.from_dict(raw)  # last write wins
        return found


def load_index(memory: Memory, *, branch: str | None = None, journal: Journal | None = None) -> BranchIndex:
    """Assemble the index for a branch with a single ``git log``.

    Cards come from notes where present. A commit without one still gets a
    row, synthesized from the checkpoint trailer — a partial index beats a
    missing one, and ``degraded`` reports how much was reconstructed.
    """
    ref = branch or memory.branch
    subjects = memory.commit_subjects(ref)
    checkpoints = {c.sha: c for c in memory.log(branch=ref)}

    cards: list[CheckpointCard] = []
    degraded = 0
    for sha, subject in reversed(subjects):  # oldest first
        note = memory.read_note(sha, ref=NOTES_REF)
        if note:
            try:
                cards.append(CheckpointCard.from_dict(json.loads(note)))
                continue
            except (ValueError, TypeError):
                pass
        if journal is not None:
            recovered = journal._from_jsonl(sha)
            if recovered is not None:
                cards.append(recovered)
                continue
        degraded += 1
        cp = checkpoints.get(sha)
        cards.append(
            CheckpointCard(
                session="",
                step_id=cp.step_id if cp else sha[:7],
                sha=sha,
                branch=ref,
                intent=subject,
                parent_sha=cp.parent_sha if cp else None,
            )
        )
    return BranchIndex(ref=ref, cards=tuple(cards), degraded=degraded)


# ---------------------------------------------------------------- helpers


def card_from_step(
    *,
    session: str,
    branch: str,
    sha: str,
    parent_sha: str | None,
    step: Any,
    attempt: int,
    verdict_passed: bool,
    verdict_reason: str,
    files: tuple[FileChange, ...],
    diff_lines: int,
    worker: Any = None,
    cost_usd: float | None = None,
    elapsed_s: float = 0.0,
    failure_class: str | None = None,
    action: str | None = None,
) -> CheckpointCard:
    """Build a step card from the objects the kernel already has in hand."""
    return CheckpointCard(
        session=session,
        step_id=step.id,
        sha=sha,
        kind="step",
        attempt=attempt,
        parent_sha=parent_sha,
        branch=branch,
        intent=step.description,
        depends_on=tuple(step.depends_on),
        verdict="pass" if verdict_passed else "fail",
        verdict_reason=verdict_reason,
        verification_kind=step.verification.kind,
        verification_command=step.verification.command,
        files=files,
        diff_lines=diff_lines,
        tool_calls=getattr(worker, "tool_calls", 0) or 0,
        # Declared on the card since it existed and never populated, so every
        # card in every run reported zero tool errors. Caught on a real run
        # where the recovery table diagnosed R1.tool_errors -- correctly, the
        # tooling really had raised -- while the audit trail beside it said
        # there had been none. Anyone reading the cards would have concluded
        # that tool errors never happen.
        tool_errors=getattr(worker, "tool_errors", 0) or 0,
        stopped_reason=getattr(worker, "stopped_reason", "") or "",
        summary=getattr(worker, "summary", "") or "",
        cost_usd=cost_usd,
        elapsed_s=elapsed_s,
        failure_class=failure_class,
        action=action,
    )


def parse_numstat(raw: str) -> tuple[tuple[FileChange, ...], int]:
    """Turn ``git diff --numstat`` output into file changes + total lines."""
    changes: list[FileChange] = []
    total = 0
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added_raw, deleted_raw, path = parts
        # "-" marks a binary file; count it as changed but contributing no lines.
        added = int(added_raw) if added_raw.isdigit() else 0
        deleted = int(deleted_raw) if deleted_raw.isdigit() else 0
        changes.append(FileChange(path=path.strip(), added=added, deleted=deleted))
        total += added + deleted
    return tuple(changes), total


def _bounded(card: CheckpointCard) -> CheckpointCard:
    """Clamp a card to index size. Truncation is recorded, never silent."""
    updates: dict[str, Any] = {}
    if len(card.files) > MAX_FILES:
        updates["files"] = card.files[:MAX_FILES]
        updates["files_truncated"] = True
    if len(card.summary) > MAX_SUMMARY:
        updates["summary"] = card.summary[: MAX_SUMMARY - 1] + "…"
    return replace(card, **updates) if updates else card


def _yaml_scalar(text: str) -> str:
    """Quote a scalar when plain style would not round-trip."""
    flat = " ".join(text.split())
    if flat != text or any(ch in flat for ch in ':#"\'{}[]&*!|>%@`') or not flat:
        return json.dumps(flat)
    return flat
