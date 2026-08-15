"""Observational checkpointing: recording what happened, without changing it.

The kernel commits when a step *passes*. That is the right unit for the
agent and the wrong unit for measurement — it is exactly the granularity at
which a silent regression hides, because a step that broke something and
then fixed it looks identical to a step that never broke anything.

This module writes a commit after every file mutation, on a ref no agent can
see, in every arm. Two properties make it usable as an instrument:

**It is invisible to the run.** Shadow commits live on ``refs/taste/shadow/*``
and are built from the index without touching the working tree, the session
branch, or ``HEAD``. Nothing the agent reads changes; nothing the Monitor
grades changes. An arm with shadow commits enabled must produce byte-identical
agent-visible behaviour to one without.

**It is uniform across arms.** A self-verifying arm has no checkpoints of its
own, so any timeline built from *its* commits would be incomparable with an
arm that checkpoints per step. Shadow commits give every arm the same
granularity, which is the only way ``detection latency`` means the same thing
in both — and why latency is denominated in shadow commits and dollars rather
than in harness-native "steps".

What this buys, concretely: given a run and a set of held-out probes, you can
replay the probes against each shadow commit and reconstruct exactly when a
regression entered — even in an arm that never checkpointed, never rolled
back, and never noticed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from taste.memory import Memory

# A top-level pseudo-ref, in the same family as ORIG_HEAD and FETCH_HEAD.
# Anything under refs/ is enumerated by `git for-each-ref`, `git branch -a`
# and `git log --all` — all read-only commands the guardrails permit — so a
# ref there both perturbs what the agent sees AND hands it a way to read back
# work the harness rolled away. A top-level ref is not enumerated, and one
# pointer suffices because each shadow commit parents the previous, keeping
# the whole chain reachable and safe from gc.
#
# This is containment, not a boundary: an agent that guesses the name can
# still resolve it. The boundary is a container whose mount excludes the git
# directory; that is an ops task, not a code one, and is recorded as such.
SHADOW_HEAD = "TASTE_SHADOW_HEAD"
SHADOW_PREFIX = SHADOW_HEAD  # retained for callers that report the ref name


@dataclass(frozen=True)
class ShadowCommit:
    """One observation point in a run's timeline."""

    seq: int
    sha: str
    session: str
    step_id: str
    attempt: int
    trigger: str
    """What produced this observation: ``tool`` (a mutating tool call),
    ``step`` (a step boundary), ``run`` (start or end)."""
    tool: str | None = None
    files: tuple[str, ...] = ()
    cost_billed_usd: float = 0.0
    cost_work_usd: float = 0.0
    """Cumulative at this point. Latency between two observations is their
    difference — a unit that means the same thing in every arm, unlike a
    count of steps."""
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class ShadowLog:
    """Writes shadow commits for one run.

    Every write is best-effort and non-fatal: an instrument that can fail the
    experiment it measures is worse than no instrument.
    """

    def __init__(
        self,
        memory: Memory,
        *,
        gitdir: Path,
        session: str,
        enabled: bool = True,
        cost_reader: Any = None,
    ) -> None:
        self.memory = memory
        self.session = session
        self.enabled = enabled
        self.cost_reader = cost_reader
        self.path = Path(gitdir) / "shadow.jsonl"
        # Private staging area, outside the repository's own index.
        self._index_path = Path(gitdir) / f"shadow-index-{session}"
        self.commits: list[ShadowCommit] = []
        self.errors = 0
        self._seq = 0

    # ------------------------------------------------------------ writing

    def observe(
        self,
        *,
        step_id: str,
        attempt: int,
        trigger: str,
        tool: str | None = None,
    ) -> ShadowCommit | None:
        """Record the working tree as it stands right now.

        Returns None when disabled, when nothing changed since the last
        observation, or when the write failed — none of which is an error the
        caller needs to handle.
        """
        if not self.enabled:
            return None
        try:
            sha, files = self._write_tree_commit()
        except Exception:
            self.errors += 1
            return None
        if sha is None:
            return None  # nothing changed; an observation would be noise

        billed, work = self._costs()
        self._seq += 1
        commit = ShadowCommit(
            seq=self._seq,
            sha=sha,
            session=self.session,
            step_id=step_id,
            attempt=attempt,
            trigger=trigger,
            tool=tool,
            files=files,
            cost_billed_usd=billed,
            cost_work_usd=work,
        )
        self.commits.append(commit)
        self._persist(commit)
        return commit

    def _write_tree_commit(self) -> tuple[str | None, tuple[str, ...]]:
        """Commit the current tree onto the shadow ref, touching nothing else.

        Built from ``write-tree`` / ``commit-tree`` / ``update-ref`` rather
        than ``git commit``: those write objects and move one ref without
        checking anything out, moving HEAD, or running hooks.

        **Staged into a private index, never the repository's own.** Using the
        real index looks harmless and is not: ``git add --all`` stages the
        agent's edits, so an agent that changes a file and then runs ``git
        diff`` to review its own work — a read-only command the guardrails
        explicitly permit — sees an empty diff. Measured directly: a 103-char
        diff became empty, and ``git status`` went from ``M file`` to ``M
        file`` staged. An instrument that changes what the subject observes is
        not an instrument. GIT_INDEX_FILE points the staging at a throwaway
        file so the repository's index never moves.
        """
        repo = self.memory.repo
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        # A fresh empty index each time: "everything currently in the working
        # tree" is exactly the snapshot semantics we want, and it makes
        # deletions fall out naturally (an absent file is absent from the tree).
        if self._index_path.exists():
            self._index_path.unlink()

        with repo.git.custom_environment(GIT_INDEX_FILE=str(self._index_path)):
            repo.git.add("--all", ".")
            tree = repo.git.write_tree().strip()

        parent = self._current_head()
        if parent is not None:
            parent_tree = repo.git.rev_parse(f"{parent}^{{tree}}").strip()
            if parent_tree == tree:
                return None, ()

        args = [tree, "-m", f"shadow {self._seq + 1}"]
        if parent is not None:
            args = [tree, "-p", parent, "-m", f"shadow {self._seq + 1}"]
        sha = repo.git.commit_tree(*args).strip()
        repo.git.update_ref(self.ref, sha)

        files = ()
        if parent is not None:
            raw = repo.git.diff("--name-only", parent, sha)
            files = tuple(line for line in raw.splitlines() if line.strip())
        return sha, files

    def _current_head(self) -> str | None:
        try:
            return self.memory.repo.git.rev_parse(self.ref).strip()
        except Exception:
            return None

    def _costs(self) -> tuple[float, float]:
        if self.cost_reader is None:
            return 0.0, 0.0
        try:
            return self.cost_reader()
        except Exception:
            return 0.0, 0.0

    def _persist(self, commit: ShadowCommit) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(commit.to_json() + "\n")
        except OSError:
            self.errors += 1

    # ------------------------------------------------------------ reading

    @property
    def ref(self) -> str:
        # One pointer per session, kept out of the refs/ namespace so the
        # agent's view of its own repository is unchanged.
        return f"{SHADOW_HEAD}_{self.session.upper().replace('-', '_')}"

    def timeline(self) -> tuple[ShadowCommit, ...]:
        return tuple(self.commits)


def load_timeline(gitdir: Path, session: str) -> tuple[ShadowCommit, ...]:
    """Read a run's observation timeline back from disk.

    Reads the JSONL rather than the ref, because the JSONL survives even when
    the commits have been garbage-collected — and because a timeline is a
    sequence, which a single ref cannot express.
    """
    path = Path(gitdir) / "shadow.jsonl"
    if not path.exists():
        return ()
    out: list[ShadowCommit] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except ValueError:
            continue
        if raw.get("session") != session:
            continue
        known = {f for f in ShadowCommit.__dataclass_fields__}
        data = {k: v for k, v in raw.items() if k in known}
        data["files"] = tuple(raw.get("files", ()))
        out.append(ShadowCommit(**data))
    return tuple(sorted(out, key=lambda c: c.seq))
