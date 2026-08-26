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

**It is uniform across arms.** A no-recovery arm has no checkpoints of its
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
# pointer suffices because each shadow commit parents the previous.
#
# Being outside refs/ also puts the chain outside git's gc roots, so the ref
# is written with --create-reflog: reflogs are gc roots and are not walked by
# --all, which is the only construction that keeps the chain both durable and
# unenumerable. See _write_tree_commit.
#
# This is containment, not a boundary: an agent that guesses the name can
# still resolve it. The boundary is a container whose mount excludes the git
# directory; that is an ops task, not a code one, and is recorded as such.
def _assert_pair_shape(reader: Any) -> None:
    """Reject a scalar cost reader at construction rather than at use.

    The failure this prevents is silent by nature: the reader is only called
    from inside a fail-open wrapper, so a wrong-shaped one degrades to
    0.0/0.0 on every observation instead of raising. Two headline metrics
    then read as exactly zero for every arm, which looks like a finding.
    """
    try:
        sample = reader()
    except Exception as exc:
        raise TypeError(f"cost_pair_reader raised on call: {exc!r}") from exc
    try:
        billed, work = sample
        float(billed), float(work)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "cost_pair_reader must return (billed_usd, work_usd); "
            f"got {sample!r}. A bare float is the Guardrails reader shape."
        ) from exc


# The shadow chain's author. Fixed, so the timeline does not depend on whose
# machine it was produced on.
_IDENTITY_NAME = "taste-shadow"
_IDENTITY_EMAIL = "shadow@taste.localhost"

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
    ``worker`` (a step boundary), ``run`` (start), ``rollback`` (a reset that
    discarded an attempt), ``final`` (run end — always recorded, see
    :meth:`ShadowLog.observe`)."""
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
        cost_pair_reader: Any = None,
    ) -> None:
        self.memory = memory
        self.session = session
        self.enabled = enabled
        # Named for its *shape*, not its subject. Guardrails takes a reader of
        # the same name and the same `Any` type that returns a bare float, and
        # handing that one to this one raises TypeError on unpack — which the
        # fail-open wrapper in _costs swallows, recording 0.0/0.0 for every
        # observation and silently zeroing wasted_work_usd and
        # cost_to_detect_usd, two of the four headline metrics.
        self.cost_pair_reader = cost_pair_reader
        if cost_pair_reader is not None:
            _assert_pair_shape(cost_pair_reader)
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
        dedupe: bool = True,
    ) -> ShadowCommit | None:
        """Record the working tree as it stands right now.

        Returns None when disabled, when nothing changed since the last
        observation, or when the write failed — none of which is an error the
        caller needs to handle.

        ``dedupe=False`` records the observation even when the tree is
        byte-identical to the previous one. The run-end observation needs it:
        "this is the tree the run ended on" is a claim about the *timeline's
        shape*, and a timeline whose last entry is whichever mutation
        happened to come last cannot state it. Everything mid-run keeps the
        default — an unchanged tree mid-run genuinely is noise.
        """
        if not self.enabled:
            return None
        try:
            sha, files = self._write_tree_commit(dedupe=dedupe)
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

    def _write_tree_commit(self, *, dedupe: bool = True) -> tuple[str | None, tuple[str, ...]]:
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
        if parent is not None and dedupe:
            parent_tree = repo.git.rev_parse(f"{parent}^{{tree}}").strip()
            if parent_tree == tree:
                return None, ()

        args = [tree, "-m", f"shadow {self._seq + 1}"]
        if parent is not None:
            args = [tree, "-p", parent, "-m", f"shadow {self._seq + 1}"]
        # Identity is supplied explicitly rather than inherited from the
        # machine's global git config. `commit-tree` refuses to run without
        # one, and on a fresh host -- which is exactly what a reproduction
        # is -- there is no global identity, so every observation raised,
        # the fail-open wrapper below swallowed it, and the timeline came
        # back EMPTY. Zero observations reads as "the run did nothing",
        # not as "the instrument could not run".
        #
        # It also makes the shadow chain independent of who is running it,
        # which is the property a reproducibility claim actually needs.
        with repo.git.custom_environment(
            GIT_AUTHOR_NAME=_IDENTITY_NAME,
            GIT_AUTHOR_EMAIL=_IDENTITY_EMAIL,
            GIT_COMMITTER_NAME=_IDENTITY_NAME,
            GIT_COMMITTER_EMAIL=_IDENTITY_EMAIL,
        ):
            sha = repo.git.commit_tree(*args).strip()
        # --create-reflog is load-bearing, not hygiene. A top-level pseudo-ref
        # is deliberately outside refs/ so the agent cannot enumerate it, but
        # that also puts it outside the set git treats as gc roots: a forced
        # `git gc --prune=now` deletes the entire shadow chain, and the
        # timeline then replays as "error" at every observation — which scores
        # as a clean run rather than as the destroyed measurement it is.
        # Reflogs *are* gc roots and are *not* enumerated by --all, so a reflog
        # on the pseudo-ref buys durability without giving the isolation back.
        repo.git.update_ref("--create-reflog", self.ref, sha)

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
        if self.cost_pair_reader is None:
            return 0.0, 0.0
        try:
            return self.cost_pair_reader()
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
