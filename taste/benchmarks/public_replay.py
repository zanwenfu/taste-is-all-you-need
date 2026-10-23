"""Public mini-swe-agent runs, replayed under the instrument.

SWE-bench's bash-only leaderboard publishes, for every model, the complete
record of each run: every command the agent executed, in order, with what it
printed. Re-executing those commands in the benchmark's own image, under the
recorder the live mini-swe-agent arm used, yields the timeline the original
run would have produced -- every intermediate tree, testable after the fact --
for no model spend. Forty live runs become thousands of replayed ones.

What makes a replay evidence rather than a reconstruction is checked per run,
not assumed:

- the replayed run must submit the patch the original submitted, compared
  per file and ignoring only the files the image build itself left modified
  (:func:`same_patch`);
- every command's exit status is compared with the recorded one;
- a recorder error (a shadow observation that failed to write) disqualifies
  the run, because a missing observation is a missing state.

The replay drives :class:`TasteEnvironment` directly rather than through the
scaffold's agent loop. The loop adds message bookkeeping and step and cost
limits, none of which touches the tree; each command goes through the same
``execute`` the live arm used -- container, working directory, environment
variables, merged streams, timeout, submission sentinel -- and the shadow log
observes after each one, exactly as live.

Two deliberate differences from the live arm, both for fidelity to the runs
being replayed: the agent's container has the network the leaderboard's had
(a recorded ``pip install`` must succeed again, or the trees diverge), and
each run keeps its own recorded environment variables and timeout. The probe
containers that test each state stay severed.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import re
import shlex
import subprocess
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from taste.benchmarks import swebench
from taste.benchmarks.miniswe import (
    SCAFFOLD,
    ScaffoldStats,
    TasteEnvironment,
    _scaffold_version,
    build_run_result,
)
from taste.benchmarks.public_trajectories import PublicTrajectory, replay_commands
from taste.kernel import RunResult
from taste.memory import Memory
from taste.replay import SandboxProbeExecutor, SuiteProbe, SuiteRun
from taste.resources import close_sandbox
from taste.shadow import ShadowLog

SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


class NetworkedAgentProvider:
    """A sandbox provider that opens the replay's agent container with a network.

    The live arm severs the agent's container so a model cannot fetch its own
    fix. A replay has no model -- its commands are fixed -- and the runs it
    reproduces had the network. Only the agent container (key ``agent:...``)
    is affected: the probe containers keep the default, and the grader asks
    for its own.
    """

    def __init__(self, inner: Any, network_mode: str = "bridge") -> None:
        self.inner = inner
        self.network_mode = network_mode

    def open(self, *, key: str, image: str = "", network_mode: str = "none"):
        if key.startswith("agent:"):
            network_mode = self.network_mode
        return self.inner.open(key=key, image=image, network_mode=network_mode)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class _WatchingSandbox:
    """Passes every call through; keeps the raw result of one watched command."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.watch: str | None = None
        self.seen: Any = None

    def exec(self, command: str, *args: Any, **kwargs: Any) -> Any:
        result = self._inner.exec(command, *args, **kwargs)
        if self.watch is not None and command == self.watch:
            self.seen = result
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class StateLogExecutor:
    """The probe executor, plus one line per tested state in ``log_path``.

    The executor reduces a suite run to per-test verdicts, and that reduction
    is right for what it is for: a test the runner never reported is a *hole*,
    never a failure. But why it went unreported is itself a measurement here
    -- an edit that leaves a test module unable to import has broken every
    test in it, as a developer would see it -- so the raw exit status, the
    unreported members and the output's tail are kept beside the verdicts.
    The analysis reports such states as their own kind of break and never
    folds them into test failures.
    """

    def __init__(self, sandbox: Any, memory: Memory, base_commit: str, *, log_path: Path, tail_chars: int = 6000) -> None:
        self.watching = _WatchingSandbox(sandbox)
        self.inner = SandboxProbeExecutor(self.watching, memory, base_commit)
        self.log_path = Path(log_path)
        self.tail_chars = tail_chars

    def run(self, sha: str, suite: SuiteProbe) -> SuiteRun:
        self.watching.watch, self.watching.seen = suite.command, None
        started = time.monotonic()
        try:
            run = self.inner.run(sha, suite)
        finally:
            raw, self.watching.watch = self.watching.seen, None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(state_record(sha, run, raw, time.monotonic() - started, self.tail_chars), sort_keys=True) + "\n")
        return run

    def close(self) -> None:
        self.inner.close()


def state_record(sha: str, run: SuiteRun, raw: Any, seconds: float, tail_chars: int = 6000) -> dict[str, Any]:
    """One tested state, as the analysis reads it.

    ``ran`` says whether the suite command executed at all: when it did not
    (the reset or the patch failed), ``infra_error`` names why and the state
    is infrastructure, not evidence. Passing members are counted rather than
    listed -- they are the complement of the three lists.
    """
    by_verdict: dict[str, list[str]] = {"fail": [], "error": [], "skip": []}
    passed = 0
    for member, verdict in run.statuses.items():
        if verdict == "pass":
            passed += 1
        else:
            by_verdict.setdefault(verdict, []).append(member)
    output = str(getattr(raw, "output", "") or "") if raw is not None else ""
    clean = raw is not None and run.infra_error is None and not by_verdict["fail"] and not by_verdict["error"]
    return {
        "sha": sha,
        "seconds": round(seconds, 2),
        "ran": raw is not None,
        "exit_code": run.exit_code,
        "timed_out": bool(getattr(raw, "timed_out", False)) if raw is not None else None,
        "infra_error": run.infra_error,
        "passed": passed,
        "failed": sorted(by_verdict["fail"]),
        "unreported": sorted(by_verdict["error"]),
        "skipped": sorted(by_verdict["skip"]),
        # The tail is for auditing a state that is not clean; a clean state's
        # output says nothing the counts do not.
        "output_tail": "" if clean else output[-tail_chars:],
    }


def make_state_log_factory(log_name: str = "states.jsonl") -> Callable[[Any, Memory, Any], StateLogExecutor]:
    """``executor_factory`` for :func:`taste.benchmarks.swebench_run.make_score`."""

    def factory(sandbox: Any, memory: Memory, ctx: Any) -> StateLogExecutor:
        return StateLogExecutor(sandbox, memory, ctx.instance.base_commit, log_path=Path(ctx.gitdir) / log_name)

    return factory


def needs_plain_shell(trajectory: PublicTrajectory) -> bool:
    """Whether the run's commands ran outside the project's environment.

    The scaffold's v2.0-2.3 SWE-bench config ran ``bash -c`` with no
    ``BASH_ENV``: no login shell, so the image's ``conda activate testbed``
    in ``.bashrc`` never ran and ``python`` was the image's base interpreter
    (3.11 where a Django 3.2 project's env has 3.6). v1 (``bash -lc``) and
    v2.4 (``BASH_ENV=/root/.bashrc``) both activated the project's env,
    which is what the routed sandbox always does.
    """
    return tuple(trajectory.interpreter) == ("bash", "-c") and "BASH_ENV" not in trajectory.env


def plain_shell(command: str, image_env: dict[str, str], run_env: dict[str, str]) -> str:
    """The command as ``docker exec -e K=V ... bash -c CMD`` ran it.

    A clean environment holding exactly what docker exec gives a process --
    the image's ``Config.Env``, ``HOME`` for root, the container's
    ``HOSTNAME`` -- plus the run's own variables, and a non-login shell. The
    routed sandbox's activation stays in the outer shell, invisible inside.
    """
    pairs = {**image_env, "HOME": "/root", **run_env}
    assigns = " ".join(f"{k}={shlex.quote(v)}" for k, v in pairs.items())
    return f'env -i {assigns} HOSTNAME="$HOSTNAME" bash -c {shlex.quote(command)}'


def _text_actions(trajectory: PublicTrajectory) -> bool:
    return trajectory.trajectory_format in ("", "mini-swe-agent-1")


def _submitted_type() -> type | None:
    try:
        from minisweagent.exceptions import Submitted
    except ImportError:  # the scaffold is installed wherever replays run
        return None
    return Submitted


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def command_record(
    index: int, recorded: Any, outcome: dict[str, Any], observed: bool, seconds: float, exception: str | None
) -> dict[str, Any]:
    """One replayed command beside the recorded one: the per-command fidelity line."""
    replay_rc = outcome.get("returncode") if outcome else None
    replay_timed_out = "timed out" in str(outcome.get("exception_info", "")) if outcome else False
    replay_output = str(outcome.get("output", "")) if outcome else ""
    return {
        "i": index,
        "observed": observed,
        "seconds": round(seconds, 2),
        "recorded_rc": recorded.returncode,
        "replay_rc": replay_rc,
        "recorded_timed_out": recorded.timed_out,
        "replay_timed_out": replay_timed_out,
        # Nothing to compare when either side has no exit status: a timeout, or
        # a command that was never recorded (the planted check).
        "rc_match": None if (recorded.returncode is None or replay_rc is None or recorded.timed_out or replay_timed_out)
        else recorded.returncode == replay_rc,
        "replay_output_sha": _digest(replay_output),
        "replay_output_chars": len(replay_output),
        "replay_output_head": replay_output[:2000],
        "exception": exception,
    }


def make_replay_execute(
    trajectory_for: Callable[[Any], PublicTrajectory], *, source: str = "", router_factory: Callable[[Any], Any] | None = None,
    image_env: dict[str, str] | None = None,
):
    """Build the ``execute`` callable for ``run_sweep``: one recorded run, re-executed.

    ``trajectory_for(cell)`` returns the parsed public run for a cell. The
    replayed submission is left on the context as ``replay_submission`` (None
    when the run never submitted) for :func:`make_replay_grade`.
    ``router_factory(ctx)`` exists for the hermetic tests; a replay uses the
    live arm's :class:`~taste.routing.SandboxRouter`. ``image_env`` is the
    image's ``Config.Env``, required for a run that ran in a plain shell
    (:func:`needs_plain_shell`).
    """

    def execute(cell, ctx) -> RunResult:
        if ctx.agent_sandbox is None:
            raise RuntimeError("a replay must run routed (a pinned container); refusing the host path")
        try:
            trajectory = trajectory_for(cell)
            session_id = uuid.uuid4().hex[:8]
            ctx.session = session_id
            return _replay_cell(ctx, trajectory, session_id, time.time(), source, router_factory, image_env)
        finally:
            if ctx.agent_sandbox is not None:
                close_sandbox(ctx.agent_sandbox)
                ctx.agent_sandbox = None

    return execute


def _live_router(ctx) -> Any:
    from taste.routing import SandboxRouter

    # Transparent sync, as the live arm: the recorded commands inspect their
    # work with git diff / git status, so the baseline must not advance.
    return SandboxRouter(ctx.agent_sandbox, ctx.workspace, workdir=ctx.agent_sandbox.workdir, advance_baseline=False)


def _replay_cell(
    ctx, trajectory: PublicTrajectory, session_id: str, started: float, source: str,
    router_factory: Callable[[Any], Any] | None = None, image_env: dict[str, str] | None = None,
) -> RunResult:
    memory = Memory.open_session(ctx.workspace, session_id, base_ref="HEAD")
    try:
        router = (router_factory or _live_router)(ctx)
        ctx.router = router
        gitdir = Path(ctx.gitdir)
        gitdir.mkdir(parents=True, exist_ok=True)
        env = TasteEnvironment(
            router, None,
            cwd=trajectory.cwd or ctx.agent_sandbox.workdir,
            timeout=int(trajectory.timeout or 60),
            env=dict(trajectory.env) or None,
        )
        shadow = ShadowLog(memory, gitdir=gitdir, session=session_id, cost_pair_reader=env.cost_pair)
        env.shadow = shadow
        ctx.shadow_ref = shadow.ref if hasattr(shadow, "ref") else f"TASTE_SHADOW_HEAD_{session_id.upper()}"
        shadow.observe(step_id="run", attempt=0, trigger="run")

        commands = replay_commands(trajectory)
        shell = "login"
        if needs_plain_shell(trajectory):
            if image_env is None:
                raise RuntimeError("this run ran in a plain shell; replaying it needs the image's environment")
            commands = [plain_shell(command, image_env, dict(trajectory.env)) for command in commands]
            shell = "plain"
        else:
            # Each command is its own shell string, as ``bash -lc CMD`` parsed it:
            # spliced raw into the environment's wrapper, an unterminated heredoc
            # (which bash -c only warns about) swallows the rest of the wrapper.
            commands = [f"bash -c {shlex.quote(command)}" for command in commands]
        submitted_type = _submitted_type()
        submitted, submission, exception, replayed = False, "", None, 0
        with (gitdir / "replay.jsonl").open("w", encoding="utf-8") as log:
            for index, command in enumerate(commands):
                before, t0, outcome = env.observed, time.monotonic(), {}
                try:
                    outcome = env.execute({"command": command})
                except Exception as exc:
                    if submitted_type is not None and isinstance(exc, submitted_type):
                        message = exc.messages[0] if exc.messages else {}
                        submission = str(message.get("extra", {}).get("submission", ""))
                        submitted = True
                        outcome = {"returncode": 0, "output": f"{SENTINEL}\n{submission}", "exception_info": ""}
                    else:
                        exception = f"{type(exc).__name__}: {exc}"
                if not submitted and not exception and index == trajectory.submit_index and _text_actions(trajectory):
                    # The text-action scaffold (v1) ended a run on the sentinel whatever
                    # the exit status; today's environment also demands a zero exit.
                    lines = str(outcome.get("output", "")).lstrip().splitlines(keepends=True)
                    if lines and lines[0].strip() == SENTINEL:
                        submission, submitted = "".join(lines[1:]), True
                replayed += 1
                log.write(json.dumps(command_record(
                    index, trajectory.commands[index], outcome, env.observed > before, time.monotonic() - t0, exception,
                ), sort_keys=True) + "\n")
                if submitted or exception:
                    break

        ctx.replay_submission = submission if submitted else None
        ctx.llm_stats = ScaffoldStats()
        with contextlib.suppress(Exception):
            shadow.observe(step_id="final", attempt=0, trigger="final", dedupe=False)
        # As the live arm: grading and the state export diff against the root
        # commit, and a new file is invisible to that diff until it is staged.
        with contextlib.suppress(Exception):
            memory.repo.git.add("--all", "--", ".", ":(exclude)patch.txt")
        manifest = {
            "mode": "replay",
            "session": session_id,
            "scaffold": SCAFFOLD,
            "scaffold_version_recorded": trajectory.mini_version,
            "scaffold_version_installed": _scaffold_version(),
            "source": source,
            "instance_id": trajectory.instance_id,
            "trajectory_format": trajectory.trajectory_format,
            "recorded_exit_status": trajectory.exit_status,
            "recorded_environment_class": trajectory.environment_class,
            "commands_recorded": len(trajectory.commands),
            "commands_to_replay": len(commands),
            "commands_replayed": replayed,
            "submit_index": trajectory.submit_index,
            "unexecuted_turns": trajectory.unexecuted_turns,
            "submitted": submitted,
            "submission_chars": len(submission),
            "exception": exception,
            "observations": env.observed,
            "shadow_errors": int(getattr(shadow, "errors", 0)),
            "sync_skipped_paths": list(getattr(router, "skipped", [])),
            "environment": env.get_template_vars(),
            "network_mode": getattr(ctx.agent_sandbox, "network_mode", None),
            "streams": "merged (2>&1 at the shell)",
            "shell": shell,
            "interpreter_recorded": list(trajectory.interpreter),
            "created_at": started,
            "elapsed_s": round(time.time() - started, 2),
        }
        (gitdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        if submitted:
            (gitdir / "replay_submission.diff").write_text(submission)
        exit_status = "Submitted" if submitted else (trajectory.exit_status or "Replayed")
        return build_run_result(
            task=ctx.instance.problem_statement, session_id=session_id, branch=memory.branch,
            final_sha=memory.head().sha, started=started, exit_status=exit_status, cost=0.0, exception=exception,
        )
    finally:
        memory.close()


def make_replay_grade(*, timeout: int = 1800, grader=None):
    """The official verdict on the replayed *submission*, as the leaderboard graded the original.

    Not the working tree: the leaderboard graded the patch the run submitted,
    which for the scaffold's later versions is a file list the agent chose,
    and a scratch script left in the tree is not part of it. A run that never
    submitted is not graded (the leaderboard scored it unresolved).
    """
    run_grader = grader or swebench.grade_in_sandbox

    def grade(ctx, result) -> bool | None:
        submission = getattr(ctx, "replay_submission", None)
        if ctx.provider is None or submission is None:
            return None
        sandbox = ctx.provider.open(key=f"grade:{ctx.instance.instance_id}", image=ctx.instance.image, network_mode="bridge")
        try:
            report = run_grader(sandbox, ctx.instance, submission, timeout=timeout)
        finally:
            close_sandbox(sandbox)
        if report is None:
            return None
        ctx.grade_report = report
        return report.resolved

    return grade


_DIFF_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+)$")


def patch_files(patch: str) -> dict[str, str]:
    """A unified diff split per file, normalised for comparison.

    ``index`` lines are dropped (blob abbreviations depend on the repository,
    not on the change) and trailing whitespace is stripped; everything else,
    mode lines included, is content.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in patch.splitlines():
        header = _DIFF_HEADER.match(line)
        if header:
            current = header.group(2)
            sections.setdefault(current, [])
            continue
        if current is None or line.startswith("index "):
            continue
        sections[current].append(line.rstrip())
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def build_edit_paths(porcelain: str) -> tuple[str, ...]:
    """Paths ``git status --porcelain`` reports in a fresh container of the image.

    These are the image build's own edits and leftovers (sphinx's tox.ini, a
    generated file): the original run's ``git diff`` saw them, the replay's
    cannot (its baseline is the image's tree), so the patch comparison skips
    them. A directory entry ends in ``/`` and covers everything under it.
    """
    paths: list[str] = []
    for line in porcelain.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path.strip().strip('"'))
    return tuple(paths)


def _ignored(path: str, ignore: tuple[str, ...]) -> bool:
    return any(path == entry or (entry.endswith("/") and path.startswith(entry)) for entry in ignore)


def same_patch(original: str, replayed: str, *, ignore: tuple[str, ...] = (), build: str = "") -> tuple[bool, list[str]]:
    """Whether two submissions change the same files the same way; and which files differ.

    ``build`` is the image build's own edit of tracked files (``git diff
    <base_commit>`` in a fresh container). The run being replayed may have
    seen those edits as uncommitted and submitted them -- today's images
    commit sphinx's dependency pins, July 2025's did not -- while the replay's
    baseline already holds them. So a file the replay never touched and the
    build did is excused only when the original's diff of it is exactly the
    build's edit; a file both the agent and the build changed is compared
    as is (and differs, conservatively). ``ignore`` holds untracked build
    leftovers, skipped by path.
    """
    a = {name: body for name, body in patch_files(original).items() if not _ignored(name, ignore)}
    b = {name: body for name, body in patch_files(replayed).items() if not _ignored(name, ignore)}
    built = patch_files(build) if build else {}
    differing = []
    for name in sorted(set(a) | set(b)):
        if name not in b and name in built:
            if a.get(name) != built[name]:
                differing.append(name)
        elif a.get(name) != b.get(name):
            differing.append(name)
    return not differing, differing


def export_states(workspace: Path, timeline: Any, out_path: Path) -> int:
    """Every observed tree, as a patch against the run's root commit, gzipped.

    The workspace is deleted after a replay (a thousand Django checkouts do
    not fit), and these patches are what lets any state be re-tested later
    against the same image.
    """

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=workspace, capture_output=True, text=True, check=True).stdout

    root = git("rev-list", "--max-parents=0", "HEAD").split()[0]
    records = []
    for commit in timeline:
        records.append({
            "seq": commit.seq, "sha": commit.sha, "step_id": commit.step_id, "trigger": commit.trigger,
            "files": list(commit.files), "patch": git("diff", "--binary", root, commit.sha),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as handle:
        json.dump({"root": root, "states": records}, handle)
    return len(records)
