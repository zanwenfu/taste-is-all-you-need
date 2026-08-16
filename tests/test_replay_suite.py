"""Per-test verdicts, and the line between a hole and a regression.

The instrument's worst available failure is reporting a regression that did
not happen. Almost every way that can occur runs through this file: a suite
that could not start, a patch that would not apply, a log that parsed to
nothing, a test the runner never mentioned. Each of those is a *missing
observation*. None of them is evidence that a test broke, and the tests here
exist to keep that distinction from eroding.

Hermetic: no Docker, no network. The sandbox path runs against
:class:`ScriptedSandbox`, which proves the commands we issue and their order.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from taste.execution import ExecResult, ScriptedSandbox
from taste.memory import Memory
from taste.replay import (
    LocalWorktreeExecutor,
    Probe,
    Replayer,
    SandboxProbeExecutor,
    SuiteProbe,
    SuiteRun,
    reconstruct,
    verdicts_from,
)
from taste.shadow import ShadowLog


def _parse_pairs(log: str) -> dict[str, str]:
    """A stand-in runner grammar: ``STATUS test_id`` per line."""
    out: dict[str, str] = {}
    for line in log.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[1].strip()] = parts[0].strip()
    return out


SUITE = SuiteProbe(
    name="p2p::demo",
    command="run-tests",
    members=("t_a", "t_b", "t_c"),
    parse=_parse_pairs,
)


# ------------------------------------------------------------------ verdicts


def test_each_member_gets_its_own_verdict() -> None:
    """An aggregate says only that something broke — not which thing, which
    makes it impossible to match against the Monitor's coverage."""
    run = verdicts_from(SUITE, ExecResult(1, "PASSED t_a\nFAILED t_b\nSKIPPED t_c", ""))
    assert run.statuses == {"t_a": "pass", "t_b": "fail", "t_c": "skip"}
    assert run.infra_error is None


def test_a_test_the_runner_never_mentioned_is_a_hole_not_a_failure() -> None:
    """It can go missing because collection crashed or the runner renamed it.
    Neither is evidence the test broke."""
    run = verdicts_from(SUITE, ExecResult(1, "PASSED t_a\nFAILED t_b", ""))
    assert run.statuses["t_c"] == "error"


def test_an_expected_failure_is_not_a_regression() -> None:
    """XFAIL is a passing outcome for the suite. Counting it as a failure
    invents a regression at every observation on repos that use it."""
    run = verdicts_from(SUITE, ExecResult(0, "XFAIL t_a\nPASSED t_b\nPASSED t_c", ""))
    assert run.statuses["t_a"] == "pass"


def test_a_log_that_parses_to_nothing_is_infrastructure_not_mass_failure() -> None:
    """This is the exact shape of the django defect: the wrong runner emits a
    usage error, nothing parses, and 'everything failed' would report a
    contamination event at every observation of the instance."""
    run = verdicts_from(SUITE, ExecResult(4, "usage: pytest [options]", ""))
    assert run.infra_error == "no results for any graded test"
    assert set(run.statuses.values()) == {"error"}


def test_a_timeout_is_a_hole_for_every_member() -> None:
    run = verdicts_from(SUITE, ExecResult(124, "", "", timed_out=True))
    assert run.infra_error == "timed out"
    assert set(run.statuses.values()) == {"error"}


def test_a_parser_that_raises_does_not_take_the_scan_down() -> None:
    def explode(_log: str) -> dict[str, str]:
        raise RuntimeError("bad grammar")

    suite = SuiteProbe(name="s", command="c", members=("t_a",), parse=explode)
    run = verdicts_from(suite, ExecResult(0, "whatever", ""))
    assert run.statuses["t_a"] == "error"
    assert "parse failed" in (run.infra_error or "")


def test_exit_code_semantics_still_apply_without_a_parser() -> None:
    """What Gate 0's single-assertion probes rely on."""
    suite = SuiteProbe(name="s", command="c", members=("only",))
    assert verdicts_from(suite, ExecResult(0, "", "")).statuses["only"] == "pass"
    assert verdicts_from(suite, ExecResult(1, "", "")).statuses["only"] == "fail"


def test_a_plain_probe_round_trips_as_a_one_member_suite() -> None:
    """Gate 0 and the baseline probes keep working unchanged."""
    suite = Probe(name="check", command="pytest", timeout=30).as_suite()
    assert suite.members == ("check",)
    assert suite.parse is None
    assert suite.timeout == 30


# ------------------------------------------------------------------ sandbox


def _repo(tmp_path: Path) -> tuple[Path, Memory, str]:
    ws = tmp_path / "repo"
    ws.mkdir()
    (ws / "src.py").write_text("x = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t.co", "commit", "-qm", "base"],
        cwd=ws, check=True,
    )
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ws, capture_output=True, text=True
    ).stdout.strip()
    return ws, Memory.open_session(ws, "s"), base


def test_the_tree_is_patched_into_the_image_in_the_right_order(tmp_path: Path) -> None:
    """Mounting a checkout over /testbed would replace a prepared environment
    with an unprepared one. Patching keeps what the image installed."""
    _ws, memory, base = _repo(tmp_path)
    sandbox = ScriptedSandbox().on("run-tests", ExecResult(0, "PASSED t_a", ""))
    executor = SandboxProbeExecutor(sandbox, memory, base)

    executor.run(base, SuiteProbe(name="s", command="run-tests", members=("t_a",),
                                  parse=_parse_pairs))

    assert "/tmp/taste.diff" in sandbox.files
    joined = " || ".join(sandbox.commands)
    assert joined.index("git checkout") < joined.index("run-tests")
    assert "git clean" in joined


def test_a_patch_that_will_not_apply_is_infrastructure_never_failure(
    tmp_path: Path,
) -> None:
    """The single most dangerous confusion available to this pipeline."""
    ws, memory, base = _repo(tmp_path)
    (ws / "src.py").write_text("x = 2\n")
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t.co", "commit", "-qm", "next"],
        cwd=ws, check=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ws, capture_output=True, text=True
    ).stdout.strip()

    sandbox = ScriptedSandbox().on("git apply", ExecResult(1, "", "patch does not apply"))
    run = SandboxProbeExecutor(sandbox, memory, base).run(
        head, SuiteProbe(name="s", command="run-tests", members=("t_a", "t_b"))
    )

    assert run.infra_error is not None
    assert set(run.statuses.values()) == {"error"}
    assert "fail" not in run.statuses.values()


def test_a_reset_failure_is_also_a_hole(tmp_path: Path) -> None:
    _ws, memory, base = _repo(tmp_path)
    sandbox = ScriptedSandbox().on("git checkout", ExecResult(1, "", "cannot reset"))
    run = SandboxProbeExecutor(sandbox, memory, base).run(
        base, SuiteProbe(name="s", command="run-tests", members=("t_a",))
    )
    assert "reset failed" in (run.infra_error or "")
    assert run.statuses["t_a"] == "error"


def test_closing_the_executor_closes_the_sandbox(tmp_path: Path) -> None:
    _ws, memory, base = _repo(tmp_path)
    sandbox = ScriptedSandbox()
    SandboxProbeExecutor(sandbox, memory, base).close()
    assert sandbox.closed


# ------------------------------------------------------------------ scanning


class _RecordingExecutor:
    """Counts preparations, so tree-major iteration is observable."""

    def __init__(self, series: dict[str, dict[str, str]]) -> None:
        self.series = series
        self.calls: list[str] = []

    def run(self, sha: str, suite: SuiteProbe) -> SuiteRun:
        self.calls.append(sha)
        statuses = self.series.get(sha, {})
        return SuiteRun(statuses={m: statuses.get(m, "error") for m in suite.members})

    def close(self) -> None:
        return None


def _timeline(tmp_path: Path, n: int) -> tuple[Memory, list]:
    ws = tmp_path / "repo"
    ws.mkdir()
    (ws / "f.py").write_text("v = 0\n")
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t.co", "commit", "-qm", "base"],
        cwd=ws, check=True,
    )
    memory = Memory.open_session(ws, "scan")
    log = ShadowLog(memory, gitdir=Path(memory.repo.git_dir) / "taste", session="scan")
    for i in range(n):
        (ws / "f.py").write_text(f"v = {i + 1}\n")
        log.observe(step_id="s", attempt=1, trigger="worker")
    return memory, list(log.timeline())


def test_one_preparation_per_tree_not_per_test(tmp_path: Path) -> None:
    """The cost argument for scanning exhaustively rather than bisecting."""
    memory, timeline = _timeline(tmp_path, 4)
    suite = SuiteProbe(
        name="s", command="c", members=("t_a", "t_b", "t_c"), parse=_parse_pairs
    )
    executor = _RecordingExecutor({c.sha: {"t_a": "pass"} for c in timeline})

    replayer = Replayer(memory, [suite], executor=executor)
    replayer.matrix(timeline, suite)

    assert len(executor.calls) == len(timeline), "must not scale with member count"
    assert replayer.replays == len(timeline)


def test_members_regress_independently(tmp_path: Path) -> None:
    """Two tests in one suite, one breaking and one not — the reason a single
    aggregate verdict is not enough."""
    memory, timeline = _timeline(tmp_path, 4)
    shas = [c.sha for c in timeline]
    series = {
        shas[0]: {"t_a": "pass", "t_b": "pass"},
        shas[1]: {"t_a": "pass", "t_b": "pass"},
        shas[2]: {"t_a": "fail", "t_b": "pass"},   # only t_a breaks
        shas[3]: {"t_a": "fail", "t_b": "pass"},
    }
    suite = SuiteProbe(name="s", command="c", members=("t_a", "t_b"), parse=_parse_pairs)

    report = reconstruct(
        memory, timeline, [suite], session="scan",
        executor=_RecordingExecutor(series),
    )

    assert [e.probe for e in report.episodes] == ["t_a"]
    assert report.episodes[0].onset_seq == timeline[2].seq


def test_holes_do_not_open_or_close_an_episode(tmp_path: Path) -> None:
    memory, timeline = _timeline(tmp_path, 4)
    shas = [c.sha for c in timeline]
    series = {
        shas[0]: {"t_a": "pass"},
        shas[1]: {},                # hole
        shas[2]: {"t_a": "pass"},
        shas[3]: {"t_a": "pass"},
    }
    suite = SuiteProbe(name="s", command="c", members=("t_a",), parse=_parse_pairs)
    report = reconstruct(
        memory, timeline, [suite], session="scan", executor=_RecordingExecutor(series)
    )
    assert report.contamination_events == 0


def test_the_executor_is_injectable_so_replay_never_forces_a_worktree(
    tmp_path: Path,
) -> None:
    """Without this seam a container executor cannot be reached at all."""
    memory, _timeline_unused = _timeline(tmp_path, 2)
    replayer = Replayer(memory, [SUITE], executor=_RecordingExecutor({}))
    assert not isinstance(replayer.executor, LocalWorktreeExecutor)


def test_the_default_executor_is_the_local_one(tmp_path: Path) -> None:
    memory, _unused = _timeline(tmp_path, 1)
    assert isinstance(Replayer(memory, [SUITE]).executor, LocalWorktreeExecutor)


@pytest.mark.parametrize("bad", ["", "no results here"])
def test_an_unusable_log_never_yields_a_regression(bad: str) -> None:
    """Belt and braces on the defect class that motivated all of this."""
    run = verdicts_from(SUITE, ExecResult(4, bad, ""))
    assert "fail" not in set(run.statuses.values())


def test_an_xfail_reason_suffix_becomes_a_hole_not_a_regression() -> None:
    """The one upstream hazard that could manufacture our dependent variable.

    The vendored parser (upstream `main`) keys a pytest line by everything
    after the status, so an xfail whose reason prints on the same line keys as
    ``"<id> - reason: ..."`` and no longer matches its PASS_TO_PASS id. Under
    the official grader that reads as a regression. Here it cannot, because a
    member the parser did not mention is `error`.

    The cost is a lost observation rather than a false event, and that is the
    right side to fail on. The rate is worth counting in the first real sweep
    on astropy / scikit-learn / sphinx, since it bounds coverage there.
    """
    from taste.benchmarks import swebench_log

    log = (
        "PASSED pkg/test_x.py::test_ok\n"
        "XFAIL pkg/test_x.py::test_flaky - reason: upstream dask bug\n"
    )
    suite = SuiteProbe(
        name="s",
        command="c",
        members=("pkg/test_x.py::test_ok", "pkg/test_x.py::test_flaky"),
        parse=lambda body: swebench_log.parse("parse_log_pytest_v2", body),
    )
    run = verdicts_from(suite, ExecResult(1, log, ""))

    assert run.statuses["pkg/test_x.py::test_ok"] == "pass"
    assert run.statuses["pkg/test_x.py::test_flaky"] == "error", (
        "an unmatched id must be a hole; 'fail' here would fabricate an event"
    )
