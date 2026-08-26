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
    RegressionEpisode,
    Replayer,
    ReplayReport,
    SandboxProbeExecutor,
    SuiteProbe,
    SuiteRun,
    base_test_id,
    collapse_episodes,
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


# ------------------------------------------------------------------ the declared unit


def _episode(probe: str, onset: int) -> RegressionEpisode:
    return RegressionEpisode(probe=probe, onset_seq=onset, onset_sha=f"sha{onset:04d}")


def test_parametrised_variants_of_one_function_are_one_declared_event() -> None:
    """The pre-declared unit is (instance, test function, onset). Counting
    raw ids instead multiplies the primary endpoint by however many
    parameters a suite happens to sweep — one broken function on a
    parametrised suite would report as dozens of events, differentially by
    instance."""
    collapsed = collapse_episodes(
        [
            _episode("tests/test_x.py::test_foo[a]", 3),
            _episode("tests/test_x.py::test_foo[b]", 3),
        ]
    )
    assert len(collapsed) == 1
    assert collapsed[0].members == 2, "the fold must be recorded, not silent"
    assert collapsed[0].probe == "tests/test_x.py::test_foo"


def test_the_same_function_breaking_at_two_onsets_is_two_events() -> None:
    """A regress-recover-regress pattern is two events in every unit; the
    collapse folds parameters, never time."""
    collapsed = collapse_episodes(
        [
            _episode("tests/test_x.py::test_foo[a]", 3),
            _episode("tests/test_x.py::test_foo[b]", 7),
        ]
    )
    assert len(collapsed) == 2
    assert all(c.members == 1 for c in collapsed)


def test_two_functions_breaking_at_one_onset_are_two_events() -> None:
    collapsed = collapse_episodes(
        [
            _episode("tests/test_x.py::test_foo", 3),
            _episode("tests/test_x.py::test_bar", 3),
        ]
    )
    assert len(collapsed) == 2


def test_a_unittest_style_id_is_already_its_own_function() -> None:
    assert base_test_id("test_total (app.tests.MathTests)") == (
        "test_total (app.tests.MathTests)"
    )
    assert base_test_id("tests/test_x.py::test_foo[1-2]") == "tests/test_x.py::test_foo"


def test_the_report_exposes_both_units_and_leaves_the_raw_list_alone() -> None:
    """Attribution joins on raw ids and the verdict matrix is measured per
    raw id, so the collapse must be a view, never a rewrite."""
    raw = [
        _episode("t::test_foo[a]", 2),
        _episode("t::test_foo[b]", 2),
        _episode("t::test_bar", 2),
    ]
    report = ReplayReport(session="s", observations=4, episodes=list(raw))

    assert report.contamination_events == 3
    assert report.contamination_events_declared == 2
    assert report.episodes == raw


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


def test_the_upstream_sha_is_never_used_for_the_local_diff(tmp_path: Path) -> None:
    """The bug that made the first real dry run report nothing.

    The workspace is built with `git archive` so no upstream objects reach it
    -- otherwise an agent could read the fix for its own issue. So the
    benchmark's base_commit is NOT resolvable locally. Diffing against it
    fails with "bad object", which becomes an infra error, which makes every
    graded test a hole, which reports as zero regressions on a clean-looking
    run. All five graded tests came back "never passed".
    """
    _ws, memory, _base = _repo(tmp_path)
    upstream = "22623bd8c265b78b161542663ee980738441c307"  # not in this repo

    executor = SandboxProbeExecutor(ScriptedSandbox(), memory, upstream)

    assert executor.local_base and executor.local_base != upstream, (
        "the local base must default to the workspace root, not the upstream sha"
    )
    run = executor.run(executor.local_base, SuiteProbe(name="s", command="c", members=("t",)))
    assert run.infra_error is None, f"diff should succeed locally, got {run.infra_error}"


def test_the_container_resets_to_the_image_snapshot_not_the_upstream_sha(tmp_path: Path) -> None:
    """This test used to assert the opposite — that the reset uses the
    benchmark's upstream sha — and the behaviour it certified was bug B8:
    images apply tracked-file edits after checking out base_commit, and a
    reset to the upstream sha reverts them before every probe. The contract
    now is: reset to the snapshot baseline of the tree the image ships;
    the upstream sha appears in no reset command."""
    _ws, memory, _base = _repo(tmp_path)
    upstream = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    sandbox = ScriptedSandbox()

    SandboxProbeExecutor(sandbox, memory, upstream).run(
        "HEAD", SuiteProbe(name="s", command="run-tests", members=("t",))
    )

    resets = [c for c in sandbox.commands if "git checkout -q" in c]
    assert resets, "the executor no longer resets the tree at all"
    assert all("scriptedbaseline" in c for c in resets), resets
    assert not any(upstream in c for c in resets), (
        "the reset went back to the upstream sha, reverting the image's own "
        "build-time edits — bug B8"
    )


def test_no_resolvable_local_base_yields_a_hole_not_a_crash(tmp_path: Path) -> None:
    """A workspace with nothing to diff against reports a hole and says why,
    rather than raising through the middle of a scan."""
    _ws, memory, _base = _repo(tmp_path)

    run = SandboxProbeExecutor(ScriptedSandbox(), memory, "abc", local_base="").run(
        "HEAD", SuiteProbe(name="s", command="c", members=("t",))
    )
    # An explicit empty local_base falls back to the root commit, so force the
    # degenerate case the guard exists for.
    executor = SandboxProbeExecutor(ScriptedSandbox(), memory, "abc")
    executor.local_base = ""
    degenerate = executor.run("HEAD", SuiteProbe(name="s", command="c", members=("t",)))

    assert run.infra_error is None or "t" in run.statuses
    assert degenerate.statuses["t"] == "error"
    assert "no local base" in (degenerate.infra_error or "")


def test_the_reset_target_is_the_image_state_not_the_upstream_base(tmp_path: Path) -> None:
    """Bug B8. Images apply tracked-file edits AFTER checking out base_commit
    (sphinx's pre_install rewrites source in place). A per-observation reset
    of `git checkout <base_commit> -- .` reverts those edits, the suite dies
    on the un-edited source at every observation, and the whole family's
    oracle reads as flake. The reset target must be a snapshot of the tree
    the image actually ships."""
    import subprocess as sp

    from taste.execution import LocalSandbox
    from taste.replay import SandboxProbeExecutor

    # The "image": a repo at base_commit, then a pre_install edit on top,
    # deliberately uncommitted — exactly how the image build leaves it.
    bed = tmp_path / "bed"
    bed.mkdir()
    (bed / "lib.py").write_text("PATCHED = False\n")
    def bedgit(*args: str) -> str:
        return sp.run(["git", "-C", str(bed), "-c", "user.name=t", "-c", "user.email=t@l", *args],
                      capture_output=True, text=True).stdout
    bedgit("init", "-q")
    bedgit("add", "-A")
    bedgit("commit", "-qm", "upstream base")
    upstream_base = bedgit("rev-parse", "HEAD").strip()
    (bed / "lib.py").write_text("PATCHED = True\n")  # the pre_install edit

    # The host workspace: independent single-commit repo, as materialize builds.
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "lib.py").write_text("PATCHED = False\n")
    sp.run(["git", "-C", str(ws), "init", "-q"], check=True)
    sp.run(["git", "-C", str(ws), "-c", "user.name=t", "-c", "user.email=t@l",
            "add", "-A"], check=True)
    sp.run(["git", "-C", str(ws), "-c", "user.name=t", "-c", "user.email=t@l",
            "commit", "-qm", "root"], check=True)
    memory = Memory(ws, "main")
    head = memory.repo.head.commit.hexsha

    executor = SandboxProbeExecutor(LocalSandbox(bed), memory, upstream_base)
    suite = Probe(name="pre_install_survives",
                  command="grep -q 'PATCHED = True' lib.py", timeout=30).as_suite()
    first = executor.run(head, suite)
    second = executor.run(head, suite)  # after one reset cycle — the moment B8 bit
    memory.close()

    assert first.infra_error is None, first.infra_error
    assert second.infra_error is None, second.infra_error
    assert set(first.statuses.values()) == {"pass"}
    assert set(second.statuses.values()) == {"pass"}, (
        "the second observation saw the pre_install edit reverted: the reset "
        "target is the upstream base, not the image state"
    )
