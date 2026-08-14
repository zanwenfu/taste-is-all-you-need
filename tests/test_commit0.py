"""The Commit0 adapter, and the held-out oracle it bolts on.

Commit0's own tests are visible by design and are also its score. A harness
whose Monitor runs the scoring suite has made "monitor passed" and "task
solved" the same event, which makes any success claim unfalsifiable. These
tests pin the split that fixes it — and, more importantly, pin every route by
which the hidden half could leak back to the agent.

Built against a synthetic library rather than a downloaded one, so the suite
stays hermetic and offline.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from taste.benchmarks.commit0 import (
    LITE,
    Commit0Task,
    choose_holdout,
    discover_tests,
    fractional_score,
    materialize,
)
from taste.memory import Memory
from taste.replay import Replayer


@pytest.fixture
def fake_commit0(tmp_path: Path) -> Path:
    """A source_root shaped like `commit0 setup` output.

    Four test files of different sizes, mirroring the real distribution: a
    coarse suite is what makes the split arithmetic interesting.
    """
    repo = tmp_path / "source" / "repos" / "widget"
    (repo / "widget").mkdir(parents=True)
    (repo / "tests").mkdir()

    (repo / "widget" / "__init__.py").write_text("from .core import value, name\n")
    (repo / "widget" / "core.py").write_text(
        'def value():\n    """Return the answer."""\n    pass\n\n\n'
        'def name():\n    """Return the name."""\n    pass\n'
    )
    sizes = {"test_core.py": 12, "test_extra.py": 6, "test_edge.py": 3, "test_tiny.py": 1}
    for filename, count in sizes.items():
        body = "from widget import value, name\n\n\n" + "\n\n".join(
            f"def test_{filename[:-3]}_{i}():\n    assert value() == 42" for i in range(count)
        )
        (repo / "tests" / filename).write_text(body + "\n")

    # A .git that must NOT survive into the workspace — its history would
    # carry the held-out tests.
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t.co", "commit", "-qm", "orig"],
        cwd=repo,
        check=True,
    )
    return tmp_path / "source"


def _task(fake_commit0: Path, tmp_path: Path, **kwargs) -> Commit0Task:
    return materialize(
        "widget", source_root=fake_commit0, dest=tmp_path / "work", **kwargs
    )


# ------------------------------------------------------------------ discovery


def test_discovers_every_test_file_and_counts(fake_commit0: Path, tmp_path: Path) -> None:
    task = _task(fake_commit0, tmp_path)
    assert task.visible_count + task.holdout_count == 22


def test_lite_split_names_the_sixteen_libraries() -> None:
    assert len(LITE) == 16
    assert "wcwidth" in LITE and "marshmallow" in LITE


# ------------------------------------------------------------------ the split


def test_holdout_is_near_the_target_fraction(fake_commit0: Path, tmp_path: Path) -> None:
    """Greedy selection must not overshoot.

    Admitting a file merely because the running total is still under target
    blows past it when the next file is large — on a real 4-file suite that
    turned a 30% target into 67% held out.
    """
    task = _task(fake_commit0, tmp_path, holdout_fraction=0.30)
    total = task.visible_count + task.holdout_count
    assert 0.05 <= task.holdout_count / total <= 0.40


def test_a_visible_signal_always_remains(fake_commit0: Path, tmp_path: Path) -> None:
    """The agent needs feedback and the Monitor needs something to run."""
    for fraction in (0.3, 0.6, 0.95):
        task = materialize(
            "widget",
            source_root=fake_commit0,
            dest=tmp_path / f"w{fraction}",
            holdout_fraction=fraction,
        )
        assert task.visible_count > 0, f"nothing visible at fraction={fraction}"
        assert task.visible_tests


def test_the_split_is_deterministic_for_a_seed(fake_commit0: Path, tmp_path: Path) -> None:
    a = materialize("widget", source_root=fake_commit0, dest=tmp_path / "a", seed=7)
    b = materialize("widget", source_root=fake_commit0, dest=tmp_path / "b", seed=7)
    assert a.holdout_tests == b.holdout_tests
    assert a.split_id == b.split_id


def test_a_different_seed_redraws_the_split(fake_commit0: Path, tmp_path: Path) -> None:
    """Re-drawable on purpose: a conclusion that depends on which tests were
    hidden is a conclusion about the split."""
    ids = {
        materialize(
            "widget", source_root=fake_commit0, dest=tmp_path / f"s{seed}", seed=seed
        ).split_id
        for seed in range(6)
    }
    assert len(ids) > 1


def test_choose_holdout_never_takes_everything() -> None:
    from taste.benchmarks.commit0 import TestFile

    files = [TestFile("only.py", 10)]
    assert choose_holdout(files, fraction=0.9, seed=0) == []


def test_choose_holdout_falls_back_to_the_smallest_file() -> None:
    """A coarse suite can admit nothing under the ceiling; an empty oracle is
    worse than a small one."""
    from taste.benchmarks.commit0 import TestFile

    files = [TestFile("big.py", 50), TestFile("small.py", 9)]
    chosen = choose_holdout(files, fraction=0.05, seed=0)
    assert [f.path for f in chosen] == ["small.py"]


# ------------------------------------------------------------------ the seal


def test_held_out_tests_are_not_on_disk(fake_commit0: Path, tmp_path: Path) -> None:
    task = _task(fake_commit0, tmp_path)
    for path in task.holdout_tests:
        assert not (task.workspace / path).exists(), f"{path} still readable"


def test_held_out_tests_are_not_recoverable_from_git(
    fake_commit0: Path, tmp_path: Path
) -> None:
    """The leak that mattered.

    Deleting a file from the working tree leaves it at ``git show HEAD:path``
    — and read-only git is deliberately allowed to workers, so the oracle was
    one command away. The workspace is therefore a fresh repository whose
    first commit is the already-split state.
    """
    task = _task(fake_commit0, tmp_path)

    def git(args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            f"git {args}", shell=True, cwd=task.workspace, capture_output=True, text=True
        )

    for path in task.holdout_tests:
        assert git(f"show HEAD:{path}").returncode != 0, f"{path} readable at HEAD"
        assert not git(f"log --all --oneline -- {path}").stdout.strip(), f"{path} in history"

    # Nothing in the object store mentions them either.
    objects = git("rev-list --objects --all").stdout
    for path in task.holdout_tests:
        assert path not in objects


def test_the_original_history_is_not_carried_over(fake_commit0: Path, tmp_path: Path) -> None:
    task = _task(fake_commit0, tmp_path)
    log = subprocess.run(
        "git log --oneline", shell=True, cwd=task.workspace, capture_output=True, text=True
    )
    assert len(log.stdout.strip().splitlines()) == 1, "exactly one starting commit"


def test_visible_tests_remain_readable(fake_commit0: Path, tmp_path: Path) -> None:
    task = _task(fake_commit0, tmp_path)
    assert task.visible_tests
    for path in task.visible_tests:
        assert (task.workspace / path).exists()


# ------------------------------------------------------------------ scoring


def test_the_scorer_can_still_run_the_held_out_tests(
    fake_commit0: Path, tmp_path: Path
) -> None:
    """Sealed from the agent, reachable by the scorer — both must hold."""
    task = _task(fake_commit0, tmp_path)
    memory = Memory.open_session(task.workspace, "score")
    replayer = Replayer(memory, list(task.probes))

    for probe in task.probes:
        # Everything is stubbed, so a held-out test must actually FAIL —
        # not error, which would mean the probe never ran.
        assert replayer.verdict_at(memory.head().sha, probe) == "fail"


def test_fractional_score_is_zero_on_the_stub(fake_commit0: Path, tmp_path: Path) -> None:
    """The floor, so the full dynamic range is available to an arm."""
    task = _task(fake_commit0, tmp_path)
    memory = Memory.open_session(task.workspace, "frac")
    assert fractional_score(memory, memory.head().sha, task) == 0.0


def test_fractional_score_rises_with_a_real_implementation(
    fake_commit0: Path, tmp_path: Path
) -> None:
    """Fractional rather than binary: a from-scratch library is rarely fully
    solved, and a binary endpoint would compress every arm onto one value."""
    task = _task(fake_commit0, tmp_path)
    memory = Memory.open_session(task.workspace, "solve")
    (task.workspace / "widget" / "core.py").write_text(
        "def value():\n    return 42\n\n\ndef name():\n    return 'widget'\n"
    )
    solved = memory.checkpoint("step-01", "implement it")

    assert fractional_score(memory, solved.sha, task) == 1.0


def test_probing_leaves_the_workspace_untouched(fake_commit0: Path, tmp_path: Path) -> None:
    """A restored probe file must not leak into the agent's tree."""
    task = _task(fake_commit0, tmp_path)
    memory = Memory.open_session(task.workspace, "clean")
    fractional_score(memory, memory.head().sha, task)

    for path in task.holdout_tests:
        assert not (task.workspace / path).exists(), f"{path} leaked back in"
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=task.workspace, capture_output=True, text=True
    )
    assert porcelain.stdout.strip() == ""


# ------------------------------------------------------------------ provenance


def test_split_id_distinguishes_otherwise_identical_tasks(
    fake_commit0: Path, tmp_path: Path
) -> None:
    """Two runs on 'the same library' with different holdouts are not the
    same task, and the manifest has to say so."""
    a = materialize("widget", source_root=fake_commit0, dest=tmp_path / "a", seed=1)
    b = materialize("widget", source_root=fake_commit0, dest=tmp_path / "b", seed=2)
    if a.holdout_tests != b.holdout_tests:
        assert a.split_id != b.split_id


def test_manifest_records_the_whole_split(fake_commit0: Path, tmp_path: Path) -> None:
    task = _task(fake_commit0, tmp_path)
    manifest = task.manifest()
    assert manifest["benchmark"] == "commit0"
    assert manifest["library"] == "widget"
    assert manifest["holdout_count"] > 0
    assert set(manifest["holdout_tests"]) == set(task.holdout_tests)


def test_task_text_tells_the_agent_not_to_edit_tests(
    fake_commit0: Path, tmp_path: Path
) -> None:
    task = _task(fake_commit0, tmp_path)
    assert "Do not" in task.task_text and "tests" in task.task_text


def test_missing_library_fails_with_a_useful_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="commit0 setup"):
        materialize("nope", source_root=tmp_path, dest=tmp_path / "w")


def test_discover_tests_ignores_vcs_and_build_dirs(fake_commit0: Path, tmp_path: Path) -> None:
    task = _task(fake_commit0, tmp_path)
    found = discover_tests(task.workspace)
    assert all(".git" not in f.path for f in found)
