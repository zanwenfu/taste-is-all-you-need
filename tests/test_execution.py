"""The execution seam.

Two properties carry real weight here. First, that a sandbox never converts an
infrastructure failure into a test verdict — that conversion is how an
instrument fabricates the phenomenon it measures. Second, that the harness's
own environment, including its API keys, does not follow a probe command into
execution.

Hermetic: no Docker daemon, no network. The Docker path is exercised against a
fake client, which proves the wire calls we make and nothing about whether a
real container would answer them.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from taste.execution import (
    DEFAULT_ENV_ALLOWLIST,
    DockerProvider,
    DockerSandbox,
    ExecResult,
    LocalProvider,
    LocalSandbox,
    Sandbox,
    ScriptedSandbox,
)

# ------------------------------------------------------------------ results


def test_output_interleaves_both_streams() -> None:
    """Django reports results on stderr; a parser reading stdout alone sees
    an empty log and would grade every test as absent."""
    assert ExecResult(0, "out", "err").output == "out\nerr"
    assert ExecResult(0, "", "err").output == "err"
    assert ExecResult(0, "out", "").output == "out"


def test_ok_is_false_on_timeout_even_with_exit_zero() -> None:
    assert not ExecResult(0, "", "", timed_out=True).ok


# ------------------------------------------------------------------ local


def test_the_harness_environment_does_not_follow_the_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe command is built from dataset content and runs whatever the
    tree contains. Handing it our environment leaks the key and points it at
    the wrong interpreter."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-appear")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-nor-this")

    result = LocalSandbox(tmp_path).exec("env")

    assert result.ok
    assert "should-never-appear" not in result.output
    assert "ANTHROPIC_API_KEY" not in result.output
    assert "OPENAI_API_KEY" not in result.output


def test_the_allowlist_is_what_a_shell_needs_and_no_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("SECRET_THING", "leak")
    result = LocalSandbox(tmp_path).exec("env")
    assert "SECRET_THING" not in result.output
    assert "PATH" in DEFAULT_ENV_ALLOWLIST


def test_explicit_env_is_still_honoured(tmp_path: Path) -> None:
    """The eval script needs LANG and friends set deliberately."""
    result = LocalSandbox(tmp_path).exec("echo $TASTE_MARKER", env={"TASTE_MARKER": "x1"})
    assert "x1" in result.stdout


def test_a_timeout_is_a_result_not_an_exception(tmp_path: Path) -> None:
    """A raising timeout would abort the scan of an entire run. It is one
    missing observation, and the caller decides what that means."""
    result = LocalSandbox(tmp_path).exec("sleep 5", timeout=1)
    assert result.timed_out
    assert not result.ok


def test_a_nonzero_exit_is_reported_not_interpreted(tmp_path: Path) -> None:
    result = LocalSandbox(tmp_path).exec("exit 3")
    assert result.exit_code == 3
    assert not result.timed_out  # a failure, not a hole


def test_text_round_trips(tmp_path: Path) -> None:
    sandbox = LocalSandbox(tmp_path)
    sandbox.put_text("nested/dir/patch.diff", "diff --git a/x b/x\n")
    assert sandbox.get_text("nested/dir/patch.diff") == "diff --git a/x b/x\n"
    assert (tmp_path / "nested" / "dir" / "patch.diff").exists()


def test_local_provider_isolates_by_key(tmp_path: Path) -> None:
    provider = LocalProvider(tmp_path)
    a = provider.open(key="inst-a")
    b = provider.open(key="inst-b")
    a.put_text("f.txt", "A")
    b.put_text("f.txt", "B")
    assert a.get_text("f.txt") == "A"
    assert b.get_text("f.txt") == "B"


def test_local_sandbox_satisfies_the_protocol(tmp_path: Path) -> None:
    assert isinstance(LocalSandbox(tmp_path), Sandbox)


# ------------------------------------------------------------------ scripted


def test_scripted_records_every_command_in_order() -> None:
    sandbox = ScriptedSandbox()
    sandbox.exec("git apply /tmp/x.diff")
    sandbox.exec("./tests/runtests.py queries")
    assert sandbox.commands == ["git apply /tmp/x.diff", "./tests/runtests.py queries"]


def test_scripted_matches_by_substring_in_insertion_order() -> None:
    sandbox = ScriptedSandbox().on("git apply", ExecResult(1, "", "conflict"))
    assert sandbox.exec("cd /testbed && git apply -v /tmp/p.diff").exit_code == 1
    assert sandbox.exec("pytest").exit_code == 0  # falls through to default


def test_scripted_satisfies_the_protocol() -> None:
    assert isinstance(ScriptedSandbox(), Sandbox)


# ------------------------------------------------------------------ docker


class _FakeContainer:
    def __init__(self) -> None:
        self.execs: list[dict[str, object]] = []
        self.archives: list[tuple[str, bytes]] = []
        self.removed = False
        self.next_result: tuple[int, tuple[bytes, bytes]] = (0, (b"ok", b""))
        self.workdir_ok = 0

    def exec_run(self, cmd, workdir=None, environment=None, demux=False):
        self.execs.append({"cmd": cmd, "workdir": workdir, "environment": environment})
        # The workdir probe is issued once at open and is not the command
        # under test; a fake that answered it with the canned result would
        # make every timeout/exit-code test fail at construction instead.
        if isinstance(cmd, list) and cmd[:2] == ["test", "-d"]:
            return (self.workdir_ok, (b"", b""))
        return self.next_result

    def put_archive(self, path, data):
        self.archives.append((path, data))
        return True

    def remove(self, force=False):
        self.removed = True


class _FakeContainers:
    def __init__(self) -> None:
        self.container = _FakeContainer()
        self.run_kwargs: dict[str, object] = {}

    def run(self, image, **kwargs):
        self.run_kwargs = {"image": image, **kwargs}
        return self.container

    def get(self, name):
        raise KeyError(name)


class _FakeClient:
    def __init__(self) -> None:
        self.containers = _FakeContainers()


def test_the_container_is_pinned_to_x86_and_severed_from_the_network() -> None:
    """The images are x86_64-first, and a probe that can reach the network
    could resolve a different dependency set than the image pins."""
    client = _FakeClient()
    DockerSandbox(image="swebench/x@sha256:abc", name="t", client=client)

    kwargs = client.containers.run_kwargs
    assert kwargs["platform"] == "linux/amd64"
    assert kwargs["network_mode"] == "none", (
        "network_disabled removes loopback too, which kills any test plugin "
        "that binds a localhost socket -- 23.5% of the dev-slice oracle"
    )
    assert "network_disabled" not in kwargs
    assert kwargs["command"] == "sleep infinity"
    assert kwargs["detach"] is True


def test_timeouts_are_enforced_inside_the_container() -> None:
    """Docker cannot cancel a running exec. A host-side timer would return
    while the command kept writing to the tree, corrupting the next
    observation."""
    client = _FakeClient()
    sandbox = DockerSandbox(image="i", name="t", client=client)
    sandbox.exec("pytest -q", timeout=42)

    issued = client.containers.container.execs[-1]["cmd"][-1]  # [0] is the workdir probe
    assert "timeout --signal=KILL 42" in issued
    assert "pytest -q" in issued


def test_the_timeout_exit_code_is_reported_as_a_timeout() -> None:
    client = _FakeClient()
    client.containers.container.next_result = (124, (b"", b""))
    sandbox = DockerSandbox(image="i", name="t", client=client)
    assert sandbox.exec("sleep 999", timeout=1).timed_out


def test_a_patch_is_shipped_as_an_archive_not_through_the_shell() -> None:
    """Payloads are unified diffs of arbitrary repository content; shell
    quoting that correctly for every instance is not achievable."""
    client = _FakeClient()
    sandbox = DockerSandbox(image="i", name="t", client=client)
    sandbox.put_text("/tmp/taste.diff", "diff --git a/f b/f\n+x\n")

    path, blob = client.containers.container.archives[0]
    assert path == "/tmp"
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        member = archive.next()
        assert member is not None and member.name == "taste.diff"
        extracted = archive.extractfile(member)
        assert extracted is not None
        assert extracted.read().decode() == "diff --git a/f b/f\n+x\n"


def test_close_removes_the_container() -> None:
    client = _FakeClient()
    sandbox = DockerSandbox(image="i", name="t", client=client)
    sandbox.close()
    assert client.containers.container.removed


def test_the_provider_reuses_one_container_per_key() -> None:
    """One start per instance is affordable; one per observation is not."""
    client = _FakeClient()
    provider = DockerProvider(client=client)
    first = provider.open(key="django__django-1", image="img")
    second = provider.open(key="django__django-1", image="img")
    assert first is second


def test_the_provider_refuses_an_unpinned_image() -> None:
    with pytest.raises(ValueError, match="no image pinned"):
        DockerProvider(client=_FakeClient()).open(key="x")


def test_importing_this_module_does_not_require_docker() -> None:
    """The development machine has no docker package; the hermetic suite must
    still import and run."""
    import sys

    assert "docker" not in sys.modules


def test_a_missing_workdir_fails_loudly_at_open() -> None:
    """Found by running against a real daemon for the first time.

    Docker resolves the exec working directory at start, so a missing workdir
    makes every later command exit 127 with the OCI error on *stdout*.
    Downstream that classifies as "no results for any graded test" -- correct,
    but discovered far too late: the whole instance replays as holes and
    scores as a clean run.
    """
    client = _FakeClient()
    client.containers.container.workdir_ok = 1  # `test -d` reports missing

    with pytest.raises(RuntimeError, match="does not exist in image"):
        DockerSandbox(image="i", name="t", client=client)

    assert client.containers.container.removed, "a failed open must not leak the container"


def test_the_workdir_is_checked_before_any_command_runs() -> None:
    client = _FakeClient()
    DockerSandbox(image="i", name="t", workdir="/testbed", client=client)
    first = client.containers.container.execs[0]["cmd"]
    assert first == ["test", "-d", "/testbed"]
