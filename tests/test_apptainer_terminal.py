"""The Apptainer port's transport and replay checks, on made-up files and a stand-in ``apptainer`` (no container)."""

from __future__ import annotations

import os
import stat
import textwrap
import time

import pytest

from taste.benchmarks import apptainer_replay as replay
from taste.brains import apptainer_terminal as at
from taste.brains.terminal_broker import TerminalRequest

STUB = textwrap.dedent("""\
    #!/bin/sh
    # Stand-in for `apptainer exec ... --pwd CWD instance://NAME CMD...`: runs CMD in CWD on this host.
    [ "$1" = exec ] || exit 0
    shift
    while [ $# -gt 0 ]; do
      case "$1" in
        --pwd) cwd=$2; shift 2 ;;
        instance://*) shift; break ;;
        *) shift ;;
      esac
    done
    cd "$cwd" || exit 255
    exec "$@"
    """)


def _spec(tmp_path, **changes):
    values = dict(image=str(tmp_path / "image.sif"), overlay=str(tmp_path / "overlay"), name="replay-test",
                  env=(("PATH", os.environ["PATH"]), ("HOME", "/root")))
    values.update(changes)
    return at.ApptainerSpec(**values)


@pytest.fixture
def backend(tmp_path):
    stub = tmp_path / "apptainer"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    item = at.ApptainerTerminalBackend(_spec(tmp_path, apptainer=str(stub)), deadline_unix=time.time() + 600,
                                       output_limit=64)
    item.started = True
    return item


def test_start_runs_root_mapped_without_the_fakeroot_wrapper_and_offline(tmp_path):
    argv = at.start_argv(_spec(tmp_path, binds=(("/x/agent", "/logs/agent"),)))
    assert argv[:5] == ["apptainer", "instance", "start", "--fakeroot", "--ignore-fakeroot-command"]
    assert argv[argv.index("--no-mount") + 1] == "home,tmp,cwd" and "--no-home" in argv and "--cleanenv" in argv
    assert argv[argv.index("--network") + 1] == "none" and "--net" in argv
    assert argv[argv.index("--bind") + 1] == "/x/agent:/logs/agent"
    assert argv[-2:] == [str(tmp_path / "image.sif"), "replay-test"]
    online = at.start_argv(_spec(tmp_path, network="host"))
    assert "--net" not in online and "--network" not in online


def test_exec_gives_exactly_dockers_environment(tmp_path):
    argv = at.exec_argv(_spec(tmp_path, env=(("PATH", "/bin"), ("HOME", "/root"))), "/app", "ls")
    assert argv[argv.index("--pwd") + 1] == "/app"
    assert argv[argv.index("instance://replay-test") + 1:] == ["/usr/bin/env", "-i", "PATH=/bin", "HOME=/root",
                                                                "/bin/sh", "-c", "ls"]


def test_docker_env_is_the_images_then_hostname_and_home():
    env = dict(at.docker_env(["PATH=/usr/bin", "LANG=C.UTF-8", "PATH=/ignored"], "abc123def456"))
    assert env == {"PATH": "/usr/bin", "LANG": "C.UTF-8", "HOSTNAME": "abc123def456", "HOME": "/root"}
    assert dict(at.docker_env(["HOME=/home/app"], "h"))["HOME"] == "/home/app"
    assert "PATH" in dict(at.docker_env([], "h"))


def test_spec_refuses_another_user_and_a_bad_network(tmp_path):
    with pytest.raises(ValueError, match="one uid"):
        _spec(tmp_path, exec_user="app")
    with pytest.raises(ValueError, match="network"):
        _spec(tmp_path, network="bridge")
    with pytest.raises(ValueError, match="instance name"):
        _spec(tmp_path, name="a b")


def test_execute_captures_streams_codes_and_caps(backend):
    done = backend.execute(TerminalRequest("r1", "actor", "echo out; echo err >&2; exit 3", "/", 30))
    assert (done.return_code, done.stdout, done.stderr, done.terminated) == (3, b"out\n", b"err\n", "")
    big = backend.execute(TerminalRequest("r2", "actor", "head -c 1000 /dev/zero", "/", 30))
    assert len(big.stdout) == 64 and big.stdout_dropped_bytes == 936


def test_execute_past_its_time_is_killed_as_docker_reports(backend):
    started = time.monotonic()
    done = backend.execute(TerminalRequest("r3", "actor", "sleep 30; echo late", "/", 1))
    assert done.terminated == "timeout" and done.return_code == at.KILLED and b"late" not in done.stdout
    assert time.monotonic() - started < 15


def test_run_feeds_standard_input(backend):
    done = backend.run("cat", "/", 30, stdin=b"through stdin")
    assert done.stdout == b"through stdin" and done.code == 0


def test_a_stopped_instance_runs_nothing(backend):
    backend.started = False
    with pytest.raises(at.TerminalFenced):
        backend.run("true", "/", 5)


def _upper(tmp_path):
    upper = tmp_path / "overlay" / "upper"
    (upper / "app" / "src").mkdir(parents=True)
    (upper / "app" / "src" / "edited.js").write_text("x")             # changed: the image has it
    (upper / "app" / "newdir" / "deep").mkdir(parents=True)            # added: a new folder, with its marker
    (upper / "app" / "newdir" / at.OPAQUE_MARK).write_text("")
    (upper / "app" / "newdir" / "deep" / "a.py").write_text("y")
    (upper / "app" / ".wh.gone.txt").write_text("")                    # deleted (a whiteout file)
    (upper / "app" / "replaced").mkdir()                               # replaced: deleted, then made again
    (upper / "app" / "replaced" / at.OPAQUE_MARK).write_text("")
    (upper / "app" / "replaced" / "new").write_text("z")
    (upper / "app" / ".git").mkdir()
    (upper / "app" / ".git" / "index").write_text("i")
    (upper / "logs" / "agent").mkdir(parents=True)                     # Harbor's mount, under an added /logs
    (upper / "logs" / "agent" / "trajectory.json").write_text("{}")
    (upper / "proc").mkdir()
    (upper / "proc" / "x").write_text("")
    (upper / "usr" / "bin").mkdir(parents=True)                        # a folder the runtime made: changed dir
    return upper


LOWER = {"/app": "d", "/app/src": "d", "/app/src/edited.js": "f", "/app/replaced": "d", "/app/.git": "d",
         "/app/.git/index": "f", "/usr": "d", "/usr/bin": "d", "/proc": "d"}


def test_changes_read_from_the_overlay_in_dockers_kinds(tmp_path):
    entries = list(at.upper_entries(_upper(tmp_path)))
    kinds = dict(((path, kind) for path, kind in at.changes_from(entries, LOWER) if path != "/app/replaced"))
    assert kinds["/app/src/edited.js"] == at.CHANGED and kinds["/app/newdir"] == at.ADDED
    assert kinds["/app/gone.txt"] == at.DELETED and kinds["/logs"] == at.ADDED
    replaced = [kind for path, kind in at.changes_from(entries, LOWER) if path == "/app/replaced"]
    assert replaced == [at.DELETED, at.ADDED]
    assert all(at.OPAQUE_MARK not in path for path, _, _ in entries)


def test_changed_paths_select_as_a_docker_checkpoint_does(tmp_path):
    entries = list(at.upper_entries(_upper(tmp_path)))
    types = {path: kind for path, kind, _ in entries}
    paths = at.select(at.changes_from(entries, LOWER), types, mounts=replay.LOG_MOUNTS)
    assert paths == ["/app/.git/index", "/app/newdir", "/app/replaced", "/app/src/edited.js", "/logs",
                     "/app/gone.txt (deleted)", "/app/replaced (deleted)"]


def test_same_image_takes_a_config_or_manifest_digest():
    image = {"config_digest": "sha256:c", "manifest_digest": "sha256:m", "index_digest": None}
    assert replay.same_image(["sha256:c"], image)["same"] is True
    assert replay.same_image(["sha256:m"], image)["same"] is True
    assert replay.same_image(["sha256:x"], image)["same"] is False
    assert replay.same_image([], image)["same"] is None


def test_judged_as_the_replay_driver_judges():
    recorded = {"reward": 1.0, "changed_paths": ["/app/a.py", "/logs"], "image_ids": ["sha256:c"]}
    summary = {"faithful": True, "reward": 1.0, "cost_usd": 0.0, "changed_paths": ["/app/a.py", "/logs"],
               "image_id": "sha256:c", "checkpoint_partial": False}
    assert replay.judge(recorded, summary, workdir="/app")["kept"] is True
    assert replay.judge(recorded, {**summary, "changed_paths": ["/app/b.py"]}, workdir="/app")["why"] == "files_differ"
    assert replay.judge(recorded, {**summary, "reward": 0.0}, workdir="/app")["why"] == "grade_differs"
    assert replay.judge(recorded, {**summary, "faithful": False}, workdir="/app")["why"] == "unfaithful"


def test_tests_go_in_as_roots_files(tmp_path):
    (tmp_path / "tests" / "sub").mkdir(parents=True)
    (tmp_path / "tests" / "sub" / "t.sh").write_text("echo")
    import io
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(replay._tar_of(tmp_path / "tests"))) as archive:
        members = archive.getmembers()
    assert sorted(member.name for member in members) == ["sub", "sub/t.sh"]
    assert {(member.uid, member.gid) for member in members} == {(0, 0)}


def test_the_in_namespace_walker_reads_the_upper_folder_as_the_host_reader_does(tmp_path):
    import json
    import subprocess
    import sys
    upper = _upper(tmp_path)
    done = subprocess.run([sys.executable, "-c", at.WALKER, str(upper)], capture_output=True, check=True)
    walked = sorted((path, kind, bool(opaque)) for path, kind, opaque in json.loads(done.stdout))
    assert walked == sorted(at.upper_entries(upper))
    argv = at.walk_argv(_spec(tmp_path))
    assert argv[argv.index("--bind") + 1] == f"{upper}:{at.UPPER_MOUNT}:ro" and "--fakeroot" in argv
