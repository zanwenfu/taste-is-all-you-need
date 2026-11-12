"""Only the records a replay needs come back from a trial archive (taste.recovery_study.archive_restore).

The fixture is a small archive made the way the study's kits made theirs: a tar of paths relative to /, in the order
of its sorted file list, compressed with zstd and cut into parts, with SHA256SUMS. No network, no Docker.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from taste.recovery_study import archive_restore as restore

ROOT = "data/taste-trials"
JOURNAL = "agent-state/workspace/.git/taste.azure.terminal-trial.{run}/worker/calls.sqlite3"
LEDGER = "controller/terminal/terminal.sqlite3"
TRAJECTORY = "controller/trajectory.json"
SCRIPT = "branch-script.json"
needs_zstd = pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is not installed")


def cli():
    spec = importlib.util.spec_from_file_location(
        "restore_trial_records_cli", Path(__file__).resolve().parents[1] / "scripts" / "restore_trial_records.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trial_files(token, *, branch=False, journal=True):
    """A trial directory's files as the archive holds them: the four a replay needs, and others it does not."""
    files = {f"{token}/{LEDGER}": b"ledger of " + token.encode(),
             f"{token}/{TRAJECTORY}": json.dumps({"session_id": token}).encode(),
             f"{token}/agent-state/workspace/src/app.py": b"print('work')\n" * 50,
             f"{token}/agent-state/workspace/.git/HEAD": b"ref: refs/heads/main\n",
             f"{token}/controller/run/credentials/taste-azure-goal.json": b'{"key": "not a real key"}',
             f"{token}/agent-state/workspace/.git/other/worker/calls.sqlite3": b"not the worker's journal"}
    if journal:
        files[f"{token}/{JOURNAL.format(run=token[:8])}"] = b"journal of " + token.encode() * 20
    if branch:
        files[f"{token}/{SCRIPT}"] = b'{"schema": "parent script"}'
    return files


def archive(directory, name, trials, *, part_bytes=700):
    """A set of the archive: its parts, its file list and SHA256SUMS. ``trials``: {path under the root: bytes}."""
    directory.mkdir(parents=True, exist_ok=True)
    members = sorted((f"{ROOT}/{path}", data) for path, data in trials.items())
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for path, data in members:
            info = tarfile.TarInfo(path)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    packed = subprocess.run(["zstd", "-q", "-c", "-6"], input=raw.getvalue(), capture_output=True, check=True).stdout
    names = []
    for number, start in enumerate(range(0, len(packed), part_bytes)):
        names.append(f"{name}.tar.zst.part-{number:03d}")
        (directory / names[-1]).write_bytes(packed[start:start + part_bytes])
    (directory / f"{name}.files.tsv").write_text("".join(f"{path}\t{len(data)}\n" for path, data in members))
    with open(directory / "SHA256SUMS", "a") as handle:
        for part in names:
            handle.write(f"{hashlib.sha256((directory / part).read_bytes()).hexdigest()}  {part}\n")
    return {path: data for path, data in members}


def tokens_file(tmp_path, *tokens):
    path = tmp_path / "tokens.txt"
    path.write_text("# the trials to restore\n" + "".join(token + "\n" for token in tokens))
    return path


def restored(out):
    return sorted(str(path.relative_to(out)) for path in Path(out).rglob("*") if path.is_file())


@pytest.fixture
def three(tmp_path):
    """One set holding three trials: aaa (a fresh trial), bbb (not asked for), ccc (a branch)."""
    files = {**trial_files("aaa111"), **trial_files("bbb222"), **trial_files("ccc333", branch=True)}
    return tmp_path / "archive", archive(tmp_path / "archive", "trials", files)


@needs_zstd
def test_only_the_four_records_of_each_named_trial_are_written(tmp_path, three):
    parts, members = three
    out = tmp_path / "out"
    status = cli().main(["--tokens", str(tokens_file(tmp_path, "aaa111", "ccc333")), "--out", str(out),
                         "--parts", str(parts), "--json", str(tmp_path / "account.json")])
    assert status == 0
    assert restored(out) == sorted([f"aaa111/{JOURNAL.format(run='aaa111')}", f"aaa111/{LEDGER}",
                                    f"aaa111/{TRAJECTORY}", f"ccc333/{JOURNAL.format(run='ccc333')}",
                                    f"ccc333/{LEDGER}", f"ccc333/{TRAJECTORY}", f"ccc333/{SCRIPT}"])
    for path in restored(out):
        assert (out / path).read_bytes() == members[f"{ROOT}/{path}"]
        assert (out / path).stat().st_mode & 0o777 == 0o600
    account = json.loads((tmp_path / "account.json").read_text())
    assert account["whole"] == 2 and account["missing"] == {}
    assert account["sets"]["trials"]["written"] == 7 and account["sets"]["trials"]["parts_checked"] >= 2


@needs_zstd
def test_reading_stops_after_the_last_wanted_member(tmp_path, three):
    parts, members = three
    account = {}
    for tokens in (["aaa111"], ["ccc333"]):
        planned = restore.plan(restore.read_index(parts / "trials.files.tsv"), tokens)
        with restore.decompressed(paths=restore.parts(parts, "trials")) as stream:
            account[tokens[0]] = restore.extract(stream, tokens, tmp_path / "out", members=planned["members"])
    # aaa's files come first in the archive: reading ends long before its end.
    assert account["aaa111"]["stopped_early"] and account["aaa111"]["read_members"] < len(members) / 2
    assert account["ccc333"]["written"] == 4 and account["ccc333"]["read_members"] <= len(members)


def raw_tar(entries):
    """An uncompressed tar of (TarInfo, bytes or None)."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for info, data in entries:
            tar.addfile(info, None if data is None else io.BytesIO(data))
    raw.seek(0)
    return raw


def test_a_link_a_path_outside_or_a_name_not_listed_is_never_written(tmp_path):
    link = tarfile.TarInfo(f"{ROOT}/aaa111/{TRAJECTORY}")
    link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
    escape = tarfile.TarInfo(f"{ROOT}/aaa111/../../evil/{LEDGER}")
    escape.size = 4
    ledger = tarfile.TarInfo(f"{ROOT}/aaa111/{LEDGER}")
    ledger.size = 6
    unlisted = tarfile.TarInfo(f"elsewhere/aaa111/{SCRIPT}")
    unlisted.size = 2
    stream = raw_tar([(link, None), (escape, b"evil"), (ledger, b"ledger"), (unlisted, b"{}")])
    members = {f"{ROOT}/aaa111/{TRAJECTORY}": 0, f"{ROOT}/aaa111/{LEDGER}": 6}
    account = restore.extract(stream, ["aaa111"], tmp_path / "out", members=members)
    assert restored(tmp_path / "out") == [f"aaa111/{LEDGER}"]
    assert account["refused"] == [{"member": f"{ROOT}/aaa111/{TRAJECTORY}", "why": "not a regular file"}]
    assert account["missing"] == {"aaa111": ["journal", "trajectory"]}
    assert not (tmp_path / "evil").exists()
    # A folder planted as a link out of the restore folder is not written through.
    (tmp_path / "out2").mkdir()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "out2" / "aaa111").symlink_to(tmp_path / "elsewhere")
    account = restore.extract(raw_tar([(ledger, b"ledger")]), ["aaa111"], tmp_path / "out2")
    assert account["written"] == 0 and account["refused"] and not list((tmp_path / "elsewhere").rglob("*"))


@needs_zstd
def test_a_second_run_leaves_the_files_and_other_bytes_are_kept_and_reported(tmp_path, three):
    parts, _ = three
    argv = ["--tokens", str(tokens_file(tmp_path, "aaa111")), "--out", str(tmp_path / "out"), "--parts", str(parts),
            "--json", str(tmp_path / "account.json")]
    assert cli().main(argv) == 0
    assert cli().main(argv) == 0
    again = json.loads((tmp_path / "account.json").read_text())["sets"]["trials"]
    assert (again["written"], again["present"], again["conflicts"]) == (0, 3, [])
    (tmp_path / "out" / "aaa111" / LEDGER).write_bytes(b"changed since")
    assert cli().main(argv) == 0
    third = json.loads((tmp_path / "account.json").read_text())["sets"]["trials"]
    assert third["conflicts"] == [f"{ROOT}/aaa111/{LEDGER}"]
    assert (tmp_path / "out" / "aaa111" / LEDGER).read_bytes() == b"changed since"


@needs_zstd
def test_a_part_that_differs_from_its_sum_stops_the_restore_before_anything_is_written(tmp_path, three):
    parts, _ = three
    last = restore.parts(parts, "trials")[-1]
    last.write_bytes(last.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="differs from its SHA-256"):
        cli().main(["--tokens", str(tokens_file(tmp_path, "aaa111")), "--out", str(tmp_path / "out"),
                    "--parts", str(parts)])
    assert restored(tmp_path / "out") == []
    (parts / "trials.tar.zst.part-000").unlink()
    with pytest.raises(ValueError, match="without a gap"):
        restore.parts(parts, "trials")


@needs_zstd
def test_a_trial_missing_a_required_record_is_reported_and_the_exit_status_says_so(tmp_path):
    parts = tmp_path / "archive"
    archive(parts, "trials", {**trial_files("aaa111"), **trial_files("ddd444", journal=False)})
    status = cli().main(["--tokens", str(tokens_file(tmp_path, "aaa111", "ddd444", "eee555")), "--out",
                         str(tmp_path / "out"), "--parts", str(parts), "--json", str(tmp_path / "account.json")])
    account = json.loads((tmp_path / "account.json").read_text())
    assert status == 1 and account["whole"] == 1
    assert account["missing"] == {"ddd444": ["journal"], "eee555": ["journal", "ledger", "trajectory"]}


@needs_zstd
def test_a_dry_run_reads_the_file_lists_alone_and_sets_holding_none_of_the_trials_are_skipped(tmp_path):
    parts = tmp_path / "archive"
    archive(parts, "trials-s00", trial_files("aaa111"))
    archive(parts, "trials-s01", {**trial_files("bbb222"), **trial_files("ccc333", branch=True)})
    argv = ["--tokens", str(tokens_file(tmp_path, "ccc333")), "--out", str(tmp_path / "out"), "--parts", str(parts),
            "--set", "trials-s00", "--set", "trials-s01", "--json", str(tmp_path / "account.json")]
    assert cli().main([*argv, "--dry-run"]) == 0
    account = json.loads((tmp_path / "account.json").read_text())
    assert not (tmp_path / "out").exists()
    assert account["sets"]["trials-s00"] == {"skipped": "holds none of the trials"}
    assert account["sets"]["trials-s01"]["members"] == 4 and account["sets"]["trials-s01"]["trials"] == 1
    # The real run reads only the set that holds the trial.
    (parts / "trials-s00.tar.zst.part-000").write_bytes(b"not read")
    assert cli().main(argv) == 0
    assert restored(tmp_path / "out") == sorted([f"ccc333/{JOURNAL.format(run='ccc333')}", f"ccc333/{LEDGER}",
                                                 f"ccc333/{TRAJECTORY}", f"ccc333/{SCRIPT}"])


@needs_zstd
def test_a_set_streamed_on_standard_input_is_restored_the_same_way(tmp_path, three, monkeypatch):
    parts, members = three
    stream = b"".join(path.read_bytes() for path in restore.parts(parts, "trials"))
    monkeypatch.setattr(sys, "stdin", type("Input", (), {"buffer": io.BytesIO(stream)})())
    status = cli().main(["--tokens", str(tokens_file(tmp_path, "ccc333")), "--out", str(tmp_path / "out"), "--stdin",
                         "--index", str(parts / "trials.files.tsv")])
    assert status == 0 and len(restored(tmp_path / "out")) == 4
    assert all((tmp_path / "out" / path).read_bytes() == members[f"{ROOT}/{path}"]
               for path in restored(tmp_path / "out"))


def test_a_tokens_file_holds_tokens_only(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text("aaa111\n\n# note\naaa111\nccc333\n")
    assert restore.tokens_from(path) == ["aaa111", "ccc333"]
    path.write_text("../etc\n")
    with pytest.raises(ValueError, match="not a trial token"):
        restore.tokens_from(path)
