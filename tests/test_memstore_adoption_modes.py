"""Adoption preserves file semantics as well as blob content."""

from __future__ import annotations

import os
from contextlib import closing
from pathlib import Path

import pytest

from taste.memstore import PublishError, Store


@pytest.mark.parametrize("source_kind", ["file", "executable", "symlink"])
@pytest.mark.parametrize("destination_kind", ["file", "executable", "symlink"])
def test_adoption_replaces_the_entry_and_preserves_its_exact_mode(
    tmp_path: Path, source_kind: str, destination_kind: str,
) -> None:
    with closing(Store.open(tmp_path / "repo", "s")) as store:
        source = store.branch("producer")
        source.write("artifact", b"#!/bin/sh\nprintf adopted\\n\n")
        if source_kind == "executable":
            source.path("artifact").chmod(0o755)
        elif source_kind == "symlink":
            source.path("artifact").unlink()
            source.path("artifact").symlink_to("missing-target")
        pinned = source.checkpoint("publish input")
        expected = store.backend.entry_at(pinned.id, "artifact")

        destination = store.branch("consumer")
        victim = tmp_path / "unrelated.txt"
        victim.write_text("must survive")
        destination.write("input", "old input")
        if destination_kind == "executable":
            destination.path("input").chmod(0o755)
        elif destination_kind == "symlink":
            destination.path("input").unlink()
            destination.path("input").symlink_to(victim)
        destination.checkpoint("previous input")
        # A common Git setting must not override the pinned source mode.
        destination.backend.repo.git.config("core.filemode", "false")
        destination.adopt(pinned, "artifact", as_="input")
        adopted = destination.checkpoint("adopt exact input")
        actual = store.backend.entry_at(adopted.id, "input")
        assert (actual.mode, actual.sha) == (expected.mode, expected.sha)
        assert victim.read_text() == "must survive"
        assert destination.path("input").is_symlink() == (source_kind == "symlink")
        if source_kind == "symlink":
            assert os.readlink(destination.path("input")) == "missing-target"
        else:
            assert destination.path("input").read_bytes() == pinned.read_bytes("artifact")
            assert bool(destination.path("input").stat().st_mode & 0o100) == (source_kind == "executable")
        assert store.origin(adopted, "input") == pinned


def test_adoption_never_traverses_a_parent_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "input"
    protected.write_text("keep")
    with closing(Store.open(tmp_path / "repo", "s")) as store:
        source = store.branch("source")
        source.write("input", "replacement")
        pinned = source.checkpoint("source")
        destination = store.branch("destination")
        destination.path("nested").symlink_to(outside, target_is_directory=True)
        with pytest.raises(PublishError, match="traverses a symlink"):
            destination.adopt(pinned, "input", as_="nested/input")
    assert protected.read_text() == "keep"


def test_changing_only_mode_changes_the_artifact_origin(tmp_path: Path) -> None:
    with closing(Store.open(tmp_path / "repo", "s")) as store:
        source = store.branch("source")
        source.write("script", "exit 0\n")
        original = source.checkpoint("ordinary file")
        consumer = store.branch("consumer")
        consumer.adopt(original, "script")
        copied = consumer.checkpoint("copied ordinary file")
        consumer.path("script").chmod(0o755)
        executable = consumer.checkpoint("made executable")
        assert copied.blob("script") == executable.blob("script")
        assert store.origin(executable, "script") == executable
