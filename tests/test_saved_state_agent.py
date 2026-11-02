"""A saved state goes back exactly or is not graded; the verifier then runs as after any agent."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from taste.benchmarks.saved_state_agent import (
    SavedStateAgent,
    load_saved_state,
    restore_saved_state,
)
from taste.brains.docker_checkpoint import CheckpointManifest


def manifest(**overrides):
    return CheckpointManifest(**{
        "environment_id": "c" * 64, "image": "sha256:" + "d" * 64, "tar": "a" * 64 + ".tar", "tar_sha256": "a" * 64,
        "tar_bytes": 100, "changes": (("A", "/app/x"),), "copied": ("/app/x",), "deleted": (), "left_out": (),
        "over_cap": (), "taken_at": 1.0, **overrides})


def saved(tmp_path, **overrides):
    path = tmp_path / "cp1.json"
    path.write_text(json.dumps(manifest(**overrides).to_dict()))
    return path


class Backend:
    def __init__(self, *, exact=True):
        self.exact, self.calls = exact, []

    def restore(self, restored, directory, *, any_container=False):
        self.calls.append((restored.tar_sha256, directory, any_container))
        return SimpleNamespace(exact=self.exact, mismatches=() if self.exact else ("/app/x",),
                               summary=lambda: {"exact": self.exact, "seconds": 1.0})


def test_a_saved_state_is_read_from_its_manifest_beside_its_tar(tmp_path):
    loaded, directory = load_saved_state(saved(tmp_path))
    assert loaded.tar_sha256 == "a" * 64 and directory == tmp_path


def test_a_partial_save_is_not_graded(tmp_path):
    with pytest.raises(ValueError, match="partial"):
        load_saved_state(saved(tmp_path, over_cap=("/app/huge",)))


def test_a_restore_may_use_another_container_of_the_same_image(tmp_path):
    backend = Backend()
    account = restore_saved_state(backend, saved(tmp_path))
    assert backend.calls == [("a" * 64, tmp_path, True)]
    assert account["exact"] is True and account["tar_sha256"] == "a" * 64


def test_files_that_did_not_come_back_exactly_are_not_graded(tmp_path):
    with pytest.raises(RuntimeError, match="differ"):
        restore_saved_state(Backend(exact=False), saved(tmp_path))


def test_the_agent_must_be_told_which_state(tmp_path):
    with pytest.raises(ValueError, match="saved_state"):
        SavedStateAgent(tmp_path)
