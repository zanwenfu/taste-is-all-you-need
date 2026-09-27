"""Credential bytes cross the controller/service boundary without public leaks."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace

import pytest

from taste.brains.process_credentials import ScopeCredential, load_credential_properties
from taste.brains.process_scope import OwnedProcessScope, ScopeSpec
from taste.resources import ResourceCleanupError
from tests.test_process_scope import Manager


class CredentialManager(Manager):
    def start(self, unit, description, spec, *, credential_directory=None):
        self.properties = load_credential_properties(credential_directory, spec.credentials)
        super().start(unit, description, spec)


@pytest.fixture
def admitted(tmp_path):
    secret = b"dummy-secret-never-public"
    spec = ScopeSpec(("/bin/true",), str(tmp_path), 1000, 6,
                     credentials=(ScopeCredential.from_bytes("azure.json", secret),))
    return spec, {"azure.json": secret}


def create(tmp_path, admitted):
    spec, values = admitted
    return OwnedProcessScope.create(tmp_path / "owner", spec, credentials=values, manager=CredentialManager())


def test_legacy_scope_metadata_digest_and_recovery_are_unchanged(tmp_path):
    old_wire = {"argv": ["/bin/true"], "cwd": "/tmp", "uid": 1000, "runtime_seconds": 6.0,
                "grace_seconds": 2.0, "python_path": None}
    digest = hashlib.sha256(json.dumps(old_wire, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    spec = ScopeSpec(("/bin/true",), "/tmp", 1000, 6)
    scope = OwnedProcessScope.create(tmp_path / "owner", spec, manager=Manager())
    saved = json.loads((scope.directory / "state.json").read_text())
    assert saved["spec"] == old_wire and saved["spec_digest"] == digest == spec.digest
    assert not (scope.directory / "credentials").exists()
    assert OwnedProcessScope(scope.directory, manager=scope.manager).stop()["processes_stopped"]


def test_private_snapshots_survive_reopen_and_do_not_enter_metadata_or_arguments(tmp_path, admitted):
    scope = create(tmp_path, admitted)
    descriptor = scope.spec.credentials[0]
    source = scope.directory / "credentials" / descriptor.name
    assert source.read_bytes() == admitted[1][descriptor.name]
    assert source.stat().st_mode & 0o777 == 0o600
    assert source.parent.stat().st_mode & 0o777 == 0o700
    saved = (scope.directory / "state.json").read_text()
    assert descriptor.sha256 in saved and admitted[1][descriptor.name].decode() not in saved
    reopened = OwnedProcessScope(scope.directory, manager=scope.manager)
    reopened.start()
    assert scope.manager.properties == [f"--property=LoadCredential={descriptor.name}:{source}"]
    assert admitted[1][descriptor.name].decode() not in repr(scope.spec)
    assert admitted[1][descriptor.name].decode() not in repr(scope.manager.properties)
    # Shutdown/recovery needs process identity, not credential bytes. A missing
    # secret must never prevent containment after a successful launch.
    source.unlink()
    assert OwnedProcessScope(scope.directory, manager=scope.manager).stop()["processes_stopped"]


@pytest.mark.parametrize("values", [None, {}, {"wrong": b"dummy"}, {"azure.json": b"wrong"},
                                    {"azure.json": "dummy-secret-never-public"}])
def test_missing_extra_or_changed_credentials_cannot_create_launch_intent(tmp_path, admitted, values):
    with pytest.raises(ValueError):
        OwnedProcessScope.create(tmp_path / "owner", admitted[0], credentials=values)
    assert not (tmp_path / "owner").exists()


@pytest.mark.parametrize("name", ["", "..", "../secret", "a/b", "a:b", "a%n", "a\nb", "a b", "x" * 65])
def test_credential_names_cannot_escape_or_expand(name):
    with pytest.raises(ValueError):
        ScopeCredential.from_bytes(name, b"dummy")


@pytest.mark.parametrize("value", [b"", b"x" * 65537, "text", bytearray(b"mutable")])
def test_credential_size_and_type_are_bounded(value):
    with pytest.raises(ValueError):
        ScopeCredential.from_bytes("azure", value)


def test_duplicate_or_unbounded_credential_sets_are_rejected(admitted):
    spec, _ = admitted
    with pytest.raises(ValueError):
        replace(spec, credentials=spec.credentials * 2)
    with pytest.raises(ValueError):
        replace(spec, credentials=tuple(ScopeCredential.from_bytes(f"item{i}", b"v") for i in range(9)))
    with pytest.raises(ValueError):
        replace(spec, credentials=list(spec.credentials))


@pytest.mark.parametrize("mutation", ["bytes", "large", "readable", "hardlink", "symlink", "fifo",
                                     "directory", "public_directory", "writable_ancestor", "symlink_ancestor"])
def test_changed_private_material_fails_before_systemd_effects(tmp_path, admitted, mutation):
    scope = create(tmp_path, admitted)
    source = scope.directory / "credentials" / "azure.json"
    if mutation == "bytes":
        source.write_bytes(b"changed dummy")
    elif mutation == "large":
        source.write_bytes(b"x" * 65537)
    elif mutation == "readable":
        source.chmod(0o644)
    elif mutation == "hardlink":
        os.link(source, tmp_path / "alias")
    elif mutation in {"symlink", "fifo", "directory"}:
        source.unlink()
        if mutation == "symlink":
            target = tmp_path / "secret"
            target.write_bytes(admitted[1]["azure.json"])
            target.chmod(0o600)
            source.symlink_to(target)
        elif mutation == "fifo":
            os.mkfifo(source, 0o600)
        else:
            source.mkdir(mode=0o700)
    elif mutation == "public_directory":
        source.parent.chmod(0o750)
    elif mutation == "writable_ancestor":
        tmp_path.chmod(0o777)
    elif mutation == "symlink_ancestor":
        actual = scope.directory / "actual"
        source.parent.rename(actual)
        source.parent.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ResourceCleanupError):
        scope.start()
    assert not scope.manager.starts
    saved = json.loads((scope.directory / "state.json").read_text())
    assert saved["termination"] is None and not saved["launch_acknowledged"]


def test_foreign_owned_ancestor_is_not_admitted_even_when_read_only(tmp_path, admitted, monkeypatch):
    scope = create(tmp_path, admitted)
    real_fstat = os.fstat
    ancestor = tmp_path.stat()

    def foreign(fd):
        value = real_fstat(fd)
        if (value.st_dev, value.st_ino) == (ancestor.st_dev, ancestor.st_ino):
            fields = list(value)
            fields[4] = 23456
            return os.stat_result(fields)
        return value

    monkeypatch.setattr(os, "fstat", foreign)
    with pytest.raises(ResourceCleanupError, match="replaceable"):
        scope.start()
    assert not scope.manager.starts


def test_partial_credential_copy_never_publishes_a_ready_scope(tmp_path, admitted, monkeypatch):
    def partial(parent, values):
        os.mkdir("credentials", mode=0o700, dir_fd=parent)
        raise OSError("injected disk failure")

    monkeypatch.setattr("taste.brains.process_scope.write_credentials", partial)
    with pytest.raises(OSError, match="disk failure"):
        create(tmp_path, admitted)
    assert not (tmp_path / "owner" / "state.json").exists()
    with pytest.raises(FileNotFoundError):
        OwnedProcessScope(tmp_path / "owner", manager=Manager())


def test_credential_digest_tampering_refuses_recovery(tmp_path, admitted):
    scope = create(tmp_path, admitted)
    state_path = scope.directory / "state.json"
    saved = json.loads(state_path.read_text())
    saved["spec"]["credentials"][0]["sha256"] = "0" * 64
    state_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="digest differs"):
        OwnedProcessScope(scope.directory, manager=scope.manager)
    assert not scope.manager.starts
