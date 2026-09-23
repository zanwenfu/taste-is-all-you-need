"""Process ownership survives ambiguous replies, faults and controller restart."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from taste.brains.process_scope import OwnedProcessScope, ScopeSpec
from taste.resources import ResourceCleanupError, resource_failures


class Manager:
    def __init__(self):
        self.units = {}
        self.populated = set()
        self.starts = []
        self.stops = []
        self.releases = []
        self.start_error = None
        self.stop_error = None
        self.release_error = None
        self.preserve_population = False
        self.before_start = lambda: None
        self.before_stop = lambda: None

    def start(self, unit, description, spec):
        self.before_start()
        self.starts.append(unit)
        self.units[unit] = {
            "Description": description, "User": str(spec.uid), "KillMode": "control-group",
            "SendSIGKILL": "yes", "Restart": "no", "InvocationID": "a" * 32,
            "ControlGroup": f"/system.slice/{unit}", "ActiveState": "active", "Result": "success",
            "RuntimeMaxUSec": f"{spec.runtime_seconds:g}s", "TimeoutStopUSec": f"{spec.grace_seconds:g}s",
            "Delegate": "no", "NoNewPrivileges": "yes", "ProtectControlGroups": "yes",
        }
        self.populated.add(unit)
        if self.start_error:
            raise self.start_error

    def inspect(self, unit):
        value = self.units.get(unit)
        return None if value is None else dict(value)

    def stop(self, unit, grace_seconds):
        self.before_stop()
        self.stops.append(unit)
        if self.stop_error:
            raise self.stop_error
        self.units[unit].update(ActiveState="inactive", ControlGroup="")
        if not self.preserve_population:
            self.populated.discard(unit)

    def empty(self, unit):
        return unit not in self.populated

    def release(self, unit):
        self.releases.append(unit)
        if self.release_error:
            raise self.release_error
        self.units.pop(unit, None)


@pytest.fixture
def scope(tmp_path):
    return OwnedProcessScope.create(
        tmp_path / "owner", ScopeSpec(("/usr/bin/python3", "-V"), str(tmp_path), 1000, 6, 0.5),
        manager=Manager(),
    )


def state(scope):
    return json.loads((scope.directory / "state.json").read_text())


def restore(scope):
    return OwnedProcessScope(scope.directory, manager=scope.manager)


def test_stop_before_launch_is_durable_and_never_contacts_the_service_manager(scope):
    receipt = scope.stop("cancelled before dispatch")
    assert receipt["processes_stopped"] and not receipt["goal_settlement_required"]
    assert restore(scope).stop("later reason") == receipt
    with pytest.raises(RuntimeError, match="already admitted"):
        restore(scope).start()
    assert scope.manager.starts == scope.manager.stops == scope.manager.releases == []


def test_launch_and_stop_intents_precede_effects_and_restart_cannot_launch_twice(scope):
    def before_start():
        assert state(scope)["phase"] == "launch_pending"
        assert state(scope)["launch_attempted"] and not state(scope)["launch_acknowledged"]

    def before_stop():
        saved = state(scope)
        assert saved["phase"] == "stop_pending" and saved["stop_reason"] == "cancel"
        assert saved["invocation_id"] == "a" * 32

    scope.manager.before_start = before_start
    scope.manager.before_stop = before_stop
    scope.start()
    with pytest.raises(RuntimeError, match="already admitted"):
        restore(scope).start()
    receipt = restore(scope).stop("cancel")
    assert receipt["processes_stopped"] and receipt["goal_settlement_required"]
    assert len(scope.manager.starts) == len(scope.manager.stops) == len(scope.manager.releases) == 1
    assert restore(scope).stop("retry") == receipt


def test_lost_start_acknowledgement_recovers_observed_service_without_relaunch(scope):
    scope.manager.start_error = ConnectionError("start reply lost")
    with pytest.raises(ResourceCleanupError, match="reply lost"):
        scope.start()
    assert state(scope)["phase"] == "launch_pending"
    with pytest.raises(RuntimeError, match="already admitted"):
        restore(scope).start()
    receipt = restore(scope).stop("recover ambiguous start")
    assert receipt["goal_settlement_required"] and receipt["invocation_id"] == "a" * 32
    assert len(scope.manager.starts) == 1 and not scope.manager.populated


def test_absence_after_lost_ack_remains_fenced_until_delayed_service_is_observed(scope):
    scope.manager.start_error = TimeoutError("DBus reply lost")
    with pytest.raises(ResourceCleanupError):
        scope.start()
    delayed = scope.manager.units.pop(scope.unit)
    scope.manager.populated.clear()
    with pytest.raises(ResourceCleanupError, match="absent service is ambiguous"):
        restore(scope).stop("cancel while launch is uncertain")
    assert state(scope)["phase"] == "stop_pending" and state(scope)["termination"] is None
    assert not scope.manager.stops
    scope.manager.units[scope.unit] = delayed
    scope.manager.populated.add(scope.unit)
    receipt = restore(scope).stop("later retry")
    assert receipt["reason"] == "cancel while launch is uncertain"
    assert not scope.manager.populated and len(scope.manager.starts) == 1


def test_lost_ack_with_observed_incarnation_can_recover_after_release_reply_is_lost(scope):
    scope.manager.start_error = ConnectionError("start reply lost")
    with pytest.raises(ResourceCleanupError):
        scope.start()
    release = scope.manager.release

    def lost_release(unit):
        release(unit)
        raise ConnectionError("release reply lost")

    scope.manager.release = lost_release
    with pytest.raises(ResourceCleanupError, match="release reply lost"):
        restore(scope).stop("recover once")
    assert state(scope)["invocation_id"] and not scope.manager.units
    receipt = restore(scope).stop("recover twice")
    assert receipt["reason"] == "recover once" and receipt["goal_settlement_required"]
    assert len(scope.manager.starts) == 1


@pytest.mark.parametrize(("key", "value"), [
    ("Description", "someone else's service"), ("User", "0"),
    ("InvocationID", "b" * 32), ("ControlGroup", "/system.slice/foreign.service"),
    ("RuntimeMaxUSec", "infinity"), ("RuntimeMaxUSec", "1h"),
    ("TimeoutStopUSec", "1min"), ("KillMode", "process"),
    ("SendSIGKILL", "no"), ("Delegate", "yes"), ("NoNewPrivileges", "no"),
])
def test_changed_identity_or_enforcement_refuses_to_signal_or_report_cleanup(scope, key, value):
    scope.start()
    original = scope.manager.units[scope.unit][key]
    scope.manager.units[scope.unit][key] = value
    with pytest.raises(ResourceCleanupError):
        restore(scope).stop("refuse wrong incarnation")
    assert not scope.manager.stops and state(scope)["termination"] is None
    assert scope.unit in scope.manager.populated
    scope.manager.units[scope.unit][key] = original
    assert restore(scope).stop()["processes_stopped"]


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_failed_stop_retains_resource_evidence_and_retries_the_same_first_reason(scope, failure):
    scope.start()
    scope.manager.stop_error = failure("termination failed")
    with pytest.raises(ResourceCleanupError) as caught:
        scope.stop("first cancellation")
    assert isinstance(caught.value.__cause__, failure)
    assert resource_failures(caught.value)[0].resource_id == scope.unit
    assert state(scope)["phase"] == "stop_pending" and scope.manager.populated
    scope.manager.stop_error = None
    receipt = restore(scope).stop("second cancellation")
    assert receipt["reason"] == "first cancellation"
    assert len(scope.manager.starts) == 1


def test_successful_stop_reply_is_insufficient_while_descendants_remain(scope):
    scope.start()
    scope.manager.preserve_population = True
    with pytest.raises(ResourceCleanupError, match="live descendants"):
        scope.stop()
    assert state(scope)["termination"] is None and scope.manager.releases == []
    scope.manager.preserve_population = False
    assert restore(scope).stop()["processes_stopped"]


def test_unit_metadata_cleanup_failure_also_remains_pending(scope):
    scope.start()
    scope.manager.release_error = OSError("manager cleanup unavailable")
    with pytest.raises(ResourceCleanupError, match="manager cleanup unavailable"):
        scope.stop()
    assert not scope.manager.populated and state(scope)["termination"] is None
    scope.manager.release_error = None
    assert restore(scope).stop()["processes_stopped"]


def test_stop_intent_persistence_failure_never_signals_and_preserves_resource_failure(scope, monkeypatch):
    scope.start()

    def no_disk(*_args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(scope, "_write", no_disk)
    with pytest.raises(BaseExceptionGroup) as caught:
        scope.stop()
    assert resource_failures(caught.value)[0].resource_id == scope.unit
    assert not scope.manager.stops and scope.manager.populated
    assert state(scope)["phase"] == "running"
    assert restore(scope).stop()["processes_stopped"]


def test_crash_after_persisting_a_never_launched_stop_does_not_invent_ambiguous_execution(scope):
    saved = state(scope)
    saved.update(phase="stop_pending", stop_reason="stop before dispatch")
    (scope.directory / "state.json").write_text(json.dumps(saved))
    receipt = restore(scope).stop()
    assert not receipt["goal_settlement_required"]
    assert not scope.manager.starts and not scope.manager.stops


def test_two_controller_objects_serialize_the_launch_transaction(scope):
    entered, release = threading.Event(), threading.Event()

    def stalled_start():
        entered.set()
        assert release.wait(5)

    scope.manager.before_start = stalled_start
    second = restore(scope)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(scope.start)
        try:
            assert entered.wait(5)
            other = pool.submit(second.start)
            assert not other.done()
        finally:
            release.set()
        first.result(timeout=5)
        with pytest.raises(RuntimeError, match="already admitted"):
            other.result(timeout=5)
    assert len(scope.manager.starts) == 1
    scope.stop()


def test_wait_expiry_stops_and_releases_live_scope(scope):
    scope.start()
    receipt = scope.wait(timeout_seconds=0.001)
    assert receipt["reason"] == "outside controller wait deadline expired"
    assert not scope.manager.populated and not scope.manager.units


def test_restored_wait_retries_pending_stop_before_sleeping(scope, monkeypatch):
    scope.start()
    scope.manager.stop_error = OSError("temporary stop failure")
    with pytest.raises(ResourceCleanupError):
        scope.stop("durable cancellation")
    scope.manager.stop_error = None

    def cannot_wait(_seconds):
        pytest.fail("a persisted cancellation must resume cleanup before waiting for execution")

    monkeypatch.setattr("taste.brains.process_scope.time.sleep", cannot_wait)
    receipt = restore(scope).wait(timeout_seconds=1)
    assert receipt["reason"] == "durable cancellation" and not scope.manager.populated


def test_fast_successful_execution_does_not_require_an_existing_unit(scope):
    start = scope.manager.start

    def fast(*args):
        start(*args)
        scope.manager.units.clear()
        scope.manager.populated.clear()

    scope.manager.start = fast
    scope.start()
    assert scope.wait(timeout_seconds=1)["goal_settlement_required"]
    assert not scope.manager.stops


@pytest.mark.parametrize("mutation", [
    {"launch_acknowledged": 1}, {"phase": "running", "launch_attempted": False},
    {"phase": "stopped", "termination": {"processes_stopped": True}},
    {"spec_digest": "corrupt"}, {"owner": "../../foreign"},
])
def test_corrupt_durable_state_is_rejected_before_any_manager_effect(scope, mutation):
    saved = state(scope)
    saved.update(mutation)
    (scope.directory / "state.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        restore(scope)
    assert not scope.manager.starts and not scope.manager.stops


@pytest.mark.parametrize("changes", [
    {"uid": 0}, {"uid": True}, {"runtime_seconds": float("inf")},
    {"grace_seconds": 0}, {"argv": ("relative",)}, {"cwd": "relative"},
    {"argv": ("/bin/echo", "%n")}, {"argv": ("/bin/echo", "x" * 65536)},
])
def test_invalid_launch_configuration_is_rejected_before_durable_admission(tmp_path, changes):
    values = {"argv": ("/bin/true",), "cwd": str(tmp_path), "uid": 1000, "runtime_seconds": 6}
    with pytest.raises(ValueError):
        ScopeSpec(**(values | changes))
