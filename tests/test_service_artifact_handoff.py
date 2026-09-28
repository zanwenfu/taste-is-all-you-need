"""Service-specific evidence cannot bypass original main-container drainage."""
import asyncio
import copy
import json
import tarfile
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from taste.benchmarks.artifact_handoff import ArtifactHandoff, ArtifactTarget
from taste.benchmarks.output_snapshot import OutputSnapshotError, SnapshotLimits
from taste.brains.docker_terminal import DockerTerminalBackend
from taste.brains.terminal_broker import TerminalConflict
from tests.test_docker_terminal import backend
from tests.test_docker_terminal import daemon as docker_daemon
from tests.test_output_snapshot import file_archive, serve_archive
from tests.test_thread_shutdown_ownership import cancel_loop_tasks

daemon = docker_daemon
HELPER = "d" * 64


@pytest.fixture
def services(daemon, tmp_path):
    main = backend(daemon)
    helper = DockerTerminalBackend(replace(main.binding, container_id=HELPER, cgroup_path=f"/docker/{HELPER}"))
    info = copy.deepcopy(daemon.info)
    info["Id"] = HELPER
    serve_archive(daemon, file_archive(("result", b"correct", tarfile.REGTYPE)), source="/tmp/result")
    main_wire = daemon.override
    serve_archive(daemon, file_archive(("result.json", b"helper", tarfile.REGTYPE)),
                  source="/audit/result.json", container_id=HELPER)
    helper_wire = daemon.override

    def dispatch(handler, path):
        if path == f"/containers/{HELPER}/json":
            daemon.reply(handler, info)
            return True
        if path.startswith(f"/containers/{HELPER}/archive?"):
            return helper_wire(handler, path)
        return main_wire(handler, path)
    daemon.override = dispatch
    targets = [ArtifactTarget("/tmp/result", tmp_path / "main", "file"),
               ArtifactTarget("/audit/result.json", tmp_path / "helper", "file", "evidence")]
    item = ArtifactHandoff(main, targets, service_backends={"evidence": helper})
    return SimpleNamespace(main=main, helper=helper, targets=targets, item=item, info=info, daemon=daemon)


async def capture(case, index, **kwargs):
    target = case.targets[index]
    return await case.item.collect(target.source, target.destination, target.kind,
                                   deadline_unix=time.time() + 5, service=target.service, **kwargs)


async def main_done(case):
    await capture(case, 0)
    return await case.item.drain_main()


def manifest(case):
    return [{"source": target.source, "destination": "artifacts/" + target.destination.name,
             "type": target.kind, "status": "ok", "service": None if target.service == "main" else target.service,
             "exclude": []} for target in case.targets]


def test_real_http_copies_from_each_original_service_after_main_drains(services):
    async def scenario():
        binding = await main_done(services)
        assert binding["container_id"] == services.main.environment_id
        assert services.daemon.info["State"]["Running"] is False
        # Mutating the caller's backend cannot redirect an admitted service.
        services.helper.binding = services.main.binding
        receipt = await capture(services, 1)
        assert receipt["binding"]["container_id"] == HELPER and receipt["service"] == "evidence"
        rows = services.item.seal_harbor_manifest(json.dumps(manifest(services)).encode(), services.targets[0].destination.parent)
        assert len(rows) == 2 and services.targets[1].destination.read_bytes() == b"helper"
        assert rows[0]["binding"] != rows[1]["binding"]
    asyncio.run(scenario())


@pytest.mark.parametrize("damage", ["omitted_drain", "failed_stop", "early_drain", "wrong_service"])
def test_best_effort_collection_cannot_bypass_a_missing_main_stop_barrier(services, damage):
    async def scenario():
        if damage != "early_drain":
            await capture(services, 0)
        if damage == "failed_stop":
            services.daemon.stop_failure = True
        if damage in {"failed_stop", "early_drain"}:
            with pytest.raises((OutputSnapshotError, RuntimeError)):
                await services.item.drain_main()
        elif damage == "wrong_service":
            await services.item.drain_main()
        before = len(services.daemon.calls)
        with pytest.raises(OutputSnapshotError):
            target = services.targets[1]
            await services.item.collect(target.source, target.destination, "file", deadline_unix=time.time() + 5,
                                        service=None if damage == "wrong_service" else "evidence")
        assert len(services.daemon.calls) == before and not services.targets[1].destination.exists()
        with pytest.raises(OutputSnapshotError):
            services.item.seal()
        assert services.item.phase == "failed"
    asyncio.run(scenario())


@pytest.mark.parametrize("restart", ["main", "helper"])
def test_restart_after_main_barrier_cannot_change_captured_service_identity(services, restart):
    async def scenario():
        await main_done(services)
        info = services.daemon.info if restart == "main" else services.info
        info["State"]["StartedAt"] = "a different original start"
        with pytest.raises(TerminalConflict, match="restarted"):
            await capture(services, 1)
        assert not services.targets[1].destination.exists() and services.item.phase == "failed"
    asyncio.run(scenario())


@pytest.mark.parametrize("service", [None, "main", "other"])
def test_manifest_cannot_relabel_successful_helper_evidence_as_another_service(services, service):
    async def scenario():
        await main_done(services)
        await capture(services, 1)
    asyncio.run(scenario())
    rows = manifest(services)
    rows[1]["service"] = service
    with pytest.raises(OutputSnapshotError, match="exact successful"):
        services.item.seal_harbor_manifest(json.dumps(rows).encode(), services.targets[0].destination.parent)
    assert services.item.phase == "failed"


def test_aggregate_capture_limits_span_service_boundaries(services):
    services.item = ArtifactHandoff(services.main, services.targets,
        service_backends={"evidence": services.helper}, limits=SnapshotLimits(total_bytes=8))

    async def scenario():
        await main_done(services)
        with pytest.raises(OutputSnapshotError, match="aggregate"):
            await capture(services, 1)
        assert services.item.phase == "failed"
    asyncio.run(scenario())


@pytest.mark.parametrize("whole_loop", [False, True])
def test_cancelled_main_drain_retains_writer_ownership_and_poisons_collection(services, monkeypatch, whole_loop):
    entered, release = threading.Event(), threading.Event()

    def blocked(_self):
        entered.set()
        assert release.wait(5)
        return services.main.environment_id
    monkeypatch.setattr(DockerTerminalBackend, "stop_and_confirm", blocked)

    async def scenario():
        await capture(services, 0)
        task = asyncio.create_task(services.item.drain_main())
        while not entered.is_set():
            await asyncio.sleep(0.001)
        cancel_loop_tasks() if whole_loop else task.cancel()
        try:
            await asyncio.sleep(0.02)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(OutputSnapshotError):
            await capture(services, 1)
        assert services.item.phase == "failed"
    asyncio.run(scenario())


@pytest.mark.parametrize("mapping", ["missing", "extra", "alias"])
def test_unbound_extra_or_aliased_services_are_rejected_before_io(services, mapping):
    values = {} if mapping == "missing" else {"evidence": services.main if mapping == "alias" else services.helper}
    if mapping == "extra":
        values["other"] = services.helper
    before = len(services.daemon.calls)
    with pytest.raises(ValueError):
        ArtifactHandoff(services.main, services.targets, service_backends=values)
    assert len(services.daemon.calls) == before


def test_admitted_artifact_targets_are_read_only(services):
    with pytest.raises(TypeError):
        services.item.targets["/tmp/result"] = services.targets[1]
