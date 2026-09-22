#!/usr/bin/env python3
"""Serial checks against real, disposable Docker containers; no model calls.

Run on the test server with PYTHONPATH pointing to the committed checkout and
the Docker SDK installed from requirements-docker-check-lock.txt. The image
must already be cached. Every container has a unique validation label, disabled
network, no host mounts, and bounded CPU/memory/PID limits. Transport failures
are injected in this client's calls; the Docker daemon is never disrupted.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import signal
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class TrackedContainers:
    def __init__(self, client: Any, token: str):
        self.client = client
        self.token = token
        self.created: list[str] = []
        self.lose_reply_once = False

    def get(self, identifier: str) -> Any:
        return self.client.containers.get(identifier)

    def run(self, image: str, **kwargs: Any) -> Any:
        require(len(self.created) < 12, "container creation limit exceeded")
        require(kwargs.get("network_mode") == "none", "probe forbids container networking")
        require(not kwargs.get("volumes") and not kwargs.get("mounts"), "probe forbids host mounts")
        kwargs["labels"] = {**kwargs.get("labels", {}), "taste.validation": self.token}
        container = self.client.containers.run(
            image, **kwargs, mem_limit="256m", nano_cpus=500_000_000, pids_limit=128,
        )
        self.created.append(container.id)
        if self.lose_reply_once:
            self.lose_reply_once = False
            raise ConnectionError("injected loss of container creation reply")
        return container


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Already-cached image containing bash, timeout and /testbed")
    parser.add_argument("--owner-token", required=True, help="Unique 32-character lowercase hex validation identity")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{32}", args.owner_token) is None:
        parser.error("--owner-token must be 32 lowercase hex characters")

    import docker
    from docker.errors import NotFound

    from taste.execution import DockerProvider, DockerSandbox
    from taste.resources import ResourceCleanupError

    client = docker.from_env(timeout=15)
    tracking = TrackedContainers(client, args.owner_token)
    wrapped = SimpleNamespace(containers=tracking)
    label = {"label": f"taste.validation={args.owner_token}"}
    report: dict[str, Any] = {"owner_token": args.owner_token, "python": platform.python_version(),
                              "docker_sdk": version("docker"), "cases": [], "cleanup_errors": []}
    owned_scope = False

    def terminated(signum, frame):
        raise TimeoutError(f"validation interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, terminated)

    def absent(identifier: str) -> bool:
        try:
            client.containers.get(identifier)
        except NotFound:
            return True
        return False

    def provider() -> DockerProvider:
        return DockerProvider(client=wrapped, prefix=f"taste-check-{args.owner_token[:8]}")

    def expect_cleanup_error(call, detail: str) -> ResourceCleanupError:
        try:
            call()
        except ResourceCleanupError as exc:
            require(detail in str(exc), f"wrong cleanup error: {exc}")
            return exc
        raise AssertionError("operation did not report its resource failure")

    try:
        # A reused validation token must not authorize removing a previous run.
        require(not client.containers.list(all=True, filters=label), "owner token already has containers")
        owned_scope = True
        image = client.images.get(args.image).id  # Resolve locally; never pull an image here.
        report.update(image=image, docker_server=client.version()["Version"])

        one, two = provider(), provider()
        first = one.open(key="same", image=image)
        second = two.open(key="same", image=image)
        require(first.container.name != second.container.name, "provider names collided")
        first_live, second_live = tracking.get(first.container.id), tracking.get(second.container.id)
        require(first_live.status == second_live.status == "running", "one provider removed the other's container")
        require(first_live.labels["taste.owner"] != second_live.labels["taste.owner"], "owner labels collided")
        one.close_all()
        require(absent(first.container.id), "first provider did not remove its container")
        command = second.exec("printf 'second-still-live'", timeout=10)
        require(command.ok and command.stdout == "second-still-live", "second provider was damaged")
        two.close_all()
        require(absent(second.container.id), "second provider did not clean up")
        report["cases"].append("separate providers preserve each other's real containers")

        shared = provider()
        first = shared.open(key="a/b", image=image)
        second = shared.open(key="a_b", image=image)
        require(first.container.name != second.container.name, "normalized names collided")
        require(tracking.get(first.container.id).status == "running", "normalization removed a live container")
        first.put_text("/testbed/taste-validation.txt", "owned bytes\n")
        require(first.get_text("/testbed/taste-validation.txt") == "owned bytes\n", "file transport changed bytes")
        shared.close_all()
        require(absent(first.container.id) and absent(second.container.id), "normalized-key cleanup incomplete")
        report["cases"].append("normalized keys and file transport retain separate ownership")

        shared = provider()
        sandbox = shared.open(key="configuration", image=image)
        expect_cleanup_error(lambda: shared.open(key="configuration", image=image, network_mode="bridge"),
                             "configuration")
        actual = tracking.get(sandbox.container.id)
        require(actual.status == "running" and actual.attrs["HostConfig"]["NetworkMode"] == "none",
                "configuration refusal changed the live container")
        require(sandbox.network_mode == "none", "reported network mode differs from launch")
        shared.close_all()
        require(absent(sandbox.container.id), "configuration case leaked a container")
        report["cases"].append("cached network mismatch is refused before changing the live container")

        shared = provider()
        first, second = shared.open(key="failure", image=image), shared.open(key="other", image=image)
        remove = first.container.remove

        def fail_remove(*args, **kwargs):
            raise ConnectionError("injected removal transport failure")

        first.container.remove = fail_remove
        try:
            expect_cleanup_error(first.close, "transport failure")
            require(tracking.get(first.container.id).status == "running", "fault injection did not leave a live container")
            expect_cleanup_error(lambda: shared.open(key="failure", image=image), "cleanup")
            expect_cleanup_error(shared.close_all, "transport failure")
            require(absent(second.container.id), "bulk close abandoned another container")
            require(tracking.get(first.container.id).status == "running", "failed owner disappeared")
        finally:
            first.container.remove = remove
        shared.close_all()
        require(absent(first.container.id), "explicit cleanup retry did not remove its container")
        report["cases"].append("failed removal retains a live owner, closes others, and supports explicit cleanup")

        before = len(tracking.created)
        try:
            DockerSandbox(image=image, name=f"taste-check-missing-{args.owner_token}",
                          workdir=f"/taste-missing-{args.owner_token}", client=wrapped)
        except RuntimeError as exc:
            require("workdir" in str(exc), f"unexpected construction failure: {exc}")
        else:
            raise AssertionError("missing workdir was accepted")
        require(len(tracking.created) == before + 1 and absent(tracking.created[-1]),
                "failed workdir probe leaked its real container")
        report["cases"].append("failed construction removes its already-created real container")

        tracking.lose_reply_once = True
        lost = expect_cleanup_error(lambda: provider().open(key="lost-reply", image=image), "acknowledgement")
        unsettled = tracking.get(tracking.created[-1])
        require(unsettled.status == "running" and lost.failures[0].resource_id == unsettled.name,
                "ambiguous creation did not retain the actual container name")
        report["cases"].append("lost create acknowledgement reports the actual unsettled container identity")
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if owned_scope:
            # Fresh SDK objects bypass only this probe's injected client faults.
            # The exact unique label is the authority; never prune by name/PID.
            try:
                for container in client.containers.list(all=True, filters=label):
                    try:
                        require(container.labels.get("taste.validation") == args.owner_token, "cleanup owner mismatch")
                        container.remove(force=True)
                    except BaseException as exc:
                        report["cleanup_errors"].append(f"{container.id}: {type(exc).__name__}: {exc}")
                report["remaining_containers"] = [item.id for item in client.containers.list(all=True, filters=label)]
            except BaseException as exc:
                report["cleanup_errors"].append(f"cleanup inspection: {type(exc).__name__}: {exc}")
            if report["cleanup_errors"] or report.get("remaining_containers"):
                report["status"] = "failed"
        report["created_containers"] = tracking.created
        client.close()
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
