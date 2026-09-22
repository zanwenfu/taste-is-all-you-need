"""Resource cleanup failures must stop admissions independently of known costs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any


def close_resources(callbacks: Iterable[Callable[[], None]]) -> None:
    """Attempt every owned close and preserve all failures, including interrupts."""
    errors: list[BaseException] = []
    for close in callbacks:
        try:
            close()
        except BaseException as exc:
            errors.append(exc)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("resource cleanup failed", errors)


@dataclass(frozen=True)
class ResourceFailure:
    resource_type: str
    resource_id: str
    operation: str
    error: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class ResourceCleanupError(RuntimeError):
    """A resource cannot be confirmed released; its owner must remain recorded."""

    def __init__(self, failures: tuple[ResourceFailure, ...]):
        if not failures:
            raise ValueError("resource cleanup failure needs evidence")
        self.failures = failures
        super().__init__("; ".join(
            f"{item.resource_type} {item.resource_id}: {item.operation}: {item.error}"
            for item in failures
        ))


def resource_error(resource_type: str, resource_id: str, operation: str,
                   error: BaseException) -> ResourceCleanupError:
    return ResourceCleanupError((ResourceFailure(
        resource_type, resource_id, operation, f"{type(error).__name__}: {error}",
    ),))


def resource_failures(error: BaseException) -> tuple[ResourceFailure, ...]:
    """Find failures even when a later finally block masks the original error."""
    pending = [error]
    seen: set[int] = set()
    found: dict[ResourceFailure, None] = {}
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ResourceCleanupError):
            found.update(dict.fromkeys(current.failures))
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        for nested in (current.__cause__, current.__context__):
            if nested is not None:
                pending.append(nested)
    return tuple(found)


def close_sandbox(sandbox: Any) -> None:
    """Adapt a sandbox close failure to the sweep's resource-failure contract."""
    try:
        sandbox.close()
    except ResourceCleanupError:
        raise
    except BaseException as exc:
        # Interrupting close also leaves cleanup unconfirmed. Keep the original
        # interrupt as the cause, and stop instead of admitting another trial.
        container = getattr(sandbox, "container", None)
        identity = getattr(container, "id", None)
        resource_type = "docker_container" if identity is not None else "sandbox"
        if identity is None:
            identity = f"{type(sandbox).__module__}.{type(sandbox).__qualname__}"
        raise resource_error(resource_type, str(identity), "close", exc) from exc
