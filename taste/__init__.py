"""Agent OS: a git-native harness for long-running agents.

Public surface:
    from taste import Kernel, Memory, agent, tool

Those names resolve lazily, and that is load-bearing rather than an
optimisation. ``taste.memstore`` is a subpackage, so Python executes this
module before it, and an eager re-export here pulled the entire legacy
harness -- seventeen modules, GitPython and dotenv -- into every process that
imported the memory layer. The layer built to replace the kernel could not be
loaded without the kernel, a brain "consuming only the memstore API" could not
actually do so, and any import-time error in the legacy stack would have
surfaced as a memstore failure. ``tests/test_package_isolation.py`` holds the
line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for type checkers and editors only
    from taste.agent import AgentSpec, agent
    from taste.kernel import Kernel, RunResult
    from taste.llm import BudgetExceeded, InfraFailure, PricingError
    from taste.memory import Checkpoint, Memory
    from taste.tools import Tool, tool

_LAZY = {
    "AgentSpec": "taste.agent",
    "agent": "taste.agent",
    "Kernel": "taste.kernel",
    "RunResult": "taste.kernel",
    "BudgetExceeded": "taste.llm",
    "InfraFailure": "taste.llm",
    "PricingError": "taste.llm",
    "Checkpoint": "taste.memory",
    "Memory": "taste.memory",
    "Tool": "taste.tools",
    "tool": "taste.tools",
}

__all__ = [
    "AgentSpec",
    "BudgetExceeded",
    "Checkpoint",
    "InfraFailure",
    "Kernel",
    "Memory",
    "PricingError",
    "RunResult",
    "Tool",
    "agent",
    "tool",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)
