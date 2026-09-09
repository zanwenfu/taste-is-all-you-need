"""The memory layer must be loadable without the harness it replaces.

The build order is bottom-up, one robust layer at a time, and the layer above
memstore is meant to consume only the memstore API. That was not enforceable
while ``taste/__init__.py`` re-exported the legacy stack eagerly: because
``taste.memstore`` is a subpackage, Python ran the parent first and pulled in
seventeen legacy modules, so "consumes only memstore" was false the moment
anyone imported it.

These run in subprocesses because import isolation cannot be observed from
inside a process that has already imported everything.
"""

from __future__ import annotations

import subprocess
import sys

LEGACY = (
    "taste.kernel",
    "taste.memory",
    "taste.cores",
    "taste.recovery",
    "taste.journal",
    "taste.shadow",
    "taste.integrate",
    "taste.agent",
    "taste.tools",
    "taste.llm",
)


def _run(code: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout.strip()


def test_importing_memstore_does_not_load_the_legacy_harness() -> None:
    leaked = _run(
        "import sys, taste.memstore\n"
        f"legacy = {LEGACY!r}\n"
        "print(','.join(m for m in legacy if m in sys.modules))\n"
    )
    assert leaked == "", f"importing taste.memstore dragged in: {leaked}"


def test_memstore_does_not_pull_in_the_llm_or_provider_stack() -> None:
    """A memory layer has no business importing an API client."""
    leaked = _run(
        "import sys, taste.memstore\n"
        "print(','.join(sorted(m for m in sys.modules "
        "if m.startswith('anthropic') or m.startswith('openai'))))\n"
    )
    assert leaked == "", f"memstore pulled in a model client: {leaked}"


def test_the_legacy_public_surface_still_works() -> None:
    """Laziness must not be a breaking change: the documented imports stand."""
    out = _run(
        "from taste import Kernel, Memory, agent, tool, AgentSpec, RunResult\n"
        "from taste import BudgetExceeded, InfraFailure, PricingError, Checkpoint, Tool\n"
        "import taste\n"
        "print(taste.__version__, Kernel.__name__, Memory.__name__)\n"
    )
    assert out == "0.1.0 Kernel Memory"


def test_an_unknown_attribute_still_raises_attribute_error() -> None:
    out = _run(
        "import taste\n"
        "try:\n"
        "    taste.NoSuchThing\n"
        "except AttributeError as exc:\n"
        "    print('ok')\n"
    )
    assert out == "ok"
