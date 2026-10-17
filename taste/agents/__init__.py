"""Agents written by others, run unchanged on Taste.

An agent keeps its own loop, prompts, tools and parsing. Taste replaces only
the two things every agent needs from outside itself: a model to ask and a
shell to run commands in. Both are given to it through a ``Host``:

- ``ask`` sends one model request through Taste's journaled, budgeted model
  session and returns the reply's output items as the provider gave them.
- ``run`` runs one shell command in the task's container through Taste's
  terminal broker, so the command ledger, time limits and monitors apply.

An adapter for an agent translates between that agent's own model and
environment interfaces and these two calls, and nothing else. What Taste does
with the record of those calls (monitoring, certification, planning, rollback)
happens outside the agent.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class HostedStop(Exception):
    """Taste ended the agent's run: its budget, deadline or a supervisor's decision.

    Never a model or command error the agent should see and handle. Adapters
    let it propagate so the agent's loop ends where it stands.
    """


@dataclass(frozen=True)
class ModelReply:
    """One model reply, as the provider's Responses API returned it."""

    output: tuple[Mapping[str, Any], ...]
    status: str
    incomplete_reason: str | None
    usage: Mapping[str, Any]
    model: str
    cost_usd: float


@dataclass(frozen=True)
class ShellResult:
    """What one command printed (standard error folded in) and how it ended."""

    output: str
    returncode: int
    timed_out: bool = False


@dataclass(frozen=True)
class AgentExit:
    """How a hosted agent's run ended, in its own words."""

    exit_status: str
    submission: str
    messages: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    model_calls: int = 0


class Host(Protocol):
    def ask(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
            effort: str) -> ModelReply: ...

    def run(self, command: str, *, cwd: str, timeout_seconds: float,
            shown: str | None = None) -> ShellResult:
        """Run ``command``; ``shown`` is the command as the agent wrote it, for the record."""


# Agents that can be hosted, by the name a trial or policy uses. Loaded only
# when a worker runs one, so importing this package imports none of them. The
# checker is the recovery study's reviewer, written here and hosted the same
# way; it is run alone (services none) on a copy of an agent's final files.
HOSTED_AGENTS = {"mini-swe-agent": ("taste.agents.mini_swe_agent", "MiniSweAgent"),
                 "swe-agent": ("taste.agents.swe_agent", "SweAgent"),
                 "checker": ("taste.agents.checker", "Checker")}


# The fields of ``platform.uname()`` mini-swe-agent writes into its task message; node is not among them, but may be
# given so that a host's whole uname can be named.
UNAME_FIELDS = ("system", "release", "version", "machine")


def uname_override(value) -> dict[str, str]:
    """``agent_uname``: the machine a hosted mini-swe-agent is told it runs on, instead of its host's.

    Empty for the host's own. Otherwise one JSON object (or the object itself, as Harbor hands over a JSON
    ``--ak`` value) with exactly the string fields system, release, version and machine, and optionally node:
    a study shows its agent the host its recorded runs had, so a branch's requests match the record's.
    """
    if value in ("", None):
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError("agent_uname must be a JSON object of uname's fields") from None
    if not isinstance(value, Mapping) or not set(UNAME_FIELDS) <= set(value) <= {*UNAME_FIELDS, "node"}:
        raise ValueError("agent_uname must name exactly system, release, version and machine (and optionally node)")
    for item in value.values():
        if type(item) is not str or not item.strip() or any(ord(char) < 32 for char in item):
            raise ValueError("agent_uname's fields must be nonempty one-line strings")
    return dict(value)


def canonical_uname(value) -> str:
    """``agent_uname`` in its one written form (compact JSON, sorted keys), or ""."""
    found = uname_override(value)
    return json.dumps(found, sort_keys=True, separators=(",", ":")) if found else ""


def hosted_agent(name: str, *, effort: str, uname: str = ""):
    """The named agent, configured as its own project configures it, at this effort.

    ``uname`` (``agent_uname``) is mini-swe-agent's alone: the other agents do not report their host, and are
    refused one rather than silently ignoring it.
    """
    if name not in HOSTED_AGENTS:
        raise ValueError(f"no hosted agent named {name!r}")
    module, attribute = HOSTED_AGENTS[name]
    if uname and name != "mini-swe-agent":
        raise ValueError(f"agent_uname is mini-swe-agent's; {name} does not report its host")
    options = {"uname": canonical_uname(uname)} if uname else {}
    return getattr(importlib.import_module(module), attribute)(effort=effort, **options)
