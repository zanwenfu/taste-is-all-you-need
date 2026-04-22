"""Agent definitions: markdown spec + Python decorator.

The blog's DX target: register an agent in ~50 lines of code plus a markdown
file that describes capability, tools, and model assignment. This module
implements that pipe — :class:`AgentSpec` parses the markdown, :func:`agent`
registers the Python side in a module-level registry that the kernel reads.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

T = TypeVar("T", bound=Callable[..., object])


@dataclass
class AgentSpec:
    """Declarative description of an agent's capability.

    Mirrors the `agent_desp.md` format from the blog: a YAML-ish frontmatter
    with ``name``, ``description``, ``tools``, ``model``, ``triggers``, plus
    a free-form markdown body used as the system prompt.
    """

    name: str
    description: str
    tools: list[str] = field(default_factory=list)
    model: str | None = None
    triggers: list[str] = field(default_factory=list)
    system_prompt: str = ""
    source_path: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> AgentSpec:
        md = Path(path).resolve()
        if not md.exists():
            raise FileNotFoundError(f"agent spec not found: {md}")
        frontmatter, body = _split_frontmatter(md.read_text())
        meta = _parse_frontmatter(frontmatter)

        if "name" not in meta:
            raise ValueError(f"{md}: spec must declare a `name`")

        return cls(
            name=str(meta["name"]),
            description=str(meta.get("description", "")),
            tools=_as_list(meta.get("tools", [])),
            model=str(meta["model"]) if meta.get("model") else None,
            triggers=_as_list(meta.get("triggers", [])),
            system_prompt=body.strip(),
            source_path=md,
        )


# Module-level registry. The kernel reads this when a user invokes a
# registered agent by name from the CLI.
AGENT_REGISTRY: dict[str, AgentSpec] = {}


def agent(config: str | Path) -> Callable[[T], T]:
    """Bind a Python function to a markdown spec and register it globally.

    The ``config`` path is resolved relative to the caller's file, matching
    the blog's `@agent(config="agent_desp.md")` ergonomics — drop the md
    next to the python file and it Just Works.
    """

    def decorator(fn: T) -> T:
        caller_dir = Path(inspect.stack()[1].filename).resolve().parent
        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = caller_dir / config_path

        spec = AgentSpec.from_file(config_path)
        AGENT_REGISTRY[spec.name] = spec
        fn.__agent_spec__ = spec  # type: ignore[attr-defined]
        return fn

    return decorator


# ---------------------------------------------------------------- parsing


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter, body). Tolerates files with or without --- fences."""
    stripped = text.lstrip()
    if stripped.startswith("---"):
        parts = stripped.split("---", 2)
        if len(parts) >= 3:
            return parts[1], parts[2]
    # No fence: treat the leading `key: value` lines as frontmatter until a blank line.
    lines = text.splitlines()
    split_idx = next(
        (i for i, line in enumerate(lines) if not line.strip() and i > 0),
        len(lines),
    )
    return "\n".join(lines[:split_idx]), "\n".join(lines[split_idx:])


def _parse_frontmatter(raw: str) -> dict[str, object]:
    """Minimal YAML-ish parser: `key: value` and `key: [a, b, c]`.

    Kept dependency-free on purpose. If/when the schema outgrows this, swap
    in ``pyyaml`` without changing the public API.
    """
    out: dict[str, object] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                out[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                out[key] = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
        else:
            out[key] = value.strip("'\"")
    return out


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str) and value:
        return [value]
    return []
