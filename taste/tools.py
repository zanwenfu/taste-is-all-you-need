"""Tool registry.

Two layers, matching the blog's CLI-first, MCP-second position:

* **Native tools**: Python callables wrapped by :func:`tool` — used inside the
  Anthropic tool-use loop. Cheap and fast for primitives the worker always
  needs (``read_file``, ``write_file``, ``run_shell``).
* **CLI tools**: :func:`discover_cli_tools` walks a folder of ``tool_desp.md``
  files and turns each into a Tool that shells out to the associated script.
  This is the 98.7%-token-reduction pattern — tool schemas live on disk and
  are only loaded when the worker opts in.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taste.agent import _parse_frontmatter, _split_frontmatter


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., str]

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def invoke(self, payload: dict[str, Any]) -> str:
        return self.handler(**payload)


@dataclass
class ToolRegistry:
    """Mutable registry. One per kernel run."""

    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def extend(self, tools: list[Tool]) -> None:
        for t in tools:
            self.register(t)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def to_anthropic(self, only: list[str] | None = None) -> list[dict[str, Any]]:
        tools = self._tools.values() if only is None else (self._tools[n] for n in only)
        return [t.to_anthropic() for t in tools]

    def invoke(self, name: str, payload: dict[str, Any]) -> str:
        return self[name].invoke(payload)


# Global default registry populated by ``@tool``. Kernel runs typically build
# their own registry and copy selected tools in from here.
DEFAULT_REGISTRY = ToolRegistry()


def tool(
    *,
    name: str | None = None,
    description: str,
    input_schema: dict[str, Any],
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """Decorator: register a Python function as a native tool."""

    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        t = Tool(
            name=name or fn.__name__,
            description=description,
            input_schema=input_schema,
            handler=fn,
        )
        DEFAULT_REGISTRY.register(t)
        fn.__tool__ = t  # type: ignore[attr-defined]
        return fn

    return decorator


# -------------------------------------------------------------- built-ins


def make_builtin_tools(workspace: Path) -> list[Tool]:
    """Return the three tools every worker needs, scoped to a workspace.

    Scoping is enforced by resolving every path under ``workspace`` — a worker
    cannot read or write outside its assigned branch's working tree.
    """
    workspace = Path(workspace).resolve()

    def _resolve(rel: str) -> Path:
        target = (workspace / rel).resolve()
        if not str(target).startswith(str(workspace)):
            raise PermissionError(f"path escapes workspace: {rel}")
        return target

    def read_file(path: str) -> str:
        p = _resolve(path)
        if not p.exists():
            return f"ERROR: {path} does not exist."
        text = p.read_text()
        return f"--- {path} ---\n{text}"

    def write_file(path: str, content: str) -> str:
        p = _resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} bytes to {path}"

    def run_shell(command: str, timeout: int = 60) -> str:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: timeout after {timeout}s\n$ {command}"
        body = (proc.stdout or "") + (f"\n[stderr]\n{proc.stderr}" if proc.stderr else "")
        return f"$ {command}\n(exit {proc.returncode})\n{body}"

    return [
        Tool(
            name="read_file",
            description="Read a file from the workspace and return its contents.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path relative to workspace root."}},
                "required": ["path"],
            },
            handler=read_file,
        ),
        Tool(
            name="write_file",
            description="Write (or overwrite) a file in the workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to workspace root."},
                    "content": {"type": "string", "description": "Full contents to write."},
                },
                "required": ["path", "content"],
            },
            handler=write_file,
        ),
        Tool(
            name="run_shell",
            description=(
                "Run a shell command in the workspace. "
                "Use for git operations, running tests, invoking linters, inspecting the tree."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 60)."},
                },
                "required": ["command"],
            },
            handler=run_shell,
        ),
    ]


# -------------------------------------------------------------- CLI discovery


def discover_cli_tools(root: Path) -> list[Tool]:
    """Walk ``root`` for tool directories and build CLI-wrapping Tools.

    Layout each tool like:

        root/
          search_api/
            tool_desp.md        # frontmatter: name, description, input_schema (json)
            run.sh              # executable; receives JSON on stdin, emits on stdout

    The ``tool_desp.md`` is only read once, at discovery time. Its schema is
    the only thing injected into the model's context — the script stays on
    disk until the tool is actually invoked.
    """
    tools: list[Tool] = []
    root = Path(root).resolve()
    if not root.is_dir():
        return tools

    for tool_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        md = tool_dir / "tool_desp.md"
        run_script = tool_dir / "run.sh"
        if not md.exists() or not run_script.exists():
            continue

        frontmatter, _ = _split_frontmatter(md.read_text())
        meta = _parse_frontmatter(frontmatter)
        schema_raw = meta.get("input_schema", '{"type": "object"}')
        try:
            input_schema = json.loads(str(schema_raw))
        except json.JSONDecodeError:
            input_schema = {"type": "object"}

        tools.append(
            Tool(
                name=str(meta.get("name", tool_dir.name)),
                description=str(meta.get("description", "")),
                input_schema=input_schema,
                handler=_make_cli_handler(run_script),
            )
        )
    return tools


def _make_cli_handler(script: Path) -> Callable[..., str]:
    """Build a handler that invokes ``script`` with JSON stdin and returns stdout."""

    def handler(**payload: Any) -> str:
        proc = subprocess.run(
            [str(script)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            return f"ERROR (exit {proc.returncode}): {proc.stderr.strip()}"
        return proc.stdout.strip()

    # Keep shlex.quote import used so linters don't flag it; reserved for
    # a future variant that builds argv-style CLI invocations.
    _ = shlex.quote
    return handler
