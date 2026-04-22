"""Tool registry — built-ins + CLI discovery."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from taste.tools import ToolRegistry, discover_cli_tools, make_builtin_tools


def test_builtin_read_write(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("hello")
    tools = {t.name: t for t in make_builtin_tools(ws)}

    assert "hello" in tools["read_file"].invoke({"path": "a.txt"})
    tools["write_file"].invoke({"path": "b/nested.txt", "content": "yo"})
    assert (ws / "b" / "nested.txt").read_text() == "yo"


def test_builtin_rejects_path_escape(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "outside.txt").write_text("secret")
    tools = {t.name: t for t in make_builtin_tools(ws)}
    with pytest.raises(PermissionError):
        tools["read_file"].invoke({"path": "../outside.txt"})


def test_run_shell_captures_output(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    tools = {t.name: t for t in make_builtin_tools(ws)}
    out = tools["run_shell"].invoke({"command": "echo taste-os"})
    assert "taste-os" in out
    assert "(exit 0)" in out


def test_registry_anthropic_schema() -> None:
    reg = ToolRegistry()
    reg.extend(make_builtin_tools(Path("/")))
    schemas = reg.to_anthropic()
    names = {s["name"] for s in schemas}
    assert names == {"read_file", "write_file", "run_shell"}
    assert all("input_schema" in s for s in schemas)


def test_discover_cli_tools(tmp_path: Path) -> None:
    """The CLI-first discovery pattern: tool_desp.md + run.sh in a subdir."""
    root = tmp_path / "tools"
    tool_dir = root / "echo_tool"
    tool_dir.mkdir(parents=True)
    (tool_dir / "tool_desp.md").write_text(
        "name: echo_tool\n"
        "description: echoes its stdin payload\n"
        'input_schema: {"type": "object", "properties": {"msg": {"type": "string"}}}\n'
    )
    run_script = tool_dir / "run.sh"
    run_script.write_text("#!/bin/sh\ncat\n")
    run_script.chmod(run_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    tools = discover_cli_tools(root)
    assert len(tools) == 1
    t = tools[0]
    assert t.name == "echo_tool"
    assert "msg" in t.input_schema["properties"]

    out = t.invoke({"msg": "hi"})
    assert '"msg": "hi"' in out or '"msg":"hi"' in out


def test_discover_ignores_incomplete_tools(tmp_path: Path) -> None:
    root = tmp_path / "tools"
    (root / "only_md").mkdir(parents=True)
    (root / "only_md" / "tool_desp.md").write_text("name: partial\n")
    assert discover_cli_tools(root) == []
