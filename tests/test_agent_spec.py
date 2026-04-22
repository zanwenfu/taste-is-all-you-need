"""AgentSpec parsing — markdown frontmatter → typed spec."""

from __future__ import annotations

from pathlib import Path

import pytest

from taste.agent import AGENT_REGISTRY, AgentSpec


def test_parses_yaml_frontmatter(tmp_path: Path) -> None:
    spec_file = tmp_path / "agent_desp.md"
    spec_file.write_text(
        "---\n"
        "name: sec_retriever\n"
        "description: fetches SEC filings\n"
        "tools: [search_api, validate]\n"
        "model: claude-sonnet-4-6\n"
        "triggers: [\"10-K\", \"filing\"]\n"
        "---\n\n"
        "You are a careful retriever of financial data.\n"
    )
    spec = AgentSpec.from_file(spec_file)
    assert spec.name == "sec_retriever"
    assert spec.description == "fetches SEC filings"
    assert spec.tools == ["search_api", "validate"]
    assert spec.model == "claude-sonnet-4-6"
    assert spec.triggers == ["10-K", "filing"]
    assert "careful retriever" in spec.system_prompt


def test_parses_headerless_frontmatter(tmp_path: Path) -> None:
    """Blog example: no `---` fences, just leading key: value lines."""
    spec_file = tmp_path / "agent_desp.md"
    spec_file.write_text(
        "name: minimal\n"
        "description: just a name\n"
        "\n"
        "Prompt body goes here.\n"
    )
    spec = AgentSpec.from_file(spec_file)
    assert spec.name == "minimal"
    assert spec.description == "just a name"
    assert spec.system_prompt == "Prompt body goes here."


def test_decorator_registers_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent_desp.md").write_text("name: demo\ndescription: demo agent\n\nbody\n")

    AGENT_REGISTRY.clear()

    # Emulate a user file defining a decorated agent.
    user_file = tmp_path / "my_agent.py"
    user_file.write_text(
        "from taste.agent import agent\n\n"
        "@agent(config='agent_desp.md')\n"
        "def demo(query):\n"
        "    return query\n"
    )
    import runpy
    runpy.run_path(str(user_file))

    assert "demo" in AGENT_REGISTRY
    assert AGENT_REGISTRY["demo"].description == "demo agent"


def test_missing_name_raises(tmp_path: Path) -> None:
    spec_file = tmp_path / "bad.md"
    spec_file.write_text("description: no name\n\nbody\n")
    with pytest.raises(ValueError, match="name"):
        AgentSpec.from_file(spec_file)
