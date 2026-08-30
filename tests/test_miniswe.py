"""mini-swe-agent under the instrument: the environment seam and the result shape."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from taste.benchmarks.miniswe import ScriptedModel, TasteEnvironment
from taste.config import HarnessConfig


@dataclass
class _Result:
    exit_code: int
    stdout: str
    stderr: str = ""
    timed_out: bool = False


class _Router:
    """Records the commands it was given; pretends the second one mutated the tree."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def exec(self, command: str, *, timeout: int = 600) -> _Result:
        self.commands.append(command)
        if "timeout" in command:
            return _Result(exit_code=124, stdout="", timed_out=True)
        return _Result(exit_code=0, stdout=f"ran {len(self.commands)}", stderr="warn" if len(self.commands) == 1 else "")


class _Shadow:
    """Observes only when the router says the tree changed (every second call here)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def observe(self, **kwargs):
        self.calls.append(kwargs)
        return object() if len(self.calls) % 2 == 0 else None


def test_environment_routes_commands_and_observes_after_each() -> None:
    router, shadow = _Router(), _Shadow()
    env = TasteEnvironment(router, shadow, cwd="/testbed", timeout=7)
    out = env.execute({"command": "echo hi"})
    assert out["returncode"] == 0 and "ran 1" in out["output"] and "warn" in out["output"]
    assert out["exception_info"] == ""
    # the scaffold's environment variables travel with every command; the command itself is intact
    assert router.commands[0].startswith("export ") and router.commands[0].endswith("echo hi")
    assert "PAGER=cat" in router.commands[0]
    env.execute({"command": "touch x"})
    assert [c["trigger"] for c in shadow.calls] == ["tool", "tool"]
    assert shadow.calls[0]["step_id"] == "cmd-001" and shadow.calls[1]["step_id"] == "cmd-002"
    assert env.observed == 1  # only the mutating command produced a commit


def test_environment_raises_submitted_on_the_scaffold_sentinel() -> None:
    pytest.importorskip("minisweagent")
    from minisweagent.exceptions import Submitted

    class Echo:
        def exec(self, command, *, timeout=600):
            return _Result(exit_code=0, stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a/x b/x\n")

    env = TasteEnvironment(Echo(), None)
    with pytest.raises(Submitted) as raised:
        env.execute({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff"})
    exit_message = raised.value.messages[0]
    assert exit_message["extra"]["exit_status"] == "Submitted"
    assert exit_message["extra"]["submission"] == "diff --git a/x b/x\n"


def test_environment_reports_timeouts_the_scaffold_way() -> None:
    env = TasteEnvironment(_Router(), None)
    out = env.execute({"command": "sleep timeout"})
    assert out["returncode"] == 124 and out["exception_info"].startswith("TimeoutError")


def test_environment_never_raises_into_the_scaffold() -> None:
    class Boom:
        def exec(self, command, *, timeout=600):
            raise RuntimeError("container gone")

    out = TasteEnvironment(Boom(), None).execute({"command": "ls"})
    assert out["returncode"] == -1 and "container gone" in out["exception_info"]


def test_scripted_model_walks_its_commands_then_submits() -> None:
    pytest.importorskip("minisweagent")
    from minisweagent.exceptions import Submitted

    model = ScriptedModel(["ls", "echo done"])
    first = model.query([])
    assert first["extra"]["actions"] == [{"command": "ls"}]
    model.query([])
    with pytest.raises(Submitted) as raised:
        model.query([])
    assert raised.value.messages[0]["role"] == "exit"
    assert raised.value.messages[0]["extra"]["exit_status"] == "Submitted"
    assert "model" in model.serialize()["info"]["config"]


def test_scaffold_pricing_comes_from_our_verified_table() -> None:
    from taste.benchmarks.miniswe import scaffold_pricing

    entry = scaffold_pricing("anthropic/claude-sonnet-4-6")
    assert entry is not None and entry["litellm_provider"] == "anthropic"
    assert entry["input_cost_per_token"] == 3.00 / 1e6 and entry["output_cost_per_token"] == 15.00 / 1e6
    assert scaffold_pricing("openai/gpt-5.6-sol")["input_cost_per_token"] == 5.00 / 1e6
    assert scaffold_pricing("anthropic/claude-sonnet-4-5-20250929") is None  # litellm's own map covers it


def test_mswe_is_an_arm_with_no_recovery_and_a_shadow() -> None:
    cfg = HarnessConfig.arm("MSWE")
    assert cfg.shadow and not cfg.regression_gate
    assert "MSWE" in HarnessConfig.arm_names()
    assert cfg.hash() != HarnessConfig.arm("A0").hash()
