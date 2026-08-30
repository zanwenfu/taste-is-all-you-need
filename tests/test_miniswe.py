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
    # the scaffold's environment variables travel with every command; the command itself is
    # intact inside a brace group whose streams are merged the way the scaffold's docker exec merges them
    assert router.commands[0].startswith("export ") and router.commands[0].endswith("{\necho hi\n} 2>&1")
    assert "PAGER=cat" in router.commands[0]


def test_the_merged_stream_wrapper_is_valid_bash_for_heredocs_and_comments() -> None:
    import subprocess

    for command in ("python3 - <<'PY'\nimport sys; print('out'); print('err', file=sys.stderr)\nPY", "echo hi # trailing"):
        seen = []

        class Rec:
            def exec(self, c, *, timeout=600):
                seen.append(c)
                p = subprocess.run(["bash", "-lc", c], capture_output=True, text=True)
                return _Result(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)

        out = TasteEnvironment(Rec(), None, env={}).execute({"command": command})
        assert out["returncode"] == 0, out
        assert "delimited by end-of-file" not in out["output"] and "syntax error" not in out["output"]
    assert "err" in out["output"] or "hi" in out["output"]
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
    assert out["returncode"] == -1 and "timed out" in out["exception_info"]


def test_environment_reports_a_one_off_failure_the_scaffold_way_then_declares_death() -> None:
    class Boom:
        def exec(self, command, *, timeout=600):
            raise RuntimeError("container gone")

    env = TasteEnvironment(Boom(), None)
    out = env.execute({"command": "ls"})
    assert out["returncode"] == -1 and "container gone" in out["exception_info"]
    env.execute({"command": "ls"})
    with pytest.raises(RuntimeError, match="environment dead"):
        env.execute({"command": "ls"})


def test_observations_carry_the_scaffolds_running_cost() -> None:
    env = TasteEnvironment(_Router(), None)
    env.agent = type("Agent", (), {"cost": 0.42})()
    env.execute({"command": "true"})
    assert env.cost_pair() == (0.42, 0.42)


@pytest.mark.parametrize(
    "command",
    [
        "python3 - <<'PY'\nprint('heredoc')\nPY",
        "echo trailing # a comment at the very end",
        "printf 'x\\n' | cat",
    ],
)
def test_routed_command_survives_heredoc_terminators_and_comments(command: str) -> None:
    """Defect 38: ``(cmd)`` on one line makes ``EOF)`` a non-terminator."""
    import subprocess

    from taste.execution import routed_command

    wrapped = routed_command("", ".", command)
    proc = subprocess.run(["bash", "-lc", wrapped], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "delimited by end-of-file" not in proc.stderr and "syntax error" not in proc.stderr


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
