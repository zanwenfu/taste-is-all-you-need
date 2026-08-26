"""The Worker's tool-use loop — the code that actually talks to the model.

Every other kernel test bypasses this via ``worker_override``, so until now
:func:`taste.cores.execute` was the least-tested and most load-bearing
function in the project. These tests drive it for real with a scripted
:class:`~tests.fakes.FakeLLM` and zero API calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taste import cores
from taste.agent import AgentSpec
from taste.cores import Step, Verification
from taste.llm import MODEL_WORKER
from taste.tools import ToolRegistry, make_builtin_tools
from tests.fakes import FakeLLM, FakeTurn, ScriptExhausted


def _spec() -> AgentSpec:
    return AgentSpec(name="w", description="a worker", system_prompt="be careful")


def _step() -> Step:
    return Step(
        id="step-01",
        description="write the module",
        verification=Verification(kind="shell", command="true"),
    )


def _tools(workspace: Path) -> ToolRegistry:
    registry = ToolRegistry()
    registry.extend(make_builtin_tools(workspace))
    return registry


def _run(llm: FakeLLM, workspace: Path, **kwargs):
    return cores.execute(
        llm,
        spec=_spec(),
        step=_step(),
        plan_context="Plan: do the thing",
        tools=_tools(workspace),
        **kwargs,
    )


# ------------------------------------------------------------------ happy path


def test_single_turn_with_no_tools_returns_summary(tmp_path: Path) -> None:
    llm = FakeLLM([FakeTurn(text="nothing to do")])
    result = _run(llm, tmp_path)

    assert result.summary == "nothing to do"
    assert result.tool_calls == 0
    assert result.stopped_reason == "end_turn"
    assert llm.call_count == 1


def test_tool_call_is_executed_and_loop_continues(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            FakeTurn(tool_calls=[("write_file", {"path": "mod.py", "content": "x = 1\n"})]),
            FakeTurn(text="wrote the module"),
        ]
    )
    result = _run(llm, tmp_path)

    assert (tmp_path / "mod.py").read_text() == "x = 1\n"
    assert result.tool_calls == 1
    assert result.summary == "wrote the module"
    assert llm.call_count == 2


def test_multiple_tools_in_one_turn_all_execute(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            FakeTurn(
                tool_calls=[
                    ("write_file", {"path": "a.py", "content": "a = 1\n"}),
                    ("write_file", {"path": "b.py", "content": "b = 2\n"}),
                ]
            ),
            FakeTurn(text="both written"),
        ]
    )
    result = _run(llm, tmp_path)

    assert (tmp_path / "a.py").exists() and (tmp_path / "b.py").exists()
    assert result.tool_calls == 2


# ------------------------------------------------------------------ transcript shape


def test_tool_results_are_returned_with_matching_tool_use_id(tmp_path: Path) -> None:
    """A mismatched tool_use_id is an API error; the loop must echo it exactly."""
    llm = FakeLLM(
        [
            FakeTurn(tool_calls=[("read_file", {"path": "missing.py"})]),
            FakeTurn(text="ok"),
        ]
    )
    _run(llm, tmp_path)

    # The assistant turn the loop appended, then the user turn carrying results.
    second_call_messages = llm.calls[1]["messages"]
    assistant = next(m for m in second_call_messages if m["role"] == "assistant")
    tool_use = next(b for b in assistant["content"] if b["type"] == "tool_use")
    results = llm.last_user_content()

    assert results[0]["type"] == "tool_result"
    assert results[0]["tool_use_id"] == tool_use["id"]


def test_assistant_turns_accumulate_in_the_transcript(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            FakeTurn(text="first", tool_calls=[("write_file", {"path": "a", "content": "1"})]),
            FakeTurn(text="second", tool_calls=[("write_file", {"path": "b", "content": "2"})]),
            FakeTurn(text="done"),
        ]
    )
    _run(llm, tmp_path)

    roles = [m["role"] for m in llm.calls[-1]["messages"]]
    # user(task), assistant, user(results), assistant, user(results)
    assert roles == ["user", "assistant", "user", "assistant", "user"]


def test_text_and_tool_use_blocks_both_survive_round_trip(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            FakeTurn(text="thinking", tool_calls=[("write_file", {"path": "a", "content": "1"})]),
            FakeTurn(text="done"),
        ]
    )
    _run(llm, tmp_path)

    assistant = next(m for m in llm.calls[1]["messages"] if m["role"] == "assistant")
    kinds = [b["type"] for b in assistant["content"]]
    assert kinds == ["text", "tool_use"]
    assert assistant["content"][0]["text"] == "thinking"


# ------------------------------------------------------------------ robustness


def test_tool_error_is_surfaced_to_the_model_not_raised(tmp_path: Path) -> None:
    """A failing tool must become an observation, not kill the run."""
    llm = FakeLLM(
        [
            FakeTurn(tool_calls=[("write_file", {"path": "../escape.py", "content": "x"})]),
            FakeTurn(text="understood, staying inside"),
        ]
    )
    result = _run(llm, tmp_path)

    results = llm.last_user_content()
    assert "TOOL ERROR" in results[0]["content"]
    assert "PermissionError" in results[0]["content"]
    assert result.summary == "understood, staying inside"


def test_unknown_tool_name_is_surfaced_not_raised(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            FakeTurn(tool_calls=[("teleport", {"destination": "mars"})]),
            FakeTurn(text="no such tool"),
        ]
    )
    _run(llm, tmp_path)
    assert "TOOL ERROR" in llm.last_user_content()[0]["content"]


def test_max_turns_bounds_the_loop(tmp_path: Path) -> None:
    """A model that never stops calling tools must not loop forever."""
    llm = FakeLLM(
        [FakeTurn(tool_calls=[("write_file", {"path": "a", "content": "1"})])] * 10
    )
    result = _run(llm, tmp_path, max_turns=3)

    assert llm.call_count == 3
    assert result.tool_calls == 3


def test_script_exhaustion_is_loud(tmp_path: Path) -> None:
    """Guards the tests themselves: a silent replay would hide runaway loops."""
    llm = FakeLLM([FakeTurn(tool_calls=[("write_file", {"path": "a", "content": "1"})])])
    with pytest.raises(ScriptExhausted):
        _run(llm, tmp_path, max_turns=5)


def test_no_summary_emitted_yields_placeholder(tmp_path: Path) -> None:
    llm = FakeLLM([FakeTurn(stop_reason="end_turn")])
    assert _run(llm, tmp_path).summary == "(no summary emitted)"


# ------------------------------------------------------------------ wiring


def test_worker_calls_are_tagged_with_the_worker_role(tmp_path: Path) -> None:
    """Per-role cost attribution depends on every call site tagging itself."""
    llm = FakeLLM([FakeTurn(text="done")])
    _run(llm, tmp_path)

    assert llm.roles_called() == ["worker"]
    assert llm.stats.per_role_model[("worker", MODEL_WORKER)].calls == 1
    assert llm.stats.total_cost_usd > 0


def test_tools_and_system_prompt_reach_the_model(tmp_path: Path) -> None:
    llm = FakeLLM([FakeTurn(text="done")])
    _run(llm, tmp_path)

    call = llm.calls[0]
    assert {t["name"] for t in call["tools"]} == {"read_file", "write_file", "run_shell"}
    system_text = " ".join(block["text"] for block in call["system"])
    assert "Worker core" in system_text
    assert "be careful" in system_text          # the agent spec's own prompt
    assert "Plan: do the thing" in system_text  # plan context


def test_step_description_is_in_the_first_user_message(tmp_path: Path) -> None:
    llm = FakeLLM([FakeTurn(text="done")])
    _run(llm, tmp_path)

    first_user = llm.calls[0]["messages"][0]
    assert first_user["role"] == "user"
    assert "write the module" in first_user["content"]
    assert "step-01" in first_user["content"]


def test_cache_tokens_are_recorded_from_usage(tmp_path: Path) -> None:
    llm = FakeLLM([FakeTurn(text="done", input_tokens=10, cache_read_tokens=990)])
    _run(llm, tmp_path)
    assert llm.stats.cache_hit_rate == pytest.approx(0.99)


def test_a_plan_step_sent_as_a_json_string_still_parses() -> None:
    """Found on a real planner call, which killed the run before a single
    step executed.

    The tool schema asks for objects and models mostly comply, but one
    returned `steps` as a list of JSON strings -- "string indices must be
    integers". Nested-object stringification differs across providers, so this
    is portability, not a one-off.
    """
    from taste.cores import _coerce_step

    encoded = '{"id": "step-01", "description": "do it", "verification": {"kind": "shell"}}'
    assert _coerce_step(encoded)["id"] == "step-01"
    assert _coerce_step({"id": "step-02"})["id"] == "step-02"


def test_a_step_that_is_neither_object_nor_encoded_object_raises() -> None:
    """Decoding a string is not the same as accepting anything. A plan we
    cannot read must fail loudly rather than run half of itself."""
    from taste.cores import _coerce_step

    for bad in (["step-01"], 42, '"just a string"', "[1,2]"):
        with pytest.raises((TypeError, ValueError)):
            _coerce_step(bad)


def test_the_error_names_the_shape_that_arrived() -> None:
    """Without it the failure reads "string indices must be integers" with no
    hint of which layer was wrong, and diagnosing it costs a run."""
    from taste.cores import _describe_shape

    assert _describe_shape({"steps": ["a", "b"]}) == "steps=list[str] n=2"
    assert _describe_shape({"steps": [{"id": "x"}]}) == "steps=list[dict] n=1"
    assert _describe_shape({"steps": "oops"}) == "steps=str"
    assert _describe_shape("not a dict") == "str"


def test_a_malformed_plan_is_retried_with_the_complaint_fed_back() -> None:
    """Measured on real tasks: a single bad tool call killed 2 of 6 runs.

    Once with steps as JSON strings, once with `steps` missing entirely. Each
    was recorded as a *task* failure, so it would have counted as evidence
    about whichever arm happened to draw it.
    """
    from taste.cores import plan
    from tests.fakes import FakeLLM, FakeTurn

    good = {"steps": [{"id": "step-01", "description": "d",
                       "verification": {"kind": "shell", "command": "true"}}]}
    llm = FakeLLM([
        FakeTurn(tool_calls=[("submit_plan", {"steps": None})]),
        FakeTurn(tool_calls=[("submit_plan", good)]),
    ])
    spec = AgentSpec(name="a", description="", system_prompt="p")

    result = plan(llm, "task", spec, "summary")

    assert [s.id for s in result.steps] == ["step-01"]
    assert len(llm.calls) == 2, "the planner must be asked again"
    retry = llm.calls[-1]["messages"][-1]["content"]
    assert "could not be read" in retry and "steps=NoneType" in retry, (
        "the model must be told what actually arrived, not just asked again"
    )


def test_a_planner_that_never_complies_still_fails() -> None:
    """Retrying is not the same as accepting. A genuine refusal must surface."""
    import pytest

    from taste.cores import PlannerError, plan
    from tests.fakes import FakeLLM, FakeTurn

    llm = FakeLLM([FakeTurn(tool_calls=[("submit_plan", {"steps": None})]) for _ in range(3)])
    with pytest.raises(PlannerError, match="malformed plan payload"):
        plan(llm, "task", AgentSpec(name="a", description="", system_prompt="p"), "s")
    assert len(llm.calls) == 3, "bounded, not infinite"


# ------------------------------------------------------------ truncation


def test_a_capped_turn_is_discarded_and_its_tool_calls_never_execute(tmp_path: Path) -> None:
    """A turn that hit the output ceiling may carry a tool call whose JSON is
    valid but whose CONTENT was cut mid-file. Executing it writes the mangled
    artifact into the tree, the Monitor fails it, and the whole thing reads
    as the agent's incompetence — measured once as a 78-event regression
    storm. The turn is discarded wholesale and the model told to split."""
    llm = FakeLLM([
        FakeTurn(
            tool_calls=[("write_file", {"path": "half.py", "content": "def f(:"})],
            stop_reason="max_tokens",
        ),
        FakeTurn(
            tool_calls=[("write_file", {"path": "whole.py", "content": "def f():\n    return 1\n"})],
        ),
        FakeTurn(text="done", stop_reason="end_turn"),
    ])
    result = _run(llm, tmp_path)

    assert not (tmp_path / "half.py").exists(), (
        "the truncated turn's write executed — the mangled file is in the tree"
    )
    assert (tmp_path / "whole.py").exists()
    assert result.tool_calls == 1, "only the intact turn's call may count"
    feedback = llm.calls[1]["messages"][-1]
    assert feedback["role"] == "user"
    assert "output limit" in str(feedback["content"]), (
        "the model was not told why its turn vanished"
    )


def test_worker_turns_request_a_ceiling_far_above_a_whole_file(tmp_path: Path) -> None:
    """4096 was the original cap and a single write_file of a real module
    exceeds it routinely; the request must carry the raised ceiling."""
    llm = FakeLLM([FakeTurn(text="done", stop_reason="end_turn")])
    _run(llm, tmp_path)
    assert llm.calls[0]["max_tokens"] >= 32_000
