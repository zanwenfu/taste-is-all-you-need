"""AgentRewind as the recovery study runs it: upstream's text unchanged, and how its two tools are answered."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from taste.agents import agentrewind
from taste.agents.agentrewind_upstream import BACKTRACK_TOOL_DEFINITIONS, DEFAULT_MEMORY_PREFIX

# sha256 of each block of replay-agent-recorder's replay/backtrack.py at commit 6661046 (the file's own sha256 is
# 31d12e7466ec96777a59c1023255a3a415f188f3b54eb608a23069c43ffa7dd9), lines as named, each line ending "\n".
UPSTREAM_BLOCKS = {
    (25, 37): "5cf3e5bf5efd2772241e163b1217a9291e3f867a339ede0202cf081d68bb6f90",
    (61, 81): "d38afda9f8ac57b40e34f03a5b5492c7635018e392c320723ce392e8c5083e9e",
    (84, 252): "89cd0a2c5e01ad43b491756c7ac4d8643affe19bb6ae48e3e5111f4b3e9fba35",
    (306, 451): "b3e8059a14b3a62cc8a5fb519f5c2b9d0943a7d7b1146426619b8452a0beb375",
    (831, 907): "622ef1950acc5d64f080a859b53d3561b4679f0cbbca4015c5737010b35216ff",
    (1306, 1318): "ff00e0cf5b2d11f35496845e4390ba2270c3fec387e176e4c11974de71478b1e",
    (1610, 1722): "7e01dfa41ded9a71b601e18bae698cedf2926129531c19523b7305e67a1c8ed6",
    (1910, 2005): "49f326c303c5ff267df3ffa26fb29e49bbc204f1a0aa36b9ce9b28b5af6292a7",
}
# From upstream's own tests (replay/tests/test_backtrack_memory_discipline.py).
FACTUAL_MEMORY = (
    "failed_check: P05_master_signature_valid. "
    "observed_loop: rewrote delivery PDF via PdfWriter().append().write() twice. "
    "evidence_observed: verifier said 'the signed byte range of the original "
    "signature was not preserved'. root_cause: full rewrite re-serializes bytes; "
    "the master cannot be re-created without the signer's private key, so the "
    "original file bytes must be preserved. corrected_strategy: use "
    "PdfWriter(reader, incremental=True) and only append after the signed range. "
    "do_not_repeat: PdfWriter().append().write() over delivery/contract.pdf."
)
VERDICTS = ("root_cause: Stage B is structurally blocked, no fidelity method exists.",
            "why_local_repair_failed: preserving the signature is impossible here.",
            "corrected_strategy: the P05 requirement is infeasible; skip it.",
            "observed_loop: this is a dead end, give up on the checkpoint.")


def steps():
    return [agentrewind.Step(1, ("Let me look at the tests.",), ({"name": "bash", "arguments": {"command": "ls"}},),
                             ({"name": "bash", "arguments": {"command": "ls"},
                               "output": "<returncode>0</returncode>\n<output>\nsrc\ntests\n</output>"},)),
            agentrewind.Step(2, ("I will edit the parser.",),
                             ({"name": "bash", "arguments": {"command": "sed -i 's/x/y/' src/p.py"}},),
                             ({"name": "bash", "arguments": {"command": "sed -i 's/x/y/' src/p.py"},
                               "output": "<returncode>0</returncode>\n<output>\n</output>"},))]


def test_the_copied_blocks_are_upstreams_text_byte_for_byte():
    text = (Path(agentrewind.__file__).parent / "agentrewind_upstream.py").read_text(encoding="utf-8")
    marks = list(re.finditer(r"^# --- upstream lines (\d+)-(\d+) ---\n", text, re.MULTILINE))
    found = {}
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        block = text[mark.end():end].rstrip("\n") + "\n"
        found[(int(mark.group(1)), int(mark.group(2)))] = hashlib.sha256(block.encode()).hexdigest()
        assert len(block.splitlines()) == int(mark.group(2)) - int(mark.group(1)) + 1
    assert found == UPSTREAM_BLOCKS
    # Outside the blocks: the licence and provenance header and the imports the blocks need, nothing else.
    head = text[:marks[0].start()]
    assert "MIT License" in head and "Copyright (c) 2026 Replay contributors" in head and "6661046" in head
    assert [line for line in head.splitlines() if line and not line.startswith("#")] == [
        "from __future__ import annotations", "import json", "import re", "from typing import Any, Literal"]


def test_the_tools_are_upstreams_two_in_the_responses_apis_form():
    offered = agentrewind.tools()
    assert [tool["name"] for tool in offered] == ["backtrack_candidates", "backtrack_commit"]
    for tool, definition in zip(offered, BACKTRACK_TOOL_DEFINITIONS, strict=True):
        assert tool == {"type": "function", **definition}
    offered[0]["parameters"]["required"].append("changed")
    assert agentrewind.tools()[0]["parameters"]["required"] == ["reason"]  # a copy each time
    assert agentrewind.with_guidance("You are a helpful assistant.") == (
        "You are a helpful assistant.\n\n" + agentrewind.GUIDANCE)
    assert agentrewind.GUIDANCE.startswith("Backtrack is a deliberate timeline fork escape hatch.")


def test_the_notebook_is_rendered_as_upstream_renders_it_after_a_rewind():
    notes = [{"memory_summary": "  first   note  ", "target_record_uid": "llm-3", "reason": "tests failed"},
             {"memory_summary": "x" * 1000, "target_record_uid": "llm-5", "reason": None}]
    rendered = agentrewind.notebook(notes)
    assert rendered.split("\n") == [
        DEFAULT_MEMORY_PREFIX.rstrip(), "", "1. first note (target=llm-3; reason=tests failed)",
        "2. " + "x" * 897 + "... (target=llm-5)"]
    assert agentrewind.notebook_message(notes) == {"role": "user", "content": rendered}
    assert agentrewind.is_notebook(agentrewind.notebook_message(notes))
    assert not agentrewind.is_notebook({"role": "system", "content": rendered})
    # The last 20 notes, the whole cut to 8,000 characters.
    many = agentrewind.notebook([{"memory_summary": f"note {index} " + "y" * 800} for index in range(25)])
    assert "1. note 5 " in many and "note 4 " not in many
    assert many.endswith("\n...[backtrack memory truncated]") and len(many) <= 8_000 + 32


def test_a_request_is_compared_without_what_the_arm_adds_to_it():
    plain = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Fix it."}]
    bash = {"type": "function", "name": "bash"}
    added = [{"role": "system", "content": agentrewind.with_guidance(plain[0]["content"])}, plain[1],
             agentrewind.notebook_message([{"memory_summary": "Do not edit the tests."}])]
    assert agentrewind.strip(added, [bash, *agentrewind.tools()]) == (plain, [bash])
    assert agentrewind.strip(plain, [bash]) == (plain, [bash]) and agentrewind.strip(plain, None) == (plain, None)


def test_a_tool_call_travels_as_a_command_no_bash_command_can_be():
    command = agentrewind.tool_command("backtrack_commit", {"record_uid": "llm-2", "memory_summary": "m"})
    assert command.startswith("\x00") and agentrewind.parse_tool_command(command) == (
        "backtrack_commit", {"record_uid": "llm-2", "memory_summary": "m"})
    assert agentrewind.parse_tool_command("{\nls\n} 2>&1") is None


def test_the_candidates_are_the_timelines_model_calls_rendered_by_upstream():
    listed = json.loads(agentrewind.answer("backtrack_candidates", {"reason": "the reviewer rejected it twice"},
                                           steps())[0])
    assert [item["record_uid"] for item in listed] == ["llm-1", "llm-2"]
    assert listed[1] == {
        "record_uid": "llm-2", "record_kind": "llm", "created_at": None, "decision_hint": "I will edit the parser.",
        "tool_calls_preview": json.dumps([{"name": "bash", "arguments": json.dumps({"command": "sed -i 's/x/y/' src/p.py"})}],
                                         ensure_ascii=False, sort_keys=True),
        "tool_records_preview": [{"tool_name": "bash", "arguments_preview": '{"command": "sed -i \'s/x/y/\' src/p.py"}',
                                  "status": "ok", "output_preview": '"<returncode>0</returncode>\\n<output>\\n</output>"',
                                  "error_preview": "", "effects_preview": ""}],
        "source_mode": "deterministic"}
    long = [agentrewind.Step(number) for number in range(1, 101)]
    assert [item["record_uid"] for item in agentrewind.candidates(long)][:2] == ["llm-21", "llm-22"]
    assert agentrewind.answer("backtrack_candidates", {"reason": " "}, steps())[1:] == (1, None)


def test_a_commit_passes_upstreams_checks_in_their_order_then_the_studys_budget_rule():
    ok = {"record_uid": "llm-1", "memory_summary": FACTUAL_MEMORY, "reason": "wrong file"}
    output, code, commit = agentrewind.answer("backtrack_commit", ok, steps())
    assert code == 0 and commit == agentrewind.Commit(1, FACTUAL_MEMORY, "wrong file")
    assert commit.note() == {"memory_summary": FACTUAL_MEMORY, "target_record_uid": "llm-1", "reason": "wrong file"}
    for text in VERDICTS:
        output, code, commit = agentrewind.answer("backtrack_commit", {**ok, "memory_summary": text}, steps())
        assert code == 1 and commit is None and "feasibility verdict" in output
    assert agentrewind.answer("backtrack_commit", {**ok, "memory_summary": ""}, steps())[0] == (
        "backtrack_commit requires a non-empty memory_summary.")
    assert agentrewind.answer("backtrack_commit", {**ok, "record_uid": ""}, steps())[0] == (
        "backtrack_commit requires a non-empty record_uid.")
    for missing in ("llm-3", "rec_000001", "llm-0"):
        assert agentrewind.answer("backtrack_commit", {**ok, "record_uid": missing}, steps())[:2] == (
            f"backtrack target not found: {missing}", 1)
    # A verdict is refused before the target is looked up, as upstream orders its checks.
    assert "feasibility" in agentrewind.answer("backtrack_commit", {"record_uid": "llm-9",
                                                                    "memory_summary": VERDICTS[0]}, steps())[0]
    again, code, _ = agentrewind.answer("backtrack_commit", ok, steps(), notes=[commit_note(FACTUAL_MEMORY)])
    assert code == 1 and again.startswith("memory_summary rejected: it is near-identical (token overlap 100%)")
    assert agentrewind.answer("backtrack_commit", ok, steps(), can_restart=False)[:2] == (
        agentrewind.REFUSED_FOR_BUDGET, 1)
    assert agentrewind.answer("bash", {}, steps())[1] == 1


def commit_note(summary):
    return {"memory_summary": summary, "target_record_uid": "llm-1", "reason": None}


def test_the_timeline_is_the_runs_model_calls_replayed_then_live_with_what_each_returned():
    """From a hosted run's record: one step per model reply the agent received, replayed ones first, as the
    study numbers steps everywhere; a setup command, a reply cut off by the run's end and a lost call are not
    steps."""
    listing = agentrewind.tool_command("backtrack_candidates", {"reason": "stuck"})
    events = [
        {"kind": "hosted_replay_output", "step": 0, "command": "cd /app", "returncode": 0, "output": "",
         "timed_out": False},
        {"kind": "hosted_replay", "step": 1, "text": ["Look first."],
         "calls": [{"id": "c1", "name": "bash", "arguments": {"command": "ls"}}]},
        {"kind": "hosted_replay_output", "step": 1, "command": "ls", "returncode": 0, "output": "src\n",
         "timed_out": False},
        {"kind": "hosted_request", "id": "r1"},
        {"kind": "hosted_lost", "id": "r1"},
        {"kind": "hosted_request", "id": "r2"},
        {"kind": "hosted_completion", "id": "r2", "text": ["Which point?"],
         "calls": [{"id": "t1", "name": "backtrack_candidates", "arguments": {"reason": "stuck"}}]},
        {"kind": "hosted_command", "effect_id": "e1", "command": listing},
        {"kind": "hosted_output", "effect_id": "e1", "returncode": 0, "output": "[]", "terminated": ""},
        {"kind": "hosted_completion", "id": "r3", "text": ["Again."], "calls": [], "cut_off": True},
        {"kind": "hosted_command", "effect_id": "e2", "command": "make test"},
        {"kind": "hosted_output", "effect_id": "e2", "returncode": -1, "output": "", "terminated": "timeout"}]
    steps = agentrewind.timeline(events)
    assert [step.number for step in steps] == [1, 2]
    assert steps[0] == agentrewind.Step(1, ("Look first.",), ({"name": "bash", "arguments": {"command": "ls"}},),
                                        ({"name": "bash", "arguments": {"command": "ls"},
                                          "output": {"returncode": 0, "output": "src\n"}, "error": None},))
    assert steps[1].calls == ({"name": "backtrack_candidates", "arguments": {"reason": "stuck"}},)
    assert steps[1].results == (
        {"name": "backtrack_candidates", "arguments": {"reason": "stuck"}, "output": {"returncode": 0, "output": "[]"},
         "error": None},
        {"name": "bash", "arguments": {"command": "make test"}, "output": {"returncode": -1, "output": ""},
         "error": "timeout"})


def test_a_runs_final_commit_and_where_it_began_to_rewind_skip_every_step_that_ran_no_container_command():
    from taste.agents.trajectory_reader import Step as Taken

    listing = agentrewind.tool_command("backtrack_candidates", {"reason": "stuck"})
    commit = agentrewind.tool_command("backtrack_commit", {"record_uid": "llm-2", "memory_summary": "m",
                                                           "reason": "why"})
    plain = [Taken(1, "ls"), Taken(2, "sed -i s/a/b/ x"), Taken(3, listing), Taken(4, ""), Taken(5, commit)]
    found = agentrewind.ended_at_commit(plain)
    assert found == (agentrewind.Commit(2, "m", "why"), 3, False)
    # A step with bash and a tool call ran a command: the rewind began after it.
    mixed = [Taken(1, "ls"), Taken(2, "sed -i s/a/b/ x\n" + listing), Taken(3, commit)]
    assert agentrewind.ended_at_commit(mixed)[1:] == (3, False)
    # A commit that ran a command itself: the twin goes back to the commit's own call, which undoes it (flagged).
    together = [Taken(1, "ls"), Taken(2, "make test\n" + commit)]
    assert agentrewind.ended_at_commit(together)[1:] == (2, True)
    assert agentrewind.ended_at_commit([Taken(1, "ls"), Taken(2, listing)]) is None
    assert agentrewind.ended_at_commit([]) is None


def test_a_tool_call_whose_recorded_command_was_cut_still_counts_as_a_call():
    cut = agentrewind.tool_command("backtrack_commit", {"record_uid": "llm-1", "memory_summary": "m" * 100})[:60]
    events = [{"kind": "hosted_replay", "step": 1, "text": ["x"], "calls": []},
              {"kind": "hosted_replay_output", "step": 1, "command": cut, "returncode": 1, "output": "refused",
               "timed_out": False}]
    [step] = agentrewind.timeline(events)
    assert step.results[0]["name"] == "unreadable tool call" and step.results[0]["arguments"] == {"command": cut}
